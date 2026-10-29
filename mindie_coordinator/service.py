"""Core coordinator service: lifecycle, IPC dispatch, and shared helpers.

[Role]
Owns `CoordinatorService` core lifecycle (`admit`/`advance`/`progress`/`serve`),
shared aggregation helpers, and re-exports the gang, evidence, and client APIs
so `mindie_coordinator.service` remains the stable public module.

[Invariants]
- Group executions persist `group_start_authorized = True` only after every
  role holds an active lease and a prepared/running job before any start gate
  is released.
- Topology-gang executions never launch from a count-only plan; the gang mixin
  observes concrete hosts/devices and rejects substituted grants.
- `wait()` never holds an execution lock while sleeping; `refresh=False` status
  reads never block on remote SSH I/O.
- A lost IPC reply leaves `submission_state="uncertain"` and is never replayed.

[Timeouts & Locks]
- Tick interval `TICK_SECONDS = 2.0`s; status cache TTL `2.0`s; stop wait
  `STOP_WAIT_SECONDS = 30.0`s; IPC timeout `CLIENT_TIMEOUT_SECONDS = 60.0`s.
- Startup lock has a 6s deadline; daemon lock is non-blocking.
- Gang observation retries are gated by `GANG_PLAN_RETRY_SECONDS`; fleet probes
  are bounded by `MAX_GANG_PROBE_HOSTS` and serialized.

[Side-Effect Policy]
- Persists execution state through `AgentSessions`, runtime leases through
  `RuntimePool`, and per-run logs under `<state_dir>/runs/<execution_id>/`.
- The daemon accept/tick loop lives in `service_client.py` and is bound onto
  `CoordinatorService` as module functions to avoid a third mixin.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from remote_dev.observability import observed_operation
from remote_dev.runtime import process_identity, runtime_status
from mindie_diagnostics import bind_context, current_context, wrap_context

from mindie_coordinator.agent_session import AgentSessions
from mindie_coordinator.backend import RemoteBackend
from mindie_coordinator.build_inputs import BUILD_INPUT_ENV_KEYS
from mindie_coordinator.client_paths import client_path
from mindie_coordinator.execution_sources import source_paths, validate_source_snapshot
from mindie_coordinator.managed_execution import ExecutionRequestError, JOB_TERMINAL
from mindie_coordinator.parity import materialize_command
from mindie_coordinator.placement import extract_topology_spec, requested_hardware, role_plan, runtime_matches
from mindie_coordinator.provision.task_environment import TaskRootBusy
from mindie_coordinator.ready_runtime import RuntimePool
from mindie_coordinator.task_messages import TaskMessages

from mindie_coordinator.service_client import (
    CLIENT_TIMEOUT_SECONDS,
    IPC_NAME,
    LOCK_NAME,
    CoordinatorClient,
    _daemon_runtime_matches,
    _dispatch_progress as _daemon_dispatch_progress,
    _dispatch_progress_active as _daemon_dispatch_progress_active,
    _ensure_daemon_locked,
    _lock_daemon,
    _runtime_identity,
    _runtime_update,
    _serve_conn as _daemon_serve_conn,
    _tick_loop as _daemon_tick_loop,
    _tick_one as _daemon_tick_one,
    _tick_one_correlated as _daemon_tick_one_correlated,
    ensure_daemon,
    lock_path,
    main,
    require_native_owner,
    serve as _daemon_serve,
    socket_path,
)
from mindie_coordinator.service_evidence import ServiceEvidenceMixin
from mindie_coordinator.service_gang import (
    GANG_PLAN_RETRY_SECONDS,
    GANG_RETRYABLE_CODES,
    MAX_GANG_PROBE_HOSTS,
    ServiceGangMixin,
)

TICK_SECONDS = 2.0
_PROCESS_CLOCK_DOMAIN = uuid.uuid4().hex
STATUS_CACHE_SECONDS = 2.0
DONE = {"succeeded", "failed", "timeout", "cancelled", "inconclusive"}
LIVE = {"running"}
PERMANENT_ERRORS = (ValueError, PermissionError, ExecutionRequestError, TaskRootBusy)
STOP_WAIT_SECONDS = 30.0
LOADED_RUNTIMES = [process_identity(name) for name in ("mindie-coordinator", "remote-dev")]


def _service_spec_key(spec: dict) -> dict:
    result = {key: spec.get(key) for key in ("command", "env", "environment", "resources", "topology",
                                            "roles", "timeout_seconds", "service", "preflight")}
    result["sources"] = (spec.get("source_snapshot") or {}).get("id")
    return result


def _map_roles(operation, roles):
    """Keep independent remote role I/O within one bounded worker group."""
    if len(roles) <= 1:
        return [operation(role) for role in roles]
    with ThreadPoolExecutor(max_workers=min(4, len(roles)), thread_name_prefix="mindie-role") as workers:
        return list(workers.map(wrap_context(operation), roles))


def aggregate_job_states(states: list[str | None]) -> str:
    """One policy for run/observe/stop. Mixed terminal is failed, not states[0]."""
    present = [state for state in states if state]
    if not present:
        return "waiting"
    if any(state == "uncertain" for state in present):
        return "uncertain"
    if any(state == "stopping" for state in present):
        return "stopping"
    if all(state == "running" for state in present):
        return "running"
    terminal = [state for state in present if state in JOB_TERMINAL]
    live = [state for state in present if state not in JOB_TERMINAL]
    if terminal and not live:
        if all(state == "succeeded" for state in terminal):
            return "succeeded"
        if all(state == "cancelled" for state in terminal):
            return "cancelled"
        if all(state == "timeout" for state in terminal):
            return "timeout"
        if all(state == "inconclusive" for state in terminal):
            return "inconclusive"
        return "failed"
    if any(state == "queued" for state in live):
        return "queued"
    if any(state in {"waiting", "pending", "granted", "starting", "prepared"} for state in live):
        return "waiting"
    if any(state == "running" for state in live):
        return "running"
    return live[0]

class CoordinatorService(ServiceGangMixin, ServiceEvidenceMixin, TaskMessages):
    # The accept/tick loop lives in service_client; binding its module functions
    # keeps a single service class (two mixins) and acyclic imports.
    serve = _daemon_serve
    _tick_loop = _daemon_tick_loop
    _dispatch_progress = _daemon_dispatch_progress
    _dispatch_progress_active = _daemon_dispatch_progress_active
    _tick_one = _daemon_tick_one
    _tick_one_correlated = _daemon_tick_one_correlated
    _serve_conn = _daemon_serve_conn

    def __init__(self, state_dir: Path, *, pool: RuntimePool | None = None, backend=None,
                 sessions: AgentSessions | None = None):
        self.state_dir = Path(client_path(state_dir)).expanduser().resolve()
        require_native_owner(self.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.backend = backend or (pool.backend if pool is not None else RemoteBackend())
        self.pool = pool or RuntimePool(self.state_dir, self.backend)
        self._sessions = sessions
        self._session_dirs: set[str] = set()
        self._lock_registry: dict[tuple[str, str], threading.Lock] = {}
        self._registry_guard = threading.Lock()
        self._changed = threading.Condition()
        self._stopped = threading.Event()
        self._mail_stopped = threading.Event()
        self._mail_workers: set[threading.Thread] = set()
        self._lifecycle_lock = threading.Lock()
        self._topology_probe_lock = threading.Lock()
        self._active_requests = 0
        self._async_progress = False
        self._ipc_token: str | None = None
        self._load_session_dirs()
        if sessions is not None:
            self._remember_session_dir(str(sessions.state_dir))
        default_sessions = self.state_dir.parent / "agent-sessions"
        if default_sessions.is_dir():
            self._remember_session_dir(str(default_sessions))

    def _lock_for(self, kind: str, key: str) -> threading.Lock:
        with self._registry_guard:
            return self._lock_registry.setdefault((kind, key), threading.Lock())

    def _save_execution(self, store, row):
        """Publish persisted facts before waking observers; no execution lock is held by a wait."""
        phase = row.get("phase")
        if phase:
            row.setdefault("phase_observed_at", {}).setdefault(phase, time.time())
        try:
            store.save_execution(row)
        except TimeoutError:
            raise
        except (OSError, sqlite3.Error) as exc:
            if phase not in DONE or not row.get("roles") or any(
                    not (role.get("observation", {}).get("remote") or {}).get("quiet")
                    for role in row["roles"]):
                raise
            row["recording_error"] = {"artifact": "execution-state", "error": str(exc)[:500],
                                      "timeout": isinstance(exc, TimeoutError)}
        with self._changed:
            self._changed.notify_all()

    def _load_session_dirs(self) -> None:
        try:
            with self.pool.transaction() as db:
                row = self.pool.get(db, "meta", "session_dirs")
            self._session_dirs.update(row.get("paths") or [])
        except ValueError:
            return

    def _remember_session_dir(self, directory: str) -> None:
        path = str(Path(directory))
        if path in self._session_dirs:
            return
        self._session_dirs.add(path)
        with self.pool.lock, self.pool.transaction() as db:
            self.pool.put(db, "meta", {"id": "session_dirs", "paths": sorted(self._session_dirs)})

    def store(self, sessions_dir: str | Path | None = None) -> AgentSessions:
        if sessions_dir:
            self._remember_session_dir(str(sessions_dir))
            if self._sessions is not None and Path(self._sessions.state_dir) == Path(sessions_dir):
                return self._sessions
            return AgentSessions(Path(sessions_dir))
        if self._sessions is not None:
            return self._sessions
        raise ValueError("sessions directory is required")

    def reconcile(self) -> None:
        self.pool.tick()
        for directory in list(self._session_dirs):
            try:
                store = self.store(directory)
            except Exception as exc:
                self._record_daemon_error(f"sessions {directory}: {exc}")
                continue
            finishing = {session["id"] for session in store.sessions() if session.get("state") == "finishing"}
            for row in store.all_executions():
                if row.get("session_id") in finishing:
                    continue
                if not row.get("admitted") or row.get("phase") in DONE:
                    continue
                try:
                    self.advance(directory, row.get("user") or "", row["id"], action="progress")
                except Exception as exc:
                    self._record_execution_error(store, row, exc)
            self._resume_finishing_sessions(directory, store)

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        with bind_context(request.get("diagnostics_context") or {}):
            return self._handle_correlated(request)

    @observed_operation(lambda self, request: "ipc." + str(request.get("op", "unknown")), component="mindie-coordinator", level="DEBUG")
    def _handle_correlated(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("op") == "ping":
            return {"ok": True, "value": {"runtime": [runtime_status(item) for item in LOADED_RUNTIMES]}}
        if request.get("op") == "restart_if_idle":
            return {"ok": True, "value": self.restart_if_idle()}
        with self._lifecycle_lock:
            if self._stopped.is_set():
                raise RuntimeError("coordinator is restarting; retry with the new daemon")
            self._active_requests += 1
        try:
            return self._handle(request)
        finally:
            with self._lifecycle_lock:
                self._active_requests -= 1

    def active_execution_refs(self):
        return [{"execution_id": row["id"], "session_id": row["session_id"], "state": row["phase"]}
                for directory in self._session_dirs for row in self.store(directory).all_executions()
                if row.get("admitted") and row.get("phase") not in DONE]

    def restart_if_idle(self):
        with self._lifecycle_lock:
            active = self.active_execution_refs()
            if self._active_requests or any(lock.locked() for lock in self._lock_registry.values()):
                return {"status": "busy", "reason": "coordinator work is in progress", "active_executions": active}
            if active:
                return {"status": "busy", "reason": "nonterminal executions remain", "active_executions": active}
            from mindie_coordinator.ready_runtime import TERMINAL
            with self.pool.transaction() as db:
                if any(row.get("state") not in TERMINAL for row in self.pool.rows(db, "run")):
                    return {"status": "busy", "reason": "unreleased resource leases remain"}
            self._stopped.set()
            return {"status": "stopping"}

    def _handle(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("op")
        if op == "tick":
            self.reconcile()
            return {"ok": True}
        if op == "admit":
            selected = request.get("client_runtime") or []
            if _runtime_identity(selected) != _runtime_identity(LOADED_RUNTIMES):
                return {"ok": True, "value": _runtime_update(
                    selected, [{"loaded": row} for row in LOADED_RUNTIMES],
                    {"reason": "daemon code differs from the selected caller", "active_executions": self.active_execution_refs()})}
            value = self.admit(request["sessions_dir"], request["user"], request["session_id"],
                               request["spec"], restart=bool(request.get("restart")),
                               wait=not self._async_progress)
            return {"ok": True, "value": value}
        if op == "advance":
            value = self.advance(request["sessions_dir"], request["user"], request["execution_id"],
                                 action=request.get("action", "status"), force=bool(request.get("force")),
                                 role=request.get("role"), refresh=request.get("refresh", True))
            return {"ok": True, "value": value}
        if op == "wait":
            return {"ok": True, "value": self.wait(request["sessions_dir"], request["user"],
                request["execution_id"], until=request.get("until", "released"),
                timeout_seconds=request.get("timeout_seconds", 30), role=request.get("role"))}
        if op == "evidence":
            return {"ok": True, "value": self.evidence(request["sessions_dir"], request["user"],
                request["execution_id"], role=request.get("role"), section=request.get("section", "all"),
                path=request.get("path"))}
        if op == "finish":
            value = self.finish(request["sessions_dir"], request["user"], request["session_id"],
                                force=bool(request.get("force")))
            return {"ok": True, "value": value}
        if op == "notifications":
            return {"ok": True, "value": self.notifications(request["sessions_dir"], request["user"], request["session_id"])}
        if op == "message":
            return {"ok": True, "value": self.message(request["sessions_dir"], request["user"], request["session_id"],
                                                       request["recipient"], request["text"])}
        if op == "pool":
            method = getattr(self.pool, request["method"])
            value = method(*request.get("args", []), **request.get("kwargs", {}))
            return {"ok": True, "value": value}
        if op == "runtime_register":
            return {"ok": True, "value": self.pool.register(request["runtime_id"], request["spec"])}
        if op == "provision":
            from mindie_coordinator.provision import provision_user_container
            return {"ok": True, "value": provision_user_container(**request.get("kwargs", {}))}
        raise ValueError(f"unsupported coordinator operation: {op!r}")

    def admit(self, sessions_dir, user, session_id, spec, restart=False, wait=None) -> dict[str, Any]:
        store = self.store(sessions_dir)
        with store.transaction() as db:
            session = store.get(db, "session", session_id)
        if session["state"] != "open":
            raise ValueError("task admission is closed; resume the task before starting another execution")
        validate_source_snapshot(spec.get("source_snapshot"))
        execution_id = None
        created = False
        with self._lock_for("session", session_id):
            if spec.get("service"):
                live = [row for row in store.executions(session_id)
                        if row.get("spec", {}).get("service") == spec["service"]
                        and row.get("phase") not in DONE]
                if live:
                    existing = live[0]
                    same = _service_spec_key(existing.get("spec") or {}) == _service_spec_key(spec)
                    if same and not restart:
                        execution_id = existing["id"]
                    elif not same and not restart:
                        previous, wanted = _service_spec_key(existing.get("spec") or {}), _service_spec_key(spec)
                        changed = [key for key in wanted if previous.get(key) != wanted[key]]
                        raise ValueError(
                            "service is already running with different inputs: " + ", ".join(changed) + "; "
                            "pass restart=True to replace it"
                        )
                    else:
                        stopped = self._stop_and_wait(store, sessions_dir, user, existing)
                        if stopped.get("state") not in DONE or stopped.get("resources_released") is not True:
                            return stopped
            if execution_id is None:
                request_id = uuid.uuid4().hex
                row = store.admit_execution(session_id, request_id, spec, user=user)
                execution_id = row["id"]
                created = True
        if wait is None:
            wait = not self._async_progress
        if not wait:
            if created:
                self._schedule_progress(sessions_dir, user, execution_id)
            return self._observe(sessions_dir, user, execution_id, "status")
        return self.advance(sessions_dir, user, execution_id, action="progress")

    def _schedule_progress(self, sessions_dir, user, execution_id) -> None:
        lock = self._lock_for("execution", execution_id)
        if lock.locked():
            return
        threading.Thread(
            target=wrap_context(self._tick_one),
            args=(sessions_dir, {"id": execution_id, "user": user, "diagnostics_context": current_context()}, lock),
            name=f"mindie-progress-{execution_id[:12]}",
            daemon=True,
        ).start()

    def advance(self, sessions_dir, user, execution_id, action="status", force=False, role=None, refresh=True) -> dict[str, Any]:
        if action in {"status", "target", "tail"}:
            return self._observe(sessions_dir, user, execution_id, action, role=role, refresh=refresh)
        if action == "stop":
            return self._request_stop(sessions_dir, user, execution_id, force)
        with self._lock_for("execution", execution_id):
            return self._advance_locked(sessions_dir, user, execution_id, action=action, force=force)

    def _progress(self, store, user, row) -> dict[str, Any]:
        halted = self._halt_if_cancelled(store, user, row)
        if halted is not None:
            return halted
        spec = row["spec"]
        session = None
        with store.transaction() as db:
            session = store.get(db, "session", row["session_id"])
        if session["state"] != "open" and not row.get("roles"):
            if session["state"] == "finishing":
                return self._stop(
                    store, user, row,
                    bool(row.get("force")) or bool((session.get("finish") or {}).get("force")),
                )
            raise ValueError("task admission is closed; resume the task before starting another execution")
        source_snapshot = validate_source_snapshot(spec["source_snapshot"])
        sources = source_paths(source_snapshot)
        if "remote_session" not in row:
            # Remote ownership is per execution. A task only supplies mutable
            # defaults and never owns a rematerializable shared source root.
            row["remote_session"] = self.pool.session_open(user, row["id"], sources)
            row["sources"] = sources
            self._save_execution(store, row)

        environment = spec.get("environment") or {}
        topology = spec.get("topology") or {}
        gang_requested = extract_topology_spec(topology) is not None
        roles = list(row.get("roles") or [])
        if roles and gang_requested and any(
            not role.get("devices") or not role.get("host") for role in roles
        ):
            # A persisted count-only plan from an older execution can never be
            # upgraded safely: the host may already have granted arbitrary
            # devices. Stop visibly instead of pretending the gang constraint
            # was observed.
            row["phase"] = "failed"
            row["error"] = (
                "topology-gang execution has a count-only role plan; stop and "
                "resubmit so observed host/device bindings are recorded"
            )
            row["error_details"] = {"code": "host_observation_required"}
            self._save_execution(store, row)
            return self._reply(row)
        if not row.get("roles"):
            if row.get("preparation_jobs"):
                row["phase"] = "uncertain"
                row["error"] = "retained preparation jobs require stop/quiet cleanup before a new execution"
                self._save_execution(store, row)
                return self._reply(row)
            retry_at = row.get("topology_retry_at")
            if isinstance(retry_at, (int, float)) and time.time() < retry_at:
                return self._reply(row)
            if gang_requested:
                planned = self._plan_gang_roles(store, user, row, spec)
                if planned.get("status") == "waiting":
                    return self._record_topology_wait(store, row, planned)
                roles = planned["roles"]
                row["topology_retry_at"] = None
                row.pop("error", None)
                row.pop("error_details", None)
            else:
                roles = spec.get("roles") or role_plan(
                    topology, spec.get("resources") or {}, spec["command"]
                )
            row["phase"] = "preparing"
            self._save_execution(store, row)
            placed = self._place_or_prepare(store, user, row, roles, environment)
            halted = self._halt_if_cancelled(store, user, row)
            if halted is not None:
                return halted
            if placed.get("status") == "waiting":
                row["phase"] = "waiting"
                row["error"] = placed.get("reason")
                self._save_execution(store, row)
                return self._reply(row)
            if placed.get("status") == "cache_miss":
                row["phase"] = "waiting_for_runtime"
                row["placement"] = placed
                if any(requested_hardware(role) for role in roles):
                    # A failed fleet probe has no reusable success cache. Reuse
                    # the existing placement retry window instead of SSH per tick.
                    row["topology_retry_at"] = time.time() + GANG_PLAN_RETRY_SECONDS
                self._save_execution(store, row)
                return self._reply(row)
            role_rows = []
            existing_bindings = placed.get("bindings") or []
            catalog = {item["runtime_id"]: item for item in self.pool.catalog()}
            # Reuse the placement decision's host snapshot; an accepted
            # placement is never re-proved with a second probe.
            host_specs = placed.get("host_specs") or {}
            for index, role in enumerate(roles):
                runtime_id = placed["runtime_ids"][index]
                catalog_item = catalog.get(runtime_id)
                if catalog_item is None or not runtime_matches(catalog_item, environment, role, host_specs):
                    row["phase"] = "waiting_for_runtime"
                    row["placement"] = {
                        "reason": "prepared environment does not match requested constraints",
                        "provisioning_started": bool(placed.get("provisioning_started")),
                    }
                    self._save_execution(store, row)
                    return self._reply(row)
                if index < len(existing_bindings) and existing_bindings[index]:
                    binding = existing_bindings[index]
                else:
                    request_id = hashlib.sha256(
                        f"{row['id']}:{runtime_id}:{role['name']}".encode()
                    ).hexdigest()
                    binding = self.pool.checkout(user, row["remote_session"]["id"], catalog_item["profile_key"],
                                                 request_id, runtime_id)
                if binding.get("status") == "cache_miss":
                    row["phase"] = "waiting_for_runtime"
                    row["placement"] = binding
                    self._save_execution(store, row)
                    return self._reply(row)
                role_rows.append({"name": role["name"], "command": role["command"], "runtime_id": runtime_id,
                                  "preflight": role.get("preflight") or spec.get("preflight"),
                                  "binding": binding, "npu_count": role.get("npu_count"),
                                  "devices": role.get("devices") or [],
                                  "host": role.get("host"),
                                  "service_port": role.get("service_port"),
                                  **({"per_die_hbm_mb": role["per_die_hbm_mb"]} if role.get("per_die_hbm_mb") else {}),
                                  **({"allow_external_busy": True} if role.get("allow_external_busy") else {}),
                                  "env": dict(role.get("env") or {})})
                prepared_snapshots = placed.get("snapshots") or {}
                if runtime_id in prepared_snapshots:
                    role_rows[-1]["snapshots"] = prepared_snapshots[runtime_id]
            row["roles"] = role_rows
            row["assignment"] = {"runtime_ids": placed["runtime_ids"],
                                 "roles": [{"name": item["name"], "runtime_id": item["runtime_id"],
                                            "host": item["binding"]["endpoint"]["host"],
                                            "root": item["binding"]["endpoint"]["cwd"],
                                            "rank": index}
                                           for index, item in enumerate(role_rows)]}
            row["binding"] = role_rows[0]["binding"]
            row["phase"] = "bound"
            self._save_execution(store, row)

        for role in row["roles"]:
            if "snapshots" in role:
                continue
            if not source_snapshot["records"]:
                role["snapshots"] = {}
                self._save_execution(store, row)
                continue
            cwd = role["binding"]["endpoint"]["cwd"]
            with self._lock_for("root", cwd):
                if self.pool.runtime_busy(role["runtime_id"]):
                    row["phase"] = "waiting"
                    row["error"] = "task root is in use by another execution; not overwriting sources"
                    self._save_execution(store, row)
                    return self._reply(row)
                self._save_progress(store, row, role["name"], {
                    "step": "sync-sources", "log_ref": str(self.state_dir / "runs" / row["id"] / role["runtime_id"] / "parity.log")})
                role["snapshots"] = self.sync_binding(role["binding"], source_snapshot, row["id"])
                self._save_execution(store, row)
                halted = self._halt_if_cancelled(store, user, row)
                if halted is not None:
                    return halted
        halted = self._halt_if_cancelled(store, user, row)
        if halted is not None:
            return halted
        if not row.get("op_preflight_passed") and (spec.get("env") or {}).get("MINDIE_SKIP_OP_PREFLIGHT") != "1":
            from mindie_coordinator.op_preflight import check_snapshot_op_contracts, format_op_violations
            violations = check_snapshot_op_contracts(source_snapshot)
            if violations:
                raise ExecutionRequestError(format_op_violations(violations))
            row["op_preflight_passed"] = True
            self._save_execution(store, row)
        for role in row["roles"]:
            if role.get("preflight") and not role.get("preflight_passed"):
                self._save_progress(store, row, role["name"], {"step": "preflight"})
                request_id = row["id"] + "-" + role["name"] if len(row["roles"]) > 1 else row["id"]
                from mindie_coordinator.runtime_profile import digest
                slot_id = digest([user, role["binding"]["id"], request_id])
                self.backend.preflight({**role["binding"], "slot_id": slot_id}, role["preflight"],
                                       {**(spec.get("env") or {}), **(role.get("env") or {})})
                role["preflight_passed"] = True
                self._save_execution(store, row)
        halted = self._halt_if_cancelled(store, user, row)
        if halted is not None:
            return halted
        if any(not role.get("managed_job") for role in row["roles"]):
            row["phase"] = "launch_pending"
            self._save_progress(store, row, None, {"step": "allocate-and-launch"})
        halted = self._halt_if_cancelled(store, user, row)
        if halted is not None:
            return halted

        hold_go = len(row["roles"]) > 1
        def start_role(role):
            if role.get("managed_job"):
                observed = self.pool.managed_control(user, role["managed_job"], "status")
                role["status_observed_at"] = time.time()
                return observed
            if self._adopt_cancel(store, row):
                return None
            merged_env = {**(spec.get("env") or {}), **(role.get("env") or {})}
            job = self.pool.managed_start(
                user, role["binding"]["id"], row["id"] + "-" + role["name"] if hold_go else row["id"],
                role["snapshots"], role["binding"]["build_key"],
                role.get("devices") or [], role.get("npu_count") or 0,
                role["command"], merged_env, spec.get("timeout_seconds"),
                service_port=role.get("service_port"), hold_go=hold_go, execution_id=row["id"],
                native_cache_receipt=(row.get("native_cache_receipts") or {}).get(role["name"]),
                proof_path=(role.get("proof_path") or (row.get("spec") or {}).get("proof_path")),
                source_snapshot=spec.get("source_snapshot"),
                **({"per_die_hbm_mb": role["per_die_hbm_mb"]} if role.get("per_die_hbm_mb") else {}),
                **({"allow_external_busy": True} if role.get("allow_external_busy") else {}),
            )
            with self._lock_for("progress", row["id"]):
                role["managed_job"] = job["id"]
                role["observation"] = job
                role["status_observed_at"] = time.time()
                self._save_execution(store, row)
            return job

        jobs = _map_roles(start_role, row["roles"])
        halted = self._halt_if_cancelled(store, user, row)
        if halted is not None:
            return halted
        for role, job in zip(row["roles"], jobs):
            role["observation"] = job
        if hold_go:
            # Same-host acquisitions honor the host FIFO. A later role may
            # have observed its sibling still queued during parallel submit;
            # recheck only those queued roles once after all submissions finish.
            if any(job.get("lease_state") == "queued" for job in jobs):
                def recheck_queued(role):
                    job = role["observation"]
                    return self.pool.managed_control(user, role["managed_job"], "status") if job.get("lease_state") == "queued" else job
                jobs = _map_roles(recheck_queued, row["roles"])
                for role, job in zip(row["roles"], jobs):
                    role["observation"] = job
            failed = [job for job in jobs if job["state"] in {"failed", "timeout", "cancelled", "inconclusive"}]
            if failed:
                self._stop_jobs(user, row, False)
                row["phase"] = "failed"
                self._save_execution(store, row)
                return self._reply(row)
            problem = self._gang_grant_problem(row, jobs) if gang_requested else None
            if problem:
                # The queue must grant exactly the observed physical Dies on
                # the observed host. A substitution is a failed constraint,
                # never an acceptable count-only placement.
                self._stop_jobs(user, row, False)
                row["phase"] = "failed"
                row["error"] = "host grant/binding does not match the observed gang plan"
                row["error_details"] = problem
                self._save_execution(store, row)
                return self._reply(row)
            if not row.get("group_start_authorized") and all(
                    job.get("lease_state") == "active" and job.get("remote", {}).get("state") in {"prepared", "running"}
                    for job in jobs):
                # Persist the group decision before sending any go command.
                # Partial replies/restarts retain this authorization and the
                # original jobs; already completed members are never relaunched.
                row["group_start_authorized"] = True
                self._save_execution(store, row)
            if row.get("group_start_authorized"):
                def release_role(role):
                    job = role["observation"]
                    if job["state"] in JOB_TERMINAL or job["state"] == "running":
                        return job
                    return self.pool.managed_release_gate(user, role["managed_job"])

                jobs = _map_roles(release_role, row["roles"])
                for role, job in zip(row["roles"], jobs):
                    role["observation"] = job
            elif not all(job["state"] in JOB_TERMINAL for job in jobs):
                row["phase"] = "queued" if any(job["state"] == "queued" for job in jobs) else "waiting"
                row["managed_job"] = jobs[0]["id"]
                row["observation"] = jobs[0]
                self._save_execution(store, row)
                return self._reply(row)
        elif gang_requested:
            # A single-role gang is launched by its own managed_start call.
            # Observe the real grant before recording a running assignment.
            problem = self._gang_grant_problem(row, jobs)
            if problem:
                self._stop_jobs(user, row, False)
                row["phase"] = "failed"
                row["error"] = "host grant/binding does not match the observed gang plan"
                row["error_details"] = problem
                self._save_execution(store, row)
                return self._reply(row)

        row["managed_job"] = jobs[0]["id"]
        row["observation"] = jobs[0]
        row["jobs_observed_at"] = time.time()
        row["phase"] = aggregate_job_states([job["state"] for job in jobs])
        if row["phase"] == "running":
            self._record_assignment(row, jobs)
            self._save_progress(store, row, None, {"step": "running"})
        self._save_execution(store, row)
        return self._reply(row)

    def sync_binding(self, binding, source_snapshot, execution_id):
        directory = self.state_dir / "runs" / execution_id / binding["runtime_id"]
        directory.mkdir(parents=True, exist_ok=True)
        endpoint = binding["endpoint"]
        args = materialize_command(workspace_id=binding["intent"]["session"],
                                   runtime_id=binding["runtime_id"], endpoint=endpoint,
                                   sources=source_paths(source_snapshot), source_snapshot=source_snapshot,
                                   workspace_root=directory)
        environment = {key: value for key, value in os.environ.items() if key not in BUILD_INPUT_ENV_KEYS}
        environment.update(binding.get("build_env", {}))
        environment.update(binding["environment"])
        with (directory / "parity.json").open("w") as stdout, (directory / "parity.log").open("w") as stderr:
            result = subprocess.run(args, stdout=stdout, stderr=stderr,
                                    env=environment, timeout=600, check=False)
        try:
            payload = json.loads((directory / "parity.json").read_text(encoding="utf-8"))
        except TimeoutError:
            raise
        except (ValueError, OSError):
            if not result.returncode:
                raise
            payload = {}
        if result.returncode:
            reason = payload.get("reason") or f"inspect {directory / 'parity.log'}"
            error = ValueError if payload.get("retryable") is False else RuntimeError
            raise error(f"source synchronization failed: {reason}")
        if payload.get("status") not in {"ready", "materialized"}:
            raise RuntimeError("source staging alone does not authorize execution")
        return payload["snapshot_commits"]
