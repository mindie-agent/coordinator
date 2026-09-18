"""One persistent coordinator process per user/state directory.

CLI, MCP and TaskClient call this process. It owns the runtime pool AND
admitted task-execution progression (placement, preparation, sync, group
leases, launch). Tick runs on its own thread so an idle client cannot freeze it.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from mindie_diagnostics import get_recorder, current_context, bind_context, wrap_context
from remote_dev.observability import observed_operation, detached_tool
from remote_dev.core.errors import error_details, RemoteExecutionError
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from remote_dev.runtime import process_identity, runtime_status
from remote_dev.processes import control as preparation_control

from mindie_coordinator.agent_session import AgentSessions
from mindie_coordinator.backend import RemoteBackend
from mindie_coordinator.build_inputs import BUILD_INPUT_ENV_KEYS
from mindie_coordinator.client_paths import client_path
from mindie_coordinator.execution_sources import source_paths, validate_source_snapshot
from mindie_coordinator.managed_execution import ExecutionRequestError, JOB_TERMINAL
from mindie_coordinator.parity import materialize_command
from mindie_coordinator.parity_support import RemoteCommandError
from mindie_coordinator.placement import (
    can_prepare,
    distinct_hosts_required,
    host_key,
    provisionable_recipe,
    role_plan,
    runtime_matches,
    validate_wait,
)
from mindie_coordinator.provision.task_environment import TaskRootBusy
from mindie_coordinator.preparation_process import (
    PreparationCancelled, PreparationUncertain, stop_preparation_process,
)
from mindie_coordinator.ready_runtime import RuntimePool, user_container_name
from mindie_coordinator.state_paths import coordinator_state_dir
from mindie_coordinator.task_messages import TaskMessages

SOCKET_NAME = "coordinator.sock"
LOCK_NAME = "coordinator.lock"
IPC_NAME = "coordinator.ipc"
TICK_SECONDS = 2.0
_PROCESS_CLOCK_DOMAIN = uuid.uuid4().hex
STATUS_CACHE_SECONDS = 2.0
DONE = {"succeeded", "failed", "timeout", "cancelled", "inconclusive"}
LIVE = {"running"}
LEASE_READY = {"granted", "starting", "active"}
PERMANENT_ERRORS = (ValueError, PermissionError, ExecutionRequestError, TaskRootBusy)
CLIENT_TIMEOUT_SECONDS = 60.0
STOP_WAIT_SECONDS = 30.0
LOADED_RUNTIMES = [process_identity(name) for name in ("mindie-coordinator", "remote-dev")]


def socket_path(state_dir: Path) -> Path:
    """Short per-user/state socket. macOS AF_UNIX paths cap near 104 bytes."""
    if os.name == "nt":
        return Path(state_dir) / IPC_NAME
    resolved = str(Path(state_dir).expanduser().resolve())
    digest = hashlib.sha256(resolved.encode()).hexdigest()[:16]
    user = "".join(ch if ch.isalnum() else "-" for ch in getpass.getuser())[:12] or "user"
    return Path("/tmp") / f"vc-{user}-{digest}.sock"


def _lock_daemon(handle, *, release=False):
    if os.name == "nt":
        import msvcrt

        if not release and os.fstat(handle.fileno()).st_size == 0:
            handle.write(b" ")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK if release else msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle, fcntl.LOCK_UN if release else fcntl.LOCK_EX | fcntl.LOCK_NB)


def lock_path(state_dir: Path) -> Path:
    return Path(state_dir) / LOCK_NAME


def require_native_owner(state_dir: Path, *, platform: str | None = None) -> None:
    if (platform or os.name) == "nt":
        return
    try:
        address = json.loads((Path(state_dir) / IPC_NAME).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    except (OSError, ValueError) as exc:
        raise RuntimeError("cannot determine coordinator ownership from its IPC marker") from exc
    if isinstance(address, dict) and "port" in address:
        raise RuntimeError(
            "this shared state directory belongs to a Windows coordinator; configure the WSL "
            "coordinator MCP and session hooks to invoke the managed Windows Python executable "
            "on /mnt/<drive>. A second Linux coordinator cannot manage the same state."
        )


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


class CoordinatorService(TaskMessages):
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
        store.save_execution(row)
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

    def wait(self, sessions_dir, user, execution_id, *, until="released", timeout_seconds=30, role=None):
        validate_wait(until, timeout_seconds)
        deadline = time.monotonic() + timeout_seconds
        while True:
            # Holding the condition across the cached read prevents a save/notify
            # between inspection and sleep. The execution worker owns all remote IO.
            with self._changed:
                reply = self._observe(sessions_dir, user, execution_id, "status", role=role, refresh=False)
                terminal = reply.get("state") in DONE
                reached = ((until == "running" and (terminal or reply.get("state") == "running"))
                           or (until == "released" and terminal and reply.get("resources_released") is True))
                if reached:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stopped.is_set():
                    break
                # The bounded wake also resumes supervision after a restart or
                # a missed external update. No client-side status RPC loop.
                self._changed.wait(min(remaining, STATUS_CACHE_SECONDS))
        if terminal:
            if reply.get("resources_released") is True:
                reply = self._wait_terminal_logs(sessions_dir, user, execution_id, reply, deadline, role)
            else:
                # Terminal business state can precede descendant/lease cleanup.
                # Its log is still growing; only the final released view may
                # populate the reusable completion log cache.
                reply = {**reply, "logs_pending": True}
        return {**reply, "wait_until": until, **({"wait_timed_out": True} if not reached else {})}

    def _owned_row(self, sessions_dir, user, execution_id):
        store = self.store(sessions_dir)
        with store.transaction() as db:
            row = store.get(db, "execution", execution_id)
        if user and row.get("user") and row["user"] != user:
            raise PermissionError("execution belongs to another principal")
        return store, row

    def _wait_terminal_logs(self, sessions_dir, user, execution_id, reply, deadline, role):
        store, row = self._owned_row(sessions_dir, user, execution_id)
        lock = self._lock_for("terminal-logs", execution_id)
        if "terminal_logs" not in row and lock.acquire(blocking=False):
            def collect():
                try:
                    _, current = self._owned_row(sessions_dir, user, execution_id)
                    # Another completed waiter may have populated this between reads.
                    if "terminal_logs" in current:
                        return
                    try:
                        value = self._tail(store, user, current)
                        logs = {key: value[key] for key in ("stdout", "stderr", "tail", "preparation_logs") if key in value}
                        logs["roles"] = [{key: item[key] for key in ("name", "stdout", "stderr", "tail") if key in item}
                                         for item in value.get("roles", [])]
                    except Exception as exc:
                        logs = {"tail_error": f"{type(exc).__name__}: {exc}"[:500]}
                    # The progress worker owns this lock for every lifecycle
                    # read/modify/save. Tail IO is outside it; only publication
                    # joins that serialization to preserve newer release facts.
                    with self._lock_for("execution", execution_id):
                        _, current = self._owned_row(sessions_dir, user, execution_id)
                        current["terminal_logs"] = logs
                        self._save_execution(store, current)
                finally:
                    lock.release()
            threading.Thread(target=collect, name=f"mindie-logs-{execution_id[:12]}", daemon=True).start()
        with self._changed:
            while True:
                _, row = self._owned_row(sessions_dir, user, execution_id)
                if "terminal_logs" in row:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {**reply, "logs_pending": True}
                self._changed.wait(remaining)
        # Enrich logs only; the collector cannot replace fresh lifecycle facts.
        reply = self._reply(row, role=role)
        logs = row["terminal_logs"]
        reply.update({key: value for key, value in logs.items() if key != "roles"})
        by_name = {item["name"]: item for item in logs.get("roles", [])}
        for view in reply.get("roles", []):
            view.update(by_name.get(view.get("name"), {}))
        return reply

    def _observe(self, sessions_dir, user, execution_id, action, role=None, refresh=True) -> dict[str, Any]:
        store = self.store(sessions_dir)
        with store.transaction() as db:
            row = store.get(db, "execution", execution_id)
        if user and row.get("user") and row["user"] != user:
            raise PermissionError("execution belongs to another principal")
        user = user or row.get("user")
        if action == "status" and not refresh:
            # Observation reads persisted facts. A stale sample asks the
            # existing execution worker to refresh asynchronously, so neither
            # a slow link nor a dead host blocks status or a bounded wait.
            stale = not self._observation_freshness(row)["fresh"]
            deferred = stale and bool(row.get("admitted")) and not self._reply(row).get("resources_released")
            if deferred:
                self._schedule_progress(sessions_dir, user, execution_id)
            reply = self._reply(row, role=role)
            reply["observation_freshness"].update(
                source="cache" if row.get("jobs_observed_at") else "local",
                refresh_requested=False, refresh_deferred=deferred,
            )
            return reply
        lock = self._lock_for("execution", execution_id)
        acquired = lock.acquire(blocking=False)
        refreshed = False
        try:
            if acquired:
                # Another observer may have refreshed between the first read
                # and lock acquisition. Do not overwrite its newer snapshot.
                with store.transaction() as db:
                    row = store.get(db, "execution", execution_id)
                current = self._observation_freshness(row)["fresh"]
                if row.get("roles") and (action != "status" or refresh or not current):
                    row = self._refresh_jobs(store, user, row)
                    refreshed = bool(row.get("jobs_observed_at"))
                elif row.get("preparation_jobs") and not row.get("roles") and (action != "status" or refresh):
                    self._refresh_preparation_jobs(store, row)
            if action == "tail":
                return self._tail(store, user, row, role=role)
            reply = self._reply(row, role=role)
            reply["observation_freshness"].update(
                source="refreshed" if refreshed else "cache" if row.get("jobs_observed_at") else "local",
                refresh_requested=bool(refresh), refresh_deferred=not acquired,
            )
            return reply
        finally:
            if acquired:
                lock.release()

    def _request_stop(self, sessions_dir, user, execution_id, force) -> dict[str, Any]:
        store = self.store(sessions_dir)
        with store.transaction() as db:
            row = store.get(db, "execution", execution_id)
        if user and row.get("user") and row["user"] != user:
            raise PermissionError("execution belongs to another principal")
        user = user or row.get("user")
        row["cancel_requested"] = True
        if force:
            row["force"] = True
        self._save_execution(store, row)
        lock = self._lock_for("execution", execution_id)
        if lock.acquire(blocking=False):
            try:
                with store.transaction() as db:
                    row = store.get(db, "execution", execution_id)
                return self._stop(store, user, row, force)
            finally:
                lock.release()
        with store.transaction() as db:
            latest = store.get(db, "execution", execution_id)
        latest["cancel_requested"] = True
        if force:
            latest["force"] = True
        return self._reply(latest)

    def _advance_locked(self, sessions_dir, user, execution_id, action="progress", force=False) -> dict[str, Any]:
        store = self.store(sessions_dir)
        with store.transaction() as db:
            row = store.get(db, "execution", execution_id)
        if user and row.get("user") and row["user"] != user:
            raise PermissionError("execution belongs to another principal")
        user = user or row.get("user")
        if row.get("phase") in DONE:
            if not self._reply(row).get("resources_released"):
                # Pool cleanup can complete after the business state becomes
                # terminal. Refresh its owned facts without restarting work.
                row = self._refresh_jobs(store, user, row)
                self._refresh_preparation_jobs(store, row)
            return self._reply(row)
        with store.transaction() as db:
            session = store.get(db, "session", row["session_id"])
        if session.get("state") == "finishing" or row.get("cancel_requested"):
            force = force or bool(row.get("force")) or bool((session.get("finish") or {}).get("force"))
            return self._stop(store, user, row, force)
        try:
            return self._progress(store, user, row)
        except Exception as exc:
            return self._record_execution_error(store, row, exc)

    def _adopt_cancel(self, store, row) -> bool:
        with store.transaction() as db:
            latest = store.get(db, "execution", row["id"])
            session = store.get(db, "session", row["session_id"])
        if latest.get("cancel_requested"):
            row["cancel_requested"] = True
        if latest.get("force"):
            row["force"] = True
        if session.get("state") == "finishing":
            row["cancel_requested"] = True
            if (session.get("finish") or {}).get("force"):
                row["force"] = True
        return bool(row.get("cancel_requested"))

    def _halt_if_cancelled(self, store, user, row):
        if self._adopt_cancel(store, row):
            return self._stop(store, user, row, bool(row.get("force")))
        return None

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

        roles = spec.get("roles") or role_plan(spec.get("topology"), spec.get("resources") or {}, spec["command"])
        environment = spec.get("environment") or {}
        if not row.get("roles"):
            if row.get("preparation_jobs"):
                row["phase"] = "uncertain"
                row["error"] = "retained preparation jobs require stop/quiet cleanup before a new execution"
                self._save_execution(store, row)
                return self._reply(row)
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
                self._save_execution(store, row)
                return self._reply(row)
            role_rows = []
            existing_bindings = placed.get("bindings") or []
            catalog = {item["runtime_id"]: item for item in self.pool.catalog()}
            for index, role in enumerate(roles):
                runtime_id = placed["runtime_ids"][index]
                catalog_item = catalog.get(runtime_id)
                if catalog_item is None or not runtime_matches(catalog_item, environment, role):
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
                                  "service_port": role.get("service_port"),
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
        for role in row["roles"]:
            if role.get("preflight") and not role.get("preflight_passed"):
                self._save_progress(store, row, role["name"], {"step": "preflight"})
                self.backend.preflight(role["binding"], role["preflight"],
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
                service_port=role.get("service_port"), hold_go=hold_go,
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

        row["managed_job"] = jobs[0]["id"]
        row["observation"] = jobs[0]
        row["jobs_observed_at"] = time.time()
        row["phase"] = aggregate_job_states([job["state"] for job in jobs])
        if row["phase"] == "running":
            self._record_assignment(row, jobs)
            self._save_progress(store, row, None, {"step": "running"})
        self._save_execution(store, row)
        return self._reply(row)

    def _place_or_prepare(self, store, user, row, roles, environment) -> dict[str, Any]:
        if row.get("preparation_jobs"):
            # A restarted/failed preparation retains durable job references.
            # Re-materializing here could replace sources under a live compiler.
            # Keep this execution observable/stoppable without replaying work.
            raise PreparationUncertain("retained preparation jobs require stop/quiet cleanup before a new execution")
        catalog = self.pool.catalog()
        # Catalog entries supply compatible environments and artifact donors,
        # never an arbitrary writable cwd to repurpose for the current sources.
        # Preparation resolves the execution's own root and reusable artifacts.
        topology = (row.get("spec") or {}).get("topology") or {}
        need_distinct = distinct_hosts_required(roles, topology)

        if need_distinct:
            known = {host_key(item) for item in catalog if item.get("user") == user and host_key(item)}
            for record in self._configured_machines():
                ip = (record.get("host") or {}).get("ip")
                if ip:
                    known.add(ip)
            if len(known) < len(roles):
                return {"status": "cache_miss", "reason":
                        "not enough distinct hosts with a matching prepared environment for this topology",
                        "provisioning_started": False}
        placements = []
        used_hosts: set[str] = set()
        for role in roles:
            donor = self._donor_for_role(user, environment, role, used_hosts, need_distinct)
            if donor is None:
                donor = self._ensure_user_container(user, environment, role, used_hosts, need_distinct,
                    on_progress=lambda event, role=role: self._save_container_progress(store, row, role['name'], event))
            if donor is None:
                return {"status": "cache_miss", "reason": "no allowed host provides the requested environment",
                        "provisioning_started": False}
            host = host_key(donor)
            if need_distinct and host:
                used_hosts.add(host)
            placements.append((role, donor))

        def prepare(placement):
            role, donor = placement
            # Private roots and immutable cache reads can prepare concurrently.
            # Only actual compiler stages share the host CPU budget below.
            if self._adopt_cancel(store, row):
                return None
            try:
                return self._prepare_role(store, user, row, role, environment, donor)
            except PreparationCancelled:
                return None

        try:
            if len(placements) == 1:
                prepared_roles = [prepare(placements[0])]
            else:
                with ThreadPoolExecutor(max_workers=min(4, len(placements)),
                                        thread_name_prefix="mindie-prepare") as workers:
                    prepared_roles = list(workers.map(prepare, placements))
        except TaskRootBusy as exc:
            return {"status": "waiting", "reason": str(exc)}
        ids = []
        bindings = []
        snapshots = {}
        for (role, _donor), prepared in zip(placements, prepared_roles):
            if prepared is None:
                return {"status": "waiting", "reason": "execution cancellation requested during preparation"}
            catalog = {item["runtime_id"]: item for item in self.pool.catalog()}
            item = catalog.get(prepared["id"])
            if item is None or not runtime_matches(item, environment, role):
                return {"status": "cache_miss",
                        "reason": "prepared environment does not match requested constraints",
                        "provisioning_started": True}
            ids.append(prepared["id"])
            bindings.append(prepared.get("binding"))
            # Completed preparation already published and checked this exact
            # input. A second materialization would remove generated version
            # metadata/native copies and repeat successful preparation work.
            if (prepared.get("attestation", {}).get("preparation") or {}).get("source_id") == row["spec"]["source_snapshot"]["id"]:
                snapshots[prepared["id"]] = {record["relpath"]: record["commit"]
                                             for record in row["spec"]["source_snapshot"]["records"]}
        return {"runtime_ids": ids, "bindings": bindings, "snapshots": snapshots,
                "reason": None, "provisioning_started": True}

    def _donor_for_role(self, user, environment, role, used_hosts, require_distinct=False):
        catalog = self.pool.catalog()
        for item in catalog:
            if item.get("user") != user or not item.get("host"):
                continue
            if require_distinct and item["host"] in used_hosts:
                continue
            if not can_prepare(environment, item, role):
                continue
            return item
        return None

    def _configured_machines(self):
        machines = getattr(self.backend, "machines", None)
        if machines is None:
            return []
        try:
            return list(machines.machines())
        except Exception:
            return []

    def _ensure_user_container(self, user, environment, role, used_hosts, require_distinct=False, *, on_progress=None):
        recipe = environment.get("recipe") or environment.get("image")
        if recipe and not provisionable_recipe(recipe):
            return None
        wanted_host = str(role["host"]) if role.get("host") else None
        for record in self._configured_machines():
            host_info = record.get("host") or {}
            host_ip = host_info.get("ip")
            if not host_ip or (require_distinct and host_ip in used_hosts):
                continue
            if wanted_host and host_ip != wanted_host:
                continue
            machine_type = host_info.get("machine_type")
            if environment.get("machine_type") and machine_type != environment["machine_type"]:
                continue
            configured = record.get("container") or {}
            owned = record.get("user") in (None, user) and configured.get("name") == user_container_name(user)
            ssh_port = configured.get("ssh_port") if owned else None
            host_endpoint = {
                "host": host_ip,
                "port": int(host_info.get("port") or 22),
                "user": host_info.get("user") or "root",
            }
            if not ssh_port or recipe:
                if not recipe:
                    # An existing configured container needs no image choice.
                    # Creating a container still requires an explicit recipe.
                    continue
                from mindie_coordinator.provision import provision_user_container

                def reserve_port(*, user, container_name, port):
                    stub = {
                        "host_endpoint": host_endpoint,
                        "container_name": container_name,
                        "endpoint": {"host": host_ip, "port": port, "user": "root"},
                    }
                    return self.backend.host(stub, {
                        "action": "container-ssh-reserve", "user": user,
                        "container_name": container_name, "port": port,
                    })

                result = provision_user_container(
                    host=host_ip, image=recipe, user=user,
                    host_user=host_endpoint["user"], host_port=host_endpoint["port"],
                    # A configured name/port does not prove the requested
                    # image. The provision owner verifies existing containers
                    # and rejects mismatches without replacing them.
                    ssh_port=int(ssh_port) if ssh_port else None,
                    machine_type=machine_type or environment.get("machine_type"),
                    machines=getattr(self.backend, "machines", None),
                    reserve_port=reserve_port,
                    **({'on_progress': on_progress} if on_progress is not None else {}),
                )
                ssh_port = result["ssh_port"]
            else:
                stub = {
                    "host_endpoint": host_endpoint,
                    "container_name": user_container_name(user),
                    "endpoint": {"host": host_ip, "port": int(ssh_port), "user": "root"},
                }
                self.backend.host(stub, {
                    "action": "container-ssh-reserve", "user": user,
                    "container_name": user_container_name(user), "port": int(ssh_port),
                })
            return {
                "user": user,
                "host": host_ip,
                "host_endpoint": host_endpoint,
                "ssh_port": int(ssh_port),
                "endpoint": {"host": host_ip, "port": int(ssh_port), "user": "root"},
                "container_name": user_container_name(user),
                "recipe": recipe,
                "machine_type": machine_type,
                "service_ports": [],
            }
        return None

    def _save_container_progress(self, store, row, role, event):
        # Keep bounded phase facts in the existing execution log even after
        # prepare-root replaces the current progress. No remote payloads/keys.
        log = self.state_dir / 'runs' / row['id'] / role / 'prepare-container.log'
        self._save_progress(store, row, role, {**event, 'log_ref': str(log)})
        try:
            with log.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
        except OSError as exc:
            get_recorder("mindie-coordinator").event("WARNING", "diagnostics.persistence_failed", error_type=type(exc).__name__)

    def _save_progress(self, store, row, role, event):
        now = time.time()
        monotonic = time.monotonic()
        clock_domain = _PROCESS_CLOCK_DOMAIN
        if event.get("log_ref"):
            path = Path(event["log_ref"])
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch(exist_ok=True)
            except OSError as exc:
                get_recorder("mindie-coordinator").event("WARNING", "diagnostics.persistence_failed", error_type=type(exc).__name__)
        with self._lock_for("progress", row["id"]):
            previous = (row.get("role_progress") or {}).get(role, {}) if role else row.get("progress") or {}
            same_step = (previous.get("step"), previous.get("role")) == (event.get("step"), role)
            if previous.get("started_at") and not same_step:
                row.setdefault("stage_history", []).append({
                    **previous, "ended_at": now,
                    **({"elapsed_seconds": round(max(0, monotonic - previous["started_monotonic"]), 3)}
                       if previous.get("clock_domain") == clock_domain and "started_monotonic" in previous else
                       {"elapsed_seconds": None, "timing_incomplete": "owner clock domain changed"})})
                row["stage_history"] = row["stage_history"][-64:]
            progress = {**(previous if same_step else {}), **event, "role": role,
                        "started_at": previous.get("started_at", now) if same_step else now,
                        "updated_at": now,
                        "started_monotonic": previous.get("started_monotonic", monotonic) if same_step and previous.get("clock_domain") == clock_domain else monotonic,
                        "clock_domain": clock_domain}
            if not same_step:
                get_recorder("mindie-coordinator").event("INFO", "execution.phase", execution_id=row["id"], role=role, phase=event.get("step"))
            row["progress"] = progress
            if role:
                row.setdefault("role_progress", {})[role] = progress
            self._save_execution(store, row)

    def _prepare_role(self, store, user, row, role, environment, donor):
        from mindie_coordinator.provision import prepare_task_environment
        from contextlib import contextmanager
        @contextmanager
        def compile_scope(step):
            # Each compiler already bounds its internal parallel variants. One
            # compiler per host avoids multiplying that whole-machine budget.
            lock = self._lock_for('compile-host', host_key(donor))
            started = time.monotonic()
            while not lock.acquire(timeout=0.1):
                if self._adopt_cancel(store, row):
                    raise PreparationCancelled('cancelled while waiting for the host compiler')
            try:
                if self._adopt_cancel(store, row):
                    raise PreparationCancelled('cancelled before compilation')
                self._save_progress(store, row, role['name'], {'step': step,
                    'compile_wait_seconds': round(time.monotonic() - started, 6)})
                yield
            finally:
                lock.release()
        prepared = prepare_task_environment(
            self.pool, user=user, session_id=row["id"], role_name=role["name"],
            environment=environment, donor=donor, sources=row.get("sources") or {},
            source_snapshot=row["spec"]["source_snapshot"],
            on_progress=lambda event: self._save_progress(store, row, role["name"], event),
            log_dir=self.state_dir / "runs" / row["id"] / role["name"],
            on_preparation_job=lambda job: self._save_preparation_job(store, row, role["name"], job),
            cancel_requested=lambda: self._adopt_cancel(store, row),
            checkout_session=row["remote_session"]["id"],
            compile_scope=compile_scope,
        )
        # Each finished role remains recoverable while its siblings prepare.
        # A crash before this save is also covered by the pool's durable
        # execution-session binding, which finish reads below.
        if prepared.get("binding"):
            with self._lock_for("progress", row["id"]):
                row.setdefault("prepared_bindings", {})[role["name"]] = prepared["binding"]
                if prepared.get("attestation"):
                    row.setdefault("preparation_results", {})[role["name"]] = self._manifest_summary(prepared["attestation"])
                self._save_execution(store, row)
        return prepared

    def _save_preparation_job(self, store, row, role, job):
        with self._lock_for("progress", row["id"]):
            row.setdefault("preparation_jobs", {}).setdefault(role, {})[job["job_id"]] = dict(job)
            progress = (row.get("role_progress") or {}).get(role)
            if progress and progress.get("step") == job.get("step"):
                progress["process"] = {key: job.get(key) for key in ("job_id", "state", "quiet", "observed_at")}
                current = row.get("progress") or {}
                if current.get("role") == role and current.get("step") == job.get("step"):
                    current["process"] = progress["process"]
            self._save_execution(store, row)

    def _refresh_preparation_jobs(self, store, row):
        for role, records in (row.get("preparation_jobs") or {}).items():
            for record in records.values():
                if record.get("quiet"):
                    continue
                try:
                    observed = preparation_control(record["endpoint"], record["job_id"], "status")
                except Exception as exc:
                    observed = {"state": "uncertain", "quiet": False, "error": str(exc)[:500]}
                record.update({key: value for key, value in observed.items() if key != "processes"})
                record["observed_at"] = time.time()
                self._save_preparation_job(store, row, role, record)

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

    def _refresh_jobs(self, store, user, row):
        if not row.get("roles"):
            return row
        def observe_role(role):
            if not role.get("managed_job"):
                return
            job = self.pool.managed_control(user, role["managed_job"], "status")
            role["observation"] = job
            role["status_observed_at"] = time.time()
        _map_roles(observe_role, row["roles"])
        if row["roles"]:
            if any(role.get("managed_job") for role in row["roles"]):
                row["jobs_observed_at"] = time.time()
            row["observation"] = row["roles"][0].get("observation")
            row["managed_job"] = row["roles"][0].get("managed_job")
            states = [role.get("observation", {}).get("state") for role in row["roles"] if role.get("observation")]
            if states:
                row["phase"] = aggregate_job_states(states)
        self._save_execution(store, row)
        return row

    def _stop(self, store, user, row, force):
        row["cancel_requested"] = True
        if force:
            row["force"] = True
        self._stop_jobs(user, row, force)
        preparation_quiet = self._stop_preparation_jobs(store, row, force)
        row = self._refresh_jobs(store, user, row)
        job_states = [role.get("observation", {}).get("state")
                      for role in row.get("roles") or [] if role.get("managed_job")]
        if not preparation_quiet:
            row["phase"] = "uncertain"
            row["error"] = "preparation stop has not verified quiet; retained jobs remain owned"
            self._save_execution(store, row)
        elif not job_states:
            row["phase"] = "cancelled"
            self._save_execution(store, row)
        return self._reply(row)

    def _stop_preparation_jobs(self, store, row, force):
        jobs = [(role, job) for role, records in (row.get("preparation_jobs") or {}).items()
                for job in records.values()]
        def stop_job(item):
            role, job = item
            if job.get("quiet"):
                return True
            return stop_preparation_process(job,
                lambda value: self._save_preparation_job(store, row, role, value), force=force)
        if len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as workers:
                return all(list(workers.map(stop_job, jobs)))
        return all(stop_job(item) for item in jobs)

    def _stop_and_wait(self, store, sessions_dir, user, row, timeout=None):
        if timeout is None:
            timeout = STOP_WAIT_SECONDS
        deadline = time.time() + timeout
        reply = self._request_stop(sessions_dir, user, row["id"], False)
        while reply.get("state") not in DONE and time.time() < deadline:
            time.sleep(0.05)
            reply = self._request_stop(sessions_dir, user, row["id"], False)
        return reply

    def _stop_jobs(self, user, row, force):
        def stop_role(role):
            if role.get("managed_job"):
                try:
                    role["observation"] = self.pool.managed_control(user, role["managed_job"], "stop", force)
                except Exception:
                    pass
        _map_roles(stop_role, row.get("roles") or [])

    def _record_assignment(self, row, jobs):
        assignment = row.get("assignment") or {"roles": []}
        for index, (role, job) in enumerate(zip(row["roles"], jobs)):
            devices = (job.get("environment") or {}).get("ASCEND_RT_VISIBLE_DEVICES")
            entry = (assignment.get("roles") or [{}] * len(jobs))
            if index < len(entry):
                entry[index].update({
                    "name": role["name"],
                    "runtime_id": role["runtime_id"],
                    "host": role["binding"]["endpoint"]["host"],
                    "root": role["binding"]["endpoint"]["cwd"],
                    "rank": index,
                    "devices": devices,
                    **({"allow_external_busy": True} if role.get("allow_external_busy") else {}),
                    "service_port": job.get("service_port"),
                    "state": job["state"],
                })
        row["assignment"] = assignment

    def _record_execution_error(self, store, row, exc) -> dict[str, Any]:
        permanent = isinstance(exc, PERMANENT_ERRORS)
        # A build/preparation command that exited nonzero has a known result.
        # Repeating the same source preparation cannot resolve that failure.
        # Lost SSH transport (255) and already-bound/live work remain recoverable.
        if (isinstance(exc, RemoteCommandError) and exc.returncode != 255
                and row.get("phase") == "preparing" and not row.get("roles")):
            permanent = True
        if any(not job.get("quiet") for records in (row.get("preparation_jobs") or {}).values()
               for job in records.values()):
            permanent = False
        path = self.state_dir / "runs" / row["id"] / "error.log"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)) + "\n")
            row["error_ref"] = str(path)
        except OSError as log_exc:
            get_recorder("mindie-coordinator").event("WARNING", "diagnostics.persistence_failed", error_type=type(log_exc).__name__)
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
        row["error_details"] = error_details(exc)
        row["phase"] = "failed" if permanent else "uncertain"
        self._save_execution(store, row)
        return self._reply(row)

    def _record_daemon_error(self, message: str) -> None:
        get_recorder("mindie-coordinator").event("ERROR", "daemon.error", message=message)
        path = self.state_dir / "daemon.log"
        try:
            with path.open("a") as stream:
                stream.write(datetime.now(timezone.utc).isoformat() + " " + message[:500] + "\n")
        except OSError:
            return

    def _role_view(self, row, role) -> dict[str, Any]:
        job = role.get("observation") or {}
        binding = role.get("binding") or {}
        endpoint = dict(binding.get("endpoint") or {})
        view = {
            "name": role.get("name"),
            "state": job.get("state") or row.get("phase"),
            "runtime_id": role.get("runtime_id"),
            "service_port": job.get("service_port") if job.get("service_port") is not None else role.get("service_port"),
            **({"allow_external_busy": True} if role.get("allow_external_busy") else {}),
            "host": endpoint.get("host"),
            "root": endpoint.get("cwd") or endpoint.get("root"),
            "endpoint": endpoint or None,
            "env": dict(role.get("env") or {}),
            "error": job.get("error"),
            "error_details": job.get("error_details"),
            "lease_state": job.get("lease_state"),
            "quiet": (job.get("remote") or {}).get("quiet"),
            "descendants_drained": (job.get("remote") or {}).get("descendants_drained"),
            "status_observed_at": role.get("status_observed_at"),
            "command_timings": ((job.get("remote") or {}).get("result") or {}).get("timings"),
        }
        if binding.get("endpoint"):
            target = self._target(row, binding, job)
            role_state = job.get("state") or row.get("phase")
            target = {**target, "state": role_state, "live": role_state in LIVE}
            view["target"] = target
            view["environment"] = target.get("environment") or {}
            if view["service_port"] is None:
                view["service_port"] = target.get("service_port")
        return view

    @staticmethod
    def _observation_freshness(row):
        sampled = row.get("jobs_observed_at")
        age = time.time() - sampled if isinstance(sampled, (int, float)) else None
        return {"snapshot_completed_at": sampled, "age_seconds": round(age, 3) if age is not None else None,
                "max_age_seconds": STATUS_CACHE_SECONDS,
                "fresh": age is not None and 0 <= age <= STATUS_CACHE_SECONDS}

    def _reply(self, row, role=None) -> dict[str, Any]:
        job = row.get("observation") or {}
        state = row.get("phase") or job.get("state")
        payload = {"execution_id": row["id"], "state": state, "service": (row.get("spec") or {}).get("service"),
                   "assignment": row.get("assignment"), "observed_at": time.time(),
                   "progress": row.get("progress"), "observation_freshness": self._observation_freshness(row)}
        if row.get("role_progress"):
            payload["role_progress"] = row["role_progress"]
        snapshot = (row.get("spec") or {}).get("source_snapshot")
        if snapshot:
            payload["source_snapshot_id"] = snapshot["id"]
            payload["sources"] = snapshot["sources"]
        if row.get("cancel_requested"):
            payload["cancel_requested"] = True
        if row.get("error"):
            payload["error"] = str(row["error"])[:500]
        if row.get("error_details"):
            payload["error_details"] = row["error_details"]
            payload["error_ref"] = row.get("error_ref")
        if row.get("placement") and state == "waiting_for_runtime":
            payload.update({k: row["placement"].get(k) for k in ("reason", "provisioning_started") if k in (row["placement"] or {})})
        roles = row.get("roles") or []
        from mindie_coordinator.ready_runtime import TERMINAL
        payload["resources_released"] = state in DONE and all(
            not item.get("managed_job") or (item.get("observation") or {}).get("lease_state") in TERMINAL
            for item in roles) and all(job.get("quiet") for records in (row.get("preparation_jobs") or {}).values()
                                      for job in records.values())
        role_views = [self._role_view(row, item) for item in roles]
        if role:
            role_views = [item for item in role_views if item.get("name") == role]
        if len(roles) > 1 or role_views:
            payload["roles"] = role_views
        binding = row.get("binding") or (roles[0]["binding"] if roles else None)
        first_job = (roles[0].get("observation") if roles else None) or job
        if binding and state in LIVE | DONE:
            payload["target"] = self._target(row, binding, first_job)
            payload["service_port"] = payload["target"]["service_port"]
        return payload

    def _tail(self, store, user, row, role=None) -> dict[str, Any]:
        payload = self._reply(row, role=role)
        selected = row.get("roles") or []
        if role:
            selected = [item for item in selected if item.get("name") == role]
        collected = []
        for item in selected:
            if not item.get("managed_job"):
                continue
            observed = self.pool.managed_control(user, item["managed_job"], "tail")
            remote = observed.get("remote") or {}
            stdout = remote.get("stdout") or ""
            stderr = remote.get("stderr") or ""
            text = stdout or stderr or remote.get("tail") or ""
            collected.append({"name": item.get("name"), "stdout": stdout, "stderr": stderr, "tail": text})
        if len(collected) == 1:
            payload["stdout"] = collected[0]["stdout"]
            payload["stderr"] = collected[0]["stderr"]
            payload["tail"] = collected[0]["tail"] or collected[0]["stdout"]
        if payload.get("roles"):
            by_name = {item["name"]: item for item in collected}
            for view in payload["roles"]:
                extra = by_name.get(view.get("name"))
                if extra:
                    view.update(extra)
        if not selected and row.get("role_progress"):
            preparation_logs = []
            for name, progress in row["role_progress"].items():
                if role and name != role:
                    continue
                path = progress.get("log_ref")
                if path and Path(path).is_file():
                    with Path(path).open("rb") as stream:
                        stream.seek(max(0, Path(path).stat().st_size - 32768))
                        preparation_logs.append({"name": name, "step": progress.get("step"),
                                                 "tail": stream.read().decode("utf-8", errors="replace")})
            payload["preparation_logs"] = preparation_logs
            if len(preparation_logs) == 1:
                payload["tail"] = preparation_logs[0]["tail"]
        return payload

    @staticmethod
    def _manifest_summary(manifest):
        return {**{key: manifest[key] for key in ("runtime_root", "profile_key", "build_key", "build_inputs",
                    "preparation", "execution_view", "source_id") if key in manifest},
                "file_count": len(manifest.get("files", {}))}

    def evidence(self, sessions_dir, user, execution_id, *, role=None, section="all", path=None):
        """Read this execution's existing local source/build receipts; never probe a mutable remote root."""
        if section not in {"all", "sources", "preparation", "build", "diagnostics"}:
            raise ValueError("section must be all, sources, preparation, build or diagnostics")
        if path is not None and (not isinstance(path, str) or not path or len(path) > 1000):
            raise ValueError("path must be a nonempty artifact path substring of at most 1000 characters")
        store, row = self._owned_row(sessions_dir, user, execution_id)
        payload = self._reply(row, role=role)
        snapshot = row.get("spec", {}).get("source_snapshot") or {}
        evidence = {"basis": "recorded execution inputs and preparation receipts; no live revalidation"}
        runs = [item["managed_job"] for item in row.get("roles", []) if item.get("managed_job") and (role is None or item.get("name") == role)]
        try:
            diagnostic_events = self.pool.diagnostic_events(user, runs)
        except PermissionError:
            raise
        except Exception as exc:
            diagnostic_events = {"operations": [], "unavailable": type(exc).__name__}
        evidence["diagnostics"] = {"context": row.get("diagnostics_context") or {}, **diagnostic_events}
        if section == "diagnostics":
            from mindie_coordinator.diagnostics import execution_bundle
            try:
                evidence["support_bundle"] = execution_bundle(row, payload, diagnostic_events,
                    self.state_dir / "results" / ("support-" + uuid.uuid4().hex + ".json"))
            except Exception as exc:
                evidence["support_bundle"] = {"unavailable": type(exc).__name__}
        if section in {"all", "sources"}:
            evidence["sources"] = {
                "snapshot_id": snapshot.get("id"), "captured_at": snapshot.get("captured_at"),
                "record_ref": str(store.state_dir / "source-inputs" / (snapshot.get("id", "unknown") + ".json")),
                "repositories": [{key: record[key] for key in ("relpath", "commit", "tree", "source_head", "ref", "build_inputs")
                                  if key in record} for record in snapshot.get("records", [])],
            }
            if row.get("spec", {}).get("script"):
                evidence["script"] = row["spec"]["script"]
        if section in {"all", "preparation"}:
            evidence["preparation"] = {"stages": row.get("stage_history", []),
                "current": row.get("role_progress") or row.get("progress"),
                "phase_observed_at": row.get("phase_observed_at", {}),
                "jobs": [{"name": name, **{key: job[key] for key in ("step", "state", "quiet", "job_id", "observed_at", "stage_timings", "elapsed_seconds", "timings", "command_timings", "transport_totals", "diagnostics_context") if key in job}}
                         for name, jobs in row.get("preparation_jobs", {}).items() if role is None or role == name
                         for job in jobs.values()]}
        if section in {"all", "preparation", "build"}:
            names = list(dict.fromkeys([*(item["name"] for item in row.get("roles", [])),
                          *row.get("role_progress", {}), *row.get("preparation_results", {})]))
            if role:
                names = [name for name in names if name == role]
            evidence["roles"] = [self._role_evidence(row, name, path) for name in names]
        payload["evidence"] = evidence
        return payload

    def _role_evidence(self, row, name, path):
        directory = (self.state_dir / "runs" / row["id"] / name).resolve()
        run_root = (self.state_dir / "runs" / row["id"]).resolve()
        if not directory.is_relative_to(run_root):
            raise ValueError("invalid recorded role log directory")
        result = {"name": name, "logs": []}
        summary = row.get("preparation_results", {}).get(name)
        if summary:
            result["build"] = summary
        # Only this owned execution's direct role logs are considered. Raw
        # receipts are decoded internally, with the backend's existing bound
        # and digest checks; their base64 text never goes to the Agent.
        from mindie_coordinator.backend import _captured_manifest
        manifest = None
        selected = next((item for item in row.get("roles", []) if item.get("name") == name), {})
        binding = selected.get("binding") or row.get("prepared_bindings", {}).get(name) or {}
        runtime_id = selected.get("runtime_id") or binding.get("runtime_id")
        if runtime_id:
            try:
                with self.pool.transaction() as db:
                    runtime = self.pool.get(db, "runtime", runtime_id)
                retained = runtime.get("attestation") or {}
                if self._matching_receipt(row, binding, retained):
                    manifest = retained
                    result["build"] = {**self._manifest_summary(manifest), "runtime_id": runtime_id,
                                       "record_ref": str(self.pool.db_path)}
                else:
                    result["receipt_error"] = "Retained runtime profile differs from this execution's fixed binding"
            except (ValueError, OSError) as exc:
                result["receipt_error"] = f"{type(exc).__name__}: {exc}"[:500]
        for log in sorted(directory.glob("*.log"))[:32]:
            if log.is_symlink() or not log.is_file():
                continue
            entry = {"step": log.stem, "log_ref": str(log), "bytes": log.stat().st_size}
            try:
                with log.open("rb") as stream:
                    size = log.stat().st_size
                    limit = 32 * 1024 * 1024
                    stream.seek(max(0, size - limit))
                    text = stream.read(limit).decode("utf-8", errors="replace")
                captured = None
                for receipt in reversed(text.splitlines()):
                    if not receipt.startswith('{'):
                        continue
                    try:
                        value = json.loads(receipt)
                    except ValueError:
                        continue
                    for key in ('preparation_timings', 'native_cache', 'native_smoke_executed'):
                        if key in value:
                            entry.setdefault(key, value[key])
                    if "manifest_zlib_base64" in value:
                        captured = _captured_manifest(receipt)
                    elif value.get("profile_key") and value.get("build_key") and "files" in value:
                        captured = value
                    if captured is not None:
                        break
                if captured is not None:
                    if not self._matching_receipt(row, binding, captured):
                        raise ValueError("profile receipt does not match this execution's source snapshot and binding")
                    manifest = captured
                    result["build"] = {**self._manifest_summary(manifest), "record_ref": str(log)}
                    entry["summary"] = "Captured profile decoded; source snapshot matches"
                else:
                    entry["tail"] = text[-2000:]
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"[:500]
            result["logs"].append(entry)
        if path is not None:
            if manifest is None:
                result["artifact_status"] = "No retained complete profile receipt; artifact hashes unavailable"
            else:
                matches = [(key, value) for key, value in manifest.get("files", {}).items() if path in key]
                result["artifacts"] = [{"path": key, **value} for key, value in matches[:32]]
                result["artifacts_total"] = len(matches)
        return result

    @staticmethod
    def _matching_receipt(row, binding, manifest):
        source_id = (manifest.get("execution_view") or {}).get("source_id") or manifest.get("source_id") or (manifest.get("preparation") or {}).get("source_id")
        endpoint = binding.get("endpoint") or {}
        root = endpoint.get("root") or endpoint.get("cwd")
        return (bool(manifest) and source_id == row.get("spec", {}).get("source_snapshot", {}).get("id")
                and (not binding.get("build_key") or manifest.get("build_key") == binding["build_key"])
                and (not root or manifest.get("runtime_root") == root))

    def _target(self, row, binding, job) -> dict[str, Any]:
        state = row.get("phase") or (job or {}).get("state")
        environment = {}
        lease_env = (job or {}).get("environment") or {}
        if lease_env.get("ASCEND_RT_VISIBLE_DEVICES"):
            environment["ASCEND_RT_VISIBLE_DEVICES"] = lease_env["ASCEND_RT_VISIBLE_DEVICES"]
        service_port = (job or {}).get("service_port")
        if service_port is None and lease_env.get("MINDIE_SERVICE_PORT"):
            service_port = int(lease_env["MINDIE_SERVICE_PORT"])
        if service_port is not None:
            environment["MINDIE_SERVICE_PORT"] = str(service_port)
        # A completed job keeps its original attestation even if its binding
        # is subsequently refreshed for another source version.
        launch_observation = dict((job or {}).get("launch_observation") or {})
        return {
            "execution_id": row["id"], "session_id": row["session_id"],
            "runtime_id": binding["runtime_id"], "binding_id": binding["id"],
            "user": binding.get("user"), "container_name": binding.get("container_name"),
            "endpoint": dict(binding["endpoint"]), "host_endpoint": binding.get("host_endpoint"),
            "container_id": binding.get("container_id"), "python": binding.get("python"),
            "profile_key": binding.get("profile_key"), "build_key": binding.get("build_key"),
            "launch_env": binding.get("launch_env") or {},
            "launch_preamble": binding.get("launch_preamble") or "",
            "launch_observation": launch_observation,
            **({"allow_external_busy": True} if (job or {}).get("request", {}).get("allow_external_busy") else {}),
            "environment": environment, "service_port": service_port,
            "state": state, "live": state in LIVE,
            "assignment": row.get("assignment"),
        }

    def finish(self, sessions_dir, user, session_id, force=False) -> dict[str, Any]:
        store = self.store(sessions_dir)
        with store.transaction() as db:
            session = store.get(db, "session", session_id)
            session["state"] = "finishing"
            session["finish"] = {"user": user, "force": bool(force), "at": time.time()}
            store.put(db, "session", session)
        states = []
        for row in store.executions(session_id):
            if row.get("phase") in DONE:
                continue
            states.append(self.advance(sessions_dir, user, row["id"], action="stop", force=force))
        completed = self._complete_finishing_session(sessions_dir, store, session_id)
        if completed["state"] == "finished":
            return {**completed, "executions": states or completed.get("executions") or []}
        return {"state": "finishing", "executions": states, "worktrees_preserved": True}

    def _resume_finishing_sessions(self, sessions_dir, store) -> None:
        for session in store.sessions():
            if session.get("state") != "finishing":
                continue
            intent = session.get("finish") or {}
            user = intent.get("user") or ""
            force = bool(intent.get("force"))
            for row in store.executions(session["id"]):
                if not row.get("admitted") or row.get("phase") in DONE:
                    continue
                if self._lock_for("execution", row["id"]).locked():
                    continue
                try:
                    self.advance(sessions_dir, user or row.get("user") or "", row["id"],
                                 action="stop", force=force)
                except Exception as exc:
                    self._record_daemon_error(f"finish stop {row.get('id')}: {exc}")
            try:
                self._complete_finishing_session(sessions_dir, store, session["id"])
            except Exception as exc:
                self._record_daemon_error(f"finish {session.get('id')}: {exc}")

    def _complete_finishing_session(self, sessions_dir, store, session_id) -> dict[str, Any]:
        with store.transaction() as db:
            session = store.get(db, "session", session_id)
        if session["state"] == "finished":
            return {"state": "finished", "executions": [self._reply(row) for row in store.executions(session_id)],
                    "worktrees_preserved": True}
        if session["state"] != "finishing":
            return {"state": session["state"], "executions": [], "worktrees_preserved": True}
        rows = store.executions(session_id)
        finishing = {"state": "finishing", "executions": [self._reply(row) for row in rows],
                     "worktrees_preserved": True}
        if any(row.get("admitted") and row.get("phase") not in DONE for row in rows):
            return finishing
        acquired = []
        try:
            for row in sorted(rows, key=lambda item: item["id"]):
                lock = self._lock_for("execution", row["id"])
                if not lock.acquire(blocking=False):
                    return finishing
                acquired.append(lock)
            rows = store.executions(session_id)
            finishing = {"state": "finishing", "executions": [self._reply(row) for row in rows],
                         "worktrees_preserved": True}
            if any(row.get("admitted") and row.get("phase") not in DONE for row in rows):
                return finishing
            user = (session.get("finish") or {}).get("user") or next(
                (row.get("user") for row in rows if row.get("user")), "")
            seen = set()
            for row in rows:
                bindings = [role.get("binding") for role in row.get("roles") or []]
                bindings.extend((row.get("prepared_bindings") or {}).values())
                if row.get("remote_session"):
                    bindings.extend(self.pool.session_bindings(user, row["remote_session"]["id"]))
                for binding in bindings:
                    if not binding or binding["id"] in seen:
                        continue
                    seen.add(binding["id"])
                    try:
                        self.pool.return_runtime(user, binding["id"])
                    except ValueError:
                        return finishing
                    except PermissionError:
                        continue
            with store.transaction() as db:
                session = store.get(db, "session", session_id)
                session["state"] = "finished"
                store.put(db, "session", session)
            return {"state": "finished", "executions": [self._reply(row) for row in rows],
                    "worktrees_preserved": True}
        finally:
            for lock in acquired:
                lock.release()

    def serve(self) -> int:
        lock = lock_path(self.state_dir).open("a+b")
        try:
            _lock_daemon(lock)
        except OSError:
            lock.close()
            raise RuntimeError(f"coordinator already running for {self.state_dir}")
        path = socket_path(self.state_dir)
        server = None
        marker = self.state_dir / IPC_NAME
        temporary = marker.with_name(marker.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            path.unlink(missing_ok=True)
            windows = os.name == "nt"
            server = socket.socket(socket.AF_INET if windows else socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                server.bind(("127.0.0.1", 0) if windows else str(path))
            except OSError as exc:
                raise RuntimeError(f"coordinator socket bind failed at {path} ({len(str(path))} bytes): {exc}") from exc
            if not windows:
                os.chmod(path, 0o700)
            self._ipc_token = secrets.token_hex(32) if windows else None
            address = {"host": "127.0.0.1", "port": server.getsockname()[1],
                       "token": self._ipc_token} if windows else {"socket": str(path)}
            server.listen(16)
            server.settimeout(1.000_000)
            # Publish only a complete record after the listener is ready.
            temporary.write_text(
                json.dumps({**address, "state_dir": str(self.state_dir.resolve())}) + "\n", encoding="utf-8"
            )
            temporary.chmod(0o600)
            os.replace(temporary, marker)
            print(f"mindie-coordinator listening socket={path} state={self.state_dir} lock={lock_path(self.state_dir)}",
                  flush=True)
            self._async_progress = True
            ticker = threading.Thread(target=self._tick_loop, name="mindie-coordinator-tick", daemon=True)
            ticker.start()
            while not self._stopped.is_set():
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                threading.Thread(target=self._serve_conn, args=(conn,), daemon=True).start()
        finally:
            self._stopped.set()
            self.close_messages()
            if server is not None:
                server.close()
            try:
                path.unlink(missing_ok=True)
                marker.unlink(missing_ok=True)
                temporary.unlink(missing_ok=True)
            finally:
                try:
                    _lock_daemon(lock, release=True)
                finally:
                    lock.close()
        return 0

    def _tick_loop(self) -> None:
        while not self._stopped.wait(TICK_SECONDS):
            try:
                self._dispatch_progress()
            except Exception as exc:
                self._record_daemon_error(f"tick: {type(exc).__name__}: {exc}")

    def _dispatch_progress(self) -> None:
        with self._lifecycle_lock:
            if self._stopped.is_set():
                return
            self._active_requests += 1
        try:
            self._dispatch_progress_active()
        finally:
            with self._lifecycle_lock:
                self._active_requests -= 1

    def _dispatch_progress_active(self) -> None:
        sessions = []
        execution_jobs = set()
        for directory in list(self._session_dirs):
            try:
                store = self.store(directory)
                executions = store.all_executions()
                sessions.append((directory, store, executions))
                execution_jobs.update(role["managed_job"] for row in executions
                                      if row.get("admitted") and row.get("phase") not in DONE
                                      and not self._lock_for("execution", row["id"]).locked()
                                      for role in row.get("roles", []) if role.get("managed_job"))
            except Exception as exc:
                self._record_daemon_error(f"sessions {directory}: {exc}")
                continue
        # Each execution worker supervises its own roles. The pool services
        # standalone jobs and free roles of busy executions. One stalled host
        # must not consume healthy siblings' renewal budget while their shared
        # execution worker remains busy. Per-job locks prevent overlap.
        self.pool.tick(exclude_managed=execution_jobs, background=True)
        for directory, store, executions in sessions:
            finishing = {session["id"] for session in store.sessions() if session.get("state") == "finishing"}
            for row in executions:
                if row.get("session_id") in finishing:
                    continue
                if not row.get("admitted") or row.get("phase") in DONE:
                    continue
                lock = self._lock_for("execution", row["id"])
                if lock.locked():
                    continue
                thread = threading.Thread(
                    target=wrap_context(self._tick_one), args=(directory, row, lock),
                    name=f"mindie-progress-{row['id'][:12]}", daemon=True,
                )
                thread.start()
            self._resume_finishing_sessions(directory, store)

    def _tick_one(self, directory, row, lock: threading.Lock) -> None:
        with bind_context(row.get("diagnostics_context") or {}), detached_tool():
            return self._tick_one_correlated(directory, row, lock)

    def _tick_one_correlated(self, directory, row, lock: threading.Lock) -> None:
        if not lock.acquire(blocking=False):
            return
        try:
            self._advance_locked(directory, row.get("user") or "", row["id"])
        except Exception as exc:
            try:
                store = self.store(directory)
                self._record_execution_error(store, row, exc)
            except Exception as record_exc:
                self._record_daemon_error(f"execution {row.get('id')}: {exc}; persist {record_exc}")
        finally:
            lock.release()
        if self._async_progress:
            try:
                store = self.store(directory)
                with store.transaction() as db:
                    current = store.get(db, 'execution', row['id'])
                for role in current.get('roles', []):
                    if role.get('managed_job') and (role.get('observation') or {}).get('state') == 'running':
                        self.pool.watch_completion(role['managed_job'],
                            lambda _job: self.advance(directory, row.get('user') or '', row['id'], action='progress'), self._stopped)
            except Exception as exc:
                self._record_daemon_error('completion observer: ' + str(exc))

    def _serve_conn(self, conn: socket.socket) -> None:
        conn.settimeout(CLIENT_TIMEOUT_SECONDS)
        try:
            data = b""
            while True:
                chunk = conn.recv(1 << 16)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break
            if not data.strip():
                return
            try:
                request = json.loads(data.decode())
                if self._ipc_token is not None and not secrets.compare_digest(
                    str(request.pop("_ipc_token", "")), self._ipc_token
                ):
                    raise PermissionError("coordinator IPC authentication failed")
                reply = self.handle(request)
            except Exception as exc:
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "error_details": error_details(exc)}
            conn.sendall((json.dumps(reply, default=str) + "\n").encode())
        except (OSError, socket.timeout, json.JSONDecodeError) as exc:
            get_recorder("mindie-coordinator").event("WARNING", "ipc.response_unavailable", error_type=type(exc).__name__)
            return
        finally:
            try:
                conn.close()
            except OSError:
                pass


class CoordinatorClient:
    def __init__(self, state_dir: Path, *, timeout=120):
        self.state_dir = Path(client_path(state_dir)).expanduser().resolve()
        self.timeout = timeout

    def call(self, op: str, **payload) -> Any:
        payload["diagnostics_context"] = current_context()
        path = socket_path(self.state_dir)
        if not path.exists():
            raise RuntimeError(f"coordinator daemon is not running at {path}")
        if os.name == "nt":
            try:
                address = json.loads(path.read_text(encoding="utf-8"))
                port, token = int(address["port"]), address["token"]
                if not 0 < port < 65536 or not isinstance(token, str) or not token:
                    raise ValueError("invalid port or token")
            except (ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(f"invalid coordinator IPC marker at {path}") from exc
            payload["_ipc_token"] = token
            target = ("127.0.0.1", port)
        else:
            target = str(path)
        conn = socket.socket(socket.AF_INET if os.name == "nt" else socket.AF_UNIX, socket.SOCK_STREAM)
        conn.settimeout(max(self.timeout, payload.get("timeout_seconds", 0) + 10) if op == "wait" else self.timeout)
        submission_state = "not_sent"
        try:
            conn.connect(target)
            submission_state = "uncertain"
            conn.sendall((json.dumps({"op": op, **payload}) + "\n").encode())
            data = b""
            while b"\n" not in data:
                chunk = conn.recv(1 << 16)
                if not chunk:
                    break
                data += chunk
        except OSError as exc:
            try:
                conn.close()
            except OSError:
                pass
            raise RemoteExecutionError("coordinator IPC response unavailable; do not resubmit an uncertain admission",
                category="ipc_transport", submission_state=submission_state,
                retryable=submission_state == "not_sent") from exc
        try:
            if not data or b"\n" not in data:
                raise RemoteExecutionError("coordinator IPC reply was lost; original request outcome is uncertain",
                    category="ipc_response", submission_state="uncertain")
            try:
                reply = json.loads(data.decode())
            except (ValueError, UnicodeError) as exc:
                raise RemoteExecutionError("coordinator IPC reply was invalid; original request outcome is uncertain",
                    category="ipc_protocol", submission_state="uncertain") from exc
        finally:
            try:
                conn.close()
            except OSError:
                pass
        if not reply.get("ok"):
            detail = reply.get("error_details") or {}
            raise RemoteExecutionError(reply.get("error") or "coordinator request failed",
                                       category=detail.get("category", "coordinator"),
                                       submission_state=detail.get("submission_state"), retryable=detail.get("retryable", False))
        return reply.get("value")

    def admit(self, sessions_dir, user, session_id, spec, restart=False):
        # A long-lived Python client must also retry an earlier busy upgrade.
        # Existing execution controls bypass this admission-only boundary.
        owner = ensure_daemon(self.state_dir)
        self.runtime = owner.runtime
        if not _daemon_runtime_matches(self):
            return _runtime_update(LOADED_RUNTIMES, self.runtime, owner.runtime_update)
        return self.call("admit", sessions_dir=str(sessions_dir), user=user,
                         session_id=session_id, spec=spec, restart=restart, client_runtime=LOADED_RUNTIMES)

    def advance(self, sessions_dir, user, execution_id, action="status", force=False, role=None, refresh=True):
        payload = dict(sessions_dir=str(sessions_dir), user=user,
                       execution_id=execution_id, action=action, force=force)
        if role:
            payload["role"] = role
        if not refresh:
            payload["refresh"] = False
        return self.call("advance", **payload)

    def wait(self, sessions_dir, user, execution_id, *, until="released", timeout_seconds=30, role=None):
        validate_wait(until, timeout_seconds)
        return self.call("wait", sessions_dir=str(sessions_dir), user=user, execution_id=execution_id,
                         until=until, timeout_seconds=timeout_seconds, role=role)

    def evidence(self, sessions_dir, user, execution_id, *, role=None, section="all", path=None):
        return self.call("evidence", sessions_dir=str(sessions_dir), user=user, execution_id=execution_id,
                         role=role, section=section, path=path)

    def finish(self, sessions_dir, user, session_id, force=False):
        return self.call("finish", sessions_dir=str(sessions_dir), user=user,
                         session_id=session_id, force=force)

    def runtime_register(self, runtime_id, spec):
        return self.call("runtime_register", runtime_id=runtime_id, spec=spec)

    def notifications(self, sessions_dir, user, session_id):
        return self.call("notifications", sessions_dir=str(sessions_dir), user=user, session_id=session_id)

    def message(self, sessions_dir, user, session_id, recipient, text):
        return self.call("message", sessions_dir=str(sessions_dir), user=user, session_id=session_id,
                         recipient=recipient, text=text)


def _runtime_identity(rows):
    return {row["package"]: {key: row.get(key) for key in ("version", "commit", "python", "location")}
            for row in rows if row.get("package")}


def _daemon_runtime_matches(client: CoordinatorClient) -> bool:
    return _runtime_identity(LOADED_RUNTIMES) == _runtime_identity(
        [row.get("loaded") or {} for row in client.runtime or []])


def _runtime_update(selected, daemon, facts):
    return {"state": "needs_runtime_update", "reason": facts.get("reason", "daemon update is pending"),
            "runtime_update": {"selected": selected, "daemon": daemon},
            **({"active_executions": facts["active_executions"]} if "active_executions" in facts else {})}


def ensure_daemon(state_dir: Path) -> CoordinatorClient:
    state_dir = Path(client_path(state_dir)).expanduser().resolve()
    require_native_owner(state_dir)
    client = CoordinatorClient(state_dir)
    try:
        client.runtime = (client.call("ping") or {}).get("runtime")
        if _daemon_runtime_matches(client):
            return client
    except (RuntimeError, FileNotFoundError, ConnectionError, OSError):
        pass
    Path(state_dir).mkdir(parents=True, exist_ok=True, mode=0o700)
    # A separate short-lived startup lock avoids spawning losing daemons that
    # are still importing when the caller has already received another's ping.
    with (Path(state_dir) / "coordinator.start.lock").open("a+b") as guard:
        deadline = time.monotonic() + 6
        while True:
            try:
                _lock_daemon(guard)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"coordinator startup is still in progress at {state_dir}")
                time.sleep(0.05)
        try:
            return _ensure_daemon_locked(state_dir, client)
        finally:
            _lock_daemon(guard, release=True)


def _ensure_daemon_locked(state_dir: Path, client: CoordinatorClient) -> CoordinatorClient:
    require_native_owner(state_dir)
    try:
        client.runtime = (client.call("ping") or {}).get("runtime")
    except (RuntimeError, FileNotFoundError, ConnectionError, OSError):
        pass
    else:
        if _daemon_runtime_matches(client):
            return client
        try:
            reply = client.call("restart_if_idle")
        except (RuntimeError, FileNotFoundError, ConnectionError, OSError) as exc:
            client.runtime_update = {"reason": f"daemon idle restart unavailable: {exc}"}
            return client
        if not isinstance(reply, dict) or reply.get("status") != "stopping":
            client.runtime_update = reply if isinstance(reply, dict) else {"reason": "daemon idle restart returned no status"}
            return client
        deadline = time.monotonic() + 5
        while socket_path(state_dir).exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        if socket_path(state_dir).exists():
            raise RuntimeError("daemon shutdown is still pending")
    log_path = Path(state_dir) / "daemon.log"
    environment = dict(os.environ)
    environment.setdefault("REMOTE_DEV_STATE_DIR", str(Path(state_dir).resolve().parent / "remote-dev-state"))
    options = ({"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
               if os.name == "nt" else {"start_new_session": True})
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "mindie_coordinator", "daemon", "--state-dir", str(state_dir)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=environment, **options,
        )
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            client.runtime = (client.call("ping") or {}).get("runtime")
            return client
        except (RuntimeError, FileNotFoundError, ConnectionError, OSError):
            if process.poll() is not None:
                detail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
                if "coordinator already running for" not in detail:
                    raise RuntimeError(f"coordinator daemon exited ({process.returncode}): {detail}")
            time.sleep(0.05)
    raise RuntimeError(f"could not start coordinator daemon at {state_dir}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the persistent local MindIE coordinator")
    parser.add_argument("--state-dir", default="", help="Coordinator state directory")
    parser.add_argument("--action", choices=("serve", "status", "restart-if-idle"), default="serve")
    args = parser.parse_args(argv)
    state = Path(client_path(args.state_dir)).expanduser().resolve() if args.state_dir else coordinator_state_dir()
    if args.action == "status":
        print(json.dumps(CoordinatorClient(state).call("ping")))
        return 0
    if args.action == "restart-if-idle":
        reply = CoordinatorClient(state).call("restart_if_idle")
        if reply["status"] == "busy":
            print(json.dumps(reply))
            return 1
        deadline = time.monotonic() + 5
        while socket_path(state).exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        if socket_path(state).exists():
            raise RuntimeError("daemon shutdown is still pending")
        print(json.dumps({"status": "restarted", "runtime": ensure_daemon(state).runtime}))
        return 0
    return CoordinatorService(state).serve()
