#!/usr/bin/env python3
"""Public host NPU coordination API, dispatcher, and compatibility re-exports.

[Role]
Defines `NpuCoordinator` from the state and lease mixins, host snapshots,
confirmed-free release probing, and the `handle_request` entry point. Every
name formerly exported by `mindie_npu_coordination` is re-exported here.

[Invariants]
- Public names, error messages, and response shapes are unchanged from the
  single-file authority.
- Compile-slot capacity is bounded by both CPU quota/affinity and available
  RAM; explicit values and the env override clamp to the hardware cap (>=1).
- `DEFAULT_MAX_COMPILE_SLOTS = 4` remains the upper default.

[Timeouts & Locks]
- SQLite behavior is inherited; request probes keep their 5s-15s timeouts.

[Side-Effect Policy]
- Dispatch preserves the original advisory, non-destructive host contract.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sqlite3
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from .mindie_npu_coordination_leases import CoordinationLeaseMixin
from .mindie_npu_coordination_probe import (
    CONTAINER_SSH_TASK_PREFIX,
    COMPILE_WORKER_MEMORY_MB,
    DEFAULT_CONTAINER_SSH_PORT_RANGE,
    DEFAULT_ESTIMATED_DURATION_SECONDS,
    DEFAULT_GRANT_TTL_SECONDS,
    DEFAULT_HEARTBEAT_TTL_SECONDS,
    DEFAULT_MAX_COMPILE_SLOTS,
    DEFAULT_QUEUE_TTL_SECONDS,
    DEFAULT_SERVING_PORT_RANGE,
    DEFAULT_START_TTL_SECONDS,
    DEFAULT_STATE_DIR,
    HBM_BUSY_THRESHOLD_MB,
    HOST_STATE_DIR_ENV,
    PORT_KINDS,
    SAFE_ID_RE,
    SCHEMA_VERSION,
    ContextVar,
    CoordinationError,
    _DIAGNOSTIC_PHASES,
    _DIAGNOSTIC_PROCESS,
    _SINGLE_ID_METRICS,
    _SINGLE_ID_TABLE_HEADER,
    _available_memory_mb,
    _cgroup_cpu_cores,
    _diagnostic_elapsed,
    _json_devices,
    _load_devices,
    _managed_completion,
    _npu_smi_cells,
    _parse_listening_port_text,
    _parse_npu_smi_device_table,
    _single_id_metric_row,
    container_ssh_task_id,
    host_worker_cap,
    parse_devices,
    parse_instant,
    parse_npu_smi_hbm,
    parse_npu_smi_info,
    parse_port_range,
    prepared_supervisor_host_pid,
    probe_listening_ports,
    probe_npu_device,
    probe_npu_occupancy,
    process_guard_busy,
    require_safe_id,
    resolve_host_state_dir,
    unguarded_cpu_process,
    unguarded_shared_process,
    user_container_name,
    utc_now_iso,
)
from .mindie_npu_coordination_state import CoordinationStateMixin

# Kept as a source-visible constant: operators and tests grep the public host
# module for the live marker name, and the probe defines the same value.
JOB_TOKEN_ENV = "REMOTE_DEV_JOB_TOKEN"


class NpuCoordinator(CoordinationStateMixin, CoordinationLeaseMixin):
    """SQLite-backed cooperative queue for one bare-metal host."""

    def __init__(
        self,
        state_dir: str | Path = DEFAULT_STATE_DIR,
        *,
        clock: Callable[[], float] = time.time,
        expected_epoch: str | None = None,
        max_compile_slots: int | None = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.db_path = self.state_dir / "coordinator.sqlite3"
        self.clock = clock
        self.expected_epoch = expected_epoch
        if max_compile_slots is None:
            configured_slots = int(os.environ.get(
                "MINDIE_COORDINATOR_MAX_COMPILE_SLOTS", DEFAULT_MAX_COMPILE_SLOTS))
            if type(configured_slots) is not int or configured_slots < 1:
                raise CoordinationError("max_compile_slots must be a positive integer")
            # The default/env cap is bounded by both CPU budget and available
            # RAM; an explicit constructor argument keeps its caller value.
            self.max_compile_slots = max(1, min(configured_slots, host_worker_cap()))
        else:
            if type(max_compile_slots) is not int or max_compile_slots < 1:
                raise CoordinationError("max_compile_slots must be a positive integer")
            self.max_compile_slots = max_compile_slots
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def admission_probe(self, task_id, probe, device_probe):
        """Use the immutable sharing policy; unknown and reclamation keep full probes."""
        with self._transaction() as connection:
            row = self._task_row(connection, require_safe_id(task_id, label="task id"))
            requested = _load_devices(row["requested_devices"])
            shared = bool(row["allow_external_busy"]) and len(requested) == 1
            # A visibility-only observation cannot reclaim a previous lease.
            # Retain full observation when this device may need reclamation,
            # so repeated shared requests cannot strand an expired reservation.
            reclaim = False
            if shared:
                now = self.clock()
                stale = connection.execute(
                    """SELECT granted_devices FROM tasks WHERE (
                        state='orphaned_busy' OR
                        (state IN ('granted','starting') AND activation_deadline<=?) OR
                        (state='active' AND heartbeat_deadline<=?))""",
                    (now, now),
                ).fetchall()
                reclaim = any(requested[0] in _load_devices(item["granted_devices"]) for item in stale)
        if shared and not reclaim:
            observed = device_probe(requested[0])
            if observed.get("status") == "ok":
                return observed
        return probe()

    def probe_requirements(self, task_id: str) -> tuple[bool, bool]:
        """Derive observation needs from immutable host-owned requests, not caller hints."""
        with self._transaction() as connection:
            row = self._task_row(connection, require_safe_id(task_id, label="task id"))
            return (bool(int(row["requested_count"]) or _load_devices(row["granted_devices"])),
                    row["requested_service_port"] is not None or bool(self._task_service_ports(connection, task_id)))

    def release_probe(self, task_id, probe, *, samples=2, interval_seconds=2.0, completion_confirmed=False):
        with self._transaction() as connection:
            row = self._task_row(connection, require_safe_id(task_id, label="task id"))
            shared = bool(row["allow_external_busy"])
            if _managed_completion(row, completion_confirmed):
                return None
        # A legacy/unconfirmed shared lease still needs visibility. One fresh
        # sample establishes device visibility; repeating a free-device
        # confirmation cannot add evidence about the owned process family.
        # release() still requires its guard to be quiet and its ports clear.
        return probe() if shared else _confirmed_free_probe(
            samples=samples, interval_seconds=interval_seconds, probe=probe)

    def startup_context(self, task_id, container_name):
        """New-run discovery needs exact facts, not global recovery/housekeeping.

        Actual acquisition performs the original housekeeping and epoch check.
        An existing task is returned explicitly and cannot use an absent proof.
        """
        task_id = require_safe_id(task_id, label="task id")
        container_name = require_safe_id(container_name, label="container name")
        with self._transaction() as connection:
            epoch = connection.execute("SELECT value FROM meta WHERE key='coordination_epoch'").fetchone()[0]
            task = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        try:
            info = json.loads(subprocess.check_output(
                ["docker", "inspect", "--format", '{"Id":{{json .Id}},"State":{{json .State}}}', container_name],
                text=True, encoding="utf-8", timeout=5))
        except subprocess.TimeoutExpired as exc:
            raise CoordinationError(f"docker inspect timed out after {exc.timeout}s for {container_name}") from exc
        except TimeoutError:
            raise
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            raise CoordinationError(f"docker inspect failed for {container_name}: {exc}") from exc
        return {"status": "ok", "coordination_epoch": epoch,
                "tasks": [self._serialize_task(task)] if task is not None else [], "container": info}

    def snapshot(
        self,
        observed: dict[str, Any] | None,
        *,
        task_id: str | None = None,
        event_limit: int = 50,
    ) -> dict[str, Any]:
        now = self.clock()
        with self._transaction() as connection:
            changes = self._housekeep(connection, observed, now=now)
            meta = {row["key"]: row["value"] for row in connection.execute("SELECT key, value FROM meta")}
            if task_id:
                task_id = require_safe_id(task_id, label="task id")
                tasks = [self._task_row(connection, task_id)]
            else:
                tasks = connection.execute(
                    "SELECT * FROM tasks ORDER BY submitted_at ASC, task_id ASC"
                ).fetchall()
            holds = connection.execute("SELECT * FROM holds ORDER BY not_before ASC, hold_id ASC").fetchall()
            if task_id:
                port_rows = connection.execute(
                    "SELECT * FROM ports WHERE task_id=? ORDER BY port ASC", (task_id,)
                ).fetchall()
            else:
                port_rows = connection.execute("SELECT * FROM ports ORDER BY port ASC").fetchall()
            events = connection.execute(
                "SELECT * FROM events ORDER BY event_id DESC LIMIT ?", (max(0, event_limit),)
            ).fetchall()
        return {
            "status": "ok",
            "schema_version": int(meta.get("schema_version", SCHEMA_VERSION)),
            "coordination_epoch": meta.get("coordination_epoch"),
            "state_dir": str(self.state_dir),
            "database": str(self.db_path),
            "ephemeral": True,
            "max_compile_slots": self.max_compile_slots,
            "tasks": [self._serialize_task(row) for row in tasks],
            "holds": [self._serialize_hold(row) for row in holds],
            "ports": [item for item in (self._serialize_port(row) for row in port_rows) if item is not None],
            "events": [
                {
                    "event_id": row["event_id"],
                    "at": utc_now_iso(float(row["at"])),
                    "kind": row["kind"],
                    "task_id": row["task_id"],
                    "data": json.loads(row["data"]),
                }
                for row in events
            ],
            "occupancy": observed,
            "gc": changes,
        }


def _confirmed_free_probe(
    *,
    samples: int,
    interval_seconds: float,
    probe: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    if samples < 1:
        raise CoordinationError("free_samples must be >= 1")
    observations: list[dict[str, Any]] = []
    for index in range(samples):
        if index:
            time.sleep(interval_seconds)
        observations.append(probe())
    latest = dict(observations[-1])
    failed = [item for item in observations if item.get("status") != "ok"]
    if failed:
        latest.update(
            {
                "status": "failed",
                "error": "one or more release confirmation probes failed",
                "busy": {},
                "free": [],
            }
        )
    else:
        # A device is releasable only when every confirmation sample observed
        # it free.  Unioning busy reasons makes one transient busy sample keep
        # the cooperative lease protected.
        combined_busy: dict[str, list[dict[str, Any]]] = {}
        common_devices = set(int(device) for device in observations[0].get("devices", []))
        for item in observations:
            common_devices.intersection_update(int(device) for device in item.get("devices", []))
            for device, reasons in item.get("busy", {}).items():
                combined_busy.setdefault(str(device), []).extend(reasons)
        latest["devices"] = sorted(common_devices)
        latest["busy"] = combined_busy
        latest["free"] = sorted(
            int(device)
            for device in latest.get("devices", [])
            if str(device) not in combined_busy
        )
    if samples > 1:
        latest["confirmation_samples"] = observations
        latest["confirmation_sample_count"] = samples
    return latest


def handle_request(request, *, probe=probe_npu_occupancy, device_probe=probe_npu_device,
                   clock=time.time, listening_ports=probe_listening_ports):
    """Self-contained host timing; no diagnostic package or remote log install."""
    phases = {}
    token = _DIAGNOSTIC_PHASES.set(phases)
    started = time.monotonic()
    def timed(name, function):
        def call(*args, **kwargs):
            before = time.monotonic()
            try:
                return function(*args, **kwargs)
            finally:
                _diagnostic_elapsed(name, before)
        return call
    try:
        result = _handle_request(request, probe=timed("occupancy_probe_ms", probe),
            device_probe=timed("device_probe_ms", device_probe), clock=clock,
            listening_ports=timed("ports_probe_ms", listening_ports))
        return {**result, "diagnostics": {"clock_domain": _DIAGNOSTIC_PROCESS,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
                "phases": {name: round(value, 3) for name, value in phases.items()}}}
    finally:
        _DIAGNOSTIC_PHASES.reset(token)


def _handle_request(
    request: dict[str, Any],
    *,
    probe: Callable[[], dict[str, Any]] = probe_npu_occupancy,
    device_probe: Callable[[int], dict[str, Any]] = probe_npu_device,
    clock: Callable[[], float] = time.time,
    listening_ports: Callable[[], dict[str, Any]] = probe_listening_ports,
) -> dict[str, Any]:
    """Execute one structured coordinator request on the host."""
    action = request.get("action")
    if action in {"submit-acquire", "submit-acquire-preflight"} and not request.get("coordination_epoch"):
        raise CoordinationError(f"{action} requires a previously observed coordination epoch")
    coordinator = NpuCoordinator(
        resolve_host_state_dir(request.get("state_dir")),
        clock=clock,
        expected_epoch=request.get("coordination_epoch"),
    )
    if action in {"message", "reply"}:
        if action == "reply" and not request.get("reply_to"):
            raise CoordinationError("reply_to is required")
        return coordinator.message(request)
    if action == "message-events":
        return coordinator.message_events(request)
    if action == "startup-context":
        return coordinator.startup_context(request["task_id"], request["container_name"])
    if action == "submit":
        return coordinator.submit(request)
    if action in {"submit-acquire", "submit-acquire-preflight"}:
        submitted = coordinator.submit(request)
        if submitted["task"]["state"] != "queued":
            return submitted
        # Keep the normal host transitions and authoritative probes. The
        # caller has one reply boundary; partial success stays discoverable
        # under this exact task ID if the probe or transport fails.
        needs_npu, needs_ports = coordinator.probe_requirements(request["task_id"])
        observed = coordinator.admission_probe(request["task_id"], probe, device_probe) if needs_npu else None
        acquired = coordinator.acquire(
            request["task_id"], observed,
            grant_ttl_seconds=int(request.get("grant_ttl_seconds") or DEFAULT_GRANT_TTL_SECONDS),
            listening=listening_ports() if needs_ports else None,
        )
        if action != "submit-acquire-preflight" or acquired.get("granted_now") is not True:
            return acquired
        # Only this acquire's new grant uses its immediately preceding sample.
        # Existing grants returned above and queued retries keep normal preflight.
        # The original transition still checks epoch, fence, expiry and conflicts.
        started = coordinator.preflight(
            request["task_id"], int(acquired["task"]["fence_token"]), observed,
            start_ttl_seconds=int(request.get("start_ttl_seconds") or DEFAULT_START_TTL_SECONDS),
        )
        return {**started, "granted_task": acquired["task"]}
    needs_npu, needs_ports = (coordinator.probe_requirements(request["task_id"])
                              if action in {"acquire", "preflight", "release", "cancel", "status", "gc"} and request.get("task_id")
                              else (True, True))
    if action == "acquire":
        return coordinator.acquire(
            request["task_id"],
            coordinator.admission_probe(request["task_id"], probe, device_probe) if needs_npu else None,
            grant_ttl_seconds=int(request.get("grant_ttl_seconds") or DEFAULT_GRANT_TTL_SECONDS),
            listening=listening_ports() if needs_ports else None,
        )
    if action == "preflight":
        return coordinator.preflight(
            request["task_id"],
            int(request["fence_token"]),
            coordinator.admission_probe(request["task_id"], probe, device_probe) if needs_npu else None,
            start_ttl_seconds=int(request.get("start_ttl_seconds") or DEFAULT_START_TTL_SECONDS),
        )
    if action == "activate":
        return coordinator.activate(
            request["task_id"],
            int(request["fence_token"]),
            pid=None if "prepared_supervisor" in request else int(request["pid"]),
            process_guard=request.get("process_guard"),
            prepared_supervisor=request.get("prepared_supervisor"),
            heartbeat_ttl_seconds=int(
                request.get("heartbeat_ttl_seconds") or DEFAULT_HEARTBEAT_TTL_SECONDS
            ),
        )
    if action == "heartbeat":
        return coordinator.heartbeat(
            request["task_id"],
            int(request["fence_token"]),
            heartbeat_ttl_seconds=int(
                request.get("heartbeat_ttl_seconds") or DEFAULT_HEARTBEAT_TTL_SECONDS
            ),
        )
    if action == "release":
        observed = coordinator.release_probe(
            request["task_id"], probe,
            samples=int(request.get("free_samples") or 2),
            interval_seconds=float(request.get("interval_seconds") or 2.0),
            completion_confirmed=request.get("completion_confirmed") is True,
        ) if needs_npu else None
        return coordinator.release(
            request["task_id"],
            int(request["fence_token"]),
            observed,
            completion_confirmed=request.get("completion_confirmed") is True,
            listening=listening_ports() if needs_ports else None,
        )
    if action == "cancel":
        return coordinator.cancel(request["task_id"], probe() if needs_npu else None,
                                  listening=listening_ports() if needs_ports else None)
    if action == "hold-add":
        return coordinator.add_hold(request, probe())
    if action == "hold-remove":
        return coordinator.remove_hold(request["hold_id"])
    if action in {"status", "gc"}:
        observed = None if request.get("no_probe") or not needs_npu else probe()
        return coordinator.snapshot(
            observed,
            task_id=request.get("task_id"),
            event_limit=int(request.get("event_limit") or 50),
        )
    if action == "container-ssh-reserve":
        # Fixed ports can describe an existing container listener. Automatic
        # selection excludes both the live sample and reservations, under the
        # same write transaction that claims the chosen port.
        return coordinator.reserve_container_ssh(
            request, listening=listening_ports() if int(request["port"]) == 0 else None,
        )
    if action == "container-ssh-release":
        return coordinator.release_container_ssh(request)
    raise CoordinationError(f"unsupported action: {action!r}")
