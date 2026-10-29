"""Lease lifecycle, admission, holds, and port-reservation mixin.

[Role]
Implements task submission, hold windows, container SSH reservations, and the
queued -> granted -> starting -> active -> released lease state machine.

[Invariants]
- Fencing tokens gate activation/heartbeat/release; a stale token never wins.
- Managed `process_guard` leases are retained until the process family is
  verified quiet and completion is confirmed when required.
- Container SSH release removes only its exact owner/container reservation.

[Timeouts & Locks]
- All state changes use the inherited `BEGIN IMMEDIATE` transaction; docker
  inspection in the public context path stays bounded at 5s.

[Side-Effect Policy]
- Mutates only coordination rows; never kills processes or containers.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from .mindie_npu_coordination_probe import (
    DEFAULT_ESTIMATED_DURATION_SECONDS,
    DEFAULT_GRANT_TTL_SECONDS,
    DEFAULT_HEARTBEAT_TTL_SECONDS,
    DEFAULT_QUEUE_TTL_SECONDS,
    DEFAULT_START_TTL_SECONDS,
    CoordinationError,
    COMPILE_WORKER_MEMORY_MB,
    compile_slot_budget,
    hbm_eligible_devices,
    _authority_attribute,
    _json_devices,
    _load_devices,
    _managed_completion,
    container_ssh_task_id,
    parse_devices,
    parse_instant,
    parse_port_range,
    require_safe_id,
    unguarded_cpu_process,
    unguarded_shared_process,
    user_container_name,
    utc_now_iso,
)


class CoordinationLeaseMixin:
    def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        now = self.clock()
        task_id = require_safe_id(request.get("task_id") or str(uuid.uuid4()), label="task id")
        agent_id = require_safe_id(request.get("agent_id"), label="agent id")
        agent_alias = request.get("agent_alias")
        if agent_alias is not None:
            agent_alias = require_safe_id(agent_alias, label="agent alias")
        session_id = request.get("session_id")
        if session_id is not None:
            session_id = require_safe_id(session_id, label="session id")
        devices = parse_devices(request.get("devices"))
        count_raw = request.get("npu_count")
        if devices is not None and count_raw is not None:
            raise CoordinationError("use only one of devices or npu_count")
        if devices is None:
            count = 0 if count_raw is None else count_raw
            if type(count) is not int or count < 0:
                raise CoordinationError("npu_count must be a nonnegative integer")
        else:
            count = len(devices)
        per_die_hbm_mb = request.get("per_die_hbm_mb", 0)
        per_worker_memory_mb = request.get("per_worker_memory_mb", COMPILE_WORKER_MEMORY_MB)
        if type(per_die_hbm_mb) is not int or per_die_hbm_mb < 0:
            raise CoordinationError("per_die_hbm_mb must be a nonnegative integer")
        if type(per_worker_memory_mb) is not int or per_worker_memory_mb < 1:
            raise CoordinationError("per_worker_memory_mb must be a positive integer")
        allow_external_busy = request.get("allow_external_busy", False)
        if type(allow_external_busy) is not bool:
            raise CoordinationError("allow_external_busy must be a boolean")
        if allow_external_busy and (devices is None or len(devices) != 1):
            raise CoordinationError("allow_external_busy requires exactly one explicit physical device")
        requested_service_port = request.get("service_port")
        if requested_service_port is not None:
            requested_service_port = int(requested_service_port)
            if requested_service_port < 0:
                raise CoordinationError("service_port must be 0 or a positive TCP port")
        service_port_choices = request.get("service_ports") or []
        if not isinstance(service_port_choices, list) or any(
            type(port) is not int or not 0 < port < 65536 for port in service_port_choices
        ) or len(set(service_port_choices)) != len(service_port_choices):
            raise CoordinationError("service_ports must contain distinct TCP ports")
        if requested_service_port is not None and requested_service_port != 0 and requested_service_port not in service_port_choices:
            raise CoordinationError("requested service_port is not in declared service_ports")
        duration = int(request.get("estimated_duration_seconds") or DEFAULT_ESTIMATED_DURATION_SECONDS)
        if duration < 1:
            raise CoordinationError("estimated_duration_seconds must be >= 1")
        not_before = parse_instant(request.get("not_before"))
        if not_before is None:
            not_before = now
        latest_start = parse_instant(request.get("latest_start"))
        if latest_start is None:
            queue_ttl = int(request.get("queue_ttl_seconds") or DEFAULT_QUEUE_TTL_SECONDS)
            if queue_ttl < 1:
                raise CoordinationError("queue_ttl_seconds must be >= 1")
            latest_start = now + queue_ttl
        if latest_start < not_before:
            raise CoordinationError("latest_start must not be earlier than not_before")
        priority = int(request.get("priority") or 0)
        container_name = request.get("container_name")
        with self._transaction() as connection:
            existing = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if existing is not None:
                expected = {
                    "agent_id": agent_id,
                    "agent_alias": agent_alias,
                    "session_id": session_id,
                    "container_name": container_name,
                    "requested_count": count,
                    "per_die_hbm_mb": per_die_hbm_mb,
                    "per_worker_memory_mb": per_worker_memory_mb,
                    "requested_devices": _json_devices(devices),
                    "allow_external_busy": int(allow_external_busy),
                    "requested_service_port": requested_service_port,
                    "service_port_choices": json.dumps(service_port_choices, separators=(",", ":")) if service_port_choices else None,
                }
                mismatched = {
                    key: {"existing": existing[key], "requested": value}
                    for key, value in expected.items()
                    if existing[key] != value
                }
                if mismatched:
                    raise CoordinationError(
                        f"task id {task_id} already exists with different ownership or resources: "
                        f"{json.dumps(mismatched, sort_keys=True)}"
                    )
                return {"status": "ok", "reused": True, "task": self._serialize_task(existing)}
            connection.execute(
                """
                INSERT INTO tasks(
                    task_id, agent_id, agent_alias, session_id, container_name,
                    requested_count, requested_devices, not_before, latest_start,
                    estimated_duration_seconds, preemptible, priority, state,
                    submitted_at, updated_at, message, requested_service_port, service_port_choices,
                    allow_external_busy, per_die_hbm_mb, per_worker_memory_mb
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    agent_id,
                    agent_alias,
                    session_id,
                    container_name,
                    count,
                    _json_devices(devices),
                    not_before,
                    latest_start,
                    duration,
                    1 if request.get("preemptible") else 0,
                    priority,
                    now,
                    now,
                    request.get("message"),
                    requested_service_port,
                    json.dumps(service_port_choices, separators=(",", ":")) if service_port_choices else None,
                    int(allow_external_busy), per_die_hbm_mb, per_worker_memory_mb,
                ),
            )
            self._event(connection, "task-submitted", task_id=task_id,
                        data={"count": count, "allow_external_busy": allow_external_busy}, now=now)
            row = self._task_row(connection, task_id)
        return {"status": "queued", "reused": False, "task": self._serialize_task(row)}

    def add_hold(self, request: dict[str, Any], observed: dict[str, Any] | None = None) -> dict[str, Any]:
        now = self.clock()
        hold_id = require_safe_id(request.get("hold_id") or str(uuid.uuid4()), label="hold id")
        owner = require_safe_id(request.get("owner"), label="hold owner")
        devices = parse_devices(request.get("devices"), allow_none=False) or []
        not_before = parse_instant(request.get("not_before"))
        if not_before is None:
            not_before = now
        end_at = parse_instant(request.get("end_at"))
        if end_at is None:
            duration = int(request.get("duration_seconds") or 0)
            if duration < 1:
                raise CoordinationError("a hold requires end_at or duration_seconds >= 1")
            end_at = not_before + duration
        if end_at <= not_before:
            raise CoordinationError("hold end_at must be later than not_before")
        with self._transaction() as connection:
            existing = connection.execute("SELECT * FROM holds WHERE hold_id=?", (hold_id,)).fetchone()
            if existing is not None:
                expected = {
                    "owner": owner,
                    "devices": _json_devices(devices),
                    "not_before": not_before,
                    "end_at": end_at,
                }
                mismatched = {
                    key: {"existing": existing[key], "requested": value}
                    for key, value in expected.items()
                    if existing[key] != value
                }
                if mismatched:
                    raise CoordinationError(
                        f"hold id {hold_id} already exists with different ownership, devices, or window: "
                        f"{json.dumps(mismatched, sort_keys=True)}"
                    )
                return {"status": "ok", "reused": True, "hold": self._serialize_hold(existing)}
            connection.execute(
                """
                INSERT INTO holds(
                    hold_id, owner, devices, not_before, end_at, reason,
                    state, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, 'active', ?, ?)
                """,
                (hold_id, owner, _json_devices(devices), not_before, end_at, request.get("reason"), now, now),
            )
            self._event(connection, "hold-added", data={"hold_id": hold_id, "devices": devices}, now=now)
            conflicts: list[dict[str, Any]] = []
            for task in connection.execute(
                "SELECT * FROM tasks WHERE state IN ('granted','starting','active','orphaned_busy')"
            ).fetchall():
                overlap = sorted(set(devices) & set(_load_devices(task["granted_devices"])))
                if overlap:
                    conflicts.append({"task_id": task["task_id"], "devices": overlap, "state": task["state"]})
            busy = self._busy_set(observed)
            if busy:
                overlap = sorted(set(devices) & busy)
                if overlap:
                    conflicts.append({"external_busy": True, "devices": overlap})
            row = connection.execute("SELECT * FROM holds WHERE hold_id=?", (hold_id,)).fetchone()
        return {
            "status": "recorded_with_conflicts" if conflicts else "recorded",
            "hold": self._serialize_hold(row),
            "conflicts": conflicts,
        }

    def remove_hold(self, hold_id: str) -> dict[str, Any]:
        hold_id = require_safe_id(hold_id, label="hold id")
        now = self.clock()
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM holds WHERE hold_id=?", (hold_id,)).fetchone()
            if row is None:
                raise CoordinationError(f"hold not found: {hold_id}")
            connection.execute(
                "UPDATE holds SET state='cancelled', updated_at=? WHERE hold_id=?",
                (now, hold_id),
            )
            self._event(connection, "hold-cancelled", data={"hold_id": hold_id}, now=now)
            row = connection.execute("SELECT * FROM holds WHERE hold_id=?", (hold_id,)).fetchone()
        return {"status": "cancelled", "hold": self._serialize_hold(row)}

    def reserve_container_ssh(
        self, request: dict[str, Any], *, listening: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        user = require_safe_id(request.get("user"), label="user")
        container_name = require_safe_id(request.get("container_name"), label="container name")
        expected = user_container_name(user)
        if container_name != expected:
            raise CoordinationError(f"container name must be {expected}")
        port = int(request["port"])
        if not 0 <= port < 65536:
            raise CoordinationError(f"invalid TCP port: {port}")
        task_id = container_ssh_task_id(user)
        now = self.clock()
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM ports WHERE task_id=? AND kind='container_ssh'",
                (task_id,),
            ).fetchone()
            if existing is not None:
                if (port and int(existing["port"]) != port) or existing["owner"] != user:
                    raise CoordinationError(
                        f"user {user} already has SSH port {existing['port']} reserved",
                        error_code="container_ssh_port_mismatch", port=int(existing["port"]),
                    )
                port = int(existing["port"])
                reused = True
            else:
                automatic = port == 0
                if automatic:
                    if listening is None or listening.get("status") != "ok":
                        raise CoordinationError("host listening ports are unavailable",
                                                error_code="listening_unavailable")
                    live = {int(item) for item in listening.get("ports", [])}
                    allocated = self._allocated_ports(connection)
                    first, last = parse_port_range(_authority_attribute("DEFAULT_CONTAINER_SSH_PORT_RANGE"))
                    port = next((candidate for candidate in range(first, last + 1)
                                 if candidate not in live and candidate not in allocated), 0)
                    if not port:
                        raise CoordinationError("no free container SSH port",
                                                error_code="container_ssh_port_exhausted")
                self._claim_port(
                    connection,
                    port=port,
                    kind="container_ssh",
                    task_id=task_id,
                    owner=user,
                    now=now,
                    listening=listening,
                    allow_listening=not automatic,
                )
                reused = False
            self._event(
                connection,
                "container-ssh-reserved",
                task_id=task_id,
                data={"user": user, "container_name": container_name, "port": port, "reused": reused},
                now=now,
            )
            return {
                "status": "reserved",
                "reused": reused,
                "user": user,
                "container_name": container_name,
                "port": port,
                "task_id": task_id,
                "coordination_epoch": self._coordination_epoch(connection),
            }

    def release_container_ssh(self, request: dict[str, Any]) -> dict[str, Any]:
        """Release only this container owner's persistent SSH reservation."""
        user = require_safe_id(request.get("user"), label="user")
        container_name = require_safe_id(request.get("container_name"), label="container name")
        expected = user_container_name(user)
        if container_name != expected:
            raise CoordinationError(f"container name must be {expected}")
        port = int(request.get("port") or 0)
        if not 0 <= port < 65536:
            raise CoordinationError(f"invalid TCP port: {port}")
        task_id = container_ssh_task_id(user)
        now = self.clock()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM ports WHERE task_id=? AND kind='container_ssh'",
                (task_id,),
            ).fetchone()
            if row is None:
                self._event(connection, "container-ssh-release-absent", task_id=task_id,
                            data={"user": user, "container_name": container_name, "port": port}, now=now)
                return {"status": "released", "released": False, "reason": "absent",
                        "user": user, "container_name": container_name, "port": 0, "task_id": task_id}
            actual_port = int(row["port"])
            if row["owner"] != user:
                raise CoordinationError(
                    f"container SSH reservation for {user} is owned by another principal",
                    error_code="container_ssh_port_mismatch", port=actual_port,
                )
            if port and port != actual_port:
                raise CoordinationError(
                    f"user {user} has SSH port {actual_port} reserved, not {port}",
                    error_code="container_ssh_port_mismatch", port=actual_port,
                )
            connection.execute(
                "DELETE FROM ports WHERE port=? AND kind='container_ssh' AND task_id=? AND owner=?",
                (actual_port, task_id, user),
            )
            self._event(connection, "container-ssh-released", task_id=task_id,
                        data={"user": user, "container_name": container_name, "port": actual_port}, now=now)
            return {"status": "released", "released": True, "user": user,
                    "container_name": container_name, "port": actual_port, "task_id": task_id}

    def acquire(
        self,
        task_id: str,
        observed: dict[str, Any] | None,
        *,
        grant_ttl_seconds: int = DEFAULT_GRANT_TTL_SECONDS,
        listening: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        task_id = require_safe_id(task_id, label="task id")
        if grant_ttl_seconds < 1:
            raise CoordinationError("grant_ttl_seconds must be >= 1")
        now = self.clock()
        with self._transaction() as connection:
            requested_row = self._task_row(connection, task_id)
            if int(requested_row["requested_count"]) and (observed is None or observed.get("status") != "ok"
                    or (not requested_row["allow_external_busy"] and self._busy_set(observed) is None)):
                return {"status": "probe_failed", "error": (observed or {}).get("error", "NPU occupancy is unknown"), "occupancy": observed}
            changes = self._housekeep(connection, observed, now=now)
            row = self._task_row(connection, task_id)
            if row["state"] != "queued":
                return {"status": row["state"], "task": self._serialize_task(row), "gc": changes}
            if float(row["not_before"]) > now:
                return {"status": "waiting", "reason": "not_before", "task": self._serialize_task(row), "gc": changes}
            head = connection.execute(
                """
                SELECT * FROM tasks
                WHERE state='queued' AND not_before<=? AND latest_start>?
                  AND (requested_count=0)=?
                ORDER BY priority DESC, submitted_at ASC, task_id ASC
                LIMIT 1
                """,
                (now, now, int(row["requested_count"] == 0)),
            ).fetchone()
            if head is None or head["task_id"] != task_id:
                return {
                    "status": "waiting",
                    "reason": "strict_fifo",
                    "ahead_task_id": None if head is None else head["task_id"],
                    "task": self._serialize_task(row),
                    "gc": changes,
                }
            budget = None
            if int(row["requested_count"]) == 0:
                active = connection.execute(
                    "SELECT COUNT(*) FROM tasks WHERE requested_count=0 "
                    "AND state IN ('granted','starting','active','orphaned_busy')"
                ).fetchone()[0]
                if active >= self.max_compile_slots:
                    return {"status": "waiting", "reason": "compile_slots_full",
                            "max_compile_slots": self.max_compile_slots, "active_compile_slots": active,
                            "task": self._serialize_task(row), "gc": changes}
                budget = compile_slot_budget(self.max_compile_slots,
                    _authority_attribute("_cgroup_cpu_cores")(),
                    _authority_attribute("_available_memory_mb")(), row["per_worker_memory_mb"])
                if budget["jobs_per_slot"] == 0:
                    return {"status": "waiting", "reason": "compile_budget_insufficient",
                            "compile_budget": budget, "task": self._serialize_task(row), "gc": changes}
            estimated_end = now + int(row["estimated_duration_seconds"])
            available = self._available_devices(
                connection,
                observed,
                exclude_task=task_id,
                start_at=now,
                end_at=estimated_end,
                allow_external_busy=bool(row["allow_external_busy"]),
            ) if int(row["requested_count"]) else set()
            available = hbm_eligible_devices(observed, available, row["per_die_hbm_mb"])
            requested = _load_devices(row["requested_devices"])
            selected, missing = self._select_granted_devices(
                requested=requested,
                count=int(row["requested_count"]),
                available=available,
            )
            if selected is None:
                if requested:
                    return {
                        "status": "waiting",
                        "reason": "requested_devices_unavailable",
                        "unavailable_devices": missing,
                        "available_devices": sorted(available),
                        "task": self._serialize_task(row),
                        "gc": changes,
                    }
                return {
                    "status": "waiting",
                    "reason": "not_enough_devices",
                    "needed": int(row["requested_count"]),
                    "available_devices": sorted(available),
                    "task": self._serialize_task(row),
                    "gc": changes,
                }
            granted_service_port = None
            if row["requested_service_port"] is not None:
                choices = json.loads(row["service_port_choices"] or "[]")
                try:
                    granted_service_port = self._select_service_port(
                        connection,
                        choices=[int(item) for item in choices],
                        requested=int(row["requested_service_port"]),
                        task_id=task_id,
                        owner=str(row["agent_id"]),
                        now=now,
                        listening=listening or {"status": "failed", "error": "host listening ports were not probed"},
                    )
                except CoordinationError as exc:
                    return {
                        "status": "waiting",
                        "reason": "service_port_unavailable",
                        "error": str(exc),
                        "task": self._serialize_task(row),
                        "occupancy": observed,
                        "gc": changes,
                    }
            fence = self._next_fence(connection)
            deadline = now + grant_ttl_seconds
            connection.execute(
                """
                UPDATE tasks
                SET state='granted', granted_devices=?, activation_deadline=?,
                    fence_token=?, granted_service_port=?, compile_budget=?, updated_at=?, message=?
                WHERE task_id=?
                """,
                (_json_devices(selected), deadline, fence, granted_service_port,
                 json.dumps(budget) if budget else None, now, "short-lived grant issued", task_id),
            )
            self._event(
                connection,
                "task-granted",
                task_id=task_id,
                data={"devices": selected, "fence_token": fence, "deadline": utc_now_iso(deadline),
                      "service_port": granted_service_port},
                now=now,
            )
            row = self._task_row(connection, task_id)
        environment = {"ASCEND_RT_VISIBLE_DEVICES": ",".join(str(item) for item in selected)}
        if granted_service_port is not None:
            environment["MINDIE_SERVICE_PORT"] = str(granted_service_port)
        return {
            "status": "granted",
            "granted_now": True,
            "task": self._serialize_task(row),
            "environment": environment,
            "occupancy": observed,
            "gc": changes,
        }

    def preflight(
        self,
        task_id: str,
        token: int,
        observed: dict[str, Any] | None,
        *,
        start_ttl_seconds: int = DEFAULT_START_TTL_SECONDS,
    ) -> dict[str, Any]:
        task_id = require_safe_id(task_id, label="task id")
        if start_ttl_seconds < 1:
            raise CoordinationError("start_ttl_seconds must be >= 1")
        now = self.clock()
        busy = self._busy_set(observed) or set()
        visible = {int(device) for device in (observed or {}).get("devices", [])}
        with self._transaction() as connection:
            requested_row = self._task_row(connection, task_id)
            if int(requested_row["requested_count"]) and (observed is None or observed.get("status") != "ok"
                    or (not requested_row["allow_external_busy"] and self._busy_set(observed) is None)):
                return {"status": "probe_failed", "error": (observed or {}).get("error", "NPU occupancy is unknown"), "occupancy": observed}
            changes = self._housekeep(connection, observed, now=now)
            row = self._task_row(connection, task_id)
            self._check_token(row, token)
            if row["state"] != "granted":
                raise CoordinationError(f"task {task_id} is {row['state']}, expected granted")
            devices = _load_devices(row["granted_devices"])
            if row["allow_external_busy"]:
                available = self._available_devices(
                    connection, observed, exclude_task=task_id, start_at=now,
                    end_at=now + int(row["estimated_duration_seconds"]), allow_external_busy=True,
                )
                conflicts = sorted(set(devices) - set(available))
            else:
                conflicts = sorted((set(devices) & busy) | (set(devices) - visible))
            conflicts = sorted(set(conflicts) | (set(devices) - hbm_eligible_devices(
                observed, devices, row["per_die_hbm_mb"])))
            if conflicts:
                next_state = "queued" if float(row["latest_start"]) > now else "expired"
                connection.execute(
                    """
                    UPDATE tasks
                    SET state=?, granted_devices=NULL, activation_deadline=NULL,
                        fence_token=NULL, updated_at=?, message=?
                    WHERE task_id=?
                    """,
                    (next_state, now, f"preflight found unavailable devices or insufficient HBM: {conflicts}", task_id),
                )
                self._event(
                    connection,
                    "preflight-conflict",
                    task_id=task_id,
                    data={"devices": conflicts, "to": next_state},
                    now=now,
                )
                row = self._task_row(connection, task_id)
                return {
                    "status": "waiting" if next_state == "queued" else "expired",
                    "reason": "device_headroom" if row["per_die_hbm_mb"] else "external_occupancy",
                    "conflicting_devices": conflicts,
                    "task": self._serialize_task(row),
                    "gc": changes,
                }
            deadline = now + start_ttl_seconds
            connection.execute(
                "UPDATE tasks SET state='starting', activation_deadline=?, updated_at=?, message=? WHERE task_id=?",
                (deadline, now, "preflight passed; launch before activation deadline", task_id),
            )
            self._event(connection, "preflight-passed", task_id=task_id, data={"devices": devices}, now=now)
            row = self._task_row(connection, task_id)
        environment = {"ASCEND_RT_VISIBLE_DEVICES": ",".join(str(item) for item in devices)}
        if row["granted_service_port"] is not None:
            environment["MINDIE_SERVICE_PORT"] = str(int(row["granted_service_port"]))
        return {
            "status": "starting",
            "task": self._serialize_task(row),
            "environment": environment,
            "occupancy": observed,
            "gc": changes,
        }

    def activate(
        self,
        task_id: str,
        token: int,
        *,
        pid: int | None = None,
        process_guard: dict | None = None,
        prepared_supervisor: dict | None = None,
        heartbeat_ttl_seconds: int = DEFAULT_HEARTBEAT_TTL_SECONDS,
    ) -> dict[str, Any]:
        task_id = require_safe_id(task_id, label="task id")
        if heartbeat_ttl_seconds < 1:
            raise CoordinationError("heartbeat_ttl_seconds must be >= 1")
        if process_guard is not None:
            if (not isinstance(process_guard, dict)
                    or not {"marker", "boot_id"}.issubset(process_guard)
                    or set(process_guard) - {"marker", "boot_id", "retain_until_release"}
                    or not re.fullmatch(r"[0-9a-f]{32}", str(process_guard.get("marker", "")))
                    or not isinstance(process_guard.get("boot_id"), str)
                    or ("retain_until_release" in process_guard and not isinstance(process_guard["retain_until_release"], bool))):
                raise CoordinationError("invalid managed process guard")
        if prepared_supervisor is not None:
            # This exact container/boot/PID/start-time/marker proof establishes
            # the waiting supervisor's presence. Scanning every host process
            # again adds no identity evidence (and can only report unknown).
            pid = _authority_attribute("prepared_supervisor_host_pid")(prepared_supervisor, process_guard)
        elif process_guard is not None:
            if not _authority_attribute("process_guard_busy")(process_guard, completion_confirmed=True):
                raise CoordinationError("managed supervisor is no longer present")
        if pid is None or pid < 1:
            raise CoordinationError("pid must be >= 1")
        now = self.clock()
        with self._transaction() as connection:
            row = self._task_row(connection, task_id)
            self._check_token(row, token)
            if row["state"] != "starting":
                raise CoordinationError(f"task {task_id} is {row['state']}, expected starting")
            if int(row['requested_count']) == 0 and process_guard is None:
                raise CoordinationError('CPU task activation requires a valid process_guard; a PID alone cannot prove process completion')
            if row['allow_external_busy'] and (process_guard is None or process_guard.get('retain_until_release') is not True):
                raise CoordinationError('shared NPU activation requires a managed process_guard retained until confirmed completion')
            if row["activation_deadline"] is not None and float(row["activation_deadline"]) <= now:
                raise CoordinationError(f"activation deadline elapsed for task {task_id}")
            expected_end = now + int(row["estimated_duration_seconds"])
            heartbeat_deadline = now + heartbeat_ttl_seconds
            connection.execute(
                """
                UPDATE tasks
                SET state='active', pid=?, started_at=?, expected_end=?,
                    heartbeat_at=?, heartbeat_deadline=?, process_guard=?, activation_deadline=NULL,
                    updated_at=?, message=?
                WHERE task_id=?
                """,
                (pid, now, expected_end, now, heartbeat_deadline,
                 json.dumps(process_guard) if process_guard else None, now, "task reported active", task_id),
            )
            self._event(connection, "task-activated", task_id=task_id, data={"pid": pid}, now=now)
            row = self._task_row(connection, task_id)
        return {"status": "active", "task": self._serialize_task(row)}

    def heartbeat(
        self,
        task_id: str,
        token: int,
        *,
        heartbeat_ttl_seconds: int = DEFAULT_HEARTBEAT_TTL_SECONDS,
    ) -> dict[str, Any]:
        task_id = require_safe_id(task_id, label="task id")
        if heartbeat_ttl_seconds < 1:
            raise CoordinationError("heartbeat_ttl_seconds must be >= 1")
        now = self.clock()
        with self._transaction() as connection:
            row = self._task_row(connection, task_id)
            self._check_token(row, token)
            if row["state"] not in {"active", "orphaned_busy", "granted", "starting"}:
                raise CoordinationError(f"task {task_id} is {row['state']}, expected active or orphaned_busy")
            if row["state"] in {"granted", "starting"}:
                connection.execute(
                    """
                    UPDATE tasks
                    SET activation_deadline=?, heartbeat_at=?, heartbeat_deadline=?,
                        updated_at=?, message=?
                    WHERE task_id=?
                    """,
                    (now + heartbeat_ttl_seconds, now, now + heartbeat_ttl_seconds, now, "heartbeat received", task_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE tasks
                    SET state='active', heartbeat_at=?, heartbeat_deadline=?,
                        updated_at=?, message=?
                    WHERE task_id=?
                    """,
                    (now, now + heartbeat_ttl_seconds, now, "heartbeat received", task_id),
                )
            row = self._task_row(connection, task_id)
        return {"status": row["state"], "task": self._serialize_task(row)}

    def release(self, task_id: str, token: int, observed: dict[str, Any] | None, *,
                completion_confirmed: bool = False, listening: dict[str, Any] | None = None) -> dict[str, Any]:
        task_id = require_safe_id(task_id, label="task id")
        now = self.clock()
        busy = self._busy_set(observed)
        visible = (
            {int(device) for device in observed.get("devices", [])}
            if observed is not None and observed.get("status") == "ok"
            else None
        )
        with self._transaction() as connection:
            row = self._task_row(connection, task_id)
            self._check_token(row, token)
            devices = set(_load_devices(row["granted_devices"]))
            managed_completion = _managed_completion(row, completion_confirmed)
            conflicts = [] if managed_completion else (
                sorted((set() if row["allow_external_busy"] else devices & busy) | (devices - visible))
                if busy is not None and visible is not None else sorted(devices))
            busy_ports = self._service_ports_busy(connection, task_id, listening)
            if ((devices and not managed_completion and (busy is None or visible is None or conflicts)) or busy_ports or unguarded_cpu_process(row)
                    or unguarded_shared_process(row)
                    or _authority_attribute("process_guard_busy")(row["process_guard"], completion_confirmed=completion_confirmed)):
                connection.execute(
                    "UPDATE tasks SET state='orphaned_busy', updated_at=?, message=? WHERE task_id=?",
                    (now, 'CPU task has no process guard; process completion is unknown' if unguarded_cpu_process(row)
                     else "release requested but processes or hardware remained busy or unknown", task_id),
                )
                self._event(connection, "release-deferred", task_id=task_id,
                            data={"devices": conflicts, "service_ports": busy_ports}, now=now)
                row = self._task_row(connection, task_id)
                return {
                    "status": "orphaned_busy",
                    "conflicting_devices": conflicts,
                    "conflicting_service_ports": busy_ports,
                    "task": self._serialize_task(row),
                    "occupancy": observed,
                }
            self._clear_service_ports(connection, task_id)
            connection.execute(
                "UPDATE tasks SET state='released', process_guard=NULL, updated_at=?, message=? WHERE task_id=?",
                (now, "owned process and ports are clear; managed lease released" if managed_completion
                 else "requested resources are free; cooperative lease released", task_id),
            )
            self._event(connection, "task-released", task_id=task_id, now=now)
            row = self._task_row(connection, task_id)
        return {"status": "released", "task": self._serialize_task(row), "occupancy": observed}

    def cancel(self, task_id: str, observed: dict[str, Any] | None = None,
               listening: dict[str, Any] | None = None) -> dict[str, Any]:
        task_id = require_safe_id(task_id, label="task id")
        now = self.clock()
        busy = self._busy_set(observed)
        visible = (
            {int(device) for device in observed.get("devices", [])}
            if observed is not None and observed.get("status") == "ok"
            else None
        )
        with self._transaction() as connection:
            row = self._task_row(connection, task_id)
            if row["state"] in {"released", "expired", "cancelled"}:
                return {"status": row["state"], "task": self._serialize_task(row)}
            devices = set(_load_devices(row["granted_devices"]))
            busy_ports = self._service_ports_busy(connection, task_id, listening)
            still_busy = (
                (
                    bool(devices)
                    and (
                        busy is None
                        or visible is None
                        or not devices.issubset(visible)
                        or (bool(devices & busy) and not row["allow_external_busy"])
                    )
                )
                or bool(busy_ports)
                or unguarded_cpu_process(row)
                or unguarded_shared_process(row)
                or _authority_attribute("process_guard_busy")(row["process_guard"])
            )
            next_state = "orphaned_busy" if still_busy else "cancelled"
            if next_state == "cancelled":
                self._clear_service_ports(connection, task_id)
            connection.execute(
                "UPDATE tasks SET state=?, process_guard=CASE WHEN ?='cancelled' THEN NULL ELSE process_guard END, updated_at=?, message=? WHERE task_id=?",
                (next_state, next_state, now, "cancel requested", task_id),
            )
            self._event(connection, "task-cancelled", task_id=task_id, data={"to": next_state}, now=now)
            row = self._task_row(connection, task_id)
        return {"status": next_state, "task": self._serialize_task(row)}
