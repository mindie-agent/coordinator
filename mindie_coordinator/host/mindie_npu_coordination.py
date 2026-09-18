#!/usr/bin/env python3
"""Cooperative host-local NPU task coordination.

The coordinator is deliberately advisory.  It gives independent MindIE agents a
shared queue and lease ledger on one host, but it does not prevent an operator
or a non-participating process from using an NPU. Observed occupancy blocks
admission unless a request explicitly allows external use of one named device.
Managed leases, holds, owned processes and service ports remain protected.

The module is stdlib-only because the agent-facing wrapper sends this source to
the bare-metal host and executes it there.  State defaults to
``/tmp/mindie-npu-coordinator/v1`` (override with ``MINDIE_NPU_COORDINATOR_STATE_DIR``
or ``request["state_dir"]``) and is expected to disappear after a host or
``/tmp`` reset; a missing database simply starts a new coordination epoch.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
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

SCHEMA_VERSION = 6
_DIAGNOSTIC_PHASES = ContextVar("host_diagnostic_phases", default=None)
_DIAGNOSTIC_PROCESS = uuid.uuid4().hex


def _diagnostic_elapsed(name, started):
    phases = _DIAGNOSTIC_PHASES.get()
    if phases is not None:
        phases[name] = phases.get(name, 0.0) + (time.monotonic() - started) * 1000
DEFAULT_STATE_DIR = "/tmp/mindie-npu-coordinator/v1"
HOST_STATE_DIR_ENV = "MINDIE_NPU_COORDINATOR_STATE_DIR"
DEFAULT_CONTAINER_SSH_PORT_RANGE = "46000:46999"
DEFAULT_SERVING_PORT_RANGE = "30000:45999"


def resolve_host_state_dir(explicit: str | Path | None = None) -> str:
    """Return the host coordination state directory.

    Precedence: explicit argument, then ``MINDIE_NPU_COORDINATOR_STATE_DIR``,
    then ``DEFAULT_STATE_DIR``.
    """
    if explicit:
        return str(explicit)
    return os.environ.get(HOST_STATE_DIR_ENV) or DEFAULT_STATE_DIR
DEFAULT_QUEUE_TTL_SECONDS = 3600
DEFAULT_GRANT_TTL_SECONDS = 60
DEFAULT_START_TTL_SECONDS = 60
DEFAULT_HEARTBEAT_TTL_SECONDS = 120
DEFAULT_ESTIMATED_DURATION_SECONDS = 3600
HBM_BUSY_THRESHOLD_MB = 4096
JOB_TOKEN_ENV = "REMOTE_DEV_JOB_TOKEN"

TASK_STATES = {
    "queued",
    "granted",
    "starting",
    "active",
    "orphaned_busy",
    "released",
    "expired",
    "cancelled",
}
RESERVING_TASK_STATES = {"granted", "starting", "active", "orphaned_busy"}
PORT_KINDS = {"container_ssh", "service"}
CONTAINER_SSH_TASK_PREFIX = "ssh."
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")


class CoordinationError(RuntimeError):
    """Raised for deterministic coordinator input or state failures."""

    def __init__(self, message: str, *, error_code: str | None = None, port: int | None = None):
        super().__init__(message)
        self.error_code = error_code
        self.port = port


def prepared_supervisor_host_pid(prepared: dict, process_guard: dict | None) -> int:
    """Resolve a container receipt immediately before fenced activation."""
    receipt = prepared["receipt"]
    if (not isinstance(process_guard, dict) or process_guard != receipt.get("process_guard")
            or process_guard.get("boot_id") != receipt.get("boot_id")
            or not re.fullmatch(r"[0-9a-f]{32}", str(process_guard.get("marker", "")))):
        raise CoordinationError("prepared receipt and process guard disagree")
    container_name = prepared["container_name"]
    container_id = prepared["container_id"]
    if not isinstance(container_name, str) or not container_name or not isinstance(container_id, str) or not container_id:
        raise CoordinationError("prepared activation requires the expected container identity")
    info = json.loads(subprocess.check_output(
        ["docker", "inspect", "--format", "{{json .}}", container_name], text=True, encoding="utf-8"))
    if info["Id"] != container_id:
        raise CoordinationError("container identity changed before activation")
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != receipt["boot_id"]:
        raise CoordinationError("host boot identity changed")
    # Use the inspected immutable ID if the name is reassigned during lookup.
    rows = subprocess.check_output(["docker", "top", container_id, "-eo", "pid"],
                                   text=True, encoding="utf-8").splitlines()[1:]
    marker = (JOB_TOKEN_ENV + "=" + process_guard["marker"]).encode()
    matches = []
    for row in rows:
        pid = int(row.strip())
        try:
            process = Path(f"/proc/{pid}")
            fields = (process / "stat").read_text().rsplit(") ", 1)[1].split()
            status = (process / "status").read_text().splitlines()
            namespace = next(line.split()[1:] for line in status if line.startswith("NSpid:"))
            if (int(namespace[-1]) == receipt["pid"] and fields[19] == receipt["start_ticks"]
                    and fields[0] != "Z" and marker in (process / "environ").read_bytes().split(b"\0")):
                matches.append(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
    if len(matches) != 1:
        raise CoordinationError("cannot identify a unique host PID for the waiting supervisor")
    return matches[0]


def process_guard_busy(value: str | dict | None, *, completion_confirmed: bool = False) -> bool:
    """Retain a managed lease while its process family can still use devices.

    Empty NPU occupancy during CPU initialization is not proof of completion.
    Subreaper jobs additionally require the manager's drained-process receipt;
    a crashed supervisor may have lost clean-environment descendants. A marker
    scan alone can never release these leases. Unreadable state retains them.
    """
    if not value:
        return False
    try:
        guard = json.loads(value) if isinstance(value, str) else value
        if not re.fullmatch(r"[0-9a-f]{32}", guard["marker"]):
            return True
        if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != guard["boot_id"]:
            return True
        marker = (JOB_TOKEN_ENV + "=" + guard["marker"]).encode()
        unknown = False
        for process in Path("/proc").iterdir():
            if not process.name.isdigit():
                continue
            try:
                state = (process / "stat").read_text().rsplit(") ", 1)[1].split()[0]
                if state != "Z" and marker in (process / "environ").read_bytes().split(b"\0"):
                    return True
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError:
                unknown = True
        return unknown or (guard.get("retain_until_release") is True and not completion_confirmed)
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return True


def unguarded_cpu_process(row: sqlite3.Row) -> bool:
    """Legacy pid-only CPU activations have no reliable completion evidence."""
    return (int(row['requested_count']) == 0 and not row['process_guard']
            and (row['state'] == 'active' or (row['state'] == 'orphaned_busy' and row['pid'] is not None)))


def unguarded_shared_process(row: sqlite3.Row) -> bool:
    """External occupancy cannot prove that an activated shared job drained."""
    return (bool(row['allow_external_busy']) and not row['process_guard']
            and (row['state'] == 'active' or (row['state'] == 'orphaned_busy' and row['started_at'] is not None)))


def _managed_shared_completion(row, completion_confirmed):
    """A drained managed shared lease releases ownership, not whole-card idleness.

    This selects the hardware-independent path, not proof of process completion:
    release still checks the stored guard's boot/marker and live family, ports
    and fence. Malformed or legacy guards retain the original hardware path.
    """
    if completion_confirmed is not True or not row['allow_external_busy'] or row['state'] not in {'active', 'orphaned_busy'}:
        return False
    try:
        guard = json.loads(row['process_guard']) if row['process_guard'] else None
        return (isinstance(guard, dict) and guard.get('retain_until_release') is True
                and isinstance(guard.get('marker'), str) and re.fullmatch(r'[0-9a-f]{32}', guard['marker']) is not None
                and isinstance(guard.get('boot_id'), str) and bool(guard['boot_id']))
    except (ValueError, TypeError):
        return False


def utc_now_iso(epoch: float | None = None) -> str:
    value = time.time() if epoch is None else epoch
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_instant(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = value.strip()
    if not text:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise CoordinationError(f"invalid timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise CoordinationError(f"timestamp must include a timezone: {value!r}")
    return parsed.timestamp()


def require_safe_id(value: str | None, *, label: str) -> str:
    if not isinstance(value, str) or not SAFE_ID_RE.fullmatch(value):
        raise CoordinationError(
            f"invalid {label}: use 3-128 characters from A-Z a-z 0-9 _ . : -"
        )
    return value


def user_container_name(user: str) -> str:
    return "mindie-" + require_safe_id(user, label="user")


def container_ssh_task_id(user: str) -> str:
    return CONTAINER_SSH_TASK_PREFIX + require_safe_id(user, label="user")


def parse_devices(value: Any, *, allow_none: bool = True) -> list[int] | None:
    if value is None:
        if allow_none:
            return None
        raise CoordinationError("devices are required")
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, list):
        raw_items = value
    else:
        raise CoordinationError("devices must be a comma-separated string or list")
    devices: list[int] = []
    seen: set[int] = set()
    for raw in raw_items:
        token = str(raw).strip()
        if not token:
            raise CoordinationError("devices contains an empty device id")
        try:
            device = int(token, 10)
        except ValueError as exc:
            raise CoordinationError(f"devices contains a non-integer id: {token!r}") from exc
        if device < 0:
            raise CoordinationError(f"devices contains a negative id: {device}")
        if device in seen:
            raise CoordinationError(f"devices contains a duplicate id: {device}")
        seen.add(device)
        devices.append(device)
    if not devices:
        raise CoordinationError("devices must not be empty")
    return sorted(devices)


def _json_devices(value: list[int] | None) -> str | None:
    return None if value is None else json.dumps(sorted(value), separators=(",", ":"))


def _load_devices(value: str | None) -> list[int]:
    if not value:
        return []
    loaded = json.loads(value)
    return [int(item) for item in loaded]


def parse_port_range(value: str) -> tuple[int, int]:
    start_s, sep, end_s = str(value).partition(":")
    if not sep:
        raise CoordinationError(f"port range must be START:END, got {value!r}")
    try:
        start = int(start_s)
        end = int(end_s)
    except ValueError as exc:
        raise CoordinationError(f"invalid port range: {value!r}") from exc
    if start <= 0 or end <= 0 or start > end or end > 65535:
        raise CoordinationError(f"invalid port range: {value!r}")
    return start, end


def _parse_listening_port_text(output: str) -> set[int]:
    ports: set[int] = set()
    for line in output.splitlines():
        match = re.search(r"[:.](\d+)$", line.strip())
        if match:
            ports.add(int(match.group(1)))
    return ports


def probe_listening_ports() -> dict[str, Any]:
    """Return host TCP listen ports from ss, then netstat."""
    commands = (
        ["ss", "-ltnH"],
        ["netstat", "-ltn"],
    )
    last_error = "ss and netstat are unavailable"
    for command in commands:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            last_error = str(exc)
            continue
        if result.returncode != 0:
            last_error = (result.stderr or result.stdout or f"{command[0]} failed")[-2000:]
            continue
        stdout = result.stdout or ""
        if command[0] == "netstat":
            addresses = []
            for line in stdout.splitlines()[2:]:
                fields = line.split()
                if len(fields) >= 4:
                    addresses.append(fields[3])
            stdout = "\n".join(addresses)
        else:
            addresses = []
            for line in stdout.splitlines():
                fields = line.split()
                if len(fields) >= 4:
                    addresses.append(fields[3])
            stdout = "\n".join(addresses)
        return {
            "status": "ok",
            "ports": sorted(_parse_listening_port_text(stdout)),
            "source": command[0],
        }
    return {"status": "failed", "error": last_error, "ports": []}


def probe_named_container(name: str) -> dict[str, Any]:
    """Report whether a docker container name exists and is running."""
    if not name:
        return {"status": "ok", "exists": False, "running": False, "name": name}
    try:
        listed = subprocess.run(
            ["docker", "container", "ls", "-a", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "failed", "error": str(exc), "exists": None, "running": None, "name": name}
    if listed.returncode != 0:
        return {
            "status": "failed",
            "error": "host Docker state unavailable",
            "exists": None,
            "running": None,
            "name": name,
            "stderr": (listed.stderr or listed.stdout)[-2000:],
        }
    exists = name in listed.stdout.splitlines()
    running = False
    if exists:
        try:
            inspected = subprocess.run(
                ["docker", "inspect", "--type", "container", name],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"status": "failed", "error": str(exc), "exists": True, "running": None, "name": name}
        if inspected.returncode != 0:
            return {
                "status": "failed",
                "error": "docker inspect failed",
                "exists": True,
                "running": None,
                "name": name,
            }
        try:
            payload = json.loads(inspected.stdout)
            running = bool(payload[0]["State"]["Running"])
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            return {"status": "failed", "error": str(exc), "exists": True, "running": None, "name": name}
    return {"status": "ok", "exists": exists, "running": running, "name": name}


def _parse_npu_smi_device_table(
    output: str,
) -> tuple[set[int], dict[int, dict[str, int]], dict[tuple[int, int], int]]:
    """Visible Phy-IDs, HBM used/total, and (NPU, Chip) -> Phy-ID mapping."""
    dev_ids: set[int] = set()
    header_ids: set[int] = set()
    hbm: dict[int, dict[str, int]] = {}
    chip_devices: dict[tuple[int, int], int] = {}
    current_npu: int | None = None
    for line in output.splitlines():
        if "0000:" in line:
            chip = re.match(r"\|\s*(\d+)\s+(\d+)\s+\|.*0000:", line)
            if chip:
                dev_id = int(chip.group(2))
                if current_npu is not None:
                    chip_devices[(current_npu, int(chip.group(1)))] = dev_id
            elif current_npu is not None:
                dev_id = current_npu
            else:
                continue
            dev_ids.add(dev_id)
            pairs = re.findall(r"(\d+)\s*/\s*(\d+)", line)
            if len(pairs) >= 2:
                hbm[dev_id] = {
                    "used_mb": int(pairs[-1][0]),
                    "total_mb": int(pairs[-1][1]),
                }
            continue
        header = re.match(r"\|\s*(\d+)\s+\d*\w+\d+\w*\s+\|", line)
        if header:
            current_npu = int(header.group(1))
            header_ids.add(current_npu)
    if not dev_ids:
        dev_ids.update(header_ids)
    return dev_ids, hbm, chip_devices


def parse_npu_smi_hbm(output: str) -> dict[int, dict[str, int]]:
    """Physical-device HBM used/total from npu-smi info text.

    Does not require a process table; occupancy failure still returns HBM.
    """
    _dev_ids, hbm, _chip_devices = _parse_npu_smi_device_table(output)
    return dict(sorted(hbm.items()))


def parse_npu_smi_info(output: str) -> dict[str, Any]:
    """Parse visible devices plus process/HBM occupancy from common layouts."""
    dev_ids, hbm, chip_devices = _parse_npu_smi_device_table(output)
    lines = output.splitlines()

    process_busy: dict[int, list[dict[str, Any]]] = {}
    in_process_table = False
    process_error = None
    for line in lines:
        if "Process name" in line or "Process memory" in line:
            in_process_table = True
            continue
        if in_process_table and "No running processes" in line:
            continue
        if in_process_table and line.startswith("|"):
            columns = [column.strip() for column in line.split("|")[1:-1]]
            # A3: | NPU Chip | PID | Process name | Memory |. Phy-ID from
            # the device table is the allocation identity, not the NPU index.
            if len(columns) >= 3 and re.fullmatch(r"\d+(?:\s+\d+)?", columns[0]):
                identity = [int(value) for value in columns[0].split()]
                if len(identity) == 2:
                    device = chip_devices.get(tuple(identity))
                elif chip_devices:
                    # A parsed (NPU, Chip) -> Phy-ID table means a multi-chip
                    # layout. A lone column is then an NPU index, not a phy-id;
                    # guessing would attach the process to the wrong device and
                    # report the real one as free. Fail closed instead.
                    device = None
                else:
                    device = identity[0]
                if device not in dev_ids or not columns[1].isdigit() or not columns[2]:
                    process_error = "npu-smi process row has an unknown device or invalid PID/name"
                    break
                process_busy.setdefault(device, []).append(
                    {"kind": "process", "pid": int(columns[1]), "name": columns[2]}
                )
                continue
            # Preserve the older flat device/chip/PID/owner/name layout.
            match = re.match(r"\|\s*(\d+)\s+\S+\s+(\d+)\s+(\S+)\s+(\S+)", line)
            if match:
                device = int(match.group(1))
                if device not in dev_ids:
                    process_error = "npu-smi process row has an unknown device"
                    break
                process_busy.setdefault(device, []).append(
                    {
                        "kind": "process",
                        "pid": int(match.group(2)),
                        "owner": match.group(3),
                        "name": match.group(4),
                    }
                )
            elif any(columns):
                process_error = "npu-smi process row could not be parsed"
                break

    if process_error or not in_process_table:
        return {"status": "failed", "error": process_error or "npu-smi process table is missing",
                "devices": sorted(dev_ids), "busy": {}, "free": []}

    busy: dict[int, list[dict[str, Any]]] = {}
    for device in sorted(dev_ids):
        reasons = list(process_busy.get(device, []))
        usage = hbm.get(device)
        if usage and usage["used_mb"] >= HBM_BUSY_THRESHOLD_MB and not reasons:
            reasons.append(
                {
                    "kind": "hbm_threshold",
                    "hbm_used_mb": usage["used_mb"],
                    "threshold_mb": HBM_BUSY_THRESHOLD_MB,
                }
            )
        if reasons:
            busy[device] = reasons
    return {
        "status": "ok",
        "collected_at": utc_now_iso(),
        "devices": sorted(dev_ids),
        "busy": {str(key): value for key, value in sorted(busy.items())},
        "free": sorted(device for device in dev_ids if device not in busy),
        "hbm": {str(key): value for key, value in sorted(hbm.items())},
        "hbm_busy_threshold_mb": HBM_BUSY_THRESHOLD_MB,
    }


def probe_npu_device(device: int) -> dict[str, Any]:
    """Fresh physical-device visibility without querying unrelated occupancy."""
    failed = {"status": "failed", "error": "physical device mapping is unknown",
              "devices": [], "busy": None, "free": []}
    try:
        result = subprocess.run(
            ["npu-smi", "info", "-t", "phyid-remap", "-p", str(device)],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if result.returncode != 0:
            return failed
        fields = {}
        for line in result.stdout.splitlines():
            match = re.fullmatch(r"\s*(Chip Physical ID|Chip Logic ID|NPU ID|Chip ID)\s*:\s*(\d+)\s*", line)
            if not match or match[1] in fields:
                if line.strip():
                    return failed
                continue
            fields[match[1]] = int(match[2])
        if (set(fields) != {"Chip Physical ID", "Chip Logic ID", "NPU ID", "Chip ID"}
                or fields["Chip Physical ID"] != device):
            return failed
        # None deliberately means unobserved: this must never become evidence
        # that another task's hardware is free during global housekeeping.
        return {"status": "ok", "collected_at": utc_now_iso(), "devices": [device],
                "busy": None, "free": []}
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return failed


def probe_npu_occupancy() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "failed", "error": str(exc), "devices": [], "busy": {}, "free": []}
    if result.returncode != 0:
        return {
            "status": "failed",
            "error": "npu-smi info failed",
            "returncode": result.returncode,
            "stderr": (result.stderr or result.stdout)[-2000:],
            "devices": [],
            "busy": {},
            "free": [],
        }
    parsed = parse_npu_smi_info(result.stdout)
    if not parsed["devices"]:
        return {
            "status": "failed",
            "error": "npu-smi output did not contain any parseable devices",
            "devices": [],
            "busy": {},
            "free": [],
        }
    return parsed


class NpuCoordinator:
    """SQLite-backed cooperative queue for one bare-metal host."""

    def __init__(
        self,
        state_dir: str | Path = DEFAULT_STATE_DIR,
        *,
        clock: Callable[[], float] = time.time,
        expected_epoch: str | None = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.db_path = self.state_dir / "coordinator.sqlite3"
        self.clock = clock
        self.expected_epoch = expected_epoch
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    agent_alias TEXT,
                    session_id TEXT,
                    container_name TEXT,
                    requested_count INTEGER NOT NULL,
                    requested_devices TEXT,
                    allow_external_busy INTEGER NOT NULL DEFAULT 0,
                    not_before REAL NOT NULL,
                    latest_start REAL NOT NULL,
                    estimated_duration_seconds INTEGER NOT NULL,
                    preemptible INTEGER NOT NULL DEFAULT 0,
                    priority INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL,
                    submitted_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    granted_devices TEXT,
                    activation_deadline REAL,
                    fence_token INTEGER,
                    started_at REAL,
                    expected_end REAL,
                    heartbeat_at REAL,
                    heartbeat_deadline REAL,
                    pid INTEGER,
                    message TEXT,
                    requested_service_port INTEGER,
                    service_port_choices TEXT,
                    granted_service_port INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_queue
                    ON tasks(state, priority DESC, submitted_at ASC);
                CREATE TABLE IF NOT EXISTS holds (
                    hold_id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    devices TEXT NOT NULL,
                    not_before REAL NOT NULL,
                    end_at REAL NOT NULL,
                    reason TEXT,
                    state TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_holds_window
                    ON holds(state, not_before, end_at);
                CREATE TABLE IF NOT EXISTS ports (
                    port INTEGER PRIMARY KEY,
                    kind TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_ports_task
                    ON ports(task_id);
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    at REAL NOT NULL,
                    kind TEXT NOT NULL,
                    task_id TEXT,
                    data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS message_peers (
                    user TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    seen_at REAL NOT NULL,
                    PRIMARY KEY(user, session_id)
                );
                CREATE TABLE IF NOT EXISTS messages (
                    cursor INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT NOT NULL UNIQUE,
                    sender TEXT NOT NULL,
                    sender_session TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    recipient_session TEXT NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_recipient
                    ON messages(recipient, recipient_session, cursor);
                """
            )
            task_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(tasks)")
            }
            if "agent_alias" not in task_columns:
                try:
                    connection.execute("ALTER TABLE tasks ADD COLUMN agent_alias TEXT")
                except sqlite3.OperationalError as exc:
                    if "duplicate column" not in str(exc).lower():
                        raise
            if "process_guard" not in task_columns:
                try:
                    connection.execute("ALTER TABLE tasks ADD COLUMN process_guard TEXT")
                except sqlite3.OperationalError as exc:
                    if "duplicate column" not in str(exc).lower():
                        raise
            for column, decl in (
                ("allow_external_busy", "INTEGER NOT NULL DEFAULT 0"),
                ("requested_service_port", "INTEGER"),
                ("service_port_choices", "TEXT"),
                ("granted_service_port", "INTEGER"),
            ):
                if column not in task_columns:
                    try:
                        connection.execute(f"ALTER TABLE tasks ADD COLUMN {column} {decl}")
                    except sqlite3.OperationalError as exc:
                        if "duplicate column" not in str(exc).lower():
                            raise
            # Older clients send the host protocol source with each request.
            # They do not know about process guards. Do not let their generic
            # "hardware free" UPDATE release a guarded CPU-initializing job.
            connection.execute("""
                CREATE TRIGGER IF NOT EXISTS protect_managed_process_release
                BEFORE UPDATE OF state ON tasks
                WHEN OLD.process_guard IS NOT NULL AND NEW.process_guard IS NOT NULL
                     AND NEW.state IN ('released', 'cancelled', 'expired')
                BEGIN
                    SELECT RAISE(ABORT, 'managed process guard must be verified before release');
                END
            """)
            connection.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            connection.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'",
                (str(SCHEMA_VERSION),),
            )
            connection.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('coordination_epoch', ?)",
                (str(uuid.uuid4()),),
            )
            connection.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('next_fence', '0')")
        finally:
            connection.close()

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        started = time.monotonic()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            _diagnostic_elapsed("state_lock_wait_ms", started)
            if self.expected_epoch is not None:
                epoch = connection.execute("SELECT value FROM meta WHERE key='coordination_epoch'").fetchone()
                if epoch is None or epoch["value"] != self.expected_epoch:
                    raise CoordinationError("coordination epoch changed; reconcile ownership before retrying")
            yield connection
            connection.execute("COMMIT")
        except Exception:
            with contextlib.suppress(sqlite3.Error):
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _event(
        self,
        connection: sqlite3.Connection,
        kind: str,
        *,
        task_id: str | None = None,
        data: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> None:
        connection.execute(
            "INSERT INTO events(at, kind, task_id, data) VALUES(?, ?, ?, ?)",
            (self.clock() if now is None else now, kind, task_id, json.dumps(data or {}, sort_keys=True)),
        )

    def _next_fence(self, connection: sqlite3.Connection) -> int:
        row = connection.execute("SELECT value FROM meta WHERE key='next_fence'").fetchone()
        value = int(row["value"]) + 1
        connection.execute("UPDATE meta SET value=? WHERE key='next_fence'", (str(value),))
        return value

    @staticmethod
    def _busy_set(observed: dict[str, Any] | None) -> set[int] | None:
        if observed is None or observed.get("status") != "ok" or observed.get("busy", {}) is None:
            return None
        return {int(key) for key in observed.get("busy", {})}

    @staticmethod
    def _serialize_task(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        payload = dict(row)
        payload["requested_devices"] = _load_devices(payload.get("requested_devices")) or None
        payload["granted_devices"] = _load_devices(payload.get("granted_devices"))
        payload["preemptible"] = bool(payload.get("preemptible"))
        payload["allow_external_busy"] = bool(payload.get("allow_external_busy"))
        payload["process_guarded"] = bool(payload.pop("process_guard", None))
        for key in (
            "not_before",
            "latest_start",
            "submitted_at",
            "updated_at",
            "activation_deadline",
            "started_at",
            "expected_end",
            "heartbeat_at",
            "heartbeat_deadline",
        ):
            if payload.get(key) is not None:
                payload[key] = utc_now_iso(float(payload[key]))
        return payload

    @staticmethod
    def _serialize_hold(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        payload = dict(row)
        payload["devices"] = _load_devices(payload.get("devices"))
        for key in ("not_before", "end_at", "created_at", "updated_at"):
            payload[key] = utc_now_iso(float(payload[key]))
        return payload

    def _task_row(self, connection: sqlite3.Connection, task_id: str) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            raise CoordinationError(f"task not found: {task_id}")
        return row

    def _check_token(self, row: sqlite3.Row, token: int) -> None:
        if row["fence_token"] is None or int(row["fence_token"]) != int(token):
            raise CoordinationError(f"stale or invalid fencing token for task {row['task_id']}")

    @staticmethod
    def _message_actor(request: dict[str, Any]) -> tuple[str, str]:
        user = request.get("user")
        if not isinstance(user, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", user):
            raise CoordinationError("invalid message user")
        return user, require_safe_id(request.get("session_id"), label="message session")

    def message(self, request: dict[str, Any]) -> dict[str, Any]:
        """Store plain coordination text. This never changes a task or lease.

        Identity is supplied by the cooperating client, as with the host queue;
        shared root access is not an adversarial authentication boundary.
        """
        user, session_id = self._message_actor(request)
        text = request.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise CoordinationError("message must contain 1..4000 characters")
        message_id = require_safe_id(request.get("message_id"), label="message id")
        with self._transaction() as db:
            previous = db.execute("SELECT * FROM messages WHERE message_id=?", (message_id,)).fetchone()
            if previous is not None:
                value = json.loads(previous["data"])
                expected = (user, session_id, text, request.get("reply_to"),
                            request.get("recipient"), request.get("recipient_session"))
                original = (value["sender"], value["sender_session"], value["text"], value.get("reply_to"),
                            value.get("requested_recipient"), value.get("requested_session"))
                if expected != original:
                    raise CoordinationError("message id was already used for different content")
                return {"status": "sent", "message": {"cursor": previous["cursor"], **value}}
            reply_to = request.get("reply_to")
            if reply_to:
                original = db.execute("SELECT * FROM messages WHERE message_id=?", (reply_to,)).fetchone()
                if original is None or (original["recipient"], original["recipient_session"]) != (user, session_id):
                    raise CoordinationError("reply does not belong to this task")
                recipient, recipient_session = original["sender"], original["sender_session"]
                thread_id = json.loads(original["data"])["thread_id"]
            else:
                recipient, recipient_session = self._message_actor({
                    "user": request.get("recipient"), "session_id": request.get("recipient_session")})
                if db.execute("SELECT 1 FROM message_peers WHERE user=? AND session_id=?",
                              (recipient, recipient_session)).fetchone() is None:
                    raise CoordinationError("recipient has not used this host; use a current coordination reference")
                thread_id = message_id
            value = {"message_id": message_id, "thread_id": thread_id,
                     "kind": "coordination-reply" if reply_to else "coordination-message",
                     "sender": user, "sender_session": session_id,
                     "recipient": recipient, "recipient_session": recipient_session,
                     "requested_recipient": request.get("recipient"),
                     "requested_session": request.get("recipient_session"),
                     "reply_to": reply_to, "text": text, "at": utc_now_iso(self.clock())}
            cursor = db.execute("INSERT INTO messages(message_id,sender,sender_session,recipient,recipient_session,data) "
                                "VALUES(?,?,?,?,?,?)", (message_id, user, session_id, recipient,
                                                       recipient_session, json.dumps(value))).lastrowid
            return {"status": "sent", "message": {"cursor": cursor, **value}}

    def message_events(self, request: dict[str, Any]) -> dict[str, Any]:
        """Register this task's return address and read its persistent inbox.

        Cursor belongs to this mailbox epoch. Peer timestamps are observations,
        never evidence that a process ended or that devices can be reclaimed.
        """
        user, session_id = self._message_actor(request)
        after = request.get("after", 0)
        limit = request.get("limit", 2)
        if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 20:
            raise CoordinationError("invalid message cursor or limit")
        with self._transaction() as db:
            epoch = db.execute("SELECT value FROM meta WHERE key='coordination_epoch'").fetchone()[0]
            if request.get("mailbox_epoch") != epoch:
                after = 0
            db.execute("INSERT INTO message_peers(user,session_id,seen_at) VALUES(?,?,?) "
                       "ON CONFLICT(user,session_id) DO UPDATE SET seen_at=excluded.seen_at",
                       (user, session_id, self.clock()))
            rows = db.execute("SELECT cursor,data FROM messages WHERE recipient=? AND recipient_session=? "
                              "AND cursor>? ORDER BY cursor LIMIT ?", (user, session_id, after, limit)).fetchall()
            peers = []
            if request.get("include_peers"):
                peers = [dict(row) for row in db.execute(
                    "SELECT user,session_id,seen_at FROM message_peers WHERE NOT (user=? AND session_id=?) "
                    "ORDER BY seen_at DESC LIMIT 5", (user, session_id))]
            return {"status": "ok", "mailbox_epoch": epoch,
                    "cursor": rows[-1]["cursor"] if rows else after,
                    "events": [{"cursor": row["cursor"], **json.loads(row["data"])} for row in rows],
                    "peers": peers}

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
                    allow_external_busy
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?)
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
                    int(allow_external_busy),
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

    def _housekeep(
        self,
        connection: sqlite3.Connection,
        observed: dict[str, Any] | None,
        *,
        now: float,
    ) -> list[dict[str, Any]]:
        changes: list[dict[str, Any]] = []
        busy = self._busy_set(observed)
        visible = (
            {int(device) for device in observed.get("devices", [])}
            if observed is not None and observed.get("status") == "ok"
            else None
        )

        for row in connection.execute("SELECT * FROM holds WHERE state='active' AND end_at<=?", (now,)).fetchall():
            connection.execute("UPDATE holds SET state='expired', updated_at=? WHERE hold_id=?", (now, row["hold_id"]))
            changes.append({"hold_id": row["hold_id"], "from": "active", "to": "expired"})
            self._event(connection, "hold-expired", data={"hold_id": row["hold_id"]}, now=now)

        for row in connection.execute(
            "SELECT * FROM tasks WHERE state='queued' AND latest_start<=?", (now,)
        ).fetchall():
            connection.execute(
                "UPDATE tasks SET state='expired', updated_at=?, message=? WHERE task_id=?",
                (now, "latest_start elapsed before grant", row["task_id"]),
            )
            changes.append({"task_id": row["task_id"], "from": "queued", "to": "expired"})
            self._event(connection, "task-expired-in-queue", task_id=row["task_id"], now=now)

        deadline_rows = connection.execute(
            """
            SELECT * FROM tasks
            WHERE state IN ('granted','starting')
              AND activation_deadline IS NOT NULL
              AND activation_deadline<=?
            """,
            (now,),
        ).fetchall()
        for row in deadline_rows:
            devices = set(_load_devices(row["granted_devices"]))
            still_busy = (
                busy is None
                or visible is None
                or not devices.issubset(visible)
                or (bool(devices & busy) and not row["allow_external_busy"])
                or process_guard_busy(row["process_guard"])
            )
            next_state = "orphaned_busy" if still_busy else "expired"
            message = (
                "activation deadline elapsed; occupancy still busy or unknown"
                if still_busy
                else "activation deadline elapsed before task became active"
            )
            connection.execute(
                "UPDATE tasks SET state=?, updated_at=?, message=? WHERE task_id=?",
                (next_state, now, message, row["task_id"]),
            )
            changes.append({"task_id": row["task_id"], "from": row["state"], "to": next_state})
            self._event(
                connection,
                "activation-deadline-elapsed",
                task_id=row["task_id"],
                data={"to": next_state},
                now=now,
            )

        heartbeat_rows = connection.execute(
            """
            SELECT * FROM tasks
            WHERE state='active'
              AND heartbeat_deadline IS NOT NULL
              AND heartbeat_deadline<=?
            """,
            (now,),
        ).fetchall()
        for row in heartbeat_rows:
            devices = set(_load_devices(row["granted_devices"]))
            still_busy = (
                busy is None
                or visible is None
                or not devices.issubset(visible)
                or (bool(devices & busy) and not row["allow_external_busy"])
                or unguarded_cpu_process(row)
                or unguarded_shared_process(row)
                or process_guard_busy(row["process_guard"])
            )
            next_state = "orphaned_busy" if still_busy else "released"
            message = 'CPU task has no process guard; process completion is unknown' if unguarded_cpu_process(row) else (
                "heartbeat expired while hardware remained busy or unknown"
                if still_busy
                else "heartbeat expired and hardware was observed free"
            )
            connection.execute(
                "UPDATE tasks SET state=?, process_guard=CASE WHEN ?='released' THEN NULL ELSE process_guard END, updated_at=?, message=? WHERE task_id=?",
                (next_state, next_state, now, message, row["task_id"]),
            )
            changes.append({"task_id": row["task_id"], "from": "active", "to": next_state})
            self._event(connection, "heartbeat-expired", task_id=row["task_id"], data={"to": next_state}, now=now)

        if busy is not None and visible is not None:
            for row in connection.execute("SELECT * FROM tasks WHERE state='orphaned_busy'").fetchall():
                devices = set(_load_devices(row["granted_devices"]))
                # Service-port rows stay until an explicit release confirms the
                # listener is gone. Age and occupancy alone are not enough.
                if self._task_service_ports(connection, row["task_id"]):
                    continue
                if (devices.issubset(visible) and (not devices.intersection(busy) or row["allow_external_busy"])
                        and not unguarded_cpu_process(row) and not unguarded_shared_process(row)
                        and not process_guard_busy(row["process_guard"])):
                    connection.execute(
                        "UPDATE tasks SET state='released', process_guard=NULL, updated_at=?, message=? WHERE task_id=?",
                        (now, "orphaned shared task has no pending owned process" if row["allow_external_busy"]
                         else "orphaned task hardware is now free", row["task_id"]),
                    )
                    changes.append({"task_id": row["task_id"], "from": "orphaned_busy", "to": "released"})
                    self._event(connection, "orphaned-task-released", task_id=row["task_id"], now=now)

        for row in connection.execute(
            "SELECT * FROM tasks WHERE state='active' AND expected_end IS NOT NULL AND expected_end<=?",
            (now,),
        ).fetchall():
            if row["message"] != "estimated duration exceeded; lease remains protected":
                connection.execute(
                    "UPDATE tasks SET updated_at=?, message=? WHERE task_id=?",
                    (now, "estimated duration exceeded; lease remains protected", row["task_id"]),
                )
                changes.append({"task_id": row["task_id"], "state": "active", "overdue": True})
                self._event(connection, "task-overdue", task_id=row["task_id"], now=now)
        return changes

    @staticmethod
    def _reserved_devices(connection: sqlite3.Connection, *, exclude_task: str | None = None) -> set[int]:
        reserved: set[int] = set()
        query = (
            "SELECT task_id, granted_devices FROM tasks "
            "WHERE state IN ('granted','starting','active','orphaned_busy')"
        )
        for row in connection.execute(query).fetchall():
            if exclude_task and row["task_id"] == exclude_task:
                continue
            reserved.update(_load_devices(row["granted_devices"]))
        return reserved

    @staticmethod
    def _hold_conflicts(
        connection: sqlite3.Connection,
        *,
        device: int,
        start_at: float,
        end_at: float,
    ) -> bool:
        for row in connection.execute(
            """
            SELECT devices FROM holds
            WHERE state='active' AND not_before<? AND end_at>?
            """,
            (end_at, start_at),
        ).fetchall():
            if device in _load_devices(row["devices"]):
                return True
        return False

    def _coordination_epoch(self, connection: sqlite3.Connection) -> str:
        row = connection.execute("SELECT value FROM meta WHERE key='coordination_epoch'").fetchone()
        if row is None:
            raise CoordinationError("coordination epoch is missing")
        return str(row["value"])

    @staticmethod
    def _serialize_port(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        payload = dict(row)
        payload["created_at"] = utc_now_iso(float(payload["created_at"]))
        return payload

    def _task_ports(self, connection: sqlite3.Connection, task_id: str) -> list[dict[str, Any]]:
        rows = connection.execute(
            "SELECT * FROM ports WHERE task_id=? ORDER BY port ASC", (task_id,)
        ).fetchall()
        return [item for item in (self._serialize_port(row) for row in rows) if item is not None]

    def _available_devices(
        self,
        connection: sqlite3.Connection,
        observed: dict[str, Any],
        *,
        exclude_task: str | None,
        start_at: float,
        end_at: float,
        allow_external_busy: bool = False,
    ) -> list[int]:
        busy = self._busy_set(observed) or set()
        reserved = self._reserved_devices(connection, exclude_task=exclude_task)
        available: list[int] = []
        for device in observed.get("devices", []):
            device = int(device)
            if (device in busy and not allow_external_busy) or device in reserved:
                continue
            if self._hold_conflicts(connection, device=device, start_at=start_at, end_at=end_at):
                continue
            available.append(device)
        return available

    def _select_granted_devices(
        self,
        *,
        requested: list[int],
        count: int,
        available: list[int],
    ) -> tuple[list[int] | None, list[int]]:
        if requested:
            missing = sorted(set(requested) - set(available))
            if missing:
                return None, missing
            return requested, []
        if len(available) < count:
            return None, []
        return sorted(available)[:count], []

    def _allocated_ports(self, connection: sqlite3.Connection, *, exclude_task: str | None = None) -> set[int]:
        allocated: set[int] = set()
        for row in connection.execute("SELECT port, task_id FROM ports").fetchall():
            if exclude_task and row["task_id"] == exclude_task:
                continue
            allocated.add(int(row["port"]))
        return allocated

    def _task_service_ports(self, connection: sqlite3.Connection, task_id: str) -> list[int]:
        return [
            int(row["port"])
            for row in connection.execute(
                "SELECT port FROM ports WHERE task_id=? AND kind='service' ORDER BY port",
                (task_id,),
            ).fetchall()
        ]

    def _claim_port(
        self,
        connection: sqlite3.Connection,
        *,
        port: int,
        kind: str,
        task_id: str,
        owner: str,
        now: float,
        listening: dict[str, Any] | None = None,
        allow_listening: bool = False,
    ) -> int:
        if kind not in PORT_KINDS:
            raise CoordinationError(f"unsupported port kind: {kind!r}")
        if type(port) is not int or not 0 < port < 65536:
            raise CoordinationError(f"invalid TCP port: {port!r}")
        existing = connection.execute("SELECT kind, task_id FROM ports WHERE port=?", (port,)).fetchone()
        if existing is not None:
            if existing["task_id"] == task_id and existing["kind"] == kind:
                return port
            raise CoordinationError(f"port {port} is already reserved as {existing['kind']}",
                                    error_code="port_reserved", port=port)
        if not allow_listening:
            if listening is None or listening.get("status") != "ok":
                raise CoordinationError(
                    (listening or {}).get("error") or "host listening ports are unavailable"
                )
            live = {int(item) for item in listening.get("ports", [])}
            if port in live:
                raise CoordinationError(f"port {port} is already bound on the host")
        connection.execute(
            "INSERT INTO ports(port, kind, task_id, owner, created_at) VALUES(?, ?, ?, ?, ?)",
            (port, kind, task_id, owner, now),
        )
        return port

    def _select_service_port(
        self,
        connection: sqlite3.Connection,
        *,
        choices: list[int],
        requested: int,
        task_id: str,
        owner: str,
        now: float,
        listening: dict[str, Any],
    ) -> int:
        if listening.get("status") != "ok":
            raise CoordinationError(
                listening.get("error") or "host listening ports are unavailable"
            )
        if not choices:
            if requested != 0:
                raise CoordinationError("service port requested but no declared runtime service ports")
            # Prepared task roots can share a host-managed automatic port pool.
            # The port is claimed below only after checking live and leased ports.
            first, last = parse_port_range(DEFAULT_SERVING_PORT_RANGE)
            choices = list(range(first, last + 1))
        live = {int(item) for item in listening.get("ports", [])}
        allocated = self._allocated_ports(connection, exclude_task=task_id)
        preferred = None if requested == 0 else requested
        if preferred is not None and preferred not in choices:
            raise CoordinationError(f"service port {preferred} is not a declared runtime service port")
        candidates = [preferred] if preferred is not None else list(choices)
        owned = {
            int(row["port"]): str(row["kind"])
            for row in connection.execute(
                "SELECT port, kind FROM ports WHERE task_id=?", (task_id,)
            ).fetchall()
        }
        for port in candidates:
            existing_kind = owned.get(port)
            if existing_kind is not None:
                if existing_kind != "service":
                    if preferred is not None:
                        raise CoordinationError(f"port {port} is already reserved as {existing_kind}")
                    continue
                return port
            if port in allocated or port in live:
                if preferred is not None:
                    raise CoordinationError(f"port {port} is already reserved or bound on the host")
                continue
            connection.execute(
                "INSERT INTO ports(port, kind, task_id, owner, created_at) VALUES(?, ?, ?, ?, ?)",
                (port, "service", task_id, owner, now),
            )
            return port
        raise CoordinationError("no free declared service port")

    def _service_ports_busy(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        listening: dict[str, Any] | None,
    ) -> list[int]:
        ports = self._task_service_ports(connection, task_id)
        if not ports:
            return []
        if listening is None or listening.get("status") != "ok":
            return ports
        live = {int(item) for item in listening.get("ports", [])}
        return sorted(port for port in ports if port in live)

    def _clear_service_ports(self, connection: sqlite3.Connection, task_id: str) -> None:
        connection.execute("DELETE FROM ports WHERE task_id=? AND kind='service'", (task_id,))
        connection.execute(
            "UPDATE tasks SET granted_service_port=NULL WHERE task_id=?",
            (task_id,),
        )

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
                    first, last = parse_port_range(DEFAULT_CONTAINER_SSH_PORT_RANGE)
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
                ORDER BY priority DESC, submitted_at ASC, task_id ASC
                LIMIT 1
                """,
                (now, now),
            ).fetchone()
            if int(row["requested_count"]) and (head is None or head["task_id"] != task_id):
                return {
                    "status": "waiting",
                    "reason": "strict_fifo",
                    "ahead_task_id": None if head is None else head["task_id"],
                    "task": self._serialize_task(row),
                    "gc": changes,
                }
            estimated_end = now + int(row["estimated_duration_seconds"])
            available = self._available_devices(
                connection,
                observed,
                exclude_task=task_id,
                start_at=now,
                end_at=estimated_end,
                allow_external_busy=bool(row["allow_external_busy"]),
            ) if int(row["requested_count"]) else set()
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
                    fence_token=?, granted_service_port=?, updated_at=?, message=?
                WHERE task_id=?
                """,
                (_json_devices(selected), deadline, fence, granted_service_port, now, "short-lived grant issued", task_id),
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
            if conflicts:
                next_state = "queued" if float(row["latest_start"]) > now else "expired"
                connection.execute(
                    """
                    UPDATE tasks
                    SET state=?, granted_devices=NULL, activation_deadline=NULL,
                        fence_token=NULL, updated_at=?, message=?
                    WHERE task_id=?
                    """,
                    (next_state, now, f"preflight found externally busy devices: {conflicts}", task_id),
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
                    "reason": "external_occupancy",
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
            pid = prepared_supervisor_host_pid(prepared_supervisor, process_guard)
        elif process_guard is not None:
            if not process_guard_busy(process_guard, completion_confirmed=True):
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
            if row["state"] not in {"active", "orphaned_busy"}:
                raise CoordinationError(f"task {task_id} is {row['state']}, expected active or orphaned_busy")
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
        return {"status": "active", "task": self._serialize_task(row)}

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
            managed_completion = _managed_shared_completion(row, completion_confirmed)
            conflicts = [] if managed_completion else (
                sorted((set() if row["allow_external_busy"] else devices & busy) | (devices - visible))
                if busy is not None and visible is not None else sorted(devices))
            busy_ports = self._service_ports_busy(connection, task_id, listening)
            if ((devices and not managed_completion and (busy is None or visible is None or conflicts)) or busy_ports or unguarded_cpu_process(row)
                    or unguarded_shared_process(row)
                    or process_guard_busy(row["process_guard"], completion_confirmed=completion_confirmed)):
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
                (now, "owned process and ports are clear; shared lease released" if row["allow_external_busy"]
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
                or process_guard_busy(row["process_guard"])
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
            if _managed_shared_completion(row, completion_confirmed):
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
        info = json.loads(subprocess.check_output(
            ["docker", "inspect", "--format", '{"Id":{{json .Id}},"State":{{json .State}}}', container_name],
            text=True, encoding="utf-8"))
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
    raise CoordinationError(f"unsupported action: {action!r}")
