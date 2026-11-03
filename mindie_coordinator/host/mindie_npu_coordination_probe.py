#!/usr/bin/env python3
"""Standalone probes and shared helpers for host NPU coordination.

[Role]
Stdlib-only parsers, device/port probes, process-guard checks, and resource
helpers used by the state, lease, and public coordination modules.

[Invariants]
- Physical NPU identity comes from `npu-smi` device/chip tables, never guessed.
- A5 busy state comes from process rows; measured HBM usage remains capacity evidence.
- Probe helpers never reclaim a lease; `busy=None` means unobserved.

[Timeouts & Locks]
- `docker`/`ss`/`netstat`/`npu-smi` probes use bounded 5s-15s subprocesses.
- No SQLite or lease state is touched here.

[Side-Effect Policy]
- Read-only observation; no host process or container is mutated.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

SCHEMA_VERSION = 7
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
DEFAULT_GRANT_TTL_SECONDS = 180
DEFAULT_START_TTL_SECONDS = 180
DEFAULT_HEARTBEAT_TTL_SECONDS = 120
DEFAULT_ESTIMATED_DURATION_SECONDS = 3600
DEFAULT_MAX_COMPILE_SLOTS = 4


from .mindie_npu_coordination_resources import (
    COMPILE_WORKER_MEMORY_MB, _available_memory_mb, _cgroup_cpu_cores, host_worker_cap,
    compile_slot_budget, hbm_eligible_devices,
)


_AUTHORITY_MODULE = ".".join(("mindie_coordinator", "host", "mindie_npu_coordination"))


def _authority_attribute(name: str) -> Any:
    """Resolve a patchable helper at call time, falling back to shared globals."""
    module = sys.modules.get(_AUTHORITY_MODULE)
    if module is not None:
        value = getattr(module, name, None)
        if value is not None:
            return value
    return globals()[name]

HBM_BUSY_THRESHOLD_MB = 4096
JOB_TOKEN_ENV = "REMOTE_DEV_JOB_TOKEN"

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
    try:
        info = json.loads(subprocess.check_output(
            ["docker", "inspect", "--format", "{{json .}}", container_name],
            text=True, encoding="utf-8", timeout=5))
    except subprocess.TimeoutExpired as exc:
        raise CoordinationError(f"docker inspect timed out after {exc.timeout}s for {container_name}") from exc
    except TimeoutError:
        raise
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        raise CoordinationError(f"docker inspect failed for {container_name}: {exc}") from exc
    if info["Id"] != container_id:
        raise CoordinationError("container identity changed before activation")
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != receipt["boot_id"]:
        raise CoordinationError("host boot identity changed")
    # Use the inspected immutable ID if the name is reassigned during lookup.
    try:
        rows = subprocess.check_output(
            ["docker", "top", container_id, "-eo", "pid"],
            text=True, encoding="utf-8", timeout=5).splitlines()[1:]
    except subprocess.TimeoutExpired as exc:
        raise CoordinationError(f"docker top timed out after {exc.timeout}s for {container_id}") from exc
    except TimeoutError:
        raise
    except (subprocess.SubprocessError, OSError) as exc:
        raise CoordinationError(f"docker top failed for {container_id}: {exc}") from exc
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
    except TimeoutError:
        raise
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


def _managed_completion(row, completion_confirmed):
    """A drained managed lease releases ownership, not whole-card idleness.

    This selects the hardware-independent path, not proof of process completion:
    release still checks the stored guard's boot/marker and live family, ports
    and fence. Admission keeps the stored exclusive/shared occupancy policy.
    Malformed or legacy guards retain the original hardware path.
    """
    if completion_confirmed is not True or row['state'] not in {'active', 'orphaned_busy'}:
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
        except TimeoutError:
            raise
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


# A5 (`npu-smi` 26.2) renders one logical NPU as a four-cell header row plus a
# four-cell metric row.  The metric row has no PCIe Bus-Id and carries
# `AICore Memory-Usage HBM-Usage` in its final cell.
_SINGLE_ID_TABLE_HEADER = re.compile(
    r"\|\s*NPU(?:\s+ID)?\s*\|\s*Name\s*\|\s*Health\s*\|"
)
_SINGLE_ID_METRICS = re.compile(
    r"(?P<aicore>\d+(?:\.\d+)?|NA|-)\s+"
    r"(?P<memory_used>\d+)\s*/\s*(?P<memory_total>\d+)\s+"
    r"(?P<hbm_used>\d+)\s*/\s*(?P<hbm_total>\d+)"
)


def _npu_smi_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _single_id_metric_row(cells: list[str]) -> re.Match[str] | None:
    """Return A5 metric groups when the trailing cell is a full A5 row."""
    if not cells:
        return None
    return _SINGLE_ID_METRICS.fullmatch(" ".join(cells[-1].split()))


def _parse_npu_smi_device_table(
    output: str,
) -> tuple[set[int], dict[int, dict[str, int]], dict[tuple[int, int], int], str | None]:
    """Visible devices, HBM, explicit chip mapping and any ambiguous layout.

    Handles A2/A3 rows with a PCIe Bus-Id and A5's four-cell single-ID rows.
    """
    dev_ids: set[int] = set()
    header_ids: set[int] = set()
    hbm: dict[int, dict[str, int]] = {}
    chip_devices: dict[tuple[int, int], int] = {}
    implicit_chips: dict[int, list[int]] = {}
    single_id_headers: set[int] = set()
    single_id_details: set[int] = set()
    layout_error = None
    current_npu: int | None = None
    for line in output.splitlines():
        if "0000:" in line:
            chip = re.match(r"\|\s*(\d+)\s+(\d+)\s+\|.*0000:", line)
            if chip:
                dev_id = int(chip.group(2))
                if current_npu is not None:
                    chip_devices[(current_npu, int(chip.group(1)))] = dev_id
            elif current_npu is not None:
                chip = re.match(r"\|\s*(\d+)\s*\|.*0000:", line)
                if not chip:
                    layout_error = "npu-smi device row has no recognized chip identity"
                    continue
                dev_id = current_npu
                implicit_chips.setdefault(current_npu, []).append(int(chip.group(1)))
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
        cells = _npu_smi_cells(line)
        metrics = _single_id_metric_row(cells)
        if metrics is not None and current_npu is not None:
            # A5 has one Die per NPU and no explicit Chip/Phy-ID column.  The
            # metric row follows its NPU header; a present first cell is the
            # Chip ID, otherwise the sole Die is Chip 0.
            chip_id = int(cells[0]) if cells[0].isdigit() else 0
            dev_id = current_npu
            chip_devices[(current_npu, chip_id)] = dev_id
            dev_ids.add(dev_id)
            single_id_details.add(dev_id)
            # HBM remains measured even when the process table reports idle;
            # these observations do not identify the source of existing usage.
            hbm[dev_id] = {"used_mb": int(metrics.group("hbm_used")),
                           "total_mb": int(metrics.group("hbm_total"))}
            continue
        if (len(cells) >= 3 and cells[0].isdigit() and cells[1]
                and not cells[1].isdigit() and ":" not in cells[1]):
            # A5 logical-device header: | 0 | Ascend950DT | OK | ... |
            current_npu = int(cells[0])
            header_ids.add(current_npu)
            single_id_headers.add(current_npu)
            continue
        header = re.match(r"\|\s*(\d+)\s+\d*\w+\d+\w*\s+\|", line)
        if header:
            current_npu = int(header.group(1))
            header_ids.add(current_npu)
    if not dev_ids:
        dev_ids.update(header_ids)
    if single_id_headers and single_id_details != single_id_headers:
        # A truncated A5 table must not make unobserved cards look free.
        layout_error = "npu-smi A5 device table is missing a metric row"
    explicit_mapping_present = bool(chip_devices)
    for npu, chips in implicit_chips.items():
        # A2 omits Phy-ID: one explicit chip row belongs to its NPU header.
        # Multiple chips or mixed layouts cannot establish that identity.
        if len(chips) != 1 or explicit_mapping_present:
            layout_error = "npu-smi device table has an ambiguous single-chip mapping"
        else:
            chip_devices[(npu, chips[0])] = npu
    return dev_ids, hbm, chip_devices, layout_error


def parse_npu_smi_hbm(output: str) -> dict[int, dict[str, int]]:
    """Physical-device HBM used/total from npu-smi info text.

    Does not require a process table; occupancy failure still returns HBM.
    A5 reports a static firmware reservation rather than task usage, so its
    ``used_mb`` is normalized to zero while total capacity is retained.
    """
    _dev_ids, hbm, _chip_devices, _layout_error = _parse_npu_smi_device_table(output)
    return dict(sorted(hbm.items()))


def parse_npu_smi_info(output: str) -> dict[str, Any]:
    """Parse visible devices plus process/HBM occupancy from common layouts."""
    dev_ids, hbm, chip_devices, layout_error = _parse_npu_smi_device_table(output)
    if layout_error:
        return {"status": "failed", "error": layout_error,
                "devices": sorted(dev_ids), "busy": {}, "free": []}
    lines = output.splitlines()

    single_id_layout = bool(_SINGLE_ID_TABLE_HEADER.search(output))
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
            # A2/A3: | NPU Chip | PID | Process name | Memory |. Resolve
            # allocation identity through the actual device-table layout.
            if len(columns) >= 3 and re.fullmatch(r"\d+(?:\s+\d+)?", columns[0]):
                identity = [int(value) for value in columns[0].split()]
                if len(identity) == 1 and single_id_layout and columns[1].isdigit():
                    # A5: | NPU | Process id | Process name | Memory |.
                    # There is no Chip/Phy-ID column, so the lone first cell
                    # is already the logical NPU (and its sole Phy-ID).
                    device = identity[0]
                    reason: dict[str, Any] = {
                        "kind": "process",
                        "pid": int(columns[1]),
                        "name": columns[2],
                    }
                    if len(columns) > 3 and columns[3].isdigit():
                        reason["memory_mb"] = int(columns[3])
                    if device not in dev_ids or not columns[2]:
                        process_error = "npu-smi process row has an unknown device or invalid PID/name"
                        break
                    process_busy.setdefault(device, []).append(reason)
                    continue
                if len(identity) == 2:
                    device = chip_devices.get(tuple(identity))
                elif chip_devices:
                    # A chip table requires both identity columns. In a
                    # multi-chip layout the lone NPU index is not a Phy-ID;
                    # guessing could report the occupied physical device free.
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
        if usage and usage["used_mb"] >= HBM_BUSY_THRESHOLD_MB and not reasons and not single_id_layout:
            reasons.append(
                {
                    "kind": "hbm_threshold",
                    "hbm_used_mb": usage["used_mb"],
                    "threshold_mb": HBM_BUSY_THRESHOLD_MB,
                }
            )
        if reasons:
            busy[device] = reasons
    free = sorted(device for device in dev_ids if device not in busy)
    active_pids = sorted({
        int(reason["pid"])
        for reasons in process_busy.values()
        for reason in reasons
        if isinstance(reason.get("pid"), int)
    })
    return {
        "status": "ok",
        "collected_at": utc_now_iso(),
        "devices": sorted(dev_ids),
        "busy": {str(key): value for key, value in sorted(busy.items())},
        "busy_devices": sorted(busy),
        "free": free,
        "free_devices": free,
        "active_pids": active_pids,
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
    except TimeoutError:
        raise
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
    except TimeoutError:
        raise
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
