"""Agent-facing NPU, server, container, and mount probe for mindie-coordinator.

[Role]
Executes a single-pass agentless SSH probe (`REMOTE_FLEET_PROBE_SCRIPT`) across
Ascend NPU hosts and parses `npu-smi info` (910B A2, dual-die 910C A3, 950 A5,
310P), Docker containers, PID cgroup/owner mapping, system load/memory, and
large model-weight mounts.

[Invariants]
- Dual-die 910C (A3) cards expose per-die `phy_id` (`0..15`) in `phy_ids`,
  `free_phy_ids`, and `busy_phy_ids` so `ASCEND_RT_VISIBLE_DEVICES` always uses
  physical device IDs rather than logical module IDs (`0..7`).
- A card/die is marked `busy` when `hbm_used_mb >= 8192`, `aicore_pct >= 5.0`,
  or any attached process consumes `>= 8192` MB.
- A5 (950DT) prints one identity row per card in a single-column device table
  (`| NPU ID | Name | Health | ... |`) instead of the A2/A3 header+chip pair.
  Its static ~4.7 GiB HBM reservation is firmware baseline, not workload usage,
  so A5 occupancy comes from the process table: any attached process means
  busy and `hbm_used_mb` is normalized to `0` while total capacity is kept.
  Misreading this layout as "no cards" makes every idle A5 card look like zero
  free capacity.
- `npu-top` capacity observations only order host probes when no explicit host
  was requested. The SSH/NPU queue evidence still decides actual availability;
  a monitor outage is reported in `npu_top_observation`, never as zero capacity.

[Timeouts & Locks]
- Remote script bounds: `npu-smi info` `10s`, `docker ps` `5s`, `df` `3s`;
  outer `remote_bash` timeout is `30000ms` (`30s`).
- Local snapshot cache (`probe_cache.json`) has default TTL `30.0s`.
- `npu-top` capacity reads are bounded by the bridge (`2.0s` default).

[Side-Effect Policy]
- Remote probe is strictly read-only; local cache writes `<state_dir>/probe_cache.json`
  atomically via `.json.tmp` + `replace()`.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from mindie_coordinator.machine_directory import MachineDirectory, MachineDirectoryUnavailable
from mindie_coordinator.npu_top_bridge import NpuTopObservationClient, observation_host_order
from mindie_coordinator.state_paths import coordinator_state_dir

PROBE_CACHE_FILENAME = "probe_cache.json"
DEFAULT_CACHE_TTL_SECONDS = 30.0
BUSY_HBM_THRESHOLD_MB = 8192
BUSY_AICORE_THRESHOLD_PCT = 5.0

_RE_NPU_VERSION = re.compile(r"npu-smi\s+([\w.-]+).*?Version:\s*([\w.-]+)")
_RE_USAGE_CAPACITY = re.compile(r"HBM Capacity\(MB\)\s*:\s*(\d+)", re.IGNORECASE)
_RE_USAGE_RATE = re.compile(r"HBM Usage Rate\(%\)\s*:\s*(\d+)", re.IGNORECASE)
_RE_USAGE_AICORE = re.compile(r"Aicore Usage Rate\(%\)\s*:\s*(\d+)", re.IGNORECASE)
_RE_DOCKER_PORT = re.compile(r":(\d+)->22/tcp")
_RE_CONTAINER_ID_64 = re.compile(r"([0-9a-f]{64})")

WEIGHT_MOUNT_KEYWORDS = ("weight", "weights", "model", "models", "hf", "huggingface", "llm", "data", "nfs", "ceph", "obs", "sfs")

REMOTE_FLEET_PROBE_SCRIPT = r"""
export PATH="/usr/local/Ascend/driver/tools:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"

echo "===SECTION:NPU_SMI==="
npu_smi_out=""
if command -v npu-smi >/dev/null 2>&1; then
    if command -v timeout >/dev/null 2>&1; then
        npu_smi_out=$(timeout 10 npu-smi info 2>/dev/null || true)
    else
        npu_smi_out=$(npu-smi info 2>/dev/null || true)
    fi
elif [ -x /usr/local/Ascend/driver/tools/npu-smi ]; then
    if command -v timeout >/dev/null 2>&1; then
        npu_smi_out=$(timeout 10 /usr/local/Ascend/driver/tools/npu-smi info 2>/dev/null || true)
    else
        npu_smi_out=$(/usr/local/Ascend/driver/tools/npu-smi info 2>/dev/null || true)
    fi
fi
[ -n "$npu_smi_out" ] && printf '%s\n' "$npu_smi_out"

echo "===SECTION:CONTAINERS==="
if command -v docker >/dev/null 2>&1; then
    timeout 5 docker ps --no-trunc --format '{{.ID}}|{{.Names}}|{{.Image}}|{{.Status}}|{{.Ports}}' 2>/dev/null || true
fi

echo "===SECTION:PID_MAP==="
npu_pids=$(printf '%s\n' "$npu_smi_out" | awk -F'|' 'NF>=4 && $2 ~ /^[[:space:]]*[0-9]+[[:space:]]*$/ {gsub(/[[:space:]]/, "", $2); print $2}' | sort -u)
for p in $npu_pids; do
    [ -d "/proc/$p" ] || continue
    owner=$(stat -c '%U' "/proc/$p" 2>/dev/null || echo "")
    cg=$(tr '\n' ' ' < "/proc/$p/cgroup" 2>/dev/null || echo "")
    pstate=$(awk '/^State:/ {print $2}' "/proc/$p/status" 2>/dev/null || echo "")
    wchan=$(tr -d '\n' < "/proc/$p/wchan" 2>/dev/null || echo "")
    echo "$p|$owner|$cg|$pstate|$wchan"
done

echo "===SECTION:SYSTEM==="
echo "HOSTNAME:$(hostname 2>/dev/null || echo unknown)"
echo "UPTIME:$(uptime -p 2>/dev/null || uptime 2>/dev/null || echo unknown)"
echo "LOADAVG:$(cat /proc/loadavg 2>/dev/null || echo '0 0 0')"
echo "CORES:$(nproc 2>/dev/null || echo 0)"
echo "CPU_MODEL:$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^[ \t]*//' || lscpu 2>/dev/null | grep -i 'Model name:' | head -n1 | cut -d: -f2 | sed 's/^[ \t]*//' || uname -m)"
free -m 2>/dev/null | awk '/^Mem:/ {print "MEM:" $2 ":" $3 ":" $7}'

echo "===SECTION:MOUNTS==="
timeout 3 df -B1 -T -x tmpfs -x devtmpfs -x overlay -x squashfs -x nsfs 2>/dev/null | tail -n +2 || true

echo "===SECTION:TOP_PROCS==="
ps -eo pid,user,%cpu,%mem,rss,comm --sort=-%cpu 2>/dev/null | head -n 11 | tail -n +2 || true
"""


def _split_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|") and stripped.endswith("|"):
        stripped = stripped[1:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _parse_int(val: str, default: int = 0) -> int:
    try:
        return int(val.strip())
    except (ValueError, AttributeError):
        return default


def _parse_float(val: str | None) -> float | None:
    if not val:
        return None
    cleaned = val.strip()
    if cleaned in ("-", "N/A", "NA", ""):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


_DISK_WAIT_WCHAN_TOKENS = ("lock_page", "folio_wait", "nfs_", "ceph_", "blk_mq", "io_schedule", "wait_on_page")


def is_disk_io_wait(proc_state: str | None, wchan: str | None) -> bool:
    """Return True when a Linux process is in uninterruptible disk/NFS page-cache sleep (`D` state)."""
    st = str(proc_state or "").strip().upper()
    wc = str(wchan or "").strip().lower()
    return st == "D" or any(tok in wc for tok in _DISK_WAIT_WCHAN_TOKENS)


def is_npu_busy(
    *,
    hbm_used_mb: int,
    aicore_pct: float = 0.0,
    processes: list[dict[str, Any]] | None = None,
    process_presence_is_busy: bool = False,
) -> bool:
    """Apply MindIE's standard occupancy threshold (>= 8192 MB HBM or active compute).

    `process_presence_is_busy` is A5 (950DT) policy: the card reports a static
    firmware HBM reservation and small test workloads, so any attributed process
    is workload occupancy regardless of its reported memory.
    """
    if hbm_used_mb >= BUSY_HBM_THRESHOLD_MB:
        return True
    if aicore_pct >= BUSY_AICORE_THRESHOLD_PCT:
        return True
    attached = list(processes or [])
    if process_presence_is_busy and attached:
        return True
    for proc in attached:
        if int(proc.get("memory_mb") or 0) >= BUSY_HBM_THRESHOLD_MB:
            return True
    return False


def parse_npu_smi_usages(raw: str, current: dict[str, Any]) -> dict[str, Any]:
    """Enrich an NPU card dict with `npu-smi info -t usages -i <id>` output."""
    updated = dict(current)
    cap_match = _RE_USAGE_CAPACITY.search(raw)
    rate_match = _RE_USAGE_RATE.search(raw)
    aic_match = _RE_USAGE_AICORE.search(raw)

    if cap_match and rate_match:
        cap = int(cap_match.group(1))
        rate = int(rate_match.group(1))
        if cap > 0 and int(updated.get("hbm_total_mb") or 0) == 0:
            used = round(cap * rate / 100)
            updated["hbm_total_mb"] = cap
            updated["hbm_used_mb"] = used
    if aic_match and float(updated.get("aicore_pct") or 0.0) == 0.0:
        updated["aicore_pct"] = float(aic_match.group(1))
    updated["busy"] = is_npu_busy(
        hbm_used_mb=int(updated.get("hbm_used_mb") or 0),
        aicore_pct=float(updated.get("aicore_pct") or 0.0),
        processes=updated.get("processes") or [],
    )
    return updated


def _a5_single_column_rows(lines: list[str]) -> list[tuple[list[str], list[str], bool]]:
    """Extract A5 (950DT) device rows from its single-column `npu-smi info` table.

    A5 prints one identity row per card (`| 0 | Ascend950DT | OK | power temp |`)
    followed by one metric row (`| | | Bus-Id | util mem hbm |`); it has no
    Chip/Phy-ID column. The A2/A3 header+chip walk ignores the digit-only
    identity cell and would report the host as having no NPUs at all, which is
    why idle A5 cards previously looked like zero free capacity. The returned
    tuples already use the canonical (identity, metric, first-for-card) shape.
    """
    rows: list[tuple[list[str], list[str], bool]] = []
    pending: list[str] | None = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line.startswith("|") or line.startswith("+"):
            continue
        cells = _split_row(line)
        if len(cells) < 4:
            continue
        identity = (
            cells[0].strip().isdigit()
            and cells[1].strip()
            and not cells[1].strip().isdigit()
            and cells[2].strip()
            and not cells[2].strip().isdigit()
        )
        if identity:
            if pending is not None:
                # Identity row without a metric row: keep the card visible and
                # let the process table decide occupancy instead of dropping it.
                rows.append((pending, [], True))
            pending = [f"{cells[0].strip()} {cells[1].strip()}", cells[2].strip(),
                       cells[3].strip()]
            continue
        if pending is not None and not cells[0].strip() and cells[3].strip():
            rows.append((pending, ["", cells[2].strip(), cells[3].strip()], True))
            pending = None
    if pending is not None:
        rows.append((pending, [], True))
    return rows


def _is_a5_single_column_layout(lines: list[str]) -> bool:
    """True when the capture carries A5's `NPU ID` device caption.

    A2/A3 captions are `NPU   Name` / `NPU   Chip`, so this stays specific to
    the 950DT single-column table.
    """
    for raw_line in lines:
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = _split_row(line)
        if cells and cells[0].strip() == "NPU ID":
            return True
    return False


def parse_npu_smi_info(
    raw: str,
    *,
    pid_owners: Mapping[int, str] | None = None,
    pid_containers: Mapping[int, str] | None = None,
    pid_states: Mapping[int, str] | None = None,
    pid_wchans: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Parse `npu-smi info` output for 910B (A2), dual-die 910C (A3), 950 (A5), and 310P."""
    npu_smi_version = ""
    driver_version = ""
    ver_match = _RE_NPU_VERSION.search(raw)
    if ver_match:
        npu_smi_version = ver_match.group(1)
        driver_version = ver_match.group(2)

    lines = raw.splitlines()
    device_rows: list[tuple[list[str], list[str], bool]] = []
    process_rows: list[list[str]] = []

    in_process_section = False
    current_header: list[str] | None = None
    header_has_chip = False

    for raw_line in lines:
        line = raw_line.strip()
        if " running processes " in line.lower() or "process id" in line.lower():
            if current_header is not None and not header_has_chip:
                device_rows.append((current_header, [], True))
                current_header = None
            in_process_section = True
            continue
        if not line.startswith("|") or line.startswith("+"):
            continue
        cells = _split_row(line)
        if not in_process_section:
            if len(cells) >= 3 and any(
                kw in cells[0].lower() for kw in ("npu", "chip", "===")
            ):
                continue
            if len(cells) >= 3:
                first_tokens = cells[0].split()
                if not first_tokens or not first_tokens[0].isdigit():
                    continue
                is_header = len(first_tokens) >= 2 and not first_tokens[1].isdigit()
                is_chip = all(tok.isdigit() for tok in first_tokens) and len(first_tokens) <= 2
                if is_header:
                    if current_header is not None and not header_has_chip:
                        device_rows.append((current_header, [], True))
                    current_header = cells
                    header_has_chip = False
                elif is_chip and current_header is not None:
                    device_rows.append((current_header, cells, not header_has_chip))
                    header_has_chip = True
        else:
            if len(cells) >= 4:
                first_tokens = cells[0].split()
                if first_tokens and first_tokens[0].isdigit():
                    process_rows.append(cells)

    if current_header is not None and not header_has_chip:
        device_rows.append((current_header, [], True))

    a5_layout = _is_a5_single_column_layout(lines)
    if a5_layout:
        a5_rows = _a5_single_column_rows(lines)
        if a5_rows:
            device_rows = a5_rows

    die_records: dict[int, list[dict[str, Any]]] = {}
    for id_cells, metric_cells, is_first_for_header in device_rows:
        tokens0 = id_cells[0].split()
        npu_id = int(tokens0[0])
        chip_name = " ".join(tokens0[1:]) if len(tokens0) > 1 else "Ascend"
        health = id_cells[1].strip() if len(id_cells) > 1 else "Unknown"

        power_temp_str = id_cells[2].strip() if len(id_cells) > 2 else ""
        pt_tokens = power_temp_str.split()
        power_w: float | None = None
        temp_c: float | None = None
        if len(pt_tokens) >= 2:
            power_w = _parse_float(pt_tokens[0]) if is_first_for_header else None
            temp_c = _parse_float(pt_tokens[1])
        elif len(pt_tokens) == 1:
            val = _parse_float(pt_tokens[0])
            if val is not None:
                if val < 100 and "." not in pt_tokens[0]:
                    temp_c = val
                elif is_first_for_header:
                    power_w = val

        chip_tokens = metric_cells[0].split() if metric_cells else []
        chip_id = (
            int(chip_tokens[0])
            if chip_tokens and chip_tokens[0].isdigit()
            else len(die_records.get(npu_id, []))
        )
        phy_id = (
            int(chip_tokens[1])
            if len(chip_tokens) >= 2 and chip_tokens[1].isdigit()
            else None
        )

        bus_id = ""
        aicore_pct = 0.0
        mem_used_mb = 0
        mem_total_mb = 0
        hbm_used_mb = 0
        hbm_total_mb = 0

        if len(metric_cells) >= 3:
            bus_id = metric_cells[1].strip()
            detail_str = metric_cells[2].strip()
            slash_matches = list(re.finditer(r"(\d+)\s*/\s*(\d+)", detail_str))
            if len(slash_matches) >= 2:
                mem_used_mb = int(slash_matches[0].group(1))
                mem_total_mb = int(slash_matches[0].group(2))
                hbm_used_mb = int(slash_matches[1].group(1))
                hbm_total_mb = int(slash_matches[1].group(2))
                prefix = detail_str[: slash_matches[0].start()].strip().split()
                if prefix:
                    aicore_pct = _parse_float(prefix[0]) or 0.0
            elif len(slash_matches) == 1:
                hbm_used_mb = int(slash_matches[0].group(1))
                hbm_total_mb = int(slash_matches[0].group(2))
                prefix = detail_str[: slash_matches[0].start()].strip().split()
                if prefix:
                    aicore_pct = _parse_float(prefix[0]) or 0.0
        elif len(id_cells) >= 4:
            slash_matches = list(re.finditer(r"(\d+)\s*/\s*(\d+)", " ".join(id_cells[2:])))
            if slash_matches:
                last = slash_matches[-1]
                hbm_used_mb = int(last.group(1))
                hbm_total_mb = int(last.group(2))

        # Host memory is not device HBM; leave hbm_total_mb=0 when absent so
        # `npu-smi info -t usages` enrichment can populate true device HBM.
        die_records.setdefault(npu_id, []).append(
            {
                "chip_id": chip_id,
                "phy_id": phy_id,
                "chip_name": chip_name,
                "health": health,
                "power_w": power_w,
                "temp_c": temp_c,
                "bus_id": bus_id,
                "aicore_pct": aicore_pct,
                "memory_used_mb": mem_used_mb,
                "memory_total_mb": mem_total_mb,
                # A5's ~4.7 GiB figure at idle is a firmware reservation, not
                # workload usage; occupancy there comes from the process table.
                "hbm_used_mb": 0 if a5_layout else hbm_used_mb,
                "hbm_total_mb": hbm_total_mb,
            }
        )

    # Resolve physical IDs per NPU before attributing processes
    phy_by_npu_chip: dict[tuple[int, int], int] = {}
    for npu_id, dies in die_records.items():
        die_count = len(dies)
        has_explicit = any(d["phy_id"] is not None for d in dies)
        for idx, d in enumerate(dies):
            if d["phy_id"] is not None:
                resolved_phy = int(d["phy_id"])
            elif has_explicit:
                resolved_phy = npu_id * die_count + idx
            else:
                resolved_phy = npu_id
            d["phy_id"] = resolved_phy
            phy_by_npu_chip[(npu_id, int(d["chip_id"]))] = resolved_phy

    procs_by_npu: dict[int, list[dict[str, Any]]] = {}
    procs_by_phy: dict[int, list[dict[str, Any]]] = {}
    owners_map = pid_owners or {}
    containers_map = pid_containers or {}
    states_map = pid_states or {}
    wchans_map = pid_wchans or {}
    for cells in process_rows:
        npu_tokens = cells[0].split()
        if not npu_tokens or not npu_tokens[0].isdigit():
            continue
        npu_id = int(npu_tokens[0])
        chip_id = int(npu_tokens[1]) if len(npu_tokens) >= 2 and npu_tokens[1].isdigit() else None
        pid = _parse_int(cells[1])
        proc_name = cells[2] if len(cells) > 2 else ""
        mem_mb = _parse_int(cells[3]) if len(cells) > 3 else 0
        if pid <= 0:
            continue
        pstate = states_map.get(pid, "")
        pwchan = wchans_map.get(pid, "")
        proc_entry: dict[str, Any] = {
            "pid": pid,
            "process_name": proc_name,
            "memory_mb": mem_mb,
            "owner": owners_map.get(pid, ""),
            "container_name": containers_map.get(pid, ""),
            "proc_state": pstate,
            "wchan": pwchan,
            "disk_io_wait": is_disk_io_wait(pstate, pwchan),
        }
        if chip_id is not None:
            proc_entry["chip_id"] = chip_id
            resolved_phy = phy_by_npu_chip.get((npu_id, chip_id))
            if resolved_phy is not None:
                proc_entry["phy_id"] = resolved_phy
                procs_by_phy.setdefault(resolved_phy, []).append(proc_entry)
        else:
            for d in die_records.get(npu_id, []):
                procs_by_phy.setdefault(int(d["phy_id"]), []).append(proc_entry)
        procs_by_npu.setdefault(npu_id, []).append(proc_entry)

    npus: list[dict[str, Any]] = []
    all_procs: list[dict[str, Any]] = []
    for npu_id in sorted(die_records):
        dies = die_records[npu_id]
        die_count = len(dies)
        first = dies[0]
        chip_name = first["chip_name"]
        if die_count >= 2 and "910" in chip_name and "910c" not in chip_name.lower():
            chip_name = f"{chip_name} (910C dual-die)"
        health = "OK" if all(d["health"].upper() == "OK" for d in dies) else first["health"]
        powers = [d["power_w"] for d in dies if d["power_w"] is not None]
        temps = [d["temp_c"] for d in dies if d["temp_c"] is not None]
        power_w = round(sum(powers), 1) if powers else None
        temp_c = round(max(temps), 1) if temps else None
        aicore_pct = round(sum(d["aicore_pct"] for d in dies) / die_count, 1)
        mem_used_mb = sum(d["memory_used_mb"] for d in dies)
        mem_total_mb = sum(d["memory_total_mb"] for d in dies)
        hbm_used_mb = sum(d["hbm_used_mb"] for d in dies)
        hbm_total_mb = sum(d["hbm_total_mb"] for d in dies)
        procs = procs_by_npu.get(npu_id, [])
        all_procs.extend(procs)
        hbm_pct = round((hbm_used_mb / hbm_total_mb) * 100.0, 1) if hbm_total_mb > 0 else 0.0
        die_entries: list[dict[str, Any]] = []
        for d in dies:
            pid_list = procs_by_phy.get(int(d["phy_id"]), [])
            die_busy = is_npu_busy(
                hbm_used_mb=int(d["hbm_used_mb"]),
                aicore_pct=float(d["aicore_pct"]),
                processes=pid_list,
                process_presence_is_busy=a5_layout,
            )
            die_entries.append(
                {
                    "chip_id": int(d["chip_id"]),
                    "phy_id": int(d["phy_id"]),
                    "bus_id": d["bus_id"],
                    "aicore_pct": d["aicore_pct"],
                    "hbm_used_mb": d["hbm_used_mb"],
                    "hbm_total_mb": d["hbm_total_mb"],
                    "busy": die_busy,
                }
            )
        busy = any(d["busy"] for d in die_entries) or is_npu_busy(
            hbm_used_mb=hbm_used_mb, aicore_pct=aicore_pct, processes=procs,
            process_presence_is_busy=a5_layout,
        )
        phy_ids = list(dict.fromkeys(int(d["phy_id"]) for d in die_entries))
        npus.append(
            {
                "npu_id": npu_id,
                "phy_ids": phy_ids,
                "chip_name": chip_name,
                "health": health,
                "power_w": power_w,
                "temp_c": temp_c,
                "bus_id": first["bus_id"],
                "aicore_pct": aicore_pct,
                "memory_used_mb": mem_used_mb,
                "memory_total_mb": mem_total_mb,
                "hbm_used_mb": hbm_used_mb,
                "hbm_total_mb": hbm_total_mb,
                "hbm_pct": hbm_pct,
                "die_count": die_count,
                "dies": die_entries,
                "busy": busy,
                "disk_io_wait_pids": [int(p["pid"]) for p in procs if p.get("disk_io_wait")],
                "processes": procs,
            }
        )

    return {
        "driver_version": driver_version,
        "npu_smi_version": npu_smi_version,
        "smi_version": npu_smi_version,
        "npus": npus,
        "processes": all_procs,
    }


def parse_fleet_probe_output(
    raw: str,
    *,
    alias: str,
    host: str,
    port: int = 22,
    user: str = "root",
    container_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse the combined multi-section stdout from `REMOTE_FLEET_PROBE_SCRIPT`."""
    sections: dict[str, list[str]] = {}
    current = "PREAMBLE"
    for line in raw.splitlines():
        if line.startswith("===SECTION:") and line.endswith("==="):
            current = line[len("===SECTION:") : -3].strip()
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)

    containers: list[dict[str, Any]] = []
    id_to_name: dict[str, str] = {}
    for line in sections.get("CONTAINERS", []):
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = line.split("|", 4)
        if len(parts) < 4:
            continue
        cid, cname, cimage, cstatus = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
        cports = parts[4].strip() if len(parts) > 4 else ""
        port_match = _RE_DOCKER_PORT.search(cports)
        ssh_port = int(port_match.group(1)) if port_match else None
        containers.append(
            {
                "id": cid[:12],
                "full_id": cid,
                "name": cname,
                "image": cimage,
                "status": cstatus,
                "ports": cports,
                "ssh_port": ssh_port,
            }
        )
        if cid:
            id_to_name[cid.lower()] = cname
            id_to_name[cid[:12].lower()] = cname

    pid_owners: dict[int, str] = {}
    pid_containers: dict[int, str] = {}
    pid_states: dict[int, str] = {}
    pid_wchans: dict[int, str] = {}
    for line in sections.get("PID_MAP", []):
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = line.split("|", 4)
        if len(parts) < 3:
            continue
        pid = _parse_int(parts[0])
        if pid <= 0:
            continue
        pid_owners[pid] = parts[1].strip()
        cgroup = parts[2].strip()
        if len(parts) >= 4:
            pid_states[pid] = parts[3].strip()
        if len(parts) >= 5:
            pid_wchans[pid] = parts[4].strip()
        match_64 = _RE_CONTAINER_ID_64.search(cgroup)
        if match_64:
            cid = match_64.group(1).lower()
            cname = id_to_name.get(cid) or id_to_name.get(cid[:12]) or cid[:12]
            pid_containers[pid] = cname

    npu_raw = "\n".join(sections.get("NPU_SMI", []))
    npu_parsed = parse_npu_smi_info(
        npu_raw,
        pid_owners=pid_owners,
        pid_containers=pid_containers,
        pid_states=pid_states,
        pid_wchans=pid_wchans,
    )
    npus = npu_parsed["npus"]
    total_npus = len(npus)
    busy_ids = [card["npu_id"] for card in npus if card.get("busy")]
    free_ids = [card["npu_id"] for card in npus if not card.get("busy")]
    free_phy_ids = [
        int(die["phy_id"])
        for card in npus
        for die in (card.get("dies") or [{"phy_id": pid, "busy": card.get("busy")} for pid in (card.get("phy_ids") or [card["npu_id"]])])
        if not die.get("busy")
    ]
    busy_phy_ids = [
        int(die["phy_id"])
        for card in npus
        for die in (card.get("dies") or [{"phy_id": pid, "busy": card.get("busy")} for pid in (card.get("phy_ids") or [card["npu_id"]])])
        if die.get("busy")
    ]

    hostname = ""
    uptime_str = ""
    cpu_model = ""
    cpu_cores = 0
    load_1m, load_5m, load_15m = 0.0, 0.0, 0.0
    mem_total_mb, mem_used_mb, mem_avail_mb = 0, 0, 0
    for line in sections.get("SYSTEM", []):
        if line.startswith("HOSTNAME:"):
            hostname = line.partition(":")[2].strip()
        elif line.startswith("UPTIME:"):
            uptime_str = line.partition(":")[2].strip()
        elif line.startswith("CPU_MODEL:"):
            cpu_model = line.partition(":")[2].strip()
        elif line.startswith("CORES:"):
            cpu_cores = _parse_int(line.partition(":")[2])
        elif line.startswith("LOADAVG:"):
            tokens = line.partition(":")[2].strip().split()
            if len(tokens) >= 3:
                load_1m = _parse_float(tokens[0]) or 0.0
                load_5m = _parse_float(tokens[1]) or 0.0
                load_15m = _parse_float(tokens[2]) or 0.0
        elif line.startswith("MEM:"):
            m_parts = line.partition(":")[2].strip().split(":")
            if len(m_parts) >= 3:
                mem_total_mb = _parse_int(m_parts[0])
                mem_used_mb = _parse_int(m_parts[1])
                mem_avail_mb = _parse_int(m_parts[2])

    mounts: list[dict[str, Any]] = []
    for line in sections.get("MOUNTS", []):
        tokens = line.split()
        if len(tokens) < 7:
            continue
        source, fstype = tokens[0], tokens[1]
        total_b = _parse_int(tokens[2])
        used_b = _parse_int(tokens[3])
        avail_b = _parse_int(tokens[4])
        pct_str = tokens[5].rstrip("%")
        target_dir = " ".join(tokens[6:])
        lower_target = target_dir.lower()
        lower_source = source.lower()
        likely_weights = (
            total_b >= 100 * (1024**3)
            and (
                any(kw in lower_target or kw in lower_source for kw in WEIGHT_MOUNT_KEYWORDS)
                or fstype.lower() in {"nfs", "nfs4", "ceph", "lustre", "glusterfs", "obsfs", "sfs"}
            )
        )
        mounts.append(
            {
                "source": source,
                "fstype": fstype,
                "total_gb": round(total_b / (1024**3), 1),
                "used_gb": round(used_b / (1024**3), 1),
                "avail_gb": round(avail_b / (1024**3), 1),
                "used_pct": _parse_float(pct_str) or 0.0,
                "mountpoint": target_dir,
                "likely_weights_dir": likely_weights,
            }
        )

    top_processes: list[dict[str, Any]] = []
    for line in sections.get("TOP_PROCS", []):
        tokens = line.split(None, 5)
        if len(tokens) < 6:
            continue
        top_processes.append(
            {
                "pid": _parse_int(tokens[0]),
                "user": tokens[1],
                "cpu_pct": _parse_float(tokens[2]) or 0.0,
                "mem_pct": _parse_float(tokens[3]) or 0.0,
                "rss_mb": round(_parse_int(tokens[4]) / 1024.0, 1),
                "command": tokens[5],
            }
        )

    chip_names = sorted({card["chip_name"] for card in npus if card.get("chip_name")})
    return {
        "alias": alias,
        "host": host,
        "port": port,
        "user": user,
        "online": True,
        "error": None,
        "probed_at": time.time(),
        "hostname": hostname,
        "uptime": uptime_str,
        "driver_version": npu_parsed["driver_version"],
        "npu_smi_version": npu_parsed["npu_smi_version"],
        "chip_summary": ", ".join(chip_names) if chip_names else "Unknown",
        "total_npus": total_npus,
        "free_npus": len(free_ids),
        "busy_npus": len(busy_ids),
        "free_npu_ids": free_ids,
        "busy_npu_ids": busy_ids,
        "total_phy_devices": len(free_phy_ids) + len(busy_phy_ids),
        "free_phy_ids": free_phy_ids,
        "busy_phy_ids": busy_phy_ids,
        "npus": npus,
        "server": {
            "hostname": hostname,
            "uptime": uptime_str,
            "cpu_model": cpu_model,
            "cpu_cores": cpu_cores,
            "load_1m": load_1m,
            "load_5m": load_5m,
            "load_15m": load_15m,
            "mem_total_mb": mem_total_mb,
            "mem_used_mb": mem_used_mb,
            "mem_available_mb": mem_avail_mb,
            "mem_pct": round((mem_used_mb / mem_total_mb) * 100.0, 1) if mem_total_mb > 0 else 0.0,
            "top_processes": top_processes,
        },
        "containers": containers,
        "bound_container": dict(container_binding) if container_binding else None,
        "mounts": mounts,
    }


def _default_ssh_runner(host_spec: Mapping[str, Any], script: str, timeout_ms: int = 30000) -> str:
    from remote_dev.core.endpoint import resolve_endpoint
    from remote_dev.core.shell_ops import remote_bash

    endpoint = resolve_endpoint(
        {
            "host": host_spec["host"],
            "port": int(host_spec.get("port") or 22),
            "user": str(host_spec.get("user") or "root"),
            "root": "/",
            "cwd": "/",
            **(
                {"identity_file": str(host_spec["identity_file"])}
                if host_spec.get("identity_file")
                else {}
            ),
        }
    )
    response = remote_bash(
        endpoint,
        command=script,
        timeout_ms=timeout_ms,
        wait=True,
        runtime_env=False,
    )
    result = response.get("result") or {}
    if result.get("outcome") != "success":
        stderr = (result.get("preview") or {}).get("stderr") or result.get("summary") or "SSH probe failed"
        raise RuntimeError(str(stderr).strip())
    stdout_ref = (result.get("refs") or {}).get("stdout")
    if stdout_ref and Path(stdout_ref).is_file():
        return Path(stdout_ref).read_text(encoding="utf-8", errors="replace")
    return str((result.get("preview") or {}).get("stdout") or "")


def probe_cache_path(state_dir: Path | str | None = None) -> Path:
    base = Path(state_dir).expanduser().resolve() if state_dir is not None else coordinator_state_dir()
    return base / PROBE_CACHE_FILENAME


def load_probe_cache(state_dir: Path | str | None = None) -> dict[str, dict[str, Any]]:
    path = probe_cache_path(state_dir)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    snapshots = payload.get("snapshots") if isinstance(payload, dict) else None
    return dict(snapshots) if isinstance(snapshots, dict) else {}


def save_probe_cache(
    snapshots: Mapping[str, Mapping[str, Any]],
    *,
    state_dir: Path | str | None = None,
) -> Path:
    path = probe_cache_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "updated_at": time.time(),
        "snapshots": {k: dict(v) for k, v in snapshots.items()},
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def probe_machine_record(
    record: Mapping[str, Any],
    *,
    mode: str = "auto",
    cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
    state_dir: Path | str | None = None,
    runner: Callable[[Mapping[str, Any], str], str] | None = None,
) -> dict[str, Any]:
    """Probe one machine record using cache (`cache`/`auto`) or live SSH (`live`/`auto`)."""
    host_info = record.get("host") or {}
    host_ip = str(host_info.get("ip") or host_info.get("host") or "").strip()
    alias = str(record.get("alias") or host_ip).strip()
    port = int(host_info.get("port") or 22)
    user = str(host_info.get("user") or "root").strip() or "root"
    container_binding = record.get("container") if isinstance(record.get("container"), Mapping) else None

    cache = load_probe_cache(state_dir)
    cached = cache.get(alias) or cache.get(host_ip)
    now = time.time()

    if mode == "cache":
        if cached is not None:
            return {**cached, "cache_hit": True, "cache_age_seconds": round(max(0.0, now - float(cached.get("probed_at") or now)), 1)}
        return {
            "alias": alias,
            "host": host_ip,
            "port": port,
            "user": user,
            "online": False,
            "error": f"no cached probe snapshot for {alias!r}; use mode='live' or mode='auto'",
            "cache_hit": False,
            "total_npus": 0,
            "free_npus": 0,
            "busy_npus": 0,
            "free_npu_ids": [],
            "busy_npu_ids": [],
            "free_phy_ids": [],
            "busy_phy_ids": [],
            "npus": [],
            "containers": [],
            "mounts": [],
        }

    if mode == "auto" and cached is not None and cached.get("online"):
        age = now - float(cached.get("probed_at") or 0.0)
        if 0 <= age <= cache_ttl_seconds:
            return {**cached, "cache_hit": True, "cache_age_seconds": round(age, 1)}

    exec_runner = runner or _default_ssh_runner
    host_spec = {
        "host": host_ip,
        "port": port,
        "user": user,
        **({"identity_file": host_info["identity_file"]} if host_info.get("identity_file") else {}),
    }
    try:
        raw_output = exec_runner(host_spec, REMOTE_FLEET_PROBE_SCRIPT)
        snapshot = parse_fleet_probe_output(
            raw_output,
            alias=alias,
            host=host_ip,
            port=port,
            user=user,
            container_binding=container_binding,
        )
        # Fallback for cards where `npu-smi info` reported 0/0 HBM
        zero_hbm_ids = [c["npu_id"] for c in snapshot["npus"] if int(c.get("hbm_total_mb") or 0) == 0]
        if zero_hbm_ids and runner is None:
            usages_cmd = " ; ".join(
                f"echo '===NPU_USAGE:{nid}==='; timeout 3 npu-smi info -t usages -i {nid} 2>/dev/null || true"
                for nid in zero_hbm_ids[:8]
            )
            try:
                usages_raw = exec_runner(host_spec, usages_cmd)
                by_id: dict[int, list[str]] = {}
                cur_id: int | None = None
                for uline in usages_raw.splitlines():
                    if uline.startswith("===NPU_USAGE:") and uline.endswith("==="):
                        cur_id = _parse_int(uline.split(":", 1)[1][:-3], -1)
                    elif cur_id is not None and cur_id >= 0:
                        by_id.setdefault(cur_id, []).append(uline)
                updated_npus = []
                for card in snapshot["npus"]:
                    nid = card["npu_id"]
                    if nid in by_id:
                        card = parse_npu_smi_usages("\n".join(by_id[nid]), card)
                    updated_npus.append(card)
                snapshot["npus"] = updated_npus
                snapshot["free_npu_ids"] = [c["npu_id"] for c in updated_npus if not c.get("busy")]
                snapshot["busy_npu_ids"] = [c["npu_id"] for c in updated_npus if c.get("busy")]
                snapshot["free_npus"] = len(snapshot["free_npu_ids"])
                snapshot["busy_npus"] = len(snapshot["busy_npu_ids"])
            except Exception as exc:
                snapshot["usage_parse_error"] = f"{type(exc).__name__}: {exc}"[:500]
        cache[alias] = snapshot
        save_probe_cache(cache, state_dir=state_dir)
        return {**snapshot, "cache_hit": False, "cache_age_seconds": 0.0}
    except Exception as exc:
        failed_snapshot = {
            "alias": alias,
            "host": host_ip,
            "port": port,
            "user": user,
            "online": False,
            "error": str(exc),
            "probed_at": now,
            "cache_hit": False,
            "total_npus": 0,
            "free_npus": 0,
            "busy_npus": 0,
            "free_npu_ids": [],
            "busy_npu_ids": [],
            "free_phy_ids": [],
            "busy_phy_ids": [],
            "npus": [],
            "containers": [],
            "bound_container": dict(container_binding) if container_binding else None,
            "mounts": [],
        }
        return failed_snapshot


def _max_contiguous_run(ids: list[int]) -> list[int]:
    if not ids:
        return []
    sorted_ids = sorted(set(ids))
    best: list[int] = [sorted_ids[0]]
    current: list[int] = [sorted_ids[0]]
    for nid in sorted_ids[1:]:
        if nid == current[-1] + 1:
            current.append(nid)
        else:
            if len(current) > len(best):
                best = current
            current = [nid]
    if len(current) > len(best):
        best = current
    return best


def _recommend_devices(free_ids: list[int], count: int) -> tuple[list[int], bool]:
    """Select `count` free NPU IDs, preferring power-of-two aligned or contiguous ranges."""
    sorted_ids = sorted(set(free_ids))
    if count <= 0 or len(sorted_ids) < count:
        return sorted_ids, False
    free_set = set(sorted_ids)
    # First try aligned contiguous blocks (e.g., 0..3, 4..7)
    for start in sorted_ids:
        if count > 1 and start % count != 0:
            continue
        candidate = list(range(start, start + count))
        if all(x in free_set for x in candidate):
            return candidate, True
    # Next try any contiguous block
    for start in sorted_ids:
        candidate = list(range(start, start + count))
        if all(x in free_set for x in candidate):
            return candidate, True
    return sorted_ids[:count], False


def _npu_top_capacity(
    *,
    count: int,
    host: str | None,
    client: Any | None,
    use_npu_top: bool | None,
    min_weight_mount_avail_gb: float,
    prefer_same_package: bool,
    exclude_throttled: bool,
) -> dict:
    """Observe capacity only; failures remain visible evidence."""
    requirements = {
        "min_idle_npus": count,
        "min_weight_mount_avail_gb": float(min_weight_mount_avail_gb),
        "prefer_same_package": bool(prefer_same_package),
        "exclude_throttled": bool(exclude_throttled),
    }
    if host:
        return {"source": "npu-top", "status": "skipped", "allocation_authority": False,
                "reason": "explicit host request preserves its original semantics",
                "candidates": [], "measurement_limitations": [], "requirements": requirements}
    if use_npu_top is False or os.environ.get("MINDIE_TOP_DISABLE") == "1":
        return {"source": "npu-top", "status": "disabled", "allocation_authority": False,
                "reason": "observation ranking disabled by caller/environment",
                "candidates": [], "measurement_limitations": [], "requirements": requirements}
    try:
        if client is None:
            # Default to the local observation API. The bridge keeps the call
            # short and returns a visible unavailable status if no monitor runs.
            client = NpuTopObservationClient(timeout=1.0)
        observation = client.capacity(
            min_idle_npus=count,
            min_weight_mount_avail_gb=min_weight_mount_avail_gb,
            prefer_same_package=prefer_same_package,
            exclude_throttled=exclude_throttled,
        )
    except Exception as exc:
        return {"source": "npu-top", "status": "unavailable", "allocation_authority": False,
                "error": str(exc)[:500], "candidates": [], "requirements": requirements,
                "measurement_limitations": [f"npu-top observation unavailable: {exc}"][:1]}
    return observation if isinstance(observation, dict) else {
        "source": "npu-top", "status": "unavailable", "allocation_authority": False,
        "error": "npu-top client returned a non-object observation",
        "candidates": [], "measurement_limitations": ["npu-top observation unavailable"],
        "requirements": requirements}


def _record_observation_rank(record: Mapping[str, Any], order: Mapping[str, int]) -> int | None:
    host = record.get("host") or {}
    identities = [record.get("alias"), host.get("ip") if isinstance(host, Mapping) else None,
                  host.get("host") if isinstance(host, Mapping) else None]
    ranks = [order[str(value).strip().casefold()] for value in identities
             if value and str(value).strip().casefold() in order]
    return min(ranks) if ranks else None


def _order_records_by_observation(records: list[dict], observation: Mapping[str, Any]) -> list[dict]:
    order = observation_host_order(observation)
    if not order:
        return list(records)

    def key(record: dict) -> int:
        rank = _record_observation_rank(record, order)
        return len(order) + 1 if rank is None else rank

    # Only reorder. Missing/stale observations never remove a host from the
    # probe set; the host queue still decides actual availability.
    return sorted(records, key=key)


def _observation_summary(observation: Mapping[str, Any] | None) -> dict | None:
    if not isinstance(observation, Mapping):
        return None
    candidates = []
    for row in observation.get("candidates") or []:
        if not isinstance(row, Mapping):
            continue
        candidates.append({key: row.get(key) for key in (
            "host", "name", "idle_npu_count", "npu_top_rank", "npu_top_eligible",
            "npu_top_reasons", "IdlePackagePairs", "MaxWeightMountAvailGb",
            "ThermalHealth", "observed_at", "age_seconds") if row.get(key) is not None})
    return {
        "source": observation.get("source", "npu-top"),
        "status": observation.get("status"),
        "allocation_authority": False,
        "error": observation.get("error"),
        "reason": observation.get("reason"),
        "requirements": observation.get("requirements"),
        "observation": observation.get("observation"),
        "measurement_limitations": list(observation.get("measurement_limitations") or []),
        "candidates": candidates,
    }


def find_fleet_capacity(
    *,
    count: int = 1,
    chip_query: str | None = None,
    host: str | None = None,
    mode: str = "auto",
    directory: MachineDirectory | None = None,
    state_dir: Path | str | None = None,
    runner: Callable[[Mapping[str, Any], str], str] | None = None,
    npu_top_client: Any | None = None,
    use_npu_top: bool | None = None,
    min_weight_mount_avail_gb: float = 0.0,
    prefer_same_package: bool = True,
    exclude_throttled: bool = True,
) -> dict[str, Any]:
    """Find and rank machines with at least `count` free NPUs across the fleet.

    `npu-top` observations (when configured) order the host probes; they never
    replace the SSH/NPU queue evidence or allocate a device.
    """
    store = directory if directory is not None else MachineDirectory()
    try:
        records = [store.record(host)] if host else store.machines()
    except MachineDirectoryUnavailable as exc:
        raise ValueError(str(exc)) from exc

    observation = _npu_top_capacity(
        count=count, host=host, client=npu_top_client, use_npu_top=use_npu_top,
        min_weight_mount_avail_gb=min_weight_mount_avail_gb,
        prefer_same_package=prefer_same_package,
        exclude_throttled=exclude_throttled,
    )
    records = _order_records_by_observation(records, observation)
    observation_order = observation_host_order(observation)
    chip_filter = (chip_query or "").strip().lower()
    candidates: list[dict[str, Any]] = []
    unsatisfied: list[dict[str, Any]] = []

    for rec in records:
        snap = probe_machine_record(rec, mode=mode, state_dir=state_dir, runner=runner)
        if not snap.get("online"):
            unsatisfied.append(
                {
                    "alias": snap["alias"],
                    "host": snap["host"],
                    "online": False,
                    "error": snap.get("error"),
                    "free_npus": 0,
                    "total_npus": 0,
                }
            )
            continue
        if chip_filter and chip_filter not in str(snap.get("chip_summary") or "").lower():
            unsatisfied.append(
                {
                    "alias": snap["alias"],
                    "host": snap["host"],
                    "online": True,
                    "reason": f"chip_summary {snap.get('chip_summary')!r} does not match {chip_query!r}",
                    "free_npus": snap["free_npus"],
                    "total_npus": snap["total_npus"],
                }
            )
            continue
        free_ids = list(snap.get("free_npu_ids") or [])
        if len(free_ids) < count:
            unsatisfied.append(
                {
                    "alias": snap["alias"],
                    "host": snap["host"],
                    "online": True,
                    "reason": f"only {len(free_ids)}/{snap['total_npus']} NPUs free (requested {count})",
                    "free_npus": len(free_ids),
                    "free_npu_ids": free_ids,
                    "free_phy_ids": list(snap.get("free_phy_ids") or []),
                    "total_npus": snap["total_npus"],
                }
            )
            continue
        recommended_npus, contiguous = _recommend_devices(free_ids, count)
        npu_by_id = {int(c["npu_id"]): c for c in (snap.get("npus") or [])}
        recommended_phy_ids: list[int] = []
        for nid in recommended_npus:
            card = npu_by_id.get(nid)
            if card and card.get("phy_ids"):
                recommended_phy_ids.extend(int(p) for p in card["phy_ids"])
            else:
                recommended_phy_ids.append(nid)
        visible_str = ",".join(str(x) for x in recommended_phy_ids)
        contig_run = _max_contiguous_run(free_ids)
        observation_rank = _record_observation_rank(rec, observation_order)
        candidates.append(
            {
                "alias": snap["alias"],
                "host": snap["host"],
                "chip_summary": snap.get("chip_summary"),
                "total_npus": snap["total_npus"],
                "free_npus": len(free_ids),
                "free_npu_ids": free_ids,
                "free_phy_ids": list(snap.get("free_phy_ids") or free_ids),
                "recommended_npu_ids": recommended_npus,
                "recommended_phy_ids": recommended_phy_ids,
                "recommended_devices": recommended_phy_ids,
                "ascend_rt_visible_devices": visible_str,
                "recommended_ascend_rt_visible_devices": visible_str,
                "contiguous": contiguous,
                "max_contiguous_free": len(contig_run),
                "bound_container": snap.get("bound_container"),
                "remote_dev_alias": snap["alias"],
                **({"npu_top_rank": observation_rank} if observation_rank is not None else {}),
            }
        )

    # Observation order is primary only when available; host queue evidence
    # (contiguous/free devices) still determines the recommended device set.
    candidates.sort(key=lambda item: (
        item.get("npu_top_rank") if item.get("npu_top_rank") is not None else 10 ** 9,
        -int(item["contiguous"]),
        -item["max_contiguous_free"],
        -item["free_npus"],
    ))
    return {
        "requested_npus": count,
        "chip_query": chip_query,
        "mode": mode,
        "matched_count": len(candidates),
        "recommended": candidates[0] if candidates else None,
        "candidates": candidates,
        "unsatisfied": unsatisfied,
        "npu_top_observation": _observation_summary(observation),
    }
