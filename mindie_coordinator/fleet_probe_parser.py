"""Pure parsers for agentless NPU fleet probe captures.

[Role]
Decodes `npu-smi info` captures for 910B A2, dual-die 910C A3, 950 A5, and
310P, the combined multi-section SSH probe stdout, Docker container rows,
PID/owner maps, system load, and large model-weight mounts.

[Invariants]
- Pure functions only: no SSH, filesystem, clock mutation, or cache access.
- Dual-die 910C physical IDs (`phy_id`) are preserved; A5 single-column tables
  derive occupancy from the process table, not the firmware HBM reservation.
- A card/die is busy at `hbm_used_mb >= 8192`, `aicore_pct >= 5.0`, or an
  attached process consuming `>= 8192` MB; A5 treats any attached process as
  busy. Thresholds and regexes are exported so the probe facade can re-export
  them unchanged.

[Timeouts & Locks]
- In-memory conversion only; no locks, sleeps, subprocesses, or timeouts.
  `probed_at` is the only clock read (`time.time()`).

[Side-Effect Policy]
- Side-effect free. Callers own remote probing and cache writes.
"""

from __future__ import annotations

import re
import time
from typing import Any, Mapping, Sequence

BUSY_HBM_THRESHOLD_MB = 8192
BUSY_AICORE_THRESHOLD_PCT = 5.0

_RE_NPU_VERSION = re.compile(r"npu-smi\s+([\w.-]+).*?Version:\s*([\w.-]+)")
_RE_USAGE_CAPACITY = re.compile(r"HBM Capacity\(MB\)\s*:\s*(\d+)", re.IGNORECASE)
_RE_USAGE_RATE = re.compile(r"HBM Usage Rate\(%\)\s*:\s*(\d+)", re.IGNORECASE)
_RE_USAGE_AICORE = re.compile(r"Aicore Usage Rate\(%\)\s*:\s*(\d+)", re.IGNORECASE)
_RE_DOCKER_PORT = re.compile(r":(\d+)->22/tcp")
_RE_CONTAINER_ID_64 = re.compile(r"([0-9a-f]{64})")

WEIGHT_MOUNT_KEYWORDS = ("weight", "weights", "model", "models", "hf", "huggingface", "llm", "data", "nfs", "ceph", "obs", "sfs")


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


def device_occupancy_inventory(npus: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Card and physical-Die free/busy inventory for one parsed `npus` list.

    Both the raw `npu-smi info` parse and the `-t usages` supplement derive
    their card counts and physical-Die lists from these same observed rows.
    """
    cards = list(npus)
    dies = [
        die
        for card in cards
        for die in (card.get("dies") or [{"phy_id": pid, "busy": card.get("busy")}
                                         for pid in (card.get("phy_ids") or [card["npu_id"]])])
    ]
    free_ids = [card["npu_id"] for card in cards if not card.get("busy")]
    busy_ids = [card["npu_id"] for card in cards if card.get("busy")]
    return {
        "total_npus": len(cards),
        "free_npus": len(free_ids),
        "busy_npus": len(busy_ids),
        "free_npu_ids": free_ids,
        "busy_npu_ids": busy_ids,
        "total_phy_devices": len(dies),
        "free_phy_ids": [int(die["phy_id"]) for die in dies if not die.get("busy")],
        "busy_phy_ids": [int(die["phy_id"]) for die in dies if die.get("busy")],
    }


def parse_npu_smi_usages(raw: str, current: dict[str, Any]) -> dict[str, Any]:
    """Enrich an NPU card dict with `npu-smi info -t usages -i <id>` output."""
    updated = dict(current)
    cap_match = _RE_USAGE_CAPACITY.search(raw)
    rate_match = _RE_USAGE_RATE.search(raw)
    aic_match = _RE_USAGE_AICORE.search(raw)

    # `-t usages` reports one card-wide HBM figure. Applying it is only sound
    # when it maps to exactly one leasable unit: a single-Die card, or a legacy
    # card row with no `dies` at all. A multi-Die card is left unmeasured
    # rather than split across Dies or left with card/Die occupancy disagreeing;
    # downstream already treats unmeasured HBM as no headroom.
    dies = updated.get("dies")
    attributable = not isinstance(dies, list) or len(dies) <= 1
    if cap_match and rate_match and attributable:
        cap = int(cap_match.group(1))
        rate = int(rate_match.group(1))
        if cap > 0 and int(updated.get("hbm_total_mb") or 0) == 0:
            updated["hbm_total_mb"] = cap
            updated["hbm_used_mb"] = round(cap * rate / 100)
    if aic_match and attributable and float(updated.get("aicore_pct") or 0.0) == 0.0:
        updated["aicore_pct"] = float(aic_match.group(1))
    if not attributable:
        return updated

    if isinstance(dies, list) and len(dies) == 1:
        die = dict(dies[0])
        if int(die.get("hbm_total_mb") or 0) == 0 and int(updated.get("hbm_total_mb") or 0) > 0:
            die.update(hbm_total_mb=int(updated["hbm_total_mb"]),
                       hbm_used_mb=int(updated["hbm_used_mb"]))
        if float(die.get("aicore_pct") or 0.0) == 0.0:
            die["aicore_pct"] = float(updated.get("aicore_pct") or 0.0)
        # A compute observation is useful even when the supplement could not
        # recover HBM. Capacity enrichment never clears process-derived busy.
        die["busy"] = bool(die.get("busy")) or is_npu_busy(
            hbm_used_mb=int(die.get("hbm_used_mb") or 0),
            aicore_pct=float(die.get("aicore_pct") or 0.0))
        updated["dies"] = [die]
    # Same card aggregation as the raw parse: any busy Die, or card-wide
    # occupancy evidence.
    updated["busy"] = bool(current.get("busy")) or any(
        die.get("busy") for die in (updated.get("dies") or [])) or is_npu_busy(
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
                # Measured total used HBM, kept factual for allocation headroom.
                # On an idle A5 card this is nonzero (firmware/driver/runtime
                # reservations and anything else resident); how much belongs to
                # each is not established here and varies by machine, so no
                # baseline is attributed or subtracted. Occupancy is a separate
                # question: A5 decides it from the process table
                # (`process_presence_is_busy`), so an idle A5 card stays idle
                # and free for leasing while its used HBM is reported as used.
                "hbm_used_mb": hbm_used_mb,
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
    inventory = device_occupancy_inventory(npus)

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
        **inventory,
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
