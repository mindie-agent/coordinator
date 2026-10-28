"""Agent-facing NPU, server, container, and mount probe for mindie-coordinator.

[Role]
Executes a single-pass agentless SSH probe (`REMOTE_FLEET_PROBE_SCRIPT`) across
Ascend NPU hosts, owns the local snapshot cache, device recommendation, and
fleet capacity ranking. Pure parsing lives in `fleet_probe_parser`, re-exported
here so existing `fleet_probe.parse_*` imports keep working.

[Invariants]
- `npu-top` observations only order host probes when no explicit host was
  requested. SSH/NPU queue evidence still decides actual availability; a
  monitor outage is reported in `npu_top_observation`, never as zero capacity.
- Dual-die 910C (A3) physical IDs (`phy_id`) drive `ASCEND_RT_VISIBLE_DEVICES`.
- `probe_machine_record` never masks a failed live probe as an online host.

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
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from mindie_coordinator.fleet_probe_parser import (
    BUSY_AICORE_THRESHOLD_PCT,
    BUSY_HBM_THRESHOLD_MB,
    WEIGHT_MOUNT_KEYWORDS,
    _DISK_WAIT_WCHAN_TOKENS,
    _RE_CONTAINER_ID_64,
    _RE_DOCKER_PORT,
    _RE_NPU_VERSION,
    _RE_USAGE_AICORE,
    _RE_USAGE_CAPACITY,
    _RE_USAGE_RATE,
    _a5_single_column_rows,
    _is_a5_single_column_layout,
    _parse_float,
    _parse_int,
    _split_row,
    device_occupancy_inventory,
    is_disk_io_wait,
    is_npu_busy,
    parse_fleet_probe_output,
    parse_npu_smi_info,
    parse_npu_smi_usages,
)
from mindie_coordinator.machine_directory import MachineDirectory, MachineDirectoryUnavailable
from mindie_coordinator.npu_top_bridge import NpuTopObservationClient, observation_host_order
from mindie_coordinator.state_paths import coordinator_state_dir

PROBE_CACHE_FILENAME = "probe_cache.json"
DEFAULT_CACHE_TTL_SECONDS = 30.0

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
            **({"auth_ref": host_spec["auth_ref"]} if host_spec.get("auth_ref") else {}),
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
    except TimeoutError:
        raise
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
        **({"auth_ref": host_info["auth_ref"]} if host_info.get("auth_ref") else {}),
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
                # Rebuild card *and* physical-Die inventory from the same
                # derivation, so a supplement can never leave a busy card with
                # its Phy-ID still listed free.
                snapshot.update(device_occupancy_inventory(updated_npus))
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
