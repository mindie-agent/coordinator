"""Project captured DCMI and owned-process observations into exact NPU run windows.

[Role] Produce reproducible C2 receipts from a completed Stage 1 collection.
[Invariants] Only complete in-window reads supply values. Expected sample indices
come from the continuous collector schedule; missing reads remain coverage gaps.
[Timeouts & Locks] Local files only; no waits, locks, remote calls or retries.
[Side-Effect Policy] Writes derived receipts beside immutable raw evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


REQUIRED = ("power", "temperature", "current_freq", "freq_reduc_cause")


def load_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def flag(values):
    if any(value is True for value in values):
        return True
    return False if values and all(value is False for value in values) else None


def throttle(getter):
    value = getter.get("decoded") or {}
    if getter.get("ret") != 0 or value.get("state") != "observed":
        return None
    if any(value.get(name) is True for name in ("thermal_limit", "power_limit", "svfd")):
        return True
    return None if value.get("unknown_bits") else False


def getter_window(rows, name, low, high, interval_ns):
    points = []
    for row in rows:
        getter = row["cards"]["0"]["getters"].get(name, {})
        start, end = getter.get("started_at_ns"), getter.get("finished_at_ns")
        if start is not None and end is not None:
            points.append((row["sample_index"], start, end, getter))
    before = [point for point in points if point[1] < low]
    after = [point for point in points if point[2] > high]
    left = before[-1] if before else None
    right = after[0] if after else None
    # The immediately adjacent reads establish the getter's actual schedule phase.
    # An absent interior sample index remains expected, even if its whole row is absent.
    expected = list(range(left[0] + 1, right[0])) if left and right else []
    inside = [point for point in points if low <= point[1] and point[2] <= high]
    observed = [point for point in inside if point[3].get("ret") == 0 and not point[3].get("error")]
    observed_indices = {point[0] for point in observed}
    missing = sorted(set(expected) - observed_indices)
    coverage = (len(expected) - len(missing)) / len(expected) if expected else 0.0
    neighborhood = ([left] if left else []) + inside + ([right] if right else [])
    gaps = [(b[1] - a[1]) / 1e9 for a, b in zip(neighborhood, neighborhood[1:])]
    schedule_targets = [row["monotonic_ns"] - row["schedule_slip_ns"] -
                        row["sample_index"] * interval_ns for row in rows]
    schedule_consistent = len(set(schedule_targets)) == 1
    limitations = []
    if not left or not right:
        limitations.append(name + "_window_not_bracketed_by_collector")
    if not expected:
        limitations.append(name + "_no_scheduled_sample_point_in_window")
    if not schedule_consistent:
        limitations.append(name + "_collector_schedule_inconsistent")
    if missing:
        limitations.append(name + "_scheduled_reads_missing_or_failed")
    return {
        "coverage_basis": "scheduled_sample_points", "coverage_ratio": coverage,
        "schedule_consistent": schedule_consistent, "configured_interval_seconds": interval_ns / 1e9,
        "expected_sample_indices": expected, "observed_sample_indices": sorted(observed_indices),
        "missing_sample_indices": missing, "observed_count": len(observed),
        "first_read_start_ns": inside[0][1] if inside else None,
        "last_read_end_ns": inside[-1][2] if inside else None,
        "start_boundary_gap_seconds": (inside[0][1] - low) / 1e9 if inside else None,
        "end_boundary_gap_seconds": (high - inside[-1][2]) / 1e9 if inside else None,
        "max_adjacent_read_gap_seconds": max(gaps) if gaps else None,
        "neighbor_before": {"sample_index": left[0], "start_ns": left[1], "end_ns": left[2]} if left else None,
        "neighbor_after": {"sample_index": right[0], "start_ns": right[1], "end_ns": right[2]} if right else None,
        "reads": [{"sample_index": point[0], **point[3]} for point in inside],
        "measurement_limitations": limitations,
    }


def project(base):
    plan = json.loads((base / "planned-run.json").read_text())
    projections = json.loads((base / "comparison-stage-projections.json").read_text())
    dcmi = load_lines(base / "dcmi-metrics.jsonl")
    processes = load_lines(base / "process-stream.jsonl")
    meta = next(row for row in dcmi if row.get("type") == "meta")
    rows = [row for row in dcmi if row.get("type") == "sample"]
    assert meta["run_id"] == plan["run_id"]
    assert all(row["run_id"] == plan["run_id"] for row in rows + processes)
    assert meta["cards"] == [0] and meta["device_id"] == 0
    interval_ns = round(meta["interval_s"] * 1e9)
    out = base / "telemetry-windows"
    out.mkdir(exist_ok=True)
    index = {}
    for stage, projection in projections.items():
        run_id = projection["stage_run_id"]
        result = json.loads((base / "producer-receipts/stages" / stage / "result.json").read_text())
        assert result["run_id"] == run_id
        assert run_id.startswith(plan["run_id"] + "-")
        window = projection["projection"]["window"]
        low, high = math.ceil(window["start_ts"] * 1e9), math.floor(window["end_ts"] * 1e9)
        metrics = {name: getter_window(rows, name, low, high, interval_ns)
                   for name in rows[0]["cards"]["0"]["getters"]}
        process_reads = [row for row in processes if
                         low <= row.get("query_started_at_ns", -1) and
                         row.get("query_finished_at_ns", high + 1) <= high]
        interference = flag([row.get("resource_interference_detected") for row in process_reads])
        limitations = [message for name in REQUIRED for message in metrics[name]["measurement_limitations"]]
        if not process_reads:
            limitations.append("selected_npu_process_ownership_not_observed_in_window")
        if any(row.get("state") != "observed" for row in process_reads):
            limitations.append("selected_npu_process_query_unavailable")
        summary = {
            "run_id": run_id, "host": plan["host"], "devices": [0], "window": window,
            "coverage_ratio": min(metrics[name]["coverage_ratio"] for name in REQUIRED),
            "throttling_detected": flag([throttle(read) for read in metrics["freq_reduc_cause"]["reads"]]),
            "resource_interference_detected": interference,
            "freq_mhz_min": min((read["value"] for read in metrics["current_freq"]["reads"] if read["ret"] == 0), default=None),
            "power_max_w": max((read["value"] for read in metrics["power"]["reads"] if read["ret"] == 0), default=None),
            "temp_max_c": max((read["value"] for read in metrics["temperature"]["reads"] if read["ret"] == 0), default=None),
            "measurement_limitations": limitations,
            "evidence_ref": str(out / (stage + ".raw-window.json")),
        }
        raw = {
            "schema_version": "mindie-stage1-c2-raw-window/1", "run_id": run_id,
            "collector_run_id": plan["run_id"], "host": plan["host"], "physical_devices": [0],
            "window": window, "coverage_basis": "scheduled_sample_points",
            "coverage_description": "Completed scheduled getter reads within the exact window; no temporal interpolation or continuous-health assertion.",
            "measurement_projection": projection, "metrics": metrics,
            "process_scope": "Observed processes using selected physical NPU 0, with stable PID identity and the admitted job marker; excludes host CPU/network contention.",
            "process_reads": process_reads, "process_observed_count": len(process_reads),
            "sources": {name: {"path": str(base / name), "sha256": hashlib.sha256((base / name).read_bytes()).hexdigest()}
                        for name in ("dcmi-metrics.jsonl", "process-stream.jsonl", "comparison-stage-projections.json")},
            "collector_metadata": meta, "summary": summary,
        }
        write(out / (stage + ".raw-window.json"), raw)
        write(out / (stage + ".telemetry.json"), summary)
        index[stage] = summary
    write(out / "index.json", index)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    args = parser.parse_args()
    print(json.dumps(project(args.base), indent=2))
