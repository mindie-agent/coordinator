"""Read-only selected-NPU process membership and ownership stream.

[Role] Binds actual `proc-mem` PIDs to one admitted remote-dev job marker.
[Invariants] PID reuse, unknown formats and unavailable process metadata stay
unknown. A complete empty device process list is a measured empty observation.
[Timeouts & Locks] 4s per npu-smi call, finite duration, caller outer adds 5s.
[Side-Effect Policy] Reads npu-smi and /proc; prints JSONL and never kills work.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


PROCESS = re.compile(r"^Process id:\s*(\d+)\s+Process name:(.*?)\s+Process memory\(MB\):\s*(\d+)$")


def parse_process_memory(output):
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if lines == ["No process in device."]:
        return []
    rows = []
    for line in lines:
        match = PROCESS.fullmatch(line)
        if match is None:
            raise ValueError("unrecognized npu-smi proc-mem row: " + line)
        rows.append({"pid": int(match[1]), "name": match[2].strip(), "memory_mb": int(match[3])})
    if not rows:
        raise ValueError("empty proc-mem output is not an observed empty process table")
    return rows


def _process_identity(pid, marker):
    root = Path("/proc") / str(pid)
    first = (root / "stat").read_text().rsplit(") ", 1)[1].split()
    environment = (root / "environ").read_bytes().split(b"\0")
    last = (root / "stat").read_text().rsplit(") ", 1)[1].split()
    return {"pid": pid, "ppid": int(last[1]), "start_ticks": last[19],
            "identity_stable": first[19] == last[19] and first[1] == last[1],
            "job_marker_matches": ("REMOTE_DEV_JOB_TOKEN=" + marker).encode() in environment}


def ownership_from_chain(chain, *, complete):
    """A live child may receive its token after exec; a stable marked parent owns it."""
    if not chain or any(not item["identity_stable"] for item in chain):
        return None
    if any(child["ppid"] != parent["pid"] for child, parent in zip(chain, chain[1:])):
        return None
    if any(item["job_marker_matches"] for item in chain):
        return True
    return False if complete else None


def observe_ownership(pid, marker):
    chain = []
    try:
        current = pid
        for _ in range(32):
            if current <= 1:
                break
            item = _process_identity(current, marker)
            chain.append(item)
            if not item["identity_stable"] or item["job_marker_matches"]:
                break
            current = item["ppid"]
        # Recheck the entire observed parent chain after traversal. PID reuse
        # or reparenting during the query cannot become ownership evidence.
        for item in chain:
            check = _process_identity(item["pid"], marker)
            item["identity_stable"] = bool(item["identity_stable"] and check["identity_stable"]
                                           and (item["start_ticks"], item["ppid"])
                                           == (check["start_ticks"], check["ppid"]))
        owned = ownership_from_chain(chain, complete=current <= 1)
        return {"identity_stable": all(item["identity_stable"] for item in chain),
                "start_ticks": chain[0]["start_ticks"] if chain else None,
                "owned": owned, "ownership_basis": "stable_process_ancestry",
                "ownership_chain": chain}
    except (FileNotFoundError, ProcessLookupError, PermissionError) as exc:
        return {"identity_stable": False, "owned": None,
                "ownership_chain": chain, "error_type": type(exc).__name__}


def observe_processes(card, chip, marker):
    before = time.time_ns()
    result = subprocess.run(["npu-smi", "info", "-t", "proc-mem", "-i", str(card), "-c", str(chip)],
                            capture_output=True, text=True, timeout=4)
    after = time.time_ns()
    sample = {"query_started_at_ns": before, "query_finished_at_ns": after,
              "stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode,
              "marker_known": True, "processes": [], "resource_interference_detected": None}
    if result.returncode:
        return {**sample, "state": "unavailable", "error": "npu-smi exited nonzero"}
    try:
        processes = parse_process_memory(result.stdout)
    except ValueError as exc:
        return {**sample, "state": "unavailable", "error": str(exc)}
    for process in processes:
        process.update(observe_ownership(process["pid"], marker))
    if any(process["owned"] is False for process in processes):
        interference = True
    elif all(process["owned"] is True for process in processes):
        interference = False
    else:
        interference = None
    return {**sample, "state": "observed", "processes": processes,
            "resource_interference_detected": interference, "finished_at_ns": time.time_ns()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--marker-file", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=1230)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--chip", type=int, default=0)
    parser.add_argument("--marker-wait-seconds", type=float, default=0,
                        help="wait for an owned prepared-job marker before sampling")
    args = parser.parse_args()
    marker_deadline = time.monotonic() + args.marker_wait_seconds
    while not args.marker_file.exists() and time.monotonic() < marker_deadline:
        time.sleep(min(0.05, max(0, marker_deadline - time.monotonic())))
    marker = args.marker_file.read_text().strip()
    if not marker:
        raise ValueError("owned job marker is required")
    deadline = time.monotonic() + args.duration
    next_sample = time.monotonic()
    number = 0
    observed = 0
    while time.monotonic() < deadline and not args.stop_file.exists():
        try:
            sample = observe_processes(args.card, args.chip, marker)
        except subprocess.TimeoutExpired as exc:
            sample = {"state": "unavailable", "error": str(exc), "error_type": type(exc).__name__,
                      "resource_interference_detected": None}
        sample.update(run_id=args.run_id, card=args.card, chip=args.chip, sample_index=number)
        print(json.dumps(sample), flush=True)
        number += 1
        observed += sample["state"] == "observed"
        next_sample = max(next_sample + args.interval, time.monotonic())
        time.sleep(max(0, next_sample - time.monotonic()))
    print(json.dumps({"type": "summary", "run_id": args.run_id, "samples": number,
                      "observed_samples": observed, "unavailable_samples": number-observed,
                      "stopped": args.stop_file.exists()}), flush=True)
    return 0 if observed else 2


if __name__ == "__main__":
    raise SystemExit(main())
