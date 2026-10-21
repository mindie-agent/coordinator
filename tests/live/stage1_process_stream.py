"""Read-only selected-NPU process membership and ownership stream.

[Role] Binds actual `proc-mem` PIDs to one admitted remote-dev job marker.
[Invariants] PID reuse, unknown formats and unavailable process metadata stay
unknown. A complete empty device process list is a measured empty observation,
which includes A5 (950DT) header-only/empty sections and A2/A3 whole-card
headers; only exit 0, empty stderr and recognized header/sentinel/process lines
make that claim. Non-zero exit, stderr, an unrecognized line or a reported NPU
ID that disagrees with the requested card stays unavailable.
[Timeouts & Locks] 4s per npu-smi call and at most two calls per sample (A5
rejects the A2/A3 `-c` selector and is re-read without it); finite duration,
caller outer timeout adds 5s.
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
SENTINEL = "No process in device."
CARD_HEADER = re.compile(r"^NPU ID\s*:\s*(\d+)$")
CHIP_COUNT_HEADER = re.compile(r"^Chip Count\s*:\s*\d+$")
CHIP_HEADER = re.compile(r"^Chip ID\s*:\s*\d+$")
CHIP_SELECTOR_UNSUPPORTED = "does not support input parameter of -c"


def parse_process_memory(output, *, returncode=0, stderr="", card=None):
    """Parse one `npu-smi info -t proc-mem` result.

    Exit 0 with no process rows is a measured empty table: A5 prints only
    `NPU ID : N` (25.1.rc3 adds `No process in device.`) and A2/A3 whole-card
    queries print `NPU ID`/`Chip Count`/`No process in device.`/`Chip ID`.
    Anything unrecognized, a non-zero exit, or stderr keeps the observation
    unavailable rather than turning it into an idle claim.
    """
    if returncode != 0:
        raise ValueError(f"npu-smi proc-mem exited {returncode}")
    if stderr.strip():
        raise ValueError("npu-smi proc-mem wrote to stderr: " + stderr.strip()[-300:])
    rows = []
    observed_cards = set()
    for line in (line.strip() for line in output.splitlines() if line.strip()):
        if line == SENTINEL:
            continue
        match = PROCESS.fullmatch(line)
        if match is not None:
            rows.append({"pid": int(match[1]), "name": match[2].strip(), "memory_mb": int(match[3])})
            continue
        header = CARD_HEADER.fullmatch(line)
        if header is not None:
            observed_cards.add(int(header[1]))
            continue
        if CHIP_COUNT_HEADER.fullmatch(line) or CHIP_HEADER.fullmatch(line):
            continue
        raise ValueError("unrecognized npu-smi proc-mem row: " + line)
    if card is not None and observed_cards and observed_cards != {int(card)}:
        raise ValueError(
            f"npu-smi proc-mem reported NPU {sorted(observed_cards)} for requested card {card}")
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


def _query_proc_memory(card, chip):
    command = ["npu-smi", "info", "-t", "proc-mem", "-i", str(card)]
    if chip is not None:
        command += ["-c", str(chip)]
    return subprocess.run(command, capture_output=True, text=True, timeout=4)


def observe_processes(card, chip, marker):
    before = time.time_ns()
    result = _query_proc_memory(card, chip)
    selected_chip = chip
    if result.returncode and CHIP_SELECTOR_UNSUPPORTED in (result.stdout + result.stderr):
        # A5 (950DT) exposes one process table per card and rejects the A2/A3
        # `-c` selector. Re-read the per-card table; the result is parsed with
        # the same exit-code/stderr/row validation, so a failed second query
        # still cannot become an idle observation.
        result = _query_proc_memory(card, None)
        selected_chip = None
    after = time.time_ns()
    sample = {"query_started_at_ns": before, "query_finished_at_ns": after,
              "stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode,
              "query_card": card, "query_chip": selected_chip,
              "marker_known": True, "processes": [], "resource_interference_detected": None}
    try:
        processes = parse_process_memory(
            result.stdout, returncode=result.returncode, stderr=result.stderr, card=card)
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
