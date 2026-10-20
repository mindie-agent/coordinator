"""High-cadence, read-only cause-register evidence for an admitted NPU run.

[Role] Samples the production DCMI getter with the physical host's wall clock.
[Invariants] Keeps every raw return code and mask; no temperature inference.
[Timeouts & Locks] Finite duration; caller owns the SSH timeout and stop file.
[Side-Effect Policy] Reads the NPU driver and prints JSONL; no device setters.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-module", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=1230)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--card", type=int, default=0)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("dcmi_telemetry", args.probe_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    deadline = time.monotonic() + args.duration
    next_sample = time.monotonic()
    count = 0
    while time.monotonic() < deadline and not args.stop_file.exists():
        started = time.time()
        sample = module.run_probe([args.card])
        sample.update(run_id=args.run_id, started_at=started, finished_at=time.time(),
                      monotonic=time.monotonic(), interval_seconds=args.interval)
        print(json.dumps(sample), flush=True)
        count += 1
        next_sample += args.interval
        time.sleep(max(0, next_sample - time.monotonic()))
    print(json.dumps({"type": "finished", "run_id": args.run_id, "samples": count,
                      "stopped": args.stop_file.exists(), "finished_at": time.time()}), flush=True)


if __name__ == "__main__":
    main()
