"""Release microbenchmark for the actual multi-role placement solver.

[Role] Measure in-memory placement against a synthetic 64-runtime catalog.
[Invariants] Preserve distinct-host constraints and refuse partial placements.
[Timeouts & Locks] Fixed iteration count; no subprocess, network, or sleep.
[Side-Effect Policy] Write only the requested benchmark result JSON.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mindie_coordinator.placement import select_runtimes

ITERATIONS = 5000
CATALOG = [
    {
        "runtime_id": f"runtime-{index}",
        "host": f"host-{index // 4}",
        "state": "ready",
        "user": "benchmark",
        "recipe": "stable",
        "python_abi": "cp311",
        "soc": "Ascend910B" if index < 32 else "Ascend910C",
    }
    for index in range(64)
]
ROLES = [{"name": "prefill", "npu_count": 2}, {"name": "decode", "npu_count": 2}]
ENVIRONMENT = {"recipe": "stable", "python_abi": "cp311", "soc": "Ascend910C"}


def placement_pair() -> None:
    selected = select_runtimes(
        CATALOG, user="benchmark", roles=ROLES, environment=ENVIRONMENT,
        topology={"distinct_hosts": True},
    )
    assert selected["runtime_ids"] == ["runtime-32", "runtime-36"], selected
    unavailable = select_runtimes(
        CATALOG, user="benchmark", roles=ROLES, environment=ENVIRONMENT,
        topology={"distinct_hosts": True},
        busy_runtime_ids={f"runtime-{index}" for index in range(36, 64)},
    )
    assert unavailable["runtime_ids"] == [] and unavailable["reason"], unavailable


def main() -> None:
    for _ in range(100):
        placement_pair()
    started = perf_counter_ns()
    for _ in range(ITERATIONS):
        placement_pair()
    elapsed = perf_counter_ns() - started
    result = {
        "status": "pass",
        "metrics": [{
            "name": "placement_success_and_capacity_rejection",
            "unit": "us/op",
            "value": elapsed / ITERATIONS / 1000,
            "iterations": ITERATIONS,
        }],
    }
    Path(os.environ["MINDIE_BENCH_RESULT_PATH"]).write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
