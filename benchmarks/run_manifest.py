"""[Role] Measure canonical C3 manifest and bench projection for one actual C++ call.
[Invariants] The binary is compiled, loaded and called inside the recorded window.
[Timeouts & Locks] One compile (10s) and Git read (3s); bounded pure projection loop.
[Side-Effect Policy] A temporary CPU fixture and caller-requested result JSON only.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mindie_coordinator.run_manifest import (  # noqa: E402
    bench_execution_evidence,
    project_run_manifest,
    utc_from_epoch,
)

SOURCE = 'extern "C" long mindie_probe(long value) { return value + 1; }\n'


def main():
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, timeout=3,
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        text=True, timeout=3,
    ).strip())
    with tempfile.TemporaryDirectory(prefix="manifest-benchmark-") as directory:
        root = Path(directory)
        source, binary = root / "probe.cpp", root / "libprobe.so"
        source.write_text(SOURCE)
        subprocess.run(
            ["c++", "-shared", "-fPIC", "-o", str(binary), str(source)],
            check=True, timeout=10, capture_output=True,
        )
        library = ctypes.CDLL(str(binary))
        library.mindie_probe.argtypes = [ctypes.c_long]
        library.mindie_probe.restype = ctypes.c_long
        started = time.time()
        call_timestamp = time.time()
        assert library.mindie_probe(41) == 42
        ended = time.time()
        binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        proof = {
            "run_id": "release-cpu-cpp-fixture", "patch_digest": source_sha,
            "loaded_binary_sha256": binary_sha, "loaded_module_path": str(binary),
            "calls": [{"input_id": "integer-41", "call_path": "mindie_probe",
                       "timestamp": call_timestamp}],
            "measurement_window": {"start_ts": started, "end_ts": ended},
            "candidate_build_key": "build-" + source_sha,
            "evidence_ref": "synthetic-cpu-fixture",
        }

        def project():
            manifest = project_run_manifest(
                run_id=proof["run_id"],
                code={"source_head": head, "snapshot_commit": head, "dirty": dirty},
                created_at=utc_from_epoch(started), started_at=utc_from_epoch(started),
                ended_at=utc_from_epoch(ended), terminal_status="passed",
                command=["mindie_probe", "41"], topology={"local": True},
                active_execution_proof=proof,
            )
            evidence = bench_execution_evidence(
                manifest["active_execution_proof"], manifest_run_id=manifest["run_id"],
            )
            assert manifest["status"] == "passed"
            assert evidence["loaded_binary_sha256"] == binary_sha
            assert evidence["run_id"] == proof["run_id"]
            assert evidence["calls"][0]["input_id"] == "integer-41"

        project()
        iterations = 1000
        timer = time.perf_counter()
        for _ in range(iterations):
            project()
        elapsed = time.perf_counter() - timer
    Path(os.environ["MINDIE_BENCH_RESULT_PATH"]).write_text(json.dumps({
        "status": "pass", "metrics": [{
            "name": "canonical_run_manifest_and_bench_projection",
            "unit": "us/op", "value": elapsed * 1e6 / iterations,
            "iterations": iterations,
        }],
    }) + "\n")


if __name__ == "__main__":
    main()
