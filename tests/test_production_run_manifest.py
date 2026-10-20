"""Actual C++ proof → production manifest → bench projection and cold receipt."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from mindie_coordinator.ready_runtime import RuntimePool
from mindie_coordinator.execution_sources import capture_sources
from mindie_coordinator.run_manifest import (
    bench_execution_evidence,
    project_run_manifest,
    utc_from_epoch,
    utc_now,
)


def _compiled_fixture(tmp_path: Path, *, dirty: bool = False) -> tuple[Path, str, str]:
    repo = tmp_path / "fixture"
    repo.mkdir()
    source = repo / "probe.cpp"
    source.write_text('extern "C" long mindie_probe(long value) { return value + 1; }\n',
                      encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, timeout=10)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=repo, check=True, timeout=10)
    subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], cwd=repo, check=True, timeout=10)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, timeout=10)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True, timeout=10)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, timeout=10).strip()
    if dirty:
        source.write_text(source.read_text() + "// Accepted uncommitted fixture edit.\n", encoding="utf-8")
    artifact = repo / "libprobe.so"
    subprocess.run(["c++", "-shared", "-fPIC", "-o", str(artifact), str(source)], check=True, timeout=10)
    library = ctypes.CDLL(str(artifact))
    library.mindie_probe.argtypes = [ctypes.c_long]
    library.mindie_probe.restype = ctypes.c_long
    assert library.mindie_probe(41) == 42
    return artifact, commit, hashlib.sha256(source.read_bytes()).hexdigest()


def _proof(artifact: Path, patch_digest: str, run_id: str) -> dict:
    loaded = artifact.resolve()
    library = ctypes.CDLL(str(loaded))
    library.mindie_probe.argtypes = [ctypes.c_long]
    library.mindie_probe.restype = ctypes.c_long
    start = time.time()
    call_ts = time.time()
    probe_value = library.mindie_probe(41)
    end = time.time()
    assert probe_value == 42
    return {
        "run_id": run_id,
        "patch_digest": patch_digest,
        "loaded_binary_sha256": hashlib.sha256(loaded.read_bytes()).hexdigest(),
        "loaded_module_path": str(loaded),
        "calls": [{"input_id": "fixture-input", "call_path": "mindie_probe", "timestamp": call_ts}],
        "measurement_window": {"start_ts": start, "end_ts": end},
        "candidate_build_key": "build-" + "c" * 64,
        "evidence_ref": "fixture-proof.json",
        "probe_value": probe_value,
    }


@pytest.mark.skipif(os.name != "posix", reason="The actual Linux C++ proof requires a POSIX compiler")
def test_actual_cpp_proof_projects_terminal_manifest_and_bench(tmp_path):
    artifact, commit, patch = _compiled_fixture(tmp_path)
    proof = _proof(artifact, patch, "debug-production-proof")
    manifest = project_run_manifest(
        run_id=proof["run_id"],
        code={"source_head": commit, "snapshot_commit": commit, "dirty": False},
        created_at=utc_from_epoch(proof["measurement_window"]["start_ts"]),
        started_at=utc_from_epoch(proof["measurement_window"]["start_ts"]),
        ended_at=utc_from_epoch(proof["measurement_window"]["end_ts"]),
        terminal_status="passed",
        command=["mindie_probe"],
        topology={"local": True},
        active_execution_proof=proof,
    )
    assert manifest["status"] == "passed"
    assert manifest["active_execution_proof"]["loaded_binary_sha256"] == proof["loaded_binary_sha256"]
    assert "probe_value" not in manifest["active_execution_proof"]
    bench = bench_execution_evidence(manifest["active_execution_proof"], manifest_run_id=manifest["run_id"])
    assert bench["loaded_binary_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()


@pytest.mark.skipif(os.name != "posix", reason="The actual Linux C++ proof requires a POSIX compiler")
def test_completed_job_exports_manifest_bench_and_cold_store_receipt(tmp_path):
    artifact, commit, patch = _compiled_fixture(tmp_path, dirty=True)
    accepted = capture_sources({"vllm-ascend": str(artifact.parent)}, tmp_path / "sources")
    source = next(item for item in accepted["records"] if item["relpath"] == "vllm-ascend")
    assert source["source_head"] == commit and source["changed_paths"]
    run_id = "debug-production-record"
    proof = _proof(artifact, patch, run_id)
    state = tmp_path / "state"
    pool = RuntimePool(state, backend=object())
    execution_id = run_id
    role_dir = state / "runs" / execution_id / "default"
    role_dir.mkdir(parents=True)
    cache_key = "d" * 64
    (role_dir / "finalize-runtime.log").write_text(json.dumps({
        "native_cache": {"status": "stored", "cache_key": cache_key, "bundle": "e" * 64,
                         "linked_files": 0}}) + "\n", encoding="utf-8")
    start = time.time()
    end = start + 1.0
    run = {"id": "f" * 64, "created_at": utc_now(), "state": "released", "task_id": "task",
           "epoch": "epoch", "intent": {"snapshots": {"vllm-ascend": source["commit"]}}}
    binding = {"profile_key": "profile", "build_key": "build", "runtime_id": "runtime",
               "endpoint": {"host": "host", "root": "/root"}}
    job = {"id": run["id"], "job_id": "mindie-job", "execution_id": execution_id,
           "state": "succeeded",
           "spec": {"command": "python -c 'import vllm_ascend'", "source_snapshot": accepted},
           "remote": {"result": {"started_at": start, "finished_at": end}},
           "active_execution_proof": proof,
           "hardware_telemetry": {"source": "npu-top", "status": "partial",
                                   "measurement_limitations": ["fixture"]}}
    pool.export_execution_record(run, binding, job=job)
    record = json.loads((state / "runs" / (run["id"] + ".json")).read_text())
    manifest = json.loads((state / "runs" / (run["id"] + ".run-manifest.json")).read_text())
    bench = json.loads((state / "runs" / (run["id"] + ".bench-execution-evidence.json")).read_text())
    assert record["run_manifest"]["status"] == "passed"
    assert manifest["status"] == "passed"
    assert manifest["code"] == {"source_head": commit, "snapshot_commit": source["commit"], "dirty": True}
    assert manifest["native_cache_receipt"] == {"hit": False, "cache_key": cache_key, "linked_files": 0}
    assert manifest["active_execution_proof"]["loaded_binary_sha256"] == proof["loaded_binary_sha256"]
    assert "probe_value" not in manifest["active_execution_proof"]
    assert bench["run_id"] == manifest["run_id"]


@pytest.mark.skipif(os.name != "posix", reason="The actual Linux C++ proof requires a POSIX compiler")
def test_missing_proof_is_explicitly_unavailable_not_fabricated(tmp_path):
    artifact, commit, patch = _compiled_fixture(tmp_path)  # noqa: F841 - fixture is actual
    state = tmp_path / "state"
    pool = RuntimePool(state, backend=object())
    accepted = capture_sources({"vllm-ascend": str(artifact.parent)}, tmp_path / "sources")
    source = next(item for item in accepted["records"] if item["relpath"] == "vllm-ascend")
    run = {"id": "a" * 64, "created_at": utc_now(), "state": "released", "task_id": "task",
           "epoch": "epoch", "intent": {"snapshots": {"vllm-ascend": source["commit"]}}}
    binding = {"profile_key": "profile", "build_key": "build", "runtime_id": "runtime",
               "endpoint": {"host": "host", "root": "/root"}}
    start = time.time()
    job = {"id": run["id"], "job_id": "mindie-missing", "execution_id": "debug-missing-proof",
           "state": "succeeded",
           "spec": {"command": "true", "source_snapshot": accepted},
           "remote": {"result": {"started_at": start, "finished_at": start + 1.0}}}
    pool.export_execution_record(run, binding, job=job)
    manifest = json.loads((state / "runs" / (run["id"] + ".run-manifest.json")).read_text())
    assert "active_execution_proof" not in manifest
    assert not (state / "runs" / (run["id"] + ".bench-execution-evidence.json")).exists()
    job["spec"].pop("source_snapshot")
    record = {"environment": {}, "resources": {}}
    pool._attach_run_manifest(record, run, binding, job)
    assert "run_manifest" not in record
    assert "original Git identity is unavailable" in record["run_manifest_unavailable"]


def test_service_role_evidence_promotes_actual_cold_store_log(tmp_path):
    from mindie_coordinator.service import CoordinatorService

    service = CoordinatorService.__new__(CoordinatorService)
    service.state_dir = tmp_path / "coordinator"
    role = service.state_dir / "runs" / "execution" / "default"
    role.mkdir(parents=True)
    cache_key = "9" * 64
    (role / "finalize-runtime.log").write_text(json.dumps({
        "native_cache": {"status": "stored", "cache_key": cache_key,
                         "bundle": "8" * 64, "linked_files": 0}}) + "\n", encoding="utf-8")
    row = {"id": "execution", "roles": [], "role_progress": {}, "preparation_results": {}}
    evidence = service._role_evidence(row, "default", None)
    assert evidence["native_cache_receipt"] == {
        "hit": False, "cache_key": cache_key, "linked_files": 0}
