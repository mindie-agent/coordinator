"""C3 additive evidence fields keep execution state semantics and run identity."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from mindie_coordinator.run_manifest import (
    BENCH_EXECUTION_EVIDENCE_FIELDS,
    RunManifestError,
    add_evidence,
    bench_execution_evidence,
    execution_proof,
    load_manifest,
    native_cache_receipt,
    new_manifest,
    transition_status,
    validate_manifest,
    write_manifest,
)

NOW = "2026-09-09T00:00:00Z"
LATER = "2026-09-09T00:00:05Z"
START_TS = 1_000.0
END_TS = 1_010.0
CODE = {"source_head": "a" * 40, "snapshot_commit": "b" * 40, "dirty": False}
SCHEMA = json.loads(
    (Path(__file__).parents[1] / "mindie_coordinator/schemas/run-manifest-v1.schema.json").read_text()
)
VALIDATOR = Draft202012Validator(SCHEMA)


def _telemetry(status="observed", limitations=None):
    return {
        "source": "npu-top",
        "status": status,
        "host": "10.0.0.1",
        "devices": [0, 1],
        "start_ts": 100,
        "end_ts": 200,
        "coverage_ratio": 1.0 if status == "observed" else 0.4,
        "throttling_detected": False if status == "observed" else None,
        "measurement_limitations": list(limitations or []),
    }


def _proof(run_id="perf-evidence-1", **overrides):
    values = {
        "run_id": run_id,
        "patch_digest": "1" * 64,
        "loaded_binary_sha256": "2" * 64,
        "loaded_module_path": "/view/vllm_ascend/_cann_ops_custom/libfixture.so",
        "calls": [
            {"input_id": "fixed-input-1", "call_path": "fixture.target",
             "timestamp": START_TS + 1.0},
            {"input_id": "fixed-input-2", "call_path": "fixture.target",
             "timestamp": START_TS + 2.0},
        ],
        "measurement_window": {"start_ts": START_TS, "end_ts": END_TS},
        "candidate_build_key": "build-" + "c" * 64,
        "evidence_ref": "artifacts/execution-evidence.json",
    }
    values.update(overrides)
    return execution_proof(**values)


def test_schema_and_validator_accept_all_optional_evidence():
    manifest = new_manifest(
        run_type="performance", run_id="perf-evidence-1", created_at=NOW, code=CODE,
        hardware_telemetry=_telemetry(),
        native_cache_receipt=native_cache_receipt(hit=True, cache_key="f" * 64, linked_files=12),
        active_execution_proof=_proof(),
        profile_summary_ref="artifacts/profile-summary.json",
    )
    VALIDATOR.validate(manifest)
    validate_manifest(manifest)
    assert manifest["run_id"] == "perf-evidence-1"
    assert manifest["native_cache_receipt"]["linked_files"] == 12


def test_optional_evidence_is_not_required_and_round_trips(tmp_path):
    manifest = new_manifest(run_type="debug", run_id="debug-no-evidence", created_at=NOW, code=CODE)
    VALIDATOR.validate(manifest)
    assert "hardware_telemetry" not in manifest
    manifest = add_evidence(
        manifest,
        hardware_telemetry=_telemetry(),
        native_cache_receipt={"hit": False, "cache_key": "a" * 64, "linked_files": 0},
        active_execution_proof=_proof(run_id="debug-no-evidence"),
        profile_summary_ref={"uri": "artifacts/profile.json", "sha256": "b" * 64},
        updated_at=LATER,
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    assert load_manifest(path) == manifest
    VALIDATOR.validate(json.loads(path.read_text()))


def test_add_evidence_never_replaces_submitted_run_id_or_terminal_status():
    manifest = new_manifest(run_type="correctness", run_id="correctness-submitted-1",
                            created_at=NOW, code=CODE)
    running = transition_status(manifest, "running", updated_at=NOW)
    passed = transition_status(running, "passed", updated_at=NOW)
    updated = add_evidence(
        passed,
        hardware_telemetry=_telemetry("unavailable", ["npu-top telemetry-window API is not installed (HTTP 404)"]),
        native_cache_receipt={"hit": True, "cache_key": "c" * 64, "linked_files": 3},
        updated_at=LATER,
    )
    assert updated["run_id"] == "correctness-submitted-1"
    assert updated["status"] == "passed"
    assert updated["created_at"] == NOW
    assert updated["updated_at"] == LATER
    # A monitor outage is evidence, not a reason to rewrite a successful run.
    validate_manifest(updated)
    VALIDATOR.validate(updated)


def test_unavailable_telemetry_requires_visible_limitations():
    manifest = new_manifest(run_type="performance", run_id="perf-gap-1", created_at=NOW, code=CODE)
    with pytest.raises(RunManifestError, match="must explain"):
        add_evidence(manifest, hardware_telemetry=_telemetry("unavailable", []))
    partial = add_evidence(
        manifest,
        hardware_telemetry=_telemetry("partial", ["coverage is incomplete (0.400)"]),
        updated_at=LATER,
    )
    assert partial["hardware_telemetry"]["measurement_limitations"] == ["coverage is incomplete (0.400)"]
    assert partial["hardware_telemetry"]["throttling_detected"] is None


def test_native_cache_receipt_rejects_impossible_hit_or_counter():
    manifest = new_manifest(run_type="performance", run_id="perf-receipt-1", created_at=NOW, code=CODE)
    for receipt in (
        {"hit": True, "cache_key": "", "linked_files": 0},
        {"hit": False, "cache_key": "a" * 64, "linked_files": -1},
        {"hit": True, "cache_key": "a" * 64, "linked_files": 1, "extra": True},
    ):
        with pytest.raises(RunManifestError, match="native_cache_receipt"):
            add_evidence(manifest, native_cache_receipt=receipt)
    with pytest.raises(RunManifestError, match="cache_key"):
        native_cache_receipt(hit=True, cache_key="", linked_files=0)


def test_requested_command_alone_cannot_be_execution_proof():
    manifest = new_manifest(run_type="correctness", run_id="correctness-proof-1",
                            created_at=NOW, code=CODE)
    requested_only = {
        "run_id": "correctness-proof-1",
        "requested_command": ["python", "run.py"],
    }
    with pytest.raises(RunManifestError, match="missing"):
        add_evidence(manifest, active_execution_proof=requested_only)
    with pytest.raises(RunManifestError, match="unknown"):
        add_evidence(manifest, active_execution_proof={
            **_proof(run_id="correctness-proof-1"), "requested_command": ["python", "run.py"]})
    # Schema validation independently rejects a bare requested command.
    invalid = {**manifest, "active_execution_proof": requested_only}
    assert list(VALIDATOR.iter_errors(invalid))


def test_execution_proof_uses_bench_canonical_observed_fields():
    proof = _proof(run_id="perf-proof-1")
    manifest = new_manifest(run_type="performance", run_id="perf-proof-1",
                            created_at=NOW, code=CODE, active_execution_proof=proof)
    assert manifest["active_execution_proof"]["loaded_binary_sha256"] == "2" * 64
    assert manifest["active_execution_proof"]["measurement_window"] == {
        "start_ts": START_TS, "end_ts": END_TS}
    assert manifest["active_execution_proof"]["calls"][0] == {
        "input_id": "fixed-input-1", "call_path": "fixture.target",
        "timestamp": START_TS + 1.0}
    VALIDATOR.validate(manifest)
    with pytest.raises(RunManifestError, match="sha256"):
        _proof(run_id="perf-proof-1", loaded_binary_sha256="not-a-digest")
    with pytest.raises(RunManifestError, match="calls"):
        _proof(run_id="perf-proof-1", calls=[])
    with pytest.raises(RunManifestError, match="finite|inside"):
        _proof(run_id="perf-proof-1", calls=[
            {"input_id": "fixed-input-1", "call_path": "fixture.target",
             "timestamp": END_TS + 1.0}])


def test_proof_run_id_must_match_manifest_run_id():
    manifest = new_manifest(run_type="performance", run_id="perf-proof-2", created_at=NOW, code=CODE)
    with pytest.raises(RunManifestError, match="must equal manifest.run_id"):
        add_evidence(manifest, active_execution_proof=_proof(run_id="another-run"))
    proof = _proof(run_id="perf-proof-2")
    with pytest.raises(RunManifestError, match="must equal manifest.run_id"):
        bench_execution_evidence(proof, manifest_run_id="another-run")


def test_each_call_timestamp_must_fall_inside_measured_window():
    with pytest.raises(RunManifestError, match="inside measurement_window"):
        _proof(run_id="perf-window-1", calls=[
            {"input_id": "fixed-input-1", "call_path": "fixture.target",
             "timestamp": START_TS - 0.5}])
    with pytest.raises(RunManifestError, match="positive duration"):
        _proof(run_id="perf-window-1", measurement_window={
            "start_ts": END_TS, "end_ts": END_TS})


def test_bench_adapter_emits_only_execution_evidence_and_keeps_patch_separate():
    proof = _proof(run_id="bench-proof-1", patch_digest="9" * 64)
    evidence = bench_execution_evidence(proof, manifest_run_id="bench-proof-1")
    assert set(evidence) == set(BENCH_EXECUTION_EVIDENCE_FIELDS)
    assert {"candidate_build_key", "measurement_window"} & set(evidence) == set()
    assert set(evidence["calls"][0]) == {"input_id", "call_path", "timestamp"}
    # A patch digest is not a build key: both identities stay distinct.
    assert evidence["patch_digest"] == "9" * 64
    assert proof["candidate_build_key"] == "build-" + "c" * 64
    assert evidence["patch_digest"] != proof["candidate_build_key"]
    # Changing the patch alone never mutates the candidate build identity.
    changed = _proof(run_id="bench-proof-1", patch_digest="8" * 64)
    assert changed["candidate_build_key"] == proof["candidate_build_key"]
