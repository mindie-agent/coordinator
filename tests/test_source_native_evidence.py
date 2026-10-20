"""Real evidence-envelope and terminal-manifest projection contracts."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "live"))

from source_native_evidence import cache_facts, execution_times, normalize_receipt, project_manifest


def _proof(run_id="debug-evidence"):
    return {
        "run_id": run_id,
        "patch_digest": "1" * 64,
        "loaded_binary_sha256": "2" * 64,
        "loaded_module_path": "/view/libprobe.so",
        "calls": [{"input_id": "input", "call_path": "probe.call", "timestamp": 100.0}],
        "measurement_window": {"start_ts": 99.0, "end_ts": 101.0},
        "candidate_build_key": "build-" + "3" * 64,
        "evidence_ref": "proof.json",
    }


def test_cache_facts_reads_taskclient_evidence_envelope_and_store_log():
    payload = {"evidence": {"roles": [{"native_cache_receipt": {"hit": True, "cache_key": "a" * 64,
                                                                "linked_files": 7}}]}}
    facts = cache_facts(payload)
    assert facts["receipt"] == {"hit": True, "cache_key": "a" * 64, "linked_files": 7}

    store_only = {"evidence": {"roles": [{"logs": [{"native_cache": {
        "status": "stored", "cache_key": "b" * 64, "linked_files": 0}}]}]}}
    assert cache_facts(store_only)["receipt"] == {
        "hit": False, "cache_key": "b" * 64, "linked_files": 0}


def test_normalize_receipt_accepts_normalized_and_store_results():
    assert normalize_receipt({"hit": True, "cache_key": "c" * 64, "linked_files": 2}) == {
        "hit": True, "cache_key": "c" * 64, "linked_files": 2}
    assert normalize_receipt({"status": "hit", "cache_key": "d" * 64, "linked_files": 3}) == {
        "hit": True, "cache_key": "d" * 64, "linked_files": 3}


def test_projection_uses_actual_terminal_state_and_timestamps():
    row = {"created_at": 1_000.0, "phase_observed_at": {"running": 1_001.0, "succeeded": 1_010.0}}
    reply = {"state": "succeeded", "observed_at": 1_011.0}
    created, started, ended = execution_times(row, reply, "succeeded")
    assert created == "1970-01-01T00:16:40.000Z"
    assert started == "1970-01-01T00:16:41.000Z"
    assert ended == "1970-01-01T00:16:50.000Z"
    manifest, _proof_obj, bench = project_manifest(
        run_id="debug-projection", code={"source_head": "a" * 40, "snapshot_commit": "b" * 40,
                                         "dirty": False},
        proof=_proof("debug-projection"), created_at=created, started_at=started, ended_at=ended,
        terminal_status="passed", topology={"host": "host", "npu_count": 0},
        receipt={"hit": False, "cache_key": "e" * 64, "linked_files": 0},
        profile_summary_ref="ready-profile.json",
    )
    assert manifest["status"] == "passed"
    assert manifest["updated_at"] == ended
    assert manifest["native_cache_receipt"]["cache_key"] == "e" * 64
    assert set(bench) == {"run_id", "patch_digest", "loaded_binary_sha256", "loaded_module_path",
                          "calls", "evidence_ref"}
