"""Actual proof → Run Manifest v1 → bench ExecutionEvidence projection.

[Role]
Normalizes TaskClient evidence/cache payloads and projects an observed proof
into a terminal Run Manifest without inventing timestamps or cache receipts.

[Invariants]
- `TaskClient.observe(action="evidence")` wraps role data under `evidence`.
- A cache receipt is accepted from a normalized result or a store/miss result;
  the keyword-only manifest builder is never called with a result dict.
- Manifest status uses the actual terminal execution state and actual execution
  timestamps, never a caller-supplied success label.
- The bench projection is the same helper used by live and local proof paths.

[Side-Effect Policy]
- Pure construction; callers write files.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from mindie_coordinator.native_cache_pool import native_cache_receipt as cache_result_receipt
from mindie_coordinator.run_manifest import (
    bench_execution_evidence,
    execution_proof,
    project_run_manifest,
)

TERMINAL = {
    "succeeded": "passed",
    "failed": "failed",
    "cancelled": "cancelled",
    "inconclusive": "inconclusive",
    "timeout": "inconclusive",
}


def epoch_iso(value: Any) -> str:
    timestamp = float(value)
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


def execution_times(row: Mapping[str, Any], reply: Mapping[str, Any], state: str) -> tuple[str, str, str]:
    """Actual created/running/terminal timestamps from the persisted row."""
    created_ts = row.get("created_at") or reply.get("observed_at")
    observed = row.get("phase_observed_at") or {}
    running_ts = observed.get("running") or created_ts
    terminal_ts = observed.get(state) or observed.get(row.get("phase")) or reply.get("observed_at") or running_ts
    created = epoch_iso(created_ts)
    started = epoch_iso(max(float(created_ts), float(running_ts)))
    ended = epoch_iso(max(float(running_ts), float(terminal_ts)))
    return created, started, ended


def normalize_receipt(value: Any) -> dict:
    if not isinstance(value, Mapping):
        raise RuntimeError("native cache receipt is not an object")
    if "hit" in value:
        receipt = {"hit": bool(value.get("hit")), "cache_key": str(value.get("cache_key") or ""),
                   "linked_files": int(value.get("linked_files") or 0)}
    else:
        receipt = cache_result_receipt(dict(value))
    return receipt


def cache_facts(payload: Any, *, required: bool = True) -> dict:
    """Read the real `evidence` envelope, not a mock top-level shape."""
    if not isinstance(payload, Mapping):
        raise RuntimeError("TaskClient evidence is not an object")
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), Mapping) else payload
    receipt = None
    store = None
    for role in evidence.get("roles") or []:
        receipt = receipt or role.get("native_cache_receipt")
        for log in role.get("logs") or []:
            if isinstance(log.get("native_cache"), dict):
                store = log["native_cache"]
    if receipt is None and store is not None:
        receipt = cache_result_receipt(store)
    if receipt is None and not required:
        return {"receipt": None, "store": store}
    if not isinstance(receipt, Mapping):
        raise RuntimeError("no native cache receipt was recorded")
    return {"receipt": normalize_receipt(receipt), "store": store}


def project_manifest(*, run_id: str, code: Mapping[str, Any], proof: Mapping[str, Any],
                     created_at: str, started_at: str, ended_at: str, terminal_status: str,
                     topology: Mapping[str, Any], receipt: Mapping[str, Any] | None = None,
                     profile_summary_ref: str | None = None) -> tuple[dict, dict, dict]:
    if terminal_status not in {"passed", "failed", "inconclusive", "cancelled"}:
        raise RuntimeError("unsupported terminal status")
    proof_obj = execution_proof(
        run_id=run_id, patch_digest=proof["patch_digest"],
        loaded_binary_sha256=proof["loaded_binary_sha256"],
        loaded_module_path=proof["loaded_module_path"], calls=proof["calls"],
        measurement_window=proof["measurement_window"],
        candidate_build_key=proof["candidate_build_key"],
        evidence_ref=proof["evidence_ref"],
    )
    manifest = project_run_manifest(
        run_id=run_id, code=code, created_at=created_at, started_at=started_at,
        ended_at=ended_at, terminal_status=terminal_status,
        command=[proof["calls"][0]["call_path"]], topology=dict(topology),
        native_cache_receipt=receipt, active_execution_proof=proof_obj,
        profile_summary_ref=profile_summary_ref,
    )
    bench = bench_execution_evidence(proof_obj, manifest_run_id=run_id)
    if set(bench) != {"run_id", "patch_digest", "loaded_binary_sha256", "loaded_module_path",
                      "calls", "evidence_ref"}:
        raise RuntimeError("bench adapter shape changed")
    return manifest, proof_obj, bench
