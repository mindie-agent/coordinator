#!/usr/bin/env python3
"""Run Manifest v1: Git-identity evidence records for workspace workflows.

[Role] Validates and writes immutable Git-identity run manifests (`source_head`, `snapshot_commit`).
[Invariants] `code` is always Git commit identity, never a content hash of source files.
- Optional C3 evidence (`hardware_telemetry`, `native_cache_receipt`,
  `active_execution_proof`, `profile_summary_ref`) is additive. It never creates
  a second execution state machine or changes `passed / failed / inconclusive /
  cancelled` semantics; a submitted `run_id` is never replaced.
- `active_execution_proof` is the canonical observed execution shape consumed by
  mindie-bench (`run_id`, `patch_digest`, `loaded_binary_sha256`,
  `loaded_module_path`, `calls[]`, `evidence_ref`) plus the C3
  `candidate_build_key` and numeric `measurement_window`. Every call timestamp
  must fall inside that measured window, and proof.run_id must equal manifest
  run_id. `patch_digest` is a content identity, never a native build key. A
  requested command alone is not proof.
[Timeouts & Locks] Atomic temp-file write + `os.replace` on local filesystem; no network waits.
[Side-Effect Policy] Local manifest file creation only.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

SCHEMA_VERSION = 1
RUN_TYPES = frozenset(
    {
        "change-validation",
        "correctness",
        "debug",
        "performance",
        "profiling",
    }
)
RUN_STATUSES = frozenset(
    {"planned", "running", "passed", "failed", "inconclusive", "cancelled"}
)
TERMINAL_STATUSES = frozenset({"passed", "failed", "inconclusive", "cancelled"})
STATUS_TRANSITIONS = {
    "planned": frozenset({"running", "failed", "cancelled", "inconclusive"}),
    "running": frozenset({"passed", "failed", "inconclusive", "cancelled"}),
    "passed": frozenset(),
    "failed": frozenset(),
    "inconclusive": frozenset(),
    "cancelled": frozenset(),
}
SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
RFC3339_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
ZERO_SHA = "0" * 40
OPTIONAL_EVIDENCE_FIELDS = (
    "hardware_telemetry",
    "native_cache_receipt",
    "active_execution_proof",
    "profile_summary_ref",
)
PROOF_FIELDS = frozenset({
    "run_id", "patch_digest", "loaded_binary_sha256", "loaded_module_path",
    "calls", "measurement_window", "candidate_build_key", "evidence_ref",
})
BENCH_EXECUTION_EVIDENCE_FIELDS = frozenset({
    "run_id", "patch_digest", "loaded_binary_sha256", "loaded_module_path",
    "calls", "evidence_ref",
})
_UNSET = object()
SECRET_ENV_RE = re.compile(
    r"(?:^|_)(?:API_?KEY|ACCESS_?KEY|AUTH|CREDENTIAL|"
    r"PASS(?:WD|WORD)?|SECRET|TOKEN|KEY)(?:_|$)",
    re.IGNORECASE,
)


class RunManifestError(ValueError):
    """Raised when a manifest violates the v1 contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def utc_from_epoch(value: Any) -> str:
    return datetime.fromtimestamp(float(value), timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


def generate_run_id(run_type: str, *, now: str | None = None) -> str:
    if run_type not in RUN_TYPES:
        raise RunManifestError(f"unsupported run_type: {run_type!r}")
    timestamp = (now or utc_now()).replace("-", "").replace(":", "")
    timestamp = timestamp.replace("T", "-").replace("Z", "").split(".", 1)[0]
    return f"{run_type}-{timestamp}-{uuid.uuid4().hex[:8]}"


def _resolve_code(
    code: Mapping[str, Any] | None,
    workspace_root: Path | str | None,
) -> dict[str, Any]:
    if code is not None:
        return {
            "source_head": code["source_head"],
            "snapshot_commit": code["snapshot_commit"],
            "dirty": bool(code["dirty"]),
        }
    if workspace_root is not None:
        from mindie_coordinator.code_identity import manifest_code

        return manifest_code(workspace_root)
    raise RunManifestError(
        "code identity is required: pass code= or workspace_root=; "
        "a fabricated all-zero SHA is not an identity"
    )


def new_manifest(
    *,
    run_type: str,
    run_id: str | None = None,
    parent_run_id: str | None = None,
    code: Mapping[str, Any] | None = None,
    workspace_root: Path | str | None = None,
    workspace_snapshot: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    model: Mapping[str, Any] | None = None,
    topology: Mapping[str, Any] | None = None,
    command: Sequence[str] | None = None,
    environment_variables: Mapping[str, str] | None = None,
    hardware_telemetry: Mapping[str, Any] | None = None,
    native_cache_receipt: Mapping[str, Any] | None = None,
    active_execution_proof: Mapping[str, Any] | None = None,
    profile_summary_ref: Any = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    timestamp = created_at or utc_now()
    optional = {
        "hardware_telemetry": hardware_telemetry,
        "native_cache_receipt": native_cache_receipt,
        "active_execution_proof": active_execution_proof,
        "profile_summary_ref": profile_summary_ref,
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id or generate_run_id(run_type, now=timestamp),
        "parent_run_id": parent_run_id,
        "run_type": run_type,
        "code": _resolve_code(code, workspace_root),
        "workspace_snapshot": dict(workspace_snapshot or {}),
        "environment": dict(environment or {}),
        "model": dict(model or {}),
        "topology": dict(topology or {}),
        "command": list(command or []),
        "environment_variables": dict(environment_variables or {}),
        "artifacts": [],
        "status": "planned",
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    manifest.update({field: deepcopy(value) for field, value in optional.items() if value is not None})
    validate_manifest(manifest)
    return manifest


def _require_mapping(value: Any, path: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{path} must be an object")
        return {}
    return value


def _is_json_native(value: Any) -> bool:
    """True when ``value`` survives JSON dump/load without coercion.

    The published schema is the contract; this check is a strict subset
    so integer keys, NaN/Inf, tuples and other non-JSON values cannot
    validate and then round-trip as something else.
    """
    if value is None or isinstance(value, str):
        return True
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_native(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_native(item)
            for key, item in value.items()
        )
    return False


def _validate_safe_id(value: Any, path: str, errors: list[str], *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str) or not SAFE_ID_RE.fullmatch(value):
        errors.append(f"{path} must match {SAFE_ID_RE.pattern}")


def _validate_artifacts(value: Any, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append("artifacts must be an array")
        return
    names: set[str] = set()
    for index, artifact in enumerate(value):
        item = _require_mapping(artifact, f"artifacts[{index}]", errors)
        for field in ("name", "kind", "uri"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                errors.append(f"artifacts[{index}].{field} must be a non-empty string")
        name = item.get("name")
        if isinstance(name, str):
            if name in names:
                errors.append(f"artifact name is duplicated: {name!r}")
            names.add(name)
        unknown = sorted(set(item) - {"name", "kind", "uri"})
        if unknown:
            errors.append(
                f"artifacts[{index}] has unknown fields: {', '.join(unknown)}"
            )


def _validate_native_cache_receipt(value: Any, errors: list[str]) -> None:
    item = _require_mapping(value, "native_cache_receipt", errors)
    if not item:
        return
    unknown = sorted(set(item) - {"hit", "cache_key", "linked_files"})
    if unknown:
        errors.append("native_cache_receipt has unknown fields: " + ", ".join(unknown))
    if type(item.get("hit")) is not bool:
        errors.append("native_cache_receipt.hit must be a boolean")
    cache_key = item.get("cache_key")
    if not isinstance(cache_key, str) or len(cache_key) > 256:
        errors.append("native_cache_receipt.cache_key must be a string of at most 256 characters")
    elif item.get("hit") is True and not cache_key:
        errors.append("native_cache_receipt.cache_key is required for a hit")
    linked = item.get("linked_files")
    if type(linked) is not int or linked < 0:
        errors.append("native_cache_receipt.linked_files must be a nonnegative integer")


def _validate_hardware_telemetry(value: Any, errors: list[str]) -> None:
    item = _require_mapping(value, "hardware_telemetry", errors)
    if not item:
        return
    if not isinstance(item.get("source"), str) or not item["source"].strip():
        errors.append("hardware_telemetry.source must be a nonempty string")
    status = item.get("status")
    if status not in {"observed", "partial", "unavailable"}:
        errors.append("hardware_telemetry.status must be observed, partial, or unavailable")
    limitations = item.get("measurement_limitations")
    if not isinstance(limitations, list) or any(not isinstance(row, str) or not row.strip() for row in limitations):
        errors.append("hardware_telemetry.measurement_limitations must be an array of nonempty strings")
    elif status in {"partial", "unavailable"} and not limitations:
        errors.append("hardware_telemetry.measurement_limitations must explain a partial or unavailable observation")
    if "coverage_ratio" in item:
        coverage = item["coverage_ratio"]
        if isinstance(coverage, bool) or not isinstance(coverage, (int, float)) or not 0.0 <= float(coverage) <= 1.0:
            errors.append("hardware_telemetry.coverage_ratio must be a number from 0 to 1")
    if "throttling_detected" in item and item["throttling_detected"] is not None:
        if type(item["throttling_detected"]) is not bool:
            errors.append("hardware_telemetry.throttling_detected must be a boolean or null")
    if "devices" in item:
        devices = item["devices"]
        if (not isinstance(devices, list) or any(type(device) is not int or device < 0 for device in devices)
                or len(set(devices)) != len(devices)):
            errors.append("hardware_telemetry.devices must be distinct nonnegative integers")
    for field in ("start_ts", "end_ts"):
        if field in item and (not _finite_number(item[field]) or item[field] < 0):
            errors.append(f"hardware_telemetry.{field} must be finite nonnegative seconds")
    if _finite_number(item.get("start_ts")) and _finite_number(item.get("end_ts")):
        if item["end_ts"] < item["start_ts"]:
            errors.append("hardware_telemetry.end_ts must not precede start_ts")
    for field in ("start_ts_ns", "end_ts_ns"):
        if field in item and (type(item[field]) is not int or item[field] < 0):
            errors.append(f"hardware_telemetry.{field} must be nonnegative integer nanoseconds")
    if type(item.get("start_ts_ns")) is int and type(item.get("end_ts_ns")) is int:
        if item["end_ts_ns"] < item["start_ts_ns"]:
            errors.append("hardware_telemetry.end_ts_ns must not precede start_ts_ns")
    if not _is_json_native(dict(item)):
        errors.append("hardware_telemetry must be JSON-native")


def _finite_number(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)))


def _validate_active_execution_proof(value: Any, errors: list[str], *,
                                     manifest_run_id: str | None = None) -> None:
    """Validate the canonical observed proof consumed by mindie-bench.

    `patch_digest` identifies the candidate patch, not a native build key.
    `candidate_build_key` remains the separate C3 build identity. Every call
    timestamp must be inside the measured window; an unmeasured/requested
    command is not execution evidence.
    """
    item = _require_mapping(value, "active_execution_proof", errors)
    if not item:
        return
    unknown = sorted(set(item) - PROOF_FIELDS)
    if unknown:
        errors.append("active_execution_proof has unknown fields: " + ", ".join(unknown))
    missing = sorted(PROOF_FIELDS - set(item))
    if missing:
        errors.append("active_execution_proof is missing: " + ", ".join(missing))
    proof_run_id = item.get("run_id")
    if not isinstance(proof_run_id, str) or not SAFE_ID_RE.fullmatch(proof_run_id):
        errors.append(f"active_execution_proof.run_id must match {SAFE_ID_RE.pattern}")
    elif manifest_run_id is not None and proof_run_id != manifest_run_id:
        errors.append("active_execution_proof.run_id must equal manifest.run_id")
    for field in ("patch_digest", "loaded_binary_sha256"):
        digest = item.get(field)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"active_execution_proof.{field} must be 64 lowercase hex characters")
    for field in ("loaded_module_path", "candidate_build_key", "evidence_ref"):
        text = item.get(field)
        if not isinstance(text, str) or not text.strip():
            errors.append(f"active_execution_proof.{field} must be a nonempty string")
    window = _require_mapping(item.get("measurement_window"), "active_execution_proof.measurement_window", errors)
    start_ts = end_ts = None
    if window:
        unknown = sorted(set(window) - {"start_ts", "end_ts"})
        if unknown:
            errors.append("active_execution_proof.measurement_window has unknown fields: " + ", ".join(unknown))
        start_ts, end_ts = window.get("start_ts"), window.get("end_ts")
        if not _finite_number(start_ts) or not _finite_number(end_ts):
            errors.append("active_execution_proof.measurement_window start_ts/end_ts must be finite numbers")
            start_ts = end_ts = None
        elif float(end_ts) <= float(start_ts):
            errors.append("active_execution_proof.measurement_window must have positive duration")
    calls = item.get("calls")
    if not isinstance(calls, list) or not calls:
        errors.append("active_execution_proof.calls must be a nonempty array")
        return
    for index, call in enumerate(calls):
        path = f"active_execution_proof.calls[{index}]"
        row = _require_mapping(call, path, errors)
        if not row:
            continue
        unknown = sorted(set(row) - {"input_id", "call_path", "timestamp"})
        if unknown:
            errors.append(f"{path} has unknown fields: " + ", ".join(unknown))
        missing = sorted({"input_id", "call_path", "timestamp"} - set(row))
        if missing:
            errors.append(f"{path} is missing: " + ", ".join(missing))
        for field in ("input_id", "call_path"):
            text = row.get(field)
            if not isinstance(text, str) or not text.strip():
                errors.append(f"{path}.{field} must be a nonempty string")
        timestamp = row.get("timestamp")
        if not _finite_number(timestamp):
            errors.append(f"{path}.timestamp must be a finite number")
        elif start_ts is not None and end_ts is not None:
            if not float(start_ts) <= float(timestamp) <= float(end_ts):
                errors.append(f"{path}.timestamp must fall inside measurement_window")


def _validate_profile_summary_ref(value: Any, errors: list[str]) -> None:
    if isinstance(value, str):
        if not value.strip():
            errors.append("profile_summary_ref must be a nonempty string")
        return
    item = _require_mapping(value, "profile_summary_ref", errors)
    if item and not _is_json_native(dict(item)):
        errors.append("profile_summary_ref must be JSON-native")


def _validate_code(value: Any, errors: list[str]) -> None:
    item = _require_mapping(value, "code", errors)
    if not item:
        return
    for field in ("source_head", "snapshot_commit"):
        sha = item.get(field)
        if not isinstance(sha, str) or not GIT_SHA_RE.fullmatch(sha):
            errors.append(f"code.{field} must be 40 lowercase hex characters")
        elif sha == ZERO_SHA:
            errors.append(
                f"code.{field} is the all-zero sentinel, not a Git commit"
            )
    if type(item.get("dirty")) is not bool:
        errors.append("code.dirty must be a boolean")
    unknown = sorted(set(item) - {"source_head", "snapshot_commit", "dirty"})
    if unknown:
        errors.append(f"code has unknown fields: {', '.join(unknown)}")


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    errors: list[str] = []
    version = manifest.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    _validate_safe_id(manifest.get("run_id"), "run_id", errors)
    _validate_safe_id(manifest.get("parent_run_id"), "parent_run_id", errors, nullable=True)

    run_type = manifest.get("run_type")
    if run_type not in RUN_TYPES:
        errors.append(f"run_type must be one of: {', '.join(sorted(RUN_TYPES))}")
    status = manifest.get("status")
    if status not in RUN_STATUSES:
        errors.append(f"status must be one of: {', '.join(sorted(RUN_STATUSES))}")

    for field in ("workspace_snapshot", "environment", "model", "topology"):
        mapping = _require_mapping(manifest.get(field), field, errors)
        if mapping and not _is_json_native(
            mapping if isinstance(mapping, dict) else dict(mapping)
        ):
            errors.append(
                f"{field} must be JSON-native (string keys, no NaN/Inf)"
            )

    command = manifest.get("command")
    if not isinstance(command, list) or any(not isinstance(part, str) for part in command):
        errors.append("command must be an array of strings")

    env = _require_mapping(manifest.get("environment_variables"), "environment_variables", errors)
    for key, value in env.items():
        if not isinstance(key, str) or not isinstance(value, str):
            errors.append("environment_variables keys and values must be strings")
            continue
        if SECRET_ENV_RE.search(key):
            errors.append(f"environment_variables must not contain secret-like key: {key}")

    _validate_code(manifest.get("code"), errors)
    _validate_artifacts(manifest.get("artifacts"), errors)

    if "hardware_telemetry" in manifest:
        _validate_hardware_telemetry(manifest["hardware_telemetry"], errors)
    if "native_cache_receipt" in manifest:
        _validate_native_cache_receipt(manifest["native_cache_receipt"], errors)
    if "active_execution_proof" in manifest:
        manifest_run_id = manifest.get("run_id") if isinstance(manifest.get("run_id"), str) else None
        _validate_active_execution_proof(manifest["active_execution_proof"], errors,
                                         manifest_run_id=manifest_run_id)
    if "profile_summary_ref" in manifest:
        _validate_profile_summary_ref(manifest["profile_summary_ref"], errors)

    for field in ("created_at", "updated_at"):
        timestamp = manifest.get(field)
        if not isinstance(timestamp, str) or not RFC3339_UTC_RE.fullmatch(timestamp):
            errors.append(f"{field} must be an RFC3339 UTC timestamp ending in Z")

    required = {
        "schema_version",
        "run_id",
        "parent_run_id",
        "run_type",
        "code",
        "workspace_snapshot",
        "environment",
        "model",
        "topology",
        "command",
        "environment_variables",
        "artifacts",
        "status",
        "created_at",
        "updated_at",
    }
    allowed = required | set(OPTIONAL_EVIDENCE_FIELDS)
    unknown = sorted(set(manifest) - allowed)
    if unknown:
        errors.append(f"unknown top-level fields: {', '.join(unknown)}")
    missing = sorted(required - set(manifest))
    if missing:
        errors.append(f"missing top-level fields: {', '.join(missing)}")

    if errors:
        raise RunManifestError("; ".join(errors))


def transition_status(
    manifest: Mapping[str, Any], status: str, *, updated_at: str | None = None
) -> dict[str, Any]:
    validate_manifest(manifest)
    current = manifest["status"]
    if status not in STATUS_TRANSITIONS[current]:
        raise RunManifestError(f"invalid status transition: {current} -> {status}")
    updated = deepcopy(dict(manifest))
    updated["status"] = status
    updated["updated_at"] = updated_at or utc_now()
    validate_manifest(updated)
    return updated


def add_artifact(
    manifest: Mapping[str, Any],
    *,
    name: str,
    kind: str,
    uri: str,
    updated_at: str | None = None,
) -> dict[str, Any]:
    validate_manifest(manifest)
    updated = deepcopy(dict(manifest))
    updated["artifacts"].append({"name": name, "kind": kind, "uri": uri})
    updated["updated_at"] = updated_at or utc_now()
    validate_manifest(updated)
    return updated


def native_cache_receipt(*, hit: bool, cache_key: str, linked_files: int) -> dict[str, Any]:
    """Build the exact C3 native-cache receipt without changing execution state."""
    receipt = {"hit": bool(hit), "cache_key": str(cache_key), "linked_files": int(linked_files)}
    errors: list[str] = []
    _validate_native_cache_receipt(receipt, errors)
    if errors:
        raise RunManifestError("; ".join(errors))
    return receipt


def execution_proof(
    *,
    run_id: str,
    patch_digest: str,
    loaded_binary_sha256: str,
    loaded_module_path: str,
    calls: Sequence[Mapping[str, Any]],
    measurement_window: Mapping[str, Any],
    candidate_build_key: str,
    evidence_ref: str,
) -> dict[str, Any]:
    """Build canonical observed proof, never a requested-command assertion.

    Attach it with `add_evidence`; `run_id` must equal the manifest run_id.
    All call timestamps must be inside `measurement_window`.
    """
    proof = {
        "run_id": run_id,
        "patch_digest": patch_digest,
        "loaded_binary_sha256": loaded_binary_sha256,
        "loaded_module_path": loaded_module_path,
        "calls": [deepcopy(dict(call)) for call in calls],
        "measurement_window": deepcopy(dict(measurement_window)),
        "candidate_build_key": candidate_build_key,
        "evidence_ref": evidence_ref,
    }
    errors: list[str] = []
    _validate_active_execution_proof(proof, errors)
    if errors:
        raise RunManifestError("; ".join(errors))
    return proof


def bench_execution_evidence(proof: Mapping[str, Any], *,
                             manifest_run_id: str | None = None) -> dict[str, Any]:
    """The single adapter from C3 proof to mindie-bench ExecutionEvidence.

    The returned object contains only the canonical bench fields. The C3-only
    `candidate_build_key` and `measurement_window` remain in the manifest proof;
    `patch_digest` is never substituted for a native build key.
    """
    errors: list[str] = []
    if not isinstance(proof, Mapping):
        raise RunManifestError("active_execution_proof must be an object")
    _validate_active_execution_proof(proof, errors, manifest_run_id=manifest_run_id)
    if errors:
        raise RunManifestError("; ".join(errors))
    return {
        "run_id": proof["run_id"],
        "patch_digest": proof["patch_digest"],
        "loaded_binary_sha256": proof["loaded_binary_sha256"],
        "loaded_module_path": proof["loaded_module_path"],
        "calls": [deepcopy(dict(call)) for call in proof["calls"]],
        "evidence_ref": proof["evidence_ref"],
    }


def project_run_manifest(
    *,
    run_id: str,
    code: Mapping[str, Any],
    created_at: str,
    started_at: str,
    ended_at: str,
    terminal_status: str,
    environment: Mapping[str, Any] | None = None,
    workspace_snapshot: Mapping[str, Any] | None = None,
    command: Sequence[str] | None = None,
    topology: Mapping[str, Any] | None = None,
    hardware_telemetry: Mapping[str, Any] | None = None,
    native_cache_receipt: Mapping[str, Any] | None = None,
    active_execution_proof: Mapping[str, Any] | None = None,
    profile_summary_ref: Any = None,
) -> dict[str, Any]:
    """Project actual persisted execution facts into a terminal Run Manifest.

    Optional evidence is added only when the caller has actual observed data.
    A missing active proof stays absent; it is not a reason to fail a generic
    command or to synthesize a call.
    """
    if terminal_status not in TERMINAL_STATUSES:
        raise RunManifestError(f"terminal_status must be terminal: {terminal_status!r}")
    manifest = new_manifest(
        run_type="debug", run_id=run_id, code=code, environment=environment or {},
        workspace_snapshot=workspace_snapshot or {}, command=command or [],
        topology=topology or {}, created_at=created_at,
    )
    manifest = transition_status(manifest, "running", updated_at=started_at)
    if active_execution_proof is not None:
        observed = dict(active_execution_proof)
        if observed.get("run_id") != run_id:
            raise RunManifestError("active execution proof run_id differs from the manifest")
        active_execution_proof = execution_proof(
            run_id=run_id, patch_digest=observed["patch_digest"],
            loaded_binary_sha256=observed["loaded_binary_sha256"],
            loaded_module_path=observed["loaded_module_path"],
            calls=observed["calls"], measurement_window=observed["measurement_window"],
            candidate_build_key=observed["candidate_build_key"],
            evidence_ref=observed["evidence_ref"],
        )
    evidence = {}
    if hardware_telemetry is not None:
        evidence["hardware_telemetry"] = hardware_telemetry
    if native_cache_receipt is not None:
        evidence["native_cache_receipt"] = native_cache_receipt
    if active_execution_proof is not None:
        evidence["active_execution_proof"] = active_execution_proof
    if profile_summary_ref is not None:
        evidence["profile_summary_ref"] = profile_summary_ref
    if evidence:
        manifest = add_evidence(manifest, updated_at=started_at, **evidence)
    return transition_status(manifest, terminal_status, updated_at=ended_at)


def add_evidence(
    manifest: Mapping[str, Any],
    *,
    hardware_telemetry: Any = _UNSET,
    native_cache_receipt: Any = _UNSET,
    active_execution_proof: Any = _UNSET,
    profile_summary_ref: Any = _UNSET,
    updated_at: str | None = None,
) -> dict[str, Any]:
    """Attach optional evidence while preserving run_id, status and history."""
    validate_manifest(manifest)
    updated = deepcopy(dict(manifest))
    for field, value in (
        ("hardware_telemetry", hardware_telemetry),
        ("native_cache_receipt", native_cache_receipt),
        ("active_execution_proof", active_execution_proof),
        ("profile_summary_ref", profile_summary_ref),
    ):
        if value is _UNSET:
            continue
        if value is None:
            updated.pop(field, None)
        else:
            updated[field] = deepcopy(value)
    updated["updated_at"] = updated_at or utc_now()
    validate_manifest(updated)
    return updated


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except TimeoutError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise RunManifestError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(payload, MutableMapping):
        raise RunManifestError("manifest root must be an object")
    validate_manifest(payload)
    return dict(payload)


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    validate_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                manifest,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
