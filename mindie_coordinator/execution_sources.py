"""Fix execution input once using the existing Git object capture implementation.

Worktree paths below locate object stores after capture; no execution phase may
read their current files. Retained refs keep dirty trees and true SCM ancestry
reachable without changing the user's HEAD, index, branches or worktree.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

SCHEMA_VERSION = "mindie.execution-sources.v1"


def source_identity(snapshot: dict) -> str:
    value = {
        "records": [{key: record.get(key) for key in
                     ("relpath", "commit", "tree", "source_head", "scm_version", "submodules", "build_inputs")}
                    for record in snapshot.get("records", [])],
        "build_env": snapshot.get("build_env", {}),
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_paths(snapshot: dict) -> dict[str, str]:
    return {name: source["path"] for name, source in snapshot["sources"].items()}


def validate_source_snapshot(snapshot: dict) -> dict:
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("run requires an accepted execution source snapshot")
    if not isinstance(snapshot.get("sources"), dict) or not isinstance(snapshot.get("records"), list):
        raise ValueError("invalid execution source snapshot")
    if snapshot.get("id") != source_identity(snapshot):
        raise ValueError("execution source snapshot identity does not match its fixed inputs")
    return snapshot


def capture_sources(sources: dict[str, str], state_dir: Path, *, attempts: int = 3) -> dict:
    """Capture selected repositories before durable admission, with bounded retry.

Fresh Git state and dirty content detect edits during capture, including a
second edit of an already dirty file. Retained trees for unchanged repositories
are reused. Source-free commands do not import Git/parity or scan anything.
"""
    if not isinstance(sources, dict):
        raise ValueError("sources must map repository names to actual worktrees")
    snapshot = {"schema_version": SCHEMA_VERSION, "sources": {}, "records": [], "build_env": {}}
    if sources:
        from mindie_coordinator.agent_session import worktree_reference
        from mindie_coordinator.build_inputs import BUILD_INPUT_ENV_KEYS
        from mindie_coordinator.execution_capture import capture_records, cleanup, save_cache
        from mindie_coordinator.parity_support import git

        roots = {}
        for name, path in sorted(sources.items()):
            if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name or "\\" in name:
                raise ValueError("source names must be single repository names")
            roots[name] = Path(worktree_reference(path)["path"])
        snapshot["build_env"] = {key: os.environ[key] for key in BUILD_INPUT_ENV_KEYS if key in os.environ}
        records, changed, cache = capture_records(roots, state_dir, attempts=attempts, build_env=snapshot["build_env"])
        changed_names = {record.relpath for record in changed}
        snapshot["records"] = [asdict(record) for record in records]
        snapshot["id"] = source_identity(snapshot)
        for record, data in zip(records, snapshot["records"]):
            if record.relpath in changed_names:
                retained_ref = f"refs/mindie/inputs/{snapshot['id']}/{record.repo_id}"
                git(Path(record.source_path), ["update-ref", retained_ref, record.commit])
                if record.source_head:
                    git(Path(record.source_path), ["update-ref", retained_ref + "-scm", record.source_head])
                data["ref"] = retained_ref
            if record.relpath in roots:
                snapshot["sources"][record.relpath] = {
                    "path": str(roots[record.relpath]), "source_head": record.source_head,
                    "commit": record.commit, "tree": record.tree,
                }
        cleanup(changed)
        for record, data in zip(records, snapshot["records"]):
            record.ref = data["ref"]
        try:
            save_cache(*cache, records)
        except OSError:
            # This rebuildable acceleration is not an admission prerequisite.
            # The immutable source records and retained refs remain authoritative.
            pass
    snapshot["id"] = source_identity(snapshot)
    snapshot["captured_at"] = time.time()
    directory = Path(state_dir) / "source-inputs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (snapshot["id"] + ".json")
    temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(snapshot, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
    return snapshot
