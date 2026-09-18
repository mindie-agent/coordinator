#!/usr/bin/env python3
"""Workspace code identity from git objects.

Code identity is a git commit: HEAD when the worktree is clean, otherwise a
deterministic parentless snapshot of the current tree. The snapshot ref keeps
that orphan commit reachable for experiment records.

Owned by mindie-coordinator. Snapshot construction uses the in-package parity
module; this never locates a consumer skill script by filesystem path.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

GIT_SHA_RE = r"^[0-9a-f]{40}$"
IDENTITY_WORKSPACE_ID = "identity"
IDENTITY_SNAPSHOT_ID = "current"


class CodeIdentityError(RuntimeError):
    """Raised when the workspace has no usable git identity."""


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdin=subprocess.DEVNULL, capture_output=True,
        text=True, encoding="utf-8",
        check=False,
    )
    if check and result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        raise CodeIdentityError(message or f"git {' '.join(args)} failed in {repo}")
    return result


def _git_head(repo: Path) -> str:
    result = _git(repo, "rev-parse", "--verify", "HEAD", check=False)
    head = result.stdout.strip()
    if result.returncode != 0 or len(head) != 40:
        raise CodeIdentityError(f"no HEAD in {repo}")
    return head


def _repo_dirty(repo: Path) -> bool:
    status = _git(
        repo, "status", "--porcelain=v1", "--untracked-files=normal"
    ).stdout
    return bool(status.strip())


def _load_parity():
    from mindie_coordinator import parity

    return parity


def _create_identity_snapshot(workspace_root: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    """Build a deterministic parentless snapshot and keep the identity refs."""
    parity = _load_parity()
    records = parity.build_snapshot_records(
        workspace_root,
        IDENTITY_WORKSPACE_ID,
        IDENTITY_SNAPSHOT_ID,
        tuple(parity.DEFAULT_DENYLIST),
        unpopulated="gitlink",
        with_build_inputs=False,
    )
    repos: dict[str, dict[str, Any]] = {}
    snapshot = ""
    for record in records:
        dirty = bool(record.changed_paths) or (
            record.source_head is not None and record.commit != record.source_head
        )
        repos[record.relpath] = {
            "source_head": record.source_head,
            "snapshot_commit": record.commit,
            "dirty": dirty,
        }
        if record.relpath == ".":
            snapshot = record.commit
    if not snapshot:
        raise CodeIdentityError("workspace snapshot record is missing")
    return snapshot, repos


def _clean_repos(source_head: str) -> dict[str, dict[str, Any]]:
    return {
        ".": {
            "source_head": source_head,
            "snapshot_commit": source_head,
            "dirty": False,
        }
    }


def code_identity(workspace_root: Path | str) -> dict[str, Any]:
    """Return git-object identity for ``workspace_root``.

    When the worktree is clean, ``snapshot_commit`` equals ``source_head``.
    When it is dirty, a deterministic parentless snapshot of the current tree
    is created and kept under ``refs/parity/identity/current/``. Unpopulated
    nested submodules contribute their parent-index gitlink SHA.
    """
    root = Path(workspace_root).resolve()
    if not (root / ".git").exists():
        raise CodeIdentityError(f"not a git worktree: {root}")
    source_head = _git_head(root)
    dirty = _repo_dirty(root)

    if not dirty:
        repos = _clean_repos(source_head)
        return {
            "source_head": source_head,
            "snapshot_commit": source_head,
            "dirty": False,
            "repos": repos,
        }

    snapshot, repos = _create_identity_snapshot(root)
    return {
        "source_head": source_head,
        "snapshot_commit": snapshot,
        "dirty": True,
        "repos": repos,
    }


def manifest_code(workspace_root: Path | str) -> dict[str, Any]:
    """The three fields stored on a Run Manifest ``code`` object."""
    identity = code_identity(workspace_root)
    return {
        "source_head": identity["source_head"],
        "snapshot_commit": identity["snapshot_commit"],
        "dirty": identity["dirty"],
    }


def identity_workspace(sources: Mapping[str, str]) -> Path:
    """Choose the worktree whose Git identity goes on Run Manifest ``code``.

    Session ``sources`` are the bound business trees (typically ``vllm`` and
    ``vllm-ascend``). Their commits already populate ``workspace_snapshot``.
    ``code`` answers which containing workspace produced the run.

    When more than one source is bound, that workspace is the nearest git
    worktree that contains every source path — the session or scaffold
    checkout. This function does not pick ``vllm`` or ``vllm-ascend`` by
    name. A single source with no containing parent uses that source's own
    worktree. Multiple sources that share no containing worktree are an error.
    """
    if not sources:
        raise CodeIdentityError(
            "session has no source worktrees; cannot resolve code identity"
        )
    paths = [Path(path).expanduser().resolve() for path in sources.values()]
    try:
        common = Path(os.path.commonpath([str(path) for path in paths]))
    except ValueError as exc:
        raise CodeIdentityError(
            "session source worktrees do not share a filesystem root"
        ) from exc
    cursor = common
    while True:
        if (cursor / ".git").exists() and all(
            path == cursor or cursor in path.parents for path in paths
        ):
            return cursor
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    if len(paths) == 1:
        cursor = paths[0]
        while True:
            if (cursor / ".git").exists():
                return cursor
            if cursor.parent == cursor:
                break
            cursor = cursor.parent
        raise CodeIdentityError(f"not a git worktree: {paths[0]}")
    raise CodeIdentityError(
        f"session has {len(sources)} source worktrees "
        f"({', '.join(sorted(sources))}) with no common containing git "
        "worktree; code identity is that containing workspace, not a "
        "business tree picked by name"
    )


def collect_referenced_snapshot_commits(workspace_root: Path) -> set[str]:
    """Collect ``code.snapshot_commit`` values from Run Manifests under ``.mindie-local/``."""
    referenced: set[str] = set()
    local = Path(workspace_root) / ".mindie-local"
    if not local.is_dir():
        return referenced
    for path in local.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        code = payload.get("code")
        if not isinstance(code, Mapping):
            continue
        snapshot = code.get("snapshot_commit")
        if isinstance(snapshot, str) and len(snapshot) == 40:
            referenced.add(snapshot)
    return referenced


def gc_parity_refs(
    workspace_root: Path,
    *,
    max_age_days: int = 7,
    now: float | None = None,
) -> dict[str, Any]:
    """Delete ``refs/parity/`` older than ``max_age_days`` and not referenced."""
    import time

    parity = _load_parity()
    root = Path(workspace_root).resolve()
    referenced = collect_referenced_snapshot_commits(root)
    cutoff = (time.time() if now is None else now) - max_age_days * 86400
    deleted: list[dict[str, str]] = []
    kept: list[dict[str, str]] = []
    tree = parity.discover_repo_tree(root, ".", None, None, unpopulated="gitlink")
    for node in parity.iter_postorder(tree):
        if node.gitlink_commit or not parity.is_git_worktree(node.repo_path):
            continue
        listed = parity.git(
            node.repo_path,
            ["for-each-ref", "--format=%(objectname)\t%(refname)", "refs/parity"],
            check=False,
        )
        git_dir = Path(
            parity.git(node.repo_path, ["rev-parse", "--git-dir"]).stdout.strip()
        )
        if not git_dir.is_absolute():
            git_dir = node.repo_path / git_dir
        for line in listed.stdout.splitlines():
            if "\t" not in line:
                continue
            sha, ref = line.split("\t", 1)
            ref_path = git_dir / ref
            if ref_path.is_file():
                mtime = ref_path.stat().st_mtime
            else:
                mtime = 0.0
            row = {
                "repo": node.relpath,
                "ref": ref,
                "commit": sha,
            }
            if sha in referenced or mtime >= cutoff:
                kept.append(row)
                continue
            parity.git(node.repo_path, ["update-ref", "-d", ref], check=False)
            deleted.append(row)
    return {
        "status": "ok",
        "deleted": deleted,
        "kept": kept,
        "referenced": sorted(referenced),
    }
