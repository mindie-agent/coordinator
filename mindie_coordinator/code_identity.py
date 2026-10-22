#!/usr/bin/env python3
"""Workspace code identity from git objects.

[Role]
Computes deterministic Git commit / worktree / native `.so` tree identities for
Run Manifests and native runtime inspection.

[Invariants]
- When the worktree is clean, `snapshot_commit == source_head`.
- When dirty, creates a deterministic parentless snapshot commit via `parity`
  and retains it under `refs/parity/identity/current/`.
- Each Git capture observes the current worktree. A submission retains that
  captured identity in its manifest; separate submissions never share a cache.
- Non-Git native runtime inspection caches metadata by root/signature for 30s.

[Timeouts & Locks]
- Local git status/rev-parse subprocesses only; no network I/O or file locks.

[Side-Effect Policy]
- Clean worktrees are strictly read-only; dirty worktrees write only synthetic
  Git objects and `refs/parity/identity/current/*` refs using a temporary index.
"""

from __future__ import annotations

import copy
import hashlib
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

GIT_SHA_RE = r"^[0-9a-f]{40}$"
IDENTITY_WORKSPACE_ID = "identity"
IDENTITY_SNAPSHOT_ID = "current"
DEFAULT_IDENTITY_CACHE_TTL_SECONDS = 30.0

_SOURCE_IDENTITY_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}


class CodeIdentityError(RuntimeError):
    """Raised when the workspace has no usable git identity."""


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdin=subprocess.DEVNULL, capture_output=True,
        text=True, encoding="utf-8",
        check=False, timeout=30,
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


def _repo_status_porcelain(repo: Path) -> str:
    return _git(
        repo, "status", "--porcelain=v1", "--untracked-files=normal"
    ).stdout


def _so_tree_mtime_signature(root: Path) -> tuple[str, dict[str, str]]:
    hasher = hashlib.sha256()
    so_hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*.so*")):
        if not path.is_file():
            continue
        if path.suffix == ".so" or ".so." in path.name:
            try:
                st = path.stat()
            except TimeoutError:
                raise
            except OSError:
                continue
            rel = path.relative_to(root).as_posix()
            entry = f"{rel}:{st.st_mtime_ns}:{st.st_size}"
            hasher.update(f"{entry}\0".encode("utf-8"))
            so_hashes[rel] = hashlib.sha256(entry.encode("utf-8")).hexdigest()
    return hasher.hexdigest(), so_hashes


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
    status_output = _repo_status_porcelain(root)
    dirty = bool(status_output.strip())

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


def inspect_source_identity(
    workspace_root: Path | str,
    *,
    ttl_seconds: float = DEFAULT_IDENTITY_CACHE_TTL_SECONDS,
) -> dict[str, Any]:
    """Capture Git identity or inspect cached non-Git native runtime metadata."""
    root = Path(workspace_root).resolve()
    if (root / ".git").exists():
        return code_identity(root)
    if not root.is_dir():
        raise CodeIdentityError(f"workspace root does not exist: {root}")
    mtime_sig, so_hashes = _so_tree_mtime_signature(root)
    cache_key = (str(root), mtime_sig)
    now = time.monotonic()
    cached = _SOURCE_IDENTITY_CACHE.get(cache_key)
    if cached is not None and (now - cached[0]) <= ttl_seconds:
        return copy.deepcopy(cached[1])
    result = {
        "root": str(root),
        "source_head": None,
        "snapshot_commit": mtime_sig,
        "dirty": False,
        "so_hashes": so_hashes,
    }
    _SOURCE_IDENTITY_CACHE[cache_key] = (now, copy.deepcopy(result))
    return result


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

