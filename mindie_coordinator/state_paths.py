"""Local state locations for this coordinator (stdlib only).

The scaffold's `mindie_local_state` stays scaffold-owned. Only the path
resolvers the task registry needs live here. Defaults follow the current
working tree, not the installed package location.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from mindie_coordinator.client_paths import client_path

STATE_DIRNAME = ".mindie-local"
COORDINATOR_STATE_ENV = "MINDIE_COORDINATOR_STATE_DIR"


def shared_workspace_root(repo_root: Path | None = None) -> Path:
    """Resolve the primary worktree of `repo_root`, or `repo_root` itself.

    Linked Git worktrees share one Git common dir, so they share one local
    task registry. An unrelated clone resolves to itself.
    """
    repo_root = Path(client_path(repo_root or Path.cwd())).expanduser().resolve()
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--git-common-dir"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return repo_root
    if result.returncode != 0 or not result.stdout.strip():
        return repo_root
    common_dir = Path(result.stdout.strip()).expanduser()
    if not common_dir.is_absolute():
        common_dir = repo_root / common_dir
    common_dir = common_dir.resolve()
    if common_dir.name.lower() == ".git" and common_dir.is_dir():
        return common_dir.parent
    return repo_root


def agent_sessions_root(repo_root: Path | None = None) -> Path:
    """Return the local native-attachment registry directory.

    `MINDIE_AGENT_SESSIONS_DIR` is the explicit override. Without it this
    follows the current working tree's primary worktree.
    """
    override = os.environ.get("MINDIE_AGENT_SESSIONS_DIR", "")
    if override:
        return Path(client_path(override)).expanduser()
    return shared_workspace_root(repo_root) / STATE_DIRNAME / "agent-sessions"


def coordinator_state_dir(sessions_dir: Path | None = None) -> Path:
    """Local runtime-pool state, next to the task registry unless overridden."""
    override = os.environ.get(COORDINATOR_STATE_ENV, "")
    if override:
        return Path(client_path(override)).expanduser()
    base = sessions_dir or agent_sessions_root()
    return Path(client_path(base)).expanduser().resolve().parent / "coordinator"
