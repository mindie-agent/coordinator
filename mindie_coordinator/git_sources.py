"""Populated-worktree and nested-submodule discovery for attestation.

Behaviour-preserving port of the three helpers `prepare_runtime.py` used to
import from the scaffold's remote-code-parity script
(`ensure_populated_worktree`, `list_submodules`, `discover_repo_tree`,
`iter_postorder`). Attestation runs inside a prepared container where only
this repository is deployed, so it cannot import a scaffold skill script.
The messages and traversal order are kept identical, because the attestation
tests assert on them.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SubmoduleEntry:
    name: str
    path: str


@dataclass
class RepoNode:
    relpath: str
    repo_path: Path
    submodule_name: str | None
    children: list["RepoNode"] = field(default_factory=list)


def git(repo: Path, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], check=check,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8")


def is_git_worktree(path: Path) -> bool:
    result = git(path, ["rev-parse", "--is-inside-work-tree"], check=False)
    if result.returncode != 0 or result.stdout.strip() != "true":
        return False
    top = git(path, ["rev-parse", "--show-toplevel"], check=False)
    if top.returncode != 0:
        return False
    try:
        return Path(top.stdout.strip()).resolve() == path.resolve()
    except FileNotFoundError:
        return False


def ensure_populated_worktree(repo: Path, relpath: str) -> None:
    if not repo.exists():
        raise RuntimeError(
            f"required repo path {relpath} is missing; initialize submodules before attestation"
        )
    if not is_git_worktree(repo):
        raise RuntimeError(
            f"required repo path {relpath} is not a populated Git worktree; "
            "run git submodule update --init --recursive before attestation"
        )


def list_submodules(repo: Path) -> list[SubmoduleEntry]:
    if not (repo / ".gitmodules").exists():
        return []
    result = git(repo, ["config", "--file", ".gitmodules", "--get-regexp",
                        r"^submodule\..*\.path$"], check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return []
    entries: list[SubmoduleEntry] = []
    for line in result.stdout.splitlines():
        key, path = line.split(maxsplit=1)
        entries.append(SubmoduleEntry(name=key.removeprefix("submodule.").removesuffix(".path"),
                                      path=path.strip()))
    return entries


def discover_repo_tree(repo: Path, relpath: str = ".", submodule_name: str | None = None) -> RepoNode:
    ensure_populated_worktree(repo, relpath)
    node = RepoNode(relpath=relpath, repo_path=repo, submodule_name=submodule_name)
    for entry in list_submodules(repo):
        child_relpath = entry.path if relpath in ("", ".") else f"{relpath}/{entry.path}"
        child_repo = repo / entry.path
        ensure_populated_worktree(child_repo, child_relpath)
        node.children.append(discover_repo_tree(child_repo, child_relpath, entry.name))
    return node


def iter_postorder(node: RepoNode):
    for child in node.children:
        yield from iter_postorder(child)
    yield node
