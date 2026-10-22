"""Local Git snapshot identity and creation for remote code parity.

[Role]
- Owns `RepoNode`/`SnapshotRecord` and builds parentless synthetic Git
  snapshots for workspace and execution inputs, plus their retained refs.

[Invariants]
- Snapshots are pinned by exact Git commit/tree object IDs; capture never
  mutates the source worktree HEAD or index.
- Child gitlinks carry exact commits; transport-only child paths do not appear
  in `changed_paths`.

[Timeouts & Locks]
- Optional clean-HEAD base discovery is bounded to 5s and retained-ref hints
  to 10s; no network I/O or locks are used here.

[Side-Effect Policy]
- Writes only synthetic Git objects/refs and temporary indexes; failures in
  optional base hints never gate materialization.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from mindie_coordinator.build_inputs import (
    VLLM_ASCEND_REINSTALL_PATTERNS, VLLM_REINSTALL_PATTERNS, build_input_fingerprints,
)
from mindie_coordinator.parity_support import (
    DEFAULT_DENYLIST, PROGRESS_SENTINEL, WORKSPACE_ID_PATTERN,
    ensure_local_git_identity, git, is_git_worktree, sanitize_repo_id,
)

class ParityUnavailable(RuntimeError):
    """Source materialization failed; no job may be launched."""


# Remote runtime script constants are shared by the install/transport layers.
# Keeping them in this lowest module avoids a parity <-> install import cycle.
PYTHON_METADATA_PREAMBLE = (
    'export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"',
    'PYTHON_CANDIDATE="$(ls -1d /usr/local/python*/bin/python3 2>/dev/null | sort -V | tail -n 1 || true)"',
    'if [ -n "$PYTHON_CANDIDATE" ]; then export PYTHON="$PYTHON_CANDIDATE"; elif command -v python3 >/dev/null 2>&1; then export PYTHON="$(command -v python3)"; elif command -v python >/dev/null 2>&1; then export PYTHON="$(command -v python)"; else echo "python not found" >&2; exit 127; fi',
    # Metadata, hashing and venv creation need libpython, not CANN/ATB setup.
    # In paired images ATB set_env itself imports torch and costs seconds.
    'MINDIE_IMAGE_PYTHON_ROOT="$(dirname "$(dirname "$(readlink -f "$PYTHON")")")"',
    'if [ -d "$MINDIE_IMAGE_PYTHON_ROOT/lib" ]; then export LD_LIBRARY_PATH="$MINDIE_IMAGE_PYTHON_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi',
    'export PATH="$(dirname "$PYTHON"):$PATH"',
)


ATB_CXX_ABI_PROBE = '''import importlib.metadata, json, sys, sysconfig
try:
    row = json.load(open(sys.argv[1])).get('atb_abi', {})
    if (row.get('cxx_abi') in ('0', '1') and row.get('torch') == importlib.metadata.version('torch')
            and row.get('python_abi') == sysconfig.get_config_var('SOABI')):
        print(row['cxx_abi'])
except TimeoutError:
    raise
except (OSError, ValueError, TypeError, AttributeError, importlib.metadata.PackageNotFoundError):
    pass
'''


DEFAULT_ENV_PREAMBLE = (
    'export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"',
    'export MINDIE_RUNTIME_ROOT="${MINDIE_RUNTIME_ROOT:-/vllm-workspace}"',
    'prepend_ld_path() {',
    '  dir="$1"',
    '  if [ -d "$dir" ]; then',
    '    export LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"',
    '  fi',
    '}',
    'prepend_ld_path /usr/local/Ascend/driver/lib64',
    'prepend_ld_path /usr/local/Ascend/driver/lib64/driver',
    'safe_source() {',
    '  file="$1"',
    '  if [ -f "$file" ]; then',
    '    set +u',
    '    local -a _mindie_source_args=()',
    '    if [ "$file" = /usr/local/Ascend/nnal/atb/set_env.sh ] && [ -f "$MINDIE_RUNTIME_ROOT/.mindie-runtime/reuse.json" ]; then',
    '      local _mindie_atb_abi',
    '      _mindie_atb_abi="$(python3 - "$MINDIE_RUNTIME_ROOT/.mindie-runtime/reuse.json" <<\'MINDIE_ATB_CXX_ABI\'',
    ATB_CXX_ABI_PROBE,
    'MINDIE_ATB_CXX_ABI',
    '      )" || _mindie_atb_abi=""',
    '      case "$_mindie_atb_abi" in 0|1) _mindie_source_args=("--cxx_abi=$_mindie_atb_abi");; esac',
    '    fi',
    '    source "$file" "${_mindie_source_args[@]}" >/dev/null 2>&1 || true',
    '    set -u',
    '  fi',
    '}',
    'for _ascend_env in '
    '/etc/profile.d/mindie-ascend-env.sh '
    '/usr/local/Ascend/cann-*/set_env.sh '
    '/usr/local/Ascend/ascend-toolkit/latest/set_env.sh '
    '/usr/local/Ascend/ascend-toolkit/set_env.sh '
    '/usr/local/Ascend/nnal/atb/set_env.sh '
    '"$MINDIE_RUNTIME_ROOT/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/bin/set_env.bash"; do',
    '  safe_source "$_ascend_env"',
    'done',
    'for _ascend_lib in '
    '/usr/local/Ascend/cann-*/lib64 '
    '/usr/local/Ascend/cann-*/runtime/lib64 '
    '/usr/local/Ascend/ascend-toolkit/latest/lib64 '
    '/usr/local/Ascend/ascend-toolkit/lib64; do',
    '  prepend_ld_path "$_ascend_lib"',
    'done',
    'unset _ascend_env _ascend_lib',
    'PYTHON_CANDIDATE="$(ls -1d /usr/local/python*/bin/python3 2>/dev/null | sort -V | tail -n 1 || true)"',
    'if [ -n "$PYTHON_CANDIDATE" ]; then export PYTHON="$PYTHON_CANDIDATE"; elif command -v python3 >/dev/null 2>&1; then export PYTHON="$(command -v python3)"; elif command -v python >/dev/null 2>&1; then export PYTHON="$(command -v python)"; else echo "python not found" >&2; exit 127; fi',
    'PYTHON_BIN_DIR="$(dirname "$PYTHON")"',
    'MINDIE_PYTHON_SHIM_DIR="$(mktemp -d /tmp/mindie-python-shim.XXXXXX)"',
    'trap "rm -rf \"$MINDIE_PYTHON_SHIM_DIR\"" EXIT',
    'ln -sf "$PYTHON" "$MINDIE_PYTHON_SHIM_DIR/python"',
    'ln -sf "$PYTHON" "$MINDIE_PYTHON_SHIM_DIR/python3"',
    'export PATH="$MINDIE_PYTHON_SHIM_DIR:$PYTHON_BIN_DIR:$PATH"',
    'hash -r',
    'export HI_PYTHON="$PYTHON"',
    'export Python3_EXECUTABLE="$PYTHON"',
    'export Python_EXECUTABLE="$PYTHON"',
    'export CMAKE_ARGS="-DPython3_EXECUTABLE=$PYTHON -DPython_EXECUTABLE=$PYTHON ${CMAKE_ARGS:-}"',
    'if [ -z "${MINDIE_BUILD_JOBS:-}" ]; then',
    '  MINDIE_BUILD_JOBS="$("$PYTHON" - <<\'PY\'',
    'import os',
    'cpu = 0',
    'try:',
    '    with open("/sys/fs/cgroup/cpu.max", encoding="utf-8") as f:',
    '        quota, period = f.read().split()[:2]',
    '    if quota != "max" and int(period) > 0:',
    '        cpu = max(1, int(quota) // int(period))',
    'except (OSError, ValueError, AttributeError):',
    '    cpu = 0',
    'if not cpu:',
    '    try:',
    '        cpu = len(os.sched_getaffinity(0))',
    '    except (OSError, ValueError, AttributeError):',
    '        cpu = max(1, min(os.cpu_count() or 1, 128))',
    'cpu = max(1, min(int(cpu), 128))',
    'mem_cap = 128',
    'try:',
    '    with open("/proc/meminfo", encoding="utf-8") as f:',
    '        for line in f:',
    '            if line.startswith("MemAvailable:"):',
    '                avail_mb = int(line.split()[1]) // 1024',
    '                mem_cap = max(1, (avail_mb - 2048) // 2048)',
    '                break',
    'except (OSError, ValueError, AttributeError):',
    '    pass',
    'print(max(1, min(cpu, mem_cap, 128)))',
    'PY',
    ')"',
    'fi',
    'export MINDIE_BUILD_JOBS',
    'export MAX_JOBS="${MAX_JOBS:-$MINDIE_BUILD_JOBS}"',
    'export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$MINDIE_BUILD_JOBS}"',
    'export PIP="$PYTHON -m pip"',
    'export PIP_DISABLE_PIP_VERSION_CHECK=1',
    'export PIP_NO_INPUT=1',
    'export PIP_DEFAULT_TIMEOUT=60',
    'export PIP_RETRIES=1',
    'export PIP_PROGRESS_BAR=off',
    'export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/.cache}"',
    'export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"',
    'export FETCHCONTENT_BASE_DIR="${FETCHCONTENT_BASE_DIR:-$XDG_CACHE_HOME/mindie/fetchcontent}"',
    'export PIP_CONFIG_FILE=/dev/null',
    'unset PIP_EXTRA_INDEX_URL',
    'export PIP_INDEX_URL="https://repo.huaweicloud.com/repository/pypi/simple"',
    'export PIP_TRUSTED_HOST="repo.huaweicloud.com"',
    'export CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"',
    'if [ -n "${MINDIE_SOC_VERSION:-}" ]; then export SOC_VERSION="$MINDIE_SOC_VERSION"; fi',
    'if [ -n "${MINDIE_COMPILE_CUSTOM_KERNELS:-}" ]; then export COMPILE_CUSTOM_KERNELS="$MINDIE_COMPILE_CUSTOM_KERNELS"; fi',
    'if [ "${MINDIE_USE_CLANG15:-0}" = "1" ] && command -v clang-15 >/dev/null 2>&1 && command -v clang++-15 >/dev/null 2>&1; then export C_COMPILER="${C_COMPILER:-$(command -v clang-15)}"; export CXX_COMPILER="${CXX_COMPILER:-$(command -v clang++-15)}"; fi',
    'export VLLM_WORKER_MULTIPROC_METHOD=spawn',
    'export OMP_NUM_THREADS=1',
    'export MKL_NUM_THREADS=1',
)

PIP_INDEX_NAME = 'huaweicloud'
PIP_INDEX_URL = 'https://repo.huaweicloud.com/repository/pypi/simple'
PIP_TRUSTED_HOST = 'repo.huaweicloud.com'

# Hardware-coupled packages that the paired vLLM-Ascend base image ships at
# CANN-matched versions. vllm-ascend/requirements.txt pins these
# (e.g. torch==2.9.0, torch-npu==2.9.0, triton-ascend==3.2.0), but per
# vllm-ascend's own Dockerfile they are NOT installed from the standard
# huaweicloud PyPI mirror -- torch(+cpu) comes from download.pytorch.org and
# torch_npu / triton-ascend come from Huawei OBS wheels and the Ascend repo
# index (see ASCEND_PIP_EXTRA_INDEX_URL). So a plain
# ``pip install -r requirements.txt`` against our mirror hard-fails with
# "No matching distribution".
#
# Handling (decided per package at install time, see
# ``install-vllm-ascend-requirements``):
#   * If the image already provides the package (any version installed), drop
#     the pinned line -- reinstalling would downgrade the image's tested stack
#     (image ships torch 2.10.0 / triton-ascend 3.2.1) and break the runtime.
#     ``verify-deps`` already treats torch-npu this way.
#   * If the package is genuinely missing, keep the line and add the Ascend
#     extra-index so it can be pulled from the public Ascend repo, matching
#     how the Dockerfile obtains it.
# Canonical (PEP 503) names.
IMAGE_PROVIDED_REQUIREMENT_NAMES = (
    'torch',
    'torchvision',
    'torchaudio',
    'torch-npu',
    'triton',
    'triton-ascend',
)

# Public Ascend package index, as used by vllm-ascend/Dockerfile
# (``PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi``).
# torch_npu / triton-ascend live here (and on OBS), not on the standard mirror.
ASCEND_PIP_EXTRA_INDEX_URL = 'https://mirrors.huaweicloud.com/ascend/repos/pypi'
ASCEND_PIP_TRUSTED_HOST = 'mirrors.huaweicloud.com'

DEFAULT_CONTAINER_CACHE_ROOT = '/root/.cache/mindie/remote-code-parity'
DEFAULT_MARKER_DIRNAME = '.remote-code-parity'
DEFAULT_GIT_TRANSPORT_TIMEOUT_SECONDS = 900.0
DEFAULT_CONTAINER_LOCK_STALE_SECONDS = 3600
# Keep runtime-private state and profiling artifacts that may be needed for
# post-run analysis across parity refreshes.
DEFAULT_ROOT_PRESERVE_PATHS = ('Mooncake', '.mindie-runtime', '.venv', 'venv', 'build')
PARITY_BRANCH_NAME = 'parity-current'

REMOTE_RUNTIME_ENV_PASSTHROUGH = (
    'CMAKE_ARGS', 'LDFLAGS', 'CC', 'CXX', 'VLLM_TARGET_DEVICE',
    'CFLAGS',
    'CXXFLAGS',
    'MINDIE_ENVIRONMENT_FINGERPRINT',
    'XDG_CACHE_HOME',
    'PIP_CACHE_DIR',
    'FETCHCONTENT_BASE_DIR',
    'MINDIE_BUILD_JOBS',
    'MAX_JOBS',
    'CMAKE_BUILD_PARALLEL_LEVEL',
    'CMAKE_BUILD_TYPE',
    'MINDIE_SOC_VERSION',
    'SOC_VERSION',
    'MINDIE_COMPILE_CUSTOM_KERNELS',
    'COMPILE_CUSTOM_KERNELS',
    'MINDIE_USE_CLANG15',
    'C_COMPILER',
    'CXX_COMPILER',
    'VERBOSE',
    'ASCEND_HOME_PATH',
)

@dataclass
class SubmoduleEntry:
    name: str
    path: str


UNPOPULATED_POLICIES = ('error', 'gitlink')

@dataclass
class RepoNode:
    relpath: str
    repo_path: Path
    submodule_name: str | None
    children: list['RepoNode'] = field(default_factory=list)
    gitlink_commit: str | None = None

@dataclass
class SnapshotRecord:
    relpath: str
    repo_id: str
    source_head: str | None
    parent: str | None
    commit: str
    tree: str
    ref: str
    changed_paths: list[str]
    submodules: list[dict[str, str]]
    build_inputs: dict[str, str] = field(default_factory=dict)
    source_path: str | None = None
    scm_version: str | None = None

def normalize_workspace_id(value: str) -> str:
    cleaned = WORKSPACE_ID_PATTERN.sub('-', value).strip('.-')
    return cleaned or 'workspace'

def validate_relative_posix_path(value: str, *, label: str) -> str:
    candidate = PurePosixPath(value)
    if not value or value in ('.', '..'):
        raise RuntimeError(f'{label} must not be empty')
    if candidate.is_absolute():
        raise RuntimeError(f'{label} must be relative, got: {value!r}')
    if '..' in candidate.parts:
        raise RuntimeError(f'{label} must not contain parent traversal, got: {value!r}')
    normalized = candidate.as_posix()
    if normalized in ('.', ''):
        raise RuntimeError(f'{label} must not be empty')
    return normalized

def validate_absolute_posix_path(value: str, *, label: str) -> str:
    if not value.startswith('/'):
        raise RuntimeError(f'{label} must be an absolute POSIX path, got: {value!r}')
    return PurePosixPath(value).as_posix()

def emit_progress(phase: str, **fields: Any) -> None:
    payload = {'phase': phase, **fields}
    print(f'{PROGRESS_SENTINEL}{json.dumps(payload, ensure_ascii=False)}', file=sys.stderr, flush=True)

def ensure_populated_worktree(repo: Path, relpath: str) -> None:
    if not repo.exists():
        raise RuntimeError(
            f'required repo path {relpath} is missing; initialize submodules before remote-code-parity'
        )
    if not is_git_worktree(repo):
        raise RuntimeError(
            f'required repo path {relpath} is not a populated Git worktree; run repo-init or git submodule update --init --recursive before remote-code-parity'
        )

def index_gitlinks(repo: Path) -> dict[str, str]:
    result = git(repo, ['ls-files', '--stage'], check=False)
    links: dict[str, str] = {}
    for line in result.stdout.splitlines():
        meta, separator, path = line.partition('\t')
        if not separator:
            continue
        parts = meta.split()
        if len(parts) >= 2 and parts[0] == '160000' and len(parts[1]) == 40:
            links[path] = parts[1]
    return links

def resolve_index_gitlink(repo: Path, path: str) -> str | None:
    return index_gitlinks(repo).get(path) or gitlink_for_path(repo, git_head(repo), path)

def list_submodules(repo: Path) -> list[SubmoduleEntry]:
    gitmodules = repo / '.gitmodules'
    if not gitmodules.exists():
        return []
    result = git(repo, ['config', '--file', '.gitmodules', '--get-regexp', r'^submodule\..*\.path$'], check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return []
    entries: list[SubmoduleEntry] = []
    for line in result.stdout.splitlines():
        key, path = line.split(maxsplit=1)
        name = key.removeprefix('submodule.').removesuffix('.path')
        entries.append(SubmoduleEntry(name=name, path=path.strip()))
    return entries

def discover_repo_tree(
    repo: Path,
    relpath: str = '.',
    submodule_name: str | None = None,
    source_roots: dict[str, Path] | None = None,
    *,
    unpopulated: str = 'error',
) -> RepoNode:
    if unpopulated not in UNPOPULATED_POLICIES:
        raise ValueError(f'unpopulated must be error or gitlink, got {unpopulated!r}')
    ensure_populated_worktree(repo, relpath)
    node = RepoNode(relpath=relpath, repo_path=repo, submodule_name=submodule_name)
    entries = list(list_submodules(repo))
    seen = {entry.path for entry in entries}
    if unpopulated == 'gitlink':
        for path in index_gitlinks(repo):
            if path not in seen:
                entries.append(SubmoduleEntry(name=path, path=path))
                seen.add(path)
    for entry in entries:
        child_relpath = entry.path if relpath in ('', '.') else f'{relpath}/{entry.path}'
        child_repo = (source_roots or {}).get(child_relpath, repo / entry.path)
        if is_git_worktree(child_repo):
            node.children.append(
                discover_repo_tree(
                    child_repo,
                    child_relpath,
                    entry.name,
                    source_roots,
                    unpopulated=unpopulated,
                )
            )
            continue
        if unpopulated == 'error':
            ensure_populated_worktree(child_repo, child_relpath)
        gitlink = resolve_index_gitlink(repo, entry.path)
        if gitlink:
            node.children.append(
                RepoNode(
                    relpath=child_relpath,
                    repo_path=child_repo,
                    submodule_name=entry.name,
                    gitlink_commit=gitlink,
                )
            )
    return node

def iter_postorder(node: RepoNode):
    for child in node.children:
        yield from iter_postorder(child)
    yield node

def git_head(repo: Path) -> str | None:
    result = git(repo, ['rev-parse', '--verify', 'HEAD'], check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None

def git_tree_for_commit(repo: Path, commit: str | None) -> str | None:
    if not commit:
        return None
    result = git(repo, ['rev-parse', f'{commit}^{{tree}}'], check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None

def reset_pathspecs(node: RepoNode, denylist: tuple[str, ...]) -> list[str]:
    specs: list[str] = []
    for child in node.children:
        specs.append(PurePosixPath(child.relpath).relative_to(node.relpath).as_posix())
    for pattern in denylist:
        if any(ch in pattern for ch in '*?[]'):
            specs.append(f':(glob){pattern}')
        else:
            specs.append(pattern)
    return specs

def synthetic_ref(workspace_id: str, snapshot_id: str, relpath: str) -> str:
    return f'refs/parity/{workspace_id}/{snapshot_id}/{sanitize_repo_id(relpath)}'

def commit_message(relpath: str) -> str:
    return f'remote-code-parity tree snapshot {sanitize_repo_id(relpath)}'

def gitlink_for_path(repo: Path, commit: str | None, path: str) -> str | None:
    if not commit:
        return None
    result = git(repo, ['ls-tree', commit, '--', path], check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    first = result.stdout.splitlines()[0].split(maxsplit=3)
    if len(first) < 3 or first[0] != '160000':
        return None
    return first[2]

def filter_transport_only_child_paths(
    paths: list[str],
    transport_only_child_paths: set[str],
) -> list[str]:
    return [path for path in paths if path not in transport_only_child_paths]

def build_synthetic_snapshot(
    node: RepoNode,
    *,
    workspace_id: str,
    snapshot_id: str,
    denylist: tuple[str, ...],
    child_commits: dict[str, SnapshotRecord],
) -> SnapshotRecord:
    repo = node.repo_path
    source_head = git_head(repo)
    ref = synthetic_ref(workspace_id, snapshot_id, node.relpath)
    temp_index = tempfile.NamedTemporaryFile(prefix='parity-index-', delete=False)
    temp_index.close()
    temp_index_path = Path(temp_index.name)
    env = os.environ.copy()
    env['GIT_INDEX_FILE'] = temp_index.name
    env['GIT_OPTIONAL_LOCKS'] = '0'
    author_name, author_email = ensure_local_git_identity(repo)
    env.setdefault('GIT_AUTHOR_NAME', author_name or 'remote-code-parity')
    env.setdefault('GIT_AUTHOR_EMAIL', author_email or 'remote-code-parity@example.invalid')
    env.setdefault('GIT_AUTHOR_DATE', '1970-01-01T00:00:00Z')
    env.setdefault('GIT_COMMITTER_NAME', author_name or 'remote-code-parity')
    env.setdefault('GIT_COMMITTER_EMAIL', author_email or 'remote-code-parity@example.invalid')
    env.setdefault('GIT_COMMITTER_DATE', '1970-01-01T00:00:00Z')

    try:
        if source_head:
            git(repo, ['read-tree', source_head], env=env)
        git(repo, ['add', '-A'], env=env)
        reset_specs = reset_pathspecs(node, denylist)
        if reset_specs:
            git(repo, ['reset', '-q', '--', *reset_specs], env=env)

        submodule_records: list[dict[str, str]] = []
        transport_only_child_paths: set[str] = set()
        for child in node.children:
            child_record = child_commits[child.relpath]
            child_rel_to_repo = PurePosixPath(child.relpath).relative_to(node.relpath).as_posix()
            source_gitlink = gitlink_for_path(repo, source_head, child_rel_to_repo)
            if (
                source_gitlink
                and child_record.source_head == source_gitlink
                and not child_record.changed_paths
            ):
                transport_only_child_paths.add(child_rel_to_repo)
            git(
                repo,
                ['update-index', '--add', '--cacheinfo', f'160000,{child_record.commit},{child_rel_to_repo}'],
                env=env,
            )
            submodule_records.append(
                {
                    'name': child.submodule_name or child_rel_to_repo,
                    'path': child_rel_to_repo,
                    'commit': child_record.commit,
                    'repo_id': child_record.repo_id,
                }
            )

        tree = git(repo, ['write-tree'], env=env).stdout.strip()
        commit = git(repo, ['commit-tree', tree, '-m', commit_message(node.relpath)], env=env).stdout.strip()
        if source_head:
            diff = git(repo, ['diff', '--name-only', f'{source_head}..{commit}']).stdout.splitlines()
        else:
            diff = git(repo, ['show', '--pretty=', '--name-only', commit]).stdout.splitlines()
        diff = filter_transport_only_child_paths(diff, transport_only_child_paths)

        git(repo, ['update-ref', ref, commit])

        return SnapshotRecord(
            relpath=node.relpath,
            repo_id=sanitize_repo_id(node.relpath),
            source_head=source_head,
            parent=source_head,
            commit=commit,
            tree=tree,
            ref=ref,
            changed_paths=[path.strip() for path in diff if path.strip()],
            submodules=submodule_records,
        )
    finally:
        temp_index_path.unlink(missing_ok=True)

def cleanup_synthetic_refs(workspace_root: Path, records: list[SnapshotRecord]) -> None:
    for record in records:
        if not record.ref:
            continue
        repo = record_source(workspace_root, record)
        if not is_git_worktree(repo):
            continue
        git(repo, ['update-ref', '-d', record.ref], check=False)

def cache_workspace_root(container_cache_root: str, workspace_id: str) -> str:
    return f"{container_cache_root.rstrip('/')}/workspaces/{workspace_id}"

def mirror_path_for(container_cache_root: str, workspace_id: str, record: SnapshotRecord) -> str:
    root = PurePosixPath(cache_workspace_root(container_cache_root, workspace_id)) / 'mirrors'
    if record.repo_id == 'workspace':
        return str(root / 'workspace.git')
    return str(root / 'nested' / f'{record.repo_id}.git')

def _fixed_inline_pack(repo, commit, carrier, previous, *, limit=65536):
    """Bound small edits by a freshly observed carrier; cold/large input uses Git."""
    if not previous or not re.fullmatch(r'[0-9a-f]{40,64}', previous):
        return None
    if git(repo, ['cat-file', '-e', previous + '^{commit}'], check=False).returncode:
        return None
    # A temporary file bounds memory even when a nominal edit contains a large
    # new blob. No network or shared refs are mutated while constructing it.
    with tempfile.TemporaryFile() as output:
        result = subprocess.run(
            ['git', '-C', str(repo), 'pack-objects', '--stdout', '--revs', '--delta-base-offset', '--compression=1'],
            input=(commit + '\n' + carrier + '\n^' + previous + '\n').encode('ascii'),
            stdout=output, stderr=subprocess.PIPE, timeout=DEFAULT_GIT_TRANSPORT_TIMEOUT_SECONDS,
        )
        if result.returncode:
            raise RuntimeError('fixed source pack failed: ' + result.stderr.decode('utf-8', errors='replace'))
        if output.tell() > limit:
            return None
        output.seek(0)
        data = output.read()
    return {'previous': previous, 'carrier': carrier, 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest(), 'data': base64.b64encode(data).decode('ascii')}

def _clean_head_snapshot_base(record):
    """Optional fixed-HEAD files plus admitted gitlinks, never a new source.

    Shared clones can have this clean snapshot's objects without its retained
    ref. Reproduce its parentless metadata from the admitted commit, not the
    current author environment. Only Git objects and a temporary index change.
    """
    if not record.source_head:
        return None
    try:
        if not all(isinstance(oid, str) and re.fullmatch(r'[0-9a-f]{40,64}', oid)
                   for oid in (record.source_head, record.commit, record.tree)):
            return None
        deadline = time.monotonic() + 5

        def objects(*args, data=None, env=None):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('optional clean snapshot hint exceeded its budget')
            return subprocess.run(
                ['git', '--no-replace-objects', '-C', record.source_path, *args],
                input=data if data is not None else b'', stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=True, timeout=remaining, env=env,
            ).stdout

        raw = objects('cat-file', 'commit', record.commit)
        header, message = raw.split(b'\n\n', 1)
        lines = header.split(b'\n')
        if (len(lines) != 3 or lines[0] != b'tree ' + record.tree.encode('ascii')
                or not lines[1].startswith(b'author ') or not lines[2].startswith(b'committer ')
                or message != (commit_message(record.relpath) + '\n').encode('utf-8')):
            return None
        # A changed .gitmodules or unmatched gitlink topology is not this
        # bounded clean-HEAD candidate. Ordinary transfer remains available.
        def entries(commit):
            return dict(entry.split(b'\t', 1)[::-1]
                        for entry in objects('ls-tree', '-r', '-z', commit).split(b'\0') if entry)

        head_entries, fixed_entries = entries(record.source_head), entries(record.commit)
        if head_entries.get(b'.gitmodules') != fixed_entries.get(b'.gitmodules'):
            return None
        head_links = {path.decode('utf-8') for path, entry in head_entries.items()
                      if entry.startswith(b'160000 ')}
        fixed_links = {path.decode('utf-8'): entry.split()[2].decode('ascii')
                       for path, entry in fixed_entries.items() if entry.startswith(b'160000 ')}
        children = {}
        for child in record.submodules:
            path = validate_relative_posix_path(child['path'], label='fixed submodule path')
            if (path != child['path'] or path in children
                    or not re.fullmatch(r'[0-9a-f]{40,64}', child['commit'])):
                return None
            children[path] = child['commit']
        if head_links != children.keys() or fixed_links != children:
            return None
        with tempfile.TemporaryDirectory(prefix='mindie-base-index-') as directory:
            env = {**os.environ, 'GIT_INDEX_FILE': str(Path(directory) / 'index'),
                   'GIT_OPTIONAL_LOCKS': '0'}
            objects('read-tree', record.source_head, env=env)
            for path, commit in children.items():
                objects('update-index', '--cacheinfo', f'160000,{commit},{path}', env=env)
            tree = objects('write-tree', env=env).strip()
        body = b'tree ' + tree + b'\n' + b'\n'.join(lines[1:]) + b'\n\n' + message
        candidate = objects('hash-object', '-t', 'commit', '-w', '--stdin', data=body).decode().strip()
        return candidate if candidate != record.commit else None
    except TimeoutError:
        raise
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, subprocess.SubprocessError):
        # This is local optional discovery, not a remotely submitted operation.
        # Missing/corrupt objects, unsupported shapes and timeouts leave the
        # existing exact-object transfer and its uncertainty handling intact.
        return None

def _local_snapshot_bases(record, *, limit=64):
    """Bounded local membership hints, never source selection or authority.

    Retained input/transport refs include parentless snapshots that an ordinary
    ancestor walk misses. A cold clone may instead reproduce one clean base.
    """
    if limit <= 0:
        return []
    try:
        # Route through the public module so existing parity.git monkeypatches
        # keep observing these optional transport hints.
        from mindie_coordinator import parity
        refs = parity.git(Path(record.source_path), ['for-each-ref', '--count=256', '--sort=-refname',
            '--format=%(objectname) %(refname)', 'refs/mindie/inputs', 'refs/parity-transport', 'refs/parity'],
            check=False, timeout=10)
    except TimeoutError:
        raise
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        return []
    if refs.returncode:
        return []
    result = []
    for line in refs.stdout.splitlines():
        parts = line.split()
        if (len(parts) == 2 and parts[1].endswith('/' + record.repo_id)
                and re.fullmatch(r'[0-9a-f]{40,64}', parts[0])
                and parts[0] != record.commit and parts[0] not in result):
            result.append(parts[0])
            if len(result) == limit:
                break
    if not result and limit > 0 and getattr(record, 'changed_paths', None):
        candidate = _clean_head_snapshot_base(record)
        if candidate:
            result.append(candidate)
    return result

def record_source(workspace_root: Path, record: SnapshotRecord) -> Path:
    return Path(record.source_path) if record.source_path else workspace_root / record.relpath

def parse_sources(values: list[str]) -> dict[str, Path]:
    sources = {}
    for value in values:
        name, separator, path = value.partition('=')
        if not separator or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or name in sources:
            raise ValueError('--source must contain unique safe names: name=/actual/worktree')
        candidate = Path(path).expanduser().resolve()
        ensure_populated_worktree(candidate, name)
        sources[name] = candidate
    return sources

def gitlink_snapshot_record(node: RepoNode) -> SnapshotRecord:
    commit = node.gitlink_commit or ''
    return SnapshotRecord(
        relpath=node.relpath,
        repo_id=sanitize_repo_id(node.relpath),
        source_head=commit or None,
        parent=commit or None,
        commit=commit,
        tree='',
        ref='',
        changed_paths=[],
        submodules=[],
    )

def build_snapshot_records(
    workspace_root: Path,
    workspace_id: str,
    snapshot_id: str,
    denylist: tuple[str, ...],
    source_roots: dict[str, Path] | None = None,
    records: list[SnapshotRecord] | None = None,
    *,
    unpopulated: str = 'error',
    with_build_inputs: bool = True,
) -> list[SnapshotRecord]:
    source_roots = source_roots or {}
    child_records: dict[str, SnapshotRecord] = {}
    ordered_records = records if records is not None else []

    def consume(tree: RepoNode) -> None:
        for node in iter_postorder(tree):
            if node.gitlink_commit:
                record = gitlink_snapshot_record(node)
            else:
                record = build_synthetic_snapshot(
                    node,
                    workspace_id=workspace_id,
                    snapshot_id=snapshot_id,
                    denylist=denylist,
                    child_commits=child_records,
                )
            child_records[node.relpath] = record
            record.source_path = str(node.repo_path.resolve()) if node.repo_path.exists() else str(node.repo_path)
            ordered_records.append(record)

    if source_roots:
        for name, path in source_roots.items():
            consume(discover_repo_tree(Path(path), name, None, None, unpopulated=unpopulated))
    else:
        consume(discover_repo_tree(workspace_root, '.', None, None, unpopulated=unpopulated))
    if with_build_inputs:
        for record in ordered_records:
            if record.relpath in ('vllm', 'vllm-ascend'):
                source = record_source(workspace_root, record)
                if not is_git_worktree(source):
                    continue
                patterns = VLLM_REINSTALL_PATTERNS if record.relpath == 'vllm' else VLLM_ASCEND_REINSTALL_PATTERNS
                record.build_inputs = build_input_fingerprints(source, record.commit, patterns)
                # Synthetic snapshots deliberately have no source history.
                # Capture setuptools-scm against the actual worktree once,
                # before transfer, instead of inferring a version remotely.
                if (source / 'pyproject.toml').exists() or (source / 'setup.py').exists():
                    from setuptools_scm import get_version
                    record.scm_version = get_version(root=str(source))
    return ordered_records
