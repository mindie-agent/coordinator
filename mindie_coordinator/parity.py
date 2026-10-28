"""Transport, fixed-source materialization, and CLI for code parity.

[Role]
- Publishes synthetic snapshots to container-local Git mirrors, materializes
  fixed execution inputs, and exposes the `sync` CLI. Snapshot creation and
  remote runtime install steps live in `parity_snapshot` / `parity_install`;
  their names are re-exported here for the historical public API.

[Invariants]
- Every execution source snapshot is pinned by exact Git commit/tree SHAs
  before transfer; working-tree edits after admission never alter inputs.
- Small incremental edits (<= 64 KiB) ride inline; cold or large trees fall
  back to atomic Git push.

[Timeouts & Locks]
- Git transport enforces DEFAULT_GIT_TRANSPORT_TIMEOUT_SECONDS (900s).
- Container mirror updates are serialized by an OS lock directory.

[Side-Effect Policy]
- Uncertain remote materialization/export outcomes raise
  PreparationUncertain/ParityUnavailable and are never silently replayed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from mindie_coordinator.build_inputs import (
    VLLM_ASCEND_REINSTALL_PATTERNS, VLLM_REINSTALL_PATTERNS, build_input_fingerprints,
)
from mindie_coordinator.parity_install import (
    _export_existing_source_objects, _write_materialized_build_source,
    container_repo_path, final_manifest, make_manifest, marker_path_for,
    materialize_command, materialize_runtime, prepare_isolated_root_script,
    remote_runtime_env_exports, render_git_clean, resolve_state_root,
    resolved_root_preserve_paths, run_runtime_install_step,
    runtime_install_step_script, summary_payload, task_python_exports,
    verify_runtime_commits, verify_runtime_commits_map,
)
from mindie_coordinator.parity_snapshot import (
    ASCEND_PIP_EXTRA_INDEX_URL, ASCEND_PIP_TRUSTED_HOST, ATB_CXX_ABI_PROBE,
    DEFAULT_CONTAINER_CACHE_ROOT, DEFAULT_CONTAINER_LOCK_STALE_SECONDS,
    DEFAULT_DENYLIST, DEFAULT_ENV_PREAMBLE, DEFAULT_GIT_TRANSPORT_TIMEOUT_SECONDS,
    DEFAULT_MARKER_DIRNAME, DEFAULT_ROOT_PRESERVE_PATHS,
    IMAGE_PROVIDED_REQUIREMENT_NAMES, PARITY_BRANCH_NAME, PIP_INDEX_NAME,
    PIP_INDEX_URL, PIP_TRUSTED_HOST, PYTHON_METADATA_PREAMBLE,
    REMOTE_RUNTIME_ENV_PASSTHROUGH, UNPOPULATED_POLICIES, ParityUnavailable,
    RepoNode, SnapshotRecord, SubmoduleEntry, _clean_head_snapshot_base,
    _fixed_inline_pack, _local_snapshot_bases, build_snapshot_records,
    build_synthetic_snapshot, cache_workspace_root, cleanup_synthetic_refs,
    commit_message, discover_repo_tree, emit_progress, ensure_populated_worktree,
    filter_transport_only_child_paths, git_head, git_tree_for_commit,
    gitlink_for_path, gitlink_snapshot_record, index_gitlinks, iter_postorder,
    list_submodules, mirror_path_for, normalize_workspace_id, parse_sources,
    record_source, reset_pathspecs, resolve_index_gitlink, synthetic_ref,
    validate_absolute_posix_path, validate_relative_posix_path,
)
from mindie_coordinator.parity_support import (
    PROGRESS_SENTINEL, WORKSPACE_ID_PATTERN, SshEndpoint, ensure_local_git_identity,
    git, is_git_worktree, json_dump, now_utc, quoted, sanitize_repo_id, ssh_exec,
    ssh_exec_stream, ssh_stream_to_file,
)

def materialize_fixed_sources(*, workspace_id: str, endpoint: dict, source_snapshot: dict,
                              log_path=None, process=None, on_progress=None,
                              container_cache_root: str | None = None,
                              host_endpoint: dict | None = None,
                              shared_cache_root: str | None = None,
                              native_publication=None, build_source=None) -> dict:
    """Materialize admitted inputs directly, without the public sync lifecycle.

    One completed remote operation pins available objects, checks out the exact
    commits, validates the resulting trees and atomically records success. Only
    an explicit missing-object reply permits transfer followed by a new remote
    operation. Transport uncertainty is never retried here. ``process.run``
    supplies a fresh persisted owned job for each operation.
    """
    import inspect
    from mindie_coordinator.execution_sources import validate_source_snapshot
    from mindie_coordinator.parity_support import _materialize_fixed
    from mindie_coordinator.shared_source_objects import SHARED_SOURCE_CACHE, copy_fixed_objects

    validate_source_snapshot(source_snapshot)
    workspace_id = normalize_workspace_id(workspace_id)
    root = validate_absolute_posix_path(endpoint['root'], label='runtime root')
    cache = validate_absolute_posix_path(container_cache_root or DEFAULT_CONTAINER_CACHE_ROOT,
                                         label='container cache root')
    if root == '/' or '..' in PurePosixPath(root).parts:
        raise ValueError('fixed source root must be an isolated absolute directory')
    records = [SnapshotRecord(**row) for row in source_snapshot['records']]
    seen = set()
    for record in records:
        relpath = validate_relative_posix_path(record.relpath, label='fixed source path')
        if relpath != record.relpath or relpath in seen or '\\' in relpath:
            raise ValueError('fixed source paths must be unique canonical POSIX paths')
        if PurePosixPath(relpath).parts[0] in {'.mindie-runtime', '.venv', '.git', '.remote-dev'}:
            raise ValueError('fixed source path overlaps coordinator-owned runtime files')
        seen.add(relpath)
        if record.repo_id != sanitize_repo_id(relpath):
            raise ValueError('fixed source repository identity does not match its path')
        if not all(re.fullmatch(r'[0-9a-f]{40,64}', value) for value in (record.commit, record.tree)):
            raise ValueError('fixed source requires exact Git object identities')
    if not records:
        raise ValueError('materialization requires at least one fixed source')
    for record in records:
        for child in record.submodules:
            child_path = validate_relative_posix_path(child['path'], label='fixed submodule path')
            if str(PurePosixPath(record.relpath) / child_path) not in seen:
                raise ValueError('fixed source descriptor is missing a submodule')
    container = SshEndpoint(host=endpoint['host'], port=int(endpoint['port']), user=endpoint['user'],
                            container=endpoint.get('container'), identity_file=endpoint.get('identity_file'),
                            auth_ref=endpoint.get('auth_ref'))
    request = {'root': root, 'source_id': source_snapshot['id'],
               'carrier_ref': f'refs/parity/{workspace_id}/transport-carrier',
               'records': [{'relpath': row.relpath, 'commit': row.commit, 'tree': row.tree,
                            'submodules': row.submodules,
                            'mirror': mirror_path_for(cache, workspace_id, row)} for row in records]}
    if host_endpoint is not None or shared_cache_root is not None:
        shared = validate_absolute_posix_path(shared_cache_root or SHARED_SOURCE_CACHE,
                                             label='shared source cache')
        with ThreadPoolExecutor(max_workers=min(4, len(records))) as pool:
            for row, record, bases in zip(request['records'], records, pool.map(_local_snapshot_bases, records)):
                row['repo_id'] = record.repo_id
                row['shared_mirror'] = str(PurePosixPath(shared) / (record.repo_id + '.git'))
                row['local_bases'] = bases
    def command():
        program = (inspect.getsource(copy_fixed_objects) + '\n' + inspect.getsource(_materialize_fixed)
                   + '\nimport json, time\n_materialize_started = time.monotonic()\n'
                   + 'result = _materialize_fixed(' + repr(request) + ')\n'
                   + "result['preparation_timings'] = {'materialize': time.monotonic() - _materialize_started}\n")
        if build_source is not None:
            program += (inspect.getsource(_write_materialized_build_source)
                        + "\nif result.get('status') == 'materialized':\n"
                        + "    result['build_source_seconds'] = _write_materialized_build_source("
                        + repr(root) + ', ' + repr(build_source) + ')\n')
        if native_publication is not None:
            return native_publication.wrap_program(program)
        return ("python3 - <<'MINDIE_FIXED_SOURCES'\n" + program
                + '\nprint(json.dumps(result, separators=(",", ":")))\nMINDIE_FIXED_SOURCES\n')

    def run():
        script = preparation_command(command())
        result = ssh_exec_stream(container, script, stream_progress=False,
                                 log_path=log_path, process=process)
        try:
            payload = json.loads(result.stdout)
        except (TypeError, ValueError) as exc:
            raise ParityUnavailable('fixed source materialization returned no complete receipt') from exc
        if not isinstance(payload, dict):
            raise ParityUnavailable('fixed source materialization returned an invalid receipt')
        return payload

    def missing_rows(result):
        missing = result.get('missing')
        if (not isinstance(missing, list) or not missing
                or any(not isinstance(row, dict) or type(row.get('index')) is not int
                       or not 0 <= row['index'] < len(records) for row in missing)
                or len({row['index'] for row in missing}) != len(missing)):
            raise ParityUnavailable('fixed source materialization returned invalid missing objects')
        return missing

    from mindie_coordinator.preparation_script import preparation_command, preparation_command_fits
    if native_publication is not None and not preparation_command_fits(command()):
        # Large native metadata uses the existing separate publication job.
        # Every later inline pack is budgeted against this entire program.
        native_publication = None
    result = run()
    if result.get('status') == 'missing' and host_endpoint is not None:
        missing = missing_rows(result)
        if process is not None and process.cancel_requested():
            from mindie_coordinator.preparation_process import PreparationCancelled
            raise PreparationCancelled('fixed object discovery cancelled before launch')
        cold = [request['records'][row['index']] for row in missing if not row.get('carrier')]
        # An existing private carrier already enables cheap delta transfer.
        # A new Python edit must not rediscover every container on the host.
        if cold:
            exported = _export_existing_source_objects(host_endpoint, cold, cache)
            if log_path and (exported.get('diagnostics') or exported.get('reason')):
                with Path(log_path).open('a', encoding='utf-8') as stream:
                    stream.write('shared source discovery: ' + json.dumps(exported, sort_keys=True) + '\n')
            if on_progress:
                on_progress({'phase': 'shared-source-objects', **exported})
            if exported['status'] == 'copied':
                # A new persisted job, not a replay of the completed miss.
                result = run()
    if result.get('status') == 'missing':
        missing = missing_rows(result)
        for row in missing:
            if process is not None and process.cancel_requested():
                from mindie_coordinator.preparation_process import PreparationCancelled
                raise PreparationCancelled('fixed object upload cancelled before launch')
            record = records[row['index']]
            mirror = request['records'][row['index']]['mirror']
            repo = Path(record.source_path)
            _, carrier = build_transport_carrier(repo, container=container, mirror_path=mirror,
                workspace_id=workspace_id, record=record, remote_carrier_commit=row.get('carrier'))
            if on_progress:
                on_progress({'phase': 'push-mirror', 'relpath': record.relpath})
            pack = _fixed_inline_pack(repo, record.commit, carrier, row.get('carrier'))
            if pack is not None:
                # The next persisted preparation job receives and materializes
                # the exact objects over its existing authenticated transport.
                request.setdefault('inline_packs', {})[str(row['index'])] = pack
                # The owned worker invokes bash -c. Bound the whole argument,
                # including base64, all repositories and the remote program,
                # below Linux's per-argument limit; overflow uses Git below.
                if preparation_command_fits(command()):
                    if on_progress:
                        on_progress({'phase': 'inline-pack-ready', 'relpath': record.relpath,
                                     'transport': 'owned-rpc-pack', 'bytes': pack['bytes']})
                    continue
                del request['inline_packs'][str(row['index'])]
            # Exact OIDs are the source, including the carrier. A concurrent
            # capture can move local refs without changing this execution.
            # Git owns its receive process/atomic refs; never kill or delete a
            # shared mirror to recover a failed transfer.
            started = time.monotonic()
            transport_options = ['-c', 'protocol.ext.allow=always'] if container.container else []
            git(repo, [*transport_options, 'push', '--porcelain', '--atomic', git_push_url(container, mirror),
                       f'{record.commit}:refs/mindie/snapshots/{record.commit}',
                       f'+{carrier}:{request["carrier_ref"]}'],
                env=git_ssh_environment(container), timeout=DEFAULT_GIT_TRANSPORT_TIMEOUT_SECONDS)
            if on_progress:
                on_progress({'phase': 'push-mirror-complete', 'relpath': record.relpath,
                             'elapsed_seconds': round(time.monotonic() - started, 6)})
        result = run()
    expected = {row.relpath: row.commit for row in records}
    if (result.get('status') != 'materialized' or result.get('source_id') != source_snapshot['id']
            or result.get('root') != root or result.get('commits') != expected):
        raise ParityUnavailable('fixed source materialization did not verify the admitted inputs')
    return result

def git_remote_url(container: SshEndpoint, mirror_path: str) -> str:
    host = container.host
    if ':' in host and not host.startswith('['):
        host = f'[{host}]'
    user = quote(container.user, safe='')
    path = quote(validate_absolute_posix_path(mirror_path, label='mirror path'), safe='/')
    return f'ssh://{user}@{host}:{container.port}{path}'

def git_push_url(container: SshEndpoint, mirror_path: str) -> str:
    """Push cold objects through the existing container selector, not host SSH.

    A pinned existing container may have no published SSH port. The normal
    remote URL remains the authority for ls-remote/fetch; this ext transport
    only supplies Git's receive-pack path for a cold push.
    """
    if not container.container:
        return git_remote_url(container, mirror_path)
    path = validate_absolute_posix_path(mirror_path, label='mirror path')
    from remote_dev.core.endpoint import Endpoint
    from remote_dev.core.ssh_transport import ssh_launch

    endpoint = Endpoint(host=container.host, port=container.port, user=container.user,
                        identity_file=container.identity_file, auth_ref=container.auth_ref)
    command = shlex.join(["docker", "exec", "-i", container.container, "git", "receive-pack", path])
    prefix = ssh_launch(endpoint, command).argv
    # git-remote-ext splits its own arguments; shell quotes are literal there.
    return 'ext::' + ' '.join(token.replace('%', '%%').replace(' ', '% ') for token in prefix)

def git_ssh_environment(container: SshEndpoint) -> dict[str, str]:
    from remote_dev.core.endpoint import Endpoint
    from remote_dev.core.ssh_transport import ssh_launch

    plan = ssh_launch(Endpoint(host=container.host, port=container.port, user=container.user,
                              identity_file=container.identity_file, auth_ref=container.auth_ref))
    env = (plan.env or os.environ).copy()
    env['GIT_TERMINAL_PROMPT'] = '0'
    cmd = list(plan.argv)
    # Git appends host and the remote git command. ssh_base_cmd already
    # includes `-- host`; keep the builder's options as the ssh prefix.
    if '--' in cmd:
        cmd = cmd[: cmd.index('--')]
    if '-T' not in cmd:
        cmd = [cmd[0], '-T', *cmd[1:]]
    env['GIT_SSH_COMMAND'] = shlex.join(cmd)
    return env

def transport_carrier_ref(
    container: SshEndpoint,
    mirror_path: str,
    workspace_id: str,
    record: SnapshotRecord,
) -> str:
    target = f'{container.user}@{container.host}:{container.port}:{mirror_path}'
    token = hashlib.sha256(target.encode('utf-8')).hexdigest()[:16]
    return f'refs/parity-transport/{workspace_id}/{token}/{record.repo_id}'

def build_transport_carrier(
    repo: Path,
    *,
    container: SshEndpoint,
    mirror_path: str,
    workspace_id: str,
    record: SnapshotRecord,
    remote_carrier_commit: str | None = None,
) -> tuple[str, str]:
    ref = transport_carrier_ref(container, mirror_path, workspace_id, record)
    previous_result = git(repo, ['rev-parse', '--verify', ref], check=False)
    previous = previous_result.stdout.strip() if previous_result.returncode == 0 else None
    # A fresh recipient can already own a verified shared snapshot even when
    # this local endpoint has no carrier ref. Connect the transport history to
    # that observed base: disconnected tree snapshots otherwise make Git send
    # every object despite excluding the base in pack-objects --revs.
    if (remote_carrier_commit and previous != remote_carrier_commit
            and re.fullmatch(r'[0-9a-f]{40,64}', remote_carrier_commit)
            and git(repo, ['cat-file', '-e', remote_carrier_commit + '^{commit}'], check=False).returncode == 0):
        previous = remote_carrier_commit
    if not previous or previous != remote_carrier_commit:
        carrier = record.commit
    elif git_tree_for_commit(repo, previous) == record.tree:
        carrier = previous
    else:
        env = os.environ.copy()
        author_name, author_email = ensure_local_git_identity(repo)
        env.update(
            {
                'GIT_AUTHOR_NAME': author_name or 'remote-code-parity',
                'GIT_AUTHOR_EMAIL': author_email or 'remote-code-parity@example.invalid',
                'GIT_AUTHOR_DATE': '1970-01-01T00:00:00Z',
                'GIT_COMMITTER_NAME': author_name or 'remote-code-parity',
                'GIT_COMMITTER_EMAIL': author_email or 'remote-code-parity@example.invalid',
                'GIT_COMMITTER_DATE': '1970-01-01T00:00:00Z',
            }
        )
        carrier = git(
            repo,
            [
                'commit-tree',
                record.tree,
                '-p',
                previous,
                '-m',
                f'remote-code-parity transport carrier {workspace_id} {record.repo_id}',
            ],
            env=env,
        ).stdout.strip()
    git(repo, ['update-ref', ref, carrier])
    return ref, carrier

def remote_ref_commit(
    repo: Path,
    *,
    remote_url: str,
    remote_ref: str,
    env: dict[str, str],
) -> str | None:
    result = git(
        repo,
        ['ls-remote', remote_url, remote_ref],
        env=env,
        timeout=DEFAULT_GIT_TRANSPORT_TIMEOUT_SECONDS,
    )
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1] == remote_ref:
            return fields[0]
    return None

def manifest_path_for(container_cache_root: str, workspace_id: str, snapshot_id: str) -> str:
    return str(PurePosixPath(cache_workspace_root(container_cache_root, workspace_id)) / 'manifests' / f'{snapshot_id}.json')

def lock_path_for(container_cache_root: str, workspace_id: str, container_identity: str) -> str:
    token = re.sub(r'[^A-Za-z0-9._-]+', '-', container_identity).strip('.-') or 'container'
    return str(PurePosixPath(cache_workspace_root(container_cache_root, workspace_id)) / 'locks' / token)

def snapshot_mirror_refs(workspace_id: str) -> tuple[str, str, str]:
    return (f'refs/heads/{PARITY_BRANCH_NAME}', f'refs/parity/{workspace_id}/current',
            f'refs/parity/{workspace_id}/transport-carrier')

def ensure_remote_bare_repos(
    container: SshEndpoint, mirror_paths: list[str], dry_run: bool, *, refs: tuple[str, ...] = (),
) -> dict[str, dict[str, str]]:
    """Prepare mirrors and observe their live refs in the same remote request.

    Callers hold the parity lock through the subsequent transfer/materialize.
    Missing refs or objects remain misses; no local receipt predicts success.
    """
    if dry_run or not mirror_paths:
        return {}
    lines = ['set -eo pipefail']
    for index, mirror_path in enumerate(mirror_paths):
        lines.extend(
            [
                f'mkdir -p {quoted(str(PurePosixPath(mirror_path).parent))}',
                f'if [ -e {quoted(mirror_path)} ] && [ ! -d {quoted(str(PurePosixPath(mirror_path) / "objects"))} ]; then rm -rf {quoted(mirror_path)}; fi',
                f'if [ ! -d {quoted(mirror_path)} ]; then git init --bare {quoted(mirror_path)} >/dev/null; fi',
            ]
        )
        for ref in refs:
            lines.append(
                f'if oid=$(git --git-dir={quoted(mirror_path)} rev-parse --verify --quiet {quoted(ref + "^{commit}")}) '
                f'&& git --git-dir={quoted(mirror_path)} cat-file -e "$oid^{{tree}}" 2>/dev/null; then '
                f'printf "%s\\t%s\\t%s\\n" {index} {quoted(ref)} "$oid"; fi'
            )
    result = ssh_exec(container, '\n'.join(lines))
    observed = {path: {} for path in mirror_paths}
    for line in result.stdout.splitlines():
        fields = line.split('\t')
        if (len(fields) == 3 and fields[0].isdigit() and int(fields[0]) < len(mirror_paths)
                and fields[1] in refs and re.fullmatch(r'[0-9a-f]{40,64}', fields[2])):
            observed[mirror_paths[int(fields[0])]][fields[1]] = fields[2]
    return observed

def push_snapshot_via_git(
    repo: Path,
    *,
    container: SshEndpoint,
    mirror_path: str,
    record: SnapshotRecord,
    workspace_id: str,
    remote_refs: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    target_ref = f'refs/parity/{workspace_id}/current'
    remote_carrier_ref = f'refs/parity/{workspace_id}/transport-carrier'
    remote_url = git_remote_url(container, mirror_path)
    git_env = git_ssh_environment(container)
    remote_carrier_commit = remote_refs.get(remote_carrier_ref) if remote_refs is not None else remote_ref_commit(
        repo,
        remote_url=remote_url,
        remote_ref=remote_carrier_ref,
        env=git_env,
    )
    carrier_ref, carrier_commit = build_transport_carrier(
        repo,
        container=container,
        mirror_path=mirror_path,
        workspace_id=workspace_id,
        record=record,
        remote_carrier_commit=remote_carrier_commit,
    )
    expected_refs = {target_ref: record.commit, f'refs/heads/{PARITY_BRANCH_NAME}': record.commit,
                     remote_carrier_ref: carrier_commit}
    if remote_refs is not None and all(remote_refs.get(ref) == commit for ref, commit in expected_refs.items()):
        return {'repo': record.relpath, 'transport': 'git',
                'elapsed_seconds': round(time.monotonic() - started, 6),
                'carrier_commit': carrier_commit, 'skipped': True,
                'detail': 'live remote refs and commit objects already match the fixed snapshot'}
    result = git(
        repo,
        [
            'push',
            '--porcelain',
            '--force',
            remote_url,
            f'{record.ref}:{target_ref}',
            f'{record.ref}:refs/heads/{PARITY_BRANCH_NAME}',
            f'{carrier_ref}:{remote_carrier_ref}',
        ],
        env=git_env,
        timeout=DEFAULT_GIT_TRANSPORT_TIMEOUT_SECONDS,
    )
    return {
        'repo': record.relpath,
        'transport': 'git',
        'elapsed_seconds': round(time.monotonic() - started, 6),
        'carrier_commit': carrier_commit,
        'detail': result.stdout.strip(),
    }

def acquire_container_lock(
    container: SshEndpoint,
    lock_path: str,
    dry_run: bool,
    *,
    stale_seconds: int = DEFAULT_CONTAINER_LOCK_STALE_SECONDS,
) -> None:
    if dry_run:
        return
    script = '\n'.join(
        [
            'set -eo pipefail',
            f'lock={quoted(lock_path)}',
            f'stale_seconds={int(stale_seconds)}',
            'mkdir -p "$(dirname "$lock")"',
            'write_owner() {',
            '  {',
            '    printf "pid=%s\\n" "$$"',
            '    printf "host=%s\\n" "$(hostname 2>/dev/null || true)"',
            '    printf "started_at=%s\\n" "$(date -Is 2>/dev/null || date)"',
            '  } >"$lock/owner"',
            '}',
            'if mkdir "$lock" 2>/dev/null; then',
            '  write_owner',
            '  exit 0',
            'fi',
            'if [ -d "$lock" ]; then',
            '  now="$(date +%s)"',
            '  mtime="$(stat -c %Y "$lock" 2>/dev/null || echo 0)"',
            '  age="$((now - mtime))"',
            '  if [ "$age" -ge "$stale_seconds" ]; then',
            '    rm -rf "$lock"',
            '    if mkdir "$lock" 2>/dev/null; then',
            '      write_owner',
            '      exit 0',
            '    fi',
            '  fi',
            'fi',
            'echo "lock exists: $lock" >&2',
            'if [ -f "$lock/owner" ]; then cat "$lock/owner" >&2 || true; fi',
            'exit 1',
        ]
    )
    result = ssh_exec(container, script, check=False)
    if result.returncode != 0:
        raise RuntimeError(f'could not acquire container lock {lock_path}: {result.stderr or result.stdout}')

def release_container_lock(container: SshEndpoint, lock_path: str, dry_run: bool) -> None:
    if dry_run:
        return
    ssh_exec(container, f'rm -rf {quoted(lock_path)} >/dev/null 2>&1 || true', check=False)

def upload_manifest(container: SshEndpoint, manifest_path: str, manifest: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    ssh_stream_to_file(container, manifest_path, json_dump(manifest) + '\n')

def run_sync(args: argparse.Namespace) -> int:
    sources = parse_sources(getattr(args, 'source', []))
    fixed = json.loads(Path(args.snapshot_file).read_text(encoding="utf-8")) if getattr(args, 'snapshot_file', None) else None
    if not sources and not fixed:
        raise ValueError('materialization requires explicit source repositories')
    workspace_root = resolve_state_root(getattr(args, 'workspace_root', None), sources)
    workspace_id = normalize_workspace_id(args.workspace_id)
    runtime_root = validate_absolute_posix_path(args.runtime_root, label='runtime root')
    container_cache_root = validate_absolute_posix_path(args.container_cache_root, label='container cache root')
    marker_dirname = validate_relative_posix_path(args.marker_dirname, label='marker dirname')
    root_preserve_paths = resolved_root_preserve_paths(marker_dirname, args.preserve_path)
    snapshot_id = args.snapshot_id or now_utc().replace(':', '').replace('-', '') + '-' + uuid.uuid4().hex[:8]
    container = SshEndpoint(host=args.container_host, port=args.container_port, user=args.container_user)

    emit_progress('snapshot-build', workspace_id=workspace_id, snapshot_id=snapshot_id)
    records: list[SnapshotRecord] = []
    keep_refs = fixed is not None
    manifest_path = manifest_path_for(container_cache_root, workspace_id, snapshot_id)
    current_phase = 'snapshot-built'
    try:
        if fixed is not None:
            if fixed.get('schema_version') != 'mindie.execution-sources.v1':
                raise ValueError('unsupported fixed source descriptor')
            records = [SnapshotRecord(**record) for record in fixed['records']]
        else:
            records = build_snapshot_records(workspace_root, workspace_id, snapshot_id, tuple(DEFAULT_DENYLIST), sources, records)
        try:
            snapshot_commits = {record.relpath: record.commit for record in records}
            manifest = make_manifest(
                workspace_root=workspace_root,
                workspace_id=workspace_id,
                snapshot_id=snapshot_id,
                server_name=args.server_name,
                container_identity=args.container_identity,
                runtime_root=runtime_root,
                container_cache_root=container_cache_root,
                marker_dirname=marker_dirname,
                root_preserve_paths=root_preserve_paths,
                records=records,
            )
            manifest['apply_mode'] = args.apply_mode
            if args.print_manifest:
                print(json_dump(manifest))

            if args.dry_run:
                summary = summary_payload(
                    status='dry-run',
                    server_name=args.server_name,
                    container_identity=args.container_identity,
                    workspace_id=workspace_id,
                    container_cache_root=container_cache_root,
                    records=records,
                    reinstall_status='skipped-by-apply-mode',
                    reason=f'apply_mode={args.apply_mode} skips runtime install/rebuild',
                    first_install=False,
                    observed_runtime_commits=None,
                )
                summary['apply_mode'] = args.apply_mode
                summary['manifest_path'] = manifest_path
                print(json_dump(summary))
                keep_refs = True
                return 0

            lock_path = lock_path_for(container_cache_root, workspace_id, args.container_identity)
            current_phase = 'acquire-lock'
            emit_progress(current_phase, lock_path=lock_path, apply_mode=args.apply_mode)
            acquire_container_lock(container, lock_path, args.dry_run)
            try:
                if args.apply_mode == 'materialize':
                    # Managed retries often change only the launch options.
                    # Observe the exact current sources under the same lock
                    # before deciding that transport/reset can be omitted.
                    observed = verify_runtime_commits_map(
                        container=container, runtime_root=runtime_root,
                        expected=snapshot_commits, include_untracked=True,
                    )
                    if observed == snapshot_commits:
                        summary = summary_payload(
                            status='materialized', server_name=args.server_name,
                            container_identity=args.container_identity, workspace_id=workspace_id,
                            container_cache_root=container_cache_root, records=records,
                            reinstall_status='skipped-by-apply-mode',
                            reason='remote sources already match the clean snapshot',
                            first_install=False, runtime_install_env={},
                            observed_runtime_commits=observed,
                        )
                        summary.update(apply_mode='materialize', fast_path='materialized-snapshot', transfers=[])
                        emit_progress('complete', status='materialized', fast_path='materialized-snapshot')
                        print(json_dump(summary))
                        keep_refs = True
                        return 0
                current_phase = 'push-mirrors'
                emit_progress(current_phase, repo_count=len(records), apply_mode=args.apply_mode)
                all_mirror_paths = [mirror_path_for(container_cache_root, workspace_id, r) for r in records]
                mirror_refs = ensure_remote_bare_repos(
                    container, all_mirror_paths, args.dry_run,
                    refs=snapshot_mirror_refs(workspace_id),
                )
                transfer_reports: list[dict[str, Any]] = []
                for record in records:
                    emit_progress('push-mirror', relpath=record.relpath, transport='git')
                    transfer = push_snapshot_via_git(
                        repo=record_source(workspace_root, record),
                        container=container,
                        mirror_path=mirror_path_for(container_cache_root, workspace_id, record),
                        record=record,
                        workspace_id=workspace_id,
                        remote_refs=mirror_refs.get(mirror_path_for(container_cache_root, workspace_id, record)),
                    )
                    transfer_reports.append(transfer)
                    emit_progress('push-mirror-complete', **transfer)
                manifest['transfers'] = transfer_reports

                current_phase = 'upload-manifest'
                emit_progress(current_phase, manifest_path=manifest_path, apply_mode=args.apply_mode)
                upload_manifest(container, manifest_path, manifest, args.dry_run)

                observed_runtime_commits = None
                status = 'source-only'
                if args.apply_mode == 'materialize':
                    current_phase = 'materialize-runtime'
                    emit_progress(current_phase, runtime_root=runtime_root, install='skipped')
                    materialize_runtime(
                        container=container,
                        runtime_root=runtime_root,
                        container_cache_root=container_cache_root,
                        workspace_id=workspace_id,
                        marker_dirname=marker_dirname,
                        root_preserve_paths=root_preserve_paths,
                        records=records,
                        dry_run=args.dry_run,
                    )
                    current_phase = 'verify-runtime-commits'
                    emit_progress(current_phase, repo_count=len(records))
                    observed_runtime_commits = verify_runtime_commits(
                        container=container,
                        runtime_root=runtime_root,
                        records=records,
                        dry_run=args.dry_run,
                    )
                    if observed_runtime_commits != snapshot_commits:
                        upload_manifest(
                            container,
                            manifest_path,
                            final_manifest(
                                manifest,
                                status='failed',
                                reinstall_status='skipped-by-apply-mode',
                                runtime_commits=observed_runtime_commits,
                            ),
                            False,
                        )
                        summary = summary_payload(
                            status='failed',
                            server_name=args.server_name,
                            container_identity=args.container_identity,
                            workspace_id=workspace_id,
                            container_cache_root=container_cache_root,
                            records=records,
                            reinstall_status='skipped-by-apply-mode',
                            reason='runtime commit verification mismatch',
                            first_install=False,
                            observed_runtime_commits=observed_runtime_commits,
                        )
                        summary['apply_mode'] = args.apply_mode
                        summary['manifest_path'] = manifest_path
                        summary['transfers'] = transfer_reports
                        print(json_dump(summary))
                        return 1
                    status = 'materialized'

                current_phase = 'finalize-manifest'
                emit_progress(current_phase, manifest_path=manifest_path, apply_mode=args.apply_mode)
                upload_manifest(
                    container,
                    manifest_path,
                    final_manifest(
                        manifest,
                        status=status,
                        reinstall_status='skipped-by-apply-mode',
                        runtime_commits=observed_runtime_commits,
                    ),
                    False,
                )
                emit_progress('complete', status=status, apply_mode=args.apply_mode)
                summary = summary_payload(
                    status=status,
                    server_name=args.server_name,
                    container_identity=args.container_identity,
                    workspace_id=workspace_id,
                    container_cache_root=container_cache_root,
                    records=records,
                    reinstall_status='skipped-by-apply-mode',
                    reason=f'apply_mode={args.apply_mode} skipped runtime install/rebuild',
                    first_install=False,
                    observed_runtime_commits=observed_runtime_commits,
                )
                summary['apply_mode'] = args.apply_mode
                summary['manifest_path'] = manifest_path
                summary['transfers'] = transfer_reports
                print(json_dump(summary))
                keep_refs = True
                return 0
            finally:
                emit_progress('release-lock', lock_path=lock_path)
                release_container_lock(container, lock_path, args.dry_run)
        except Exception as exc:
            raise RuntimeError(f'{current_phase}: {exc}') from exc
    finally:
        if not keep_refs:
            cleanup_synthetic_refs(workspace_root, records)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Prepare or enforce remote code parity for a ready runtime.', allow_abbrev=False)
    subparsers = parser.add_subparsers(dest='command', required=True)

    sync = subparsers.add_parser('sync', help='Publish container-local mirrors and materialize runtime state.')
    sync.add_argument('--workspace-root', default='', help='Optional local state directory; bound --source worktrees are snapshotted directly.')
    sync.add_argument('--source', action='append', default=[], help='Use an actual external business worktree: vllm=/path or vllm-ascend=/path.')
    sync.add_argument('--workspace-id', required=True, help='Stable workspace id used for container cache namespacing.')
    sync.add_argument('--server-name', required=True)
    sync.add_argument('--runtime-root', required=True)
    sync.add_argument('--container-identity', required=True)
    sync.add_argument('--container-cache-root', default=DEFAULT_CONTAINER_CACHE_ROOT)
    sync.add_argument('--marker-dirname', default=DEFAULT_MARKER_DIRNAME)
    sync.add_argument('--preserve-path', action='append', default=[])
    sync.add_argument('--snapshot-id', default=None)
    sync.add_argument('--snapshot-file', help='Use a previously captured execution source descriptor without rereading worktrees.')
    sync.add_argument('--container-host', required=True)
    sync.add_argument('--container-port', type=int, required=True)
    sync.add_argument('--container-user', required=True)
    sync.add_argument('--dry-run', action='store_true')
    sync.add_argument('--print-manifest', action='store_true')
    sync.add_argument('--apply-mode', choices=('source-only', 'materialize'), default='materialize')
    return parser

def main() -> int:
    from mindie_coordinator._stdio import configure_stdio
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == 'sync':
            return run_sync(args)
        parser.error(f'unsupported command: {args.command}')
        return 2
    except Exception as exc:
        from mindie_coordinator.parity_support import LocalCommandError, RemoteCommandError
        payload: dict[str, Any] = {
            'status': 'failed',
            'reason': str(exc),
            'retryable': not (isinstance(exc, (LocalCommandError, ValueError, PermissionError))
                              or isinstance(exc, RemoteCommandError) and exc.returncode != 255),
        }
        for field in ('server_name', 'container_identity', 'workspace_id'):
            if hasattr(args, field):
                payload[field] = getattr(args, field)
        print(json_dump(payload))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
