"""Code parity: Git snapshot identity and remote materialization.

Owned by mindie-coordinator. Callers invoke this module as a library or as
``python -m mindie_coordinator.parity``. This never locates a consumer skill
script by filesystem path.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

from mindie_coordinator.build_inputs import (
    BUILD_INPUT_ENV_KEYS,
    DEPENDENCY_INSTALL_PATTERNS,
    VLLM_ASCEND_REINSTALL_PATTERNS,
    VLLM_REINSTALL_PATTERNS,
    build_input_fingerprints,
)
from mindie_coordinator.parity_support import (
    DEFAULT_DENYLIST,
    PROGRESS_SENTINEL,
    WORKSPACE_ID_PATTERN,
    SshEndpoint,
    ensure_local_git_identity,
    git,
    glob_match_any,
    is_git_worktree,
    json_dump,
    load_state,
    now_utc,
    quoted,
    repo_root_from,
    sanitize_repo_id,
    save_state,
    ssh_exec,
    ssh_exec_stream,
    ssh_stream_bytes_to_file,
    ssh_stream_to_file,
    update_state,
)


class ParityUnavailable(RuntimeError):
    """Source materialization failed; no job may be launched."""


def materialize_command(
    *,
    workspace_id: str,
    runtime_id: str,
    endpoint: dict,
    sources: dict[str, str],
    workspace_root: Path | str | None = None,
    source_snapshot: dict | None = None,
) -> list[str]:
    """Build the in-package parity CLI that materializes explicit source repos.

    Bound ``sources`` are the worktrees to snapshot. A Git parent of those
    trees is not required. ``workspace_root`` is only optional local state.
    """
    if not sources and not (source_snapshot or {}).get("records"):
        raise ValueError("materialization requires at least one fixed source")
    command = [
        sys.executable,
        "-m",
        "mindie_coordinator.parity",
        "sync",
        "--workspace-id",
        workspace_id,
        "--server-name",
        runtime_id,
        "--runtime-root",
        endpoint["root"],
        "--container-identity",
        runtime_id,
        "--container-host",
        endpoint["host"],
        "--container-port",
        str(endpoint["port"]),
        "--container-user",
        endpoint["user"],
        "--apply-mode",
        "materialize",
    ]
    if workspace_root:
        command.extend(["--workspace-root", str(Path(workspace_root).expanduser())])
    for name, path in sources.items():
        command.extend(["--source", name + "=" + path])
    if source_snapshot is not None:
        # The durable descriptor is already captured by submission. This file
        # transports that data; parity must not capture a newer working tree.
        state = Path(workspace_root or tempfile.gettempdir()) / ".mindie-local" / "fixed-inputs"
        state.mkdir(parents=True, exist_ok=True)
        transport_key = hashlib.sha256(json.dumps(source_snapshot, sort_keys=True).encode()).hexdigest()[:20]
        descriptor = state / (source_snapshot["id"] + '-' + transport_key + ".json")
        if not descriptor.exists():
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=state, delete=False) as out:
                json.dump(source_snapshot, out)
                temporary = Path(out.name)
            os.replace(temporary, descriptor)
        command.extend(["--snapshot-file", str(descriptor)])
    return command


def _write_materialized_build_source(root, data):
    """Publish fixed SCM/build metadata only after complete source materialization."""
    import json, os, tempfile, time
    from pathlib import Path
    started = time.monotonic()
    directory = Path(root) / '.mindie-runtime'
    if directory.is_symlink():
        raise ValueError('build metadata escaped owned runtime')
    directory.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.build-source-', dir=directory)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, sort_keys=True)
        os.replace(temporary, directory / 'build-source.json')
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return time.monotonic() - started


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
    container = SshEndpoint(host=endpoint['host'], port=int(endpoint['port']), user=endpoint['user'])
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
            git(repo, ['push', '--porcelain', '--atomic', git_remote_url(container, mirror),
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


def _export_existing_source_objects(host_endpoint, records, legacy_cache):
    import inspect
    from remote_dev.core.endpoint import resolve_endpoint
    from remote_dev.core.ssh_transport import run_remote_python
    from mindie_coordinator.preparation_process import PreparationCancelled, PreparationUncertain
    from mindie_coordinator.shared_source_objects import (
        copy_fixed_objects, export_container_objects, export_existing_objects,
    )
    source = inspect.getsource(copy_fixed_objects) + '\n' + inspect.getsource(export_container_objects)
    source += '\nimport json,sys\nprint(json.dumps(export_container_objects(json.load(sys.stdin))))\n'
    script = inspect.getsource(export_existing_objects)
    script += '\nimport json,sys\nprint(json.dumps(export_existing_objects(json.load(sys.stdin))))\n'
    result = run_remote_python(resolve_endpoint({**host_endpoint, 'root': '/', 'cwd': '/'}), script,
        {'records': records, 'legacy_cache': legacy_cache, 'export_source': source}, timeout_ms=360000)
    if result.get('status') == 'cancelled':
        raise PreparationCancelled('fixed object export cancelled; it was not replayed')
    if (result.get('status') == 'failed' and result.get('exit_code') is not None
            and result.get('remote_outcome') != 'unknown'):
        raise ParityUnavailable('fixed object export failed: ' + str(result.get('stderr_tail') or result))
    if result.get('status') not in {'copied', 'miss'} or result.get('remote_outcome') == 'unknown':
        raise PreparationUncertain('fixed object export did not complete; it was not replayed: ' + str(result))
    return result


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
        refs = git(Path(record.source_path), ['for-each-ref', '--count=256', '--sort=-refname',
            '--format=%(objectname) %(refname)', 'refs/mindie/inputs', 'refs/parity-transport', 'refs/parity'],
            check=False, timeout=10)
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
    'try:',
    '    count = len(os.sched_getaffinity(0))',
    'except Exception:',
    '    count = os.cpu_count() or 1',
    'print(max(1, min(int(count), 128)))',
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
STATE_FILENAME = 'runtime-state.json'
CONSENT_FILENAME = 'install-consents.json'
PARITY_BRANCH_NAME = 'parity-current'
TRANSFER_MODES = ('auto', 'git', 'bundle')

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

RUNTIME_INSTALL_ENV_KEYS = (
    'MAX_JOBS',
    'CMAKE_BUILD_PARALLEL_LEVEL',
    'CMAKE_BUILD_TYPE',
    'FETCHCONTENT_BASE_DIR',
    'XDG_CACHE_HOME',
    'PIP_CACHE_DIR',
    'PIP_CONFIG_FILE',
    'PIP_INDEX_URL',
    'PIP_TRUSTED_HOST',
    'MINDIE_BUILD_JOBS',
    'SOC_VERSION',
    'COMPILE_CUSTOM_KERNELS',
    'C_COMPILER',
    'CXX_COMPILER',
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


def remote_runtime_env_exports() -> list[str]:
    lines: list[str] = []
    for key in REMOTE_RUNTIME_ENV_PASSTHROUGH:
        if key in os.environ:
            lines.append(f'export {key}={quoted(os.environ[key])}')
    return lines


def redact_url_value(value: str) -> str:
    parts = value.split()
    redacted_parts: list[str] = []
    for part in parts:
        try:
            parsed = urlsplit(part)
        except ValueError:
            redacted_parts.append(part)
            continue
        if parsed.scheme and parsed.netloc and '@' in parsed.netloc:
            host = parsed.netloc.rsplit('@', 1)[1]
            redacted_parts.append(urlunsplit((parsed.scheme, f'***@{host}', parsed.path, parsed.query, parsed.fragment)))
        else:
            redacted_parts.append(part)
    return ' '.join(redacted_parts)


def redact_runtime_env(env: dict[str, str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in env.items():
        if key.endswith('_URL') or 'INDEX' in key or 'PATH' in key:
            redacted[key] = redact_url_value(value)
        else:
            redacted[key] = value
    return redacted


def resolved_root_preserve_paths(marker_dirname: str, extra_paths: list[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    for path in [*DEFAULT_ROOT_PRESERVE_PATHS, marker_dirname, *extra_paths]:
        normalized = validate_relative_posix_path(path, label='preserve path')
        if normalized not in ordered:
            ordered.append(normalized)
    return tuple(ordered)


@dataclass
class RuntimeInstallMarker:
    path: str
    record: dict[str, Any] | None


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


def load_runtime_state(repo_root: Path) -> dict[str, Any]:
    return load_state(repo_root, STATE_FILENAME, {'schema_version': 2, 'servers': {}})


def save_runtime_state(repo_root: Path, state: dict[str, Any]) -> Path:
    return save_state(repo_root, STATE_FILENAME, state)


def load_consent(repo_root: Path) -> dict[str, Any]:
    return load_state(repo_root, CONSENT_FILENAME, {'schema_version': 1, 'consents': {}})


def resolve_install_consent(repo_root: Path, server_name: str, container_identity: str) -> str:
    state = load_consent(repo_root)
    decision = (
        state.get('consents', {})
        .get(server_name, {})
        .get('containers', {})
        .get(container_identity, {})
        .get('decision')
    )
    return decision or 'unknown'


def cache_workspace_root(container_cache_root: str, workspace_id: str) -> str:
    return f"{container_cache_root.rstrip('/')}/workspaces/{workspace_id}"


def mirror_path_for(container_cache_root: str, workspace_id: str, record: SnapshotRecord) -> str:
    root = PurePosixPath(cache_workspace_root(container_cache_root, workspace_id)) / 'mirrors'
    if record.repo_id == 'workspace':
        return str(root / 'workspace.git')
    return str(root / 'nested' / f'{record.repo_id}.git')


def bundle_path_for(container_cache_root: str, workspace_id: str, record: SnapshotRecord) -> str:
    root = PurePosixPath(cache_workspace_root(container_cache_root, workspace_id)) / 'bundles'
    return str(root / f'{record.repo_id}-{record.commit}.bundle')


def git_remote_url(container: SshEndpoint, mirror_path: str) -> str:
    host = container.host
    if ':' in host and not host.startswith('['):
        host = f'[{host}]'
    user = quote(container.user, safe='')
    path = quote(validate_absolute_posix_path(mirror_path, label='mirror path'), safe='/')
    return f'ssh://{user}@{host}:{container.port}{path}'


def git_ssh_environment(container: SshEndpoint) -> dict[str, str]:
    from remote_dev.core.endpoint import Endpoint
    from remote_dev.core.ssh_transport import ssh_base_cmd

    env = os.environ.copy()
    env['GIT_TERMINAL_PROMPT'] = '0'
    cmd = list(
        ssh_base_cmd(Endpoint(host=container.host, port=container.port, user=container.user))
    )
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


def marker_path_for(runtime_root: str, marker_dirname: str) -> str:
    return str(PurePosixPath(runtime_root) / marker_dirname / 'runtime-install.json')


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


def cleanup_failed_mirror_hydration(container: SshEndpoint, mirror_path: str) -> None:
    script = '\n'.join(
        [
            'set +e',
            f'mirror={quoted(mirror_path)}',
            'for pid in $(pgrep -x git-receive-pack 2>/dev/null || true); do',
            '  cmd="$(tr "\\000" " " <"/proc/$pid/cmdline" 2>/dev/null || true)"',
            '  case "$cmd" in',
            '    *"$mirror"*)',
            '      pkill -TERM -P "$pid" >/dev/null 2>&1 || true',
            '      kill -TERM "$pid" >/dev/null 2>&1 || true',
            '      ;;',
            '  esac',
            'done',
            'sleep 1',
            'for pid in $(pgrep -x git-receive-pack 2>/dev/null || true); do',
            '  cmd="$(tr "\\000" " " <"/proc/$pid/cmdline" 2>/dev/null || true)"',
            '  case "$cmd" in',
            '    *"$mirror"*)',
            '      pkill -KILL -P "$pid" >/dev/null 2>&1 || true',
            '      kill -KILL "$pid" >/dev/null 2>&1 || true',
            '      ;;',
            '  esac',
            'done',
            'rm -rf "$mirror"',
        ]
    )
    try:
        ssh_exec(container, script, check=False)
    except Exception:
        pass


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


def push_snapshot_via_bundle(
    repo: Path,
    *,
    container: SshEndpoint,
    mirror_path: str,
    container_cache_root: str,
    record: SnapshotRecord,
    workspace_id: str,
) -> dict[str, Any]:
    target_ref = f'refs/parity/{workspace_id}/current'
    remote_bundle_path = bundle_path_for(container_cache_root, workspace_id, record)
    local_bundle = tempfile.NamedTemporaryFile(prefix='parity-bundle-', suffix='.bundle', delete=False)
    local_bundle.close()
    local_bundle_path = Path(local_bundle.name)
    started = time.monotonic()
    try:
        git(repo, ['bundle', 'create', str(local_bundle_path), record.ref], timeout=DEFAULT_GIT_TRANSPORT_TIMEOUT_SECONDS)
        bundle_payload = local_bundle_path.read_bytes()
        bundle_bytes = len(bundle_payload)
        ssh_stream_bytes_to_file(container, remote_bundle_path, bundle_payload)
        script = '\n'.join(
            [
                'set -eo pipefail',
                f'mkdir -p {quoted(str(PurePosixPath(mirror_path).parent))}',
                f'if [ ! -d {quoted(mirror_path)} ]; then git init --bare {quoted(mirror_path)} >/dev/null; fi',
                (
                    f'git -C {quoted(mirror_path)} fetch --force {quoted(remote_bundle_path)} '
                    f'{quoted(record.ref + ":" + target_ref)} '
                    f'{quoted(record.ref + ":refs/heads/" + PARITY_BRANCH_NAME)} >/dev/null'
                ),
                f'rm -f {quoted(remote_bundle_path)}',
            ]
        )
        ssh_exec(container, script)
        return {
            'repo': record.relpath,
            'transport': 'bundle',
            'elapsed_seconds': round(time.monotonic() - started, 6),
            'bundle_bytes': bundle_bytes,
        }
    except Exception:
        cleanup_failed_mirror_hydration(container, mirror_path)
        raise
    finally:
        local_bundle_path.unlink(missing_ok=True)


def push_snapshot_to_mirror(
    repo: Path,
    *,
    container: SshEndpoint,
    mirror_path: str,
    container_cache_root: str,
    record: SnapshotRecord,
    workspace_id: str,
    dry_run: bool,
    transport: str = 'auto',
    remote_refs: dict[str, str] | None = None,
) -> dict[str, Any]:
    if transport not in TRANSFER_MODES:
        raise ValueError(f'unsupported parity transport: {transport}')
    if dry_run:
        return {'repo': record.relpath, 'transport': f'{transport}-dry-run'}

    git_error: Exception | None = None
    if transport in {'auto', 'git'}:
        try:
            return push_snapshot_via_git(
                repo,
                container=container,
                mirror_path=mirror_path,
                record=record,
                workspace_id=workspace_id,
                remote_refs=remote_refs,
            )
        except Exception as exc:
            if transport == 'git':
                raise
            git_error = exc
            emit_progress(
                'push-mirror-fallback',
                relpath=record.relpath,
                from_transport='git',
                to_transport='bundle',
                reason=str(exc),
            )

    result = push_snapshot_via_bundle(
        repo,
        container=container,
        mirror_path=mirror_path,
        container_cache_root=container_cache_root,
        record=record,
        workspace_id=workspace_id,
    )
    if git_error is not None:
        result['fallback_from'] = 'git'
    return result

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


def container_repo_path(runtime_root: str, record: SnapshotRecord) -> str:
    if record.relpath in ('', '.'):
        return runtime_root
    return str(PurePosixPath(runtime_root) / record.relpath)


def prepare_isolated_root_script(runtime_root: str) -> str:
    """Create or reset an isolated task root. Never pip-uninstall image packages."""
    hostname_repair = (
        'if command -v hostname >/dev/null 2>&1; then '
        '( flock -x 9; h="$(hostname 2>/dev/null || true)"; '
        'if [ -n "$h" ] && ! grep -q -F "$h" /etc/hosts 2>/dev/null; then '
        'echo "127.0.0.1 $h" >> /etc/hosts; fi ) 9>/tmp/mindie-hostname.lock; fi'
    )
    return '\n'.join(
        [
            'set -eo pipefail',
            f'mkdir -p {quoted(runtime_root)}',
            hostname_repair,
            f'rm -rf {quoted(str(PurePosixPath(runtime_root) / "vllm"))} {quoted(str(PurePosixPath(runtime_root) / "vllm-ascend"))}',
            f'rm -rf {quoted(str(PurePosixPath(runtime_root) / ".git/modules/vllm"))} {quoted(str(PurePosixPath(runtime_root) / ".git/modules/vllm-ascend"))}',
        ]
    )


def task_python_exports(python: str) -> list[str]:
    """Pin every pip/build/executable selection to the task-owned interpreter."""
    q = quoted(python)
    return [
        f'export PYTHON={q}',
        'export PIP="$PYTHON -m pip"',
        'export HI_PYTHON="$PYTHON"',
        'export Python3_EXECUTABLE="$PYTHON"',
        'export Python_EXECUTABLE="$PYTHON"',
        'PYTHON_BIN_DIR="$(dirname "$PYTHON")"',
        'if [ -n "${MINDIE_PYTHON_SHIM_DIR:-}" ]; then',
        '  ln -sf "$PYTHON" "$MINDIE_PYTHON_SHIM_DIR/python"',
        '  ln -sf "$PYTHON" "$MINDIE_PYTHON_SHIM_DIR/python3"',
        'fi',
        'export PATH="${MINDIE_PYTHON_SHIM_DIR:+$MINDIE_PYTHON_SHIM_DIR:}$PYTHON_BIN_DIR:$PATH"',
        'hash -r',
        'export CMAKE_ARGS="-DPython3_EXECUTABLE=\\"$PYTHON\\" -DPython_EXECUTABLE=\\"$PYTHON\\""',
        f'test -x {q}',
    ]


def render_git_clean(repo_dir: str, preserve_paths: tuple[str, ...]) -> str:
    parts = ['git', '-C', quoted(repo_dir), 'clean', '-ffd']
    for path in preserve_paths:
        parts.extend(['-e', quoted(path)])
    parts.append('>/dev/null')
    return ' '.join(parts)


def materialize_runtime(
    *,
    container: SshEndpoint,
    runtime_root: str,
    container_cache_root: str,
    workspace_id: str,
    marker_dirname: str,
    root_preserve_paths: tuple[str, ...],
    records: list[SnapshotRecord],
    dry_run: bool,
) -> None:
    record_by_relpath = {record.relpath: record for record in records}
    if '.' in record_by_relpath:
        roots = [record_by_relpath['.']]
    else:
        roots = [record for record in records if '/' not in record.relpath]
    if not roots:
        raise ValueError('materialize requires explicit source records (vllm, vllm-ascend)')
    parity_tracking_ref = f'refs/remotes/parity/{PARITY_BRANCH_NAME}'

    def render_repo_step(record: SnapshotRecord) -> str:
        repo_dir = container_repo_path(runtime_root, record)
        mirror_path = mirror_path_for(container_cache_root, workspace_id, record)
        lines = ['set -eo pipefail', f'mkdir -p {quoted(str(PurePosixPath(repo_dir).parent))}']
        if record.relpath in ('', '.'):
            lines.append(f'if [ ! -e {quoted(str(PurePosixPath(repo_dir) / ".git"))} ]; then git init {quoted(repo_dir)} >/dev/null; fi')
        else:
            lines.append(
                f'if [ ! -e {quoted(str(PurePosixPath(repo_dir) / ".git"))} ]; then rm -rf {quoted(repo_dir)} && git clone --no-checkout {quoted(mirror_path)} {quoted(repo_dir)} >/dev/null; fi'
            )
        lines.extend(
            [
                f'git -C {quoted(repo_dir)} remote get-url parity >/dev/null 2>&1 || git -C {quoted(repo_dir)} remote add parity {quoted(mirror_path)}',
                f'git -C {quoted(repo_dir)} remote set-url parity {quoted(mirror_path)}',
                f'git -C {quoted(repo_dir)} fetch --force --no-recurse-submodules parity {quoted(PARITY_BRANCH_NAME + ":" + parity_tracking_ref)} >/dev/null',
                f'git -C {quoted(repo_dir)} checkout -B parity/current {quoted(parity_tracking_ref)} >/dev/null',
                f'git -C {quoted(repo_dir)} reset --hard {quoted(parity_tracking_ref)} >/dev/null',
            ]
        )
        lines.append(render_git_clean(repo_dir, root_preserve_paths))
        for child in record.submodules:
            child_relpath = child['path'] if record.relpath in ('', '.') else f"{record.relpath}/{child['path']}"
            child_record = record_by_relpath[child_relpath]
            child_mirror = mirror_path_for(container_cache_root, workspace_id, child_record)
            submodule_url_key = f"submodule.{child['name']}.url"
            lines.extend(
                [
                    f'git -C {quoted(repo_dir)} config {quoted(submodule_url_key)} {quoted(child_mirror)}',
                    f'git -C {quoted(repo_dir)} submodule sync -- {quoted(child["path"])} >/dev/null || true',
                ]
            )
        return '\n'.join(lines)

    def collect_scripts(record: SnapshotRecord, out: list[str]) -> None:
        emit_progress('materialize-repo', relpath=record.relpath)
        out.append(render_repo_step(record))
        for child in record.submodules:
            child_relpath = child['path'] if record.relpath in ('', '.') else f"{record.relpath}/{child['path']}"
            collect_scripts(record_by_relpath[child_relpath], out)

    if dry_run:
        return
    parts: list[str] = [
        'set -eo pipefail',
        f'mkdir -p {quoted(runtime_root)}',
        f'mkdir -p {quoted(str(PurePosixPath(runtime_root) / marker_dirname))}',
    ]
    repo_scripts: list[str] = []
    for root_record in roots:
        collect_scripts(root_record, repo_scripts)
    parts.extend(repo_scripts)
    ssh_exec(container, '\n'.join(parts))

def reinstall_required_for_repo(record: SnapshotRecord, patterns: tuple[str, ...]) -> bool:
    return any(glob_match_any(path, patterns) for path in record.changed_paths)


def dependency_install_required_for_repo(record: SnapshotRecord) -> bool:
    return any(glob_match_any(path, DEPENDENCY_INSTALL_PATTERNS) for path in record.changed_paths)


def committed_changed_paths(repo: Path, last_head: str, current_head: str) -> list[str] | None:
    """Paths changed between the last synced HEAD and the current HEAD.

    Returns ``None`` when the diff cannot be computed (e.g. the old commit was
    garbage-collected), in which case callers must fall back to the
    conservative reinstall behavior.
    """
    if last_head == current_head:
        return []
    result = git(repo, ['diff', '--name-only', f'{last_head}..{current_head}'], check=False)
    if result.returncode != 0:
        return None
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def runtime_install_step_script(
    *,
    runtime_root: str,
    marker_dirname: str,
    container_identity: str,
    step: str,
    uninstall_packages: tuple[str, ...] = (),
    python: str | None = None,
) -> str:
    lines = ['set -euo pipefail', f'cd {quoted(runtime_root)}']
    lines.extend(remote_runtime_env_exports())
    lines.append(f'export MINDIE_RUNTIME_ROOT={quoted(runtime_root)}')
    if step != 'write-marker':
        lines.extend(DEFAULT_ENV_PREAMBLE)
    if python and step != 'write-marker':
        lines.extend(task_python_exports(python))
    if step in {'verify-imports', 'verify-deps'}:
        # Shared native reuse copies outputs instead of installing an editable
        # package. Verify this execution view, as the final profile probe does.
        lines.extend([
            'if [ -f "$MINDIE_RUNTIME_ROOT/.mindie-runtime/shared-native.json" ]; then',
            '  export PYTHONPATH="$MINDIE_RUNTIME_ROOT/.mindie-runtime/metadata:$MINDIE_RUNTIME_ROOT/vllm:$MINDIE_RUNTIME_ROOT/vllm-ascend${PYTHONPATH:+:$PYTHONPATH}"',
            'fi',
        ])
    if step in {'install-vllm', 'install-vllm-ascend', 'install-vllm-ascend-incremental', 'install-vllm-ascend-requirements', 'check-build-compat'}:
        source_name = 'vllm' if step == 'install-vllm' else 'vllm-ascend'
        # Only package-created fixed input data enters the build shell. Never
        # derive versions from the synthetic commit's truncated history.
        lines.extend([
            'if [ -f "$MINDIE_RUNTIME_ROOT/.mindie-runtime/build-source.json" ]; then',
            '  eval "$("$PYTHON" - ' + quoted(source_name) + ' "$MINDIE_RUNTIME_ROOT/.mindie-runtime/build-source.json" <<\'MINDIE_BUILD_INPUTS\'',
            'import json, shlex, sys',
            'spec = json.load(open(sys.argv[2]))',
            'for key, value in spec.get("build_env", {}).items():',
            '    if not __import__("re").fullmatch(r"[A-Z_][A-Z0-9_]*", key): raise ValueError("invalid build environment key")',
            '    print("export " + key + "=" + shlex.quote(value))',
            'version = spec["versions"][sys.argv[1]]["version"]',
            'print("export SETUPTOOLS_SCM_PRETEND_VERSION=" + shlex.quote(version))',
            'MINDIE_BUILD_INPUTS',
            ')"',
            'fi',
        ])
    if step in {'install-vllm', 'install-vllm-ascend', 'install-vllm-ascend-requirements'}:
        lines.extend(
            [
                'emit_progress() {',
                "  python3 - \"$1\" \"$2\" \"$3\" \"${4:-}\" <<'PY' >&2",
                'import json',
                'import sys',
                'payload = {"phase": sys.argv[1], "message": sys.argv[2]}',
                'if sys.argv[4]: payload["remote_log"] = sys.argv[4]',
                'if len(sys.argv) > 3 and sys.argv[3]:',
                '    try:',
                '        payload["expected_seconds"] = int(sys.argv[3])',
                '    except ValueError:',
                '        pass',
                f'print("{PROGRESS_SENTINEL}" + json.dumps(payload, ensure_ascii=False))',
                'PY',
                '}',
                'run_with_log_progress() {',
                '  phase="$1"',
                '  message="$2"',
                '  expected_seconds="$3"',
                '  log_file="$4"',
                '  shift 4',
                '  "$@" >"$log_file" 2>&1 &',
                '  pid=$!',
                '  emit_progress "$phase" "$message" "$expected_seconds" "$log_file"',
                '  start_ts=$(date +%s)',
                '  while kill -0 "$pid" 2>/dev/null; do',
                '    sleep 8',
                '    if ! kill -0 "$pid" 2>/dev/null; then',
                '      break',
                '    fi',
                '    elapsed=$(( $(date +%s) - start_ts ))',
                '    if [ -s "$log_file" ]; then',
                '      last_line="$(tail -n 1 "$log_file" 2>/dev/null | tr -d \"\\r\" | sed \"s/[^[:print:]\\t]//g\" | cut -c1-180)"',
                '      if [ -n "$last_line" ]; then',
                '        emit_progress "$phase" "$message - $last_line" "$expected_seconds"',
                '      else',
                '        emit_progress "$phase" "$message - still working (elapsed ${elapsed}s)" "$expected_seconds"',
                '      fi',
                '    else',
                '      emit_progress "$phase" "$message - still working (elapsed ${elapsed}s)" "$expected_seconds"',
                '    fi',
                '  done',
                '  set +e',
                '  wait "$pid"',
                '  status=$?',
                '  set -e',
                '  if [ "$status" -ne 0 ]; then',
                '    tail -n 160 "$log_file" >&2 || true',
                '  fi',
                '  return "$status"',
                '}',
                'run_with_progress() {',
                '  phase="$1"',
                '  message="$2"',
                '  expected_seconds="$3"',
                '  shift 3',
                '  mkdir -p "$MINDIE_RUNTIME_ROOT/.mindie-runtime/prepare-logs"',
                '  log_file="$(mktemp "$MINDIE_RUNTIME_ROOT/.mindie-runtime/prepare-logs/${phase}.XXXXXX")"',
                '  set +e',
                '  run_with_log_progress "$phase" "$message" "$expected_seconds" "$log_file" "$@"',
                '  status=$?',
                '  set -e',
                '  return "$status"',
                '}',
                'configure_pip_index() {',
                '  unset PIP_EXTRA_INDEX_URL',
                '  export PIP_CONFIG_FILE=/dev/null',
                f'  export PIP_INDEX_URL={quoted(PIP_INDEX_URL)}',
                f'  export PIP_TRUSTED_HOST={quoted(PIP_TRUSTED_HOST)}',
                f'  emit_progress "runtime-pip-index" "using pip index {PIP_INDEX_NAME}" 30',
                '}',
                'pip_install_fast() {',
                '  phase="$1"',
                '  message="$2"',
                '  expected_seconds="$3"',
                '  shift 3',
                '  configure_pip_index',
                f'  run_with_progress "$phase" "$message via {PIP_INDEX_NAME}" "$expected_seconds" "$PYTHON" -m pip "$@"',
                '}',
                'install_editable_fast() {',
                '  phase="$1"',
                '  message="$2"',
                '  target_dir="$3"',
                '  expected_seconds="$4"',
                '  shift 4',
                '  cd "$target_dir"',
                '  configure_pip_index',
                f'  run_with_progress "$phase" "$message via {PIP_INDEX_NAME}" "$expected_seconds" "$@"',
                '}',
            ]
        )

    if step == 'uninstall':
        pkg_args = ' '.join(uninstall_packages) if uninstall_packages else 'vllm vllm-ascend vllm_ascend'
        lines.append(f'"$PYTHON" -m pip uninstall -y {pkg_args} >/dev/null 2>&1 || true')
    elif step == 'install-vllm':
        lines.extend(
            [
                f'cd {quoted(str(PurePosixPath(runtime_root) / "vllm"))}',
                'export VLLM_TARGET_DEVICE=empty',
                'export TORCH_DEVICE_BACKEND_AUTOLOAD=0',
                'install_editable_fast "runtime-install-vllm" "building editable vllm" . 900 "$PYTHON" -m pip install --no-deps -e . --no-build-isolation',
            ]
        )
    elif step == 'install-vllm-ascend-requirements':
        # The hardware-coupled stack (torch / torch_npu / triton-ascend /
        # torchvision / torchaudio) is decided per package at install time:
        #   - already installed in the image -> drop the pinned line (avoid
        #     downgrading the image's tested runtime);
        #   - genuinely missing -> keep it and add the public Ascend
        #     extra-index so it resolves from mirrors.huaweicloud.com/ascend,
        #     exactly like vllm-ascend/Dockerfile pulls torch_npu/triton.
        # Regex matches the requirement name at line start, terminated by a
        # version op / marker / comment / EOL, so ``torch`` never swallows
        # ``torchvision``. Constant names are already PEP 503 canonical.
        _names = sorted(
            {name.strip().lower() for name in IMAGE_PROVIDED_REQUIREMENT_NAMES},
            key=len,
            reverse=True,
        )
        _pkg_group = '(' + '|'.join(_names) + ')'
        _line_re = r'^[[:space:]]*' + _pkg_group + r'[[:space:]]*($|[<>=!~;#[])'
        # awk uses the captured package name to test installability on the
        # image, so keep the group in a shell-friendly single expression.
        # The terminated line regex is the MATCH gate: without it a bare
        # prefix would misclassify e.g. ``torchao`` as image-provided ``torch``
        # and silently drop the line from the install list.
        lines.extend(
            [
                f'cd {quoted(str(PurePosixPath(runtime_root) / "vllm-ascend"))}',
                'filtered_req="$(mktemp -t mindie-req.XXXXXX.txt)"',
                'dropped=""',
                'kept_hw=""',
                'while IFS= read -r req_line || [ -n "$req_line" ]; do',
                f'  if printf "%s" "$req_line" | grep -qiE {quoted(_line_re)}; then',
                f'    pkg="$(printf "%s" "$req_line" | grep -oiE {quoted("^[[:space:]]*" + _pkg_group)} | tr -d "[:space:]" | tr "[:upper:]" "[:lower:]")"',
                '  else',
                '    pkg=""',
                '  fi',
                '  if [ -n "$pkg" ]; then',
                # Treat the package as image-provided if pip knows about it,
                # regardless of the pinned version (prevents downgrade).
                '    if "$PYTHON" -m pip show "$pkg" >/dev/null 2>&1; then',
                '      dropped="$dropped $pkg"',
                '      continue',
                '    else',
                '      kept_hw="$kept_hw $pkg"',
                '    fi',
                '  fi',
                '  printf "%s\\n" "$req_line" >> "$filtered_req"',
                'done < requirements.txt',
                '[ -n "$dropped" ] && emit_progress "runtime-req-filter" '
                '"image provides:$dropped" 5 || true',
                # If any hardware package is genuinely missing, allow the
                # public Ascend index for this install only.
                'req_index_args=""',
                '[ -n "$kept_hw" ] && emit_progress "runtime-req-ascend-index" '
                f'"pulling from ascend index:$kept_hw" 5 && req_index_args={quoted("--extra-index-url " + ASCEND_PIP_EXTRA_INDEX_URL + " --trusted-host " + ASCEND_PIP_TRUSTED_HOST)} || true',
                'pip_install_fast "runtime-install-vllm-ascend-requirements" '
                '"installing vllm-ascend requirements (image stack reconciled)" 900 '
                'install $req_index_args -r "$filtered_req"',
                'rm -f "$filtered_req"',
            ]
        )
    elif step == 'check-build-compat':
        # Fail fast BEFORE the multi-minute custom-ops build when the
        # checked-out vllm-ascend submodule is not version-matched to the base
        # image. vllm-ascend/CMakeLists.txt hard-pins the expected torch
        # version (``if(NOT TORCH_VERSION VERSION_EQUAL "X.Y.Z")
        # message(FATAL_ERROR ...)``); if it disagrees with the image's torch
        # the cmake configure aborts deep in the build (the confusing
        # "kineto_LIBRARY-NOTFOUND" + FATAL_ERROR we hit on qwen35-125). Detect
        # the mismatch up front and tell the user how to resolve it instead of
        # wasting the build time. This is a preflight, never a silent skip.
        lines.extend(
            [
                f'cd {quoted(str(PurePosixPath(runtime_root) / "vllm-ascend"))}',
                'required_torch="$(grep -oE \'VERSION_EQUAL[[:space:]]*"[0-9]+\\.[0-9]+\\.[0-9]+"\' CMakeLists.txt 2>/dev/null | grep -oE \'[0-9]+\\.[0-9]+\\.[0-9]+\' | head -1)"',
                'if [ -z "$required_torch" ]; then',
                '  echo "build-compat: no torch version pin in CMakeLists.txt; skipping preflight"',
                'else',
                '  installed_torch="$("$PYTHON" -c \'import torch; print(torch.__version__.split("+")[0])\' 2>/dev/null || true)"',
                '  if [ -z "$installed_torch" ]; then',
                '    echo "ERROR: cannot import torch in the image to check build compatibility" >&2',
                '    exit 1',
                '  fi',
                '  if [ "$required_torch" != "$installed_torch" ]; then',
                '    echo "ERROR: vllm-ascend custom-ops build requires torch==$required_torch but this image ships torch==$installed_torch." >&2',
                '    echo "The bound vllm-ascend sources are not version-matched to this environment." >&2',
                '    echo "Bind a vllm-ascend worktree whose CMakeLists.txt expects torch $installed_torch," >&2',
                '    echo "or request an environment recipe/image that provides torch $required_torch." >&2',
                '    echo "The coordinator will not skip the custom-ops build or retarget the image stack." >&2',
                '    exit 1',
                '  fi',
                '  echo "build-compat: image torch $installed_torch matches vllm-ascend requirement"',
                'fi',
            ]
        )
    elif step == 'install-vllm-ascend-incremental':
        import inspect
        from mindie_coordinator import native_incremental, native_kernel_recipe, runtime_profile
        script = Path(native_incremental.__file__).read_text(encoding='utf-8')
        compiler = Path(native_kernel_recipe.__file__).read_text(encoding='utf-8').replace('from __future__ import annotations\n', '')
        lines.extend([
            '"$PYTHON" - "$MINDIE_RUNTIME_ROOT" <<\'MINDIE_NATIVE_INCREMENTAL\'',
            script,
            compiler,
            inspect.getsource(runtime_profile.compiled_opc_recipe),
            'import sys',
            'build_incremental_kernel(Path(sys.argv[1]), compile_recipe=lambda root, plan, env: '
            'compile_kernel_recipe(root, plan, env, read_recipe=compiled_opc_recipe))',
            'MINDIE_NATIVE_INCREMENTAL',
        ])
    elif step == 'install-vllm-ascend':
        lines.extend(
            [
                f'cd {quoted(str(PurePosixPath(runtime_root) / "vllm-ascend"))}',
                'install_editable_fast "runtime-install-vllm-ascend" "building editable vllm-ascend custom ops" . 2400 "$PYTHON" -m pip install --no-deps -v -e . --no-build-isolation',
            ]
        )
    elif step == 'verify-imports':
        lines.extend(
            [
                '"$PYTHON" - <<\'PY\'',
                'import sys',
                'import torch',
                'import torch_npu  # noqa: F401',
                'import vllm',
                'import vllm_ascend',
                'print(f"editable-import-smoke=ok python={sys.executable} torch={torch.__version__} vllm={getattr(vllm, \'__version__\', \'unknown\')}")',
                'PY',
            ]
        )
    elif step == 'verify-deps':
        # Check vllm-ascend deps only.  vllm-ascend intentionally overrides
        # some vllm constraints (e.g. opencv-python-headless) to keep numpy
        # compatible with CANN, so checking vllm deps would false-positive.
        lines.extend(
            [
                '"$PYTHON" - <<\'PY\'',
                'import sys',
                'from importlib.metadata import requires, version as pkg_version',
                'from packaging.requirements import Requirement',
                'from packaging.utils import canonicalize_name',
                'from packaging.version import InvalidVersion, Version',
                '',
                'def importable(module):',
                '    try:',
                '        __import__(module)',
                '        return True',
                '    except Exception:',
                '        return False',
                '',
                'def public_version(value):',
                '    try:',
                '        return Version(value).public',
                '    except InvalidVersion:',
                '        return value.split("+", 1)[0]',
                '',
                'def requirement_satisfied(req, installed):',
                '    if not req.specifier:',
                '        return True',
                '    if req.specifier.contains(installed, prereleases=True):',
                '        return True',
                '    public = public_version(installed)',
                '    if public != installed and req.specifier.contains(public, prereleases=True):',
                '        return True',
                '    # Paired vLLM Ascend images provide torch_npu as runtime state.',
                '    if canonicalize_name(req.name) == "torch-npu" and importable("torch_npu"):',
                '        return True',
                '    return False',
                '',
                'errors = []',
                'try:',
                '    reqs = requires("vllm-ascend") or []',
                'except Exception:',
                '    reqs = []',
                'for raw in reqs:',
                '    try:',
                '        r = Requirement(raw)',
                '        if r.marker and not r.marker.evaluate():',
                '            continue',
                '        try:',
                '            installed = pkg_version(r.name)',
                '        except Exception:',
                '            if canonicalize_name(r.name) == "torch-npu" and importable("torch_npu"):',
                '                continue',
                '            raise',
                '        if not requirement_satisfied(r, installed):',
                '            errors.append(f"{r.name}{r.specifier} (installed {installed})")',
                '    except Exception as exc:',
                '        # Do NOT swallow: a missing package (raised above) or an',
                '        # unparseable requirement is a real verification failure.',
                '        errors.append(f"{raw!r} could not be verified: {exc!r}")',
                'if errors:',
                '    for e in errors:',
                '        print(f"MISMATCH: {e}", file=sys.stderr)',
                '    sys.exit(1)',
                'print("dependency-check=ok")',
                'PY',
            ]
        )
    elif step == 'write-marker':
        lines.extend(
            [
                f'mkdir -p {quoted(str(PurePosixPath(runtime_root) / marker_dirname))}',
                (
                    'cat > '
                    + quoted(marker_path_for(runtime_root, marker_dirname))
                    + " <<'JSON'\n"
                    + json.dumps(
                        {
                            'container_identity': container_identity,
                            'runtime_root': runtime_root,
                            'updated_at': now_utc(),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + '\nJSON'
                ),
            ]
        )
    else:
        raise ValueError(f'unknown runtime install step: {step}')
    return '\n'.join(lines)


def run_runtime_install_step(
    *,
    container: SshEndpoint,
    runtime_root: str,
    marker_dirname: str,
    container_identity: str,
    step: str,
    stream_progress: bool = False,
    uninstall_packages: tuple[str, ...] = (),
    python: str | None = None,
    on_progress=None,
    log_path=None,
    process=None,
) -> None:
    script = runtime_install_step_script(
        runtime_root=runtime_root,
        marker_dirname=marker_dirname,
        container_identity=container_identity,
        step=step,
        uninstall_packages=uninstall_packages,
        python=python,
    )
    if stream_progress or on_progress is not None or log_path is not None or process is not None:
        ssh_exec_stream(container, script, stream_progress=stream_progress,
                        on_progress=on_progress, log_path=log_path, process=process)
    else:
        ssh_exec(container, script)


def read_runtime_install_marker(
    *,
    container: SshEndpoint,
    runtime_root: str,
    marker_dirname: str,
) -> RuntimeInstallMarker:
    # The marker read is a read-only remote cat, so it is safe under dry-run
    # and takes no dry_run switch.
    path = marker_path_for(runtime_root, marker_dirname)
    script = '\n'.join(
        [
            'set -eo pipefail',
            f'if [ -f {quoted(path)} ]; then cat {quoted(path)}; fi',
        ]
    )
    result = ssh_exec(container, script)
    content = result.stdout.strip()
    if not content:
        return RuntimeInstallMarker(path=path, record=None)
    try:
        return RuntimeInstallMarker(path=path, record=json.loads(content))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'container runtime-install marker at {path} is invalid JSON: {exc}') from exc


def first_install_needed(marker: RuntimeInstallMarker, container_identity: str, runtime_root: str) -> bool:
    if marker.record is None:
        return True
    return marker.record.get('container_identity') != container_identity or marker.record.get('runtime_root') != runtime_root


def verify_runtime_commits_map(
    *,
    container: SshEndpoint,
    runtime_root: str,
    expected: dict[str, str],
    include_untracked: bool = False,
) -> dict[str, str]:
    lines = ['set -eo pipefail']
    for relpath in expected:
        repo_dir = runtime_root if relpath in ('', '.') else str(PurePosixPath(runtime_root) / relpath)
        clean = f"git -C {quoted(repo_dir)} diff --quiet HEAD --"
        if include_untracked:
            clean += f' && test -z "$(git -C {quoted(repo_dir)} ls-files --others --exclude-standard)"'
        lines.append(
            f"if {clean}; then "
            f"printf '%s %s\\n' {quoted(relpath)} \"$(git -C {quoted(repo_dir)} rev-parse HEAD)\"; "
            f"else printf '%s %s\\n' {quoted(relpath)} dirty-runtime; fi"
        )
    result = ssh_exec(container, '\n'.join(lines))
    observed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        relpath, commit = line.split(maxsplit=1)
        observed[relpath] = commit
    return observed


def verify_runtime_commits(
    *,
    container: SshEndpoint,
    runtime_root: str,
    records: list[SnapshotRecord],
    dry_run: bool,
) -> dict[str, str]:
    expected = {record.relpath: record.commit for record in records}
    if dry_run:
        return expected
    return verify_runtime_commits_map(container=container, runtime_root=runtime_root, expected=expected)


def workspace_fingerprint(workspace_root: Path, source_roots: dict[str, Path] | None = None) -> str:
    """Hash dirty content, not just the names/status of dirty files.

    A second edit of an already modified file has identical porcelain status.
    Hash tracked diffs and untracked bytes so it cannot disappear in the fast
    path. Prefix the format to invalidate the old status-only cache.
    """
    tree = discover_repo_tree(workspace_root, '.', None, source_roots)
    hasher = hashlib.sha256()
    for node in iter_postorder(tree):
        repo = node.repo_path
        head = git_head(repo) or ''
        dirty = git(repo, ['diff', '--binary', '--no-ext-diff', 'HEAD', '--']).stdout if head else ''
        for token in (node.relpath, head, dirty):
            hasher.update(token.encode('utf-8'))
            hasher.update(b'\0')
        untracked = git(repo, ['ls-files', '--others', '--exclude-standard', '-z']).stdout
        for relpath in sorted(filter(None, untracked.split('\0'))):
            path = repo / relpath
            hasher.update(relpath.encode('utf-8') + b'\0')
            hasher.update(str(path.lstat().st_mode).encode('ascii') + b'\0')
            if path.is_symlink():
                hasher.update(os.readlink(path).encode('utf-8'))
            else:
                with path.open('rb') as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                        hasher.update(chunk)
            hasher.update(b'\0')
    build_env = {key: os.environ[key] for key in BUILD_INPUT_ENV_KEYS if key in os.environ}
    hasher.update(json.dumps(build_env, sort_keys=True).encode('utf-8'))
    return 'content-v2:' + hasher.hexdigest()


def changed_build_inputs(current: dict[str, str], installed: dict[str, str] | None) -> tuple[bool, bool]:
    if not installed:
        return True, True  # One conservative rebuild migrates legacy state.
    dependencies = current.get('dependencies') != installed.get('dependencies')
    native = dependencies or any(current.get(key) != installed.get(key) for key in ('native', 'build_env'))
    return native, dependencies


def read_runtime_install_env(
    *,
    container: SshEndpoint,
    runtime_root: str,
    dry_run: bool,
) -> dict[str, str]:
    if dry_run:
        return {}
    lines = ['set -euo pipefail', f'mkdir -p {quoted(runtime_root)}', f'cd {quoted(runtime_root)}']
    lines.extend(remote_runtime_env_exports())
    lines.append(f'export MINDIE_RUNTIME_ROOT={quoted(runtime_root)}')
    lines.extend(DEFAULT_ENV_PREAMBLE)
    lines.extend(
        [
            '"$PYTHON" - <<\'PY\'',
            'import json',
            'import os',
            f'keys = {json.dumps(RUNTIME_INSTALL_ENV_KEYS)}',
            'env = {key: os.environ[key] for key in keys if key in os.environ}',
            'print(json.dumps(env, sort_keys=True))',
            'PY',
        ]
    )
    result = ssh_exec(container, '\n'.join(lines))
    raw = json.loads(result.stdout.strip() or '{}')
    return redact_runtime_env({str(key): str(value) for key, value in raw.items()})


def update_runtime_state(
    *,
    repo_root: Path,
    server_name: str,
    container_identity: str,
    runtime_root: str,
    container_cache_root: str,
    marker_dirname: str,
    records: list[SnapshotRecord],
    first_reinstall_completed: bool,
    runtime_install_env: dict[str, str] | None,
) -> None:
    def apply_update(state: dict[str, Any]) -> None:
        server_state = state.setdefault('servers', {}).setdefault(server_name, {})
        containers = server_state.setdefault('containers', {})
        containers[container_identity] = {
            'runtime_root': runtime_root,
            'container_cache_root': container_cache_root,
            'marker_dirname': marker_dirname,
            'last_sync_at': now_utc(),
            'first_reinstall_completed': first_reinstall_completed,
            'last_snapshot_commits': {record.relpath: record.commit for record in records},
            'last_head_commits': {record.relpath: record.source_head for record in records},
            'installed_build_inputs': {record.relpath: record.build_inputs for record in records if record.build_inputs},
            'last_runtime_install_env': runtime_install_env or {},
        }

    update_state(repo_root, STATE_FILENAME, {'schema_version': 2, 'servers': {}}, apply_update)


def make_manifest(
    *,
    workspace_root: Path,
    workspace_id: str,
    snapshot_id: str,
    server_name: str,
    container_identity: str,
    runtime_root: str,
    container_cache_root: str,
    marker_dirname: str,
    root_preserve_paths: tuple[str, ...],
    records: list[SnapshotRecord],
    runtime_install_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    git_name, git_email = ensure_local_git_identity(workspace_root)
    return {
        'schema_version': 2,
        'generated_at': now_utc(),
        'workspace_root': str(workspace_root),
        'workspace_id': workspace_id,
        'snapshot_id': snapshot_id,
        'server_name': server_name,
        'container_identity': container_identity,
        'runtime_root': runtime_root,
        'container_cache_root': container_cache_root,
        'marker_dirname': marker_dirname,
        'root_preserve_paths': list(root_preserve_paths),
        'git_identity': {'name': git_name, 'email': git_email},
        'repos': [asdict(record) for record in records],
        'runtime_install_env': runtime_install_env or {},
        'local_source_of_truth': 'tracked + staged + unstaged + untracked-nonignored',
    }


def summary_payload(
    *,
    status: str,
    server_name: str,
    container_identity: str,
    workspace_id: str,
    container_cache_root: str | None,
    records: list[SnapshotRecord],
    reinstall_status: str,
    reason: str | None,
    first_install: bool,
    runtime_install_env: dict[str, str] | None = None,
    observed_runtime_commits: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        'status': status,
        'server_name': server_name,
        'container_identity': container_identity,
        'workspace_id': workspace_id,
        'container_cache_root': container_cache_root,
        'first_install': first_install,
        'snapshot_commits': {record.relpath: record.commit for record in records},
        'build_inputs': {record.relpath: record.build_inputs for record in records if record.build_inputs},
        'runtime_commits': observed_runtime_commits,
        'reinstall': reinstall_status,
        'runtime_install_env': runtime_install_env or {},
        'reason': reason,
    }


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


def final_manifest(manifest: dict[str, Any], *, status: str, reinstall_status: str, runtime_commits: dict[str, str] | None) -> dict[str, Any]:
    enriched = dict(manifest)
    enriched['completed_at'] = now_utc()
    enriched['status'] = status
    enriched['reinstall'] = reinstall_status
    enriched['runtime_commits'] = runtime_commits
    return enriched


def resolve_state_root(workspace_root: str | None, sources: dict[str, Path]) -> Path:
    if workspace_root:
        return Path(workspace_root).expanduser().resolve()
    if sources:
        return next(iter(sources.values())).resolve().parent
    raise ValueError('bind the actual vllm and vllm-ascend worktrees')


def run_plan(args: argparse.Namespace) -> int:
    sources = parse_sources(getattr(args, 'source', []))
    if not {'vllm', 'vllm-ascend'}.issubset(sources):
        raise ValueError('bind the actual vllm and vllm-ascend worktrees')
    workspace_root = resolve_state_root(getattr(args, 'workspace_root', None), sources)
    workspace_id = normalize_workspace_id(args.workspace_id)
    runtime_root = validate_absolute_posix_path(args.runtime_root, label='runtime root')
    container_cache_root = validate_absolute_posix_path(args.container_cache_root, label='container cache root')
    marker_dirname = validate_relative_posix_path(args.marker_dirname, label='marker dirname')
    root_preserve_paths = resolved_root_preserve_paths(marker_dirname, args.preserve_path)
    snapshot_id = args.snapshot_id or now_utc().replace(':', '').replace('-', '')
    records: list[SnapshotRecord] = []
    try:
        records = build_snapshot_records(workspace_root, workspace_id, snapshot_id, tuple(DEFAULT_DENYLIST), sources, records)
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
        print(json_dump(manifest))
        return 0
    finally:
        cleanup_synthetic_refs(workspace_root, records)


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
            record_map = {record.relpath: record for record in records}
            prior_runtime_state = load_runtime_state(workspace_root)
            last_container_state = (
                prior_runtime_state
                .get('servers', {})
                .get(args.server_name, {})
                .get('containers', {})
                .get(args.container_identity, {})
            )
            last_commits = last_container_state.get('last_snapshot_commits', {})
            installed_inputs = last_container_state.get('installed_build_inputs', {})

            def changes(relpath: str) -> tuple[bool, bool]:
                record = record_map.get(relpath)
                return changed_build_inputs(record.build_inputs, installed_inputs.get(relpath)) if record else (False, False)

            reinstall_vllm, vllm_dependency_changed = changes('vllm')
            reinstall_vllm_ascend, vllm_ascend_dependency_changed = changes('vllm-ascend')
            if reinstall_vllm and 'vllm-ascend' in record_map:
                reinstall_vllm_ascend = True

            if args.force_reinstall:
                if 'vllm' in record_map:
                    reinstall_vllm = True
                if 'vllm-ascend' in record_map:
                    reinstall_vllm_ascend = True
            install_vllm_ascend_deps = vllm_ascend_dependency_changed or vllm_dependency_changed

            snapshot_commits = {record.relpath: record.commit for record in records}

            if (
                args.apply_mode in ('auto', 'install') and not args.dry_run
                and not args.force_reinstall and not reinstall_vllm and not reinstall_vllm_ascend
                and snapshot_commits == last_commits and last_container_state.get('first_reinstall_completed')
            ):
                marker = read_runtime_install_marker(container=container, runtime_root=runtime_root, marker_dirname=marker_dirname)
                if not first_install_needed(marker, args.container_identity, runtime_root):
                    observed = verify_runtime_commits_map(container=container, runtime_root=runtime_root, expected=snapshot_commits)
                    if observed == snapshot_commits:
                        summary = summary_payload(status='ready', server_name=args.server_name, container_identity=args.container_identity, workspace_id=workspace_id, container_cache_root=container_cache_root, records=records, reinstall_status='not-needed', reason='snapshot and installed build inputs unchanged', first_install=False, runtime_install_env=last_container_state.get('last_runtime_install_env', {}), observed_runtime_commits=observed)
                        summary['fast_path'] = 'snapshot'
                        print(json_dump(summary))
                        keep_refs = True
                        return 0

            auto_selected_materialize = False
            if args.apply_mode == 'auto':
                needs_install = args.force_reinstall or reinstall_vllm or reinstall_vllm_ascend or install_vllm_ascend_deps
                if not needs_install and not args.dry_run:
                    current_phase = 'auto-apply-mode'
                    marker = read_runtime_install_marker(
                        container=container,
                        runtime_root=runtime_root,
                        marker_dirname=marker_dirname,
                    )
                    needs_install = first_install_needed(marker, args.container_identity, runtime_root)
                args.apply_mode = 'install' if needs_install else 'materialize'
                auto_selected_materialize = args.apply_mode == 'materialize'
                emit_progress(
                    'auto-apply-mode',
                    selected=args.apply_mode,
                    reinstall_vllm=reinstall_vllm,
                    reinstall_vllm_ascend=reinstall_vllm_ascend,
                    dependency_install=install_vllm_ascend_deps,
                )

            if args.apply_mode in {'source-only', 'materialize'}:
                runtime_install_env: dict[str, str] = {}
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
                    runtime_install_env=runtime_install_env,
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
                        runtime_install_env=runtime_install_env,
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
                    if args.apply_mode == 'materialize' and not auto_selected_materialize:
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
                        refs=snapshot_mirror_refs(workspace_id) if args.transport in {'auto', 'git'} else (),
                    )
                    transfer_reports: list[dict[str, Any]] = []
                    for record in records:
                        emit_progress('push-mirror', relpath=record.relpath, transport=args.transport)
                        transfer = push_snapshot_to_mirror(
                            repo=record_source(workspace_root, record),
                            container=container,
                            mirror_path=mirror_path_for(container_cache_root, workspace_id, record),
                            container_cache_root=container_cache_root,
                            record=record,
                            workspace_id=workspace_id,
                            dry_run=args.dry_run,
                            transport=args.transport,
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
                        expected_runtime_commits = {record.relpath: record.commit for record in records}
                        if observed_runtime_commits != expected_runtime_commits:
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
                                runtime_install_env=runtime_install_env,
                                observed_runtime_commits=observed_runtime_commits,
                            )
                            summary['apply_mode'] = args.apply_mode
                            summary['manifest_path'] = manifest_path
                            summary['transfers'] = transfer_reports
                            print(json_dump(summary))
                            return 1
                        status = 'materialized'
                        if auto_selected_materialize:
                            # Auto mode proved no native/dependency changes, so
                            # recording this snapshot keeps future runs on the
                            # fingerprint fast path without hiding install needs.
                            current_phase = 'update-local-state'
                            emit_progress(current_phase, server_name=args.server_name)
                            update_runtime_state(
                                repo_root=workspace_root,
                                server_name=args.server_name,
                                container_identity=args.container_identity,
                                runtime_root=runtime_root,
                                container_cache_root=container_cache_root,
                                marker_dirname=marker_dirname,
                                records=records,
                                first_reinstall_completed=last_container_state.get('first_reinstall_completed', False),
                                runtime_install_env=last_container_state.get('last_runtime_install_env', {}),
                            )

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
                        runtime_install_env=runtime_install_env,
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

            current_phase = 'read-runtime-marker'
            emit_progress(current_phase, runtime_root=runtime_root)
            marker = read_runtime_install_marker(
                container=container,
                runtime_root=runtime_root,
                marker_dirname=marker_dirname,
            )
            first_install = first_install_needed(marker, args.container_identity, runtime_root)
            if first_install:
                current_phase = 'check-consent'
                emit_progress(current_phase, container_identity=args.container_identity)
                consent = resolve_install_consent(workspace_root, args.server_name, args.container_identity)
                if consent != 'allow':
                    summary = summary_payload(
                        status='blocked',
                        server_name=args.server_name,
                        container_identity=args.container_identity,
                        workspace_id=workspace_id,
                        container_cache_root=container_cache_root,
                        records=records,
                        reinstall_status='blocked-by-consent',
                        reason='first-time runtime replacement requires explicit consent',
                        first_install=True,
                        observed_runtime_commits=None,
                    )
                    print(json_dump(summary))
                    return 2
                reinstall_vllm = True if 'vllm' in record_map else reinstall_vllm
                reinstall_vllm_ascend = True if 'vllm-ascend' in record_map else reinstall_vllm_ascend
                install_vllm_ascend_deps = True if 'vllm-ascend' in record_map else install_vllm_ascend_deps

            runtime_install_env: dict[str, str] = {}
            if not args.dry_run:
                current_phase = 'runtime-install-env'
                emit_progress(current_phase, runtime_root=runtime_root)
                runtime_install_env = read_runtime_install_env(
                    container=container,
                    runtime_root=runtime_root,
                    dry_run=args.dry_run,
                )

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
                runtime_install_env=runtime_install_env,
            )
            if args.print_manifest:
                print(json_dump(manifest))

            if args.dry_run:
                reinstall_status = 'would-perform' if (reinstall_vllm or reinstall_vllm_ascend) else 'not-needed'
                summary = summary_payload(
                    status='dry-run',
                    server_name=args.server_name,
                    container_identity=args.container_identity,
                    workspace_id=workspace_id,
                    container_cache_root=container_cache_root,
                    records=records,
                    reinstall_status=reinstall_status,
                    reason=None,
                    first_install=first_install,
                    runtime_install_env=runtime_install_env,
                    observed_runtime_commits=None,
                )
                print(json_dump(summary))
                keep_refs = True
                return 0

            lock_path = lock_path_for(container_cache_root, workspace_id, args.container_identity)
            current_phase = 'acquire-lock'
            emit_progress(current_phase, lock_path=lock_path)
            acquire_container_lock(container, lock_path, args.dry_run)
            try:
                current_phase = 'push-mirrors'
                emit_progress(current_phase, repo_count=len(records))
                all_mirror_paths = [mirror_path_for(container_cache_root, workspace_id, r) for r in records]
                mirror_refs = ensure_remote_bare_repos(
                    container, all_mirror_paths, args.dry_run,
                    refs=snapshot_mirror_refs(workspace_id) if args.transport in {'auto', 'git'} else (),
                )
                transfer_reports: list[dict[str, Any]] = []
                for record in records:
                    emit_progress('push-mirror', relpath=record.relpath, transport=args.transport)
                    transfer = push_snapshot_to_mirror(
                        repo=record_source(workspace_root, record),
                        container=container,
                        mirror_path=mirror_path_for(container_cache_root, workspace_id, record),
                        container_cache_root=container_cache_root,
                        record=record,
                        workspace_id=workspace_id,
                        dry_run=args.dry_run,
                        transport=args.transport,
                        remote_refs=mirror_refs.get(mirror_path_for(container_cache_root, workspace_id, record)),
                    )
                    transfer_reports.append(transfer)
                    emit_progress('push-mirror-complete', **transfer)
                manifest['transfers'] = transfer_reports

                current_phase = 'upload-manifest'
                emit_progress(current_phase, manifest_path=manifest_path)
                upload_manifest(container, manifest_path, manifest, args.dry_run)

                if first_install:
                    current_phase = 'first-install-prepare'
                    emit_progress(current_phase, runtime_root=runtime_root)
                    ssh_exec(container, prepare_isolated_root_script(runtime_root))

                current_phase = 'materialize-runtime'
                emit_progress(current_phase, runtime_root=runtime_root)
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

                reinstall_status = 'not-needed'
                if reinstall_vllm or reinstall_vllm_ascend:
                    reinstall_status = 'performed'
                    current_phase = 'runtime-install'
                    emit_progress(
                        current_phase,
                        reinstall_vllm=reinstall_vllm,
                        reinstall_vllm_ascend=reinstall_vllm_ascend,
                    )
                    if not first_install:
                        uninstall_pkgs: list[str] = []
                        if reinstall_vllm:
                            uninstall_pkgs.append('vllm')
                        if reinstall_vllm_ascend:
                            uninstall_pkgs.extend(['vllm-ascend', 'vllm_ascend'])
                        emit_progress('runtime-install-uninstall', packages=uninstall_pkgs)
                        run_runtime_install_step(
                            container=container,
                            runtime_root=runtime_root,
                            marker_dirname=marker_dirname,
                            container_identity=args.container_identity,
                            step='uninstall',
                            stream_progress=False,
                            uninstall_packages=tuple(uninstall_pkgs),
                        )
                    if reinstall_vllm:
                        emit_progress('runtime-install-vllm', package='vllm')
                        run_runtime_install_step(
                            container=container,
                            runtime_root=runtime_root,
                            marker_dirname=marker_dirname,
                            container_identity=args.container_identity,
                            step='install-vllm',
                            stream_progress=True,
                        )
                    if reinstall_vllm_ascend:
                        emit_progress('runtime-install-check-build-compat')
                        run_runtime_install_step(
                            container=container,
                            runtime_root=runtime_root,
                            marker_dirname=marker_dirname,
                            container_identity=args.container_identity,
                            step='check-build-compat',
                            stream_progress=True,
                        )
                        if install_vllm_ascend_deps:
                            emit_progress('runtime-install-vllm-ascend-requirements', requirements='requirements.txt')
                            run_runtime_install_step(
                                container=container,
                                runtime_root=runtime_root,
                                marker_dirname=marker_dirname,
                                container_identity=args.container_identity,
                                step='install-vllm-ascend-requirements',
                                stream_progress=True,
                            )
                        else:
                            emit_progress(
                                'runtime-install-vllm-ascend-requirements',
                                requirements='skipped-paired-image-deps',
                            )
                        emit_progress('runtime-install-vllm-ascend', package='vllm-ascend')
                        run_runtime_install_step(
                            container=container,
                            runtime_root=runtime_root,
                            marker_dirname=marker_dirname,
                            container_identity=args.container_identity,
                            step='install-vllm-ascend',
                            stream_progress=True,
                        )
                    emit_progress('runtime-install-verify-imports')
                    run_runtime_install_step(
                        container=container,
                        runtime_root=runtime_root,
                        marker_dirname=marker_dirname,
                        container_identity=args.container_identity,
                        step='verify-imports',
                        stream_progress=True,
                    )
                    emit_progress('runtime-install-verify-deps')
                    run_runtime_install_step(
                        container=container,
                        runtime_root=runtime_root,
                        marker_dirname=marker_dirname,
                        container_identity=args.container_identity,
                        step='verify-deps',
                        stream_progress=True,
                    )
                    emit_progress('runtime-install-marker')
                    run_runtime_install_step(
                        container=container,
                        runtime_root=runtime_root,
                        marker_dirname=marker_dirname,
                        container_identity=args.container_identity,
                        step='write-marker',
                        stream_progress=False,
                    )

                current_phase = 'verify-runtime-commits'
                emit_progress(current_phase, repo_count=len(records))
                observed_runtime_commits = verify_runtime_commits(
                    container=container,
                    runtime_root=runtime_root,
                    records=records,
                    dry_run=args.dry_run,
                )
                expected_runtime_commits = {record.relpath: record.commit for record in records}
                if observed_runtime_commits != expected_runtime_commits:
                    upload_manifest(
                        container,
                        manifest_path,
                        final_manifest(
                            manifest,
                            status='failed',
                            reinstall_status=reinstall_status,
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
                        reinstall_status=reinstall_status,
                        reason='runtime commit verification mismatch',
                        first_install=first_install,
                        runtime_install_env=runtime_install_env,
                        observed_runtime_commits=observed_runtime_commits,
                    )
                    summary['transfers'] = transfer_reports
                    print(json_dump(summary))
                    return 1

                current_phase = 'update-local-state'
                emit_progress(current_phase, server_name=args.server_name)
                update_runtime_state(
                    repo_root=workspace_root,
                    server_name=args.server_name,
                    container_identity=args.container_identity,
                    runtime_root=runtime_root,
                    container_cache_root=container_cache_root,
                    marker_dirname=marker_dirname,
                    records=records,
                    first_reinstall_completed=first_install
                    or last_container_state.get('first_reinstall_completed', False)
                    or reinstall_status == 'performed',
                    runtime_install_env=runtime_install_env,
                )
                current_phase = 'finalize-manifest'
                emit_progress(current_phase, manifest_path=manifest_path)
                upload_manifest(
                    container,
                    manifest_path,
                    final_manifest(
                        manifest,
                        status='ready',
                        reinstall_status=reinstall_status,
                        runtime_commits=observed_runtime_commits,
                    ),
                    False,
                )
                emit_progress('complete', status='ready')
                summary = summary_payload(
                    status='ready',
                    server_name=args.server_name,
                    container_identity=args.container_identity,
                    workspace_id=workspace_id,
                    container_cache_root=container_cache_root,
                    records=records,
                    reinstall_status=reinstall_status,
                    reason=None,
                    first_install=first_install,
                    runtime_install_env=runtime_install_env,
                    observed_runtime_commits=observed_runtime_commits,
                )
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


def run_gc(args: argparse.Namespace) -> int:
    lib = Path(__file__).resolve().parents[3] / 'lib'
    if str(lib) not in sys.path:
        sys.path.insert(0, str(lib))
    from mindie_coordinator.code_identity import gc_parity_refs

    result = gc_parity_refs(
        Path(args.workspace_root),
        max_age_days=args.max_age_days,
        now=args.now,
    )
    print(json_dump(result))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Prepare or enforce remote code parity for a ready runtime.', allow_abbrev=False)
    subparsers = parser.add_subparsers(dest='command', required=True)

    def add_shared_arguments(target: argparse.ArgumentParser) -> None:
        target.add_argument('--workspace-root', default='', help='Optional local state directory; bound --source worktrees are snapshotted directly.')
        target.add_argument('--source', action='append', default=[], help='Use an actual external business worktree: vllm=/path or vllm-ascend=/path.')
        target.add_argument('--workspace-id', required=True, help='Stable workspace id used for container cache namespacing.')
        target.add_argument('--server-name', required=True)
        target.add_argument('--runtime-root', required=True)
        target.add_argument('--container-identity', required=True)
        target.add_argument('--container-cache-root', default=DEFAULT_CONTAINER_CACHE_ROOT)
        target.add_argument('--marker-dirname', default=DEFAULT_MARKER_DIRNAME)
        target.add_argument('--preserve-path', action='append', default=[])

    plan = subparsers.add_parser('plan', help='Build a synthetic snapshot manifest without remote mutations.')
    add_shared_arguments(plan)
    plan.add_argument('--snapshot-id', default=None)

    sync = subparsers.add_parser('sync', help='Publish container-local mirrors, materialize runtime state, and reinstall when required.')
    add_shared_arguments(sync)
    sync.add_argument('--snapshot-id', default=None)
    sync.add_argument('--snapshot-file', help='Use a previously captured execution source descriptor without rereading worktrees.')
    sync.add_argument('--container-host', required=True)
    sync.add_argument('--container-port', type=int, required=True)
    sync.add_argument('--container-user', required=True)
    sync.add_argument('--force-reinstall', action='store_true', help='Force reinstall of vllm and vllm-ascend regardless of what changed.')
    sync.add_argument('--dry-run', action='store_true')
    sync.add_argument('--print-manifest', action='store_true')
    sync.add_argument(
        '--transport',
        choices=TRANSFER_MODES,
        default='auto',
        help='auto prefers incremental Git push and falls back to the full-bundle transport.',
    )
    sync.add_argument(
        '--apply-mode',
        choices=('auto', 'source-only', 'materialize', 'install'),
        default='auto',
        help='auto picks materialize for pure-Python changes and install only when native/dependency files changed (or first install); source-only publishes snapshots only; materialize updates runtime sources without install/rebuild; install forces the full parity behavior.',
    )

    gc = subparsers.add_parser(
        'gc',
        help='Delete unused refs/parity refs older than seven days.',
    )
    gc.add_argument('--workspace-root', required=True, help='Local workspace root.')
    gc.add_argument('--max-age-days', type=int, default=7)
    gc.add_argument(
        '--now',
        type=float,
        default=None,
        help='Unix timestamp used as now; tests pin this so age is deterministic.',
    )

    return parser


def main() -> int:
    from mindie_coordinator._stdio import configure_stdio
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == 'plan':
            return run_plan(args)
        if args.command == 'sync':
            return run_sync(args)
        if args.command == 'gc':
            return run_gc(args)
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
