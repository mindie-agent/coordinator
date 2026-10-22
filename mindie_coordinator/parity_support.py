"""Git, SSH transport, and remote fixed-source materialization helpers.

[Role]
- Provides subprocess Git execution (`git`), SSH command/stream wrappers
  (`ssh_exec`, `ssh_exec_stream`, `ssh_stream_to_file`),
  and the self-contained remote materialization payload (`_materialize_fixed`).

[Invariants]
- Remote source mirrors (`*.git`) store immutable commit refs under
  `refs/mindie/snapshots/<oid>`; checkouts use `--local` clone hardlinks and
  detached HEAD without mutating donor mirrors.
- `RemoteCommandError` represents a completed remote exit status, distinct from
  transport timeouts (`returncode == 255` or `TimeoutError`).

[Timeouts & Locks]
- Stream uploads enforce explicit deadlines (`120s`) and
  raise `TimeoutError` without blind replay on timeout.
- Remote mirror init and target worktree checkouts use OS-released `flock` files.

[Side-Effect Policy]
- Never silently retry timed-out remote stream writes.
- Shared object publication failures degrade gracefully to private mirrors
  without blocking materialization, whereas subprocess timeouts propagate.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from mindie_coordinator.shared_source_objects import copy_fixed_objects

WORKSPACE_ID_PATTERN = re.compile(r'[^A-Za-z0-9._-]+')
DEFAULT_DENYLIST = (
    '.mindie-local/',
    '.mindie-runtime/',
    '.remote-code-parity/',
    '.workspace.local/',
    '.machine-inventory.json',
    '.codex/',
    '.claude/settings.local.json',
    '.env',
    '.env.*',
    '.venv/',
    'venv/',
    '__pycache__/',
    '.pytest_cache/',
    '.mypy_cache/',
    '.ruff_cache/',
    '*.log',
    '*.out',
    '.DS_Store',
    '._*',
    'Thumbs.db',
)

PROGRESS_SENTINEL = '__MINDIE_PARITY_PROGRESS__='


@dataclass(frozen=True)
class SshEndpoint:
    host: str
    port: int
    user: str
    container: str | None = None

    def destination(self) -> str:
        return f'{self.user}@{self.host}'


@dataclass(frozen=True)
class SshStreamingResult:
    returncode: int
    stdout: str
    stderr: str
    progress_events: list[dict[str, Any]]


class LocalCommandError(RuntimeError):
    """A local child process completed with a nonzero exit code."""


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture_output: bool = True,
    timeout: float | None = None,
    stdin: int | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            check=False,
            capture_output=capture_output,
            text=True, encoding="utf-8",
            timeout=timeout,
            stdin=stdin,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"command timed out after {timeout:.0f}s: "
            f"{' '.join(shlex.quote(part) for part in cmd)}\n"
            f'stdout:\n{exc.stdout or ""}\n'
            f'stderr:\n{exc.stderr or ""}'
        ) from exc
    if check and result.returncode != 0:
        raise LocalCommandError(
            f"command failed ({result.returncode}): {' '.join(shlex.quote(part) for part in cmd)}\n"
            f'stdout:\n{result.stdout}\n'
            f'stderr:\n{result.stderr}'
        )
    return result


def git(
    repo: Path,
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    options = ['-c', 'core.longpaths=true'] if os.name == 'nt' else []
    return run(['git', *options, '-C', str(repo), *args], env=env, check=check, timeout=timeout,
               stdin=subprocess.DEVNULL)



def now_utc() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def sanitize_repo_id(relpath: str) -> str:
    return 'workspace' if relpath in ('', '.') else relpath.replace('/', '__')


def json_dump(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def quoted(script: str) -> str:
    return shlex.quote(script)


def parse_progress_event(line: str) -> dict[str, Any] | None:
    if not line.startswith(PROGRESS_SENTINEL):
        return None
    try:
        payload = json.loads(line[len(PROGRESS_SENTINEL) :])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _remote_endpoint(endpoint: SshEndpoint, *, long_stream: bool = False):
    from remote_dev.core.endpoint import Endpoint

    if long_stream:
        return Endpoint.for_long_stream(host=endpoint.host, port=endpoint.port, user=endpoint.user,
                                         container=endpoint.container)
    return Endpoint(host=endpoint.host, port=endpoint.port, user=endpoint.user,
                    container=endpoint.container)


class RemoteCommandError(RuntimeError):
    """A completed SSH command status, distinct from a lost transport."""

    def __init__(self, returncode: int, message: str):
        super().__init__(message)
        self.returncode = returncode


def _ssh_failure(returncode: int, stdout: str, stderr: str, *, what: str) -> RemoteCommandError:
    return RemoteCommandError(returncode,
        f'command failed ({returncode}): {what}\n'
        f'stdout:\n{stdout}\n'
        f'stderr:\n{stderr}'
    )


def ssh_exec(
    endpoint: SshEndpoint,
    script: str,
    *,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    from remote_dev.core.ssh_transport import run_script

    del capture_output
    completed = run_script(_remote_endpoint(endpoint), script)
    stdout = completed.stdout or ''
    stderr = completed.stderr or ''
    if completed.timed_out:
        returncode = 255
        stderr = stderr or 'remote command timed out'
    else:
        returncode = 0 if completed.returncode is None else int(completed.returncode)
    result = subprocess.CompletedProcess(
        ['ssh', endpoint.destination(), str(endpoint.port)],
        returncode,
        stdout,
        stderr,
    )
    if check and result.returncode != 0:
        raise _ssh_failure(result.returncode, stdout, stderr, what=f'ssh_exec {endpoint.destination()}')
    return result


def ssh_exec_stream(
    endpoint: SshEndpoint,
    script: str,
    *,
    check: bool = True,
    stream_progress: bool = True,
    on_progress=None,
    log_path=None,
    process=None,
) -> SshStreamingResult:
    from remote_dev.core.ssh_transport import run_stream
    from mindie_coordinator.preparation_script import preparation_command

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    progress_events: list[dict[str, Any]] = []

    def on_output(channel: str, text: str) -> None:
        if log_path is not None:
            with Path(log_path).open('a', encoding='utf-8') as log:
                log.write(text)
        if channel == 'stdout':
            stdout_parts.append(text)
            return
        event = parse_progress_event(text)
        if event is not None:
            progress_events.append(event)
            if on_progress is not None:
                on_progress(event)
            if stream_progress:
                sys.stderr.write(text if text.endswith('\n') else text + '\n')
                sys.stderr.flush()
            return
        stderr_parts.append(text)

    completed = (process.run(script, on_output=on_output) if process is not None else run_stream(
        _remote_endpoint(endpoint, long_stream=True), preparation_command(script),
        merge_stderr=False, on_output=on_output,
    ))
    stdout = ''.join(stdout_parts)
    stderr = ''.join(stderr_parts)
    if completed.timed_out:
        returncode = 255
        if completed.stderr and completed.stderr not in stderr:
            stderr = f'{stderr}{completed.stderr}' if stderr else completed.stderr
    else:
        returncode = 0 if completed.returncode is None else int(completed.returncode)
    if check and returncode != 0:
        raise _ssh_failure(returncode, stdout, stderr, what=f'ssh_exec_stream {endpoint.destination()}')
    return SshStreamingResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        progress_events=progress_events,
    )


def ssh_stream_to_file(endpoint: SshEndpoint, remote_path: str, payload: str) -> None:
    from remote_dev.core.ssh_transport import run_bytes

    script = f'mkdir -p {quoted(str(PurePosixPath(remote_path).parent))} && cat > {quoted(remote_path)}'
    try:
        result = run_bytes(_remote_endpoint(endpoint, long_stream=True), script,
                           stdin=payload.encode('utf-8'), timeout_ms=120000)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f'SSH upload to {remote_path} timed out; remote outcome is unknown; payload was not replayed') from exc
    stdout = (result.stdout or b'').decode('utf-8', errors='replace')
    stderr = (result.stderr or b'').decode('utf-8', errors='replace')
    if result.returncode != 0:
        raise RuntimeError(
            f'failed to stream payload to {remote_path}\nstdout:\n{stdout}\nstderr:\n{stderr}'
        )


def _materialize_fixed(request):
    """Self-contained remote program for a coordinator-owned fixed source root.

    Mirrors hold immutable commit refs, so concurrent roots need no lock across
    transfer requests. Only initialization and one root's checkout are locked;
    both locks are released by the OS when this process dies.
    """
    import base64
    import contextlib
    import fcntl
    import hashlib
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import tempfile

    def git(path, *args, check=True):
        result = subprocess.run(['git', '-C', str(path), *args], capture_output=True,
                                text=True, encoding='utf-8', timeout=120, stdin=subprocess.DEVNULL)
        if check and result.returncode:
            raise RuntimeError(f'git {args[0]} failed in {path}: {result.stderr.strip()}')
        return result

    @contextlib.contextmanager
    def lock(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    def atomic_json(path, value):
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as output:
                json.dump(value, output, sort_keys=True)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    root = Path(request['root'])
    rows = request['records']
    missing = []
    for index, row in enumerate(rows):
        mirror = Path(row['mirror'])
        with lock(mirror.with_suffix('.init.lock')):
            if not mirror.exists():
                git(mirror.parent, 'init', '--bare', str(mirror))
            if git(mirror, 'rev-parse', '--is-bare-repository').stdout.strip() != 'true':
                raise ValueError(f'expected a bare mirror: {mirror}')
            pack = request.get('inline_packs', {}).get(str(index))
            if pack is not None:
                data = base64.b64decode(pack['data'], validate=True)
                if (len(data) > 65536 or len(data) != pack['bytes']
                        or hashlib.sha256(data).hexdigest() != pack['sha256']):
                    raise ValueError('fixed source pack size or digest differs')
                if git(mirror, 'cat-file', '-e', pack['previous'] + '^{commit}', check=False).returncode:
                    raise ValueError('fixed source pack prerequisite disappeared')
                received = subprocess.run(['git', '-C', str(mirror), 'index-pack', '--stdin'],
                                          input=data, capture_output=True, timeout=60)
                if received.returncode:
                    raise ValueError('fixed source pack rejected: ' + received.stderr.decode(errors='replace'))
                if git(mirror, 'rev-parse', row['commit'] + '^{tree}').stdout.strip() != row['tree']:
                    raise ValueError('fixed source pack has the wrong tree')
                git(mirror, 'cat-file', '-e', pack['carrier'] + '^{commit}')
                # Match atomic Git push semantics. Every accepted snapshot has
                # its own immutable pin even if concurrent carriers advance.
                refs = ('start\nupdate refs/mindie/snapshots/' + row['commit'] + ' ' + row['commit']
                        + '\nupdate ' + request['carrier_ref'] + ' ' + pack['carrier'] + '\nprepare\ncommit\n')
                updated = subprocess.run(['git', '-C', str(mirror), 'update-ref', '--stdin'],
                                         input=refs, text=True, capture_output=True, timeout=30)
                if updated.returncode:
                    raise ValueError('fixed source refs rejected: ' + updated.stderr)
        tree = git(mirror, 'rev-parse', '--verify', row['commit'] + '^{tree}', check=False)
        shared = row.get('shared_mirror')
        if tree.returncode and shared:
            try:
                if copy_fixed_objects(shared, mirror, row):
                    tree = git(mirror, 'rev-parse', '--verify', row['commit'] + '^{tree}')
            except TimeoutError:
                raise
            except (OSError, ValueError, RuntimeError) as exc:
                # These are completed local errors, not subprocess timeouts or
                # cancellation. The exact private input is still missing.
                print(f'shared source candidate unavailable: {exc}', file=sys.stderr, flush=True)
        if tree.returncode:
            carrier = git(mirror, 'rev-parse', '--verify', request['carrier_ref'], check=False)
            local_bases = row.get('local_bases', [])
            if (shared and local_bases and
                    (carrier.returncode or carrier.stdout.strip() not in local_bases)):
                try:
                    # Only a snapshot available on both sides is useful for
                    # bounded delta packing. Cache recency cannot prove local
                    # membership, especially for independent tree snapshots.
                    hints = (git(shared, 'for-each-ref', '--format=%(objectname) %(refname)',
                                 *['refs/mindie/snapshots/' + oid for oid in local_bases])
                            if Path(shared).is_dir() else None)
                    available = set()
                    for line in hints.stdout.splitlines() if hints else []:
                        parts = line.split()
                        if len(parts) == 2 and parts[1] == 'refs/mindie/snapshots/' + parts[0]:
                            available.add(parts[0])
                    for oid in local_bases:
                        if oid not in available:
                            continue
                        try:
                            base = {'commit': oid, 'tree': git(shared, 'rev-parse', oid + '^{tree}').stdout.strip()}
                            if copy_fixed_objects(shared, mirror, base):
                                previous = carrier.stdout.strip() if not carrier.returncode else '0' * len(oid)
                                git(mirror, 'update-ref', request['carrier_ref'], oid, previous, check=False)
                                carrier = git(mirror, 'rev-parse', '--verify', request['carrier_ref'], check=False)
                                break
                        except TimeoutError:
                            raise
                        except (OSError, ValueError, RuntimeError) as exc:
                            print(f'shared source negotiation base unavailable: {exc}', file=sys.stderr, flush=True)
                except TimeoutError:
                    raise
                except (OSError, ValueError, RuntimeError) as exc:
                    print(f'shared source negotiation base unavailable: {exc}', file=sys.stderr, flush=True)
            missing.append({'index': index, 'carrier': carrier.stdout.strip() if not carrier.returncode else None})
            continue
        if tree.stdout.strip() != row['tree']:
            raise ValueError(f'fixed commit tree mismatch: {row["relpath"]}')
        # Pin before materialization; another execution may move legacy refs.
        fixed_ref = 'refs/mindie/snapshots/' + row['commit']
        pinned = git(mirror, 'rev-parse', '--verify', fixed_ref, check=False)
        if pinned.returncode or pinned.stdout.strip() != row['commit']:
            git(mirror, 'update-ref', fixed_ref, row['commit'])
        if shared:
            # Publish immutable objects only. The shared mirror never replaces
            # this owner's refs/objects or becomes a live alternate dependency.
            try:
                copy_fixed_objects(mirror, shared, row)
            except TimeoutError:
                raise
            except (OSError, ValueError, RuntimeError) as exc:
                # Cache publication cannot gate a valid private snapshot.
                # Its checkout and complete source validation still run below.
                print(f'shared source publication unavailable: {exc}', file=sys.stderr, flush=True)
            # A shared hit also provides this owner's first negotiation base.
            # Create only; concurrent work may already have advanced its ref.
            # Neither this mutable hint nor any shared ref selects a checkout.
            git(mirror, 'update-ref', request['carrier_ref'], row['commit'],
                '0' * len(row['commit']), check=False)
    if missing:
        return {'status': 'missing', 'missing': missing}

    # The lock inode is outside anything checkout/clean can remove. It remains
    # stable across retries and across workspace names targeting the same root.
    token = hashlib.sha256(str(root).encode()).hexdigest()
    with lock(root.parent / '.mindie-source-locks' / (token + '.lock')):
        root.mkdir(parents=True, exist_ok=True)
        marker = root / '.mindie-runtime'
        marker.mkdir(exist_ok=True)
        owner = marker / 'fixed-source-owner.json'
        expected_owner = {'source_id': request['source_id'], 'root': str(root)}
        atomic_json(owner, expected_owner)
        receipt = marker / 'source-materialization.json'
        # A failed retry cannot leave an earlier success receipt for this run.
        receipt.unlink(missing_ok=True)
        for row in sorted(rows, key=lambda item: (len(Path(item['relpath']).parts), item['relpath'])):
            repo = root / row['relpath']
            repo.parent.mkdir(parents=True, exist_ok=True)
            if not (repo / '.git').exists():
                # Local clones hardlink immutable Git objects; fetching a whole
                # repository into every new root would repack those same bytes.
                git(repo.parent, 'clone', '--local', '--no-checkout', row['mirror'], str(repo))
            git(repo, 'config', 'core.autocrlf', 'false')
            if git(repo, 'cat-file', '-e', row['commit'] + '^{tree}', check=False).returncode:
                git(repo, 'fetch', '--no-tags', '--no-recurse-submodules', row['mirror'], row['commit'])
            git(repo, 'checkout', '--detach', '--force', row['commit'])
            git(repo, 'clean', '-ffd')
            for child in row['submodules']:
                child_path = str(Path(row['relpath']) / child['path'])
                child_row = next(item for item in rows if item['relpath'] == child_path)
                git(repo, 'config', 'submodule.' + child['name'] + '.url', child_row['mirror'])
        observed = {row['relpath']: row['commit'] for row in rows}
        result = {'status': 'materialized', **expected_owner, 'commits': observed}
        atomic_json(receipt, result)
        return result


def is_git_worktree(path: Path) -> bool:
    result = git(path, ['rev-parse', '--is-inside-work-tree'], check=False)
    if result.returncode != 0 or result.stdout.strip() != 'true':
        return False
    top = git(path, ['rev-parse', '--show-toplevel'], check=False)
    if top.returncode != 0:
        return False
    try:
        return Path(top.stdout.strip()).resolve() == path.resolve()
    except FileNotFoundError:
        return False


def ensure_local_git_identity(repo: Path) -> tuple[str | None, str | None]:
    # Despite the name this is read-only: it never sets git config, it only
    # reports the identity (if any) that snapshot commits would record.
    name = git(repo, ['config', '--get', 'user.name'], check=False).stdout.strip() or None
    email = git(repo, ['config', '--get', 'user.email'], check=False).stdout.strip() or None
    return name, email


def glob_match_any(path: str, patterns: Iterable[str]) -> bool:
    import fnmatch

    normalized = path.replace('\\', '/')
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in patterns)
