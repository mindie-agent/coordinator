"""Host-local, immutable Git objects; execution refs and checkouts stay private."""

SHARED_SOURCE_CACHE = '/tmp/mindie-source-objects/v1'


def copy_fixed_objects(source, destination, row):
    """Copy one exact object closure, without alternates or donor ref changes.

    Git fetch checks received objects and connectivity before publishing the
    immutable ref. A missing source is a cache miss; a failed transfer is not.
    Locks protect only the receiving mirror and remain stable across retries.
    This function is also embedded in package-owned remote programs.
    """
    import fcntl
    from pathlib import Path
    import subprocess

    def git(path, *args, check=True):
        result = subprocess.run(['git', '-C', str(path), *args], capture_output=True,
                                text=True, timeout=60)
        if result.returncode < 0:
            raise subprocess.SubprocessError('fixed object copy interrupted: ' + result.stderr.strip())
        if check and result.returncode:
            raise RuntimeError('fixed object copy failed: ' + result.stderr.strip())
        return result

    source, destination = Path(source), Path(destination)
    if source.resolve() != source or destination.resolve() != destination:
        raise ValueError('fixed object mirror must not traverse symlinks')
    if not source.exists():
        return False
    tree = git(source, 'rev-parse', '--verify', row['commit'] + '^{tree}', check=False)
    if tree.returncode:
        return False
    if tree.stdout.strip() != row['tree']:
        raise ValueError('fixed object source has the wrong tree')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination.with_suffix('.init.lock'), 'a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if not destination.exists():
            git(destination.parent, 'init', '--bare', str(destination))
        if git(destination, 'rev-parse', '--is-bare-repository').stdout.strip() != 'true':
            raise ValueError('fixed object destination is not a bare mirror')
        fixed_ref = 'refs/mindie/snapshots/' + row['commit']
        pinned = git(destination, 'rev-parse', '--verify', fixed_ref, check=False)
        if not pinned.returncode and pinned.stdout.strip() == row['commit']:
            return True
        git(destination, '-c', 'fetch.fsckObjects=true', '-c', 'transfer.fsckObjects=true',
            '-c', 'gc.auto=0', 'fetch', '--no-tags', '--no-recurse-submodules',
            '--no-write-fetch-head', str(source), row['commit'] + ':' + fixed_ref)
        if git(destination, 'rev-parse', row['commit'] + '^{tree}').stdout.strip() != row['tree']:
            raise ValueError('fixed object destination has the wrong tree')
    return True


def export_existing_objects(request, run=None):
    """On a cold miss, read requested objects from existing managed containers.

    Docker IDs and MindIE labels bound discovery. Only the coordinator mirror
    location and requested repository names are inspected, never worktrees or
    arbitrary user directories. The container program writes the shared cache
    only; all donor paths and refs remain untouched.
    """
    import json
    import re
    import subprocess
    import sys
    if run is None:
        run = subprocess.run
    try:
        listed = run(['docker', 'ps', '--filter', 'label=com.mindie.managed=true', '--format', '{{.ID}}'],
                     capture_output=True, text=True, check=True, timeout=15)
        identifiers = listed.stdout.split()
        if not identifiers:
            return {'status': 'miss', 'copied': []}
        containers = json.loads(run(['docker', 'inspect', *identifiers], capture_output=True,
                                    text=True, check=True, timeout=15).stdout)
    except subprocess.TimeoutExpired as exc:
        return {'status': 'uncertain', 'reason': str(exc)}
    except subprocess.CalledProcessError as exc:
        if exc.returncode < 0 or exc.returncode >= 128:
            return {'status': 'uncertain', 'reason': str(exc)}
        return {'status': 'miss', 'copied': [], 'reason': str(exc)}
    except (OSError, ValueError) as exc:
        return {'status': 'miss', 'copied': [], 'reason': str(exc)}
    pending = list(request['records'])
    copied = []
    diagnostics = []
    for info in containers:
        identifier = info.get('Id', '')
        labels = (info.get('Config') or {}).get('Labels') or {}
        if (not re.fullmatch('[0-9a-f]{64}', identifier)
                or not re.fullmatch('sha256:[0-9a-f]{64}', info.get('Image', ''))
                or labels.get('com.mindie.managed') != 'true'
                or not (info.get('State') or {}).get('Running')
                or not re.fullmatch('/mindie-[A-Za-z0-9_.-]+', info.get('Name', ''))):
            continue
        if not any(m.get('Type') == 'bind' and m.get('Source') == '/tmp'
                   and m.get('Destination') == '/tmp' and m.get('RW') for m in info.get('Mounts', [])):
            continue
        try:
            reply = run(['docker', 'exec', '-i', identifier, 'python3', '-c', request['export_source']],
                        input=json.dumps({'records': pending, 'legacy_cache': request['legacy_cache']}),
                        capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired as exc:
            return {'status': 'uncertain', 'reason': str(exc)}
        except OSError as exc:
            print(f'fixed object donor unavailable: {exc}', file=sys.stderr, flush=True)
            diagnostics.append(str(exc))
            continue
        if reply.returncode < 0 or reply.returncode >= 128:
            return {'status': 'uncertain', 'reason': 'donor export interrupted: ' + reply.stderr[-2000:]}
        if reply.returncode:
            print('fixed object donor unavailable: ' + reply.stderr[-2000:], file=sys.stderr, flush=True)
            diagnostics.append(reply.stderr[-2000:])
            continue
        try:
            result = json.loads(reply.stdout)
            if not isinstance(result, dict):
                raise ValueError('fixed object donor reply is not an object')
            if result.get('status') in {'uncertain', 'cancelled'} or result.get('remote_outcome') == 'unknown':
                return result
            found = result.get('copied')
            if (not isinstance(found, list) or any(not isinstance(commit, str) or
                    commit not in {r['commit'] for r in pending} for commit in found)):
                raise ValueError('fixed object donor returned unexpected commits')
            reported = result.get('diagnostics', [])
            if not isinstance(reported, list) or any(not isinstance(value, str) for value in reported):
                raise ValueError('fixed object donor diagnostics are invalid')
        except (ValueError, TypeError) as exc:
            # This donor command has exited. An unusable optional cache reply
            # must not discard other donors or block the normal Git upload.
            message = 'fixed object donor reply unavailable: ' + str(exc)
            print(message, file=sys.stderr, flush=True)
            diagnostics.append(message[-2000:])
            continue
        diagnostics.extend(reported)
        copied.extend(found)
        pending = [row for row in pending if row['commit'] not in found]
        if not pending:
            break
    return {'status': 'copied' if copied else 'miss', 'copied': copied,
            **({'diagnostics': diagnostics[-3:]} if diagnostics else {})}


def export_container_objects(request):
    """Read only requested legacy mirror names; copy exact closures to /tmp."""
    from pathlib import Path
    import subprocess
    import sys
    copied = []
    diagnostics = []
    base = Path(request['legacy_cache']) / 'workspaces'
    for row in request['records']:
        # Workspace identities differ by user, but only this requested repo
        # name is considered. No source file or mutable ref is inspected.
        for mirror in base.glob('*/mirrors/nested/' + row['repo_id'] + '.git'):
            try:
                if copy_fixed_objects(mirror, row['shared_mirror'], row):
                    copied.append(row['commit'])
                    break
            except subprocess.SubprocessError as exc:
                return {'status': 'uncertain', 'reason': str(exc)}
            except (OSError, ValueError, RuntimeError) as exc:
                print(f'fixed object candidate unavailable: {exc}', file=sys.stderr, flush=True)
                diagnostics.append(str(exc)[-2000:])
    return {'copied': copied, **({'diagnostics': diagnostics[-3:]} if diagnostics else {})}
