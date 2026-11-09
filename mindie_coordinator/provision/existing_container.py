"""Bounded read-only reuse of an already configured fixed-image container.

[Role] Observes one existing container and decides whether it can be reused
without provisioning another.

[Invariants] Observation is read-only. Determinate "not reusable" facts return
`unusable` so the provision owner may create a container. Observation errors,
timeouts, docker daemon/permission failures, and an SSH listener that exists
but is not this container's sshd return `failed` and must abort; they are
never `unknown`. `docker inspect` is `unusable` only when the tool reports the
named container absent. Observation SSH forwards `identity_file`/`auth_ref`.

[Timeouts & Locks] The whole inspection is bounded to 8 seconds and each
subprocess receives `min(3, remaining)` seconds.

[Side-Effect Policy] Runs read-only docker inspection commands only.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import time

_MISSING_CONTAINER_MARKERS = ('no such object', 'no such container')


def _inspect_reports_absent(exc: subprocess.CalledProcessError) -> bool:
    text = f"{exc.stderr or ''} {exc.stdout or ''}".lower()
    return any(marker in text for marker in _MISSING_CONTAINER_MARKERS)


def inspect_existing(request, run=subprocess.run):
    """One host observation; unusable facts may fall through, failures abort."""
    deadline = time.monotonic() + 8

    def output(argv):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('existing container observation timed out')
        return run(argv, capture_output=True, text=True, check=True,
                   timeout=min(3, remaining)).stdout

    try:
        container, port = request['container'], int(request['ssh_port'])
        rows = json.loads(output(['docker', 'inspect', container]))
        if len(rows) != 1:
            return {'status': 'unusable', 'reason': 'container identity unavailable'}
        info = rows[0]
        identifier = info.get('Id', '')
        if info.get('Name') != '/' + container or not re.fullmatch('[0-9a-f]{64}', identifier):
            return {'status': 'unusable', 'reason': 'container identity unavailable'}
        config = info.get('Config') or {}
        image = config.get('Image')
        image_id = info.get('Image', '')
        if not image or not re.fullmatch('sha256:[0-9a-f]{64}', image_id):
            return {'status': 'unusable', 'reason': 'container image unavailable'}
        # Same explicit-reference comparison as bootstrap with its current
        # use_prepared_image_cache=false. A base-image label alone is not proof.
        if image not in request['image_request']['candidates']:
            return {'status': 'mismatch', 'container_id': identifier, 'image': image,
                    'image_id': image_id,
                    'reason': 'existing container image does not match the requested selector'}
        state, labels = info.get('State') or {}, config.get('Labels') or {}
        if (state.get('Running') is not True or state.get('Paused') or state.get('Restarting')
                or int(state.get('Pid') or 0) <= 0):
            return {'status': 'unusable', 'reason': 'container is not running normally'}
        if (labels.get('com.mindie.managed') != 'true'
                or labels.get('com.mindie.namespace') != request['user']
                or labels.get('com.mindie.workdir') != request['workdir']
                or labels.get('com.mindie.container_ssh_port') != str(port)
                or (info.get('HostConfig') or {}).get('NetworkMode') != 'host'):
            return {'status': 'unusable', 'reason': 'managed container or SSH port facts are incomplete'}
        expected_machine = (request.get('image_request') or {}).get('machine_type')
        actual_machine = (labels.get('com.mindie.machine_type') or '').strip()
        if expected_machine and actual_machine and actual_machine.upper() != str(expected_machine).strip().upper():
            return {'status': 'mismatch', 'container_id': identifier, 'image': image,
                    'image_id': image_id,
                    'reason': f'existing container machine type {actual_machine} does not match requested {expected_machine}'}
        # Host networking needs evidence that this container's sshd owns the
        # configured listening socket, not just a label or another user's port.
        processes = output(['docker', 'top', identifier, '-eo', 'pid']).splitlines()[1:]
        owned = {int(line.strip()) for line in processes if line.strip().isdigit()}
        listeners = output(['ss', '-ltnpH', 'sport = :' + str(port)]).splitlines()
        if not listeners:
            return {'status': 'unusable', 'reason': 'configured SSH port is not listening'}
        listener_pids = set()
        for line in listeners:
            pids = {int(value) for value in re.findall(r'pid=(\d+)', line)}
            sshd = {int(value) for value in re.findall(r'\("sshd[^\"]*",pid=(\d+)', line)}
            if not pids or pids != sshd or not pids.issubset(owned):
                return {'status': 'failed', 'reason': 'SSH listener ownership is unconfirmed'}
            listener_pids.update(pids)
        return {'status': 'match', 'container_id': identifier, 'image': image,
                'image_id': image_id, 'base_image': labels.get('com.mindie.base_image'),
                'base_image_id': labels.get('com.mindie.base_image_id'), 'ssh_port': port,
                'listener_pids': sorted(listener_pids), 'observed_at': time.time()}
    except TimeoutError:
        raise
    except subprocess.TimeoutExpired as exc:
        return {'status': 'failed', 'reason': f'existing container observation timed out: {exc}'[:300]}
    except FileNotFoundError as exc:
        return {'status': 'failed', 'reason': f'host observation command missing: {exc}'[:300]}
    except subprocess.CalledProcessError as exc:
        argv = list(exc.cmd) if isinstance(exc.cmd, (list, tuple)) else [str(exc.cmd)]
        # Missing container is a determinate non-reuse; docker daemon,
        # permission, and timeout-as-nonzero failures abort as failed.
        if argv[:2] == ['docker', 'inspect'] and _inspect_reports_absent(exc):
            return {'status': 'unusable', 'reason': str(exc)[:300]}
        return {'status': 'failed', 'reason': str(exc)[:300]}
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        return {'status': 'failed', 'reason': str(exc)[:300]}


def observe_existing(target, *, container, user, ssh_port, workdir, image_request,
                     identity_file=None, auth_ref=None):
    """Use the existing cached Python transport, with no host mutation or probe."""
    from remote_dev.core.endpoint import resolve_endpoint
    from remote_dev.core.ssh_transport import run_remote_python
    from remote_dev.core.cancellation import current_event

    source = Path(__file__).read_text(encoding='utf-8')
    source += '\nimport sys\nprint(json.dumps(inspect_existing(json.load(sys.stdin))))\n'
    endpoint = resolve_endpoint({
        'host': target.host, 'port': target.port, 'user': target.user,
        'root': '/', 'cwd': '/',
        **({'identity_file': identity_file} if identity_file else {}),
        **({'auth_ref': auth_ref} if auth_ref else {}),
    })
    try:
        result = run_remote_python(endpoint, source,
            {'container': container, 'user': user, 'ssh_port': ssh_port,
             'workdir': workdir, 'image_request': image_request}, timeout_ms=10000)
    except TimeoutError:
        raise
    except Exception as exc:
        result = {'status': 'failed', 'reason': f'{type(exc).__name__}: {exc}'[:300]}
    cancellation = current_event()
    if cancellation is not None and cancellation.is_set():
        return {'status': 'cancelled'}
    if isinstance(result, dict):
        return result
    return {'status': 'failed', 'reason': 'invalid host observation'}
