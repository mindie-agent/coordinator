"""Bounded read-only reuse of an already configured fixed-image container."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import time


def inspect_existing(request, run=subprocess.run):
    """One host observation; ambiguity falls back to the provision owner."""
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
            return {'status': 'unknown', 'reason': 'container identity unavailable'}
        info = rows[0]
        identifier = info.get('Id', '')
        if info.get('Name') != '/' + container or not re.fullmatch('[0-9a-f]{64}', identifier):
            return {'status': 'unknown', 'reason': 'container identity unavailable'}
        config = info.get('Config') or {}
        image = config.get('Image')
        image_id = info.get('Image', '')
        if not image or not re.fullmatch('sha256:[0-9a-f]{64}', image_id):
            return {'status': 'unknown', 'reason': 'container image unavailable'}
        # Same explicit-reference comparison as bootstrap with its current
        # use_prepared_image_cache=false. A base-image label alone is not proof.
        if image not in request['image_request']['candidates']:
            return {'status': 'mismatch', 'container_id': identifier, 'image': image,
                    'image_id': image_id,
                    'reason': 'existing container image does not match the requested selector'}
        state, labels = info.get('State') or {}, config.get('Labels') or {}
        if (state.get('Running') is not True or state.get('Paused') or state.get('Restarting')
                or int(state.get('Pid') or 0) <= 0):
            return {'status': 'unknown', 'reason': 'container is not running normally'}
        if (labels.get('com.mindie.managed') != 'true'
                or labels.get('com.mindie.namespace') != request['user']
                or labels.get('com.mindie.workdir') != request['workdir']
                or labels.get('com.mindie.container_ssh_port') != str(port)
                or (info.get('HostConfig') or {}).get('NetworkMode') != 'host'):
            return {'status': 'unknown', 'reason': 'managed container or SSH port facts are incomplete'}
        # Host networking needs evidence that this container's sshd owns the
        # configured listening socket, not just a label or another user's port.
        processes = output(['docker', 'top', identifier, '-eo', 'pid']).splitlines()[1:]
        owned = {int(line.strip()) for line in processes if line.strip().isdigit()}
        listeners = output(['ss', '-ltnpH', 'sport = :' + str(port)]).splitlines()
        if not listeners:
            return {'status': 'unknown', 'reason': 'configured SSH port is not listening'}
        listener_pids = set()
        for line in listeners:
            pids = {int(value) for value in re.findall(r'pid=(\d+)', line)}
            sshd = {int(value) for value in re.findall(r'\("sshd[^\"]*",pid=(\d+)', line)}
            if not pids or pids != sshd or not pids.issubset(owned):
                return {'status': 'unknown', 'reason': 'SSH listener ownership is unconfirmed'}
            listener_pids.update(pids)
        return {'status': 'match', 'container_id': identifier, 'image': image,
                'image_id': image_id, 'base_image': labels.get('com.mindie.base_image'),
                'base_image_id': labels.get('com.mindie.base_image_id'), 'ssh_port': port,
                'listener_pids': sorted(listener_pids), 'observed_at': time.time()}
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError, TimeoutError) as exc:
        return {'status': 'unknown', 'reason': str(exc)[:300]}


def observe_existing(target, *, container, user, ssh_port, workdir, image_request):
    """Use the existing cached Python transport, with no host mutation or probe."""
    from remote_dev.core.endpoint import resolve_endpoint
    from remote_dev.core.ssh_transport import run_remote_python
    from remote_dev.core.cancellation import current_event

    source = Path(__file__).read_text(encoding='utf-8')
    source += '\nimport sys\nprint(json.dumps(inspect_existing(json.load(sys.stdin))))\n'
    endpoint = resolve_endpoint({'host': target.host, 'port': target.port,
                                 'user': target.user, 'root': '/', 'cwd': '/'})
    try:
        result = run_remote_python(endpoint, source,
            {'container': container, 'user': user, 'ssh_port': ssh_port,
             'workdir': workdir, 'image_request': image_request}, timeout_ms=10000)
    except Exception as exc:
        result = {'status': 'unknown', 'reason': str(exc)[:300]}
    cancellation = current_event()
    if cancellation is not None and cancellation.is_set():
        return {'status': 'cancelled'}
    return result if isinstance(result, dict) else {'status': 'unknown', 'reason': 'invalid host observation'}
