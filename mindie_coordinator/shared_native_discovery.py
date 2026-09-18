"""On-demand export from existing verified MindIE roots on one Docker host.

Run only after a shared-cache miss. Discovery reads donor metadata; export
writes only the host-shared cache. No donor interpreter, source or container
configuration is modified, and unqualified historical installs are skipped.
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess


DISCOVER = r'''
import json,sys,pathlib
args=json.loads(sys.argv[1])
base=pathlib.Path('/vllm-workspace')
result=[]
for pattern in ('tasks/*/*/*/.mindie-runtime/ready-profile.json','executions/*/*/*/.mindie-runtime/ready-profile.json'):
    for marker in base.glob(pattern):
        if marker.is_symlink():
            continue
        try:
            manifest=json.loads(marker.read_text())
            profile=manifest['profile']
            previous=manifest['preparation']
            requested=args['preparation']
            if (profile['image_digest'] != args['image_digest'] or
                previous.get('dependencies') != requested.get('dependencies') or
                previous.get('native',{}).get('vllm') != requested.get('native',{}).get('vllm') or
                previous.get('environment',{}).get('build_env',{}) != requested.get('environment',{}).get('build_env',{})):
                continue
            constraints=requested.get('environment',{}).get('environment',{})
            if any(constraints.get(key) and constraints[key] != profile.get(key)
                   for key in ('soc','cann','python_abi','machine_type')):
                continue
            if args.get('machine_type') and args['machine_type'] != profile.get('machine_type'):
                continue
            root=marker.parents[1]
            if str(root.resolve()) != manifest['runtime_root']:
                continue
            result.append({'root':str(root),'profile':profile,
                           'exact':previous.get('native') == requested.get('native')})
        except (ValueError,KeyError,TypeError,OSError):
            continue
print(json.dumps(sorted(result,key=lambda row:not row['exact'])))
'''


def discover_donors(request: dict, run, reasons: list) -> list[dict]:
    listed = run(['docker', 'ps', '--filter', 'name=mindie-', '--format', '{{.ID}}'],
                 capture_output=True, text=True, check=True, timeout=15)
    identifiers = listed.stdout.split()
    if not identifiers:
        return []
    containers = json.loads(run(['docker', 'inspect', *identifiers], capture_output=True,
                                text=True, check=True, timeout=15).stdout)
    donors = []
    for container in containers:
        name = container.get('Name', '').lstrip('/')
        if not re.fullmatch(r'mindie-[A-Za-z0-9_.-]+', name) or container.get('Image') != request['image_digest']:
            continue
        # Only a shared host /tmp mount can publish a bundle visible to the
        # recipient. Private container paths are never promoted as host cache.
        if not any(row.get('Type') == 'bind' and row.get('Source') == '/tmp' and
                   row.get('Destination') == '/tmp' and row.get('RW', False)
                   for row in container.get('Mounts', [])):
            continue
        # Retain Docker's immutable identity through metadata/export. A new
        # container taking the old name is not part of this discovery pass.
        identifier = container['Id']
        try:
            query = run(['docker', 'exec', identifier, 'python3', '-c', DISCOVER,
                         json.dumps({'preparation': request['preparation'], 'image_digest': request['image_digest'],
                                     'machine_type': request.get('machine_type')})],
                        capture_output=True, text=True, timeout=30)
            found = json.loads(query.stdout) if not query.returncode else []
        except (subprocess.TimeoutExpired, ValueError):
            # This is a read-only metadata query. One unavailable container
            # must not discard valid candidates already found on this host.
            reasons.append(name + ': metadata unavailable')
            continue
        for donor in found:
            donors.append({'container_id': identifier, 'container_name': name, 'root': donor['root'],
                           'environment': donor['profile'].get('launch_env', {}), 'exact': donor['exact']})
    # An exact root in a later container must outrank an incremental baseline
    # in the first one. Discovery order is not a compatibility decision.
    return sorted(donors, key=lambda donor: not donor['exact'])


def export_verified_donor(request: dict, run=subprocess.run) -> dict:
    reasons = []
    # Subsequent attempts consume this finite inventory only after a completed
    # recipient miss. They never discover new roots or replay an uncertain job.
    donors = request.get('donors')
    if donors is None:
        donors = discover_donors(request, run, reasons)
    for index, donor in enumerate(donors):
        environment = donor['environment']
        if not isinstance(environment, dict) or any(not re.fullmatch('[A-Z_][A-Z0-9_]*', key) or
                                                   not isinstance(value, str) for key, value in environment.items()):
            continue
        exports = '\n'.join('export ' + key + '=' + shlex.quote(value) for key, value in environment.items())
        command = exports + '\nexec python3 - ' + shlex.quote(json.dumps({'root': donor['root']}))
        exported = run(['docker', 'exec', '-i', donor['container_id'], 'bash', '-c', command],
                       input=request['export_source'], capture_output=True, text=True, timeout=60)
        if exported.returncode == 0:
            result = json.loads(exported.stdout)
            if result.get('status') == 'stored':
                return {**result, 'donor_container': donor['container_name'], 'donor_root': donor['root'],
                        'remaining_donors': donors[index + 1:]}
        reasons.append(donor['container_name'] + ': ' + exported.stderr[-500:])
    return {'status': 'miss', 'reason': '; '.join(reasons[-3:]) or 'no compatible verified donor'}
