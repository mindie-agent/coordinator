import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mindie_coordinator.shared_native_discovery import DISCOVER, export_verified_donor


@pytest.mark.parametrize('mismatch', [None, 'image', 'mount', 'unmanaged', 'no-profile', 'failed-export'])
def test_discovery_exports_only_compatible_verified_roots(mismatch):
    calls = []
    request = {'image_digest': 'sha256:same', 'preparation': {'native': {'vllm': 'same'}},
               'export_source': 'publish verified outputs only'}
    container = {'Id': 'container-id', 'Name': '/mindie-alice', 'Image': 'sha256:same',
                 'Mounts': [{'Type': 'bind', 'Source': '/tmp', 'Destination': '/tmp', 'RW': True}]}
    if mismatch == 'image':
        container['Image'] = 'sha256:other'
    elif mismatch == 'mount':
        container['Mounts'] = []
    elif mismatch == 'unmanaged':
        container['Name'] = '/production-unrelated'
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        output, code = '', 0
        if argv[1] == 'ps':
            output = 'container-id\n'
        elif argv[1] == 'inspect':
            output = json.dumps([container])
        elif argv[2] != '-i':
            output = json.dumps([] if mismatch == 'no-profile' else [{'root': '/vllm-workspace/executions/alice/host/default',
                                   'profile': {'launch_env': {'PATH': '/donor/.venv/bin:/usr/bin'}}, 'exact': True}])
        else:
            assert kwargs['input'] == request['export_source']
            assert 'export PATH=/donor/.venv/bin:/usr/bin' in argv[-1]
            assert 'exec python3 - ' in argv[-1]
            output, code = json.dumps({'status': 'stored'}), int(mismatch == 'failed-export')
        return SimpleNamespace(stdout=output, stderr='fixture failure' if code else '', returncode=code)
    reply = export_verified_donor(request, run)
    assert reply['status'] == ('stored' if mismatch is None else 'miss')
    if mismatch in ('image', 'mount', 'unmanaged'):
        assert len(calls) == 2
    assert all('rm' not in argv and 'stop' not in argv and 'start' not in argv for argv, _ in calls)


@pytest.mark.parametrize('failed_exact', [False, True])
@pytest.mark.parametrize('unavailable', [None, 'timeout', 'invalid-json'])
def test_global_exact_priority_and_finite_donor_inventory(failed_exact, unavailable):
    calls, exported = [], []
    request = {'image_digest': 'sha256:same', 'preparation': {}, 'export_source': 'verify then publish'}
    containers = [{'Id': name + '-id', 'Name': '/mindie-' + name, 'Image': 'sha256:same',
                   'Mounts': [{'Type': 'bind', 'Source': '/tmp', 'Destination': '/tmp', 'RW': True}]}
                  for name in ('base', 'exact', 'other-base', 'unavailable')]
    def run(argv, **kwargs):
        calls.append(argv)
        code, output = 0, ''
        if argv[1] == 'ps':
            output = 'base-id exact-id other-base-id'
        elif argv[1] == 'inspect':
            output = json.dumps(containers)
        elif argv[2] != '-i':
            if argv[2] == 'unavailable-id':
                if unavailable == 'timeout':
                    raise subprocess.TimeoutExpired(argv, 30)
                return SimpleNamespace(stdout='invalid-json' if unavailable else '[]', stderr='', returncode=0)
            output = json.dumps([{'root': '/vllm-workspace/executions/' + argv[2],
                                  'profile': {'launch_env': {}}, 'exact': argv[2] == 'exact-id'}])
        else:
            exported.append(argv[3])
            code = int(failed_exact and argv[3] == 'exact-id')
            output = json.dumps({'status': 'stored', 'bundle': argv[3], 'native_key': argv[3]})
        return SimpleNamespace(stdout=output, stderr='export failed' if code else '', returncode=code)
    result = export_verified_donor(request, run)
    assert exported == (['exact-id', 'base-id'] if failed_exact else ['exact-id'])
    # A completed receiver miss can consume later candidates without another
    # Docker scan. Changing/reusing a container name cannot redirect its ID.
    inventory_calls = len(calls)
    remaining = result['remaining_donors']
    containers.clear()
    while remaining:
        result = export_verified_donor({**request, 'donors': remaining}, run)
        assert result['status'] == 'stored'
        assert len(result['remaining_donors']) < len(remaining)
        remaining = result['remaining_donors']
    assert exported == ['exact-id', 'base-id', 'other-base-id']
    assert all(argv[:3] == ['docker', 'exec', '-i'] for argv in calls[inventory_calls:])
    before_empty = len(calls)
    assert export_verified_donor({**request, 'donors': []}, run)['status'] == 'miss'
    assert len(calls) == before_empty


@pytest.mark.parametrize('failure', ['cancelled', 'uncertain', 'remote-unknown'])
def test_export_interruption_is_not_an_ordinary_cache_miss(monkeypatch, failure):
    from mindie_coordinator.backend import RemoteBackend
    from mindie_coordinator.preparation_process import PreparationCancelled, PreparationUncertain
    backend = RemoteBackend()
    spec = {'container_name': 'mindie-recipient', 'host_endpoint': {'host': 'host', 'port': 22, 'user': 'root'}}
    monkeypatch.setattr(backend, 'bash', lambda *args: json.dumps({
        'Id': 'recipient-generation', 'Image': 'sha256:same', 'State': {'Running': True}}))
    result = {'status': 'failed', 'remote_outcome': 'unknown'} if failure == 'remote-unknown' else {'status': failure}
    monkeypatch.setattr('remote_dev.core.ssh_transport.run_remote_python', lambda *args, **kwargs: result)
    with pytest.raises(PreparationCancelled if failure == 'cancelled' else PreparationUncertain):
        backend._export_shared_native(spec)


@pytest.mark.parametrize('mismatch', [None, 'soc', 'build_env', 'native', 'unqualified'])
def test_actual_metadata_discovery_accepts_equivalent_image_request(tmp_path, mismatch):
    root = tmp_path / 'executions/alice/host/default'
    marker = root / '.mindie-runtime/ready-profile.json'
    marker.parent.mkdir(parents=True)
    prior = {'dependencies': {'vllm': 'deps'}, 'native': {'vllm': 'v', 'vllm-ascend': 'a'},
             'environment': {'environment': {}, 'build_env': {}}}
    manifest = {'runtime_root': str(root.resolve()), 'profile': {'image_digest': 'sha256:same', 'soc': 'A3'},
                'preparation': prior}
    if mismatch == 'unqualified':
        manifest.pop('preparation')
    marker.write_text(json.dumps(manifest))
    request = json.loads(json.dumps(prior))
    request['environment']['environment'] = {'image': 'registry/image@sha256:reference', 'soc': 'A3'}
    if mismatch == 'soc':
        request['environment']['environment']['soc'] = 'A2'
    elif mismatch == 'build_env':
        request['environment']['build_env']['CXXFLAGS'] = '-different'
    elif mismatch == 'native':
        request['native']['vllm'] = 'other-vllm'
    script = DISCOVER.replace("pathlib.Path('/vllm-workspace')", 'pathlib.Path(' + repr(str(tmp_path)) + ')')
    result = subprocess.run([sys.executable, '-c', script, json.dumps({'preparation': request, 'image_digest': 'sha256:same'})],
                            capture_output=True, text=True, check=True)
    found = json.loads(result.stdout)
    assert bool(found) == (mismatch is None)
    if found:
        assert found[0]['root'] == str(root)
