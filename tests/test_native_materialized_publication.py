"""The actual composed Bash argument publishes only after fixed Git success."""
import copy
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from test_native_view_publication import donor
from mindie_coordinator import parity, parity_support
from mindie_coordinator.native_publication import NativeViewPublication


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def fixed_plan(donor, *, missing=False):
    source, view, manifest, args, _ = donor
    # The native donor fixture initially includes an empty source view. Real
    # materialization instead creates it under its existing ownership lock.
    shutil.rmtree(view)
    rows = []
    for name in ('vllm', 'vllm-ascend'):
        repo = source / name
        (repo / '.gitignore').write_text('*.so\n_cann_ops_custom/\n_build_info.py\n_version.py\n')
        git(repo, 'init', '-q')
        git(repo, 'config', 'user.name', 'Native fixture')
        git(repo, 'config', 'user.email', 'native@example.invalid')
        git(repo, 'add', '.')
        git(repo, 'commit', '-qm', 'fixed Python')
        mirror = source.parent / (name + '.git')
        if missing:
            git(source.parent, 'init', '--bare', str(mirror))
        else:
            git(source.parent, 'clone', '--bare', str(repo), str(mirror))
        rows.append({'relpath': name, 'commit': git(repo, 'rev-parse', 'HEAD'),
                     'tree': git(repo, 'rev-parse', 'HEAD^{tree}'), 'submodules': [], 'mirror': str(mirror)})
    snapshot = {'id': args['source_id'], 'build_env': {}, 'records': [
        {**row, 'build_inputs': args['build_inputs'][row['relpath']]} for row in rows]}
    spec = {'python': sys.executable, 'endpoint': {'root': str(view)},
            'preparation': args['preparation'], 'source_snapshot': snapshot}
    previous = {'python': sys.executable, 'endpoint': {'root': str(source)},
                'attestation': {**manifest, 'container_id': 'same-container'}}
    publication = NativeViewPublication(spec, previous, args['versions'])
    request = {'root': str(view), 'source_id': args['source_id'], 'carrier_ref': 'refs/parity/test/carrier', 'records': rows}
    return publication, request


def materialize_program(request):
    extra = ''
    # The materializer's shared-object owner supplies this helper in current
    # composition; older materializers have no cross-container object call.
    try:
        from mindie_coordinator.shared_source_objects import copy_fixed_objects
    except ImportError:
        pass
    else:
        extra = inspect.getsource(copy_fixed_objects) + '\n'
    return extra + inspect.getsource(parity_support._materialize_fixed) + '\nimport json\nresult = _materialize_fixed(' + repr(request) + ')\n'


@pytest.mark.parametrize('missing', [False, True])
def test_actual_bash_argument_composes_materialization_and_fixed_native_proof(donor, missing):
    publication, request = fixed_plan(donor, missing=missing)
    from mindie_coordinator.preparation_script import preparation_command
    command = preparation_command(publication.wrap_program(materialize_program(request)))
    size = len(command.encode('utf-8'))
    assert size <= 96 * 1024
    completed = subprocess.run(['/bin/bash', '-c', command], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    if missing:
        assert result['status'] == 'missing' and 'native_view' not in result
        assert not Path(request['root']).exists()
    else:
        assert result['status'] == 'materialized'
        observed = publication.accept(result['native_view'])
        marker = Path(request['root']) / '.mindie-runtime/ready-profile.json'
        actual = json.loads(marker.read_text())
        assert observed['runtime_root'] == request['root'] and observed['container_id'] == 'same-container'
        assert actual['files'] == publication.donor['files']
        assert actual['execution_view']['source_id'] == request['source_id']
        assert result['commits'] == {row['relpath']: row['commit'] for row in request['records']}
        timings = result['preparation_timings']
        assert 0 <= timings['source_metadata'] <= timings['native_publication']
    print(json.dumps({'composite_command_bytes': size, 'status': result['status'], 'native_published': not missing}))


def test_failed_native_publication_never_returns_a_materialized_only_success(donor):
    publication, request = fixed_plan(donor)
    publication.request['donor_manifest_digest'] = 'f' * 64
    completed = subprocess.run(['/bin/bash', '-c', publication.wrap_program(materialize_program(request))],
                               capture_output=True, text=True)
    assert completed.returncode != 0 and 'native donor manifest changed' in completed.stderr
    assert not (Path(request['root']) / '.mindie-runtime/ready-profile.json').exists()
    assert not completed.stdout.strip()


@pytest.mark.parametrize('edit_bytes', [0, 64, 24000, 100000])
def test_real_materializer_api_budgets_native_modules_and_inline_git_pack(donor, monkeypatch, edit_bytes):
    from test_fixed_materialization import make_record, snapshot
    publication, request = fixed_plan(donor)
    source = Path(publication.request['source_root'])
    mirrors = {row['relpath']: row['mirror'] for row in request['records']}
    for row in request['records']:
        git(row['mirror'], 'update-ref', 'refs/parity/test/transport-carrier', row['commit'])
    if edit_bytes:
        repo = source / 'vllm'
        (repo / 'data.bin').write_bytes(os.urandom(edit_bytes))
        git(repo, 'add', '.')
        git(repo, 'commit', '-qm', 'fixed source edit')
    records = [make_record(source / name, name) for name in ('vllm', 'vllm-ascend')]
    fixed = snapshot(records)
    publication.request['source_id'] = fixed['id']
    publication.request['preparation']['source_id'] = fixed['id']
    commands, pushes = [], []
    original_git = parity.git
    def transfer(repo, args, **kwargs):
        if args[0] == 'push':
            pushes.append(args)
        return original_git(repo, args, **kwargs)
    def stream(endpoint, command, **kwargs):
        commands.append(command)
        assert len(command.encode('utf-8')) <= 96 * 1024
        completed = subprocess.run(['/bin/bash', '-c', command], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        return SimpleNamespace(stdout=completed.stdout)
    monkeypatch.setattr(parity, 'git', transfer)
    monkeypatch.setattr(parity, 'ssh_exec_stream', stream)
    monkeypatch.setattr(parity, 'mirror_path_for', lambda cache, workspace, row: mirrors[row.relpath])
    monkeypatch.setattr(parity, 'git_remote_url', lambda endpoint, mirror: mirror)
    monkeypatch.setattr(parity, 'git_ssh_environment', lambda endpoint: os.environ.copy())
    result = parity.materialize_fixed_sources(workspace_id='test', source_snapshot=fixed,
        endpoint={'host': 'fixture', 'port': 22, 'user': 'fixture', 'root': request['root']},
        native_publication=publication, container_cache_root=str(source.parent / 'cache'))
    assert publication.accept(result['native_view'])['execution_view']['source_id'] == fixed['id']
    assert set(result['preparation_timings']) == {'materialize', 'source_metadata', 'native_publication'}
    assert all(value >= 0 for value in result['preparation_timings'].values())
    assert len(commands) == (2 if edit_bytes else 1)
    assert len(pushes) == (1 if edit_bytes == 100000 else 0)
    if edit_bytes:
        assert (Path(request['root']) / 'vllm/data.bin').read_bytes() == (source / 'vllm/data.bin').read_bytes()
    print(json.dumps({'edit_bytes': edit_bytes, 'command_bytes': [len(row.encode('utf-8')) for row in commands],
                      'git_pushes': len(pushes), 'native_published': True}))


def test_materializer_accounts_for_entire_composite_before_inline_pack(monkeypatch, tmp_path):
    """Incompressible over-budget composition retains the two-job fallback."""
    from test_fixed_materialization import make_record, snapshot
    if sys.platform != 'linux':
        pytest.skip('actual Git peer runs under Linux')
    source = tmp_path / 'source'
    source.mkdir()
    git(source, 'init', '-q')
    git(source, 'config', 'user.name', 'Test')
    git(source, 'config', 'user.email', 'test@example.invalid')
    (source / 'model.py').write_text('value = 1')
    git(source, 'add', '.')
    git(source, 'commit', '-qm', 'fixed')
    record = make_record(source, 'project')
    fixed = snapshot([record])
    commands = []
    oversized = os.urandom(100000).hex()
    class OversizedPublication:
        def wrap_program(self, program):
            return oversized
    def stream(endpoint, command, **kwargs):
        commands.append(command)
        assert len(command.encode()) < 96 * 1024 and command != 'x' * len(command)
        return SimpleNamespace(stdout=json.dumps({'status': 'materialized', 'root': str(tmp_path / 'view'),
                                                  'source_id': fixed['id'], 'commits': {'project': record.commit}}))
    monkeypatch.setattr(parity, 'ssh_exec_stream', stream)
    result = parity.materialize_fixed_sources(workspace_id='test', source_snapshot=fixed,
        endpoint={'host': 'fixture', 'port': 22, 'user': 'fixture', 'root': str(tmp_path / 'view')},
        native_publication=OversizedPublication(), container_cache_root=str(tmp_path / 'cache'))
    assert 'native_view' not in result and len(commands) == 1


@pytest.mark.skipif(sys.platform != 'linux', reason='executes the Linux container worker with /proc process identities')
def test_backend_prepares_missing_root_before_real_owned_worker_launch_and_quiet(donor, monkeypatch):
    from remote_dev.core.ssh_transport import RemoteCompleted
    from remote_dev.processes.client import worker_source
    from remote_dev.processes.worker import control_job
    from test_fixed_materialization import make_record, snapshot
    from mindie_coordinator.backend import PreparedNativeView, RemoteBackend
    from mindie_coordinator.preparation_process import stop_preparation_process

    publication, request = fixed_plan(donor)
    source, root = Path(publication.request['source_root']), Path(request['root'])
    records = [make_record(source / name, name) for name in ('vllm', 'vllm-ascend')]
    for row in records:
        row.scm_version = '2.1'
        row.build_inputs = publication.request['build_inputs'][row.relpath]
    fixed = snapshot(records)
    endpoint = {'host': 'fixture', 'port': 22, 'user': 'fixture', 'root': str(root), 'cwd': str(root)}
    spec = {'user': 'fixture', 'python': sys.executable, 'endpoint': endpoint, 'host_endpoint': None,
            'source_snapshot': fixed, 'preparation': {**publication.request['preparation'], 'source_id': fixed['id']}}
    mirrors = {row['relpath']: row['mirror'] for row in request['records']}
    monkeypatch.setattr(parity, 'mirror_path_for', lambda cache, workspace, row: mirrors[row.relpath])
    saved, actions = [], []
    worker = worker_source()
    assert not root.exists()
    # This is the real worker precondition that a direct bash-c test misses.
    with pytest.raises(FileNotFoundError):
        control_job({'root': str(root), 'job_id': 'absent-root', 'action': 'status'}, worker)

    bootstrap = str(source.parent)
    monkeypatch.setattr('mindie_coordinator.provision.host_ops.DEFAULT_WORKDIR', bootstrap)
    original_root_script = parity.prepare_isolated_root_script
    monkeypatch.setattr(parity, 'prepare_isolated_root_script',
                        lambda *a, **k: 'hostname() { return 1; }\n' + original_root_script(*a, **k))
    def control(actual_endpoint, job_id, action, **kwargs):
        assert actual_endpoint == {**endpoint, 'root': bootstrap, 'cwd': bootstrap}
        if action == 'launch':
            assert saved and not root.exists()
        actions.append(action)
        return control_job({'root': actual_endpoint['root'], 'job_id': job_id, 'action': action, **kwargs}, worker)

    monkeypatch.setattr('remote_dev.core.ssh_transport.run_rpc_script',
                        lambda *a, **k: pytest.fail('root setup must share the owned job'))
    monkeypatch.setattr('mindie_coordinator.preparation_process.control', control)
    backend = RemoteBackend()
    monkeypatch.setattr(backend, '_prepare_native_view', lambda *a, **k: pytest.fail('publication must share the owned job'))
    monkeypatch.setattr(backend, 'bash', lambda *a, **k: pytest.fail('no extra shell operation'))
    try:
        result = backend.prepare_task_root(spec, sources={name: str(source / name) for name in mirrors},
            environment={}, source_snapshot=fixed, reuse={'kind': 'native', 'runtime': publication.previous},
            on_preparation_job=lambda row: saved.append(copy.deepcopy(row)))
        assert isinstance(result, PreparedNativeView)
        assert result.attestation['execution_view']['source_id'] == fixed['id']
        assert actions[0] == 'launch' and actions.count('launch') == 1
        assert len({row['job_id'] for row in saved}) == 1
        assert saved[-1]['endpoint']['root'] == bootstrap and saved[-1]['quiet'] is True
        assert saved[-1]['stage_timings']['prepare-root'] >= 0
        assert saved[-1]['result']['exit_code'] == 0
        # A restarted controller can observe the exact job under the same root.
        final = control(saved[-1]['endpoint'], saved[-1]['job_id'], 'status')
        assert final['quiet'] is True and final['processes'] == []
        assert final['result']['descendants_drained'] is True
    finally:
        if saved:
            assert stop_preparation_process(saved[-1], lambda _: None, force=True)
