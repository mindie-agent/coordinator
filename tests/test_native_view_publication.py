"""One native publication feeds managed binding without another adoption pass."""
import copy
import importlib.machinery
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor

import pytest

from mindie_coordinator import preparation_cache as cache, runtime_profile as profile
from mindie_coordinator.build_inputs import BUILD_INPUT_ENV_KEYS


@pytest.fixture
def donor(tmp_path, monkeypatch):
    if os.name == 'nt':
        pytest.skip('native publication runs in a Linux container with POSIX loader paths')
    source, view = tmp_path / 'donor', tmp_path / 'execution'
    for root in (source, view):
        for name, package in (('vllm', 'vllm'), ('vllm-ascend', 'vllm_ascend')):
            path = root / name / package
            path.mkdir(parents=True)
            (path / '__init__.py').write_text("raise RuntimeError('changed Python ran')\n")
    for name, package in (('vllm', 'vllm'), ('vllm-ascend', 'vllm_ascend')):
        metadata = source / '.mindie-runtime/metadata' / (package + '-2.0.dist-info')
        metadata.mkdir(parents=True)
        (metadata / 'METADATA').write_text('Name: ' + name + '\nVersion: 2.0\n')
    dependencies = tmp_path / 'dependencies'
    for name in ('torch', 'torch-npu'):
        metadata = dependencies / (name.replace('-', '_') + '-1.0.dist-info')
        metadata.mkdir(parents=True)
        (metadata / 'METADATA').write_text('Name: ' + name + '\nVersion: 1.0\n')
    package = 'vllm-ascend/vllm_ascend/'
    files = {package + 'vllm_ascend_C' + importlib.machinery.EXTENSION_SUFFIXES[0]: 'library',
             package + '_cann_ops_custom/kernel.so': 'library',
             package + '_cann_ops_custom/config.json': 'metadata', package + '_build_info.py': 'metadata'}
    for name in files:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'original native output')
    settings = {name: '1.0' for name in profile.PROFILE_FIELDS}
    settings.update(vllm='2.0', vllm_ascend='2.0', python_abi=sysconfig.get_config_var('SOABI'),
                    build_env={}, launch_env={'PYTHONPATH': ':'.join([*(str(source / suffix) for suffix in
                        ('.mindie-runtime/metadata', 'vllm', 'vllm-ascend')), str(dependencies)])},
                    compatibility_evidence='.mindie-runtime/profile-evidence/smoke.json', system_files={})
    settings['launch_env'].update(
        LD_LIBRARY_PATH=str(source / 'vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/test') + ':/image/lib',
        ASCEND_CUSTOM_OPP_PATH=str(source / 'vllm-ascend/vllm_ascend/_cann_ops_custom'))
    evidence = {}
    for name in ('cann', 'driver'):
        system = tmp_path / name
        system.write_text('system version')
        settings['system_files'][name] = {'path': str(system), 'sha256': profile.file_digest(system)}
        evidence[name] = '.mindie-runtime/profile-evidence/' + name + '.json'
        path = source / evidence[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings['system_files'][name]))
    evidence['smoke'] = '.mindie-runtime/profile-evidence/smoke.json'
    original = {'passed': True, 'profile_key': profile.profile_key(settings), 'stdout': 'original import proof'}
    (source / evidence['smoke']).write_text(json.dumps(original))
    inputs = {name: {'native': 'a' * 64, 'dependencies': 'b' * 64, 'build_env': 'c' * 64}
              for name in ('vllm', 'vllm-ascend')}
    manifest = profile.capture(source, settings, inputs, files, evidence)
    marker = source / '.mindie-runtime/ready-profile.json'
    marker.write_text(json.dumps(manifest))
    for name in ('vllm', 'vllm_ascend'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    for suffix in ('vllm', 'vllm-ascend', '.mindie-runtime/metadata'):
        monkeypatch.syspath_prepend(str(source / suffix))
    monkeypatch.syspath_prepend(str(dependencies))
    monkeypatch.setenv('PYTHONPATH', str(dependencies))
    real_version = profile.importlib.metadata.version
    monkeypatch.setattr(profile.importlib.metadata, 'version',
                        lambda name: real_version(name) if name in ('vllm', 'vllm-ascend') else '1.0')
    for name in ('verify_environment', 'native_compatibility_receipt', 'native_source_mapping', 'profile_key',
                 'build_key', 'verified_preparation', 'verify_native_compatibility', 'checked_file',
                 'file_digest', 'digest', 'build_toolchain_from_logs'):
        monkeypatch.setattr(cache, name, getattr(profile, name), raising=False)
    for name in ('native_vendor_paths', 'native_vendor_launch_environment', 'native_import_smoke', 'native_compatibility_key'):
        monkeypatch.setattr(cache, name, getattr(profile, name), raising=False)
    monkeypatch.setattr(cache, 'BUILD_INPUT_ENV_KEYS', BUILD_INPUT_ENV_KEYS, raising=False)
    monkeypatch.setattr(cache, 'verify', lambda *a, **k: pytest.fail('native outputs must not be re-attested'), raising=False)
    args = {'root': str(view), 'source_root': str(source), 'source_id': 'accepted-source',
            'donor_manifest_digest': profile.digest(manifest), 'build_env': {}, 'build_inputs': inputs,
            'preparation': {'environment': {'environment': {}, 'build_env': {}},
                            'dependencies': {}, 'native': {}, 'source_id': 'accepted-source'},
            'versions': {name: {'version': '2.1', 'source_head': 'accepted-head'} for name in inputs}}
    return source, view, manifest, args, original


def test_publication_reuses_native_proof_once_and_changed_python_still_fails(donor, monkeypatch):
    source, view, old, args, original = donor
    hashes = []
    monkeypatch.setattr(cache, 'file_digest', lambda path: hashes.append(path) or profile.file_digest(path))
    monkeypatch.setattr(sys, 'argv', ['publication', json.dumps(args)])
    namespace = vars(cache).copy()
    outputs = []
    namespace['print'] = outputs.append
    exec(compile(cache.REMOTE_NATIVE_VIEW_SUFFIX, '<native-view>', 'exec'), namespace)
    reply = json.loads(outputs[0])
    assert 'files' not in reply['manifest']
    current = {**reply['manifest'], 'files': old['files']}
    assert reply['manifest_digest'] == profile.digest(current)
    assert json.loads((view / '.mindie-runtime/ready-profile.json').read_text()) == current
    assert hashes == [view / name for name in old['files']]
    assert current['build_key'] != old['build_key']
    assert current['profile']['vllm'] == current['profile']['vllm_ascend'] == '2.1'
    smoke = json.loads((view / current['evidence']['smoke']['path']).read_text())
    assert smoke['compatibility']['origin']['smoke'] == original
    assert smoke['python_import_executed'] is False and 'passed' not in smoke
    profile.verify(view, current)  # An explicit complete adoption still accepts this proof.
    profile.verify_execution_view(view, current)
    result = subprocess.run([sys.executable, '-c', 'import vllm'], capture_output=True, text=True,
                            env={**os.environ, 'PYTHONPATH': str(view / 'vllm')})
    assert result.returncode != 0 and 'changed Python ran' in result.stderr
    (view / next(iter(old['files']))).write_bytes(b'execution-local change')
    assert (source / next(iter(old['files']))).read_bytes() == b'original native output'


def test_successive_hot_views_move_only_the_current_loader_paths_and_keep_flat_origin(donor, tmp_path):
    source, view, old, args, original = donor
    first = cache.prepare_native_view(view, source, old, args)
    manifest = {**first['manifest'], 'files': old['files']}
    later = tmp_path / 'second-execution'
    for name, package in (('vllm', 'vllm'), ('vllm-ascend', 'vllm_ascend')):
        path = later / name / package
        path.mkdir(parents=True)
        (path / '__init__.py').write_text("raise RuntimeError('second Python view')\n")
    following = {**args, 'root': str(later), 'source_root': str(view), 'source_id': 'second-inputs',
                 'preparation': {**args['preparation'], 'source_id': 'second-inputs'},
                 'donor_manifest_digest': profile.digest(manifest), 'build_inputs': manifest['build_inputs'],
                 'versions': {name: {'version': '2.2', 'source_head': 'second-head'} for name in args['versions']}}
    second = cache.prepare_native_view(later, view, manifest, following)
    current = {**second['manifest'], 'files': manifest['files']}
    for name in ('PYTHONPATH', 'LD_LIBRARY_PATH', 'ASCEND_CUSTOM_OPP_PATH'):
        parts = current['profile']['launch_env'][name].split(':')
        assert not any(part.startswith(str(root) + '/') for root in (source, view) for part in parts)
    assert current['profile']['launch_env']['LD_LIBRARY_PATH'].endswith(':/image/lib')
    smoke = json.loads((later / current['evidence']['smoke']['path']).read_text())
    assert smoke['compatibility']['origin']['smoke'] == original
    assert smoke['compatibility']['origin']['build_key'] == old['build_key']
    assert smoke['python_import_executed'] is False and 'passed' not in smoke
    profile.verify_execution_view(later, current)


@pytest.mark.parametrize('passed', [False, True])
def test_missing_vendor_loader_path_requires_one_owned_import_then_reuses_new_proof(donor, monkeypatch, passed):
    source, view, old, args, original = donor
    library = 'vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/libopapi.so'
    path = source / library
    path.parent.mkdir(parents=True)
    path.write_bytes(b'verified donor library')
    old['files'][library] = {'role': 'library', 'sha256': profile.file_digest(path)}
    args['donor_manifest_digest'] = profile.digest(old)
    original_bytes = {name: (source / name).read_bytes() for name in old['files']}
    calls = []
    def smoke(root, settings, inputs):
        calls.append(root)
        assert root == view
        assert settings['launch_env']['LD_LIBRARY_PATH'].split(':')[0] == str((view / library).parent)
        assert not any(part.startswith(str(source) + '/') for part in settings['launch_env']['LD_LIBRARY_PATH'].split(':'))
        return {'passed': passed, 'python_import_executed': True, 'profile_key': profile.profile_key(settings),
                'build_inputs': inputs, 'stdout': 'new owned import', 'stderr': '', 'elapsed_seconds': .01}
    monkeypatch.setattr(cache, 'native_import_smoke', smoke)
    if not passed:
        with pytest.raises(ValueError, match='loader upgrade import smoke failed'):
            cache.prepare_native_view(view, source, old, args)
        assert not (view / '.mindie-runtime/ready-profile.json').exists()
    else:
        first = cache.prepare_native_view(view, source, old, args)
        current = {**first['manifest'], 'files': old['files']}
        proof = profile.native_compatibility_receipt(view, current)
        assert proof['origin']['smoke']['stdout'] == 'new owned import'
        # The new profile needs no more path additions. Its certificate is
        # applicable to the next owned view without an upgrade/import.
        assert profile.native_vendor_launch_environment(view, current['files'], current['profile']['launch_env']) == current['profile']['launch_env']
        assert proof['key'] == profile.native_compatibility_key(current)
        later = view.parent / 'next-execution'
        for name, package in (('vllm', 'vllm'), ('vllm-ascend', 'vllm_ascend')):
            target = later / name / package
            target.mkdir(parents=True)
            (target / '__init__.py').write_text("raise RuntimeError('normal hot view imported again')\n")
        following = {**args, 'root': str(later), 'source_root': str(view), 'source_id': 'next-inputs',
                     'donor_manifest_digest': profile.digest(current), 'build_inputs': current['build_inputs'],
                     'versions': {name: {'version': '2.2', 'source_head': 'next-head'} for name in args['versions']}}
        second = cache.prepare_native_view(later, view, current, following)
        next_profile = {**second['manifest'], 'files': current['files']}
        assert profile.native_compatibility_receipt(later, next_profile) == proof
    evidence = json.loads((view / '.mindie-runtime/profile-evidence/smoke.json').read_text())
    assert evidence['reason'] == 'native-loader-environment-upgrade'
    assert evidence['elapsed_seconds'] == .01 and evidence['passed'] is passed
    assert calls == [view]
    assert {name: (source / name).read_bytes() for name in old['files']} == original_bytes


@pytest.mark.parametrize('change', ['native', 'build-env', 'manifest', 'artifact', 'evidence'])
def test_failed_publication_never_marks_the_view_ready(donor, change):
    source, view, old, args, _ = donor
    if change == 'native':
        args['build_inputs'] = copy.deepcopy(args['build_inputs'])
        args['build_inputs']['vllm']['native'] = 'f' * 64
    elif change == 'build-env':
        args['build_env'] = {'CXX': 'other-compiler'}
    elif change == 'manifest':
        old['build_key'] = 'changed'
    else:
        path = (next(iter(old['files'])) if change == 'artifact' else old['evidence']['smoke']['path'])
        (source / path).write_bytes(b'changed after original proof')
    with pytest.raises(ValueError):
        cache.prepare_native_view(view, source, old, args)
    assert not (view / '.mindie-runtime/ready-profile.json').exists()


def test_actual_backend_script_publishes_once_and_launch_checks_only_mutable_facts(donor, monkeypatch, tmp_path):
    from mindie_coordinator import backend as adapters, parity_support
    from mindie_coordinator.parity_support import SshStreamingResult, RemoteCommandError
    source, view, old, args, _ = donor
    backend = adapters.RemoteBackend()
    previous = {'python': sys.executable, 'endpoint': {'root': str(source)},
                'attestation': {**old, 'container_id': 'same-container'}}
    spec = {'python': sys.executable, 'endpoint': {'host': 'local.invalid', 'port': 46001,
            'user': 'root', 'root': str(view), 'cwd': str(view)}, 'preparation': args['preparation'],
            'source_snapshot': {'id': args['source_id'], 'build_env': {}, 'records': [
                {'relpath': name, 'build_inputs': row} for name, row in args['build_inputs'].items()]}}
    calls = []

    def execute(script):
        result = subprocess.run(['bash', '-c', script], cwd=view, capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RemoteCommandError(result.returncode, result.stderr)
        return result

    def stream(endpoint, script, **kwargs):
        calls.append(script)
        result = execute(script)
        assert len(result.stdout) < 10000
        return SshStreamingResult(result.returncode, result.stdout, result.stderr, [])

    monkeypatch.setattr(parity_support, 'ssh_exec_stream', stream)
    attestation = backend._prepare_native_view(spec, previous, args['versions'])
    assert len(calls) == 1 and 'MINDIE_CAPTURE_PROBE' not in calls[0]
    assert attestation['files'] == old['files'] and attestation['container_id'] == 'same-container'
    # The real launch adapter must not run either whole-bundle verification or
    # native-input recapture. Fixed Git checks remain in its common runner.
    package_file = adapters._package_file
    guarded = {}
    for name, function in (('runtime_profile.py', 'verify'), ('build_inputs.py', 'runtime_build_inputs')):
        path = tmp_path / name
        path.write_text(package_file(name).read_text() + '\ndef ' + function + '(*a, **k):\n    raise AssertionError("unexpected full native recheck")\n')
        guarded[name] = path
    monkeypatch.setattr(adapters, '_package_file', lambda name: guarded.get(name) or package_file(name))
    runtime = {**spec, 'host_endpoint': {}, 'container_name': 'owned',
               'attestation': attestation, 'prepared_native_view': True}
    monkeypatch.setattr(backend, 'bash', lambda target, script: json.dumps(
        {'Id': 'same-container', 'State': {'Running': True}}) if script.startswith('docker inspect') else execute(script).stdout)
    snapshots = {}
    from test_compact_preflight import git
    for name in ('vllm', 'vllm-ascend'):
        repo = view / name
        git(repo, 'init')
        git(repo, 'config', 'user.name', 'Publication test')
        git(repo, 'config', 'user.email', 'publication@example.invalid')
        git(repo, 'add', '.')
        git(repo, 'commit', '-m', 'fixed Python view')
        snapshots[name] = git(repo, 'rev-parse', 'HEAD')
    assert backend.verify_preflight(runtime, snapshots=snapshots) is True
    (view / 'vllm/vllm/__init__.py').write_text('changed_after_admission = True\n')
    with pytest.raises(RemoteCommandError, match='source differs from pinned snapshot'):
        backend.verify_preflight(runtime, snapshots=snapshots)
    system = Path(attestation['profile']['system_files']['driver']['path'])
    system.write_text('changed driver')
    with pytest.raises(RemoteCommandError, match='support file changed'):
        backend.verify_preflight(runtime)


@pytest.mark.parametrize('mode', ['combined', 'oversized', 'uncertain', 'cancelled-before'])
def test_hot_preparation_has_one_publication_and_no_separate_capture(monkeypatch, mode):
    from mindie_coordinator import backend as adapters, parity, parity_support
    backend = adapters.RemoteBackend()
    seen = []
    published = {'completed': 'native-view'}
    from types import SimpleNamespace
    publication = SimpleNamespace(accept=lambda reply: published if reply == {'published': True} else pytest.fail('wrong native proof'))
    monkeypatch.setattr('mindie_coordinator.native_publication.NativeViewPublication', lambda *args: publication)
    snapshot = {'id': 'accepted', 'records': [
        {'relpath': name, 'scm_version': '2.1', 'source_head': 'accepted-head'}
        for name in ('vllm', 'vllm-ascend')]}
    spec = {'user': 'alice', 'python': '/donor/bin/python',
            'endpoint': {'host': 'local.invalid', 'port': 46001, 'user': 'root', 'root': '/execution'},
            'host_endpoint': {'host': 'local.invalid', 'port': 22, 'user': 'root'},
            'source_snapshot': snapshot}
    def materialize(**kwargs):
        from mindie_coordinator.preparation_process import PreparationProcess
        assert kwargs['source_snapshot'] is snapshot
        assert kwargs['endpoint'] is spec['endpoint']
        assert isinstance(kwargs['process'], PreparationProcess)
        assert kwargs['process'].setup[0][0] == 'prepare-root'
        assert kwargs['native_publication'] is publication
        seen.append('materialize')
        if mode == 'uncertain':
            from mindie_coordinator.preparation_process import PreparationUncertain
            raise PreparationUncertain('lost composite operation')
        return {'native_view': {'published': True}} if mode == 'combined' else {}
    monkeypatch.setattr(parity, 'materialize_fixed_sources', materialize)
    from remote_dev.core.ssh_transport import RemoteCompleted
    monkeypatch.setattr('remote_dev.core.ssh_transport.run_rpc_script',
                        lambda *a, **k: seen.append('prepare-root') or RemoteCompleted(0, '', ''))
    monkeypatch.setattr(adapters, 'native_compatibility_key', lambda manifest: 'existing-proof')
    monkeypatch.setattr(backend, '_prepare_native_view', lambda *a, **k: seen.append('publish') or published)
    monkeypatch.setattr(backend, 'bash', lambda *a, **k: pytest.fail('hot preparation has no extra shell probe'))
    monkeypatch.setattr(backend, '_write_ready_profile', lambda *a, **k: pytest.fail('hot preparation must not recapture'))
    monkeypatch.setattr(backend, '_shared_native', lambda *a, **k: pytest.fail('hot donor needs no shared cache lookup'))
    progress = []
    def prepare():
        return backend.prepare_task_root(spec, sources={'vllm': '/local/vllm', 'vllm-ascend': '/local/ascend'},
            environment={}, source_snapshot=snapshot, reuse={'kind': 'native', 'runtime': {'attestation': {}}},
            on_progress=lambda event: progress.append(event['step']), on_preparation_job=lambda *a, **k: None,
            cancel_requested=lambda: mode == 'cancelled-before')
    if mode == 'cancelled-before':
        from mindie_coordinator.preparation_process import PreparationCancelled
        with pytest.raises(PreparationCancelled):
            prepare()
        assert seen == [] and progress == []
        return
    if mode == 'uncertain':
        from mindie_coordinator.preparation_process import PreparationUncertain
        with pytest.raises(PreparationUncertain):
            prepare()
        assert seen == ['materialize']
        return
    result = prepare()
    assert isinstance(result, adapters.PreparedNativeView) and result.attestation is published
    assert seen == (['materialize'] if mode == 'combined' else ['materialize', 'publish'])
    assert progress == (['materialize'] if mode == 'combined' else ['materialize', 'publish-native-view'])


@pytest.mark.parametrize('uncertain', [False, True])
def test_preparation_handoff_binds_only_a_completed_result(tmp_path, monkeypatch, uncertain):
    from test_coordinator import Backend, RuntimePool
    from mindie_coordinator.backend import PreparedNativeView
    from mindie_coordinator.preparation_process import PreparationUncertain
    from mindie_coordinator.provision.task_environment import prepare_task_environment
    backend = Backend(tmp_path / 'host')
    pool = RuntimePool(tmp_path / 'pool', backend)
    session = pool.session_open('alice', 'execution', {})

    def prepare(spec, **kwargs):
        if uncertain:
            raise PreparationUncertain('lost publication reply')
        return PreparedNativeView({**backend.attestation, 'runtime_root': spec['endpoint']['root'],
            'container_id': 'container-a', 'execution_view': {'source_id': spec['source_snapshot']['id']}})

    monkeypatch.setattr(backend, 'prepare_task_root', prepare)
    donor = {'host': '192.0.2.1', 'host_endpoint': {'host': '192.0.2.1', 'port': 22, 'user': 'root'},
             'ssh_port': 46001, 'python': '/original/bin/python', 'container_name': 'mindie-alice'}
    kwargs = dict(user='alice', session_id='execution', role_name='default', environment={}, donor=donor,
                  sources={}, source_snapshot={'id': 'accepted', 'records': []}, checkout_session=session['id'])
    if uncertain:
        with pytest.raises(PreparationUncertain, match='lost publication reply'):
            prepare_task_environment(pool, **kwargs)
        assert pool.catalog() == []
    else:
        row = prepare_task_environment(pool, **kwargs)
        assert row['state'] == 'bound' and row['prepared_native_view']
        assert row['binding']['runtime_id'] == row['id']
    assert backend.calls == []


@pytest.fixture
def binding(tmp_path):
    from test_coordinator import Backend, RuntimePool, runtime_spec
    backend = Backend(tmp_path / 'host')
    pool = RuntimePool(tmp_path / 'pool', backend)
    session = pool.session_open('alice', 'prepared-view', {})
    spec = {**runtime_spec(1), 'source_snapshot': {'id': 'accepted'}}
    attestation = {**backend.attestation, 'runtime_root': spec['endpoint']['root'],
                   'container_id': 'container-a', 'execution_view': {'source_id': 'accepted'}}
    return backend, pool, session, spec, attestation


def test_prepared_binding_is_atomic_replayable_and_never_probes_or_reserves_again(binding):
    backend, pool, session, spec, attestation = binding
    with ThreadPoolExecutor(3) as workers:
        results = list(workers.map(lambda _: pool._bind_prepared(
            'view', spec, 'alice', session['id'], 'same', attestation), range(3)))
    assert len({row['binding']['id'] for row in results}) == 1
    assert all(row['state'] == 'bound' and row['prepared_native_view'] for row in results)
    assert backend.calls == []
    from test_coordinator import RuntimePool
    restarted = RuntimePool(pool.state_dir, backend)
    assert restarted._bind_prepared('view', spec, 'alice', session['id'], 'same', attestation) == results[0]
    assert backend.calls == []
    with pytest.raises(ValueError, match='differs from the already bound'):
        restarted._bind_prepared('view', spec, 'alice', session['id'], 'same', {**attestation, 'build_key': 'wrong'})


@pytest.mark.parametrize('change', ['source', 'root', 'container', 'owner'])
def test_foreign_or_incomplete_prepared_result_cannot_bind(binding, change):
    backend, pool, session, spec, attestation = binding
    if change == 'source':
        attestation['execution_view']['source_id'] = 'other-inputs'
    elif change == 'root':
        attestation['runtime_root'] = '/another/view'
    elif change == 'container':
        attestation.pop('container_id')
    with pytest.raises((ValueError, PermissionError)):
        pool._bind_prepared('view', spec, 'bob' if change == 'owner' else 'alice', session['id'], 'same', attestation)
    assert pool.catalog() == [] and backend.calls == []
