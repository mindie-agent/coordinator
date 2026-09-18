"""Shared outputs remain copies; execution roots, interpreters and users stay separate."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

import pytest

from mindie_coordinator import preparation_cache as cache, runtime_profile as profile
from mindie_coordinator.backend import RemoteBackend
from mindie_coordinator.provision.task_environment import reusable_preparation


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, 'native_import_receipt', profile.native_import_receipt, raising=False)
    monkeypatch.setattr(cache, 'native_import_receipt', profile.native_import_receipt, raising=False)
    source, target, shared = tmp_path / 'alice-build', tmp_path / 'bob-execution', tmp_path / 'host-cache'
    system = tmp_path / 'image-system'
    system.mkdir()
    settings = {field: '1.0' for field in profile.PROFILE_FIELDS}
    settings.update(image_digest='sha256:same-image', python_abi=sysconfig.get_config_var('SOABI'),
                    build_env={}, launch_env={}, compatibility_evidence='smoke.json', system_files={})
    for name in ('cann', 'driver'):
        path = system / name
        path.write_text('same image ' + name)
        settings['system_files'][name] = {'path': str(path), 'sha256': profile.file_digest(path)}
    for root in (source, target):
        for name in ('vllm', 'vllm-ascend'):
            package = root / name / name.replace('-', '_')
            package.mkdir(parents=True)
            (package / '__init__.py').write_text('value = ' + repr(root.name))
    extension = Path(importlib.util.find_spec('_ctypes_test').origin)
    relative = 'vllm-ascend/vllm_ascend/' + extension.name
    shutil.copy2(extension, source / relative)
    metadata = 'vllm-ascend/vllm_ascend/_build_info.py'
    (source / metadata).write_text('SOC_VERSION = "1.0"\n')
    for name in ('cann', 'driver'):
        (source / (name + '.txt')).write_text(name)
    (source / 'smoke.json').write_text(json.dumps({'passed': True}))
    request = reusable_preparation({'id': 'bob-source', 'records': [
        {'relpath': name, 'build_inputs': {'native': 'native-' + name, 'dependencies': 'deps-' + name}}
        for name in ('vllm', 'vllm-ascend')]}, {'recipe': 'test'}, {})
    manifest = profile.capture(source, settings, {'vllm': 'native-vllm', 'vllm-ascend': 'native-ascend'},
                               {relative: 'library', metadata: 'metadata'},
                               {'cann': 'cann.txt', 'driver': 'driver.txt', 'smoke': 'smoke.json'})
    manifest['preparation'] = profile.verified_preparation(request, settings)
    marker = source / '.mindie-runtime/ready-profile.json'
    marker.parent.mkdir()
    marker.write_text(json.dumps(manifest))
    class Distribution:
        version = '1.0'
        def __init__(self, name):
            self.name = name
        def read_text(self, name):
            return {'METADATA': f'Name: {self.name}\nVersion: 1.0\n',
                    'entry_points.txt': '[vllm.platform_plugins]\nnpu=vllm_ascend:register\n'}.get(name)
    monkeypatch.setattr(cache.importlib.metadata, 'distribution', Distribution)
    monkeypatch.setattr(cache.importlib.metadata, 'version', lambda _: '1.0')
    for name in ('digest', 'publish', 'verify', 'file_digest', 'checked_file', 'verified_preparation'):
        monkeypatch.setattr(cache, name, getattr(profile, name), raising=False)
    monkeypatch.setattr(cache, 'native_tree_entries', lambda *args: {}, raising=False)
    monkeypatch.setattr(cache, 'kernel_rebuild_plan', lambda *args: None, raising=False)
    monkeypatch.setattr(cache, 'VLLM_ASCEND_REINSTALL_PATTERNS', (), raising=False)
    monkeypatch.setattr(cache, 'submodule_content', None, raising=False)
    versions = {name: {'version': '2.0.dev1', 'source_head': 'bob-head'} for name in ('vllm', 'vllm-ascend')}
    return source, target, shared, request, manifest, relative, versions


def test_cross_user_reuse_loads_real_extension_without_donor_runtime(bundle):
    source, target, shared, request, manifest, relative, versions = bundle
    assert cache.store_shared_native(source, shared)['status'] == 'stored'
    shutil.rmtree(source)  # No donor interpreter, editable source or runtime is reused.
    assert cache.restore_shared_native(target, shared, request, 'sha256:same-image', versions)['status'] == 'hit'
    command = "import json,vllm,vllm_ascend._ctypes_test as extension;from vllm._version import __commit_id__;print(json.dumps({'source':vllm.value,'extension':extension.__file__,'head':__commit_id__}))"
    result = subprocess.run([sys.executable, '-c', command], capture_output=True, text=True,
                            env={**os.environ, 'PYTHONPATH': os.pathsep.join(map(str, [target / '.mindie-runtime/metadata',
                                                                                target / 'vllm', target / 'vllm-ascend']))})
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout)
    assert loaded['source'] == 'bob-execution' and loaded['head'] == 'bob-head'
    assert Path(loaded['extension']).is_relative_to(target)
    original = next((shared / 'bundles').iterdir()) / relative
    expected = original.read_bytes()
    (target / relative).write_bytes(b'local user edit')
    assert original.read_bytes() == expected


def test_explicit_reference_reuses_same_measured_image_with_current_request_identity(bundle):
    source, target, shared, request, manifest, relative, versions = bundle
    cache.store_shared_native(source, shared)
    request = copy.deepcopy(request)
    request['environment']['environment'] = {'image': 'registry/image@sha256:reference',
                                           'cann': '1.0', 'soc': '1.0',
                                           'python_abi': sysconfig.get_config_var('SOABI')}
    result = cache.restore_shared_native(target, shared, request, 'sha256:same-image', versions)
    assert result['status'] == 'hit'
    assert result['native_key'] == profile.verified_preparation(request, manifest['profile'])['native_key']
    assert result['native_key'] != manifest['preparation']['native_key']


@pytest.mark.parametrize('home,expected', [('/image/atb/cxx_abi_1', '1'), ('/image/atb/unknown', None)])
def test_verified_atb_selection_is_carried_without_guessing_abi(bundle, home, expected):
    source, target, shared, request, manifest, relative, versions = bundle
    manifest['profile']['launch_env']['ATB_HOME_PATH'] = home
    manifest['profile_key'] = profile.profile_key(manifest['profile'])
    manifest['build_key'] = profile.build_key(manifest['profile'], manifest['build_inputs'])
    (source / '.mindie-runtime/ready-profile.json').write_text(json.dumps(manifest))
    cache.store_shared_native(source, shared)
    assert cache.restore_shared_native(target, shared, request, 'sha256:same-image', versions)['status'] == 'hit'
    receipt = json.loads((target / '.mindie-runtime/reuse.json').read_text())
    assert receipt.get('atb_abi', {}).get('cxx_abi') == expected
    if expected:
        assert receipt['atb_abi']['torch'] == manifest['profile']['torch']
        assert receipt['atb_abi']['python_abi'] == manifest['profile']['python_abi']


@pytest.mark.parametrize('constraint', ['soc', 'cann', 'python_abi', 'machine_type'])
def test_coarse_image_lookup_still_checks_explicit_measured_constraints(bundle, constraint):
    source, target, shared, request, manifest, relative, versions = bundle
    cache.store_shared_native(source, shared)
    request = copy.deepcopy(request)
    request['environment']['environment'][constraint] = 'incompatible'
    with pytest.raises(ValueError, match='requested ' + constraint):
        cache.restore_shared_native(target, shared, request, 'sha256:same-image', versions)
    assert not (target / relative).exists()


@pytest.mark.parametrize('fact', ['soc', 'machine_type'])
def test_shared_lookup_rejects_recipient_hardware_mismatch(bundle, monkeypatch, fact):
    source, target, shared, request, manifest, relative, versions = bundle
    cache.store_shared_native(source, shared)
    if fact == 'soc':
        monkeypatch.setenv('SOC_VERSION', 'different')
    with pytest.raises(ValueError, match='recipient'):
        cache.restore_shared_native(target, shared, request, 'sha256:same-image', versions,
                                    machine_type='different' if fact == 'machine_type' else None)
    assert not (target / relative).exists()


@pytest.mark.parametrize('requirement,installed,satisfied', [
    ('numpy>=2', '2.1+image', True), ('numpy>=2', '1.9', False),
    ('missing>=1', None, False), ('broken=version', '1.0', False),
    ('torch-npu==2.10.0.post4', '2.10.0.post4.dev20260715', True),
    ('missing>=1; python_version < "2"', None, True),
])
def test_restore_dependency_check_uses_actual_recipient_metadata(monkeypatch, requirement, installed, satisfied):
    def version(name):
        if installed is None:
            raise cache.importlib.metadata.PackageNotFoundError(name)
        return installed
    monkeypatch.setattr(cache.importlib.metadata, 'version', version)
    distributions = {'vllm-ascend': {'files': {'METADATA': 'Name: vllm-ascend\nRequires-Dist: ' + requirement + '\n'}}}
    result = cache.recipient_dependencies(distributions, {'torch_npu': installed})
    assert result['satisfied'] is satisfied
    assert result['interpreter'] == sys.executable and result['prefix'] == sys.prefix
    assert result['purelib'] == sysconfig.get_paths()['purelib']
    assert bool(result['errors']) is not satisfied


@pytest.mark.parametrize('selected', [False, True])
@pytest.mark.parametrize('change', ['native', 'dependencies', 'image', 'abi', 'cann', 'corrupt'])
def test_mismatches_do_not_copy_artifacts(bundle, monkeypatch, change, selected):
    source, target, shared, request, manifest, relative, versions = bundle
    candidate = cache.store_shared_native(source, shared)
    image = 'sha256:same-image'
    if change in ('native', 'dependencies'):
        request = copy.deepcopy(request)
        request[change]['vllm'] = 'different-input'
    elif change == 'image':
        image = 'sha256:another-image'
    elif change == 'abi':
        monkeypatch.setattr(cache.sysconfig, 'get_config_var', lambda _: 'different-abi')
    elif change == 'cann':
        Path(manifest['profile']['system_files']['cann']['path']).write_text('CANN changed')
    else:
        (next((shared / 'bundles').iterdir()) / relative).write_bytes(b'changed output')
    try:
        result = cache.restore_shared_native(target, shared, request, image, versions,
                                             candidate=candidate if selected else None)
    except ValueError:
        pass
    else:
        assert result['status'] == 'miss'
    assert not (target / relative).exists()


def test_exported_bundle_restore_ignores_later_shared_index_writes(bundle):
    source, target, shared, request, manifest, relative, versions = bundle
    candidate = cache.store_shared_native(source, shared)
    # A concurrent donor publishes a different index. The current selection
    # still restores exactly the exported, independently verified bundle.
    for index in shared.glob('*.json'):
        index.write_text(json.dumps({'bundle': 'f' * 64, 'native_key': 'different'}))
    result = cache.restore_shared_native(target, shared, request, 'sha256:same-image', versions, candidate=candidate)
    assert result['status'] == 'hit' and result['bundle'] == candidate['bundle']
    assert (target / relative).read_bytes() == (source / relative).read_bytes()


def test_cached_outputs_can_be_discarded_before_normal_build(bundle):
    source, target, shared, request, manifest, relative, versions = bundle
    cache.store_shared_native(source, shared)
    cache.restore_shared_native(target, shared, request, 'sha256:same-image', versions)
    cache.discard_shared_native(target)
    assert not (target / relative).exists()
    assert not (target / '.mindie-runtime/metadata').exists()
    assert (target / 'vllm/vllm/__init__.py').read_text() == "value = 'bob-execution'"
    assert (next((shared / 'bundles').iterdir()) / relative).is_file()


@pytest.mark.parametrize('failure', [None, 'profile', 'cache-miss', 'missing-dependency', 'broken-dependency', 'repaired-abi',
                                    'cancel-profile', 'uncertain-profile', 'cancel-revalidate', 'uncertain-revalidate'])
def test_normal_preparation_uses_cache_and_rebuilds_failed_hit(tmp_path, monkeypatch, failure):
    import mindie_coordinator.parity as parity
    import mindie_coordinator.parity_support as transport
    from remote_dev.core.ssh_transport import RemoteCompleted
    from mindie_coordinator.preparation_process import PreparationCancelled, PreparationUncertain
    backend = RemoteBackend()
    spec = {'user': 'bob', 'container_name': 'mindie-bob', 'python': '/bob/.venv/bin/python',
            'endpoint': {'host': 'host', 'port': 2202, 'user': 'root', 'root': '/bob'},
            'host_endpoint': {'host': 'host', 'port': 22, 'user': 'root'}}
    snapshot = {'id': 'source', 'records': [{'relpath': name, 'scm_version': '1.0', 'source_head': 'head'}
                                          for name in ('vllm', 'vllm-ascend')]}
    monkeypatch.setattr(parity, 'materialize_fixed_sources', lambda **kwargs: None)
    monkeypatch.setattr('remote_dev.core.ssh_transport.run_rpc_script',
                        lambda *a, **k: RemoteCompleted(0, '', ''))
    monkeypatch.setattr(transport, 'ssh_exec_stream', lambda *a, **k: None)
    monkeypatch.setattr(backend, 'bash', lambda *a: '')
    monkeypatch.setattr(backend, 'inspect', lambda *a, **k: pytest.fail('registration owns the full attestation'))
    events, installed, operations = [], [], []
    failed = False
    def install(**kwargs):
        nonlocal failed
        step = kwargs['step']
        installed.append(step)
        if step == 'verify-deps' and failure == 'broken-dependency':
            raise RuntimeError('declared dependency cannot be satisfied')
    def write_profile(*args, **kwargs):
        nonlocal failed
        if failure in ('cancel-profile', 'uncertain-profile'):
            raise (PreparationCancelled if failure.startswith('cancel') else PreparationUncertain)('owned process interrupted')
        if failure == 'profile' and not failed:
            failed = True
            raise RuntimeError('cached profile import does not load')
    def operation(spec, action, versions, **kwargs):
        operations.append(action)
        if action == 'revalidate' and failure in ('cancel-revalidate', 'uncertain-revalidate'):
            raise (PreparationCancelled if failure.startswith('cancel') else PreparationUncertain)('owned process interrupted')
        return {'status': {'restore': 'miss' if failure == 'cache-miss' else 'hit',
                           'store': 'miss', 'discard': 'discarded',
                           'revalidate': 'miss' if failure == 'repaired-abi' else 'validated'}[action],
                'reason': 'fixture cache unavailable',
                'dependencies': {'satisfied': failure not in {'missing-dependency', 'broken-dependency', 'repaired-abi',
                                                              'cancel-revalidate', 'uncertain-revalidate'}}}
    monkeypatch.setattr(parity, 'run_runtime_install_step', install)
    monkeypatch.setattr(backend, '_write_ready_profile', write_profile)
    monkeypatch.setattr(backend, '_shared_native', operation)
    monkeypatch.setattr(backend, '_export_shared_native', lambda spec: {'status': 'miss'})
    if failure == 'broken-dependency':
        with pytest.raises(RuntimeError, match='dependency cannot be satisfied'):
            backend.prepare_task_root(spec, sources={'vllm': '/a', 'vllm-ascend': '/b'}, environment={},
                                      source_snapshot=snapshot, on_progress=events.append)
        assert installed == ['install-vllm-ascend-requirements', 'verify-deps']
        assert operations == ['restore']
        return
    if failure and failure.startswith(('cancel-', 'uncertain-')):
        with pytest.raises(PreparationCancelled if failure.startswith('cancel') else PreparationUncertain):
            backend.prepare_task_root(spec, sources={'vllm': '/a', 'vllm-ascend': '/b'}, environment={},
                                      source_snapshot=snapshot, on_progress=events.append)
        assert operations == (['restore', 'revalidate'] if failure.endswith('-revalidate') else ['restore'])
        assert 'install-vllm-ascend' not in installed
        return
    assert backend.prepare_task_root(spec, sources={'vllm': '/a', 'vllm-ascend': '/b'}, environment={},
                                     source_snapshot=snapshot, on_progress=events.append) is None
    assert installed.count('install-vllm-ascend') == int(failure in {'profile', 'cache-miss', 'repaired-abi'})
    assert installed.count('install-vllm-ascend-requirements') == int(failure in {'cache-miss', 'missing-dependency', 'repaired-abi'})
    if failure is None:
        assert installed == []
    if failure == 'missing-dependency':
        assert installed == ['install-vllm-ascend-requirements', 'verify-deps']
    assert 'verify-imports' not in installed and 'write-marker' not in installed
    assert operations == (['restore', 'discard'] if failure == 'profile' else ['restore']
                          if failure == 'cache-miss' else ['restore', 'revalidate', 'discard']
                          if failure == 'repaired-abi' else ['restore', 'revalidate']
                          if failure == 'missing-dependency' else ['restore'])
    assert spec['python'] == '/bob/.venv/bin/python'


@pytest.mark.parametrize('change', [None, 'torch', 'torch-npu', 'python', 'cann', 'manifest', 'index'])
def test_dependency_repair_rechecks_fixed_bundle_and_preserves_native_proof(bundle, monkeypatch, change):
    source, target, shared, request, manifest, relative, versions = bundle
    cache.store_shared_native(source, shared)
    restored = cache.restore_shared_native(target, shared, request, 'sha256:same-image', versions)
    receipt = target / '.mindie-runtime/reuse.json'
    receipt.write_text(json.dumps({'kind': 'dependencies', 'copied_packages': ['torch']}))
    original = (source / relative).read_bytes()
    if change in {'torch', 'torch-npu'}:
        monkeypatch.setattr(cache.importlib.metadata, 'version', lambda name: '2.0' if name == change else '1.0')
    elif change == 'python':
        monkeypatch.setattr(cache.sysconfig, 'get_config_var', lambda _: 'different-abi')
    elif change == 'cann':
        Path(manifest['profile']['system_files']['cann']['path']).write_text('changed during repair')
    elif change == 'manifest':
        path = shared / 'bundles' / restored['bundle'] / 'manifest.json'
        path.write_text(path.read_text() + '\n')
    elif change == 'index':
        # A newer publisher can update discovery indexes, not this selection.
        for path in shared.glob('*.json'):
            path.write_text(json.dumps({'bundle': 'f' * 64}))
    if change not in {None, 'index'}:
        with pytest.raises(ValueError, match='ABI|support|manifest changed'):
            cache.revalidate_shared_native(target, shared)
        assert json.loads(receipt.read_text())['kind'] == 'dependencies'
    else:
        assert cache.revalidate_shared_native(target, shared) == {'status': 'validated', 'bundle': restored['bundle']}
        proof = json.loads(receipt.read_text())
        assert proof == {'kind': 'shared-native', 'native_key': restored['native_key'],
                         'soc': manifest['profile']['soc'], 'compiler': manifest['profile']['compiler']}
    assert (source / relative).read_bytes() == original
    assert (target / relative).read_bytes() == original


def test_known_host_weight_mounts_keep_original_paths():
    from mindie_coordinator.provision import host_ops
    probe = host_ops.render_host_probe_script()
    bootstrap = host_ops.render_bootstrap_host_script()
    assert '["/home", "/tmp", "/weight", "/weights", "/models", "/data", "/mnt"]' in probe
    assert 'for optional in /home /tmp /weight /weights /models /data /mnt; do' in bootstrap
    assert 'mount_args+=("-v" "$optional:$optional")' in bootstrap
    assert 'if [ -e "$optional" ]; then' in bootstrap


@pytest.mark.parametrize('action', ['restore', 'store', 'discard'])
def test_cache_metadata_steps_do_not_activate_cann_or_atb(monkeypatch, action):
    backend = RemoteBackend()
    commands = []
    def bash(endpoint, command):
        commands.append(command)
        return ('{"Id":"fixed-container","Image":"sha256:image","State":{"Running":true}}'
                if command.startswith('docker inspect') else '{"status":"fixture"}')
    monkeypatch.setattr(backend, 'bash', bash)
    spec = {'endpoint': {'root': '/execution'}, 'host_endpoint': {},
            'container_name': 'mindie-fixture', 'python': '/execution/.venv/bin/python'}
    assert backend._shared_native(spec, action, {})['status'] == 'fixture'
    assert 'safe_source()' not in commands[-1]
    assert '/nnal/atb/set_env.sh' not in commands[-1]
    assert 'readlink -f "$PYTHON"' in commands[-1]


def test_marker_and_venv_do_not_activate_native_runtime():
    from mindie_coordinator.parity import runtime_install_step_script
    from mindie_coordinator.provision.task_environment import create_venv_script
    marker = runtime_install_step_script(runtime_root='/execution', marker_dirname='.runtime',
                                         container_identity='mindie-fixture', step='write-marker', python='/unused-python')
    venv = create_venv_script('/execution', '/execution/.venv/bin/python')
    assert '/unused-python' not in marker
    assert 'safe_source()' not in marker and 'safe_source()' not in venv
    assert '/nnal/atb/set_env.sh' not in marker and '/nnal/atb/set_env.sh' not in venv


@pytest.mark.skipif(os.name == 'nt', reason='generated shell payload executes in the Linux recipient')
def test_actual_verification_payload_reads_execution_source_and_metadata(tmp_path):
    from mindie_coordinator.parity import runtime_install_step_script
    root, image = tmp_path / 'execution', tmp_path / 'image-python'
    image.mkdir()
    for name, body in {'torch': '__version__ = "1.0"', 'torch_npu': '',
                       'vllm': 'raise RuntimeError("loaded old image vllm")',
                       'vllm_ascend': 'raise RuntimeError("loaded old image ascend")'}.items():
        (image / (name + '.py')).write_text(body)
    for name in ('vllm', 'vllm-ascend'):
        package = root / name / name.replace('-', '_')
        package.mkdir(parents=True)
        (package / '__init__.py').write_text('__version__ = "current-source"\n')
    marker = root / '.mindie-runtime/shared-native.json'
    marker.parent.mkdir()
    marker.write_text('{}')
    for base, requires in ((root / '.mindie-runtime/metadata', ''),
                           (image, 'Requires-Dist: definitely-missing-fixture-package>=999\n')):
        metadata = base / 'vllm_ascend-1.0.dist-info/METADATA'
        metadata.parent.mkdir(parents=True)
        metadata.write_text('Name: vllm-ascend\nVersion: 1.0\n' + requires)
    for step in ('verify-imports', 'verify-deps'):
        script = runtime_install_step_script(runtime_root=str(root), marker_dirname='.runtime',
                                              container_identity='mindie-bob', step=step, python=sys.executable)
        result = subprocess.run(['bash', '-c', script], capture_output=True, text=True,
                                env={**os.environ, 'PYTHONPATH': str(image)}, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        assert ('current-source' if step == 'verify-imports' else 'dependency-check=ok') in result.stdout


def test_shared_baseline_recognizes_kernel_delta_and_keeps_its_interpreter(bundle, monkeypatch):
    from mindie_coordinator import native_incremental as native
    source, target, shared, request, manifest, relative, versions = bundle
    old, new = b'int result = 1;\n', b'int result = 2;\n'
    def blob(data):
        import hashlib
        return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    path = 'csrc/moe/add_rms_norm_bias/op_kernel/add_rms_norm_bias.cpp'
    prefix = 'vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe'
    installed = prefix + '/custom_transformer_impl/ascendc/add_rms_norm_bias/add_rms_norm_bias.cpp'
    config = prefix + '/kernel/config/ascend910_93/add_rms_norm_bias.json'
    for root, name, data in ((source, installed, old), (source, config, b'{}'),
                             (target, 'vllm-ascend/' + path, new)):
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)
    for name in (installed, config):
        manifest['files'][name] = {'sha256': profile.file_digest(source / name), 'role': 'metadata'}
    current = {path: ('100644', blob(new))}
    request['native']['vllm-ascend'] = native.native_tree_digest({path: ('100644', blob(old))})
    manifest['preparation'] = profile.verified_preparation(request, manifest['profile'])
    (source / '.mindie-runtime/ready-profile.json').write_text(json.dumps(manifest))
    cache.store_shared_native(source, shared)
    request['native']['vllm-ascend'] = native.native_tree_digest(current)
    monkeypatch.setattr(cache, 'native_tree_entries', lambda *args: current)
    monkeypatch.setattr(cache, 'kernel_rebuild_plan', native.kernel_rebuild_plan)
    result = cache.restore_shared_native(target, shared, request, 'sha256:same-image', versions)
    assert result['status'] == 'incremental' and result['operator'] == 'add_rms_norm_bias'
    assert (target / relative).read_bytes() == (source / relative).read_bytes()
    plan = json.loads((target / '.mindie-runtime/native-incremental.json').read_text())
    assert plan['source'] == path
    assert plan['native_to'] != plan['native_from']
    # No compiled result is published by restoration alone.
    assert not (target / '.mindie-runtime/ready-profile.json').exists()


@pytest.mark.parametrize('failure', [None, 'build', 'imports', 'profile', 'export', 'later-export',
                                    'exhausted', 'cancel-export-restore', 'uncertain-export-restore'])
@pytest.mark.parametrize('dependency_donor', [False, True])
def test_incremental_recipe_never_silently_falls_back_to_full_build(tmp_path, monkeypatch, failure, dependency_donor):
    import mindie_coordinator.parity as parity
    import mindie_coordinator.parity_support as transport
    from remote_dev.core.ssh_transport import RemoteCompleted
    from mindie_coordinator.preparation_process import PreparationCancelled, PreparationUncertain
    backend = RemoteBackend()
    spec = {'user': 'bob', 'container_name': 'mindie-bob', 'python': '/bob/.venv/bin/python',
            'endpoint': {'host': 'host', 'port': 2202, 'user': 'root', 'root': '/bob'},
            'host_endpoint': {'host': 'host', 'port': 22, 'user': 'root'}}
    snapshot = {'id': 'source', 'records': [{'relpath': name, 'scm_version': '1.0', 'source_head': 'head'}
                                          for name in ('vllm', 'vllm-ascend')]}
    monkeypatch.setattr(parity, 'materialize_fixed_sources', lambda **kwargs: None)
    monkeypatch.setattr('remote_dev.core.ssh_transport.run_rpc_script', lambda *a, **k: RemoteCompleted(0, '', ''))
    monkeypatch.setattr(transport, 'ssh_exec_stream', lambda *a, **k: None)
    monkeypatch.setattr(backend, 'bash', lambda *a: '')
    installed, operations, process_steps, exports = [], [], [], []
    def install(**kwargs):
        installed.append(kwargs['step'])
        if failure == 'build' and kwargs['step'] == 'install-vllm-ascend-incremental':
            raise RuntimeError('actual incremental failure')
    def capture_profile(*args, **kwargs):
        if failure in {'profile', 'imports'}:
            raise RuntimeError('actual incremental failure')
    def operation(spec, action, versions, **kwargs):
        operations.append(action)
        process_steps.append(kwargs['process'].step)
        if failure in {'export', 'later-export', 'exhausted', 'cancel-export-restore', 'uncertain-export-restore'} and len(operations) == 1:
            return {'status': 'miss'}
        if action == 'restore' and exports:
            if kwargs['process'].step == 'shared-native-restore-after-export':
                assert kwargs['candidate'] == {'bundle': str(len(exports)), 'native_key': 'selected'}
            if failure in {'cancel-export-restore', 'uncertain-export-restore'}:
                raise (PreparationCancelled if failure.startswith('cancel') else PreparationUncertain)('interrupted')
            if failure == 'exhausted' or (failure == 'later-export' and len(exports) == 1):
                return {'status': 'miss', 'reason': 'base changes cannot be rebuilt incrementally'}
        return {'status': 'incremental' if action == 'restore' else 'stored', 'dependencies': {'satisfied': True}}
    def export(spec, **kwargs):
        assert failure in {'export', 'later-export', 'exhausted', 'cancel-export-restore', 'uncertain-export-restore'}
        assert kwargs == ({'donors': ['remaining']} if exports else {})
        exports.append(kwargs)
        assert len(exports) <= 2
        return {'status': 'stored', 'bundle': str(len(exports)), 'native_key': 'selected',
                'remaining_donors': ['remaining'] if len(exports) == 1 and failure != 'export' else []}
    monkeypatch.setattr(parity, 'run_runtime_install_step', install)
    monkeypatch.setattr(backend, '_write_ready_profile', capture_profile)
    monkeypatch.setattr(backend, '_shared_native', operation)
    monkeypatch.setattr(backend, '_export_shared_native', export)
    def prepare():
        return backend.prepare_task_root(spec, sources={'vllm': '/a', 'vllm-ascend': '/b'}, environment={},
                                         source_snapshot=snapshot, on_preparation_job=lambda record: None,
                                         reuse={'kind': 'dependencies', 'runtime': {
                                             'endpoint': {'root': '/dependency-donor'},
                                             'attestation': {'profile': {**dict.fromkeys(profile.PROFILE_FIELDS, '1.0'),
                                                 'build_env': {}, 'launch_env': {}, 'compatibility_evidence': 'smoke.json',
                                                 'system_files': {key: {'path': '/' + key, 'sha256': '1' * 64}
                                                                  for key in ('cann', 'driver')}}},
                                             'python': '/dependency-donor/.venv/bin/python'}} if dependency_donor else None)
    if failure in {'cancel-export-restore', 'uncertain-export-restore'}:
        with pytest.raises(PreparationCancelled if failure.startswith('cancel') else PreparationUncertain):
            prepare()
        assert len(exports) == 1 and operations == ['restore', 'restore']
        assert not installed
        return
    if failure == 'exhausted':
        assert prepare() is None
        assert len(exports) == 2
        assert installed.count('install-vllm-ascend') == 1
        assert 'install-vllm-ascend-incremental' not in installed
        return
    if failure in {'build', 'imports', 'profile'}:
        with pytest.raises(RuntimeError, match='actual incremental failure'):
            prepare()
        assert operations == ['restore']
    else:
        assert prepare() is None
        assert operations == (['restore', 'restore', 'restore'] if failure == 'later-export' else
                              ['restore', 'restore'] if failure == 'export' else ['restore'])
        if failure == 'export':
            assert process_steps == ['shared-native-restore', 'shared-native-restore-after-export']
    assert 'install-vllm-ascend' not in installed
    assert 'install-vllm' not in installed and 'install-vllm-ascend-requirements' not in installed
    assert 'verify-imports' not in installed and 'verify-deps' not in installed
    assert installed.count('install-vllm-ascend-incremental') == 1
