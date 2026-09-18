"""Import proof reuse excludes device kernels, never Python or CPU loaders."""
import copy
import json
import subprocess

import pytest

from test_native_compatibility_reuse import prepared
from mindie_coordinator import runtime_profile as profile
from mindie_coordinator.build_inputs import build_input_fingerprints, VLLM_ASCEND_REINSTALL_PATTERNS


def test_source_closure_changes_for_python_resources_and_added_files(tmp_path):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], text=True).strip()
    git('init', '-q')
    git('config', 'user.email', 'test@example.invalid')
    git('config', 'user.name', 'Test')
    (tmp_path / 'op.cpp').write_text('int op;')
    (tmp_path / 'module.py').write_text('value=1')
    def capture():
        git('add', '.')
        git('commit', '-qm', 'fixed input')
        return build_input_fingerprints(tmp_path, 'HEAD', VLLM_ASCEND_REINSTALL_PATTERNS, build_env={})
    first = capture()
    (tmp_path / 'op.cpp').write_text('int renamed;')
    kernel = capture()
    assert kernel['native'] != first['native'] and kernel['imports'] == first['imports']
    (tmp_path / 'module.py').write_text('value=2')
    python = capture()
    assert python['imports'] != kernel['imports']
    (tmp_path / 'config.json').write_text('{}')
    assert capture()['imports'] != python['imports']


@pytest.mark.parametrize('change', ['python', 'dependency', 'dependency-origin', 'extension', 'metadata', 'loader', 'kernel'])
def test_only_device_kernel_change_retains_import_proof(prepared, change):
    donor = copy.deepcopy(prepared['donor'])
    for row in donor['build_inputs'].values():
        row['imports'] = 'd' * 64
    payload = 'vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/test/op_impl/ai_core/tbe/kernel/ascend910b/op/a.o'
    donor['files'][payload] = {'sha256': '1' * 64, 'role': 'library'}
    current = copy.deepcopy(donor)
    if change == 'python':
        current['build_inputs']['vllm']['imports'] = 'e' * 64
    elif change == 'dependency':
        current['build_inputs']['vllm']['dependencies'] = 'e' * 64
    elif change == 'loader':
        current['profile']['launch_env']['LD_LIBRARY_PATH'] = '/unexpected/lib'
    elif change == 'dependency-origin':
        current['profile']['dependency_origins'] = {'torch': '/other/venv/site-packages'}
    elif change == 'kernel':
        current['build_inputs']['vllm-ascend']['native'] = 'e' * 64
        current['files'][payload]['sha256'] = '2' * 64
    else:
        name = next(name for name, value in current['files'].items()
                    if value['role'] == ('library' if change == 'extension' else 'metadata'))
        current['files'][name]['sha256'] = '2' * 64
    assert (profile.native_import_closure_key(current) == profile.native_import_closure_key(donor)) == (change == 'kernel')


def test_new_kernel_evidence_still_hashes_all_outputs(prepared):
    root = prepared['original']
    donor = copy.deepcopy(prepared['donor'])
    for row in donor['build_inputs'].values():
        row['imports'] = 'd' * 64
    donor['build_key'] = profile.build_key(donor['profile'], donor['build_inputs'])
    certificate = profile.native_import_receipt(root, donor)
    smoke = {'kind': 'native-import-closure-reuse', 'python_import_executed': False,
             'profile_key': donor['profile_key'], 'build_inputs': donor['build_inputs'],
             'compatibility': certificate}
    smoke_path = root / donor['evidence']['smoke']['path']
    smoke_path.write_text(json.dumps(smoke))
    donor['evidence']['smoke']['sha256'] = profile.file_digest(smoke_path)
    profile.verify(root, donor, check_environment=False)
    # Subsequent proof reuse remains flat and can also donate a native view.
    assert profile.native_import_receipt(root, donor) == certificate
    assert profile.native_compatibility_receipt(root, donor)['origin'] == certificate['origin']
    output = root / next(iter(donor['files']))
    output.write_bytes(b'changed after compilation')
    with pytest.raises(ValueError, match='artifact hash mismatch'):
        profile.verify(root, donor, check_environment=False)


def test_legacy_receipts_do_not_invent_a_source_import_proof(prepared):
    assert profile.native_import_receipt(prepared['original'], prepared['donor']) is None


@pytest.mark.skipif(__import__('sys').platform != 'linux', reason='actual capture runs under Linux')
@pytest.mark.parametrize('change', [False, True])
def test_actual_finalization_reuses_only_matching_import_closure(prepared, monkeypatch, change):
    import sys
    from mindie_coordinator.prepare_runtime import REMOTE_CAPTURE_SUFFIX
    root = prepared['view']
    for name in profile.LAUNCH_PATH_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('SOC_VERSION', 'test-soc')
    monkeypatch.setenv('CXX', 'test-compiler')
    settings = copy.deepcopy(prepared['settings'])
    settings['launch_env'] = {'SOC_VERSION': 'test-soc', 'PYTHONPATH': ':'.join(
        str(prepared['original'] / name) for name in ('.mindie-runtime/metadata', 'vllm', 'vllm-ascend'))}
    inputs = copy.deepcopy(prepared['inputs'])
    for row in inputs.values():
        row['imports'] = 'd' * 64
    donor = {**prepared['donor'], 'profile': settings, 'build_inputs': inputs}
    certificate = profile.native_import_receipt(prepared['original'], donor)
    (root / '.mindie-runtime/profile-evidence/import-closure.json').write_text(json.dumps(certificate))
    (root / '.mindie-runtime/reuse.json').write_text(json.dumps({'kind': 'shared-native',
        'import_evidence': '.mindie-runtime/profile-evidence/import-closure.json'}))
    inputs = copy.deepcopy(inputs)
    inputs['vllm-ascend']['native'] = 'e' * 64
    if change:
        inputs['vllm']['imports'] = 'f' * 64
    request = {'root': str(root), 'image_digest': 'sha256:fixture',
               'cann_files': [settings['system_files']['cann']['path']],
               'driver_files': [settings['system_files']['driver']['path']]}
    monkeypatch.setattr(sys, 'argv', ['capture', json.dumps(request)])
    calls = []
    def smoke(*args):
        calls.append(True)
        return {'passed': True, 'python_import_executed': True}
    namespace = {key: value for key, value in vars(profile).items() if not key.startswith('__')}
    namespace.update(native_import_smoke=smoke, installed_native_files=lambda root: prepared['files'],
                     _build_namespace={'runtime_build_inputs': lambda *args: inputs})
    exec(compile(REMOTE_CAPTURE_SUFFIX, '<actual-finalization>', 'exec'), namespace)
    manifest = json.loads((root / '.mindie-runtime/ready-profile.json').read_text())
    profile.verify(root, manifest)
    assert bool(calls) is change
