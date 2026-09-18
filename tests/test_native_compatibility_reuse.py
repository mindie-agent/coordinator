"""Native proof reuse never claims that changed Python code was imported."""
import copy
import importlib.machinery
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from unittest.mock import Mock

import pytest

from mindie_coordinator import runtime_profile as profile
from mindie_coordinator.prepare_runtime import REMOTE_CAPTURE_SUFFIX


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(profile, 'installed_dependency_identity', lambda: {})
    original, view = tmp_path / 'original', tmp_path / 'view'
    extension = 'vllm-ascend/vllm_ascend/vllm_ascend_C' + importlib.machinery.EXTENSION_SUFFIXES[0]
    files = {extension: 'library', 'vllm-ascend/vllm_ascend/_cann_ops_custom/config.json': 'metadata'}
    for root in (original, view):
        for name in files:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'native fixture')
        for source, package in (('vllm', 'vllm'), ('vllm-ascend', 'vllm_ascend')):
            path = root / source / package
            path.mkdir(parents=True, exist_ok=True)
            (path / '__init__.py').write_text("raise RuntimeError('candidate Python failure')\n")
            dist = root / '.mindie-runtime/metadata' / (package + '-2.0.dist-info')
            dist.mkdir(parents=True)
            (dist / 'METADATA').write_text('Name: ' + source + '\nVersion: 2.0\n')
        evidence = root / '.mindie-runtime/profile-evidence'
        evidence.mkdir()
        for name in ('cann', 'driver'):
            (evidence / (name + '.json')).write_text('{}')
    settings = {name: '1.0' for name in profile.PROFILE_FIELDS}
    settings.update(image_digest='sha256:fixture', soc='test-soc', compiler='test-compiler',
                    python_abi=sysconfig.get_config_var('SOABI'), vllm='2.0', vllm_ascend='2.0',
                    build_env={}, launch_env={}, compatibility_evidence='.mindie-runtime/profile-evidence/smoke.json',
                    system_files={})
    for name in ('cann', 'driver'):
        system = tmp_path / name
        system.write_text('fixture-' + name)
        settings[name] = system.read_text()
        settings['system_files'][name] = {'path': str(system), 'sha256': profile.file_digest(system)}
    inputs = {name: {'native': 'a' * 64, 'dependencies': 'b' * 64, 'build_env': 'c' * 64}
              for name in ('vllm', 'vllm-ascend')}
    evidence = {name: '.mindie-runtime/profile-evidence/' + name + '.json' for name in ('cann', 'driver', 'smoke')}
    original_smoke = {'passed': True, 'profile_key': profile.profile_key(settings), 'stdout': 'original import output'}
    (original / evidence['smoke']).write_text(json.dumps(original_smoke))
    donor = profile.capture(original, settings, inputs, files, evidence)
    profile.verify(original, donor, check_environment=False)
    certificate = profile.native_compatibility_receipt(original, donor)
    for package in ('vllm', 'vllm_ascend'):
        monkeypatch.delitem(sys.modules, package, raising=False)
    for path in (view / 'vllm', view / 'vllm-ascend', view / '.mindie-runtime/metadata'):
        monkeypatch.syspath_prepend(str(path))
    version = profile.importlib.metadata.version
    monkeypatch.setattr(profile.importlib.metadata, 'version',
                        lambda name: version(name) if name in ('vllm', 'vllm-ascend') else '1.0')

    def capture(settings_override=None, inputs_override=None, certificate_override=None):
        selected = copy.deepcopy(settings_override or settings)
        selected_inputs = copy.deepcopy(inputs_override or inputs)
        smoke = {'kind': 'native-compatibility-reuse', 'python_import_executed': False,
                 'profile_key': profile.profile_key(selected), 'build_inputs': selected_inputs,
                 'compatibility': copy.deepcopy(certificate_override or certificate),
                 'source_mapping': profile.native_source_mapping(view)}
        (view / evidence['smoke']).write_text(json.dumps(smoke))
        return profile.capture(view, selected, selected_inputs, files, evidence), smoke

    return dict(original=original, view=view, donor=donor, settings=settings, inputs=inputs,
                files=files, evidence=evidence, certificate=certificate, capture=capture)


def test_current_mapping_is_verified_but_python_execution_can_still_fail(prepared):
    manifest, smoke = prepared['capture']()
    profile.verify(prepared['view'], manifest)
    assert 'passed' not in smoke and smoke['python_import_executed'] is False
    assert smoke['compatibility']['origin']['smoke']['stdout'] == 'original import output'
    assert profile.native_compatibility_receipt(prepared['view'], manifest) == prepared['certificate']
    result = subprocess.run([sys.executable, '-c', 'import vllm'], capture_output=True, text=True,
                            env={**os.environ, 'PYTHONPATH': str(prepared['view'] / 'vllm')})
    assert result.returncode != 0 and 'candidate Python failure' in result.stderr


@pytest.mark.parametrize('change', ['native', 'dependencies', 'torch', 'build_env', 'loader', 'other-view-loader', 'artifact'])
def test_reused_proof_rejects_changes_even_if_new_manifest_is_recaptured(prepared, change):
    settings, inputs = copy.deepcopy(prepared['settings']), copy.deepcopy(prepared['inputs'])
    if change in ('native', 'dependencies'):
        inputs['vllm-ascend'][change] = 'f' * 64
    elif change == 'torch':
        settings['torch'] = 'different'
    elif change == 'build_env':
        settings['build_env'] = {'SOC_VERSION': 'other-chip'}
    elif change in ('loader', 'other-view-loader'):
        settings['launch_env']['LD_LIBRARY_PATH'] = ('/unverified/library' if change == 'loader'
                                                     else '/vllm-workspace/executions/other/unsafe-library')
    else:
        (prepared['view'] / next(iter(prepared['files']))).write_bytes(b'replaced native bytes')
    manifest, _ = prepared['capture'](settings, inputs)
    with pytest.raises(ValueError, match='native compatibility'):
        profile.verify(prepared['view'], manifest, check_environment=False)


def test_view_mapping_rejects_foreign_sources_and_changed_scm(prepared, monkeypatch, tmp_path):
    manifest, _ = prepared['capture']()
    metadata = next((prepared['view'] / '.mindie-runtime/metadata').glob('vllm-*/METADATA'))
    metadata.write_text('Name: vllm\nVersion: 3.0\n')
    with pytest.raises(ValueError, match='SCM metadata mapping changed'):
        profile.verify(prepared['view'], manifest)
    foreign = tmp_path / 'foreign'
    (foreign / 'vllm').mkdir(parents=True)
    (foreign / 'vllm/__init__.py').write_text('value = 1')
    monkeypatch.syspath_prepend(str(foreign))
    with pytest.raises(ValueError, match='module mapping escaped'):
        profile.native_source_mapping(prepared['view'])


def test_original_receipt_is_not_restamped_or_read_after_unverified_change(prepared):
    certificate = copy.deepcopy(prepared['certificate'])
    certificate['origin']['smoke']['passed'] = False
    manifest, _ = prepared['capture'](certificate_override=certificate)
    with pytest.raises(ValueError, match='native compatibility'):
        profile.verify(prepared['view'], manifest, check_environment=False)
    original = prepared['original'] / prepared['evidence']['smoke']
    original.write_text(json.dumps({'passed': True, 'stdout': 'changed after verify'}))
    with pytest.raises(ValueError, match='evidence changed after verification'):
        profile.native_compatibility_receipt(prepared['original'], prepared['donor'])


@pytest.mark.parametrize('field', ['profile_key', 'build_inputs'])
def test_reused_receipt_requires_current_view_identity(prepared, field):
    manifest, smoke = prepared['capture']()
    smoke.pop(field)
    with pytest.raises(ValueError, match='native compatibility'):
        profile.verify_native_compatibility(prepared['view'], manifest, smoke, check_environment=False)


@pytest.mark.parametrize('vendor', ['_cann_ops_custom', '_cann_ops_custom/vendors/test'])
def test_only_known_overlay_paths_are_normalized(vendor):
    inputs = {name: {'native': 'a' * 64, 'dependencies': 'b' * 64} for name in ('vllm', 'vllm-ascend')}
    settings = {name: 'fixture' for name in profile.PROFILE_FIELDS}
    settings.update(build_env={}, system_files={}, launch_env={})
    def manifest(root):
        current = copy.deepcopy(settings)
        current['launch_env'] = {'PYTHONPATH': root + '/vllm:/image/acl',
                                'LD_LIBRARY_PATH': root + '/vllm-ascend/vllm_ascend:/image/lib',
                                'ASCEND_CUSTOM_OPP_PATH': root + '/vllm-ascend/vllm_ascend/' + vendor}
        return {'runtime_root': root, 'profile': current, 'build_inputs': inputs, 'files': {'a': {'sha256': 'a' * 64}}}
    original, current = manifest('/source-a'), manifest('/source-b')
    assert profile.native_compatibility_key(original) == profile.native_compatibility_key(current)
    current['profile']['launch_env']['ASCEND_CUSTOM_OPP_PATH'] += ':/vllm-workspace/executions/other/vendor'
    assert profile.native_compatibility_key(original) != profile.native_compatibility_key(current)


@pytest.mark.parametrize('value', ['/a::/b', '/a:relative:/b', ''])
def test_cwd_dependent_loader_paths_require_a_new_import(prepared, value):
    donor = copy.deepcopy(prepared['donor'])
    donor['profile']['launch_env']['LD_LIBRARY_PATH'] = value
    with pytest.raises(ValueError, match='absolute loader search paths'):
        profile.native_compatibility_key(donor)
    assert profile.native_compatibility_receipt(prepared['original'], donor) is None


@pytest.mark.parametrize('hot', [False, True])
@pytest.mark.skipif(sys.platform != 'linux', reason='native capture runs in a Linux runtime')
def test_capture_runs_import_only_when_no_complete_native_proof_is_available(prepared, monkeypatch, hot):
    # Execute the actual remote capture suffix. The import subprocess is a
    # boundary spy: cold preparation must run it, warm preparation must not.
    for name in profile.LAUNCH_PATH_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('SOC_VERSION', 'test-soc')
    monkeypatch.setenv('CXX', 'test-compiler')
    settings = copy.deepcopy(prepared['settings'])
    settings['launch_env'] = {'SOC_VERSION': 'test-soc', 'PYTHONPATH': ':'.join(
        str(prepared['original'] / path) for path in ('.mindie-runtime/metadata', 'vllm', 'vllm-ascend'))}
    donor = copy.deepcopy(prepared['donor'])
    donor['profile'] = settings
    certificate = copy.deepcopy(prepared['certificate'])
    certificate['key'] = profile.native_compatibility_key(donor)
    proof = prepared['view'] / '.mindie-runtime/profile-evidence/native-compatibility.json'
    proof.write_text(json.dumps(certificate))
    reuse = {'kind': 'native', 'soc': 'test-soc', 'compiler': 'test-compiler'}
    if hot:
        reuse['compatibility_evidence'] = '.mindie-runtime/profile-evidence/native-compatibility.json'
    (prepared['view'] / '.mindie-runtime/reuse.json').write_text(json.dumps(reuse))
    request = {'root': str(prepared['view']), 'image_digest': 'sha256:fixture',
               'cann_files': [settings['system_files']['cann']['path']],
               'driver_files': [settings['system_files']['driver']['path']]}
    monkeypatch.setattr(sys, 'argv', ['capture', json.dumps(request)])
    run = Mock(side_effect=AssertionError('warm preparation imported business code')) if hot else Mock(
        return_value=subprocess.CompletedProcess([], 1, '', 'candidate Python failure'))
    monkeypatch.setattr(subprocess, 'run', run)
    namespace = {key: value for key, value in vars(profile).items() if not key.startswith('__')}
    namespace.update(installed_native_files=lambda root: prepared['files'],
                     _build_namespace={'runtime_build_inputs': lambda *args: prepared['inputs']})
    if hot:
        exec(compile(REMOTE_CAPTURE_SUFFIX, '<remote-profile>', 'exec'), namespace)
        actual = json.loads((prepared['view'] / '.mindie-runtime/ready-profile.json').read_text())
        profile.verify(prepared['view'], actual)
        run.assert_not_called()
    else:
        with pytest.raises(ValueError, match='import smoke failed'):
            exec(compile(REMOTE_CAPTURE_SUFFIX, '<remote-profile>', 'exec'), namespace)
        run.assert_called_once()
        assert not (prepared['view'] / '.mindie-runtime/ready-profile.json').exists()
