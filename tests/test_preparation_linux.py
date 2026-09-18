"""Functional Linux proof: reused extension and SCM metadata resolve to the view."""
from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from mindie_coordinator import preparation_cache as cache, runtime_profile as profile


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux recipient venv and shell contract')
def test_image_pip_installs_only_in_new_owned_venv(tmp_path, monkeypatch):
    import importlib.util
    import shlex
    import zipfile
    from mindie_coordinator import parity
    from mindie_coordinator.provision.task_environment import create_venv_script
    root = tmp_path / 'execution'
    interpreter = root / '.venv/bin/python'
    monkeypatch.setattr(parity, 'PYTHON_METADATA_PREAMBLE', ['PYTHON=' + shlex.quote(sys.executable)])
    script = create_venv_script(str(root), str(interpreter))
    environment = dict(os.environ)
    pip = importlib.util.find_spec('pip')
    if pip and pip.origin:
        # Expose the image's existing pip while keeping sys.prefix/purelib in
        # the new venv, exactly as system-site-packages does on the recipient.
        environment['PYTHONPATH'] = str(Path(pip.origin).parent.parent)
    subprocess.run(['bash', '-c', script], env=environment, capture_output=True, text=True, check=True, timeout=30)
    wheel = tmp_path / 'mindie_dependency_fixture-1.0-py3-none-any.whl'
    dist = 'mindie_dependency_fixture-1.0.dist-info/'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('mindie_dependency_fixture.py', 'value = "owned"\n')
        archive.writestr(dist + 'METADATA', 'Metadata-Version: 2.1\nName: mindie-dependency-fixture\nVersion: 1.0\n')
        archive.writestr(dist + 'WHEEL', 'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
        archive.writestr(dist + 'RECORD', '')
    subprocess.run([str(interpreter), '-m', 'pip', 'install', '--no-index', '--no-deps', str(wheel)],
                   env=environment, capture_output=True, text=True, check=True, timeout=30)
    probe = subprocess.run([str(interpreter), '-c',
        'import json,sys,sysconfig,mindie_dependency_fixture as p;print(json.dumps([sys.prefix,sysconfig.get_paths()["purelib"],p.__file__]))'],
        env=environment, capture_output=True, text=True, check=True)
    prefix, purelib, installed = map(Path, json.loads(probe.stdout))
    assert prefix == root / '.venv'
    assert purelib.is_relative_to(prefix) and installed.is_relative_to(purelib)


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux extension loading contract')
def test_python_source_view_loads_copied_extension_and_own_scm_metadata(tmp_path, monkeypatch):
    import _ctypes_test
    original, view = tmp_path / 'published', tmp_path / 'view'
    # Use Python's installed extension as an ABI-correct native fixture; this
    # test invokes neither pip nor a compiler or a model/device runtime.
    extension = 'vllm-ascend/vllm_ascend/' + Path(_ctypes_test.__file__).name
    source_extension = original / extension
    source_extension.parent.mkdir(parents=True)
    shutil.copy2(_ctypes_test.__file__, source_extension)
    config = original / 'vllm-ascend/vllm_ascend/config.json'
    config.write_text('{}')
    for name in ('vllm', 'vllm-ascend'):
        package = name.replace('-', '_')
        for root, value in ((original, 'original'), (view, 'changed')):
            target = root / name / package
            target.mkdir(parents=True, exist_ok=True)
            (target / '__init__.py').write_text('value = ' + repr(value) + '\n')
    installed = tmp_path / 'donor-site'
    for name in ('vllm', 'vllm_ascend'):
        metadata = installed / (name + '-1.0.dist-info')
        metadata.mkdir(parents=True)
        (metadata / 'METADATA').write_text('Name: ' + name.replace('_', '-') + '\nVersion: 1.0\n')
    monkeypatch.syspath_prepend(str(installed))
    settings = {key: '1.0' for key in profile.PROFILE_FIELDS}
    settings.update(python_abi=sysconfig.get_config_var('SOABI'), build_env={}, launch_env={},
                    compatibility_evidence='smoke.json', system_files={})
    for name in ('cann', 'driver'):
        file = original / (name + '.txt')
        file.write_text('fixture ' + name)
        settings['system_files'][name] = {'path': str(file), 'sha256': profile.file_digest(file)}
    (original / 'smoke.json').write_text(json.dumps({'passed': True}))
    inputs = {'vllm': {'native': 'same'}, 'vllm-ascend': {'native': 'same'}}
    manifest = profile.capture(original, settings, inputs,
        {extension: 'library', config.relative_to(original).as_posix(): 'metadata'},
        {'cann': 'cann.txt', 'driver': 'driver.txt', 'smoke': 'smoke.json'})
    monkeypatch.setattr(profile.importlib.metadata, 'version', lambda name: '1.0')
    monkeypatch.setattr(cache, 'verify', profile.verify, raising=False)
    monkeypatch.setattr(cache, 'checked_file', profile.checked_file, raising=False)
    monkeypatch.setattr(cache, 'file_digest', profile.file_digest, raising=False)
    monkeypatch.setattr(cache, 'build_toolchain_from_logs', profile.build_toolchain_from_logs, raising=False)
    original_hash = profile.file_digest(source_extension)
    cache.copy_native_view(view, original, manifest,
        {name: {'version': '2.0.dev3+g123abc', 'source_head': 'real-source-head'} for name in ('vllm', 'vllm-ascend')})
    command = '''import json, importlib.metadata, vllm, vllm_ascend._ctypes_test as extension
from vllm._version import __version__, __commit_id__
print(json.dumps({'value':vllm.value,'source':vllm.__file__,'extension':extension.__file__,
 'version':__version__,'metadata':importlib.metadata.version('vllm'),'commit':__commit_id__,
 'loaded':extension.__name__}))'''
    paths = [view / '.mindie-runtime/metadata', view / 'vllm', view / 'vllm-ascend', installed,
             original / 'vllm', original / 'vllm-ascend']
    result = subprocess.run([sys.executable, '-c', command], text=True, capture_output=True,
                            env={**os.environ, 'PYTHONPATH': os.pathsep.join(map(str, paths))})
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    assert actual['value'] == 'changed'
    assert Path(actual['source']).is_relative_to(view)
    assert Path(actual['extension']).is_relative_to(view)
    assert actual['version'] == actual['metadata'] == '2.0.dev3+g123abc'
    assert actual['commit'] == 'real-source-head'
    assert actual['loaded'] == 'vllm_ascend._ctypes_test'
    assert profile.file_digest(source_extension) == original_hash


def test_destination_symlinks_cannot_escape_execution_view(tmp_path):
    external = tmp_path / 'external'
    external.mkdir()
    root = tmp_path / 'view'
    root.mkdir()
    (root / 'vllm-ascend').symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        cache.safe_destination(root, 'vllm-ascend/new.so')
    assert not (external / 'new.so').exists()


@pytest.mark.parametrize('smoke', [
    {'passed': False}, {'passed': True, 'profile_key': 'another-profile'},
    {'passed': True, 'build_inputs': {'different': 'inputs'}}, 'not-json',
])
def test_malformed_smoke_cannot_qualify_native_outputs(tmp_path, smoke):
    settings = {key: 'fixture' for key in profile.PROFILE_FIELDS}
    settings.update(build_env={}, launch_env={}, compatibility_evidence='smoke.json', system_files={})
    for name in ('cann', 'driver', 'extension.so', 'config.json'):
        (tmp_path / name).write_text(name)
    for name in ('cann', 'driver'):
        settings['system_files'][name] = {'path': str(tmp_path / name), 'sha256': profile.file_digest(tmp_path / name)}
    (tmp_path / 'smoke.json').write_text(json.dumps(smoke) if isinstance(smoke, dict) else smoke)
    manifest = profile.capture(tmp_path, settings, {'vllm': 'a', 'vllm-ascend': 'b'},
        {'extension.so': 'library', 'config.json': 'metadata'},
        {'cann': 'cann', 'driver': 'driver', 'smoke': 'smoke.json'})
    with pytest.raises(ValueError, match='import-smoke'):
        profile.verify(tmp_path, manifest, check_environment=False)
