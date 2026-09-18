"""The actual Bash/Python probe resolves the registered loader environment."""
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import pytest

from mindie_coordinator import backend, runtime_profile
from mindie_coordinator.build_inputs import runtime_build_inputs


@pytest.mark.skipif(os.name == 'nt', reason='executes the actual remote Bash entry')
@pytest.mark.parametrize('operation', ['inspect', 'qualify'])
@pytest.mark.parametrize('change', [None, 'version', 'missing', 'origin'])
def test_probe_uses_bound_pythonpath_and_rejects_changed_distribution(tmp_path, monkeypatch, operation, change):
    root = tmp_path / 'view'
    root.mkdir()
    for name in ('vllm', 'vllm-ascend'):
        repo = root / name
        repo.mkdir()
        for arguments in [('init', '-q'), ('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                                          'commit', '--allow-empty', '-qm', 'fixture')]:
            subprocess.run(['git', '-C', str(repo), *arguments], check=True, capture_output=True)
    base, cann = tmp_path / 'base-metadata', tmp_path / 'cann-python'
    for name in [*runtime_profile.PACKAGES.values(), 'affinity-sched']:
        directory = (cann if name == 'affinity-sched' else base) / (name.replace('-', '_') + '-1.0.dist-info')
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'METADATA').write_text('Name: ' + name + '\nVersion: 1.0\n')
    environment = {**os.environ, 'PYTHONPATH': os.pathsep.join(map(str, (base, cann)))}
    # Capture the real metadata inventory in a fresh interpreter, with the
    # same loader paths used at preparation; no torch/device import is needed.
    code = Path(runtime_profile.__file__).read_text() + '\nprint(json.dumps(installed_dependency_identity()))\n'
    identity = json.loads(subprocess.check_output([sys.executable, '-c', code], env=environment, text=True))
    profile = dict.fromkeys(runtime_profile.PROFILE_FIELDS, '1.0')
    profile.update(image_digest='sha256:fixture', python_abi=sysconfig.get_config_var('SOABI'),
                   build_env={}, launch_env={'PYTHONPATH': environment['PYTHONPATH']},
                   compatibility_evidence='fixture-smoke', system_files={
                       name: {'path': str(tmp_path / name), 'sha256': '1' * 64} for name in ('cann', 'driver')}, **identity)
    for name, row in profile['system_files'].items():
        Path(row['path']).write_text(name)
        row['sha256'] = runtime_profile.file_digest(Path(row['path']))
    manifest = {'runtime_root': str(root), 'profile': profile, 'profile_key': runtime_profile.profile_key(profile),
                'build_key': 'fixture', 'files': {}, 'build_inputs': runtime_build_inputs(root, profile, runtime_profile.profile_key(profile))}
    marker = root / '.mindie-runtime/ready-profile.json'
    marker.parent.mkdir()
    if change == 'origin':
        profile['dependency_origins']['affinity-sched'] = str(tmp_path / 'other-origin')
    marker.write_text(json.dumps(manifest))
    # Keep environment verification unmodified. Artifact/source verification
    # is covered by the existing complete-profile and snapshot tests.
    module = tmp_path / 'environment_probe.py'
    module.write_text(Path(runtime_profile.__file__).read_text() + '\ndef verify(root, manifest): verify_environment(root, manifest)\nverify_execution_view = verify\n')
    package_file = backend._package_file
    monkeypatch.setattr(backend, '_package_file', lambda name: module if name == 'runtime_profile.py' else package_file(name))
    client = backend.RemoteBackend()
    def local_bash(target, script):
        env = {**os.environ, 'PYTHONPATH': str(base)}
        result = subprocess.run(['bash', '--noprofile', '--norc'], input=script, text=True,
                                capture_output=True, env=env, cwd=root, timeout=30)
        if result.returncode:
            raise RuntimeError(result.stderr)
        return result.stdout
    monkeypatch.setattr(client, 'bash', local_bash)
    runtime = {'endpoint': {'root': str(root)}, 'python': sys.executable, 'attestation': manifest}
    metadata = cann / 'affinity_sched-1.0.dist-info'
    if change == 'version':
        (metadata / 'METADATA').write_text('Name: affinity-sched\nVersion: 2.0\n')
    elif change == 'missing':
        metadata.rename(cann / 'removed')
    def probe():
        return client._inspect_manifest(runtime, expected_digest=runtime_profile.digest(manifest), prepared_view=True) if operation == 'inspect' else client.qualify_prepared_inputs(runtime, {'records': []})
    if change:
        with pytest.raises(RuntimeError, match='profile dependency changed: affinity-sched|metadata was found for affinity-sched|dependency locations changed'):
            probe()
    else:
        reply = probe()
        assert reply == ({'manifest_digest': runtime_profile.digest(manifest)} if operation == 'inspect' else {'qualified': True, 'build_key': 'fixture'})
