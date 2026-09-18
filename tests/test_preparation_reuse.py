"""Content reuse and independent writable execution views."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import shlex
from pathlib import Path
from unittest import mock

import pytest

from mindie_coordinator import preparation_cache
from mindie_coordinator.build_inputs import VLLM_ASCEND_REINSTALL_PATTERNS, build_input_fingerprints
from mindie_coordinator.provision.task_environment import reusable_preparation
from mindie_coordinator.prepare_runtime import REMOTE_COMMAND_CAPTURE_SUFFIX
from mindie_coordinator.runtime_profile import command_launch_environment, file_digest, profile_key, verify


def test_content_keys_ignore_task_source_view_and_python_only_change():
    donor = {'profile': {'image_digest': 'sha256:image', 'soc': 'ascend-test', 'python_abi': 'cp312',
                         'launch_env': {'PYTHONPATH': '/old/execution'}}}
    snapshot = {'id': 'source-A', 'build_env': {}, 'records': [
        {'relpath': 'vllm', 'build_inputs': {'native': 'n1', 'dependencies': 'd1'}},
        {'relpath': 'vllm-ascend', 'build_inputs': {'native': 'n2', 'dependencies': 'd2'}}]}
    baseline = reusable_preparation(snapshot, {'recipe': 'test'}, donor)
    snapshot['id'] = 'source-B-python-change'
    donor['profile']['launch_env']['PYTHONPATH'] = '/other/execution'
    changed = reusable_preparation(snapshot, {'recipe': 'test'}, donor)
    assert baseline['native_key'] == changed['native_key']
    assert baseline['dependency_key'] == changed['dependency_key']
    snapshot['records'][1]['build_inputs']['native'] = 'new-native'
    native = reusable_preparation(snapshot, {'recipe': 'test'}, donor)
    assert native['native_key'] != baseline['native_key']
    assert native['dependency_key'] == baseline['dependency_key']
    snapshot['records'][1]['build_inputs']['native'] = 'n2'
    assert reusable_preparation(snapshot, {'recipe': 'test'}, donor)['native_key'] == baseline['native_key']
    snapshot['build_env']['SOC_VERSION'] = 'other-chip'
    assert reusable_preparation(snapshot, {'recipe': 'test'}, donor)['native_key'] != baseline['native_key']


def test_generic_launch_retains_image_paths_without_old_execution_sources():
    result = command_launch_environment({
        'PATH': '/tmp/mindie-python-shim.abc:/old/.venv/bin:/usr/local/python/bin:/usr/bin',
        'PYTHONPATH': '/old/vllm:/vllm-workspace/executions/other/vllm:/image/acl',
        'LD_LIBRARY_PATH': '/image/cann/lib64:/old/vllm-ascend/lib',
        'ASCEND_CUSTOM_OPP_PATH': '/old/vllm-ascend/custom',
        'ASCEND_OPP_PATH': '/image/cann/opp',
    }, roots=['/old'])
    assert result == {'PATH': '/usr/local/python/bin:/usr/bin', 'PYTHONPATH': '/image/acl',
                      'LD_LIBRARY_PATH': '/image/cann/lib64', 'ASCEND_OPP_PATH': '/image/cann/opp'}


def test_supported_recipe_build_inputs_include_python_codegen_and_build_configuration(tmp_path):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], text=True).strip()
    git('init')
    git('config', 'user.name', 'Test')
    git('config', 'user.email', 'test@example.invalid')
    for name in ('csrc/generate.py', 'vllm_ascend/envs.py', 'vllm_ascend/model.py', 'requirements.txt'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original')
    def commit():
        git('add', '.')
        git('commit', '-m', 'case')
        return build_input_fingerprints(tmp_path, 'HEAD', VLLM_ASCEND_REINSTALL_PATTERNS, build_env={})
    baseline = commit()
    (tmp_path / 'vllm_ascend/model.py').write_text('business Python edit')
    edited = commit()
    assert {k: v for k, v in edited.items() if k != 'imports'} == {k: v for k, v in baseline.items() if k != 'imports'}
    assert edited['imports'] != baseline['imports']
    (tmp_path / 'csrc/generate.py').write_text('code generator edit')
    generated = commit()
    assert generated['native'] != baseline['native']
    assert generated['dependencies'] == baseline['dependencies']
    (tmp_path / 'vllm_ascend/envs.py').write_text('build selection edit')
    assert commit()['native'] != generated['native']


def test_dependency_copy_never_carries_editables_and_does_not_share_writable_files(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.mkdir()
    for name in ('dep.py', '__editable__.vllm.pth', 'vllm-1.dist-info', 'old.pth'):
        (source / name).write_text('/old/vllm' if name == 'old.pth' else 'contents')
    copied = preparation_cache.copy_dependencies(source, target)
    assert copied == ['dep.py']
    (target / 'dep.py').write_text('execution modification')
    assert (source / 'dep.py').read_text() == 'contents'


class Distribution:
    version = '0.27.1+empty'
    def read_text(self, name):
        return {'METADATA': 'Name: vllm\nVersion: 0.27.1+empty\nRequires-Dist: torch\n',
                'entry_points.txt': '[vllm.platform_plugins]\nnpu=vllm_ascend:register\n'}.get(name)


def test_native_view_copies_outputs_and_routes_exact_source_metadata(tmp_path, monkeypatch):
    source, target = tmp_path / 'published', tmp_path / 'execution'
    extension = 'vllm-ascend/vllm_ascend/vllm_ascend_C.so'
    output = source / extension
    output.parent.mkdir(parents=True)
    output.write_bytes(b'compiled')
    (target / 'vllm/vllm').mkdir(parents=True)
    manifest = {'files': {extension: {'sha256': file_digest(output)}}, 'build_key': 'native-key',
                'profile': {'soc': 'test-soc', 'compiler': 'test-compiler'}}
    calls = []
    monkeypatch.setattr(preparation_cache, 'verify', lambda *a: calls.append(a), raising=False)
    monkeypatch.setattr(preparation_cache, 'checked_file', lambda root, name: root / name, raising=False)
    monkeypatch.setattr(preparation_cache, 'file_digest', file_digest, raising=False)
    monkeypatch.setattr(preparation_cache, 'build_toolchain_from_logs', lambda root: {}, raising=False)
    monkeypatch.setattr(preparation_cache.importlib.metadata, 'distribution', lambda name: Distribution())
    receipt = preparation_cache.copy_native_view(target, source, manifest,
        {'vllm': {'version': '0.28.0.dev2+g123456', 'source_head': 'real-head'}})
    assert receipt['build_key'] == 'native-key'
    assert calls == [(source, manifest)]
    (target / extension).write_bytes(b'local change')
    assert output.read_bytes() == b'compiled'
    namespace = {}
    exec((target / 'vllm/vllm/_version.py').read_text(), namespace)
    assert namespace['__version__'] == '0.28.0.dev2+g123456'
    assert namespace['__commit_id__'] == 'real-head'
    metadata = next((target / '.mindie-runtime/metadata').glob('*/METADATA')).read_text()
    assert 'Version: 0.28.0.dev2+g123456.empty' in metadata
    assert 'Requires-Dist: torch' in metadata


def test_fixed_source_metadata_retry_rewrites_safely(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation_cache.importlib.metadata, 'distribution', lambda name: Distribution())
    package = tmp_path / 'vllm/vllm'
    package.mkdir(parents=True)
    versions = {'vllm': {'version': '0.28.0.dev2+g123456', 'source_head': 'fixed-head'}}
    preparation_cache.write_source_metadata(tmp_path, versions)
    metadata = next((tmp_path / '.mindie-runtime/metadata').glob('*/METADATA'))
    expected = metadata.read_text()
    metadata.write_text('interrupted prior preparation')
    preparation_cache.write_source_metadata(tmp_path, versions)
    assert metadata.read_text() == expected
    assert len(list((tmp_path / '.mindie-runtime/metadata').glob('*.dist-info'))) == 1
    external = tmp_path.parent / (tmp_path.name + '-outside.txt')
    external.write_text('must remain unchanged')
    metadata.unlink()
    metadata.symlink_to(external)
    with pytest.raises(ValueError, match='symlink'):
        preparation_cache.write_source_metadata(tmp_path, versions)
    assert external.read_text() == 'must remain unchanged'


@pytest.mark.parametrize('artifact', ['vllm_ascend_C.so', '_build_info.py'])
def test_native_copy_rejects_donor_changed_after_verification(tmp_path, monkeypatch, artifact):
    source, target = tmp_path / 'published', tmp_path / 'execution'
    relative = 'vllm-ascend/vllm_ascend/' + artifact
    path = source / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b'verified artifact')
    manifest = {'files': {relative: {'sha256': file_digest(path)}}}
    monkeypatch.setattr(preparation_cache, 'verify', lambda *args: None, raising=False)
    monkeypatch.setattr(preparation_cache, 'checked_file', lambda root, name: root / name, raising=False)
    monkeypatch.setattr(preparation_cache, 'file_digest', file_digest, raising=False)
    actual_copy = preparation_cache.shutil.copy2
    def concurrent_replacement(original, destination):
        original.write_bytes(b'donor rewrote artifact after validation')
        return actual_copy(original, destination)
    monkeypatch.setattr(preparation_cache.shutil, 'copy2', concurrent_replacement)
    with pytest.raises(ValueError, match='copied artifact differs'):
        preparation_cache.copy_native_view(target, source, manifest, {})


def test_native_copy_requires_generated_metadata_proof(tmp_path, monkeypatch):
    source = tmp_path / 'published'
    path = source / 'vllm-ascend/vllm_ascend/_build_info.py'
    path.parent.mkdir(parents=True)
    path.write_text('SOC_VERSION = "unexpected"')
    monkeypatch.setattr(preparation_cache, 'verify', lambda *args: None, raising=False)
    with pytest.raises(ValueError, match='generated build metadata has no verified donor hash'):
        preparation_cache.copy_native_view(tmp_path / 'view', source, {'files': {}}, {})


def test_installed_outputs_include_import_time_build_metadata(tmp_path):
    from mindie_coordinator.runtime_profile import installed_native_files
    package = tmp_path / 'vllm-ascend/vllm_ascend'
    vendor = package / '_cann_ops_custom/vendors/test'
    vendor.mkdir(parents=True)
    for path in (package / 'vllm_ascend_C.so', package / '_build_info.py', vendor / 'custom.so', vendor / 'config.json'):
        path.write_text('fixture')
    assert installed_native_files(tmp_path)['vllm-ascend/vllm_ascend/_build_info.py'] == 'metadata'


def test_long_role_names_have_distinct_runtime_identity():
    from mindie_coordinator.provision.task_environment import isolated_root, task_runtime_id
    names = ['tensor-parallel-worker-001', 'tensor-parallel-worker-002']
    roots = [isolated_root('execution', name, 'host') for name in names]
    identities = [task_runtime_id('execution', 'host', name) for name in names]
    assert roots[0] != roots[1]
    assert identities[0] != identities[1]
    assert identities[0] == task_runtime_id('execution', 'host', names[0])


def test_historical_qualification_rejects_clean_native_commit_without_rebuild(tmp_path, monkeypatch):
    from mindie_coordinator import backend
    from mindie_coordinator.build_inputs import runtime_build_inputs
    for name in ('vllm', 'vllm-ascend'):
        repo = tmp_path / name
        repo.mkdir()
        def git(*args):
            return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()
        git('init')
        git('config', 'user.name', 'Test')
        git('config', 'user.email', 'test@example.invalid')
        (repo / 'native.cpp').write_text('int compiled = 1;\n')
        git('add', '.')
        git('commit', '-m', 'compiled source')
    manifest = {'profile': {'build_env': {}}, 'profile_key': 'profile', 'build_key': 'original-artifact',
                'build_inputs': runtime_build_inputs(tmp_path, {'build_env': {}}, 'profile')}
    marker = tmp_path / '.mindie-runtime/ready-profile.json'
    marker.parent.mkdir()
    marker.write_text(json.dumps(manifest))
    # Artifact/environment verification already passed; this test exercises
    # the independent link between those bytes and the current Git source.
    probe = tmp_path / 'profile_probe.py'
    probe.write_text('import json\nfrom pathlib import Path\ndef verify(root, manifest): pass\n')
    package_file = backend._package_file
    monkeypatch.setattr(backend, '_package_file', lambda name: probe if name == 'runtime_profile.py' else package_file(name))
    client = backend.RemoteBackend()
    def local_probe(endpoint, script):
        prefix, body = script.split(" <<'MINDIE_QUALIFY'\n", 1)
        request = shlex.split(prefix.splitlines()[-1])[-1]
        code = body.rsplit('\nMINDIE_QUALIFY', 1)[0]
        result = subprocess.run([sys.executable, '-', request], input=code, text=True, capture_output=True, check=True)
        return result.stdout
    monkeypatch.setattr(client, 'bash', local_probe)
    def snapshot():
        return {'records': [{'relpath': name, 'tree': subprocess.check_output(
            ['git', '-C', str(tmp_path / name), 'rev-parse', 'HEAD^{tree}'], text=True).strip()} for name in ('vllm', 'vllm-ascend')]}
    runtime = {'endpoint': {'root': str(tmp_path)}, 'python': sys.executable}
    assert client.qualify_prepared_inputs(runtime, snapshot())['qualified'] is True
    (repo / 'native.cpp').write_text('int compiled = 2;\n')
    git('add', '.')
    git('commit', '-m', 'source changed without rebuilding artifact')
    result = client.qualify_prepared_inputs(runtime, snapshot())
    assert result['qualified'] is False
    assert 'native artifacts do not match' in result['reason']


def test_generic_profile_executes_without_native_packages_and_detects_identity_tampering(tmp_path):
    import mindie_coordinator.runtime_profile as module
    code = Path(module.__file__).read_text() + REMOTE_COMMAND_CAPTURE_SUFFIX
    request = {'root': str(tmp_path), 'image_digest': 'sha256:verified-image', 'source_id': 'fixed-source'}
    result = subprocess.run([sys.executable, '-c', 'import sys; exec(sys.stdin.read())', json.dumps(request)],
                            input=code, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert manifest['profile']['kind'] == 'command'
    verify(tmp_path, manifest)
    manifest['source_id'] = 'different'
    with pytest.raises(ValueError, match='source identity'):
        verify(tmp_path, manifest)
    unsafe = copy.deepcopy(manifest['profile'])
    unsafe['launch_env'] = {'API_TOKEN': 'secret'}
    with pytest.raises(ValueError, match='secrets'):
        profile_key(unsafe)


def test_artifact_donor_registration_preserves_fixed_proof_and_cannot_launch(tmp_path):
    from mindie_coordinator.execution_sources import SCHEMA_VERSION, source_identity
    from mindie_coordinator.ready_runtime import RuntimePool
    class Backend:
        def __init__(self):
            self.inspected = []
            self.host_requests = []
        def inspect(self, spec, **kwargs):
            self.inspected.append(spec)
            return {'profile_key': 'profile', 'build_key': 'build', 'container_id': 'container',
                    'profile': {'launch_env': {}}}
        def host(self, runtime, request):
            self.host_requests.append(request)
    backend = Backend()
    pool = RuntimePool(tmp_path / 'pool', backend)
    fixed = {'schema_version': SCHEMA_VERSION,
             'sources': {name: {'path': '/source/' + name, 'tree': 'a' * 40, 'commit': 'b' * 40} for name in ('vllm', 'vllm-ascend')},
             'records': [{'relpath': name, 'tree': 'a' * 40, 'commit': 'b' * 40} for name in ('vllm', 'vllm-ascend')], 'build_env': {}}
    fixed['id'] = source_identity(fixed)
    spec = {'user': 'alice', 'python': '/donor/.venv/bin/python', 'container_name': 'mindie-alice',
            'host_endpoint': {'host': '192.0.2.1', 'port': 22, 'user': 'root'},
            'endpoint': {'host': '192.0.2.1', 'port': 46001, 'user': 'root', 'root': '/donor', 'cwd': '/donor'},
            'reuse_only': True, 'source_snapshot': fixed}
    registered = pool.register('artifact-donor', spec)
    assert registered['reuse_only'] is True
    assert backend.inspected[0]['source_snapshot'] == fixed
    assert backend.host_requests == []
    session = pool.session_open('alice', 'session', {})
    assert pool.checkout('alice', session['id'], 'profile', 'checkout', 'artifact-donor')['status'] == 'cache_miss'
    malformed = {**spec, 'source_snapshot': {**fixed, 'id': 'tampered'}}
    with pytest.raises(ValueError, match='identity'):
        pool.register('bad-donor', malformed)
