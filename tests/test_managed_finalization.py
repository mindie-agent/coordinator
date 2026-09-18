"""Owned finalization reuses current source hashes, never foreign byte claims."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import subprocess
import sys
import threading
from unittest.mock import Mock

import pytest

from mindie_coordinator import backend, parity, preparation_cache as cache, runtime_profile as profile
from mindie_coordinator.prepare_runtime import REMOTE_CAPTURE_SUFFIX
from test_native_compatibility_reuse import prepared
from test_shared_preparation import bundle
from test_captured_profile_handoff import receipt


def test_current_capture_and_publish_hash_source_once_and_destination_once(bundle, monkeypatch):
    root, _, shared, _, original, _, _ = bundle
    files = {name: row['role'] for name, row in original['files'].items()}
    # A large realistic cardinality catches an accidental extra whole traversal.
    for index in range(1104 - len(files)):
        name = f'vllm-ascend/vllm_ascend/_cann_ops_custom/{index}.o'
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(str(index).encode())
        files[name] = 'library'
    evidence = {name: row['path'] for name, row in original['evidence'].items()}
    counts = Counter()
    real_digest = profile.file_digest
    def counted(path):
        path = Path(path)
        if path.is_relative_to(root) and path.relative_to(root).as_posix() in files:
            counts['source'] += 1
        elif path.is_relative_to(shared) and path.name.endswith(('.o', '.pyd', '.so', '.py')):
            counts['destination'] += 1
        return real_digest(path)
    monkeypatch.setattr(profile, 'file_digest', counted)
    monkeypatch.setattr(cache, '_publish_captured_bundle', profile._publish_captured_bundle, raising=False)
    manifest = profile.capture(root, original['profile'], original['build_inputs'], files, evidence)
    manifest['preparation'] = original['preparation']
    profile._verify_manifest_proof(root, manifest)
    timings = {}
    result = cache._store_captured_native(root, shared, manifest, timings)
    assert result['status'] == 'stored'
    assert counts == {'source': 1104, 'destination': 1104}
    assert timings['bundle_copy'] >= 0 and timings['bundle_verify'] >= 0


@pytest.mark.parametrize('public', [False, True])
def test_copy_corruption_never_publishes_bundle_or_index(bundle, monkeypatch, public):
    root, _, shared, _, manifest, relative, _ = bundle
    monkeypatch.setattr(cache, '_publish_captured_bundle', profile._publish_captured_bundle, raising=False)
    real_copy = profile.shutil.copy2
    def corrupt(source, target, *args, **kwargs):
        result = real_copy(source, target, *args, **kwargs)
        if Path(source) == root / relative:
            Path(target).write_bytes(b'corrupt copy')
        return result
    monkeypatch.setattr(profile.shutil, 'copy2', corrupt)
    before = (root / relative).read_bytes()
    with pytest.raises(ValueError, match='artifact hash mismatch'):
        if public:
            cache.store_shared_native(root, shared)
        else:
            cache._store_captured_native(root, shared, manifest, {})
    assert (root / relative).read_bytes() == before
    assert not list(shared.glob('*.json'))
    assert not list((shared / 'bundles').iterdir())


def test_public_publish_still_rejects_changed_source(bundle):
    root, _, shared, _, manifest, relative, _ = bundle
    (root / relative).write_bytes(b'changed after independent attestation')
    with pytest.raises(ValueError, match='artifact hash mismatch'):
        profile.publish(root, shared, manifest)
    assert not shared.exists()


def test_concurrent_complete_publications_preserve_one_verified_bundle(bundle, monkeypatch):
    root, _, shared, _, manifest, _, _ = bundle
    barrier = threading.Barrier(2)
    real_rename = profile.os.rename
    def simultaneous(source, target):
        barrier.wait(timeout=10)
        return real_rename(source, target)
    monkeypatch.setattr(profile.os, 'rename', simultaneous)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(profile.publish, root, shared, manifest) for _ in range(2)]
        destinations = [future.result() for future in futures]
    assert destinations[0] == destinations[1]
    assert list(shared.iterdir()) == [destinations[0]]
    profile.verify(destinations[0], manifest, check_environment=False)


@pytest.mark.skipif(sys.platform != 'linux', reason='actual capture payload runs in a Linux runtime')
@pytest.mark.parametrize('failure', [None, 'smoke', 'copy', 'filesystem', 'marker'])
def test_actual_managed_suffix_finishes_smoke_copy_and_atomic_ready(prepared, monkeypatch, tmp_path, failure):
    root = prepared['view']
    for name in profile.LAUNCH_PATH_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('SOC_VERSION', 'test-soc')
    monkeypatch.setenv('CXX', 'test-compiler')
    request = {'root': str(root), 'source_id': 'fixed', 'image_digest': 'sha256:fixture',
               'preparation': {'native': {'vllm': 'a', 'vllm-ascend': 'b'}, 'dependencies': {},
                               'native_key': 'fixture', 'environment': {}},
               'managed_finalize': True, 'publish_native_cache': True,
               'installation_marker': str(root / '.remote-code-parity/runtime-install.json'),
               'container_identity': 'fixture',
               'cann_files': [prepared['settings']['system_files']['cann']['path']],
               'driver_files': [prepared['settings']['system_files']['driver']['path']]}
    monkeypatch.setattr(sys, 'argv', ['capture', json.dumps(request)])
    imports = Mock(return_value=subprocess.CompletedProcess([], 1 if failure == 'smoke' else 0, 'fixture smoke', 'failed import'))
    monkeypatch.setattr(subprocess, 'run', imports)
    outputs, errors = [], []
    namespace = {key: value for key, value in vars(profile).items() if not key.startswith('__')}
    exec(compile(Path(cache.__file__).read_text(), '<cache>', 'exec'), namespace)
    namespace.update(installed_dependency_identity=lambda: {}, installed_native_files=lambda root: prepared['files'],
                     print=lambda message, **kwargs: (errors if kwargs.get('file') else outputs).append(message),
                     SHARED_NATIVE_CACHE=str(tmp_path / 'cache'),
                     _build_namespace={'runtime_build_inputs': lambda *args: prepared['inputs']})
    if failure == 'filesystem':
        namespace['_store_captured_native'] = Mock(side_effect=PermissionError('read-only optional cache'))
    if failure == 'copy':
        real_copy = profile.shutil.copy2
        def corrupt(source, target, *a, **k):
            real_copy(source, target, *a, **k)
            Path(target).write_bytes(b'corrupted destination')
        monkeypatch.setattr(profile.shutil, 'copy2', corrupt)
    if failure == 'marker':
        real_replace = profile.os.replace
        def fail_marker(source, target):
            if Path(target).name == 'ready-profile.json':
                raise PermissionError('ready marker failed')
            return real_replace(source, target)
        monkeypatch.setattr(profile.os, 'replace', fail_marker)
    if failure in {'smoke', 'copy', 'marker'}:
        with pytest.raises((ValueError, PermissionError)):
            exec(compile(REMOTE_CAPTURE_SUFFIX, '<capture>', 'exec'), namespace)
        assert not outputs and not (root / '.mindie-runtime/ready-profile.json').exists()
        assert not (root / '.mindie-runtime/ready-profile.tmp').exists()
    else:
        exec(compile(REMOTE_CAPTURE_SUFFIX, '<capture>', 'exec'), namespace)
        manifest = backend._captured_manifest(outputs[0])
        reply = json.loads(outputs[0])
        assert manifest == json.loads((root / '.mindie-runtime/ready-profile.json').read_text())
        assert reply['native_smoke_executed'] is True
        assert reply['native_cache']['status'] == ('miss' if failure else 'stored')
        assert all(value >= 0 for value in reply['preparation_timings'].values())
        assert {'native_smoke', 'capture_hash', 'cache_store', 'finalize_total'} <= reply['preparation_timings'].keys()
        profile.verify(root, manifest)
    imports.assert_called_once()


def test_image_is_reused_only_within_preparation_and_generation_is_checked(receipt, monkeypatch):
    _, info, spec = receipt
    adapter = backend.RemoteBackend()
    query = Mock(return_value=json.dumps(info))
    monkeypatch.setattr(adapter, 'bash', query)
    assert adapter._preparation_container(spec) == {'Id': info['Id'], 'Image': info['Image']}
    assert adapter._preparation_container(spec)['Image'] == info['Image']
    query.assert_called_once()
    query.return_value = json.dumps({**info, 'Id': 'replacement'})
    with pytest.raises(ValueError, match='generation changed'):
        adapter._write_ready_profile(spec, {}, managed_finalize=True)
