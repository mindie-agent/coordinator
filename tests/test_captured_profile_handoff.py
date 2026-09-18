"""A complete fresh capture feeds registration without repeating its probe."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import zlib

import pytest

from remote_dev.core.ssh_transport import RemoteCompleted
from mindie_coordinator import backend as adapters, runtime_profile as profile
from mindie_coordinator.prepare_runtime import REMOTE_CAPTURE_SUFFIX
from mindie_coordinator.preparation_process import PreparationCancelled, PreparationUncertain
from test_native_compatibility_reuse import prepared


def encode(manifest):
    raw = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
    return json.dumps({'manifest_zlib_base64': base64.b64encode(zlib.compress(raw)).decode(),
                       'manifest_bytes': len(raw), 'manifest_digest': profile.digest(manifest)})


@pytest.fixture
def receipt():
    settings = dict.fromkeys(profile.PROFILE_FIELDS, '1.0')
    settings.update(image_digest='sha256:image', build_env={}, launch_env={},
                    compatibility_evidence='smoke.json', system_files={
                        name: {'path': '/' + name, 'sha256': '1' * 64} for name in ('cann', 'driver')})
    inputs = {name: {'native': 'a' * 64, 'dependencies': 'b' * 64, 'build_env': 'c' * 64}
              for name in ('vllm', 'vllm-ascend')}
    manifest = {'schema_version': 1, 'profile': settings, 'profile_key': profile.profile_key(settings),
                'build_key': profile.build_key(settings, inputs), 'build_inputs': inputs,
                'runtime_root': '/execution', 'execution_view': {'source_id': 'fixed', 'python': '/execution/.venv/bin/python'},
                'files': {name: {'role': role, 'sha256': hashlib.sha256(name.encode()).hexdigest()}
                          for name, role in [('extension.so', 'library'), ('config.json', 'metadata')]},
                'evidence': {name: {'path': name + '.json', 'sha256': '2' * 64} for name in ('cann', 'driver', 'smoke')}}
    info = {'Id': 'actual-container', 'Image': 'sha256:image', 'State': {'Running': True}}
    spec = {'python': '/execution/.venv/bin/python', 'endpoint': {'host': 'fixture.invalid', 'port': 46001,
             'user': 'root', 'root': '/execution', 'cwd': '/execution'}, 'host_endpoint': {},
            'container_name': 'owned', 'source_snapshot': {'id': 'fixed'}}
    return manifest, info, spec


def test_complete_1103_file_manifest_roundtrip_is_compressed(receipt):
    manifest, _, _ = receipt
    manifest['files'] = {'vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/op/kernel/' + str(i) + '.o': {
        'role': 'library', 'sha256': hashlib.sha256(str(i).encode()).hexdigest()} for i in range(1103)}
    output = encode(manifest)
    assert adapters._captured_manifest(output) == manifest
    assert len(output) < len(json.dumps(manifest)) // 2


@pytest.mark.parametrize('change', ['encoding', 'truncated', 'trailing', 'size', 'digest', 'limit', 'bomb'])
def test_incomplete_or_unbounded_capture_receipt_cannot_be_used(receipt, change):
    manifest, _, _ = receipt
    reply = json.loads(encode(manifest))
    if change == 'encoding':
        reply['manifest_zlib_base64'] = 'not base64!'
    elif change in {'truncated', 'trailing'}:
        compressed = base64.b64decode(reply['manifest_zlib_base64'])
        reply['manifest_zlib_base64'] = base64.b64encode(compressed[:-1] if change == 'truncated' else compressed + b'other').decode()
    elif change == 'size':
        reply['manifest_bytes'] += 1
    elif change == 'digest':
        reply['manifest_digest'] = '0' * 64
    elif change == 'limit':
        reply['manifest_bytes'] = 16 * 1024 * 1024 + 1
    else:
        reply['manifest_bytes'] = 2
        reply['manifest_zlib_base64'] = base64.b64encode(zlib.compress(b'x' * 1000000)).decode()
    with pytest.raises(ValueError):
        adapters._captured_manifest(json.dumps(reply))


@pytest.mark.parametrize('owned', [False, True])
def test_capture_returns_actual_container_and_full_proof_from_one_completed_command(receipt, monkeypatch, owned):
    manifest, info, spec = receipt
    backend = adapters.RemoteBackend()
    output = encode(manifest)
    shell = Mock(side_effect=[json.dumps(info), output])
    monkeypatch.setattr(backend, 'bash', shell)
    stream = Mock(return_value=SimpleNamespace(stdout=output))
    monkeypatch.setattr('mindie_coordinator.parity_support.ssh_exec_stream', stream)
    process = object() if owned else None
    result = backend._write_ready_profile(spec, {}, process=process)
    assert result == {**manifest, 'container_id': info['Id'],
                      'launch_preamble': profile.launch_preamble(manifest['profile'], spec['python'])}
    assert shell.call_count == (1 if owned else 2)
    assert 'json .Id' in shell.call_args_list[0].args[1] and 'json .Image' in shell.call_args_list[0].args[1]
    if owned:
        assert stream.call_count == 1 and stream.call_args.kwargs['process'] is process


@pytest.mark.parametrize('change', ['root', 'source', 'python', 'image', 'files', 'evidence', 'build-key'])
def test_capture_handoff_rejects_foreign_or_incomplete_proof(receipt, monkeypatch, change):
    manifest, info, spec = receipt
    if change == 'root': manifest['runtime_root'] = '/other'
    elif change in {'source', 'python'}: manifest['execution_view']['source_id' if change == 'source' else 'python'] = 'other'
    elif change == 'image': info['Image'] = 'sha256:other'
    elif change == 'files': manifest['files'] = {}
    elif change == 'evidence': manifest['evidence'].pop('smoke')
    else: manifest['build_key'] = '0' * 64
    backend = adapters.RemoteBackend()
    monkeypatch.setattr(backend, 'bash', Mock(side_effect=[json.dumps(info), encode(manifest)]))
    with pytest.raises(ValueError, match='capture did not return'):
        backend._write_ready_profile(spec, {})


@pytest.mark.parametrize('state', [{'Running': False}, {'Running': True, 'Paused': True}, {'Running': True, 'Restarting': True}])
def test_container_not_running_normally_never_starts_capture(receipt, monkeypatch, state):
    _, info, spec = receipt
    info['State'] = state
    backend = adapters.RemoteBackend()
    shell = Mock(return_value=json.dumps(info))
    monkeypatch.setattr(backend, 'bash', shell)
    with pytest.raises(ValueError, match='not running normally'):
        backend._write_ready_profile(spec, {})
    shell.assert_called_once()


@pytest.mark.parametrize('corrupt', [False, True])
@pytest.mark.skipif(sys.platform != 'linux', reason='capture runs inside a Linux runtime')
def test_actual_capture_suffix_returns_only_after_import_hashes_and_atomic_marker(prepared, monkeypatch, corrupt):
    root = prepared['view']
    for name in profile.LAUNCH_PATH_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('SOC_VERSION', 'test-soc')
    monkeypatch.setenv('CXX', 'test-compiler')
    request = {'root': str(root), 'source_id': 'fixed', 'image_digest': 'sha256:fixture',
               'cann_files': [prepared['settings']['system_files']['cann']['path']],
               'driver_files': [prepared['settings']['system_files']['driver']['path']]}
    monkeypatch.setattr(sys, 'argv', ['capture', json.dumps(request)])
    imports = Mock(return_value=subprocess.CompletedProcess([], 0, 'fixture import result', ''))
    monkeypatch.setattr(subprocess, 'run', imports)
    outputs, verified = [], []
    def verify(root, manifest):
        verified.append(copy.deepcopy(manifest))
        if corrupt:
            (root / next(iter(manifest['files']))).write_bytes(b'changed after capture')
        profile.verify(root, manifest)
    namespace = {key: value for key, value in vars(profile).items() if not key.startswith('__')}
    namespace.update(installed_native_files=lambda root: prepared['files'], print=outputs.append,
                     verify=verify, _build_namespace={'runtime_build_inputs': lambda *args: prepared['inputs']})
    if corrupt:
        with pytest.raises(ValueError, match='artifact hash mismatch'):
            exec(compile(REMOTE_CAPTURE_SUFFIX, '<capture>', 'exec'), namespace)
        assert not outputs and not (root / '.mindie-runtime/ready-profile.json').exists()
    else:
        exec(compile(REMOTE_CAPTURE_SUFFIX, '<capture>', 'exec'), namespace)
        manifest = adapters._captured_manifest(outputs[0])
        assert manifest == json.loads((root / '.mindie-runtime/ready-profile.json').read_text()) == verified[0]
        assert manifest['execution_view'] == {'source_id': 'fixed', 'python': sys.executable}
        profile.verify_execution_view(root, manifest)
        assert len(manifest['files']) == len(prepared['files'])
    imports.assert_called_once()
    assert 'import torch_npu, vllm, vllm_ascend, acl' in imports.call_args.args[0][2]


@pytest.mark.parametrize('owned_prefix,foreign_site,editable', [
    (True, False, False), (False, False, False), (True, True, False), (True, False, True), (False, False, True)])
def test_fresh_venv_metadata_is_allowed_only_inside_this_execution(prepared, monkeypatch, tmp_path, owned_prefix, foreign_site, editable):
    root = prepared['view']
    purelib = tmp_path / 'foreign-site' if foreign_site else root / '.venv/lib/site-packages'
    purelib.mkdir(parents=True)
    for dist in (root / '.mindie-runtime/metadata').iterdir():
        if editable:
            package = dist.name.split('-')[0]
            source = 'vllm-ascend' if package == 'vllm_ascend' else 'vllm'
            moved = dist.rename(root / source / (package + '.egg-info'))
            (moved / 'METADATA').rename(moved / 'PKG-INFO')
        else:
            dist.rename(purelib / dist.name)
    monkeypatch.syspath_prepend(str(purelib))
    monkeypatch.setattr(sys, 'prefix', str(root / '.venv' if owned_prefix else tmp_path / 'donor/.venv'))
    monkeypatch.setattr(profile.sysconfig, 'get_paths', lambda: {'purelib': str(purelib)})
    if owned_prefix and not foreign_site:
        assert profile.native_source_mapping(root)['vllm_version'] == '2.0'
    else:
        with pytest.raises(ValueError, match='metadata escaped'):
            profile.native_source_mapping(root)


@pytest.mark.parametrize('failure', [None, 'capture-cancel', 'store-cancel', 'store-unknown'])
def test_fresh_handoff_waits_for_store_and_cancellation_checks(receipt, monkeypatch, failure):
    manifest, _, spec = receipt
    backend = adapters.RemoteBackend()
    spec.update(user='alice')
    snapshot = {'id': 'fixed', 'records': [{'relpath': name, 'scm_version': '1.0', 'source_head': 'fixed-head'}
                                         for name in ('vllm', 'vllm-ascend')]}
    monkeypatch.setattr('remote_dev.core.ssh_transport.run_rpc_script', lambda *a, **k: RemoteCompleted(0, '', ''))
    monkeypatch.setattr('mindie_coordinator.parity.materialize_fixed_sources', lambda **k: None)
    monkeypatch.setattr('mindie_coordinator.parity_support.ssh_exec_stream', lambda *a, **k: None)
    monkeypatch.setattr('mindie_coordinator.parity.run_runtime_install_step', lambda **k: None)
    monkeypatch.setattr(backend, 'bash', lambda *a, **k: '')
    monkeypatch.setattr(backend, '_export_shared_native', lambda *a, **k: {'status': 'miss'})
    steps, cancelled = [], [False]
    def capture(*a, **k):
        steps.append('capture')
        assert k['managed_finalize'] and k['publish_native_cache']
        assert k['process'].step == 'finalize-runtime'
        if failure == 'store-unknown':
            raise PreparationUncertain('unknown owned finalization during store')
        cancelled[0] = failure in {'capture-cancel', 'store-cancel'}
        return manifest
    def cache(spec, action, *a, **k):
        steps.append(action)
        assert action != 'store', 'publication belongs to the same owned finalization'
        return {'status': 'miss'}
    monkeypatch.setattr(backend, '_write_ready_profile', capture)
    monkeypatch.setattr(backend, '_shared_native', cache)
    def run():
        return backend.prepare_task_root(spec, sources={'vllm': '/a', 'vllm-ascend': '/b'}, environment={},
            source_snapshot=snapshot, on_preparation_job=lambda record: None, cancel_requested=lambda: cancelled[0])
    if failure:
        with pytest.raises(PreparationUncertain if failure == 'store-unknown' else PreparationCancelled): run()
        assert steps == ['restore', 'capture']
    else:
        result = run()
        assert isinstance(result, adapters.PreparedNativeView) and result.attestation is manifest
        assert steps == ['restore', 'capture']
