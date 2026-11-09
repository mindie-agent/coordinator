"""Trim only the internal wire reply; durable restore/rollback proof is complete."""
import copy
import json
import sys

import pytest

from mindie_coordinator import preparation_cache as cache
from test_shared_preparation import bundle


@pytest.mark.parametrize('incremental', [False, True])
def test_restore_wire_reply_keeps_fixed_bundle_and_dependencies_but_not_copy_list(bundle, monkeypatch, incremental):
    source, target, shared, preparation, manifest, relative, versions = bundle
    cache.store_shared_native(source, shared)
    preparation = copy.deepcopy(preparation)
    if incremental:
        preparation['native']['vllm-ascend'] = 'changed-kernel'
        monkeypatch.setattr(cache, 'kernel_rebuild_plan', lambda *args: {'operator': 'fixture-operator'})
    args = {'action': 'restore', 'root': str(target), 'cache': str(shared), 'preparation': preparation,
            'image_digest': 'sha256:same-image', 'versions': versions}
    monkeypatch.setattr(sys, 'argv', ['shared-native', json.dumps(args)])
    outputs = []
    namespace = vars(cache).copy()
    namespace['print'] = outputs.append
    exec(compile(cache.REMOTE_SHARED_SUFFIX, '<shared-native>', 'exec'), namespace)
    reply = json.loads(outputs[0])
    saved = json.loads((target / '.mindie-runtime/shared-native.json').read_text())
    assert reply['status'] == ('incremental' if incremental else 'hit')
    assert 'copied' not in reply and saved['copied'] == list(manifest['files'])
    for field in ('native_key', 'bundle', 'bundle_manifest_sha256', 'dependencies'):
        assert reply[field] == saved[field]
    if incremental:
        assert reply['operator'] == 'fixture-operator'
    assert cache.revalidate_shared_native(target, shared) == {'status': 'validated', 'bundle': reply['bundle']}
    original = (source / relative).read_bytes()
    assert (target / relative).read_bytes() == original
    cache.discard_shared_native(target)
    assert not (target / relative).exists()
    assert (source / relative).read_bytes() == original
    assert (shared / 'bundles' / reply['bundle'] / relative).read_bytes() == original


def test_large_internal_copy_list_is_not_emitted_or_removed_from_result_object(monkeypatch):
    copied = ['vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/test/kernel/' + str(i) + '.o'
              for i in range(1103)]
    receipt = {'status': 'hit', 'copied': copied, 'native_key': 'native', 'bundle': 'fixed',
               'bundle_manifest_sha256': 'a' * 64, 'dependencies': {'satisfied': True, 'versions': {'torch': '2.8'}}}
    monkeypatch.setattr(sys, 'argv', ['shared-native', json.dumps({'action': 'restore', 'root': '/execution',
        'preparation': {}, 'image_digest': 'image', 'versions': {}})])
    namespace = vars(cache).copy()
    outputs = []
    namespace.update(print=outputs.append, restore_shared_native=lambda *a, **k: receipt)
    exec(compile(cache.REMOTE_SHARED_SUFFIX, '<shared-native>', 'exec'), namespace)
    assert json.loads(outputs[0]) == {key: value for key, value in receipt.items() if key != 'copied'}
    assert receipt['copied'] is copied and len(copied) == 1103
    assert len(outputs[0]) < 1000 < len(json.dumps(receipt))


def _run_shared_suffix(monkeypatch, *, action='restore', restore=None, error=None):
    monkeypatch.setattr(sys, 'argv', ['shared-native', json.dumps({
        'action': action, 'root': '/execution', 'preparation': {},
        'image_digest': 'image', 'versions': {},
    })])
    namespace = vars(cache).copy()
    outputs = []
    def restore_fn(*a, **k):
        if error is not None:
            raise error
        return restore
    namespace.update(print=outputs.append, restore_shared_native=restore_fn)
    exec(compile(cache.REMOTE_SHARED_SUFFIX, '<shared-native>', 'exec'), namespace)
    return json.loads(outputs[0])


def test_shared_restore_disk_error_is_failed_not_miss(monkeypatch):
    reply = _run_shared_suffix(monkeypatch, error=OSError(28, 'No space left on device'))
    assert reply['status'] == 'failed'
    assert reply['error_type'] == 'OSError'
    assert 'No space left on device' in reply['reason']


def test_shared_restore_permission_error_is_failed_not_miss(monkeypatch):
    reply = _run_shared_suffix(monkeypatch, error=PermissionError('cache not writable'))
    assert reply['status'] == 'failed'
    assert 'PermissionError' in reply['reason']


def test_shared_restore_missing_file_is_failed_not_miss(monkeypatch):
    reply = _run_shared_suffix(monkeypatch, error=FileNotFoundError('bundle missing'))
    assert reply['status'] == 'failed'
    assert reply['error_type'] == 'FileNotFoundError'
    assert 'bundle missing' in reply['reason']


def test_shared_restore_corrupt_index_is_failed_not_miss(monkeypatch):
    reply = _run_shared_suffix(monkeypatch, error=json.JSONDecodeError('Expecting value', '', 0))
    assert reply['status'] == 'failed'
    assert reply['error_type'] == 'JSONDecodeError'


def test_shared_restore_abi_mismatch_is_still_miss(monkeypatch):
    reply = _run_shared_suffix(monkeypatch, error=ValueError('cached Python ABI differs'))
    assert reply['status'] == 'miss'
    assert 'Python ABI' in reply['reason']


def test_shared_restore_integrity_valueerror_is_failed_not_miss(monkeypatch):
    reply = _run_shared_suffix(monkeypatch, error=ValueError('invalid shared native cache pointer'))
    assert reply['status'] == 'failed'
    assert reply['error_type'] == 'ValueError'
    assert 'invalid shared native cache pointer' in reply['reason']


def test_shared_restore_timeout_is_not_converted_to_miss(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['shared-native', json.dumps({
        'action': 'restore', 'root': '/execution', 'preparation': {},
        'image_digest': 'image', 'versions': {},
    })])
    namespace = vars(cache).copy()
    namespace['restore_shared_native'] = lambda *a, **k: (_ for _ in ()).throw(TimeoutError('cache read timed out'))
    with pytest.raises(TimeoutError, match='timed out'):
        exec(compile(cache.REMOTE_SHARED_SUFFIX, '<shared-native>', 'exec'), namespace)
