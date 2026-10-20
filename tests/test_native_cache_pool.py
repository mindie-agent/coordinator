"""Native content-hash pool behavior: reuse, invalidation, linking, receipts."""
from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
from unittest import mock

import pytest

from mindie_coordinator import preparation_cache as cache
from test_shared_preparation import bundle as _bundle_fixture

bundle = _bundle_fixture


def _preparation(**overrides):
    preparation = {
        'source_id': 'source-a',
        'environment': {
            'cann': '8.2.RC1',
            'soc': 'Ascend910B',
            'python_abi': 'cp312',
            'build_env': {'CMAKE_BUILD_TYPE': 'Release', 'CXXFLAGS': '-O3'},
        },
        'native': {'vllm': 'v' * 64, 'vllm-ascend': 'a' * 64},
    }
    preparation.update(overrides)
    return preparation


def test_content_key_tracks_exact_native_tokens_and_ignores_task_workspace():
    baseline = cache.native_content_key(_preparation())
    # Source workspace/task identity never changes a compiled-byte key.
    assert cache.native_content_key(_preparation(source_id='source-b')) == baseline
    assert cache.native_content_key(_preparation(role='other-slot')) == baseline
    for path, replacement in (
        ('native', {'vllm': 'v' * 64, 'vllm-ascend': 'b' * 64}),
        ('environment', {'cann': '8.3.RC1', 'soc': 'Ascend910B', 'python_abi': 'cp312',
                         'build_env': {'CMAKE_BUILD_TYPE': 'Release', 'CXXFLAGS': '-O3'}}),
        ('environment', {'cann': '8.2.RC1', 'soc': 'Ascend910C', 'python_abi': 'cp312',
                         'build_env': {'CMAKE_BUILD_TYPE': 'Release', 'CXXFLAGS': '-O3'}}),
        ('environment', {'cann': '8.2.RC1', 'soc': 'Ascend910B', 'python_abi': 'cp313',
                         'build_env': {'CMAKE_BUILD_TYPE': 'Release', 'CXXFLAGS': '-O3'}}),
        ('environment', {'cann': '8.2.RC1', 'soc': 'Ascend910B', 'python_abi': 'cp312',
                         'build_env': {'CMAKE_BUILD_TYPE': 'Release', 'CXXFLAGS': '-O2'}}),
    ):
        changed = _preparation()
        changed[path] = replacement
        assert cache.native_content_key(changed) != baseline


def test_content_key_requires_measured_cann_soc_and_abi():
    for field in ('cann', 'soc', 'python_abi'):
        broken = _preparation()
        broken['environment'][field] = None
        with pytest.raises(ValueError, match=field):
            cache.native_content_key(broken)


def test_store_restore_uses_content_pool_and_hardlinks_execution_view(bundle):
    source, target, shared, request, manifest, relative, versions = bundle
    stored = cache.store_shared_native(source, shared)
    key = stored['cache_key']
    assert len(key) == 64
    entry = shared / key
    assert (entry / 'native-content.json').is_file()
    assert cache.native_content_key(manifest['preparation']) == key
    # The replaceable pool entry is a copy, not a hard link into the immutable
    # verified bundle; a writable view can only damage the former.
    bundle_file = next((shared / 'bundles').iterdir()) / relative
    assert os.stat(bundle_file).st_ino != os.stat(entry / relative).st_ino

    restored = cache.restore_shared_native(target, shared, request, 'sha256:same-image', versions)
    assert restored['status'] == 'hit'
    assert restored['cache_key'] == key
    assert restored['linked_files'] == len(manifest['files'])
    assert os.stat(target / relative).st_ino == os.stat(entry / relative).st_ino
    assert cache.native_cache_receipt(restored) == {
        'hit': True, 'cache_key': key, 'linked_files': len(manifest['files'])}


def test_corrupt_pool_entry_is_repaired_from_verified_bundle_not_trusted(bundle):
    source, target, shared, request, manifest, relative, versions = bundle
    stored = cache.store_shared_native(source, shared)
    key = stored['cache_key']
    entry = shared / key
    original = (entry / relative).read_bytes()
    first = cache.restore_shared_native(target, shared, request, 'sha256:same-image', versions)
    assert first['status'] == 'hit'
    # In-place write through the writable view damages the replaceable pool
    # inode, exactly the hazard hard links introduce.
    (target / relative).write_bytes(b'corrupted through slot hardlink')
    assert (entry / relative).read_bytes() != original
    bundle_file = next((shared / 'bundles').iterdir()) / relative
    assert bundle_file.read_bytes() == original

    replacement = target.parent / 'replacement-view'
    (replacement / 'vllm/vllm').mkdir(parents=True)
    (replacement / 'vllm-ascend/vllm_ascend').mkdir(parents=True)
    repaired = cache.restore_shared_native(replacement, shared, request, 'sha256:same-image', versions)
    assert repaired['status'] == 'hit'
    assert repaired['cache_key'] == key
    assert (replacement / relative).read_bytes() == original
    assert (entry / relative).read_bytes() == original


def test_same_content_key_with_different_artifacts_is_refused(bundle):
    source, target, shared, request, manifest, relative, versions = bundle
    stored = cache.store_shared_native(source, shared)
    bundle_dir = next((shared / 'bundles').iterdir())
    altered = copy.deepcopy(manifest)
    altered['files'][relative]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='same native content key produced different artifacts'):
        cache.publish_native_content(bundle_dir, shared, altered)
    assert (shared / stored['cache_key'] / relative).read_bytes() == (source / relative).read_bytes()


@pytest.mark.parametrize('different', [False, True])
def test_native_publication_race_compares_the_winners_actual_bytes(bundle, monkeypatch, different):
    source, _, shared, _, _, relative, _ = bundle
    stored = cache.store_shared_native(source, shared)
    original = shared / 'bundles' / stored['bundle']
    manifest = json.loads((original / 'manifest.json').read_text())
    winner = shared / 'other-verified-bundle'
    shutil.copytree(original, winner)
    winner_manifest = copy.deepcopy(manifest)
    if different:
        metadata = next(name for name, row in manifest['files'].items() if row['role'] == 'metadata')
        path = winner / metadata
        path.write_bytes(path.read_bytes() + b'\n# second publisher output\n')
        winner_manifest['files'][metadata]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        (winner / 'manifest.json').write_text(json.dumps(winner_manifest))
    cache.verify(winner, winner_manifest, check_environment=False)
    pool = shared / 'race-pool'
    destination = pool / stored['cache_key']
    rename = os.rename
    intervened = False

    def interleave(source_path, destination_path):
        nonlocal intervened
        if Path(destination_path) == destination and not intervened:
            intervened = True
            cache.publish_native_content(winner, pool, winner_manifest)
        return rename(source_path, destination_path)

    monkeypatch.setattr(os, 'rename', interleave)
    if different:
        with pytest.raises(ValueError, match='different artifacts'):
            cache.publish_native_content(original, pool, manifest)
    else:
        assert cache.publish_native_content(original, pool, manifest)['status'] == 'present'
    assert intervened
    assert json.loads((destination / 'manifest.json').read_text()) == winner_manifest
    assert (destination / relative).read_bytes() == (winner / relative).read_bytes()
    assert not list(pool.glob('.native-content-*'))


@pytest.mark.parametrize('race', [False, True])
@pytest.mark.parametrize('different', [False, True])
def test_kernel_publication_compares_output_for_the_same_inputs(tmp_path, monkeypatch, race, different):
    from mindie_coordinator import native_kernel_recipe as recipe

    plan = {'operator': 'fixture', 'unit': 'ascend910b', 'source': 'csrc/fixture.cpp',
            'source_sha256': 'b' * 64, 'native_from': 'a' * 64, 'native_to': 'a' * 64}
    roots = [tmp_path / name for name in ('first', 'second')]
    for index, root in enumerate(roots):
        binary = root / 'vllm-ascend/csrc/build/binary/ascend910b'
        (binary / 'bin').mkdir(parents=True)
        (binary / 'gen').mkdir()
        # Byte publication contract only; this is not an Ascend compiler fixture.
        (binary / 'bin/output').write_bytes(f'publisher {index if different else 0}'.encode())
        (binary / 'gen/recipe.sh').write_text('same recipe\n')
    pool = tmp_path / 'pool'
    key = recipe.kernel_content_key(plan, {}, cann_version='8.2.RC1',
                                    soc_version='Ascend910B', python_abi='cp312')
    recipes = [{'fixture': 'same inputs'}]
    rename = os.rename
    intervened = False

    def interleave(source_path, destination_path):
        nonlocal intervened
        if Path(destination_path) == pool / key and not intervened:
            intervened = True
            recipe.publish_kernel_recipe_content(roots[1], plan, pool, key, recipes)
        return rename(source_path, destination_path)

    if race:
        monkeypatch.setattr(os, 'rename', interleave)
    else:
        recipe.publish_kernel_recipe_content(roots[1], plan, pool, key, recipes)
    if different:
        with pytest.raises(ValueError, match='different artifacts'):
            recipe.publish_kernel_recipe_content(roots[0], plan, pool, key, recipes)
    else:
        assert recipe.publish_kernel_recipe_content(roots[0], plan, pool, key, recipes)['status'] == 'present'
    assert (pool / key / 'bin/output').read_bytes() == (b'publisher 1' if different else b'publisher 0')
    assert not list(pool.glob('.kernel-content-*'))


def test_hardlink_fallback_is_limited_to_unsupported_filesystems(monkeypatch):
    source = cache.Path('/cache/native.bin')
    target = cache.Path('/view/native.bin')
    fallback = mock.Mock()
    monkeypatch.setattr(cache.os, 'link', mock.Mock(side_effect=OSError(errno.EXDEV, 'cross-device')))
    monkeypatch.setattr(cache.shutil, 'copy2', fallback)
    assert cache._link_or_copy(source, target) is False
    fallback.assert_called_once_with(source, target)

    fallback.reset_mock()
    monkeypatch.setattr(cache.os, 'link', mock.Mock(side_effect=OSError(errno.EIO, 'io failure')))
    with pytest.raises(OSError) as raised:
        cache._link_or_copy(source, target)
    assert raised.value.errno == errno.EIO
    fallback.assert_not_called()


def test_restore_falls_back_to_copy_when_hardlink_unsupported(bundle, monkeypatch):
    source, target, shared, request, manifest, relative, versions = bundle
    cache.store_shared_native(source, shared)
    monkeypatch.setattr(cache.os, 'link', mock.Mock(side_effect=OSError(errno.EXDEV, 'cross-device')))
    restored = cache.restore_shared_native(target, shared, request, 'sha256:same-image', versions)
    assert restored['status'] == 'hit'
    assert restored['linked_files'] == 0
    assert (target / relative).read_bytes() == (source / relative).read_bytes()


def test_corrupt_source_bundle_never_produces_a_hit(bundle, monkeypatch):
    source, target, shared, request, manifest, relative, versions = bundle
    cache.store_shared_native(source, shared)
    bundle_dir = next((shared / 'bundles').iterdir())
    (bundle_dir / relative).write_bytes(b'changed after publication')
    with pytest.raises(ValueError):
        cache.restore_shared_native(target, shared, request, 'sha256:same-image', versions)
    assert not (target / relative).exists()


def test_content_receipt_shape_is_explicit_for_miss_and_incremental():
    assert cache.native_cache_receipt({'status': 'miss', 'cache_key': 'k' * 64}) == {
        'hit': False, 'cache_key': 'k' * 64, 'linked_files': 0}
    assert cache.native_cache_receipt({'status': 'incremental', 'cache_key': 'k' * 64,
                                       'linked_files': 4}) == {
        'hit': True, 'cache_key': 'k' * 64, 'linked_files': 4}


def test_kernel_recipe_content_key_reuses_the_same_native_tokens():
    from mindie_coordinator import native_kernel_recipe as recipe

    plan = {'operator': 'fixture', 'unit': 'ascend910b', 'native_to': 'a' * 64,
            'source_sha256': 'b' * 64}
    environment = {'CMAKE_ARGS': '-DCMAKE_BUILD_TYPE=Release', 'CXXFLAGS': '-O3'}
    baseline = recipe.kernel_content_key(plan, environment, cann_version='8.2.RC1',
                                         soc_version='Ascend910B', python_abi='cp312')
    assert recipe.kernel_content_key(plan, dict(environment), cann_version='8.2.RC1',
                                     soc_version='Ascend910B', python_abi='cp312') == baseline
    assert recipe.kernel_content_key({**plan, 'native_to': 'c' * 64}, environment,
                                     cann_version='8.2.RC1', soc_version='Ascend910B',
                                     python_abi='cp312') != baseline
    assert recipe.kernel_content_key(plan, {**environment, 'CXXFLAGS': '-O2'},
                                     cann_version='8.2.RC1', soc_version='Ascend910B',
                                     python_abi='cp312') != baseline
    assert recipe.kernel_content_key(plan, environment, cann_version='8.3.RC1',
                                     soc_version='Ascend910B', python_abi='cp312') != baseline
    assert recipe.kernel_content_key(plan, environment, cann_version='8.2.RC1',
                                     soc_version='Ascend910C', python_abi='cp312') != baseline


def test_kernel_recipe_content_publishes_and_restores_actual_outputs(tmp_path):
    from mindie_coordinator import native_kernel_recipe as recipe

    root, cache = tmp_path / 'owned', tmp_path / 'cache'
    destination = root / 'vllm-ascend/csrc/build/binary/ascend910b'
    (destination / 'bin/fixture').mkdir(parents=True)
    (destination / 'bin/fixture/fixture.o').write_bytes(b'compiled object')
    (destination / 'gen').mkdir()
    (destination / 'gen/fixture.sh').write_text('opc --fixture\n')
    plan = {'operator': 'fixture', 'unit': 'ascend910b', 'source': 'csrc/fixture.cpp',
            'source_sha256': 'b' * 64, 'native_from': 'a' * 64, 'native_to': 'a' * 64}

    def read_recipe(path):
        return {'name': path.name}

    key = recipe.kernel_content_key(plan, {}, cann_version='8.2.RC1',
                                    soc_version='Ascend910B', python_abi='cp312')
    published = recipe.publish_kernel_recipe_content(root, plan, cache, key, [read_recipe(destination / 'gen/fixture.sh')])
    assert published['status'] == 'published'
    assert (cache / key / 'bin/fixture/fixture.o').read_bytes() == b'compiled object'
    with pytest.raises(ValueError, match='different compile'):
        recipe.publish_kernel_recipe_content(
            root, {**plan, 'operator': 'other'}, cache, key,
            [read_recipe(destination / 'gen/fixture.sh')])
    import shutil
    shutil.rmtree(destination)
    assert recipe.restore_kernel_recipe_content(root, plan, cache, key, read_recipe=read_recipe) is True
    assert (destination / 'bin/fixture/fixture.o').read_bytes() == b'compiled object'
    assert (destination / 'gen/fixture.sh').read_text() == 'opc --fixture\n'
    # A same-name entry whose bytes changed is refused, not returned as a hit.
    (cache / key / 'bin/fixture/fixture.o').write_bytes(b'changed object')
    shutil.rmtree(destination)
    with pytest.raises(ValueError, match='bytes changed'):
        recipe.restore_kernel_recipe_content(root, plan, cache, key, read_recipe=read_recipe)


def test_remote_concatenation_publishes_and_reuses_actual_bundle(bundle):
    from mindie_coordinator import native_cache_pool, preparation_cache, runtime_profile

    namespace = {}
    for module in (runtime_profile, native_cache_pool, preparation_cache):
        exec(compile(Path(module.__file__).read_text(), str(module.__file__), "exec"), namespace)
    source, _, shared, _, manifest, relative, _ = bundle
    stored = cache.store_shared_native(source, shared)
    immutable = shared / 'bundles' / stored['bundle']
    manifest = json.loads((immutable / 'manifest.json').read_text())
    other_pool = shared / 'remote-pool'
    published = namespace['publish_native_content'](immutable, other_pool, manifest)
    reused = namespace['publish_native_content'](immutable, other_pool, manifest)
    assert published['status'] == 'published' and reused['status'] == 'present'
    assert (other_pool / published['cache_key'] / relative).read_bytes() == (source / relative).read_bytes()
