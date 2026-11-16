"""Native content-hash pool behavior: reuse, invalidation, linking, receipts."""
from __future__ import annotations

import contextlib
import copy
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest import mock

import pytest

from mindie_coordinator import native_cache_pool, preparation_cache as cache
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
    monkeypatch.setattr(native_cache_pool.os, 'link', mock.Mock(side_effect=OSError(errno.EXDEV, 'cross-device')))
    monkeypatch.setattr(cache.shutil, 'copy2', fallback)
    assert cache._link_or_copy(source, target) is False
    fallback.assert_called_once_with(source, target)

    fallback.reset_mock()
    monkeypatch.setattr(native_cache_pool.os, 'link', mock.Mock(side_effect=OSError(errno.EIO, 'io failure')))
    with pytest.raises(OSError) as raised:
        cache._link_or_copy(source, target)
    assert raised.value.errno == errno.EIO
    fallback.assert_not_called()


def test_restore_falls_back_to_copy_when_hardlink_unsupported(bundle, monkeypatch):
    source, target, shared, request, manifest, relative, versions = bundle
    cache.store_shared_native(source, shared)
    monkeypatch.setattr(native_cache_pool.os, 'link', mock.Mock(side_effect=OSError(errno.EXDEV, 'cross-device')))
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


def _kernel_cache_entry(tmp_path):
    from mindie_coordinator import native_kernel_recipe as recipe

    root, pool = tmp_path / 'owned', tmp_path / 'pool'
    content_cache = pool / 'kernel'
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
    recipe.publish_kernel_recipe_content(
        root, plan, content_cache, key, [read_recipe(destination / 'gen/fixture.sh')])
    return recipe, root, pool, content_cache, destination, plan, key, read_recipe


def test_kernel_restore_shared_lock_blocks_concurrent_eviction(tmp_path):
    recipe, _root, pool, content_cache, _destination, _plan, key, _read = _kernel_cache_entry(tmp_path)
    with recipe._kernel_entry_lock(content_cache, key) as acquired:
        assert acquired
        result = cache.evict_native_cache(pool, keep=set(), max_entries=1,
                                          max_total_bytes=1)
        assert result['removed'] == []
        assert (content_cache / key).is_dir()
    assert (content_cache / key).is_dir()


def test_kernel_restore_failure_cleans_partial_output_and_falls_back(tmp_path, monkeypatch):
    import errno

    recipe, root, pool, content_cache, destination, plan, key, read_recipe = _kernel_cache_entry(tmp_path)
    shutil.rmtree(destination)
    original_copy = recipe.shutil.copy2
    calls = {'count': 0}

    def faulty_copy(source, target):
        calls['count'] += 1
        if calls['count'] == 2:
            raise OSError(errno.EIO, 'injected restore failure')
        return original_copy(source, target)

    monkeypatch.setattr(recipe.os, 'link',
                        lambda *args: (_ for _ in ()).throw(OSError(errno.EXDEV, 'cross-device')))
    monkeypatch.setattr(recipe.shutil, 'copy2', faulty_copy)
    with pytest.raises(OSError, match='injected restore failure'):
        recipe.restore_kernel_recipe_content(root, plan, content_cache, key, read_recipe=read_recipe)
    assert not destination.exists()
    if destination.parent.exists():
        assert not list(destination.parent.iterdir())

    # A stale partial destination must not make the compile fallback raise
    # FileExistsError at its own mkdir boundary.
    (destination / 'bin/fixture').mkdir(parents=True)
    (destination / 'bin/fixture/partial.o').write_bytes(b'partial')
    monkeypatch.setattr(recipe, '_recipe_inputs', lambda *args: ([], 'dynamic.py', 'ini'))
    monkeypatch.setattr(recipe, 'kernel_content_key', lambda *args, **kwargs: key)
    monkeypatch.setattr(recipe.tempfile, 'mkdtemp',
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            RuntimeError('compiler path reached')))
    with pytest.raises(RuntimeError, match='compiler path reached'):
        recipe.compile_kernel_recipe(
            root, plan, {}, read_recipe=read_recipe, content_cache=content_cache,
            content_metadata={'cann_version': '8.2.RC1', 'soc_version': 'Ascend910B',
                              'python_abi': 'cp312'})
    assert not destination.exists()


def test_cache_scan_never_follows_symlinked_subroots(tmp_path):
    cache_root = tmp_path / 'cache'
    cache_root.mkdir()
    outside = tmp_path / 'outside'
    outside_entry = outside / ('a' * 64)
    outside_entry.mkdir(parents=True)
    (outside_entry / 'blob').write_bytes(b'x' * 10)
    try:
        (cache_root / 'bundles').symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip('directory symlinks are unavailable')
    result = cache.evict_native_cache(cache_root, keep=set(), max_entries=1,
                                      max_total_bytes=1)
    assert result['removed'] == []
    assert (outside_entry / 'blob').read_bytes() == b'x' * 10
    link = tmp_path / 'cache-link'
    link.symlink_to(cache_root, target_is_directory=True)
    assert cache.evict_native_cache(link, keep=set(), max_entries=1,
                                    max_total_bytes=1)['status'] == 'unsafe'


def test_trash_removal_clears_read_only_files(tmp_path):
    import stat

    cache_root = tmp_path / 'cache'
    entry = cache_root / ('a' * 64)
    entry.mkdir(parents=True)
    blob = entry / 'blob'
    blob.write_bytes(b'x' * 10)
    blob.chmod(stat.S_IREAD)
    result = cache.evict_native_cache(cache_root, keep=set(), max_entries=1,
                                      max_total_bytes=1)
    assert result['status'] == 'evicted'
    assert not entry.exists()
    assert not list(cache_root.glob('.trash-*'))
    assert result['entries'] == 0 and result['bytes'] == 0


def test_undeleted_trash_keeps_honest_entry_and_byte_accounting(tmp_path, monkeypatch):
    import errno
    from mindie_coordinator import native_cache_pool

    cache_root = tmp_path / 'cache'
    for name in ('a' * 64, 'b' * 64):
        entry = cache_root / name
        entry.mkdir(parents=True)
        (entry / 'blob').write_bytes(b'x' * 10)
    monkeypatch.setattr(native_cache_pool, '_remove_tree',
                        lambda path: (_ for _ in ()).throw(OSError(errno.EIO, 'trash locked')))
    result = cache.evict_native_cache(cache_root, keep=set(), max_entries=1,
                                      max_total_bytes=1)
    assert result['removed'] == []
    assert result['entries'] == 2
    assert result['bytes'] == 20
    assert len(list(cache_root.glob('.trash-*'))) == 2


def test_eviction_is_fail_closed_when_cache_lock_cannot_be_created(tmp_path):
    cache_root = tmp_path / 'cache'
    entry = cache_root / ('a' * 64)
    entry.mkdir(parents=True)
    (entry / 'blob').write_bytes(b'x' * 10)
    # A regular file where the `.locks` directory belongs makes lock-file
    # creation fail; destructive eviction must not run unlocked.
    (cache_root / '.locks').write_text('not a directory', encoding='utf-8')
    result = cache.evict_native_cache(cache_root, keep=set(), max_entries=1,
                                      max_total_bytes=1)
    assert result['status'] == 'busy'
    assert (entry / 'blob').read_bytes() == b'x' * 10


@contextlib.contextmanager
def _busy_cache_lock(*_args, **_kwargs):
    yield False


_EVICTION_CHILD = r'''
import json, sys
from pathlib import Path
from mindie_coordinator import native_cache_pool
pool, content_cache, key = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
result = native_cache_pool.evict_native_cache(
    pool, keep=set(), max_entries=1, max_total_bytes=1)
print(json.dumps({
    'result': result,
    'lock_path': str(native_cache_pool.cache_entry_lock_path(content_cache, 'kernel', key)),
    'pool_lock_path': str(native_cache_pool.cache_entry_lock_path(pool, 'kernel', key)),
    'exists': (content_cache / key).is_dir(),
}))
'''


def _run_eviction_child(tmp_path, pool, content_cache, key):
    env = {**os.environ}
    root = str(Path(__file__).resolve().parents[1])
    env['PYTHONPATH'] = root + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    process = subprocess.Popen([sys.executable, '-c', _EVICTION_CHILD, str(pool),
                                str(content_cache), key], cwd=str(tmp_path), env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        raise
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def test_kernel_restore_and_eviction_resolve_one_lock_file(tmp_path):
    recipe, _root, pool, content_cache, _destination, _plan, key, _read = _kernel_cache_entry(tmp_path)
    restore_lock = native_cache_pool.cache_entry_lock_path(content_cache, 'kernel', key)
    eviction_lock = native_cache_pool.cache_entry_lock_path(pool, 'kernel', key)
    assert restore_lock == eviction_lock
    assert restore_lock.parent == pool / '.locks'
    assert not restore_lock.is_relative_to(content_cache / '.locks')
    process_key = os.path.normcase(os.path.realpath(os.fspath(restore_lock)))
    with recipe._kernel_entry_lock(content_cache, key) as acquired:
        assert acquired is True
        assert process_key in native_cache_pool._PROCESS_LOCKS
    assert process_key not in native_cache_pool._PROCESS_LOCKS


def test_cross_process_kernel_restore_lock_blocks_eviction(tmp_path):
    recipe, _root, pool, content_cache, _destination, _plan, key, _read = _kernel_cache_entry(tmp_path)
    expected_lock = str(native_cache_pool.cache_entry_lock_path(pool, 'kernel', key))
    with recipe._kernel_entry_lock(content_cache, key) as acquired:
        assert acquired is True
        completed = _run_eviction_child(tmp_path, pool, content_cache, key)
        assert completed.returncode == 0, completed.stderr
        observed = json.loads(completed.stdout)
        assert observed['lock_path'] == observed['pool_lock_path'] == expected_lock
        assert observed['result']['removed'] == []
        assert observed['exists'] is True
        assert (content_cache / key).is_dir()
    completed = _run_eviction_child(tmp_path, pool, content_cache, key)
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert observed['result']['removed'] == ['kernel:' + key]
    assert observed['exists'] is False
    assert not (content_cache / key).exists()


def test_store_skips_cleanly_when_cache_lock_is_unavailable(bundle, monkeypatch):
    source, _target, shared, _request, manifest, _relative, _versions = bundle
    monkeypatch.setattr(cache, '_cache_operation', _busy_cache_lock)
    monkeypatch.setattr(cache, '_store_shared_native',
                        lambda *_args, **_kwargs: pytest.fail('store body ran without the cache lock'))
    result = cache.store_shared_native(source, shared)
    assert result['status'] == 'busy'
    assert result['linked_files'] == 0
    assert result['cache_key'] == ''
    assert not shared.exists()
    assert 'distributions' not in manifest


def test_captured_store_skips_cleanly_when_cache_lock_is_unavailable(bundle, monkeypatch):
    source, _target, shared, _request, manifest, _relative, _versions = bundle
    monkeypatch.setattr(cache, '_cache_operation', _busy_cache_lock)
    monkeypatch.setattr(cache, '_store_captured_native_unlocked',
                        lambda *_args, **_kwargs: pytest.fail('store body ran without the cache lock'))
    timings = {}
    result = cache._store_captured_native(source, shared, manifest, timings)
    assert result['status'] == 'busy'
    assert result['linked_files'] == 0
    assert not shared.exists()
    assert timings == {}


def test_restore_treats_unavailable_cache_lock_as_a_miss(bundle, monkeypatch):
    source, target, shared, request, _manifest, relative, versions = bundle
    cache.store_shared_native(source, shared)
    monkeypatch.setattr(cache, '_cache_operation', _busy_cache_lock)
    monkeypatch.setattr(cache, '_restore_shared_native_unlocked',
                        lambda *_args, **_kwargs: pytest.fail('restore body ran without the cache lock'))
    result = cache.restore_shared_native(target, shared, request, 'sha256:same-image', versions)
    assert result['status'] == 'miss'
    assert 'busy' in result['reason']
    assert isinstance(result['cache_key'], str)
    assert result['linked_files'] == 0
    assert not (target / relative).exists()
    assert not (target / '.mindie-runtime/shared-native.json').exists()


@pytest.mark.parametrize('snapshot, expected', [
    (None, FileNotFoundError),
    ('not-json', FileNotFoundError),
    ([], ValueError),
    ({'bundle': 'b' * 64, 'manifest_text': '{}'}, ValueError),
    ({'bundle': 'a' * 64, 'manifest_text': 42}, ValueError),
])
def test_revalidate_snapshot_failure_never_raises_unbound_local(tmp_path, snapshot, expected):
    root = tmp_path / 'root'
    name = 'a' * 64
    marker = root / '.mindie-runtime/shared-native.json'
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({'bundle': name, 'bundle_manifest_sha256': 'c' * 64,
                                  'native_key': 'native-key'}))
    if snapshot is not None:
        (root / '.mindie-runtime/shared-native-bundle.json').write_text(
            snapshot if isinstance(snapshot, str) else json.dumps(snapshot))
    with pytest.raises(expected) as raised:
        cache.revalidate_shared_native(root, tmp_path / 'cache')
    assert not isinstance(raised.value, UnboundLocalError)


@pytest.mark.parametrize('exc_info', [
    (TimeoutError, TimeoutError('injected lstat timeout'), None),
    TimeoutError('injected lstat timeout'),
])
def test_clear_readonly_never_swallows_timeout(exc_info):
    calls = []
    with pytest.raises(TimeoutError, match='injected lstat timeout'):
        native_cache_pool._clear_readonly(calls.append, '/tmp/cache-entry', exc_info)
    assert calls == []


@pytest.mark.parametrize('func, exc', [
    (os.lstat, PermissionError(errno.EACCES, 'denied')),
    (os.scandir, PermissionError(errno.EACCES, 'denied')),
    (os.listdir, PermissionError(errno.EACCES, 'denied')),
    (os.unlink, OSError(errno.EIO, 'io failure')),
])
def test_clear_readonly_only_retries_unlink_and_rmdir_permissions(func, exc):
    with pytest.raises(type(exc)) as raised:
        native_cache_pool._clear_readonly(func, '/tmp/cache-entry', (type(exc), exc, None))
    assert raised.value is exc


def test_rmtree_lstat_timeout_propagates_instead_of_false_deletion(tmp_path, monkeypatch):
    target = tmp_path / 'cache-entry'
    target.mkdir(parents=True)
    (target / 'blob').write_bytes(b'x' * 10)
    real_lstat = os.lstat

    def lstat(path, *args, **kwargs):
        if Path(path) == target:
            raise TimeoutError('injected lstat timeout')
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(native_cache_pool.os, 'lstat', lstat)
    with pytest.raises(TimeoutError):
        native_cache_pool._remove_tree(target)
    assert target.is_dir()
    assert (target / 'blob').read_bytes() == b'x' * 10


def test_remove_tree_rejects_a_silent_rmtree_noop(tmp_path, monkeypatch):
    target = tmp_path / 'cache-entry'
    target.mkdir(parents=True)
    (target / 'blob').write_bytes(b'x' * 10)
    monkeypatch.setattr(native_cache_pool.shutil, 'rmtree', lambda *_args, **_kwargs: None)
    with pytest.raises(OSError, match='still exists after removal'):
        native_cache_pool._remove_tree(target)
    assert (target / 'blob').read_bytes() == b'x' * 10


def test_eviction_does_not_subtract_bytes_when_the_target_survives(tmp_path, monkeypatch):
    cache_root = tmp_path / 'cache'
    entry = cache_root / ('a' * 64)
    entry.mkdir(parents=True)
    (entry / 'blob').write_bytes(b'x' * 10)
    # Simulate a remover that reports success without deleting the renamed tree.
    monkeypatch.setattr(native_cache_pool, '_remove_tree', lambda _path: None)
    result = cache.evict_native_cache(cache_root, keep=set(), max_entries=1,
                                      max_total_bytes=1)
    assert result['removed'] == []
    assert result['entries'] == 1
    assert result['bytes'] == 10
    assert not entry.exists()
    assert len(list(cache_root.glob('.trash-*'))) == 1


def test_eviction_does_not_subtract_bytes_when_unlink_fails(tmp_path, monkeypatch):
    cache_root = tmp_path / 'cache'
    entry = cache_root / ('a' * 64)
    entry.mkdir(parents=True)
    (entry / 'blob').write_bytes(b'x' * 10)
    real_unlink = os.unlink

    def failing_unlink(path, *args, **kwargs):
        if Path(path).name == 'blob':
            raise PermissionError(errno.EACCES, 'injected unlink failure')
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(native_cache_pool.os, 'unlink', failing_unlink)
    result = cache.evict_native_cache(cache_root, keep=set(), max_entries=1,
                                      max_total_bytes=1)
    assert result['removed'] == []
    assert result['entries'] == 1
    assert result['bytes'] == 10
    assert not entry.exists()
    assert len(list(cache_root.glob('.trash-*'))) == 1


def test_kernel_publish_lock_unavailable_skips_the_shared_write(tmp_path, monkeypatch):
    recipe, root, _pool, content_cache, _destination, plan, key, _read = _kernel_cache_entry(tmp_path)
    shutil.rmtree(content_cache / key)
    monkeypatch.setattr(recipe, '_kernel_entry_lock', _busy_cache_lock)
    result = recipe.publish_kernel_recipe_content(root, plan, content_cache, key, [])
    assert result == {'status': 'busy', 'cache_key': key}
    assert not (content_cache / key).exists()
    assert not list(content_cache.glob('.kernel-content-*'))
