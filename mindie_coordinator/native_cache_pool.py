"""Content-addressed native pool: publish, verify, link and invalidate bytes.

[Role]
Owns the `/tmp/mindie-native-cache/<key>` pool mechanics used by
`preparation_cache`: key material, atomic publication, hash verification,
hard-link mapping with copy fallback, corruption repair and cache receipts.

[Invariants]
- The verified bundle is immutable and authoritative; a pool entry is only a
  replaceable copy keyed by `sha256(csrc tree + CANN + SoC + ABI + flags)`.
- A hit re-hashes the pool entry and never trusts a matching name.
- A valid entry with a different artifact set is an integrity conflict, not a
  hit; it is never overwritten or silently preferred.
- Only EXDEV/EPERM/EACCES/ENOTSUP/EMLINK/EINVAL (and cross-volume Windows
  errors) fall back from `os.link` to `copy2`; all other OSErrors propagate.
- Global eviction is bounded by entry count/bytes and never removes pinned,
  locked, or content-referenced immutable entries.

[Timeouts & Locks]
- Cross-process cache locks use `fcntl.flock` on POSIX and `msvcrt.locking` on
  Windows; failure to create/acquire a lock is fail-closed and skips eviction.
- `cache-root` is a shared lock during restore/store and a non-blocking
  exclusive lock during eviction; every entry has its own shared/exclusive lock.
- Every `(kind, name)` entry lock resolves to `<cache-root>/.locks/` and
  `_PROCESS_LOCKS` is keyed by that resolved path, matching cross-process locks.
- Publication uses a private `mkdtemp` plus `.staging-*` sibling lock and one
  `os.rename`; no partial key is visible. Staging is swept only after a
  non-blocking builder-lock acquisition and the age grace both succeed.
- `_remove_tree` propagates TimeoutError/non-retryable OSError and verifies the
  path is gone, so eviction never reports bytes it did not reclaim.

[Side-Effect Policy]
- Writes only under the supplied cache root and returns paths/receipts.
"""
from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

try:  # Local package use; remote concatenation executes runtime_profile first.
    from .runtime_profile import checked_file, verify
except (ImportError, KeyError):  # pragma: no cover - concatenated namespace
    # The remote namespace already contains these runtime_profile functions.
    pass

NATIVE_CONTENT_SCHEMA = 'mindie-native-content/1'
NATIVE_CONTENT_MANIFEST = 'native-content.json'
_HARDLINK_FALLBACK_ERRNOS = {
    value for value in (
        getattr(errno, 'EXDEV', None), getattr(errno, 'EPERM', None),
        getattr(errno, 'EACCES', None), getattr(errno, 'ENOTSUP', None),
        getattr(errno, 'EOPNOTSUPP', None), getattr(errno, 'EMLINK', None),
        getattr(errno, 'EINVAL', None),
    ) if value is not None
}
_HARDLINK_FALLBACK_WINERRORS = {1, 17, 50}

NATIVE_CACHE_MAX_ENTRIES = 16
NATIVE_CACHE_MAX_BYTES = 8 * 1024 ** 3
CACHE_STAGING_MAX_AGE = 3600
_CACHE_STAGING_PREFIXES = ('.publish-', '.native-content-', '.kernel-content-',
                           '.trash-', '.index-')
_ENTRY_NAME = re.compile(r'[0-9a-f]{64}\Z')


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def safe_destination(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('unsafe native cache destination')
    return root / path


def native_content_key_material(preparation: dict) -> dict:
    """Exact tokens identifying compiled native bytes; no task/donor identity."""
    environment = preparation.get('environment') or {}
    native = preparation.get('native') or {}
    if not isinstance(environment, dict) or not isinstance(native, dict):
        raise ValueError('native content key requires preparation environment and native inputs')
    if not native.get('vllm-ascend'):
        raise ValueError('native content key requires the vllm-ascend csrc tree')
    csrc = {name: native.get(name) for name in ('vllm', 'vllm-ascend') if native.get(name)}
    build_env = environment.get('build_env') or {}
    if not isinstance(build_env, dict) or any(not isinstance(key, str) for key in build_env):
        raise ValueError('native content key requires string-keyed compile flags')
    material = {
        'csrc_tree_sha': hashlib.sha256(_canonical_json(csrc).encode()).hexdigest(),
        'cann_version': environment.get('cann'),
        'soc_version': environment.get('soc'),
        'python_abi': environment.get('python_abi'),
        'compile_flags': {key: build_env[key] for key in sorted(build_env)},
    }
    missing = [name for name in ('cann_version', 'soc_version', 'python_abi') if not material[name]]
    if missing:
        raise ValueError('native content key is missing measured ' + ', '.join(missing))
    return material


def native_content_key(preparation: dict) -> str:
    """sha256(csrc_tree_sha + cann_version + soc_version + python_abi + flags)."""
    material = native_content_key_material(preparation)
    tokens = [NATIVE_CONTENT_SCHEMA, material['csrc_tree_sha'], str(material['cann_version']),
              str(material['soc_version']), str(material['python_abi']),
              _canonical_json(material['compile_flags'])]
    return hashlib.sha256('\x00'.join(tokens).encode()).hexdigest()


def native_cache_receipt(result: dict | None) -> dict:
    """The only Run Manifest native-cache evidence shape."""
    result = result or {}
    return {
        'hit': result.get('status') in {'hit', 'incremental'},
        'cache_key': str(result.get('cache_key') or ''),
        'linked_files': int(result.get('linked_files') or 0),
    }


def cache_busy_result(preparation: dict | None = None) -> dict:
    """Honest result when a shared cache lock could not be acquired."""
    try:
        key = native_content_key(preparation) if preparation is not None else ''
    except (TypeError, ValueError):
        key = ''
    return {'status': 'miss' if preparation is not None else 'busy',
            'reason': 'shared native cache is busy', 'bundle': '',
            'cache_key': key, 'linked_files': 0}


def _link_or_copy(source: Path, target: Path) -> bool:
    try:
        os.link(source, target)
        return True
    except TimeoutError:
        raise
    except OSError as exc:
        unsupported = exc.errno in _HARDLINK_FALLBACK_ERRNOS or getattr(exc, 'winerror', None) in _HARDLINK_FALLBACK_WINERRORS
        if not unsupported:
            raise
    shutil.copy2(source, target)
    return False


def _is_link(path: Path) -> bool:
    """True for POSIX symlinks and NTFS junctions/reparse points."""
    try:
        if path.is_symlink():
            return True
        isjunction = getattr(os.path, 'isjunction', None)
        return bool(isjunction is not None and isjunction(path))
    except TimeoutError:
        raise
    except OSError:
        # An unreadable path cannot prove it is a real directory.
        return True


def _lock_byte(handle) -> None:
    """msvcrt locks a byte range; every advisory lock file carries one byte."""
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b'\0')
        handle.flush()
    handle.seek(0)


def _lock_handle(handle, *, exclusive: bool, blocking: bool) -> None:
    _lock_byte(handle)
    if os.name == 'nt':  # pragma: no cover - exercised on Windows CI
        import msvcrt
        mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
        msvcrt.locking(handle.fileno(), mode, 1)
        return
    import fcntl
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    if not blocking:
        operation |= fcntl.LOCK_NB
    fcntl.flock(handle.fileno(), operation)


def _unlock_handle(handle) -> None:
    try:
        if os.name == 'nt':  # pragma: no cover - exercised on Windows CI
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except TimeoutError:
        raise
    except OSError:
        pass


def cache_lock_file(cache: Path, name: str) -> Path:
    """Canonical `<cache>/.locks/<sha256(name)>.lock` path."""
    digest = hashlib.sha256(name.encode()).hexdigest()
    return Path(cache) / '.locks' / (digest + '.lock')


def cache_entry_lock_path(cache: Path, kind: str, name: str) -> Path:
    """Canonical entry lock shared by publish, restore and eviction.

    `native_kernel_recipe` receives the `<root>/kernel` content sub-cache while
    eviction receives `<root>`; both must resolve to `<root>/.locks/`, or a
    cross-process eviction can rename the entry during restore.
    """
    cache = Path(cache)
    if kind == 'kernel' and cache.name == 'kernel':
        cache = cache.parent
    return cache_lock_file(cache, 'entry-' + kind + '-' + name)


@contextlib.contextmanager
def _advisory_lock_file(lock_path: Path, *, exclusive: bool, blocking: bool = True):
    """Yield True only while a real cross-process advisory lock is held.

    Lock creation or acquisition failure is fail-closed: callers that delete
    cache bytes skip the operation instead of running unlocked.
    """
    handle = None
    acquired = False
    lock_path = Path(lock_path)
    try:
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = lock_path.open('a+b')
        except TimeoutError:
            raise
        except OSError:
            yield False
            return
        try:
            _lock_handle(handle, exclusive=exclusive, blocking=blocking)
            acquired = True
        except TimeoutError:
            raise
        except OSError:
            yield False
            return
        yield True
    finally:
        if handle is not None:
            if acquired:
                _unlock_handle(handle)
            handle.close()


def _advisory_lock(cache: Path, name: str, *, exclusive: bool, blocking: bool = True):
    return _advisory_lock_file(cache_lock_file(cache, name), exclusive=exclusive,
                               blocking=blocking)


_PROCESS_LOCKS: dict[str, dict] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


@contextlib.contextmanager
def _cache_entry_lock(cache: Path, kind: str, name: str, *, exclusive: bool = False,
                      blocking: bool = True):
    """Exclude same-process callers with the same topology as the file lock."""
    lock_path = cache_entry_lock_path(cache, kind, name)
    token = os.path.normcase(os.path.realpath(os.fspath(lock_path)))
    with _PROCESS_LOCKS_GUARD:
        state = _PROCESS_LOCKS.setdefault(token, {'lock': threading.Lock(), 'users': 0})
        state['users'] += 1
    lock = state['lock']
    acquired = lock.acquire(blocking)
    try:
        if not acquired:
            yield False
            return
        try:
            with _advisory_lock_file(lock_path, exclusive=exclusive,
                                     blocking=blocking) as file_acquired:
                yield file_acquired
        finally:
            lock.release()
    finally:
        with _PROCESS_LOCKS_GUARD:
            state['users'] -= 1
            if state['users'] == 0:
                _PROCESS_LOCKS.pop(token, None)


def _cache_operation(cache: Path):
    return _advisory_lock(Path(cache), 'cache-root', exclusive=False)


def staging_lock_path(path: Path) -> Path:
    """Sibling lock for one private staging directory/file."""
    path = Path(path)
    return path.parent / ('.staging-' + path.name + '.lock')


@contextlib.contextmanager
def hold_staging_lock(path: Path):
    """Hold one exclusive advisory lock for a complete staging build.

    The sibling lock remains on disk until the staging path is gone, so a
    sweeper can prove that no live builder still owns the directory.
    """
    path = Path(path)
    lock_path = staging_lock_path(path)
    handle = None
    acquired = False
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open('a+b')
        _lock_handle(handle, exclusive=True, blocking=True)
        acquired = True
        yield
    finally:
        if handle is not None:
            if acquired:
                _unlock_handle(handle)
            handle.close()
        if acquired and not path.exists():
            try:
                lock_path.unlink(missing_ok=True)
            except TimeoutError:
                raise
            except OSError:
                pass


def _staging_builder_alive(path: Path) -> bool:
    """Prove ownership only by acquiring the builder's exclusive lock."""
    lock_path = staging_lock_path(path)
    try:
        handle = lock_path.open('a+b')
    except TimeoutError:
        raise
    except OSError:
        # Cannot even create the probe lock: never delete the staging path.
        return True
    acquired = False
    try:
        try:
            _lock_handle(handle, exclusive=True, blocking=False)
            acquired = True
        except TimeoutError:
            raise
        except OSError:
            return True
        return False
    finally:
        if acquired:
            _unlock_handle(handle)
        handle.close()
        if acquired:
            try:
                lock_path.unlink(missing_ok=True)
            except TimeoutError:
                raise
            except OSError:
                pass


_RMTREE_RETRY_FUNCTIONS = (os.unlink, os.remove, os.rmdir)


def _clear_readonly(func, target, exc_info) -> None:
    """`shutil.rmtree` callback: clear one read-only bit and retry once.

    TimeoutError and non-permission OSErrors propagate so `rmtree` can never
    return as if it deleted a tree that still exists. `rmtree` cannot resume
    `lstat`/`scandir` traversal after this callback in Python 3.11, so those
    failures propagate instead of silently skipping the subtree.
    """
    exc = exc_info[1] if isinstance(exc_info, tuple) else exc_info
    if isinstance(exc, TimeoutError):
        raise exc
    if not isinstance(exc, PermissionError) or func not in _RMTREE_RETRY_FUNCTIONS:
        raise exc
    try:
        os.chmod(target, stat.S_IWRITE)
    except TimeoutError:
        raise
    except OSError:
        pass
    func(target)


def _remove_tree(path: Path) -> None:
    """Remove a cache tree, proving it is gone; never report false deletion."""
    path = Path(path)
    shutil.rmtree(path, onerror=_clear_readonly)
    if os.path.lexists(path):
        raise OSError('cache tree still exists after removal: ' + str(path))


def _children(path: Path) -> list[Path]:
    try:
        return list(path.iterdir())
    except TimeoutError:
        raise
    except OSError:
        return []


def sweep_staging_directories(directory: Path, prefixes, *,
                              max_age_seconds: float = CACHE_STAGING_MAX_AGE) -> list[str]:
    """Remove staging only when it is old *and* its builder lock is free."""
    directory = Path(directory)
    if _is_link(directory):
        return []
    now = time.time()
    removed = []
    for path in _children(directory):
        if not path.name.startswith(tuple(prefixes)) or path.name.startswith('.staging-'):
            continue
        try:
            age = now - path.lstat().st_mtime
        except TimeoutError:
            raise
        except OSError:
            continue
        if age < max_age_seconds:
            continue
        if _staging_builder_alive(path):
            continue
        try:
            if _is_link(path):
                path.unlink(missing_ok=True)
            elif path.is_dir():
                _remove_tree(path)
            else:
                path.unlink(missing_ok=True)
        except TimeoutError:
            raise
        except OSError:
            continue
        removed.append(path.name)
    return removed


def sweep_native_cache_staging(cache: Path, *,
                               max_age_seconds: float = CACHE_STAGING_MAX_AGE) -> list[str]:
    """Remove orphaned publication/trash staging left by a killed process."""
    return sweep_staging_directories(Path(cache), _CACHE_STAGING_PREFIXES,
                                     max_age_seconds=max_age_seconds)


def _tree_bytes(path: Path) -> int:
    total = 0
    for root, _directories, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                total += (Path(root) / name).lstat().st_size
            except TimeoutError:
                raise
            except OSError:
                continue
    return total


def native_cache_limits(max_entries: int | None = None,
                        max_total_bytes: int | None = None) -> tuple[int, int]:
    """Resolve the global entry/byte budget; `MINDIE_NATIVE_CACHE_MAX_BYTES` wins."""
    if max_entries is None:
        try:
            max_entries = int(os.environ.get('MINDIE_NATIVE_CACHE_MAX_ENTRIES',
                                             NATIVE_CACHE_MAX_ENTRIES))
        except (TypeError, ValueError):
            max_entries = NATIVE_CACHE_MAX_ENTRIES
    if max_total_bytes is None:
        try:
            max_total_bytes = int(os.environ.get('MINDIE_NATIVE_CACHE_MAX_BYTES',
                                                 NATIVE_CACHE_MAX_BYTES))
        except (TypeError, ValueError):
            max_total_bytes = NATIVE_CACHE_MAX_BYTES
    return max(1, int(max_entries)), max(1, int(max_total_bytes))


def _cache_entries(cache: Path) -> list[dict]:
    cache = Path(cache)
    entries = []
    getuid = getattr(os, 'getuid', None)
    uid = getuid() if getuid is not None else None

    def add(kind: str, name: str, path: Path) -> None:
        if not _ENTRY_NAME.fullmatch(name) or _is_link(path):
            return
        try:
            info = path.lstat()
        except TimeoutError:
            raise
        except OSError:
            return
        if not stat.S_ISDIR(info.st_mode):
            return
        if uid is not None and info.st_uid != uid:
            return
        entries.append({'kind': kind, 'name': name, 'path': path,
                        'mtime': info.st_mtime, 'size': _tree_bytes(path)})

    bundles = cache / 'bundles'
    if not _is_link(bundles) and bundles.is_dir():
        for path in _children(bundles):
            add('bundle', path.name, path)
    if not _is_link(cache):
        for path in _children(cache):
            add('content', path.name, path)
    kernel = cache / 'kernel'
    if not _is_link(kernel) and kernel.is_dir():
        for path in _children(kernel):
            add('kernel', path.name, path)
    return entries


def _content_bundle(cache: Path, name: str) -> str | None:
    try:
        metadata = json.loads((cache / name / NATIVE_CONTENT_MANIFEST).read_text(encoding='utf-8'))
    except TimeoutError:
        raise
    except (OSError, ValueError):
        return None
    bundle = metadata.get('bundle') if isinstance(metadata, dict) else None
    return bundle if isinstance(bundle, str) and _ENTRY_NAME.fullmatch(bundle) else None


def _index_files(cache: Path) -> dict[str, list[Path]]:
    mapping: dict[str, list[Path]] = {}
    for path in _children(cache):
        if path.suffix != '.json' or _is_link(path) or not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except TimeoutError:
            raise
        except (OSError, ValueError):
            continue
        bundle = data.get('bundle') if isinstance(data, dict) else None
        if isinstance(bundle, str) and _ENTRY_NAME.fullmatch(bundle):
            mapping.setdefault(bundle, []).append(path)
    return mapping


def touch_native_cache_entry(path: Path) -> None:
    """Mark an entry as recently used; mtime is the LRU ordering key."""
    try:
        os.utime(path, None)
    except TimeoutError:
        raise
    except OSError:
        pass


def evict_native_cache(cache: Path, *, keep=(), max_entries: int | None = None,
                       max_total_bytes: int | None = None) -> dict:
    """Bound `/tmp/mindie-native-cache` by entry count and total bytes.

    Candidates are ordered by mtime and evicted with an atomic rename to
    `.trash-*` followed by `rmtree`. The just-published/requested keys, any
    entry locked by another cache operation, and immutable bundles referenced
    by a content entry are never evicted.
    """
    cache = Path(cache)
    if isinstance(keep, str):
        keep = {keep}
    limits = native_cache_limits(max_entries, max_total_bytes)
    result = {'status': 'missing', 'removed': [], 'staging_removed': [],
              'entries': 0, 'bytes': 0,
              'limits': {'entries': limits[0], 'bytes': limits[1]}}
    if _is_link(cache) or not cache.is_dir():
        if cache.exists():
            result['status'] = 'unsafe'
        return result
    with _advisory_lock(cache, 'cache-root', exclusive=True, blocking=False) as acquired:
        if not acquired:
            result['status'] = 'busy'
            return result
        for directory in (cache, cache / 'bundles', cache / 'kernel'):
            result['staging_removed'].extend(sweep_native_cache_staging(directory))
        entries = _cache_entries(cache)
        indexes = _index_files(cache)
        pinned = {value for value in keep if isinstance(value, str) and value}
        total = sum(entry['size'] for entry in entries)
        removed = []
        trash_entries = 0
        while entries and (len(entries) + trash_entries > limits[0] or total > limits[1]):
            referenced = {_content_bundle(cache, entry['name'])
                          for entry in entries if entry['kind'] == 'content'}
            entries.sort(key=lambda entry: (entry['mtime'], entry['kind'], entry['name']))
            dropped = False
            for entry in entries:
                if entry['name'] in pinned:
                    continue
                if entry['kind'] == 'bundle' and entry['name'] in referenced:
                    continue
                with _cache_entry_lock(cache, entry['kind'], entry['name'],
                                       exclusive=True, blocking=False) as locked:
                    if not locked:
                        continue
                    trash = cache / ('.trash-' + uuid.uuid4().hex)
                    try:
                        os.rename(entry['path'], trash)
                    except TimeoutError:
                        raise
                    except OSError:
                        continue
                    deleted = False
                    try:
                        _remove_tree(trash)
                    except TimeoutError:
                        raise
                    except OSError as exc:
                        print(f'native cache trash cleanup failed: {exc}',
                              file=sys.stderr, flush=True)
                    else:
                        deleted = not os.path.lexists(trash)
                entries.remove(entry)
                if deleted:
                    total -= entry['size']
                    removed.append(entry['kind'] + ':' + entry['name'])
                    if entry['kind'] == 'bundle':
                        for path in indexes.pop(entry['name'], []):
                            try:
                                path.unlink(missing_ok=True)
                            except TimeoutError:
                                raise
                            except OSError:
                                continue
                else:
                    # The renamed bytes still occupy disk; keep them counted.
                    total += _tree_bytes(trash) - entry['size']
                    trash_entries += 1
                dropped = True
                break
            if not dropped:
                break
        existing_bundles = {entry['name'] for entry in entries if entry['kind'] == 'bundle'}
        for bundle, paths in indexes.items():
            if bundle in existing_bundles:
                continue
            for path in paths:
                try:
                    path.unlink(missing_ok=True)
                except TimeoutError:
                    raise
                except OSError:
                    continue
        result.update(status='evicted' if removed else 'within-limits',
                      removed=removed, entries=len(entries) + trash_entries, bytes=total)
        return result


def _content_names(manifest: dict) -> list[str]:
    names = set(manifest.get('files') or {})
    names.update(row.get('path') for row in (manifest.get('evidence') or {}).values()
                 if isinstance(row, dict) and row.get('path'))
    return sorted(names)


def _read_native_content(entry: Path, expected_key: str) -> dict | None:
    """Return a verified entry description, or None when absent.

    A present but malformed entry raises; treating corruption as a miss would
    let the next publication hide an earlier integrity failure.
    """
    if not entry.exists():
        return None
    metadata_path = entry / NATIVE_CONTENT_MANIFEST
    manifest_path = entry / 'manifest.json'
    if not entry.is_dir() or not metadata_path.is_file() or not manifest_path.is_file():
        raise ValueError('native content entry is incomplete: ' + entry.name)
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata.get('schema_version') != NATIVE_CONTENT_SCHEMA or metadata.get('cache_key') != expected_key:
        raise ValueError('native content entry key changed: ' + entry.name)
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != metadata.get('bundle_manifest_sha256'):
        raise ValueError('native content entry manifest changed: ' + entry.name)
    manifest = json.loads(manifest_bytes)
    if native_content_key(manifest.get('preparation') or {}) != expected_key:
        raise ValueError('native content entry preparation changed: ' + entry.name)
    return {'metadata': metadata, 'manifest': manifest, 'manifest_bytes': manifest_bytes}


def publish_native_content(bundle: Path, cache: Path, manifest: dict) -> dict:
    """Atomically materialize one verified bundle under its content key.

    Bytes are copied from the immutable bundle rather than hard-linked to it,
    so an in-place write through a writable view can only damage the replaceable
    pool entry; the bundle remains the recovery source.
    """
    key = native_content_key(manifest.get('preparation') or {})
    destination = cache / key
    if destination.exists():
        try:
            existing = _read_native_content(destination, key)
            verify(destination, existing['manifest'], check_environment=False)
        except ValueError:
            _drop_native_content(cache, key)
        else:
            if existing['manifest'].get('files') != manifest.get('files'):
                # Same measured inputs but different bytes are not a hit. Never
                # overwrite either artifact set; require investigation.
                raise ValueError('same native content key produced different artifacts')
            return {'status': 'present', 'cache_key': key, 'bundle': bundle.name}
    cache.mkdir(parents=True, exist_ok=True)
    sweep_native_cache_staging(cache)
    temp = Path(tempfile.mkdtemp(prefix='.native-content-', dir=cache))
    with hold_staging_lock(temp):
        try:
            for name in _content_names(manifest):
                target = safe_destination(temp, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(checked_file(bundle, name), target)
            manifest_bytes = (bundle / 'manifest.json').read_bytes()
            if json.loads(manifest_bytes) != manifest:
                raise ValueError('bundle manifest does not describe the published bytes')
            (temp / 'manifest.json').write_bytes(manifest_bytes)
            (temp / NATIVE_CONTENT_MANIFEST).write_text(json.dumps({
                'schema_version': NATIVE_CONTENT_SCHEMA,
                'cache_key': key,
                'bundle': bundle.name,
                'bundle_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
                'native_key': (manifest.get('preparation') or {}).get('native_key'),
                'files': {name: row['sha256'] for name, row in (manifest.get('files') or {}).items()},
            }, sort_keys=True, indent=2) + '\n')
            # The bundle publisher already verified these bytes; a hit re-hashes
            # the entry, so publication does not repeat the whole traversal.
            try:
                os.rename(temp, destination)
            except TimeoutError:
                raise
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY} or not destination.is_dir():
                    raise
                existing = _read_native_content(destination, key)
                verify(destination, existing['manifest'], check_environment=False)
                if existing['manifest'].get('files') != manifest.get('files'):
                    raise ValueError('same native content key produced different artifacts') from exc
                return {'status': 'present', 'cache_key': key, 'bundle': bundle.name}
            return {'status': 'published', 'cache_key': key, 'bundle': bundle.name}
        finally:
            if temp.exists():
                try:
                    _remove_tree(temp)
                except TimeoutError:
                    raise
                except OSError:
                    pass


def _drop_native_content(cache: Path, key: str) -> None:
    """Remove only this key's replaceable entry after a failed verification."""
    entry = cache / key
    if entry.parent.resolve() != cache.resolve():
        return
    if entry.is_dir() and not _is_link(entry):
        _remove_tree(entry)
    elif entry.exists() or entry.is_symlink():
        entry.unlink()
