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

[Timeouts & Locks]
- Publication uses a private `mkdtemp` and one `os.rename`; no partial key is
  visible. No network calls or locks.

[Side-Effect Policy]
- Writes only under the supplied cache root and returns paths/receipts.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import tempfile
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


def _link_or_copy(source: Path, target: Path) -> bool:
    try:
        os.link(source, target)
        return True
    except OSError as exc:
        unsupported = exc.errno in _HARDLINK_FALLBACK_ERRNOS or getattr(exc, 'winerror', None) in _HARDLINK_FALLBACK_WINERRORS
        if not unsupported:
            raise
    shutil.copy2(source, target)
    return False


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
    metadata = json.loads(metadata_path.read_text())
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
    temp = Path(tempfile.mkdtemp(prefix='.native-content-', dir=cache))
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
            shutil.rmtree(temp)


def _drop_native_content(cache: Path, key: str) -> None:
    """Remove only this key's replaceable entry after a failed verification."""
    entry = cache / key
    if entry.parent.resolve() != cache.resolve():
        return
    if entry.is_dir() and not entry.is_symlink():
        shutil.rmtree(entry)
    elif entry.exists() or entry.is_symlink():
        entry.unlink()
