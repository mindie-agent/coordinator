"""Same-container preparation reuse, executed inside a fresh execution view.

Published roots are never modified. Venvs stay at their original absolute path;
native reuse shares that interpreter and overlays fully copied package outputs.
Native changes copy only dependency packages into a fresh venv before building.
"""
from __future__ import annotations

import importlib.metadata
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import sysconfig
import tempfile
from pathlib import Path


def safe_destination(root: Path, relative: str) -> Path:
    """Published artifacts may never be redirected through a source symlink."""
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('unsafe execution artifact destination')
    for part in (root, *(root / Path(*path.parts[:index]) for index in range(1, len(path.parts) + 1))):
        if part.is_symlink():
            raise ValueError('symlink in execution artifact destination: ' + str(part))
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('execution artifact destination escaped its root')
    return target


def copy_dependencies(source_site: Path, destination_site: Path) -> list[str]:
    """Copy dependencies without reusing editable finders or business packages."""
    copied = []
    destination_site.mkdir(parents=True, exist_ok=True)
    for source in source_site.iterdir():
        normalized = source.name.lower().replace('-', '_')
        if normalized.startswith(('vllm', '__editable__', 'pip', 'setuptools', '_distutils_hack')):
            continue
        if source.suffix == '.pth' and ('editable' in source.name or 'vllm' in source.read_text(errors='replace').lower()):
            continue
        target = safe_destination(destination_site, source.name)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
        copied.append(source.name)
    return copied


def write_source_metadata(root: Path, versions: dict, distributions: dict | None = None) -> None:
    """Generate source version files and distribution overlays from real SCM."""
    metadata_root = safe_destination(root, '.mindie-runtime/metadata')
    metadata_root.mkdir(parents=True, exist_ok=True)
    for name, row in versions.items():
        version = row.get('version')
        if not isinstance(version, str) or not re.fullmatch(r'[A-Za-z0-9.!+_-]+', version):
            raise ValueError('missing captured SCM version: ' + name)
        if name not in {'vllm', 'vllm-ascend'}:
            raise ValueError('unsupported native source metadata: ' + name)
        package = name.replace('-', '_')
        saved = (distributions or {}).get(name)
        dist = importlib.metadata.distribution(name) if saved is None else None
        installed = saved['version'] if saved else dist.version
        read_text = (lambda filename: saved['files'].get(filename)) if saved else dist.read_text
        # vLLM's supported empty-device recipe adds this package suffix.
        package_version = version + ('.empty' if '+' in version else '+empty') if name == 'vllm' and installed.endswith('empty') else version
        metadata_relative = '.mindie-runtime/metadata/' + package + '-' + package_version + '.dist-info'
        destination = safe_destination(root, metadata_relative)
        destination.mkdir(exist_ok=True)
        metadata = read_text('METADATA')
        if not metadata:
            raise ValueError('installed distribution has no metadata: ' + name)
        metadata = re.sub(r'^Version: .*$', 'Version: ' + package_version, metadata, count=1, flags=re.MULTILINE)
        safe_destination(root, metadata_relative + '/METADATA').write_text(metadata, encoding='utf-8')
        for filename in ('entry_points.txt', 'top_level.txt', 'WHEEL'):
            value = read_text(filename)
            output = safe_destination(root, metadata_relative + '/' + filename)
            if value is not None:
                output.write_text(value, encoding='utf-8')
            else:
                output.unlink(missing_ok=True)
        from setuptools_scm import dump_version
        version_file = safe_destination(root, name + '/' + package + '/_version.py')
        # Use the recipe's own version writer, including prerelease/local tuple
        # semantics; preserve the real HEAD rather than synthetic ancestry.
        dump_version(str(root / name), version, package + '/_version.py')
        with version_file.open('a', encoding='utf-8') as stream:
            stream.write('\n__commit_id__ = commit_id = ' + repr(row.get('source_head')) + '\n')


def copy_native_view(root: Path, source_root: Path, manifest: dict, versions: dict, *, verify_donor=True, timings=None) -> dict:
    # `verify`, `checked_file` and `capture` are injected from runtime_profile
    # into the remote preparation script; this module has no remote imports.
    if verify_donor:
        verify(source_root, manifest)
    if root.resolve() == source_root.resolve():
        raise ValueError('native view destination must be a new execution root')
    build_info = 'vllm-ascend/vllm_ascend/_build_info.py'
    if (source_root / build_info).is_file() and build_info not in manifest['files']:
        raise ValueError('generated build metadata has no verified donor hash: ' + build_info)
    for name, expected in manifest['files'].items():
        target = safe_destination(root, name)
        if target.exists():
            raise ValueError('native output already exists in fresh view: ' + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checked_file(source_root, name), target)
        # A bound donor may change between verification and copying. The new
        # view must retain the previously verified bytes, never re-attest them.
        if file_digest(checked_file(root, name)) != expected['sha256']:
            raise ValueError('copied artifact differs from verified donor hash: ' + name)
    import time
    metadata_started = time.monotonic()
    write_source_metadata(root, versions)
    if timings is not None:
        timings['source_metadata'] = time.monotonic() - metadata_started
    toolchain = build_toolchain_from_logs(source_root)
    return {'kind': 'native', 'source_root': str(source_root), 'build_key': manifest['build_key'],
            'soc': manifest['profile']['soc'], 'compiler': manifest['profile']['compiler'],
            'toolchain': toolchain}


SHARED_NATIVE_CACHE = '/tmp/mindie-native-cache'


def shared_input_key(preparation: dict, image_digest: str) -> str:
    """One lookup from fixed inputs; a first-use donor has no measured profile."""
    if not image_digest or not all(preparation.get('native', {}).get(name) for name in ('vllm', 'vllm-ascend')):
        raise ValueError('shared native cache requires fixed native inputs and image digest')
    environment = preparation.get('environment', {})
    # A default image and its explicit immutable reference can resolve to the
    # same Docker image. Request spelling is not compiled-output identity;
    # restore checks requested constraints against the measured profile.
    return digest({'image': image_digest, 'dependencies': preparation.get('dependencies'),
                   'native': preparation['native'], 'build_env': environment.get('build_env', {})})


def shared_base_key(preparation: dict, image_digest: str) -> str:
    """Dependency-compatible candidates; kernel reuse still proves the exact delta."""
    baseline = {**preparation, 'native': {**preparation.get('native', {}), 'vllm-ascend': '*'}}
    return shared_input_key(baseline, image_digest)


def store_shared_native(root: Path, cache: Path) -> dict:
    """Automatically cache only the existing verified output bundle and metadata."""
    manifest = json.loads((root / '.mindie-runtime/ready-profile.json').read_text())
    manifest = _with_distribution_metadata(manifest)
    bundle = publish(root, cache / 'bundles', manifest)
    return _index_native_bundle(cache, manifest, bundle)


def _store_captured_native(root: Path, cache: Path, manifest: dict, timings: dict) -> dict:
    """Only the managed capture's current stack supplies this fresh manifest."""
    manifest = _with_distribution_metadata(manifest)
    bundle = _publish_captured_bundle(root, cache / 'bundles', manifest, timings=timings)
    return _index_native_bundle(cache, manifest, bundle)


def _with_distribution_metadata(manifest: dict) -> dict:
    metadata = {}
    for name in ('vllm', 'vllm-ascend'):
        distribution = importlib.metadata.distribution(name)
        metadata[name] = {'version': distribution.version, 'files': {
            filename: distribution.read_text(filename) for filename in
            ('METADATA', 'WHEEL', 'entry_points.txt', 'top_level.txt')}}
    return {**manifest, 'distributions': metadata}


def _index_native_bundle(cache: Path, manifest: dict, bundle: Path) -> dict:
    # publish already atomically verifies and installs the complete directory.
    # The small index does not register or expose any donor interpreter/runtime.
    preparation = manifest['preparation']
    key = shared_input_key(preparation, manifest['profile']['image_digest'])
    cache.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.index-', dir=cache)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump({'bundle': bundle.name, 'native_key': preparation['native_key']}, stream)
        os.replace(temporary, cache / (key + '.json'))
        # This index only selects a candidate. Its exact native delta and all
        # copied outputs must still be checked before an incremental build.
        base = shared_base_key(preparation, manifest['profile']['image_digest'])
        fd, temporary = tempfile.mkstemp(prefix='.index-', dir=cache)
        with os.fdopen(fd, 'w') as stream:
            json.dump({'bundle': bundle.name, 'native_key': preparation['native_key']}, stream)
        os.replace(temporary, cache / ('base-' + base + '.json'))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {'status': 'stored', 'native_key': preparation['native_key'], 'bundle': bundle.name}


def discard_shared_native(root: Path) -> None:
    """Remove only cache-created files in this not-yet-running execution view."""
    marker = root / '.mindie-runtime/shared-native.json'
    if not marker.is_file():
        return
    receipt = json.loads(marker.read_text())
    for relative in receipt.get('copied', []):
        safe_destination(root, relative).unlink(missing_ok=True)
    metadata = safe_destination(root, '.mindie-runtime/metadata')
    if metadata.exists():
        shutil.rmtree(metadata)
    safe_destination(root, '.mindie-runtime/reuse.json').unlink(missing_ok=True)
    safe_destination(root, '.mindie-runtime/native-incremental.json').unlink(missing_ok=True)
    marker.unlink()


def recipient_dependencies(distributions: dict, profile: dict) -> dict:
    """Check recipe metadata in the actual recipient without importing torch."""
    from email.parser import Parser
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    from packaging.version import InvalidVersion, Version

    metadata = distributions.get('vllm-ascend', {}).get('files', {}).get('METADATA')
    errors, versions = [], {}
    if not metadata:
        errors.append('verified vllm-ascend distribution metadata is missing')
    for raw in Parser().parsestr(metadata or '').get_all('Requires-Dist', []):
        try:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            installed = importlib.metadata.version(requirement.name)
            versions[canonicalize_name(requirement.name)] = installed
            if not requirement.specifier or requirement.specifier.contains(installed, prereleases=True):
                continue
            try:
                public = Version(installed).public
            except InvalidVersion:
                public = installed.split('+', 1)[0]
            if public != installed and requirement.specifier.contains(public, prereleases=True):
                continue
            # The existing recipe accepts the paired image's torch-npu dev
            # build after import. Restore already verified this exact version
            # and the donor's original successful smoke, so no repeat import.
            if canonicalize_name(requirement.name) == 'torch-npu' and installed == profile['torch_npu']:
                continue
            errors.append(f'{requirement.name}{requirement.specifier} (installed {installed})')
        except Exception as exc:
            errors.append(f'{raw!r}: {exc}')
    return {'satisfied': not errors, 'errors': errors, 'versions': versions,
            'interpreter': sys.executable, 'prefix': sys.prefix, 'purelib': sysconfig.get_paths()['purelib']}


def verify_recipient_native_abi(profile: dict) -> None:
    """A dependency repair must preserve the environment that built the bundle."""
    if profile['python_abi'] != sysconfig.get_config_var('SOABI'):
        raise ValueError('cached Python ABI differs')
    for field, package in (('torch', 'torch'), ('torch_npu', 'torch-npu')):
        if importlib.metadata.version(package) != profile[field]:
            raise ValueError('cached package ABI differs: ' + package)
    for row in profile['system_files'].values():
        if file_digest(Path(row['path'])) != row['sha256']:
            raise ValueError('cached CANN/driver support differs')


def revalidate_shared_native(root: Path, cache: Path) -> dict:
    """Recheck the fixed selected bundle after pip or dependency-overlay repair."""
    receipt = json.loads(safe_destination(root, '.mindie-runtime/shared-native.json').read_text())
    name = receipt.get('bundle', '')
    if not re.fullmatch('[0-9a-f]{64}', name):
        raise ValueError('shared native receipt has no fixed bundle')
    data = (cache / 'bundles' / name / 'manifest.json').read_bytes()
    if hashlib.sha256(data).hexdigest() != receipt.get('bundle_manifest_sha256'):
        raise ValueError('selected shared native bundle manifest changed')
    profile = json.loads(data)['profile']
    verify_recipient_native_abi(profile)
    # Dependency copying has its own receipt. Keep the original native proof
    # so later profile capture cannot lose the bundle's SoC/compiler evidence.
    safe_destination(root, '.mindie-runtime/reuse.json').write_text(json.dumps({
        'kind': 'shared-native', 'native_key': receipt['native_key'],
        'soc': profile['soc'], 'compiler': profile['compiler']}))
    return {'status': 'validated', 'bundle': name}


def restore_shared_native(root: Path, cache: Path, preparation: dict, image_digest: str, versions: dict,
                          machine_type: str | None = None, candidate: dict | None = None) -> dict:
    """Copy an ABI-compatible cached bundle into this execution's own sources."""
    key = shared_input_key(preparation, image_digest)
    if candidate is None:
        index = cache / (key + '.json')
        if not index.is_file():
            index = cache / ('base-' + shared_base_key(preparation, image_digest) + '.json')
            if not index.is_file():
                return {'status': 'miss', 'reason': 'no matching compiled outputs'}
        pointer = json.loads(index.read_text())
    else:
        # An export selects immutable content, never whichever donor happened
        # to overwrite the shared exact/base index before this restore.
        pointer = candidate
    if not re.fullmatch('[0-9a-f]{64}', pointer.get('bundle', '')):
        raise ValueError('invalid shared native cache pointer')
    bundle = cache / 'bundles' / pointer['bundle']
    manifest_bytes = (bundle / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    profile = manifest['profile']
    expected = verified_preparation(preparation, profile)
    baseline = manifest.get('preparation', {})
    verified_baseline = verified_preparation(baseline, profile)
    if (pointer.get('native_key') != baseline.get('native_key') or
            baseline.get('native_key') != verified_baseline['native_key'] or
            baseline.get('dependency_key') != verified_baseline['dependency_key'] or
            baseline.get('dependencies') != preparation.get('dependencies') or
            baseline.get('environment', {}).get('build_env', {}) != preparation.get('environment', {}).get('build_env', {}) or
            profile['image_digest'] != image_digest or
            baseline.get('native', {}).get('vllm') != preparation.get('native', {}).get('vllm')):
        raise ValueError('cached native inputs or image differ')
    requested = preparation.get('environment', {}).get('environment', {})
    for name in ('soc', 'cann', 'python_abi', 'machine_type'):
        if requested.get(name) and requested[name] != profile.get(name):
            raise ValueError('cached profile does not satisfy requested ' + name)
    current_soc = (os.environ.get('SOC_VERSION') or os.environ.get('MINDIE_SOC_VERSION') or
                   preparation.get('environment', {}).get('soc'))
    if current_soc and current_soc != profile['soc']:
        raise ValueError('cached SoC differs from recipient environment')
    if machine_type and machine_type != profile.get('machine_type'):
        raise ValueError('cached machine type differs from recipient host')
    plan = None
    if baseline.get('native') != preparation.get('native'):
        entries = native_tree_entries(root / 'vllm-ascend', VLLM_ASCEND_REINSTALL_PATTERNS, submodule_content)
        plan = kernel_rebuild_plan(root, bundle, manifest, preparation, entries)
        if plan is None:
            return {'status': 'miss', 'reason': 'native changes require a complete build'}
    verify_recipient_native_abi(profile)
    verify(bundle, manifest, check_environment=False)
    if 'vllm-ascend/vllm_ascend/_build_info.py' not in manifest['files']:
        raise ValueError('cached generated build metadata is missing')
    # Preflight all destinations before writing any cached output.
    for name in manifest['files']:
        if safe_destination(root, name).exists():
            raise ValueError('native output already exists in fresh view: ' + name)
    receipt = {'status': 'incremental' if plan else 'hit', 'native_key': expected['native_key'],
               'copied': list(manifest['files']), 'bundle': pointer['bundle'],
               'bundle_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest()}
    marker = safe_destination(root, '.mindie-runtime/shared-native.json')
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(receipt))
    try:
        for name, identity in manifest['files'].items():
            target = safe_destination(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checked_file(bundle, name), target)
            if file_digest(target) != identity['sha256']:
                raise ValueError('shared artifact changed while copying: ' + name)
        write_source_metadata(root, versions, manifest['distributions'])
        receipt['dependencies'] = recipient_dependencies(manifest['distributions'], profile)
        marker.write_text(json.dumps(receipt))
        reuse = {'kind': 'shared-native', 'native_key': expected['native_key'],
                 'soc': profile['soc'], 'compiler': profile['compiler']}
        compatibility = native_import_receipt(bundle, manifest)
        if compatibility is not None:
            relative = '.mindie-runtime/profile-evidence/import-closure.json'
            proof = safe_destination(root, relative)
            proof.parent.mkdir(parents=True, exist_ok=True)
            proof.write_text(json.dumps(compatibility, sort_keys=True) + '\n')
            reuse['import_evidence'] = relative
        atb = re.search(r'/cxx_abi_([01])/?$', profile.get('launch_env', {}).get('ATB_HOME_PATH', ''))
        if atb:
            # This comes from the verified donor's real ATB activation. The
            # consumer checks its current torch version/Python ABI before
            # giving ATB's supported argument, otherwise ATB detects normally.
            reuse['atb_abi'] = {'cxx_abi': atb[1], 'torch': profile['torch'], 'python_abi': profile['python_abi']}
        safe_destination(root, '.mindie-runtime/reuse.json').write_text(json.dumps(reuse))
        if plan:
            safe_destination(root, '.mindie-runtime/native-incremental.json').write_text(json.dumps(plan))
            receipt['operator'] = plan['operator']
    except Exception:
        discard_shared_native(root)
        raise
    return receipt


REMOTE_SHARED_SUFFIX = r'''
import sys
args = json.loads(sys.argv[1])
root = Path(args['root'])
cache = Path(args.get('cache', SHARED_NATIVE_CACHE))
try:
    if args['action'] == 'store':
        result = store_shared_native(root, cache)
    elif args['action'] == 'discard':
        discard_shared_native(root)
        result = {'status': 'discarded'}
    elif args['action'] == 'revalidate':
        result = revalidate_shared_native(root, cache)
    else:
        result = restore_shared_native(root, cache, args['preparation'], args['image_digest'], args['versions'],
                                       args.get('machine_type'), args.get('candidate'))
except Exception as exc:
    result = {'status': 'miss', 'reason': str(exc)}
if args['action'] == 'restore' and result.get('status') in {'hit', 'incremental'}:
    # Rollback owns this full list in shared-native.json. The coordinator
    # needs the outcome/compatibility facts, not every copied artifact path.
    result = {key: value for key, value in result.items() if key != 'copied'}
print(json.dumps(result))
'''


def native_view_launch_environment(manifest: dict, root: Path) -> dict:
    """Move only this view's source/metadata and native loader paths."""
    source = manifest['runtime_root'].rstrip('/')
    destination = root.as_posix().rstrip('/')
    overlays = {source + suffix for suffix in ('/.mindie-runtime/metadata', '/vllm', '/vllm-ascend')}
    native = source + '/vllm-ascend/vllm_ascend'
    vendors = [source + '/' + path for path in native_vendor_paths(manifest['files'])['ASCEND_CUSTOM_OPP_PATH']]
    result = dict(manifest['profile']['launch_env'])
    for key in ('PYTHONPATH', 'LD_LIBRARY_PATH', 'ASCEND_CUSTOM_OPP_PATH'):
        parts = []
        for part in result.get(key, '').split(':'):
            if key == 'PYTHONPATH' and part in overlays:
                part = destination + part[len(source):]
            elif key != 'PYTHONPATH' and (part in {native, native + '/_cann_ops_custom'}
                                         or part.startswith(native + '/_cann_ops_custom/')
                                         or any(part == vendor or part.startswith(vendor + '/') for vendor in vendors)):
                part = destination + part[len(source):]
            if part and part not in parts:
                parts.append(part)
        if key == 'PYTHONPATH':
            current = [destination + suffix for suffix in ('/.mindie-runtime/metadata', '/vllm', '/vllm-ascend')]
            parts = list(dict.fromkeys([*current, *parts]))
        if parts:
            result[key] = ':'.join(parts)
    return native_vendor_launch_environment(root, manifest['files'], result)


def prepare_native_view(root: Path, source_root: Path, donor: dict, args: dict) -> dict:
    """Publish a Python source view from the existing native proof in one step.

    Copying checks every output against the recorded hash once. The environment
    and original import evidence are reused, not re-captured or marked passed.
    The caller has already materialized and checked the fixed Git snapshot.
    """
    import time
    started = time.monotonic()
    timings = {}
    if digest(donor) != args['donor_manifest_digest']:
        raise ValueError('native donor manifest changed; inspect or repair that environment')
    verify_environment(source_root, donor)
    compatibility = native_compatibility_receipt(source_root, donor)
    if compatibility is None:
        raise ValueError('native donor needs an original environment compatibility proof')
    for name in ('vllm', 'vllm-ascend'):
        for field in ('native', 'dependencies'):
            if args['build_inputs'][name][field] != donor['build_inputs'][name][field]:
                raise ValueError('native reuse inputs changed: ' + name + '/' + field)
    if args['build_env'] != donor['profile']['build_env']:
        raise ValueError('native reuse build environment changed')
    reuse = copy_native_view(root, source_root, donor, args['versions'], verify_donor=False, timings=timings)
    current = copy.deepcopy(donor)
    current['runtime_root'] = str(root.resolve())
    profile = current['profile']
    profile['source_versions'] = args['versions']
    profile['compatibility_evidence'] = '.mindie-runtime/profile-evidence/smoke.json'
    profile['launch_env'] = native_view_launch_environment(donor, root)
    old_overlays = {str(source_root / suffix) for suffix in ('.mindie-runtime/metadata', 'vllm', 'vllm-ascend')}
    sys.path[:] = [str(root / suffix) for suffix in ('.mindie-runtime/metadata', 'vllm', 'vllm-ascend')] + [
        path for path in sys.path if path not in old_overlays]
    mapping = native_source_mapping(root)
    profile.update(vllm=mapping['vllm_version'], vllm_ascend=mapping['vllm-ascend_version'])
    current['profile_key'] = profile_key(profile)
    environment = {**profile['build_env'], 'MINDIE_ENVIRONMENT_FINGERPRINT': current['profile_key']}
    environment = {key: environment[key] for key in BUILD_INPUT_ENV_KEYS if key in environment}
    fingerprint = hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()
    current['build_inputs'] = {name: {**row, 'build_env': fingerprint} for name, row in args['build_inputs'].items()}
    current['build_key'] = build_key(profile, current['build_inputs'])
    current['preparation'] = verified_preparation(args['preparation'], profile)
    current['execution_view'] = {'source_id': args['source_id'], 'native_from': donor['build_key']}
    smoke = {'kind': 'native-compatibility-reuse', 'python_import_executed': False,
             'profile_key': current['profile_key'], 'build_inputs': current['build_inputs'],
             'compatibility': compatibility, 'source_mapping': mapping}
    upgraded_loader = (native_vendor_launch_environment(source_root, donor['files'], donor['profile']['launch_env'])
                       != donor['profile']['launch_env'])
    if upgraded_loader and compatibility['key'] != native_compatibility_key(current):
        # Historical imports did not prove this loader environment. Upgrade
        # only the owned view, and keep the one-time import cost explicit.
        smoke = native_import_smoke(root, profile, current['build_inputs'])
        smoke.update(reason='native-loader-environment-upgrade', source_mapping=mapping)
        if not smoke['passed']:
            failed = safe_destination(root, '.mindie-runtime/profile-evidence/smoke.json')
            failed.parent.mkdir(parents=True, exist_ok=True)
            failed.write_text(json.dumps(smoke, sort_keys=True, indent=2) + '\n')
            raise ValueError('owned native loader upgrade import smoke failed; inspect profile-evidence/smoke.json')
    else:
        verify_native_compatibility(root, current, smoke, check_environment=False)
    evidence = {}
    for name in ('cann', 'driver'):
        row = donor['evidence'][name]
        data = checked_file(source_root, row['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('native donor environment evidence changed: ' + name)
        target = safe_destination(root, row['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        evidence[name] = dict(row)
    for name, relative, value in (
        ('smoke', '.mindie-runtime/profile-evidence/smoke.json', smoke),
        ('reuse', '.mindie-runtime/reuse.json', reuse),
    ):
        target = safe_destination(root, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
        target.write_bytes(data)
        evidence[name] = {'path': relative, 'sha256': hashlib.sha256(data).hexdigest()}
    current['evidence'] = evidence
    build_source = safe_destination(root, '.mindie-runtime/build-source.json')
    build_source.write_text(json.dumps({'versions': args['versions'], 'build_env': args['build_env']}) + '\n')
    marker = safe_destination(root, '.mindie-runtime/ready-profile.json')
    temporary = marker.with_suffix('.tmp')
    temporary.write_text(json.dumps(current, sort_keys=True, indent=2) + '\n')
    os.replace(temporary, marker)
    # Files have not changed: the caller already owns their complete identity.
    # Return only the new view facts rather than another 300 KB native manifest.
    result = {'manifest': {key: value for key, value in current.items() if key != 'files'},
              'manifest_digest': digest(current)}
    timings['native_publication'] = time.monotonic() - started
    return {**result, 'preparation_timings': timings}


REMOTE_NATIVE_VIEW_SUFFIX = r'''
args = json.loads(sys.argv[1])
source_root = Path(args['source_root'])
donor = json.loads((source_root / '.mindie-runtime/ready-profile.json').read_text())
print(json.dumps(prepare_native_view(Path(args['root']), source_root, donor, args)))
'''


REMOTE_REUSE_SUFFIX = r'''
import subprocess, sys, sysconfig
args = json.loads(sys.argv[1])
root = Path(args['root'])
source_root = Path(args['source_root'])
manifest = json.loads((source_root / '.mindie-runtime/ready-profile.json').read_text())
if args['kind'] == 'native':
    result = copy_native_view(root, source_root, manifest, args['versions'])
    compatibility = native_compatibility_receipt(source_root, manifest)
    if compatibility is not None:
        relative = '.mindie-runtime/profile-evidence/native-compatibility.json'
        proof = safe_destination(root, relative)
        proof.parent.mkdir(parents=True, exist_ok=True)
        proof.write_text(json.dumps(compatibility, sort_keys=True) + '\n')
        result['compatibility_evidence'] = relative
else:
    verify(source_root, manifest)
    source_site = Path(sysconfig.get_paths()['purelib'])
    destination_site = Path(subprocess.check_output([args['python'], '-c', 'import sysconfig; print(sysconfig.get_paths()["purelib"])'], text=True).strip())
    result = {'kind': 'dependencies', 'source_root': str(source_root),
              'copied_packages': copy_dependencies(source_site, destination_site)}
receipt = root / '.mindie-runtime/reuse.json'
receipt.parent.mkdir(parents=True, exist_ok=True)
receipt.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
'''
