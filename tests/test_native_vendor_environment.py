"""Custom-op search paths exist before Python starts and point at owned files."""
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from mindie_coordinator import preparation_cache as cache, runtime_profile as profile
from mindie_coordinator.prepare_runtime import REMOTE_CAPTURE_SUFFIX


def test_vendor_paths_follow_manifest_layout_and_all_vendors():
    base = 'another-repo/another_package/_cann_ops_custom/vendors/'
    paths = profile.native_vendor_paths({
        base + 'custom_transformer/op_api/lib/libopapi.so': {},
        base + 'custom_transformer/op_api/lib/libopapi.so.1': {},
        base + 'other_vendor/op_impl/config/ops.json': {},
        'unrelated/vendor/op_api/lib/libother.so': {},
    })
    assert paths == {'ASCEND_CUSTOM_OPP_PATH': [base + 'custom_transformer', base + 'other_vendor'],
                     'LD_LIBRARY_PATH': [base + 'custom_transformer/op_api/lib']}
    with pytest.raises(ValueError, match='unsafe'):
        profile.native_vendor_paths({'../escape/_cann_ops_custom/vendors/test/op_api/lib/libopapi.so': {}})


@pytest.fixture
def native_loader(tmp_path, monkeypatch):
    if sys.platform != 'linux':
        pytest.skip('actual ELF loader and colon-delimited paths require Linux')
    root, image = tmp_path / 'owned', tmp_path / 'image'
    image.mkdir()
    for name in ('torch_npu', 'acl'):
        (image / (name + '.py')).write_text('')
    # An image/donor library with the same basename must not win the search.
    (image / 'libopapi.so').write_bytes(b'incompatible donor library')
    extension = Path(importlib.util.find_spec('_ctypes_test').origin)
    package = root / 'vllm-ascend/vllm_ascend'
    package.mkdir(parents=True)
    target_extension = package / ('vllm_ascend_C' + importlib.machinery.EXTENSION_SUFFIXES[0])
    shutil.copy2(extension, target_extension)
    vendor = package / '_cann_ops_custom/vendors/custom_transformer'
    library = vendor / 'op_api/lib/libopapi.so'
    library.parent.mkdir(parents=True)
    shutil.copy2(extension, library)
    (vendor / 'config.json').write_text('{}')
    # Reuse a real stdlib ELF fixture under the supported alias. Its original
    # PyInit name remains _ctypes_test; both imported ELF and opapi are owned.
    (package / '__init__.py').write_text(
        'import ctypes, importlib.util, pathlib, sys\n'
        f'p = pathlib.Path(__file__).parent / {target_extension.name!r}\n'
        'spec = importlib.util.spec_from_file_location("_ctypes_test", p)\n'
        'vllm_ascend_C = importlib.util.module_from_spec(spec)\n'
        'spec.loader.exec_module(vllm_ascend_C)\n'
        'sys.modules[__name__ + ".vllm_ascend_C"] = vllm_ascend_C\n'
        'ctypes.CDLL("libopapi.so")\n'
        f'assert {str(library)!r} in pathlib.Path("/proc/self/maps").read_text()\n'
        f'print("loaded-owned-opapi=" + {str(library)!r})\n')
    vllm = root / 'vllm/vllm'
    vllm.mkdir(parents=True)
    (vllm / '__init__.py').write_text('')
    for name in ('vllm', 'vllm-ascend'):
        metadata = root / '.mindie-runtime/metadata' / (name.replace('-', '_') + '-1.0.dist-info')
        metadata.mkdir(parents=True)
        (metadata / 'METADATA').write_text('Name: ' + name + '\nVersion: 1.0\n')
    system = {}
    for name in ('cann', 'driver'):
        path = tmp_path / name
        path.write_text('1.0')
        system[name] = {'path': str(path), 'sha256': profile.file_digest(path)}
    for key in profile.LAUNCH_PATH_KEYS:
        monkeypatch.delenv(key, raising=False)
    environment = {'LD_LIBRARY_PATH': str(image), 'ASCEND_CUSTOM_OPP_PATH': str(image / 'opp'),
                   'PYTHONPATH': ':'.join(str(path) for path in
                       (root / '.mindie-runtime/metadata', root / 'vllm', root / 'vllm-ascend', image)),
                   'SOC_VERSION': '1.0'}
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    # The native child reads PYTHONPATH at interpreter startup. The capture
    # suffix also runs in this already-started test process, so give its real
    # module/metadata resolver the same owned source view.
    for path in reversed(environment['PYTHONPATH'].split(':')):
        monkeypatch.syspath_prepend(path)
    monkeypatch.setenv('CXX', '1.0')
    real_version = profile.importlib.metadata.version
    monkeypatch.setattr(profile.importlib.metadata, 'version',
                        lambda name: '1.0' if name in {'torch', 'torch-npu'} else real_version(name))
    settings = {name: '1.0' for name in profile.PROFILE_FIELDS}
    settings.update(python_abi=profile.sysconfig.get_config_var('SOABI'), build_env={}, launch_env=environment,
                    compatibility_evidence='.mindie-runtime/profile-evidence/smoke.json', system_files=system)
    inputs = {name: {'native': 'a' * 64, 'dependencies': 'b' * 64, 'build_env': 'c' * 64}
              for name in ('vllm', 'vllm-ascend')}
    return root, image, library, settings, inputs


def test_native_smoke_resolves_real_owned_elf_before_image_library(native_loader):
    root, image, library, settings, inputs = native_loader
    failed = profile.native_import_smoke(root, settings, inputs)
    assert failed['passed'] is False and 'file too short' in failed['stderr']
    environment = dict(os.environ)
    files = profile.installed_native_files(root)
    settings['launch_env'] = profile.native_vendor_launch_environment(root, files, settings['launch_env'])
    assert profile.native_vendor_launch_environment(root, files, settings['launch_env']) == settings['launch_env']
    passed = profile.native_import_smoke(root, settings, inputs)
    assert passed['passed'] is True, passed['stderr']
    assert 'loaded-owned-opapi=' + str(library) in passed['stdout']
    assert settings['launch_env']['LD_LIBRARY_PATH'] == str(library.parent) + ':' + str(image)
    assert settings['launch_env']['ASCEND_CUSTOM_OPP_PATH'].endswith(':' + str(image / 'opp'))
    assert dict(os.environ) == environment


def test_actual_capture_publishes_the_environment_used_by_its_native_child(native_loader, monkeypatch):
    root, image, library, settings, inputs = native_loader
    request = {'root': str(root), 'image_digest': '1.0',
               'cann_files': [settings['system_files']['cann']['path']],
               'driver_files': [settings['system_files']['driver']['path']]}
    monkeypatch.setattr(sys, 'argv', ['capture', json.dumps(request)])
    namespace = {key: value for key, value in vars(profile).items() if not key.startswith('__')}
    namespace['_build_namespace'] = {'runtime_build_inputs': lambda *args: inputs}
    exec(compile(REMOTE_CAPTURE_SUFFIX, '<real-loader-capture>', 'exec'), namespace)
    manifest = json.loads((root / '.mindie-runtime/ready-profile.json').read_text())
    smoke = json.loads((root / '.mindie-runtime/profile-evidence/smoke.json').read_text())
    assert smoke['passed'] is True and 'loaded-owned-opapi=' + str(library) in smoke['stdout']
    assert smoke['source_mapping'] == {
        'vllm': str(root / 'vllm/vllm/__init__.py'),
        'vllm_version': '1.0',
        'vllm_ascend': str(root / 'vllm-ascend/vllm_ascend/__init__.py'),
        'vllm-ascend_version': '1.0',
        'extension': str(root / 'vllm-ascend/vllm_ascend'
                         / ('vllm_ascend_C' + importlib.machinery.EXTENSION_SUFFIXES[0])),
    }
    assert manifest['profile']['launch_env']['LD_LIBRARY_PATH'].split(':') == [str(library.parent), str(image)]
    assert manifest['profile']['launch_env']['ASCEND_CUSTOM_OPP_PATH'].split(':')[0] == str(library.parents[2])


@pytest.mark.skipif(sys.platform != 'linux', reason='native view paths describe Linux containers')
def test_native_view_upgrades_omitted_and_rebases_nonstandard_vendor_layout(monkeypatch):
    for name in ('native_vendor_paths', 'native_vendor_launch_environment'):
        monkeypatch.setattr(cache, name, getattr(profile, name), raising=False)
    relative = 'custom-repo/custom_package/_cann_ops_custom/vendors/custom_transformer/op_api/lib/libopapi.so'
    vendor = '/donor/' + str(Path(relative).parents[2])
    manifest = {'runtime_root': '/donor', 'files': {relative: {}}, 'profile': {'launch_env': {
        'LD_LIBRARY_PATH': vendor + '/op_api/lib:/image/lib', 'ASCEND_CUSTOM_OPP_PATH': '/image/opp'}}}
    current = cache.native_view_launch_environment(manifest, Path('/owned'))
    assert '/donor/' not in json.dumps(current)
    assert current['LD_LIBRARY_PATH'] == vendor.replace('/donor/', '/owned/') + '/op_api/lib:/image/lib'
    assert current['ASCEND_CUSTOM_OPP_PATH'] == vendor.replace('/donor/', '/owned/') + ':/image/opp'
