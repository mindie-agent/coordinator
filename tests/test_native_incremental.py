from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from mindie_coordinator import native_incremental as native
from mindie_coordinator.build_inputs import VLLM_ASCEND_REINSTALL_PATTERNS, build_input_fingerprints, submodule_content


def blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


@pytest.fixture
def kernel(tmp_path):
    root, bundle = tmp_path / 'execution', tmp_path / 'bundle'
    source = 'csrc/moe/add_rms_norm_bias/op_kernel/add_rms_norm_bias.cpp'
    prefix = 'vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe'
    installed = prefix + '/custom_transformer_impl/ascendc/add_rms_norm_bias/add_rms_norm_bias.cpp'
    old, new = b'int result = 1;\n', b'int result = 2;\n'
    for path, data in ((root / 'vllm-ascend' / source, new), (bundle / installed, old)):
        path.parent.mkdir(parents=True)
        path.write_bytes(data)
    entries = {source: ('100644', blob(new)), 'CMakeLists.txt': ('100644', blob(b'project(native)'))}
    prior = {**entries, source: ('100644', blob(old))}
    before = {'vllm': 'same', 'vllm-ascend': native.native_tree_digest(prior)}
    current = {'vllm': 'same', 'vllm-ascend': native.native_tree_digest(entries)}
    config = prefix + '/kernel/config/ascend910_93/add_rms_norm_bias.json'
    manifest = {'preparation': {'native': before}, 'files': {
        installed: {'sha256': hashlib.sha256(old).hexdigest()}, config: {'sha256': 'unused'}}}
    return root, bundle, source, installed, entries, manifest, {'native': current}


def test_single_translation_unit_reconstructs_complete_native_identity(kernel):
    root, bundle, source, installed, entries, manifest, preparation = kernel
    plan = native.kernel_rebuild_plan(root, bundle, manifest, preparation, entries)
    assert plan['operator'] == 'add_rms_norm_bias'
    assert plan['source'] == source and plan['installed_source'] == installed
    assert plan['native_from'] == manifest['preparation']['native']['vllm-ascend']
    assert plan['native_to'] == preparation['native']['vllm-ascend']


def test_recipe_plan_retains_actual_installed_tbe_tiling_layout(kernel):
    root, bundle, source, installed, entries, manifest, preparation = kernel
    # Paths from the captured installed manifest: tiling belongs below tbe,
    # beside kernel/config, rather than beside the vendor's op_api directory.
    tbe = installed.split('/custom_transformer_impl/ascendc/', 1)[0]
    tiling = [tbe + '/op_tiling/liboptiling.so',
              tbe + '/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so']
    manifest['files']['.mindie-runtime/kernel-compile-recipe.json'] = {'sha256': 'recipe'}
    manifest['files'].update({path: {'sha256': 'tiling-' + str(index)} for index, path in enumerate(tiling)})
    plan = native.kernel_rebuild_plan(root, bundle, manifest, preparation, entries)
    assert {path: plan['recipe_files'][path] for path in tiling} == {
        path: manifest['files'][path]['sha256'] for path in tiling}


@pytest.mark.parametrize('change', ['cmake', 'submodule', 'second-kernel', 'mode', 'add', 'vllm', 'unfixed'])
def test_other_native_changes_cannot_borrow_single_kernel_proof(kernel, change):
    root, bundle, source, installed, entries, manifest, preparation = kernel
    if change == 'cmake':
        entries['CMakeLists.txt'] = ('100644', blob(b'changed build flags'))
    elif change == 'submodule':
        entries['csrc/third_party/catlass'] = ('160000', 'new-submodule-tree')
    elif change == 'second-kernel':
        entries['csrc/moe/other/op_kernel/other.cpp'] = ('100644', blob(b'changed'))
    elif change == 'mode':
        entries[source] = ('100755', entries[source][1])
    elif change == 'add':
        entries[source + '.h'] = ('100644', blob(b'new header'))
    elif change == 'vllm':
        preparation['native']['vllm'] = 'changed-vllm'
    else:
        preparation['native']['vllm-ascend'] = 'different-admitted-snapshot'
    if change != 'unfixed':
        preparation['native']['vllm-ascend'] = native.native_tree_digest(entries)
    assert native.kernel_rebuild_plan(root, bundle, manifest, preparation, entries) is None


def test_corrupt_packaged_source_is_rejected(kernel):
    root, bundle, source, installed, entries, manifest, preparation = kernel
    (bundle / installed).write_bytes(b'corruption')
    with pytest.raises(ValueError, match='changed in bundle'):
        native.kernel_rebuild_plan(root, bundle, manifest, preparation, entries)


def test_working_source_must_still_match_admitted_git_blob(kernel):
    root, bundle, source, installed, entries, manifest, preparation = kernel
    (root / 'vllm-ascend' / source).write_bytes(b'changed after admission')
    with pytest.raises(ValueError, match='source changed before planning'):
        native.kernel_rebuild_plan(root, bundle, manifest, preparation, entries)


def test_native_tree_token_contract_is_identical_to_existing_fingerprint(tmp_path):
    for filename in ('CMakeLists.txt', 'csrc/moe/add/op_kernel/中文.cpp', 'package/python.py'):
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(filename, encoding='utf-8')
    def git(*args):
        return subprocess.run(['git', '-C', str(tmp_path), *args], capture_output=True, check=True)
    git('init')
    git('add', '.')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-m', 'fixed inputs')
    entries = native.native_tree_entries(tmp_path, VLLM_ASCEND_REINSTALL_PATTERNS, submodule_content)
    assert 'package/python.py' not in entries
    assert native.native_tree_digest(entries) == build_input_fingerprints(tmp_path, 'HEAD', VLLM_ASCEND_REINSTALL_PATTERNS)['native']


def prepared_merge(kernel):
    root, bundle, source, installed, entries, manifest, preparation = kernel
    plan = native.kernel_rebuild_plan(root, bundle, manifest, preparation, entries)
    base = root / plan['kernel_root']
    config = base / 'config/ascend910_93'
    config.mkdir(parents=True)
    op = base / 'ascend910_93/add_rms_norm_bias'
    op.mkdir(parents=True)
    (op / 'old.o').write_bytes(b'old kernel')
    host = root / 'host-library.so'
    host.write_bytes(b'full shared host library')
    def row(op, filename):
        return {'binaryList': [{'binPath': 'ascend910_93/' + op + '/' + filename + '.o',
                                'jsonPath': 'ascend910_93/' + op + '/' + filename + '.json',
                                'simplifiedKey': ['dtype=float'], 'coreType': 2}]}
    def per_op(filename):
        return json.dumps({'binList': [{'simplifiedKey': ['dtype=float'], 'binInfo': {
            'jsonFilePath': 'ascend910_93/add_rms_norm_bias/' + filename + '.json'}}]})
    original = {'AddRmsNormBias': row('add_rms_norm_bias', 'old'), 'Other': row('other', 'keep')}
    (config / 'binary_info_config.json').write_text(json.dumps(original))
    (config / 'relocatable_kernel_info_config.json').write_text('{}')
    (config / 'add_rms_norm_bias.json').write_text(per_op('old'))
    output = root / 'vllm-ascend/csrc/build/binary/ascend910_93/bin'
    (output / 'add_rms_norm_bias').mkdir(parents=True)
    (output / 'add_rms_norm_bias/new.o').write_bytes(b'new kernel')
    (output / 'add_rms_norm_bias/new.json').write_text('{}')
    (output / 'binary_info_config.json').write_text(json.dumps({'AddRmsNormBias': row('add_rms_norm_bias', 'new')}))
    (output / 'relocatable_kernel_info_config.json').write_text('{}')
    (output / 'add_rms_norm_bias.json').write_text(per_op('new'))
    target = root / installed
    target.parent.mkdir(parents=True)
    target.write_bytes((bundle / installed).read_bytes())
    return root, plan, config, output, original, host


def test_merge_replaces_only_selected_operator_and_preserves_other_dispatch(kernel):
    root, plan, config, output, original, host = prepared_merge(kernel)
    changed = native.merge_kernel_outputs(root, plan)
    assert any(path.endswith('new.o') for path in changed)
    merged = json.loads((config / 'binary_info_config.json').read_text())
    assert merged['Other'] == original['Other']
    assert merged['AddRmsNormBias']['binaryList'][0]['binPath'].endswith('new.o')
    assert host.read_bytes() == b'full shared host library'
    assert not (root / plan['kernel_root'] / 'ascend910_93/add_rms_norm_bias/old.o').exists()
    assert (root / plan['installed_source']).read_bytes() == (root / 'vllm-ascend' / plan['source']).read_bytes()


@pytest.mark.parametrize('damage', ['missing-binary', 'other-op', 'traversal', 'missing-config', 'missing-coverage',
                                   'missing-variant', 'per-op-variant', 'per-op-missing-output'])
def test_invalid_incremental_output_keeps_installed_baseline_untouched(kernel, damage):
    root, plan, config, output, original, host = prepared_merge(kernel)
    if damage == 'missing-binary':
        (output / 'add_rms_norm_bias/new.o').unlink()
    elif damage == 'other-op':
        (output / 'binary_info_config.json').write_text(json.dumps({'Other': original['Other']}))
    elif damage == 'traversal':
        row = json.loads((output / 'binary_info_config.json').read_text())
        row['AddRmsNormBias']['binaryList'][0]['binPath'] = 'ascend910_93/add_rms_norm_bias/../new.o'
        (output / 'binary_info_config.json').write_text(json.dumps(row))
    elif damage == 'missing-config':
        (output / 'add_rms_norm_bias.json').unlink()
    elif damage == 'missing-variant':
        extra = copy.deepcopy(original['AddRmsNormBias']['binaryList'][0])
        extra['simplifiedKey'] = ['dtype=half']
        original['AddRmsNormBias']['binaryList'].append(extra)
        (config / 'binary_info_config.json').write_text(json.dumps(original))
    elif damage in ('per-op-variant', 'per-op-missing-output'):
        filename = output / 'add_rms_norm_bias.json'
        value = json.loads(filename.read_text())
        if damage == 'per-op-variant':
            value['binList'][0]['simplifiedKey'] = ['dtype=half']
        else:
            value['binList'][0]['binInfo']['jsonFilePath'] = 'ascend910_93/add_rms_norm_bias/absent.json'
        filename.write_text(json.dumps(value))
    else:
        (output / 'binary_info_config.json').write_text('{}')
    with pytest.raises((ValueError, FileNotFoundError)):
        native.merge_kernel_outputs(root, plan)
    assert json.loads((config / 'binary_info_config.json').read_text()) == original
    assert (root / plan['kernel_root'] / 'ascend910_93/add_rms_norm_bias/old.o').read_bytes() == b'old kernel'
    assert host.read_bytes() == b'full shared host library'


def test_compile_requires_fixed_source_and_fresh_build_directory(kernel, monkeypatch):
    root, bundle, source, installed, entries, manifest, preparation = kernel
    plan = native.kernel_rebuild_plan(root, bundle, manifest, preparation, entries)
    marker = root / '.mindie-runtime/native-incremental.json'
    marker.parent.mkdir()
    marker.write_text(json.dumps(plan))
    monkeypatch.setattr(native.subprocess, 'run', lambda *a, **k: pytest.fail('must reject before building'))
    (root / 'vllm-ascend' / source).write_bytes(b'mutated admitted source')
    with pytest.raises(ValueError, match='source changed'):
        native.build_incremental_kernel(root)
    (root / 'vllm-ascend' / source).write_bytes(b'int result = 2;\n')
    (root / 'vllm-ascend/csrc/build').mkdir()
    with pytest.raises(ValueError, match='fresh owned build'):
        native.build_incremental_kernel(root)


def test_compile_uses_current_owned_sources_and_merges_successful_result(kernel, monkeypatch):
    root, plan, config, output, original, host = prepared_merge(kernel)
    build = root / 'vllm-ascend/csrc/build'
    generated = root / 'fixture-generated'
    build.rename(generated)
    marker = root / '.mindie-runtime/native-incremental.json'
    marker.parent.mkdir()
    marker.write_text(json.dumps(plan))
    catlass = root / 'vllm-ascend/csrc/third_party/catlass/include'
    catlass.mkdir(parents=True)
    monkeypatch.setenv('CXXFLAGS', '-captured-build-flags')
    monkeypatch.setenv('CPATH', '/image/includes')
    def compile(argv, **kwargs):
        assert argv == ['bash', 'build.sh', '--opkernel', '--ops=add_rms_norm_bias', '--soc=ascend910_93']
        assert kwargs['cwd'] == root / 'vllm-ascend/csrc'
        assert kwargs['env']['CXXFLAGS'] == '-captured-build-flags'
        assert kwargs['env']['CPATH'] == str(catlass) + ':/image/includes'
        assert kwargs['check'] is True
        generated.rename(build)
    monkeypatch.setattr(native.subprocess, 'run', compile)
    result = native.build_incremental_kernel(root)
    assert result['status'] == 'compiled'
    assert json.loads(marker.read_text()) == result
    assert host.read_bytes() == b'full shared host library'
    assert json.loads((config / 'binary_info_config.json').read_text())['Other'] == original['Other']


def test_renderer_contains_owned_single_operator_build_without_pip():
    from mindie_coordinator.parity import runtime_install_step_script
    script = runtime_install_step_script(runtime_root='/execution', marker_dirname='.runtime',
                                         container_identity='mindie-fixture', step='install-vllm-ascend-incremental', python='/python')
    assert 'build_incremental_kernel(Path(sys.argv[1]), compile_recipe=' in script
    assert 'read_recipe=compiled_opc_recipe' in script
    assert "'--opkernel', '--ops=' + plan['operator']" in script
    assert 'pip install --no-deps -v -e .' not in script
    assert 'spec.get("build_env", {})' in script
    assert 'SETUPTOOLS_SCM_PRETEND_VERSION=' in script
    program = script.split("<<'MINDIE_NATIVE_INCREMENTAL'\n", 1)[1].split('\nMINDIE_NATIVE_INCREMENTAL', 1)[0]
    compile(program, '<owned incremental program>', 'exec')
