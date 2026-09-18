import json
from pathlib import Path

import pytest

from mindie_coordinator import runtime_profile as profile


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def compiler_script(gen, index=0, flags=''):
    parameter = write(gen / f'Op_{index}_param.json', json.dumps({'op_list': [{'bin_filename': f'Op_{index}'}]}))
    return write(gen / f'Op-test_op-{index}.sh',
                 '#!/bin/bash\nres=$(opc $1 --main_func=test_op '
                 f'--input_param={parameter.as_posix()} --soc_version=Ascend910_9391 '
                 f'--output=$2 --impl_mode=high_performance,optional {flags})\n')


def build_fixture(root, count=3):
    tools = root / 'vllm-ascend/csrc/cmake/scripts/util'
    for name in ('ascendc_bin_param_build.py', 'ascendc_ops_config.py',
                 'ascendc_impl_build.py', 'opdesc_parser.py', 'const_var.py'):
        write(tools / name, '# fixed source generator\n')
    gen = root / 'vllm-ascend/csrc/build/binary/ascend910_93/gen'
    destination = root / 'vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom/op_impl/ai_core/tbe/kernel/ascend910_93/test_op'
    for index in range(count):
        compiler_script(gen, index)
        for suffix in ('.o', '.json'):
            name = f'Op_{index}' + suffix
            write(gen.parent / 'bin/test_op' / name, name)
            write(destination / name, name)
    write(root / 'vllm-ascend/vllm_ascend/vllm_ascend_C.so', 'extension')
    return gen, destination


def recipe(root):
    return root / '.mindie-runtime/kernel-compile-recipe.json'


def test_literal_flags_and_parameter_semantics_survive_normalization(tmp_path):
    script = compiler_script(tmp_path, flags='--op_debug_config=oom --tiling_key="1,2"')
    before = profile.compiled_opc_recipe(script)
    assert '--op_debug_config=oom' in before['opc_args']
    assert '--tiling_key=1,2' in before['opc_args']
    assert '--input_param=<parameter>' in before['opc_args']
    parameter = tmp_path / 'Op_0_param.json'
    parameter.write_text(json.dumps(json.loads(parameter.read_text()), indent=4))
    assert profile.compiled_opc_recipe(script) == before
    parameter.write_text('{"op_list":[{"bin_filename":"changed"}]}')
    assert profile.compiled_opc_recipe(script)['parameter_sha256'] != before['parameter_sha256']


@pytest.mark.parametrize('flags', ['--debug=$UNKNOWN', '--debug=$(touch /tmp/x)',
                                  '--debug=foo;touch x', '--main_func=other', '--debug=`id`'])
def test_dynamic_or_ambiguous_command_has_no_recipe(tmp_path, flags):
    assert profile.compiled_opc_recipe(compiler_script(tmp_path, flags=flags)) is None


def test_parameter_outside_generated_directory_is_not_read(tmp_path):
    script = compiler_script(tmp_path / 'gen')
    outside = write(tmp_path / 'outside_param.json', '{}')
    script.write_text(script.read_text().replace(str(tmp_path / 'gen' / 'Op_0_param.json').replace('\\', '/'), outside.as_posix()))
    assert profile.compiled_opc_recipe(script) is None


@pytest.mark.parametrize('option', ['--main_func', '--soc_version', '--input_param'])
def test_required_option_needs_a_literal_value(tmp_path, option):
    script = compiler_script(tmp_path)
    import re
    script.write_text(re.sub(re.escape(option) + r'=[^ ]+', option, script.read_text()))
    assert profile.compiled_opc_recipe(script) is None


def test_complete_actual_build_is_captured_as_installed_metadata(tmp_path):
    build_fixture(tmp_path)
    profile.capture_kernel_compile_recipe(tmp_path)
    data = json.loads(recipe(tmp_path).read_text())
    assert len(data['variants']['ascend910_93']['test_op']['variants']) == 3
    assert len(data['tools']) == 5
    assert profile.installed_native_files(tmp_path)['.mindie-runtime/kernel-compile-recipe.json'] == 'metadata'


@pytest.mark.parametrize('change', ['missing_object', 'wrong_installed_object', 'malformed_script', 'missing_generator', 'missing_scripts'])
def test_incomplete_new_build_cannot_retain_stale_recipe(tmp_path, change):
    gen, destination = build_fixture(tmp_path)
    profile.capture_kernel_compile_recipe(tmp_path)
    assert recipe(tmp_path).exists()
    if change == 'missing_object':
        (gen.parent / 'bin/test_op/Op_0.o').unlink()
    elif change == 'wrong_installed_object':
        (destination / 'Op_0.o').write_text('different')
    elif change == 'malformed_script':
        (gen / 'Op-test_op-0.sh').write_text('eval "$OPC"')
    elif change == 'missing_generator':
        (tmp_path / 'vllm-ascend/csrc/cmake/scripts/util/const_var.py').unlink()
    else:
        for path in gen.glob('*.sh'):
            path.unlink()
    profile.capture_kernel_compile_recipe(tmp_path)
    assert not recipe(tmp_path).exists()


def test_source_only_restore_preserves_existing_recipe_without_recapture(tmp_path):
    target = write(recipe(tmp_path), '{"verified": "existing bytes"}\n')
    before = target.read_bytes()
    profile.capture_kernel_compile_recipe(tmp_path)
    assert target.read_bytes() == before


def test_changed_generator_drops_other_operator_proofs(tmp_path):
    build_fixture(tmp_path)
    profile.capture_kernel_compile_recipe(tmp_path)
    data = json.loads(recipe(tmp_path).read_text())
    data['variants']['ascend910_93']['old_other_op'] = {'variants': []}
    recipe(tmp_path).write_text(json.dumps(data))
    generator = tmp_path / 'vllm-ascend/csrc/cmake/scripts/util/const_var.py'
    generator.write_text('# changed generator')
    profile.capture_kernel_compile_recipe(tmp_path)
    assert 'old_other_op' not in json.loads(recipe(tmp_path).read_text())['variants']['ascend910_93']


@pytest.mark.parametrize('name', ['Op_0', 'Nonexistent'])
def test_parameters_must_cover_exact_unique_output_names(tmp_path, name):
    gen, _ = build_fixture(tmp_path)
    for parameter in gen.glob('*_param.json'):
        parameter.write_text(json.dumps({'op_list': [{'bin_filename': name}]}))
    profile.capture_kernel_compile_recipe(tmp_path)
    assert not recipe(tmp_path).exists()


def test_missing_gen_cannot_transfer_old_proof_to_changed_output(tmp_path):
    import shutil
    gen, destination = build_fixture(tmp_path)
    profile.capture_kernel_compile_recipe(tmp_path)
    shutil.rmtree(gen)
    (destination / 'Op_0.o').write_text('different output')
    profile.capture_kernel_compile_recipe(tmp_path)
    assert not recipe(tmp_path).exists()


@pytest.mark.parametrize('old', [[], {'schema_version': 1, 'variants': {'ascend910_93': []}}])
def test_unknown_previous_metadata_does_not_break_normal_capture(tmp_path, old):
    build_fixture(tmp_path)
    write(recipe(tmp_path), json.dumps(old))
    profile.capture_kernel_compile_recipe(tmp_path)
    assert len(json.loads(recipe(tmp_path).read_text())['variants']['ascend910_93']['test_op']['variants']) == 3
