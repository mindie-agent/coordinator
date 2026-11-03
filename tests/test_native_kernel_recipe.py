from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mindie_coordinator import native_kernel_recipe as recipe
from mindie_coordinator.runtime_profile import compiled_opc_recipe


GENERATOR = r'''
import configparser, json, pathlib, sys
ini, output, unit = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
parser = configparser.ConfigParser()
parser.read(ini)
op_type = parser.sections()[0]
op = parser[op_type]['opFile.value']
for index, dtype in enumerate(parser[op_type]['input0.dtype'].split(',')):
    name = op_type + '_' + dtype
    param = output / (name + '_param.json')
    param.write_text(json.dumps({'op_type': op_type, 'dtype': dtype, 'op_list': [{'bin_filename': name}]}))
    (output / (op_type + '-' + op + '-' + str(index) + '.sh')).write_text(
        'res=$(opc $1 --main_func=' + op + ' --input_param=' + param.resolve().as_posix() +
        ' --soc_version=' + unit + ' --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic)\n'
        'printf "%s\\n" "$res"\n'
        'test -f $2/' + name + '.o || exit 1\ntest -f $2/' + name + '.json || exit 1\n')
'''

CONFIGURATOR = r'''
import json, os, pathlib, sys
root, unit = pathlib.Path(sys.argv[2]), sys.argv[4]
assert root.name == 'bin' and root.parent.name == unit and root.parent.parent.name == 'binary'
if os.environ.get('FAIL_CONFIG'):
    raise RuntimeError('configuration failed after all OPC variants completed')
for directory in root.iterdir():
    if not directory.is_dir():
        continue
    dispatch, variants = [], []
    for path in sorted(directory.glob('*.json')):
        data = json.loads(path.read_text())
        prefix = unit + '/' + directory.name + '/' + path.stem
        dispatch.append({'binPath': prefix + '.o', 'jsonPath': prefix + '.json', 'dtype': data['dtype']})
        variants.append({'dtype': data['dtype'], 'binInfo': {'jsonFilePath': prefix + '.json'}})
    (root / 'binary_info_config.json').write_text(json.dumps({'FixtureOp': {'binaryList': dispatch}}))
    (root / 'relocatable_kernel_info_config.json').write_text('{}')
    (root / (directory.name + '.json')).write_text(json.dumps({'binList': variants}))
'''

DYNAMIC = '''
from tbe.tikcpp import compile_op
from pathlib import Path
PYF_PATH = Path(__file__).parent
def get_kernel_source():
    for source in (PYF_PATH / 'op_kernel/fixture.cpp', PYF_PATH / '../ascendc/fixture/fixture.cpp'):
        if source.exists():
            return str(source)
def fixture():
    return compile_op(get_kernel_source(), ['-include/donor/old/vllm-ascend/csrc/common/compat.h'])
'''

OPC = r'''
import importlib.util, json, os, pathlib, sys, time, types
args = dict(arg[2:].split('=', 1) for arg in sys.argv[2:])
param = json.loads(pathlib.Path(args['input_param']).read_text())
def compile_op(source, options):
    content = pathlib.Path(source).read_bytes()
    assert content == b'new CPP\n', content
    assert pathlib.Path(options[0].removeprefix('-include')).read_bytes() == b'fixed header'
    if os.environ.get('FAIL_VARIANT') == param['dtype']:
        raise RuntimeError('compiler failed after actual new CPP was selected')
    time.sleep(0.05)
    target = pathlib.Path(args['output']) / param['op_list'][0]['bin_filename']
    target.with_suffix('.o').write_bytes(content + param['dtype'].encode())
    target.with_suffix('.json').write_text(json.dumps(param))
    if os.environ.get('FAIL_AFTER_OUTPUT') == param['dtype']:
        raise RuntimeError('nonzero compiler exit despite complete files')
    if os.environ.get('COMPILED_VARIANTS'):
        with open(os.environ['COMPILED_VARIANTS'], 'a') as stream:
            stream.write(param['dtype'] + '\n')
tbe = types.ModuleType('tbe'); tikcpp = types.ModuleType('tbe.tikcpp'); tikcpp.compile_op = compile_op
sys.modules.update({'tbe': tbe, 'tbe.tikcpp': tikcpp})
spec = importlib.util.spec_from_file_location('dynamic', sys.argv[1])
dynamic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dynamic)
getattr(dynamic, args['main_func'])()
'''


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def compiler(tmp_path):
    root = tmp_path / 'owned'
    vendor = 'vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/fixture_vendor'
    prefix = vendor + '/op_impl/ai_core/tbe'
    dynamic = prefix + '/fixture_impl/dynamic/fixture.py'
    source = 'csrc/moe/fixture/op_kernel/fixture.cpp'
    kernel = prefix + '/kernel'
    unit = 'ascend910_93'
    write(root / 'vllm-ascend' / source, 'new CPP\n')
    write(root / 'vllm-ascend/csrc/common/compat.h', 'fixed header')
    metadata = {
        dynamic: DYNAMIC,
        prefix + '/fixture_impl/ascendc/fixture/fixture.cpp': 'old CPP\n',
        prefix + '/fixture_impl/ascendc/common/include.h': 'shared header',
        prefix + '/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so': 'verified tiling',
        prefix + '/op_tiling/liboptiling.so': 'verified tiling alias',
        prefix + '/config/' + unit + '/aic-' + unit + '-ops-info.json': json.dumps({'FixtureOp': {
            'opFile': {'value': 'fixture'}, 'input0': {'dtype': 'float32,float16,bfloat16'}}}),
        kernel + '/config/' + unit + '/fixture.json': json.dumps({'binList': [1, 2, 3]}),
    }
    for dtype in ('float32', 'float16', 'bfloat16'):
        for suffix in ('.o', '.json'):
            metadata[kernel + '/' + unit + '/fixture/FixtureOp_' + dtype + suffix] = 'old output'
    for name, content in metadata.items():
        write(root / name, content)
    tools = root / 'vllm-ascend/csrc/cmake/scripts/util'
    for name in recipe.RECIPE_TOOLS:
        write(tools / name, GENERATOR if name == 'ascendc_bin_param_build.py' else CONFIGURATOR if name == 'ascendc_ops_config.py' else '# fixed generator dependency\n')
    donor_gen = tmp_path / 'donor-gen'
    donor_gen.mkdir()
    ini = write(tmp_path / 'donor.ini', '[FixtureOp]\nopFile.value=fixture\ninput0.dtype=float32,float16,bfloat16\n')
    subprocess.run([sys.executable, str(tools / 'ascendc_bin_param_build.py'), str(ini), str(donor_gen), unit], check=True)
    proof = {'schema_version': 1, 'tools': {name: digest(tools / name) for name in recipe.RECIPE_TOOLS},
             'variants': {unit: {'fixture': {'variants': [compiled_opc_recipe(path) for path in sorted(donor_gen.glob('*.sh'))],
                 'outputs': {name: digest(root / name) for name in metadata if name.startswith(kernel + '/' + unit + '/fixture/')}}}}}
    write(root / recipe.RECIPE_FILE, json.dumps(proof))
    files = {name: digest(root / name) for name in [*metadata, recipe.RECIPE_FILE]}
    plan = {'operator': 'fixture', 'unit': unit, 'source': source, 'kernel_root': kernel,
            'source_sha256': digest(root / 'vllm-ascend' / source), 'recipe_files': files}
    executable = write(tmp_path / 'bin/opc', '#!' + sys.executable + '\n' + OPC)
    executable.chmod(0o755)
    environment = {**os.environ, 'PATH': str(executable.parent) + os.pathsep + os.environ.get('PATH', '')}
    for key in ('OP_DEBUG_CONFIG', 'TILING_KEY', 'OPS_COMPILE_OPTIONS', 'CMAKE_ARGS'):
        environment.pop(key, None)
    return root, plan, environment, proof


def test_metadata_ini_roundtrip_preserves_every_string():
    import configparser
    description = {'input0': {'dtype': 'float32,float16', 'format': 'ND,ND'},
                   'attr_epsilon': {'defaultValue': '1e-06', 'type': 'float'}}
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read_string(recipe._ops_ini('FixtureOp', description))
    assert dict(parser['FixtureOp']) == {group + '.' + key: value for group, rows in description.items() for key, value in rows.items()}


@pytest.mark.parametrize('damage', ['missing-proof', 'unknown-version', 'unknown-tools', 'missing-op', 'missing-tiling', 'options', 'variant-options', 'variant-parameter', 'outputs', 'malformed', 'extra-cmake', 'other-python'])
def test_incomplete_recipe_falls_back_before_any_compiler(compiler, damage, monkeypatch):
    root, plan, environment, proof = compiler
    if damage == 'missing-proof':
        plan['recipe_files'].pop(recipe.RECIPE_FILE)
    elif damage == 'unknown-version':
        proof['schema_version'] = 2
    elif damage == 'unknown-tools':
        proof['tools']['opdesc_parser.py'] = 'different'
    elif damage == 'missing-op':
        proof['variants'] = {}
    elif damage == 'missing-tiling':
        plan['recipe_files'] = {path: sha for path, sha in plan['recipe_files'].items() if '/op_tiling/' not in path}
    elif damage == 'options':
        environment['OP_DEBUG_CONFIG'] = 'debug'
    elif damage == 'variant-options':
        proof['variants'][plan['unit']]['fixture']['variants'][0]['opc_args'].append('--op_debug_config=debug')
    elif damage == 'variant-parameter':
        proof['variants'][plan['unit']]['fixture']['variants'][0]['parameter_sha256'] = 'different'
    elif damage == 'outputs':
        proof['variants'][plan['unit']]['fixture']['outputs'] = {}
    elif damage == 'malformed':
        proof = []
    elif damage == 'extra-cmake':
        environment['CMAKE_ARGS'] = '-DEXTRA_KERNEL_OPTION=1'
    elif damage == 'other-python':
        environment['CMAKE_ARGS'] = '-DPython3_EXECUTABLE=/other/python -DPython_EXECUTABLE=/other/python'
    if recipe.RECIPE_FILE in plan['recipe_files']:
        write(root / recipe.RECIPE_FILE, json.dumps(proof))
        plan['recipe_files'][recipe.RECIPE_FILE] = digest(root / recipe.RECIPE_FILE)
    original = recipe.subprocess.run
    def run(args, **kwargs):
        assert args[0] != 'bash', 'compiler must not run for an ineligible recipe'
        return original(args, **kwargs)
    monkeypatch.setattr(recipe.subprocess, 'run', run)
    assert recipe.compile_kernel_recipe(root, plan, environment, read_recipe=compiled_opc_recipe) is False
    assert not (root / 'vllm-ascend/csrc/build').exists()
    assert not list((root / '.mindie-runtime').glob('kernel-recipe-*'))


def test_changed_verified_recipe_is_fatal(compiler):
    root, plan, environment, _ = compiler
    write(root / recipe.RECIPE_FILE, '{}')
    with pytest.raises(ValueError, match='verified kernel recipe file changed'):
        recipe.compile_kernel_recipe(root, plan, environment, read_recipe=compiled_opc_recipe)


@pytest.mark.skipif(os.name == 'nt', reason='the real owned compiler commands execute on Linux')
def test_all_variants_use_current_cpp_in_real_compiler_processes(compiler):
    root, plan, environment, _ = compiler
    before = {name: (root / name).read_bytes() for name in plan['recipe_files']}
    environment['MAX_JOBS'] = '3'
    assert recipe.compile_kernel_recipe(root, plan, environment, read_recipe=compiled_opc_recipe)
    output = root / 'vllm-ascend/csrc/build/binary' / plan['unit'] / 'bin/fixture'
    assert sorted(path.read_bytes() for path in output.glob('*.o')) == [b'new CPP\nbfloat16', b'new CPP\nfloat16', b'new CPP\nfloat32']
    assert len(list(output.glob('*.json'))) == 3
    assert {name: (root / name).read_bytes() for name in plan['recipe_files']} == before
    assert not list((root / '.mindie-runtime').glob('kernel-recipe-*'))


@pytest.mark.skipif(os.name == 'nt', reason='the real owned compiler commands execute on Linux')
def test_failed_variant_waits_for_siblings_and_never_publishes_or_replays(compiler):
    root, plan, environment, _ = compiler
    completed = root / 'completed'
    environment.update(FAIL_VARIANT='float16', COMPILED_VARIANTS=str(completed), MAX_JOBS='3')
    with pytest.raises(subprocess.CalledProcessError):
        recipe.compile_kernel_recipe(root, plan, environment, read_recipe=compiled_opc_recipe)
    assert sorted(completed.read_text().splitlines()) == ['bfloat16', 'float32']
    assert not (root / 'vllm-ascend/csrc/build').exists()
    stages = list((root / '.mindie-runtime').glob('kernel-recipe-*'))
    assert len(stages) == 1
    assert len(list((stages[0] / 'binary' / plan['unit'] / 'bin/fixture').glob('*.o'))) == 2


@pytest.mark.skipif(os.name == 'nt', reason='the real owned compiler commands execute on Linux')
def test_complete_files_do_not_hide_a_nonzero_opc_exit(compiler):
    root, plan, environment, _ = compiler
    environment['FAIL_AFTER_OUTPUT'] = 'float16'
    with pytest.raises(subprocess.CalledProcessError):
        recipe.compile_kernel_recipe(root, plan, environment, read_recipe=compiled_opc_recipe)
    assert not (root / 'vllm-ascend/csrc/build').exists()


@pytest.mark.skipif(os.name == 'nt', reason='the real owned compiler commands execute on Linux')
def test_configuration_failure_preserves_completed_outputs_and_source_proofs(compiler, capsys):
    root, plan, environment, _ = compiler
    environment['FAIL_CONFIG'] = '1'
    with pytest.raises(subprocess.CalledProcessError):
        recipe.compile_kernel_recipe(root, plan, environment, read_recipe=compiled_opc_recipe)
    stages = list((root / '.mindie-runtime').glob('kernel-recipe-*'))
    assert len(stages) == 1
    assert str(stages[0]) in capsys.readouterr().out
    output = stages[0] / 'binary' / plan['unit'] / 'bin/fixture'
    assert len(list(output.glob('*.o'))) == 3 and len(list(output.glob('*.json'))) == 3
    assert len(list(stages[0].glob('*.source.json'))) == 3
    assert not (root / 'vllm-ascend/csrc/build').exists()


@pytest.mark.skipif(os.name == 'nt', reason='the real owned compiler commands execute on Linux')
def test_rendered_owner_entry_compiles_merges_and_preserves_next_recipe(compiler):
    from mindie_coordinator.parity import runtime_install_step_script, task_python_exports
    from mindie_coordinator.runtime_profile import capture_kernel_compile_recipe
    root, plan, environment, _ = compiler
    op, unit = plan['operator'], plan['unit']
    config = root / plan['kernel_root'] / 'config' / unit
    variants, dispatch = [], []
    for dtype in ('bfloat16', 'float16', 'float32'):
        path = unit + '/' + op + '/FixtureOp_' + dtype
        variants.append({'dtype': dtype, 'binInfo': {'jsonFilePath': path + '.json'}})
        dispatch.append({'dtype': dtype, 'binPath': path + '.o', 'jsonPath': path + '.json'})
    write(config / 'fixture.json', json.dumps({'binList': variants}))
    other = {'binaryList': [{'binPath': unit + '/other/keep.o', 'jsonPath': unit + '/other/keep.json'}]}
    write(config / 'binary_info_config.json', json.dumps({'FixtureOp': {'binaryList': dispatch}, 'Other': other}))
    write(config / 'relocatable_kernel_info_config.json', '{}')
    plan['recipe_files'][(config / 'fixture.json').relative_to(root).as_posix()] = digest(config / 'fixture.json')
    plan['installed_source'] = next(name for name in plan['recipe_files'] if name.endswith('/ascendc/fixture/fixture.cpp'))
    marker = write(root / '.mindie-runtime/native-incremental.json', json.dumps(plan))
    shell = runtime_install_step_script(runtime_root=str(root), marker_dirname='.runtime',
        container_identity='fixture', step='install-vllm-ascend-incremental', python=sys.executable)
    program = shell.split("<<'MINDIE_NATIVE_INCREMENTAL'\n", 1)[1].split('\nMINDIE_NATIVE_INCREMENTAL', 1)[0]
    # Run the actual owner shell exports, including its nonempty CMAKE_ARGS,
    # before executing the rendered compiler program.
    command = '\n'.join(task_python_exports(sys.executable)) + '\n"$PYTHON" - "$1" <<\'COMPILER\'\n' + program + '\nCOMPILER\n'
    completed = subprocess.run(['bash', '-c', command, 'fixture', str(root)], env=environment, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert 'recipe OPC variants=3' in completed.stdout and 'bash build.sh' not in completed.stdout
    assert json.loads(marker.read_text())['status'] == 'compiled'
    assert (root / plan['installed_source']).read_bytes() == b'new CPP\n'
    assert json.loads((config / 'binary_info_config.json').read_text())['Other'] == other
    capture_kernel_compile_recipe(root)
    captured = json.loads((root / recipe.RECIPE_FILE).read_text())['variants'][unit][op]
    assert len(captured['variants']) == 3 and len(captured['outputs']) == 6
    assert all(digest(root / name) == sha for name, sha in captured['outputs'].items())


def test_ini_restores_streaming_input_order_from_sorted_json():
    values = {'dtype': 'float32', 'format': 'ND', 'name': 'tensor', 'paramType': 'required'}
    ini = recipe._ops_ini('FixtureOp', {'input10': values, 'input2': values, 'input0': values})
    assert ini.index('input0.name') < ini.index('input0.dtype') < ini.index('input2.name') < ini.index('input10.name')


@pytest.mark.parametrize('wrong', ['lookup', 'compile'])
def test_dynamic_guard_rejects_old_cpp_in_actual_python_process(tmp_path, wrong):
    source = write(tmp_path / 'new.cpp', 'new CPP')
    old = write(tmp_path / 'old.cpp', 'old CPP')
    lookup = repr(str(old)) if wrong == 'lookup' else repr(str(source))
    dynamic = 'from tbe.tikcpp import compile_op\ndef get_kernel_source(): return ' + lookup + '\n'
    code = recipe._guarded_dynamic(dynamic, source, digest(source), tmp_path)
    call = 'get_kernel_source()' if wrong == 'lookup' else 'compile_op(' + repr(str(old)) + ')'
    script = '''import sys, types
tbe = types.ModuleType('tbe'); tikcpp = types.ModuleType('tbe.tikcpp')
tikcpp.compile_op = lambda *a: (_ for _ in ()).throw(AssertionError('old CPP must not reach compiler'))
sys.modules.update({'tbe': tbe, 'tbe.tikcpp': tikcpp})
''' + code + '\n' + call
    result = subprocess.run([sys.executable, '-c', script], text=True, capture_output=True)
    assert result.returncode != 0
    assert 'did not receive the fixed owned CPP' in result.stderr
    assert 'old CPP must not reach compiler' not in result.stderr


def test_unknown_dynamic_absolute_source_path_falls_back(tmp_path):
    source = write(tmp_path / 'new.cpp', 'CPP')
    write(tmp_path / 'vllm-ascend/csrc/common/compat.h', 'fixed header')
    with pytest.raises(recipe.RecipeUnavailable, match='unsupported source path'):
        recipe._guarded_dynamic(DYNAMIC + "\npath='/donor/vllm-ascend/csrc/unsafe.cpp'", source, digest(source), tmp_path)
