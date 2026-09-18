"""Compile qualified installed operator metadata without a CMake build tree.

This optional path consumes only a verified compilation recipe. Legacy bundles
without complete option evidence continue through the ordinary source build.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile


RECIPE_FILE = '.mindie-runtime/kernel-compile-recipe.json'
RECIPE_TOOLS = ('ascendc_bin_param_build.py', 'ascendc_ops_config.py',
                'ascendc_impl_build.py', 'opdesc_parser.py', 'const_var.py')


class RecipeUnavailable(ValueError):
    """Known unsupported metadata, before any compiler has been launched."""


def _recipe_child(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or '..' in relative.parts:
        raise ValueError('unsafe kernel recipe path')
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('symlink in kernel recipe path')
    return path


def _recipe_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ops_ini(op_type: str, description: dict) -> str:
    """Invert the upstream nested-string INI-to-JSON transform exactly."""
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', op_type) or not isinstance(description, dict):
        raise RecipeUnavailable('unsupported operator description')
    lines = ['[' + op_type + ']']
    def section_order(name):
        indexed = re.fullmatch(r'(input|output)(\d+)', name)
        return (1 if indexed[1] == 'input' else 2, int(indexed[2])) if indexed else (0 if name == 'attr' else 3, name)
    for section in sorted(description, key=section_order):
        values = description[section]
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', section) or not isinstance(values, dict):
            raise RecipeUnavailable('unsupported operator metadata section')
        # Upstream's streaming parser advances input/output indices on .name;
        # installed JSON member order is not a semantic ordering guarantee.
        for key in sorted(values, key=lambda key: (key != 'name', key)):
            value = values[key]
            if (not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', key) or not isinstance(value, str)
                    or any(char in value for char in '\r\n\0')):
                raise RecipeUnavailable('operator metadata is not reversible string data')
            lines.append(section + '.' + key + '=' + value)
    return '\n'.join(lines) + '\n'


def _guarded_dynamic(text: str, source: Path, sha256: str, root: Path) -> str:
    """Retain generated logic but fence both source lookup and compiler entry."""
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise RecipeUnavailable('unknown generated dynamic syntax') from exc
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    imported = {name.asname or name.name for node in tree.body if isinstance(node, ast.ImportFrom)
                and node.module == 'tbe.tikcpp' for name in node.names}
    if 'get_kernel_source' not in functions or 'compile_op' not in imported:
        raise RecipeUnavailable('unknown generated dynamic compiler interface')
    # Source-derived absolute include paths follow the fixed source view. Do
    # not execute a donor's old source directory through a generated wrapper.
    class Rebase(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and '/vllm-ascend/' in node.value:
                match = re.fullmatch(r'(-include|-I)(/[^\r\n]*?)/vllm-ascend/(csrc/[^\r\n]+)', node.value)
                if not match:
                    raise RecipeUnavailable('unsupported source path in dynamic compiler')
                option, _, relative = match.groups()
                current = _recipe_child(root, 'vllm-ascend/' + relative)
                if not current.exists():
                    raise RecipeUnavailable('dynamic compiler include is absent from fixed sources')
                node.value = option + str(current)
            return node
    tree = Rebase().visit(tree)
    code = ast.unparse(tree)
    guard = '''
import hashlib as _mindie_hashlib
import json as _mindie_json
import os as _mindie_os
import pathlib as _mindie_pathlib
_mindie_lookup = get_kernel_source
_mindie_compile = compile_op
def _mindie_check_source(value):
    actual = _mindie_pathlib.Path(value).resolve()
    if actual != _mindie_pathlib.Path(__MINDIE_SOURCE_PATH_LITERAL__) or _mindie_hashlib.sha256(actual.read_bytes()).hexdigest() != __MINDIE_SOURCE_SHA_LITERAL__:
        raise RuntimeError('kernel compiler did not receive the fixed owned CPP')
    return value
def get_kernel_source(*args, **kwargs):
    return _mindie_check_source(_mindie_lookup(*args, **kwargs))
def compile_op(source, *args, **kwargs):
    _mindie_check_source(source)
    _mindie_pathlib.Path(_mindie_os.environ['MINDIE_KERNEL_SOURCE_PROOF']).write_text(_mindie_json.dumps({'source': str(_mindie_pathlib.Path(source).resolve()), 'sha256': __MINDIE_SOURCE_SHA_LITERAL__}))
    return _mindie_compile(source, *args, **kwargs)
'''.replace('__MINDIE_SOURCE_PATH_LITERAL__', repr(str(source.resolve()))).replace('__MINDIE_SOURCE_SHA_LITERAL__', repr(sha256))
    # Substitute explicit tokens, not user code or an environment-selected path.
    return code + '\n' + guard


def _recipe_inputs(root: Path, plan: dict, environment: dict) -> tuple[list, str, str]:
    files = plan.get('recipe_files') or {}
    if RECIPE_FILE not in files:
        raise RecipeUnavailable('compiled option evidence is missing')
    for name, expected in files.items():
        path = _recipe_child(root, name)
        if not path.is_file() or _recipe_digest(path) != expected:
            raise ValueError('verified kernel recipe file changed: ' + name)
    try:
        recipe = json.loads(_recipe_child(root, RECIPE_FILE).read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise RecipeUnavailable('compiled option evidence is malformed') from exc
    try:
        cmake_args = shlex.split(environment.get('CMAKE_ARGS', ''))
    except ValueError as exc:
        raise RecipeUnavailable('unrecognized CMake options') from exc
    interpreter_args = [f'-DPython3_EXECUTABLE={sys.executable}', f'-DPython_EXECUTABLE={sys.executable}']
    if (not isinstance(recipe, dict) or recipe.get('schema_version') != 1
            or any(environment.get(key) for key in ('OP_DEBUG_CONFIG', 'TILING_KEY', 'OPS_COMPILE_OPTIONS'))
            or (cmake_args and cmake_args != interpreter_args)):
        raise RecipeUnavailable('kernel compilation options are not fully supported')
    tools = root / 'vllm-ascend/csrc/cmake/scripts/util'
    if not isinstance(recipe.get('tools'), dict):
        raise RecipeUnavailable('compiler generator identity is missing')
    for name in RECIPE_TOOLS:
        if _recipe_digest(_recipe_child(tools, name)) != recipe.get('tools', {}).get(name):
            raise RecipeUnavailable('fixed source compiler generator differs from its proof')
    op, unit = plan['operator'], plan['unit']
    try:
        entry = recipe['variants'][unit][op]
        variants = entry['variants']
    except (KeyError, TypeError) as exc:
        raise RecipeUnavailable('operator has no effective variant recipe') from exc
    if not isinstance(variants, list) or not variants or any(not isinstance(row, dict) for row in variants):
        raise RecipeUnavailable('operator variant recipe is incomplete')
    outputs = {name: sha for name, sha in files.items() if name.startswith(plan['kernel_root'] + '/' + unit + '/' + op + '/')}
    if not outputs or entry.get('outputs') != outputs:
        raise RecipeUnavailable('compiler recipe is not bound to the installed operator outputs')
    prefix = plan['kernel_root'].removesuffix('/kernel')
    info = prefix + '/config/' + unit + '/aic-' + unit + '-ops-info.json'
    dynamics = [name for name in files if name.startswith(prefix + '/') and name.endswith('/dynamic/' + op + '.py')]
    tiling = [name for name in files if name.startswith(prefix + '/op_tiling/') and name.endswith('.so')]
    if info not in files or len(dynamics) != 1 or not tiling:
        raise RecipeUnavailable('installed compiler metadata or tiling is missing')
    try:
        descriptions = json.loads(_recipe_child(root, info).read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise RecipeUnavailable('operator description is malformed') from exc
    if not isinstance(descriptions, dict):
        raise RecipeUnavailable('operator description has unsupported shape')
    selected = [(name, row) for name, row in descriptions.items()
                if isinstance(row, dict) and isinstance(row.get('opFile'), dict) and row['opFile'].get('value') == op]
    if len(selected) != 1:
        raise RecipeUnavailable('operator metadata is ambiguous')
    return variants, dynamics[0], _ops_ini(*selected[0])


def compile_kernel_recipe(root: Path, plan: dict, environment: dict, *, read_recipe) -> bool:
    """Return False only before OPC starts; compiler failures never replay."""
    try:
        variants, dynamic, ini = _recipe_inputs(root, plan, environment)
    except (RecipeUnavailable, FileNotFoundError) as exc:
        print('native-incremental: recipe fallback: ' + str(exc), flush=True)
        return False
    stage = Path(tempfile.mkdtemp(prefix='kernel-recipe-', dir=root / '.mindie-runtime'))
    started = False
    completed = False
    try:
        binary = stage / 'binary' / plan['unit']
        generated = binary / 'gen'
        generated.mkdir(parents=True)
        info = stage / 'operator.ini'
        info.write_text(ini, encoding='utf-8')
        tools = root / 'vllm-ascend/csrc/cmake/scripts/util'
        result = subprocess.run([sys.executable, str(tools / 'ascendc_bin_param_build.py'),
                                 str(info), str(generated), plan['unit']], env=environment, timeout=60)
        if result.returncode:
            if result.returncode < 0:
                raise subprocess.CalledProcessError(result.returncode, result.args)
            raise RecipeUnavailable('upstream parameter generator rejected installed metadata')
        scripts = sorted(generated.glob('*.sh'))
        prior = json.loads(_recipe_child(root, plan['kernel_root'] + '/config/' + plan['unit'] + '/' + plan['operator'] + '.json').read_text())
        generated_recipes = [read_recipe(script) for script in scripts]
        if (not scripts or any(row is None for row in generated_recipes)
                or sorted(generated_recipes, key=lambda row: row['name']) != sorted(variants, key=lambda row: row.get('name', ''))):
            raise RecipeUnavailable('generated options or parameters differ from the effective build recipe')
        if len(scripts) != len(prior.get('binList', [])):
            raise RecipeUnavailable('generated variant count differs from installed dispatch')
        # A fresh private compiler view prevents any installed old CPP from
        # preceding BUILD_KERNEL_SRC in upstream's relative-path resolver.
        dynamic_path = stage / 'impl/dynamic' / (plan['operator'] + '.py')
        dynamic_path.parent.mkdir(parents=True)
        prefix = dynamic.rsplit('/dynamic/', 1)[0]
        for name in plan['recipe_files']:
            if name.startswith(prefix + '/ascendc/'):
                target = _recipe_child(stage / 'impl', name[len(prefix) + 1:])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(_recipe_child(root, name), target)
        source = _recipe_child(root, 'vllm-ascend/' + plan['source'])
        source_dir = dynamic_path.parent / 'op_kernel'
        if any(path.is_symlink() for path in source.parent.rglob('*')):
            raise RecipeUnavailable('source include tree contains symlinks')
        shutil.copytree(source.parent, source_dir)
        staged_source = source_dir / source.name
        dynamic_path.write_text(_guarded_dynamic(_recipe_child(root, dynamic).read_text(encoding='utf-8'),
            staged_source, plan['source_sha256'], root), encoding='utf-8')
        output = binary / 'bin' / plan['operator']
        output.mkdir(parents=True)
        vendor = plan['kernel_root'].split('/op_impl/ai_core/tbe/', 1)[0]
        environment = {**environment, 'ASCEND_CUSTOM_OPP_PATH': str(_recipe_child(root, vendor)) +
            (':' + environment['ASCEND_CUSTOM_OPP_PATH'] if environment.get('ASCEND_CUSTOM_OPP_PATH') else ''),
            'HI_PYTHON': sys.executable, 'TILINGKEY_PAR_COMPILE': '1', 'BIN_FILENAME_HASHED': '1'}
        print(f'native-incremental: recipe OPC variants={len(scripts)}', flush=True)
        def compile_one(script):
            proof = stage / (script.stem + '.source.json')
            # Upstream scripts check output presence but can mask OPC's exit
            # code. Preserve their setup and flags, and retain the actual exit
            # before any later shell command overwrites it.
            command = script.read_text(encoding='utf-8')
            command, count = re.subn(r'(^[ \t]*res=\$\(opc [^\n]*\)[ \t]*$)',
                lambda match: match[0] + '\nmindie_opc_status=$?\n'
                'if [ "$mindie_opc_status" -ne 0 ]; then printf "%s\\n" "$res"; exit "$mindie_opc_status"; fi',
                command, flags=re.MULTILINE)
            if count != 1:
                raise ValueError('generated OPC command is not uniquely observable')
            script.write_text(command, encoding='utf-8')
            subprocess.run(['bash', str(script), str(dynamic_path), str(output)],
                cwd=generated, env={**environment, 'MINDIE_KERNEL_SOURCE_PROOF': str(proof)}, check=True)
            if json.loads(proof.read_text()) != {'source': str(staged_source.resolve()), 'sha256': plan['source_sha256']}:
                raise ValueError('variant has no successful fixed CPP compiler proof')
        started = True
        # Wait for every owned compiler even when a sibling fails. The outer
        # preparation process remains the process-family cancellation owner.
        requested_jobs = environment.get('MAX_JOBS', '')
        workers = int(requested_jobs) if requested_jobs.isdecimal() and int(requested_jobs) > 0 else (os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=min(len(scripts), workers)) as executor:
            futures = [executor.submit(compile_one, script) for script in scripts]
            errors = []
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    errors.append(exc)
            if errors:
                raise errors[0]
        expected = {row['binary'] + suffix for row in generated_recipes for suffix in ('.o', '.json')}
        if ({path.name for path in output.iterdir()} != expected
                or any(not path.is_file() or path.is_symlink() for path in output.iterdir())):
            raise ValueError('compiler did not produce every complete variant output')
        subprocess.run([sys.executable, str(tools / 'ascendc_ops_config.py'), '-p', str(binary / 'bin'),
                        '-s', plan['unit']], env=environment, check=True)
        destination = _recipe_child(root, 'vllm-ascend/csrc/build/binary/' + plan['unit'])
        destination.mkdir(parents=True)
        (binary / 'bin').rename(destination / 'bin')
        final_gen = destination / 'gen'
        for script in scripts:
            script.write_text(script.read_text(encoding='utf-8').replace(str(generated.resolve()), str(final_gen.resolve())), encoding='utf-8')
        generated.rename(final_gen)
        if [read_recipe(path) for path in sorted(final_gen.glob('*.sh'))] != generated_recipes:
            raise ValueError('relocated compiler evidence changed its effective recipe')
        completed = True
        return True
    except RecipeUnavailable as exc:
        if started:
            raise
        print('native-incremental: recipe fallback: ' + str(exc), flush=True)
        return False
    finally:
        if started and not completed:
            print('native-incremental: retained failed compiler stage: ' + str(stage), flush=True)
        else:
            shutil.rmtree(stage)
