"""Compile qualified installed operator metadata without a CMake build tree.

[Role]
Executes fast direct `opc` compilation (`compile_kernel_recipe`) from verified
`.mindie-runtime/kernel-compile-recipe.json` metadata without invoking a full
CMake configure/build tree.

[Invariants]
- Raises/catches `RecipeUnavailable` and returns `False` ONLY before `opc`
  compilation starts; once `started = True`, any compiler failure propagates
  directly without falling back or replaying.
- Guards the generated dynamic compiler wrapper so `get_kernel_source` and
  `compile_op` prove they received the exact fixed `.cpp` SHA-256.
- A supplied content cache is only an optimization over the already-validated
  recipe: entries are keyed by native tree + CANN + SoC + Python ABI + flags and
  are re-hashed before use. A cache failure never turns a compiler success into
  a failed build, and a corrupt entry is never selected by name.

[Timeouts & Locks]
- Parameter generation subprocess (`ascendc_bin_param_build.py`) has `timeout=60`s.
- Variant `opc` compilations run concurrently via `ThreadPoolExecutor` bounded
  by `MAX_JOBS` or the CPU/RAM recommendation from `_recommend_max_jobs()`.

[Side-Effect Policy]
- Stages intermediate files in a temporary `.mindie-runtime/kernel-recipe-*`
  directory and atomically renames `bin` and `gen` into `vllm-ascend/csrc/build/binary/<unit>` on completion.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import errno
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
KERNEL_CONTENT_SCHEMA = 'mindie-native-kernel-content/1'
NATIVE_CONTENT_KEY_SCHEMA = 'mindie-native-content/1'
KERNEL_CONTENT_MANIFEST = 'manifest.json'
KERNEL_COMPILE_FLAG_KEYS = (
    'CMAKE_BUILD_TYPE', 'SOC_VERSION', 'MINDIE_SOC_VERSION',
    'COMPILE_CUSTOM_KERNELS', 'MINDIE_COMPILE_CUSTOM_KERNELS', 'MINDIE_USE_CLANG15',
    'C_COMPILER', 'CXX_COMPILER', 'CFLAGS', 'CXXFLAGS', 'ASCEND_HOME_PATH',
    'MINDIE_ENVIRONMENT_FINGERPRINT', 'CMAKE_ARGS', 'LDFLAGS', 'CC', 'CXX',
    'VLLM_TARGET_DEVICE')


class RecipeUnavailable(ValueError):
    """Known unsupported metadata, before any compiler has been launched."""


def _recipe_child(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or '..' in relative.parts:
        raise ValueError('unsafe kernel recipe path')
    return root / name


def _recipe_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cgroup_cpu_cores() -> int:
    """CPU budget from the cgroup quota or scheduler affinity."""
    try:
        fields = Path('/sys/fs/cgroup/cpu.max').read_text(encoding='utf-8').split()
        if len(fields) >= 2 and fields[0] != 'max':
            return max(1, int(fields[0]) // max(1, int(fields[1])))
    except TimeoutError:
        raise
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except TimeoutError:
        raise
    except (AttributeError, OSError):
        return 1


def _recommend_max_jobs() -> int:
    """Bound compiler workers by CPU and available RAM, without requiring remote_dev."""
    try:
        from remote_dev.core.host_spec import probe_local_hardware_spec, recommend_worker_count
    except ImportError:
        cpu_cores = _cgroup_cpu_cores()
        try:
            meminfo = Path('/proc/meminfo').read_text(encoding='utf-8')
        except TimeoutError:
            raise
        except OSError:
            return max(1, min(cpu_cores, 16))
        available_mb = None
        for line in meminfo.splitlines():
            if line.startswith('MemAvailable:'):
                fields = line.split()
                if len(fields) >= 2:
                    available_mb = int(fields[1]) // 1024
                break
        if available_mb is None:
            return max(1, min(cpu_cores, 16))
        memory_workers = max(1, (available_mb - 2048) // 2048)
        return max(1, min(cpu_cores, memory_workers, 16))
    return recommend_worker_count(
        probe_local_hardware_spec(), per_worker_memory_mb=2048,
    )


def kernel_content_key(plan: dict, environment: dict, *, cann_version: str, soc_version: str,
                       python_abi: str) -> str:
    """Same native content token set used by `preparation_cache`.

    `plan['native_to']` is the verified Git-native tree digest of the target
    source. The key never contains a task id, execution root, or donor path.
    """
    csrc_tree_sha = str(plan.get('native_to') or plan.get('source_sha256') or '')
    if not csrc_tree_sha:
        raise ValueError('kernel content key requires a fixed native source tree')
    if not all((cann_version, soc_version, python_abi)):
        raise ValueError('kernel content key requires CANN, SoC and Python ABI')
    flags = {key: environment[key] for key in KERNEL_COMPILE_FLAG_KEYS if key in environment}
    tokens = [NATIVE_CONTENT_KEY_SCHEMA, csrc_tree_sha, str(cann_version), str(soc_version),
              str(python_abi), json.dumps(flags, sort_keys=True, separators=(',', ':'))]
    return hashlib.sha256('\x00'.join(tokens).encode()).hexdigest()


def _kernel_cache_files(entry: Path) -> dict[str, str]:
    files = {}
    for name in ('bin', 'gen'):
        base = entry / name
        if not base.is_dir():
            raise ValueError('kernel content entry is missing ' + name)
        for path in sorted(base.rglob('*')):
            if path.is_file():
                files[path.relative_to(entry).as_posix()] = _recipe_digest(path)
    return files


def _kernel_cache_manifest(plan: dict, key: str, recipes: list, files: dict[str, str]) -> dict:
    return {
        'schema_version': KERNEL_CONTENT_SCHEMA,
        'cache_key': key,
        'operator': plan['operator'],
        'unit': plan['unit'],
        'source': plan.get('source'),
        'source_sha256': plan.get('source_sha256'),
        'native_from': plan.get('native_from'),
        'native_to': plan.get('native_to'),
        'generated_recipes': recipes,
        'files': files,
    }


def _drop_kernel_content(cache: Path, key: str) -> None:
    entry = cache / key
    if entry.parent.resolve() != cache.resolve():
        return
    if entry.is_dir() and not entry.is_symlink():
        shutil.rmtree(entry)
    elif entry.exists() or entry.is_symlink():
        entry.unlink()


def restore_kernel_recipe_content(root: Path, plan: dict, cache: Path, key: str, *, read_recipe) -> bool:
    """Map a previously verified kernel entry; False only when no entry exists."""
    entry = cache / key
    if not entry.is_dir():
        return False
    manifest = json.loads((entry / KERNEL_CONTENT_MANIFEST).read_text())
    if (manifest.get('schema_version') != KERNEL_CONTENT_SCHEMA or manifest.get('cache_key') != key
            or manifest.get('operator') != plan['operator'] or manifest.get('unit') != plan['unit']
            or manifest.get('native_to') != plan.get('native_to')):
        raise ValueError('kernel content entry does not describe this build')
    expected = manifest.get('files')
    if not isinstance(expected, dict) or not expected:
        raise ValueError('kernel content entry has no file proof')
    if _kernel_cache_files(entry) != expected:
        raise ValueError('kernel content entry bytes changed')
    generated = sorted((entry / 'gen').glob('*.sh'))
    if [read_recipe(path) for path in generated] != manifest.get('generated_recipes'):
        raise ValueError('kernel content entry recipe changed')
    destination = _recipe_child(root, 'vllm-ascend/csrc/build/binary/' + plan['unit'])
    if destination.exists():
        raise ValueError('kernel content restore requires a fresh build directory')
    destination.mkdir(parents=True)
    for relative in expected:
        target = _recipe_child(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        source = _recipe_child(entry, relative)
        try:
            os.link(source, target)
        except TimeoutError:
            raise
        except OSError as exc:
            fallback = {getattr(errno, name, None) for name in
                        ('EXDEV', 'EPERM', 'EACCES', 'ENOTSUP', 'EOPNOTSUPP', 'EMLINK', 'EINVAL')}
            if exc.errno not in fallback:
                raise
            shutil.copy2(source, target)
    if _kernel_cache_files(destination) != expected:
        raise ValueError('kernel content restore produced different bytes')
    return True


def publish_kernel_recipe_content(root: Path, plan: dict, cache: Path, key: str, recipes: list) -> dict:
    """Atomically publish this compile's validated `bin`/`gen` output."""
    destination = cache / key
    binary = _recipe_child(root, 'vllm-ascend/csrc/build/binary/' + plan['unit'])
    files = _kernel_cache_files(binary)
    candidate = _kernel_cache_manifest(plan, key, recipes, files)

    def check_existing(existing, observed_files):
        if any(existing.get(field) != value for field, value in candidate.items() if field != 'files'):
            raise ValueError('same kernel content key describes a different compile')
        if observed_files != existing.get('files') or observed_files != files:
            raise ValueError('same kernel content key produced different artifacts')

    if destination.exists():
        try:
            existing = json.loads((destination / KERNEL_CONTENT_MANIFEST).read_text())
            observed_files = _kernel_cache_files(destination)
        except TimeoutError:
            raise
        except (OSError, ValueError):
            # The entry is incomplete/corrupt; rebuild it from this compile.
            _drop_kernel_content(cache, key)
        else:
            check_existing(existing, observed_files)
            return {'status': 'present', 'cache_key': key}
    cache.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.kernel-content-', dir=cache))
    try:
        for relative in files:
            target = _recipe_child(temp, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_recipe_child(binary, relative), target)
        if _kernel_cache_files(temp) != files:
            raise ValueError('kernel content publication copied different bytes')
        (temp / KERNEL_CONTENT_MANIFEST).write_text(json.dumps(candidate, sort_keys=True, indent=2) + '\n')
        try:
            os.rename(temp, destination)
        except TimeoutError:
            raise
        except OSError as exc:
            if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY} or not destination.is_dir():
                raise
            check_existing(json.loads((destination / KERNEL_CONTENT_MANIFEST).read_text()),
                           _kernel_cache_files(destination))
            return {'status': 'present', 'cache_key': key}
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return {'status': 'published', 'cache_key': key}


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


def compile_kernel_recipe(root: Path, plan: dict, environment: dict, *, read_recipe,
                          content_cache: Path | None = None,
                          content_metadata: dict | None = None) -> bool:
    """Return False only before OPC starts; compiler failures never replay."""
    try:
        variants, dynamic, ini = _recipe_inputs(root, plan, environment)
    except (RecipeUnavailable, FileNotFoundError) as exc:
        print('native-incremental: recipe fallback: ' + str(exc), flush=True)
        return False
    cache_key = None
    if content_cache is not None:
        metadata = dict(content_metadata or {})
        measured = ('cann_version', 'soc_version', 'python_abi')
        if all(isinstance(metadata.get(name), str) and metadata[name].strip() for name in measured):
            cache_key = kernel_content_key(plan, environment, **metadata)
            try:
                if restore_kernel_recipe_content(root, plan, content_cache, cache_key, read_recipe=read_recipe):
                    print('native-incremental: kernel content cache hit ' + cache_key, flush=True)
                    return True
            except TimeoutError:
                raise
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                # Cache lookup is optional; a miss/corrupt entry never becomes
                # a compiler result, and the actual compiler still runs.
                print('native-incremental: kernel content cache unavailable: ' + str(exc), flush=True)
        else:
            missing = [name for name in measured if not metadata.get(name)]
            print('native-incremental: kernel content cache skipped; measured metadata missing: '
                  + ', '.join(missing), flush=True)
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
                                 str(info), str(generated), plan['unit']], env=environment, timeout=60,
                                stdin=subprocess.DEVNULL)
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
                cwd=generated, env={**environment, 'MINDIE_KERNEL_SOURCE_PROOF': str(proof)}, check=True,
                timeout=600, stdin=subprocess.DEVNULL)
            if json.loads(proof.read_text()) != {'source': str(staged_source.resolve()), 'sha256': plan['source_sha256']}:
                raise ValueError('variant has no successful fixed CPP compiler proof')
        started = True
        # Wait for every owned compiler even when a sibling fails. The outer
        # preparation process remains the process-family cancellation owner.
        requested_jobs = environment.get('MAX_JOBS', '')
        if requested_jobs.isdecimal() and int(requested_jobs) > 0:
            workers = int(requested_jobs)
        else:
            workers = _recommend_max_jobs()
        if environment.get('MINDIE_COMPILE_JOBS_LIMIT'):
            workers = min(workers, int(environment['MINDIE_COMPILE_JOBS_LIMIT']))
        if workers < 1:
            raise ValueError('insufficient memory for one compiler worker')
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
                or any(not path.is_file() for path in output.iterdir())):
            raise ValueError('compiler did not produce every complete variant output')
        subprocess.run([sys.executable, str(tools / 'ascendc_ops_config.py'), '-p', str(binary / 'bin'),
                        '-s', plan['unit']], env=environment, check=True, timeout=120,
                        stdin=subprocess.DEVNULL)
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
        if cache_key is not None:
            try:
                publish_kernel_recipe_content(root, plan, content_cache, cache_key, generated_recipes)
                print('native-incremental: kernel content cache stored ' + cache_key, flush=True)
            except TimeoutError:
                raise
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                # Compilation already succeeded; an optional cache write must
                # not turn a complete owned build into a failure.
                print('native-incremental: kernel content cache store failed: ' + str(exc), flush=True)
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
