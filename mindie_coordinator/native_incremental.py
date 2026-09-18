"""Rebuild one changed AscendC translation unit over verified installed outputs.

The recipe's installed source copy supplies the old Git blob. Replacing only
that blob must reproduce the donor's complete native fingerprint. CMake build
directories are deliberately not reused: upstream custom commands can keep a
stale .done file without depending on their source file.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess


def native_tree_entries(repo: Path, patterns, submodule_fingerprint) -> dict[str, tuple[str, str]]:
    """Use the same token and submodule identities as build_inputs."""
    entries = {}
    output = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-r', '-z', 'HEAD'])
    for entry in filter(None, output.decode('utf-8').split('\0')):
        metadata, path = entry.split('\t', 1)
        mode, kind, oid = metadata.split()
        if kind == 'commit':
            oid = submodule_fingerprint(repo / path, oid)
        if kind == 'commit' or any(fnmatch.fnmatch(path, pattern) for pattern in patterns):
            entries[path] = (mode, oid)
    return entries


def native_tree_digest(entries: dict) -> str:
    tokens = sorted(f'{mode}\0{path}\0{oid}' for path, (mode, oid) in entries.items())
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


def kernel_rebuild_plan(root: Path, bundle: Path, manifest: dict, preparation: dict, entries: dict) -> dict | None:
    """Qualify an existing single .cpp edit without trusting source mtimes."""
    old_native = manifest.get('preparation', {}).get('native', {})
    current_native = preparation.get('native', {})
    if (old_native.get('vllm') != current_native.get('vllm') or
            not old_native.get('vllm-ascend') or
            native_tree_digest(entries) != current_native.get('vllm-ascend')):
        return None
    if old_native['vllm-ascend'] == current_native.get('vllm-ascend'):
        return None
    for name, (mode, oid) in entries.items():
        match = re.fullmatch(r'csrc/(?:moe|gmm|attention|mc2|ffn|posembedding)/([a-z][a-z0-9_]*)/op_kernel/([^/]+\.cpp)', name)
        if not match or mode != '100644':
            continue
        op, filename = match.groups()
        suffix = '/ascendc/' + op + '/' + filename
        candidates = [path for path in manifest['files'] if path.endswith(suffix) and '/op_impl/ai_core/tbe/' in path]
        if len(candidates) != 1:
            continue
        installed = candidates[0]
        source = _child(root, 'vllm-ascend/' + name)
        expected = manifest['files'][installed]['sha256']
        current = source.read_bytes()
        if hashlib.sha256(current).hexdigest() == expected:
            continue
        before = _child(bundle, installed)
        if before.is_symlink() or not before.is_file():
            raise ValueError('unsafe installed kernel source in native bundle')
        original = before.read_bytes()
        if hashlib.sha256(original).hexdigest() != expected:
            raise ValueError('native kernel source changed in bundle')
        algorithm = 'sha1' if len(oid) == 40 else 'sha256' if len(oid) == 64 else None
        if algorithm is None:
            return None
        if hashlib.new(algorithm, b'blob ' + str(len(current)).encode() + b'\0' + current).hexdigest() != oid:
            raise ValueError('fixed kernel source changed before planning')
        blob = hashlib.new(algorithm, b'blob ' + str(len(original)).encode() + b'\0' + original).hexdigest()
        prior = {**entries, name: (mode, blob)}
        if native_tree_digest(prior) != old_native['vllm-ascend']:
            continue
        kernel = installed.split('/op_impl/ai_core/tbe/', 1)[0] + '/op_impl/ai_core/tbe/kernel'
        configs = [path for path in manifest['files'] if path.startswith(kernel + '/config/') and path.endswith('/' + op + '.json')]
        if len(configs) != 1:
            return None
        unit = PurePosixPath(configs[0]).parent.name
        if not re.fullmatch(r'ascend[a-z0-9_]+', unit):
            return None
        plan = {'operator': op, 'source': name, 'installed_source': installed,
                'kernel_root': kernel, 'unit': unit, 'source_sha256': hashlib.sha256(current).hexdigest(),
                'native_from': old_native['vllm-ascend'], 'native_to': current_native['vllm-ascend']}
        recipe_file = '.mindie-runtime/kernel-compile-recipe.json'
        if recipe_file in manifest['files']:
            tbe = kernel.removesuffix('/kernel')
            plan['recipe_files'] = {path: row['sha256'] for path, row in manifest['files'].items()
                if (path == recipe_file or path == configs[0]
                    or path.startswith(kernel + '/' + unit + '/' + op + '/')
                    or path.startswith(tbe + '/config/')
                    or path.startswith(tbe + '/op_tiling/')
                    or (path.startswith(tbe + '/') and ('/ascendc/' in path or path.endswith('/dynamic/' + op + '.py'))))}
        return plan
    return None


def _child(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if parts.is_absolute() or not parts.parts or '..' in parts.parts:
        raise ValueError('unsafe incremental output path')
    path = root / relative
    if any((root / Path(*parts.parts[:i])).is_symlink() for i in range(1, len(parts.parts) + 1)):
        raise ValueError('symlink in incremental output path')
    return path


def _dispatch_identity(rows: list, paths: set[str]) -> list[str]:
    # Generated binary names can change with the new source. Dispatch keys,
    # dtypes, parameter contracts and variant count must remain complete.
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError('invalid incremental dispatch entries')
    return sorted(json.dumps({key: value for key, value in row.items() if key not in paths}, sort_keys=True)
                  for row in rows)


def _validate_operator(plan: dict) -> tuple[str, str]:
    op, unit = plan['operator'], plan['unit']
    if not re.fullmatch(r'[a-z][a-z0-9_]*', op) or not re.fullmatch(r'ascend[a-z0-9_]+', unit):
        raise ValueError('invalid incremental operator or compute unit')
    return op, unit


def merge_kernel_outputs(root: Path, plan: dict) -> list[str]:
    """Replace one operator and its config entries, preserving all host libraries."""
    op, unit = _validate_operator(plan)
    generated = _child(root, f'vllm-ascend/csrc/build/binary/{unit}/bin')
    source_dir = _child(generated, op)
    outputs = list(source_dir.iterdir()) if source_dir.is_dir() else []
    if not any(path.suffix == '.o' for path in outputs) or not any(path.suffix == '.json' for path in outputs):
        raise ValueError('incremental build returned no complete kernel binaries')
    if any(not path.is_file() or path.is_symlink() for path in outputs):
        raise ValueError('incremental kernel output must contain only regular files')
    kernel = _child(root, plan['kernel_root'])
    config = _child(kernel, f'config/{unit}')
    replacements = {}
    affected_types = set()
    for name in ('binary_info_config.json', 'relocatable_kernel_info_config.json'):
        before = json.loads(_child(config, name).read_text())
        after_path = _child(generated, name)
        after = json.loads(after_path.read_text()) if after_path.is_file() else {}
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ValueError('unsupported kernel config shape')
        selected = {key for key, row in before.items() if isinstance(row, dict) and any(
            isinstance(binary, dict) and str(binary.get('binPath', '')).startswith(unit + '/' + op + '/')
            for binary in row.get('binaryList', []))}
        # A kernel-only rebuild cannot add unrelated operators or remove the
        # original operator's dispatch coverage silently.
        if set(after) != selected:
            raise ValueError('incremental configuration changed operator coverage')
        for key, row in after.items():
            if not isinstance(row, dict) or not row.get('binaryList'):
                raise ValueError('incremental config has no kernel dispatch entries')
            if (_dispatch_identity(before[key]['binaryList'], {'binPath', 'jsonPath'}) !=
                    _dispatch_identity(row['binaryList'], {'binPath', 'jsonPath'})):
                raise ValueError('incremental configuration changed variant coverage')
            for binary in row['binaryList']:
                for field, suffix in (('binPath', '.o'), ('jsonPath', '.json')):
                    value = binary.get(field, '')
                    if (not value.startswith(unit + '/' + op + '/') or not value.endswith(suffix)
                            or len(PurePosixPath(value).parts) != 3
                            or not _child(source_dir, PurePosixPath(value).name).is_file()):
                        raise ValueError('incremental config refers to missing or unrelated output')
        replacements[name] = {**before, **after}
        affected_types.update(selected)
    if not affected_types:
        raise ValueError('baseline has no matching operator dispatch')
    per_op = _child(generated, op + '.json')
    if not per_op.is_file() or not json.loads(per_op.read_text()).get('binList'):
        raise ValueError('incremental build returned no per-operator config')
    prior_variants = json.loads(_child(config, op + '.json').read_text())['binList']
    variants = json.loads(per_op.read_text())['binList']
    if _dispatch_identity(prior_variants, {'binInfo'}) != _dispatch_identity(variants, {'binInfo'}):
        raise ValueError('incremental per-operator configuration changed variant coverage')
    for variant in variants:
        value = variant.get('binInfo', {}).get('jsonFilePath', '')
        if (not value.startswith(unit + '/' + op + '/') or not value.endswith('.json')
                or len(PurePosixPath(value).parts) != 3
                or not _child(source_dir, PurePosixPath(value).name).is_file()):
            raise ValueError('incremental per-operator config refers to missing output')
    # All checks precede replacing any installed output. This execution is not
    # bound/running yet; final profile capture is its only success publication.
    destination = _child(kernel, unit + '/' + op)
    shutil.rmtree(destination)
    shutil.copytree(source_dir, destination)
    shutil.copy2(per_op, _child(config, op + '.json'))
    for name, value in replacements.items():
        _child(config, name).write_text(json.dumps(value, sort_keys=True) + '\n')
    shutil.copy2(_child(root, 'vllm-ascend/' + plan['source']), _child(root, plan['installed_source']))
    return [str(path.relative_to(root)) for path in destination.iterdir()]


def build_incremental_kernel(root: Path, *, compile_recipe=None) -> dict:
    plan_path = _child(root, '.mindie-runtime/native-incremental.json')
    plan = json.loads(plan_path.read_text())
    _validate_operator(plan)
    source = _child(root, 'vllm-ascend/' + plan['source'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != plan['source_sha256']:
        raise ValueError('fixed kernel source changed before compilation')
    repository = _child(root, 'vllm-ascend')
    if (repository / 'csrc/build').exists():
        raise ValueError('incremental compilation requires a fresh owned build directory')
    environment = dict(os.environ)
    catlass = repository / 'csrc/third_party/catlass/include'
    if catlass.is_dir():
        environment['CPATH'] = str(catlass) + (':' + environment['CPATH'] if environment.get('CPATH') else '')
    # Only an explicit metadata miss may take the ordinary build. Errors after
    # any compiler has started propagate to the existing owned-process owner.
    if compile_recipe is None or not compile_recipe(root, plan, environment):
        command = ['bash', 'build.sh', '--opkernel', '--ops=' + plan['operator'], '--soc=' + plan['unit']]
        print('native-incremental: ' + ' '.join(command), flush=True)
        subprocess.run(command, cwd=repository / 'csrc', env=environment, check=True)
    changed = merge_kernel_outputs(root, plan)
    result = {**plan, 'status': 'compiled', 'outputs': changed}
    plan_path.write_text(json.dumps(result, sort_keys=True) + '\n')
    print('native-incremental: compiled ' + plan['operator'], flush=True)
    return result
