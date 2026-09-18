"""Shared native-input identity used by parity and prepared-runtime verification."""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import subprocess
from pathlib import Path

VLLM_REINSTALL_PATTERNS = (
    'requirements*', 'pyproject.toml', 'setup.py', 'setup.cfg', 'CMakeLists.txt',
    'cmake/**', 'csrc/**', '**/*.cu', '**/*.cuh', '**/*.cpp', '**/*.cc', '**/*.h', '**/*.hpp',
    'vllm/envs.py', 'tools/build/**', 'tools/codegen/**',
    '*.c', '*.cc', '*.cpp', '*.cxx', '*.cu', '*.cuh', '*.h', '*.hpp', '*.hxx', '*.s', '*.S', '*.cmake', '*.proto',
)
VLLM_ASCEND_REINSTALL_PATTERNS = VLLM_REINSTALL_PATTERNS + ('vllm_ascend/_cann_ops_custom/**', 'vllm_ascend/envs.py')
DEPENDENCY_INSTALL_PATTERNS = ('requirements*', 'pyproject.toml', 'setup.py', 'setup.cfg')
BUILD_INPUT_ENV_KEYS = (
    'CMAKE_BUILD_TYPE', 'SOC_VERSION', 'MINDIE_SOC_VERSION',
    'COMPILE_CUSTOM_KERNELS', 'MINDIE_COMPILE_CUSTOM_KERNELS', 'MINDIE_USE_CLANG15',
    'C_COMPILER', 'CXX_COMPILER', 'CFLAGS', 'CXXFLAGS', 'ASCEND_HOME_PATH',
    'MINDIE_ENVIRONMENT_FINGERPRINT',
    'CMAKE_ARGS', 'LDFLAGS', 'CC', 'CXX', 'VLLM_TARGET_DEVICE',
)


def submodule_content(repo: Path, commit: str) -> str:
    """Hash nested source content, not task-specific synthetic commit metadata."""
    top = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', '--show-toplevel'], stdin=subprocess.DEVNULL, text=True, encoding="utf-8").strip()
    if Path(top).resolve() != repo.resolve():
        raise ValueError(f'native submodule is not populated: {repo}')
    tree = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-r', '-z', commit], stdin=subprocess.DEVNULL, text=True, encoding="utf-8")
    tokens = []
    for entry in filter(None, tree.split('\0')):
        metadata, path = entry.split('\t', 1)
        mode, kind, oid = metadata.split()
        if kind == 'commit':
            oid = submodule_content(repo / path, oid)
        tokens.append(f'{mode}\0{path}\0{oid}')
    return hashlib.sha256(json.dumps(sorted(tokens)).encode()).hexdigest()


def build_input_fingerprints(repo: Path, commit: str, patterns: tuple[str, ...], *, build_env=None) -> dict[str, str]:
    native, dependencies, imports = [], [], []
    tree = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-r', '-z', commit], stdin=subprocess.DEVNULL, text=True, encoding="utf-8")
    for entry in filter(None, tree.split('\0')):
        metadata, path = entry.split('\t', 1)
        mode, kind, oid = metadata.split()
        if kind == 'commit':
            oid = submodule_content(repo / path, oid)
        token = f'{mode}\0{path}\0{oid}'
        if kind == 'commit' or any(fnmatch.fnmatch(path, pattern) for pattern in patterns):
            native.append(token)
        # Conservative import closure: retain every tracked resource except
        # native compiler source. Loaded extensions are hashed separately.
        if kind == 'commit' or Path(path).suffix.lower() not in {'.cpp', '.cc', '.cxx', '.c', '.h', '.hpp', '.cuh', '.cu'}:
            imports.append(token)
        if any(fnmatch.fnmatch(path, pattern) for pattern in DEPENDENCY_INSTALL_PATTERNS):
            dependencies.append(token)
    source = os.environ if build_env is None else build_env
    environment = {key: source[key] for key in BUILD_INPUT_ENV_KEYS if key in source}
    return {
        'native': hashlib.sha256(json.dumps(sorted(native)).encode()).hexdigest(),
        'imports': hashlib.sha256(json.dumps(sorted(imports)).encode()).hexdigest(),
        'dependencies': hashlib.sha256(json.dumps(sorted(dependencies)).encode()).hexdigest(),
        'build_env': hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest(),
    }


def runtime_build_inputs(root: Path, profile: dict, fingerprint: str) -> dict:
    environment = {**profile['build_env']}
    environment['MINDIE_ENVIRONMENT_FINGERPRINT'] = fingerprint
    return {name: build_input_fingerprints(root / name, 'HEAD', patterns, build_env=environment)
            for name, patterns in [('vllm', VLLM_REINSTALL_PATTERNS), ('vllm-ascend', VLLM_ASCEND_REINSTALL_PATTERNS)]}
