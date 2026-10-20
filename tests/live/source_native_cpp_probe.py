"""C++ cache-probe fixtures and source-proof command for the live driver.

[Role] Builds isolated minimal vllm/vllm-ascend-named packages for the native
cache contract and emits the proof command; not an actual framework build.

[Invariants] Real C++ is compiled; no Python object is labeled native. Package
names exercise the production preparation path only and are labeled as probes.

[Side-Effect Policy] Callers choose a private sandbox directory.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sysconfig
from pathlib import Path


MATH_UTILS = '''#ifndef KERNEL_MATH_UTILS_H
#define KERNEL_MATH_UTILS_H
#include <cstdint>

namespace device_utils {

template <typename T, T roundVal>
__aicore__ __force_inline__ T RoundUp(const T &val)
{
    return (val + roundVal - 1) / roundVal * roundVal;
}

};  // namespace device_utils

#endif
'''

BINDING = '''#include <Python.h>
#define __aicore__
#define __force_inline__ inline
#include "kernels/math_utils.h"
#ifndef MINDIE_SOURCE_OFFSET
#define MINDIE_SOURCE_OFFSET __OFFSET__
#endif
#ifndef MINDIE_ACCEPTANCE_NONCE
#define MINDIE_ACCEPTANCE_NONCE 0
#endif

static PyObject* probe(PyObject*, PyObject* args) {
    long value = 7;
    if (!PyArg_ParseTuple(args, "l", &value)) return nullptr;
    long rounded = static_cast<long>(device_utils::RoundUp<long, 8>(value));
    return PyLong_FromLong(rounded + MINDIE_SOURCE_OFFSET + MINDIE_ACCEPTANCE_NONCE);
}
static PyMethodDef methods[] = {{"probe", probe, METH_VARARGS, "real C++ probe"}, {nullptr, nullptr, 0, nullptr}};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "vllm_ascend_C", nullptr, -1, methods};
PyMODINIT_FUNC PyInit_vllm_ascend_C(void) { return PyModule_Create(&module); }
'''

VENDOR = '''#define __aicore__
#define __force_inline__ inline
#include "kernels/math_utils.h"
extern "C" long mindie_vendor_probe(long value) {
    return static_cast<long>(device_utils::RoundUp<long, 8>(value));
}
'''

SETUP = r'''from pathlib import Path
import shutil
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

class BuildExt(build_ext):
    def run(self):
        super().run()
        for ext in self.extensions:
            built = Path(self.get_ext_fullpath(ext.name))
            target = Path.cwd() / Path(*ext.name.split(".")[:-1]) / built.name
            if built.resolve() == target.resolve():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(built, target)

import os
build_info = Path.cwd() / "vllm_ascend" / "_build_info.py"
build_info.write_text(
    "build_info = {\"version\": \"0.0.1\", \"soc\": %r}\n" % os.environ.get("SOC_VERSION", ""),
    encoding="utf-8")
print("-- The CXX compiler identification is " + os.environ.get("CXX", ""), flush=True)
if os.environ.get("SOC_VERSION"):
    print("-- Detected SOC version: " + os.environ["SOC_VERSION"], flush=True)

setup(
    name="vllm-ascend",
    version="0.0.1",
    packages=find_packages(),
    package_data={"vllm_ascend": ["_cann_ops_custom/**/*.json"]},
    include_package_data=True,
    ext_modules=[
        Extension("vllm_ascend.vllm_ascend_C", ["csrc/binding.cpp"]),
        Extension("vllm_ascend._cann_ops_custom.vendors.probe.op_api.lib.libprobe",
                  ["csrc/vendor.cpp"]),
    ],
    cmdclass={"build_ext": BuildExt},
)
'''


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                            encoding="utf-8", check=True, timeout=60)
    return result.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=MindIE Acceptance", "-c", "user.email=acceptance@example.invalid",
         "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def write_cpp_probe_repos(parent: Path, *, offset: int = 1, tag: bool = True) -> dict[str, Path]:
    """Create isolated minimal C++ probe packages for the cache contract.

    These use the real production package names only to exercise the native
    preparation path. They are not actual vllm/vllm-ascend source deployment and
    must never be reported as a framework build.
    """
    parent.mkdir(parents=True, exist_ok=True)
    vllm = parent / "vllm"
    ascend = parent / "vllm-ascend"
    _write(vllm / "pyproject.toml", '[build-system]\nrequires = ["setuptools>=61"]\n'
           'build-backend = "setuptools.build_meta"\n')
    _write(vllm / "setup.py", 'from setuptools import setup\nsetup(name="vllm", version="0.0.1", packages=["vllm"])\n')
    _write(vllm / "vllm/__init__.py", '__version__ = "0.0.1"\n')
    _write(ascend / "pyproject.toml", '[build-system]\nrequires = ["setuptools>=61"]\n'
           'build-backend = "setuptools.build_meta"\n')
    _write(ascend / "setup.py", SETUP)
    _write(ascend / "requirements.txt", "")
    _write(ascend / "CMakeLists.txt",
           "cmake_minimum_required(VERSION 3.20)\n"
           'if(NOT TORCH_VERSION VERSION_EQUAL "2.10.0")\n'
           '  message(FATAL_ERROR "fixture torch compatibility marker")\n'
           "endif()\n")
    _write(ascend / "vllm_ascend/__init__.py", '__version__ = "0.0.1"\n')
    _write(ascend / "vllm_ascend/_cann_ops_custom/vendors/probe/op_impl/ai_core/tbe/kernel/"
           "config/ascend910b/probe.json", '{"operator": "probe", "schema": 1}\n')
    for package in ("vllm_ascend/_cann_ops_custom",
                    "vllm_ascend/_cann_ops_custom/vendors",
                    "vllm_ascend/_cann_ops_custom/vendors/probe",
                    "vllm_ascend/_cann_ops_custom/vendors/probe/op_api",
                    "vllm_ascend/_cann_ops_custom/vendors/probe/op_api/lib"):
        _write(ascend / package / ".gitkeep", "")
    _write(ascend / "csrc/kernels/math_utils.h", MATH_UTILS)
    _write(ascend / "csrc/binding.cpp", BINDING.replace("__OFFSET__", str(offset)))
    _write(ascend / "csrc/vendor.cpp", VENDOR)
    for repo in (vllm, ascend):
        _git(repo, "init")
        _git(repo, "config", "user.name", "MindIE Acceptance")
        _git(repo, "config", "user.email", "acceptance@example.invalid")
        _commit(repo, "initial isolated native probe")
        if tag:
            _git(repo, "tag", "v0.0.1")
    return {"vllm": vllm, "vllm-ascend": ascend}


def set_source_offset(repo: Path, offset: int) -> str:
    _write(repo / "csrc/binding.cpp", BINDING.replace("__OFFSET__", str(offset)))
    return _commit(repo, f"change native source offset to {offset}")


def set_vllm_probe_marker(repo: Path) -> str:
    """Make the live fixture's vllm tree distinct from a prepared donor."""
    _write(repo / "vllm/__init__.py",
           '__version__ = "0.0.1"\n# wave4 live acceptance fixture\n')
    return _commit(repo, "mark live vllm fixture")


def set_scm_tag(repo: Path, tag: str) -> None:
    existing = _git(repo, "tag", "--list", tag)
    if not existing:
        _git(repo, "tag", tag)


def patch_digest(repo: Path) -> str:
    return hashlib.sha256(
        b"vllm-ascend-cpp-probe\0" + (repo / "csrc/binding.cpp").read_bytes()
        + b"\0" + (repo / "csrc/kernels/math_utils.h").read_bytes()
    ).hexdigest()


def source_file_hashes(sources: dict[str, Path]) -> dict[str, str]:
    files = {}
    for name, base in sources.items():
        for relative in ("README.md", "CMakeLists.txt", "csrc"):
            path = base / relative
            if path.is_file():
                files[name + "/" + relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            elif path.is_dir():
                for child in sorted(p for p in path.rglob("*") if p.is_file())[:64]:
                    files[name + "/" + child.relative_to(base).as_posix()] = hashlib.sha256(child.read_bytes()).hexdigest()
    return files


def local_candidate_build_key(repo: Path) -> str:
    """Documented local C++ proof identity; never presented as a production build key."""
    value = {
        "kind": "local-cpp-proof",
        "commit": _git(repo, "rev-parse", "HEAD"),
        "patch_digest": patch_digest(repo),
        "python_abi": sysconfig.get_config_var("SOABI"),
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def proof_command(expected: int) -> str:
    code = f'''
import hashlib, json, os, pathlib, subprocess, time
root = pathlib.Path.cwd()
expected = {int(expected)}
source = root / "vllm-ascend/csrc/binding.cpp"
header = root / "vllm-ascend/csrc/kernels/math_utils.h"
patch = hashlib.sha256(
    b"vllm-ascend-cpp-probe\\0" + source.read_bytes() + b"\\0" + header.read_bytes()
).hexdigest()
start = time.time()
import vllm_ascend.vllm_ascend_C as extension
loaded = pathlib.Path(extension.__file__).resolve()
value = extension.probe(41)
call_ts = time.time()
if value != expected:
    raise SystemExit("compiled probe returned %r, expected %r" % (value, expected))
profile_path = root / ".mindie-runtime/ready-profile.json"
profile = json.loads(profile_path.read_text()) if profile_path.is_file() else {{}}
candidate_build_key = profile.get("build_key") or os.environ.get("MINDIE_ACCEPTANCE_BUILD_KEY", "")
if not candidate_build_key:
    raise SystemExit("no actual build key or documented local proof key")
link = {{}}
cache_key = os.environ.get("MINDIE_ACCEPTANCE_CACHE_KEY", "")
relative = os.environ.get("MINDIE_ACCEPTANCE_CACHE_RELATIVE", "")
if cache_key and relative:
    cache_file = pathlib.Path("/tmp/mindie-native-cache") / cache_key / relative
    link = {{"path": str(cache_file), "exists": cache_file.is_file()}}
    if cache_file.is_file():
        link.update(same_inode=os.stat(cache_file).st_ino == os.stat(loaded).st_ino,
                    nlink=os.stat(loaded).st_nlink)
proof = {{
    "run_id": os.environ["MINDIE_RUN_ID"],
    "patch_digest": patch,
    "loaded_binary_sha256": hashlib.sha256(loaded.read_bytes()).hexdigest(),
    "loaded_module_path": str(loaded),
    "loaded_module_relative": str(loaded.relative_to(root)),
    "calls": [{{"input_id": os.environ["MINDIE_ACCEPTANCE_INPUT"],
                "call_path": "vllm_ascend.vllm_ascend_C.probe", "timestamp": call_ts}}],
    "measurement_window": {{"start_ts": start, "end_ts": time.time()}},
    "candidate_build_key": candidate_build_key,
    "ready_profile": str(profile_path.resolve()) if profile_path.is_file() else "",
    "evidence_ref": str((root / "acceptance-proof.json").resolve()),
    "source_commit": subprocess.check_output(
        ["git", "-C", "vllm-ascend", "rev-parse", "HEAD"], text=True).strip(),
    "probe_value": value,
    "cache_link": link,
}}
(root / "acceptance-proof.json").write_text(json.dumps(proof, sort_keys=True) + "\\n")
print("MINDIE_ACCEPTANCE_PROOF " + json.dumps(proof, sort_keys=True))
'''
    return '"$MINDIE_PYTHON" - <<\'MINDIE_NATIVE_PROOF\'\n' + code + '\nMINDIE_NATIVE_PROOF\n'
