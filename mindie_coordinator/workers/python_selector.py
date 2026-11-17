"""Deterministic remote Python interpreter & build-job discovery (`python3 -I` stdlib only).

[Role]
Selects the remote Python interpreter deterministically by ABI and installed
distributions (`torch` + `torch-npu`) and computes cgroup/RAM-aware `MINDIE_BUILD_JOBS`.

[Invariants]
- Pure Python stdlib; zero `import torch` or NPU device-context initialization.
- Priority order:
  1. Explicit `MINDIE_PYTHON` (or `PYTHON` when `MINDIE_RESPECT_PYTHON_ENV=1`).
  2. Active `VIRTUAL_ENV` / `CONDA_PREFIX` interpreter.
  3. Recorded `execution_view.python` in `$MINDIE_RUNTIME_ROOT/.mindie-runtime/ready-profile.json`.
  4. Installed candidates ranked by `(has_torch_and_torch_npu, has_torch, version_tuple, is_usr_local)`.

[Timeouts & Locks]
- Pure local filesystem metadata inspection (`< 10ms`); zero subprocesses or locks.

[Side-Effect Policy]
- Read-only filesystem checks; prints selected path or integer job count to stdout.
"""
from __future__ import annotations

import glob
import json
import os
import pathlib
import re
import shutil
import sys


def _is_exec(path_str: str | None) -> bool:
    if not path_str:
        return False
    p = pathlib.Path(path_str)
    try:
        return p.is_file() and os.access(p, os.X_OK)
    except OSError:
        return False


def _parse_ver(text: str) -> tuple[int, ...]:
    nums = [int(part) for part in re.findall(r"\d+", text)[:4]]
    return tuple(nums) if nums else (0,)


def _inspect_prefix(binary: pathlib.Path) -> tuple[int, int, tuple[int, ...], int]:
    prefixes = []
    for base in (binary.parent.parent, binary.resolve().parent.parent):
        if base not in prefixes:
            prefixes.append(base)
    has_torch = 0
    has_torch_npu = 0
    best_ver = _parse_ver(binary.parent.parent.name) or _parse_ver(binary.name)
    for prefix in prefixes:
        lib_dir = prefix / "lib"
        if not lib_dir.is_dir():
            continue
        try:
            py_dirs = sorted(lib_dir.glob("python*"))
        except OSError:
            py_dirs = []
        for py_dir in py_dirs:
            ver = _parse_ver(py_dir.name)
            if ver > best_ver:
                best_ver = ver
            for sp_name in ("site-packages", "dist-packages"):
                sp = py_dir / sp_name
                if not sp.is_dir():
                    continue
                try:
                    entries = [e.name.lower() for e in os.scandir(sp) if e.name.endswith((".dist-info", ".egg-info", ".egg-link"))]
                except OSError:
                    continue
                for name in entries:
                    if name.startswith(("torch-", "torch.")):
                        has_torch = 1
                    elif name.startswith(("torch_npu-", "torch_npu.", "torch-npu-", "torch-npu.")):
                        has_torch_npu = 1
    is_usr_local = 1 if str(binary).startswith("/usr/local/python") else 0
    return (1 if (has_torch and has_torch_npu) else 0, has_torch, best_ver, is_usr_local)


def select_python() -> str:
    explicit = os.environ.get("MINDIE_PYTHON") or (
        os.environ.get("PYTHON") if os.environ.get("MINDIE_RESPECT_PYTHON_ENV") == "1" else None
    )
    if _is_exec(explicit):
        return str(explicit)

    for env_key in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        env_root = os.environ.get(env_key)
        if env_root:
            for rel in ("bin/python3", "bin/python"):
                cand = str(pathlib.Path(env_root) / rel)
                if _is_exec(cand):
                    return cand

    runtime_root = os.environ.get("MINDIE_RUNTIME_ROOT")
    if runtime_root:
        profile_path = pathlib.Path(runtime_root) / ".mindie-runtime" / "ready-profile.json"
        try:
            data = json.loads(profile_path.read_text(encoding="utf-8"))
            recorded = (data.get("execution_view") or {}).get("python")
            if _is_exec(recorded):
                return str(recorded)
        except (OSError, ValueError, TypeError, AttributeError):
            pass

    raw_candidates: list[str] = []
    raw_candidates.extend(sorted(glob.glob("/usr/local/python*/bin/python3")))
    for fixed in ("/usr/local/bin/python3", shutil.which("python3"), "/usr/bin/python3", shutil.which("python")):
        if fixed and fixed not in raw_candidates:
            raw_candidates.append(fixed)

    scored: list[tuple[tuple[int, int, tuple[int, ...], int, int], str]] = []
    for idx, cand in enumerate(raw_candidates):
        if not _is_exec(cand):
            continue
        p = pathlib.Path(cand)
        both, torch_only, ver, usr_local = _inspect_prefix(p)
        scored.append(((both, torch_only, ver, usr_local, -idx), cand))

    if not scored:
        return ""
    scored.sort(reverse=True)
    return scored[0][1]


def compute_build_jobs() -> int:
    cpu = 0
    try:
        with open("/sys/fs/cgroup/cpu.max", encoding="utf-8") as fh:
            quota, period = fh.read().split()[:2]
        if quota != "max" and int(period) > 0:
            cpu = max(1, int(quota) // int(period))
    except (OSError, ValueError, AttributeError):
        cpu = 0
    if not cpu:
        try:
            cpu = len(os.sched_getaffinity(0))
        except (OSError, ValueError, AttributeError):
            cpu = max(1, min(os.cpu_count() or 1, 128))
    cpu = max(1, min(int(cpu), 128))
    mem_cap = 128
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    avail_mb = int(line.split()[1]) // 1024
                    mem_cap = max(1, (avail_mb - 2048) // 2048)
                    break
    except (OSError, ValueError, AttributeError):
        pass
    return max(1, min(cpu, mem_cap, 128))


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "python"
    if mode == "build-jobs":
        print(compute_build_jobs())
        return 0
    chosen = select_python()
    if not chosen:
        return 127
    print(chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
