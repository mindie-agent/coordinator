"""xLLM C++ Ascend runtime and build-cache readiness preflight.

[Role]
Validate and probe an ``xllm`` / ``xllm-service`` runtime and C++ build-cache
environment (``xllm`` binary/package, ``$ASCEND_OPP_PATH/vendors/custom_xllm_math/.xllm_ops_git_head``,
``libcust_opapi.so`` in ``LD_LIBRARY_PATH``, ``ATB_HOME_PATH``, and ``ccache`` /
``vcpkg`` cache readiness), returning deterministic violations and recommended
environment overrides.

[Invariants]
- Pure over supplied environment mappings and optional ``Path`` roots; never
  spawns subprocesses or mutates host/container state.
- Distinguishes ``xllm_ops_vendor_missing`` vs ``xllm_ops_revision_mismatch``
  when ``.xllm_ops_git_head`` is compared against ``expected_xllm_ops_git_head``.
- ``valid`` is True iff no ``error`` severity violations exist.

[Timeouts & Locks]
Zero subprocesses, zero network calls, and no locks.

[Side-Effect Policy]
Read-only inspection of caller-supplied ``Path`` roots when provided; never
writes files or mutates caller dictionaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


DEFAULT_ASCEND_OPP_PATH = "/usr/local/Ascend/ascend-toolkit/latest/opp"
DEFAULT_ATB_HOME_PATH = "/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1"
XLLM_VENDOR_SUBDIR = "vendors/custom_xllm_math"
XLLM_OPS_GIT_HEAD_FILE = ".xllm_ops_git_head"


def probe_xllm_runtime_readiness(
    env: Mapping[str, Any] | None = None,
    *,
    xllm_binary_present: bool | None = None,
    xllm_package_present: bool | None = None,
    xllm_ops_git_head: str | None = None,
    expected_xllm_ops_git_head: str | None = None,
    opp_root: Path | str | None = None,
    atb_root: Path | str | None = None,
    ccache_dir: Path | str | None = None,
    vcpkg_root: Path | str | None = None,
    require_build_cache: bool = False,
) -> dict[str, Any]:
    """Deterministically validate xLLM runtime readiness and C++ build caches."""
    env_map = {str(k): str(v) for k, v in (env or {}).items() if v is not None}
    violations: list[dict[str, Any]] = []
    recommended_env: dict[str, str] = {}

    def _add(code: str, severity: str, message: str, remediation: str, **extra: Any) -> None:
        violations.append({
            "code": code,
            "rule": code,
            "severity": severity,
            "message": message,
            "remediation": remediation,
            **extra,
        })

    # 1. xllm package / binary presence
    bin_ok = xllm_binary_present if xllm_binary_present is not None else bool(env_map.get("XLLM_BIN") or env_map.get("XLLM_HOME"))
    pkg_ok = xllm_package_present if xllm_package_present is not None else bool(env_map.get("XLLM_PYTHON_PKG"))
    runtime_present = bool(bin_ok or pkg_ok)
    if (xllm_binary_present is False and xllm_package_present is False) or (
        xllm_binary_present is None and xllm_package_present is None and not runtime_present
        and "XLLM_REQUIRED" in env_map
    ):
        _add(
            "xllm_runtime_missing",
            "error",
            "Neither the `xllm` C++ binary nor the `xllm` Python package was detected in the target environment.",
            "Install `xllm` (`pip install xllm` or build via `python setup.py bdist_wheel`) or set `XLLM_BIN` to the compiled `build/xllm/core/server/xllm` binary.",
            env_keys=["PATH", "XLLM_BIN"],
        )

    # 2. ASCEND_OPP_PATH & vendors/custom_xllm_math/.xllm_ops_git_head
    ascend_opp = str(opp_root or env_map.get("ASCEND_OPP_PATH") or "").strip()
    vendor_dir = f"{ascend_opp.rstrip('/')}/{XLLM_VENDOR_SUBDIR}" if ascend_opp else f"{DEFAULT_ASCEND_OPP_PATH}/{XLLM_VENDOR_SUBDIR}"
    vendor_op_api_lib = f"{vendor_dir}/op_api/lib"
    observed_head = xllm_ops_git_head.strip() if isinstance(xllm_ops_git_head, str) and xllm_ops_git_head.strip() else None
    vendor_exists = observed_head is not None

    if opp_root is not None:
        opp_path = Path(opp_root)
        head_file = opp_path / XLLM_VENDOR_SUBDIR / XLLM_OPS_GIT_HEAD_FILE
        if head_file.is_file():
            vendor_exists = True
            if observed_head is None:
                try:
                    observed_head = head_file.read_text(encoding="utf-8").strip()
                except TimeoutError:
                    raise
                except OSError:
                    observed_head = None

    if not ascend_opp:
        recommended_env["ASCEND_OPP_PATH"] = DEFAULT_ASCEND_OPP_PATH
        _add(
            "ascend_opp_path_missing",
            "error",
            "`ASCEND_OPP_PATH` is not set; xLLM cannot locate `vendors/custom_xllm_math` custom AscendC kernels.",
            f"Source CANN `set_env.sh` and export `ASCEND_OPP_PATH={DEFAULT_ASCEND_OPP_PATH}`.",
            env_keys=["ASCEND_OPP_PATH"],
        )
    elif not vendor_exists:
        _add(
            "xllm_ops_vendor_missing",
            "error",
            f"Incomplete NPU operator vendor: `{vendor_dir}/{XLLM_OPS_GIT_HEAD_FILE}` is missing.",
            "Build and install `third_party/xllm_ops` into `$ASCEND_OPP_PATH/vendors/custom_xllm_math` and record `.xllm_ops_git_head`.",
            env_keys=["ASCEND_OPP_PATH"],
        )
    elif expected_xllm_ops_git_head and observed_head != expected_xllm_ops_git_head.strip():
        _add(
            "xllm_ops_revision_mismatch",
            "error",
            f"NPU operator revision mismatch: `{XLLM_OPS_GIT_HEAD_FILE}` is {observed_head!r}, expected {expected_xllm_ops_git_head.strip()!r}.",
            "Re-run the `third_party/xllm_ops` custom operator installer so `vendors/custom_xllm_math` matches the active submodule commit.",
            env_keys=["ASCEND_OPP_PATH"],
            observed_revision=observed_head,
            expected_revision=expected_xllm_ops_git_head.strip(),
        )

    # 3. LD_LIBRARY_PATH must include custom_xllm_math/op_api/lib (libcust_opapi.so)
    ld_library_path = str(env_map.get("LD_LIBRARY_PATH") or "").strip()
    ld_entries = [p.strip().rstrip("/") for p in ld_library_path.split(":") if p.strip()]
    has_xllm_op_api_lib = any(
        entry == vendor_op_api_lib.rstrip("/") or entry.endswith("custom_xllm_math/op_api/lib")
        for entry in ld_entries
    )
    if not has_xllm_op_api_lib:
        recommended_env["LD_LIBRARY_PATH"] = (
            f"{vendor_op_api_lib}:{ld_library_path}" if ld_library_path else vendor_op_api_lib
        )
        _add(
            "missing_xllm_custom_op_lib_path",
            "error",
            f"`{vendor_op_api_lib}` (`libcust_opapi.so`) is missing from `LD_LIBRARY_PATH`.",
            f"Prepend `{vendor_op_api_lib}` to `LD_LIBRARY_PATH` before launching `xllm`.",
            env_keys=["LD_LIBRARY_PATH"],
        )

    # 4. ATB_HOME_PATH check
    atb_home = str(atb_root or env_map.get("ATB_HOME_PATH") or "").strip()
    if not atb_home:
        recommended_env["ATB_HOME_PATH"] = DEFAULT_ATB_HOME_PATH
        _add(
            "atb_home_path_missing",
            "error",
            "`ATB_HOME_PATH` is not configured; xLLM requires Ascend Transformer Boost (NNAL ATB) at runtime.",
            f"Source `/usr/local/Ascend/nnal/atb/set_env.sh` or export `ATB_HOME_PATH={DEFAULT_ATB_HOME_PATH}`.",
            env_keys=["ATB_HOME_PATH"],
        )

    # 5. ccache / vcpkg build-cache status
    resolved_ccache = str(ccache_dir or env_map.get("CCACHE_DIR") or "").strip()
    resolved_vcpkg = str(vcpkg_root or env_map.get("VCPKG_ROOT") or "").strip()
    ccache_ready = bool(resolved_ccache) and (ccache_dir is None or Path(ccache_dir).is_dir())
    vcpkg_ready = bool(resolved_vcpkg) and (vcpkg_root is None or Path(vcpkg_root).is_dir())
    if not ccache_ready or not vcpkg_ready:
        severity = "error" if require_build_cache else "warning"
        if not resolved_ccache:
            recommended_env["CCACHE_DIR"] = "/root/.ccache"
        if not resolved_vcpkg:
            recommended_env["VCPKG_ROOT"] = "/opt/vcpkg"
        _add(
            "xllm_build_cache_unconfigured",
            severity,
            "C++ build acceleration (`CCACHE_DIR` and/or `VCPKG_ROOT`) is not fully configured for `xllm`.",
            "Mount persistent `CCACHE_DIR` and `VCPKG_ROOT` directories before running `setup.py` / CMake builds to avoid cold C++ recompilation.",
            env_keys=["CCACHE_DIR", "VCPKG_ROOT"],
        )

    errors = [v for v in violations if v["severity"] == "error"]
    warnings = [v for v in violations if v["severity"] == "warning"]
    return {
        "framework": "xllm",
        "valid": len(errors) == 0,
        "status": "invalid" if errors else ("warning" if warnings else "ok"),
        "runtime_present": runtime_present,
        "xllm_ops_vendor": {
            "vendor_dir": vendor_dir,
            "op_api_lib_dir": vendor_op_api_lib,
            "git_head_present": vendor_exists,
            "observed_git_head": observed_head,
            "expected_git_head": expected_xllm_ops_git_head.strip() if expected_xllm_ops_git_head else None,
        },
        "cache_status": {
            "ccache_ready": ccache_ready,
            "ccache_dir": resolved_ccache or None,
            "vcpkg_ready": vcpkg_ready,
            "vcpkg_root": resolved_vcpkg or None,
        },
        "violations": violations,
        "violation_codes": [v["code"] for v in violations],
        "errors": errors,
        "warnings": warnings,
        "recommended_env": recommended_env,
    }


validate_xllm_runtime_env = probe_xllm_runtime_readiness

__all__ = [
    "DEFAULT_ASCEND_OPP_PATH",
    "DEFAULT_ATB_HOME_PATH",
    "XLLM_OPS_GIT_HEAD_FILE",
    "XLLM_VENDOR_SUBDIR",
    "probe_xllm_runtime_readiness",
    "validate_xllm_runtime_env",
]
