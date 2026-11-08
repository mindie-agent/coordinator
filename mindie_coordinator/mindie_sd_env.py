"""MindIE-SD (mindiesd) multimodal diffusion runtime and USP topology preflight.

[Role]
Probe and validate a ``MindIE-SD`` (``mindiesd``) multimodal diffusion runtime
environment (``mindiesd`` Python package, ``libpta_extend.so`` PTA plugin,
custom AscendC/TIK operator vendor ``vendors/mindie_sd`` / ``custom_ascendc_ops``,
CANN toolkit root, USP Ulysses/Ring/CFG/TP hybrid parallelism topology, and
Atlas A5 ``quant_four_over_six_a5`` SoC requirements).

[Invariants]
- ``probe_mindie_sd_runtime_readiness`` executes a single bounded Python
  subprocess (default ``10.0s``) when ``python_executable`` is probed;
  ``subprocess.TimeoutExpired`` always raises ``TimeoutError`` and
  ``except TimeoutError: raise`` always precedes ``except OSError:``.
  Probe execution failures (missing interpreter, nonzero exit, invalid JSON)
  raise with the interpreter path and stderr; they are never reported as
  "package is not installed". Import errors are recorded as issues, never as
  ``version='installed'``.
- ``validate_mindie_sd_runtime_env`` is pure in-memory and emits stable
  ``MIE-USP-001``, ``MIE-USP-002``, and ``MIE-QNT-A5`` diagnostic codes.

[Timeouts & Locks]
Single bounded subprocess timeout (``timeout_sec=10.0``) in readiness probe;
zero locks or waits in ``validate_mindie_sd_runtime_env``.

[Side-Effect Policy]
Read-only probe of supplied paths/environment; never mutates host or caller state.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CANN_ROOT = "/usr/local/Ascend/ascend-toolkit/latest"
_PROBE_SNIPPET = (
    "import importlib.util, json\n"
    "spec = importlib.util.find_spec('mindiesd')\n"
    "origin = getattr(spec, 'origin', None) if spec else None\n"
    "payload = {'available': False, 'version': None, 'origin': origin}\n"
    "if spec is not None:\n"
    "    try:\n"
    "        import mindiesd as _m\n"
    "        payload['available'] = True\n"
    "        payload['version'] = getattr(_m, '__version__', None)\n"
    "    except Exception as exc:\n"
    "        payload['import_error'] = f'{type(exc).__name__}: {exc}'\n"
    "print(json.dumps(payload))\n"
)


def probe_mindie_sd_runtime_readiness(
    env: Mapping[str, str] | None = None,
    *,
    python_executable: str | None = None,
    timeout_sec: float = 10.0,
) -> dict[str, Any]:
    """Probe ``mindiesd`` package, ``libpta_extend.so``, and custom opp vendor."""
    merged_env = dict(os.environ) if env is None else {str(k): str(v) for k, v in env.items() if v is not None}
    issues: list[str] = []
    mindiesd_available = False
    mindiesd_version: str | None = None
    pkg_dir: Path | None = None

    explicit_root = merged_env.get("MINDIE_SD_ROOT") or merged_env.get("MINDIESD_HOME")
    if explicit_root and Path(explicit_root).is_dir():
        cand = Path(explicit_root)
        pkg_dir = cand / "mindiesd" if (cand / "mindiesd").is_dir() else cand
        mindiesd_available = True
        mindiesd_version = merged_env.get("MINDIE_SD_VERSION") or "source"

    if python_executable is not None or not mindiesd_available:
        exe = python_executable or sys.executable
        proc_env = {**os.environ, **merged_env}
        try:
            completed = subprocess.run(
                [exe, "-c", _PROBE_SNIPPET],
                capture_output=True,
                text=True,
                timeout=float(timeout_sec),
                env=proc_env,
                check=False,
                **({"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"mindiesd readiness probe timed out after {timeout_sec}s") from exc
        except TimeoutError:
            raise
        except OSError as exc:
            raise RuntimeError(
                f"mindiesd readiness probe could not execute {exe!r}: {type(exc).__name__}: {exc}"
            ) from exc
        if completed.returncode != 0:
            stderr = (completed.stderr or "")[-500:]
            raise RuntimeError(
                f"mindiesd readiness probe failed: interpreter={exe!r} "
                f"exit={completed.returncode} stderr={stderr!r}"
            )
        stdout = (completed.stdout or "").strip()
        if not stdout:
            raise RuntimeError(f"mindiesd readiness probe returned empty stdout from {exe!r}")
        try:
            payload = json.loads(stdout.splitlines()[-1])
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"mindiesd readiness probe returned invalid JSON from {exe!r}: {exc}"
            ) from exc
        if payload.get("import_error"):
            issues.append(f"mindiesd import failed via {exe!r}: {payload['import_error']}")
        elif payload.get("available"):
            mindiesd_available = True
            version = payload.get("version")
            mindiesd_version = None if version in (None, "") else str(version)
            if payload.get("origin"):
                pkg_dir = Path(str(payload["origin"])).resolve().parent

    cann_path = (merged_env.get("ASCEND_HOME_PATH") or merged_env.get("ASCEND_TOOLKIT_HOME") or DEFAULT_CANN_ROOT).strip()

    # Check libPTAExtensionOPS.so / libpta_extend.so under package plugin dir or LD_LIBRARY_PATH
    pta_plugin_path: str | None = None
    plugin_names = ("libPTAExtensionOPS.so", "libpta_extend.so")
    pta_candidates: list[Path] = []
    if pkg_dir is not None:
        for name in plugin_names:
            pta_candidates.extend((pkg_dir / "plugin" / name, pkg_dir / "lib" / name))
    for entry in (merged_env.get("LD_LIBRARY_PATH") or "").split(":"):
        if entry.strip():
            for name in plugin_names:
                pta_candidates.append(Path(entry.strip()) / name)
    for cand in pta_candidates:
        try:
            if cand.is_file():
                pta_plugin_path = str(cand)
                break
        except TimeoutError:
            raise
        except OSError:
            continue
    pta_plugin_available = pta_plugin_path is not None

    # Check custom AscendC/TIK opp vendor path
    custom_opp_path: str | None = None
    vendor_names = ("mindie_sd", "customize", "aie_ascendc", "customize_transformer", "custom_ascendc_ops")
    opp_candidates = [Path(raw.strip()) for raw in (merged_env.get("ASCEND_CUSTOM_OPP_PATH") or "").split(":") if raw.strip()]
    if pkg_dir is not None:
        for vname in vendor_names:
            opp_candidates.extend((pkg_dir / "ops" / "vendors" / vname, pkg_dir / "vendors" / vname))
    if cann_path:
        for vname in vendor_names:
            opp_candidates.append(Path(cann_path) / "opp" / "vendors" / vname)
    for cand in opp_candidates:
        try:
            if cand.is_dir():
                custom_opp_path = str(cand)
                break
        except TimeoutError:
            raise
        except OSError:
            continue
    custom_opp_available = custom_opp_path is not None

    if not mindiesd_available and not any(item.startswith("mindiesd import failed") for item in issues):
        issues.append("mindiesd Python package is not installed or importable")
    if not pta_plugin_available and not custom_opp_available:
        issues.append(
            "neither mindiesd/plugin/{libPTAExtensionOPS.so,libpta_extend.so} nor custom AscendC operator vendor "
            "(ASCEND_CUSTOM_OPP_PATH / vendors/{mindie_sd,customize,aie_ascendc,customize_transformer}) was found"
        )

    ready = bool(mindiesd_available and (pta_plugin_available or custom_opp_available))
    return {
        "framework": "mindie-sd",
        "ready": ready,
        "mindiesd_available": mindiesd_available,
        "mindiesd_version": mindiesd_version,
        "pta_plugin_available": pta_plugin_available,
        "pta_plugin_path": pta_plugin_path,
        "custom_opp_available": custom_opp_available,
        "custom_opp_path": custom_opp_path,
        "cann_path": cann_path,
        "issues": issues,
    }


def validate_mindie_sd_runtime_env(
    env: Mapping[str, str] | None = None,
    *,
    soc_version: str = "",
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    cfg_degree: int = 1,
    tp_degree: int = 1,
    world_size: int | None = None,
    num_q_heads: int | None = None,
    num_kv_heads: int | None = None,
    use_quant_four_over_six_a5: bool = False,
) -> dict[str, Any]:
    """Validate MindIE-SD USP hybrid parallel degrees, head divisibility, and A5 MXFP4 constraints."""
    env_map = {str(k): str(v) for k, v in (env or {}).items() if v is not None}
    issues: list[str] = []
    warnings: list[str] = []
    issue_codes: list[str] = []

    for name, deg in (("ulysses_degree", ulysses_degree), ("ring_degree", ring_degree), ("cfg_degree", cfg_degree), ("tp_degree", tp_degree)):
        if int(deg) < 1:
            issues.append(f"MIE-USP-002: {name} must be >= 1 (got {deg})")
            issue_codes.append("MIE-USP-002")

    expected_world = int(ulysses_degree) * int(ring_degree) * int(cfg_degree) * int(tp_degree)
    if world_size is not None and int(world_size) != expected_world:
        issues.append(
            f"MIE-USP-002: ulysses_degree({ulysses_degree}) * ring_degree({ring_degree}) * "
            f"cfg_degree({cfg_degree}) * tp_degree({tp_degree}) = {expected_world} != world_size({world_size})"
        )
        issue_codes.append("MIE-USP-002")

    if num_q_heads is not None and int(ulysses_degree) >= 1 and int(num_q_heads) % int(ulysses_degree) != 0:
        issues.append(f"MIE-USP-001: num_q_heads ({num_q_heads}) must be divisible by ulysses_degree ({ulysses_degree})")
        issue_codes.append("MIE-USP-001")

    if num_kv_heads is not None and int(ulysses_degree) >= 1 and int(num_kv_heads) >= 1:
        if int(num_kv_heads) % int(ulysses_degree) != 0 and int(ulysses_degree) % int(num_kv_heads) != 0:
            issues.append(
                f"MIE-USP-001: num_kv_heads ({num_kv_heads}) and ulysses_degree ({ulysses_degree}) "
                "must divide each other cleanly for GQA Ulysses all-to-all"
            )
            issue_codes.append("MIE-USP-001")

    soc_lower = str(soc_version or "").strip().lower()
    if use_quant_four_over_six_a5 and ("ascend950" not in soc_lower and "a5" not in soc_lower):
        issues.append(f"MIE-QNT-A5: quant_four_over_six_a5 requires Atlas A5 / Ascend950 SoC (got {soc_version!r})")
        issue_codes.append("MIE-QNT-A5")

    custom_opp_env = env_map.get("ASCEND_CUSTOM_OPP_PATH") or ""
    if not custom_opp_env:
        warnings.append("ASCEND_CUSTOM_OPP_PATH is not set; custom MindIE-SD AscendC operators may not resolve")
    elif "fla/ops/ascend" in custom_opp_env and env_map.get("FLA_NPU_DISABLE_PTH") != "1":
        warnings.append(
            "ASCEND_CUSTOM_OPP_PATH references fla/ops/ascend without FLA_NPU_DISABLE_PTH=1; "
            "fla_npu_opp_env.pth may shadow MindIE-SD custom operator vendors"
        )
    if not env_map.get("LD_LIBRARY_PATH"):
        warnings.append("LD_LIBRARY_PATH is empty; libPTAExtensionOPS.so / libpta_extend.so / libcust_opapi.so may not resolve")

    return {
        "framework": "mindie-sd",
        "valid": len(issues) == 0,
        "issues": issues,
        "issue_codes": issue_codes,
        "warnings": warnings,
        "expected_world_size": expected_world,
    }


__all__ = [
    "DEFAULT_CANN_ROOT",
    "probe_mindie_sd_runtime_readiness",
    "validate_mindie_sd_runtime_env",
]
