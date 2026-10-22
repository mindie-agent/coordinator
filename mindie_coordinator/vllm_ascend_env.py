"""vllm-ascend runtime environment preflight validation.

[Role]
Validate a `vllm-ascend` / CANN runtime environment mapping and launch flags,
returning deterministic conflict violations and recommended env overrides.

[Invariants]
- Pure in-memory: no I/O, subprocesses, network, or filesystem access.
- Each violation preserves its stable `code` (mirrored as `rule`) and severity.
- `valid` is true iff no `error` severity violations exist.
- `recommended_env` is a fresh mapping; caller inputs are never mutated.

[Timeouts & Locks]
- None: zero waits, zero I/O, and no locks are acquired.

[Side-Effect Policy]
- Side-effect free: reads arguments only and never mutates caller-owned state.
"""

from __future__ import annotations

from typing import Any, Mapping


_LOOPBACK_IFNAMES = frozenset({"lo", "lo0", "localhost", "127.0.0.1", "::1"})


def validate_vllm_ascend_runtime_env(
    env: Mapping[str, Any] | None = None,
    *,
    additional_config: Mapping[str, Any] | None = None,
    enforce_eager: bool = False,
    cudagraph_mode: str | None = None,
    enable_custom_ops: bool | None = None,
    custom_op_vendor_dir: str | None = None,
    multi_node: bool = False,
    dynamic_shapes: bool = False,
    is_moe: bool = False,
    enable_sleep_mode: bool = False,
) -> dict[str, Any]:
    """Deterministically validate `vllm-ascend` & CANN runtime environment variables."""
    env_map = {str(k): str(v) for k, v in (env or {}).items() if v is not None}
    add_cfg = dict(additional_config or {})
    violations: list[dict[str, Any]] = []
    recommended_env: dict[str, str] = {}

    def _add(code: str, severity: str, message: str, remediation: str, **extra: Any) -> None:
        violations.append({"code": code, "rule": code, "severity": severity, "message": message, "remediation": remediation, **extra})

    # 1. Graph mode (`ACLGraph` / `torchair` / `xlite` / `npugraph_ex`) vs `ASCEND_LAUNCH_BLOCKING=1`
    torchair_cfg, xlite_cfg, comp_cfg = add_cfg.get("torchair_graph_config") or {}, add_cfg.get("xlite_graph_config") or {}, add_cfg.get("ascend_compilation_config") or {}
    cg_mode = str(cudagraph_mode or env_map.get("VLLM_CUDAGRAPH_MODE") or ("NONE" if enforce_eager else "PIECEWISE")).strip().upper()
    graph_enabled = (
        (not enforce_eager and cg_mode not in {"NONE", "0", "FALSE", ""})
        or bool(isinstance(torchair_cfg, Mapping) and torchair_cfg.get("enabled"))
        or bool(isinstance(xlite_cfg, Mapping) and xlite_cfg.get("enabled"))
        or bool(isinstance(comp_cfg, Mapping) and comp_cfg.get("enable_npugraph_ex"))
    )
    if graph_enabled and env_map.get("ASCEND_LAUNCH_BLOCKING", "").strip().lower() in {"1", "true", "yes"}:
        recommended_env["ASCEND_LAUNCH_BLOCKING"] = "0"
        _add(
            "aclgraph_launch_blocking_conflict", "error",
            "ACLGraph / Torchair / npugraph_ex graph capture is enabled while ASCEND_LAUNCH_BLOCKING=1 is set; vllm-ascend raises `ValueError: ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1`.",
            "Set `ASCEND_LAUNCH_BLOCKING=0` (or unset it) for ACLGraph/npugraph_ex execution, or pass `--enforce-eager` when isolating single-op faults.",
            env_keys=["ASCEND_LAUNCH_BLOCKING"],
        )
    elif graph_enabled and (env_map.get("DUMP_GE_GRAPH", "").strip() in {"1", "2", "3"} or env_map.get("ASCEND_GLOBAL_LOG_LEVEL", "").strip() in {"0", "1"}):
        _add(
            "aclgraph_launch_blocking_conflict", "warning",
            "Verbose GE graph dump (`DUMP_GE_GRAPH`) or debug log level (`ASCEND_GLOBAL_LOG_LEVEL=0/1`) is active during graph mode.",
            "Set `ASCEND_GLOBAL_LOG_LEVEL=3` and unset `DUMP_GE_GRAPH` after graph capture diagnosis to avoid capture timeouts.",
            env_keys=["DUMP_GE_GRAPH", "ASCEND_GLOBAL_LOG_LEVEL"],
        )

    # 2. Custom AscendC op library path (`_cann_ops_custom` / `op_api/lib` in `LD_LIBRARY_PATH`)
    rl_cfg = add_cfg.get("rl_config") or {}
    batch_invariant = env_map.get("VLLM_BATCH_INVARIANT", "").strip().lower() in {"1", "true", "yes"} or bool(isinstance(rl_cfg, Mapping) and rl_cfg.get("enable_batch_invariant"))
    custom_opp, ld_library_path = str(env_map.get("ASCEND_CUSTOM_OPP_PATH") or custom_op_vendor_dir or "").strip(), str(env_map.get("LD_LIBRARY_PATH") or "").strip()
    ld_entries = [p.strip().rstrip("/") for p in ld_library_path.split(":") if p.strip()]
    custom_ops_requested = ((enable_custom_ops is True) or (enable_custom_ops is None and (bool(custom_opp) or env_map.get("COMPILE_CUSTOM_KERNELS", "").strip() in {"1", "true"}))) and not batch_invariant
    if custom_ops_requested:
        vendor_dirs = [p.strip().rstrip("/") for p in custom_opp.split(":") if p.strip()]
        expected_lib_dirs = [f"{v}/op_api/lib" if not v.endswith("/op_api/lib") else v for v in vendor_dirs]
        has_op_api_lib = any(entry.endswith("/op_api/lib") or "_cann_ops_custom" in entry for entry in ld_entries)
        missing_specific_libs = [lib_dir for lib_dir in expected_lib_dirs if lib_dir not in ld_entries]
        if missing_specific_libs or not has_op_api_lib or not env_map.get("ASCEND_CUSTOM_OPP_PATH", "").strip():
            target_vendor = vendor_dirs[0] if vendor_dirs else "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize"
            target_lib = f"{target_vendor}/op_api/lib" if not target_vendor.endswith("/op_api/lib") else target_vendor
            recommended_env["ASCEND_CUSTOM_OPP_PATH"] = custom_opp or target_vendor
            recommended_env["LD_LIBRARY_PATH"] = f"{target_lib}:{ld_library_path}" if ld_library_path else target_lib
            _add(
                "missing_custom_op_library_path", "error",
                "Custom AscendC operators (`_C_ascend` / `_cann_ops_custom`) are enabled, but `ASCEND_CUSTOM_OPP_PATH` or `<vendor_dir>/op_api/lib` (`libcust_opapi.so`) is missing from `LD_LIBRARY_PATH`.",
                f"Export `ASCEND_CUSTOM_OPP_PATH={recommended_env['ASCEND_CUSTOM_OPP_PATH']}` and prepend `{target_lib}` to `LD_LIBRARY_PATH`.",
                env_keys=["ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH"],
            )

    # 3. Multi-node / HCCL / Gloo / TP socket interface consistency & loopback check
    hccl_if, gloo_if, tp_if = env_map.get("HCCL_SOCKET_IFNAME", "").strip(), env_map.get("GLOO_SOCKET_IFNAME", "").strip(), env_map.get("TP_SOCKET_IFNAME", "").strip()
    hccl_ip = (env_map.get("HCCL_IF_IP") or env_map.get("VLLM_HOST_IP") or "").strip()
    if_map = {k: v for k, v in (("HCCL_SOCKET_IFNAME", hccl_if), ("GLOO_SOCKET_IFNAME", gloo_if), ("TP_SOCKET_IFNAME", tp_if)) if v}
    loopback_vars = [k for k, v in (*if_map.items(), ("HCCL_IF_IP", hccl_ip)) if v and (v.lower() in _LOOPBACK_IFNAMES or v.startswith("127."))]
    if loopback_vars:
        _add(
            "multinode_socket_ifname_mismatch", "error",
            f"Loopback interface or address configured in {', '.join(loopback_vars)}.",
            "Bind `HCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, and `HCCL_IF_IP` to the routable host/RoCE NIC (never `lo` or `127.0.0.1`).",
            env_keys=loopback_vars,
        )
    elif len(set(if_map.values())) > 1:
        canonical_if = hccl_if or gloo_if or tp_if
        for key in ("HCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME"):
            recommended_env[key] = canonical_if
        _add(
            "multinode_socket_ifname_mismatch", "error",
            f"Inconsistent collective socket interfaces across {if_map}: HCCL, Gloo, and TP must bind the same NIC.",
            f"Align `HCCL_SOCKET_IFNAME={canonical_if}`, `GLOO_SOCKET_IFNAME={canonical_if}`, and `TP_SOCKET_IFNAME={canonical_if}`.",
            env_keys=list(if_map.keys()),
        )
    elif multi_node and (not hccl_if or not gloo_if):
        if hccl_if or gloo_if:
            recommended_env["HCCL_SOCKET_IFNAME"] = recommended_env["GLOO_SOCKET_IFNAME"] = hccl_if or gloo_if
        _add(
            "multinode_socket_ifname_mismatch", "error",
            "Multi-node execution requires both `HCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` to be explicitly set to the routable NIC.",
            "Export matching `HCCL_SOCKET_IFNAME=<nic>`, `GLOO_SOCKET_IFNAME=<nic>`, and `HCCL_IF_IP=<node_ip>` on every node.",
            env_keys=["HCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"],
        )

    # 4. NPU memory allocator fragmentation (`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`)
    alloc_conf = env_map.get("PYTORCH_NPU_ALLOC_CONF", "").strip()
    alloc_compact = alloc_conf.lower().replace(" ", "")
    has_expandable_true, has_expandable_false = "expandable_segments:true" in alloc_compact, "expandable_segments:false" in alloc_compact
    sleep_active = bool(enable_sleep_mode or (isinstance(rl_cfg, Mapping) and rl_cfg.get("enabled")))
    moe_or_dynamic = bool(dynamic_shapes or is_moe or env_map.get("VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP", "").strip().lower() in {"1", "true"} or add_cfg.get("enable_shared_expert_dp") or add_cfg.get("multistream_overlap_shared_expert"))
    if sleep_active and has_expandable_true:
        recommended_env["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"
        _add(
            "npu_allocator_fragmentation_hazard", "error",
            "`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` conflicts with `vllm-ascend` CaMem sleep mode (`enable_sleep_mode=True`).",
            "Unset `expandable_segments:True` (or set `PYTORCH_NPU_ALLOC_CONF=expandable_segments:False`) when `enable_sleep_mode` / RL sleep mode is enabled.",
            env_keys=["PYTORCH_NPU_ALLOC_CONF"],
        )
    elif moe_or_dynamic and not sleep_active and not has_expandable_true:
        recommended_env["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
        _add(
            "npu_allocator_fragmentation_hazard", "error" if has_expandable_false else "warning",
            "Dynamic-shape or MoE workload is running without `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`, risking NPU HBM virtual/physical block fragmentation.",
            "Export `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` before launching `vllm-ascend` on dynamic-shape or MoE models.",
            env_keys=["PYTORCH_NPU_ALLOC_CONF"],
        )
    elif has_expandable_true and ("max_split_size_mb" in alloc_compact or "garbage_collection_threshold" in alloc_compact):
        recommended_env["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
        _add(
            "npu_allocator_fragmentation_hazard", "warning",
            f"`PYTORCH_NPU_ALLOC_CONF={alloc_conf}` combines `expandable_segments:True` with legacy block-splitting knobs.",
            "Keep only `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` to avoid conflicting allocator behavior.",
            env_keys=["PYTORCH_NPU_ALLOC_CONF"],
        )

    errors = [v for v in violations if v["severity"] == "error"]
    warnings = [v for v in violations if v["severity"] == "warning"]
    return {
        "valid": len(errors) == 0, "status": "invalid" if errors else ("warning" if warnings else "ok"),
        "violations": violations, "violation_codes": [v["code"] for v in violations],
        "errors": errors, "warnings": warnings, "recommended_env": recommended_env,
    }
