"""Tier 0 unit tests for vllm-ascend runtime env, SoC feature compatibility, and NUMA/CPU binding preflight."""
from __future__ import annotations

from mindie_coordinator.env_manager import validate_vllm_ascend_runtime_env
from mindie_coordinator.op_preflight import (
    validate_model_soc_feature_compatibility,
    validate_numa_cpu_binding_plan,
)


def test_validate_vllm_ascend_runtime_env_catches_aclgraph_and_launch_blocking_conflict():
    report = validate_vllm_ascend_runtime_env(
        {"ASCEND_LAUNCH_BLOCKING": "1"},
        enforce_eager=False,
        cudagraph_mode="FULL_DECODE_ONLY",
    )
    assert report["valid"] is False
    assert report["status"] == "invalid"
    assert "aclgraph_launch_blocking_conflict" in report["violation_codes"]
    assert report["recommended_env"]["ASCEND_LAUNCH_BLOCKING"] == "0"

    # When enforce_eager=True and no graph mode is enabled, ASCEND_LAUNCH_BLOCKING=1 is allowed
    eager_report = validate_vllm_ascend_runtime_env(
        {"ASCEND_LAUNCH_BLOCKING": "1"},
        enforce_eager=True,
    )
    assert "aclgraph_launch_blocking_conflict" not in eager_report["violation_codes"]


def test_validate_vllm_ascend_runtime_env_catches_missing_custom_op_lib_path():
    report = validate_vllm_ascend_runtime_env(
        {
            "ASCEND_CUSTOM_OPP_PATH": "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize",
            "LD_LIBRARY_PATH": "/usr/local/Ascend/driver/lib64",
        },
        enforce_eager=True,
    )
    assert report["valid"] is False
    assert "missing_custom_op_library_path" in report["violation_codes"]
    assert "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib" in report["recommended_env"]["LD_LIBRARY_PATH"]

    # Valid when op_api/lib is present in LD_LIBRARY_PATH
    valid_report = validate_vllm_ascend_runtime_env(
        {
            "ASCEND_CUSTOM_OPP_PATH": "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize",
            "LD_LIBRARY_PATH": "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib:/usr/local/Ascend/driver/lib64",
        },
        enforce_eager=True,
    )
    assert valid_report["valid"] is True
    assert "missing_custom_op_library_path" not in valid_report["violation_codes"]


def test_validate_vllm_ascend_runtime_env_catches_multinode_socket_ifname_mismatch_and_loopback():
    mismatch_report = validate_vllm_ascend_runtime_env(
        {
            "HCCL_SOCKET_IFNAME": "enp189s0f0",
            "GLOO_SOCKET_IFNAME": "docker0",
        },
        multi_node=True,
        enforce_eager=True,
    )
    assert mismatch_report["valid"] is False
    assert "multinode_socket_ifname_mismatch" in mismatch_report["violation_codes"]
    assert mismatch_report["recommended_env"]["GLOO_SOCKET_IFNAME"] == "enp189s0f0"

    loopback_report = validate_vllm_ascend_runtime_env(
        {"HCCL_SOCKET_IFNAME": "lo", "GLOO_SOCKET_IFNAME": "lo", "HCCL_IF_IP": "127.0.0.1"},
        multi_node=True,
        enforce_eager=True,
    )
    assert loopback_report["valid"] is False
    assert "multinode_socket_ifname_mismatch" in loopback_report["violation_codes"]


def test_validate_vllm_ascend_runtime_env_catches_allocator_fragmentation_and_sleep_mode_conflict():
    moe_warn = validate_vllm_ascend_runtime_env(
        {},
        is_moe=True,
        dynamic_shapes=True,
        enforce_eager=True,
    )
    assert "npu_allocator_fragmentation_hazard" in moe_warn["violation_codes"]
    assert moe_warn["recommended_env"]["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:True"

    sleep_conflict = validate_vllm_ascend_runtime_env(
        {"PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True"},
        enable_sleep_mode=True,
        enforce_eager=True,
    )
    assert sleep_conflict["valid"] is False
    assert "npu_allocator_fragmentation_hazard" in sleep_conflict["violation_codes"]
    assert sleep_conflict["recommended_env"]["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:False"


def test_validate_model_soc_feature_compatibility_across_310p_910b_910c_950():
    # Ascend310P rejects bf16, mxfp8/mxfp4, MLA/FIA TND, MC2, and static_kernel
    res_310p = validate_model_soc_feature_compatibility(
        "Ascend310P3",
        ["bf16", "mxfp8", "mla", "mc2", "static_kernel"],
    )
    assert res_310p["valid"] is False
    assert res_310p["device_family"] == "Ascend310P"
    assert {
        "soc_310p_bf16_unsupported",
        "soc_310p_microscaling_quant_unsupported",
        "soc_310p_mla_fia_tnd_unsupported",
        "soc_310p_mc2_unsupported",
        "soc_310p_graph_static_kernel_unsupported",
    }.issubset(res_310p["violation_codes"])

    # Ascend910B rejects A5 native mxfp4/simt, A3 dual-die SIO, and MC2 > 24 experts/rank
    res_910b = validate_model_soc_feature_compatibility(
        "Ascend910B4",
        {"quantization": "mxfp4", "topology_policy": "same_npu_dual_die", "moe_comm_type": "mc2"},
        experts_per_rank=32,
    )
    assert res_910b["valid"] is False
    assert res_910b["device_family"] == "Ascend910B"
    assert {
        "soc_910b_a5_native_feature_unsupported",
        "soc_910b_dual_die_sio_unsupported",
        "soc_910b_mc2_expert_limit_exceeded",
    }.issubset(res_910b["violation_codes"])

    # Ascend910_93 (A3 910C) allows dual-die SIO + MLA + W8A8_DYNAMIC + MC2
    res_910c_ok = validate_model_soc_feature_compatibility(
        "Ascend910_93",
        {"dtype": "bf16", "quantization": "w8a8_dynamic", "topology_policy": "same_npu_dual_die", "mla": True, "mc2": True},
        experts_per_rank=16,
    )
    assert res_910c_ok["valid"] is True
    assert res_910c_ok["device_family"] == "Ascend910_93"

    # Ascend950 (A5) allows native mxfp4 and simt, but rejects HCCL_INTRA_ROCE_ENABLE / SFA DCP
    res_950 = validate_model_soc_feature_compatibility(
        "Ascend950",
        ["mxfp4", "simt", "hccl_intra_roce_enable"],
    )
    assert res_950["valid"] is False
    assert res_950["device_family"] == "Ascend950"
    assert res_950["violation_codes"] == ["soc_950_feature_unsupported"]


def test_validate_numa_cpu_binding_plan_verifies_irq_acl_release_and_catches_hazards():
    # Valid 2-rank binding on Ascend910B sharing NUMA 0 (0-23)
    valid_plan = validate_numa_cpu_binding_plan(
        npu_ids=[0, 1],
        npu_affinity_map={0: "0-23", 1: "0-23"},
        allowed_cpus="0-23",
        soc_version="Ascend910B4",
        omp_num_threads=8,
        numa_node_cpus={0: "0-23", 1: "24-47"},
    )
    assert valid_plan["valid"] is True
    assert valid_plan["binding_mode"] == "topo_affinity"
    assert valid_plan["allocations"][0] == {
        "all_cpus": list(range(0, 12)),
        "irq_cpus": [0, 1],
        "main_cpus": list(range(2, 10)),
        "acl_thread_cpu": 10,
        "release_thread_cpu": 11,
    }
    assert valid_plan["allocations"][1]["irq_cpus"] == [12, 13]
    assert valid_plan["allocations"][1]["acl_thread_cpu"] == 22
    assert valid_plan["allocations"][1]["release_thread_cpu"] == 23

    # Invalid plan: cgroup conflict, core overlap, < 5 CPUs on A2, cross-NUMA, and OMP oversubscription
    bad_plan = validate_numa_cpu_binding_plan(
        npu_ids=[0, 1, 2],
        allowed_cpus="0-15",
        soc_version="Ascend910B4",
        rank_cpu_bindings={
            0: "0-7",
            1: "6-13",   # overlaps with rank 0 on cores 6,7 and spans NUMA 0 (0-7) + NUMA 1 (8-15)
            2: "14-16",  # only 3 cores (< 5 min required on A2) and core 16 is outside allowed_cpus 0-15
        },
        omp_num_threads=12,
        numa_node_cpus={0: "0-7", 1: "8-15"},
    )
    assert bad_plan["valid"] is False
    assert {
        "cgroup_cpuset_conflict",
        "multi_rank_cpu_core_overlap",
        "insufficient_cpus_per_npu",
        "cross_numa_cpu_binding",
        "omp_threads_oversubscription",
    }.issubset(bad_plan["violation_codes"])
