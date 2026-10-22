"""Tier 0 unit tests for hardware-aware resource normalization, runtime matching, and execution target selection."""
from __future__ import annotations

import pytest

from mindie_coordinator.placement import (
    normalize_resources,
    role_plan,
    runtime_matches,
    select_execution_target,
    select_runtimes,
)


def test_normalize_resources_and_role_plan_preserve_hardware_expectations():
    normalized = normalize_resources({"npu_count": 2, "cpu_cores": 16, "memory_mb": 65536, "hbm_mb": 131072})
    assert normalized == {
        "npu_count": 2,
        "cpu_cores": 16,
        "memory_mb": 65536,
        "hbm_mb": 131072,
    }

    for field in ("cpu_cores", "memory_mb", "hbm_mb"):
        for bad_val in (-1, 1.5, "16", True):
            with pytest.raises(ValueError, match=f"{field} must be a nonnegative integer"):
                normalize_resources({field: bad_val})

    roles = role_plan(
        {
            "roles": [
                {"name": "prefill", "npu_count": 2, "memory_mb": 131072},
                {"name": "decode", "npu_count": 2},
            ]
        },
        normalize_resources({"cpu_cores": 16, "memory_mb": 65536, "hbm_mb": 65536}),
        "python serve.py",
    )
    assert roles[0]["cpu_cores"] == 16
    assert roles[0]["memory_mb"] == 131072
    assert roles[0]["hbm_mb"] == 65536
    assert roles[1]["cpu_cores"] == 16
    assert roles[1]["memory_mb"] == 65536
    assert roles[1]["hbm_mb"] == 65536


def test_runtime_matches_and_select_runtimes_filter_undersized_hosts():
    small_host_row = {
        "runtime_id": "rt-small",
        "state": "ready",
        "user": "alice",
        "host": "10.0.0.10",
        "recipe": "rc",
        "server": {"cpu_cores": 8, "mem_total_mb": 32768, "mem_available_mb": 16384},
        "npus": [{"npu_id": 0, "busy": False, "hbm_total_mb": 32768, "hbm_used_mb": 0}],
    }
    large_host_row = {
        "runtime_id": "rt-large",
        "state": "ready",
        "user": "alice",
        "host": "173.125.1.2",
        "recipe": "rc",
        "server": {"cpu_cores": 192, "mem_total_mb": 1048576, "mem_available_mb": 900000},
        "npus": [
            {"npu_id": 0, "busy": False, "hbm_total_mb": 65536, "hbm_used_mb": 0},
            {"npu_id": 1, "busy": False, "hbm_total_mb": 65536, "hbm_used_mb": 0},
        ],
    }

    role = {"name": "default", "command": "pytest", "npu_count": 2, "cpu_cores": 32, "memory_mb": 65536, "hbm_mb": 65536}
    assert runtime_matches(small_host_row, {"recipe": "rc"}, role) is False
    assert runtime_matches(large_host_row, {"recipe": "rc"}, role) is True

    chosen = select_runtimes(
        [small_host_row, large_host_row],
        user="alice",
        roles=[role],
        environment={"recipe": "rc"},
    )
    assert chosen == {"runtime_ids": ["rt-large"], "reason": None}


def test_select_execution_target_across_local_and_fleet_probe_hosts():
    fleet_hosts = [
        {
            "alias": "a2-busy",
            "host": "173.125.1.2",
            "online": True,
            "total_npus": 8,
            "free_npus": 0,
            "free_phy_ids": [],
            "server": {"cpu_cores": 192, "load_1m": 160.0, "mem_total_mb": 524288, "mem_available_mb": 32000},
            "npus": [{"npu_id": i, "busy": True, "hbm_total_mb": 65536, "hbm_used_mb": 62000} for i in range(8)],
        },
        {
            "alias": "a3-165",
            "host": "192.168.13.165",
            "online": True,
            "total_npus": 8,
            "free_npus": 4,
            "free_phy_ids": [0, 1, 2, 3],
            "server": {"cpu_cores": 192, "load_1m": 10.0, "mem_total_mb": 1048576, "mem_available_mb": 850000},
            "npus": [
                {
                    "npu_id": 0,
                    "busy": False,
                    "dies": [
                        {"phy_id": 0, "busy": False, "hbm_total_mb": 65536, "hbm_used_mb": 0},
                        {"phy_id": 1, "busy": False, "hbm_total_mb": 65536, "hbm_used_mb": 0},
                    ],
                },
                {
                    "npu_id": 1,
                    "busy": False,
                    "dies": [
                        {"phy_id": 2, "busy": False, "hbm_total_mb": 65536, "hbm_used_mb": 0},
                        {"phy_id": 3, "busy": False, "hbm_total_mb": 65536, "hbm_used_mb": 0},
                    ],
                },
            ],
        },
    ]

    # 1. High-spec local workstation runs CPU/RAM compilation locally without SSH overhead
    workstation_local = {
        "os_family": "macos",
        "cpu_cores": 24,
        "usable_cpu_cores": 20.0,
        "mem_total_mb": 131072,
        "mem_available_mb": 96000,
    }
    local_target = select_execution_target(
        {"tier": "heavy", "cpu_cores": 16, "memory_mb": 32768},
        local_spec=workstation_local,
        fleet_hosts=fleet_hosts,
    )
    assert local_target["decision"] == "local"
    assert local_target["target"] == "local"

    # 2. NPU / HBM workload routes to the free A3 fleet host (`a3-165`) and skips busy `a2-busy`
    remote_target = select_execution_target(
        {"tier": "npu", "npu_count": 2, "hbm_mb": 120000, "memory_mb": 65536},
        local_spec=workstation_local,
        fleet_hosts=fleet_hosts,
    )
    assert remote_target["decision"] == "remote"
    assert remote_target["host"] == "192.168.13.165"
    assert remote_target["alias"] == "a3-165"

    # 3. Impossible request returns reject with structured shortfall
    rejected = select_execution_target(
        {"tier": "npu", "npu_count": 16, "hbm_mb": 1048576},
        local_spec=workstation_local,
        fleet_hosts=fleet_hosts,
    )
    assert rejected["decision"] == "reject"
    assert rejected["target"] is None
    assert "npu_count" in rejected["shortfall"]
