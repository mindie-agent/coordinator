"""Tier 0 tests for the pure HCCL/RoCE/HCCS preflight module."""

from __future__ import annotations

import copy

from mindie_coordinator.hccl_preflight import (
    audit_hccl_environment,
    parse_hccn_link_and_optical,
    validate_hccl_ranktable,
)


def _two_node_table() -> dict:
    servers = []
    rank = 0
    for node in range(2):
        devices = []
        for index in range(8):
            devices.append({
                "device_id": str(index),
                "device_ip": f"192.168.{10 + node}.{index + 1}",
                "rank_id": str(rank),
            })
            rank += 1
        servers.append({"server_id": f"node-{node}", "device": devices})
    return {
        "version": "1.0",
        "server_count": "2",
        "server_list": servers,
        "status": "completed",
    }


def _a3_superpod_table() -> dict:
    servers = []
    for node in range(4):
        offset = (node % 2) * 4
        devices = []
        for index in range(4):
            devices.append({
                "device_id": str(index),
                "super_device_id": str(offset + index),
                "device_ip": f"192.168.{20 + node}.{index + 1}",
                "rank_id": str(node * 4 + index),
            })
        servers.append({
            "server_id": f"node-{node}",
            "host_ip": f"172.16.0.{100 + node}",
            "device": devices,
        })
    return {
        "version": "1.2",
        "server_count": "4",
        "server_list": servers,
        "super_pod_list": [
            {"super_pod_id": "0", "server_list": [{"server_id": "node-0"}, {"server_id": "node-1"}]},
            {"super_pod_id": "1", "server_list": [{"server_id": "node-2"}, {"server_id": "node-3"}]},
        ],
        "status": "completed",
    }


def test_parse_healthy_link_optical_stat_and_ip():
    result = parse_hccn_link_and_optical(
        0,
        link_output="link status: UP\n",
        optical_output=(
            "optical present: present\n"
            "Tx power: -1.5 dBm\n"
            "Rx power: -2.0 dBm\n"
            "optical temp: 42 C\n"
        ),
        stat_output="mac_rx_fcs_pkt_num: 0\npcs_err_cnt: 0\n",
        ip_output="ipaddr: 192.168.10.2\nnetmask: 255.255.255.0\n",
    )

    assert result["status"] == "healthy"
    assert result["link_up"] is True
    assert result["optical_present"] is True
    assert result["crc_errors"] == 0
    assert result["tx_power_dbm"] == -1.5
    assert result["rx_power_dbm"] == -2.0
    assert result["optical_temperature_c"] == 42.0
    assert result["ip_address"] == "192.168.10.2"
    assert result["netmask"] == "255.255.255.0"
    assert result["issues"] == []


def test_parse_crc_and_pcs_errors_mark_link_degraded():
    result = parse_hccn_link_and_optical(
        3,
        link_output="link status: UP\n",
        optical_output="present: present\nTx power: 1.0 dBm\nRx power: 0.5 dBm\n",
        stat_output=(
            "mac_rx_fcs_pkt_num: 42\n"
            "pcs_err_cnt: 3\n"
            "mac_rx_symbol_err_pkt_num: 1\n"
        ),
        ip_output="ipaddr: 192.168.10.4\n",
    )

    assert result["status"] == "degraded"
    assert result["link_up"] is True
    assert result["crc_errors"] == 42
    assert result["pcs_errors"] == 3
    assert result["symbol_errors"] == 1
    issues = " ".join(result["issues"])
    assert "CRC/FCS errors: 42" in issues
    assert "PCS errors: 3" in issues


def test_parse_down_link_and_absent_optical_are_down():
    result = parse_hccn_link_and_optical(
        7,
        link_output="link status: DOWN\n",
        optical_output="optical present: absent\n",
        stat_output="mac_rx_fcs_pkt_num: 0\n",
        ip_output="ipaddr: 192.168.10.8\n",
    )

    assert result["status"] == "down"
    assert result["link_up"] is False
    assert result["optical_present"] is False
    issues = " ".join(result["issues"]).lower()
    assert "down" in issues
    assert "optical module absent" in issues


def test_parse_absent_optical_on_up_link_degrades_health():
    result = parse_hccn_link_and_optical(
        1,
        link_output="link status: UP\n",
        optical_output="optical present: absent\n",
    )

    assert result["status"] == "degraded"
    assert result["optical_present"] is False
    assert any("optical module absent" in issue for issue in result["issues"])


def test_audit_multinode_moe_catches_small_buffsize_missing_nic_and_launch_blocking():
    env = {
        "HCCL_BUFFSIZE": "128",
        "HCCL_RDMA_TC": "132",
        "HCCL_RDMA_SL": "4",
        "HCCL_CONNECT_TIMEOUT": "1200",
        "HCCL_EXEC_TIMEOUT": "1200",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "ASCEND_LAUNCH_BLOCKING": "1",
        "MINDIE_MOE": "1",
    }

    result = audit_hccl_environment(env, multi_node=True, pd_disaggregated=True)

    assert result["status"] == "invalid"
    assert any("HCCL_BUFFSIZE=128" in warning for warning in result["warnings"])
    assert any("HCCL_SOCKET_IFNAME" in warning for warning in result["warnings"])
    assert any("GLOO_SOCKET_IFNAME" in warning for warning in result["warnings"])
    assert any("ASCEND_LAUNCH_BLOCKING" in error for error in result["errors"])
    assert result["recommended_env"]["HCCL_BUFFSIZE"] == "1024"
    assert result["recommended_env"]["ASCEND_LAUNCH_BLOCKING"] == "0"


def test_audit_pd_disaggregated_requires_gloo_nic():
    env = {
        "HCCL_SOCKET_IFNAME": "=eth0,enp0",
        "HCCL_BUFFSIZE": "1024",
        "HCCL_EXEC_TIMEOUT": "1200",
        "HCCL_CONNECT_TIMEOUT": "1200",
    }

    result = audit_hccl_environment(env, pd_disaggregated=True)

    assert result["status"] == "warning"
    assert any("GLOO_SOCKET_IFNAME" in warning for warning in result["warnings"])
    assert result["errors"] == []


def test_audit_a3_a5_intra_roce_and_pcie_conflicts():
    base = {
        "HCCL_SOCKET_IFNAME": "eth0",
        "GLOO_SOCKET_IFNAME": "eth0",
        "HCCL_BUFFSIZE": "1024",
        "HCCL_CONNECT_TIMEOUT": "1200",
        "HCCL_EXEC_TIMEOUT": "1200",
    }
    a3 = audit_hccl_environment({**base, "HCCL_INTRA_ROCE_ENABLE": "1"}, multi_node=True, soc="Ascend910_93")
    a5 = audit_hccl_environment({**base, "HCCL_INTRA_ROCE_ENABLE": "1"}, multi_node=True, soc="Ascend910_95")
    conflict = audit_hccl_environment(
        {**base, "HCCL_INTRA_ROCE_ENABLE": "1", "HCCL_INTRA_PCIE_ENABLE": "1"},
        soc="Ascend910B",
    )

    assert a3["status"] == "warning"
    assert any("A3" in warning for warning in a3["warnings"])
    assert a5["status"] == "invalid"
    assert any("A5" in error for error in a5["errors"])
    assert conflict["status"] == "invalid"
    assert any("cannot both be 1" in error for error in conflict["errors"])


def test_audit_rejects_invalid_rdma_tc_sl_and_timeout_order():
    result = audit_hccl_environment(
        {
            "HCCL_RDMA_TC": "130",
            "HCCL_RDMA_SL": "9",
            "HCCL_CONNECT_TIMEOUT": "1200",
            "HCCL_EXEC_TIMEOUT": "120",
        },
        multi_node=True,
    )

    assert result["status"] == "invalid"
    assert any("multiple of 4" in error for error in result["errors"])
    assert any("HCCL_RDMA_SL" in error for error in result["errors"])
    assert any("HCCL_EXEC_TIMEOUT" in error for error in result["errors"])


def test_validate_two_node_sixteen_rank_table_is_valid():
    result = validate_hccl_ranktable(_two_node_table())

    assert result["valid"] is True
    assert result["status"] == "ok"
    assert result["rank_count"] == 16
    assert result["server_count"] == 2
    assert result["errors"] == []


def test_validate_rejects_duplicate_rank_id():
    table = _two_node_table()
    table["server_list"][1]["device"][0]["rank_id"] = "0"

    result = validate_hccl_ranktable(table)

    assert result["valid"] is False
    assert any("rank_id" in error and "unique" in error for error in result["errors"])


def test_validate_rejects_missing_device_ip():
    table = _two_node_table()
    del table["server_list"][0]["device"][2]["device_ip"]

    result = validate_hccl_ranktable(table)

    assert result["valid"] is False
    assert any("device_ip" in error for error in result["errors"])


def test_validate_rejects_non_contiguous_rank_ids():
    table = _two_node_table()
    table["server_list"][1]["device"][-1]["rank_id"] = "99"

    result = validate_hccl_ranktable(table)

    assert result["valid"] is False
    assert any("contiguous" in error for error in result["errors"])


def test_validate_a3_superpod_phy_id_must_be_unique_within_pod():
    table = _a3_superpod_table()
    assert validate_hccl_ranktable(table, soc="Ascend910_93")["valid"] is True

    duplicate = copy.deepcopy(table)
    duplicate["server_list"][1]["device"][1]["super_device_id"] = "0"
    result = validate_hccl_ranktable(duplicate, soc="Ascend910_93")

    assert result["valid"] is False
    assert any("duplicate super_device_id" in error for error in result["errors"])


def test_validate_a3_superpod_requires_every_server_assigned():
    table = _a3_superpod_table()
    table["super_pod_list"][1]["server_list"].pop()

    result = validate_hccl_ranktable(table, soc="Ascend910_93")

    assert result["valid"] is False
    assert any("not assigned to any super pod" in error for error in result["errors"])


def test_audit_moe_single_node_small_buffsize_warns():
    result = audit_hccl_environment({"HCCL_BUFFSIZE": "128", "MINDIE_MOE": "1"})

    assert result["status"] == "warning"
    assert result["moe_inferred"] is True
    assert any("HCCL_BUFFSIZE=128" in warning for warning in result["warnings"])
    assert result["recommended_env"]["HCCL_BUFFSIZE"] == "1024"


def test_audit_missing_buffsize_on_large_all2all_warns():
    result = audit_hccl_environment(
        {"HCCL_SOCKET_IFNAME": "eth0", "GLOO_SOCKET_IFNAME": "eth0"},
        multi_node=True,
        pd_disaggregated=True,
    )

    assert result["status"] == "warning"
    assert any("HCCL_BUFFSIZE is unset" in warning for warning in result["warnings"])


def test_audit_rejects_invalid_whitelist_and_expansion_values():
    result = audit_hccl_environment({
        "HCCL_WHITELIST_DISABLE": "2",
        "HCCL_OP_EXPANSION_MODE": "not-a-mode",
    })

    assert result["status"] == "invalid"
    assert any("HCCL_WHITELIST_DISABLE" in error for error in result["errors"])
    assert any("HCCL_OP_EXPANSION_MODE" in error for error in result["errors"])
