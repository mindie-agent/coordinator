"""Topology-aware multi-Die gang lease tests (Tier 0, Tier 1, Tier 2).

Tier 0 replays real A2/A3 `npu-smi info` Golden Captures in memory.
Tier 1 executes the sandbox hygiene/cleanup command boundary locally.
Tier 2 (`-m live_npu`) probes standing A2/A3 hosts over SSH, computes S1/S2
leases, persists them in isolated `/tmp/mindie-npu-test-<uuid>` sandboxes, and
removes only those sandboxes.
"""

from __future__ import annotations

import json
import shlex
import shutil
import os
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from mindie_coordinator.placement import (
    distinct_hosts_required,
    extract_topology_spec,
    plan_gang_roles,
    role_plan,
    select_runtimes,
)
from mindie_coordinator.topology_gang import (
    GangLeaseError,
    TopologySpec,
    allocate_gang_lease,
    classify_topology_scale,
    cleanup_sandbox_command,
    parse_npu_smi_capture,
    parse_sandbox_hygiene,
    sandbox_hygiene_command,
    validate_sandbox_path,
)

FIXTURES = Path(__file__).parent / "fixtures"
A2_FIXTURE = FIXTURES / "a2-173-125-1-2.txt"
A3_IDLE_FIXTURE = FIXTURES / "a3-192-168-13-165-idle.txt"
# `.156` keeps the standing host's real 25.5.3 schema/process rows and the
# documented partial-busy state: only NPU 2,3 (Phy-ID 4..7) are free.
A3_PARTIAL_FIXTURE = FIXTURES / "a3-192-168-13-156-partial.txt"
A3_BUSY_FIXTURE = FIXTURES / "a3-192-168-13-154-busy.txt"

A2_HOST = "173.125.1.2"
A3_IDLE_HOST = "192.168.13.165"
A3_PARTIAL_HOST = "192.168.13.156"
A3_BUSY_HOST = "192.168.13.154"


def _capture(path: Path, host: str):
    return parse_npu_smi_capture(path.read_text(encoding="utf-8"), host=host)


def _inject_process(raw: str, npu_id: int, chip_id: int, pid: int, name: str) -> str:
    marker = f"| No running processes found in NPU {npu_id}"
    assert marker in raw
    row = (
        f"| {npu_id}       {chip_id}                 | {pid}        | "
        f"{name:<24} | 1024                    | {pid}                  |"
    )
    return raw.replace(marker, row, 1)


def _between(raw: str, begin: str, end: str) -> str:
    start = raw.index(begin) + len(begin)
    stop = raw.index(end, start)
    return raw[start:stop].strip()


@pytest.mark.unit
def test_tier0_golden_a2_and_a3_topology_mapping():
    a2 = _capture(A2_FIXTURE, A2_HOST)
    assert len(a2.cards) == 8
    assert [card.phy_ids for card in a2.cards] == [(phy,) for phy in range(8)]
    assert a2.free_phy_ids == tuple(range(8))
    assert all(card.sibling_pair is False for card in a2.cards)

    a3_idle = _capture(A3_IDLE_FIXTURE, A3_IDLE_HOST)
    assert len(a3_idle.cards) == 8
    assert [card.phy_ids for card in a3_idle.cards] == [
        (2 * index, 2 * index + 1) for index in range(8)
    ]
    assert all(card.sibling_pair for card in a3_idle.cards)
    assert a3_idle.free_phy_ids == tuple(range(16))
    assert a3_idle.free_dual_die_pairs == tuple(
        (2 * index, 2 * index + 1) for index in range(8)
    )

    a3_partial = _capture(A3_PARTIAL_FIXTURE, A3_PARTIAL_HOST)
    assert [card.phy_ids for card in a3_partial.cards] == [
        (2 * index, 2 * index + 1) for index in range(8)
    ]
    assert a3_partial.free_phy_ids == (4, 5, 6, 7)
    assert a3_partial.free_dual_die_pairs == ((4, 5), (6, 7))
    for card in a3_partial.cards:
        if card.npu_id in (2, 3):
            assert card.fully_free is True
            assert not card.dies[0].processes
        else:
            assert card.fully_free is False
            assert all(
                any("VLLMWorker" in reason for reason in die.exclusion_reasons)
                for die in card.dies
            )

    a3_busy = _capture(A3_BUSY_FIXTURE, A3_BUSY_HOST)
    assert a3_busy.free_phy_ids == ()
    assert a3_busy.busy_phy_ids == tuple(range(16))
    assert all(
        any("VLLMWorker_DP" in reason for reason in die.exclusion_reasons)
        for die in a3_busy.dies
    )


@pytest.mark.unit
def test_tier0_production_scale_classification_keeps_16_die_ambiguity_explicit():
    assert classify_topology_scale(1) == "S0"
    assert classify_topology_scale(2) == "S1"
    assert classify_topology_scale(4) == "S1"
    assert classify_topology_scale(8) == "S2"
    assert classify_topology_scale(16, node_count=1) == "S2"
    assert classify_topology_scale(16, node_count=2) == "S3"
    assert classify_topology_scale(32, node_count=2) == "S3"
    with pytest.raises(GangLeaseError):
        classify_topology_scale(6)
    with pytest.raises(GangLeaseError):
        classify_topology_scale(4, node_count=2)
    with pytest.raises(GangLeaseError):
        classify_topology_scale(1, node_count=2)


@pytest.mark.unit
def test_tier0_topology_spec_infers_scale_from_die_and_node_counts():
    s1 = TopologySpec.from_mapping(
        {"die_count": 4, "placement_policy": "intra_node_hccs"}
    )
    assert s1.scale == "S1"

    s2 = TopologySpec.from_mapping(
        {"die_count": 16, "placement_policy": "disaggregated_pd_pair"}
    )
    assert s2.scale == "S2"
    assert s2.prefill_die_count == 8 and s2.decode_die_count == 8

    s3 = TopologySpec.from_mapping(
        {"die_count": 16, "placement_policy": "multi_node_roce"}
    )
    assert s3.scale == "S3" and s3.node_count == 2


@pytest.mark.unit
def test_tier0_s0_s1_allocations_on_golden_captures():
    a2 = _capture(A2_FIXTURE, A2_HOST)
    a3_idle = _capture(A3_IDLE_FIXTURE, A3_IDLE_HOST)
    a3_partial = _capture(A3_PARTIAL_FIXTURE, A3_PARTIAL_HOST)

    s0_a2 = allocate_gang_lease([a2], TopologySpec("S0", 1, "intra_node_hccs"))
    assert s0_a2.visible_devices_by_role == {"default": "0"}
    assert s0_a2.total_die_count == 1

    s0_a3 = allocate_gang_lease([a3_idle], TopologySpec("S0", 1, "intra_node_hccs"))
    assert s0_a3.visible_devices_by_role == {"default": "0"}

    s1_a3 = allocate_gang_lease(
        [a3_idle], TopologySpec("S1", 4, "intra_node_hccs")
    )
    assert s1_a3.visible_devices_by_role == {"default": "0,1,2,3"}
    assert set(s1_a3.members[0].phy_ids) <= set(a3_idle.free_phy_ids)

    s1_partial_pair = allocate_gang_lease(
        [a3_partial], TopologySpec("S1", 4, "same_npu_dual_die")
    )
    assert s1_partial_pair.visible_devices_by_role == {"default": "4,5,6,7"}

    s1_partial_pd = allocate_gang_lease(
        [a3_partial], TopologySpec("S1", 4, "disaggregated_pd_pair")
    )
    assert s1_partial_pd.visible_devices_by_role == {
        "prefill": "4,5",
        "decode": "6,7",
    }
    assert not (set(s1_partial_pd.groups["prefill"]) & set(s1_partial_pd.groups["decode"]))

    s1_a2 = allocate_gang_lease([a2], TopologySpec("S1", 4, "intra_node_hccs"))
    assert s1_a2.visible_devices_by_role == {"default": "0,1,2,3"}


@pytest.mark.unit
def test_tier0_s2_full_node_and_same_node_pd_allocations():
    a2 = _capture(A2_FIXTURE, A2_HOST)
    a3_idle = _capture(A3_IDLE_FIXTURE, A3_IDLE_HOST)

    s2_a2 = allocate_gang_lease([a2], TopologySpec("S2", 8, "intra_node_hccs"))
    assert s2_a2.visible_devices_by_role == {"default": "0,1,2,3,4,5,6,7"}

    s2_a2_pd = allocate_gang_lease(
        [a2], TopologySpec("S2", 8, "disaggregated_pd_pair")
    )
    assert s2_a2_pd.visible_devices_by_role == {
        "prefill": "0,1,2,3",
        "decode": "4,5,6,7",
    }

    s2_a3 = allocate_gang_lease([a3_idle], TopologySpec("S2", 16, "intra_node_hccs"))
    assert s2_a3.visible_devices_by_role == {"default": ",".join(map(str, range(16)))}

    s2_a3_pairs = allocate_gang_lease(
        [a3_idle], TopologySpec("S2", 16, "same_npu_dual_die")
    )
    assert s2_a3_pairs.visible_devices_by_role == {
        "default": ",".join(map(str, range(16)))
    }

    s2_a3_pd = allocate_gang_lease(
        [a3_idle], TopologySpec("S2", 16, "disaggregated_pd_pair")
    )
    assert s2_a3_pd.visible_devices_by_role == {
        "prefill": ",".join(map(str, range(8))),
        "decode": ",".join(map(str, range(8, 16))),
    }


@pytest.mark.unit
def test_tier0_busy_golden_capture_rejects_gang_and_keeps_state_honest():
    a3_busy = _capture(A3_BUSY_FIXTURE, A3_BUSY_HOST)
    for scale, count, policy in (
        ("S1", 4, "intra_node_hccs"),
        ("S2", 16, "same_npu_dual_die"),
    ):
        with pytest.raises(GangLeaseError) as excinfo:
            allocate_gang_lease([a3_busy], TopologySpec(scale, count, policy))
        assert excinfo.value.code in {
            "insufficient_free_dies",
            "insufficient_dual_die_pairs",
        }
        assert a3_busy.free_phy_ids == ()

    a3_idle = _capture(A3_IDLE_FIXTURE, A3_IDLE_HOST)
    with pytest.raises(GangLeaseError) as excinfo:
        allocate_gang_lease(
            [a3_busy, a3_idle], TopologySpec("S3", 16, "multi_node_roce")
        )
    assert excinfo.value.code == "insufficient_nodes"

    with pytest.raises(GangLeaseError) as excinfo:
        allocate_gang_lease(
            [a3_idle],
            TopologySpec("S1", 4, "intra_node_hccs", visible_phy_ids=(0, 1, 2, 99)),
        )
    assert excinfo.value.code == "requested_dies_unavailable"


@pytest.mark.unit
def test_tier0_same_card_pairs_skip_half_busy_card_and_active_user_process():
    raw = A3_IDLE_FIXTURE.read_text(encoding="utf-8")
    occupied = _inject_process(raw, npu_id=0, chip_id=1, pid=424242, name="python3")
    topology = parse_npu_smi_capture(occupied, host=A3_IDLE_HOST)
    assert topology.cards[0].free_phy_ids == (0,)
    assert topology.cards[0].fully_free is False
    assert any(
        "active user process python3" in reason
        for reason in topology.cards[0].dies[1].exclusion_reasons
    )

    lease = allocate_gang_lease(
        [topology], TopologySpec("S1", 2, "same_npu_dual_die")
    )
    assert lease.members[0].phy_ids == (2, 3)
    assert 0 not in lease.members[0].phy_ids
    assert 1 not in lease.members[0].phy_ids


@pytest.mark.unit
def test_tier0_s3_multi_node_roce_keeps_per_host_visible_ids():
    a3_idle = _capture(A3_IDLE_FIXTURE, A3_IDLE_HOST)
    a3_partial = _capture(A3_PARTIAL_FIXTURE, A3_PARTIAL_HOST)

    lease = allocate_gang_lease(
        [a3_idle, a3_partial], TopologySpec("S3", 16, "multi_node_roce")
    )
    assert lease.node_count == 2
    assert lease.total_die_count == 16
    by_host = lease.visible_phy_ids_by_host
    assert by_host[A3_IDLE_HOST] == tuple(range(12))
    assert by_host[A3_PARTIAL_HOST] == (4, 5, 6, 7)
    assert set(by_host[A3_IDLE_HOST]) <= set(a3_idle.free_phy_ids)
    assert set(by_host[A3_PARTIAL_HOST]) <= set(a3_partial.free_phy_ids)
    assert lease.visible_devices_by_role["node-0"] == ",".join(map(str, range(12)))
    assert lease.visible_devices_by_role["node-1"] == "4,5,6,7"
    assert json.loads(lease.to_json())["visible_phy_ids_by_host"][A3_PARTIAL_HOST] == [
        4,
        5,
        6,
        7,
    ]


@pytest.mark.unit
def test_tier0_role_plan_s1_same_npu_dual_die_materializes_only_sibling_pairs():
    a3_idle = _capture(A3_IDLE_FIXTURE, A3_IDLE_HOST)
    a3_partial = _capture(A3_PARTIAL_FIXTURE, A3_PARTIAL_HOST)
    topology = {
        "scale": "S1",
        "die_count": 2,
        "placement_policy": "same_npu_dual_die",
        "visible_phy_ids": [4, 5],
    }

    spec = extract_topology_spec(topology)
    assert spec is not None and spec.scale == "S1"
    with pytest.raises(ValueError):
        extract_topology_spec({**topology, "scale": "S9"})
    assert role_plan(topology, {}, "serve") == [
        {"name": "default", "command": "serve", "devices": [4, 5]}
    ]
    assert role_plan(topology, {}, "serve", host_topologies=[a3_partial]) == [
        {
            "name": "default",
            "command": "serve",
            "devices": [4, 5],
            "host": A3_PARTIAL_HOST,
        }
    ]

    straddling = {**topology, "visible_phy_ids": [1, 2]}
    with pytest.raises(GangLeaseError) as excinfo:
        role_plan(straddling, {}, "serve")
    assert excinfo.value.code == "insufficient_dual_die_pairs"
    with pytest.raises(GangLeaseError) as excinfo:
        role_plan(straddling, {}, "serve", host_topologies=[a3_idle])
    assert excinfo.value.code == "insufficient_dual_die_pairs"
    assert isinstance(excinfo.value, ValueError)


@pytest.mark.unit
def test_tier0_role_plan_s2_pd_pair_splits_disjoint_groups_and_selects_roots():
    topology = {
        "scale": "S2",
        "die_count": 16,
        "placement_policy": "disaggregated_pd_pair",
    }
    structural = role_plan(topology, {}, "serve")
    assert [(role["name"], role["npu_count"]) for role in structural] == [
        ("prefill", 8),
        ("decode", 8),
    ]

    visible = {**topology, "visible_phy_ids": list(range(16))}
    sliced = plan_gang_roles(visible, {}, "serve")
    assert sliced[0]["devices"] == list(range(8))
    assert sliced[1]["devices"] == list(range(8, 16))

    a3_idle = _capture(A3_IDLE_FIXTURE, A3_IDLE_HOST)
    allocated = role_plan(topology, {}, "serve", host_topologies=[a3_idle])
    prefill, decode = allocated
    assert prefill["name"] == "prefill" and decode["name"] == "decode"
    assert prefill["host"] == decode["host"] == A3_IDLE_HOST
    assert prefill["env"] == {"MINDIE_GANG_ROLE": "prefill"}
    assert decode["env"] == {"MINDIE_GANG_ROLE": "decode"}
    assert set(prefill["devices"]).isdisjoint(decode["devices"])
    assert sorted(prefill["devices"] + decode["devices"]) == list(range(16))

    catalog = [
        {"runtime_id": "pd-a", "state": "ready", "user": "alice",
         "host": A3_IDLE_HOST, "profile": {}},
        {"runtime_id": "pd-b", "state": "ready", "user": "alice",
         "host": A3_IDLE_HOST, "profile": {}},
    ]
    selected = select_runtimes(
        catalog, user="alice", roles=allocated, topology=topology
    )
    assert selected == {"runtime_ids": ["pd-a", "pd-b"], "reason": None}


@pytest.mark.unit
def test_tier0_role_plan_s3_multi_node_roce_requires_distinct_hosts():
    topology = {
        "scale": "S3",
        "die_count": 16,
        "placement_policy": "multi_node_roce",
        "node_count": 2,
    }
    structural = role_plan(topology, {}, "serve")
    assert [role["name"] for role in structural] == ["node-0", "node-1"]
    assert [role["npu_count"] for role in structural] == [8, 8]
    assert distinct_hosts_required(structural, topology) is True
    assert distinct_hosts_required(structural, {}) is False

    a3_idle = _capture(A3_IDLE_FIXTURE, A3_IDLE_HOST)
    a3_partial = _capture(A3_PARTIAL_FIXTURE, A3_PARTIAL_HOST)
    allocated = role_plan(
        topology, {}, "serve", host_topologies=[a3_idle, a3_partial]
    )
    assert {role["host"] for role in allocated} == {A3_IDLE_HOST, A3_PARTIAL_HOST}
    assert distinct_hosts_required(allocated, topology) is True
    free_by_host = {
        A3_IDLE_HOST: set(a3_idle.free_phy_ids),
        A3_PARTIAL_HOST: set(a3_partial.free_phy_ids),
    }
    assert sum(len(role["devices"]) for role in allocated) == 16
    for role in allocated:
        assert len(role["devices"]) == len(set(role["devices"]))
        assert set(role["devices"]) <= free_by_host[role["host"]]

    same_host_catalog = [
        {"runtime_id": "only-a", "state": "ready", "user": "alice",
         "host": A3_IDLE_HOST, "profile": {}},
        {"runtime_id": "only-b", "state": "ready", "user": "alice",
         "host": A3_IDLE_HOST, "profile": {}},
    ]
    blocked = select_runtimes(
        same_host_catalog, user="alice", roles=structural, topology=topology
    )
    assert blocked["runtime_ids"] == []
    assert "distinct hosts" in blocked["reason"]

    two_host_catalog = same_host_catalog + [
        {"runtime_id": "other", "state": "ready", "user": "alice",
         "host": A3_PARTIAL_HOST, "profile": {}},
    ]
    selected = select_runtimes(
        two_host_catalog, user="alice", roles=structural, topology=topology
    )
    assert len(selected["runtime_ids"]) == 2
    chosen_hosts = {
        next(row["host"] for row in two_host_catalog if row["runtime_id"] == runtime_id)
        for runtime_id in selected["runtime_ids"]
    }
    assert chosen_hosts == {A3_IDLE_HOST, A3_PARTIAL_HOST}


@pytest.mark.unit
def test_tier0_hygiene_commands_are_scoped_and_shm_is_read_only():
    sandbox = "/tmp/mindie-npu-test-0123456789abcdef"
    probe = sandbox_hygiene_command(sandbox)
    assert sandbox in probe
    assert "/dev/shm/hccl*" in probe
    assert "rm -rf" not in probe
    assert "docker" not in probe
    assert "kill" not in probe

    cleanup = cleanup_sandbox_command(sandbox)
    assert 'rm -rf -- "$sandbox"' in cleanup
    assert f"sandbox={shlex.quote(sandbox)}" in cleanup
    assert "/dev/shm" not in cleanup
    assert "docker" not in cleanup

    for unsafe in (
        "/tmp/other-sandbox",
        "/tmp/mindie-npu-test-",
        "/tmp/mindie-npu-test-../escape",
        "relative/path",
    ):
        with pytest.raises(GangLeaseError):
            validate_sandbox_path(unsafe)

    raw = "\n".join(
        [
            f"SANDBOX|{sandbox}",
            "SANDBOX_STATE|present",
            f"SANDBOX_ENTRY|file|42|{sandbox}/gang-leases.json",
            "SHM_STALE_SECONDS|900",
            "SHM_ENTRY|4096|1000|/dev/shm/hccl_old",
            "SHM_ENTRY|2048|1990|/dev/shm/hccl_active",
            "SHM_ENTRY|0|0|/dev/shm/hccl_unknown",
        ]
    )
    report = parse_sandbox_hygiene(raw, now=2000.0)
    assert report.sandbox_present is True
    assert report.sandbox_entries[0].path.endswith("gang-leases.json")
    assert report.stale_shm_paths == ("/dev/shm/hccl_old",)
    assert report.deletion_authorized is False
    assert report.shm_entries[2].stale_candidate is False


@pytest.mark.integration
@pytest.mark.skipif(os.name == "nt", reason="executes the POSIX remote-host /tmp and bash contract")
def test_tier1_local_sandbox_hygiene_and_cleanup_process_boundary():
    sandbox = f"/tmp/mindie-npu-test-{uuid.uuid4().hex}"
    validate_sandbox_path(sandbox)
    try:
        Path(sandbox).mkdir()
        lease_path = Path(sandbox) / "gang-leases.json"
        lease_path.write_text('{"ok": true}\n', encoding="utf-8")

        hygiene = subprocess.run(
            ["/bin/bash", "-c", sandbox_hygiene_command(sandbox)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        assert hygiene.returncode == 0, hygiene.stderr
        report = parse_sandbox_hygiene(hygiene.stdout)
        assert report.sandbox_present is True
        assert any(entry.path.endswith("gang-leases.json") for entry in report.sandbox_entries)
        assert report.deletion_authorized is False

        cleanup = subprocess.run(
            ["/bin/bash", "-c", cleanup_sandbox_command(sandbox)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        assert cleanup.returncode == 0, cleanup.stderr
        assert "CLEANUP|removed" in cleanup.stdout
        assert Path(sandbox).exists() is False
    finally:
        if Path(sandbox).exists():
            shutil.rmtree(sandbox, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tier 2: standing A2/A3 live probes. Never touches /dev/shm or containers.
# ---------------------------------------------------------------------------

SSH_BASE = [
    "ssh",
    "-o",
    "ControlMaster=no",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=5",
]


def _ssh_script(host: str, script: str, *, timeout: int = 8) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*SSH_BASE, f"root@{host}", "bash -s"],
        input=script,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _probe_script(sandbox: str) -> str:
    hygiene = sandbox_hygiene_command(sandbox)
    return f"""set -u
sandbox={shlex.quote(sandbox)}
mkdir -p -- "$sandbox"
echo "===NPU_SMI_BEGIN==="
if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info 2>/dev/null || true
elif [ -x /usr/local/Ascend/driver/tools/npu-smi ]; then
  /usr/local/Ascend/driver/tools/npu-smi info 2>/dev/null || true
fi
echo "===NPU_SMI_END==="
echo "===HYGIENE_BEGIN==="
{hygiene}
echo "===HYGIENE_END==="
"""


def _write_plan_script(sandbox: str, payload: str) -> str:
    hygiene = sandbox_hygiene_command(sandbox)
    delimiter = "MINDIE_GANG_LEASE_JSON"
    return f"""set -u
sandbox={shlex.quote(sandbox)}
mkdir -p -- "$sandbox"
cat > "$sandbox/gang-leases.json" <<'{delimiter}'
{payload}
{delimiter}
echo "LEASE_BYTES|$(wc -c < "$sandbox/gang-leases.json")"
echo "===LEASE_BEGIN==="
cat "$sandbox/gang-leases.json"
echo "===LEASE_END==="
echo "===HYGIENE_BEGIN==="
{hygiene}
echo "===HYGIENE_END==="
"""


@pytest.mark.live_npu
def test_tier2_live_a2_and_a3_multi_die_gang_lease_in_sandbox():
    hosts = (A2_HOST, A3_IDLE_HOST, A3_PARTIAL_HOST)
    sandboxes = {
        host: f"/tmp/mindie-npu-test-coord-{uuid.uuid4().hex}" for host in hosts
    }
    topologies = {}
    cleanup_failures: list[str] = []
    try:
        for host in hosts:
            sandbox = sandboxes[host]
            result = _ssh_script(host, _probe_script(sandbox), timeout=8)
            if result.returncode != 0:
                pytest.skip(f"SSH probe failed for {host}: {result.stderr[-300:]}")
            smi_raw = _between(
                result.stdout, "===NPU_SMI_BEGIN===", "===NPU_SMI_END==="
            )
            topology = parse_npu_smi_capture(smi_raw, host=host)
            initial_hygiene = parse_sandbox_hygiene(
                _between(result.stdout, "===HYGIENE_BEGIN===", "===HYGIENE_END===")
            )
            assert initial_hygiene.sandbox_present is True, host
            topologies[host] = topology

        a2 = topologies[A2_HOST]
        assert len(a2.cards) == 8
        assert all(len(card.phy_ids) == 1 for card in a2.cards)
        assert len(a2.dies) == 8

        a3_idle = topologies[A3_IDLE_HOST]
        a3_partial = topologies[A3_PARTIAL_HOST]
        for topology in (a3_idle, a3_partial):
            assert len(topology.cards) == 8
            assert all(card.sibling_pair for card in topology.cards)
            assert len(topology.dies) == 16

        s1_host = max(
            (topology for topology in (a3_idle, a3_partial, a2) if len(topology.free_phy_ids) >= 4),
            key=lambda topology: len(topology.free_phy_ids),
            default=None,
        )
        if s1_host is None:
            pytest.skip("no standing host has 4 free Dies for an S1 gang lease")
        s1 = allocate_gang_lease(
            [s1_host], TopologySpec("S1", 4, "intra_node_hccs")
        )
        assert s1.total_die_count == 4
        assert set(s1.members[0].phy_ids) <= set(s1_host.free_phy_ids)

        s2_host = max(
            (topology for topology in (a3_idle, a3_partial, a2) if len(topology.free_phy_ids) >= 8),
            key=lambda topology: len(topology.free_phy_ids),
            default=None,
        )
        if s2_host is None:
            pytest.skip("no standing host has 8 free Dies for an S2 gang lease")
        s2_die_count = 16 if len(s2_host.free_phy_ids) >= 16 else 8
        s2 = allocate_gang_lease(
            [s2_host], TopologySpec("S2", s2_die_count, "intra_node_hccs")
        )
        assert s2.total_die_count == s2_die_count
        assert set(s2.members[0].phy_ids) <= set(s2_host.free_phy_ids)

        plans = {"s1": s1.to_dict(), "s2": s2.to_dict()}
        if s2_die_count == 16:
            s2_pd = allocate_gang_lease(
                [s2_host], TopologySpec("S2", 16, "disaggregated_pd_pair")
            )
            assert s2_pd.visible_devices_by_role["prefill"] == ",".join(
                map(str, range(8))
            )
            assert s2_pd.visible_devices_by_role["decode"] == ",".join(
                map(str, range(8, 16))
            )
            plans["s2_pd"] = s2_pd.to_dict()

        a3_pair_host = next(
            (
                topology
                for topology in (a3_idle, a3_partial)
                if len(topology.free_dual_die_pairs) >= 2
            ),
            None,
        )
        if a3_pair_host is not None:
            s1_pair = allocate_gang_lease(
                [a3_pair_host], TopologySpec("S1", 4, "same_npu_dual_die")
            )
            pair_ids = set(s1_pair.members[0].phy_ids)
            assert len(pair_ids) % 2 == 0
            assert all((phy ^ 1) in pair_ids for phy in pair_ids)
            plans["s1_pair"] = s1_pair.to_dict()

        for host in hosts:
            payload = json.dumps(
                {
                    "host": host,
                    "sandbox": sandboxes[host],
                    "probed_at_unix": int(time.time()),
                    "topology": topologies[host].to_dict(),
                    "plans": plans,
                    "deletion_authorized": False,
                },
                indent=2,
                sort_keys=True,
            )
            result = _ssh_script(host, _write_plan_script(sandboxes[host], payload), timeout=8)
            assert result.returncode == 0, f"{host}: {result.stderr[-300:]}"
            written = json.loads(
                _between(result.stdout, "===LEASE_BEGIN===", "===LEASE_END===")
            )
            assert written["host"] == host
            assert written["deletion_authorized"] is False
            assert written["plans"]["s1"]["die_count"] == 4
            report = parse_sandbox_hygiene(
                _between(result.stdout, "===HYGIENE_BEGIN===", "===HYGIENE_END===")
            )
            assert report.sandbox_present is True
            assert any(
                entry.path.endswith("gang-leases.json")
                for entry in report.sandbox_entries
            ), host
            assert report.deletion_authorized is False
    finally:
        for host in hosts:
            try:
                result = _ssh_script(
                    host, cleanup_sandbox_command(sandboxes[host]), timeout=8
                )
                if result.returncode != 0 or "CLEANUP|removed" not in result.stdout:
                    cleanup_failures.append(
                        f"{host}: rc={result.returncode} out={result.stdout[-200:]!r} "
                        f"err={result.stderr[-200:]!r}"
                    )
            except Exception as exc:  # noqa: BLE001 - report cleanup failure explicitly
                cleanup_failures.append(f"{host}: {exc.__class__.__name__}: {exc}")

    if cleanup_failures:
        pytest.fail("sandbox cleanup failed: " + "; ".join(cleanup_failures))
