"""Tier 0/1 tests for explicit host hardware floors in runtime placement.

Host capacity comes from the production `parse_fleet_probe_output` parser over
checked-in A2/A3 `npu-smi` captures, and prepared-root rows come from a real
`RuntimePool.catalog()`. No hand-built catalog shape stands in for either.
"""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from mindie_coordinator.agent_session import AgentSessions
from mindie_coordinator.fleet_probe import (
    device_occupancy_inventory,
    parse_fleet_probe_output,
    parse_npu_smi_usages,
)
from mindie_coordinator.placement import (
    hardware_shortfall,
    normalize_resources,
    role_plan,
    runtime_matches,
    select_execution_target,
)
from mindie_coordinator.ready_runtime import RuntimePool
from mindie_coordinator.task_client import TaskClient
from remote_dev.core.host_spec import normalize_remote_probe_spec
from test_topology_gang_service import (
    A2,
    A3_PARTIAL,
    HOST,
    GangBackend,
    _runtime_spec,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _host_snapshot(capture: Path, *, cores: int, mem_total_mb: int, mem_avail_mb: int, host: str = HOST) -> dict:
    """One production fleet-probe snapshot: real NPU capture + real SYSTEM section."""
    raw = (
        "===SECTION:SYSTEM===\n"
        f"HOSTNAME:{host}\nCORES:{cores}\nLOADAVG:12.0 10.0 9.0\n"
        f"MEM:{mem_total_mb}:{mem_total_mb - mem_avail_mb}:{mem_avail_mb}\n"
        "===SECTION:NPU_SMI===\n" + capture.read_text(encoding="utf-8")
    )
    return parse_fleet_probe_output(raw, alias=host, host=host)


A3_HOST = dict(capture=A3_PARTIAL, cores=192, mem_total_mb=1031168, mem_avail_mb=900096)
A2_HOST = dict(capture=A2, cores=64, mem_total_mb=196608, mem_avail_mb=150000)
A5_IDLE = FIXTURES / "a5-25rc3-info-idle.txt"
A5_HOST = dict(capture=A5_IDLE, cores=192, mem_total_mb=1031168, mem_avail_mb=900096)
# Measured total used HBM on cards 0..7 of the committed A5 capture:
# 4778/4785/4778/4782/4779/4783/4778/4784 MiB of 98304 MiB each, with every
# card idle. The split between firmware, driver and runtime reservations is not
# established by `npu-smi info` and differs by machine, so these tests only
# require that measured usage is not reported as free.
A5_MEASURED_FREE_HBM_MB = 8 * 98304 - (4778 + 4785 + 4778 + 4782 + 4779 + 4783 + 4778 + 4784)


@pytest.fixture
def pool_catalog_row(tmp_path):
    """A prepared-root row straight out of the production RuntimePool catalog."""
    backend = GangBackend(tmp_path / "host", devices=list(range(16)), busy=set())
    pool = RuntimePool(tmp_path / "manager", backend)
    spec = _runtime_spec(service_ports=[48000])
    backend.mark_prepared(spec)
    pool.register("runtime-a", spec)
    rows = pool.catalog()
    assert len(rows) == 1 and rows[0]["host"] == HOST
    return rows[0]


def test_normalize_resources_and_role_plan_preserve_hardware_expectations():
    normalized = normalize_resources({"npu_count": 2, "cpu_cores": 16, "memory_mb": 65536, "hbm_mb": 131072})
    assert normalized == {"npu_count": 2, "cpu_cores": 16, "memory_mb": 65536, "hbm_mb": 131072}

    for field in ("cpu_cores", "memory_mb", "hbm_mb"):
        for bad_val in (-1, 1.5, "16", True):
            with pytest.raises(ValueError, match=f"{field} must be a nonnegative integer"):
                normalize_resources({field: bad_val})

    roles = role_plan(
        {"roles": [{"name": "prefill", "npu_count": 2, "memory_mb": 131072},
                   {"name": "decode", "npu_count": 2}]},
        normalize_resources({"cpu_cores": 16, "memory_mb": 65536, "hbm_mb": 65536}),
        "python serve.py",
    )
    assert [role["memory_mb"] for role in roles] == [131072, 65536]
    assert [role["cpu_cores"] for role in roles] == [16, 16]
    assert [role["hbm_mb"] for role in roles] == [65536, 65536]


def test_catalog_row_hardware_floor_is_decided_by_the_fleet_snapshot(pool_catalog_row):
    """A prepared-root row carries no capacity; the host snapshot supplies it."""
    a3 = {HOST: normalize_remote_probe_spec(_host_snapshot(**A3_HOST))}
    a2 = {HOST: normalize_remote_probe_spec(_host_snapshot(**A2_HOST))}

    fits_a3 = {"name": "default", "command": "pytest", "cpu_cores": 160, "memory_mb": 786432, "hbm_mb": 196608}
    assert runtime_matches(pool_catalog_row, {}, fits_a3, a3) is True
    # The same prepared root on the smaller A2 host cannot satisfy it.
    assert runtime_matches(pool_catalog_row, {}, fits_a3, a2) is False
    assert hardware_shortfall(pool_catalog_row, fits_a3, a2)["gaps"] == {
        "cpu_cores": {"required": 160, "available": 64},
        "memory_mb": {"required": 786432, "available": 150000},
    }

    # Without an observed snapshot an explicit floor is a gap, not a pass.
    assert runtime_matches(pool_catalog_row, {}, fits_a3, {}) is False
    assert hardware_shortfall(pool_catalog_row, fits_a3, {})["unobserved"] == [
        "cpu_cores", "hbm_mb", "memory_mb"]

    # Legacy unconstrained roles keep matching with no snapshot at all.
    assert runtime_matches(pool_catalog_row, {}, {"name": "default", "command": "pytest"}, None) is True
    assert hardware_shortfall(pool_catalog_row, {"name": "default", "npu_count": 2}, None) is None


def test_aggregate_free_hbm_never_falls_back_to_total_capacity(pool_catalog_row):
    busy_host = {HOST: normalize_remote_probe_spec(
        {"host": HOST, "hbm_total_mb": 16 * 65536, "free_npus": 8,
         "server": {"cpu_cores": 192, "mem_total_mb": 1031168, "mem_available_mb": 900096}})}
    role = {"name": "default", "command": "pytest", "hbm_mb": 65536}
    assert runtime_matches(pool_catalog_row, {}, role, busy_host) is False
    gap = hardware_shortfall(pool_catalog_row, role, busy_host)["gaps"]["hbm_mb"]
    assert gap == {"required": 65536, "available": 0, "observed": False}


def test_a5_idle_measured_usage_is_not_free_headroom(pool_catalog_row):
    """Real A5 capture -> normalizer -> hardware floor, measured end to end."""
    snapshot = _host_snapshot(**A5_HOST)
    spec = normalize_remote_probe_spec(snapshot)
    # All eight A5 Dies are idle and free for leasing...
    assert snapshot["free_npus"] == 8 and snapshot["free_phy_ids"] == list(range(8))
    assert spec["npu_count"] == 8 and spec["free_npus"] == 8
    # ...yet their measured used HBM is really used, so the nameplate
    # 786432 MiB is not available.
    assert spec["vram_total_mb"] == 8 * 98304
    assert spec["vram_free_mb"] == A5_MEASURED_FREE_HBM_MB == 748185
    assert spec["vram_free_observed"] is True

    host_specs = {HOST: spec}
    over = {"name": "default", "command": "pytest", "hbm_mb": 8 * 96000}
    assert runtime_matches(pool_catalog_row, {}, over, host_specs) is False
    assert hardware_shortfall(pool_catalog_row, over, host_specs)["gaps"]["hbm_mb"] == {
        "required": 768000, "available": 748185}
    # The same request stays admissible just under measured aggregate headroom.
    fits = {"name": "default", "command": "pytest", "hbm_mb": A5_MEASURED_FREE_HBM_MB}
    assert runtime_matches(pool_catalog_row, {}, fits, host_specs) is True


def test_aggregate_hbm_floor_does_not_prove_one_die_can_host_it(pool_catalog_row):
    """`hbm_mb` is aggregate across free Dies; a single-Die view is stricter."""
    snapshot = _host_snapshot(**A5_HOST)
    aggregate = {HOST: normalize_remote_probe_spec(snapshot)}
    # Restrict the same capture to its first physical Die.
    one_die = {HOST: normalize_remote_probe_spec(
        {**snapshot, "npus": snapshot["npus"][:1], "free_phy_ids": [0], "total_phy_devices": 1})}
    assert one_die[HOST]["free_npus"] == 1
    assert one_die[HOST]["vram_free_mb"] == 98304 - 4778

    role = {"name": "default", "command": "pytest", "npu_count": 1, "hbm_mb": 96000}
    # 96000 MiB fits the 8-Die aggregate but not the Die that would host it.
    assert runtime_matches(pool_catalog_row, {}, role, aggregate) is True
    assert runtime_matches(pool_catalog_row, {}, role, one_die) is False
    assert hardware_shortfall(pool_catalog_row, role, one_die)["gaps"]["hbm_mb"] == {
        "required": 96000, "available": 93526}


def _zero_hbm_single_die_card():
    """A single-Die card as the raw parse leaves it when `npu-smi info` shows 0/0.

    No committed capture reports a `0 / 0` HBM column and there is no committed
    `npu-smi info -t usages` capture, so this card dict and the supplemental
    text below are protocol-shaped constructed inputs, not hardware captures.
    They drive the real `parse_npu_smi_usages` and `device_occupancy_inventory`
    producers on the shape `fleet_probe.probe_machine_record` hands them.
    """
    return {"npu_id": 0, "phy_ids": [0], "busy": False, "hbm_used_mb": 0, "hbm_total_mb": 0,
            "die_count": 1, "aicore_pct": 0.0, "processes": [],
            "dies": [{"chip_id": 0, "phy_id": 0, "busy": False, "hbm_used_mb": 0,
                      "hbm_total_mb": 0, "aicore_pct": 0.0}]}


def _usages(rate_pct: int, aicore_pct: int = 0) -> str:
    return ("\tHBM Capacity(MB)                  : 98304\n"
            f"\tHBM Usage Rate(%)                 : {rate_pct}\n"
            f"\tAicore Usage Rate(%)              : {aicore_pct}\n")


def test_usages_supplement_publishes_capacity_to_the_die_row():
    """A card-only update would leave the normalizer reading 0/0 per Die."""
    enriched = parse_npu_smi_usages(_usages(5), _zero_hbm_single_die_card())
    assert (enriched["hbm_total_mb"], enriched["hbm_used_mb"]) == (98304, 4915)
    assert enriched["busy"] is False
    assert enriched["dies"][0]["hbm_total_mb"] == 98304
    assert enriched["dies"][0]["hbm_used_mb"] == 4915
    assert enriched["dies"][0]["busy"] is False
    snapshot = {"host": HOST, "npus": [enriched], **device_occupancy_inventory([enriched])}
    assert snapshot["free_phy_ids"] == [0] and snapshot["busy_phy_ids"] == []
    spec = normalize_remote_probe_spec(snapshot)
    assert spec["vram_free_mb"] == 98304 - 4915
    assert spec["vram_free_observed"] is True



def test_compute_only_supplement_preserves_die_busy_when_hbm_is_unmeasured():
    # The supplementary fields are independent observations: an unavailable
    # HBM figure cannot cancel a positive compute observation on the same Die.
    enriched = parse_npu_smi_usages(
        "\tAicore Usage Rate(%)              : 30\n", _zero_hbm_single_die_card())
    snapshot = {"host": HOST, "npus": [enriched], **device_occupancy_inventory([enriched])}
    assert enriched["busy"] is True and enriched["dies"][0]["busy"] is True
    assert snapshot["free_phy_ids"] == [] and snapshot["busy_phy_ids"] == [0]
    spec = normalize_remote_probe_spec(snapshot)
    assert spec["free_npus"] == 0 and spec["vram_free_mb"] == 0
    assert spec["vram_free_observed"] is False

def test_usages_supplement_that_reports_occupancy_frees_no_die(pool_catalog_row):
    """A newly busy supplement must not leave its Phy-ID listed free.

    `free_phy_ids` is authoritative for execution-unit capacity, so a card the
    supplement turns busy has to lose its Die from the free inventory and stop
    contributing headroom.
    """
    enriched = parse_npu_smi_usages(_usages(80), _zero_hbm_single_die_card())
    assert enriched["dies"][0]["hbm_used_mb"] == 78643
    assert enriched["busy"] is True and enriched["dies"][0]["busy"] is True

    snapshot = {"host": HOST, "npus": [enriched], **device_occupancy_inventory([enriched])}
    assert snapshot["free_npu_ids"] == [] and snapshot["busy_npu_ids"] == [0]
    assert snapshot["free_phy_ids"] == [] and snapshot["busy_phy_ids"] == [0]
    assert snapshot["total_phy_devices"] == 1

    spec = normalize_remote_probe_spec(snapshot)
    assert spec["npu_count"] == 1 and spec["free_npus"] == 0
    assert spec["vram_free_mb"] == 0
    role = {"name": "default", "command": "pytest", "npu_count": 1, "hbm_mb": 8192}
    assert runtime_matches(pool_catalog_row, {}, role, {HOST: spec}) is False
    assert hardware_shortfall(pool_catalog_row, role, {HOST: spec})["gaps"]["hbm_mb"] == {
        "required": 8192, "available": 0}


def test_usages_supplement_keeps_observed_facts_and_ambiguous_dies_unsplit():
    """Capacity telemetry only adds evidence; it never clears observed facts."""
    occupied = _zero_hbm_single_die_card()
    occupied["dies"][0]["busy"] = True
    occupied["busy"] = True
    occupied["processes"] = [{"pid": 4242, "process_name": "reduce_test", "memory_mb": 345}]
    enriched = parse_npu_smi_usages(_usages(5), occupied)
    assert enriched["busy"] is True and enriched["dies"][0]["busy"] is True
    assert device_occupancy_inventory([enriched])["free_phy_ids"] == []

    # An already measured card is untouched, so the real A5 capture's values
    # survive a supplemental round trip unchanged.
    measured = _host_snapshot(**A5_HOST)["npus"][0]
    assert parse_npu_smi_usages(_usages(50), measured) == measured

    # An unattributable card-wide figure is not applied to a dual-Die card at
    # all: no fabricated split, and no card/Die occupancy disagreement that
    # would let a busy card keep offering free Phy-IDs.
    dual = {"npu_id": 0, "phy_ids": [0, 1], "busy": False, "hbm_used_mb": 0, "hbm_total_mb": 0,
            "die_count": 2, "aicore_pct": 0.0, "processes": [],
            "dies": [{"chip_id": 0, "phy_id": 0, "busy": False, "hbm_used_mb": 0, "hbm_total_mb": 0},
                     {"chip_id": 1, "phy_id": 1, "busy": False, "hbm_used_mb": 0, "hbm_total_mb": 0}]}
    assert parse_npu_smi_usages(_usages(80), dual) == dual
    # The Dies stay leasable with unmeasured headroom, crediting no capacity.
    dual_spec = normalize_remote_probe_spec(
        {"host": HOST, "npus": [dual], **device_occupancy_inventory([dual])})
    assert dual_spec["free_npus"] == 2
    assert dual_spec["vram_free_mb"] == 0
    assert dual_spec["vram_free_observed"] is False


def _hardware_client(tmp_path, *, busy=(), prepared=True):
    backend = GangBackend(tmp_path / "host", devices=list(range(16)), busy=set(busy))
    pool = RuntimePool(tmp_path / "manager", backend)
    spec = _runtime_spec(service_ports=[48000])
    if prepared:
        backend.mark_prepared(spec)
        pool.register("runtime-a", spec)
    store = AgentSessions(tmp_path / "sessions")
    context = store.attach("codex", "native-hardware", str(tmp_path))
    return TaskClient(context["context_file"], pool=pool, user="alice"), backend


def test_oversized_memory_request_waits_with_the_observed_shortfall(tmp_path):
    client, _backend = _hardware_client(tmp_path)
    snapshot = _host_snapshot(**A2_HOST)
    with mock.patch.object(client.coordinator, "_probe_host_record", return_value=snapshot) as probe:
        reply = client.run("true", sources={}, resources={"memory_mb": 786432, "cpu_cores": 160})

    assert reply["state"] == "waiting_for_runtime", reply
    reason = reply["reason"]
    assert "requested host hardware is not available" in reason
    assert "memory_mb needs 786432, 150000 available" in reason
    assert "cpu_cores needs 160, 64 available" in reason
    # One probe per candidate host for the whole placement decision.
    assert probe.call_count == 1
    assert reply["host_observations"] == [{"host": HOST, "state": "observed", "cache_hit": False}]


def test_satisfiable_hardware_request_runs_and_is_not_reproved(tmp_path):
    # hbm_mb is an aggregate free-HBM floor: the A3 capture has 4 free Dies
    # holding ~249 GiB free in total. It does not assert that one Die holds
    # 196 GiB; the host queue still owns per-device allocation.
    client, backend = _hardware_client(tmp_path)
    snapshot = _host_snapshot(**A3_HOST)
    with mock.patch.object(client.coordinator, "_probe_host_record", return_value=snapshot) as probe:
        reply = client.run("true", sources={}, resources={"memory_mb": 786432, "cpu_cores": 160, "hbm_mb": 196608})

    assert reply["state"] == "running", reply
    assert ("host", "submit") in backend.calls
    # The accepted placement is not re-proved by the post-preparation check.
    assert probe.call_count == 1


def test_unconstrained_job_never_probes_host_capacity(tmp_path):
    client, backend = _hardware_client(tmp_path)
    with mock.patch.object(client.coordinator, "_probe_host_record") as probe:
        reply = client.run("true", sources={}, resources={"npu_count": 0})

    assert reply["state"] == "running", reply
    assert ("host", "submit") in backend.calls
    assert probe.call_count == 0


def test_probe_failure_stays_visible_instead_of_an_empty_fleet(tmp_path):
    client, _backend = _hardware_client(tmp_path)
    with mock.patch.object(client.coordinator, "_probe_host_record",
                           side_effect=OSError("ssh: connect to host 192.0.2.1 port 22: No route to host")):
        reply = client.run("true", sources={}, resources={"memory_mb": 4096})

    assert reply["state"] == "waiting_for_runtime", reply
    observation = reply["host_observations"][0]
    assert observation["state"] == "unavailable"
    assert "No route to host" in observation["error"]
    assert "capacity not observed for memory_mb" in reply["reason"]


def test_offline_snapshot_is_reported_rather_than_treated_as_capacity(tmp_path):
    client, _backend = _hardware_client(tmp_path)
    offline = {"alias": HOST, "host": HOST, "online": False, "error": "probe timed out after 20s"}
    with mock.patch.object(client.coordinator, "_probe_host_record", return_value=offline):
        reply = client.run("true", sources={}, resources={"memory_mb": 4096})

    assert reply["state"] == "waiting_for_runtime", reply
    assert reply["host_observations"][0] == {
        "host": HOST, "state": "unavailable", "error": "probe timed out after 20s"}


def test_select_execution_target_counts_physical_dies_across_local_and_fleet():
    fleet_hosts = [_host_snapshot(**A3_HOST, host="192.168.13.165")]
    workstation_local = {"os_family": "macos", "cpu_cores": 24, "usable_cpu_cores": 20.0,
                         "mem_total_mb": 131072, "mem_available_mb": 96000}

    local_target = select_execution_target(
        {"tier": "heavy", "cpu_cores": 16, "memory_mb": 32768},
        local_spec=workstation_local, fleet_hosts=fleet_hosts)
    assert local_target["decision"] == "local"

    # The A3 capture has 4 free Dies on 2 free cards; a 4-Die request fits.
    remote_target = select_execution_target(
        {"tier": "npu", "npu_count": 4, "hbm_mb": 240000, "memory_mb": 65536},
        local_spec=workstation_local, fleet_hosts=fleet_hosts)
    assert remote_target["decision"] == "remote"
    assert remote_target["host"] == "192.168.13.165"
    assert remote_target["target_spec"]["npu_count"] == 16
    assert remote_target["target_spec"]["free_npus"] == 4

    rejected = select_execution_target(
        {"tier": "npu", "npu_count": 16, "hbm_mb": 1048576},
        local_spec=workstation_local, fleet_hosts=fleet_hosts)
    assert rejected["decision"] == "reject"
    assert rejected["shortfall"]["remote"][0]["gaps"]["npu_count"] == {"required": 16, "available": 4}


def test_failed_hardware_probe_retries_on_the_existing_placement_window(tmp_path, monkeypatch):
    import time
    client, _backend = _hardware_client(tmp_path)
    now = [time.time()]
    monkeypatch.setattr("mindie_coordinator.service.time.time", lambda: now[0])
    with mock.patch.object(client.coordinator, "_probe_host_record", side_effect=OSError("ssh unavailable")) as probe:
        reply = client.run("true", sources={}, resources={"memory_mb": 1024})
        assert reply["state"] == "waiting_for_runtime"
        for _ in range(3):
            client.coordinator.reconcile()
        assert probe.call_count == 1
        from mindie_coordinator.service_gang import GANG_PLAN_RETRY_SECONDS
        now[0] += GANG_PLAN_RETRY_SECONDS + 0.1
        client.coordinator.reconcile()
        assert probe.call_count == 2


def test_first_run_reports_the_configured_hosts_actual_hardware_gap(tmp_path):
    client, backend = _hardware_client(tmp_path, prepared=False)
    # The same configured-machine seam used by production provisioning; no
    # prepared root exists yet and no container operation should be attempted.
    configured = [{"host": {"ip": HOST, "port": 22, "user": "root"},
                   "user": "alice", "container": {"name": "mindie-alice", "ssh_port": 2201}}]
    with mock.patch.object(client.coordinator, "_configured_machines", return_value=configured), \
         mock.patch.object(client.coordinator, "_probe_host_record", return_value=_host_snapshot(**A2_HOST)):
        reply = client.run("true", sources={}, resources={"memory_mb": 786432})
    assert reply["state"] == "waiting_for_runtime"
    assert "memory_mb needs 786432, 150000 available" in reply["reason"]
    assert not any(action == "submit" for _host, action in backend.calls)
