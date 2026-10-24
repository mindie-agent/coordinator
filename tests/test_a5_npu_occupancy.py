"""A5 (Ascend950DT) idle/occupied occupancy regressions from live captures.

Tier 0: parses golden `npu-smi` captures only; no subprocess, network or sleep.

The failure these lock down: an idle A5 card must be reported free. A5 prints a
single-column device table (no Chip/Phy-ID pair) and, on some firmware, a
process section with no `No process in device.` sentinel. Reading either shape
as unparseable makes every idle card look occupied (`free=0`).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mindie_coordinator import fleet_probe
from mindie_coordinator.host.mindie_npu_coordination import parse_npu_smi_info as parse_coordinator_info
from mindie_coordinator.topology_gang_observation import (
    parse_npu_smi_capture,
    topology_from_fleet_snapshot,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load_helper(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / "live" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROCESSES = load_helper("stage1_process_stream")


def fixture(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


def device_sections(name):
    """Split the concatenated per-device proc-mem capture on its DEVICE markers."""
    sections = {}
    current = None
    for line in fixture(name).splitlines():
        if line.startswith("=====DEVICE "):
            current = int(line.split()[1])
            sections[current] = []
        elif line.startswith("=====RC "):
            current = None
        elif current is not None:
            sections[current].append(line)
    return {device: "\n".join(lines) for device, lines in sections.items()}


def busy_ids(parsed):
    return [card["npu_id"] for card in parsed["npus"] if card["busy"]]


def free_ids(parsed):
    return [card["npu_id"] for card in parsed["npus"] if not card["busy"]]


def test_a5_idle_summary_exposes_all_cards_as_free():
    parsed = fleet_probe.parse_npu_smi_info(fixture("a5-25rc3-info-idle.txt"))
    assert [card["npu_id"] for card in parsed["npus"]] == list(range(8))
    assert free_ids(parsed) == list(range(8))
    assert busy_ids(parsed) == []
    card = parsed["npus"][0]
    assert card["chip_name"] == "Ascend950DT"
    assert card["health"] == "OK"
    assert card["phy_ids"] == [0]
    assert card["die_count"] == 1
    assert card["power_w"] is not None and card["temp_c"] is not None
    # An idle A5 card still reports used HBM. The capture's measured total used
    # value is kept verbatim (`4778 / 98304` on card 0) rather than rewritten to
    # zero; this test does not attribute it to firmware, driver or runtime.
    # Occupancy stays with the process table: every card above is still free.
    assert card["hbm_used_mb"] == 4778
    assert card["hbm_total_mb"] == 98304
    assert card["dies"] == [
        {"chip_id": 0, "phy_id": 0, "bus_id": "NA", "aicore_pct": 0.0,
         "hbm_used_mb": 4778, "hbm_total_mb": 98304, "busy": False}
    ]
    assert [c["hbm_used_mb"] for c in parsed["npus"]] == [4778, 4785, 4778, 4782, 4779, 4783, 4778, 4784]


def test_a5_occupied_summary_marks_every_card_busy():
    parsed = fleet_probe.parse_npu_smi_info(fixture("a5-25rc3-info-occupied.txt"))
    assert busy_ids(parsed) == list(range(8))
    assert free_ids(parsed) == []
    assert {process["process_name"] for process in parsed["processes"]} == {"reduce_test"}


def test_a5_mixed_summary_reports_only_the_occupied_card():
    parsed = fleet_probe.parse_npu_smi_info(fixture("a5-25rc3-info-mixed.txt"))
    # NPU 0 runs a 345 MB python3 job: below the generic 8192 MB threshold, but
    # any process on an A5 card is workload occupancy.
    assert busy_ids(parsed) == [0]
    assert free_ids(parsed) == list(range(1, 8))
    assert parsed["npus"][0]["processes"][0]["process_name"] == "python3"
    assert parsed["npus"][0]["processes"][0]["memory_mb"] == 345


def test_a5_lease_observation_sees_cards_and_free_dies():
    idle = fixture("a5-25rc3-info-idle.txt")
    capture = parse_npu_smi_capture(idle, host="a5-idle")
    assert len(capture.cards) == 8
    assert sorted(die.phy_id for card in capture.cards for die in card.dies) == list(range(8))
    assert all(not die.observed_busy for card in capture.cards for die in card.dies)

    mixed_raw = "===SECTION:NPU_SMI===\n" + fixture("a5-25rc3-info-mixed.txt")
    snapshot = fleet_probe.parse_fleet_probe_output(mixed_raw, alias="a5-mixed", host="a5-mixed")
    mixed = topology_from_fleet_snapshot(snapshot, host="a5-mixed")
    assert len(mixed.cards) == 8
    assert [die.phy_id for card in mixed.cards for die in card.dies if die.observed_busy] == [0]
    assert [die.phy_id for card in mixed.cards for die in card.dies if not die.observed_busy] == list(range(1, 8))


def test_a2_a3_summaries_keep_their_existing_occupancy():
    a2 = fleet_probe.parse_npu_smi_info(fixture("a2-173-125-1-2.txt"))
    assert free_ids(a2) == list(range(8))
    a3_idle = fleet_probe.parse_npu_smi_info(fixture("a3-192-168-13-165-idle.txt"))
    assert free_ids(a3_idle) == list(range(8))
    a3_busy = fleet_probe.parse_npu_smi_info(fixture("a3-192-168-13-154-busy.txt"))
    assert busy_ids(a3_busy) == list(range(8))
    a3_partial = fleet_probe.parse_npu_smi_info(fixture("a3-192-168-13-156-partial.txt"))
    assert free_ids(a3_partial) == [2, 3]


def test_a5_idle_proc_mem_forms_are_zero_processes():
    assert PROCESSES.parse_process_memory(fixture("a5-25rc3-proc-mem-idle.txt"), card=0) == []
    # Reported A5 26.x header-only form: no sentinel row at all.
    assert PROCESSES.parse_process_memory(
        fixture("a5-26rc-proc-mem-idle-header-only.txt"), card=0) == []
    # A wholly empty section with exit 0 is the same measured empty table.
    assert PROCESSES.parse_process_memory("", card=0) == []
    # Whole-card A2/A3 queries carry headers around the sentinel.
    assert PROCESSES.parse_process_memory(fixture("a2-proc-mem-idle.txt"), card=0) == []
    assert PROCESSES.parse_process_memory(fixture("a3-proc-mem-idle.txt"), card=0) == []
    # `-c <chip>` queries keep the bare sentinel they always produced.
    assert PROCESSES.parse_process_memory(fixture("a2-proc-mem-idle-chip.txt"), card=0) == []
    assert PROCESSES.parse_process_memory(fixture("a3-proc-mem-idle-chip.txt"), card=0) == []


def test_a5_proc_mem_occupied_and_mixed_rows():
    occupied = PROCESSES.parse_process_memory(fixture("a5-25rc3-proc-mem-occupied.txt"), card=0)
    assert occupied == [{"pid": 904683, "name": "reduce_test", "memory_mb": 6656}]

    sections = device_sections("a5-25rc3-proc-mem-mixed.txt")
    assert sections
    parsed = {
        device: PROCESSES.parse_process_memory(output, card=device)
        for device, output in sections.items()
    }
    assert parsed[0] == [{"pid": 2360798, "name": "python3", "memory_mb": 345}]
    assert all(parsed[device] == [] for device in range(1, 8))


@pytest.mark.parametrize("stdout,stderr,returncode", [
    (fixture("a5-proc-mem-error-invalid-card.txt"), "", 215),
    (fixture("a5-proc-mem-error-unsupported-chip.txt"), "", 215),
    (fixture("a5-info-proc-unsupported.txt"), "", 0),
    ("", "npu-smi: driver not loaded", 0),
    ("", "", 1),
    ("\tNPU ID : 0\n\tquery failed\n", "", 0),
])
def test_proc_mem_command_failures_never_become_idle(stdout, stderr, returncode):
    with pytest.raises(ValueError):
        PROCESSES.parse_process_memory(stdout, returncode=returncode, stderr=stderr, card=0)


def test_proc_mem_rejects_a_different_card_identity():
    with pytest.raises(ValueError):
        PROCESSES.parse_process_memory(fixture("a5-25rc3-proc-mem-idle.txt"), card=3)


def test_a5_observe_processes_rereads_without_the_unsupported_chip_selector(monkeypatch):
    calls = []

    class Result:
        def __init__(self, stdout, returncode):
            self.stdout = stdout
            self.stderr = ""
            self.returncode = returncode

    def query(card, chip):
        calls.append((card, chip))
        if chip is not None:
            return Result(fixture("a5-proc-mem-error-unsupported-chip.txt"), 215)
        return Result(fixture("a5-25rc3-proc-mem-idle.txt"), 0)

    monkeypatch.setattr(PROCESSES, "_query_proc_memory", query)
    sample = PROCESSES.observe_processes(0, 0, "marker")
    assert calls == [(0, 0), (0, None)]
    assert sample["state"] == "observed"
    assert sample["processes"] == []
    assert sample["query_chip"] is None
    # A measured empty table means "no foreign process", not "unknown".
    assert sample["resource_interference_detected"] is False


def test_a5_observe_processes_keeps_a_failed_second_query_unavailable(monkeypatch):
    class Result:
        def __init__(self, stdout, returncode):
            self.stdout = stdout
            self.stderr = ""
            self.returncode = returncode

    def query(card, chip):
        if chip is not None:
            return Result(fixture("a5-proc-mem-error-unsupported-chip.txt"), 215)
        return Result("", 215)

    monkeypatch.setattr(PROCESSES, "_query_proc_memory", query)
    sample = PROCESSES.observe_processes(0, 0, "marker")
    assert sample["state"] == "unavailable"
    assert sample["processes"] == []


def test_coordinator_summary_treats_a_sentinel_free_process_section_as_idle():
    # Some A5 firmware prints an empty process section instead of one
    # `No running processes found in NPU N` row per device.
    capture = [
        line for line in fixture("a5-25rc3-info-idle.txt").splitlines()
        if "No running processes" not in line
    ]
    parsed = parse_coordinator_info("\n".join(capture) + "\n")
    assert parsed["status"] == "ok"
    assert parsed["devices"] == list(range(8))
    assert parsed["free"] == list(range(8))
    assert parsed["busy"] == {}
