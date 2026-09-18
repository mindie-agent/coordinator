"""Shared admission needs fresh device identity, not external occupancy."""
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mindie_coordinator.host import mindie_npu_coordination as host


MAPPING = """\tChip Physical ID              :6
\tChip Logic ID                 :6
\tNPU ID                        :3
\tChip ID                       :0
"""
FREE = {"status": "ok", "devices": [6, 7], "busy": {}, "free": [6, 7]}
VISIBLE = {"status": "ok", "devices": [6], "busy": None, "free": []}


def test_device_probe_uses_physical_id_and_leaves_occupancy_unknown(monkeypatch):
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout=MAPPING))
    monkeypatch.setattr(host.subprocess, "run", run)
    result = host.probe_npu_device(6)
    assert result["status"] == "ok" and result["devices"] == [6]
    assert result["busy"] is None and result["free"] == []
    assert host.NpuCoordinator._busy_set(result) is None
    assert run.call_args.args == (["npu-smi", "info", "-t", "phyid-remap", "-p", "6"],)
    assert run.call_args.kwargs["timeout"] == 15


@pytest.mark.parametrize("output,rc", [
    (MAPPING, 215), ("", 0), (MAPPING.replace(":6", ":7", 1), 0),
    (MAPPING.replace(":3", ":-1"), 0),
    (MAPPING.replace("\tChip ID                       :0\n", ""), 0),
    (MAPPING + "Chip Physical ID:6\n", 0), (MAPPING + "Unknown device\n", 0),
])
def test_missing_invalid_or_unsupported_mapping_is_unknown(monkeypatch, output, rc):
    monkeypatch.setattr(host.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=rc, stdout=output)))
    result = host.probe_npu_device(6)
    assert result["status"] == "failed" and result["devices"] == [] and result["busy"] is None


@pytest.mark.parametrize("error", [OSError("unavailable"), subprocess.TimeoutExpired("npu-smi", 15)])
def test_mapping_transport_failure_is_unknown(monkeypatch, error):
    monkeypatch.setattr(host.subprocess, "run", Mock(side_effect=error))
    assert host.probe_npu_device(6)["status"] == "failed"


@pytest.fixture
def queue(tmp_path):
    now = [1000.0]
    coordinator = host.NpuCoordinator(tmp_path, clock=lambda: now[0])
    with coordinator._transaction() as db:
        epoch = coordinator._coordination_epoch(db)
    args = {"state_dir": str(tmp_path), "coordination_epoch": epoch,
            "task_id": "shared", "agent_id": "owner", "devices": [6], "allow_external_busy": True}
    return coordinator, now, args


def call(queue, action, *, full=None, device=None, **extra):
    _, now, args = queue
    return host.handle_request({**args, "action": action, **extra}, clock=lambda: now[0],
                               probe=full or Mock(side_effect=AssertionError("unexpected occupancy query")),
                               device_probe=device or Mock(return_value=VISIBLE),
                               listening_ports=lambda: {"status": "ok", "ports": []})


def test_combined_admission_uses_one_fresh_targeted_sample(queue):
    device = Mock(return_value=VISIBLE)
    reply = call(queue, "submit-acquire-preflight", device=device)
    assert reply["status"] == "starting"
    assert reply["task"]["granted_devices"] == [6]
    assert reply["task"]["fence_token"] == reply["granted_task"]["fence_token"]
    assert reply["environment"]["ASCEND_RT_VISIBLE_DEVICES"] == "6"
    device.assert_called_once_with(6)
    # Lost reply recovery must not advance or resample an existing grant/start.
    device.reset_mock()
    assert call(queue, "submit-acquire-preflight", device=device)["task"] == reply["task"]
    device.assert_not_called()


def test_queued_preflight_resamples_device_and_rejects_disappearance(queue):
    grant = call(queue, "submit-acquire")
    device = Mock(return_value={"status": "failed"})
    full = Mock(return_value={"status": "ok", "devices": [7], "busy": {}})
    reply = call(queue, "preflight", fence_token=grant["task"]["fence_token"], device=device, full=full)
    assert reply["status"] == "waiting" and reply["task"]["fence_token"] is None
    device.assert_called_once_with(6)
    full.assert_called_once_with()


def test_unsupported_targeted_query_keeps_successful_full_probe_fallback(queue):
    full = Mock(return_value=FREE)
    assert call(queue, "submit-acquire-preflight", device=Mock(return_value={"status": "failed"}), full=full)["status"] == "starting"
    full.assert_called_once_with()


def test_caller_hints_cannot_relax_strict_host_policy(queue):
    call(queue, "submit", allow_external_busy=False)
    device = Mock(side_effect=AssertionError("strict lease cannot use visibility only"))
    full = Mock(return_value={**FREE, "busy": {"6": ["external"]}})
    assert call(queue, "acquire", full=full, device=device, allow_external_busy=True)["status"] == "waiting"
    full.assert_called_once_with()


@pytest.mark.parametrize("preflight", [False, True])
def test_strict_transitions_reject_visibility_only_observations(queue, preflight):
    coordinator, _, args = queue
    coordinator.submit({**args, "allow_external_busy": False})
    if preflight:
        token = coordinator.acquire("shared", FREE)["task"]["fence_token"]
        result = coordinator.preflight("shared", token, VISIBLE)
    else:
        result = coordinator.acquire("shared", VISIBLE)
    assert result["status"] == "probe_failed"


@pytest.mark.parametrize("condition", ["strict_peer", "shared_peer", "hold", "fifo"])
def test_targeted_admission_keeps_reservations_holds_and_fifo(queue, condition):
    coordinator, now, args = queue
    if condition.endswith("peer"):
        coordinator.submit({**args, "task_id": "peer", "allow_external_busy": condition == "shared_peer"})
        coordinator.acquire("peer", FREE)
    elif condition == "hold":
        coordinator.add_hold({"hold_id": "other-hold", "owner": "other", "devices": [6],
                              "not_before": now[0], "duration_seconds": 60})
    else:
        coordinator.submit({**args, "task_id": "ahead", "priority": 100})
    reply = call(queue, "submit-acquire-preflight")
    assert reply["status"] == "waiting" and reply["task"]["state"] == "queued"


def test_hold_added_after_grant_is_checked_again(queue):
    coordinator, now, _ = queue
    grant = call(queue, "submit-acquire")
    coordinator.add_hold({"hold_id": "other-hold", "owner": "other", "devices": [6],
                          "not_before": now[0], "duration_seconds": 60})
    assert call(queue, "preflight", fence_token=grant["task"]["fence_token"])["status"] == "waiting"


def test_inventory_does_not_reclaim_other_expired_strict_lease(queue):
    coordinator, now, args = queue
    coordinator.submit({**args, "task_id": "peer", "devices": [7], "allow_external_busy": False})
    token = coordinator.acquire("peer", FREE)["task"]["fence_token"]
    coordinator.preflight("peer", token, FREE)
    coordinator.activate("peer", token, pid=123, heartbeat_ttl_seconds=1)
    now[0] += 2
    assert call(queue, "submit-acquire-preflight")["status"] == "starting"
    with coordinator._transaction() as db:
        peer = coordinator._task_row(db, "peer")
        assert peer["state"] == "orphaned_busy" and peer["fence_token"] == token
        assert host._load_devices(peer["granted_devices"]) == [7]


@pytest.mark.parametrize("state", ["granted", "starting", "active", "orphaned_busy"])
def test_expired_conflicting_reservation_keeps_full_probe_reclamation(queue, state):
    coordinator, now, args = queue
    coordinator.submit({**args, "task_id": "peer", "allow_external_busy": False})
    token = coordinator.acquire("peer", FREE)["task"]["fence_token"]
    if state != "granted":
        coordinator.preflight("peer", token, FREE)
    if state in {"active", "orphaned_busy"}:
        coordinator.activate("peer", token, pid=123, heartbeat_ttl_seconds=1)
    now[0] += 61
    if state == "orphaned_busy":
        coordinator.snapshot(None)
    device = Mock(side_effect=AssertionError("reclamation needs full occupancy"))
    full = Mock(return_value=FREE)
    assert call(queue, "submit-acquire-preflight", device=device, full=full)["status"] == "starting"
    full.assert_called_once_with()


def test_targeted_probe_does_not_bypass_epoch_or_fence(queue):
    coordinator, _, _ = queue
    grant = call(queue, "submit-acquire")
    with pytest.raises(host.CoordinationError, match="fencing token"):
        call(queue, "preflight", fence_token=grant["task"]["fence_token"] + 1)
    with pytest.raises(host.CoordinationError, match="epoch"):
        call(queue, "preflight", fence_token=grant["task"]["fence_token"], coordination_epoch="changed")
    with coordinator._transaction() as db:
        assert coordinator._task_row(db, "shared")["state"] == "granted"
