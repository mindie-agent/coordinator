"""A completed owned family releases its shared lease without a device scan."""
import json

import pytest

from mindie_coordinator.host import mindie_npu_coordination as protocol
from test_external_busy import BUSY, GUARD, FREE, start


@pytest.fixture
def active(tmp_path, monkeypatch):
    host = protocol.NpuCoordinator(tmp_path, clock=lambda: 1000)
    token, family, grant = start(host, monkeypatch)
    args = {"action": "release", "task_id": "shared", "state_dir": str(tmp_path),
            "coordination_epoch": host.snapshot(None)["coordination_epoch"],
            "fence_token": token, "completion_confirmed": True}
    return host, token, family, args


def test_completed_shared_family_skips_device_probe_and_keeps_next_acquire_fresh(active):
    host, token, family, args = active
    family["alive"] = False
    reply = protocol.handle_request(args, clock=host.clock,
                                    probe=lambda: pytest.fail("completed shared family must not scan all hardware"))
    assert reply["status"] == "released" and reply["occupancy"] is None
    assert reply["task"]["process_guarded"] is False
    host.submit({"task_id": "next", "agent_id": "owner", "devices": [0], "allow_external_busy": True})
    assert host.acquire("next", None)["status"] == "probe_failed"
    assert host.acquire("next", BUSY)["status"] == "granted"


def test_live_or_unknown_family_cannot_release_even_with_confirmation(active, monkeypatch):
    host, token, family, args = active
    for unknown in (False, True):
        if unknown:
            monkeypatch.setattr(protocol, "process_guard_busy", lambda *a, **kw: True)
        reply = protocol.handle_request(args, clock=host.clock,
                                        probe=lambda: pytest.fail("family check should decide, without a device scan"))
        assert reply["status"] == "orphaned_busy" and reply["task"]["process_guarded"]


@pytest.mark.parametrize("guard", [None, {}, {**GUARD, "retain_until_release": False},
                                  {**GUARD, "marker": "malformed"}])
def test_missing_or_legacy_guard_keeps_original_probe_path(active, guard):
    host, token, family, args = active
    family["alive"] = False
    with host._transaction() as db:
        db.execute("UPDATE tasks SET process_guard=? WHERE task_id='shared'", (json.dumps(guard) if guard is not None else None,))
    sampled = []
    protocol.handle_request(args, clock=host.clock, probe=lambda: sampled.append(True) or BUSY)
    assert sampled == [True]


def test_unconfirmed_completion_keeps_sampling_and_the_retained_guard(active):
    host, token, family, args = active
    family["alive"] = False
    sampled = []
    reply = protocol.handle_request({**args, "completion_confirmed": False}, clock=host.clock,
                                    probe=lambda: sampled.append(True) or BUSY)
    assert sampled == [True] and reply["status"] == "orphaned_busy"


def test_wrong_fence_rejects_without_clearing_the_owned_guard(active):
    host, token, family, args = active
    family["alive"] = False
    with pytest.raises(protocol.CoordinationError, match="fencing"):
        protocol.handle_request({**args, "fence_token": token + 1}, clock=host.clock, probe=lambda: FREE)
    assert host.snapshot(None)["tasks"][0]["process_guarded"]


def test_actual_boot_mismatch_remains_unknown(active, monkeypatch):
    host, token, family, args = active
    monkeypatch.undo()  # Use the real host guard implementation, not the fixture's family model.
    reply = protocol.handle_request(args, clock=host.clock, probe=lambda: pytest.fail("no whole-device scan"))
    assert reply["status"] == "orphaned_busy" and reply["task"]["process_guarded"]
