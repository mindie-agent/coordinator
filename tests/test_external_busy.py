"""Explicit sharing permits external occupancy while retaining owned leases."""
import json
import sqlite3

import pytest

from mindie_coordinator.host import mindie_npu_coordination as host_protocol
from mindie_coordinator.host.mindie_npu_coordination import CoordinationError, NpuCoordinator, handle_request
from mindie_coordinator.placement import normalize_resources, role_plan


BUSY = {"status": "ok", "devices": [0, 1], "busy": {"0": [{"kind": "process", "pid": 4321}]}, "free": [1]}
FREE = {"status": "ok", "devices": [0, 1], "busy": {}, "free": [0, 1]}
GUARD = {"marker": "a" * 32, "boot_id": "test-boot", "retain_until_release": True}


@pytest.fixture
def queue(tmp_path):
    now = [1000.0]
    return NpuCoordinator(tmp_path, clock=lambda: now[0]), now


def submit(host, task="shared", **extra):
    return host.submit({"task_id": task, "agent_id": "test-owner", "devices": [0],
                        "allow_external_busy": True, **extra})


def start(host, monkeypatch, *, task="shared", **extra):
    submit(host, task, **extra)
    grant = host.acquire(task, BUSY, listening={"status": "ok", "ports": []})["task"]
    token = grant["fence_token"]
    host.preflight(task, token, BUSY)
    state = {"alive": True}

    def owned_process(value, *, completion_confirmed=False):
        value = json.loads(value) if isinstance(value, str) else value
        return bool(value) and (state["alive"] or (value.get("retain_until_release") and not completion_confirmed))

    monkeypatch.setattr(host_protocol, "process_guard_busy", owned_process)
    host.activate(task, token, pid=9876, process_guard=GUARD, heartbeat_ttl_seconds=1)
    return token, state, grant


@pytest.mark.parametrize("resources", [
    {"allow_external_busy": True}, {"npu_count": 1, "allow_external_busy": True},
    {"devices": [], "allow_external_busy": True}, {"devices": [0, 1], "allow_external_busy": True},
    {"devices": [0], "allow_external_busy": "true"}, {"devices": [0], "allow_external_busy": 1},
])
def test_sharing_requires_one_explicit_device_and_a_real_boolean(resources):
    with pytest.raises(ValueError, match="allow_external_busy"):
        normalize_resources(resources)


def test_sharing_is_preserved_in_default_and_explicit_roles():
    resources = normalize_resources({"devices": [0], "npu_count": 1, "allow_external_busy": True})
    assert resources == {"devices": [0], "allow_external_busy": True}
    assert role_plan({"host": "selected-host"}, resources, "run")[0]["allow_external_busy"] is True
    roles = role_plan({"roles": [{"name": "shared"}, {"name": "strict", "devices": [1], "allow_external_busy": False}]}, resources, "run")
    assert roles[0]["allow_external_busy"] is True and roles[1]["allow_external_busy"] is False
    with pytest.raises(ValueError, match="explicit physical device"):
        role_plan({"roles": [{"name": "auto", "npu_count": 1}]}, resources, "run")


def test_default_occupancy_policy_cannot_be_relaxed_by_acquire_hints(queue, tmp_path):
    host, _ = queue
    submit(host, "strict", allow_external_busy=False)
    reply = handle_request({"action": "acquire", "state_dir": str(tmp_path), "task_id": "strict", "allow_external_busy": True},
                           clock=host.clock, probe=lambda: BUSY)
    assert reply["status"] == "waiting" and reply["task"]["allow_external_busy"] is False
    with pytest.raises(CoordinationError, match="different ownership or resources"):
        submit(host, "strict")


@pytest.mark.parametrize("shared,expected_samples", [(True, 0), (False, 2)])
def test_release_samples_follow_host_owned_sharing_policy(queue, monkeypatch, tmp_path, shared, expected_samples):
    host, _ = queue
    submit(host, allow_external_busy=shared)
    grant = host.acquire("shared", FREE)["task"]
    token = grant["fence_token"]
    host.preflight("shared", token, FREE)
    monkeypatch.setattr(host_protocol, "process_guard_busy", lambda *args, **kwargs: True)
    host.activate("shared", token, pid=9876, process_guard=GUARD)
    monkeypatch.setattr(host_protocol, "process_guard_busy", lambda *args, **kwargs: False)
    samples = []
    def probe():
        samples.append(True)
        return BUSY if shared else FREE
    result = handle_request({"action": "release", "task_id": "shared", "state_dir": str(tmp_path),
                             "coordination_epoch": host.snapshot(None)["coordination_epoch"], "fence_token": token,
                             "completion_confirmed": True, "allow_external_busy": not shared,
                             "interval_seconds": 0.001}, clock=host.clock, probe=probe)
    assert result["status"] == "released"
    assert len(samples) == expected_samples


@pytest.mark.parametrize("extra", [
    {"devices": None}, {"devices": []}, {"devices": [0, 1]}, {"allow_external_busy": "true"},
])
def test_host_also_rejects_ambiguous_sharing_requests(queue, extra):
    host, _ = queue
    with pytest.raises(CoordinationError, match="allow_external_busy|devices must not be empty"):
        submit(host, **extra)


def test_sharing_grants_the_real_device_and_keeps_other_leases_exclusive(queue):
    host, _ = queue
    assert submit(host)["task"]["allow_external_busy"] is True
    grant = host.acquire("shared", BUSY)
    assert grant["status"] == "granted"
    assert grant["task"]["requested_count"] == 1
    assert grant["environment"]["ASCEND_RT_VISIBLE_DEVICES"] == "0"
    submit(host, "other")
    assert host.acquire("other", BUSY)["reason"] == "requested_devices_unavailable"
    assert host.cancel("shared", BUSY)["status"] == "cancelled"
    assert host.acquire("other", BUSY)["status"] == "granted"


def test_a_shared_request_does_not_overlap_an_existing_strict_managed_lease(queue):
    host, _ = queue
    submit(host, "strict", allow_external_busy=False)
    assert host.acquire("strict", FREE)["status"] == "granted"
    submit(host)
    assert host.acquire("shared", BUSY)["status"] == "waiting"


@pytest.mark.parametrize("observed", [None, {"status": "failed", "error": "probe failed"}, {"status": "ok", "devices": [1], "busy": {}}])
def test_sharing_still_requires_successful_probe_and_visible_device(queue, observed):
    host, _ = queue
    submit(host)
    assert host.acquire("shared", observed)["status"] != "granted"
    token = host.acquire("shared", BUSY)["task"]["fence_token"]
    assert host.preflight("shared", token, observed)["status"] != "starting"


@pytest.mark.parametrize("hold_after_grant", [False, True])
def test_sharing_respects_holds_at_acquire_and_preflight(queue, hold_after_grant):
    host, now = queue
    submit(host)
    token = host.acquire("shared", BUSY)["task"]["fence_token"] if hold_after_grant else None
    host.add_hold({"hold_id": "reserved", "owner": "other", "devices": [0], "not_before": now[0], "duration_seconds": 60})
    if hold_after_grant:
        assert host.preflight("shared", token, BUSY)["status"] == "waiting"
    else:
        assert host.acquire("shared", BUSY)["status"] == "waiting"


def test_shared_activation_requires_retained_owned_guard(queue, monkeypatch):
    host, _ = queue
    submit(host)
    token = host.acquire("shared", BUSY)["task"]["fence_token"]
    host.preflight("shared", token, BUSY)
    monkeypatch.setattr(host_protocol, "process_guard_busy", lambda *args, **kwargs: True)
    for guard in (None, {"marker": "a" * 32, "boot_id": "test-boot"}, {**GUARD, "retain_until_release": False}):
        with pytest.raises(CoordinationError, match="shared NPU activation requires"):
            host.activate("shared", token, pid=9876, process_guard=guard)


def test_external_worker_is_not_a_reason_to_retain_a_drained_owned_lease(queue, monkeypatch):
    host, _ = queue
    token, owned, _ = start(host, monkeypatch)
    assert host.release("shared", token, BUSY, completion_confirmed=True)["status"] == "orphaned_busy"
    owned["alive"] = False
    assert host.release("shared", token, BUSY)["status"] == "orphaned_busy"
    assert host.cancel("shared", BUSY)["status"] == "orphaned_busy"
    assert host.release("shared", token, BUSY, completion_confirmed=True)["status"] == "released"
    assert BUSY["busy"]["0"][0]["pid"] == 4321
    submit(host, "strict", allow_external_busy=False)
    assert host.acquire("strict", BUSY)["status"] == "waiting"


def test_completed_shared_release_preserves_ports_without_requiring_device_visibility(queue, monkeypatch):
    host, _ = queue
    token, owned, grant = start(host, monkeypatch, service_port=0)
    owned["alive"] = False
    port = grant["granted_service_port"]
    for observed, listening in ((BUSY, None), (BUSY, {"status": "ok", "ports": [port]})):
        result = host.release("shared", token, observed, completion_confirmed=True, listening=listening)
        assert result["status"] == "orphaned_busy"
        assert result["task"]["granted_service_port"] == port
    result = host.release("shared", token, None, completion_confirmed=True, listening={"status": "ok", "ports": []})
    assert result["status"] == "released" and result["task"]["granted_service_port"] is None


@pytest.mark.parametrize("preflight", [False, True])
@pytest.mark.parametrize("action", ["cancel", "expire"])
def test_unactivated_shared_grant_can_end_with_external_worker_still_running(queue, preflight, action):
    host, now = queue
    submit(host)
    token = host.acquire("shared", BUSY)["task"]["fence_token"]
    if preflight:
        host.preflight("shared", token, BUSY)
    if action == "cancel":
        assert host.cancel("shared", BUSY)["status"] == "cancelled"
    else:
        now[0] += 61
        assert host.snapshot(BUSY, task_id="shared")["tasks"][0]["state"] == "expired"


def test_shared_heartbeat_expiry_retains_owned_guard_until_confirmed_completion(queue, monkeypatch):
    host, now = queue
    token, owned, _ = start(host, monkeypatch)
    owned["alive"] = False
    now[0] += 2
    assert host.snapshot(BUSY, task_id="shared")["tasks"][0]["state"] == "orphaned_busy"
    assert host.release("shared", token, BUSY, completion_confirmed=True)["status"] == "released"


@pytest.mark.parametrize("action", ["release", "cancel", "gc"])
def test_unguarded_shared_legacy_activation_cannot_be_released_from_occupancy(queue, action):
    host, _ = queue
    submit(host)
    token = host.acquire("shared", BUSY)["task"]["fence_token"]
    with host._transaction() as db:
        db.execute("UPDATE tasks SET state='active', started_at=1, heartbeat_deadline=0 WHERE task_id='shared'")
    if action == "release":
        result = host.release("shared", token, FREE, completion_confirmed=True)["task"]
    elif action == "cancel":
        result = host.cancel("shared", FREE)["task"]
    else:
        result = host.snapshot(FREE, task_id="shared")["tasks"][0]
    assert result["state"] == "orphaned_busy"


def test_existing_host_database_keeps_strict_policy_on_upgrade(queue, tmp_path):
    host, _ = queue
    submit(host, "existing", allow_external_busy=False)
    with sqlite3.connect(host.db_path) as db:
        db.execute("ALTER TABLE tasks DROP COLUMN allow_external_busy")
        db.execute("UPDATE meta SET value='5' WHERE key='schema_version'")
    upgraded = NpuCoordinator(tmp_path, clock=host.clock)
    result = upgraded.acquire("existing", BUSY)
    assert result["status"] == "waiting" and result["task"]["allow_external_busy"] is False
