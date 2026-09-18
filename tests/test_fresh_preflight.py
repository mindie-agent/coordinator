"""One fresh admission sample, with the original queue and recovery boundaries."""
from contextlib import closing
from types import MethodType
import sqlite3

import pytest

import test_coordinator as fixtures
from mindie_coordinator.backend import RemoteBackend
from mindie_coordinator.host.mindie_npu_coordination import CoordinationError, NpuCoordinator, handle_request


@pytest.fixture
def case():
    value = fixtures.PoolTests(methodName="runTest")
    value.setUp()
    value.backend.submit_and_acquire = MethodType(RemoteBackend.submit_and_acquire, value.backend)
    value.backend.submit_and_preflight = MethodType(RemoteBackend.submit_and_preflight, value.backend)
    try:
        yield value
    finally:
        value.tearDown()


def request(tmp_path, **kwargs):
    coordinator = NpuCoordinator(tmp_path)
    return {"action": "submit-acquire-preflight", "state_dir": str(tmp_path),
            "coordination_epoch": coordinator.snapshot(None)["coordination_epoch"],
            "task_id": "one", "agent_id": "owner", "devices": [0], **kwargs}


def free():
    return {"status": "ok", "devices": [0, 1], "busy": {}}


def test_fresh_grant_uses_one_sample_and_the_original_fenced_preflight(tmp_path):
    probes = []
    reply = handle_request(request(tmp_path), probe=lambda: probes.append(True) or free())
    assert len(probes) == 1
    assert reply["task"]["state"] == "starting"
    assert reply["granted_task"]["state"] == "granted"
    assert reply["task"]["fence_token"] == reply["granted_task"]["fence_token"]
    assert reply["environment"]["ASCEND_RT_VISIBLE_DEVICES"] == "0"


def test_existing_grant_is_not_preflighted_from_a_new_sample(tmp_path):
    args = request(tmp_path)
    grant = handle_request({**args, "action": "submit-acquire"}, probe=free)
    replay = handle_request(args, probe=lambda: pytest.fail("existing grant must not be resampled"))
    assert replay["task"] == grant["task"] and "granted_task" not in replay


def test_competing_acquire_does_not_reuse_its_sample_for_an_existing_grant(tmp_path, monkeypatch):
    original = NpuCoordinator.acquire

    def raced(self, *args, **kwargs):
        original(self, *args, **kwargs)  # Another caller got the grant first.
        return original(self, *args, **kwargs)

    monkeypatch.setattr(NpuCoordinator, "acquire", raced)
    reply = handle_request(request(tmp_path), probe=free)
    assert reply["task"]["state"] == "granted" and "granted_task" not in reply


@pytest.mark.parametrize("condition", ["busy", "not_before", "fifo", "port"])
def test_waiting_results_never_enter_starting(tmp_path, condition):
    args = request(tmp_path)
    ports = {"status": "ok", "ports": []}
    observation = free()
    if condition == "busy":
        observation["busy"] = {"0": ["external"]}
    elif condition == "not_before":
        args["not_before"] = 4102444800  # 2100, representable on Windows too.
        args["latest_start"] = 4102444810
    elif condition == "fifo":
        handle_request({**args, "action": "submit", "task_id": "ahead", "priority": 100}, probe=free)
    else:
        args.update(service_port=50001, service_ports=[50001])
        ports["ports"] = [50001]
    reply = handle_request(args, probe=lambda: observation, listening_ports=lambda: ports)
    assert reply["task"]["state"] == "queued" and "granted_task" not in reply


def test_epoch_change_between_grant_and_preflight_is_rejected(tmp_path, monkeypatch):
    original = NpuCoordinator.preflight

    def restarted(self, *args, **kwargs):
        with closing(sqlite3.connect(tmp_path / "coordinator.sqlite3")) as db, db:
            db.execute("UPDATE meta SET value='new-epoch' WHERE key='coordination_epoch'")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(NpuCoordinator, "preflight", restarted)
    with pytest.raises(CoordinationError, match="epoch changed"):
        handle_request(request(tmp_path), probe=free)


def test_expired_grant_cannot_enter_starting_with_the_earlier_sample(tmp_path, monkeypatch):
    original = NpuCoordinator.preflight

    def delayed(self, *args, **kwargs):
        now = self.clock()
        self.clock = lambda: now + 120
        return original(self, *args, **kwargs)

    monkeypatch.setattr(NpuCoordinator, "preflight", delayed)
    with pytest.raises(CoordinationError, match="fence|expected granted"):
        handle_request(request(tmp_path), probe=free)


def test_probe_failure_retains_the_queued_task_and_no_starting_transition(tmp_path):
    args = request(tmp_path)
    reply = handle_request(args, probe=lambda: {"status": "failed", "error": "unknown visibility"})
    assert reply["status"] == "probe_failed" and "granted_task" not in reply
    tasks = NpuCoordinator(tmp_path).snapshot(None)["tasks"]
    assert [(row["task_id"], row["state"]) for row in tasks] == [("one", "queued")]


def test_fresh_cpu_start_avoids_unneeded_hardware_probes(tmp_path):
    args = request(tmp_path)
    args.pop("devices")
    args["npu_count"] = 0
    reply = handle_request(args, probe=lambda: pytest.fail("CPU NPU probe"),
                           listening_ports=lambda: pytest.fail("no requested port"))
    assert reply["task"]["state"] == "starting" and reply["environment"]["ASCEND_RT_VISIBLE_DEVICES"] == ""


def test_first_managed_run_verifies_before_admission_and_records_both_transitions(case):
    binding = case.bind("alice", case.root / "a")
    case.backend.calls.clear()
    job = case.managed("alice", binding)
    assert job["state"] == "running"
    calls = case.backend.calls
    assert [action for kind, action in calls if kind == "host"] == ["status", "submit-acquire-preflight", "activate"]
    assert next(i for i, call in enumerate(calls) if call[0] == "inspect") < calls.index(("host", "submit-acquire-preflight"))
    assert ("job", "status") not in calls
    events = [item["state"] for item in case.pool.events("alice")["events"]
              if item["kind"] == "run-state" and item.get("run") == job["id"]]
    assert events == ["granted", "starting", "active"]


def test_queued_run_discards_the_early_verification_and_checks_again_after_acquire(case):
    binding = case.bind("alice", case.root / "a")
    case.backend.busy = [0]
    job = case.managed("alice", binding)
    assert job["state"] == "queued"
    case.backend.busy = []
    case.backend.calls.clear()
    job = case.pool.managed_advance(job["id"])
    assert job["state"] == "running"
    assert ("host", "submit-acquire-preflight") not in case.backend.calls
    assert ("host", "acquire") in case.backend.calls and ("host", "preflight") in case.backend.calls
    assert any(kind == "inspect" for kind, _ in case.backend.calls)
    assert ("job", "status") in case.backend.calls


def test_lost_combined_reply_observes_same_task_and_job_before_preparing(case):
    binding = case.bind("alice", case.root / "a")
    case.backend.fail_after = "submit-acquire-preflight"
    job = case.managed("alice", binding)
    assert job["state"] == "uncertain" and not case.backend.jobs
    with case.pool.transaction() as db:
        task_id = case.pool.get(db, "run", job["id"])["task_id"]
    case.backend.calls.clear()
    restarted = fixtures.RuntimePool(case.root / "manager", case.backend)
    recovered = restarted.managed_advance(job["id"])
    assert recovered["state"] == "running"
    assert case.backend.calls.index(("job", "status")) < case.backend.calls.index(("job", "prepare"))
    assert ("host", "submit-acquire-preflight") not in case.backend.calls
    with restarted.transaction() as db:
        assert restarted.get(db, "run", job["id"])["task_id"] == task_id


def test_verification_failure_never_submits_or_prepares(case):
    binding = case.bind("alice", case.root / "a")
    case.backend.verify_preflight = lambda *args, **kwargs: False
    case.backend.calls.clear()
    job = case.managed("alice", binding)
    assert job["state"] == "failed" and job["lease_state"] == "cancelled"
    assert "did not confirm" in job["preflight_error"]
    assert not case.backend.jobs
    assert not any(kind == "host" and action.startswith("submit") for kind, action in case.backend.calls)


@pytest.mark.parametrize("during", ["verify", "admission", "activate"])
def test_cancel_at_each_new_boundary_prevents_business_go(case, during):
    binding = case.bind("alice", case.root / "a")

    def cancel():
        with case.pool.transaction() as db:
            job = case.pool.rows(db, "job")[0]
            job["cancel_requested"] = True
            case.pool.put(db, "job", job)

    if during == "verify":
        case.backend.verify_preflight = lambda *args, **kwargs: (cancel(), True)[1]
    else:
        original = case.backend.host

        def host(runtime, args):
            reply = original(runtime, args)
            if args["action"] == ("submit-acquire-preflight" if during == "admission" else "activate"):
                cancel()
            return reply

        case.backend.host = host
    case.backend.calls.clear()
    job = case.managed("alice", binding)
    if job["state"] == "stopping":
        job = case.pool.managed_advance(job["id"])
    assert job["state"] == "cancelled" and job["lease_state"] in {"cancelled", "released"}
    assert ("job", "go") not in case.backend.calls
    if during != "activate":
        assert ("job", "prepare") not in case.backend.calls


def test_combined_managed_hold_go_still_waits_for_the_group_gate(case):
    binding = case.bind("alice", case.root / "a")
    case.backend.calls.clear()
    job = case.pool.managed_start("alice", binding["id"], "group", {}, "native-a", [0], 0,
                                  "true", {}, hold_go=True)
    assert job["state"] == "waiting" and job["lease_state"] == "active"
    assert ("job", "go") not in case.backend.calls
    assert case.pool.managed_release_gate("alice", job["id"])["state"] == "running"
    assert case.backend.calls.count(("job", "prepare")) == 1
    assert case.backend.calls.count(("job", "go")) == 1
