"""Prepared activation preserves host fencing and supervisor identity in one RPC."""
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import test_coordinator as fixtures
from mindie_coordinator.backend import RemoteBackend
from mindie_coordinator.host import mindie_npu_coordination as host


@pytest.fixture
def supervisor(monkeypatch):
    guard = {"marker": "a" * 32, "boot_id": "boot-one", "retain_until_release": True}
    receipt = {"pid": 42, "start_ticks": "9001", "boot_id": "boot-one", "process_guard": guard}
    prepared = {"container_name": "mindie-test", "container_id": "container-one", "receipt": receipt}
    state = {"container_id": "container-one", "boot_id": "boot-one", "pids": [810],
             "namespace": 42, "start_ticks": "9001", "state": "S", "marker": "a" * 32}
    original_text, original_bytes = Path.read_text, Path.read_bytes

    def read_text(path, *args, **kwargs):
        name = path.as_posix()
        if name == "/proc/sys/kernel/random/boot_id":
            return state["boot_id"]
        if name.startswith("/proc/") and name.endswith("/stat"):
            fields = [state["state"]] + ["0"] * 18 + [state["start_ticks"]]
            return "810 (supervisor with spaces) " + " ".join(fields)
        if name.startswith("/proc/") and name.endswith("/status"):
            return "Name:\tsupervisor\nNSpid:\t810\t" + str(state["namespace"]) + "\n"
        return original_text(path, *args, **kwargs)

    def read_bytes(path, *args, **kwargs):
        if path.as_posix().startswith("/proc/") and path.name == "environ":
            return (host.JOB_TOKEN_ENV + "=" + state["marker"]).encode() + b"\0"
        return original_bytes(path, *args, **kwargs)

    def docker(argv, **kwargs):
        if argv == ["docker", "inspect", "--format", "{{json .}}", "mindie-test"]:
            return json.dumps({"Id": state["container_id"]})
        assert argv == ["docker", "top", "container-one", "-eo", "pid"]
        return "PID\n" + "\n".join(map(str, state["pids"]))

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(host.subprocess, "check_output", docker)
    return prepared, guard, state


@pytest.mark.parametrize("change,match", [
    ({"container_id": "container-replaced"}, "container identity"),
    ({"boot_id": "boot-two"}, "boot identity"),
    ({"start_ticks": "9002"}, "unique host PID"),
    ({"namespace": 43}, "unique host PID"),
    ({"state": "Z"}, "unique host PID"),
    ({"marker": "b" * 32}, "unique host PID"),
    ({"pids": []}, "unique host PID"),
    ({"pids": [810, 811]}, "unique host PID"),
])
def test_prepared_identity_rejects_changed_or_ambiguous_process(supervisor, change, match):
    prepared, guard, state = supervisor
    state.update(change)
    with pytest.raises(host.CoordinationError, match=match):
        host.prepared_supervisor_host_pid(prepared, guard)


def test_guard_must_match_receipt_and_its_boot(supervisor):
    prepared, guard, _ = supervisor
    with pytest.raises(host.CoordinationError, match="guard disagree"):
        host.prepared_supervisor_host_pid(prepared, {**guard, "marker": "b" * 32})
    prepared["receipt"]["boot_id"] = "other-boot"
    with pytest.raises(host.CoordinationError, match="guard disagree"):
        host.prepared_supervisor_host_pid(prepared, guard)


@pytest.fixture
def authority(tmp_path, monkeypatch):
    clock = [1000.0]
    coordinator = host.NpuCoordinator(tmp_path / "authority", clock=lambda: clock[0])
    coordinator.submit({"task_id": "cpu-task", "agent_id": "owner", "npu_count": 0})
    granted = coordinator.acquire("cpu-task", None)
    token = granted["task"]["fence_token"]
    coordinator.preflight("cpu-task", token, None)
    with coordinator._transaction() as connection:
        epoch = coordinator._coordination_epoch(connection)
    request = {"state_dir": str(tmp_path / "authority"), "action": "activate", "task_id": "cpu-task",
               "fence_token": token, "coordination_epoch": epoch}
    busy = Mock(return_value=True)
    monkeypatch.setattr(host, "process_guard_busy", busy)
    return coordinator, request, clock, busy


def test_prepared_activation_preserves_authority_and_uses_resolved_pid(authority, supervisor):
    coordinator, request, clock, busy = authority
    prepared, guard, _ = supervisor
    result = host.handle_request({**request, "prepared_supervisor": prepared, "process_guard": guard, "pid": 999},
                                 clock=lambda: clock[0])
    assert result["task"]["state"] == "active"
    assert result["task"]["pid"] == 810
    with coordinator._transaction() as connection:
        assert json.loads(coordinator._task_row(connection, "cpu-task")["process_guard"]) == guard
    busy.assert_not_called()


@pytest.mark.parametrize("failure", ["fence", "epoch", "deadline", "guard_gone"])
def test_prepared_activation_keeps_fence_epoch_deadline_and_guard_checks(authority, supervisor, failure):
    coordinator, request, clock, busy = authority
    prepared, guard, state = supervisor
    if failure == "fence":
        request["fence_token"] += 1
    elif failure == "epoch":
        request["coordination_epoch"] = "another-epoch"
    elif failure == "deadline":
        clock[0] += 61
    else:
        state["marker"] = "b" * 32
    with pytest.raises(host.CoordinationError):
        host.handle_request({**request, "prepared_supervisor": prepared, "process_guard": guard},
                            clock=lambda: clock[0])
    # Read without housekeeping: rejection must not write an active task.
    with coordinator._transaction() as connection:
        assert coordinator._task_row(connection, "cpu-task")["state"] == "starting"


def test_explicit_pid_activation_keeps_legacy_path(authority, monkeypatch):
    _, request, clock, busy = authority
    monkeypatch.setattr(host, "prepared_supervisor_host_pid", Mock(side_effect=AssertionError("unexpected resolver")))
    guard = {"marker": "a" * 32, "boot_id": "boot-one"}
    result = host.handle_request({**request, "pid": 123, "process_guard": guard}, clock=lambda: clock[0])
    assert result["task"]["pid"] == 123
    busy.assert_called_once_with(guard, completion_confirmed=True)


@pytest.mark.parametrize("guard_change", [{"extra": True}, {"retain_until_release": "true"}])
def test_prepared_activation_still_validates_guard_shape(authority, supervisor, guard_change):
    coordinator, request, clock, busy = authority
    prepared, guard, _ = supervisor
    guard.update(guard_change)
    with pytest.raises(host.CoordinationError, match="invalid managed process guard"):
        host.handle_request({**request, "prepared_supervisor": prepared, "process_guard": guard},
                            clock=lambda: clock[0])
    with coordinator._transaction() as connection:
        assert coordinator._task_row(connection, "cpu-task")["state"] == "starting"
    busy.assert_not_called()


def test_backend_sends_one_fixed_code_rpc_without_pid_shell(monkeypatch):
    python = Mock(return_value={"status": "active", "task": {"state": "active", "pid": 810}})
    monkeypatch.setattr("remote_dev.core.ssh_transport.run_remote_python", python)
    backend = RemoteBackend()
    backend.bash = Mock(side_effect=AssertionError("separate PID shell must not run"))
    runtime = {"host_endpoint": {"host": "host.invalid", "port": 22, "user": "root"},
               "container_name": "mindie-test", "attestation": {"container_id": "container-one"}}
    receipt = {"pid": 42}
    request = {"action": "activate", "task_id": "cpu-task", "fence_token": 7, "coordination_epoch": "epoch-one"}
    result = backend.activate_prepared(runtime, request, receipt)
    assert result["task"]["pid"] == 810
    python.assert_called_once()
    sent = python.call_args.args[2]
    assert all(sent[key] == value for key, value in request.items())
    assert sent["prepared_supervisor"] == {"container_name": "mindie-test", "container_id": "container-one", "receipt": receipt}
    backend.bash.assert_not_called()


@pytest.fixture
def pool_case():
    case = fixtures.PoolTests()
    case.setUp()
    try:
        yield case
    finally:
        case.tearDown()


def test_first_run_skips_status_but_resume_without_saved_receipt_observes(pool_case):
    case = pool_case
    binding = case.bind("alice", case.root / "source")
    case.backend.calls.clear()
    job = case.managed("alice", binding)
    assert job["state"] == "running"
    assert ("job", "status") not in case.backend.calls
    with case.pool.transaction() as db:
        row = case.pool.get(db, "job", job["id"])
        row.pop("remote", None)
        row.pop("had_receipt", None)
        case.pool.put(db, "job", row)
    case.backend.calls.clear()
    restarted = fixtures.RuntimePool(case.root / "manager", case.backend)
    assert restarted.managed_advance(job["id"])["state"] == "running"
    assert ("job", "status") in case.backend.calls
    assert ("job", "prepare") not in case.backend.calls


def test_lost_prepare_reply_probes_before_reusing_same_supervisor(pool_case):
    case = pool_case
    binding = case.bind("alice", case.root / "source")
    case.backend.fail_job_after = "prepare"
    job = case.managed("alice", binding)
    assert job["state"] == "uncertain"
    assert not (job.get("remote") or {}).get("receipt")
    case.backend.calls.clear()
    restarted = fixtures.RuntimePool(case.root / "manager", case.backend)
    recovered = restarted.managed_advance(job["id"])
    assert recovered["state"] == "running"
    assert case.backend.calls.index(("job", "status")) < case.backend.calls.index(("job", "prepare"))
    assert list(case.backend.jobs) == [job["job_id"]]


def test_combined_activation_lost_reply_reconciles_without_replaying(pool_case):
    case = pool_case
    binding = case.bind("alice", case.root / "source")
    calls = []

    def activate(runtime, request, receipt):
        calls.append(copy.deepcopy(request))
        return case.backend.host(runtime, {**request, "pid": receipt["pid"]})

    case.backend.activate_prepared = activate
    case.backend.job_host_pid = Mock(side_effect=AssertionError("combined path must not separately resolve PID"))
    case.backend.fail_after = "activate"
    job = case.managed("alice", binding)
    assert job["state"] == "uncertain"
    assert ("job", "go") not in case.backend.calls
    restarted = fixtures.RuntimePool(case.root / "manager", case.backend)
    recovered = restarted.managed_advance(job["id"])
    assert recovered["state"] == "running"
    assert len(calls) == 1
    assert case.backend.calls.count(("job", "prepare")) == 1
    assert case.backend.calls.count(("job", "go")) == 1
    case.backend.job_host_pid.assert_not_called()


def test_cancel_after_lost_activation_stops_without_go_and_preserves_peer(pool_case):
    case = pool_case
    first_binding = case.bind("alice", case.root / "first")
    peer_binding = case.bind("bob", case.root / "peer")
    peer = case.managed("bob", peer_binding, 1)
    case.backend.activate_prepared = lambda runtime, request, receipt: case.backend.host(
        runtime, {**request, "pid": receipt["pid"]})
    case.backend.fail_after = "activate"
    job = case.managed("alice", first_binding)
    assert job["state"] == "uncertain"
    case.backend.calls.clear()
    case.pool.managed_control("alice", job["id"], "stop")
    ended = case.pool.managed_control("alice", job["id"])
    assert ended["state"] == "cancelled"
    assert ("job", "go") not in case.backend.calls
    assert case.backend.jobs[peer["job_id"]]["state"] == "running"
