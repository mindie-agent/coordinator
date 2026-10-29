"""Two-stage preparation uses the real host CPU queue before the NPU run."""
import pytest

pytestmark = pytest.mark.usefixtures("host_compile_capacity")
import copy


from mindie_coordinator.compile_lease import CompileLease, finish_compile_lease
from mindie_coordinator.host import mindie_npu_coordination as host_module
from mindie_coordinator.preparation_process import PreparationCancelled, PreparationProcess


ENDPOINT = {"host": "example.invalid", "port": 22, "user": "root", "root": "/tmp/test", "cwd": "/tmp/test"}


def host_request(tmp_path, monkeypatch):
    monkeypatch.setenv("MINDIE_COORDINATOR_MAX_COMPILE_SLOTS", "1")
    host = host_module.NpuCoordinator(tmp_path)
    actions = []
    def request(endpoint, payload):
        actions.append(payload["action"])
        if payload["action"] == "startup-context":
            return {**host.snapshot(None), "container": {"Id": "container-id", "State": {"Running": True}}}
        return host_module.handle_request({**payload, "state_dir": str(tmp_path)},
            probe=lambda: pytest.fail("CPU preparation must never probe NPU devices"),
            listening_ports=lambda: pytest.fail("CPU preparation must never reserve service ports"))
    return host, request, actions


def test_cancel_waiting_compile_keeps_other_compiler_and_never_launches(tmp_path, monkeypatch):
    host, request, actions = host_request(tmp_path, monkeypatch)
    host.submit({"task_id": "other-compiler", "agent_id": "other-owner", "npu_count": 0})
    host.acquire("other-compiler", None)
    lease = CompileLease(ENDPOINT, "mindie-test", request)
    record = {"job_id": "prepare-cancelled", "state": "pending", "quiet": False}
    saved = []
    with pytest.raises(PreparationCancelled, match="CPU compilation capacity"):
        lease.acquire(record, lambda value: saved.append(copy.deepcopy(value)), lambda: True)
    states = {row["task_id"]: row["state"] for row in host.snapshot(None)["tasks"]}
    assert states == {"other-compiler": "granted", "prepare-cancelled": "cancelled"}
    assert saved[-1]["quiet"] and saved[-1]["compile_lease"]["state"] == "cancelled"
    assert "activate" not in actions


def test_compiler_start_gate_is_fenced_and_completion_releases_cpu_capacity(tmp_path, monkeypatch):
    host, request, actions = host_request(tmp_path, monkeypatch)
    alive = [True]
    monkeypatch.setattr(host_module, "process_guard_busy",
        lambda value, *, completion_confirmed=False: bool(value) and (alive[0] or not completion_confirmed))
    monkeypatch.setattr(host_module, "prepared_supervisor_host_pid", lambda prepared, guard: prepared["receipt"]["pid"])
    saved, output, remote, cleaned = [], [], [], []
    receipt = {"pid": 4242, "boot_id": "test-boot", "start_time": 1,
               "process_guard": {"marker": "a" * 32, "boot_id": "test-boot", "retain_until_release": True}}
    def control(endpoint, job_id, action, **parameters):
        remote.append(action)
        if action == "prepare":
            assert saved[-1]["compile_lease"]["state"] == "starting"
            assert parameters["spec"]["env"]["ASCEND_RT_VISIBLE_DEVICES"] == ""
            assert parameters["spec"]["env"]["MINDIE_SLOT_ID"] == job_id
            assert saved[-1]["compile_lease"]["compile_budget"]["jobs_per_slot"] == 16
            for name in ("MAX_JOBS", "MINDIE_BUILD_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL", "MINDIE_COMPILE_JOBS_LIMIT"):
                assert parameters["spec"]["env"][name] == "2"
            return {"state": "prepared", "quiet": False, "receipt": receipt}
        if action == "go":
            task = host.snapshot(None, task_id=job_id)["tasks"][0]
            assert task["state"] == "active" and parameters["authorization"]["fence"] == task["fence_token"]
            return {"state": "running", "quiet": False, "receipt": receipt}
        assert action == "exchange"
        alive[0] = False
        return {"state": "succeeded", "quiet": True, "receipt": receipt, "stdout": "compiled\n",
                "result": {"exit_code": 0}, "stdout_offset": 9, "stderr_offset": 0}
    monkeypatch.setattr("mindie_coordinator.preparation_process.control", control)
    result = PreparationProcess(ENDPOINT, "install-vllm-ascend",
        lambda record: saved.append(copy.deepcopy(record)), lambda: False,
        env={"MAX_JOBS": "2", "CMAKE_BUILD_PARALLEL_LEVEL": "99"},
        compile_lease=CompileLease(ENDPOINT, "mindie-test", request,
            cleanup=lambda endpoint, slot: cleaned.append((endpoint, slot)))).run("compile", on_output=lambda *item: output.append(item))
    assert result.returncode == 0 and output == [("stdout", "compiled\n")]
    assert saved[-1]["compile_lease"]["state"] == "released"
    assert saved[-1]["slot_cleanup"]["state"] == "completed"
    assert cleaned == [(ENDPOINT, saved[-1]["job_id"])]
    assert remote == ["prepare", "go", "exchange"]
    assert actions.index("submit-acquire-preflight") < actions.index("activate") < actions.index("release")
    host.submit({"task_id": "next-compiler", "agent_id": "next-owner", "npu_count": 0})
    assert host.acquire("next-compiler", None)["status"] == "granted"


def test_release_failure_preserves_compiler_outcome_for_later_reconciliation(tmp_path, monkeypatch):
    host, request, actions = host_request(tmp_path, monkeypatch)
    host.submit({"task_id": "prepare-finished", "agent_id": "owner", "npu_count": 0})
    host.acquire("prepare-finished", None)
    record = {"job_id": "prepare-finished", "slot_id": "prepare-finished", "endpoint": ENDPOINT,
        "state": "succeeded", "quiet": True, "result": {"exit_code": 0},
        "compile_lease": {"task_id": "prepare-finished", "state": "granted", "host_endpoint": ENDPOINT,
                          "coordination_epoch": host.snapshot(None)["coordination_epoch"]}}
    def unavailable(*args):
        raise TimeoutError("host reply unavailable")
    assert not finish_compile_lease(record, lambda value: None, request=unavailable)
    assert record["state"] == "succeeded" and record["result"]["exit_code"] == 0
    assert record["compile_lease"]["cleanup_error"] == "host reply unavailable"
    assert finish_compile_lease(record, lambda value: None, request=request, cleanup=unavailable)
    assert record["compile_lease"]["state"] == "cancelled"
    assert record["slot_cleanup"]["state"] == "failed" and record["state"] == "succeeded"
    cleaned = []
    assert finish_compile_lease(record, lambda value: None, request=unavailable,
        cleanup=lambda endpoint, slot: cleaned.append(slot))
    assert cleaned == ["prepare-finished"] and record["slot_cleanup"]["state"] == "completed"


def test_legacy_host_explicitly_reports_missing_compile_budget(tmp_path, monkeypatch):
    host, request, _ = host_request(tmp_path, monkeypatch)
    def legacy(endpoint, payload):
        reply = request(endpoint, payload)
        if reply.get('task'):
            reply['task'].pop('compile_budget', None)
        return reply
    record = {'job_id': 'legacy-compiler', 'state': 'pending', 'quiet': False}
    lease = CompileLease(ENDPOINT, 'mindie-test', legacy)
    lease.acquire(record, lambda _: None, lambda: False)
    assert record['compile_lease']['compile_budget'] == {
        'status': 'unavailable', 'reason': 'host did not provide a per-slot compile budget'}
    assert host.snapshot(None, task_id='legacy-compiler')['tasks'][0]['state'] == 'starting'
