"""CPU admission and arbitrary source execution use the real host authority."""
import json
from pathlib import Path
import pytest

from mindie_coordinator.host.mindie_npu_coordination import NpuCoordinator, CoordinationError, handle_request
from mindie_coordinator.managed_execution import task_preamble
from mindie_coordinator.ready_runtime import RuntimePool


def test_cpu_command_bypasses_waiting_npu_job_without_claiming_a_device(tmp_path):
    host = NpuCoordinator(tmp_path)
    host.submit({"task_id": "blocked", "agent_id": "gpu", "npu_count": 1})
    host.submit({"task_id": "cpu", "agent_id": "cpu", "npu_count": 0, "service_port": 0})
    observed = {"status": "ok", "devices": [0], "busy": {"0": [{"kind": "external"}]}, "free": []}
    waiting = host.acquire("blocked", observed)
    assert waiting["status"] == "waiting"
    granted = host.acquire("cpu", observed, listening={"status": "ok", "ports": []})
    assert granted["task"]["state"] == "granted"
    assert granted["task"]["granted_devices"] == []
    assert granted["task"]["granted_service_port"] > 0
    start = host.preflight("cpu", granted["task"]["fence_token"], observed)
    assert start["status"] == "starting"
    assert start["environment"]["ASCEND_RT_VISIBLE_DEVICES"] == ""
    cancelled = host.cancel("cpu", observed, listening={"status": "ok", "ports": []})
    assert cancelled["task"]["state"] == "cancelled"


@pytest.mark.parametrize("value", [-1, True, 0.5, "1"])
def test_host_rejects_invalid_counts(tmp_path, value):
    with pytest.raises(CoordinationError, match="nonnegative integer"):
        NpuCoordinator(tmp_path).submit({"task_id": "invalid", "agent_id": "caller", "npu_count": value})


def test_source_free_execution_context_does_not_require_a_git_checkout(tmp_path):
    pool = RuntimePool(tmp_path, object())
    session = pool.session_open("user", "execution", {})
    assert session["sources"] == {}


def test_preamble_uses_only_bound_source_names():
    base = {"python": "/env/bin/python", "endpoint": {"cwd": "/runs/one"}}
    command = task_preamble({**base, "source_names": ["example", "other-code"]})
    assert "/runs/one/example:/runs/one/other-code" in command
    assert "vllm" not in command
    assert "PYTHONPATH" not in task_preamble({**base, "source_names": []})


def test_cpu_lifecycle_never_probes_npu_and_cannot_release_unknown_npu_lease(tmp_path, monkeypatch):
    host = NpuCoordinator(tmp_path)
    host.submit({"task_id": "npu", "agent_id": "npu", "npu_count": 1})
    observed = {"status": "ok", "devices": [0], "busy": {}, "free": [0]}
    npu = host.acquire("npu", observed)["task"]
    host.submit({"task_id": "cpu", "agent_id": "cpu", "npu_count": 0})
    def forbidden():
        raise AssertionError("CPU command must not inspect hardware or listening ports")
    def call(action, **values):
        return handle_request({"action": action, "task_id": "cpu", "state_dir": str(tmp_path), **values},
                              probe=forbidden, listening_ports=forbidden)
    granted=call("acquire")
    token=granted["task"]["fence_token"]
    assert granted["environment"]["ASCEND_RT_VISIBLE_DEVICES"] == ""
    assert granted["occupancy"] is None
    assert call("preflight", fence_token=token)["status"] == "starting"
    assert call("status")["occupancy"] is None
    monkeypatch.setattr("mindie_coordinator.host.mindie_npu_coordination.process_guard_busy", lambda *args, **kwargs: not kwargs.get("completion_confirmed",False))
    assert call("release", fence_token=token)["status"] == "orphaned_busy"
    assert call("release", fence_token=token, completion_confirmed=True)["status"] == "released"
    assert host.release("npu", npu["fence_token"], None, completion_confirmed=True)["status"] == "orphaned_busy"


def test_cpu_service_still_checks_live_port_before_release(tmp_path):
    host=NpuCoordinator(tmp_path)
    host.submit({"task_id":"service", "agent_id":"cpu", "npu_count":0, "service_port":0})
    def forbidden():raise AssertionError("no NPU request")
    def call(action, ports, **values):
        return handle_request({"action":action,"task_id":"service","state_dir":str(tmp_path),**values},probe=forbidden,
                              listening_ports=lambda:{"status":"ok","ports":ports})
    granted=call("acquire",[])["task"]
    token=granted["fence_token"];port=granted["granted_service_port"]
    assert call("release",[port],fence_token=token,completion_confirmed=True)["status"] == "orphaned_busy"
    assert call("release",[],fence_token=token,completion_confirmed=True)["status"] == "released"
