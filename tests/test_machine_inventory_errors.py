"""Inventory errors reach real task receipts and remain recoverable."""
import json
from pathlib import Path

import pytest

from mindie_coordinator.agent_session import AgentSessions
from mindie_coordinator.machine_directory import MachineDirectory
from mindie_coordinator.ready_runtime import RuntimePool
from mindie_coordinator.task_client import TaskClient
from test_coordinator import Backend, runtime_spec


@pytest.fixture
def inventory_client(tmp_path):
    backend = Backend(tmp_path / "host")
    pool = RuntimePool(tmp_path / "pool", backend)
    sessions = AgentSessions(tmp_path / "sessions")
    context = sessions.attach("codex", "inventory-errors", str(tmp_path))
    client = TaskClient(context["context_file"], pool=pool, user="alice")
    path = tmp_path / "machines.json"
    backend.machines = MachineDirectory(path=path)
    return client, pool, backend, path


@pytest.mark.parametrize("document", ["{invalid-json", '{"machines": null}'])
def test_broken_inventory_reports_failure_and_repair_resumes_same_execution(inventory_client, document):
    client, pool, backend, path = inventory_client
    path.write_text(document)
    failed = client.run("true", sources={})
    assert failed["state"] == "uncertain"
    assert "machines.json" in failed["error"]
    assert backend.jobs == {}
    path.write_text(json.dumps({"machines": []}))
    client.coordinator.reconcile()
    waiting = client.observe(failed["execution_id"])
    assert waiting["state"] == "waiting_for_runtime"
    spec = runtime_spec(1, user="alice", recipe="rc")
    backend.mark_prepared(spec)
    pool.register("available-after-repair", spec)
    client.coordinator.reconcile()
    running = client.observe(failed["execution_id"])
    assert running["state"] == "running"
    assert running["execution_id"] == failed["execution_id"]
    assert len(backend.jobs) == 1
    client.observe(failed["execution_id"])
    assert len(backend.jobs) == 1
    assert client.observe(failed["execution_id"], "stop")["resources_released"]


@pytest.mark.parametrize("error", [PermissionError("denied"), OSError("read failed")])
def test_inventory_read_failure_is_not_an_empty_inventory(inventory_client, monkeypatch, error):
    client, _, backend, path = inventory_client
    path.write_text(json.dumps({"machines": []}))
    read_text = Path.read_text

    def fail_read(self, *args, **kwargs):
        if self == path:
            raise error
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_read)
    failed = client.run("true", sources={})
    assert failed["state"] == "uncertain"
    assert type(error).__name__ in failed["error"]
    assert "machines.json" in failed["error"]
    assert backend.jobs == {}


@pytest.mark.parametrize("configured", [False, True])
def test_optional_or_valid_empty_inventory_keeps_waiting_contract(inventory_client, configured):
    client, _, backend, path = inventory_client
    if configured:
        path.write_text(json.dumps({"machines": []}))
    else:
        backend.machines = None
    result = client.run("true", sources={})
    assert result["state"] == "waiting_for_runtime"
    assert backend.jobs == {}
