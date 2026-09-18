"""Selected code upgrades the idle owner; busy owners keep execution control."""
from pathlib import Path
from unittest.mock import Mock
import sys

import pytest

from mindie_coordinator import service
from mindie_coordinator.agent_session import AgentSessions

IPCClient = service.CoordinatorClient


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    state = tmp_path / "coordinator"
    owner = service.CoordinatorService(state)
    marker = state / "test-listener"
    marker.touch()
    current = {"package": "mindie-coordinator", "version": "0.4.1.dev1", "commit": "current-commit",
               "python": sys.executable, "location": "/current/site-packages", "pid": 999}
    loaded = {**current, "commit": "previous-commit", "pid": 100}
    calls, launches, restart_error = [], [], []
    monkeypatch.setattr(service, "LOADED_RUNTIMES", [current])
    monkeypatch.setattr(service, "socket_path", lambda _: marker)

    class Client:
        def __init__(self, state_dir):
            self.state_dir = Path(state_dir)

        def call(self, operation, **payload):
            calls.append(operation)
            if operation == "ping":
                if not marker.exists():
                    raise FileNotFoundError("stopped")
                return {"runtime": [{"loaded": dict(loaded)}]}
            if operation == "restart_if_idle":
                if restart_error:
                    raise restart_error[0]
                reply = owner.restart_if_idle()
                if reply["status"] == "stopping":
                    marker.unlink()
                return reply
            return owner.handle({"op": operation, **payload})["value"]

    def launch(command, **kwargs):
        assert not marker.exists(), "new daemon started before old listener closed"
        launches.append((command, kwargs))
        loaded.update({**current, "pid": loaded["pid"] + 1})
        owner._stopped.clear()
        marker.touch()
        return Mock(poll=lambda: None)

    monkeypatch.setattr(service, "CoordinatorClient", Client)
    monkeypatch.setattr(service.subprocess, "Popen", launch)
    return state, owner, current, loaded, calls, launches, restart_error


@pytest.mark.parametrize("difference", ["commit", "python", "location", "version"])
@pytest.mark.parametrize("package", ["mindie-coordinator", "remote-dev"])
def test_idle_different_daemon_restarts_with_selected_interpreter(daemon, difference, package):
    state, owner, current, loaded, calls, launches, *_ = daemon
    current["package"] = package
    loaded.update(current)
    loaded.update({difference: "other", "pid": 100})
    client = service.ensure_daemon(state)
    assert calls == ["ping", "ping", "restart_if_idle", "ping"]
    assert client.runtime[0]["loaded"] == {**current, "pid": 101}
    assert len(launches) == 1
    assert launches[0][0] == [sys.executable, "-m", "mindie_coordinator", "daemon", "--state-dir", str(state)]


def test_identical_code_different_pid_does_not_restart(daemon):
    state, owner, current, loaded, calls, launches, *_ = daemon
    loaded.update({**current, "pid": 100})
    assert service.ensure_daemon(state).runtime[0]["loaded"]["pid"] == 100
    assert calls == ["ping"]
    assert not launches


def test_busy_daemon_retries_immediately_after_work_finishes(daemon):
    state, owner, current, loaded, calls, launches, _ = daemon
    owner._active_requests = 1
    first = service.ensure_daemon(state)
    assert first.runtime[0]["loaded"]["pid"] == 100
    assert first.runtime_update["reason"] == "coordinator work is in progress"
    assert not owner._stopped.is_set() and not launches
    owner._active_requests = 0
    assert service.ensure_daemon(state).runtime[0]["loaded"]["commit"] == current["commit"]
    assert calls.count("restart_if_idle") == 2
    assert len(launches) == 1


def test_busy_new_admission_has_runtime_and_execution_facts_but_control_stays_with_owner(daemon, tmp_path, monkeypatch):
    state, owner, current, loaded, calls, launches, _ = daemon
    store = AgentSessions(tmp_path / "sessions")
    context = store.attach("claude", "native", str(tmp_path))
    owner.store(store.state_dir)
    execution = {"id": "e" * 64, "session_id": context["session"]["id"], "phase": "running", "admitted": True}
    with store.transaction() as db:
        store.put(db, "execution", execution)
    client = IPCClient(state)
    client.call = service.CoordinatorClient(state).call
    monkeypatch.setattr(owner, "admit", Mock(side_effect=AssertionError("old daemon must not admit new work")))
    before = store.all_executions()
    pending = client.admit(store.state_dir, "user", context["session"]["id"], {"command": "new"})
    assert pending["state"] == "needs_runtime_update"
    assert pending["active_executions"] == [{"execution_id": execution["id"], "session_id": execution["session_id"], "state": "running"}]
    assert pending["runtime_update"]["selected"] == [current]
    assert pending["runtime_update"]["daemon"][0]["loaded"] == loaded
    assert not launches and not owner._stopped.is_set()
    assert store.all_executions() == before
    owner.admit.assert_not_called()
    advance = Mock(side_effect=lambda *a, action, **kw: {"execution_id": execution["id"], "state": "running", "action": action})
    monkeypatch.setattr(owner, "advance", advance)
    for action in ("status", "tail", "stop"):
        reply = client.advance(store.state_dir, "user", execution["id"], action=action)
        assert reply["action"] == action
    assert advance.call_count == 3


def test_server_rechecks_selected_identity_before_admitting(daemon, monkeypatch):
    state, owner, current, loaded, calls, launches, _ = daemon
    admit = Mock(side_effect=AssertionError("mismatched input must not be admitted"))
    monkeypatch.setattr(owner, "admit", admit)
    for selected in ([], [loaded]):
        result = owner.handle({"op": "admit", "client_runtime": selected})["value"]
        assert result["state"] == "needs_runtime_update"
        assert result["runtime_update"]["selected"] == selected
    admit.assert_not_called()


def test_restart_failure_retains_raw_reason_and_never_admits(daemon):
    state, owner, current, loaded, calls, launches, restart_error = daemon
    restart_error.append(RuntimeError("idle restart unavailable"))
    client = IPCClient(state)
    client.call = service.CoordinatorClient(state).call
    result = client.admit("unused", "user", "task", {})
    assert result["state"] == "needs_runtime_update"
    assert "idle restart unavailable" in result["reason"]
    assert "admit" not in calls and not launches
