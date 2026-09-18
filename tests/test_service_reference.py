from unittest.mock import Mock

import pytest

from mindie_coordinator.agent_session import AgentSessions
from mindie_coordinator.task_client import TaskClient


@pytest.fixture
def clients(tmp_path):
    store = AgentSessions(tmp_path / "sessions")
    contexts = [store.attach(client="codex", native_session_id=name, cwd=str(tmp_path))
                for name in ("first", "second")]
    owner = Mock()
    owner.advance.side_effect = lambda _store, _user, eid, **kw: {"execution_id": eid, "state": "running"}
    return store, [TaskClient(ctx["context_file"], service=owner) for ctx in contexts], owner


def add(store, client, request, phase="running", created=1):
    row = store.execution(client.context, request, {"service": "model"})
    row.update(phase=phase, created_at=created)
    store.save_execution(row)
    return row["id"]


def test_reference_is_scoped_and_missing_does_not_contact_runtime(clients):
    store, (first, second), owner = clients
    eid = add(store, first, "one")
    assert second.observe(service="model", action="stop")["state"] == "not_found"
    owner.advance.assert_not_called()
    with pytest.raises(ValueError, match="another MindIE task"):
        second.observe(eid, "stop")
    assert first.observe(service="model")["execution_id"] == eid


def test_live_wins_over_newer_terminal_and_stop_uses_selected_execution(clients):
    store, (first, _), owner = clients
    live = add(store, first, "live")
    add(store, first, "terminal", "cancelled", created=5)
    assert first.observe(service="model", action="stop", force=True)["execution_id"] == live
    assert owner.advance.call_args.kwargs["force"] is True
    assert owner.advance.call_args.kwargs["action"] == "stop"


def test_latest_terminal_and_ambiguous_live_are_deterministic(clients):
    store, (first, _), owner = clients
    add(store, first, "old", "failed")
    latest = add(store, first, "new", "cancelled", created=2)
    assert first.resolve_execution(service="model") == latest
    add(store, first, "live-one")
    add(store, first, "live-two")
    with pytest.raises(ValueError, match="multiple live"):
        first.observe(service="model", action="stop")
    owner.advance.assert_not_called()


def test_missing_or_conflicting_reference_is_rejected(clients):
    store, (first, _), owner = clients
    eid = add(store, first, "one")
    for kwargs in ({}, {"execution_id": eid, "service": "model"}):
        with pytest.raises(ValueError, match="exactly one"):
            first.observe(**kwargs)
    owner.advance.assert_not_called()


def test_release_wait_does_not_treat_terminal_but_leased_as_released(clients):
    store, (first, _), owner = clients
    eid = add(store, first, "one")
    owner.wait.return_value = {"execution_id": eid, "state": "cancelled", "resources_released": False, "wait_timed_out": True}
    assert first.wait(eid, until="released", timeout_seconds=0)["wait_timed_out"] is True
    owner.wait.return_value = {"execution_id": eid, "state": "cancelled", "resources_released": True}
    assert "wait_timed_out" not in first.wait(eid, until="released", timeout_seconds=0)


def test_running_wait_returns_failure_without_waiting_or_relaunching(clients):
    store, (first, _), owner = clients
    eid = add(store, first, "one")
    owner.wait.return_value = {"execution_id": eid, "state": "failed"}
    assert first.wait(eid)["state"] == "failed"
    owner.admit.assert_not_called()
    owner.advance.assert_not_called()
