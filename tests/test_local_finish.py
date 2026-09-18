"""Local finish and managed admission share one atomic ownership boundary."""
import os
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from mindie_coordinator.agent_session import AgentSessions
from mindie_coordinator.task_client import TaskClient


def task(tmp_path, **kwargs):
    store = AgentSessions(tmp_path / "sessions")
    context = store.attach("codex", "local-task", str(tmp_path))
    return store, context, TaskClient(context["context_file"], user="user", **kwargs)


def test_empty_finish_has_no_service_or_remote_transport_import_in_fresh_process(tmp_path):
    script = '''import importlib.abc,json,sys
from pathlib import Path
class BlockManagedImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        pure_diagnostics = {"remote_dev", "remote_dev.observability", "remote_dev.core", "remote_dev.core.errors"}
        if fullname == "mindie_coordinator.service" or (fullname.startswith("remote_dev.") and fullname not in pure_diagnostics):
            raise AssertionError("unmanaged finish imported " + fullname)
sys.meta_path.insert(0, BlockManagedImports())
from mindie_coordinator.agent_session import AgentSessions
from mindie_coordinator.task_client import TaskClient
store=AgentSessions(Path(sys.argv[1]) / "sessions")
context=store.attach("codex", "fresh-local", sys.argv[1])
client=TaskClient(context["context_file"], user="user")
for kwargs in ({"command": " "}, {"command": " ", "wait_until": "released"}):
    try:
        client.run(**kwargs)
    except ValueError as exc:
        assert str(exc) == "command is required", exc
    else:
        raise AssertionError("empty command was accepted")
try:
    client.wait("invalid-id", until="released")
except ValueError as exc:
    assert str(exc) == "invalid local execution id", exc
else:
    raise AssertionError("invalid execution was accepted")
reply=client.finish(force=True)
business = lambda result: {key: value for key, value in result.items() if key != "diagnostics"}
assert business(reply) == {"state":"finished", "executions":[], "worktrees_preserved":True}, reply
again = client.finish()
assert business(again) == business(reply)
assert reply["diagnostics"]["operation_id"] != again["diagnostics"]["operation_id"]
assert all(result["diagnostics"]["status"] == "success" and result["diagnostics"]["finished_at"]
           for result in (reply, again))
assert client._service is None
assert not (Path(sys.argv[1]) / "coordinator").exists()
assert client.status()["session"]["state"] == "finished"
pure_diagnostics = {"remote_dev", "remote_dev.observability", "remote_dev.core", "remote_dev.core.errors"}
assert {name for name in sys.modules if name == "remote_dev" or name.startswith("remote_dev.")} <= pure_diagnostics
assert "mindie_coordinator.service" not in sys.modules
print(json.dumps(reply))
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)],
                            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                                 "MINDIE_DIAGNOSTICS_ROOT": str(tmp_path / "diagnostics"),
                                 "MINDIE_COORDINATOR_STATE_DIR": str(tmp_path / "coordinator")},
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_planned_work_finishes_locally_but_any_managed_history_delegates(tmp_path):
    owner = Mock()
    store, context, client = task(tmp_path, service=owner)
    planned = store.execution(context, "draft", {"command": "true"})
    result = client.finish()
    assert result["executions"] == [{"execution_id": planned["id"], "state": "cancelled", "resources_released": True}]
    owner.finish.assert_not_called()
    context = store.attach("codex", "local-task", str(tmp_path))
    admitted = store.admit_execution(context["session"]["id"], "managed", {"command": "true"}, user="user")
    admitted["phase"] = "succeeded"
    store.save_execution(admitted)
    client.finish(force=True)
    owner.finish.assert_called_once_with(str(store.state_dir), "user", context["session"]["id"], force=True)


def test_legacy_remote_facts_do_not_use_local_finish(tmp_path):
    owner = Mock()
    store, context, client = task(tmp_path, service=owner)
    record = store.execution(context, "legacy", {"command": "true"})
    record["preparation_jobs"] = {"worker": {"job": {"quiet": False}}}
    store.save_execution(record)
    client.finish()
    owner.finish.assert_called_once()


@pytest.mark.parametrize("first", ["finish", "admit"])
def test_finish_and_admission_are_serialized_by_the_database(tmp_path, monkeypatch, first):
    store, context, client = task(tmp_path)
    session_id = context["session"]["id"]
    persisted, release, second_started = threading.Event(), threading.Event(), threading.Event()
    outcome = {}
    original_put = store.put
    def paused_put(db, kind, value):
        original_put(db, kind, value)
        boundary = kind == "session" and value.get("state") == "finished" if first == "finish" else kind == "execution" and value.get("admitted")
        if boundary:
            persisted.set()
            assert release.wait(5)
    monkeypatch.setattr(store, "put", paused_put)
    def finish():
        outcome["finish"] = store.close_if_unmanaged(session_id, user="user")
    def admit():
        try:
            outcome["admit"] = store.admit_execution(session_id, "race", {"command": "true"}, user="user")
        except ValueError as exc:
            outcome["admit_error"] = str(exc)
    actions = {"finish": finish, "admit": admit}
    other = "admit" if first == "finish" else "finish"
    def second():
        second_started.set()
        actions[other]()
    threads = [threading.Thread(target=actions[first]), threading.Thread(target=second)]
    try:
        threads[0].start()
        assert persisted.wait(3)
        threads[1].start()
        assert second_started.wait(3)
        assert other not in outcome
    finally:
        release.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(5)
    assert all(not thread.is_alive() for thread in threads)
    if first == "finish":
        assert outcome["finish"]["state"] == "finished"
        assert "admission is closed" in outcome["admit_error"]
        assert store.executions(session_id) == []
    else:
        assert outcome["admit"]["admitted"] is True
        assert outcome["finish"] is None
        assert store.context(context["attachment"]["id"])["session"]["state"] == "open"


def test_finish_wins_after_service_initial_open_check_before_admission(tmp_path, monkeypatch):
    from mindie_coordinator.service import CoordinatorService
    store, context, client = task(tmp_path)
    service = CoordinatorService(tmp_path / "coordinator", pool=MagicMock(), sessions=store)
    checked, release = threading.Event(), threading.Event()
    errors = []
    def source_check(snapshot):
        checked.set()
        assert release.wait(5)
    monkeypatch.setattr("mindie_coordinator.service.validate_source_snapshot", source_check)
    def admit():
        try:
            service.admit(str(store.state_dir), "user", context["session"]["id"], {"source_snapshot": {}}, wait=False)
        except ValueError as exc:
            errors.append(str(exc))
    worker = threading.Thread(target=admit)
    try:
        worker.start()
        assert checked.wait(3)
        assert client.finish()["state"] == "finished"
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert len(errors) == 1 and "admission is closed" in errors[0]
    assert store.executions(context["session"]["id"]) == []


def test_admission_idempotency_checks_principal_spec_and_closed_state(tmp_path):
    store, context, _ = task(tmp_path)
    sid = context["session"]["id"]
    row = store.admit_execution(sid, "same", {"command": "true"}, user="user")
    assert store.admit_execution(sid, "same", {"command": "true"}, user="user") == row
    with pytest.raises(ValueError, match="different arguments"):
        store.admit_execution(sid, "same", {"command": "false"}, user="user")
    with pytest.raises(PermissionError, match="another principal"):
        store.admit_execution(sid, "same", {"command": "true"}, user="other")
    with store.transaction() as db:
        session = store.get(db, "session", sid)
        session["state"] = "finishing"
        store.put(db, "session", session)
    with pytest.raises(ValueError, match="admission is closed"):
        store.admit_execution(sid, "same", {"command": "true"}, user="user")
