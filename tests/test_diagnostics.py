"""Read-only scope, correlation, clocks and logging-failure behavior."""
import contextlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from mindie_diagnostics import configure, current_context, bind_context
from remote_dev.core.errors import RemoteExecutionError, error_details
from mindie_coordinator import ops, service, task_server
from mindie_coordinator.ready_runtime import RuntimePool
from test_agent_wait_evidence import owned


@pytest.mark.parametrize("name", list(ops.TOOL_SCHEMAS))
def test_all_task_tools_have_entry_identity_even_on_validation_error(tmp_path, name):
    recorder = configure("mindie-coordinator", root=tmp_path)
    result = ops.mindie_call(name, {"invalid_field": True})["result"]
    records = [json.loads(line) for line in Path(recorder.record_ref).read_text(encoding="utf-8").splitlines()]
    start = next(row for row in records if row["event"] == "operation.start")
    assert result["invocation_id"] == start["operation_id"] == result["diagnostics"]["operation_id"]
    assert result["error_details"]["category"] == "caller"
    assert records[-1]["status"] == "error"


def test_owned_bundle_reads_only_selected_records_without_backend(owned, monkeypatch):
    owner, store, client, row = owned
    configure("mindie-coordinator", root=store.state_dir / "diagnostic-log")
    row["roles"] = [{"name": "default", "managed_job": "ours", "binding": {}}]
    with configure("mindie-coordinator", root=store.state_dir / "diagnostic-log").operation("fixture"):
        row["diagnostics_context"] = current_context()
        with owner.pool.transaction() as db:
            for key, principal in [("ours", "alice"), ("unrelated", "bob")]:
                owner.pool.put(db, "run", {"id": key, "owner": principal})
                owner.pool.event(db, principal, "run-operation", run=key, operation="release", elapsed_seconds=.123)
    owner._save_execution(store, row)
    owner.pool.backend = Mock(side_effect=AssertionError("diagnostics must not contact a host"))
    monkeypatch.setattr(owner.pool, "status", Mock(side_effect=AssertionError("no global status")), raising=False)
    reply = client.observe(row["id"], action="evidence", section="diagnostics")
    evidence = reply["evidence"]
    assert [item["run"] for item in evidence["diagnostics"]["operations"]] == ["ours"]
    bundle = json.loads(Path(evidence["support_bundle"]["record_ref"]).read_text(encoding="utf-8"))
    assert bundle["summary"]["event_count"] == 2
    assert "unrelated" not in json.dumps(bundle) and "alice" not in json.dumps(bundle)
    assert any(event["attributes"].get("duration_ms") == 123 for event in bundle["events"])
    with pytest.raises(PermissionError):
        owner.pool.diagnostic_events("alice", ["unrelated"])
    owner.pool.status.assert_not_called()


def test_stage_duration_uses_monotonic_and_restart_has_explicit_gap(owned, monkeypatch):
    owner, store, client, row = owned
    clock = SimpleNamespace(time=lambda: 1000, monotonic=lambda: 10)
    monkeypatch.setattr(service, "time", clock)
    owner._save_progress(store, row, None, {"step": "first"})
    clock.time = lambda: 1  # wall clock moved backwards
    clock.monotonic = lambda: 12.5
    owner._save_progress(store, row, None, {"step": "second"})
    assert row["stage_history"][-1]["elapsed_seconds"] == 2.5
    monkeypatch.setattr(service, "_PROCESS_CLOCK_DOMAIN", "b" * 32)
    owner._save_progress(store, row, None, {"step": "third"})
    assert row["stage_history"][-1]["elapsed_seconds"] is None
    assert row["stage_history"][-1]["timing_incomplete"]


def test_known_cache_failure_is_terminal_failed_not_uncertain(owned, monkeypatch):
    from mindie_coordinator.managed_execution import ExecutionRequestError
    from mindie_coordinator.service import DONE
    owner, store, _, row = owned
    row["preparation_jobs"] = {"worker": {"j1": {
        "quiet": False, "job_id": "j1", "step": "shared-native-restore",
        "endpoint": {"host": "host", "port": 22, "user": "root"}}}}
    owner._save_execution(store, row)
    stopped = []

    def stop_jobs(store, row, force):
        stopped.append((force, list((row.get("preparation_jobs") or {}).get("worker") or {})))
        return True

    monkeypatch.setattr(owner, "_stop_preparation_jobs", stop_jobs)
    reply = owner._record_execution_error(
        store, row, ExecutionRequestError("shared native cache restore failed: OSError: No space left on device"))
    assert reply["state"] == "failed"
    assert row["phase"] == "failed"
    assert row["phase"] in DONE
    assert stopped == [(False, ["j1"])]


def test_unconfirmed_stop_retries_reclaim_without_demoting_failed(owned, monkeypatch):
    from mindie_coordinator.managed_execution import ExecutionRequestError
    owner, store, _, row = owned
    row["preparation_jobs"] = {"worker": {"j1": {
        "quiet": False, "job_id": "j1", "step": "shared-native-restore",
        "endpoint": {"host": "host", "port": 22, "user": "root"}}}}
    owner._save_execution(store, row)
    stopped = []

    def stop_jobs(store, row, force):
        stopped.append(force)
        return False

    monkeypatch.setattr(owner, "_stop_preparation_jobs", stop_jobs)
    reply = owner._record_execution_error(
        store, row, ExecutionRequestError("shared native cache restore failed: OSError: No space left on device"))
    assert reply["state"] == "failed"
    assert reply["resources_released"] is False
    assert stopped == [False]
    reply = owner._advance_locked(str(store.state_dir), "alice", row["id"])
    assert reply["state"] == "failed"
    assert stopped == [False, False]
    reply = owner._observe(str(store.state_dir), "alice", row["id"], action="status", refresh=True)
    assert reply["state"] == "failed"
    assert stopped == [False, False, False]


def test_reconcile_retries_unconfirmed_stop_on_terminal_failed(owned, monkeypatch):
    from mindie_coordinator.managed_execution import ExecutionRequestError
    owner, store, _, row = owned
    row["preparation_jobs"] = {"worker": {"j1": {
        "quiet": False, "job_id": "j1", "step": "shared-native-restore",
        "endpoint": {"host": "host", "port": 22, "user": "root"}}}}
    owner._save_execution(store, row)
    stopped = []

    def stop_jobs(store, row, force):
        stopped.append(force)
        return False

    monkeypatch.setattr(owner, "_stop_preparation_jobs", stop_jobs)
    owner._record_execution_error(
        store, row, ExecutionRequestError("shared native cache restore failed"))
    assert stopped == [False]
    stopped.clear()
    owner.reconcile()
    assert stopped == [False]
    with store.transaction() as db:
        latest = store.get(db, "execution", row["id"])
    assert latest["phase"] == "failed"


def test_explicit_stop_does_not_demote_failed_to_uncertain(owned, monkeypatch):
    from mindie_coordinator.managed_execution import ExecutionRequestError
    owner, store, _, row = owned
    row["preparation_jobs"] = {"worker": {"j1": {
        "quiet": False, "job_id": "j1", "step": "shared-native-restore",
        "endpoint": {"host": "host", "port": 22, "user": "root"}}}}
    owner._save_execution(store, row)
    monkeypatch.setattr(owner, "_stop_preparation_jobs", lambda *a, **k: False)
    owner._record_execution_error(
        store, row, ExecutionRequestError("shared native cache restore failed"))
    reply = owner._request_stop(str(store.state_dir), "alice", row["id"], False)
    assert reply["state"] == "failed"
    assert reply["resources_released"] is False
    with store.transaction() as db:
        row = store.get(db, "execution", row["id"])
    row["roles"] = [{"name": "default", "managed_job": "job", "binding": {},
                     "observation": {"state": "uncertain", "quiet": False}}]
    owner._save_execution(store, row)
    monkeypatch.setattr(owner.pool, "managed_control",
                        lambda *a, **k: {"state": "uncertain", "quiet": False, "lease_state": "active"})
    reply = owner._observe(str(store.state_dir), "alice", row["id"], action="status", refresh=True)
    assert reply["state"] == "failed"


def test_finish_stays_finishing_while_failed_jobs_are_unreleased(owned, monkeypatch):
    from mindie_coordinator.managed_execution import ExecutionRequestError
    owner, store, _, row = owned
    row["preparation_jobs"] = {"worker": {"j1": {
        "quiet": False, "job_id": "j1", "step": "shared-native-restore",
        "endpoint": {"host": "host", "port": 22, "user": "root"}}}}
    owner._save_execution(store, row)
    monkeypatch.setattr(owner, "_stop_preparation_jobs", lambda *a, **k: False)
    owner._record_execution_error(
        store, row, ExecutionRequestError("shared native cache restore failed"))
    result = owner.finish(str(store.state_dir), "alice", row["session_id"])
    assert result["state"] == "finishing"
    with store.transaction() as db:
        session = store.get(db, "session", row["session_id"])
        latest = store.get(db, "execution", row["id"])
    assert session["state"] == "finishing"
    assert latest["phase"] == "failed"


def test_failed_phase_survives_uncertain_job_observation(owned, monkeypatch):
    from mindie_coordinator.managed_execution import ExecutionRequestError
    owner, store, _, row = owned
    row["preparation_jobs"] = {"worker": {"j1": {
        "quiet": False, "job_id": "j1", "step": "shared-native-restore",
        "endpoint": {"host": "host", "port": 22, "user": "root"}}}}
    owner._save_execution(store, row)
    monkeypatch.setattr(owner, "_stop_preparation_jobs", lambda *a, **k: False)
    owner._record_execution_error(
        store, row, ExecutionRequestError("shared native cache restore failed"))
    row["roles"] = [{"name": "default", "managed_job": "job", "binding": {},
                     "observation": {"state": "uncertain", "quiet": False}}]
    owner._save_execution(store, row)
    monkeypatch.setattr(owner.pool, "managed_control",
                        lambda *a, **k: {"state": "uncertain", "quiet": False, "lease_state": "active"})
    reply = owner._advance_locked(str(store.state_dir), "alice", row["id"])
    assert reply["state"] == "failed"


def test_task_root_busy_waits_instead_of_permanent_failed(owned):
    from mindie_coordinator.provision.task_environment import TaskRootBusy
    from mindie_coordinator.service import DONE, PERMANENT_ERRORS
    owner, store, _, row = owned
    assert TaskRootBusy not in PERMANENT_ERRORS
    reply = owner._record_execution_error(store, row, TaskRootBusy("execution root is busy"))
    assert reply["state"] == "waiting"
    assert row["phase"] == "waiting"
    assert row["phase"] not in DONE
    assert "TaskRootBusy" in reply["error"]


def test_diagnostic_disk_failure_preserves_error_and_authoritative_failure(owned, monkeypatch):
    owner, store, client, row = owned
    blocked = owner.state_dir / "runs"
    blocked.write_text("not a directory")
    result = owner._record_execution_error(store, row, RemoteExecutionError("lost reply", submission_state="uncertain"))
    assert result["state"] == "uncertain"
    assert result["error_details"]["submission_state"] == "uncertain"
    monkeypatch.setattr(store, "save_execution", Mock(side_effect=OSError("authoritative state disk full")))
    with pytest.raises(OSError, match="authoritative"):
        owner._save_execution(store, row)


def test_projection_write_failure_does_not_replace_business_exception(owned, monkeypatch):
    owner, _, _, _ = owned
    @contextlib.contextmanager
    def failed_transaction():
        raise OSError("diagnostic event write failed")
        yield
    monkeypatch.setattr(owner.pool, "transaction", failed_transaction)
    with owner.pool._run_operation({"id": "ours", "owner": "alice"}, "release"):
        released = True
    assert released
    with pytest.raises(ValueError, match="original business failure"):
        with owner.pool._run_operation({"id": "ours", "owner": "alice"}, "prepare"):
            raise ValueError("original business failure")


def test_background_failure_has_separate_lifetime_from_completed_admission(owned, monkeypatch):
    from remote_dev.observability import observed_tool, observed_operation, current_tool
    from mindie_diagnostics import wrap_context
    owner, store, _, row = owned
    recorder = configure("mindie-coordinator", root=store.state_dir / "logs")
    gate = threading.Event()
    @observed_operation("background.prepare", component="mindie-coordinator")
    def advance(*args):
        assert current_tool() is None
        raise RuntimeError("background preparation failure")
    monkeypatch.setattr(owner, "_advance_locked", advance)
    threads = []
    @observed_tool("mindie.run", component="mindie-coordinator")
    def admit():
        row["diagnostics_context"] = current_context()
        def later():
            assert gate.wait(2)
            owner._tick_one(str(store.state_dir), row, threading.Lock())
        thread = threading.Thread(target=wrap_context(later))
        threads.append(thread)
        thread.start()
        return {"state": "queued", "execution_id": row["id"]}
    reply = admit()
    gate.set()
    threads[0].join(3)
    assert not threads[0].is_alive()
    assert reply["diagnostics"]["status"] == "success"
    records = [json.loads(line) for line in Path(recorder.record_ref).read_text().splitlines()]
    finish = next(item for item in records if item["event"] == "operation.end" and item["operation"] == "mindie.run")
    failure = next(item for item in records if item["event"] == "operation.end" and item["operation"] == "background.prepare")
    assert finish["status"] == "success" and failure["status"] == "error"
    assert finish["trace_id"] == failure["trace_id"]
    assert finish["operation_id"] != failure["operation_id"]


@pytest.mark.parametrize("connected", [False, True])
def test_ipc_disconnect_certainty_and_no_automatic_replay(tmp_path, monkeypatch, connected):
    marker = service.socket_path(tmp_path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"port": 12345, "token": "opaque"}))
    conn = Mock()
    if connected:
        conn.recv.return_value = b""
    else:
        conn.connect.side_effect = ConnectionRefusedError()
    monkeypatch.setattr(service.socket, "socket", Mock(return_value=conn))
    with pytest.raises(RemoteExecutionError) as caught:
        service.CoordinatorClient(tmp_path).call("admit", session_id="same-session")
    details = error_details(caught.value)
    assert details["submission_state"] == ("uncertain" if connected else "not_sent")
    assert details["retryable"] is (not connected)
    assert conn.connect.call_count == 1
    assert conn.sendall.call_count == int(connected)
