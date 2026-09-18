"""Bounded owner waits and recorded evidence; no remote resources or business readiness."""
import base64
import hashlib
import io
import json
import shutil
import subprocess
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from mindie_coordinator import service, task_server, mindie
from mindie_coordinator.agent_session import AgentSessions
from mindie_coordinator.execution_sources import capture_sources
from mindie_coordinator.runtime_profile import digest
from mindie_coordinator.task_client import TaskClient


@pytest.fixture
def owned(tmp_path, monkeypatch):
    store = AgentSessions(tmp_path / "sessions")
    context = store.attach(client="codex", native_session_id="wait-owner", cwd=str(tmp_path))
    owner = service.CoordinatorService(tmp_path / "coordinator", sessions=store)
    monkeypatch.setattr(owner, "_schedule_progress", Mock())
    client = TaskClient(context["context_file"], service=owner, user="alice")
    snapshot = capture_sources({}, store.state_dir)
    row = store.admit_execution(client.context["session"]["id"], "case", {"command": "true", "source_snapshot": snapshot}, user="alice")
    row.update(phase="preparing", roles=[])
    owner._save_execution(store, row)
    return owner, store, client, row


def test_wait_is_not_a_client_status_loop_and_rejects_other_task(owned):
    owner, store, client, row = owned
    reply = client.wait(row["id"], until="released", timeout_seconds=0)
    assert reply["execution_id"] == row["id"] and reply["wait_timed_out"] is True
    assert reply["resources_released"] is False
    assert store.executions(row["session_id"])[0]["id"] == row["id"]
    other = store.attach(client="codex", native_session_id="other", cwd=str(store.state_dir))
    other_client = TaskClient(other["context_file"], service=owner, user="alice")
    for action in ("wait", "evidence"):
        with pytest.raises(ValueError, match="another MindIE task"):
            other_client.observe(row["id"], action=action)
    with pytest.raises(PermissionError, match="another principal"):
        owner.wait(str(store.state_dir), "bob", row["id"], timeout_seconds=0)
    with pytest.raises(PermissionError, match="another principal"):
        owner.evidence(str(store.state_dir), "bob", row["id"])


def test_condition_wakes_for_running_without_holding_execution_lock(owned):
    owner, store, client, row = owned
    started = threading.Event()
    original = owner._observe
    def observe(*args, **kwargs):
        value = original(*args, **kwargs)
        started.set()
        return value
    owner._observe = observe
    with ThreadPoolExecutor() as workers:
        pending = workers.submit(client.wait, row["id"], until="running", timeout_seconds=3)
        assert started.wait(1)
        lock = owner._lock_for("execution", row["id"])
        assert lock.acquire(blocking=False)
        try:
            row["phase"] = "running"
            owner._save_execution(store, row)
        finally:
            lock.release()
        reply = pending.result(timeout=1)
    assert reply["state"] == "running" and "wait_timed_out" not in reply


def test_release_wait_keeps_cleanup_facts_and_reads_terminal_logs_once(owned):
    owner, store, client, row = owned
    row.update(phase="failed", roles=[{"name": "default", "binding": {}, "managed_job": "job",
                                      "observation": {"state": "failed", "lease_state": "active", "remote": {"quiet": False}}}])
    owner._save_execution(store, row)
    owner.pool.managed_control = Mock(return_value={"remote": {"stdout": "early output", "stderr": "early failure"}})
    first = client.wait(row["id"], until="released", timeout_seconds=0)
    assert first["wait_timed_out"] is True and first["resources_released"] is False
    # The business failure can be reported immediately for a running wait.
    failure = client.wait(row["id"], until="running", timeout_seconds=2)
    assert failure["state"] == "failed" and failure["roles"][0]["quiet"] is False
    assert failure["logs_pending"] is True and "stderr" not in failure
    owner.pool.managed_control.assert_not_called()
    _, current = owner._owned_row(store.state_dir, "alice", row["id"])
    assert "terminal_logs" not in current
    current["roles"][0]["observation"].update(lease_state="released", remote={"quiet": True})
    owner._save_execution(store, current)
    owner.pool.managed_control.return_value = {"remote": {"stdout": "complete output", "stderr": "failure details after cleanup"}}
    last = client.wait(row["id"], until="released", timeout_seconds=2)
    assert last["resources_released"] is True and last["state"] == "failed"
    assert last["roles"][0]["quiet"] is True and last["stdout"] == "complete output"
    assert last["stderr"] == "failure details after cleanup"
    again = client.wait(row["id"], until="released", timeout_seconds=2)
    assert again["stderr"] == last["stderr"]
    assert owner.pool.managed_control.call_count == 1
    assert owner.pool.managed_control.call_args.args == ("alice", "job", "tail")


def test_terminal_tail_failure_and_slow_tail_do_not_change_outcome_or_block_control(owned):
    owner, store, client, row = owned
    row["phase"] = "succeeded"
    owner._save_execution(store, row)
    started, release = threading.Event(), threading.Event()
    def tail(*args, **kwargs):
        started.set()
        assert release.wait(3)
        raise RuntimeError("log transport unavailable")
    owner._tail = tail
    with ThreadPoolExecutor() as workers:
        pending = workers.submit(client.wait, row["id"], until="released", timeout_seconds=.05)
        assert started.wait(1)
        reply = pending.result(timeout=1)
        assert reply["state"] == "succeeded" and reply["resources_released"] is True and reply["logs_pending"]
        assert owner.handle({"op": "ping"})["ok"] is True
        release.set()
    result = client.wait(row["id"], until="released", timeout_seconds=2)
    assert result["state"] == "succeeded" and result["resources_released"] is True
    assert "log transport unavailable" in result["tail_error"]


def test_terminal_log_publication_serializes_with_latest_lifecycle_facts(owned):
    owner, store, client, row = owned
    row.update(phase="failed", roles=[{"name": "default", "binding": {}, "managed_job": "job",
        "observation": {"state": "failed", "lease_state": "released", "remote": {"quiet": True}}}])
    owner._save_execution(store, row)
    entered, released = threading.Event(), threading.Event()
    def tail(*args, **kwargs):
        entered.set()
        assert released.wait(3)
        return {"stdout": "business exit", "roles": []}
    owner._tail = tail
    original = owner._save_execution
    def save(store, current):
        if "terminal_logs" in current:
            lock = owner._lock_for("execution", row["id"])
            assert not lock.acquire(blocking=False), "tail publication must join lifecycle serialization"
        return original(store, current)
    owner._save_execution = save
    with ThreadPoolExecutor() as workers:
        result = workers.submit(client.wait, row["id"], until="running", timeout_seconds=3)
        assert entered.wait(1)
        with owner._lock_for("execution", row["id"]):
            _, current = owner._owned_row(store.state_dir, "alice", row["id"])
            current["error"] = "latest cleanup detail"
            owner._save_execution(store, current)
        released.set()
        reply = result.result(timeout=1)
    assert reply["resources_released"] is True and reply["roles"][0]["quiet"] is True
    assert reply["error"] == "latest cleanup detail"
    assert reply["stdout"] == "business exit"


def test_legacy_same_step_progress_without_timestamp_is_upgraded(owned):
    owner, store, client, row = owned
    row["role_progress"] = {"default": {"step": "compile", "role": "default"}}
    owner._save_progress(store, row, "default", {"step": "compile"})
    assert row["role_progress"]["default"]["started_at"] > 0
    owner._save_progress(store, row, "default", {"step": "finalize"})
    assert row["stage_history"][0]["step"] == "compile"
    assert row["stage_history"][0]["elapsed_seconds"] >= 0


@pytest.mark.parametrize("budget", [-1, float("nan"), float("inf"), 601, True, "30"])
def test_invalid_wait_budget_does_not_admit(owned, budget):
    owner, store, client, row = owned
    owner.admit = Mock()
    with pytest.raises(ValueError, match="wait timeout_seconds"):
        client.run("true", sources={}, wait_until="released", wait_timeout_seconds=budget)
    owner.admit.assert_not_called()


def test_script_file_is_fixed_once_and_run_waits_on_its_admitted_reference(owned, tmp_path):
    owner, store, client, row = owned
    script = tmp_path / "business.sh"
    raw = '#!/bin/bash\nprintf "业务\\n"\n'.encode("utf-8")
    script.write_bytes(raw)
    def admit(directory, user, session, spec, restart=False):
        assert spec["command"].encode("utf-8") == raw
        assert spec["script"]["sha256"] == hashlib.sha256(raw).hexdigest()
        script.write_text("exit 99\n", encoding="utf-8")
        return {"execution_id": row["id"], "state": "preparing"}
    owner.admit = Mock(side_effect=admit)
    result = client.run(script_file=script, sources={}, wait_until="released", wait_timeout_seconds=0)
    assert result["execution_id"] == row["id"] and result["wait_timed_out"]
    owner.admit.assert_called_once()
    with pytest.raises(ValueError, match="exactly one"):
        client.run("true", script_file=script, sources={})


def test_script_client_path_mapping_size_bound_and_service_identity(owned, tmp_path, monkeypatch):
    owner, store, client, row = owned
    script = tmp_path / "script.sh"
    script.write_text("true\n", encoding="utf-8")
    monkeypatch.setattr("mindie_coordinator.task_client.client_path", lambda path: str(script))
    owner.admit = Mock(return_value={"execution_id": row["id"], "state": "running"})
    client.run(script_file="/mnt/d/script.sh", sources={})
    spec = owner.admit.call_args.args[3]
    other = {**spec, "script": {**spec["script"], "path": "/another/file.sh"}}
    assert service._service_spec_key(spec) == service._service_spec_key(other)
    script.write_bytes(b"a" * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="at most 1 MiB"):
        client.run(script_file="/mnt/d/script.sh", sources={})
    assert owner.admit.call_count == 1


def test_wait_transport_error_after_admission_retains_reference(owned):
    owner, store, client, row = owned
    owner.admit = Mock(return_value={"execution_id": row["id"], "state": "preparing"})
    owner.wait = Mock(side_effect=TimeoutError("IPC observation lost"))
    reply = client.run("true", sources={}, wait_until="released")
    assert reply["execution_id"] == row["id"] and reply["state"] == "preparing"
    assert "IPC observation lost" in reply["wait_error"]
    owner.admit.assert_called_once()


def test_windows_script_bom_and_crlf_executes_as_shell_text(owned, tmp_path):
    owner, store, client, row = owned
    normalized = "set -e\ncat <<'DATA'\nexact output\nDATA\n"
    raw = b"\xef\xbb\xbf" + normalized.replace("\n", "\r\n").encode("utf-8")
    script = tmp_path / "windows.sh"
    script.write_bytes(raw)
    owner.admit = Mock(return_value={"execution_id": row["id"], "state": "running"})
    client.run(script_file=script, sources={})
    spec = owner.admit.call_args.args[3]
    assert spec["command"] == normalized
    assert spec["script"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert spec["script"]["bytes"] == len(raw)
    assert spec["script"]["command_sha256"] == hashlib.sha256(normalized.encode()).hexdigest()
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is not available for the plain-shell execution check")
    result = subprocess.run([bash, "-c", spec["command"]], capture_output=True, text=True, encoding="utf-8", timeout=15)
    assert result.returncode == 0 and result.stdout == "exact output\n", result.stderr


def receipt(row):
    manifest = {"profile_key": "profile", "build_key": "build", "runtime_root": "/owned/root",
                "execution_view": {"source_id": row["spec"]["source_snapshot"]["id"]},
                "build_inputs": {"repo": {"native": "fixed-native"}},
                "files": {"repo/kernel/a.o": {"sha256": "a" * 64, "role": "library"}}}
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return manifest, {"manifest_zlib_base64": base64.b64encode(zlib.compress(encoded)).decode(),
                      "manifest_bytes": len(encoded), "manifest_digest": digest(manifest),
                      "preparation_timings": {"native_compile": 4.5}, "native_cache": {"status": "stored"}}


@pytest.mark.parametrize('previous_miss', [False, True])
def test_materialization_timing_is_readable_without_a_full_manifest(owned, previous_miss):
    owner, store, client, row = owned
    row['role_progress'] = {'default': {'step': 'materialize'}}
    owner._save_execution(store, row)
    directory = owner.state_dir / 'runs' / row['id'] / 'default'
    directory.mkdir(parents=True)
    timing = {'materialize': 2.0, 'source_metadata': 0.1, 'native_publication': 0.8}
    previous = (json.dumps({'status': 'missing', 'preparation_timings': {'materialize': 0.01}}) + '\n'
                if previous_miss else '')
    (directory / 'materialize.log').write_text(previous + json.dumps({'status': 'materialized',
        'preparation_timings': timing, 'native_smoke_executed': False}))
    owner.pool.managed_control = Mock(side_effect=AssertionError('no remote probe'))
    result = client.observe(row['id'], action='evidence', section='preparation')
    log = result['evidence']['roles'][0]['logs'][0]
    assert log['preparation_timings'] == timing and log['native_smoke_executed'] is False


@pytest.mark.parametrize("log_name", ["verify-profile.log", "finalize-runtime.log", "materialize-sources.log"])
def test_evidence_decodes_owned_receipts_without_remote_reads(owned, log_name):
    owner, store, client, row = owned
    row["role_progress"] = {"default": {"step": "finished"}}
    owner._save_execution(store, row)
    manifest, envelope = receipt(row)
    directory = owner.state_dir / "runs" / row["id"] / "default"
    directory.mkdir(parents=True)
    log = directory / log_name
    log.write_text(json.dumps(envelope), encoding="utf-8")
    owner.pool.managed_control = Mock(side_effect=AssertionError("no remote probe"))
    result = client.observe(row["id"], action="evidence", path="kernel")
    evidence = result["evidence"]
    assert evidence["sources"]["snapshot_id"] == row["spec"]["source_snapshot"]["id"]
    role = evidence["roles"][0]
    assert role["build"]["build_inputs"] == manifest["build_inputs"]
    assert role["build"]["record_ref"] == str(log)
    assert role["artifacts"][0]["sha256"] == "a" * 64
    assert role["logs"][0]["preparation_timings"] == {"native_compile": 4.5}
    assert "manifest_zlib_base64" not in json.dumps(result)
    owner.pool.managed_control.assert_not_called()


def test_evidence_uses_exact_retained_binding_and_rejects_changed_runtime(owned):
    owner, store, client, row = owned
    manifest, _ = receipt(row)
    row["roles"] = [{"name": "default", "runtime_id": "runtime", "binding": {"build_key": "build"}}]
    owner._save_execution(store, row)
    with owner.pool.transaction() as db:
        owner.pool.put(db, "runtime", {"id": "runtime", "attestation": manifest})
    evidence = client.observe(row["id"], action="evidence", section="build", path="kernel")["evidence"]
    assert evidence["roles"][0]["artifacts_total"] == 1
    manifest["build_key"] = "new-build"
    with owner.pool.transaction() as db:
        owner.pool.put(db, "runtime", {"id": "runtime", "attestation": manifest})
    role = client.observe(row["id"], action="evidence", section="build", path="kernel")["evidence"]["roles"][0]
    assert "differs" in role["receipt_error"] and "artifacts" not in role


def test_corrupt_receipt_reports_local_error_and_ref(owned):
    owner, store, client, row = owned
    row["role_progress"] = {"default": {"step": "failed"}}
    owner._save_execution(store, row)
    _, envelope = receipt(row)
    envelope["manifest_digest"] = "wrong"
    directory = owner.state_dir / "runs" / row["id"] / "default"
    directory.mkdir(parents=True)
    log = directory / "finalize-runtime.log"
    log.write_text(json.dumps(envelope), encoding="utf-8")
    entry = client.observe(row["id"], action="evidence")["evidence"]["roles"][0]["logs"][0]
    assert "digest differs" in entry["error"] and entry["log_ref"] == str(log)
    assert "tail" not in entry


def test_cli_local_script_and_existing_execution_wait(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(mindie, "mindie_call", lambda name, args: calls.append((name, args)) or {"result": {"outcome": "success"}})
    monkeypatch.setattr("sys.argv", ["mindie", "run", "--script-file", "case.sh", "--no-sources", "--wait", "released", "--wait-timeout-seconds", "180"])
    assert mindie.main() == 0
    assert calls[-1][1] == {"script_file": "case.sh", "sources": {}, "wait_until": "released", "wait_timeout_seconds": 180.0}
    monkeypatch.setattr("sys.argv", ["mindie", "execution", "--execution-id", "e" * 64, "--wait", "running"])
    assert mindie.main() == 0
    assert calls[-1][1] == {"execution_id": "e" * 64, "action": "wait", "until": "running"}
    assert len(capsys.readouterr().out.splitlines()) == 2


@pytest.mark.parametrize("framed", [False, True])
def test_mcp_wait_saturation_keeps_stop_ping_and_ids_responsive(monkeypatch, framed):
    entered = 0
    guard = threading.Lock()
    all_waiting, release = threading.Event(), threading.Event()
    def call(name, arguments, metadata=None):
        nonlocal entered
        if arguments.get("action") == "wait":
            with guard:
                entered += 1
                if entered == 8:
                    all_waiting.set()
            assert release.wait(3), "stop was blocked behind wait workers"
        else:
            assert all_waiting.wait(2)
            release.set()
        return {"structuredContent": {"state": arguments["action"]}, "content": []}
    monkeypatch.setattr(task_server, "call_tool", call)
    messages = [{"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {
                    "name": "mindie_execution", "arguments": {"action": "wait", "execution_id": "e" * 64}}} for i in range(8)]
    messages += [{"jsonrpc": "2.0", "id": 99, "method": "tools/call", "params": {
                    "name": "mindie_execution", "arguments": {"action": "stop", "execution_id": "e" * 64}}},
                 {"jsonrpc": "2.0", "id": 100, "method": "ping"}]
    raw = b""
    for message in messages:
        encoded = json.dumps(message).encode()
        raw += (f"Content-Length: {len(encoded)}\r\n\r\n".encode() + encoded) if framed else encoded + b"\n"
    output = io.BytesIO()
    assert task_server.serve(io.BytesIO(raw), output) == 0
    responses = list(task_server.StdioTransport(io.BytesIO(output.getvalue()), io.BytesIO()).messages())
    assert {message["id"] for message in responses} == {*range(8), 99, 100}
    assert len(responses) == 10 and all("result" in message for message in responses)
    assert next(message for message in responses if message["id"] == 99)["result"]["structuredContent"]["state"] == "stop"


def test_ipc_wait_timeout_includes_budget_margin(tmp_path, monkeypatch):
    marker = tmp_path / "ipc"
    marker.write_text(json.dumps({"port": 12345, "token": "fixture"}), encoding="utf-8")
    monkeypatch.setattr(service, "socket_path", lambda _: marker)
    socket = Mock()
    socket.recv.return_value = b'{"ok": true, "value": {"state": "running"}}\n'
    monkeypatch.setattr(service.socket, "socket", lambda *a: socket)
    client = service.CoordinatorClient(tmp_path)
    assert client.wait(tmp_path, "alice", "e" * 64, timeout_seconds=600)["state"] == "running"
    socket.settimeout.assert_called_once_with(610)
