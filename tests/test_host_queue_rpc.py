"""Host requests use the native Python transport without bypassing the authority."""
import json
import subprocess
import sys
from unittest.mock import Mock

import pytest

from mindie_coordinator.backend import RemoteBackend
from mindie_coordinator.host_queue import HostQueue, HostQueueUnavailable


HOST = {"host": "host.invalid", "port": 22, "user": "root", "cwd": "/ignored"}


def test_default_backend_executes_fixed_host_code_with_separate_payloads(tmp_path, monkeypatch):
    calls = []

    def python(endpoint, code, payload, *, timeout_ms):
        calls.append((endpoint, code, payload, timeout_ms))
        # Windows command lines cannot hold the full Linux host module.
        program = tmp_path / "host_program.py"
        program.write_text(code, encoding="utf-8")
        result = subprocess.run([sys.executable, str(program)], input=json.dumps(payload),
                                capture_output=True, text=True, encoding="utf-8", timeout=10)
        return json.loads(result.stdout)

    monkeypatch.setattr("remote_dev.core.ssh_transport.run_remote_python", python)
    backend = RemoteBackend()
    backend.bash = Mock(side_effect=AssertionError("normal host requests must not embed shell code"))
    state_dir = str(tmp_path / "authority")
    status = backend.host({"host_endpoint": HOST}, {"action": "status", "no_probe": True, "state_dir": state_dir})
    epoch = status["coordination_epoch"]
    assert status["diagnostics"]["elapsed_ms"] >= 0
    assert len(status["diagnostics"]["clock_domain"]) == 32
    common = {"state_dir": state_dir, "coordination_epoch": epoch, "task_id": "cpu-task"}
    submitted = backend.host({"host_endpoint": HOST},
                             {**common, "action": "submit", "agent_id": "owner", "npu_count": 0})
    assert submitted["status"] == "queued"
    assert submitted["diagnostics"]["phases"]["state_lock_wait_ms"] >= 0
    granted = backend.host({"host_endpoint": HOST}, {**common, "action": "acquire"})
    assert granted["status"] == "granted"
    # Execute the unchanged authority's fencing check in its real SQLite state.
    with pytest.raises(RuntimeError, match="fencing"):
        backend.host({"host_endpoint": HOST}, {**common, "action": "preflight", "fence_token": -1})
    cancelled = backend.host({"host_endpoint": HOST}, {**common, "action": "cancel"})
    assert cancelled["status"] == "cancelled"
    assert len(calls) == 5
    assert len({code for _, code, _, _ in calls}) == 1
    assert "def handle_request(" in calls[0][1]
    assert "cpu-task" not in calls[0][1]
    assert all(endpoint.root == "/" and endpoint.cwd == "/" and timeout == 45000
               for endpoint, _, _, timeout in calls)
    backend.bash.assert_not_called()


@pytest.mark.parametrize("reply", [
    {"status": "failed", "error": "unknown remote outcome"},
    {"status": "needs_input", "error": "wrong fence"},
    {"status": "probe_failed", "error": "unknown occupancy"},
    {"status": "timeout", "error": "no deadline reply"},
    {"status": "cancelled", "error": "interrupted"},
    [],
])
def test_native_failure_never_replays_through_shell(monkeypatch, reply):
    python = Mock(return_value=reply)
    monkeypatch.setattr("remote_dev.core.ssh_transport.run_remote_python", python)
    backend = RemoteBackend()
    backend.bash = Mock(side_effect=AssertionError("a mutation must not be replayed"))
    with pytest.raises(RuntimeError):
        backend.host({"host_endpoint": HOST}, {"action": "activate", "fence_token": 7})
    python.assert_called_once()
    backend.bash.assert_not_called()


def test_native_transport_exception_is_uncertain_without_fallback(monkeypatch):
    python = Mock(side_effect=OSError("connection lost after send"))
    monkeypatch.setattr("remote_dev.core.ssh_transport.run_remote_python", python)
    with pytest.raises(RuntimeError, match="reconcile before retry"):
        HostQueue().request(HOST, {"action": "release"})
    python.assert_called_once()


def test_native_override_resolves_before_transport_and_keeps_code_stable(tmp_path, monkeypatch):
    module = tmp_path / "override.py"
    module.write_text("import json\nclass CoordinationError(Exception): pass\n"
                      "def handle_request(request): return {'status': 'ok'}\n")
    python = Mock(return_value={"status": "ok"})
    monkeypatch.setattr("remote_dev.core.ssh_transport.run_remote_python", python)
    queue = HostQueue(module_path=module)
    queue.request(HOST, {"action": "first", "value": "quotes ' 中文\n"})
    queue.request(HOST, {"action": "second"})
    assert python.call_args_list[0].args[1] == python.call_args_list[1].args[1]
    assert python.call_args_list[0].args[2]["value"] == "quotes ' 中文\n"
    with pytest.raises(HostQueueUnavailable):
        HostQueue(module_path=tmp_path / "missing.py").request(HOST, {"action": "status"})
    assert python.call_count == 2


def test_explicit_shell_and_host_queue_injection_remain_authoritative(tmp_path, monkeypatch):
    stdout = tmp_path / "stdout.json"
    stdout.write_text(json.dumps({"status": "ok", "tasks": []}))
    shell = Mock()
    shell.run.return_value = {"outcome": "success", "refs": {"stdout": str(stdout)}}
    python = Mock(side_effect=AssertionError("explicit adapters remain in charge"))
    monkeypatch.setattr("remote_dev.core.ssh_transport.run_remote_python", python)
    backend = RemoteBackend(shell=shell)
    assert backend.host({"host_endpoint": HOST}, {"action": "status"})["status"] == "ok"
    shell.run.assert_called_once()
    assert "def handle_request(" in shell.run.call_args.args[1]
    injected = Mock()
    assert RemoteBackend(host_queue=injected).host_queue is injected
    python.assert_not_called()


def test_real_host_failure_survives_native_nonzero_wrapper_without_output_leak(tmp_path, monkeypatch):
    calls = []

    def rpc(endpoint, operation, code, payload, *, timeout_ms):
        assert operation == 'python'
        calls.append(payload)
        program = tmp_path / 'host_failure.py'
        program.write_text(code, encoding='utf-8')
        completed = subprocess.run([sys.executable, str(program)], input=json.dumps(payload),
                                   capture_output=True, text=True, encoding='utf-8', timeout=10)
        return {'returncode': completed.returncode, 'stdout': completed.stdout,
                'stderr': completed.stderr}

    # Keep run_remote_python's actual exit-code/JSON handling. Only the SSH
    # peer is replaced with a local process running the exact shipped source.
    monkeypatch.setattr('remote_dev.core.rpc_transport.request', rpc)
    queue = HostQueue()
    common = {'state_dir': str(tmp_path / 'authority'), 'action': 'container-ssh-reserve', 'port': 46002}
    queue.request(HOST, {**common, 'user': 'alice', 'container_name': 'mindie-alice'})
    with pytest.raises(RuntimeError, match=r'already reserved \(port 46002\) \[port_reserved\]'):
        queue.request(HOST, {**common, 'user': 'bob', 'container_name': 'mindie-bob'})
    assert len(calls) == 2


@pytest.mark.parametrize('port', [46002, 'SECRET', True, -1, 65536])
def test_known_reservation_error_uses_only_allowed_code_and_port(monkeypatch, port):
    reply = {'status': 'failed', 'error': 'remote python failed', 'exit_code': 2,
             'stdout_tail': json.dumps({'status': 'needs_input', 'error_code': 'port_reserved',
                                       'port': port, 'error': 'SECRET', 'credentials': 'SECRET'}),
             'stderr_tail': 'SECRET'}
    python = Mock(return_value=reply)
    monkeypatch.setattr('remote_dev.core.ssh_transport.run_remote_python', python)
    with pytest.raises(RuntimeError, match='host port is already reserved') as caught:
        HostQueue().request(HOST, {'action': 'container-ssh-reserve'})
    assert 'SECRET' not in str(caught.value)
    assert ('(port ' in str(caught.value)) is (type(port) is int and 0 < port < 65536)
    python.assert_called_once()


@pytest.mark.parametrize('change', [
    {'remote_outcome': 'unknown'}, {'status': 'timeout'}, {'status': 'cancelled'}, {'exit_code': 1},
    {'stdout_tail': 'not JSON'}, {'stdout_tail': '[]'}, {'stdout_tail': 'x' * 4001},
    {'stdout_tail': json.dumps({'status': 'needs_input', 'error_code': 'SECRET'})},
])
def test_unrecognized_or_uncertain_failure_is_not_reinterpreted_or_replayed(monkeypatch, change):
    reply = {'status': 'failed', 'error': 'original failure', 'exit_code': 2,
             'stdout_tail': json.dumps({'status': 'needs_input', 'error_code': 'port_reserved', 'port': 46002}),
             **change}
    python = Mock(return_value=reply)
    monkeypatch.setattr('remote_dev.core.ssh_transport.run_remote_python', python)
    with pytest.raises(RuntimeError, match='^original failure$'):
        HostQueue().request(HOST, {'action': 'container-ssh-reserve'})
    python.assert_called_once()
