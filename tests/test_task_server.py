"""The stdio task server: listing, dispatch, capability, and offline degradation."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from importlib.metadata import version
from pathlib import Path
from unittest import mock

from mindie_coordinator import task_server
from mindie_coordinator.agent_session import AgentSessions
from mindie_coordinator.host_queue import SCHEMA_VERSION
from mindie_coordinator.service import CoordinatorClient, socket_path, _lock_daemon
from mindie_coordinator.task_server import (
    ALIASES,
    SERVICE_NAME,
    StdioTransport,
    call_tool,
    handle,
    initialize_result,
    list_tools,
    package_version,
)

LEAKY = ("MINDIE_COORDINATOR_STATE_DIR", "MINDIE_AGENT_SESSIONS_DIR", "MINDIE_CONTEXT_FILE")


def clean_environment(**extra):
    environment = {key: value for key, value in os.environ.items() if key not in LEAKY}
    environment.update(extra)
    return environment


def stop_test_daemon(root):
    """Stop only the daemon belonging to this test before removing its files."""
    state = root / "coordinator"
    if not socket_path(state).exists():
        return
    client = CoordinatorClient(state, timeout=5)
    pid = client.call("ping")["runtime"][0]["loaded"]["pid"]
    assert client.call("restart_if_idle")["status"] == "stopping"
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if not socket_path(state).exists():
            if os.name == "nt":
                from test_windows_service import pid_alive
                if pid_alive(pid):
                    time.sleep(0.05)
                    continue
            try:
                with (state / "coordinator.lock").open("a+b") as guard:
                    _lock_daemon(guard)
                    _lock_daemon(guard, release=True)
                return
            except OSError:
                pass
        time.sleep(0.05)
    raise AssertionError("test daemon did not release its state directory")


class TaskRegistry:
    def __init__(self, root: Path):
        self.state_dir = root / "agent-sessions"
        self.store = AgentSessions(self.state_dir)
        self.context = self.store.attach("codex", "native-test", str(root))
        self.context_file = self.context["context_file"]


class CapabilityTests(unittest.TestCase):
    def test_shared_device_schema_requires_explicit_single_card_opt_in(self):
        from jsonschema import Draft202012Validator, ValidationError

        schema = next(tool["inputSchema"] for tool in list_tools() if tool["name"] == "mindie_run")
        validator = Draft202012Validator(schema)
        validator.validate({"command": "serve", "resources": {"devices": [0], "allow_external_busy": True}})
        for resources in ({"allow_external_busy": True}, {"devices": [0, 1], "allow_external_busy": True},
                          {"devices": [0], "allow_external_busy": "true"}):
            with self.subTest(resources=resources), self.assertRaises(ValidationError):
                validator.validate({"command": "serve", "resources": resources})

    def test_tools_list_advertises_portable_names_with_their_own_schemas(self):
        tools = list_tools()
        self.assertEqual([tool["name"] for tool in tools], ["mindie_session", "mindie_run", "mindie_execution", "mindie_finish", "mindie_message"])
        from jsonschema import Draft202012Validator, ValidationError

        examples = {"mindie_session": {"sources": {}}, "mindie_run": {"command": "echo ready"},
                    "mindie_execution": {"execution_id": "a" * 64}, "mindie_finish": {},
                    "mindie_message": {"recipient": {"host": "host-ref", "user": "bob", "session_id": "task-bob"}, "text": "hello"}}
        for tool in tools:
            with self.subTest(tool=tool["name"]):
                self.assertTrue(tool["description"])
                validator = Draft202012Validator(tool["inputSchema"])
                validator.validate(examples[tool["name"]])
                with self.assertRaises(ValidationError):
                    validator.validate({**examples[tool["name"]], "unsupported": True})
        self.assertEqual(handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"], {"tools": tools})

    def test_initialize_declares_the_package_version(self):
        result = initialize_result({"protocolVersion": "2025-03-26", "capabilities": {}})
        self.assertEqual(result["protocolVersion"], "2025-03-26")
        declared = result["capabilities"]["experimental"][SERVICE_NAME]
        self.assertEqual(declared["version"], version("mindie-coordinator"))
        self.assertEqual(declared["version"], package_version())
        self.assertEqual(declared["tools"], sorted(ALIASES))
        self.assertEqual(result["serverInfo"]["version"], package_version())
        self.assertEqual(result["serverInfo"]["name"], SERVICE_NAME)
        self.assertIn("tools", result["capabilities"])
        self.assertEqual(initialize_result({"protocolVersion": "9999-01-01"})["protocolVersion"],
                         task_server.PROTOCOL_VERSIONS[-1])

    def test_initialize_declares_the_bundled_host_protocol_schema_version(self):
        declared = initialize_result({})["capabilities"]["experimental"][SERVICE_NAME]
        self.assertEqual(declared["host_protocol_schema_version"], SCHEMA_VERSION)

    def test_json_rpc_plumbing_notifications_ping_and_unknown_methods(self):
        self.assertIsNone(handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        self.assertEqual(handle({"jsonrpc": "2.0", "id": 7, "method": "ping"}), {"jsonrpc": "2.0", "id": 7, "result": {}})
        error = handle({"jsonrpc": "2.0", "id": 8, "method": "resources/list"})["error"]
        self.assertEqual(error["code"], -32601)
        error = handle({"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "remote_bash", "arguments": {}}})["error"]
        self.assertEqual(error["code"], -32602)
        self.assertIn("unknown task tool", error["message"])
        error = handle({"jsonrpc": "2.0", "id": 10, "method": "tools/call", "params": {"name": "mindie_session", "arguments": []}})["error"]
        self.assertEqual(error["code"], -32000)
        self.assertEqual(handle({"jsonrpc": "2.0", "id": 11})["error"]["code"], -32600)

    def test_describe_is_the_offline_view_of_initialize_and_tools_list(self):
        proc = subprocess.run(
            [sys.executable, "-m", "mindie_coordinator", "task-server", "--describe"],
            capture_output=True, text=True, check=False, env=clean_environment(),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["serverInfo"]["version"], package_version())
        self.assertEqual([tool["name"] for tool in payload["tools"]], list(ALIASES))
        proc = subprocess.run(
            [sys.executable, "-m", "mindie_coordinator", "task-server", "--help"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("usage:", proc.stdout)


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(stop_test_daemon, self.root)
        self.registry = TaskRegistry(self.root)
        patcher = mock.patch.dict("os.environ", clean_environment(), clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def call(self, name, **arguments):
        response = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": name, "arguments": {"context_file": self.registry.context_file, **arguments}}})
        self.assertNotIn("error", response, response)
        result = response["result"]
        self.assertEqual(json.loads(result["content"][0]["text"]), result["structuredContent"])
        return result

    def test_mindie_session_is_local_and_the_same_task_answers_twice(self):
        first = self.call("mindie_session")
        second = self.call("mindie_session")
        for result in (first, second):
            self.assertFalse(result["isError"], result)
            self.assertEqual(result["structuredContent"]["outcome"], "success")
            self.assertEqual(result["structuredContent"]["status"], "open")
            self.assertEqual(result["structuredContent"]["tool"], "mindie.session")
        self.assertEqual(first["structuredContent"]["data"]["session"]["id"],
                         second["structuredContent"]["data"]["session"]["id"])
        self.assertEqual(first["structuredContent"]["target"]["session_id"], self.registry.context["session"]["id"])

    def test_mindie_session_binds_an_actual_worktree_without_any_manager(self):
        repo = self.root / "repo"
        repo.mkdir()
        for args in (["init"], ["config", "user.name", "T"], ["config", "user.email", "t@example.invalid"],
                     ["commit", "--allow-empty", "-m", "base"]):
            subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
        result = self.call("mindie_session", sources={"repo": str(repo)})
        self.assertFalse(result["isError"])
        self.assertEqual(result["structuredContent"]["data"]["session"]["sources"]["repo"]["path"], str(repo.resolve()))

    def test_mindie_run_without_bound_sources_admits_an_empty_fixed_input(self):
        owner = mock.Mock()
        owner.admit.return_value = {"execution_id": "e" * 64, "state": "queued"}
        owner.runtime = None
        with mock.patch("mindie_coordinator.service.ensure_daemon", return_value=owner):
            result = self.call("mindie_run", command="true")
        self.assertFalse(result["isError"])
        self.assertEqual(result["structuredContent"]["tool"], "mindie.run")
        self.assertEqual((result["structuredContent"]["outcome"], result["structuredContent"]["status"]), ("success", "queued"))
        spec = owner.admit.call_args.args[3]
        self.assertEqual(spec["source_snapshot"]["records"], [])
        self.assertEqual(spec["resources"], {"npu_count": 0})

    def test_mindie_run_forwards_explicit_shared_device_in_fixed_admission(self):
        owner = mock.Mock()
        owner.admit.return_value = {"execution_id": "e" * 64, "state": "queued"}
        owner.runtime = None
        with mock.patch("mindie_coordinator.service.ensure_daemon", return_value=owner):
            result = self.call("mindie_run", command="serve", sources={},
                               resources={"devices": [0], "allow_external_busy": True})
        self.assertFalse(result["isError"])
        spec = owner.admit.call_args.args[3]
        self.assertEqual(spec["resources"], {"devices": [0], "allow_external_busy": True})
        self.assertTrue(spec["roles"][0]["allow_external_busy"])

    def test_accepted_execution_progress_is_not_an_mcp_error(self):
        from mindie_coordinator.task_client import TaskClient

        for state, outcome in (("queued", "success"), ("preparing", "success"), ("waiting", "success"),
                               ("running", "success"), ("cancelled", "cancelled"),
                               ("uncertain", "blocked"), ("inconclusive", "failed"), ("timeout", "timeout")):
            with self.subTest(state=state), mock.patch.object(TaskClient, "observe", return_value={"execution_id": "e" * 64, "state": state}):
                result = self.call("mindie_execution", execution_id="e" * 64)
                self.assertEqual(result["isError"], outcome not in {"success", "cancelled"})
                self.assertEqual(result["structuredContent"]["outcome"], outcome)
                self.assertEqual(result["structuredContent"]["data"]["state"], state)

    def test_pending_runtime_update_preserves_blocking_facts_without_claiming_admission(self):
        from mindie_coordinator.task_client import TaskClient

        pending = {"state": "needs_runtime_update", "reason": "nonterminal executions remain",
                   "runtime_update": {"selected": [{"commit": "new"}], "daemon": [{"loaded": {"commit": "old"}}]},
                   "active_executions": [{"execution_id": "e" * 64, "state": "running"}]}
        with mock.patch.object(TaskClient, "run", return_value=pending):
            result = self.call("mindie_run", command="true", sources={})
        self.assertTrue(result["isError"])
        observation = result["structuredContent"]
        self.assertEqual((observation["outcome"], observation["status"]), ("blocked", "needs_runtime_update"))
        self.assertEqual(observation["data"], pending)
        self.assertNotIn("execution_id", observation["data"])

    def test_mindie_execution_rejects_an_id_it_does_not_own_before_any_network(self):
        result = self.call("mindie_execution", execution_id="not-an-id", action="status")
        self.assertTrue(result["isError"])
        self.assertEqual(result["structuredContent"]["tool"], "mindie.execution")
        self.assertEqual(result["structuredContent"]["outcome"], "blocked")
        self.assertIn("invalid local execution id", result["structuredContent"]["summary"])

    def test_mindie_execution_rejects_another_task_before_the_coordinator(self):
        other = self.registry.store.attach("codex", "native-other", str(self.root))
        row = self.registry.store.execution(self.registry.context, "owned-fixture", {"command": "true"})
        row.update(phase="cancelled", user="alice", roles=[])
        self.registry.store.save_execution(row)
        with mock.patch("mindie_coordinator.service.ensure_daemon",
                        side_effect=AssertionError("coordinator must not start for a foreign execution")):
            for action in ("status", "tail", "stop", "target"):
                with self.subTest(action=action):
                    response = handle({
                        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": "mindie_execution",
                                   "arguments": {"context_file": other["context_file"],
                                                 "execution_id": row["id"], "action": action}},
                    })
                    result = response["result"]
                    self.assertTrue(result["isError"], result)
                    self.assertEqual(result["structuredContent"]["tool"], "mindie.execution")
                    self.assertEqual(result["structuredContent"]["outcome"], "blocked")
                    self.assertIn("another MindIE task", result["structuredContent"]["summary"])
        with self.registry.store.transaction() as db:
            latest = self.registry.store.get(db, "execution", row["id"])
        self.assertEqual(latest["phase"], "cancelled")
        self.assertFalse(latest.get("cancel_requested"))

    def test_mindie_finish_completes_locally_and_the_dotted_name_is_still_accepted(self):
        result = self.call("mindie.finish")
        self.assertFalse(result["isError"])
        self.assertEqual(result["structuredContent"]["tool"], "mindie.finish")
        self.assertEqual((result["structuredContent"]["outcome"], result["structuredContent"]["status"]), ("success", "finished"))
        self.assertTrue(result["structuredContent"]["data"]["worktrees_preserved"])

    def test_a_missing_context_is_blocked_not_guessed(self):
        response = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "mindie_session"}})
        result = response["result"]
        self.assertTrue(result["isError"])
        self.assertEqual(result["structuredContent"]["status"], "unavailable")
        self.assertIn("context is required", result["structuredContent"]["summary"])

    def test_call_tool_uses_the_remote_dev_result_envelope(self):
        result = call_tool("mindie_session", {"context_file": self.registry.context_file})
        self.assertEqual(result["structuredContent"]["schema_version"], "remote-dev.result.v1")
        self.assertLessEqual({"tool", "invocation_id", "target", "outcome", "status", "summary", "warnings"},
                             set(result["structuredContent"]))


class StdioClient:
    def __init__(self, framed: bool, env: dict):
        self.framed = framed
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "mindie_coordinator", "task-server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        )
        self.sequence = 0

    def send(self, payload):
        body = json.dumps(payload).encode()
        if self.framed:
            self.proc.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        else:
            self.proc.stdin.write(body + b"\n")
        self.proc.stdin.flush()

    def receive(self):
        if not self.framed:
            return json.loads(self.proc.stdout.readline())
        headers = {}
        while True:
            line = self.proc.stdout.readline()
            if line in {b"\r\n", b"\n"}:
                break
            key, value = line.decode().split(":", 1)
            headers[key.strip().lower()] = value.strip()
        return json.loads(self.proc.stdout.read(int(headers["content-length"])))

    def rpc(self, method, params=None):
        self.sequence += 1
        self.send({"jsonrpc": "2.0", "id": self.sequence, "method": method, "params": params or {}})
        reply = self.receive()
        assert reply["id"] == self.sequence, reply
        return reply

    def close(self):
        self.proc.stdin.close()
        code = self.proc.wait(timeout=20)
        stderr = self.proc.stderr.read().decode()
        self.proc.stdout.close()
        self.proc.stderr.close()
        return code, stderr


class LiveStdioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.registry = TaskRegistry(Path(self.temp.name))
        self.addCleanup(stop_test_daemon, Path(self.temp.name))
        self.env = clean_environment()

    def session(self, framed):
        client = StdioClient(framed, self.env)
        init = client.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                         "clientInfo": {"name": "test", "version": "0"}})
        self.assertEqual(init["result"]["serverInfo"]["version"], package_version())
        client.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        names = [tool["name"] for tool in client.rpc("tools/list")["result"]["tools"]]
        self.assertEqual(names, ["mindie_session", "mindie_run", "mindie_execution", "mindie_finish", "mindie_message"])
        first = client.rpc("tools/call", {"name": "mindie_session", "arguments": {"context_file": self.registry.context_file}})
        second = client.rpc("tools/call", {"name": "mindie_session", "arguments": {"context_file": self.registry.context_file}})
        for reply in (first, second):
            self.assertFalse(reply["result"]["isError"], reply)
        self.assertEqual(first["result"]["structuredContent"]["data"]["session"]["id"],
                         second["result"]["structuredContent"]["data"]["session"]["id"])
        run = client.rpc("tools/call", {"name": "mindie_run", "arguments": {"context_file": self.registry.context_file,
                                                                          "request_id": "r1", "command": "true"}})
        self.assertTrue(run["result"]["isError"])
        self.assertEqual(run["result"]["structuredContent"]["status"], "unavailable")
        unknown = client.rpc("tools/call", {"name": "remote_bash", "arguments": {}})
        self.assertEqual(unknown["error"]["code"], -32602)
        code, stderr = client.close()
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")

    def test_content_length_framed_session(self):
        self.session(framed=True)

    def test_newline_delimited_session_as_native_clients_speak_it(self):
        self.session(framed=False)

    def test_malformed_frames_get_json_rpc_errors_and_the_session_continues(self):
        client = StdioClient(True, self.env)
        client.proc.stdin.write(b"Content-Length: 5\r\n\r\n{nope")
        client.proc.stdin.flush()
        self.assertEqual(client.receive()["error"]["code"], -32700)
        client.proc.stdin.write(b"Content-Length: 2\r\n\r\n[]")
        client.proc.stdin.flush()
        self.assertEqual(client.receive()["error"]["code"], -32600)
        self.assertEqual(client.rpc("ping")["result"], {})
        code, stderr = client.close()
        self.assertEqual(code, 0, stderr)


class TransportUnitTests(unittest.TestCase):
    def test_framing_is_decided_by_the_first_line_and_mirrored_in_replies(self):
        import io
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode()
        reader = io.BytesIO(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        writer = io.BytesIO()
        transport = StdioTransport(reader, writer)
        self.assertEqual(json.loads(transport.receive()), json.loads(body))
        transport.send({"ok": True})
        self.assertTrue(writer.getvalue().startswith(b"Content-Length: "))
        self.assertIsNone(transport.receive())
        reader = io.BytesIO(b"\n" + body + b"\n")
        writer = io.BytesIO()
        transport = StdioTransport(reader, writer)
        self.assertEqual(json.loads(transport.receive()), json.loads(body))
        transport.send({"ok": True})
        self.assertEqual(writer.getvalue(), b'{"ok":true}\n')

    def test_a_frame_without_content_length_is_a_protocol_error(self):
        import io
        transport = StdioTransport(io.BytesIO(b"Content-Type: x\r\n\r\n{}"), io.BytesIO())
        transport.framed = True
        with self.assertRaisesRegex(task_server.ProtocolError, "Content-Length"):
            transport.receive()


if __name__ == "__main__":
    unittest.main()
