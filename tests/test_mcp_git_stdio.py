"""Real stdio reader and worker Git processes must not share the input pipe."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

from mindie_coordinator.agent_session import AgentSessions


def test_worker_binds_and_captures_sources_while_mcp_reader_waits(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        '[project]\nname="fixture"\ndynamic=["version"]\n', encoding="utf-8")
    (repo / "kernel.cpp").write_text("base\n", encoding="utf-8")
    for args in (["init"], ["config", "user.name", "Test"],
                 ["config", "user.email", "test@example.invalid"],
                 ["add", "."], ["commit", "-m", "base"]):
        subprocess.run(["git", "-C", str(repo), *args], stdin=subprocess.DEVNULL,
                       check=True, capture_output=True, timeout=10)
    (repo / "kernel.cpp").write_text("edited\n", encoding="utf-8")
    state = tmp_path / "sessions"
    context = AgentSessions(state).attach("codex", "stdio-git", str(tmp_path))
    environment = {key: value for key, value in os.environ.items() if not key.startswith("MINDIE_")}
    environment.update(MINDIE_AGENT_SESSIONS_DIR=str(state),
                       MINDIE_DIAGNOSTICS_ROOT=str(tmp_path / "diagnostics"),
                       MINDIE_COORDINATOR_STATE_DIR=str(tmp_path / "coordinator"))
    # Keep real TaskClient input capture, but never start a daemon or contact a
    # host. Reaching the admission boundary proves every local Git call returned.
    code = """
from mindie_coordinator.task_client import TaskClient
from mindie_coordinator.task_server import main
def offline(self):
    import subprocess, sys
    echo = [sys.executable, '-c', 'import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())']
    assert subprocess.run(echo, capture_output=True, check=True, timeout=5).stdout == b''
    assert subprocess.run(echo, input=b'explicit input', capture_output=True,
                          check=True, timeout=5).stdout == b'explicit input'
    raise RuntimeError('captured inputs; admission disabled by test')
TaskClient.coordinator = property(offline)
raise SystemExit(main())
"""
    process = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
    replies = queue.Queue()
    def read():
        for line in process.stdout:
            replies.put(json.loads(line))
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    def rpc(identifier, method, params=None):
        process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": identifier,
            "method": method, "params": params or {}}).encode() + b"\n")
        process.stdin.flush()
        # Leave stdin open and send no follow-up message while the worker runs.
        # Inheriting this pipe deadlocks Git for Windows against the MCP reader.
        response = replies.get(timeout=10)
        assert response["id"] == identifier, response
        return response
    try:
        assert "result" in rpc(1, "initialize")
        params = {"context_file": context["context_file"], "sources": {"vllm": str(repo)}}
        session = rpc(2, "tools/call", {"name": "mindie_session", "arguments": params})
        assert not session["result"]["isError"], session
        run = rpc(3, "tools/call", {"name": "mindie_run", "arguments": {**params, "command": "true"}})
        assert "captured inputs; admission disabled" in run["result"]["structuredContent"]["summary"], run
        snapshots = [json.loads(path.read_text(encoding="utf-8")) for path in (state / "source-inputs").glob("*.json")]
        snapshot = next(item for item in snapshots if item.get("schema_version") == "mindie.execution-sources.v1")
        record = snapshot["records"][0]
        assert record["changed_paths"] == ["kernel.cpp"]
        assert record["build_inputs"]["native"]
        assert record["scm_version"]
        assert rpc(4, "ping")["result"] == {}
    finally:
        # EOF also releases an inherited Git input pipe on the failing version.
        process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True, timeout=10)
            else:
                process.kill()
            process.wait(timeout=5)
        reader.join(timeout=5)
        errors = process.stderr.read().decode("utf-8", errors="replace")
        process.stdout.close()
        process.stderr.close()
    assert process.returncode == 0, errors
    assert errors == ""
