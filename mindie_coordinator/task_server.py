"""Stdio MCP server for task lifecycle and optional coordination messages.

`mindie_session`, `mindie_run`, `mindie_execution` and `mindie_finish` are coordinator
semantics. This process is their home. It is local-first: `mindie_session` and
`mindie_finish` need no remote resources; `mindie_run` uses this process's own
runtime pool.

Wire contract: JSON-RPC 2.0 over stdio. The first bytes decide the framing
for the whole session: Content-Length framed or one JSON object per line.
Standard library only for the transport; results use `remote_dev.result`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, BinaryIO
from mindie_diagnostics import get_recorder, wrap_context, bind_context
from remote_dev.observability import protocol_streams, observed_tool

from mindie_coordinator.host_queue import SCHEMA_VERSION
from mindie_coordinator.agent_session import AgentSessions
from mindie_coordinator.ops import TOOL_DESCRIPTIONS, TOOL_SCHEMAS, mindie_call, LOADED_RUNTIMES

SERVICE_NAME = "mindie-coordinator-task"
PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")
ALIASES = {name.replace(".", "_"): name for name in TOOL_SCHEMAS}
INSTRUCTIONS = (
    "MindIE task tools. Native hooks automatically attach the task; no session-creation call is needed. "
    "Supported tool hooks supply context_file; otherwise use the native hook's context. "
    "Never guess a task from cwd or history. Sessions with no managed "
    "hosts stay local. Normal run/status calls receive coordination messages "
    "opportunistically; use mindie_message only for a substantive request or reply."
)


def package_version() -> str:
    return LOADED_RUNTIMES[0]["version"] or "unknown"


class ProtocolError(ValueError):
    """A malformed frame or message that gets a JSON-RPC error, not a crash."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def capabilities() -> dict[str, Any]:
    return {
        "tools": {"listChanged": False},
        "experimental": {
            SERVICE_NAME: {
                "version": package_version(),
                "host_protocol_schema_version": SCHEMA_VERSION,
                "tools": sorted(ALIASES),
            }
        },
    }


def server_info() -> dict[str, Any]:
    return {"name": SERVICE_NAME, "version": package_version()}


def initialize_result(params: dict[str, Any]) -> dict[str, Any]:
    requested = params.get("protocolVersion")
    return {
        "protocolVersion": requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[-1],
        "capabilities": capabilities(),
        "serverInfo": server_info(),
        "instructions": INSTRUCTIONS,
    }


def list_tools() -> list[dict[str, Any]]:
    return [
        {"name": alias, "description": TOOL_DESCRIPTIONS[name], "inputSchema": TOOL_SCHEMAS[name]}
        for alias, name in ALIASES.items()
    ]


def canonical_name(name: str) -> str:
    return ALIASES.get(name, name)


def call_tool(name: str, arguments: dict[str, Any] | None, metadata: dict | None = None) -> dict[str, Any]:
    with bind_context((metadata or {}).get("mindie_diagnostics") or {}):
        return _call_tool(name, arguments, metadata)


@observed_tool(lambda name, arguments, metadata=None: canonical_name(name), component="mindie-coordinator")
def _call_tool(name, arguments, metadata=None):
    canonical = canonical_name(name)
    if canonical not in TOOL_SCHEMAS:
        raise ProtocolError(-32602, f"unknown task tool: {name}")
    # A persistent MCP server's environment can outlive the native caller.
    # It must use the caller's context, never the thread that launched it.
    arguments = dict(arguments or {})
    if metadata and "kimi_code/session_id" in metadata:
        native = metadata["kimi_code/session_id"]
        agent = metadata.get("kimi_code/agent_id", "")
        if not isinstance(native, str) or not native.strip() or not isinstance(agent, str):
            raise ValueError("invalid native Kimi call identity")
        if agent == "main":
            agent = ""
        context = AgentSessions().native_context("kimi", native, agent)
        supplied = arguments.get("context_file")
        if supplied and supplied != context["context_file"]:
            raise ValueError("context_file differs from this native Kimi caller")
        arguments["context_file"] = context["context_file"]
    elif metadata and "x-codex-turn-metadata" in metadata:
        # Native Codex tools/call supplies this object per invocation, including
        # calls inside functions.exec which do not pass through PreToolUse.
        turn = metadata["x-codex-turn-metadata"]
        native = turn.get("thread_id") if isinstance(turn, dict) else None
        if not isinstance(native, str) or not native.strip():
            raise ValueError("invalid native Codex call identity")
        context = AgentSessions().native_context("codex", native)
        supplied = arguments.get("context_file")
        if supplied and supplied != context["context_file"]:
            raise ValueError("context_file differs from this native Codex caller")
        arguments["context_file"] = context["context_file"]
    payload = mindie_call(canonical, arguments, allow_native_context=False)
    result = payload["result"]
    return {
        # Some native clients expose text only, including Grok 1.0.25.
        # Keep the same compact/full facts visible through both MCP channels.
        "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
        "structuredContent": result,
        "isError": result.get("outcome") not in {"success", "cancelled"},
    }


def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params") or {}
    if request_id is None and isinstance(method, str) and method.startswith("notifications/"):
        return None
    if not isinstance(method, str):
        return _error(request_id, -32600, "request must carry a string method")
    try:
        if method == "initialize":
            return _result(request_id, initialize_result(params))
        if method == "ping":
            return _result(request_id, {})
        if method == "tools/list":
            return _result(request_id, {"tools": list_tools()})
        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments")
            arguments = {} if arguments is None else arguments
            if not isinstance(name, str):
                raise ValueError("tools/call requires a string name")
            if not isinstance(arguments, dict):
                raise ValueError("tools/call arguments must be an object")
            metadata = params.get("_meta")
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError("tools/call _meta must be an object")
            return _result(request_id, call_tool(name, arguments, metadata))
        return _error(request_id, -32601, f"method not found: {method}")
    except ProtocolError as exc:
        return _error(request_id, exc.code, str(exc))
    except Exception as exc:  # noqa: BLE001
        from remote_dev.core.errors import error_details
        return _error(request_id, -32000, str(exc), {**error_details(exc),
            **({"diagnostics": exc.diagnostics} if hasattr(exc, "diagnostics") else {})})


def _result(request_id: Any, value: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": value}


def _error(request_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
    if data is not None:
        payload["error"]["data"] = data
    return payload


class StdioTransport:
    def __init__(self, reader: BinaryIO, writer: BinaryIO):
        self.reader = reader
        self.writer = writer
        self.framed: bool | None = None
        self._write_lock = threading.Lock()

    def send(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        with self._write_lock:
            if self.framed:
                self.writer.write(f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii"))
                self.writer.write(encoded)
            else:
                self.writer.write(encoded + b"\n")
            self.writer.flush()

    def _read_exact(self, length: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < length:
            chunk = self.reader.read(length - len(chunks))
            if not chunk:
                break
            chunks.extend(chunk)
        return bytes(chunks)

    def receive(self) -> bytes | None:
        line = self.reader.readline()
        if not line:
            return None
        if self.framed is None:
            self.framed = line.lower().startswith(b"content-length:")
        if not self.framed:
            while not line.strip():
                line = self.reader.readline()
                if not line:
                    return None
            return line
        headers: dict[str, str] = {}
        while line not in {b"\r\n", b"\n"}:
            text = line.decode("ascii", errors="replace").strip()
            if ":" in text:
                key, value = text.split(":", 1)
                headers[key.lower()] = value.strip()
            line = self.reader.readline()
            if not line:
                return None
        try:
            length = int(headers["content-length"])
        except (KeyError, ValueError) as exc:
            raise ProtocolError(-32600, "missing or invalid Content-Length header") from exc
        return self._read_exact(length)

    def messages(self):
        while True:
            try:
                body = self.receive()
            except ProtocolError as exc:
                self.send(_error(None, exc.code, str(exc)))
                continue
            if body is None:
                return
            try:
                message = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self.send(_error(None, -32700, f"parse error: {exc}"))
                continue
            if not isinstance(message, dict):
                self.send(_error(None, -32600, "request must be an object"))
                continue
            yield message


def serve(reader: BinaryIO, writer: BinaryIO) -> int:
    transport = StdioTransport(reader, writer)
    def respond(message):
        response = handle(message)
        if response is not None:
            transport.send(response)
    # Separate capacity keeps stop/status usable even when all wait workers
    # are occupied. JSON-RPC responses are correlated by id, not arrival order.
    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="mindie-mcp-wait") as waits, \
            ThreadPoolExecutor(max_workers=4, thread_name_prefix="mindie-mcp-control") as controls:
        for message in transport.messages():
            if message.get("method") != "tools/call":
                respond(message)
                continue
            params = message.get("params") or {}
            args = (params.get("arguments") or {}) if isinstance(params, dict) else {}
            raw_name = params.get("name") if isinstance(params, dict) else None
            name = canonical_name(raw_name) if isinstance(raw_name, str) else ""
            waiting = (name == "mindie.run" or (name == "mindie.execution" and isinstance(args, dict)
                                              and args.get("action") == "wait"))
            (waits if waiting else controls).submit(wrap_context(respond), message)
    return 0


def describe() -> dict[str, Any]:
    return {"serverInfo": server_info(), "capabilities": capabilities(), "tools": list_tools()}


def main(argv: list[str] | None = None) -> int:
    from mindie_coordinator._stdio import configure_stdio
    configure_stdio()
    parser = argparse.ArgumentParser(
        description="Serve the MindIE task tools over stdio (JSON-RPC 2.0, "
                    "Content-Length framed or newline-delimited)."
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="print the capability declaration and tool list as JSON, then exit",
    )
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps(describe(), ensure_ascii=False, indent=2))
        return 0
    # Reserve the protocol pipe for this reader before any read or worker
    # starts. Children (including third-party Git/version helpers) must not
    # inherit RPC stdin: on Windows, their startup can block behind its read.
    # os.dup creates a non-inheritable descriptor; dup2 also updates Windows'
    # standard-input handle. Explicit subprocess input still uses its own pipe.
    with protocol_streams() as (reader, writer):
        return serve(reader, writer)


if __name__ == "__main__":
    raise SystemExit(main())
