#!/usr/bin/env python3
"""MindIE local context and task facade; the same operations are served by task_server.py over MCP stdio."""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

from remote_dev.result import make_result
from remote_dev.observability import observed_tool

from mindie_coordinator.agent_session import CLIENTS, AgentSessions, load_context
from mindie_coordinator.ops import mindie_call
from mindie_coordinator.task_client import TaskClient


@observed_tool(lambda tool, **kwargs: tool, component="mindie-coordinator")
def error_payload(tool: str, *, outcome: str, status: str, error: str) -> dict:
    """Same result contract as the remote-dev CLI wrappers: errors print a
    result JSON (never a traceback) and exit non-zero."""
    result = make_result(
        tool=tool,
        target={"kind": "mindie-task"},
        outcome=outcome,  # type: ignore[arg-type]
        status=status,
        summary=f"{tool} {status}.",
        preview={"stderr": error[-4000:]},
        extra={"error": error, "error_details": {"category": "caller" if status in {"invalid_json", "invalid_wait", "invalid_sources"} else "internal",
                                                "submission_state": "not_sent", "retryable": False}},
    )
    return result


def main():
    from mindie_coordinator._stdio import configure_stdio
    configure_stdio()
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=inspect.cleandoc(TaskClient.__doc__),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="operation", required=True)
    attach = sub.add_parser("attach", help="Adapter entry: native root/resume, child, or explicit task association")
    attach.add_argument("--client", choices=sorted(CLIENTS), required=True)
    attach.add_argument("--native-session-id", required=True)
    attach.add_argument("--cwd", default=str(Path.cwd()))
    attach.add_argument("--parent-context", default="")
    attach.add_argument("--association", default="")
    attach.add_argument("--agent-id", default="")
    for name in ("session", "run", "execution", "finish"):
        child = sub.add_parser(name)
        child.add_argument("--context-file")
        child.add_argument("--full", action="store_true", default=None)
        child.add_argument("--json", default="{}", help="Additional structured tool arguments")
        if name in {"run", "execution"}:
            child.add_argument("--wait", dest="wait_until", choices=["running", "released"],
                               help="Owner-side wait; timeout retains the same execution and does not stop it")
            child.add_argument("--wait-timeout-seconds", type=float, help="Observation budget, 0-600 seconds (default 30)")
        if name == "run":
            command = child.add_mutually_exclusive_group()
            command.add_argument("--command")
            command.add_argument("--script-file", help="Capture a local UTF-8 shell file directly; no runner needed")
            sources = child.add_mutually_exclusive_group()
            sources.add_argument("--source", action="append", metavar="NAME=PATH",
                                 help="Capture an actual worktree for this submission; repeat for multiple repositories")
            sources.add_argument("--no-sources", action="store_true", default=None,
                                 help="Run without source dependencies, ignoring task defaults")
            child.add_argument("--service", default=None)
            child.add_argument("--restart", action="store_true", default=None)
            child.add_argument("--timeout-seconds", type=int)
        if name == "execution":
            reference = child.add_mutually_exclusive_group()
            reference.add_argument("--execution-id")
            reference.add_argument("--service")
            child.add_argument("--action", choices=["status", "wait", "evidence", "tail", "stop", "target"])
            child.add_argument("--section", choices=["all", "sources", "preparation", "build", "diagnostics"])
            child.add_argument("--path", help="Artifact path substring for evidence")
            child.add_argument("--role", default=None)
            child.add_argument("--refresh", action="store_true", default=None,
                               help="Refresh remote status; otherwise reuse a snapshot for up to two seconds")
    args = vars(parser.parse_args())
    operation = args.pop("operation")
    if operation == "attach":
        inherited = args["parent_context"] or args["association"]
        try:
            store = AgentSessions(Path(load_context(inherited)["state_dir"])) if inherited else AgentSessions()
            payload = store.attach(**args)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps(error_payload("mindie.attach", outcome="failed", status="attach_failed", error=f"{type(exc).__name__}: {exc}"), ensure_ascii=False))
            return 1
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    try:
        extra = json.loads(args.pop("json"))
    except json.JSONDecodeError as exc:
        print(json.dumps(error_payload("mindie." + operation, outcome="needs_input", status="invalid_json", error=f"invalid --json: {exc}"), ensure_ascii=False))
        return 1
    # Unset argparse defaults (None) must not silently override --json keys:
    # `--json '{"action":"stop"}'` degraded to a status query otherwise.
    merged = {**extra, **{key: value for key, value in args.items() if value is not None}}
    if operation == "execution":
        if "wait_until" in merged:
            if merged.get("action", "wait") != "wait":
                print(json.dumps(error_payload("mindie.execution", outcome="needs_input", status="invalid_wait",
                                               error="--wait requires action=wait"), ensure_ascii=False))
                return 1
            merged["action"] = "wait"
            merged["until"] = merged.pop("wait_until")
        if "wait_timeout_seconds" in merged:
            merged["timeout_seconds"] = merged.pop("wait_timeout_seconds")
    if operation == "run":
        selected = merged.pop("source", None)
        no_sources = merged.pop("no_sources", None)
        if selected:
            captured = {}
            for item in selected:
                name, separator, path = item.partition("=")
                if not separator or not name or not path or name in captured:
                    print(json.dumps(error_payload("mindie.run", outcome="needs_input", status="invalid_sources",
                                                   error="--source needs unique NAME=PATH entries"), ensure_ascii=False))
                    return 1
                captured[name] = path
            merged["sources"] = captured
        elif no_sources:
            merged["sources"] = {}
    result = mindie_call("mindie." + operation, merged)
    print(json.dumps(result["result"], ensure_ascii=False))
    return 0 if result["result"]["outcome"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
