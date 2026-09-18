"""Task-facing operations exposed identically to every MCP host.

Task lifecycle and optional coordination messages are coordinator semantics.
Clients pass command and experiment needs; this
package owns placement, environment, and recovery.
"""

from __future__ import annotations

import time

from remote_dev.result import make_result
from remote_dev.observability import observed_tool
from remote_dev.core.errors import error_details, caller_error
from remote_dev.runtime import process_identity, runtime_status
from mindie_coordinator.presentation import present
from mindie_coordinator.placement import ENVIRONMENT_KEYS
from mindie_coordinator.state_paths import coordinator_state_dir

LOADED_RUNTIMES = [process_identity(name) for name in ("mindie-coordinator", "remote-dev")]

TOOL_DESCRIPTIONS = {
    "mindie.session": "Inspect this native session's MindIE task or replace source defaults for future submissions. No machine is required. Tasks with known managed hosts also receive cached coordination messages without waiting for remote polling. Active executions retain their submitted inputs.",
    "mindie.run": "Submit shell command or a local UTF-8 script_file with fixed sources and environment/resource/topology needs. Use wait_until=running or released and wait_timeout_seconds (0-600) for an owner-side bounded wait; timeout returns the same execution_id without stopping or resubmitting. Terminal waits include logs. Omitted sources uses task defaults; {} has no source dependencies. Devices default to zero. Explicit allow_external_busy shares one named device with external processes, retaining managed ownership.",
    "mindie.execution": "Observe one owned execution_id or task-scoped service: action=status, wait, tail, evidence, stop or target. wait uses until=running or released, timeout_seconds=0-600; timeout returns the same reference and does not stop or resubmit. Terminal wait includes logs. evidence reads retained source/preparation/build facts and refs, optionally filtered by section and artifact path substring; it does not revalidate remote files. Stop retains the container and source roots.",
    "mindie.finish": "Finish this MindIE task by closing admission and stopping owned executions; the coordinator completes cleanup and returns leases. Preserve the container, worktrees and evidence.",
    "mindie.message": "Send coordination text to a reference returned by run/status (coordination_peers[].reference), or reply using notifications[].reply_reference. Sender, host and thread are filled internally. Messages never execute commands or transfer resource ownership; normal run/status calls receive replies automatically.",
}


def task_schema(properties: dict, required: tuple[str, ...] = ()) -> dict:
    return {"type": "object",
            "properties": {"context_file": {"type": "string", "description": "Local task context supplied by the native session hook; never guess from cwd or newest history."},
                           "full": {"type": "boolean", "default": False, "description": "Return the full operation record instead of its compact observation."},
                           **properties},
            "required": list(required), "additionalProperties": False}


RESOURCE_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "devices": {"type": "array", "items": {"type": "integer", "minimum": 0}, "uniqueItems": True},
    "npu_count": {"type": "integer", "minimum": 0, "description": "Number of NPUs; defaults to zero. If devices is also supplied, this must equal its length."},
    "allow_external_busy": {"type": "boolean", "default": False, "description": "Explicitly permit external occupancy on exactly one named physical device. Requires devices=[id]; other managed leases, holds and owned process/port cleanup still apply."},
    "service_port": {"type": "integer", "minimum": 0, "maximum": 65535}}}
RESOURCE_SCHEMA["allOf"] = [{
    "if": {"required": ["allow_external_busy"], "properties": {"allow_external_busy": {"const": True}}},
    "then": {"required": ["devices"], "properties": {"devices": {"minItems": 1, "maxItems": 1}}},
}]
ROLE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["name"], "properties": {
    **RESOURCE_SCHEMA["properties"], "name": {"type": "string", "minLength": 1},
    "host": {"type": "string", "minLength": 1}, "command": {"type": "string", "minLength": 1},
    "preflight": {"type": "string", "minLength": 1},
    "env": {"type": "object", "additionalProperties": {"type": "string"}}}}


TOOL_SCHEMAS = {
    "mindie.session": task_schema({"sources": {"type": "object", "additionalProperties": {"type": "string"}}}),
    "mindie.run": task_schema({
        "command": {"type": "string"},
        "script_file": {"type": "string", "description": "Local UTF-8 shell file, up to 1 MiB, captured once as the command. Mutually exclusive with command."},
        "wait_until": {"type": "string", "enum": ["running", "released"]},
        "wait_timeout_seconds": {"type": "number", "minimum": 0, "maximum": 600, "default": 30},
        "sources": {"type": "object", "additionalProperties": {"type": "string"}, "description": "Actual worktrees to capture once. Omit to use explicit task defaults or this attachment's automatic cwd binding; {} selects no sources."},
        "preflight": {"type": "string", "description": "Optional validation command in the prepared root before NPU allocation. It must not require devices or start a service."},
        "env": {"type": "object", "additionalProperties": {"type": "string"}},
        "environment": {"type": "object", "additionalProperties": False, "properties": {
            key: {"type": "string", "minLength": 1} for key in sorted(ENVIRONMENT_KEYS)}},
        "resources": RESOURCE_SCHEMA,
        "topology": {"type": "object", "additionalProperties": False, "properties": {
            "host": {"type": "string", "minLength": 1, "description": "Host constraint for one default role; use roles[].host for a role group."},
            "roles": {"type": "array", "minItems": 1, "items": ROLE_SCHEMA},
            "distinct_hosts": {"type": "boolean"}}, "not": {"required": ["host", "roles"]}},
        "timeout_seconds": {"type": ["integer", "null"], "default": 1800},
        "service": {"type": ["string", "null"], "description": "Ensure a task-scoped service with identical fixed sources and configuration. Changed inputs require restart. To connect without capture, use mindie_execution(service=...)."},
        "restart": {"type": "boolean", "description": "Replace a live named service, including one with the same spec"},
    }),
    "mindie.execution": task_schema({"execution_id": {"type": "string"}, "service": {"type": "string", "description": "Task-scoped service name, mutually exclusive with execution_id"}, "action": {"type": "string", "enum": ["status", "wait", "evidence", "tail", "stop", "target"]}, "force": {"type": "boolean"}, "refresh": {"type": "boolean", "default": False, "description": "Refresh remote status instead of reusing the last snapshot for up to two seconds. Busy executions return cache age and refresh_deferred."}, "role": {"type": "string", "description": "Optional topology role name"},
        "until": {"type": "string", "enum": ["running", "released"], "default": "released"},
        "timeout_seconds": {"type": "number", "minimum": 0, "maximum": 600, "default": 30},
        "section": {"type": "string", "enum": ["all", "sources", "preparation", "build", "diagnostics"], "default": "all",
                    "description": "diagnostics exports a bounded redacted attachment from this owned execution's local records and returns its file ref; no remote probe or global state scan"},
        "path": {"type": "string", "description": "Optional artifact path substring for retained profile hashes"}}),
    "mindie.finish": task_schema({"force": {"type": "boolean"}}),
    "mindie.message": task_schema({"recipient": {"type": "object", "description": "Use an existing coordination reference or reply_reference unchanged."},
                                   "text": {"type": "string", "minLength": 1, "maxLength": 4000}}, ("recipient", "text")),
}
TOOL_SCHEMAS["mindie.run"]["oneOf"] = [
    {"required": ["command"], "not": {"required": ["script_file"]}},
    {"required": ["script_file"], "not": {"required": ["command"]}},
]
TOOL_SCHEMAS["mindie.execution"]["oneOf"] = [
    {"required": ["execution_id"], "not": {"required": ["service"]}},
    {"required": ["service"], "not": {"required": ["execution_id"]}},
]


@observed_tool(lambda name, args, **kwargs: name, component="mindie-coordinator")
def mindie_call(name, args, *, allow_native_context=True):
    started = time.monotonic()
    target = {"kind": "mindie-task"}
    client = None
    try:
        if name not in TOOL_SCHEMAS:
            raise caller_error("unknown MindIE operation")
        unknown = set(args) - set(TOOL_SCHEMAS[name]["properties"])
        if unknown:
            raise caller_error("unsupported fields for " + name + ": " + ", ".join(sorted(unknown)))
        from mindie_coordinator.task_client import TaskClient
        client = TaskClient(args.get("context_file", ""), allow_native_context=allow_native_context)
        target["session_id"] = client.context["session"]["id"]
        if name == "mindie.session":
            if "sources" in args:
                client.sources(args["sources"])
            value = client.status()
            status = value["session"]["state"]
        elif name == "mindie.run":
            keys = ("command", "sources", "env", "environment", "resources", "topology",
                    "timeout_seconds", "service", "restart", "preflight", "script_file", "wait_until", "wait_timeout_seconds")
            value = client.run(**{key: args[key] for key in keys if key in args})
            status = value["state"]
        elif name == "mindie.execution":
            value = client.observe(args.get("execution_id"), args.get("action", "status"),
                                   args.get("force", False), role=args.get("role"), refresh=bool(args.get("refresh")),
                                   **{key: args[key] for key in ("until", "timeout_seconds", "section", "path") if key in args},
                                   **({"service": args["service"]} if "service" in args else {}))
            status = value["state"]
        elif name == "mindie.finish":
            value = client.finish(args.get("force", False))
            status = value["state"]
        elif name == "mindie.message":
            value = client.message(args.get("recipient"), args.get("text"))
            status = value["state"]
        else:
            raise ValueError("unknown MindIE operation")
        if status in {"failed", "inconclusive"}:
            outcome = "failed"
        elif status == "timeout":
            outcome = "timeout"
        elif status == "cancelled":
            outcome = "cancelled"
        elif status in {"uncertain", "waiting_for_runtime", "needs_runtime_update"}:
            outcome = "blocked"
        elif status in {"queued", "preparing", "waiting"} and not value.get("execution_id"):
            outcome = "blocked"
        else:
            # Admission and successful observation are completed tool calls.
            # A durable execution can still be queued or preparing; reporting
            # that normal progress as an MCP error makes clients retry work.
            outcome = "success"
        if name == "mindie.finish" and outcome == "success" and status != "finished":
            outcome = "blocked"
        result = make_result(tool=name, target=target, outcome=outcome, status=status,
                             summary="MindIE " + status.replace("_", " "),
                             duration_ms=int((time.monotonic() - started) * 1000), extra={"data": value})
    except Exception as exc:
        result = make_result(tool=name, target=target, outcome="blocked", status="unavailable",
                             summary=str(exc), duration_ms=int((time.monotonic() - started) * 1000),
                             warnings=["Local file and shell tools remain available. No remote success is implied."],
                             extra={"error_details": error_details(exc)})
    result["runtime"] = {"client": [runtime_status(item) for item in LOADED_RUNTIMES]}
    if client is not None:
        service = client._service
        if service is not None and getattr(service, "runtime", None):
            result["runtime"]["daemon"] = service.runtime
        result = present(result, coordinator_state_dir(client.store.state_dir),
                         full=bool(args.get("full")), target=args.get("action") == "target")
    return {"text": result["summary"], "result": result}
