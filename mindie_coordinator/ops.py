"""Task-facing operations exposed identically to every MCP host.

[Role] Validates MCP tool arguments and formats `CoordinatorService` results via `presentation.present`.
[Invariants] Returns structured `make_result` payloads; caller errors map to `isError=True` with actionable remedies.
[Timeouts & Locks] Enforces `wait_timeout_seconds` (`0..600s`) before delegating to `TaskClient`/`CoordinatorService`.
[Side-Effect Policy] Delegates all session and execution mutations to `TaskClient` and `CoordinatorService`.
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
from mindie_coordinator.topology_gang import PLACEMENT_POLICIES, SCALES

LOADED_RUNTIMES = [process_identity(name) for name in ("mindie-coordinator", "remote-dev")]

TOOL_DESCRIPTIONS = {
    "mindie.session": "Inspect this native session's MindIE task or replace source defaults for future submissions. No machine is required. Tasks with known managed hosts also receive cached coordination messages without waiting for remote polling. Active executions retain their submitted inputs.",
    "mindie.run": "Submit shell command or a local UTF-8 script_file with fixed sources and environment/resource/topology needs. Use wait_until=running or released and wait_timeout_seconds (0-600) for an owner-side bounded wait; timeout returns the same execution_id without stopping or resubmitting. Terminal waits include logs. Omitted sources uses task defaults; {} has no source dependencies. Devices default to zero. Explicit allow_external_busy shares one named device with external processes, retaining managed ownership.",
    "mindie.execution": "Observe one owned execution_id or task-scoped service: action=status, wait, tail, evidence, stop or target. wait uses until=running or released, timeout_seconds=0-600; timeout returns the same reference and does not stop or resubmit. Terminal wait includes logs. evidence reads retained source/preparation/build facts and refs, optionally filtered by section and artifact path substring; it does not revalidate remote files. Stop retains the container and source roots.",
    "mindie.finish": "Finish this MindIE task by closing admission and stopping owned executions; the coordinator completes cleanup and returns leases. Preserve the container, worktrees and evidence.",
    "mindie.message": "Send coordination text to a reference returned by run/status (coordination_peers[].reference), or reply using notifications[].reply_reference. Sender, host and thread are filled internally. Messages never execute commands or transfer resource ownership; normal run/status calls receive replies automatically.",
    "mindie.hosts": "Manage SSH hosts, key bootstrap, multi-node SSH mesh, and remote-dev endpoint aliases in coordinator's shared MachineDirectory (machines.json -> endpoints.json). Actions: list, upsert, remove, set_default, bootstrap_key, mesh.",
    "mindie.probe": "Probe Ascend NPU occupancy/HBM/AICore/processes, find fleet NPU capacity, or inspect host CPU/RAM/Docker containers and shared weight mounts over agentless SSH. Actions: hosts, capacity, npu, server, mounts.",
    "mindie.container": "Bind an existing Docker container or SSH port to a host alias, provision a per-user Ascend container (mindie-<user>), inspect running containers, or remove/unbind a container while syncing remote-dev endpoints.json. Actions: bind, provision, inspect, remove.",
    "mindie.lease": "Acquire, heartbeat, inspect, or release a lightweight bare-metal NPU and service-port lease on a host via HostQueue, returning ASCEND_RT_VISIBLE_DEVICES and env ready for remote-dev execution. Actions: status, acquire, heartbeat, release.",
}

ENV_TOOLS = frozenset({"mindie.hosts", "mindie.probe", "mindie.container", "mindie.lease"})


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
            "distinct_hosts": {"type": "boolean"},
            "scale": {"type": "string", "enum": list(SCALES)},
            "die_count": {"type": "integer", "minimum": 1, "maximum": 32},
            "placement_policy": {"type": "string", "enum": list(PLACEMENT_POLICIES)},
            "prefill_die_count": {"type": "integer", "minimum": 1},
            "decode_die_count": {"type": "integer", "minimum": 1},
            "node_count": {"type": "integer", "minimum": 1},
            "visible_phy_ids": {"type": "array", "items": {"type": "integer", "minimum": 0},
                                "minItems": 1, "maxItems": 32, "uniqueItems": True}},
         "not": {"required": ["host", "roles"]}},
        "timeout_seconds": {"type": ["integer", "null"], "default": 1800},
        "service": {"type": ["string", "null"], "description": "Ensure a task-scoped service with identical fixed sources and configuration. Changed inputs require restart. To connect without capture, use mindie_execution(service=...)."},
        "restart": {"type": "boolean", "description": "Replace a live named service, including one with the same spec"},
    }),
    "mindie.execution": task_schema({"execution_id": {"type": "string"}, "run_id": {"type": "string", "description": "Stable run_id returned at admission; identifies the same execution record."}, "service": {"type": "string", "description": "Task-scoped service name, mutually exclusive with execution_id/run_id"}, "action": {"type": "string", "enum": ["status", "wait", "evidence", "tail", "stop", "target"]}, "force": {"type": "boolean"}, "refresh": {"type": "boolean", "default": False, "description": "Refresh remote status instead of reusing the last snapshot for up to two seconds. Busy executions return cache age and refresh_deferred."}, "role": {"type": "string", "description": "Optional topology role name"},
        "until": {"type": "string", "enum": ["running", "released"], "default": "released"},
        "timeout_seconds": {"type": "number", "minimum": 0, "maximum": 600, "default": 30},
        "section": {"type": "string", "enum": ["all", "sources", "preparation", "build", "diagnostics"], "default": "all",
                    "description": "diagnostics exports a bounded redacted attachment from this owned execution's local records and returns its file ref; no remote probe or global state scan"},
        "path": {"type": "string", "description": "Optional artifact path substring for retained profile hashes"}}),
    "mindie.finish": task_schema({"force": {"type": "boolean"}}),
    "mindie.message": task_schema({"recipient": {"type": "object", "description": "Use an existing coordination reference or reply_reference unchanged."},
                                   "text": {"type": "string", "minLength": 1, "maxLength": 4000}}, ("recipient", "text")),
    "mindie.hosts": task_schema({
        "action": {"type": "string", "enum": ["list", "upsert", "add", "remove", "set_default", "bootstrap_key", "mesh"], "default": "list"},
        "alias": {"type": "string", "minLength": 1, "description": "Short machine alias exposed to remote-dev (defaults to host IP)."},
        "host": {"type": "string", "minLength": 1, "description": "Host IP address, DNS hostname, or registered alias."},
        "port": {"type": "integer", "minimum": 1, "maximum": 65535, "default": 22},
        "user": {"type": "string", "minLength": 1, "default": "root"},
        "identity_file": {"type": "string", "description": "Optional SSH private key path for this host."},
        "workdir": {"type": "string", "description": "Default bare-metal working directory on the host."},
        "machine_type": {"type": "string", "enum": ["A2", "A3", "A5", "310P"]},
        "container_name": {"type": "string", "description": "Optional bound container name on this host."},
        "container_ssh_port": {"type": "integer", "minimum": 1, "maximum": 65535},
        "container_user": {"type": "string", "default": "root"},
        "container_workdir": {"type": "string", "default": "/vllm-workspace"},
        "container_mode": {"type": "string", "enum": ["ssh", "docker_exec"]},
        "set_default": {"type": "boolean", "default": False, "description": "Mark this machine as the default remote-dev endpoint."},
        "bootstrap_key": {"type": "boolean", "default": False, "description": "Install local SSH public key using password_env before saving."},
        "password_env": {"type": "string", "description": "Environment variable holding the SSH password for non-interactive key bootstrap."},
        "public_key_file": {"type": "string", "description": "Optional local SSH .pub key path for key bootstrap."},
        "probe_hardware": {"type": "boolean", "default": False, "description": "Run a live hardware/NPU probe and store hardware summary in machines.json."},
        "peer_alias": {"type": "string", "description": "Second machine alias for action=mesh."},
        "peer_host": {"type": "string", "description": "Second machine host for action=mesh."},
        "peer_port": {"type": "integer", "minimum": 1, "maximum": 65535, "description": "Second machine container SSH port for action=mesh."},
    }),
    "mindie.probe": task_schema({
        "action": {"type": "string", "enum": ["hosts", "capacity", "npu", "server", "mounts"], "default": "hosts"},
        "alias": {"type": "string", "description": "Optional machine alias to probe (omit to probe all registered hosts)."},
        "host": {"type": "string", "description": "Optional machine alias or host IP to probe."},
        "port": {"type": "integer", "minimum": 1, "maximum": 65535},
        "user": {"type": "string"},
        "identity_file": {"type": "string"},
        "mode": {"type": "string", "enum": ["auto", "cache", "live"], "default": "auto", "description": "auto reuses local probe cache up to 30s; live forces SSH probe; cache reads local snapshot only."},
        "npu_count": {"type": "integer", "minimum": 1, "default": 1, "description": "Required free NPU count when action=capacity."},
        "chip_query": {"type": "string", "description": "Optional chip substring filter (e.g. '910B', '910C', '950') when action=capacity."},
    }),
    "mindie.container": task_schema({
        "action": {"type": "string", "enum": ["bind", "provision", "inspect", "remove"], "default": "inspect"},
        "alias": {"type": "string", "description": "Target machine alias."},
        "host": {"type": "string", "description": "Target machine alias or host IP."},
        "port": {"type": "integer", "minimum": 1, "maximum": 65535, "description": "Host SSH port (default 22)."},
        "user": {"type": "string", "description": "Host SSH user (default root)."},
        "container_name": {"type": "string", "description": "Docker container name to bind or remove."},
        "container_id": {"type": "string", "description": "Optional Docker container ID."},
        "ssh_port": {"type": "integer", "minimum": 1, "maximum": 65535, "description": "Direct container SSH port (when using container sshd)."},
        "container_user": {"type": "string", "default": "root"},
        "workdir": {"type": "string", "default": "/vllm-workspace"},
        "python": {"type": "string", "description": "Optional Python interpreter path inside the container."},
        "mode": {"type": "string", "enum": ["ssh", "docker_exec", "auto", "cache", "live"]},
        "image": {"type": "string", "description": "Image selector for action=provision ('local-latest', 'rc', 'main', 'stable', or full image ref)."},
        "machine_type": {"type": "string", "enum": ["A2", "A3", "A5", "310P"]},
        "user_namespace": {"type": "string", "description": "User namespace for mindie-<user> container provisioning."},
        "password_env": {"type": "string", "description": "Environment variable holding SSH password for initial host bootstrap."},
        "set_default": {"type": "boolean", "default": False},
        "unbind_only": {"type": "boolean", "default": False, "description": "When action=remove, only unbind container from machines.json without stopping the Docker container."},
    }),
    "mindie.lease": task_schema({
        "action": {"type": "string", "enum": ["status", "acquire", "heartbeat", "release"], "default": "status"},
        "alias": {"type": "string", "description": "Target machine alias."},
        "host": {"type": "string", "description": "Target machine alias or host IP."},
        "port": {"type": "integer", "minimum": 1, "maximum": 65535},
        "user": {"type": "string"},
        "lease_id": {"type": "string", "description": "Lease/task identifier (auto-generated on acquire if omitted; required on heartbeat/release)."},
        "owner": {"type": "string", "description": "Lease owner username."},
        "npu_count": {"type": "integer", "minimum": 0, "description": "Number of NPUs to reserve on the host."},
        "devices": {"type": "array", "items": {"type": "integer", "minimum": 0}, "uniqueItems": True, "description": "Explicit physical NPU device IDs to reserve."},
        "allow_external_busy": {"type": "boolean", "default": False},
        "service_port": {"type": "integer", "minimum": 0, "maximum": 65535, "description": "0 reserves a free host port in 18000..18999; positive int reserves that specific port."},
        "purpose": {"type": "string", "description": "Human/agent-readable description of what this lease is running."},
        "ttl_seconds": {"type": "integer", "minimum": 30, "maximum": 86400, "default": 900},
        "fence_token": {"type": "integer", "minimum": 1, "description": "Fence token issued by acquire for heartbeat/release."},
        "coordination_epoch": {"type": "string", "description": "Optional observed host coordination epoch."},
        "remote_state_dir": {"type": "string", "description": "Optional remote host coordinator state directory override."},
    }),
}
TOOL_SCHEMAS["mindie.run"]["oneOf"] = [
    {"required": ["command"], "not": {"required": ["script_file"]}},
    {"required": ["script_file"], "not": {"required": ["command"]}},
]
TOOL_SCHEMAS["mindie.execution"]["oneOf"] = [
    {"required": ["execution_id"], "not": {"anyOf": [{"required": ["service"]}, {"required": ["run_id"]}]}},
    {"required": ["run_id"], "not": {"anyOf": [{"required": ["service"]}, {"required": ["execution_id"]}]}},
    {"required": ["service"], "not": {"anyOf": [{"required": ["execution_id"]}, {"required": ["run_id"]}]}},
]


@observed_tool(lambda name, args, **kwargs: name, component="mindie-coordinator")
def mindie_call(name, args, *, allow_native_context=True):
    started = time.monotonic()
    target = {"kind": "mindie-env" if name in ENV_TOOLS else "mindie-task"}
    client = None
    state_dir = None
    try:
        if name not in TOOL_SCHEMAS:
            raise caller_error("unknown MindIE operation")
        unknown = set(args) - set(TOOL_SCHEMAS[name]["properties"])
        if unknown:
            raise caller_error("unsupported fields for " + name + ": " + ", ".join(sorted(unknown)))
        from mindie_coordinator.task_client import TaskClient
        if name in ENV_TOOLS:
            if args.get("context_file"):
                client = TaskClient(args["context_file"], allow_native_context=allow_native_context)
                target["session_id"] = client.context["session"]["id"]
                state_dir = coordinator_state_dir(client.store.state_dir)
            else:
                state_dir = coordinator_state_dir()
            from mindie_coordinator.env_manager import (
                handle_container,
                handle_hosts,
                handle_lease,
                handle_probe,
            )
            if name == "mindie.hosts":
                value = handle_hosts(args, state_dir=state_dir)
            elif name == "mindie.probe":
                value = handle_probe(args, state_dir=state_dir)
            elif name == "mindie.container":
                value = handle_container(args, state_dir=state_dir)
            elif name == "mindie.lease":
                value = handle_lease(args, state_dir=state_dir)
            status = str(value.get("state") or "ready")
        else:
            client = TaskClient(args.get("context_file", ""), allow_native_context=allow_native_context)
            target["session_id"] = client.context["session"]["id"]
            state_dir = coordinator_state_dir(client.store.state_dir)
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
                if sum(key in args for key in ("execution_id", "run_id", "service")) != 1:
                    raise caller_error("supply exactly one of execution_id, run_id or service")
                value = client.observe(args.get("execution_id") or args.get("run_id"), args.get("action", "status"),
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
        elif status in {"uncertain", "partial", "waiting_for_runtime", "needs_runtime_update", "insufficient_capacity"}:
            outcome = "blocked"
        elif status in {"queued", "preparing", "waiting"} and name not in ENV_TOOLS and not value.get("execution_id"):
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
    if state_dir is not None:
        result = present(result, state_dir,
                         full=bool(args.get("full")), target=args.get("action") == "target")
    return {"text": result["summary"], "result": result}
