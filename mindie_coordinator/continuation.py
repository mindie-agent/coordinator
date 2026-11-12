"""Actionable continuation hints for a failed coordinator helper.

[Role] Retain existing execution references and expose native/script routes.
[Invariants] Hints do not launch work, grant leases or clear uncertain effects.
[Timeouts & Locks] Pure projection of one result; no waits or locks.
[Side-Effect Policy] No writes or retries; the caller chooses its next action.
"""


def continuation(name, arguments, value=None, details=None):
    value = value or {}
    execution_id = value.get("execution_id") or value.get("run_id")
    if not execution_id and name == "mindie.execution":
        execution_id = arguments.get("execution_id") or arguments.get("run_id")
    context = {"context_file": arguments["context_file"]} if arguments.get("context_file") else {}
    result = {
        "suggested_action": "continue_with_native_tools_or_script",
        "message": (
            "This helper failure does not block independent task work. Use existing "
            "CLIs or task-local scripts through remote-dev or authorized native SSH. "
            "Retain source/interpreter choices and confirmed results. Device work "
            "still needs its actual lease or verified reservation; uncertain writes "
            "and live jobs must be observed before any replacement or retry. "
            "Independent hosts/probe/container/lease operations need no managed task "
            "binding; omit context_file when using those environment interfaces."
        ),
        "environment_operations": [
            {"tool": "mindie.hosts", "arguments": {"action": "list"}},
            {"tool": "mindie.probe", "arguments": {"action": "hosts"}},
        ],
        "custom_script": {"tool": "mindie.run", "parameter": "script_file",
                          "scope": "new managed executions after existing work is reconciled"},
        "native_tools": ["remote.bash", "native_shell_or_ssh"],
    }
    if (details or {}).get("error_code") == "managed_task_context_required":
        result.update(
            suggested_action="continue_independent_environment_work",
            submission_state="not_sent",
            task_context={
                "reason": details["reason"],
                "recovery": "Use a registered context_file belonging to this native caller, "
                            "or repair its native task hook for managed execution.",
                "knowledge_activation_repairs_binding": False,
                "repeat_activation_required": False,
            },
            environment_routes=[
                {"tool": "mindie.container", "action": "inspect", "requires": ["host_or_alias"],
                 "omit": ["context_file"]},
                {"tool": "mindie.lease", "action": "status", "requires": ["host_or_alias"],
                 "omit": ["context_file"]},
            ],
        )
        result["custom_script"] = {
            "tool": "remote.bash", "scope": "authorized existing endpoint and source directory",
            "device_work": "Acquire and retain the actual host lease before device execution.",
        }
    if (details or {}).get("submission_state") == "uncertain":
        result.update(submission_state="uncertain",
                      suggested_action="inspect_existing_task_before_replacement",
                      observe={"tool": "mindie.session", "arguments": context})
    if execution_id:
        result["execution_id"] = execution_id
        result["observe"] = {"tool": "mindie.execution", "arguments": {
            **context, "execution_id": execution_id, "action": "status"}}
        result["inspect_target"] = {"tool": "mindie.execution", "arguments": {
            **context, "execution_id": execution_id, "action": "target"},
            "when": "a runtime binding is available"}
        # A failed process can still hold resources. Only observed release
        # supports an immediate replacement; missing facts remain visible.
        result["resources_released"] = value.get("resources_released")
        if value.get("resources_released") is not True:
            result["suggested_action"] = "observe_existing_execution_before_replacement"
    return result
