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
            "and live jobs must be observed before any replacement or retry."
        ),
        "custom_script": {"tool": "mindie.run", "parameter": "script_file",
                          "scope": "new managed executions after existing work is reconciled"},
        "native_tools": ["remote.bash", "native_shell_or_ssh"],
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
