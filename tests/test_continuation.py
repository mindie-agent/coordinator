"""Failure guidance retains the observation route before any replacement job."""
from remote_dev.core.errors import RemoteExecutionError
from mindie_coordinator import ops, task_client
from mindie_coordinator.continuation import continuation


def test_retained_execution_and_unreleased_resources_require_observation():
    result = continuation("mindie.run", {"context_file": "/task/context.json"}, {
        "execution_id": "existing", "state": "failed", "resources_released": False})
    assert result["suggested_action"] == "observe_existing_execution_before_replacement"
    assert result["observe"] == {"tool": "mindie.execution", "arguments": {
        "context_file": "/task/context.json", "execution_id": "existing", "action": "status"}}
    assert result["resources_released"] is False


def test_confirmed_release_exposes_custom_script_without_losing_reference():
    result = continuation("mindie.execution", {"run_id": "old"}, {
        "state": "failed", "resources_released": True})
    assert result["suggested_action"] == "continue_with_native_tools_or_script"
    assert result["execution_id"] == "old"
    assert result["custom_script"]["parameter"] == "script_file"


def test_unknown_submit_observes_task_and_does_not_resubmit(monkeypatch):
    class DisconnectedClient:
        def __init__(self, *args, **kwargs):
            raise RemoteExecutionError("reply lost", submission_state="uncertain")
    monkeypatch.setattr(task_client, "TaskClient", DisconnectedClient)
    result = ops.mindie_call("mindie.run", {"context_file": "/task/context.json", "command": "custom-job"})["result"]
    assert result["outcome"] == "blocked"
    assert result["error_details"]["submission_state"] == "uncertain"
    assert result["next"]["suggested_action"] == "inspect_existing_task_before_replacement"
    assert result["next"]["observe"] == {"tool": "mindie.session", "arguments": {"context_file": "/task/context.json"}}


def test_missing_attachment_exposes_working_environment_continuation(monkeypatch):
    monkeypatch.delenv("MINDIE_CONTEXT_FILE", raising=False)
    result = ops.mindie_call("mindie.session", {}, allow_native_context=False)["result"]
    assert result["outcome"] == "blocked"
    assert result["status"] == "managed_task_context_required"
    assert result["next"]["custom_script"]["tool"] == "remote.bash"
    assert result["next"]["task_context"]["repeat_activation_required"] is False
    for operation in result["next"]["environment_operations"]:
        response = ops.mindie_call(
            operation["tool"], operation["arguments"], allow_native_context=False,
        )["result"]
        assert response["outcome"] == "success", response
        assert response["data"]["state"] == "ready"


def test_native_boundary_recovery_reaches_lease_validation_without_attachment(tmp_path, monkeypatch):
    from mindie_coordinator.agent_session import AgentSessions
    from mindie_coordinator.task_server import call_tool

    monkeypatch.setenv("MINDIE_AGENT_SESSIONS_DIR", str(tmp_path / "sessions"))
    metadata = {"x-codex-turn-metadata": {"thread_id": "unattached"}}
    result = call_tool("mindie_run", {"command": "never-sent"}, metadata)
    assert result["isError"]
    assert result["structuredContent"]["error_details"]["submission_state"] == "not_sent"
    recovery = result["structuredContent"]["next"]
    assert recovery["task_context"]["knowledge_activation_repairs_binding"] is False
    route = next(item for item in recovery["environment_routes"] if item["tool"] == "mindie.lease")
    lease = call_tool(route["tool"], {"action": route["action"]}, metadata)
    assert lease["isError"]
    assert "no machine registered or specified" in lease["structuredContent"]["summary"]
    assert lease["structuredContent"]["status"] != "managed_task_context_required"
    assert AgentSessions(tmp_path / "sessions").sessions() == []
