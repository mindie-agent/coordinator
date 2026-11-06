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
