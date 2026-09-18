import json
from types import SimpleNamespace

import pytest

from mindie_coordinator.service import CoordinatorService


@pytest.mark.parametrize("retryable,error", [(False, ValueError), (True, RuntimeError), (None, RuntimeError)])
def test_completed_sync_failure_preserves_reason_and_recovery(tmp_path, monkeypatch, retryable, error):
    service = object.__new__(CoordinatorService)
    service.state_dir = tmp_path
    monkeypatch.setattr("mindie_coordinator.service.materialize_command", lambda **kwargs: ["fixture"])

    def run(args, *, stdout, stderr, **kwargs):
        json.dump({"status": "failed", "reason": "nested source path too long", "retryable": retryable}, stdout)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("mindie_coordinator.service.subprocess.run", run)
    binding = {"runtime_id": "runtime", "endpoint": {}, "intent": {"session": "task"}, "environment": {}}
    with pytest.raises(error, match="nested source path too long"):
        service.sync_binding(binding, {"sources": {"vllm": {"path": "one"}, "vllm-ascend": {"path": "two"}}}, "execution")
