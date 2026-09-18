import json
import pytest
from mindie_coordinator.service import CoordinatorService
from mindie_coordinator.launch_observation import launch_observation, ENV_NAME
from mindie_coordinator.placement import validate_user_env


def test_launch_receipt_is_detached_from_mutable_sources_and_omits_secrets():
    binding = {"profile_key": "profile", "host_endpoint": {"host": "machine"}}
    request = {"snapshots": {"vllm": "verified", "vllm-ascend": "ascend"}, "expected_build_key": "build"}
    spec = {"command": "run actual model", "env": {"TOKEN": "never-copy-this-secret"}}
    receipt = launch_observation(binding, request, spec, {"ASCEND_RT_VISIBLE_DEVICES": "1,2"})
    request["snapshots"]["vllm"] = "later"
    assert receipt["workspace_snapshot"]["vllm_commit"] == "verified"
    assert receipt["npu_devices"] == [1, 2]
    assert "never-copy-this-secret" not in json.dumps(receipt)
    assert receipt["native_digest"]["build_key"] == "build"


def test_target_retains_actual_receipt_after_stop_and_binding_refresh():
    owner = object.__new__(CoordinatorService)
    binding = {"id": "binding", "runtime_id": "runtime", "endpoint": {"host": "container"}, "build_key": "new-build"}
    row = {"id": "execution", "session_id": "task", "phase": "cancelled"}
    receipt = {"source_commits": {"vllm": "old-code"}, "build_key": "old-build"}
    assert owner._target(row, binding, {"launch_observation": receipt})["launch_observation"] == receipt
    assert owner._target(row, binding, {})["launch_observation"] == {}


def test_caller_cannot_spoof_owner_observation():
    with pytest.raises(ValueError, match="cannot override"):
        validate_user_env({ENV_NAME: "{}"})
