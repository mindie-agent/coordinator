"""Business age alone never consumes a queued task or a live start gate."""
import time
from unittest.mock import patch

from mindie_coordinator.host.mindie_npu_coordination import NpuCoordinator
from mindie_coordinator.state_paths import shared_workspace_root
from test_group_start import pool_fixture

FREE = {"status": "ok", "devices": [0], "busy": {}, "free": [0]}
BUSY = {"status": "ok", "devices": [0], "busy": {"0": [{"pid": 123}]}, "free": []}


def test_host_queue_without_deadline_survives_months_but_explicit_limit_expires(tmp_path):
    now = [1000.0]
    queue = NpuCoordinator(tmp_path / "host", clock=lambda: now[0])
    queue.submit({"task_id": "unbounded", "agent_id": "owner", "devices": [0]})
    queue.submit({"task_id": "limited", "agent_id": "owner", "devices": [0], "queue_ttl_seconds": 7})
    assert queue.acquire("unbounded", BUSY)["task"]["state"] == "queued"
    now[0] += 90 * 86400
    records = {row["task_id"]: row for row in queue.snapshot(BUSY)["tasks"]}
    assert records["unbounded"]["latest_start"] is None
    assert records["unbounded"]["state"] == "queued"
    assert records["limited"]["state"] == "expired"
    assert queue.acquire("unbounded", FREE)["task"]["state"] == "granted"


def test_pool_default_gate_and_command_have_no_deadline(tmp_path):
    pool, backend = pool_fixture(tmp_path)
    session = pool.session_open("alice", "execution", {})
    binding = pool.checkout("alice", session["id"], "profile-a", "one", "donor-0")
    with patch.object(backend, "job", wraps=backend.job) as remote:
        job = pool.managed_start("alice", binding["id"], "one", {}, "native-a", [], 0, "true", {}, hold_go=True)
    prepared = next(call.kwargs["spec"] for call in remote.call_args_list if call.args[2] == "prepare")
    assert prepared["prepared_timeout_seconds"] is None
    assert prepared["timeout_seconds"] is None
    with pool.transaction() as db:
        run = pool.get(db, "run", job["id"])
    assert run["deadline"] is None


def test_git_authority_failure_does_not_select_another_state_root(tmp_path):
    import subprocess
    import pytest
    (tmp_path / ".git").mkdir()
    with patch("mindie_coordinator.state_paths.subprocess.run", side_effect=OSError("git unavailable")):
        with pytest.raises(OSError, match="git unavailable"):
            shared_workspace_root(tmp_path)
