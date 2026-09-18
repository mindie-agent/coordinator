from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from mindie_coordinator import ops as mindie_ops
from mindie_coordinator import task_client as mindie_task_client


class FakeTaskClient:
    state = "finishing"

    def __init__(self, *_args, **_kwargs):
        self.context = {"session": {"id": "sess-test"}}
        self._service = None
        self._temporary = tempfile.TemporaryDirectory()
        self.store = SimpleNamespace(state_dir=Path(self._temporary.name) / "sessions")

    def finish(self, _force=False):
        return {"state": self.state, "executions": [], "worktrees_preserved": True}


class VawsOpsTests(unittest.TestCase):
    def test_finish_non_terminal_state_is_blocked_not_success(self) -> None:
        with mock.patch.object(mindie_task_client, "TaskClient", FakeTaskClient):
            payload = mindie_ops.mindie_call("mindie.finish", {})
        self.assertEqual(payload["result"]["status"], "finishing")
        self.assertEqual(payload["result"]["outcome"], "blocked")

    def test_finish_terminal_state_stays_success(self) -> None:
        class DoneClient(FakeTaskClient):
            state = "finished"

        with mock.patch.object(mindie_task_client, "TaskClient", DoneClient):
            payload = mindie_ops.mindie_call("mindie.finish", {})
        self.assertEqual(payload["result"]["status"], "finished")
        self.assertEqual(payload["result"]["outcome"], "success")


if __name__ == "__main__":
    unittest.main()
