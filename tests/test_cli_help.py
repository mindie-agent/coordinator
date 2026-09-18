from __future__ import annotations

import json
import subprocess
import sys
import unittest


class CliHelpTests(unittest.TestCase):
    def test_coordinator_entry_and_task_server_help(self):
        for args in (["--help"], ["task-server", "--help"], ["runtime-register", "--help"],
                     ["daemon", "--help"], ["provision", "--help"]):
            with self.subTest(args=args):
                proc = subprocess.run(
                    [sys.executable, "-m", "mindie_coordinator", *args],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn("usage:", proc.stdout)

    def test_task_facade_uses_one_cli_without_endpoint_or_network_requirements(self):
        for args in (["--help"], ["attach", "--help"], ["session", "--help"],
                     ["run", "--help"], ["execution", "--help"], ["finish", "--help"]):
            with self.subTest(args=args):
                proc = subprocess.run(
                    [sys.executable, "-m", "mindie_coordinator.mindie", *args],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn("usage:", proc.stdout)

    def test_json_arguments_are_not_overridden_by_argparse_defaults(self) -> None:
        code = """
import json, sys
from unittest import mock
import mindie_coordinator.mindie as module
captured = {}
def fake(name, args, **kwargs):
    captured.update(args)
    return {"result": {"outcome": "success"}}
argv = ["mindie", "execution", "--execution-id", "e1", "--json", json.dumps({"action": "stop"})]
with mock.patch.object(module, "mindie_call", side_effect=fake), mock.patch.object(sys, "argv", argv):
    module.main()
print(json.dumps(captured))
"""
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        merged = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(merged["action"], "stop")

    def test_mindie_cli_bad_json_returns_result_contract_without_traceback(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "mindie_coordinator.mindie", "session", "--json", "{not json"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertNotIn("Traceback", proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["schema_version"], "remote-dev.result.v1")
        self.assertEqual(payload["tool"], "mindie.session")
        self.assertEqual(payload["status"], "invalid_json")
        self.assertEqual(payload["outcome"], "needs_input")

    def test_mindie_cli_attach_error_returns_result_contract_without_traceback(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "mindie_coordinator.mindie",
                "attach",
                "--client",
                "kimi",
                "--native-session-id",
                "native-test",
                "--parent-context",
                "/nonexistent/mindie-agent-context.json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertNotIn("Traceback", proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["schema_version"], "remote-dev.result.v1")
        self.assertEqual(payload["tool"], "mindie.attach")
        self.assertEqual(payload["status"], "attach_failed")
        self.assertEqual(payload["outcome"], "failed")


if __name__ == "__main__":
    unittest.main()
