"""Happy-path and failure wiring for parity_support.ssh_exec_stream."""
from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from remote_dev.core.ssh_transport import RemoteCompleted
from mindie_coordinator.parity_support import (
    PROGRESS_SENTINEL,
    SshEndpoint,
    RemoteCommandError,
    ssh_exec_stream,
)


class SshExecStreamWiringTests(unittest.TestCase):
    def test_run_stream_captures_machine_stdout_and_progress_stderr(self) -> None:
        def fake_run_stream(endpoint, script, **kwargs):
            self.assertIs(endpoint.ssh_mux, False)
            self.assertTrue(endpoint.keepalive)
            self.assertFalse(kwargs.get("merge_stderr", True))
            on_output = kwargs["on_output"]
            on_output("stdout", '{"ok": true}\n')
            on_output("stderr", PROGRESS_SENTINEL + '{"phase": "install", "message": "pip"}\n')
            on_output("stderr", "warn\n")
            return RemoteCompleted(0, "", "", timed_out=False)

        endpoint = SshEndpoint(host="192.0.2.10", port=46000, user="root")
        events = []
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "install.log"
            with patch("remote_dev.core.ssh_transport.run_stream", fake_run_stream):
                result = ssh_exec_stream(endpoint, "true", stream_progress=False,
                                         on_progress=events.append, log_path=log)
            self.assertIn("warn", log.read_text())
            self.assertIn('"ok": true', log.read_text())
        self.assertEqual(events[0]["phase"], "install")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '{"ok": true}\n')
        self.assertEqual(result.stderr, "warn\n")
        self.assertEqual(result.progress_events[0]["phase"], "install")

    def test_nonzero_status_raises(self) -> None:
        def fake_run_stream(endpoint, script, **kwargs):
            del endpoint, script
            kwargs["on_output"]("stdout", "out\n")
            kwargs["on_output"]("stderr", "err\n")
            return RemoteCompleted(7, "out\n", "err\n", timed_out=False)

        endpoint = SshEndpoint(host="192.0.2.10", port=46000, user="root")
        with patch("remote_dev.core.ssh_transport.run_stream", fake_run_stream):
            with self.assertRaises(RemoteCommandError) as raised:
                ssh_exec_stream(endpoint, "false")
        self.assertEqual(raised.exception.returncode, 7)
        self.assertIn("command failed (7)", str(raised.exception))
        self.assertIn("out", str(raised.exception))
        self.assertIn("err", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
