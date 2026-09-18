"""Provision host_ops.run_remote_script through remote-dev run_stream.

No real SSH. stream_ssh_command is replaced with a local argv so the shared
reader, deadline, capture, and on_output callback still run.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import pytest

import remote_dev.core.ssh_transport as ssh_transport

from mindie_coordinator.provision import host_ops

TARGET = host_ops.SshTarget(host="192.0.2.10", user="root", port=22)


def _local_bash(_endpoint, script, *, timeout_ms=None):
    del _endpoint, timeout_ms
    # remote-dev streams the script through this process's binary stdin.
    assert script is None
    # Resolve PATH ourselves: CreateProcess searches System32 before PATH and
    # can otherwise select its unconfigured WSL alias instead of Git Bash.
    bash = shutil.which("bash") or "bash"
    return [bash, "-s"]


def _local_python(source: str):
    def _cmd(_endpoint, script, *, timeout_ms=None):
        del _endpoint, script, timeout_ms
        return [sys.executable, "-c", source]

    return _cmd


class RunRemoteScriptTransportTests(unittest.TestCase):
    def test_separate_channels_progress_and_sentinel(self) -> None:
        script = r"""
echo 'machine-out'
echo '__MINDIE_PROGRESS__={"phase":"probe","message":"checking"}' >&2
echo 'warn-line' >&2
echo '__MINDIE_JSON__={"success":true,"step":"probe"}'
"""
        with patch.object(ssh_transport, "stream_ssh_command", _local_bash):
            result = host_ops.run_remote_script(
                TARGET, script, stream_progress=False
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(result.timed_out)
        self.assertIn("machine-out\n", result.stdout)
        self.assertIn("warn-line\n", result.stderr)
        self.assertIn(host_ops.PROGRESS_SENTINEL, result.stderr)
        self.assertEqual(result.payload, {"success": True, "step": "probe"})
        self.assertEqual(result.progress_events[0]["phase"], "probe")
        self.assertEqual(result.progress_events[0]["message"], "checking")

    def test_nonzero_status_is_preserved(self) -> None:
        script = "echo out; echo err >&2; exit 7"
        with patch.object(ssh_transport, "stream_ssh_command", _local_bash):
            result = host_ops.run_remote_script(TARGET, script, stream_progress=False)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertFalse(result.timed_out)
        self.assertEqual(result.stdout, "out\n")
        self.assertEqual(result.stderr, "err\n")
        self.assertIsNone(result.payload)

    def test_timeout_propagates_from_shared_reader(self) -> None:
        sleeper = "import sys,time; sys.stdout.buffer.write(b'partial'); sys.stdout.buffer.flush(); time.sleep(2)"
        started = time.monotonic()
        with patch.object(ssh_transport, "stream_ssh_command", _local_python(sleeper)):
            result = host_ops.run_remote_script(
                TARGET, "unused", timeout_seconds=1, stream_progress=False
            )
        elapsed = time.monotonic() - started
        self.assertTrue(result.timed_out)
        self.assertLess(elapsed, 1.5)
        self.assertIn("partial", result.stdout)
        self.assertIn("wall-clock", result.stderr)
        self.assertEqual(result.timeout_seconds, 1)

    def test_literal_multiline_quoted_and_unicode_args(self) -> None:
        args = [
            "hello world",
            "line1\nline2",
            "quote'\"and",
            "unicøde-Δ",
        ]
        script = r"""
printf '%s\n' "$1"
printf '%s\n' "$2"
printf '%s\n' "$3"
printf '%s\n' "$4"
echo '__MINDIE_JSON__={"success":true,"n":4}'
"""
        with patch.object(ssh_transport, "stream_ssh_command", _local_bash):
            result = host_ops.run_remote_script(
                TARGET, script, args=args, stream_progress=False
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(result.timed_out)
        lines = result.stdout.splitlines()
        self.assertEqual(lines[0], "hello world")
        self.assertEqual(lines[1], "line1")
        self.assertEqual(lines[2], "line2")
        self.assertEqual(lines[3], "quote'\"and")
        self.assertEqual(lines[4], "unicøde-Δ")
        self.assertEqual(result.payload, {"success": True, "n": 4})


if __name__ == "__main__":
    unittest.main()


def test_cached_probe_executes_literal_arguments_and_preserves_channels(monkeypatch):
    calls = []
    def rpc(endpoint, script, *, timeout_ms):
        calls.append((endpoint, timeout_ms))
        done = subprocess.run([shutil.which('bash'), '-s'], input=script,
                              capture_output=True, text=True, encoding='utf-8', timeout=5)
        return SimpleNamespace(returncode=done.returncode, stdout=done.stdout, stderr=done.stderr,
                               timed_out=False, cancelled=False)
    monkeypatch.setattr(ssh_transport, 'run_rpc_script', rpc)
    monkeypatch.setattr(host_ops, 'run_stream', lambda *a, **k: pytest.fail('probe opened an extra stream'))
    result = host_ops.run_remote_script(TARGET,
        'printf "%s\\n" "$1"; echo \'__MINDIE_PROGRESS__={"phase":"read"}\' >&2; '
        'echo \'__MINDIE_JSON__={"success":true}\'',
        args=['literal\nquote\'"$()'], timeout_seconds=5, stream_progress=False, reuse_connection=True)
    assert result.stdout.split(host_ops.SENTINEL)[0] == 'literal\nquote\'"$()\n'
    assert host_ops.assert_remote_success(result)['success'] is True
    assert result.progress_events[0]['phase'] == 'read'
    assert len(calls) == 1 and calls[0][1] == 5000


@pytest.mark.parametrize('cancelled', [False, True])
def test_cached_probe_never_replays_unknown_or_cancelled_command(monkeypatch, cancelled):
    calls = []
    def rpc(*args, **kwargs):
        calls.append(True)
        return SimpleNamespace(returncode=None, stdout='', stderr='', timed_out=False, cancelled=cancelled)
    monkeypatch.setattr(ssh_transport, 'run_rpc_script', rpc)
    monkeypatch.setattr(host_ops, 'run_stream', lambda *a, **k: pytest.fail('unknown probe was replayed'))
    with pytest.raises(host_ops.MachineManagementError, match='no fallback'):
        host_ops.run_remote_script(TARGET, 'probe', reuse_connection=True)
    assert len(calls) == 1


def test_cached_probe_retains_exit_error(monkeypatch):
    monkeypatch.setattr(ssh_transport, 'run_rpc_script', lambda *a, **k:
        SimpleNamespace(returncode=7, stdout='__MINDIE_JSON__={"success":false,"error":"missing runtime"}\n',
                        stderr='detail\n', timed_out=False, cancelled=False))
    result = host_ops.run_remote_script(TARGET, 'probe', reuse_connection=True, stream_progress=False)
    assert result.returncode == 7
    with pytest.raises(host_ops.MachineManagementError, match='missing runtime'):
        host_ops.assert_remote_success(result)
