"""An SSH upload can finish remotely yet lose its exit reply; never replay it."""
import subprocess

import pytest

from mindie_coordinator.parity_support import SshEndpoint, ssh_stream_to_file


def test_upload_timeout_is_bounded_unknown_and_never_replayed(monkeypatch):
    calls = []

    def stalled(endpoint, script, *, stdin, timeout_ms):
        calls.append((endpoint, script, stdin, timeout_ms))
        raise subprocess.TimeoutExpired('ssh', timeout_ms / 1000)

    monkeypatch.setattr('remote_dev.core.ssh_transport.run_bytes', stalled)
    with pytest.raises(TimeoutError, match='remote outcome is unknown; payload was not replayed'):
        ssh_stream_to_file(SshEndpoint(host='example.invalid', port=22, user='user'), '/tmp/upload-target', 'metadata\n')
    assert len(calls) == 1
    endpoint, script, actual, timeout = calls[0]
    assert timeout == 120000
    assert endpoint.keepalive is True and endpoint.ssh_mux is False
    assert actual == b'metadata\n'
