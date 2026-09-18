"""An SSH upload can finish remotely yet lose its exit reply; never replay it."""
import subprocess

import pytest

from mindie_coordinator.parity_support import SshEndpoint, ssh_stream_to_file, ssh_stream_bytes_to_file


@pytest.mark.parametrize('upload,payload,expected_timeout', [
    (ssh_stream_to_file, 'metadata\n', 120000),
    (ssh_stream_bytes_to_file, b'archive\x00', 1800000),
])
def test_upload_timeout_is_bounded_unknown_and_never_replayed(monkeypatch, upload, payload, expected_timeout):
    calls = []

    def stalled(endpoint, script, *, stdin, timeout_ms):
        calls.append((endpoint, script, stdin, timeout_ms))
        raise subprocess.TimeoutExpired('ssh', timeout_ms / 1000)

    monkeypatch.setattr('remote_dev.core.ssh_transport.run_bytes', stalled)
    with pytest.raises(TimeoutError, match='remote outcome is unknown; payload was not replayed'):
        upload(SshEndpoint(host='example.invalid', port=22, user='user'), '/tmp/upload-target', payload)
    assert len(calls) == 1
    endpoint, script, actual, timeout = calls[0]
    assert timeout == expected_timeout
    assert endpoint.keepalive is True and endpoint.ssh_mux is False
    assert actual == (payload.encode() if isinstance(payload, str) else payload)
