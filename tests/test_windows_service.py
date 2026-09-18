"""Native loopback IPC lifecycle. No endpoints, allocation or device imports."""
import concurrent.futures
import ctypes
from ctypes import wintypes
import json
import os
import socket
import time

import pytest

from mindie_coordinator.service import CoordinatorClient, ensure_daemon, socket_path


pytestmark = pytest.mark.skipif(os.name != "nt", reason="native Windows IPC")


def pid_alive(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        return ctypes.get_last_error() == 5
    try:
        return kernel.WaitForSingleObject(handle, 0) == 0x102
    finally:
        kernel.CloseHandle(handle)


def stop(client):
    pid = client.call("ping")["runtime"][0]["loaded"]["pid"]
    assert client.call("restart_if_idle")["status"] == "stopping"
    deadline = time.monotonic() + 8
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not pid_alive(pid), "daemon still alive after graceful stop"
    assert not socket_path(client.state_dir).exists()


def test_parallel_start_authentication_restart_and_stale_marker(tmp_path):
    state = tmp_path / ("中文 directory " + "long-" * 20)
    state.mkdir()
    socket_path(state).write_text("{partial", encoding="utf-8")
    client = None
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            clients = list(pool.map(ensure_daemon, [state] * 4))
        client = clients[0]
        pids = {item.call("ping")["runtime"][0]["loaded"]["pid"] for item in clients}
        assert len(pids) == 1
        old_pid = pids.pop()
        address = json.loads(socket_path(state).read_text(encoding="utf-8"))
        assert address["host"] == "127.0.0.1"
        with socket.create_connection(("127.0.0.1", address["port"]), timeout=3) as conn:
            conn.sendall(b'{"op":"restart_if_idle","_ipc_token":"wrong"}\n')
            reply = json.loads(conn.makefile("rb").readline())
        assert not reply["ok"] and "authentication" in reply["error"]
        assert client.call("ping")["runtime"][0]["loaded"]["pid"] == old_pid
        stop(client)
        client = ensure_daemon(state)
        assert client.call("ping")["runtime"][0]["loaded"]["pid"] != old_pid
        assert not list(state.glob("*.tmp"))
    finally:
        if client is not None and socket_path(state).exists():
            stop(client)
