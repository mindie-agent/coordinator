"""Native loopback IPC lifecycle and Win32 Job breakaway integration tests."""
import concurrent.futures
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

from mindie_coordinator import service_client
from mindie_coordinator.service import CoordinatorClient, ensure_daemon, socket_path


pytestmark = pytest.mark.skipif(os.name != "nt", reason="native Windows IPC")

REPO_ROOT = Path(__file__).resolve().parents[1]
CREATE_SUSPENDED = 0x00000004
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_UNICODE_ENVIRONMENT = 0x00000400
CREATE_NO_WINDOW = 0x08000000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
PROCESS_QUERY_LIMITED_INFORMATION = 0x00001000
WAIT_OBJECT_0 = 0x00000000


if os.name == "nt":
    class _STARTUPINFOW(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("lpReserved", wintypes.LPWSTR),
            ("lpDesktop", wintypes.LPWSTR),
            ("lpTitle", wintypes.LPWSTR),
            ("dwX", wintypes.DWORD),
            ("dwY", wintypes.DWORD),
            ("dwXSize", wintypes.DWORD),
            ("dwYSize", wintypes.DWORD),
            ("dwXCountChars", wintypes.DWORD),
            ("dwYCountChars", wintypes.DWORD),
            ("dwFillAttribute", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("wShowWindow", wintypes.WORD),
            ("cbReserved2", wintypes.WORD),
            ("lpReserved2", ctypes.c_void_p),
            ("hStdInput", wintypes.HANDLE),
            ("hStdOutput", wintypes.HANDLE),
            ("hStdError", wintypes.HANDLE),
        ]

    class _PROCESS_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("hProcess", wintypes.HANDLE),
            ("hThread", wintypes.HANDLE),
            ("dwProcessId", wintypes.DWORD),
            ("dwThreadId", wintypes.DWORD),
        ]

    class _BASIC_LIMIT_INFO(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _EXTENDED_LIMIT_INFO(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BASIC_LIMIT_INFO),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


def _kernel32():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    kernel.IsProcessInJob.restype = wintypes.BOOL
    kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.TerminateJobObject.restype = wintypes.BOOL
    kernel.CreateProcessW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.LPCWSTR,
        ctypes.POINTER(_STARTUPINFOW),
        ctypes.POINTER(_PROCESS_INFORMATION),
    ]
    kernel.CreateProcessW.restype = wintypes.BOOL
    kernel.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel.ResumeThread.restype = wintypes.DWORD
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.TerminateProcess.restype = wintypes.BOOL
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    return kernel


def _create_job(kernel, limit_flags: int) -> int:
    job = int(kernel.CreateJobObjectW(None, None) or 0)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    ext = _EXTENDED_LIMIT_INFO()
    ext.BasicLimitInformation.LimitFlags = limit_flags
    if not kernel.SetInformationJobObject(
        job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(ext), ctypes.sizeof(ext)
    ):
        err = ctypes.get_last_error()
        kernel.CloseHandle(job)
        raise ctypes.WinError(err)
    return job


def _build_worker_env_block() -> ctypes.Array[ctypes.c_wchar]:
    existing_py_path = ""
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if "\x00" in key or "\x00" in value:
            continue
        if key.casefold() == "pythonpath":
            existing_py_path = value
        else:
            env[key] = value
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{existing_py_path}" if existing_py_path else str(REPO_ROOT)
    items = [f"{k}={v}" for k, v in sorted(env.items(), key=lambda kv: kv[0].upper())]
    return ctypes.create_unicode_buffer("\x00".join(items) + "\x00\x00")


def _run_python_in_jobs(kernel, jobs: tuple[int, ...], argv: list[str]) -> int:
    cmd_buf = ctypes.create_unicode_buffer(subprocess.list2cmdline(argv))
    env_buf = _build_worker_env_block()
    si = _STARTUPINFOW()
    si.cb = ctypes.sizeof(_STARTUPINFOW)
    pi = _PROCESS_INFORMATION()
    flags = CREATE_SUSPENDED | CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP | CREATE_UNICODE_ENVIRONMENT
    if not kernel.CreateProcessW(
        None,
        cmd_buf,
        None,
        None,
        False,
        flags,
        ctypes.cast(env_buf, ctypes.c_void_p),
        None,
        ctypes.byref(si),
        ctypes.byref(pi),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    proc_h = int(pi.hProcess)
    thread_h = int(pi.hThread)
    try:
        for job in jobs:
            if not kernel.AssignProcessToJobObject(job, proc_h):
                err = ctypes.get_last_error()
                kernel.TerminateProcess(proc_h, 1)
                raise ctypes.WinError(err)
            in_job = wintypes.BOOL(False)
            assert kernel.IsProcessInJob(proc_h, job, ctypes.byref(in_job)) and in_job.value
        if kernel.ResumeThread(thread_h) == 0xFFFFFFFF:
            err = ctypes.get_last_error()
            kernel.TerminateProcess(proc_h, 1)
            raise ctypes.WinError(err)
        kernel.CloseHandle(thread_h)
        thread_h = 0
        wait_res = kernel.WaitForSingleObject(proc_h, 20000)
        if wait_res != WAIT_OBJECT_0:
            kernel.TerminateProcess(proc_h, 1)
            raise TimeoutError(f"Win32 job worker timed out (wait={wait_res})")
        exit_code = wintypes.DWORD(1)
        if not kernel.GetExitCodeProcess(proc_h, ctypes.byref(exit_code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(exit_code.value)
    finally:
        if thread_h:
            kernel.CloseHandle(thread_h)
        if proc_h:
            kernel.CloseHandle(proc_h)


def _pid_in_job(kernel, pid: int, job: int) -> bool:
    handle = kernel.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        in_job = wintypes.BOOL(False)
        if not kernel.IsProcessInJob(handle, job, ctypes.byref(in_job)):
            raise ctypes.WinError(ctypes.get_last_error())
        return bool(in_job.value)
    finally:
        kernel.CloseHandle(handle)


def _write_job_worker_script(path: Path) -> Path:
    path.write_text(
        "import json, pathlib, sys, traceback\n"
        "from mindie_coordinator import service_client\n"
        "from mindie_coordinator.service import CoordinatorClient, ensure_daemon\n"
        "state = pathlib.Path(sys.argv[1])\n"
        "out = pathlib.Path(sys.argv[2])\n"
        "try:\n"
        "    ensure_daemon(state)\n"
        "    ping = CoordinatorClient(state).call('ping')\n"
        "    loaded = ping['runtime'][0]['loaded']\n"
        "    out.write_text(\n"
        "        json.dumps({\n"
        "            'ok': True,\n"
        "            'pid': loaded['pid'],\n"
        "            'service_client_file': str(pathlib.Path(service_client.__file__).resolve()),\n"
        "            'daemon_location': str(loaded.get('location') or ''),\n"
        "        }),\n"
        "        encoding='utf-8',\n"
        "    )\n"
        "except BaseException as exc:\n"
        "    out.write_text(json.dumps({'ok': False, 'error': repr(exc), 'tb': traceback.format_exc()}), encoding='utf-8')\n"
        "    raise SystemExit(1)\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


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


def test_daemon_survives_breakaway_allowed_foreground_job_termination(tmp_path):
    assert Path(service_client.__file__).resolve().is_relative_to(REPO_ROOT)
    kernel = _kernel32()
    worker_py = _write_job_worker_script(tmp_path / "job_worker.py")
    state = tmp_path / "state-breakaway-ok"
    result_file = tmp_path / "result-breakaway-ok.json"
    # Nested [outer_job (no breakaway), foreground_job (BREAKAWAY_OK)]:
    # On Windows 8+, CREATE_BREAKAWAY_FROM_JOB breaks out of foreground_job while stopping at outer_job,
    # so this test never skips under non-breakaway CI runner Jobs AND outer_job guarantees zero leaked daemons.
    outer_job = _create_job(kernel, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
    foreground_job = _create_job(kernel, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_BREAKAWAY_OK)
    client = None
    try:
        rc = _run_python_in_jobs(
            kernel,
            (outer_job, foreground_job),
            [sys.executable, str(worker_py), str(state), str(result_file)],
        )
        payload = json.loads(result_file.read_text(encoding="utf-8")) if result_file.is_file() else {}
        assert rc == 0, f"foreground job worker failed: {payload}"
        assert Path(payload["service_client_file"]).resolve().is_relative_to(REPO_ROOT)
        assert Path(payload["daemon_location"]).resolve().is_relative_to(REPO_ROOT)
        daemon_pid = int(payload["pid"])
        assert not _pid_in_job(kernel, daemon_pid, foreground_job), "daemon must break away from foreground CLI Job"
        assert _pid_in_job(kernel, daemon_pid, outer_job), "daemon remains owned by test outer_job"

        # Explicitly terminate and close the foreground Job Object (mirrors mindie-harness CLI exit)
        assert kernel.TerminateJobObject(foreground_job, 1)
        assert kernel.CloseHandle(foreground_job)
        foreground_job = 0

        # Persistent coordinator daemon MUST survive foreground Job termination
        assert pid_alive(daemon_pid), "coordinator daemon was killed by foreground TerminateJobObject"
        client = CoordinatorClient(state)
        ping = client.call("ping")
        assert ping["runtime"][0]["loaded"]["pid"] == daemon_pid
    finally:
        try:
            if state.exists() and socket_path(state).exists():
                stop(client or CoordinatorClient(state))
        finally:
            if foreground_job:
                kernel.TerminateJobObject(foreground_job, 1)
                kernel.CloseHandle(foreground_job)
            if outer_job:
                kernel.TerminateJobObject(outer_job, 1)
                kernel.CloseHandle(outer_job)


def test_daemon_falls_back_cleanly_inside_breakaway_denied_outer_job(tmp_path):
    assert Path(service_client.__file__).resolve().is_relative_to(REPO_ROOT)
    kernel = _kernel32()
    worker_py = _write_job_worker_script(tmp_path / "job_worker_denied.py")
    state = tmp_path / "state-breakaway-denied"
    result_file = tmp_path / "result-breakaway-denied.json"
    # Nested [outer_job (no breakaway), foreground_job (no breakaway)] -> CreateProcess(CREATE_BREAKAWAY_FROM_JOB)
    # fails with ERROR_ACCESS_DENIED (winerror=5) and falls back cleanly to base_flags.
    outer_job = _create_job(kernel, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
    foreground_job = _create_job(kernel, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
    client = None
    try:
        rc = _run_python_in_jobs(
            kernel,
            (outer_job, foreground_job),
            [sys.executable, str(worker_py), str(state), str(result_file)],
        )
        payload = json.loads(result_file.read_text(encoding="utf-8")) if result_file.is_file() else {}
        assert rc == 0, f"breakaway-denied fallback worker failed: {payload}"
        assert Path(payload["service_client_file"]).resolve().is_relative_to(REPO_ROOT)
        assert Path(payload["daemon_location"]).resolve().is_relative_to(REPO_ROOT)
        daemon_pid = int(payload["pid"])

        # Before closing the Jobs, daemon answered ping and resides inside both foreground_job and outer_job
        assert _pid_in_job(kernel, daemon_pid, foreground_job), "fallback daemon must reside inside foreground_job"
        assert _pid_in_job(kernel, daemon_pid, outer_job), "fallback daemon must reside inside outer_job"
        client = CoordinatorClient(state)
        assert client.call("ping")["runtime"][0]["loaded"]["pid"] == daemon_pid
        stop(client)
        client = None
    finally:
        try:
            if state.exists() and socket_path(state).exists():
                stop(client or CoordinatorClient(state))
        finally:
            if foreground_job:
                kernel.TerminateJobObject(foreground_job, 1)
                kernel.CloseHandle(foreground_job)
            if outer_job:
                kernel.TerminateJobObject(outer_job, 1)
                kernel.CloseHandle(outer_job)
