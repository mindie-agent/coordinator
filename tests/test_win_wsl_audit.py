"""Regression tests for Windows user-scoped Git Bash, WSL2 native isolation, CP936 UTF-8, and TimeoutError honesty."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from mindie_coordinator import execution_slots, local_exec, npu_top_bridge, service_client


def test_strip_windows_extended_prefix_preserves_unsafe_win32_components(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_exec, "_IS_WINDOWS_OVERRIDE", True)
    assert local_exec.strip_windows_extended_prefix(r"\\?\C:\Users\张三\repo") == r"C:\Users\张三\repo"
    assert local_exec.strip_windows_extended_prefix(r"\\?\UNC\server\share\repo") == r"\\server\share\repo"
    for unsafe in (
        r"\\?\Volume{01234567-89ab-cdef-0123-456789abcdef}\repo",
        r"\\?\C:\Users\张三\repo.",
        r"\\?\C:\trusted.\repo",
        r"\\?\C:\trusted \repo",
        r"\\?\C:\dir\NUL\repo",
        r"\\?\C:\dir\..\repo",
        r"\\?\C:\dir/sub\repo",
    ):
        assert local_exec.strip_windows_extended_prefix(unsafe) == unsafe


def test_utf8_python_env_overrides_inherited_cp936() -> None:
    env = local_exec.utf8_python_env({"PYTHONIOENCODING": "cp936", "PYTHONUTF8": "0"})
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    child_env = local_exec.utf8_python_env({**os.environ, "PYTHONIOENCODING": "cp936", "PYTHONUTF8": "0"})
    proc = subprocess.run(
        [sys.executable, "-c", "import sys; sys.stdout.write('昇腾协调器检查\\n')"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10.0,
        check=True,
        env=child_env,
        **local_exec.no_window_kwargs(),
    )
    assert proc.stdout.strip() == "昇腾协调器检查"


def test_windows_resolve_bash_rejects_wsl_stubs_and_finds_user_scoped_git_bash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(local_exec, "_IS_WINDOWS_OVERRIDE", True)
    sys32 = tmp_path / "Windows" / "System32"
    winapps = tmp_path / "Microsoft" / "WindowsApps"
    user_git = tmp_path / "LocalAppData" / "Programs" / "Git"
    for folder in (sys32, winapps, user_git / "cmd", user_git / "bin"):
        folder.mkdir(parents=True, exist_ok=True)
    (sys32 / "bash.exe").write_bytes(b"MZ" + b"\x00" * 62)
    (winapps / "bash.exe").write_bytes(b"MZ" + b"\x00" * 62)
    (user_git / "cmd" / "git.exe").write_bytes(b"MZ" + b"\x00" * 62)
    real_git_bash = user_git / "bin" / "bash.exe"
    real_git_bash.write_bytes(b"MZ" + b"\x00" * 62)

    fake_env = {
        "PATH": f"{sys32};{winapps};{user_git / 'cmd'}",
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
        "LOCALAPPDATA": str(tmp_path / "LocalAppData"),
    }
    resolved_git = local_exec.resolve_git(env=fake_env)
    resolved_bash = local_exec.resolve_bash(env=fake_env)
    assert Path(resolved_git) == user_git / "cmd" / "git.exe"
    assert Path(resolved_bash) == real_git_bash
    assert "system32" not in resolved_bash.lower()
    assert "windowsapps" not in resolved_bash.lower()

    # Without Git Bash anywhere, System32/WindowsApps stubs MUST NOT be returned:
    real_git_bash.unlink()
    with pytest.raises(FileNotFoundError):
        local_exec.resolve_bash(env={"PATH": f"{sys32};{winapps}", "PATHEXT": ".EXE"})

    # TimeoutError inside os.stat must propagate on Windows lookup too:
    with mock.patch("mindie_coordinator.local_exec.os.stat", side_effect=TimeoutError("win stat deadline")):
        with pytest.raises(TimeoutError, match="win stat deadline"):
            local_exec.resolve_git(env=fake_env)


def test_wsl2_which_native_preserves_relpath_and_symlink_dotdot_and_skips_drvfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name == "nt":
        pytest.skip("POSIX/WSL2 symlink and mountinfo isolation test")
    ext4_root = tmp_path / "ext4"
    ext4_sub = ext4_root / "sub"
    decoy_bin = ext4_root / "bin"
    drv_dir = tmp_path / "mnt_c"
    native_bin = tmp_path / "usr_bin"
    for folder in (ext4_sub, decoy_bin, drv_dir, native_bin):
        folder.mkdir(parents=True, exist_ok=True)

    for folder in (decoy_bin, native_bin):
        for tool_name in ("git", "bash"):
            tool = folder / tool_name
            tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            tool.chmod(0o755)

    pe_git = drv_dir / "git"
    pe_buf = bytearray(0x200)
    pe_buf[0:2] = b"MZ"
    pe_buf[0x3C:0x40] = (0x80).to_bytes(4, "little")
    pe_buf[0x80:0x84] = b"PE\x00\x00"
    pe_git.write_bytes(bytes(pe_buf))
    pe_git.chmod(0o755)

    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        f"30 1 8:1 / {tmp_path} rw,relatime - ext4 /dev/sdb rw\n"
        f"40 30 0:35 / {drv_dir.resolve()} rw,noatime - 9p drvfs rw,aname=drvfs\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(local_exec, "_MOUNTINFO_PATH", str(mountinfo))
    escape = ext4_sub / "escape"
    escape.symlink_to(drv_dir / "nested")

    assert local_exec.is_wsl_drvfs(f"{escape}/../git") is True
    assert local_exec.which_native("git", path=f"{ext4_sub}/escape/../bin") is None
    assert local_exec.resolve_git(path=f"{drv_dir}:{native_bin}") == str(native_bin / "git")
    assert local_exec.resolve_bash(path=f"{drv_dir}:{native_bin}") == str(native_bin / "bash")

    monkeypatch.chdir(native_bin)
    assert local_exec.which_native("./git") == "./git"


def test_execution_slots_script_handles_parentheses_and_globs_in_path(tmp_path: Path) -> None:
    from mindie_coordinator.local_exec import resolve_bash_or_none

    bash = resolve_bash_or_none()
    if bash is None:
        pytest.skip("Bash required for execution_slots shell script test")
    weird_dir = tmp_path / "Program Files (x86)" / "glob*[test]"
    weird_dir.mkdir(parents=True)
    snippet = execution_slots._SLOT_HELPER_SCRIPT if hasattr(execution_slots, "_SLOT_HELPER_SCRIPT") else None
    if snippet is None:
        src = Path(execution_slots.__file__).read_text(encoding="utf-8")
        assert "set -f" in src
    proc = subprocess.run(
        [
            bash,
            "-c",
            'old_opts=$-; set -f; IFS=:; count=0; for d in $PATH; do count=$((count+1)); done; '
            'case $old_opts in *f*) ;; *) set +f ;; esac; unset IFS; printf "%d\\n" "$count"',
        ],
        env={**os.environ, "PATH": f"{weird_dir}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10.0,
        check=True,
    )
    assert int(proc.stdout.strip()) >= 3


def test_npu_top_bridge_propagates_timeout_error_and_service_client_keeps_no_window() -> None:
    opener = mock.Mock()
    opener.open.side_effect = TimeoutError("socket deadline")
    client = npu_top_bridge.NpuTopObservationClient("http://127.0.0.1:9", opener=opener)
    with pytest.raises(TimeoutError, match="socket deadline"):
        client._get("/api/agent/capacity", {})
    with pytest.raises(TimeoutError, match="socket deadline"):
        client._post("/api/servers/batch", {})
    src = Path(service_client.__file__).read_text(encoding="utf-8")
    assert "CREATE_NO_WINDOW" in src
