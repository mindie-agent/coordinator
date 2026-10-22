"""Native fixtures come from an installed wheel, not optional CPython tests."""
import importlib
from pathlib import Path
import sys

import pytest


@pytest.fixture(autouse=True)
def isolated_diagnostics(tmp_path, monkeypatch, request):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    if request.node.get_closest_marker("live_npu") is not None:
        ssh = Path.home() / ".ssh"
        if ssh.exists():
            (home / ".ssh").symlink_to(ssh, target_is_directory=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local" / "share"))
    monkeypatch.setenv("MINDIE_COORDINATOR_STATE_DIR", str(tmp_path.resolve() / "coordinator-state"))
    monkeypatch.setenv("MINDIE_DIAGNOSTICS_ROOT", str(tmp_path.resolve() / "isolated-diagnostics"))
    monkeypatch.setenv("MINDIE_DIAGNOSTICS_CONFIG", str(tmp_path.resolve() / "reporting.json"))


@pytest.fixture
def native_extension():
    module_name = {
        "linux": "_psutil_linux",
        "darwin": "_psutil_osx",
        "win32": "_psutil_windows",
    }[sys.platform]
    module = importlib.import_module("psutil." + module_name)
    return module_name, Path(module.__file__)
