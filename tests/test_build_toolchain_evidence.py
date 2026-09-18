import os
from unittest.mock import Mock

import pytest

from mindie_coordinator.backend import RemoteBackend
from mindie_coordinator.parity_support import RemoteCommandError
from mindie_coordinator.runtime_profile import build_toolchain_from_logs, file_digest


def build_log(root, name, text):
    path = root / ".mindie-runtime/prepare-logs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_completed_build_supplies_soc_and_actual_compiler_identities(tmp_path):
    path = build_log(tmp_path, "runtime-install-vllm-ascend.one", """
  -- Detected SOC version: ascend910_9362
  -- The CXX compiler identification is GNU 13.3.0
  -- The CXX compiler identification is GNU 7.3.0
  -- The CXX compiler identification is GNU 13.3.0
Successfully installed vllm_ascend-0.1.dev1+g12345
""")
    evidence = build_toolchain_from_logs(tmp_path)
    assert evidence == {"soc": "ascend910_9362", "compilers": ["GNU 13.3.0", "GNU 7.3.0"],
                        "path": path.relative_to(tmp_path).as_posix(), "sha256": file_digest(path)}


def test_incomplete_new_install_cannot_reuse_older_evidence(tmp_path):
    old = build_log(tmp_path, "runtime-install-vllm-ascend.old", "Successfully installed vllm-ascend-1\n")
    new = build_log(tmp_path, "runtime-install-vllm-ascend.new", "-- Detected SOC version: ascend910b1\n")
    os.utime(old, ns=(1, 1))
    os.utime(new, ns=(2, 2))
    with pytest.raises(ValueError, match="no successful completion evidence"):
        build_toolchain_from_logs(tmp_path)


def test_missing_or_conflicting_soc_stays_unknown(tmp_path):
    assert build_toolchain_from_logs(tmp_path) == {}
    build_log(tmp_path, "runtime-install-vllm-ascend.conflict", """
-- Detected SOC version: ascend910b1
-- Detected SOC version: ascend950
Successfully installed vllm-ascend-1
""")
    with pytest.raises(ValueError, match="conflicting SoC"):
        build_toolchain_from_logs(tmp_path)


def test_probe_keeps_completed_exit_separate_from_transport_uncertainty():
    shell = Mock()
    backend = RemoteBackend(shell=shell)
    for code in (1, 255):
        shell.run.return_value = {"outcome": "failed", "status": "nonzero_exit", "exit_code": code}
        with pytest.raises(RemoteCommandError) as error:
            backend.bash({}, "fixture")
        assert error.value.returncode == code
    shell.run.return_value = {"outcome": "timeout", "status": "timeout", "exit_code": None}
    with pytest.raises(RuntimeError) as error:
        backend.bash({}, "fixture")
    assert not isinstance(error.value, RemoteCommandError)
