"""Actual Git output for Unicode local worktrees; no remote or device work."""
import json
import os
import subprocess
import sys
from pathlib import Path

from mindie_coordinator.agent_session import worktree_reference
from mindie_coordinator.code_identity import code_identity
from mindie_coordinator.git_sources import ensure_populated_worktree
from mindie_coordinator.parity import build_snapshot_records
from mindie_coordinator.state_paths import shared_workspace_root


def test_snapshot_ref_beyond_windows_path_limit(tmp_path):
    from mindie_coordinator.parity_support import git
    root = tmp_path / "nested-source"
    root.mkdir()
    git(root, ["init", "-q"])
    git(root, ["-c", "user.name=Test", "-c", "user.email=test@example.invalid",
               "commit", "--allow-empty", "-qm", "fixture"])
    ref = "refs/parity/" + "a" * 64 + "/" + "b" * 64 + "/" + "c" * 100
    assert len(str(root / ".git" / ref)) > 260
    git(root, ["update-ref", ref, "HEAD"])
    assert git(root, ["rev-parse", ref]).stdout == git(root, ["rev-parse", "HEAD"]).stdout
    assert git(root, ["config", "--local", "--get", "core.longpaths"], check=False).returncode == 1


def test_remote_profile_absolute_paths_do_not_depend_on_client_os():
    import pytest
    from mindie_coordinator.runtime_profile import PROFILE_FIELDS, profile_key, launch_preamble
    profile = {key: "test" for key in PROFILE_FIELDS}
    profile.update(build_env={}, launch_env={}, compatibility_evidence="fixture",
                   system_files={name: {"path": f"/usr/local/{name}/version.info", "sha256": "a" * 64}
                                 for name in ("cann", "driver")})
    assert len(profile_key(profile)) == 64
    assert "/owned/.venv/bin" in launch_preamble(profile, python="/owned/.venv/bin/python")
    profile["system_files"]["cann"]["path"] = "relative/version.info"
    with pytest.raises(ValueError, match="absolute path"):
        profile_key(profile)


def test_remote_upload_uses_posix_parent_and_exact_bytes(monkeypatch):
    from types import SimpleNamespace
    from mindie_coordinator.parity_support import SshEndpoint, ssh_stream_to_file, ssh_stream_bytes_to_file
    calls = []

    def capture(endpoint, script, *, stdin, timeout_ms):
        assert endpoint.keepalive is True
        assert endpoint.ssh_mux is False
        assert timeout_ms in (120000, 1800000)
        calls.append((script, stdin))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("remote_dev.core.ssh_transport.run_bytes", capture)
    endpoint = SshEndpoint(host="example.invalid", port=22, user="test")
    ssh_stream_to_file(endpoint, "/tmp/nested path/manifests/test.json", "中文\n")
    ssh_stream_bytes_to_file(endpoint, "/tmp/nested path/bundles/test.bundle", bytes(range(256)))
    assert calls[0] == ("mkdir -p '/tmp/nested path/manifests' && cat > '/tmp/nested path/manifests/test.json'", "中文\n".encode())
    assert calls[1] == ("mkdir -p '/tmp/nested path/bundles' && head -c 256 > '/tmp/nested path/bundles/test.bundle'", bytes(range(256)))


def test_unicode_worktree_binding_snapshot_and_cli(tmp_path):
    root = tmp_path / "中文 🧪 source"
    root.mkdir()
    for args in [["init", "-q"], ["config", "user.name", "Test"],
                 ["config", "user.email", "test@example.invalid"], ["config", "core.autocrlf", "false"]]:
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    (root / "中文.txt").write_bytes("原始\n".encode())
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True, capture_output=True)
    assert worktree_reference(str(root))["path"] == str(root.resolve())
    assert shared_workspace_root(root) == root.resolve()
    ensure_populated_worktree(root, "vllm")
    (root / "中文.txt").write_bytes("修改\n".encode())
    assert code_identity(root)["dirty"]
    rows = build_snapshot_records(root, "unicode", "test", ())
    assert rows and rows[0].commit
    proc = subprocess.run([sys.executable, "-m", "mindie_coordinator.mindie", "attach", "--client", "codex",
                           "--native-session-id", "unicode-test", "--cwd", str(root)],
                          capture_output=True, timeout=15,
                          env={**os.environ, "MINDIE_AGENT_SESSIONS_DIR": str(tmp_path / "sessions"),
                               "PYTHONIOENCODING": "cp936" if os.name == "nt" else "utf-8"})
    assert proc.returncode == 0, proc.stderr
    assert "中文 🧪" in json.dumps(json.loads(proc.stdout.decode("utf-8")), ensure_ascii=False)
