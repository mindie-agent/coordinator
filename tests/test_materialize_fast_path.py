import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mindie_coordinator import parity


@pytest.mark.parametrize("matches", [True, False])
def test_materialize_reuses_only_a_verified_remote_snapshot(tmp_path, monkeypatch, capsys, matches):
    sources = {name: tmp_path / name for name in ("vllm", "vllm-ascend")}
    rows = [parity.SnapshotRecord(name, name, "a" * 40, "a" * 40, "b" * 40, "c" * 40, "ref", [], [])
            for name in sources]
    expected = {row.relpath: row.commit for row in rows}
    monkeypatch.setattr(parity, "parse_sources", lambda values: sources)
    monkeypatch.setattr(parity, "build_snapshot_records", lambda *args: rows)
    monkeypatch.setattr(parity, "cleanup_synthetic_refs", lambda *args: None)
    monkeypatch.setattr(parity, "make_manifest", lambda **kwargs: {})
    monkeypatch.setattr(parity, "acquire_container_lock", Mock())
    released = Mock()
    monkeypatch.setattr(parity, "release_container_lock", released)
    verify = Mock(return_value=expected if matches else {"vllm": "dirty-runtime"})
    monkeypatch.setattr(parity, "verify_runtime_commits_map", verify)
    transfer = Mock(side_effect=RuntimeError("full transfer required"))
    monkeypatch.setattr(parity, "ensure_remote_bare_repos", transfer)
    args = parity.build_parser().parse_args([
        "sync", "--workspace-id", "test", "--server-name", "runtime", "--container-identity", "runtime",
        "--container-host", "example.invalid", "--container-port", "22", "--container-user", "test",
        "--runtime-root", "/task", "--workspace-root", str(tmp_path), "--apply-mode", "materialize",
    ])
    if matches:
        assert parity.run_sync(args) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["fast_path"] == "materialized-snapshot"
        assert result["snapshot_commits"] == expected
        assert result["runtime_commits"] == expected
        assert "manifest_path" not in result
        transfer.assert_not_called()
    else:
        with pytest.raises(RuntimeError, match="full transfer required"):
            parity.run_sync(args)
        transfer.assert_called_once()
    assert verify.call_args.kwargs["include_untracked"] is True
    released.assert_called_once()


def test_remote_snapshot_probe_detects_tracked_and_untracked_changes(tmp_path, monkeypatch):
    bash = str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe") if os.name == "nt" else shutil.which("bash")
    if not bash or not Path(bash).is_file():
        pytest.skip("local Linux-peer fixture requires Bash")
    repo = tmp_path / "vllm"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    (repo / "model.py").write_text("value = 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "model.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture"], check=True)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()

    def ssh(endpoint, script):
        result = subprocess.run([bash, "-s"], input=script.encode(), capture_output=True, check=True)
        return SimpleNamespace(stdout=result.stdout.decode())

    monkeypatch.setattr(parity, "ssh_exec", ssh)

    def probe():
        return parity.verify_runtime_commits_map(container=None, runtime_root=tmp_path.as_posix(),
                                                 expected={"vllm": head}, include_untracked=True)
    assert probe() == {"vllm": head}
    (repo / "extra.py").write_text("extra = 1\n")
    assert probe() == {"vllm": "dirty-runtime"}
    (repo / "extra.py").unlink()
    (repo / "model.py").write_text("value = 2\n")
    assert probe() == {"vllm": "dirty-runtime"}
