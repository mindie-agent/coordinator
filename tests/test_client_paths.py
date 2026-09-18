import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from mindie_coordinator.agent_session import AgentSessions, load_context, worktree_reference
from mindie_coordinator.client_paths import client_path
from mindie_coordinator.hooks.mindie_session import handle
from mindie_coordinator.service import ensure_daemon, require_native_owner
from test_execution_inputs import repo


def test_only_standard_wsl_drive_mounts_map_to_windows():
    assert client_path("/mnt/d/project/中文 path/repo", platform="nt") == "D:\\project\\中文 path\\repo"
    assert client_path("/mnt/c", platform="nt") == "C:\\"
    assert client_path("/home/alice/project", platform="nt") == "/home/alice/project"
    assert client_path("/mnt/data/project", platform="nt") == "/mnt/data/project"
    assert client_path("/mnt/d/project", platform="posix") == "/mnt/d/project"


@pytest.mark.skipif(os.name != "nt", reason="native Windows path resolution")
def test_wsl_context_worktree_and_hook_cwd_resolve_to_one_registry(tmp_path):
    def wsl(path):
        path = Path(path).resolve()
        return "/mnt/" + path.drive[0].lower() + path.as_posix()[2:]

    source = repo(tmp_path / "project")
    store = AgentSessions(tmp_path / "state")
    context = store.attach("grok", "native-wsl", wsl(source))
    assert context["attachment"]["cwd"] == str(source.resolve())
    assert load_context(wsl(context["context_file"]))["session"]["id"] == context["session"]["id"]
    assert worktree_reference(wsl(source))["path"] == str(source.resolve())
    second = repo(tmp_path / "business")
    store.bind_sources(context, {"business": str(second)})
    handle("grok", {"hookEventName": "SessionStart", "sessionId": "native-wsl", "cwd": wsl(source)}, store)
    assert set(store.context(context["attachment"]["id"])["session"]["sources"]) == {"business"}


def test_posix_refuses_windows_owned_state_before_starting_a_daemon(tmp_path):
    (tmp_path / "coordinator.ipc").write_text('{"host":"127.0.0.1","port":1234}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="managed Windows Python"):
        require_native_owner(tmp_path, platform="posix")
    with patch("mindie_coordinator.service.require_native_owner", side_effect=lambda path: require_native_owner(path, platform="posix")), \
         patch("mindie_coordinator.service.CoordinatorClient") as client:
        with pytest.raises(RuntimeError, match="second Linux coordinator"):
            ensure_daemon(tmp_path)
    client.assert_not_called()


def test_posix_can_resume_its_own_socket_marker(tmp_path):
    (tmp_path / "coordinator.ipc").write_text('{"socket":"/tmp/vc-user-test.sock"}', encoding="utf-8")
    require_native_owner(tmp_path, platform="posix")
