"""Local configured attribution: no network, host calls, containers or NPU work."""
import json
from unittest.mock import Mock

import pytest

from mindie_coordinator.agent_session import AgentSessions
from mindie_coordinator.hooks.mindie_session import handle
from mindie_coordinator.ready_runtime import user_container_name
from mindie_coordinator.task_client import TaskClient, coordinator_user
from mindie_coordinator.user_identity import IDENTITY_FILE_ENV, load_github_identity


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for name in (IDENTITY_FILE_ENV, "MINDIE_CONTEXT_FILE", "MINDIE_PARENT_CONTEXT", "MINDIE_ATTACH_CONTEXT",
                 "CODEX_THREAD_ID", "CODEX_SESSION_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("mindie_coordinator.task_client.getpass.getuser", lambda: "root")


def identity(tmp_path, login="Alice", user_id=42):
    path = tmp_path / f"{login} 身份.json"
    path.write_text(json.dumps({"schema": "mindie.github.v1", "login": login,
                               "github_user_id": user_id}), encoding="utf-8")
    return path


def context(tmp_path, native="native-first"):
    return AgentSessions(tmp_path / "sessions").attach("codex", native, str(tmp_path))


def test_configured_user_drives_task_and_default_container(tmp_path, monkeypatch):
    path = identity(tmp_path)
    monkeypatch.setenv(IDENTITY_FILE_ENV, str(path))
    native = context(tmp_path)
    owner = Mock()
    client = TaskClient(native["context_file"], service=owner)
    assert coordinator_user() == client.user == "alice"
    assert user_container_name(client.user) == "mindie-alice"
    assert client.context["session"]["user"] == "alice"
    assert client.context["session"]["github_identity"]["github_user_id"] == 42
    assert owner.mock_calls == []


def test_confirmed_login_only_binds_local_task_without_github_authentication(tmp_path, monkeypatch):
    path = tmp_path / 'identity-only.json'
    path.write_text(json.dumps({'schema': 'mindie.github.v1', 'login': 'Alice'}), encoding='utf-8')
    monkeypatch.setenv(IDENTITY_FILE_ENV, str(path))
    native = context(tmp_path)
    owner = Mock()
    client = TaskClient(native['context_file'], service=owner)
    assert coordinator_user() == client.user == 'alice'
    assert client.context['session']['github_identity'] == {'schema': 'mindie.github.v1', 'login': 'alice'}
    path.unlink()
    assert TaskClient(native['context_file'], service=owner).user == 'alice'
    assert owner.mock_calls == []


@pytest.mark.parametrize('user_id', [None, True, False, 0, -1, 1.5, '42'])
def test_present_numeric_id_still_requires_a_positive_integer(tmp_path, user_id):
    with pytest.raises(ValueError, match='positive numeric github_user_id'):
        load_github_identity(identity(tmp_path, user_id=user_id))


def test_resume_and_explicit_child_keep_bound_user_across_workspace_change(tmp_path, monkeypatch):
    first = context(tmp_path)
    alice = TaskClient(first["context_file"], identity_file=identity(tmp_path), service=Mock())
    monkeypatch.setenv(IDENTITY_FILE_ENV, str(identity(tmp_path, "Bob", 99)))
    resumed = TaskClient(first["context_file"], service=Mock())
    assert resumed.user == "alice"
    store = alice.store
    child = store.attach("claude", "child", str(tmp_path), parent_context=first["context_file"])
    assert TaskClient(child["context_file"], service=Mock()).user == "alice"
    second = context(tmp_path, "native-second")
    assert TaskClient(second["context_file"], service=Mock()).user == "bob"
    with pytest.raises(ValueError, match="already bound to user 'alice'"):
        TaskClient(first["context_file"], user="bob", service=Mock())
    assert store.context(first["attachment"]["id"])["session"]["user"] == "alice"


def test_explicit_user_and_standalone_package_remain_supported(tmp_path, monkeypatch):
    explicit = context(tmp_path)
    monkeypatch.setenv(IDENTITY_FILE_ENV, str(tmp_path / "bad-path.json"))
    assert coordinator_user("operator") == "operator"
    assert TaskClient(explicit["context_file"], user="operator", service=Mock()).user == "operator"
    monkeypatch.delenv(IDENTITY_FILE_ENV)
    standalone = context(tmp_path, "standalone")
    assert coordinator_user() == "root"
    assert TaskClient(standalone["context_file"], service=Mock()).user == "root"


def test_explicit_identity_path_wins_without_changing_process_environment(tmp_path, monkeypatch):
    ambient = identity(tmp_path, "Bob", 99)
    selected = identity(tmp_path)
    monkeypatch.setenv(IDENTITY_FILE_ENV, str(ambient))
    first = TaskClient(context(tmp_path)["context_file"], identity_file=selected, service=Mock())
    assert first.user == "alice"
    assert coordinator_user() == "bob"


@pytest.mark.parametrize("document,reason", [
    ("{broken", "not valid UTF-8 JSON"),
    ('{"schema":"other"}', "schema mindie.github.v1"),
    ('{"schema":"mindie.github.v1","login":"bad/login","github_user_id":42}', "invalid personal login"),
])
def test_invalid_configured_identity_explains_the_error_and_does_not_bind_root(tmp_path, document, reason):
    path = tmp_path / "identity.json"
    path.write_text(document, encoding="utf-8")
    native = context(tmp_path)
    with pytest.raises(ValueError, match=reason) as raised:
        TaskClient(native["context_file"], identity_file=path, service=Mock())
    assert str(path) in str(raised.value)
    assert "user" not in AgentSessions(tmp_path / "sessions").context(native["attachment"]["id"])["session"]


def test_missing_and_empty_configured_path_are_not_os_user_fallback(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Cannot read configured GitHub identity file"):
        load_github_identity(tmp_path / "missing.json")
    monkeypatch.setenv(IDENTITY_FILE_ENV, "")
    with pytest.raises(ValueError, match="path is empty"):
        coordinator_user()


def test_hook_binds_user_before_long_lived_mcp_can_apply_its_environment(tmp_path, monkeypatch):
    store = AgentSessions(tmp_path / "sessions")
    monkeypatch.setenv(IDENTITY_FILE_ENV, str(identity(tmp_path)))
    event = {"hook_event_name": "SessionStart", "session_id": "native", "cwd": str(tmp_path)}
    handle("codex", event, store)
    bound = store.native_context("codex", "native")
    assert bound["session"]["user"] == "alice"
    monkeypatch.setenv(IDENTITY_FILE_ENV, str(tmp_path / "now-unavailable.json"))
    handle("codex", {**event, "source": "resume"}, store)
    assert TaskClient(bound["context_file"], service=Mock()).user == "alice"
    monkeypatch.delenv(IDENTITY_FILE_ENV)
    handle("codex", {**event, "session_id": "standalone"}, store)
    assert "user" not in store.native_context("codex", "standalone")["session"]


def test_legacy_execution_attribution_is_retained(tmp_path):
    native = context(tmp_path)
    store = AgentSessions(tmp_path / "sessions")
    row = store.execution(native, "accepted-before-upgrade", {})
    row.update(user="legacy-operator", admitted=True)
    store.save_execution(row)
    client = TaskClient(native["context_file"], identity_file=identity(tmp_path), service=Mock())
    assert client.user == "legacy-operator"
    assert "github_identity" not in client.context["session"]


@pytest.mark.parametrize("via_cli", [False, True])
def test_provision_uses_configured_user_with_shared_root_transport(tmp_path, monkeypatch, via_cli, capsys):
    from mindie_coordinator import provision
    from mindie_coordinator.cli import main

    monkeypatch.setenv(IDENTITY_FILE_ENV, str(identity(tmp_path)))
    directory = Mock()
    monkeypatch.setattr(provision, "MachineDirectory", lambda: directory)
    calls = Mock(side_effect=["probe", "bootstrap", "smoke"])
    monkeypatch.setattr(provision.host_ops, "run_remote_script", calls)
    monkeypatch.setattr(provision.host_ops, "assert_remote_success",
                        Mock(side_effect=[{"free_port": 2222}, {}, {}]))
    monkeypatch.setattr(provision.host_ops, "find_public_key", lambda _: tmp_path / "key.pub")
    monkeypatch.setattr(provision.host_ops, "load_public_key", lambda _: "ssh-ed25519 test")

    if via_cli:
        assert main(["provision", "--host", "shared-host", "--image", "stable"]) == 0
        result = json.loads(capsys.readouterr().out)
    else:
        result = provision.provision_user_container(host="shared-host", image="stable")
    assert result["user"] == "alice"
    assert result["container_name"] == "mindie-alice"
    boot = calls.call_args_list[1]
    assert boot.args[0].user == "root"
    assert boot.kwargs["args"][0] == "mindie-alice"
    assert boot.kwargs["args"][5] == "alice"
    assert boot.kwargs["args"][1] == '2222'
    assert [call.kwargs.get('reuse_connection', False) for call in calls.call_args_list] == [True, False, True]
    assert 'if not False:' in calls.call_args_list[2].args[1]
    assert calls.call_args_list[2].args[0].user == "root"
    assert directory.upsert_machine.call_args.args[0]["user"] == "alice"


def test_provision_invalid_config_fails_before_remote_work(tmp_path, monkeypatch):
    from mindie_coordinator import provision

    monkeypatch.setenv(IDENTITY_FILE_ENV, str(tmp_path / "missing.json"))
    remote = Mock()
    monkeypatch.setattr(provision.host_ops, "run_remote_script", remote)
    with pytest.raises(ValueError, match="Cannot read configured GitHub identity file"):
        provision.provision_user_container(host="shared-host", image="stable")
    remote.assert_not_called()


@pytest.mark.parametrize("explicit_user,expected", [(None, "alice"), ("operator", "operator")])
def test_runtime_register_defaults_to_configured_user(tmp_path, monkeypatch, capsys, explicit_user, expected):
    from mindie_coordinator.cli import main

    monkeypatch.setenv(IDENTITY_FILE_ENV, str(identity(tmp_path)))
    owner = Mock()
    owner.runtime_register.return_value = {
        "id": "prepared", "user": expected, "container_name": f"mindie-{expected}",
        "python": "/venv/bin/python", "endpoint": {}, "state": "ready", "reuse_only": False,
    }
    monkeypatch.setattr("mindie_coordinator.service.ensure_daemon", lambda _: owner)
    args = ["runtime-register", "--runtime-id", "prepared", "--host", "shared-host",
            "--ssh-port", "2222", "--root", "/prepared", "--python", "/venv/bin/python",
            "--state-dir", str(tmp_path)]
    if explicit_user:
        args += ["--user", explicit_user]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["user"] == expected
    spec = owner.runtime_register.call_args.args[1]
    assert spec["user"] == expected
    assert spec["host_endpoint"]["user"] == spec["endpoint"]["user"] == "root"
