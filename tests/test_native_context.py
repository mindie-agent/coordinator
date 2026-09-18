from unittest.mock import Mock

import pytest

from mindie_coordinator.agent_session import AgentSessions, load_context
from mindie_coordinator.task_client import TaskClient
from test_execution_inputs import repo


@pytest.fixture
def native_env(tmp_path, monkeypatch):
    for name in ("MINDIE_CONTEXT_FILE", "MINDIE_PARENT_CONTEXT", "MINDIE_ATTACH_CONTEXT",
                 "CODEX_THREAD_ID", "CODEX_SESSION_ID", "GROK_SESSION_ID", "KIMI_SESSION_ID", "KIMI_AGENT_ID", "CURSOR_CONVERSATION_ID"):
        monkeypatch.delenv(name, raising=False)
    state = tmp_path / "sessions"
    monkeypatch.setenv("MINDIE_AGENT_SESSIONS_DIR", str(state))
    monkeypatch.setenv("CODEX_THREAD_ID", "native-first")
    return state


def test_native_resume_and_new_thread_are_local_and_distinct(native_env, monkeypatch):
    owner = Mock()
    first = TaskClient(service=owner)
    again = TaskClient(service=owner)
    assert first.context["session"]["id"] == again.context["session"]["id"]
    monkeypatch.setenv("CODEX_THREAD_ID", "native-second")
    second = TaskClient(service=owner)
    assert second.context["session"]["id"] != first.context["session"]["id"]
    assert first.status()["executions"] == []
    assert first.context["session"]["sources"] == {}
    assert owner.mock_calls == []


def test_explicit_context_wins_and_invalid_path_never_falls_back(native_env):
    expected = AgentSessions(native_env).attach("claude", "other-client", str(native_env))
    assert load_context(expected["context_file"])["session"]["id"] == expected["session"]["id"]
    with pytest.raises(FileNotFoundError):
        load_context(str(native_env / "missing.json"))
    assert len(AgentSessions(native_env).sessions()) == 1


def test_missing_or_conflicting_identity_does_not_create_state(native_env, monkeypatch):
    monkeypatch.setenv("CODEX_SESSION_ID", "another-native-id")
    with pytest.raises(ValueError, match="conflicting native"):
        load_context()
    assert not native_env.exists()
    monkeypatch.delenv("CODEX_THREAD_ID")
    with pytest.raises(ValueError, match="MindIE context is required"):
        load_context()
    assert not native_env.exists()


def test_mcp_does_not_adopt_the_server_process_native_identity(native_env):
    from mindie_coordinator.task_server import call_tool
    reply = call_tool("mindie_session", {})
    assert reply["isError"]
    assert not native_env.exists()


@pytest.mark.parametrize("client,key", [("grok", "GROK_SESSION_ID"), ("kimi", "KIMI_SESSION_ID")])
def test_native_shell_uses_existing_attachment_after_cd(native_env, monkeypatch, tmp_path, client, key):
    monkeypatch.delenv("CODEX_THREAD_ID")
    monkeypatch.setenv(key, "actual-native")
    store = AgentSessions(native_env)
    expected = store.attach(client, "actual-native", str(tmp_path / "original"))
    monkeypatch.chdir(tmp_path)
    assert load_context()["context_file"] == expected["context_file"]
    assert load_context()["attachment"]["cwd"] == str(tmp_path / "original")
    monkeypatch.setenv(key, "unknown-native")
    with pytest.raises(ValueError, match="missing or ambiguous"):
        load_context()
    assert len(store.sessions()) == 1


def test_kimi_native_child_does_not_fall_back_to_parent(native_env, monkeypatch, tmp_path):
    monkeypatch.delenv("CODEX_THREAD_ID")
    monkeypatch.setenv("KIMI_SESSION_ID", "native-kimi")
    monkeypatch.setenv("KIMI_AGENT_ID", "child")
    store = AgentSessions(native_env)
    parent = store.attach("kimi", "native-kimi", str(tmp_path))
    monkeypatch.setenv("KIMI_AGENT_ID", "main")
    assert load_context()["context_file"] == parent["context_file"]
    monkeypatch.setenv("KIMI_AGENT_ID", "child")
    with pytest.raises(ValueError, match="missing or ambiguous"):
        load_context()
    child = store.attach("kimi", "native-kimi", str(tmp_path), parent_context=parent["context_file"], agent_id="child")
    assert load_context()["context_file"] == child["context_file"]


def test_kimi_mcp_call_metadata_is_per_call_and_visible_as_text(native_env, monkeypatch, tmp_path):
    import json
    from mindie_coordinator.task_server import call_tool
    store = AgentSessions(native_env)
    first = store.attach("kimi", "first", str(tmp_path))
    second = store.attach("kimi", "second", str(tmp_path))
    for native, expected in (("first", first), ("second", second), ("first", first)):
        reply = call_tool("mindie_session", {"full": True}, {"kimi_code/session_id": native, "kimi_code/agent_id": "main"})
        assert not reply["isError"]
        assert reply["structuredContent"]["data"]["session"]["id"] == expected["session"]["id"]
        assert json.loads(reply["content"][0]["text"]) == reply["structuredContent"]
    with pytest.raises(ValueError, match="differs from this native Kimi caller"):
        call_tool("mindie_session", {"context_file": first["context_file"]}, {"kimi_code/session_id": "second"})
    with pytest.raises(ValueError, match="missing or ambiguous"):
        call_tool("mindie_session", {}, {"kimi_code/session_id": "first", "kimi_code/agent_id": "unknown-child"})


def codex_call_metadata(native):
    # Native Desktop capture: x-codex-turn-metadata is an object, not JSON text.
    # Identity and unrelated values are synthetic; no user paths are retained.
    return {"callId": "call-fixture", "threadId": native, "itemId": "item-fixture", "progressToken": 1,
            "x-codex-turn-metadata": {
                "session_id": native, "thread_id": native, "workspace_kind": "local", "turn_id": "turn-fixture",
                "turn_started_at_unix_ms": 1, "thread_source": "user", "turn_trigger": "user",
                "sandbox": "workspace-write", "sandbox_mode": "workspace-write", "auto_review_enabled": False,
                "node_repl_auto_review_required": False, "node_repl_disabled": False, "workspaces": {},
                "model": "fixture", "codex_version": "0.154.0-alpha.6.2", "reasoning_effort": "medium"}}


def test_codex_native_call_metadata_routes_each_call_to_existing_attachment(native_env, monkeypatch, tmp_path):
    import json
    from mindie_coordinator.task_server import handle
    store = AgentSessions(native_env)
    first = store.attach("codex", "first", str(tmp_path / "first"))
    second = store.attach("codex", "second", str(tmp_path / "second"))
    monkeypatch.setenv("CODEX_THREAD_ID", "unrelated-server-thread")
    monkeypatch.setenv("CODEX_SESSION_ID", "unrelated-server-session")
    monkeypatch.setattr(store, "attach", Mock(side_effect=AssertionError("call must not attach")))
    monkeypatch.setattr("mindie_coordinator.task_server.AgentSessions", lambda: store)
    for native, expected in (("first", first), ("second", second), ("first", first)):
        arguments = {"full": True}
        response = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "mindie_session", "arguments": arguments, "_meta": codex_call_metadata(native)}})
        reply = response["result"]
        assert not reply["isError"]
        assert reply["structuredContent"]["data"]["session"]["id"] == expected["session"]["id"]
        assert reply["structuredContent"]["data"]["attachment"]["cwd"] == expected["attachment"]["cwd"]
        assert json.loads(reply["content"][0]["text"]) == reply["structuredContent"]
        assert arguments == {"full": True}
    assert len(store.sessions()) == 2
    store.attach.assert_not_called()


def test_codex_native_metadata_preserves_matching_context_and_rejects_conflict(native_env, tmp_path):
    from mindie_coordinator.task_server import call_tool
    store = AgentSessions(native_env)
    first = store.attach("codex", "first", str(tmp_path))
    second = store.attach("codex", "second", str(tmp_path))
    assert not call_tool("mindie_session", {"context_file": first["context_file"]}, codex_call_metadata("first"))["isError"]
    with pytest.raises(ValueError, match="differs from this native Codex caller"):
        call_tool("mindie_session", {"context_file": second["context_file"]}, codex_call_metadata("first"))


@pytest.mark.parametrize("turn", [None, "{\"thread_id\":\"native-first\"}", {},
                                    {"session_id": "native-first"}, {"thread_id": ""}, {"thread_id": 42}])
def test_codex_invalid_metadata_never_falls_back_to_process_identity(native_env, turn):
    from mindie_coordinator.task_server import call_tool
    with pytest.raises(ValueError, match="invalid native Codex call identity"):
        call_tool("mindie_session", {}, {"threadId": "native-first", "x-codex-turn-metadata": turn})
    assert not native_env.exists()


@pytest.mark.parametrize("metadata", [None, {}, {"threadId": "native-first"}, {"session_id": "native-first"}])
def test_codex_missing_call_metadata_has_no_alias_or_process_fallback(native_env, metadata):
    from mindie_coordinator.task_server import call_tool
    assert call_tool("mindie_session", {}, metadata)["isError"]
    assert not native_env.exists()


def test_codex_unknown_metadata_does_not_create_or_reassign_attachment(native_env, tmp_path):
    from mindie_coordinator.task_server import call_tool
    store = AgentSessions(native_env)
    known = store.attach("codex", "known", str(tmp_path))
    metadata = codex_call_metadata("unknown")
    metadata["threadId"] = metadata["x-codex-turn-metadata"]["session_id"] = "known"
    with pytest.raises(ValueError, match="missing or ambiguous"):
        call_tool("mindie_session", {"context_file": known["context_file"]}, metadata)
    assert len(store.sessions()) == 1
    assert store.native_context("codex", "known")["context_file"] == known["context_file"]


def test_kimi_nested_agent_hooks_keep_exact_parent(native_env, tmp_path):
    from mindie_coordinator.hooks.mindie_session import handle
    store = AgentSessions(native_env)
    base = {"session_id": "native-kimi", "cwd": str(tmp_path)}
    handle("kimi", {**base, "hook_event_name": "SessionStart", "agent_id": "main"}, store)
    handle("kimi", {**base, "hook_event_name": "SubagentStart", "agent_id": "child", "parent_agent_id": "main"}, store)
    handle("kimi", {**base, "hook_event_name": "SubagentStart", "agent_id": "grandchild", "parent_agent_id": "child"}, store)
    child = store.native_context("kimi", "native-kimi", "child")
    grandchild = store.native_context("kimi", "native-kimi", "grandchild")
    assert grandchild["attachment"]["parent_id"] == child["attachment"]["id"]


def test_native_status_preserves_a_finished_task(native_env):
    context = load_context()
    store = AgentSessions(native_env)
    with store.transaction() as db:
        session = store.get(db, "session", context["session"]["id"])
        session["state"] = "finished"
        store.put(db, "session", session)
    assert TaskClient(service=Mock()).status()["session"]["state"] == "finished"


def test_explicit_parent_preserves_association(native_env, monkeypatch):
    parent = load_context()
    monkeypatch.setenv("CODEX_THREAD_ID", "child-native")
    monkeypatch.setenv("MINDIE_PARENT_CONTEXT", parent["context_file"])
    child = load_context()
    assert child["session"]["id"] == parent["session"]["id"]
    assert child["attachment"]["parent_id"] == parent["attachment"]["id"]


def test_first_native_cli_binds_sources_once_and_shell_cd_keeps_them(native_env, tmp_path, monkeypatch):
    project = repo(tmp_path / "project")
    other = repo(tmp_path / "other")
    monkeypatch.chdir(project)
    first = load_context()
    assert first["source_defaults"]["origin"] == "native-cwd"
    assert first["source_defaults"]["sources"]["project"]["path"] == str(project.resolve())
    assert first["session"]["sources"] == {}
    monkeypatch.chdir(other)
    later = load_context()
    assert later["context_file"] == first["context_file"]
    assert later["attachment"]["cwd"] == str(project.resolve())
    assert later["source_defaults"] == first["source_defaults"]


@pytest.mark.parametrize("association_env", ["MINDIE_PARENT_CONTEXT", "MINDIE_ATTACH_CONTEXT"])
def test_first_native_association_preserves_explicit_task_sources(native_env, tmp_path, monkeypatch, association_env):
    project = repo(tmp_path / "project")
    child_root = repo(tmp_path / "child")
    monkeypatch.chdir(project)
    parent = load_context()
    store = AgentSessions(native_env)
    explicit = store.bind_sources(parent, {"chosen": str(project)})
    monkeypatch.setenv("CODEX_THREAD_ID", "native-child")
    monkeypatch.setenv(association_env, parent["context_file"])
    monkeypatch.chdir(child_root)
    child = load_context()
    assert child["session"]["id"] == parent["session"]["id"]
    assert child["attachment"]["sources"]["child"]["path"] == str(child_root.resolve())
    assert child["source_defaults"] == explicit["source_defaults"]
    store.bind_sources(parent, {})
    monkeypatch.chdir(project)
    resumed = load_context()
    assert resumed["attachment"]["cwd"] == str(child_root.resolve())
    assert resumed["source_defaults"] == {"origin": "explicit", "sources": {}}


def test_native_cli_without_git_still_attaches_locally(native_env, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    context = load_context()
    assert context["source_defaults"]["sources"] == {}
    assert "source reference not yet bound" in capsys.readouterr().err


def test_cursor_shell_first_call_and_hook_share_exact_native_context(native_env, tmp_path, monkeypatch):
    from mindie_coordinator.hooks.mindie_session import handle
    project = repo(tmp_path / "project")
    native = "4c009c90-471a-4be5-b9b9-963f0ab6fb4b"
    monkeypatch.delenv("CODEX_THREAD_ID")
    monkeypatch.setenv("CURSOR_CONVERSATION_ID", native)
    monkeypatch.chdir(project)
    first = load_context()
    store = AgentSessions(native_env)
    handle("cursor", {"hook_event_name": "sessionStart", "conversation_id": native, "cwd": str(project)}, store)
    monkeypatch.chdir(tmp_path)
    assert load_context()["context_file"] == first["context_file"]
    assert load_context()["source_defaults"]["sources"]["project"]["path"] == str(project)
    assert len(store.sessions()) == 1
    monkeypatch.setenv("CURSOR_CONVERSATION_ID", "escaped_2Fid")
    with pytest.raises(ValueError):
        load_context()
    assert len(store.sessions()) == 1


def test_existing_user_container_needs_no_image_selection(monkeypatch):
    from mindie_coordinator import provision
    from mindie_coordinator.service import CoordinatorService
    verify = Mock(side_effect=AssertionError("implicit image must keep the existing hot path"))
    monkeypatch.setattr(provision, "provision_user_container", verify)
    service = object.__new__(CoordinatorService)
    service.backend = Mock()
    record = {"host": {"ip": "192.0.2.10", "machine_type": "A3"},
              "container": {"name": "mindie-alice", "ssh_port": 2201}, "user": "alice"}
    service._configured_machines = lambda: [record]
    donor = service._ensure_user_container("alice", {}, {}, set())
    assert donor["ssh_port"] == 2201
    assert donor["recipe"] is None
    assert service.backend.host.call_args.args[1]["action"] == "container-ssh-reserve"
    service.backend.reset_mock()
    assert service._ensure_user_container("bob", {}, {}, set()) is None
    service.backend.host.assert_not_called()
    record["container"] = {}
    assert service._ensure_user_container("alice", {}, {}, set()) is None
    service.backend.host.assert_not_called()
    verify.assert_not_called()


@pytest.mark.parametrize("image", ["registry.example/ascend:v1.2.3", "registry.example/ascend@sha256:" + "a" * 64])
def test_first_run_can_provision_an_explicit_fixed_image(image, monkeypatch):
    from mindie_coordinator import provision
    from mindie_coordinator.service import CoordinatorService
    service = object.__new__(CoordinatorService)
    service.backend = Mock()
    record = {"host": {"ip": "192.0.2.10", "machine_type": "A3"},
              "container": {"name": "mindie-donor", "ssh_port": 2201}, "user": "donor"}
    service._configured_machines = lambda: [record]
    create = Mock(return_value={"ssh_port": 2202})
    monkeypatch.setattr(provision, "provision_user_container", create)
    donor = service._ensure_user_container("recipient", {"image": image}, {"host": "192.0.2.10"}, set())
    assert donor["recipe"] == image
    assert donor["container_name"] == "mindie-recipient"
    assert donor["ssh_port"] == 2202
    assert create.call_args.kwargs["image"] == image
    assert create.call_args.kwargs["user"] == "recipient"
    assert record["container"]["name"] == "mindie-donor"


@pytest.mark.parametrize("image", ["typo", "registry.example/ascend", "registry.example/ascend:latest", "auto"])
def test_first_run_does_not_provision_implicit_or_unsupported_images(image, monkeypatch):
    from mindie_coordinator import provision
    from mindie_coordinator.service import CoordinatorService
    service = object.__new__(CoordinatorService)
    service.backend = Mock()
    service._configured_machines = lambda: [{"host": {"ip": "192.0.2.10"}}]
    create = Mock()
    monkeypatch.setattr(provision, "provision_user_container", create)
    assert service._ensure_user_container("recipient", {"image": image}, {}, set()) is None
    create.assert_not_called()


@pytest.mark.parametrize("selector", ["image", "recipe"])
@pytest.mark.parametrize("image", ["registry.example/ascend:v1.2.3", "registry.example/ascend@sha256:" + "a" * 64])
def test_existing_user_container_verifies_explicit_image_before_returning_donor(selector, image, monkeypatch):
    from mindie_coordinator import provision
    from mindie_coordinator.service import CoordinatorService
    service = object.__new__(CoordinatorService)
    service.backend = Mock()
    record = {"host": {"ip": "192.0.2.10", "machine_type": "A3"},
              "container": {"name": "mindie-alice", "ssh_port": 2201}, "user": "alice"}
    service._configured_machines = lambda: [record]
    verify = Mock(return_value={"ssh_port": 2201})
    monkeypatch.setattr(provision, "provision_user_container", verify)
    donor = service._ensure_user_container("alice", {selector: image}, {}, set())
    assert donor["recipe"] == image and donor["ssh_port"] == 2201
    verify.assert_called_once()
    assert verify.call_args.kwargs["image"] == image
    assert verify.call_args.kwargs["ssh_port"] == 2201
    assert verify.call_args.kwargs["user"] == "alice"
    # The provision owner reserves its validated container's port itself.
    service.backend.host.assert_not_called()


def test_existing_container_image_mismatch_cannot_be_relabelled_as_requested_image(monkeypatch):
    from mindie_coordinator import provision
    from mindie_coordinator.provision import existing_container
    from mindie_coordinator.provision.host_ops import MachineManagementError, RemoteResult, SshTarget
    from mindie_coordinator.service import CoordinatorService
    service = object.__new__(CoordinatorService)
    service.backend = Mock()
    service.backend.host.return_value = {"port": 2201}
    record = {"host": {"ip": "192.0.2.10"},
              "container": {"name": "mindie-alice", "ssh_port": 2201}, "user": "alice",
              "image": {"requested": "registry.example/ascend:old"}}
    service._configured_machines = lambda: [record]
    monkeypatch.setattr(existing_container, 'observe_existing', lambda *a, **k: {'status': 'unknown'})
    target = SshTarget(host="192.0.2.10", user="root", port=22)
    remote = Mock(side_effect=[
        RemoteResult(target, 0, "", "", {"success": True, "suggested_port": 2299}),
        RemoteResult(target, 21, "", "", {"success": False,
                     "error": "existing container image does not match the requested selector"}),
    ])
    monkeypatch.setattr(provision.host_ops, "run_remote_script", remote)
    monkeypatch.setattr(provision.host_ops, "find_public_key", lambda _: "unused-fixture-key")
    monkeypatch.setattr(provision.host_ops, "load_public_key", lambda _: "ssh-ed25519 test")
    with pytest.raises(MachineManagementError, match="existing container image does not match"):
        service._ensure_user_container("alice", {"image": "registry.example/ascend:new"}, {}, set())
    assert record["image"]["requested"] == "registry.example/ascend:old"
    assert record["container"] == {"name": "mindie-alice", "ssh_port": 2201}
    # Exercise the real provision owner: mismatch fails before smoke/upsert,
    # and the bootstrap is explicitly forbidden to replace an existing image.
    assert remote.call_count == 2
    bootstrap = remote.call_args_list[1]
    assert bootstrap.kwargs["args"][0:2] == ["mindie-alice", "2201"]
    assert bootstrap.kwargs["args"][6] == "false"
    service.backend.machines.upsert_machine.assert_not_called()
