"""Native cwd ownership and automatic source defaults use real Git worktrees."""
import io
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from mindie_coordinator.agent_session import AgentSessions, worktree_reference
from mindie_coordinator.hooks.mindie_session import handle, in_project_scope, main
from mindie_coordinator.presentation import compact_data
from mindie_coordinator.task_client import TaskClient
from test_execution_inputs import client, git, repo


def linked_worktree(source, target):
    git(source, "worktree", "add", "--detach", str(target), "HEAD")
    return target


def start(store, native, cwd, *, source="startup"):
    handle("codex", {"hook_event_name": "SessionStart", "session_id": native,
                     "cwd": str(cwd), "source": source}, store)
    return store.native_context("codex", native)


def test_scope_accepts_external_linked_worktree_subdir_but_not_a_clone_or_nested_repo(tmp_path):
    source = repo(tmp_path / "project")
    worktree = linked_worktree(source, tmp_path / "elsewhere")
    child = worktree / "src"
    child.mkdir()
    unrelated = repo(source / "unrelated")
    clone = tmp_path / "clone"
    git(source, "clone", str(source), str(clone))
    (source / "new-non-git-folder").mkdir()
    assert in_project_scope(child, source)
    assert in_project_scope(source / "new-non-git-folder", source)
    assert not in_project_scope(clone, source)
    assert not in_project_scope(unrelated, source)
    assert not in_project_scope(tmp_path / "missing", source)


def test_scope_preserves_registered_submodule_and_explicit_module_worktree_scope(tmp_path):
    source = repo(tmp_path / "project")
    module = repo(tmp_path / "module-source")
    git(source, "-c", "protocol.file.allow=always", "submodule", "add", str(module), "business")
    subdir = source / "business" / "src"
    subdir.mkdir()
    assert in_project_scope(subdir, source)
    worktree = linked_worktree(source / "business", tmp_path / "module-worktree")
    assert in_project_scope(worktree, source / "business")


def test_non_git_and_unborn_projects_keep_their_own_local_scope(tmp_path):
    project = tmp_path / "project"
    child = project / "child"
    child.mkdir(parents=True)
    assert in_project_scope(child, project)
    git(project, "init")
    assert in_project_scope(child, project)
    assert not in_project_scope(tmp_path, project)


def test_hook_entry_scopes_before_creating_native_attachment(tmp_path, monkeypatch, capsys):
    source = repo(tmp_path / "project")
    outside = linked_worktree(source, tmp_path / "native-worktree")
    foreign = repo(tmp_path / "other-project")
    store = AgentSessions(tmp_path / "registry")
    monkeypatch.setattr("mindie_coordinator.hooks.mindie_session.AgentSessions", lambda: store)
    monkeypatch.setattr("sys.argv", ["hook", "--client", "codex", "--project", str(source)])
    for directory, native in [(outside, "actual-native-id"), (foreign, "foreign-native-id")]:
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"hook_event_name": "SessionStart",
                                                               "session_id": native, "cwd": str(directory)})))
        assert main() == 0
        capsys.readouterr()
    accepted = store.native_context("codex", "actual-native-id")
    assert accepted["attachment"]["cwd"] == str(outside.resolve())
    assert accepted["source_defaults"]["sources"]["project"]["path"] == str(outside.resolve())
    with pytest.raises(ValueError, match="association is missing"):
        store.native_context("codex", "foreign-native-id")
    assert len(store.sessions()) == 1


def test_two_attachments_have_independent_auto_sources_and_resume_keeps_identity(tmp_path):
    source = repo(tmp_path / "project")
    first = linked_worktree(source, tmp_path / "first")
    second = linked_worktree(source, tmp_path / "second")
    moved = linked_worktree(source, tmp_path / "moved")
    store = AgentSessions(tmp_path / "registry")
    a = start(store, "native-a", first)
    b = store.attach("codex", "native-b", str(second), association=a["context_file"])
    b = start(store, "native-b", second)
    assert a["session"]["id"] == b["session"]["id"]
    assert a["source_defaults"]["sources"]["project"]["path"] == str(first.resolve())
    assert b["source_defaults"]["sources"]["project"]["path"] == str(second.resolve())
    resumed = start(store, "native-a", moved, source="resume")
    assert resumed["context_file"] == a["context_file"]
    assert resumed["session"]["id"] == a["session"]["id"]
    assert resumed["attachment"]["cwd"] == str(moved.resolve())
    assert resumed["source_defaults"]["sources"]["project"]["path"] == str(moved.resolve())
    start(store, "native-a", moved, source="compact")
    assert store.context(b["attachment"]["id"])["source_defaults"] == b["source_defaults"]
    assert store.context(a["attachment"]["id"])["session"]["sources"] == {}


def test_subagent_auto_defaults_and_explicit_empty_override_do_not_cross_attachments(tmp_path):
    source = repo(tmp_path / "project")
    child_root = linked_worktree(source, tmp_path / "child")
    store = AgentSessions(tmp_path / "registry")
    parent = start(store, "parent", source)
    handle("codex", {"hook_event_name": "SubagentStart", "session_id": "parent", "agent_id": "child-id",
                     "cwd": str(child_root)}, store)
    child = store.native_context("codex", "parent", "child-id")
    assert child["session"]["id"] == parent["session"]["id"]
    assert child["source_defaults"]["sources"]["project"]["path"] == str(child_root.resolve())
    assert store.native_context("codex", "parent")["source_defaults"] == parent["source_defaults"]
    explicit = store.bind_sources(parent, {"chosen": str(source)})
    assert store.context(child["attachment"]["id"])["source_defaults"] == explicit["source_defaults"]
    store.bind_sources(parent, {})
    start(store, "parent", child_root, source="resume")
    defaults = store.context(child["attachment"]["id"])["source_defaults"]
    assert defaults == {"origin": "explicit", "sources": {}}
    assert compact_data(store.context(child["attachment"]["id"]))["source_defaults"] == defaults


def test_prompt_handoff_refreshes_only_its_native_attachment(tmp_path):
    source = repo(tmp_path / "project")
    changed = linked_worktree(source, tmp_path / "changed")
    store = AgentSessions(tmp_path / "registry")
    original = start(store, "native", source)
    handle("codex", {"hook_event_name": "UserPromptSubmit", "session_id": "native", "cwd": str(changed)}, store)
    current = store.native_context("codex", "native")
    assert current["context_file"] == original["context_file"]
    assert current["source_defaults"]["sources"]["project"]["path"] == str(changed.resolve())
    non_git = tmp_path / "new-project"
    non_git.mkdir()
    current = start(store, "native", non_git, source="resume")
    assert current["attachment"]["cwd"] == str(non_git.resolve())
    assert current["source_defaults"]["sources"] == {}


@pytest.mark.parametrize("explicit", [None, {}, {"chosen": "source"}])
def test_claude_pretool_after_enter_worktree_refreshes_only_automatic_sources(tmp_path, explicit):
    source = repo(tmp_path / "project")
    changed = linked_worktree(source, tmp_path / "entered-worktree")
    store = AgentSessions(tmp_path / "registry")
    handle("claude", {"hook_event_name": "SessionStart", "session_id": "native", "cwd": str(source)}, store)
    original = store.native_context("claude", "native")
    sibling = store.attach("claude", "sibling", str(source), association=original["context_file"])
    sibling = store.bind_native_sources(sibling)
    if explicit is not None:
        explicit = {name: str(source) for name in explicit}
        store.bind_sources(original, explicit)
    sibling = store.context(sibling["attachment"]["id"])
    result = handle("claude", {"hook_event_name": "PreToolUse", "session_id": "native", "cwd": str(changed),
                               "tool_name": "mcp__mindie_task__mindie_run", "tool_input": {"command": "echo ready"}}, store)
    current = store.native_context("claude", "native")
    assert current["context_file"] == original["context_file"]
    assert current["session"]["id"] == original["session"]["id"]
    assert current["attachment"]["cwd"] == str(changed.resolve())
    expected = explicit if explicit is not None else {"project": str(changed.resolve())}
    assert {name: row["path"] for name, row in current["source_defaults"]["sources"].items()} == expected
    assert store.context(sibling["attachment"]["id"]) == sibling
    assert result["hookSpecificOutput"]["updatedInput"]["context_file"] == original["context_file"]


def test_claude_pretool_does_not_infer_cwd_or_unknown_identity(tmp_path, monkeypatch):
    source = repo(tmp_path / "project")
    changed = linked_worktree(source, tmp_path / "another-worktree")
    store = AgentSessions(tmp_path / "registry")
    handle("claude", {"hook_event_name": "SessionStart", "session_id": "native", "cwd": str(source)}, store)
    monkeypatch.chdir(changed)
    tool = {"hook_event_name": "PreToolUse", "session_id": "native", "tool_name": "mcp__mindie_task__mindie_session"}
    handle("claude", tool, store)
    assert store.native_context("claude", "native")["attachment"]["cwd"] == str(source.resolve())
    with pytest.raises(ValueError, match="association is missing"):
        handle("claude", {**tool, "session_id": "unknown", "cwd": str(changed)}, store)
    assert len(store.sessions()) == 1


@pytest.mark.parametrize("handoff", ["resume", "claude-pretool"])
def test_handoff_leaves_accepted_sources_and_service_identity_immutable(client, tmp_path, handoff):
    source = repo(tmp_path / "project", "accepted-A")
    changed = linked_worktree(source, tmp_path / "changed")
    if handoff == "claude-pretool":
        native = "claude-native"
        handle("claude", {"hook_event_name": "SessionStart", "session_id": native, "cwd": str(source)}, client.store)
        original = client.store.native_context("claude", native)
        client = TaskClient(original["context_file"], service=client.coordinator, user="alice")
    else:
        native = client.context["attachment"]["native_session_id"]
        original = start(client.store, native, source)
    first = client.run("serve", service="api")
    before = client.store.executions(original["session"]["id"])[0]
    (changed / "value.txt").write_text("accepted-B")
    if handoff == "claude-pretool":
        handle("claude", {"hook_event_name": "PreToolUse", "session_id": native, "cwd": str(changed),
                          "tool_name": "mcp__mindie_task__mindie_run", "tool_input": {}}, client.store)
    else:
        start(client.store, native, changed, source="resume")
    with pytest.raises(ValueError, match="different inputs: sources"):
        client.run("serve", service="api")
    second = client.run("echo new submission")
    rows = {row["id"]:row for row in client.store.executions(original["session"]["id"])}
    assert rows[first["execution_id"]] == before
    assert rows[first["execution_id"]]["spec"]["source_snapshot"]["sources"]["project"]["path"] == str(source.resolve())
    assert rows[second["execution_id"]]["spec"]["source_snapshot"]["sources"]["project"]["path"] == str(changed.resolve())
    fixed = before["spec"]["source_snapshot"]["records"][0]
    assert git(source, "show", fixed["commit"] + ":value.txt") == "accepted-A"


def test_legacy_source_mapping_is_not_guessed_and_per_run_explicit_sources_remain_usable(tmp_path):
    source = repo(tmp_path / "project")
    worktree = linked_worktree(source, tmp_path / "changed")
    store = AgentSessions(tmp_path / "registry")
    context = start(store, "legacy", source)
    legacy = {"project": worktree_reference(str(source))}
    with store.transaction() as db:
        session = store.get(db, "session", context["session"]["id"])
        session.pop("source_mode")
        session["sources"] = legacy
        store.put(db, "session", session)
    current = start(store, "legacy", worktree, source="resume")
    assert current["session"]["sources"] == legacy
    assert current["source_defaults"]["origin"] == "unknown"
    owner = Mock()
    client = TaskClient(current["context_file"], service=owner)
    with patch("mindie_coordinator.execution_sources.capture_sources", side_effect=AssertionError("must not capture old defaults")):
        with pytest.raises(ValueError, match="no provenance"):
            client.run("echo ready")
    owner.admit.assert_not_called()
    client.run("echo ready", sources={"project":str(worktree)})
    assert owner.admit.call_args.args[3]["source_snapshot"]["sources"]["project"]["path"] == str(worktree.resolve())
    client.sources({})
    client.run("echo without sources")
    assert owner.admit.call_args.args[3]["source_snapshot"]["records"] == []
