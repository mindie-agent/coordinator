"""Native tool context injection, including Cursor's asynchronous session start."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import Mock
import io
import json

import pytest

from mindie_coordinator.agent_session import AgentSessions
from mindie_coordinator.hooks.mindie_session import handle
from mindie_coordinator.hooks.mindie_session import main
from test_execution_inputs import repo


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for name in ("MINDIE_CONTEXT_FILE", "MINDIE_PARENT_CONTEXT", "MINDIE_ATTACH_CONTEXT",
                 "MINDIE_GITHUB_IDENTITY_FILE", "CODEX_THREAD_ID", "CODEX_SESSION_ID"):
        monkeypatch.delenv(name, raising=False)


def cursor_event(root, *, native="conversation", event="preToolUse", **fields):
    return {"conversation_id": native, "generation_id": "first-turn",
            "hook_event_name": event, "cursor_version": "test",
            "workspace_roots": [str(root)], "cwd": str(root),
            "tool_name": "MCP:mindie_run", "tool_input": {"command": "echo ready"}, **fields}


@pytest.mark.parametrize("first", ["sessionStart", "preToolUse"])
def test_cursor_first_tool_and_late_start_share_native_task_and_sources(tmp_path, first):
    source = repo(tmp_path / "project")
    store = AgentSessions(tmp_path / "sessions")
    handle("cursor", cursor_event(source, event=first), store)
    context = store.native_context("cursor", "conversation")
    for event in ("preToolUse", "sessionStart", "preToolUse"):
        output = handle("cursor", cursor_event(source, event=event, generation_id="later-turn"), store)
        if event == "preToolUse":
            assert output == {"updated_input": {"command": "echo ready", "context_file": context["context_file"]}}
    current = store.native_context("cursor", "conversation")
    assert current["session"]["id"] == context["session"]["id"]
    assert current["source_defaults"]["sources"]["project"]["path"] == str(source.resolve())
    assert len(store.sessions()) == 1


def test_cursor_concurrent_start_and_first_tools_create_one_task(tmp_path):
    source = repo(tmp_path / "project")
    store = AgentSessions(tmp_path / "sessions")
    barrier = Barrier(4)

    def start(event):
        barrier.wait(timeout=5)
        return handle("cursor", cursor_event(source, event=event), store)

    with ThreadPoolExecutor(4) as workers:
        results = list(workers.map(start, ["sessionStart", "preToolUse", "sessionStart", "preToolUse"]))
    context = store.native_context("cursor", "conversation")
    assert {results[index]["updated_input"]["context_file"] for index in (1, 3)} == {context["context_file"]}
    assert len(store.sessions()) == 1


@pytest.mark.parametrize("fields", [
    {"tool_name": "Shell", "tool_input": {"command": "echo mindie_run"}},
    {"tool_name": "MCP:unrelated_mindie_run"},
    {"tool_name": "MCP:mindie_run_extra"},
    {"tool_name": "mcp__unrelated__knowledge_query"},
    {"tool_name": "mcp__unrelated__remote_read"},
    {"tool_input": []},
    {"tool_input": {"command": "echo ready", "context_file": "explicit-context"}},
])
def test_cursor_unrelated_or_explicit_calls_never_open_the_task_registry(tmp_path, monkeypatch, fields):
    opening = Mock(side_effect=AssertionError("ordinary tool must not open registry"))
    monkeypatch.setattr("mindie_coordinator.hooks.mindie_session.AgentSessions", opening)
    assert handle("cursor", cursor_event(tmp_path, **fields)) == {}
    opening.assert_not_called()


def test_cursor_missing_native_id_never_uses_cwd_to_create_task(tmp_path, monkeypatch):
    opening = Mock(side_effect=AssertionError("missing identity must not open registry"))
    monkeypatch.setattr("mindie_coordinator.hooks.mindie_session.AgentSessions", opening)
    with pytest.raises(ValueError, match="no native session identity"):
        handle("cursor", cursor_event(tmp_path, native=""))
    opening.assert_not_called()


def test_cursor_first_tool_respects_explicit_association(tmp_path, monkeypatch):
    source = repo(tmp_path / "project")
    store = AgentSessions(tmp_path / "sessions")
    parent = store.attach("codex", "parent", str(source))
    monkeypatch.setenv("MINDIE_ATTACH_CONTEXT", parent["context_file"])
    output = handle("cursor", cursor_event(source), store)
    child = store.native_context("cursor", "conversation")
    assert output["updated_input"]["context_file"] == child["context_file"]
    assert child["session"]["id"] == parent["session"]["id"]
    assert len(store.sessions()) == 1


def test_cursor_unknown_subagent_is_not_attached_as_a_new_root(tmp_path):
    store = AgentSessions(tmp_path / "sessions")
    with pytest.raises(ValueError, match="association is missing"):
        handle("cursor", cursor_event(tmp_path, agent_id="child"), store)
    assert store.sessions() == []


@pytest.mark.parametrize("client", ["claude", "codex", "grok", "cursor"])
def test_message_gets_native_context_without_changing_explicit_arguments(tmp_path, client):
    store = AgentSessions(tmp_path / "sessions")
    context = store.attach(client, "native", str(tmp_path))
    arguments = {"recipient": {"reply_reference": "known-reference"}, "text": "ready"}
    payload = {"hook_event_name": "preToolUse", "session_id": "native", "cwd": str(tmp_path),
               "tool_name": "mcp__mindie-task__mindie_message", "tool_input": arguments}
    output = handle(client, payload, store)
    updated = output["updated_input"] if client == "cursor" else output["hookSpecificOutput"]["updatedInput"]
    assert updated == {**arguments, "context_file": context["context_file"]}
    assert "context_file" not in arguments
    explicit = {**arguments, "context_file": "caller-selected-context"}
    assert handle(client, {**payload, "tool_input": explicit}, store) == {}


@pytest.mark.parametrize("name", ["mindie-task__mindie_message", "mindie-knowledge__knowledge_capture", "remote-dev__remote_read"])
def test_grok_dispatcher_keeps_its_nested_envelope(tmp_path, name):
    store = AgentSessions(tmp_path / "sessions")
    context = store.attach("grok", "native", str(tmp_path))
    arguments = {"recipient": {"reply_reference": "known-reference"}, "text": "ready"}
    nested = {"tool_name": name, "tool_input": arguments}
    output = handle("grok", {"hookEventName": "pre_tool_use", "sessionId": "native", "cwd": str(tmp_path),
                             "toolName": "use_tool", "toolInput": nested}, store)
    assert output == {"hookSpecificOutput": {"hookEventName": "PreToolUse", "updatedInput": {
        **nested, "tool_input": {**arguments, "context_file": context["context_file"]}}}}


@pytest.mark.parametrize("client,name", [
    ("claude", "mcp__mindie-knowledge__knowledge_query"),
    ("claude", "mcp__mindie-knowledge__knowledge_explain"),
    ("codex", "mcp__mindie_knowledge__knowledge_capture"),
    ("codex", "mcp__remote_dev__remote_read"),
    ("cursor", "MCP:mindie-knowledge__knowledge_query"),
    ("cursor", "MCP:remote-dev__remote_job_status"),
    ("cursor", "MCP:knowledge_query"),
    ("cursor", "MCP:remote_read"),
])
def test_owned_companion_tools_receive_the_same_native_context(tmp_path, client, name):
    store = AgentSessions(tmp_path / "sessions")
    context = store.attach(client, "native", str(tmp_path))
    payload = {"hook_event_name": "preToolUse", "session_id": "native", "cwd": str(tmp_path),
               "tool_name": name, "tool_input": {"query": "existing input"}}
    output = handle(client, payload, store)
    updated = output["updated_input"] if client == "cursor" else output["hookSpecificOutput"]["updatedInput"]
    assert updated == {"query": "existing input", "context_file": context["context_file"]}
    assert handle(client, {**payload, "tool_input": {**updated, "context_file": "explicit-context"}}, store) == {}


@pytest.mark.parametrize("client", ["claude", "codex", "grok"])
@pytest.mark.parametrize("name", ["MCP:knowledge_query", "MCP:remote_read"])
def test_unqualified_companions_are_only_cursor_native_names(tmp_path, client, name):
    store = AgentSessions(tmp_path / "sessions")
    store.attach(client, "native", str(tmp_path))
    assert handle(client, {"hook_event_name": "PreToolUse", "session_id": "native", "cwd": str(tmp_path),
                           "tool_name": name, "tool_input": {}}, store) == {}


def test_hook_hint_reports_current_explicit_workspace_without_claiming_preparation(tmp_path):
    before, prepared = repo(tmp_path / "before"), repo(tmp_path / "prepared")
    store = AgentSessions(tmp_path / "sessions")
    context = store.attach("claude", "native", str(before))
    store.bind_sources(context, {"project": str(prepared)})
    output = handle("claude", {"hook_event_name": "SessionStart", "source": "compact", "session_id": "native", "cwd": str(before)}, store)
    hint = output["hookSpecificOutput"]["additionalContext"]
    assert context["context_file"] in hint
    assert f'Source defaults (explicit): {json.dumps({"project": str(prepared)})}' in hint
    assert "No session-creation call is needed" not in hint


@pytest.mark.parametrize("agent", [None, "main", "agent-1"])
def test_kimi_prompt_keeps_context_without_native_call_metadata(tmp_path, monkeypatch, capsys, agent):
    before, after = repo(tmp_path / "before"), repo(tmp_path / "after")
    store = AgentSessions(tmp_path / "sessions")
    handle("kimi", {"hook_event_name": "SessionStart", "session_id": "native", "cwd": str(before)}, store)
    if agent == "agent-1":
        handle("kimi", {"hook_event_name": "SubagentStart", "session_id": "native", "agent_id": agent,
                        "parent_agent_id": "main", "cwd": str(before)}, store)
    native_agent = agent if agent == "agent-1" else ""
    original = store.native_context("kimi", "native", native_agent)
    payload = {"hook_event_name": "UserPromptSubmit", "session_id": "native", "cwd": str(after)}
    if agent:
        payload["agent_id"] = agent
    monkeypatch.setattr("mindie_coordinator.hooks.mindie_session.AgentSessions", lambda: store)
    monkeypatch.setattr("sys.argv", ["hook", "--client", "kimi"])
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    capsys.readouterr()
    assert main() == 0
    text = capsys.readouterr().out.strip()
    if agent:
        assert text == ""
    else:
        assert original["context_file"] in text
    current = store.native_context("kimi", "native", native_agent)
    assert current["session"]["id"] == original["session"]["id"]
    assert current["attachment"]["cwd"] == str(after)
    assert {source["path"] for source in current["source_defaults"]["sources"].values()} == {str(after)}


@pytest.mark.parametrize("client", ["claude", "codex", "grok", "cursor"])
@pytest.mark.parametrize("event", ["UserPromptSubmit", "beforeSubmitPrompt"])
def test_prompt_refreshes_cwd_without_repeating_context(tmp_path, client, event):
    before, after = repo(tmp_path / "before"), repo(tmp_path / "after")
    store = AgentSessions(tmp_path / "sessions")
    payload = {"hook_event_name": "SessionStart", "session_id": "native", "cwd": str(before)}
    assert handle(client, payload, store)
    original = store.native_context(client, "native")
    for cwd in (before, after, after):
        assert handle(client, {**payload, "hook_event_name": event, "cwd": str(cwd)}, store) == {}
    current = store.native_context(client, "native")
    assert current["session"]["id"] == original["session"]["id"]
    assert current["attachment"]["cwd"] == str(after)
    assert {source["path"] for source in current["source_defaults"]["sources"].values()} == {str(after)}
    tool = {**payload, "hook_event_name": "PreToolUse", "cwd": str(after),
            "tool_name": "mindie_run", "tool_input": {"command": "echo ready"}}
    updated = handle(client, tool, store)
    arguments = updated["updated_input"] if client == "cursor" else updated["hookSpecificOutput"]["updatedInput"]
    assert arguments["context_file"] == current["context_file"]


@pytest.mark.parametrize("client", ["claude", "codex", "grok", "cursor"])
@pytest.mark.parametrize("fields", [
    {"tool_name": "Shell", "tool_input": {"command": "echo mindie_run"}},
    {"tool_name": "mcp__other__mindie_run_extra"},
    {"tool_name": "mindie_run", "tool_input": []},
    {"tool_name": "mindie_run", "tool_input": {"context_file": "explicit-context"}},
])
def test_ordinary_pretool_needs_no_identity_registry_or_git_scope(tmp_path, monkeypatch, capsys, client, fields):
    opening = Mock(side_effect=AssertionError("ordinary tool must not open registry"))
    scope = Mock(side_effect=AssertionError("ordinary tool must not probe Git scope"))
    monkeypatch.setattr("mindie_coordinator.hooks.mindie_session.AgentSessions", opening)
    monkeypatch.setattr("mindie_coordinator.hooks.mindie_session.in_project_scope", scope)
    payload = {"hook_event_name": "PreToolUse", "cwd": str(tmp_path), **fields}
    assert handle(client, payload) == {}
    monkeypatch.setattr("sys.argv", ["hook", "--client", client, "--project", str(tmp_path)])
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    assert main() == 0
    assert json.loads(capsys.readouterr().out) == {}
    opening.assert_not_called()
    scope.assert_not_called()


def test_unrelated_grok_dispatcher_never_opens_registry(monkeypatch):
    monkeypatch.setattr("mindie_coordinator.hooks.mindie_session.AgentSessions",
                        Mock(side_effect=AssertionError("ordinary nested call must not open registry")))
    assert handle("grok", {"hookEventName": "pre_tool_use", "toolName": "use_tool",
                           "toolInput": {"tool_name": "other_provider__remote_read", "tool_input": {"path": "/work/code.py"}}}) == {}


@pytest.mark.parametrize("client", ["claude", "codex", "cursor", "grok", "kimi"])
def test_prepared_roots_bind_on_native_start_resume_and_child_without_overriding_explicit_empty(tmp_path, client):
    root = repo(tmp_path / "workspace")
    child = repo(root / "business")
    sources = {"workspace": str(root), "business": str(child)}
    store = AgentSessions(tmp_path / "sessions")
    payload = {"hook_event_name": "SessionStart", "session_id": "native", "cwd": str(root)}
    handle(client, payload, store, sources=sources)
    context = store.native_context(client, "native")
    assert context["source_defaults"]["origin"] == "native-prepared"
    assert {name: row["path"] for name, row in context["source_defaults"]["sources"].items()} == sources
    handle(client, {**payload, "hook_event_name": "SubagentStart", "agent_id": "child"}, store, sources=sources)
    attached = store.native_context(client, "native", "child")
    assert attached["session"]["id"] == context["session"]["id"]
    assert attached["source_defaults"]["sources"] == context["source_defaults"]["sources"]
    store.bind_sources(context, {})
    handle(client, {**payload, "source": "resume"}, store, sources=sources)
    assert store.native_context(client, "native")["source_defaults"] == {"origin": "explicit", "sources": {}}


def test_prepared_source_map_is_replaced_after_native_cwd_change_and_not_rebound_per_tool(tmp_path, monkeypatch):
    first, second = repo(tmp_path / "first"), repo(tmp_path / "second")
    store = AgentSessions(tmp_path / "sessions")
    payload = {"hook_event_name": "SessionStart", "session_id": "native", "cwd": str(first)}
    handle("claude", payload, store, sources={"first": str(first)})
    handle("claude", {**payload, "hook_event_name": "UserPromptSubmit", "cwd": str(second)},
           store, sources={"second": str(second)})
    current = store.native_context("claude", "native")
    assert set(current["source_defaults"]["sources"]) == {"second"}
    monkeypatch.setattr(store, "bind_native_sources", Mock(side_effect=AssertionError("repeated source binding")))
    tool = {**payload, "hook_event_name": "PreToolUse", "cwd": str(second),
            "tool_name": "mindie_run", "tool_input": {}}
    assert handle("claude", tool, store, sources={"second": str(second)})


def test_selected_roots_extend_hook_scope_but_unselected_nested_clone_does_not(tmp_path):
    from mindie_coordinator.hooks.mindie_session import in_project_scope
    project, bundle = repo(tmp_path / "project"), repo(tmp_path / "bundle")
    selected, unrelated = repo(bundle / "selected"), repo(bundle / "unrelated")
    roots = {"workspace": str(bundle), "business": str(selected)}
    assert in_project_scope(selected, project, sources=roots)
    assert not in_project_scope(unrelated, project, sources=roots)


def test_explicit_empty_native_map_stays_empty(tmp_path):
    root = repo(tmp_path / "project")
    store = AgentSessions(tmp_path / "sessions")
    context = store.attach("codex", "native", str(root))
    context = store.bind_native_sources(context, sources={})
    assert context["source_defaults"] == {"origin": "native-prepared", "sources": {}}
    context = store.bind_native_sources(context)
    assert context["source_defaults"] == {"origin": "native-prepared", "sources": {}}


@pytest.mark.parametrize("client", ["claude", "codex", "cursor", "grok", "kimi"])
def test_detached_cli_resume_revalidates_prepared_sources_not_original_cwd(tmp_path, client):
    from test_execution_inputs import git
    original, editing = repo(tmp_path / "original"), repo(tmp_path / "editing")
    store = AgentSessions(tmp_path / "sessions")
    event = {"hook_event_name": "SessionStart", "session_id": "native", "cwd": str(original)}
    handle(client, event, store)
    initial = store.native_context(client, "native")
    store.bind_native_sources(initial, sources={"workspace": str(editing)})
    handle(client, {**event, "hook_event_name": "SessionEnd"}, store)
    assert store.context(initial["attachment"]["id"])["attachment"]["state"] == "detached"
    (editing / "resumed.txt").write_text("task revision advanced\n")
    git(editing, "add", "resumed.txt")
    git(editing, "commit", "-m", "advance selected repository")
    # The real CLI carries only native identity and the original cwd. It does
    # not resubmit the consumer's prepared map on SessionStart.
    handle(client, {**event, "source": "resume"}, store)
    resumed = store.native_context(client, "native")
    assert resumed["session"]["id"] == initial["session"]["id"]
    assert resumed["attachment"]["id"] == initial["attachment"]["id"]
    assert resumed["source_defaults"]["origin"] == "native-prepared"
    assert resumed["source_defaults"]["sources"]["workspace"]["path"] == str(editing)
    assert resumed["source_defaults"]["sources"]["workspace"]["head_at_bind"] == git(editing, "rev-parse", "HEAD")
    store.bind_sources(resumed, {})
    handle(client, {**event, "hook_event_name": "SessionEnd"}, store)
    handle(client, {**event, "source": "resume"}, store)
    assert store.native_context(client, "native")["source_defaults"] == {"origin": "explicit", "sources": {}}
    # A different native task in the same checkout must not inherit this W.
    handle(client, {**event, "session_id": "other"}, store)
    other = store.native_context(client, "other")
    assert other["session"]["id"] != initial["session"]["id"]
    assert {row["path"] for row in other["source_defaults"]["sources"].values()} == {str(original)}


def test_detached_resume_in_different_cwd_drops_prepared_selection(tmp_path):
    original, editing, moved = (repo(tmp_path / name) for name in ("original", "editing", "moved"))
    store = AgentSessions(tmp_path / "sessions")
    event = {"hook_event_name": "SessionStart", "session_id": "native", "cwd": str(original)}
    handle("codex", event, store, sources={"workspace": str(editing)})
    handle("codex", {**event, "hook_event_name": "SessionEnd"}, store)
    handle("codex", {**event, "cwd": str(moved), "source": "resume"}, store)
    result = store.native_context("codex", "native")["source_defaults"]
    assert result["origin"] == "native-cwd"
    assert {row["path"] for row in result["sources"].values()} == {str(moved)}


def test_lost_prepared_repository_stays_unknown_across_repeated_cli_resumes(tmp_path):
    original, editing = repo(tmp_path / "original"), repo(tmp_path / "editing")
    store = AgentSessions(tmp_path / "sessions")
    event = {"hook_event_name": "SessionStart", "session_id": "native", "cwd": str(original)}
    handle("codex", event, store, sources={"workspace": str(editing)})
    hidden = tmp_path / "temporarily-unavailable"
    editing.rename(hidden)
    for _ in range(2):
        handle("codex", {**event, "hook_event_name": "SessionEnd"}, store)
        handle("codex", {**event, "source": "resume"}, store)
        result = store.native_context("codex", "native")["source_defaults"]
        assert result["origin"] == "unknown" and result["sources"] == {}
        assert "Prepared native source roots are unavailable" in result["reason"]


def test_native_prepared_roots_capture_ignored_child_edits(tmp_path):
    from test_execution_inputs import git
    from mindie_coordinator.execution_sources import capture_sources
    root = repo(tmp_path / "workspace")
    (root / ".gitignore").write_text("/business/\n")
    git(root, "add", ".gitignore")
    git(root, "commit", "-m", "ignore independent source")
    child = repo(root / "business")
    store = AgentSessions(tmp_path / "sessions")
    handle("codex", {"hook_event_name": "SessionStart", "session_id": "native", "cwd": str(root)},
           store, sources={"workspace": str(root), "business": str(child)})
    context = store.native_context("codex", "native")
    defaults = {name: row["path"] for name, row in context["source_defaults"]["sources"].items()}
    before = capture_sources(defaults, tmp_path / "snapshots")
    (child / "value.txt").write_text("dirty child")
    assert git(root, "status", "--porcelain") == ""
    after = capture_sources(defaults, tmp_path / "snapshots")
    assert before["id"] != after["id"]
    changed = next(record for record in after["records"] if record["relpath"] == "business")
    assert git(child, "show", changed["commit"] + ":value.txt") == "dirty child"


def test_missing_prepared_child_never_becomes_implicit_source_free_execution(tmp_path):
    source = repo(tmp_path / "workspace")
    store = AgentSessions(tmp_path / "sessions")
    event = {"hook_event_name": "SessionStart", "session_id": "native", "cwd": str(source)}
    roots = {"workspace": str(source), "business": str(source / "missing")}
    handle("codex", event, store, sources=roots)
    context = store.native_context("codex", "native")
    assert context["source_defaults"]["origin"] == "unknown"
    assert "No empty source set" in context["source_defaults"]["reason"]
    store.bind_sources(context, {})
    assert store.native_context("codex", "native")["source_defaults"] == {"origin": "explicit", "sources": {}}
    # Recovered automatic references no longer retain a stale binding error.
    repo(source / "missing")
    restored = store.bind_native_sources(context, sources=roots)
    assert "source_error" not in restored["attachment"]


def test_prepared_roots_keep_global_kimi_silent_outside_git_project(tmp_path, monkeypatch, capsys):
    from mindie_coordinator.hooks.mindie_session import in_project_scope
    project = repo(tmp_path / "project")
    outside = tmp_path / "non-git"
    outside.mkdir()
    sources = {"workspace": str(project)}
    assert not in_project_scope(outside, project, sources=sources)
    monkeypatch.setattr("sys.argv", ["hook", "--client", "kimi", "--project", str(project)])
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"hook_event_name": "UserPromptSubmit",
                        "session_id": "unrelated", "cwd": str(outside)})))
    monkeypatch.setattr("mindie_coordinator.hooks.mindie_session.AgentSessions",
                        Mock(side_effect=AssertionError("outside scope must not open registry")))
    capsys.readouterr()
    assert main(sources=sources) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "" and captured.err == ""
