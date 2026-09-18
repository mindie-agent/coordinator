#!/usr/bin/env python3
"""Native-client session attachment hook. Local only; never contacts the fleet."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from mindie_coordinator.agent_session import CLIENTS, AgentSessions, git_common_directory, load_context
from mindie_coordinator.client_paths import client_path


# Cross-client payload discriminators, pinned by each client's documented hook
# contract and by the hook contract tests in session-management/tests:
# - Grok payloads use the camelCase field `hookEventName` and never carry
#   `cursor_version`; such a payload reaching the Claude or Cursor adapter is a
#   Grok/IDE compatibility import and must no-op.
# - A genuine Cursor payload carries `cursor_version` (Cursor hook input schema,
#   cursor.com/docs/hooks) and distinguishes its event with the snake_case field
#   `hook_event_name`; either marker means the payload is Cursor's own and the
#   Cursor adapter should attach.
# Only the client's own adapter may create a root attachment.
GROK_EVENT_FIELD = "hookEventName"
CURSOR_VERSION_FIELD = "cursor_version"
TASK_TOOLS = ("mindie_session", "mindie_run", "mindie_execution", "mindie_finish", "mindie_message")


def task_tool(name: str) -> bool:
    """Recognize the task tool itself, including native MCP-qualified names."""
    return any(name == tool or name.endswith(("__" + tool, ":" + tool)) for tool in TASK_TOOLS)


def context_tool(name: str, client: str = "") -> bool:
    """Include companion calls only when the native name identifies their provider."""
    if task_tool(name):
        return True
    # Cursor names the tool resolved by its MCP configuration without a
    # provider prefix. The native dispatcher owns that name resolution.
    if client == "cursor" and re.fullmatch(r"MCP:(?:knowledge_(?:query|explain|capture)|remote_[a-z_]+)", name):
        return True
    return bool(re.fullmatch(
        r"(?:MCP:)?(?:mcp__)?(?:mindie[-_]knowledge__knowledge_(?:query|explain|capture)"
        r"|remote[-_]dev__remote_[a-z_]+)", name))


def task_call(client: str, payload: dict) -> tuple[str, object, str, object]:
    """Read native tool coordinates without opening a task or probing Git."""
    name = str(payload.get("tool_name") or payload.get("toolName") or "")
    arguments = payload.get("tool_input", payload.get("toolInput", {}))
    nested_key = ""
    selected = arguments
    if client == "grok" and isinstance(arguments, dict):
        nested_name = str(arguments.get("tool_name") or arguments.get("toolName") or "")
        if nested_name:
            name = nested_name
            nested_key = "tool_input" if "tool_input" in arguments else "toolInput"
            selected = arguments.get(nested_key, {})
    return name, arguments, nested_key, selected


def needs_task_context(client: str, payload: dict) -> bool:
    name, _, _, arguments = task_call(client, payload)
    return (client in {"claude", "codex", "grok", "cursor"} and context_tool(name, client)
            and isinstance(arguments, dict) and not arguments.get("context_file"))


def normalized_event(payload: dict) -> str:
    return re.sub(r"[^a-z]", "", str(payload.get("hook_event_name") or payload.get("hookEventName") or "").lower())


def attach_native(store: AgentSessions, client: str, native: str, cwd: str) -> tuple[AgentSessions, dict]:
    parent = os.environ.get("MINDIE_PARENT_CONTEXT", "")
    association = os.environ.get("MINDIE_ATTACH_CONTEXT", "")
    if parent or association:
        inherited = load_context(parent or association)
        store = AgentSessions(Path(inherited["state_dir"]))
    return store, store.attach(client, native, cwd, parent_context=parent, association=association)


def in_project_scope(cwd: Path, project: Path, *, sources: dict[str, str] | None = None) -> bool:
    """Include actual worktrees of this repository, never a similarly named clone."""
    cwd, project = cwd.resolve(), project.resolve()
    if cwd == project:
        return True
    if sources is not None:
        # The consumer supplies explicitly prepared roots. Never discover
        # arbitrary nested repositories or infer task association from them.
        try:
            actual = subprocess.run(
                ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", timeout=5, check=True,
            ).stdout.strip()
            if Path(client_path(actual)).resolve() in {Path(client_path(path)).resolve() for path in sources.values()}:
                return True
        except (OSError, ValueError, subprocess.SubprocessError):
            pass  # Preserve the original non-Git initialization/outside scope.
    inside = project in cwd.parents
    try:
        common = git_common_directory(project)
    except (OSError, ValueError, subprocess.SubprocessError):
        # Keep local/non-Git project initialization usable in its own scope.
        return inside
    try:
        current = cwd
        for _ in range(8):
            if git_common_directory(current).samefile(common):
                return True
            # Registered submodules belong to the workspace too. An unrelated
            # nested clone has no superproject and must not inherit its hooks.
            parent = subprocess.run(
                ["git", "-C", str(current), "rev-parse", "--show-superproject-working-tree"],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", timeout=5, check=True,
            ).stdout.strip()
            if not parent:
                return False
            parent = Path(client_path(parent)).resolve()
            if parent == current:
                return False
            current = parent
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
    return False


def bind_native_defaults(store: AgentSessions, context: dict, sources: dict[str, str] | None = None) -> dict:
    try:
        return store.bind_native_sources(context, sources=sources)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        # An unborn/non-Git or inaccessible directory still has a local task.
        # attach() clears old automatic sources when the native cwd changes.
        print(f"MindIE: source reference not yet bound: {type(exc).__name__}", file=sys.stderr)
        return store.context(context["attachment"]["id"]) if sources is not None else context


def handle(client: str, payload: dict, store: AgentSessions | None = None, *,
           sources: dict[str, str] | None = None) -> dict:
    if client == "claude" and (GROK_EVENT_FIELD in payload or CURSOR_VERSION_FIELD in payload):
        return {}
    # Grok also imports Cursor hooks by default.
    if client == "cursor" and GROK_EVENT_FIELD in payload and CURSOR_VERSION_FIELD not in payload:
        return {}
    normalized = normalized_event(payload)
    cursor_pretool = client == "cursor" and normalized == "pretooluse"
    if normalized == "pretooluse" and not needs_task_context(client, payload):
        # Ordinary tools need neither a registry read nor Git scope resolution.
        return {}
    native = str((payload.get("conversation_id") if client == "cursor" else "")
                 or payload.get("session_id") or payload.get("sessionId") or payload.get("conversation_id") or "")
    cwd = client_path(payload.get("cwd") or payload.get("workspaceRoot") or (payload.get("workspace_roots") or [str(Path.cwd())])[0])
    if not native:
        raise ValueError("hook has no native session identity; no task association was guessed")
    store = store or AgentSessions()

    if normalized == "sessionstart":
        if payload.get("source") == "compact":
            context = store.native_context(client, native)
            if context["attachment"]["cwd"] != str(Path(cwd).resolve()):
                context = store.attach(client, native, str(cwd))
        else:
            store, context = attach_native(store, client, native, str(cwd))
        context = bind_native_defaults(store, context, sources)
    elif normalized in {"subagentstart", "subagentstop"}:
        parent_native = str(payload.get("parent_conversation_id") or payload.get("parentSessionId") or native)
        parent_agent = str(payload.get("parent_agent_id") or "") if client == "kimi" else ""
        parent = store.native_context(client, parent_native, "" if parent_agent == "main" else parent_agent)
        child = str(payload.get("agent_id") or payload.get("subagent_id") or payload.get("subagentId") or "")
        if not child:
            raise ValueError("client omitted the child id; do not invent a native session from its display name")
        context = store.attach(client, native, str(cwd), parent_context=parent["context_file"], agent_id=child)
        if normalized == "subagentstop":
            store.detach(context)
            return {}
        context = bind_native_defaults(store, context, sources)
    else:
        agent_id = str(payload.get("agent_id") or "")
        if client == "kimi" and agent_id == "main":
            agent_id = ""
        try:
            context = store.native_context(client, native, agent_id)
        except ValueError:
            if not cursor_pretool or agent_id:
                raise
            store, context = attach_native(store, client, native, str(cwd))
            context = bind_native_defaults(store, context, sources)
        if normalized == "sessionend":
            store.detach(context)
            return {}
        refresh_cwd = normalized in {"userpromptsubmit", "beforesubmitprompt"} or (
            client == "claude" and normalized == "pretooluse" and bool(payload.get("cwd")))
        if refresh_cwd and context["attachment"]["cwd"] != str(Path(cwd).resolve()):
            # EnterWorktree can move Claude during one prompt. Its next tool
            # carries the new native cwd; do not wait for another user prompt.
            context = store.attach(client, native, str(cwd), agent_id=context["attachment"].get("agent_id") or "")
            context = bind_native_defaults(store, context, sources)

    context = store.bind_configured_user(context)
    if client in {"claude", "codex", "grok", "cursor"} and normalized in {"userpromptsubmit", "beforesubmitprompt"}:
        # Native tool injection retains context. Refresh cwd/user above, but do
        # not append the same instructions on every user turn.
        return {}
    if client == "kimi" and normalized == "userpromptsubmit" and payload.get("agent_id"):
        # The SessionSetup extension supplies this native agent id alongside
        # MCP call metadata. Keep cwd/user binding above, without appending a
        # redundant context instruction on every prompt. Kimi clients without
        # native per-call metadata still receive context through prompt text.
        return {}
    defaults = context["source_defaults"]
    paths = {name: source["path"] for name, source in defaults["sources"].items()}
    hint = ("MindIE task automatically attached to this native session. Context:\n" + context["context_file"] + "\n"
            f"Source defaults ({defaults['origin']}): {json.dumps(paths, ensure_ascii=False)}\n"
            "Client startup owns workspace and component preparation. "
            "Native hooks supply context_file to supported MindIE tools; otherwise use this context. "
            "For a child or authorized cross-tool handoff, pass this context explicitly.")
    if normalized == "pretooluse":
        # Grok exposes MCP calls through its native `use_tool` dispatcher. Its
        # PreToolUse payload therefore puts the qualified MCP name and the
        # actual arguments one level deeper than Claude/Codex. Rewrite the
        # dispatcher envelope so the context receipt reaches the MCP server;
        # never flatten or infer the nested call from cwd/history.
        name, arguments, nested_key, nested_arguments = task_call(client, payload)
        if context_tool(name, client) and client in {"claude", "codex", "grok", "cursor"}:
            if not isinstance(nested_arguments, dict) or nested_arguments.get("context_file"):
                return {}
            updated = {**nested_arguments, "context_file": context["context_file"]}
            if client == "cursor":
                # https://cursor.com/docs/hooks#pretooluse uses snake_case at
                # the root, not Claude's hookSpecificOutput envelope.
                return {"updated_input": updated}
            if nested_key:
                return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "updatedInput": {**arguments, nested_key: updated}}}
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "updatedInput": {**arguments, "context_file": context["context_file"]}}}
        return {}
    if client == "cursor" and normalized == "sessionstart":
        return {"env": {"MINDIE_CONTEXT_FILE": context["context_file"]}, "additional_context": hint}
    if client == "claude" and normalized == "sessionstart" and os.environ.get("CLAUDE_ENV_FILE"):
        with Path(client_path(os.environ["CLAUDE_ENV_FILE"])).open("a") as stream:
            stream.write("\nexport MINDIE_CONTEXT_FILE=" + shlex.quote(context["context_file"]) + "\n")
    if normalized in {"sessionstart", "subagentstart", "userpromptsubmit"}:
        canonical = {"sessionstart": "SessionStart", "subagentstart": "SubagentStart", "userpromptsubmit": "UserPromptSubmit"}[normalized]
        return {"hookSpecificOutput": {"hookEventName": canonical, "additionalContext": hint}}
    return {}


def main(*, sources: dict[str, str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", choices=sorted(CLIENTS), required=True)
    parser.add_argument("--project", help="Scope a global hook (notably Kimi) to this project")
    args = parser.parse_args()
    try:
        payload = json.load(sys.stdin)
        if normalized_event(payload) == "pretooluse" and not needs_task_context(args.client, payload):
            print("{}")
            return 0
        if args.project:
            cwd = Path(client_path(payload.get("cwd") or payload.get("workspaceRoot") or
                       (payload.get("workspace_roots") or [str(Path.cwd())])[0])).resolve()
            project = Path(client_path(args.project)).expanduser().resolve()
            if not in_project_scope(cwd, project, sources=sources):
                # Kimi appends stdout to the user prompt; a literal {} would
                # pollute every prompt outside this project. Stay silent.
                print("")
                return 0
        output = handle(args.client, payload, sources=sources)
        # Kimi's UserPromptSubmit contract appends returned text, rather than
        # relying on Claude's additionalContext extension.
        if args.client == "kimi" and payload.get("hook_event_name") == "UserPromptSubmit":
            print(output["hookSpecificOutput"]["additionalContext"] if output else "")
        else:
            print(json.dumps(output))
    except Exception as exc:
        print(f"MindIE local association unavailable: {exc}. Local tools remain usable.", file=sys.stderr)
        print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
