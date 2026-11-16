# mindie-coordinator 0.7.12

## Zero Pre-Call Session-Binding Gates & Receipt Honesty Hardening

- **Zero pre-call session-binding gates on deterministic tools (`P2.7`)**: `mindie_hosts`, `mindie_probe`, `mindie_container`, `mindie_lease`, and handle-addressed `task.*` operations execute directly without prior host hooks, `_meta`, or pre-created `context_file`; `AgentSessions.ensure_context()` supports `cwd=None` on-demand attachment and `task_server._resolve_mcp_caller()` aligns WorkBuddy (`wb`), Claude Code, Codex, Kimi, and standard MCP connection callers.
- **Transaction & receipt honesty fixes**: Moves `_publish(key)` outside the outer `BEGIN IMMEDIATE` transaction in `AgentSessions.attach(resume=False)`, restricts `submission_state="not_sent"` strictly to pre-dispatch `TaskClient` initialization failures, and preserves succeeded `mindie.lease` `acquire`/`release` grants with `receipt_warning` + `receipt_unconfirmed=True` on local lease-cache I/O errors without overwriting unreadable caches.
