# mindie-coordinator 0.7.2

[中文](coordinator-0.7.2.md)

## Native Tencent WorkBuddy Session Identity & Hook Support

- Add `wb` and `workbuddy` (canonicalized to `"wb"`) to `CLIENTS`, `_CLIENT_ALIASES`, and `load_context()` in `mindie_coordinator/agent_session.py`, resolving `WORKBUDDY_SESSION_ID`, `WORKBUDDY_CONVERSATION_ID`, and `CODEBUDDY_SESSION_ID`.
- Support WorkBuddy `SessionStart` (exporting `MINDIE_CONTEXT_FILE` to `CODEBUDDY_ENV_FILE` / `CLAUDE_ENV_FILE`), `UserPromptSubmit`, and `PreToolUse` hooks in `mindie_coordinator/hooks/mindie_session.py`.
