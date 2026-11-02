# mindie-coordinator 0.7.2

[English](coordinator-0.7.2.en.md)

## 腾讯 WorkBuddy 原生会话标识与多端客户端统一支持

- 在 `mindie_coordinator/agent_session.py` 的 `CLIENTS`、`_CLIENT_ALIASES` 与 `load_context()` 中新增 `wb` / `workbuddy` 客户端标识（统一归并为 `"wb"`），并支持 `WORKBUDDY_SESSION_ID`、`WORKBUDDY_CONVERSATION_ID` 与 `CODEBUDDY_SESSION_ID` 原生会话环境变量解析。
- 在 `mindie_coordinator/hooks/mindie_session.py` 中支持 WorkBuddy 的 `SessionStart`（向 `CODEBUDDY_ENV_FILE` / `CLAUDE_ENV_FILE` 导出 `MINDIE_CONTEXT_FILE`）、`UserPromptSubmit` 与 `PreToolUse` 生命周期 Hook。
