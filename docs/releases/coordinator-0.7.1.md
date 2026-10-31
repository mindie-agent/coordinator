# mindie-coordinator 0.7.1

[English](coordinator-0.7.1.en.md)

## Antigravity 原生会话标识与五端客户端统一支持

- 在 `mindie_coordinator/agent_session.py` 的 `CLIENTS` 与 `load_context()` 中新增 `agy` / `antigravity` 客户端标识与 `AGY_CONVERSATION_ID` 原生会话环境变量解析，支持 Google Antigravity 宿主与 Codex、Claude Code、Kimi Code、DSH 共享同一套任务与 NPU 租约隔离契约。
- 同步更新统一品牌标识资产与中英文文档。
