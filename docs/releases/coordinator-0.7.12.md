# mindie-coordinator 0.7.12

## 确定性工具零前置会话门禁与回执诚实性加固

- **确定性工具零前置会话绑定门禁（`P2.7`）**：`mindie_hosts`、`mindie_probe`、`mindie_container`、`mindie_lease` 及句柄寻址的 `task.*` 调用在无宿主 Hook、无 `_meta`、无预置 `context_file` 时均可直接执行；`AgentSessions.ensure_context()` 支持 `cwd=None` 按需挂载，`task_server._resolve_mcp_caller()` 对齐 WorkBuddy（`wb`）、Claude Code、Codex、Kimi 与标准 MCP 连接标识。
- **事务与回执诚实性修复**：修复 `AgentSessions.attach(resume=False)` 外层 `BEGIN IMMEDIATE` 嵌套发布死锁；将 `submission_state="not_sent"` 严格收敛至 `TaskClient` 派发前初始化失败；`mindie.lease` 在远端 `acquire`/`release` 成功后若本地租约缓存读写失败，保留成功结果并附带 `receipt_warning` + `receipt_unconfirmed=True`，且绝不用空字典覆盖不可读的存量租约缓存。
