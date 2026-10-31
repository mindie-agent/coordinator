# mindie-coordinator 0.7.1

[中文](coordinator-0.7.1.md)

## Native Antigravity Session Identity & Five-Harness Client Support

- Added `agy` / `antigravity` to `CLIENTS` and wired `AGY_CONVERSATION_ID` discovery in `mindie_coordinator/agent_session.py:load_context()`, enabling Google Antigravity sessions to share the same deterministic task and NPU lease isolation contract alongside Codex, Claude Code, Kimi Code, and DSH.
- Aligned unified MindIE Agent brand assets and bilingual documentation.
