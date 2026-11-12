# Coordinator 0.7.7

Native MCP call metadata no longer makes independent host, probe, container or
bare-metal lease operations require a managed task attachment. Explicit task
contexts and managed execution tools retain native ownership checks.

The public catalog distinguishes environment operations from task lifecycle
operations. Missing attachment must not block resource discovery or authorized
CPU-only and offline work; actual device work still requires verified ownership.

Regression coverage includes unbound Codex/Kimi callers, child callers, explicit
context conflicts and real stdio sessions that remain responsive after failure.
