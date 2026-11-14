# coordinator 0.7.9

[English](coordinator-0.7.9.en.md)

- **Diagnostics 2.0 内部故障与机器目录契约违规观测（`diagnostics.py`、`cli.py`、`ops.py`、`machine_directory.py`）**：升级 `mindie-diagnostics` 至正式版 `v0.5.11`（`171623584aa812677df907518dcb55b06493241b`）与 `remote-dev` 至正式版 `v0.11.10`（`3556df571450dab481a25e0157167f1c3727dc6c`），为内部未预期异常与机器目录权威状态契约违规提供源码根目录归因及单次去重上报，严格排除调用方上下文与 SSH/远端环境错误。
