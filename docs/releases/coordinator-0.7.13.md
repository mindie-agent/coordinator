# mindie-coordinator 0.7.13：反转 `AD-04/AD-05` 反模式、恢复确定性工具零绑定直通与控制面治理收敛

[English](coordinator-0.7.13.en.md)

`mindie-coordinator v0.7.13` 完成控制面反模式清理与洁净架构收敛，并升级锁定 `mindie-diagnostics v0.5.16` 与 `remote-dev v0.11.13`：

- **彻底移除确定性工具前置会话绑定门禁（反转 `AD-04/AD-05`，贯彻 `P2.7`）**：移除历史遗留的强制会话上下文文件与前置绑定锁检查——标准 MCP `tools/call` 在零 `_meta`、零前置 Hook、零 `request_nonce`/`context_file` 下即可直接调用 `mindie_hosts`、`mindie_probe`、`mindie_container`、`mindie_lease` 与句柄寻址 `task.*`，物理卡显存与端口互斥专由 `HostQueue` 租约（`lease_id` + `fence_token`）保障。
- **消除本地单向自锁死状态与落实回执诚实性**：清理本地不可恢复故障锁态，确保远端主操作成功后若本地账本写入告警，保持 `isError=False` 并附加 `receipt_warning` + `receipt_unconfirmed=True`；全部核心模块收敛至 `< 800` 行并在 `OSError` 边界前显式重抛 `TimeoutError`。
- **底座正式版升级与 `control-plane` 门禁对齐**：升级 `mindie-diagnostics` 至 `v0.5.16`、`remote-dev` 至 `v0.11.13`，并启用 `--profile control-plane` PR 设计原则门禁。
