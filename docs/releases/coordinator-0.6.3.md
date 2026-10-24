# mindie-coordinator 0.6.3：远端准备作业状态刷新、HBM 遥测一致性与实测容量准入

[English](https://github.com/mindie-agent/coordinator/blob/v0.6.3/docs/releases/coordinator-0.6.3.en.md)

`mindie-coordinator 0.6.3` 修复了容器准备阶段远端作业的状态刷新与编译租约收尾，统一了昇腾 A5 实测已用 HBM 保留与单/多 Die 卡补充遥测的一致性，并在放置与容器准备全流程按主机实测可用容量（CPU、内存与总空闲 HBM）执行准入核对。

---

## 一、核心修复与行为变更

### 1. 远端准备作业状态刷新与编译租约收尾（`service_gang.py`、`service_evidence.py`、`service.py`，PR `#63`）

- 修复了 `service_gang.py` 拆分后遗漏 `preparation_control` 导入、导致刷新远端准备作业状态时因内部 `NameError` 被误记为 `state="uncertain"` 的问题。
- 在状态观测与终态推进中接入准备作业刷新，确保容器准备与源码编译任务及时更新远端真实状态并释放编译租约。

### 2. 实测已用 HBM 保留与卡/Die 补充遥测一致性（`fleet_probe_parser.py`，PR `#64`）

- **保留 A5 实测已用 HBM 读数**：昇腾 950DT（A5）的设备繁忙判定继续由进程表决定（无业务进程时保持空闲可租约），同时如实保留 `npu-smi info` 测得的总已用显存 `hbm_used_mb`（不再将其归零或扣除假设的固定基线），供后续可用显存余量计算。
- **卡/Die 补充遥测一致性**：`parse_npu_smi_usages` 仅在单 Die 卡或不含 `dies` 拆分的整卡记录缺少 HBM 读数时，才回填 `npu-smi info -t usages` 的整卡统计并同步更新单 Die 记录；双 Die A3 卡不再被整卡级统计覆盖或拆平，卡片与物理 Die 的空闲/繁忙汇总统一由 `device_occupancy_inventory` 派生。

### 3. 基于实测主机容量的放置与准备准入（`service_evidence.py`、`service_gang.py`、`service.py`、`placement.py`，PR `#64`）

- **按决策单次采样主机容量快照**：当角色显式声明了 `cpu_cores`、`memory_mb` 或 `hbm_mb` 下限时，`ServiceEvidenceMixin._observe_host_specs` 通过 `_probe_host_record`（`probe_machine_record`）对候选主机（默认上限 `8` 台）采样并归一化为 `host_specs` 快照。
- **全流程容量筛选**：在 `CoordinatorService` 中，同一 `host_specs` 快照依次用于：
  1. **Donor 容器挑选**：先通过 `can_prepare` 核对环境兼容性，再用 `hardware_shortfall` 核对主机容量；
  2. **用户容器开通**：在开通容器前跳过不满足 `hardware_shortfall` 的主机；
  3. **准备完成后复核**与**最终角色绑定复核**：通过 `runtime_matches(..., host_specs)` 确认目标环境与主机容量满足要求（独立辅助函数 `select_runtimes(..., host_specs=...)` 亦支持传入 `host_specs`）。
- **主机级总余量核对与缺口说明**：`hbm_mb` 与 `cpu_cores`、`memory_mb` 均作为主机级实测可用余量下限（对比主机空闲 Die 的汇总 `vram_free_mb`）进行可行性筛查，最终逐 Die 独占分配仍由主机队列完成；当主机未完成探测或可用容量不足时，`shortfall_reason` 返回按主机去重的缺口原因。

---

## 二、测试验证

- 本地单元与契约测试完整覆盖准备作业状态刷新、A5 实测已用 HBM 保留、单/双 Die `-t usages` 遥测一致性、主机容量准入与 `shortfall_reason` 缺口报告。
- 在常驻 A2（`173.125.1.2`）与 A3（`192.168.13.165`、`192.168.13.154`）主机上完成只读 Fleet 容量探测实测（`test_live_fleet_probe_and_capacity_across_a2_and_a3`，**`1 passed`，`8.01s`**）。
