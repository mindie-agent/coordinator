# 工具接口、运行时准备与可观测性参考

**中文** · [English](./configuration-and-tools.en.md) · [返回文档中心](./README.md)

本文档涵盖 `mindie-coordinator` 的任务提交与观测参数、单卡共享策略、原生构建证明与复用、跨任务协调消息、宿主机队列 Python API、守护进程版本一致性以及结构化诊断规范。

---

## 1. 任务提交、等待与证据观测

- **命令与脚本二选一**：`mindie_run` / `TaskClient.run` 支持传入 `command` 或本地 UTF-8 脚本文件 `script_file`（上限 1 MiB，自动移除 BOM 并将 CRLF 转为 LF，同时记录原始字节摘要与提交命令摘要）。
- **有界等待（`wait_until`）**：
  - 提交时可指定 `wait_until="running"` 或 `"released"` 及 `wait_timeout_seconds`（`0–600` 秒）。
  - `released` 代表进程已终止且资源已释放；业务执行成功需同时检查 `state == "succeeded"`。
  - 若等待超时（`wait_timed_out=True`），不会终止或重提任务，可继续对同一 `execution_id` 发起等待或观测。
- **非阻塞缓存与主动刷新**：
  - MCP / CLI 状态查询默认立即返回最近一次持久化的受管作业快照（附带 `observation_freshness` 说明快照年龄、来源及是否因繁忙而推迟刷新）；快照超过 2 秒会自动调度后台推进。
  - 传入 `--refresh`（或 `refresh: true`）请求实时采样；Python `TaskClient.observe()` 默认实时采样，传入 `refresh=False` 可读取非阻塞缓存。
- **执行证据查询（`action="evidence"`）**：
  - 支持按 `section="sources"`、`"preparation"`、`"build"` 或 `"diagnostics"` 查看已记录的源码快照、准备阶段耗时（`preparation_timings`）、编译回执与脱敏支持包（`support_bundle`）。
- **预检命令（`preflight`）**：支持在分配任何 NPU 租约前使用所选解释器运行无设备预检脚本（占位端口为 `0`）；预检失败将保留原始 `stderr` 引用并中止业务启动。

---

## 2. 物理 NPU 分配与显式单卡共享

- **默认零卡策略**：资源参数默认 `npu_count=0`，纯 CPU 或编译任务不占用 NPU；需要 NPU 时显式声明正整数 `npu_count` 或配合 `topology={"host": "..."}` 指定具体物理卡号 `devices`。
- **显式单卡外部共享（`allow_external_busy=True`）**：
  ```python
  resources={"devices": [0], "allow_external_busy": True}
  ```
  仅在明确指定单张物理设备时生效，允许该卡上已存在外部非受管进程或 HBM 占用，但与其他 Coordinator 受管租约依然互斥。停止或完成时仅终止本执行的受管进程族并释放自身租约，绝不干扰卡上的外部工作负载。
- **服务端口自动分配**：传入 `service_port=0` 由宿主机协调器在声明范围或默认范围（`30000–45999`）内分配空闲端口，并在执行释放时同步归还。

---

## 3. 原生构建证明与增量物化

- **增量 Git 快照物化**：复用远端镜像已有的不可变 Git 对象；小幅工作树修改通过有界 Git Pack 直接在既有 RPC 连接上传输并校验父提交与结果树哈希。
- **原生扩展证明复用**：首次构建的原生环境会执行完整框架导入冒烟测试并记录哈希；后续仅修改业务 Python 源码时，只要依赖、加载器环境与原生构建产物哈希完全匹配，即可直接复用既有导入证明（回执显式标记 `python_import_executed=false`），跳过重复的编译与全量探针。
- **外部预构建运行时注册**：可通过 `mindie-coordinator runtime-register --reuse-only --source vllm=PATH --source vllm-ascend=PATH`（或 `CoordinatorClient.runtime_register`）将已有原生构建目录注册为只读复用源。

---

## 4. 跨任务离线协调消息（`mindie_message`）

当任务在等待队列中看到 `coordination_peers[].reference` 时，可通过 `mindie_message` 向对方发送协调消息，或使用收到的 `notifications[].reply_reference` 回复：

- 消息存储在宿主机队列数据库旁，由本地 Coordinator 在常规 `run` / `status` 调用时按最短 5 秒间隔异步拉取，不阻塞远端 I/O。
- 消息仅用于协作者沟通，绝不会执行命令、释放租约或停止其他用户的任务。

---

## 5. 宿主机 NPU 队列 Python API 与守护进程管理

### 宿主机队列公共导入入口

```python
from mindie_coordinator.host_queue import (
    HostQueue,
    HostQueueUnavailable,
    SCHEMA_VERSION,
    CoordinationError,
    handle_request,
    host_queue_module_path,
    load_host_protocol,
)
```

宿主机侧持久化状态默认位于 `/tmp/mindie-npu-coordinator/v1/`（可通过 `MINDIE_NPU_COORDINATOR_STATE_DIR` 覆盖）。

### 本地守护进程状态与平滑重启

```bash
# 查看已加载与已安装的包版本/Commit 身份（不拉起新守护进程）
mindie-coordinator daemon --action status

# 当无活动执行或未释放租约时平滑重启至当前已安装版本
mindie-coordinator daemon --action restart-if-idle
```

受管任务准入要求守护进程与调用方的 `coordinator` 及 `remote-dev` Commit 与 Python 解释器一致；若空闲守护进程版本不一致会自动使用调用方解释器重启，若守护进程正忙则返回 `needs_runtime_update`。

---

## 6. 结构化诊断（Diagnostics）

- 每次工具响应均包含 `diagnostics`（`operation_id`、`trace_id`、UTC 起止时间、单调时钟耗时、有界阶段摘要与本地 `record_ref`）以及精确区分提交确定性（`not_sent` / `not_executed`、`uncertain`、`acknowledged`）的 `error_details`。
- 默认日志级别为 `MINDIE_LOG_LEVEL=INFO`，可通过 `MINDIE_DIAGNOSTICS_ROOT` 指定私有日志根目录。命令原文、文件内容、环境变量值与凭据绝不写入诊断日志参数。
- 对所属执行运行 `python -m mindie_coordinator.mindie execution --execution-id ID --action evidence --section diagnostics` 可获取脱敏的 `support_bundle`。
