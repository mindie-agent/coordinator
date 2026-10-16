# 工具参数、单卡共享、算子复用与诊断参考

**中文** · [English](./configuration-and-tools.en.md) · [返回文档中心](./README.md)

这篇指南用最直白的方式说明在使用 `mindie-coordinator` 时最常碰到的 6 个实用问题：**怎么传命令和等结果、怎么跟别人共用单张 NPU 卡、怎么跳过重复的算子编译、怎么给排队的同事发协调留言、怎么管理本机守护进程，以及出错后怎么查诊断包。**

---

## 1. 提交任务、等待完成与查看执行证据

- **传命令还是传脚本文件？二选一即可**：
  调用 `mindie_run` 或 `TaskClient.run` 时，既可以直接传一句 `command`，也可以传本地的 Shell 脚本文件路径 `script_file`（最大支持 1 MiB，会自动去掉 UTF-8 BOM 头并把 Windows 的 CRLF 换行转成 Linux 的 LF，防止脚本在 Linux 容器里报 `^M` 错误）。
- **怎么等待任务跑到指定阶段（`wait_until`）？**
  - 可以传 `wait_until="running"`（等它跑起来就返回）或 `wait_until="released"`（等它跑完并且把 NPU 卡还了再返回），配合 `wait_timeout_seconds`（`0–600` 秒）设置最长等待时间。
  - **注意区分“卡已释放”和“任务跑成功”**：`released` 只代表进程结束了、NPU 卡归还了；要判断脚本本身有没有报错退出，请同时看 `state == "succeeded"`。
  - **等待超时不会杀任务**：如果等了 180 秒还没跑完（返回 `wait_timed_out=True`），你的任务依然在远端正常跑，绝不会被中断或重复提交。你可以拿着同一个 `execution_id` 继续等待或查日志。
- **秒回缓存 vs. 实时去远端查状态**：
  - MCP 工具 `mindie_execution` 和 CLI `mindie execution` 默认直接读取本地最近一次保存的状态快照（附带 `observation_freshness` 告诉你快照是几秒前采的），保证零卡顿秒回；如果快照超过 2 秒，后台会自动异步去刷新。
  - 如果你想立刻去远端拉一次最新状态，加上 `--refresh`（或 `refresh: true`）即可。Python `TaskClient.observe()` 默认就是实时刷新（想读缓存可传 `refresh=False`）。
- **查看执行留存证据（`action="evidence"`）**：
  随时按 `section="sources"`（当时用的哪版代码）、`"preparation"`（环境准备各阶段耗时）、`"build"`（编译产物回执）或 `"diagnostics"`（脱敏诊断包）查看完整记录。
- **借卡前先跑预检（`preflight`）**：
  支持传入 `preflight` 命令，在真正去排队借 NPU 卡之前，先用选好的 Python 解释器跑一遍无卡预检（比如检查语法或配置是否正确）。如果预检报错，直接中止并返回错误输出，不会白白排队占卡。

---

## 2. 怎么申请 NPU 卡？怎么跟别人共享同一张卡？

- **不写卡数时默认 0 卡（不占 NPU）**：
  `resources` 默认 `npu_count=0`，所以跑普通编译或纯 CPU 测试不会占用任何 NPU。需要用卡时，传正整数 `npu_count`，或者配合 `topology={"host": "..."}` 指定具体的物理卡号列表 `devices`。
- **跟卡上已有的外部进程共享单张卡（`allow_external_busy=True`）**：
  ```python
  resources={"devices": [0], "allow_external_busy": True}
  ```
  当你明确指定**单张物理卡**（比如 `devices: [0]`）并加上 `allow_external_busy=True` 时，即使这张卡上已经有别人没走 Coordinator 启动的外部进程或显存占用，依然允许你启动任务（但跟其他 Coordinator 受管任务之间仍然互斥排队）。等你跑完或停止时，**只会清理你自己的进程树和租约，绝不会去杀卡上原有的外部进程**。
- **自动挑空闲服务端口**：
  传 `service_port=0` 时，宿主机协调器会自动在 `30000–45999`（或你声明的端口池）里挑一个既没人监听、也没被其他任务租走的空闲端口分给你，任务结束自动收回。

---

## 3. 代码增量同步与编译好的算子自动复用

- **改几行代码不用全量重传**：
  远端已经有 Git 镜像仓库时，本地刚改的几行代码只会打包成很小的增量 Git Pack 通过现有 SSH 连接秒传过去，自动校验父提交和文件树哈希。
- **改了 Python 不用重编 C++ 算子，也不用重跑漫长的导入自检**：
  第一次编译好原生环境时，系统会跑一次完整的框架导入冒烟测试并记下编译产物哈希。之后如果你只改了 Python 业务代码，只要底层依赖、环境变量和原生算子哈希没变，系统会直接复用上次的导入证明（回执里标明 `python_import_executed=false`），秒级进入业务执行！
- **直接复用现成编译好的目录**：
  如果容器里已经有编译好的 `vllm` / `vllm-ascend` 目录，可以用以下命令把它注册为只读复用源：
  ```bash
  mindie-coordinator runtime-register --reuse-only --source vllm=PATH --source vllm-ascend=PATH
  ```

---

## 4. 排队等卡时，给正在用卡的同事发消息（`mindie_message`）

当你提交任务发现卡被占了、返回结果里带有 `coordination_peers[].reference` 时，你可以直接调 `mindie_message` 给对方发一条留言（或者用收到的 `notifications[].reply_reference` 回复对方）：

- 留言存放在宿主机队列旁，本地守护进程在日常调用 `run` / `status` 时每 5 秒顺带异步拉取一次，不会拖慢正常操作。
- **纯沟通，零越权**：发消息只用于协作者之间打招呼商量用卡，绝不会执行任何命令，也不会强行抢走或停掉别人的任务。

---

## 5. 宿主机队列 Python API 与本机守护进程管理

### 在 Python 中直接调用宿主机队列 API

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

宿主机上的队列状态默认保存在 `/tmp/mindie-npu-coordinator/v1/`（可通过 `MINDIE_NPU_COORDINATOR_STATE_DIR` 修改）。

### 查看或平滑重启本机守护进程

```bash
# 查看当前后台跑的守护进程版本和磁盘上已安装的版本是否一致
mindie-coordinator daemon --action status

# 如果当前没有任何在跑的任务或占着的租约，平滑重启守护进程到最新版本
mindie-coordinator daemon --action restart-if-idle
```

当你升级了 `coordinator` 或 `remote-dev` 包后：如果后台守护进程正好空闲，下次提交任务时它会自动重启切换到新版本；如果后台还有任务在跑，它会保护在途任务不被中断，并返回 `needs_runtime_update` 提醒你等任务跑完再切。

---

## 6. 故障诊断与隐私保护

- 每次工具调用都会返回 `diagnostics`（含 `operation_id`、`trace_id`、各阶段耗时与本地记录引用）和 `error_details`（明确区分命令是“没发出去 `not_sent`”、“已确认 `acknowledged`”还是“断网不确定 `uncertain`”）。
- 默认日志级别为 `MINDIE_LOG_LEVEL=INFO`，日志目录可通过 `MINDIE_DIAGNOSTICS_ROOT` 配置。你的业务命令原文、源码内容、环境变量值和密钥**绝不会写进诊断日志**。
- 需要导出某次执行的脱敏诊断包时，直接运行：
  ```bash
  python -m mindie_coordinator.mindie execution --execution-id <ID> --action evidence --section diagnostics
  ```
