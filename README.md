<h1 align="center">MindIE Agent Coordinator (<code>mindie-coordinator</code>)</h1>

<p align="center">
  <strong>面向单用户的远端昇腾容器生命周期、Git 源码快照物化与宿主机 NPU 排队分配控制面</strong>
</p>

<p align="center">
  <strong>中文</strong> · <a href="./README.en.md">English</a> · <a href="./docs/README.md">文档中心</a> · <a href="https://github.com/mindie-agent/mindie-agent">MindIE Agent 主仓库</a>
</p>

---

## 项目简介

`mindie-coordinator` 是 [MindIE Agent](https://github.com/mindie-agent/mindie-agent) 的本地控制面组件（Python 包名 `mindie-coordinator`，导入名 `mindie_coordinator`，CLI 入口 `mindie-coordinator` 与 `mindie`）。它在开发者本机作为单用户进程运行，底层通过 [`remote-dev`](https://github.com/mindie-agent/remote-dev) 操控用户在远端昇腾服务器上的持久化容器与宿主机 NPU 分配队列。

- **Git 驱动的源码身份与隔离物化**：准入时固定本地 Git 工作树与子模块快照（不移动本地 `HEAD` 或暂存区），并在远端持久化用户容器（`mindie-<user>`）内生成独立的执行视图。
- **宿主机 NPU 与端口唯一权威**：通过推送到物理宿主机的零依赖 `mindie_npu_coordination.py` 模块统一仲裁 NPU 设备租约与服务端口（`30000–45999`），支持独占分配与显式单卡共享（`allow_external_busy=True`）。
- **受管执行全生命周期**：覆盖环境准备（Preparation）、预检（Preflight）、受管启动（`MINDIE_EXECUTION_OBSERVATION` 证明）、非阻塞状态观测、进程族排空停止与确定性资源释放。
- **统一 MCP、CLI 与 Python SDK**：通过 `mindie-coordinator task-server` 提供 5 个标准 MCP 工具（`mindie_session`、`mindie_run`、`mindie_execution`、`mindie_finish`、`mindie_message`），并与 `mindie` CLI 及 Python `TaskClient` 共享同一本地守护进程。

---

## 快速开始

### 1. 安装与启动

需要 Python `>= 3.11`。`pyproject.toml` 已锁定经审查的 `remote-dev` 与 `mindie-diagnostics` 版本，安装时自动一并解析：

```bash
# 安装到当前环境
uv pip install git+https://github.com/mindie-agent/coordinator@main

# 或免安装直接启动 MCP Task Server
uvx --from git+https://github.com/mindie-agent/coordinator@main mindie-coordinator task-server
```

### 2. 客户端 MCP 配置示例

在 `.mcp.json`（Claude Code / Cursor）中注册：

```json
{
  "mcpServers": {
    "mindie-coordinator": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/mindie-agent/coordinator@main",
        "mindie-coordinator",
        "task-server"
      ]
    }
  }
}
```

### 3. Python `TaskClient` 使用示例

```python
from mindie_coordinator.task_client import TaskClient

client = TaskClient("/local/task-context.json")
run = client.run(
    '"$MINDIE_PYTHON" -c "import torch_npu; print(1)"',
    sources={"vllm": "/local/vllm", "vllm-ascend": "/local/vllm-ascend"},
    resources={"devices": [0]},
    topology={"host": "npu-host"},
    wait_until="released",
    wait_timeout_seconds=180,
)
print(run["execution_id"], run["state"], run["resources_released"], run.get("stdout", ""))
```

### 4. CLI 提交与观测示例

```bash
# 直接提交本地 UTF-8 Shell 脚本并等待释放
mindie run --script-file business.sh --source app=/local/app --wait released --wait-timeout-seconds 180

# 观测已有执行的状态或等待其释放
mindie execution --execution-id EXECUTION_ID --wait released --wait-timeout-seconds 180

# 查看执行记录的构建、源码或诊断证据
mindie execution --execution-id EXECUTION_ID --action evidence --section build --path kernel_name
```

---

## MCP 工具一览

| MCP 工具 | CLI / Python 对应入口 | 职责说明 |
| --- | --- | --- |
| `mindie_session` | `TaskClient.session()` | 查询当前本地任务身份、已绑定的默认源码工作树与来源证明 |
| `mindie_run` | `mindie run` / `TaskClient.run()` | 固定 Git 源码快照，提交业务命令或脚本，申请 NPU 租约与环境准备并启动受管执行 |
| `mindie_execution` | `mindie execution` / `TaskClient.observe()` | 查询非阻塞缓存或实时执行状态、增量日志（`tail`）、构建/诊断证据（`evidence`）或停止执行（`stop`） |
| `mindie_finish` | `TaskClient.finish()` | 关闭当前任务准入，终止所属受管执行并归还全部运行时绑定与设备租约 |
| `mindie_message` | `TaskClient.message()` | 基于等待期间返回的 `coordination_peers[].reference` 或 `reply_reference` 在宿主机协作者之间收发离线協調消息 |

---

## 文档导航

| 文档 | 中文 | English |
| --- | --- | --- |
| 文档中心索引 | [docs/README.md](./docs/README.md) | [docs/README.en.md](./docs/README.en.md) |
| 持久化用户容器与任务生命周期契约 | [docs/session-lifecycle.zh.md](./docs/session-lifecycle.zh.md) | [docs/session-lifecycle.md](./docs/session-lifecycle.md) |
| 工具接口、运行时准备与可观测性参考 | [docs/configuration-and-tools.md](./docs/configuration-and-tools.md) | [docs/configuration-and-tools.en.md](./docs/configuration-and-tools.en.md) |

---

## 仓库目录结构

| 路径 | 职责说明 |
| --- | --- |
| `mindie_coordinator/cli.py` | `mindie-coordinator` / `python -m mindie_coordinator` 命令行入口 |
| `mindie_coordinator/mindie.py` | 面向任务的轻量 `mindie` CLI 入口 |
| `mindie_coordinator/task_server.py` | 暴露 5 个任务与协调工具的 stdio MCP Server |
| `mindie_coordinator/task_client.py` | Python `TaskClient` SDK |
| `mindie_coordinator/host_queue.py` | 宿主机 NPU 排队分配公共 Python API |
| `mindie_coordinator/host/` | 零依赖宿主机 NPU 与端口协调模块（通过 SSH 下发至物理机执行） |
| `mindie_coordinator/prepare_runtime.py` | 容器内环境证明、原生扩展复用与发布 |
| `mindie_coordinator/parity.py` | 本地 Git 工作树快照捕获与远端物化 |
| `mindie_coordinator/run_manifest.py` | Run Manifest v1 与代码身份记录 |
| `docs/` | 中英双语生命周期契约与工具配置参考 |
| `tests/` | 单元测试套件 |

---

## 开发与测试

```bash
uv venv
uv pip install -e ".[test]"
.venv/bin/python -m pytest
```

---

## 许可证

本项目基于 [MIT License](./LICENSE) 开源。
