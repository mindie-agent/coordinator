# 贡献指南 (`mindie-coordinator` — `control-plane`)

**中文** · [English](./CONTRIBUTING.en.md)

`mindie-coordinator` 属于 MindIE Agent **远端与集群控制面（`control-plane`）** 仓库类别，负责多机 Ascend NPU/端口互斥租约（`HostQueue`）、A3 双 Die 拓扑放置、Git 提交级代码同构同步与受管任务编排。

## 1. 核心架构红线

1. **物理资源唯一权威（`CTRL-LEASE-OR-AUTH`）**：
   - 远端宿主机的 `HostQueue`（`lease_id` + `fence_token`）是分配物理 NPU 卡（含 A3 双 Die `Phy-ID 0..15`）与服务端口的唯一权威；严禁在本地私自分卡或跳过 `fence_token`。
   - 保持 `sessions.sqlite3`、本地 `coordinator.sqlite3` 与远端宿主机队列库三者物理隔离；保持 `machines.json`（含 `.lock`）与 `endpoints.json` 原子同步，复用 `remote-dev` 共享 SSH 凭据。
   - 严禁在远端准备脚本中注入 `PIP_CONFIG_FILE=/dev/null` 或 `unset PIP_EXTRA_INDEX_URL` 破坏内网镜像源，严禁用 `sort -V` 盲猜 Python 版本（统一使用 `mindie_coordinator/workers/python_selector.py`）。
2. **零前置绑定与跨连接显式句柄（`CTRL-HANDLE`）**：
   - 所有 9 个 MCP 工具必须支持零 `_meta`、零前置 Hook、免传 `context_file` 直接调用（`P2.7`）。
   - 断线重连或子代理接管时，凭显式 `execution_id` / `run_id` 必须直接定位并恢复查询、日志拉取与停止操作；静态算子预检（`op_preflight`）仅输出非阻塞 `preflight_warnings`，严禁拦截任务提交。
3. **纯标准库远端脚本与有界超时（`CTRL-STREAM-IO`）**：
   - 发往远端宿主机或容器执行的独立脚本统一维护在 `mindie_coordinator/workers/`，保持 100% Python 标准库（`python3 -I` 可直跑，Windows SSH 启动命令 `< 8191` 字符）。
   - 严格遵守 `outer_timeout >= sum(inner_budgets) + cleanup_grace`，核心模块保持 `< 800` 行并在前 30 行声明 `[Role]` / `[Invariants]` / `[Timeouts & Locks]` / `[Side-Effect Policy]`。
4. **外部副作用与回执诚实性（`CTRL-RECEIPT`）**：
   - `except OSError:` 前必须显式 `except TimeoutError: raise`。
   - 仅在 `submit` 真正发出后（`submit_dispatched=True`）才允许在断连时进入 `uncertain`；发包前失败保持 `pending` 并在重试时自动刷新 `epoch`，严禁引入 `needs_repair` / `TaskRootBusy` 等单向本地自锁状态。
   - 远端 `provision` / `remove` / `lease` 主操作一旦成功，本地 `machines.json` / `endpoints.json` / `leases.json` 回执写入失败必须保持 `outcome="success"`（`isError=False`），并附带 `receipt_warning` 与 `receipt_unconfirmed=True`。

## 2. 本地隔离测试与打包冒烟

所有测试必须隔离宿主机真实目录（`HOME`、`USERPROFILE`、`XDG_*`、`MINDIE_CONFIG_HOME`、`MINDIE_DATA_HOME`、`MINDIE_DIAGNOSTICS_ROOT`、`REMOTE_DEV_AUTH_DIR`、`REMOTE_DEV_STATE_DIR`、`UV_CACHE_DIR`、`TMPDIR`）：

```bash
# 1. 运行本地 Tier 0 / Tier 1 测试套件
uv venv && uv pip install -e ".[test]"
.venv/bin/python -m pytest -q -m "not live_npu"

# 2. 源码目录外真实 Wheel 安装冒烟验证（防止漏打包子包/workers）
uv build
smoke_dir="$(mktemp -d)"
(cd "$smoke_dir" && uv venv .venv && VIRTUAL_ENV="$PWD/.venv" uv pip install /path/to/dist/*.whl && .venv/bin/mindie-coordinator --help)
rm -rf "$smoke_dir"

# 3. 常驻 A2 (173.125.1.2) / A3 (192.168.13.165) 非破坏性实机验证（严禁连接 .153）
.venv/bin/python -m pytest -q -m live_npu
```

## 3. 提交 PR 前自检（`control-plane` Profile）

1. 确保全部文本文件使用 Unix `LF`（`\n`）换行：
   ```bash
   python3 .github/scripts/check_lf_eol.py
   ```
2. 按 [`.github/pull_request_template.md`](.github/pull_request_template.md) 勾选全部 4 项稳定核查 ID（`CTRL-LEASE-OR-AUTH`、`CTRL-HANDLE`、`CTRL-STREAM-IO`、`CTRL-RECEIPT`），逐项填写实测证据（纯文档改动可写 `not-applicable: docs-only`），并运行本地校验：
   ```bash
   python3 .github/scripts/check_pr_checklist.py --profile control-plane --body-file <pr_body.md>
   ```
