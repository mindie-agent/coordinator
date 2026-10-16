# 持久化用户容器中的任务生命周期契约

**中文** · [English](./session-lifecycle.md) · [返回文档中心](./README.md)

持久化 Coordinator 守护进程统一管理固定的执行输入、隔离的执行根目录、多角色资源预留与经确认的资源清理。任务结束时保留用户容器及其已准备好的运行环境。每位用户在每台宿主机上拥有一个固定名称的持久化容器 `mindie-<user>`（例如 `mindie-alice`），业务代码修改、编译、服务与测试均在该容器内运行；宿主机侧操作仅限于设备/端口仲裁与容器维护。

---

## 所有权模型

| 对象 | 所拥有的资源与状态 | 生命周期 |
| --- | --- | --- |
| 本地任务 (`Local task`) | 可变默认源码映射、执行引用与证据记录 | 独立于远端执行 |
| 用户容器 (`User container`) | 单台宿主机上稳定的容器身份与 SSH 端点 | 独立于单个任务，跨任务持久保留 |
| 已准备运行时 (`Prepared runtime`) | 用户容器内单次执行的工作根目录及选定的 Python / CANN / 构建产物 | 保留验证证据；兼容的依赖与原生构建产物可跨执行共享 |
| 运行时绑定 (`Runtime binding`) | 单次执行对该准备根目录的检出使用权 | 检出（Checkout）至归还（Return） |
| 受管执行 (`Execution`) | 固定的 Git 源码输入、受管进程族、申请的 NPU 设备与服务端口 | 准入（Admission）至确认清理完成（Confirmed cleanup） |

容器的存在及其设备挂载本身不占用 NPU 租约；宿主机协调器（Host Coordinator）是 NPU 设备与宿主机服务端口的唯一分配权威。同一用户容器内的不同准备根目录只要 NPU 与服务端口租约互不重叠，即可并发绑定和运行。

---

## 环境选择与隔离

每次受管启动均明确记录工作树、Python 解释器、CANN 安装目录、构建产物、环境变量以及专属输出/缓存目录，在导入原生扩展库之前注入新进程，绝不篡改全局 Shell 配置或共享符号链接：

- `vllm` 与 `vllm-ascend` 使用各自独立的源码与构建路径。
- 不同虚拟环境可包含不同兼容组合的 `torch` / `torch_npu`，通过 `MINDIE_PYTHON` 显式选定并与全局包隔离。
- CANN 用户态工具链、算子包与动态库路径按版本目录整体选定；驱动/固件与内核能力仍受宿主机物理约束。
- 仅当输入完全匹配时才复用构建缓存；运行日志与可变输出归单次执行独占。

---

## 生命周期操作阶段

1. **设置默认绑定（`mindie_session` / 原生 Hook）**：客户端会话 Hook 自动关联本地任务，并根据实际工作目录（含 Git Linked Worktree 与已注册 Submodule）识别默认源码树。显式设置的任务级源码映射优先于自动发现，且仅影响后续提交。
2. **准入提交（`mindie_run` / `TaskClient.run`）**：在持久化准入前一次性捕获当前调用或默认源码的 Git 对象与 SCM 版本信息。无源码命令无需依赖业务树；纯 CPU 命令默认 `npu_count=0`。
3. **环境准备（Prepare）**：在用户的持久化容器内创建隔离的执行根目录，并通过运行时注册表复用兼容的依赖与原生扩展产物；准备阶段不占用运行态 NPU 租约。
4. **受管执行或常驻服务（Execute / Serve）**：持久化请求与进程身份，向宿主机申请 NPU 租约与服务端口，并注入 `MINDIE_EXECUTION_OBSERVATION` 启动。具名服务（`service="model"`）会校验输入一致性，替换已有服务需显式指定 `restart=True` 并等待旧进程完成清理。
5. **停止与释放（Stop / Release）**：仅终止当前执行所属的受管进程族（含子进程），确认进程族已排空且服务端口不再监听后释放 NPU 租约，同时保留容器与已准备好的运行环境。对于指定单张物理卡且设置 `resources.allow_external_busy=True` 的共享执行，释放时仅检查自身受管进程族与端口，不干扰外部已有工作负载。
6. **关闭任务（`mindie_finish` / `TaskClient.finish`）**：从未提交过受管执行的本地任务直接在本地注册表闭合；包含受管执行的任务由守护进程停止所属执行、归还绑定并标记为 `finished`。
7. **故障与幂等对账**：当进程或设备状态未知或清理失败时，执行继续保持所有权（`finishing`）直至守护进程完成对账，绝不因本地 PID 退出或超时而伪造资源释放。

---

## 参考资料

- [Python 虚拟环境文档](https://docs.python.org/3/library/venv.html)
- [TorchNPU 版本兼容表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)
- [CANN 环境变量参考](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha001/maintenref/envvar/envref_07_0003.html)
