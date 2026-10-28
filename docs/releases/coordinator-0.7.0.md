# mindie-coordinator 0.7.0

[English](coordinator-0.7.0.en.md)

本版本将主机接入、密码保存、公钥安装和 SSH 恢复统一交给 remote-dev。首次配置返回的 `auth_ref` 写入机器清单和 endpoint，之后新 Agent 会话无需再次提供密码。不支持公钥的主机仍使用已经验证的共享密码配置，不会因公钥安装失败而否决有效的密码登录。

coordinator 不再维护自己的 askpass、密码传递或公钥安装实现。密码只进入 remote-dev 的用户私有明文凭据文件；机器清单和 npu-top 注册请求只传认证引用。同一用户下的插件虚拟环境共享配置，不调用系统钥匙串或要求系统解锁。容器的直接 SSH 端口属于独立认证范围，不会继承另一个主机端口的认证引用。

与 npu-top 的同步读取实际 HTTP 207 逐台回执。认证尚未就绪或本地引用记录失败会明确报告为部分成功，同时保留已完成的主机登记事实。已有公钥及清单元数据继续保留，普通远端操作不会因认证恢复而被重放。

本组件锁定 remote-dev 0.11.1 正式 Release 的完整 Git 提交 `2a750ec8c74d7f053410d48dd820aeb542e7d3b1`，并继续固定 mindie-diagnostics 0.5.4（`5ae643e19223f2181ecdbaa3edaf464137c4805e`）。发布流水线校验安装来源、正式依赖 Release、源码与 Tag 对齐，并随包提供 benchmark 报告。

remote-dev 0.11.1 补齐 Git for Windows 2.56 的 `ucrt64/bin` 路径识别，使 Git Bash 启动的 Python 与 PowerShell 一样能找到同一安装目录内的 SSH。

共享认证已通过三平台真实 SSH 验收：macOS 的安装后三组件 wheel 与 Kimi 候选共 7 项通过（34.58 秒）；原生 Windows PowerShell 4 项通过（14.186 秒），真实 WSL2 Ubuntu 24.04 4 项通过（12.694 秒），后两者均无失败或跳过。[验收运行 40774308378](https://github.com/mindie-agent/remote-dev/actions/runs/40774308378) 使用与 remote-dev 候选 `a7564f49b1ff106e59f72961c851b7b11eb81130` 相同的源码树。

实际消费者路径覆盖 coordinator 主机接入、Kimi 两个会话和两个虚拟环境、npu-top 数据库重开后的连接与采集，共用隔离 `REMOTE_DEV_AUTH_DIR` 中默认的 `credentials.json`。同时验证自动安装公钥、禁用公钥的密码登录及撤销公钥后的修复，业务操作仅执行一次。Codex、Claude Code 和 DSH 的覆盖范围为 endpoint profile，未声称完整原生宿主验收。HTTP 207 回执与本地消费者合同通过；固定调度与运行清单 benchmark 工作负载各完成 5 次本地兼容性检查，正式报告由 Tag 流水线生成。
