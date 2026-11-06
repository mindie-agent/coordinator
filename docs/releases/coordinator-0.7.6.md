# coordinator 0.7.6

[English](coordinator-0.7.6.en.md)

本版本修复业务解释器选择与传递，并让准备流程不适用时仍能继续已授权任务。

- 锁定 remote-dev 0.11.7 的正式发布源码，共用有限解释器发现；容器就绪按 Torch / Torch-NPU（MindIE-SD 场景还需对应框架）元数据选择，保留活动环境和 venv 身份。
- 将已选解释器传递给后续准备。复用已验证的同一容器端点时保留业务 Python 与 SSH key；其他端点不继承绑定。
- 新任务 venv 使用选定 donor 创建，构建前检查必要依赖；另一个 venv 的私有包不会被冒充为已继承，donor 保持不变。
- 明确直接远端目录开发与受管源码物化的边界。失败结果的 `next` 保留执行查询入口和资源释放事实，支持任务脚本、CLI 或原生工具继续；未知提交先观察和对账。
- 保留 MindIE 与 MindIE-SD 镜像的就绪差异；压缩 readiness 传输，兼容 Windows SSH 命令长度限制。

先发布 remote-dev 0.11.7，再发布本版本；用户通过正式组件 Release 独立更新。CI 覆盖 Linux、macOS、Windows 的本地合同与正常依赖安装；精确 Tag 的测试与 benchmark 由发布工作流执行。A2/A3 SSH 不可达，实际原生重编译、动态库加载与 NPU Canary 尚未验收。
