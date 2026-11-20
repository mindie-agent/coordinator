# `mindie-coordinator` 文档中心

**中文** · [English](./README.en.md) · [返回项目主页](../README.md)

这里汇总了 `mindie-coordinator` 的工作原理、远端个人专属容器（`mindie-<用户名>`）的生命周期说明，以及日常调工具、传参数、共享单卡和查诊断证据的实用指南。

## 文档导航

| 文档主题 | 中文 | English | 你能在这里查到什么 |
| --- | --- | --- | --- |
| 专属容器与任务全生命周期说明 | [session-lifecycle.zh.md](./session-lifecycle.zh.md) | [session-lifecycle.md](./session-lifecycle.md) | 你的专属容器 `mindie-<user>` 是怎么管理的？一次任务从提交代码、准备环境、占卡运行到自动释放经历了哪几步？ |
| 工具参数、单卡共享、算子复用与诊断参考 | [configuration-and-tools.md](./configuration-and-tools.md) | [configuration-and-tools.en.md](./configuration-and-tools.en.md) | `mindie_run` / `TaskClient` 怎么传参、怎么跟别人的任务共用单张 NPU 卡、怎么跳过重复编译、怎么给排队的同事发消息 |

## 版本发布说明

| 版本 | 中文 | English | 核心变更摘要 |
| --- | --- | --- | --- |
| `v0.7.14` | [coordinator-0.7.14.md](./releases/coordinator-0.7.14.md) | [coordinator-0.7.14.en.md](./releases/coordinator-0.7.14.en.md) | Windows 常驻守护进程脱离单次前台 CLI 的 Win32 Job（`CREATE_BREAKAWAY_FROM_JOB` 与 `winerror=5` 受控回退）、精简 `coordinator.ipc.<hex8>.tmp` 临时标记文件名及配套底座依赖更新 |
| `v0.7.13` | [coordinator-0.7.13.md](./releases/coordinator-0.7.13.md) | [coordinator-0.7.13.en.md](./releases/coordinator-0.7.13.en.md) | 移除确定性工具前置会话绑定门禁（`P2.7` 零绑定直通）、消除本地单向锁死状态与落实回执诚实性 |
