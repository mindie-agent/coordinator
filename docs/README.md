# `mindie-coordinator` 文档中心

**中文** · [English](./README.en.md) · [返回项目主页](../README.md)

本目录收录 `mindie-coordinator` 的持久化容器任务生命周期契约，以及工具接口、运行时准备、宿主机队列与诊断参考文档。

## 文档列表

| 文档 | 中文 | English | 内容概述 |
| --- | --- | --- | --- |
| 持久化用户容器与任务生命周期契约 | [session-lifecycle.zh.md](./session-lifecycle.zh.md) | [session-lifecycle.md](./session-lifecycle.md) | 本地任务、持久化用户容器（`mindie-<user>`）、已准备运行时与受管执行的所有权边界及生命周期转换 |
| 工具接口、运行时准备与可观测性参考 | [configuration-and-tools.md](./configuration-and-tools.md) | [configuration-and-tools.en.md](./configuration-and-tools.en.md) | `TaskClient` / MCP / CLI 参数细节、单卡共享、原生构建复用、跨任务消息、守护进程版本一致性与结构化诊断 |
