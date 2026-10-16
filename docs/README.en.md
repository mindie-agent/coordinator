# `mindie-coordinator` Documentation

[中文](./README.md) · **English** · [Back to Project Homepage](../README.en.md)

This directory contains the persistent user-container task lifecycle contract and the technical reference for tools, runtime preparation, host NPU queueing, and diagnostics in `mindie-coordinator`.

## Documents

| Document | 中文 | English | Summary |
| --- | --- | --- | --- |
| Persistent User Container & Task Lifecycle | [session-lifecycle.zh.md](./session-lifecycle.zh.md) | [session-lifecycle.md](./session-lifecycle.md) | Ownership boundaries and lifecycle transitions across local tasks, persistent `mindie-<user>` containers, prepared runtimes, and managed executions |
| Tools, Runtime Preparation & Observability | [configuration-and-tools.md](./configuration-and-tools.md) | [configuration-and-tools.en.md](./configuration-and-tools.en.md) | `TaskClient` / MCP / CLI semantics, explicit single-NPU sharing, native build reuse, peer messaging, daemon identity checks, and structured diagnostics |
