# `mindie-coordinator` Documentation

[中文](./README.md) · **English** · [Back to Project Homepage](../README.en.md)

Welcome to the `mindie-coordinator` documentation hub. Here you will find clear explanations of how persistent user containers (`mindie-<user>`) work, how NPU leases are granted and released, and how to use every tool and configuration option.

## Guides & References

| Topic | 中文 | English | What You'll Learn |
| --- | --- | --- | --- |
| Persistent User Container & Task Lifecycle | [session-lifecycle.zh.md](./session-lifecycle.zh.md) | [session-lifecycle.md](./session-lifecycle.md) | How local tasks, `mindie-<user>` containers, prepared runtimes, and NPU leases are managed from submission to cleanup |
| Tools, Single-NPU Sharing, Build Reuse & Diagnostics | [configuration-and-tools.md](./configuration-and-tools.md) | [configuration-and-tools.en.md](./configuration-and-tools.en.md) | Parameters for `mindie_run` / `TaskClient`, single-NPU sharing, native build cache reuse, peer messaging, and diagnostics |

## Release Notes

| Version | 中文 | English | Highlights |
| --- | --- | --- | --- |
| `v0.7.14` | [coordinator-0.7.14.md](./releases/coordinator-0.7.14.md) | [coordinator-0.7.14.en.md](./releases/coordinator-0.7.14.en.md) | Windows persistent daemon Win32 Job breakaway (`CREATE_BREAKAWAY_FROM_JOB` with `winerror=5` fallback), compact `coordinator.ipc.<hex8>.tmp` staging markers, and companion dependency pin updates |
| `v0.7.13` | [coordinator-0.7.13.md](./releases/coordinator-0.7.13.md) | [coordinator-0.7.13.en.md](./releases/coordinator-0.7.13.en.md) | Removal of pre-call session-binding gates on deterministic tools (`P2.7`), elimination of one-way local fault lockouts, and receipt honesty hardening |
