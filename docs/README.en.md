# `mindie-coordinator` Documentation

[中文](./README.md) · **English** · [Back to Project Homepage](../README.en.md)

Welcome to the `mindie-coordinator` documentation hub. Here you will find clear explanations of how persistent user containers (`mindie-<user>`) work, how NPU leases are granted and released, and how to use every tool and configuration option.

## Guides & References

| Topic | 中文 | English | What You'll Learn |
| --- | --- | --- | --- |
| Persistent User Container & Task Lifecycle | [session-lifecycle.zh.md](./session-lifecycle.zh.md) | [session-lifecycle.md](./session-lifecycle.md) | How local tasks, `mindie-<user>` containers, prepared runtimes, and NPU leases are managed from submission to cleanup |
| Tools, Single-NPU Sharing, Build Reuse & Diagnostics | [configuration-and-tools.md](./configuration-and-tools.md) | [configuration-and-tools.en.md](./configuration-and-tools.en.md) | Parameters for `mindie_run` / `TaskClient`, single-NPU sharing, native build cache reuse, peer messaging, and diagnostics |
