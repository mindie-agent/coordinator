# mindie-coordinator 0.7.11

## 存储生命周期与保留策略修复

- **本地脏工作树 Git Ref 泄漏（C-4）**：`capture-<hash>.json` 原子替换成功后，删除已被替换的 `refs/parity/execution-inputs/...` 合成 ref（连同 `-scm` 伴随 ref）；仍被当前缓存复用的 ref 保持不变，历史脏工作树对象不再永久阻止 `git gc`。
- **`capture-*.json` 保留上限**：`source-inputs/capture-*.json` 按最新 128 份 / 14 天双上限回收，当前缓存文件永不删除。
- **`.mindie-local/fixed-inputs` 泄漏（C-6）**：传输描述文件写入失败时在 `finally` 中删除临时文件；超过 24 小时的旧描述文件在下次 materialize 前清理，并在 `parity sync` 完成后立即回收本次描述文件。
- **准备日志上限**：`.mindie-runtime/prepare-logs/<phase>.XXXXXX` 在每次准备步骤后仅保留最新 16 份，失败重试不再无限堆积日志。
- **终态运行目录保留**：`<state_dir>/runs/<run_id>/` 仅对明确终态（`succeeded`/`failed`/`timeout`/`cancelled`）按最新 128 个 / 14 天回收；活动执行与 `inconclusive`/`unknown` 结果永不删除。
- **远端原生缓存全局 LRU（C-1）**：`/tmp/mindie-native-cache` 新增 16 条目 / 8 GiB 全局预算（`MINDIE_NATIVE_CACHE_MAX_BYTES` 可覆盖字节上限），按入口 mtime 从旧到新原子改名到 `.trash-*` 后删除，严格跳过刚发布/请求的键、已锁定条目，以及仍被内容条目引用的不可变 bundle；不可变 bundle 与可写内容条目继续保持独立副本，绝不把只读缓存变成共享可写目录。
- **远端 staging 清理**：超过 1 小时的 `.publish-*`、`.native-content-*`、`.kernel-content-*`、`.trash-*` 与 `kernel-recipe-*` 孤儿 staging 会在后续发布/编译时清理；成功与失败路径都在 `finally` 中移除未消费的临时目录。
- **宿主机 SQLite 保留（C-3/C-5）**：`_housekeep()` 现在回收 7 天前的终态 tasks/holds、过期 messages，并将 events 限制为最新 4096 条；活动租约、活动 holds 与 `unknown`/`orphaned_busy` 状态不受影响。宿主协议 schema 升级到 `8`，旧数据库自动补齐 `messages.created_at` 并建立保留索引。
- **共享源快照 ref 上限**：`refs/mindie/snapshots/<commit>` 仅保留按提交时间最新的 64 个快照 ref，避免一次性提交长期钉住共享对象库。
- **回归测试**：新增 Tier 0/1 测试覆盖合成 ref 替换删除、原生缓存条目/字节上限与 staging 清理、锁定条目保护、SQLite 终态行保留、fixed-input 写失败清理，以及共享快照 ref 上限。

## 配套依赖

- 本次仅发布 coordinator 自身修复；`mindie-diagnostics` 与 `remote-dev` 的固定提交保持当前已发布版本不变。
