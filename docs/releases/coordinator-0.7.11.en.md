# mindie-coordinator 0.7.11

## Storage Lifecycle & Retention Fixes

- **Local Dirty-Worktree Git Ref Leak (C-4)**: After `capture-<hash>.json` is atomically replaced, delete the superseded `refs/parity/execution-inputs/...` synthetic refs (including their `-scm` companion). Refs still reused by the current cache remain intact, so historical dirty-tree blobs no longer block `git gc` forever.
- **`capture-*.json` Retention**: Bound `source-inputs/capture-*.json` to the newest 128 files and 14 days; the current cache file is never deleted.
- **`.mindie-local/fixed-inputs` Leak (C-6)**: Unlink transport-descriptor temporary files in `finally` when a write fails; prune descriptors older than 24 hours before the next materialization, and reclaim the completed invocation's descriptor right after `parity sync`.
- **Prepare-Log Bound**: Keep only the newest 16 `.mindie-runtime/prepare-logs/<phase>.XXXXXX` logs after each preparation step so retries cannot accumulate logs without bound.
- **Terminal Run-Directory Retention**: Reclaim `<state_dir>/runs/<run_id>/` only for definitively terminal executions (`succeeded`/`failed`/`timeout`/`cancelled`), keeping at most 128 and 14 days. Active runs and `inconclusive`/`unknown` outcomes are never deleted.
- **Remote Native-Cache Global LRU (C-1)**: `/tmp/mindie-native-cache` now enforces a 16-entry / 8 GiB global budget (`MINDIE_NATIVE_CACHE_MAX_BYTES` overrides the byte limit). Entries are evicted oldest-mtime first with an atomic rename to `.trash-*` followed by `rmtree`, strictly skipping the just-published/requested keys, locked entries, and immutable bundles still referenced by a content entry. Immutable bundles and writable content entries remain independent copies; a read-only cache is never turned into a shared mutable directory.
- **Remote Staging Cleanup**: Orphaned `.publish-*`, `.native-content-*`, `.kernel-content-*`, `.trash-*`, and `kernel-recipe-*` staging older than one hour is swept during later publications/compiles, and `finally` removes any staging directory a successful or failed rename did not consume.
- **Host SQLite Retention (C-3/C-5)**: `_housekeep()` now removes terminal tasks/holds older than seven days and expired messages, and bounds `events` to the newest 4096 rows. Active leases, active holds, and `unknown`/`orphaned_busy` states are untouched. The host protocol schema is now `8`; existing databases backfill `messages.created_at` and gain retention indexes automatically.
- **Shared Source Snapshot Ref Bound**: Keep only the newest 64 `refs/mindie/snapshots/<commit>` refs by commit time so transient commits cannot pin the shared object store indefinitely.
- **Regression Tests**: New Tier 0/1 coverage for superseded synthetic-ref deletion, native-cache entry/byte caps and staging cleanup, locked-entry protection, SQLite terminal-row retention, failed fixed-input writes, and the shared snapshot-ref window.

## Companion Dependencies

- This release publishes only the coordinator fixes; the `mindie-diagnostics` and `remote-dev` pinned commits remain at their currently published versions.
