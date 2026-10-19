# MindIE Coordinator (`mindie-coordinator`)

Task-scoped Ascend NPU runtime orchestration, parity materialization, and multi-host execution authority for MindIE Agent.

## 1. Domain Boundary & Core Principles

- **Single Device & Port Authority**: The host-side queue (`mindie_coordinator/host/mindie_npu_coordination.py`) is the sole authority for NPU device leases (`ASCEND_RT_VISIBLE_DEVICES`), service ports (`MINDIE_SERVICE_PORT`), and container process-tree draining.
- **Git-Identity Parity**: Workspace snapshots are identified and materialized strictly by Git commit / tree identity (`source_head`, `snapshot_commit`), never by ad-hoc source file content hashes.
- **State Honesty & No Silent Fallbacks**: Distinguish `completed`, `failed`, and `unknown` external effects. Never mask failures as empty dicts or `exit 0`, never let `except OSError:` swallow `TimeoutError`, and never blindly replay an uncertain external write.
- **No Self-Bricking**: Reserve `unknown` / `inconclusive` strictly for irreversible external writes (`submit` / `stop` timeout after dispatch); local errors never permanently lock a task root (`TaskRootBusy` / permanent `fault` / `suppressed` are abolished).

## 2. Module Map (`mindie_coordinator/`)

- `task_server.py`: Stdio JSON-RPC 2.0 MCP server (`mindie.session`, `mindie.run`, `mindie.execution`, `mindie.finish`).
- `ops.py` & `presentation.py`: MCP tool validation, error classification, and structured human/agent output formatting.
- `task_client.py` & `agent_session.py`: Local session attachment, task context persistence, and Git common-dir binding.
- `hooks/mindie_session.py`: Cross-client session hook adapter (Claude Code, Codex, Cursor, Kimi, OpenCode).
- `service.py`: Central `CoordinatorService` orchestrating placement, source parity, preparation, submission, and recovery.
- `placement.py`: Pure constraint solver matching recipe/ABI/SoC/topology requirements to attested host candidates.
- `fleet_probe.py`: Concurrent SSH fleet polling of host NPU queues and prepared runtime catalogs.
- `host_queue.py` & `host/mindie_npu_coordination.py`: Host-side SQLite NPU lease queue, cgroup/process-tree draining, and `npu-smi` reconciliation.
- `managed_execution.py`: Owner-side execution state machine (`submitting`, `queued`, `running`, `stopping`, `released`, `inconclusive`).
- `parity.py`, `parity_support.py`, `code_identity.py`: Git snapshot capture, bundle transport, and remote fixed-worktree materialization.
- `prepare_runtime.py`, `preparation_cache.py`, `runtime_profile.py`, `ready_runtime.py`: Remote container runtime probing, incremental build caching, and prepared activation scripts.
- `build_inputs.py`, `native_incremental.py`, `native_kernel_recipe.py`, `shared_native_discovery.py`, `shared_source_objects.py`: Incremental C++/Ascend C native build and `.so` artifact discovery helpers (concatenated into remote preparation scripts—keep top-level helpers self-contained).
- `provision/host_ops.py`, `provision/task_environment.py`, `env_manager.py`, `endpoint_bridge.py`: Host/container provisioning, `machines.json` registry, and `remote-dev` endpoint synchronization.
- `run_manifest.py` & `execution_capture.py`: Immutable Git-identity run manifests and bounded execution log/evidence capture.

## 3. Timeout Arithmetic & Concurrency Invariants

- **Outer >= Sum(Inner) + Grace**:
  - `STOP_DRAIN (15s) >= STOP_GRACE (10s) + 3s` cleanup margin.
  - Verifier `outer_timeout >= inner_timeout + inventory_grace` (`timeout + 15s`).
  - Multi-stage SSH mux diagnostics (`-O check`, `-O exit`,probe) use independent per-stage timeouts.
- **Lightweight SQLite (`host/mindie_npu_coordination.py`)**:
  - `open_database` configures `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000` once per connection; never run `:memory:` schema reflection or `.owner` `lstat` on every `SELECT`.
- **Remote Script Concatenation**:
  - `prepare_runtime.py` concatenates `runtime_profile.py`, `preparation_cache.py`, `build_inputs.py`, `native_incremental.py`, `native_kernel_recipe.py`, `shared_native_discovery.py`, and `shared_source_objects.py` via `.read_text()`. Do not introduce package-relative imports inside those concatenated modules.

## 4. Verification & Testing Pyramid

Run fast local tests (Tier 0 & Tier 1):

```bash
pytest -q -m "not live_npu"
```

Run standing live A2/A3 Ascend NPU integration tests (Tier 2, non-destructive isolated sandboxes on A2 `root@173.125.1.2` and A3 `root@192.168.13.165` / `.154` — **never connect to `.153`**):

```bash
pytest -q -m live_npu
```
