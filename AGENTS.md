# MindIE Coordinator (`mindie-coordinator`) — Control-Plane Architecture & Invariants

Task-scoped Ascend NPU runtime orchestration, Git-identity parity materialization, and multi-host execution authority (`control-plane` repository class) for MindIE Agent.

## 1. Domain Boundary & Five Iron Rules

1. **Single Device & Port Authority (`HostQueue`)**:
   - Remote `mindie_coordinator/host/mindie_npu_coordination.py` (`HostQueue`) is the sole authority for physical NPU leases (`lease_id` + `fence_token` -> `ASCEND_RT_VISIBLE_DEVICES`), A3 dual-Die `Phy-ID 0..15` placement, cooperative container/service ports (`MINDIE_SERVICE_PORT`), and container process-tree draining.
   - Keep `sessions.sqlite3`, local `coordinator.sqlite3`, and the remote host SQLite queue database strictly separate (`records(kind="session")` schemas differ; never merge destructively).
   - Keep `<state_dir>/machines.json` (+ `.lock`, read by `npu-top`), `<state_dir>/endpoints.json` (read by `remote-dev`), `.mindie-runtime/ready-profile.json`, and `vllm_ascend_env.py` backwards-compatible.
2. **Zero Pre-Call Session-Binding Gates (`P2.7`)**:
   - Standard MCP `tools/call` with zero `_meta`, zero prior session hooks, and omitted `context_file` works out of the box across all 9 tools (`mindie.hosts`, `mindie.probe`, `mindie.container`, `mindie.lease`, `mindie.session`, `mindie.run`, `mindie.execution`, `mindie.finish`, `mindie.message`).
   - `mindie.execution` resolves explicit `execution_id` / `run_id` handles directly across MCP reconnections and subagents (`context_for_execution`).
3. **State & Receipt Honesty (`Principles 9 & 14`, Iron Rule 2)**:
   - Distinguish `completed`, `failed`, and `unknown` external effects; precede `except OSError:` with `except TimeoutError: raise`.
   - Reserve `uncertain` strictly for dispatched external writes (`submit_dispatched` or `submitted`). Pre-submit read/preflight failures stay `pending` and refresh `epoch` on retry; returned/drifted runtimes remain `ready` (re-verified by `inspect()` on `checkout`) with zero local one-way self-bricking (`needs_repair` / `TaskRootBusy` abolished).
   - Static operator AST checks (`op_preflight.py`) emit non-blocking `preflight_warnings` only and never abort `mindie.run` or `preflight`.
   - Once a remote `provision`, `remove`, or `lease` operation succeeds, local registry/receipt write failures keep `outcome="success"` (`isError=False`) and attach `receipt_warning` + `receipt_unconfirmed=True`.
4. **Standalone Pure-Stdlib Remote Workers & Intranet-Safe Environment (`Iron Rule 4`)**:
   - Standalone remote helper scripts live in `mindie_coordinator/workers/` (`python3 -I` pure stdlib) and concatenated modules (`runtime_profile.py`, `preparation_cache.py`, `build_inputs.py`, `native_incremental.py`, `native_cache_pool.py`, `shared_native_discovery.py`, `shared_source_objects.py`) avoid relative package imports.
   - Remote Python discovery (`workers/python_selector.py`) selects interpreters deterministically by `MINDIE_PYTHON` -> active venv -> `ready-profile.json` -> installed `torch` + `torch-npu` metadata (never `sort -V`), and preserves intranet `pip.conf` / `PIP_INDEX_URL` / `PIP_EXTRA_INDEX_URL` settings.
5. **SQLite & Timeout Arithmetic (`Principle 10`, Iron Rule 1)**:
   - Configure SQLite `WAL` + `busy_timeout=5000` once per connection in `open_database` / `transaction`.
   - Enforce `outer_timeout >= sum(inner_budgets) + cleanup_grace` (`STOP_DRAIN 15s >= STOP_GRACE 10s + 3s`).

## 2. Module Map (`mindie_coordinator/`)

- `task_server.py`, `ops.py`, `presentation.py`: Stdio JSON-RPC 2.0 MCP server, schema validation, Receipt Honesty hoisting, and bounded client summaries.
- `task_client.py` & `agent_session.py`: Hookless session auto-attachment (`ensure_context`) and explicit `execution_id` handle lookup (`context_for_execution`).
- `service.py`, `service_gang.py`, `service_evidence.py`, `placement.py`: `CoordinatorService` placement, parity sync, advisory preflight, multi-role gang coordination, and redacted diagnostic evidence export.
- `ready_runtime.py` & `managed_execution.py`: SQLite runtime pool (`coordinator.sqlite3`), pre-submit vs post-submit lease state machine, and remote job supervision.
- `host_queue.py` & `host/`: Remote host SQLite NPU/port lease queue, cgroup/process-tree draining, and `npu-smi` topology reconciliation.
- `parity.py`, `parity_snapshot.py`, `parity_install.py`, `parity_support.py`: Git commit/tree snapshot capture, bundle transport, and intranet-safe remote materialization.
- `backend.py`, `backend_preparation.py`, `prepare_runtime.py`, `runtime_profile.py`: Remote container verification, incremental native build cache reuse, and `ready-profile.json` attestation.
- `provision/`, `env_manager.py`, `machine_directory.py`, `endpoint_bridge.py`: Container provisioning, `machines.json` inventory, `endpoints.json` sync, and direct `HostQueue` lease handlers.
- `workers/`: Pure-stdlib remote worker modules (`host_probe_worker.py`, `python_selector.py`, `ready_probe_worker.py`).

## 3. Verification & Pre-PR Gate

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) ([English](CONTRIBUTING.en.md)) before submitting changes:

```bash
# Isolated Tier 0/1 test suite
pytest -q -m "not live_npu"

# Standing A2 (173.125.1.2) & A3 (192.168.13.165) non-destructive Tier 2 suite (never use .153)
pytest -q -m live_npu

# Unix LF line-ending & control-plane PR checklist gates
python3 .github/scripts/check_lf_eol.py
python3 .github/scripts/check_pr_checklist.py --profile control-plane --body-file <pr_body.md>
```
