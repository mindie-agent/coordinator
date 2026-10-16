# Tools, Runtime Preparation & Observability Reference

[中文](./configuration-and-tools.md) · **English** · [Back to Documentation Index](./README.en.md)

This reference documents submission and observation parameters, explicit single-NPU sharing, native build attestation and reuse, peer coordination messaging, the host NPU queue Python API, daemon version identity checks, and structured diagnostics in `mindie-coordinator`.

---

## 1. Submission, Bounded Waits & Evidence Inspection

- **Command vs. Script File**: `mindie_run` / `TaskClient.run` accept either `command` or a local UTF-8 `script_file` (up to 1 MiB; BOM is stripped, CRLF is normalized to LF, and both original and normalized digests are recorded).
- **Bounded Waits (`wait_until`)**:
  - Pass `wait_until="running"` or `"released"` with `wait_timeout_seconds` (`0–600` seconds).
  - `released` confirms process termination and resource release; business success additionally requires `state == "succeeded"`.
  - A wait timeout (`wait_timed_out=True`) never stops or resubmits the execution; callers can wait on or inspect the same `execution_id` again.
- **Non-Blocking Cache vs. Fresh Observations**:
  - Task MCP and CLI status calls return the latest persisted snapshot immediately (with `observation_freshness` reporting snapshot age, source, and whether a busy execution deferred refresh) and schedule background progression when older than 2 seconds.
  - Pass `--refresh` (`refresh: true`) for a live remote observation. `TaskClient.observe()` defaults to `refresh=True`; pass `refresh=False` to read the non-blocking cache.
- **Recorded Execution Evidence (`action="evidence"`)**:
  - Inspect `section="sources"`, `"preparation"`, `"build"`, or `"diagnostics"` to read retained source snapshots, `preparation_timings`, native profile receipts, and redacted `support_bundle` data.
- **Preflight Commands (`preflight`)**: Validate the prepared environment with the selected interpreter and placeholder port `0` before any NPU lease is allocated; failures retain stderr references and abort launch before allocating devices.

---

## 2. Physical NPU Allocation & Explicit Single-Device Sharing

- **Zero-NPU Default**: `resources` defaults to `npu_count=0` so CPU or compiler tasks reserve no NPUs. Specify `npu_count` or explicit `devices` (with `topology={"host": "..."}`) for NPU workloads.
- **Explicit Single-NPU External Sharing (`allow_external_busy=True`)**:
  ```python
  resources={"devices": [0], "allow_external_busy": True}
  ```
  Requires exactly one explicit physical device and permits pre-existing external process/HBM occupancy while still conflicting with other managed coordinator leases. Stop/finish terminates only this execution's managed process family and releases its lease once its descendants and ports have drained, leaving external workloads untouched.
- **Service Port Selection**: `service_port=0` selects a free port from the declared service port list or the host default range (`30000–45999`), checking both active sockets and existing leases.

---

## 3. Incremental Git Materialization & Native Build Reuse

- **Incremental Git Snapshot Materialization**: Reuses immutable objects already present in the remote mirror and transfers small working-tree edits as a bounded Git pack over the existing RPC connection, verifying parent commits and resulting tree hashes before atomic publication.
- **Native Attestation Reuse**: The initial native environment runs a full framework import smoke test and records artifact hashes. Subsequent Python-only edits reuse that import proof (`python_import_executed=false` in the receipt) whenever dependency/native inputs, loader environment, and artifact hashes match.
- **Donor Runtime Registration**: Register an existing native artifact donor via `mindie-coordinator runtime-register --reuse-only --source vllm=PATH --source vllm-ascend=PATH` (or `CoordinatorClient.runtime_register`).

---

## 4. Peer Coordination Messaging (`mindie_message`)

While waiting on queued resources, tasks can send coordination messages to `coordination_peers[].reference` or reply via `notifications[].reply_reference` using `mindie_message`:

- Messages are stored alongside the host queue database and polled asynchronously by the local coordinator at a minimum 5-second interval during normal `run` / `status` calls.
- Messages are strictly informational—they never execute commands, release leases, or stop another user's task.

---

## 5. Host NPU Queue Python API & Daemon Lifecycle

### Public Host Queue Imports

```python
from mindie_coordinator.host_queue import (
    HostQueue,
    HostQueueUnavailable,
    SCHEMA_VERSION,
    CoordinationError,
    handle_request,
    host_queue_module_path,
    load_host_protocol,
)
```

Host state defaults to `/tmp/mindie-npu-coordinator/v1/` (overridable via `MINDIE_NPU_COORDINATOR_STATE_DIR`).

### Local Daemon Status & Graceful Restart

```bash
# Inspect loaded vs. installed package and commit identities without starting a daemon
mindie-coordinator daemon --action status

# Restart the daemon to the installed version only when no executions or leases are active
mindie-coordinator daemon --action restart-if-idle
```

Managed admission requires the daemon's `coordinator` and `remote-dev` commit and Python interpreter identities to match the caller's. An idle mismatched daemon restarts automatically; a busy daemon keeps active executions running and returns `needs_runtime_update`.

---

## 6. Structured Diagnostics

- Tool responses include `diagnostics` (`operation_id`, `trace_id`, UTC start/end, monotonic duration, bounded phase summaries, and local `record_ref`) and `error_details` (preserving category, retryability, and submission certainty: `not_sent` / `not_executed`, `uncertain`, or `acknowledged`).
- `MINDIE_LOG_LEVEL=INFO` is the default and `MINDIE_DIAGNOSTICS_ROOT` configures the local log root. Raw commands, file contents, environment values, and credentials are never passed as diagnostic logging arguments.
- Run `python -m mindie_coordinator.mindie execution --execution-id ID --action evidence --section diagnostics` to retrieve the execution's stage timings and redacted `support_bundle`.
