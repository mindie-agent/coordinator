# Tools, Single-NPU Sharing, Build Reuse & Diagnostics Reference

[中文](./configuration-and-tools.md) · **English** · [Back to Documentation Index](./README.en.md)

This guide covers six practical topics for daily work with `mindie-coordinator`: **submitting commands and waiting on results, sharing a single NPU card with external processes, reusing compiled native operators, messaging peers in the host queue, managing the local daemon, and inspecting structured diagnostics.**

---

## 1. Submitting Work, Waiting & Inspecting Evidence

- **Command String vs. Local Script File**:
  `mindie_run` and `TaskClient.run` accept either a `command` string or a local UTF-8 `script_file` path (up to 1 MiB; UTF-8 BOM is stripped and Windows CRLF line endings are automatically converted to LF so Linux shells never fail on ``).
- **Bounded Waits (`wait_until`)**:
  - Pass `wait_until="running"` (return as soon as the workload launches) or `wait_until="released"` (wait until the process finishes and NPU leases are released), along with `wait_timeout_seconds` (`0–600` seconds).
  - **`released` vs. `succeeded`**: `released` means the process has exited and hardware resources have been returned; check `state == "succeeded"` to confirm the command itself exited with code 0.
  - **Wait Timeouts Never Kill Your Job**: If `wait_timeout_seconds` expires before completion (`wait_timed_out=True`), the execution keeps running normally on the remote server. You can call `mindie execution` or `TaskClient.observe()` on the same `execution_id` to keep watching or waiting.
- **Instant Cached Status vs. Live Refresh**:
  - By default, MCP `mindie_execution` and CLI `mindie execution` return the latest locally persisted snapshot immediately (with `observation_freshness` showing its age in seconds) and trigger a background refresh if the snapshot is older than 2 seconds.
  - Pass `--refresh` (`refresh: true`) when you want an immediate live poll from the remote host. Python `TaskClient.observe()` defaults to `refresh=True` (pass `refresh=False` to read the non-blocking cache).
- **Inspecting Recorded Evidence (`action="evidence"`)**:
  Query `section="sources"`, `"preparation"`, `"build"`, or `"diagnostics"` to inspect the exact Git commit snapshot, preparation stage timings, native build receipts, or redacted `support_bundle`.
- **Pre-Lease Preflight Checks (`preflight`)**:
  Run a lightweight validation command with the selected Python interpreter and placeholder port `0` **before** acquiring any NPU lease. If preflight fails, launch aborts immediately and returns `stderr` without wasting queue time or NPU leases.

---

## 2. Allocating NPUs & Sharing a Single Card (`allow_external_busy`)

- **Zero-NPU Default for CPU/Build Tasks**:
  `resources` defaults to `npu_count=0`, so compilation or CPU-only tests never tie up NPU cards. Request NPUs by setting `npu_count` or passing explicit physical `devices` (with `topology={"host": "..."}`).
- **Explicit Single-NPU Sharing (`allow_external_busy=True`)**:
  ```python
  resources={"devices": [0], "allow_external_busy": True}
  ```
  When you target **one specific physical device** (e.g., `devices: [0]`) and set `allow_external_busy=True`, the coordinator allows your job to run even if that card already has unmanaged external processes or HBM usage (while still queuing fairly against other coordinator-managed leases). When your job finishes or stops, **only your own managed process tree and lease are cleaned up — external processes on that card are never touched.**
- **Automatic Free Port Selection**:
  Pass `service_port=0` to let the host coordinator pick an unused port from `30000–45999` (or your declared port list) that is neither bound by an active socket nor held by another lease.

---

## 3. Fast Incremental Git Sync & Native Build Reuse

- **Incremental Git Snapshot Sync**:
  Reuses immutable Git objects already present on the remote host and sends small working-tree edits as a compact Git pack over the existing SSH RPC connection, verifying parent commits and tree hashes before publishing.
- **Skip Redundant Operator Rebuilds & Import Checks**:
  When a native environment is built for the first time, `mindie-coordinator` runs a full framework import smoke test and records artifact hashes. On subsequent runs where you only edited Python files and the underlying dependencies, loader environment, and compiled operator hashes are unchanged, it reuses that verified proof (`python_import_executed=false` in the receipt) and launches your script immediately.
- **Register Existing Prebuilt Directories**:
  You can register an existing compiled `vllm` / `vllm-ascend` tree inside the container as a read-only reuse source:
  ```bash
  mindie-coordinator runtime-register --reuse-only --source vllm=PATH --source vllm-ascend=PATH
  ```

---

## 4. Messaging Teammates in the Queue (`mindie_message`)

When your task is queued waiting for NPUs and reports `coordination_peers[].reference`, you can send a coordination message to that peer (or reply via `notifications[].reply_reference`) using `mindie_message`:

- Messages live alongside the host queue database and are polled asynchronously by the local daemon (at most once every 5 seconds during normal `run` / `status` calls).
- **Strictly Informational**: Messages never execute commands, release leases, or stop another user's workload.

---

## 5. Host Queue Python API & Local Daemon Management

### Public Host Queue Python Imports

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

Host queue state defaults to `/tmp/mindie-npu-coordinator/v1/` (configurable via `MINDIE_NPU_COORDINATOR_STATE_DIR`).

### Inspecting or Gracefully Restarting the Local Daemon

```bash
# Compare the running daemon's package/commit version against the installed version
mindie-coordinator daemon --action status

# Restart the daemon to the newly installed version only when no jobs or leases are active
mindie-coordinator daemon --action restart-if-idle
```

When you upgrade `coordinator` or `remote-dev`, an idle daemon automatically restarts on the next submission to match your environment; if the daemon is currently managing active runs, it keeps them running safely and returns `needs_runtime_update`.

---

## 6. Structured Diagnostics & Privacy

- Every response includes `diagnostics` (`operation_id`, `trace_id`, start/end timestamps, monotonic duration, stage timings, and local `record_ref`) plus `error_details` reporting submission certainty (`not_sent` / `not_executed`, `uncertain`, or `acknowledged`).
- Logs default to `MINDIE_LOG_LEVEL=INFO` under `MINDIE_DIAGNOSTICS_ROOT`. Raw commands, source file contents, environment variable values, and credentials are never written to diagnostic log fields.
- Retrieve a redacted diagnostic support bundle for any owned execution with:
  ```bash
  python -m mindie_coordinator.mindie execution --execution-id <ID> --action evidence --section diagnostics
  ```
