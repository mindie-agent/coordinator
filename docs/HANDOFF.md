# Handoff: coordinator as a local installable package

Status: current package ownership, 2026-09-12. Execution behavior is defined
by README.md and session-lifecycle.md.

This repository is the MindIE coordinator extracted from
`vllm-ascend-workspace`. It is a **local process**: it coordinates the current
user's own remote containers and NPU allocation. It does not host a manager
for other people to connect to.

Install as `mindie-coordinator` (import `mindie_coordinator`). The public TaskClient
contract is described in [session-lifecycle.md](session-lifecycle.md) and
implemented in [task_client.py](../mindie_coordinator/task_client.py).

## Ownership

| Piece | Owner | Notes |
| --- | --- | --- |
| Host NPU authority | this package, `mindie_coordinator.host` | Shipped to the host and executed there. One implementation. Marker: `REMOTE_DEV_JOB_TOKEN`. |
| Host queue client | `mindie_coordinator.host_queue` | Public API. |
| Task tools / stdio MCP | `mindie_coordinator.task_server` | `mindie-coordinator task-server` |
| Runtime pool / managed jobs | `mindie_coordinator.ready_runtime`, `managed_execution` | In-process, this user. Remote process control is `remote_dev.processes.control`. |
| Result envelope | `remote_dev.result` | `schema_version: remote-dev.result.v1` |
| Remote shell | `remote_dev.core.endpoint` / `remote_dev.core.shell_ops` | Pip package `remote-dev`. |
| Run Manifest v1 | this package, `mindie_coordinator.run_manifest` | `code` is Git identity (`source_head`, `snapshot_commit`). |
| Code identity | this package, `mindie_coordinator.code_identity` | Dirty trees get a parentless snapshot commit. |
| Code parity | this package, `mindie_coordinator.parity` | In-package CLI; no consumer script path. |
| Execution preparation | Fixed execution snapshot, isolated source root, verified dependency/native cache | Generic commands use an available interpreter without a new venv. Native builds and reuse remain package-owned. |
| Machine directory | this package, `mindie_coordinator.machine_directory` | Consumers pass a document; store is coordinator-owned. |
| Persistent daemon | `mindie_coordinator.service` | Short `/tmp/vc-<user>-<sha>.sock`; lock/sqlite in state dir. |

## What this package expects from remote-dev

Import the installed `remote_dev` package. Do not locate a checkout by path.

1. `remote_dev.result.make_result` — `RESULT_SCHEMA_VERSION` is `remote-dev.result.v1`.
2. `remote_dev.core.endpoint.resolve_endpoint(mapping)` from an explicit
   `host` + `port` (this package refuses calls without both).
3. `remote_dev.core.shell_ops.remote_bash(endpoint, *, command, timeout_ms, runtime_env)`
   returning `{"result": {...}}` with `outcome` and `refs.stdout` / `refs.stderr`.
4. `remote_dev.processes.control(endpoint, job_id, action, **params)`.
5. Worker marker `REMOTE_DEV_JOB_TOKEN` and reserved prefix `REMOTE_DEV_JOB_`.

## Validation coverage

The source suite covers the control plane, local daemon, and shared transport.
The Linux `/proc` worker/guard test is skipped on macOS. Tests using fake
resources do not establish real NPU or container preparation behavior.

Behavioral tests cover immutable execution inputs, independent source defaults,
service replacement, multi-role gate activation and lease renewal, CPU-only
resource claims, process/port cleanup, daemon restart and native Windows/WSL
ownership. Preparation tests reject changed build inputs, damaged native
artifacts and identity collisions; copied artifacts must match their donor
proof. Platform-specific cases are skipped when their OS primitives are absent.

Use the installed-wheel and real-container evidence for claims about remote
execution. Unit fixtures alone cannot establish successful native builds or
NPU imports.

No-donor first machine: `_ensure_user_container` calls coordinator-owned
`provision_user_container` with HostQueue `container-ssh-reserve`, then
prepare. Requires a configured `MachineDirectory` row and a supported recipe.

Hardware and model acceptance require separate remote validation.
