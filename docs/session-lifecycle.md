# Task Lifecycle in a Persistent User Container

[中文](./session-lifecycle.zh.md) · **English** · [Back to Documentation Index](./README.en.md)

The persistent coordinator daemon owns fixed execution inputs, isolated execution
roots, multi-role reservations, and confirmed cleanup. Task completion preserves
the user container and its prepared environments. Each user has one persistent
container on each host, named `mindie-<user>` (for example, `mindie-alice` on
every host). Business file work, builds, services, and tests run in that
container; host operations are limited to device/port coordination and container
maintenance.

## Ownership Model

| Object | Owns | Lifetime |
| --- | --- | --- |
| Local task | Mutable source defaults, execution references, and evidence | Independent of remote execution |
| User container | Stable container identity and SSH endpoint on one host | Independent of tasks |
| Prepared runtime | One execution's work root and selected Python/CANN/build inside the user container | Evidence retained; compatible dependency/native artifacts may be shared |
| Runtime binding | One execution's use of that prepared root | Checkout through return |
| Execution | Fixed source inputs, managed process family, requested devices, and service-port use | Admission through confirmed cleanup |

A container's existence and device mounts do not reserve NPUs. The host
coordinator is the sole NPU and host-port authority. SSH endpoint reservations
belong to the user container and survive task completion; service-port use and
NPU leases belong to executions. Different prepared roots in the same user
container may have concurrent bindings and executions as long as their NPU and
service-port leases remain disjoint.

## Environment Selection

Each launch records an explicit worktree, Python executable, CANN installation
directory, build artifacts, environment variables, and owned output/cache
directories. A new process receives this environment before importing native
libraries, without mutating global shell configuration or shared symlinks:

- `vllm` and `vllm-ascend` use separate source/build locations.
- Python environments can hold different compatible `torch` / `torch_npu` pairs,
  selected explicitly via `MINDIE_PYTHON`.
- CANN user-space versions live in versioned installation directories; toolkit,
  operator packages, and dynamic-library paths are selected together as one
  launch environment.
- Build caches are reused only for matching inputs; logs and mutable run outputs
  remain execution-owned.

## Lifecycle Operations

1. **Set defaults (`mindie_session` / Hooks)**: Native session hooks associate the
   local task automatically and discover default source worktrees from the
   client's actual `cwd` (including linked Git worktrees and registered
   submodules). Explicit task source defaults override automatic sources for
   future submissions.
2. **Submit (`mindie_run` / `TaskClient.run`)**: Resolves sources from the call or
   effective defaults and pins their Git content and SCM provenance once before
   durable admission. Source-free commands require no vLLM trees, and CPU commands
   default to `npu_count=0`.
3. **Prepare**: Creates an isolated execution work root in the user's persistent
   container and reuses compatible dependency/native artifacts without holding a
   running NPU lease.
4. **Execute or Serve**: Persists the request and process identity, obtains the
   host NPU lease and service-port allocation, and launches with the selected
   environment (`MINDIE_EXECUTION_OBSERVATION`). Named services (`service="model"`)
   verify identical inputs and require `restart=True` plus confirmed cleanup
   before replacement.
5. **Stop or Release**: Terminates only the execution's managed process family,
   verifies that descendants have drained and service ports are closed, and then
   releases NPU leases while preserving the container and prepared environments.
   Executions admitted with `resources.allow_external_busy=True` on one explicit
   physical device leave external NPU processes untouched and release their own
   lease once their owned process family and ports are clear.
6. **Close Task (`mindie_finish` / `TaskClient.finish`)**: Tasks that never
   admitted managed work close directly in the local registry. For tasks with
   managed executions, `finish` stops admitted executions, returns remaining
   bindings once drained, and marks the task finished.
7. **Failure & Reconciliation**: Unknown process/device states or cleanup failures
   retain ownership (`finishing`) until the daemon reconciles them—local PID death
   or age never fabricates release.

## References

- [Python virtual environments](https://docs.python.org/3/library/venv.html)
- [TorchNPU version matching](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)
- [CANN package environment variables](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha001/maintenref/envvar/envref_07_0003.html)
