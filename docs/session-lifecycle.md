# Task lifecycle in a persistent user container

Status: current control-plane contract, 2026-09-12. The persistent daemon owns
fixed execution inputs, isolated execution roots, multi-role reservation and
confirmed cleanup. Actual NPU/recipe evidence still requires remote runs.

This is the contract for the breaking lifecycle change. Task completion
preserves the container and its prepared
environments. Each user has one persistent container on each host, named
`mindie-<user>` (for example, `mindie-alice` on every host). Business file work,
builds, services and tests run in that container. Host operations are limited
to device/port coordination and container maintenance. Reuse the existing
ready-runtime registry and managed executions.

## Ownership

| Object | Owns | Lifetime |
|---|---|---|
| Local task | Mutable source defaults, execution references and evidence | Independent of remote execution |
| User container | Stable container identity and SSH endpoint on one host | Independent of tasks |
| Prepared runtime | One execution's work root and selected Python/CANN/build inside the user container | Evidence retained; compatible dependency/native artifacts may be shared |
| Runtime binding | One execution's use of that prepared root | Checkout through return |
| Execution | Fixed source inputs, managed process family, requested devices and service-port use | Admission through confirmed cleanup |

A container's existence and device mounts do not reserve NPUs. The host
coordinator is the only NPU and host-port authority. SSH endpoint reservations
belong to the user container; they survive task completion. Service-port use and NPU
leases belong to executions. Port ownership must prevent overlap with runtime
SSH endpoints. The scaffold stores receipts and workflow inputs, not a second
allocation table.

The container is not exclusively checked out by one task. Different prepared
roots in the same user container may have concurrent bindings and executions;
their NPU and service-port leases remain disjoint. Keep exclusive use of a root
that source materialization or build refresh can modify. The existing runtime
rows can represent these prepared roots; no second container scheduler is needed.
Multiple rows may share one SSH endpoint only for the same host, user and actual
container identity, and must not independently allocate that SSH endpoint.

## Environment selection

Each launch records an explicit worktree, Python executable, CANN installation
directory, build artifacts, environment variables, and owned output/cache
directories. A new process receives this environment before importing native
libraries. Do not mutate global shell configuration, shared latest symlinks,
or an environment/build directory that an active execution uses.

- vLLM and vLLM-Ascend use separate source/build locations. Python source
  selection alone does not select or rebuild their native extensions.
- Python environments can contain different compatible torch/torch_npu pairs.
  Select the interpreter explicitly; isolate its packages from global installs.
- CANN user-space versions can live in versioned installation directories.
  Select the matching toolkit, operator packages and dynamic-library paths as
  one launch environment. Driver/firmware and kernel facilities remain host
  constraints and cannot be switched by a per-process environment variable.
- Prepared environments may be reused. Build caches are reused only for matching
  inputs; logs and mutable run outputs remain execution-owned.

Retain the existing profile/build verification and refresh entrypoints. A
different source tree or environment does not inherently require a new container.
Re-verify prepared artifacts before their next use. Runtime provisioning,
maintenance and explicit deletion are separate from task completion.

## Operations

**Set defaults.** Native session hooks automatically associate the local task;
Agents do not call a session-creation tool. Hook-discovered source
defaults belong to that native attachment, using its actual cwd. Resume or
handoff refreshes that attachment's cwd and automatic sources while preserving
the native task identity. Sibling attachments do not overwrite one another.
Claude also refreshes on the next pre-tool event with a native cwd, so entering
a worktree during one prompt does not leave later calls bound to its old cwd.
Explicit task source defaults override automatic sources, including an explicit
empty map. Changing either default affects future submissions only.

Scoped hooks recognize external Git linked worktrees through the actual common
Git directory, and existing registered submodules through Git's superproject
chain. A repository URL or directory name is not an identity proof. The owner
must be able to read the paths and Git metadata; a native client's independent
repository copy is not automatically associated as a linked worktree. These
hooks observe the client-selected workspace. They do not create a worktree or
change the parent client's cwd before its first tool call.

Supported pre-tool hooks inject the attachment context into MindIE MCP calls,
including `mindie_message` and provider-qualified `mindie-knowledge` / `remote-dev`
companion tools. Cursor's native MCP dispatcher resolves bare `MCP:knowledge_*`
and `MCP:remote_*` names through its configured providers; other clients require
the provider-qualified name. Client wiring owns workspace/component preparation and
consumption of this context; the hook reports its current source defaults.
Cursor's `sessionStart` runs asynchronously, so its
first MindIE `preToolUse` can attach the same stable conversation ID idempotently
before supplying `updated_input`. Other tool calls do not trigger this fallback.
See the [Cursor hook contract](https://cursor.com/docs/hooks#pretooluse).

`source_defaults` reports the effective map and its provenance. Saved task
maps from older versions without provenance are reported as unknown; the
coordinator does not guess whether they were explicit or automatic. A new
explicit task binding or per-run `sources` resolves that ambiguity.

**Submit.** Resolve sources from the current call or effective defaults and capture
their Git content and true SCM version once, before durable admission. Edits
during capture receive bounded retries; unstable input is not admitted. Each
role uses the same fixed descriptor. Source-free commands need no vLLM trees;
CPU commands default to zero NPUs. Status and connection lookups never capture.

**Prepare.** Create an execution work root in the user's fixed container and
reuse compatible dependency/native artifacts through the existing registry.
Never select an unrelated writable root and overwrite its source content.
Independent hosts prepare concurrently; preparation acquires no running NPU
lease. Successful preparation's fixed-source attestation is reused for launch,
without a second capture or materialization.

**Execute or serve.** Use the existing managed execution path. Persist the
request and process identity, obtain the host lease and service-port use, then
launch with the selected environment. Validate the current binding and fencing
information at admission. A long-running model service keeps its execution
lease for its entire process lifetime. All managed device use follows this
path; an interactive NPU command also needs an execution lease.

Named-service ensure compares fixed sources, command, environment, topology,
resources and preflight. Different inputs are reported explicitly; replacement
requires `restart=True` and confirmed old-process cleanup. An execution/service
reference connects to the original execution without rereading local sources.

**Stop or finish.** Stop only this execution's managed process family, including
children. Confirm it is terminal and its service ports are no longer listening
before releasing its resources. Strict leases also require their assigned
devices to be observable and free.
An execution admitted with `resources.allow_external_busy=true` on one explicit
physical device may leave external NPU users running. It releases only its own
lease after managed descendant completion and port checks; external occupancy
is neither owned work nor evidence that the managed process is still alive.
Its retained process guard remains authoritative after heartbeat loss. A grant
that never activated can expire or be cancelled even while external workers run.
Return and verify the prepared root for reuse through the existing pool path. Keep
the container, SSH endpoint, prepared environments, worktrees and evidence.
Thus stopping a model releases that model execution's NPUs; an idle local task
does not keep cards reserved.

**Close task.** A task that has never admitted managed work finishes directly
in the local registry without loading a coordinator service or remote runtime.
Its close check/state change and managed execution admission use the same
database write transaction boundary. If close commits first, admission fails;
if admission commits first, finish delegates to the coordinator. Admission
checks the task's open state and writes its complete admitted execution record
atomically, without an intermediate unadmitted row. For managed tasks, persist
the finish intent (owner/`force`) and stop/cancel admitted executions. The
daemon's existing tick/reconcile path returns remaining bindings and marks
the task finished once execution stop has drained. A frontend exit does not
require a second finish; the same pending finish continues after daemon
restart through the existing session-dir registry. Close must serialize with
launch: admitted work remains owned until cleanup completes, and a returned
binding cannot launch new work. Use the existing pool/run states and managed
supervisor; do not add a parallel session lease state machine.

**Failure and retry.** Unknown process/device state or cleanup failure keeps the
execution's ownership. A failed runtime verification leaves that prepared root unavailable for
checkout. An admitted finish continues on the daemon; do not require the
frontend to retry finish. Unknown or still-running processes keep the task
`finishing` until the existing return path can complete. GC reports unresolved
ownership; age, local PID death and missing local metadata
are not release evidence. The existing job identity is used to reconcile an
uncertain launch; a new explicit submission creates a new execution. No
container deletion is required to prove completion.

## Breaking cutover and deletion

Route managed task workflows through the existing pool/binding/execution APIs.
Remove the old per-session container allocation/release path, local leases.json
authority, its compatibility arguments and tests, and the unmerged candidate's
parallel durable session_reserved allocation path. Keep ordinary explicit
remote endpoints for remote-dev operations. Runtime maintenance can explicitly
remove a container after its bindings and executions are gone.

No old-record migration, auto-adoption, shared host database reset, or automatic
workload removal is part of this change. Update clients together. Retain one
CI execution per suite and distinct package installation checks.

## Minimum acceptance

1. Two successive tasks reuse the same container with their selected source and
   environment; the first leaves no managed workers, NPU lease or service port.
2. A live service retains its resources; stopping it releases them while the
   container and SSH endpoint remain available.
3. Wrong/stale ownership and close-versus-launch cannot mutate another run or
   release ahead of admitted work.
4. Unknown probes or failed cleanup retain ownership; an admitted finish
   completes on the daemon when the condition clears. A failed runtime check
   prevents reuse.
5. Existing host allocation prevents device/port conflicts. Environment switching
   does not overwrite an active environment or reuse mismatched native artifacts.

Adapt the existing focused tests and delete tests of removed behavior. Run one
full CI pass after integration. Actual NPU environment switching still needs
remote evidence; documentation and mocked tests do not establish ABI support.

## References

- [Python virtual environments](https://docs.python.org/3/library/venv.html)
- [TorchNPU version matching](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)
- [CANN package environment variables](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha001/maintenref/envvar/envref_07_0003.html)

The existing implementation anchors are ready_runtime.py (checkout,
return_runtime, refresh) and managed_execution.py (_finish_managed).
