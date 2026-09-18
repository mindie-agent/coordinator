# coordinator · MindIE Agent

Local-process coordinator for one user's remote Ascend containers and host NPU
allocation. It is not a hosted multi-user service.

Consumer agents access it on the local machine; it operates the user's remote
containers through `remote-dev`. Code identity is git. The API owns runtime
preparation, lifecycle transitions and resource release; callers provide the
business command and constraints.

## Quick start

Use the existing `mindie_run` / `mindie_execution` MCP tools when available.
Python callers can use the same task API below. The complete short example
and its semantics are also available in one local command:
`python -m mindie_coordinator.mindie --help` (or `help(TaskClient)` in Python).

```python
from mindie_coordinator.task_client import TaskClient

client = TaskClient("/local/task-context.json")
run = client.run(
    '"$MINDIE_PYTHON" -c "import torch_npu; print(1)"',
    sources={"vllm": "/local/vllm", "vllm-ascend": "/local/vllm-ascend"},
    resources={"devices": [0]},
    topology={"host": "npu-host"},
    wait_until="released", wait_timeout_seconds=180,
)
execution_id = run["execution_id"]
print(run["state"], run["resources_released"], run.get("stdout", ""))
```

A local shell file can be submitted directly, without a Python runner:

```bash
mindie run --script-file business.sh --source app=/local/app --wait released --wait-timeout-seconds 180
mindie execution --execution-id EXECUTION_ID --wait released --wait-timeout-seconds 180
mindie execution --execution-id EXECUTION_ID --action evidence --section build --path kernel_name
```

`script_file` and `command` are mutually exclusive. The UTF-8 shell file is
read once (up to 1 MiB); a BOM is removed and CRLF becomes LF for Bash. Original
bytes/digest and the submitted command digest are recorded; the local filename does not
change service identity. MCP accepts the same `script_file`, `wait_until` and
`wait_timeout_seconds` run arguments. For an existing execution, MCP/Python use
`action="wait"`, `until="released"` and `timeout_seconds=180`.

Replace the paths and host with your task's inputs. Use the native attachment's
supplied `context_file`, or `TaskClient()` to resolve `MINDIE_CONTEXT_FILE` / the
actual native task context. No separate session or attach call is needed.
`sources` captures edits from local Git worktrees; their names become remote
subdirectories on `PYTHONPATH`. The command runs in that execution root and
`MINDIE_PYTHON` selects its prepared interpreter. Omitted sources use task defaults;
`sources={}` runs without source dependencies. Resources default to no NPU;
use `npu_count` for available devices or `devices` with `topology.host` for
specific physical devices. Explicit sharing of one physical NPU adds
`allow_external_busy=True` to resources; other managed leases still conflict.

`run` returns after admission unless `wait_until` is supplied. A released result confirms termination and
resource release; business success also requires `state == "succeeded"`.
If a bounded wait returns `wait_timed_out=True`, inspect its facts and wait on
the same id again as needed; that timeout does not stop or resubmit the work.
Wait budgets are 0–600 seconds and are separate from the command's execution
timeout. The owner handles waiting; no Agent status loop is required. Terminal
wait returns `stdout` / `stderr` / `tail`, collected once and cached on the execution.
A slow log read can return `logs_pending=True`; a failed read returns `tail_error`.
Neither changes the business state or release facts. Waiting again on the same
reference reuses the collection. A terminal exchange's explicit stdout/stderr may
also satisfy tail for that same verified drained and released job; incomplete,
status-only or different-job observations still require a log read.
If observation fails after admission, `wait_error` retains the admitted execution
reference; continue observing that reference instead of submitting again.
For progress use
`observe(execution_id, refresh=False)`; to stop it use `action="stop"` and wait
for release. `finish()` closes the entire task and stops its owned executions;
completed runs release their resources without a per-run finish call.

## Agent references and recorded launch facts

`TaskClient()` uses an explicit context or `MINDIE_CONTEXT_FILE` first. Local
Codex commands can also resolve their actual `CODEX_THREAD_ID`, creating or
resuming its native attachment locally when the hook did not export a context.
Conflicting native identities require an explicit context. This does not bind
sources, discover machines, or allocate devices; ordinary code review needs no
task client.

Native prompt hooks refresh a changed cwd and configured user silently on
clients with task-tool context injection. SessionStart and subagent attachment
still supply the initial context; Kimi clients without native per-call metadata
receive context through their current prompt hook. Ordinary PreToolUse events
return before Git scope checks or opening the task registry.

`client.finish()` closes a task that has never admitted managed work directly
in the local registry, without starting or importing the coordinator service
or remote-dev. Local close and execution admission share one database write
transaction boundary: if close wins, submission is rejected; if admission wins,
finish delegates to the coordinator for owned-execution cleanup.

A configured existing user container can prepare a task without selecting an
image recipe again. Creating a container requires an explicit recipe or a
concrete image tag/digest, supplied in the same run's `environment.image`.
No separate provisioning call is required for a first run at a fixed version.
Preparation checks source/image build compatibility before installing vLLM.
A completed failing preparation command ends that execution with its diagnostic
log; an unavailable SSH transport remains uncertain. Correct the configuration
and submit a new execution instead of repeating the failed preparation forever.

`mindie_execution` accepts exactly one `execution_id` or task-scoped `service`.
The Python equivalent is `client.observe(service="model", action="status")`.
An absent service returns `state: not_found` without contacting a runtime.
An ambiguous live service requires an execution reference. Lookup never joins
another task or allocates resources.

Library workflows can use `client.wait(execution_id, until="running")` or
`until="released"`, with `timeout_seconds` between 0 and 600. A timeout returns the
last observation with `wait_timed_out: true`; a release wait requires confirmed
termination and resource release. Changing bound business worktree paths
changes defaults for future submissions. It does not affect active executions
or require their bindings to be returned.

`observe(execution_id, action="evidence")` reads retained source snapshots,
preparation stages and native profile receipts. Select `section="sources"`,
`"preparation"` or `"build"` when useful; `path` filters artifact path substrings.
The coordinator decodes existing compressed receipts and returns readable facts,
hashes and original references. Missing or mismatching receipts are explicit.
This is optional diagnosis of recorded execution facts, not a new business
validation step or a live check of mutable remote files.

`client.run(command, sources={"app": "/actual/worktree"})` captures fixed Git
content and SCM provenance before admission. Omitted `sources` uses explicit
task defaults, or this native attachment's automatic cwd binding when no
explicit defaults were set; `sources={}` runs without source dependencies.
Defaults are replaced by `client.sources(mapping)`, including `{}` to clear
them. All roles consume one accepted snapshot; later edits affect only later
submissions. Replies expose `source_snapshot_id` and the selected source map.
Capture pins Git objects without moving HEAD or changing the user's index.

Resources default to `npu_count=0`. A CPU command or compiler therefore reserves
no NPU; declare a positive `npu_count` or specific `devices` for NPU work. Each
execution has its own work directory, while compatible prepared artifacts can
be reused. Independent host preparations run concurrently, bounded to four
workers and one active preparation per host.

To share one explicitly selected physical NPU with existing external workers,
pass `resources={"devices": [0], "allow_external_busy": True}` to `client.run`
or `mindie_run`. Use `topology={"host": "selected-host"}` to bind the host too.
The option is fixed at admission and requires exactly one explicit device;
omitting it keeps the normal occupancy checks. It permits observed external
process/HBM use, without estimating or reserving free memory. Other coordinator
leases and holds still conflict, and unknown or missing hardware is not usable.
The managed supervisor must retain its own process guard until completion.
Stop/finish only terminates this execution's family and releases its lease once
that family has drained and its ports are clear; existing workers remain running.
The assignment, launch observation and execution target expose the sharing flag.

`run(..., service="model")` ensures identical fixed sources and configuration.
Changed inputs report the differing fields; `restart=True` replaces the service
only after the old execution has stopped and released resources. Connecting
with `observe(service="model")` does not capture the current worktree.

Managed launch injects `MINDIE_EXECUTION_OBSERVATION` and retains the same receipt
as `target.launch_observation`. It records the source snapshots, attested
environment/native build identity, physical host, allocated devices and actual
command. User environment values are represented by a digest. The variable is
reserved and cannot be supplied by callers. The receipt survives termination
and later binding refreshes; it proves what was verified at launch, not a later
inspection of runtime mutations. Workload collectors add their actual model,
topology and input parameters; missing facts remain unknown.

## Status observations

`mindie-coordinator runtime-register` sends verification and registration to the
running coordinator, which owns all catalog writes. Register an existing native
artifact donor with `--reuse-only --source vllm=PATH --source vllm-ascend=PATH`;
the source inputs are fixed before verification and the donor work root cannot
be checked out for execution. Library clients can submit the same explicit
specification with `CoordinatorClient.runtime_register(runtime_id, spec)`.

Task MCP/CLI execution status returns the latest persisted managed-job snapshot
immediately. A sample older than two seconds schedules background progression.
`mindie execution --refresh` (or tool argument `refresh: true`) requests
a new status observation. Replies include `observation_freshness` with snapshot
completion time, age, freshness, source and whether a busy execution deferred
refresh. Per-role `status_observed_at` preserves individual sampling times;
roles are sampled concurrently with at most four workers. The top-level `observed_at` is the
response-generation time, not proof of a new remote query.

`TaskClient.observe()` preserves its fresh-by-default library behavior; pass
`refresh=False` to read the nonblocking status cache. `TaskClient.wait()` uses one
owner RPC with condition notifications, with no execution lock held while waiting.
MCP wait and control requests use separate worker pools, keeping stop and ping
responsive. Running jobs also use the existing remote owned-job bounded wait
on an independent observer; a terminal quiet receipt immediately advances fenced
release without waiting for the next two-second supervision tick. Lease renewal
and lost-wait recovery keep that normal supervision cadence. Waiters hold no
control locks and cannot replay commands. Tail, target, stop, resource
allocation and background progression retain their existing behavior. Cached
observations neither allocate resources nor establish new ownership. A busy
execution returns its stored observation immediately and explicitly marks a
requested refresh as deferred. Cached stale or missing timestamps schedule a
refresh on the existing execution worker; state/error/release fields remain visible
and the reply reports its actual sample age rather than claiming fresh remote facts.

Recorded preparation evidence includes available `preparation_timings` in seconds.
For a combined source/native job, `materialize` measures fixed Git materialization,
`native_publication` measures verification/copy/publication, and `source_metadata`
is a subphase of native publication, not an additional duration to sum. These
remote monotonic durations exclude transport and the surrounding owned-job setup;
the existing owner stage duration includes those costs.

## Install

```bash
uv pip install git+https://github.com/mindie-agent/coordinator@main
```

Or run without a permanent install:

```bash
uvx --from git+https://github.com/mindie-agent/coordinator@main mindie-coordinator task-server
```

Replace `@main` with a commit or tag when you pin. `python -m mindie_coordinator`
is the same entry as `mindie-coordinator`.

The package depends on `remote-dev>=0.7.0` (import `remote_dev`). It
does not pin that package's git source; the workspace that installs this
library chooses the tag. `uv sync` / `uv lock` are not the developer path
here: a library that named remote-dev's git source in `pyproject.toml`
would pin every consumer to that tag.

## Start the task server

`mindie-coordinator task-server` serves task and coordination tools over stdio MCP:

`mindie_session`, `mindie_run`, `mindie_execution`, `mindie_finish`, `mindie_message`.

For a substantive coordination request, pass a `coordination_peers[].reference`
returned while waiting and the message text to `mindie_message`. Reply with the
received `notifications[].reply_reference`. The task supplies the sender;
there are no owner, host endpoint, thread, cursor or ACK parameters to fill in.
Messages never execute commands, release leases or stop another user's task.

Normal run/status calls return locally cached notifications and trigger at most
one short polling worker for the task's already known hosts, with a five-second
minimum interval after each completed poll. They do not wait for remote inbox
I/O or scan the fleet. The existing local coordinator owns the worker; there
is no new daemon, SSE service or background Agent wakeup. A later normal call
delivers newly fetched text. A task without managed hosts remains local.

The host stores messages alongside the existing queue database, so independent
local coordinator/session stores can exchange offline messages. The queue's
default `/tmp` state can disappear on host reset; this is not reboot-durable
storage. Local `mail-message` records preserve fetched text before advancing
the host cursor. Automatic delivery records that the local coordinator returned
the text, not that the native client received it or the Agent accepted it;
there is no exactly-once end-to-end or manual acknowledgement claim. Peer
timestamps are contact observations, never evidence that resources are free.
Custom backends opt in with `supports_task_messages = True`; older backends
receive no new host actions. Embedded service owners call `close_messages()`
before removing their state; the daemon closes its workers on shutdown.

Example Cursor / Claude `.mcp.json` (or `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "mindie-coordinator": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/mindie-agent/coordinator@main",
        "mindie-coordinator",
        "task-server"
      ]
    }
  }
}
```

Start `mindie-coordinator daemon` for this user/state-dir. `task-server`, the
`mindie` CLI, and `TaskClient` call that process. They do not each create a
private scheduler.

`mindie_session` exposes the local task identity. `mindie_finish` closes admission
and stops the task's owned executions; the daemon then returns remaining
bindings and marks the task finished without a second finish.
`mindie_run` admits a business command plus environment/resource/topology needs;
the daemon places, prepares, launches and observes it. Pass the `context_file` supplied by the
native session hook; never guess a task from cwd or history. Do not pass
request IDs, profile hashes, or runtime IDs.

Managed launches prepend their selected source directories to Python's import
path. This prevents repository directories
in the task cwd from shadowing editable packages, while preserving the CANN
and other support paths already supplied by the environment.

The first native environment receives a full framework import smoke. A fresh Python source view can reuse that original evidence when
its dependency/native inputs, loader environment and complete artifact hashes
match; preparation checks the current module and SCM metadata mappings without
importing the changed business code. Its receipt explicitly records
`python_import_executed=false`, keeps the original import result, and makes no
claim that the new Python source passed. The business execution reports its own
result. Incomplete old evidence or cwd-dependent loader paths retain the full
import check.

For a compatible native environment, source materialization is followed by one
native-view publication: outputs are copied and checked against the existing
hashes, source/SCM mappings are updated, and the original import proof is carried
forward. Its completed receipt goes directly into an atomic managed binding.
There is no second profile capture, full registration probe, or SSH reservation.
Fresh native builds and shared/incremental restores also hand their completed
capture directly to managed registration. A shared restore can reuse import
evidence only when captured non-compiler source/resources, resolved dependency
versions and locations, CPU libraries, generated build metadata and loader paths
match. Device kernel outputs and dispatch are still validated after compilation;
the complete bundle is still hashed. Missing older import-closure evidence takes
the normal import path. The capture verifies its environment and atomically writes
the marker; its compressed reply retains every file identity. The handoff checks
the exact execution root, interpreter, source identity and observed image and
container, and waits for any owned cache store and cancellation check. Launch
still verifies current container, environment, source mapping and pinned Git
inputs before admission.
Before launch, coordinator checks the container, environment version facts,
current source mappings and fixed Git inputs; it does not rehash all native
outputs in its private execution view. Initial builds, changed native or
dependency inputs, and explicit adoption/repair retain complete verification.
A lost, failed or cancelled publication never publishes a successful binding.

For serving, `service_port=0` asks the host coordinator to select a free port.
If a task runtime has no declared service ports, automatic selection uses the
host's default serving range (30000–45999). A nonempty declaration restricts
selection to those ports; an explicit port must be declared. Both paths check
listening sockets and existing leases, and release ports with the execution.

## Host NPU queue (Python API)

The scaffold imports the public allocation surface from one place:

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

`host/mindie_npu_coordination.py` stays stdlib-only. It is shipped over SSH and
executed on the physical host. Durable host state defaults to
`/tmp/mindie-npu-coordinator/v1/` and is overridden by
`MINDIE_NPU_COORDINATOR_STATE_DIR` or `request["state_dir"]`.
`MINDIE_HOST_QUEUE_MODULE` overrides the bundled file.

## Layout

| Path | Role |
| --- | --- |
| `mindie_coordinator/cli.py` | `mindie-coordinator` / `python -m mindie_coordinator` |
| `mindie_coordinator/task_server.py` | Stdio MCP for task and coordination tools |
| `mindie_coordinator/host_queue.py` | Public host NPU API |
| `mindie_coordinator/host/` | Host allocation module, shipped to the host |
| `mindie_coordinator/backend.py` | Container/host probes via `remote_dev` |
| `mindie_coordinator/prepare_runtime.py` | In-container attest / publish / restore |
| `mindie_coordinator/workers/` | Linux supervisor source, shipped into a container |
| `mindie_coordinator/run_manifest.py` | Run Manifest v1 (Git identity in `code`) |
| `mindie_coordinator/code_identity.py` | Git snapshot identity for manifests |
| `mindie_coordinator/parity.py` | Working-tree snapshot and remote materialization |
| `mindie_coordinator/machine_directory.py` | Coordinator-owned machine directory |

## Development

`uv sync` is not the setup path. This library declares
`remote-dev>=0.7.0` without a git source: remote-dev is not on PyPI,
so `uv sync` / `uv lock` fail with an unsatisfiable-dependency error.
That is intentional. A library that pinned remote-dev's git URL would
take the upgrade decision away from every consumer, and
`constraint-dependencies` cannot carry a git URL.

Install the git source first, then this tree without resolving
dependencies from an index:

```bash
uv venv
uv pip install "remote-dev @ git+https://github.com/mindie-agent/remote-dev@13301ef7f52b53ffca0a6702a8a3c18f2edfcd52"
uv pip install pytest "jsonschema>=4" "setuptools-scm>=8"
uv pip install -e . --no-deps
.venv/bin/python -m pytest
```

`uv venv` is the first command so an unreadable project config fails
before install. Do not add `[tool.uv.sources]` for remote-dev: that table
travels to consumers. Requires Python 3.11+.

On native Windows, run the same setup commands and use
`.venv/Scripts/python.exe -m pytest tests`. The local daemon uses a locked state
directory and token-authenticated IPv4 loopback IPC; its listener is never bound
to an external interface. Native CLI pipes and Git output use UTF-8. Source
publication preserves Linux path syntax independently of the client platform.
Git snapshot commands enable long-path support for their own Windows invocation,
including nested submodule refs, without changing repository or global Git config.
The tests that emulate a Linux peer need a working Bash. If the Windows `bash`
alias points at an unconfigured WSL installation, prepend Git for Windows'
`bin` directory to the test process's `PATH`, as the Windows CI job does.

## Progress, records, and loaded versions

Execution observations include the active preparation/sync/preflight step,
its timestamps and log reference. Installation heartbeat events reach status
while compilation is in progress. Full install logs remain in the task root;
role errors, lease state and descendant quietness remain visible after failure.
`resources_released` is separate from execution state.
Long install, native reuse and profile verification commands have persisted
remote-dev job references and receipts. Stop interrupts their owned process
families and requires verified quiet before reporting cancellation or release;
an unknown transport or ownership outcome remains uncertain. A restarted daemon
can observe and stop these retained jobs without replaying preparation or
replacing sources beneath a compiler. Queued preparation can cancel while
another execution holds the host's compiler budget. Private source preparation
and cache reads proceed concurrently; one compiler per host retains the recipe's
existing internal parallelism. Shared filesystem writes use their narrow locks
or atomic publication. Slow standalone pool jobs dispatch independently. Tail includes the current
local preparation log. Fixed source materialization runs as an owned preparation
job with bounded upload/command deadlines and cancellation support.
Completed build-compatibility failures end preparation before editable installs
and do not retry automatically. Completed source-sync failures retain their
original cause; lost transport remains uncertain. Remote profile paths are
validated independently of the client operating system.
If a successful editable build removed its CMake cache and the image did not
export SoC/compiler variables, attestation reads those build selections from
the latest completed installer log and hashes that log as profile evidence.
Incomplete or conflicting evidence stays an error.

Managed source materialization consumes the admitted Git snapshot directly in
one remote operation. Existing immutable mirror objects are reused; a completed
missing-object response uploads only those objects before a new owned job.
Small edits use a bounded Git pack in that next owned job over the existing
RPC connection. The complete command, including all encoded packs, is capped
below the remote worker's argument limit; cold or larger transfers retain Git
SSH. Pack contents, prerequisite commit and resulting tree are verified before
atomically publishing snapshot refs or materializing the execution view.
Container provisioning checks SSH and Python package metadata without allocating
an NPU. Native import validation belongs to runtime preparation; device execution
belongs to a resource-backed run. The explicit standalone smoke command retains
its device test.
Each execution retains independent working files, a stable per-root lock, and
final HEAD and dirty-state checks across parent repositories and submodules.
Uncertain jobs are observed, never replayed. Runtime compatibility, native
build checks and resource allocation still run. Host coordination uses the
remote-dev Python RPC code cache, sending only the request after the first call.

The first owned preparation job starts from the existing container workspace
and creates its execution root and venv before materialization. Large composed
programs use a digest-checked compressed bootstrap within the command budget, so
embedded package code does not force a separate publication job. Uncompressible
inputs retain a bounded transfer fallback. Preparation job evidence records the
combined local elapsed time and individual remote setup durations; these use
separate clock domains and must not be subtracted from each other. Owned jobs
wait for exit within their bounded yield rather than returning early on output. First admission persists
the observed host epoch before submitting and acquiring in one host exchange.
New managed runs read their exact task and container facts together, verify the
compact source/environment view, then reuse a newly issued grant's occupancy
sample for host preflight. Queue recovery verifies again; facts are not cached
across queue waits. Explicit shared-device admission queries the current physical
device mapping; external occupancy stays unknown. Strict leases, unsupported
device queries and conflicting expired reservations retain full occupancy
probes. Prepared activation uses the exact container, boot, PID, start-time and
process marker proof without a second scan of every host process.
Startup and release operations record monotonic durations
in the existing run events, without command contents or heartbeat log entries.
Completed managed shared leases retain the host process-guard and port checks;
they need no whole-device visibility scan to release their own ownership.

Task MCP and `python -m mindie_coordinator.mindie` return compact observations by
default, with one local `record_ref` to the full response. MCP text is a summary;
structuredContent holds the observation. Pass `full: true` / `--full` for the
full response. Python TaskClient continues to return complete records. CLI
success and error output are each a single result object, without a text/result
wrapper. Explicit target requests retain launch data exactly or point to the
full record if oversized; truncated shell setup is never returned as executable.

`mindie-coordinator daemon --action status` reads loaded and installed package
identities without starting a daemon. `--action restart-if-idle` asks the daemon
to reject restart while work or unreleased leases remain; after an idle exit it
starts the installed version. MCP tools report their own process identity and
require a native-client MCP restart when stale. Missing commit metadata is unknown.
Compact runtime observations retain each package's commit and the actual Python
interpreter; differing client/daemon identities keep their full scope evidence.
`current` compares loaded code with that process's installation, not upstream.
Knowledge runs in a separate component and is not covered by these identities.
Managed admission requires the daemon to match the caller's coordinator and
remote-dev code identities, including commit and interpreter. A different idle
daemon restarts automatically using the caller's interpreter, even when the
version number is unchanged. Busy daemons keep their existing executions;
new admission returns `needs_runtime_update` with selected/loaded identities
and available active execution references. Status, tail and stop continue
through the existing owner. No new execution is queued by this response.

A run (or each topology role) may supply a `preflight` shell command to validate
the prepared environment before any NPU lease is allocated. It uses the selected
interpreter and a placeholder service port of zero, and must not require devices
or start a service. Failure retains original stderr references and does not launch
the business command. Planned Run Manifests can record early failure/inconclusive
outcomes without inventing a running stage.

Managed runs validate fixed inputs and binding/resource parameters before queueing;
they verify the complete remote environment and source view after the grant,
before preparing or authorizing the business command. A failed verification retains
its error and drains the owned job before returning resources. Unknown process or
host state keeps cleanup pending. Standalone `RuntimePool.request_run` also retains
its remote verification before queueing.


## Diagnostics

Tool replies include `diagnostics` with an operation ID, trace ID, actual UTC
start/end, monotonic duration, bounded phase summaries and a local `record_ref`.
The operation ID exists before work starts. A run admission's duration describes
that call; its asynchronous execution and later wait have separate lifetimes.
Native MCP metadata propagates correlation automatically. Diagnostic IDs never
grant task or endpoint authority and are excluded from launch-content identity.

`MINDIE_LOG_LEVEL=INFO` is the default. `DEBUG` adds RPC, lock and transport phase
detail; `WARN`/`WARNING` retains warnings and errors. `MINDIE_DIAGNOSTICS_ROOT`
selects a private local log root. The shared diagnostics package rotates bounded
per-process JSONL files and records package versions. Commands, file contents,
environment values and credentials are not logging arguments. Business output
remains in existing job logs and tool output, separate from implementation logs.
MCP reserves its protocol descriptors: Python and native stdout writes go to
stderr, and child processes cannot inherit protocol stdin accidentally.

`error_details` preserves category, retryability and submission certainty.
`not_sent` / `not_executed` indicate a known pre-execution boundary;
`uncertain` requires observing the existing job/execution reference, without
resubmission. `acknowledged` means the remote request was accepted, not that its
business succeeded. Only an explicit `caller` category denotes known caller
input error; a generic validation or permission exception is not that judgment.
Cancellation/observation timeout does not fabricate quiet or resource release.
Logging/export disk errors do not change business results or ownership; failures
to persist authoritative execution state still fail.

For a local issue attachment, the shared `mindie-diagnostics bundle` command makes
an offline, bounded public projection from diagnostic events. Use the returned
operation ID. It excludes raw commands, paths, endpoints and business logs, and
reports missing or truncated evidence. It never replays work or uploads an issue
by itself. Monotonic clocks are process-local: do not subtract remote/local UTC
stamps or sum overlapping RPC, command and parallel-role phase durations.

For an owned execution, `python -m mindie_coordinator.mindie execution
--execution-id ID --action evidence --section diagnostics` returns its retained
pool operations, stage timing and a redacted `support_bundle`. This path reads
only the selected execution's local records, never global host status, housekeeping,
other tasks' events or a remote directory. Source/preparation/build evidence
remains available through the existing evidence sections. Preparation preserves
launch/exchange costs; completed command receipts expose spawn, shell and
process-family drain timing. Historical records without clock/correlation data
report a gap instead of inventing an elapsed time.
