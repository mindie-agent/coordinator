<h1 align="center">MindIE Agent Coordinator (<code>mindie-coordinator</code>)</h1>

<p align="center">
  <strong>Local-process control plane for remote Ascend user containers, Git source snapshot materialization, and host NPU queue allocation</strong>
</p>

<p align="center">
  <a href="./README.md">中文</a> · <strong>English</strong> · <a href="./docs/README.en.md">Documentation</a> · <a href="https://github.com/mindie-agent/mindie-agent">MindIE Agent Main Repo</a>
</p>

---

## Overview

`mindie-coordinator` is the local-process control plane of [MindIE Agent](https://github.com/mindie-agent/mindie-agent) (distribution `mindie-coordinator`, import package `mindie_coordinator`, console entries `mindie-coordinator` and `mindie`). It runs locally for a single user and drives remote Ascend containers and host NPU allocation queues over [`remote-dev`](https://github.com/mindie-agent/remote-dev).

- **Git-Backed Code Identity & Isolated Materialization**: Captures local Git worktree and submodule snapshots at admission without moving `HEAD` or altering the user's index, materializing isolated execution roots inside the user's persistent container (`mindie-<user>`).
- **Single Authority for Host NPUs & Service Ports**: Arbitrates physical NPU leases and service ports (`30000–45999`) via the stdlib-only `mindie_npu_coordination.py` host module, supporting both exclusive device grants and explicit single-device sharing (`allow_external_busy=True`).
- **End-to-End Managed Execution Lifecycle**: Owns runtime preparation, preflight checks, attested launches (`MINDIE_EXECUTION_OBSERVATION`), non-blocking status observations, process-family draining, and verified resource release.
- **Unified MCP, CLI & Python SDK**: Serves 5 stdio MCP tools (`mindie_session`, `mindie_run`, `mindie_execution`, `mindie_finish`, `mindie_message`) alongside the `mindie` CLI and Python `TaskClient`, all backed by a single per-user local daemon.

---

## Quick Start

### 1. Installation & Launch

Requires Python `>= 3.11`. `pyproject.toml` pins compatible revisions of `remote-dev` and `mindie-diagnostics`:

```bash
# Install into your environment
uv pip install git+https://github.com/mindie-agent/coordinator@main

# Or launch the stdio MCP task server directly via uvx
uvx --from git+https://github.com/mindie-agent/coordinator@main mindie-coordinator task-server
```

### 2. Client MCP Configuration

Example `.mcp.json` (Claude Code / Cursor):

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

### 3. Python `TaskClient` Example

```python
from mindie_coordinator.task_client import TaskClient

client = TaskClient("/local/task-context.json")
run = client.run(
    '"$MINDIE_PYTHON" -c "import torch_npu; print(1)"',
    sources={"vllm": "/local/vllm", "vllm-ascend": "/local/vllm-ascend"},
    resources={"devices": [0]},
    topology={"host": "npu-host"},
    wait_until="released",
    wait_timeout_seconds=180,
)
print(run["execution_id"], run["state"], run["resources_released"], run.get("stdout", ""))
```

### 4. CLI Submission & Observation

```bash
# Submit a local UTF-8 shell script directly and wait until resources are released
mindie run --script-file business.sh --source app=/local/app --wait released --wait-timeout-seconds 180

# Wait on or inspect an existing execution
mindie execution --execution-id EXECUTION_ID --wait released --wait-timeout-seconds 180

# Inspect recorded build, source, or diagnostic evidence
mindie execution --execution-id EXECUTION_ID --action evidence --section build --path kernel_name
```

---

## MCP Tools Overview

| MCP Tool | CLI / Python Entry | Purpose |
| --- | --- | --- |
| `mindie_session` | `TaskClient.session()` | Inspect local task identity, effective source defaults, and binding provenance |
| `mindie_run` | `mindie run` / `TaskClient.run()` | Pin Git source snapshots, admit a command or script, prepare the runtime, acquire NPU leases, and launch |
| `mindie_execution` | `mindie execution` / `TaskClient.observe()` | Read cached or fresh execution status, tail logs, inspect `evidence` (`sources` / `preparation` / `build` / `diagnostics`), or stop work |
| `mindie_finish` | `TaskClient.finish()` | Close task admission, stop owned executions, and return all runtime bindings and leases |
| `mindie_message` | `TaskClient.message()` | Exchange offline coordination messages with peer tasks using `coordination_peers[].reference` or `reply_reference` |

---

## Documentation

| Topic | 中文 | English |
| --- | --- | --- |
| Documentation Index | [docs/README.md](./docs/README.md) | [docs/README.en.md](./docs/README.en.md) |
| Persistent User Container & Task Lifecycle | [docs/session-lifecycle.zh.md](./docs/session-lifecycle.zh.md) | [docs/session-lifecycle.md](./docs/session-lifecycle.md) |
| Tools, Runtime Preparation & Observability | [docs/configuration-and-tools.md](./docs/configuration-and-tools.md) | [docs/configuration-and-tools.en.md](./docs/configuration-and-tools.en.md) |

---

## Repository Layout

| Path | Role |
| --- | --- |
| `mindie_coordinator/cli.py` | `mindie-coordinator` / `python -m mindie_coordinator` CLI entry |
| `mindie_coordinator/mindie.py` | Task-facing `mindie` CLI entry |
| `mindie_coordinator/task_server.py` | Stdio MCP server exposing the 5 task and coordination tools |
| `mindie_coordinator/task_client.py` | Python `TaskClient` SDK |
| `mindie_coordinator/host_queue.py` | Public host NPU queue Python API |
| `mindie_coordinator/host/` | Stdlib-only host NPU & port allocation module shipped over SSH |
| `mindie_coordinator/prepare_runtime.py` | In-container runtime attestation, native artifact reuse, and publication |
| `mindie_coordinator/parity.py` | Working-tree Git snapshot capture and remote materialization |
| `mindie_coordinator/run_manifest.py` | Run Manifest v1 and Git code identity |
| `docs/` | Bilingual lifecycle contract and tool/runtime reference documentation |
| `tests/` | Unit test suite |

---

## Development & Testing

```bash
uv venv
uv pip install -e ".[test]"
.venv/bin/python -m pytest
```

---

## License

Released under the [MIT License](./LICENSE).
