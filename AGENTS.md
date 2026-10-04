# mindie-coordinator

Local-process coordinator for one user's remote Ascend containers and host
NPU allocation. It is not a hosted service. Code identity is Git.

## What this package owns

| Concern | Module |
| --- | --- |
| Run Manifest v1 | `mindie_coordinator.run_manifest` |
| Code identity | `mindie_coordinator.code_identity` |
| Working-tree → remote parity | `mindie_coordinator.parity` |
| Machine directory | `mindie_coordinator.machine_directory` |
| Host NPU queue | `mindie_coordinator.host_queue`, `mindie_coordinator.host` |
| Persistent coordinator | `mindie_coordinator.service` (`mindie-coordinator daemon`) |
| Task facade / stdio MCP | `mindie_coordinator.task_client`, `mindie_coordinator.task_server` |
| User-container provision | `mindie_coordinator.provision` |

A consumer passes data. This package does not locate a consumer tree by
path or environment variable.

## What this package must not do

- Reach back into the scaffold. No `MINDIE_PARITY_SCRIPT`, no
  `MINDIE_MACHINE_INVENTORY`, no file-path imports of `.agents/`.
- Construct SSH options. That belongs to `remote-dev`.
- Reimplement or override `remote-dev` transport policy in the coordinator.
  Package metadata pins the reviewed public runtime combination; changes to
  those pins require the same design-principles and installation review as code.

## Developer setup

Install the package and its declared public dependencies in one resolution, as
shown in README and CI. Use a dedicated virtual environment; do not alter the
user's active environment.

```bash
uv venv
uv pip install -e ".[test]"
.venv/bin/python -m pytest
```

Requires Python 3.11+.

## Tests

```bash
.venv/bin/python -m pytest
```

Component tests use isolated temporary state and do not require NPU, Docker,
`torch` or `torch_npu`. Hardware, real SSH and native host acceptance remain
separate evidence; passing component tests does not establish those results.
