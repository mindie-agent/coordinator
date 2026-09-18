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
- Pin `remote-dev`'s git source in `pyproject.toml` or
  `[tool.uv.sources]`. The consumer chooses the tag.

## Developer setup

`uv sync` is not the path. `remote-dev` is not on PyPI, so a lock
that named its git URL would pin every consumer to that tag.

```bash
uv venv
uv pip install "remote-dev @ git+https://github.com/mindie-agent/remote-dev@13301ef7f52b53ffca0a6702a8a3c18f2edfcd52"
uv pip install pytest "jsonschema>=4" "setuptools-scm>=8"
uv pip install -e . --no-deps
```

`uv venv` is first so an unreadable project config fails before install.
Requires Python 3.11+.

## Tests

```bash
HOME="$(mktemp -d)" .venv/bin/python -m pytest
```

No NPU, no Docker, no `torch` / `torch_npu`. Empty `HOME`.
