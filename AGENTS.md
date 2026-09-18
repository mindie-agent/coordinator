# vaws-coordinator

Local-process coordinator for one user's remote Ascend containers and host
NPU allocation. It is not a hosted service. Code identity is Git.

## What this package owns

| Concern | Module |
| --- | --- |
| Run Manifest v1 | `vaws_coordinator.run_manifest` |
| Code identity | `vaws_coordinator.code_identity` |
| Working-tree → remote parity | `vaws_coordinator.parity` |
| Machine directory | `vaws_coordinator.machine_directory` |
| Host NPU queue | `vaws_coordinator.host_queue`, `vaws_coordinator.host` |
| Persistent coordinator | `vaws_coordinator.service` (`vaws-coordinator daemon`) |
| Task facade / stdio MCP | `vaws_coordinator.task_client`, `vaws_coordinator.task_server` |
| User-container provision | `vaws_coordinator.provision` |

A consumer passes data. This package does not locate a consumer tree by
path or environment variable.

## What this package must not do

- Reach back into the scaffold. No `VAWS_PARITY_SCRIPT`, no
  `VAWS_MACHINE_INVENTORY`, no file-path imports of `.agents/`.
- Construct SSH options. That belongs to `vaws-remote-dev`.
- Pin `vaws-remote-dev`'s git source in `pyproject.toml` or
  `[tool.uv.sources]`. The consumer chooses the tag.

## Developer setup

`uv sync` is not the path. `vaws-remote-dev` is not on PyPI, so a lock
that named its git URL would pin every consumer to that tag.

```bash
uv venv
uv pip install "vaws-remote-dev @ git+https://github.com/mindie-agent/remote-dev@89d197ef13bcae46f7bea809c22bbb058c4edfbf"
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
