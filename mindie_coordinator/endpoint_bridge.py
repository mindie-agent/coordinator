"""Bridge coordinator's MachineDirectory (machines.json) to remote-dev endpoints.

[Role] Synchronizes `<coordinator_state_dir>/endpoints.json` with `machines.json` and registers a `remote-dev` alias resolver.
[Invariants] Disabled or unverified hosts are excluded from active alias resolution unless explicitly requested.
[Timeouts & Locks] Atomic temp-file write + `os.replace` for `endpoints.json`; zero network I/O.
[Side-Effect Policy] Writes `<coordinator_state_dir>/endpoints.json` and registers in-process `remote-dev` resolver callbacks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from remote_dev.core.endpoint import register_resolver, resolver_setup

from mindie_coordinator.state_paths import coordinator_state_dir

ENDPOINTS_FILENAME = "endpoints.json"


def endpoints_file_path(state_dir: Path | str | None = None) -> Path:
    base = Path(state_dir).expanduser().resolve() if state_dir is not None else coordinator_state_dir()
    return base / ENDPOINTS_FILENAME


def _host_endpoint_spec(row: Mapping[str, Any]) -> dict[str, Any] | None:
    host = row.get("host")
    if not isinstance(host, Mapping):
        return None
    ip = str(host.get("ip") or host.get("host") or "").strip()
    if not ip:
        return None
    port = int(host.get("port") or 22)
    user = str(host.get("user") or "root").strip() or "root"
    workdir = str(host.get("workdir") or "/").strip() or "/"
    spec: dict[str, Any] = {
        "host": ip,
        "port": port,
        "user": user,
        "root": workdir,
        "cwd": workdir,
    }
    if host.get("identity_file"):
        spec["identity_file"] = str(host["identity_file"])
    if host.get("auth_ref"):
        spec["auth_ref"] = str(host["auth_ref"])
    if host.get("known_hosts_file"):
        spec["known_hosts_file"] = str(host["known_hosts_file"])
    if host.get("python"):
        spec["python"] = str(host["python"])
    if host.get("runtime_env") is not None:
        spec["runtime_env"] = bool(host["runtime_env"])
    if host.get("runtime_env_file") or host.get("env_file"):
        spec["runtime_env_file"] = str(host.get("runtime_env_file") or host.get("env_file"))
    return spec


def _apply_container_runtime_env(spec: dict[str, Any], container: Mapping[str, Any]) -> None:
    if container.get("runtime_env") is not None:
        spec["runtime_env"] = bool(container["runtime_env"])
    env_file = container.get("runtime_env_file") or container.get("env_file")
    if env_file:
        spec["runtime_env_file"] = str(env_file)


def _primary_endpoint_spec(row: Mapping[str, Any]) -> dict[str, Any] | None:
    host_spec = _host_endpoint_spec(row)
    if host_spec is None:
        return None
    container = row.get("container")
    if not isinstance(container, Mapping) or not container:
        return dict(host_spec)

    c_name = str(container.get("name") or "").strip()
    c_id = str(container.get("id") or "").strip()
    c_port = container.get("ssh_port")
    c_user = str(container.get("user") or "root").strip() or "root"
    c_workdir = str(container.get("workdir") or "/vllm-workspace").strip() or "/vllm-workspace"
    c_mode = str(container.get("mode") or "").strip().lower()

    # Mode 1: Direct container SSH port (classic mindie-<user> container sshd)
    if c_port and int(c_port) > 0 and c_mode != "docker_exec":
        spec: dict[str, Any] = {
            "host": host_spec["host"],
            "port": int(c_port),
            "user": c_user,
            "root": c_workdir,
            "cwd": c_workdir,
        }
        # A published container SSH port is a distinct authentication scope.
        if container.get("identity_file"):
            spec["identity_file"] = str(container["identity_file"])
        if container.get("auth_ref"):
            spec["auth_ref"] = str(container["auth_ref"])
        if container.get("known_hosts_file"):
            spec["known_hosts_file"] = str(container["known_hosts_file"])
        if container.get("python"):
            spec["python"] = str(container["python"])
        _apply_container_runtime_env(spec, container)
        return spec

    # Mode 2: Host SSH + `docker exec -i <container>` (remote-dev container endpoint)
    if c_name or c_id:
        container_spec: dict[str, Any] = {"user": c_user, "workdir": c_workdir}
        if c_name:
            container_spec["name"] = c_name
        if c_id:
            container_spec["id"] = c_id
        spec = {
            **host_spec,
            "root": c_workdir,
            "cwd": c_workdir,
            "container": container_spec,
        }
        if container.get("python"):
            spec["python"] = str(container["python"])
        _apply_container_runtime_env(spec, container)
        return spec

    return dict(host_spec)


def _docker_exec_endpoint_spec(row: Mapping[str, Any]) -> dict[str, Any] | None:
    host_spec = _host_endpoint_spec(row)
    if host_spec is None:
        return None
    container = row.get("container")
    if not isinstance(container, Mapping):
        return None
    c_name = str(container.get("name") or "").strip()
    c_id = str(container.get("id") or "").strip()
    if not c_name and not c_id:
        return None
    c_user = str(container.get("user") or "root").strip() or "root"
    c_workdir = str(container.get("workdir") or "/vllm-workspace").strip() or "/vllm-workspace"
    container_spec: dict[str, Any] = {"user": c_user, "workdir": c_workdir}
    if c_name:
        container_spec["name"] = c_name
    if c_id:
        container_spec["id"] = c_id
    spec = {
        **host_spec,
        "root": c_workdir,
        "cwd": c_workdir,
        "container": container_spec,
    }
    if container.get("python"):
        spec["python"] = str(container["python"])
    _apply_container_runtime_env(spec, container)
    return spec


def build_endpoints_document(inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a `machines.json` inventory document into `remote-dev`'s `endpoints.json` schema."""
    machines = inventory.get("machines")
    if not isinstance(machines, list):
        machines = []

    endpoints: dict[str, dict[str, Any]] = {}
    first_alias: str | None = None
    for row in machines:
        if not isinstance(row, Mapping):
            continue
        host_spec = _host_endpoint_spec(row)
        primary_spec = _primary_endpoint_spec(row)
        if host_spec is None or primary_spec is None:
            continue
        alias = str(row.get("alias") or host_spec["host"]).strip()
        if not alias:
            continue
        if first_alias is None:
            first_alias = alias
        endpoints[alias] = primary_spec
        endpoints[f"{alias}-host"] = host_spec
        docker_spec = _docker_exec_endpoint_spec(row)
        if docker_spec is not None:
            endpoints[f"{alias}-docker"] = docker_spec

    configured_default = str(inventory.get("default_alias") or "").strip()
    default_alias: str | None = None
    if configured_default and configured_default in endpoints:
        default_alias = configured_default
    elif len(machines) == 1 and first_alias and first_alias in endpoints:
        default_alias = first_alias

    if default_alias and default_alias in endpoints:
        endpoints.setdefault("default", dict(endpoints[default_alias]))
        if f"{default_alias}-host" in endpoints:
            endpoints.setdefault("default-host", dict(endpoints[f"{default_alias}-host"]))

    doc: dict[str, Any] = {"endpoints": endpoints}
    if default_alias:
        doc["default_alias"] = default_alias
    return doc


def sync_endpoints_file(
    inventory: Mapping[str, Any],
    *,
    machines_path: Path | None = None,
    target_path: Path | None = None,
) -> Path | None:
    """Write `endpoints.json` alongside `machines.json` when backed by disk."""
    if target_path is None:
        if machines_path is None or str(machines_path) == "<memory>":
            return None
        target_path = machines_path.parent / ENDPOINTS_FILENAME
    doc = build_endpoints_document(inventory)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(target_path)
    return target_path


def resolve_coordinator_endpoint(
    payload: Mapping[str, Any],
    *,
    directory: Any | None = None,
) -> dict[str, Any] | None:
    """Resolve an endpoint request against coordinator's MachineDirectory."""
    if "host" in payload and "port" in payload:
        return None
    from mindie_coordinator.machine_directory import MachineDirectory, MachineDirectoryUnavailable

    store = directory if directory is not None else MachineDirectory()
    try:
        inventory, _ = store.load()
    except MachineDirectoryUnavailable:
        return None

    doc = build_endpoints_document(inventory)
    endpoints = doc.get("endpoints") or {}
    default_alias = doc.get("default_alias")

    alias = payload.get("alias") or payload.get("target_alias") or payload.get("machine")
    if isinstance(alias, str) and alias.strip():
        key = alias.strip()
        if key in endpoints:
            return _merge_overrides(endpoints[key], payload)
        # Also allow matching by host IP
        for row in inventory.get("machines") or []:
            if isinstance(row, Mapping) and (row.get("host") or {}).get("ip") == key:
                spec = _primary_endpoint_spec(row)
                if spec is not None:
                    return _merge_overrides(spec, payload)
        return None

    host_only = payload.get("host")
    if isinstance(host_only, str) and host_only.strip() and "port" not in payload:
        key = host_only.strip()
        if key in endpoints:
            return _merge_overrides(endpoints[key], payload)
        for row in inventory.get("machines") or []:
            if isinstance(row, Mapping) and (row.get("host") or {}).get("ip") == key:
                spec = _primary_endpoint_spec(row)
                if spec is not None:
                    return _merge_overrides(spec, payload)
        return None

    if not payload or (not alias and not host_only):
        if default_alias and default_alias in endpoints:
            return _merge_overrides(endpoints[default_alias], payload)

    return None


def _merge_overrides(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key in (
        "user",
        "root",
        "cwd",
        "python",
        "connect_timeout",
        "control_path",
        "strict_host_key_checking",
        "identity_file",
        "auth_ref",
        "known_hosts_file",
        "ssh_options",
        "ssh_mux",
        "runtime_env",
        "runtime_env_file",
        "container",
    ):
        if key in overrides and overrides[key] is not None:
            merged[key] = overrides[key]
    return merged


@resolver_setup
def setup_resolvers() -> None:
    """Register coordinator's MachineDirectory resolver with remote-dev."""
    from remote_dev.core.endpoint import unregister_resolver

    unregister_resolver("mindie_coordinator")
    register_resolver(
        resolve_coordinator_endpoint,
        name="mindie_coordinator",
        fields=("alias", "target_alias", "machine", "host"),
        description="Resolve MindIE coordinator machine aliases, host IPs, and default_alias from machines.json",
    )
