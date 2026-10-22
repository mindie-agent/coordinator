"""SSH host, container, hardware probe, NPU lease, and vllm-ascend env preflight.

[Role]
Implements the four direct environment management handlers (`handle_hosts`,
`handle_probe`, `handle_container`, `handle_lease`) backing `mindie.hosts`,
`mindie.probe`, `mindie.container`, and `mindie.lease`, plus deterministic
`validate_vllm_ascend_runtime_env` runtime environment conflict preflight.

[Invariants]
- `handle_hosts` and `handle_container` keep `<state_dir>/machines.json` and
  `<state_dir>/remote_dev_endpoints.json` synchronized atomically.
- `handle_lease` delegates card allocation and fencing to `HostQueue` and caches
  lease metadata (`fence_token`, `coordination_epoch`, `devices`) in `leases.json`.
- Container removal releases only that container owner's `container_ssh` row
  after confirmed removal/absence; failed or unknown removal keeps it.
- `validate_vllm_ascend_runtime_env` is a pure in-memory deterministic validator.

[Timeouts & Locks]
- Delegates remote execution to `fleet_probe`, `host_ops`, and `HostQueue`.
- `validate_vllm_ascend_runtime_env` performs zero I/O, subprocesses, or waits.

[Side-Effect Policy]
- Persists `machines.json`, `remote_dev_endpoints.json`, and `leases.json` via
  atomic `.tmp` + `replace()`.
"""

from __future__ import annotations

import getpass
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from remote_dev.core.errors import caller_error

from mindie_coordinator.endpoint_bridge import (
    build_endpoints_document,
    endpoints_file_path,
    sync_endpoints_file,
)
from mindie_coordinator.fleet_probe import (
    find_fleet_capacity,
    probe_machine_record,
)
from mindie_coordinator.host_queue import HostQueue
from mindie_coordinator.machine_directory import (
    MachineDirectory,
    MachineDirectoryUnavailable,
    MachineNotFoundError,
)
from mindie_coordinator.provision import host_ops, provision_user_container
from mindie_coordinator.ready_runtime import safe_id
from mindie_coordinator.state_paths import coordinator_state_dir
from mindie_coordinator.vllm_ascend_env import (
    _LOOPBACK_IFNAMES as _LOOPBACK_IFNAMES,
    validate_vllm_ascend_runtime_env as validate_vllm_ascend_runtime_env,
)


def _resolve_store(
    directory: MachineDirectory | None = None,
    state_dir: Path | str | None = None,
) -> tuple[MachineDirectory, Path]:
    base = Path(state_dir).expanduser().resolve() if state_dir is not None else coordinator_state_dir()
    if directory is not None:
        return directory, base
    return MachineDirectory(path=base / "machines.json"), base


def _resolve_host_target(
    args: Mapping[str, Any],
    store: MachineDirectory,
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    """Return `(alias, host_spec, full_record_or_None)` from tool arguments."""
    target = str(args.get("alias") or args.get("host") or "").strip()
    try:
        rec = store.record(target if target else None)
        host_info = rec.get("host") or {}
        ip = str(host_info.get("ip") or host_info.get("host") or "").strip()
        alias = str(rec.get("alias") or ip).strip()
        spec = {
            "host": ip,
            "port": int(args.get("port") or host_info.get("port") or 22),
            "user": str(args.get("user") or host_info.get("user") or "root").strip() or "root",
        }
        if args.get("identity_file") or host_info.get("identity_file"):
            spec["identity_file"] = str(args.get("identity_file") or host_info["identity_file"])
        return alias, spec, rec
    except (MachineDirectoryUnavailable, MachineNotFoundError):
        if not target:
            raise caller_error(
                "no machine registered or specified; pass `host` (or `alias`) or register a machine with mindie_hosts(action='upsert', ...)"
            )
        # Allow direct ad-hoc host IP/hostname when explicitly provided
        spec = {
            "host": target,
            "port": int(args.get("port") or 22),
            "user": str(args.get("user") or "root").strip() or "root",
        }
        if args.get("identity_file"):
            spec["identity_file"] = str(args["identity_file"])
        return target, spec, None


def handle_hosts(
    args: Mapping[str, Any],
    *,
    directory: MachineDirectory | None = None,
    state_dir: Path | str | None = None,
    runner: Callable[[Mapping[str, Any], str], str] | None = None,
) -> dict[str, Any]:
    """Execute `mindie.hosts` (`list`, `upsert`, `remove`, `set_default`, `bootstrap_key`, `mesh`)."""
    action = str(args.get("action") or "list").strip()
    store, base_dir = _resolve_store(directory, state_dir)

    if action == "list":
        try:
            inventory, inv_path = store.load()
        except MachineDirectoryUnavailable:
            inventory, inv_path = {"machines": []}, store.path()
        endpoints_doc, ep_path = build_endpoints_document(inventory), endpoints_file_path(base_dir)
        if inventory.get("machines") and str(inv_path) != "<memory>" and not ep_path.is_file():
            sync_endpoints_file(inventory, machines_path=inv_path, target_path=ep_path)
        return {
            "state": "ready", "action": "list", "inventory_path": str(inv_path), "endpoints_file": str(ep_path),
            "default_alias": endpoints_doc.get("default_alias"), "machines": inventory.get("machines", []),
            "remote_dev_endpoints": endpoints_doc.get("endpoints", {}),
        }

    if action in {"upsert", "add"}:
        host_ip = str(args.get("host") or "").strip()
        if not host_ip:
            raise caller_error("mindie_hosts(action='upsert') requires `host` (IP or hostname)")
        alias, port = str(args.get("alias") or host_ip).strip(), int(args.get("port") or 22)
        user = str(args.get("user") or "root").strip() or "root"
        machine_type = host_ops.normalize_machine_type(args.get("machine_type")) if args.get("machine_type") else None
        existing_rec: dict[str, Any] = {}
        for key in (alias, host_ip):
            try:
                existing_rec = store.record(key)
                break
            except (MachineDirectoryUnavailable, MachineNotFoundError):
                continue

        if args.get("bootstrap_key") or args.get("password_env"):
            password_env = args.get("password_env")
            if not password_env:
                raise caller_error("bootstrap_key requires `password_env` naming an environment variable that holds the SSH password")
            ns = host_ops._namespace(host=host_ip, user=user, host_port=port, public_key_file=args.get("public_key_file"), password=None, password_env=str(password_env), password_stdin=False, print_command=False)
            if host_ops.cmd_bootstrap_host_key(ns) != 0:
                raise RuntimeError(f"SSH key bootstrap failed for {user}@{host_ip}:{port}")

        host_entry: dict[str, Any] = {**(existing_rec.get("host") or {}), "ip": host_ip, "port": port, "user": user}
        if machine_type:
            host_entry["machine_type"] = machine_type
        if args.get("identity_file"):
            host_entry["identity_file"] = str(args["identity_file"])
        if args.get("workdir"):
            host_entry["workdir"] = str(args["workdir"])

        container_entry = dict(existing_rec.get("container") or {})
        if args.get("container_name") or args.get("container_ssh_port"):
            if args.get("container_name"):
                container_entry["name"] = str(args["container_name"]).strip()
            if args.get("container_ssh_port"):
                container_entry["ssh_port"] = int(args["container_ssh_port"])
            container_entry.setdefault("user", str(args.get("container_user") or "root"))
            container_entry.setdefault("workdir", str(args.get("container_workdir") or "/vllm-workspace"))
            if args.get("container_mode"):
                container_entry["mode"] = str(args["container_mode"])

        record: dict[str, Any] = {**existing_rec, "alias": alias, "host": host_entry}
        if container_entry:
            record["container"] = container_entry

        probe_summary: dict[str, Any] | None = None
        if args.get("probe_hardware"):
            snap = probe_machine_record(record, mode="live", state_dir=base_dir, runner=runner)
            probe_summary = {"online": snap.get("online"), "chip_summary": snap.get("chip_summary"), "total_npus": snap.get("total_npus"), "free_npus": snap.get("free_npus"), "driver_version": snap.get("driver_version"), "error": snap.get("error")}
            if snap.get("online"):
                record["hardware"] = {"chip_summary": snap.get("chip_summary"), "total_npus": snap.get("total_npus"), "driver_version": snap.get("driver_version")}

        inv_path = store.upsert_machine(record, set_default=bool(args.get("set_default")))
        endpoints_doc = store.endpoints_document()
        return {
            "state": "ready", "action": "upsert", "alias": alias, "record": record, "probe": probe_summary,
            "inventory_path": str(inv_path), "endpoints_file": str(endpoints_file_path(inv_path.parent)),
            "default_alias": endpoints_doc.get("default_alias"), "remote_dev_endpoint": (endpoints_doc.get("endpoints") or {}).get(alias),
        }

    if action == "remove":
        target = str(args.get("alias") or args.get("host") or "").strip()
        if not target:
            raise caller_error("mindie_hosts(action='remove') requires `alias` or `host`")
        try:
            inv_path, removed = store.remove_machine(target)
        except MachineDirectoryUnavailable as exc:
            raise caller_error(str(exc)) from exc
        endpoints_doc = store.endpoints_document()
        return {"state": "ready", "action": "remove", "target": target, "removed": removed, "inventory_path": str(inv_path), "default_alias": endpoints_doc.get("default_alias")}

    if action == "set_default":
        target = str(args.get("alias") or args.get("host") or "").strip()
        if not target:
            raise caller_error("mindie_hosts(action='set_default') requires `alias` or `host`")
        try:
            inv_path = store.set_default(target)
        except (MachineDirectoryUnavailable, ValueError) as exc:
            raise caller_error(str(exc)) from exc
        endpoints_doc = store.endpoints_document()
        return {"state": "ready", "action": "set_default", "default_alias": endpoints_doc.get("default_alias"), "inventory_path": str(inv_path), "remote_dev_endpoint": (endpoints_doc.get("endpoints") or {}).get("default")}

    if action == "bootstrap_key":
        alias, host_spec, _ = _resolve_host_target(args, store)
        password_env = args.get("password_env")
        if not password_env:
            raise caller_error("mindie_hosts(action='bootstrap_key') requires `password_env`")
        ns = host_ops._namespace(host=host_spec["host"], user=host_spec["user"], host_port=host_spec["port"], public_key_file=args.get("public_key_file"), password=None, password_env=str(password_env), password_stdin=False, print_command=False)
        if host_ops.cmd_bootstrap_host_key(ns) != 0:
            raise RuntimeError(f"SSH key bootstrap failed for {host_spec['user']}@{host_spec['host']}:{host_spec['port']}")
        return {"state": "ready", "action": "bootstrap_key", "alias": alias, "host": host_spec["host"], "port": host_spec["port"], "user": host_spec["user"]}

    if action == "mesh":
        alias_a, alias_b = str(args.get("alias") or args.get("host") or "").strip(), str(args.get("peer_alias") or args.get("peer_host") or "").strip()
        if not alias_a or not alias_b:
            raise caller_error("mindie_hosts(action='mesh') requires both `alias`/`host` and `peer_alias`/`peer_host`")
        rec_a, rec_b = store.record(alias_a), store.record(alias_b)
        port_a, port_b = int((rec_a.get("container") or {}).get("ssh_port") or args.get("port") or 0), int((rec_b.get("container") or {}).get("ssh_port") or args.get("peer_port") or 0)
        if port_a <= 0 or port_b <= 0:
            raise caller_error("SSH mesh requires both machines to have a container `ssh_port` configured")
        host_a, host_b = (rec_a.get("host") or {}).get("ip"), (rec_b.get("host") or {}).get("ip")
        target_a, target_b = host_ops.SshTarget(host=host_a, user="root", port=port_a), host_ops.SshTarget(host=host_b, user="root", port=port_b)
        comment_a, comment_b = f"mindie-mesh:{rec_a.get('alias') or host_a}", f"mindie-mesh:{rec_b.get('alias') or host_b}"
        key_a = host_ops.assert_remote_success(host_ops.run_remote_script(target_a, host_ops.render_mesh_export_key_script(), args=[comment_a], batch_mode=True))
        key_b = host_ops.assert_remote_success(host_ops.run_remote_script(target_b, host_ops.render_mesh_export_key_script(), args=[comment_b], batch_mode=True))
        host_ops.assert_remote_success(host_ops.run_remote_script(target_a, host_ops.render_mesh_add_peer_script(), args=[key_b["public_key"], host_b, str(port_b)], batch_mode=True))
        host_ops.assert_remote_success(host_ops.run_remote_script(target_b, host_ops.render_mesh_add_peer_script(), args=[key_a["public_key"], host_a, str(port_a)], batch_mode=True))
        return {"state": "ready", "action": "mesh", "pair": [{"alias": rec_a.get("alias"), "host": host_a, "ssh_port": port_a}, {"alias": rec_b.get("alias"), "host": host_b, "ssh_port": port_b}]}

    raise caller_error(f"unsupported mindie.hosts action: {action!r}")


def handle_probe(
    args: Mapping[str, Any],
    *,
    directory: MachineDirectory | None = None,
    state_dir: Path | str | None = None,
    runner: Callable[[Mapping[str, Any], str], str] | None = None,
) -> dict[str, Any]:
    """Execute `mindie.probe` (`hosts`, `capacity`, `npu`, `server`, `mounts`)."""
    action = str(args.get("action") or "hosts").strip()
    mode = str(args.get("mode") or "auto").strip()
    store, base_dir = _resolve_store(directory, state_dir)
    target = str(args.get("alias") or args.get("host") or "").strip()

    if action == "capacity":
        npu_count = int(args.get("npu_count") if args.get("npu_count") is not None else 1)
        if npu_count < 1:
            raise caller_error("npu_count must be >= 1 for capacity search")
        cap = find_fleet_capacity(count=npu_count, chip_query=args.get("chip_query"), host=target or None, mode=mode, directory=store, state_dir=base_dir, runner=runner)
        return {"state": "ready" if cap["matched_count"] > 0 else "insufficient_capacity", "action": "capacity", **cap}

    if target:
        _, host_spec, rec = _resolve_host_target(args, store)
        records = [rec if rec is not None else {"alias": target, "host": {"ip": host_spec["host"], "port": host_spec["port"], "user": host_spec["user"]}}]
    else:
        try:
            records = store.machines()
        except MachineDirectoryUnavailable:
            records = []

    snapshots = [probe_machine_record(rec, mode=mode, state_dir=base_dir, runner=runner) for rec in records]
    if action == "hosts":
        hosts_summary = [
            {
                "alias": snap["alias"], "host": snap["host"], "port": snap["port"], "online": snap["online"],
                "error": snap.get("error"), "chip_summary": snap.get("chip_summary"), "driver_version": snap.get("driver_version"),
                "total_npus": snap["total_npus"], "free_npus": snap["free_npus"], "busy_npus": snap["busy_npus"],
                "free_npu_ids": snap["free_npu_ids"], "busy_npu_ids": snap["busy_npu_ids"],
                "free_phy_ids": snap.get("free_phy_ids", []), "busy_phy_ids": snap.get("busy_phy_ids", []),
                "bound_container": snap.get("bound_container"), "containers_count": len(snap.get("containers") or []), "cache_hit": snap.get("cache_hit", False),
            }
            for snap in snapshots
        ]
        return {"state": "ready", "action": "hosts", "mode": mode, "hosts_count": len(hosts_summary), "hosts": hosts_summary}

    if action == "npu":
        return {"state": "ready", "action": "npu", "mode": mode, "hosts": [
            {
                "alias": snap["alias"], "host": snap["host"], "online": snap["online"], "error": snap.get("error"),
                "chip_summary": snap.get("chip_summary"), "driver_version": snap.get("driver_version"),
                "npu_smi_version": snap.get("npu_smi_version"), "total_npus": snap["total_npus"], "free_npus": snap["free_npus"],
                "free_npu_ids": snap["free_npu_ids"], "busy_npu_ids": snap["busy_npu_ids"],
                "free_phy_ids": snap.get("free_phy_ids", []), "busy_phy_ids": snap.get("busy_phy_ids", []), "npus": snap.get("npus", []),
            }
            for snap in snapshots
        ]}

    if action == "server":
        return {"state": "ready", "action": "server", "mode": mode, "hosts": [
            {"alias": snap["alias"], "host": snap["host"], "online": snap["online"], "error": snap.get("error"), "server": snap.get("server") or {}, "containers": snap.get("containers") or [], "bound_container": snap.get("bound_container")}
            for snap in snapshots
        ]}

    if action == "mounts":
        return {"state": "ready", "action": "mounts", "mode": mode, "hosts": [
            {"alias": snap["alias"], "host": snap["host"], "online": snap["online"], "error": snap.get("error"), "mounts": snap.get("mounts") or [], "weights_mounts": [m for m in (snap.get("mounts") or []) if m.get("likely_weights_dir")]}
            for snap in snapshots
        ]}

    raise caller_error(f"unsupported mindie.probe action: {action!r}")


def handle_container(
    args: Mapping[str, Any],
    *,
    directory: MachineDirectory | None = None,
    state_dir: Path | str | None = None,
    runner: Callable[[Mapping[str, Any], str], str] | None = None,
    queue: HostQueue | None = None,
) -> dict[str, Any]:
    """Execute `mindie.container` (`bind`, `provision`, `inspect`, `remove`)."""
    action = str(args.get("action") or "inspect").strip()
    store, base_dir = _resolve_store(directory, state_dir)

    if action == "bind":
        alias, host_spec, rec = _resolve_host_target(args, store)
        container_name = str(args.get("container_name") or "").strip()
        ssh_port = args.get("ssh_port")
        if not container_name and not ssh_port:
            raise caller_error("mindie_container(action='bind') requires `container_name` and/or `ssh_port`")

        container_user = str(args.get("container_user") or "root").strip() or "root"
        workdir = str(args.get("workdir") or "/vllm-workspace").strip() or "/vllm-workspace"
        mode = str(args.get("mode") or ("ssh" if ssh_port else "docker_exec")).strip().lower()

        container_entry: dict[str, Any] = {"user": container_user, "workdir": workdir, "mode": mode}
        if container_name:
            container_entry["name"] = container_name
        if args.get("container_id"):
            container_entry["id"] = str(args["container_id"]).strip()
        if ssh_port:
            container_entry["ssh_port"] = int(ssh_port)
        if args.get("python"):
            container_entry["python"] = str(args["python"]).strip()
        if args.get("runtime_env_file") or args.get("env_file"):
            container_entry["runtime_env_file"] = str(args.get("runtime_env_file") or args.get("env_file")).strip()
        if args.get("runtime_env") is not None:
            container_entry["runtime_env"] = bool(args["runtime_env"])

        base_rec = rec if rec is not None else {"alias": alias, "host": {"ip": host_spec["host"], "port": host_spec["port"], "user": host_spec["user"]}}
        updated_rec = {
            **base_rec, "alias": alias,
            "host": {**(base_rec.get("host") or {}), "ip": host_spec["host"], "port": host_spec["port"], "user": host_spec["user"]},
            "container": container_entry,
        }
        inv_path = store.upsert_machine(updated_rec, set_default=bool(args.get("set_default")))
        endpoints_doc = store.endpoints_document()
        return {
            "state": "ready", "action": "bind", "alias": alias, "container": container_entry,
            "inventory_path": str(inv_path), "endpoints_file": str(endpoints_file_path(inv_path.parent)),
            "remote_dev_endpoint": (endpoints_doc.get("endpoints") or {}).get(alias),
        }

    if action == "provision":
        alias, host_spec, rec = _resolve_host_target(args, store)
        image = str(args.get("image") or "").strip()
        if not image:
            raise caller_error(
                "mindie_container(action='provision') requires `image` ('local-latest', 'rc', 'main', 'stable', or an explicit image reference)"
            )
        machine_type = args.get("machine_type") or ((rec or {}).get("host") or {}).get("machine_type")
        host_queue = queue if queue is not None else HostQueue()

        def _reserve_port(*, user: str, container_name: str, port: int) -> dict[str, Any]:
            return host_queue.request(
                host_spec,
                {"action": "container-ssh-reserve", "user": user, "container_name": container_name, "port": port},
            )

        provisioned = provision_user_container(
            host=host_spec["host"], image=image,
            user=args.get("container_user") or args.get("user_namespace") or args.get("user_tag") or args.get("user"),
            host_user=host_spec["user"], host_port=host_spec["port"],
            ssh_port=int(args["ssh_port"]) if args.get("ssh_port") else None,
            machine_type=machine_type, password_env=args.get("password_env"),
            machines=store, reserve_port=_reserve_port, alias=alias,
        )
        if args.get("set_default"):
            store.set_default(alias)
        endpoints_doc = store.endpoints_document()
        return {
            "state": "ready", "action": "provision", "alias": alias,
            "provisioned": provisioned, "remote_dev_endpoint": (endpoints_doc.get("endpoints") or {}).get(alias),
        }

    if action == "inspect":
        alias, host_spec, rec = _resolve_host_target(args, store)
        probe_rec = rec if rec is not None else {"alias": alias, "host": {"ip": host_spec["host"], "port": host_spec["port"], "user": host_spec["user"]}}
        snap = probe_machine_record(probe_rec, mode=str(args.get("mode") or "auto"), state_dir=base_dir, runner=runner)
        endpoints_doc = store.endpoints_document()
        return {
            "state": "ready", "action": "inspect", "alias": alias, "host": host_spec["host"],
            "online": snap.get("online"), "bound_container": (rec or {}).get("container"),
            "containers": snap.get("containers") or [],
            "remote_dev_endpoint": (endpoints_doc.get("endpoints") or {}).get(alias),
        }

    if action == "remove":
        alias, host_spec, rec = _resolve_host_target(args, store)
        bound = (rec or {}).get("container") or {}
        container_name = str(args.get("container_name") or bound.get("name") or "").strip()
        ssh_port = int(args.get("ssh_port") or bound.get("ssh_port") or 0)
        unbind_only = bool(args.get("unbind_only"))

        removal = None
        release = None
        if not unbind_only:
            if not container_name:
                raise caller_error("mindie_container(action='remove') requires `container_name` or a bound container on the target machine")
            ns = host_ops._namespace(
                host=host_spec["host"], user=host_spec["user"], host_port=host_spec["port"],
                container_name=container_name, container_ssh_port=ssh_port if ssh_port > 0 else None,
                known_hosts=str(host_ops.DEFAULT_KNOWN_HOSTS), clean_local_known_hosts=bool(ssh_port > 0),
            )
            removal = host_ops.remove_container_result(ns)
            confirmed_absent = removal.get("removed") is False and removal.get("previous_state") is None
            if removal.get("success") is not True or not (removal.get("removed") is True or confirmed_absent):
                raise RuntimeError(f"failed to confirm removal of container {container_name!r} on {host_spec['host']}")
            owner = str(args.get("container_user") or args.get("owner") or (rec or {}).get("user") or "").strip()
            if not owner and container_name.startswith("mindie-"):
                owner = container_name[len("mindie-"):]
            if not owner:
                return {
                    "state": "partial", "action": "remove", "alias": alias, "container_name": container_name,
                    "removal": removal,
                    "reservation_release": {"status": "not_attempted", "reason": "container owner is unknown; reservation not touched"},
                    "remote_dev_endpoint": (store.endpoints_document().get("endpoints") or {}).get(alias),
                }
            host_queue = queue if queue is not None else HostQueue()
            try:
                release = host_queue.request(host_spec, {
                    "action": "container-ssh-release", "user": owner,
                    "container_name": container_name, "port": ssh_port if ssh_port > 0 else 0,
                })
            except Exception as exc:
                return {
                    "state": "partial", "action": "remove", "alias": alias, "container_name": container_name,
                    "removal": removal,
                    "reservation_release": {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:800]},
                    "remote_dev_endpoint": (store.endpoints_document().get("endpoints") or {}).get(alias),
                }

        if rec is not None:
            cleaned = {k: v for k, v in rec.items() if k != "container"}
            store.upsert_machine(cleaned)
        endpoints_doc = store.endpoints_document()
        return {
            "state": "ready", "action": "remove", "alias": alias, "container_name": container_name or None,
            "unbind_only": unbind_only, "removal": removal, "reservation_release": release,
            "remote_dev_endpoint": (endpoints_doc.get("endpoints") or {}).get(alias),
        }

    raise caller_error(f"unsupported mindie.container action: {action!r}")


def _leases_file(base_dir: Path) -> Path:
    return base_dir / "leases.json"


def _load_leases(base_dir: Path) -> dict[str, dict[str, Any]]:
    import json

    p = _leases_file(base_dir)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Corrupted leases file {p}")
    leases = data.get("leases") or {}
    if not isinstance(leases, dict):
        raise ValueError(f"Corrupted leases file {p}")
    return {str(key): dict(value) for key, value in leases.items()}


def _save_leases(base_dir: Path, leases: Mapping[str, Mapping[str, Any]]) -> None:
    import json

    path = _leases_file(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"leases": {k: dict(v) for k, v in leases.items()}}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def handle_lease(
    args: Mapping[str, Any],
    *,
    directory: MachineDirectory | None = None,
    state_dir: Path | str | None = None,
    queue: HostQueue | None = None,
) -> dict[str, Any]:
    """Execute `mindie.lease` (`acquire`, `heartbeat`, `status`, `release`) via HostQueue."""
    action = str(args.get("action") or "status").strip()
    store, base_dir = _resolve_store(directory, state_dir)
    alias, host_spec, _ = _resolve_host_target(args, store)
    host_queue = queue if queue is not None else HostQueue()
    owner_user = safe_id(str(args.get("owner") or getpass.getuser()))
    remote_sd = {"state_dir": str(args["remote_state_dir"])} if args.get("remote_state_dir") else {}

    if action == "status":
        status_req: dict[str, Any] = {"action": "status", **remote_sd}
        if args.get("lease_id"):
            status_req["task_id"] = str(args["lease_id"]).strip()
        response = host_queue.request(host_spec, status_req)
        tasks_list = [
            {**t, "lease_id": t.get("lease_id") or t.get("task_id")} if isinstance(t, Mapping) else t
            for t in (response.get("tasks") or [])
        ]
        return {
            "state": "ready", "action": "status", "alias": alias, "host": host_spec["host"],
            "coordination_epoch": response.get("coordination_epoch"), "occupancy": response.get("occupancy"),
            "tasks": tasks_list, "leases": tasks_list, "count": len(tasks_list),
            "coordination_messages": response.get("coordination_messages", []),
        }

    if action == "acquire":
        lease_id = safe_id(str(args.get("lease_id") or f"lease-{owner_user}-{uuid.uuid4().hex[:10]}").strip())
        devices = args.get("devices")
        npu_count_arg = args.get("npu_count")
        if isinstance(devices, list) and npu_count_arg is not None and len(devices) != int(npu_count_arg):
            raise caller_error("when both `devices` and `npu_count` are supplied, `npu_count` must equal `len(devices)`")
        ttl_seconds = int(args.get("ttl_seconds") or 900)
        epoch = args.get("coordination_epoch")
        if not epoch:
            status_resp = host_queue.request(host_spec, {"action": "status", "no_probe": True, **remote_sd})
            epoch = status_resp.get("coordination_epoch")

        req: dict[str, Any] = {
            "action": "submit-acquire" if epoch else "submit", "task_id": lease_id, "agent_id": owner_user,
            "message": str(args.get("purpose") or "remote-dev workload"),
            "estimated_duration_seconds": ttl_seconds, "grant_ttl_seconds": ttl_seconds,
            "allow_external_busy": bool(args.get("allow_external_busy")), **remote_sd,
        }
        if epoch:
            req["coordination_epoch"] = str(epoch)
        if isinstance(devices, list):
            req["devices"] = [int(d) for d in devices]
        else:
            req["npu_count"] = int(npu_count_arg) if npu_count_arg is not None else 1
        if "service_port" in args and args["service_port"] is not None:
            sp = int(args["service_port"])
            req["service_port"] = sp
            if sp > 0:
                req["service_ports"] = [sp]

        response = host_queue.request(host_spec, req)
        if req["action"] == "submit" and (response.get("task") or {}).get("state") == "queued":
            response = host_queue.request(
                host_spec,
                {"action": "acquire", "task_id": lease_id, "grant_ttl_seconds": ttl_seconds, **remote_sd},
            )
        task = response.get("task") or {}
        raw_state = str(task.get("state") or response.get("status") or "unknown")
        allocated_devices = task.get("granted_devices") if task.get("granted_devices") is not None else (task.get("allocated_npus") or [])
        visible_str = ",".join(str(d) for d in allocated_devices)
        service_port = task.get("granted_service_port") if task.get("granted_service_port") is not None else task.get("service_port")
        fence_token = task.get("fence_token")
        env_vars: dict[str, str] = {}
        if visible_str:
            env_vars["ASCEND_RT_VISIBLE_DEVICES"] = visible_str
            env_vars["MINDIE_LEASED_DEVICES"] = visible_str
        if fence_token is not None:
            env_vars["MINDIE_FENCE_TOKEN"] = str(fence_token)
        if task.get("job_token"):
            env_vars["MINDIE_JOB_TOKEN"] = str(task["job_token"])
        if service_port is not None:
            env_vars["MINDIE_SERVICE_PORT"] = str(service_port)

        leases = _load_leases(base_dir)
        lease_record = {
            "lease_id": lease_id, "alias": alias, "host": host_spec["host"], "owner": owner_user,
            "fence_token": fence_token, "coordination_epoch": epoch,
            "devices": allocated_devices, "service_port": service_port, "state": raw_state,
        }
        leases[f"{alias}:{lease_id}"] = lease_record
        leases[lease_id] = lease_record
        _save_leases(base_dir, leases)

        return {
            "state": "active" if raw_state in {"granted", "active", "starting"} else raw_state,
            "lease_state": raw_state, "action": "acquire", "alias": alias, "host": host_spec["host"],
            "lease_id": lease_id, "owner": owner_user, "fence_token": fence_token,
            "devices": allocated_devices, "ascend_rt_visible_devices": visible_str,
            "service_port": service_port, "env": env_vars, "lease": task, "remote_dev_alias": alias,
        }

    if action == "heartbeat":
        lease_id = safe_id(str(args.get("lease_id") or "").strip())
        if not lease_id:
            raise caller_error("mindie_lease(action='heartbeat') requires `lease_id`")
        ttl_seconds = int(args.get("ttl_seconds") or 900)
        leases = _load_leases(base_dir)
        stored = leases.get(f"{alias}:{lease_id}") or leases.get(lease_id) or {}
        fence_token = args.get("fence_token") if args.get("fence_token") is not None else stored.get("fence_token")
        if fence_token is None:
            snap = host_queue.request(host_spec, {"action": "status", "task_id": lease_id, "no_probe": True, **remote_sd})
            tasks = snap.get("tasks") or []
            if tasks:
                fence_token = tasks[0].get("fence_token")
        if fence_token is None:
            raise caller_error(f"no fence_token found for lease {lease_id!r}; pass `fence_token` or acquire the lease first")
        response = host_queue.request(
            host_spec,
            {"action": "heartbeat", "task_id": lease_id, "fence_token": int(fence_token), "heartbeat_ttl_seconds": ttl_seconds, **remote_sd},
        )
        task = response.get("task") or {}
        raw_state = str(task.get("state") or response.get("status") or "active")
        allocated_devices = task.get("granted_devices") if task.get("granted_devices") is not None else (task.get("allocated_npus") or [])
        service_port = task.get("granted_service_port") if task.get("granted_service_port") is not None else task.get("service_port")
        return {
            "state": "active" if raw_state in {"granted", "active", "starting"} else raw_state,
            "lease_state": raw_state, "action": "heartbeat", "alias": alias, "host": host_spec["host"],
            "lease_id": lease_id, "fence_token": int(fence_token), "devices": allocated_devices,
            "ascend_rt_visible_devices": ",".join(str(d) for d in allocated_devices),
            "service_port": service_port, "lease": task,
        }

    if action == "release":
        lease_id = safe_id(str(args.get("lease_id") or "").strip())
        if not lease_id:
            raise caller_error("mindie_lease(action='release') requires `lease_id`")
        leases = _load_leases(base_dir)
        stored = leases.get(f"{alias}:{lease_id}") or leases.get(lease_id) or {}
        fence_token = args.get("fence_token") if args.get("fence_token") is not None else stored.get("fence_token")
        if fence_token is None:
            snap = host_queue.request(host_spec, {"action": "status", "task_id": lease_id, "no_probe": True, **remote_sd})
            tasks = snap.get("tasks") or []
            if tasks:
                fence_token = tasks[0].get("fence_token")
        if fence_token is not None:
            response = host_queue.request(
                host_spec,
                {
                    "action": "release", "task_id": lease_id, "fence_token": int(fence_token),
                    "free_samples": int(args.get("free_samples") or 1),
                    "interval_seconds": float(args.get("interval_seconds") or 0.0), **remote_sd,
                },
            )
        else:
            response = host_queue.request(host_spec, {"action": "cancel", "task_id": lease_id, **remote_sd})
        leases.pop(f"{alias}:{lease_id}", None)
        leases.pop(lease_id, None)
        _save_leases(base_dir, leases)
        task = response.get("task") or {}
        raw_state = str(task.get("state") or response.get("status") or "released")
        return {
            "state": "released" if raw_state in {"released", "cancelled"} else raw_state,
            "lease_state": raw_state, "action": "release", "alias": alias, "host": host_spec["host"],
            "lease_id": lease_id, "fence_token": int(fence_token) if fence_token is not None else None, "lease": task,
        }

    raise caller_error(f"unsupported mindie.lease action: {action!r}")
