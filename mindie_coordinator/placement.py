"""Choose hosts and prepared roots from attested catalog + constraints.

[Role] Pure constraint solver mapping environment/resource/topology requirements
to attested host candidates, including observed topology-gang role plans.
[Invariants] Never relaxes topology, recipe, ABI, or hardware constraints, and never starts a partial multi-role group.
- A topology-gang plan is executable only with observed host/card topology;
  explicit host, physical devices, and service ports are preserved or rejected.
- Optional `npu-top` observations may only reorder matching ready roots. An
  explicit role/topology host still filters by identity and is never rewritten;
  the host queue remains the allocation authority.
[Timeouts & Locks] Pure in-memory evaluation; zero I/O or locks.
[Side-Effect Policy] Side-effect free.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any, Sequence

from remote_dev.core.host_spec import evaluate_placement, normalize_remote_probe_spec
from mindie_coordinator.topology_gang import (
    POLICY_DISAGGREGATED_PD_PAIR,
    POLICY_MULTI_NODE_ROCE,
    POLICY_SAME_NPU_DUAL_DIE,
    SCALE_S3,
    GangLeaseError,
    TopologySpec,
    allocate_gang_lease,
)

SUPPORTED_RECIPES = {"rc", "main", "stable", "local-latest"}
RESERVED_ENV = {"ASCEND_RT_VISIBLE_DEVICES", "MINDIE_SERVICE_PORT", "MINDIE_PYTHON", "MINDIE_RUN_ID", "MINDIE_EXECUTION_OBSERVATION"}
ENVIRONMENT_KEYS = {"recipe", "image", "python_abi", "cann", "soc", "machine_type"}
RESOURCE_KEYS = {"devices", "npu_count", "service_port", "allow_external_busy", "cpu_cores", "memory_mb", "hbm_mb"}
TOPOLOGY_GANG_KEYS = {
    "scale", "die_count", "placement_policy", "prefill_die_count",
    "decode_die_count", "node_count", "visible_phy_ids",
}
TOPOLOGY_KEYS = {"host", "roles", "distinct_hosts"} | TOPOLOGY_GANG_KEYS
ROLE_KEYS = {"name", "command", "preflight", "devices", "npu_count", "service_port", "allow_external_busy", "cpu_cores", "memory_mb", "hbm_mb", "host", "env"}


def validate_wait(until, timeout_seconds):
    """Validate an observation request without importing the service runtime."""
    if until not in {"running", "released"}:
        raise ValueError("until must be running or released")
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or not 0 <= timeout_seconds <= 600):
        raise ValueError("wait timeout_seconds must be finite and between 0 and 600")


def provisionable_recipe(recipe: str) -> bool:
    """Use the provision owner's existing fixed-image syntax for new roots too."""
    if recipe in SUPPORTED_RECIPES:
        return True
    from mindie_coordinator.provision.host_ops import MachineManagementError, require_explicit_image_ref
    try:
        require_explicit_image_ref(recipe)
    except MachineManagementError:
        return False
    return True


def checked_mapping(value, label, supported):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    unknown = set(value) - supported
    if unknown:
        raise ValueError(f"unsupported {label} fields: {', '.join(sorted(map(str, unknown)))}; supported: {', '.join(sorted(supported))}")
    return dict(value)


def normalize_environment(environment):
    environment = checked_mapping(environment, "environment", ENVIRONMENT_KEYS)
    if any(not isinstance(value, str) or not value.strip() for value in environment.values()):
        raise ValueError("environment constraints must be nonempty strings")
    if environment.get("recipe") and environment.get("image") and environment["recipe"] != environment["image"]:
        raise ValueError("environment.recipe and environment.image specify conflicting constraints; supply one")
    return environment


def validate_user_env(env: dict[str, Any] | None) -> dict[str, str]:
    """Literal user env only. Managed device/port/interpreter/job tokens stay reserved."""
    if env is None:
        return {}
    if not isinstance(env, dict):
        raise ValueError("env must be a mapping of literal string values")
    cleaned: dict[str, str] = {}
    for key, value in env.items():
        if not isinstance(key, str) or not key:
            raise ValueError("env keys must be nonempty strings")
        if key in RESERVED_ENV or key.startswith("REMOTE_DEV_JOB_"):
            raise ValueError("cannot override managed device, service-port, interpreter, or job-token ownership")
        if not isinstance(value, str):
            raise ValueError("env values must stay literal strings")
        cleaned[key] = value
    return cleaned


def normalize_resources(resources: dict[str, Any] | None) -> dict[str, Any]:
    resources = checked_mapping(resources, "resources", RESOURCE_KEYS)
    if "devices" not in resources and "npu_count" not in resources:
        resources["npu_count"] = 0
    if "devices" in resources:
        devices = resources["devices"]
        if not isinstance(devices, list) or any(type(d) is not int or d < 0 for d in devices):
            raise ValueError("resources.devices must be distinct nonnegative integers")
        if len(set(devices)) != len(devices):
            raise ValueError("resources.devices must be distinct nonnegative integers")
    if "npu_count" in resources and (type(resources["npu_count"]) is not int or resources["npu_count"] < 0):
        raise ValueError("npu_count must be a nonnegative integer")
    if "devices" in resources and "npu_count" in resources:
        if resources["npu_count"] != len(resources["devices"]):
            raise ValueError("npu_count must equal the number of devices when both are supplied")
        resources.pop("npu_count")
    if "service_port" in resources and (type(resources["service_port"]) is not int or not 0 <= resources["service_port"] < 65536):
        raise ValueError("service_port must be an integer from 0 to 65535")
    if "allow_external_busy" in resources:
        if type(resources["allow_external_busy"]) is not bool:
            raise ValueError("allow_external_busy must be a boolean")
        if resources["allow_external_busy"] and len(resources.get("devices", [])) != 1:
            raise ValueError("allow_external_busy requires exactly one explicit physical device")
    for hw_key in ("cpu_cores", "memory_mb", "hbm_mb"):
        if hw_key in resources and (type(resources[hw_key]) is not int or resources[hw_key] < 0):
            raise ValueError(f"{hw_key} must be a nonnegative integer")
    return resources


def extract_topology_spec(topology: dict[str, Any] | None) -> TopologySpec | None:
    """Return the validated topology-gang contract, when one was requested.

    Any of `scale`, `placement_policy`, or `die_count` opts a topology into
    gang planning. `GangLeaseError` subclasses `ValueError`, so admission code
    and gang-aware callers can catch either type for the same refusal.
    """
    if topology is None:
        return None
    if not isinstance(topology, dict):
        raise ValueError("topology must be an object")
    if not TOPOLOGY_GANG_KEYS & set(topology):
        return None
    if not {"scale", "die_count", "placement_policy"} & set(topology):
        raise ValueError(
            "topology gang fields require scale, die_count, or placement_policy"
        )
    payload = {key: topology[key] for key in TOPOLOGY_GANG_KEYS if key in topology}
    return TopologySpec.from_mapping(payload)


def _gang_role(
    name: str,
    command: str,
    *,
    devices: Sequence[int] | None = None,
    npu_count: int | None = None,
    host: str | None = None,
    env: dict[str, str] | None = None,
    service_port: int | None = None,
    allow_external_busy: bool = False,
) -> dict[str, Any]:
    role: dict[str, Any] = {"name": name, "command": command}
    if devices is not None:
        role["devices"] = [int(device) for device in devices]
    else:
        role["npu_count"] = int(npu_count or 0)
    if host is not None:
        role["host"] = host
    if env is not None:
        role["env"] = dict(env)
    if service_port is not None:
        role["service_port"] = int(service_port)
    if allow_external_busy:
        role["allow_external_busy"] = True
    return role


def _require_sibling_pairs(
    phy_ids: Sequence[int], *, code: str = "insufficient_dual_die_pairs"
) -> None:
    """A3 dual-Die cards expose siblings as `(2k, 2k+1)`; never straddle cards."""
    if len(phy_ids) % 2:
        raise GangLeaseError(
            code,
            f"same_npu_dual_die requires an even Phy-ID count, got {list(phy_ids)}",
        )
    for first, second in zip(phy_ids[::2], phy_ids[1::2]):
        if first % 2 or second != first + 1:
            raise GangLeaseError(
                code,
                f"same_npu_dual_die requires sibling Phy-ID pairs, got {list(phy_ids)}",
            )


def _apply_requested_resources(
    roles: list[dict[str, Any]],
    resources: dict[str, Any],
    *,
    host: str | None = None,
) -> list[dict[str, Any]]:
    """Carry supported resource bindings onto the generated role plan.

    The host queue remains the single port/device authority. `service_port`
    applies to the first/default service; a single port cannot be duplicated
    across independent PD or S3 roles. `allow_external_busy` is only valid for
    one explicit Die and is therefore generated for one role at most.
    """
    if not roles:
        return roles
    if host:
        for role in roles:
            role["host"] = host
    if resources.get("service_port") is not None:
        roles[0]["service_port"] = int(resources["service_port"])
    if resources.get("allow_external_busy"):
        roles[0]["allow_external_busy"] = True
    for hw_key in ("cpu_cores", "memory_mb", "hbm_mb"):
        if resources.get(hw_key) is not None:
            for role in roles:
                role.setdefault(hw_key, int(resources[hw_key]))
    return roles


def _prepare_gang_request(
    topology: dict[str, Any],
    spec: TopologySpec,
    resources: dict[str, Any] | None,
    host_topologies: Sequence[Any] | None,
    *,
    require_observed: bool,
) -> tuple[TopologySpec, dict[str, Any], str | None, Sequence[Any] | None]:
    """Validate host/device/port bindings and merge them into the topology spec.

    Explicit `resources.devices` become the requested physical Phy-IDs rather
    than a count, so the host queue must grant exactly those IDs or wait. A
    conflict between `topology.visible_phy_ids`, `resources.devices`, or
    `topology.host` is rejected before any preparation or submission.
    """
    if not isinstance(topology, dict):
        raise GangLeaseError("invalid_spec", "topology must be an object")
    if "distinct_hosts" in topology and type(topology["distinct_hosts"]) is not bool:
        raise GangLeaseError("invalid_spec", "topology.distinct_hosts must be a boolean")
    host = topology.get("host")
    if host is not None and (not isinstance(host, str) or not host.strip()):
        raise GangLeaseError(
            "invalid_spec", "topology.host must be a nonempty hostname or IP address"
        )
    if host and spec.scale == SCALE_S3:
        raise GangLeaseError(
            "conflicting_host",
            "topology.host cannot bind a multi-node RoCE gang; use topology.roles "
            "with one host per node",
        )
    if not isinstance(resources, dict) and resources is not None:
        raise GangLeaseError("invalid_spec", "resources must be an object")
    declared = dict(resources or {})
    normalized = normalize_resources(declared)
    devices = list(normalized.get("devices") or [])
    if devices:
        if spec.scale == SCALE_S3:
            raise GangLeaseError(
                "conflicting_devices",
                "multi-node S3 cannot bind one flat resources.devices list across hosts",
            )
        if len(devices) != spec.die_count:
            raise GangLeaseError(
                "conflicting_devices",
                f"resources.devices has {len(devices)} entries but topology.die_count is {spec.die_count}",
            )
        if spec.visible_phy_ids is not None and set(spec.visible_phy_ids) != set(devices):
            raise GangLeaseError(
                "conflicting_devices",
                "topology.visible_phy_ids and resources.devices request different physical Dies",
            )
        spec = TopologySpec.from_mapping(
            {**spec.to_dict(), "visible_phy_ids": devices}
        )
        if spec.placement_policy == POLICY_SAME_NPU_DUAL_DIE:
            # An explicitly requested straddling pair is a request conflict,
            # not a temporary capacity shortage that waiting could fix.
            _require_sibling_pairs(
                spec.visible_phy_ids or (), code="conflicting_devices"
            )
    elif (
        "npu_count" in declared
        and declared["npu_count"] not in (0, spec.die_count)
    ):
        raise GangLeaseError(
            "conflicting_npu_count",
            f"resources.npu_count={declared['npu_count']} conflicts with topology.die_count={spec.die_count}",
        )
    if normalized.get("allow_external_busy"):
        if spec.die_count != 1 or len(devices) != 1:
            raise GangLeaseError(
                "conflicting_allow_external_busy",
                "allow_external_busy is only valid for one explicit physical Die",
            )

    if host_topologies is not None and hasattr(host_topologies, "host"):
        host_topologies = [host_topologies]
    if host_topologies is not None:
        host_topologies = list(host_topologies)
    if require_observed and host_topologies is None:
        raise GangLeaseError(
            "host_observation_required",
            "topology-gang placement requires live host/card observations; "
            "a count-only plan is not executable",
        )
    if host and host_topologies is not None:
        matching = [
            item for item in host_topologies if str(getattr(item, "host", "")) == host
        ]
        if not matching:
            raise GangLeaseError(
                "requested_host_unobserved",
                f"no topology observation for requested host {host!r}",
                host=host,
            )
        host_topologies = matching
    return spec, normalized, host, host_topologies


def _structural_gang_roles(
    spec: TopologySpec,
    command: str,
    *,
    host: str | None = None,
    resources: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Synthesize the role shape for client validation or a host probe.

    A structural plan is never executed by `CoordinatorService`: gang
    executions require observed `HostTopology` rows so physical devices and
    sibling pairs are concrete before submission.
    """
    if spec.placement_policy == POLICY_DISAGGREGATED_PD_PAIR:
        if spec.visible_phy_ids is not None:
            prefill_count = int(spec.prefill_die_count or 0)
            roles = [
                _gang_role(
                    "prefill", command,
                    devices=spec.visible_phy_ids[:prefill_count],
                ),
                _gang_role(
                    "decode", command,
                    devices=spec.visible_phy_ids[prefill_count:],
                ),
            ]
            return _apply_requested_resources(roles, resources or {}, host=host)
        roles = [
            _gang_role("prefill", command, npu_count=spec.prefill_die_count),
            _gang_role("decode", command, npu_count=spec.decode_die_count),
        ]
        return _apply_requested_resources(roles, resources or {}, host=host)
    if spec.placement_policy == POLICY_MULTI_NODE_ROCE:
        node_count = int(spec.node_count or 2)
        per_node = spec.die_count // node_count
        roles = [
            _gang_role(f"node-{index}", command, npu_count=per_node)
            for index in range(node_count)
        ]
        return _apply_requested_resources(roles, resources or {}, host=host)
    if spec.visible_phy_ids is not None:
        if spec.placement_policy == POLICY_SAME_NPU_DUAL_DIE:
            _require_sibling_pairs(spec.visible_phy_ids)
        roles = [_gang_role("default", command, devices=spec.visible_phy_ids)]
        return _apply_requested_resources(roles, resources or {}, host=host)
    roles = [_gang_role("default", command, npu_count=spec.die_count)]
    return _apply_requested_resources(roles, resources or {}, host=host)


def _lease_gang_roles(
    lease,
    spec: TopologySpec,
    command: str,
    *,
    host: str | None = None,
    resources: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Materialize the gang lease's per-role physical device groups."""
    if host and any(member.host != host for member in lease.members):
        raise GangLeaseError(
            "requested_host_mismatch",
            f"selected host(s) {list(lease.hosts)} do not satisfy topology.host={host!r}",
            host=host,
        )
    if spec.placement_policy == POLICY_DISAGGREGATED_PD_PAIR:
        roles = [
            _gang_role(
                member.role, command,
                devices=member.phy_ids,
                host=member.host,
                env={"MINDIE_GANG_ROLE": member.role},
            )
            for member in lease.members
        ]
        return _apply_requested_resources(roles, resources or {}, host=host)
    if spec.placement_policy == POLICY_MULTI_NODE_ROCE:
        roles = [
            _gang_role(
                member.role, command,
                devices=member.phy_ids,
                host=member.host,
            )
            for member in lease.members
        ]
        return _apply_requested_resources(roles, resources or {}, host=host)
    member = lease.members[0]
    roles = [
        _gang_role(
            "default", command,
            devices=member.phy_ids,
            host=member.host,
        )
    ]
    return _apply_requested_resources(roles, resources or {}, host=host)


def plan_gang_placement(
    topology: dict[str, Any] | None,
    resources: dict[str, Any],
    command: str,
    *,
    host_topologies: Sequence[Any] | None = None,
    require_observed: bool = False,
):
    """Validate one gang request and plan concrete roles.

    Returns `(lease, roles)`. Without observations the lease is `None` and the
    roles are a structural shape only. `require_observed=True` refuses that
    shape, which is how production execution avoids a count-only fallback.
    """
    topology = checked_mapping(topology, "topology", TOPOLOGY_KEYS)
    spec = extract_topology_spec(topology)
    if spec is None:
        raise GangLeaseError(
            "invalid_spec",
            "topology must specify scale, die_count, or placement_policy for gang planning",
        )
    if "roles" in topology:
        raise GangLeaseError(
            "conflicting_roles",
            "topology.roles cannot be combined with topology-gang fields",
        )
    spec, normalized, host, hosts = _prepare_gang_request(
        topology, spec, resources, host_topologies, require_observed=require_observed
    )
    if hosts is None:
        return None, _structural_gang_roles(
            spec, command, host=host, resources=normalized
        )
    lease = allocate_gang_lease(hosts, spec)
    return lease, _lease_gang_roles(
        lease, spec, command, host=host, resources=normalized
    )


def plan_gang_roles(
    topology: dict[str, Any] | None,
    resources: dict[str, Any],
    command: str,
    *,
    host_topologies: Sequence[Any] | None = None,
    require_observed: bool = False,
) -> list[dict[str, Any]]:
    """Turn a topology-gang spec into ordered role requirements.

    With host observations, `allocate_gang_lease` selects the physical Dies and
    each role carries its concrete disjoint device group plus the observed
    host. Without observations, only the structural role shape is emitted so
    clients can validate inputs; production `CoordinatorService` always calls
    with `require_observed=True` and real fleet observations.
    """
    _lease, roles = plan_gang_placement(
        topology,
        resources,
        command,
        host_topologies=host_topologies,
        require_observed=require_observed,
    )
    return roles


def role_plan(
    topology: dict[str, Any] | None,
    resources: dict[str, Any],
    command: str,
    *,
    host_topologies: Sequence[Any] | None = None,
    require_observed: bool = False,
) -> list[dict[str, Any]]:
    topology = checked_mapping(topology, "topology", TOPOLOGY_KEYS)
    if "distinct_hosts" in topology and type(topology["distinct_hosts"]) is not bool:
        raise ValueError("topology.distinct_hosts must be a boolean")
    if "host" in topology and (not isinstance(topology["host"], str) or not topology["host"].strip()):
        raise ValueError("topology.host must be a nonempty hostname or IP address")
    if "host" in topology and "roles" in topology:
        raise ValueError("use topology.host for one default role, or topology.roles with host in each role; do not combine them")
    if extract_topology_spec(topology) is not None:
        if "roles" in topology:
            raise ValueError(
                "topology.roles cannot be combined with topology-gang fields"
            )
        return plan_gang_roles(
            topology,
            resources,
            command,
            host_topologies=host_topologies,
            require_observed=require_observed,
        )
    if "roles" not in topology:
        item = {"name": "default", "command": command}
        if "host" in topology:
            item["host"] = topology["host"]
        item.update({k: resources[k] for k in RESOURCE_KEYS if k in resources})
        return [item]
    if not isinstance(topology["roles"], list) or not topology["roles"]:
        raise ValueError("topology.roles must be a nonempty array of role objects")
    roles = []
    for role in topology["roles"]:
        role = checked_mapping(role, "topology.roles[]", ROLE_KEYS)
        if not isinstance(role.get("name"), str) or not role["name"].strip():
            raise ValueError("topology.roles entries need a name")
        name = role["name"]
        if any(previous["name"] == name for previous in roles):
            raise ValueError("topology.roles names must be distinct")
        if "command" in role and (not isinstance(role["command"], str) or not role["command"].strip()):
            raise ValueError("role command must be a nonempty shell command")
        item = {"name": name, "command": role.get("command", command)}
        if role.get("preflight") is not None:
            if not isinstance(role["preflight"], str) or not role["preflight"].strip():
                raise ValueError("role preflight must be a nonempty shell command")
            item["preflight"] = role["preflight"]
        if "devices" in role:
            item.update(normalize_resources({key: role[key] for key in ("devices", "npu_count") if key in role}))
        elif "npu_count" in role or "devices" not in resources:
            item.update(normalize_resources({"npu_count": role.get("npu_count", resources.get("npu_count", 0))}))
        else:
            item.update(normalize_resources({"devices": resources["devices"]}))
        if "allow_external_busy" in role or "allow_external_busy" in resources:
            requested = {key: item[key] for key in ("devices", "npu_count") if key in item}
            requested["allow_external_busy"] = role.get("allow_external_busy", resources.get("allow_external_busy"))
            item.update(normalize_resources(requested))
        if role.get("service_port") is not None:
            item["service_port"] = normalize_resources({"service_port": role["service_port"]})["service_port"]
        elif resources.get("service_port") is not None and len(roles) == 0:
            item["service_port"] = resources["service_port"]
        for hw_key in ("cpu_cores", "memory_mb", "hbm_mb"):
            if hw_key in role:
                item[hw_key] = normalize_resources({hw_key: role[hw_key]})[hw_key]
            elif hw_key in resources:
                item[hw_key] = resources[hw_key]
        if "host" in role:
            if not isinstance(role["host"], str) or not role["host"].strip():
                raise ValueError("role host must be a nonempty hostname or IP address")
            item["host"] = role["host"]
        if "env" in role:
            item["env"] = validate_user_env(role["env"])
        roles.append(item)
    return roles


def distinct_hosts_required(roles: list[dict[str, Any]], topology: dict[str, Any] | None = None) -> bool:
    """Default allows same-host roles. Distinct hosts only when explicitly required."""
    if topology:
        if topology.get("distinct_hosts"):
            return True
        if topology.get("placement_policy") == POLICY_MULTI_NODE_ROCE:
            return True
        if topology.get("scale") == SCALE_S3:
            return True
    named = [str(role["host"]) for role in roles if role.get("host")]
    return len(set(named)) > 1


def host_key(row: dict[str, Any]) -> str:
    return str(row.get("host") or (row.get("host_endpoint") or {}).get("host") or "")


def _first_present_int(sources: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> int | None:
    for src in sources:
        for key in keys:
            val = src.get(key)
            if val is not None and not isinstance(val, bool):
                try:
                    return int(val)
                except (TypeError, ValueError):
                    continue
    return None


def _row_hardware_matches(row: dict[str, Any], role: dict[str, Any] | None) -> bool:
    if not role:
        return True
    wanted_cpu = int(role.get("cpu_cores") or 0)
    wanted_mem = int(role.get("memory_mb") or 0)
    wanted_hbm = int(role.get("hbm_mb") or 0)
    if wanted_cpu <= 0 and wanted_mem <= 0 and wanted_hbm <= 0:
        return True
    profile = row.get("profile") if isinstance(row.get("profile"), Mapping) else {}
    server = row.get("server") if isinstance(row.get("server"), Mapping) else (profile.get("server") if isinstance(profile.get("server"), Mapping) else {})
    hardware = row.get("hardware") if isinstance(row.get("hardware"), Mapping) else (profile.get("hardware") if isinstance(profile.get("hardware"), Mapping) else {})
    sources = [row, server, hardware, profile]
    if wanted_cpu > 0:
        have_cpu = _first_present_int(sources, ("cpu_cores",))
        if have_cpu is not None and have_cpu < wanted_cpu:
            return False
    if wanted_mem > 0:
        have_mem = _first_present_int(sources, ("mem_usable_mb", "mem_available_mb", "memory_mb", "mem_total_mb"))
        if have_mem is not None and have_mem < wanted_mem:
            return False
    if wanted_hbm > 0:
        have_hbm = _first_present_int(sources, ("hbm_free_mb", "vram_free_mb", "hbm_total_mb", "vram_total_mb", "hbm_mb"))
        if have_hbm is None and isinstance(row.get("npus"), Sequence) and not isinstance(row.get("npus"), (str, bytes)):
            have_hbm = int(normalize_remote_probe_spec(row)["vram_free_mb"])
        if have_hbm is not None and have_hbm < wanted_hbm:
            return False
    return True


def runtime_matches(row: dict[str, Any], environment: dict[str, Any] | None, role: dict[str, Any] | None = None) -> bool:
    environment = environment or {}
    profile = row.get("profile") or {}
    recipe = environment.get("recipe") or environment.get("image")
    if recipe and (row.get("recipe") or profile.get("recipe")) != recipe:
        return False
    if environment.get("python_abi") and (row.get("python_abi") or profile.get("python_abi")) != environment["python_abi"]:
        return False
    if environment.get("cann") and (row.get("cann") or profile.get("cann")) != environment["cann"]:
        return False
    if environment.get("soc") and (row.get("soc") or profile.get("soc")) != environment["soc"]:
        return False
    if environment.get("machine_type") and (row.get("machine_type") or profile.get("machine_type")) != environment["machine_type"]:
        return False
    if role and role.get("host") and host_key(row) != str(role["host"]):
        return False
    port = (role or {}).get("service_port")
    if port not in {None, 0}:
        declared = row.get("service_ports") or []
        if int(port) not in declared:
            return False
    if not _row_hardware_matches(row, role):
        return False
    return True


def select_execution_target(
    expectation: Mapping[str, Any] | str | None,
    *,
    local_spec: Mapping[str, Any] | None = None,
    fleet_hosts: Sequence[Mapping[str, Any]] = (),
    prefer_local: bool = True,
) -> dict[str, Any]:
    """Select `"local"`, `"remote"` (with `host` and `alias`), or `"reject"` from `fleet_probe` host records and `local_spec`."""
    return evaluate_placement(
        expectation,
        local_spec=local_spec,
        remote_specs=fleet_hosts,
        prefer_local=prefer_local,
    )


def _runtime_observation_rank(row: dict[str, Any], order: dict[str, int]) -> int:
    identities = [host_key(row), row.get("alias"), row.get("name"),
                  (row.get("endpoint") or {}).get("host") if isinstance(row.get("endpoint"), dict) else None]
    ranks = [order[str(value).strip().casefold()] for value in identities
             if value and str(value).strip().casefold() in order]
    return min(ranks) if ranks else len(order) + 1


def select_runtimes(
    catalog: list[dict[str, Any]],
    *,
    user: str,
    roles: list[dict[str, Any]],
    environment: dict[str, Any] | None = None,
    busy_runtime_ids: set[str] | None = None,
    topology: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pick one matching ready root per role.

    Same-host multi-role is allowed unless topology.distinct_hosts or the roles
    name different hosts. Distinct mutable roots are still required. An
    `npu-top` observation only reorders already-matching roots; explicit hosts
    keep their exact identity constraint.
    """
    busy_runtime_ids = busy_runtime_ids or set()
    ready = [
        row for row in catalog
        if row.get("state") == "ready" and row.get("user") == user and not row.get("draining")
        and row.get("runtime_id") not in busy_runtime_ids
    ]
    if observation:
        from mindie_coordinator.npu_top_bridge import observation_host_order
        order = observation_host_order(observation)
        if order:
            ready.sort(key=lambda row: _runtime_observation_rank(row, order))
    chosen: list[str] = []
    seen_hosts: set[str] = set()
    require_distinct_hosts = distinct_hosts_required(roles, topology)
    for role in roles:
        match = None
        for row in ready:
            rid = row["runtime_id"]
            if rid in chosen:
                continue
            if not runtime_matches(row, environment, role):
                continue
            host = host_key(row)
            if require_distinct_hosts and host and host in seen_hosts:
                continue
            match = row
            break
        if match is None:
            return {"runtime_ids": [], "reason": _gap(roles, ready, environment, topology)}
        chosen.append(match["runtime_id"])
        if host_key(match):
            seen_hosts.add(host_key(match))
    return {"runtime_ids": chosen, "reason": None}


def _gap(roles, ready, environment, topology=None) -> str:
    environment = environment or {}
    recipe = environment.get("recipe") or environment.get("image")
    if recipe and not provisionable_recipe(recipe) and not any(runtime_matches(row, environment) for row in ready):
        return f"unsupported environment recipe {recipe!r}; no matching prepared root"
    if distinct_hosts_required(roles, topology):
        hosts = {host_key(row) for row in ready if runtime_matches(row, environment)}
        if len(hosts) < len(roles):
            return "not enough distinct hosts with a matching prepared environment for this topology"
    return "no verified ready runtime matches the requested environment and hardware constraints"


def can_prepare(environment: dict[str, Any] | None, donor: dict[str, Any] | None,
                role: dict[str, Any] | None = None) -> bool:
    if donor is None:
        return False
    environment = environment or {}
    recipe = environment.get("recipe") or environment.get("image")
    have_recipe = donor.get("recipe") or (donor.get("profile") or {}).get("recipe")
    if recipe and have_recipe and have_recipe != recipe:
        return False
    if recipe and recipe not in SUPPORTED_RECIPES and have_recipe != recipe:
        return False
    if role and role.get("host") and host_key(donor) != str(role["host"]):
        return False
    for key in ("python_abi", "cann", "soc", "machine_type"):
        wanted = environment.get(key)
        if not wanted:
            continue
        have = donor.get(key) or (donor.get("profile") or {}).get(key)
        if have != wanted:
            return False
    if not _row_hardware_matches(donor, role):
        return False
    return True
