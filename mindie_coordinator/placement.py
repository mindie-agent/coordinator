"""Choose hosts and prepared roots from attested catalog + constraints.

The host NPU coordinator remains the only device/port authority. This module
never relaxes topology, recipe, ABI or hardware constraints, and never starts
a partial group.
"""

from __future__ import annotations

import math
from typing import Any

SUPPORTED_RECIPES = {"rc", "main", "stable", "local-latest"}
RESERVED_ENV = {"ASCEND_RT_VISIBLE_DEVICES", "MINDIE_SERVICE_PORT", "MINDIE_PYTHON", "MINDIE_EXECUTION_OBSERVATION"}
ENVIRONMENT_KEYS = {"recipe", "image", "python_abi", "cann", "soc", "machine_type"}
RESOURCE_KEYS = {"devices", "npu_count", "service_port", "allow_external_busy"}
TOPOLOGY_KEYS = {"host", "roles", "distinct_hosts"}
ROLE_KEYS = {"name", "command", "preflight", "devices", "npu_count", "service_port", "allow_external_busy", "host", "env"}


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
    return resources


def role_plan(topology: dict[str, Any] | None, resources: dict[str, Any], command: str) -> list[dict[str, Any]]:
    topology = checked_mapping(topology, "topology", TOPOLOGY_KEYS)
    if "distinct_hosts" in topology and type(topology["distinct_hosts"]) is not bool:
        raise ValueError("topology.distinct_hosts must be a boolean")
    if "host" in topology and (not isinstance(topology["host"], str) or not topology["host"].strip()):
        raise ValueError("topology.host must be a nonempty hostname or IP address")
    if "host" in topology and "roles" in topology:
        raise ValueError("use topology.host for one default role, or topology.roles with host in each role; do not combine them")
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
    if topology and topology.get("distinct_hosts"):
        return True
    named = [str(role["host"]) for role in roles if role.get("host")]
    return len(set(named)) > 1


def host_key(row: dict[str, Any]) -> str:
    return str(row.get("host") or (row.get("host_endpoint") or {}).get("host") or "")


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
    return True


def select_runtimes(
    catalog: list[dict[str, Any]],
    *,
    user: str,
    roles: list[dict[str, Any]],
    environment: dict[str, Any] | None = None,
    busy_runtime_ids: set[str] | None = None,
    topology: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pick one matching ready root per role.

    Same-host multi-role is allowed unless topology.distinct_hosts or the roles
    name different hosts. Distinct mutable roots are still required.
    """
    busy_runtime_ids = busy_runtime_ids or set()
    ready = [
        row for row in catalog
        if row.get("state") == "ready" and row.get("user") == user and not row.get("draining")
        and row.get("runtime_id") not in busy_runtime_ids
    ]
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
    return True
