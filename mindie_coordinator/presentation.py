"""Small task-facing observations; full operation records remain locally readable."""
from __future__ import annotations

import json
from pathlib import Path

EXECUTION_KEYS = ("execution_id", "state", "service", "error", "error_ref", "error_details", "reason", "progress",
                  "source_snapshot_id", "sources", "role_progress",
                  "observed_at", "cancel_requested", "service_port", "provisioning_started",
                  "worktrees_preserved", "resources_released", "stdout", "stderr", "tail", "preparation_logs", "observation_freshness",
                  "runtime_update", "active_executions", "wait_until", "wait_timed_out", "wait_error", "logs_pending", "tail_error", "evidence")
ROLE_KEYS = ("name", "state", "runtime_id", "host", "root", "service_port", "error", "error_details", "lease_state",
             "quiet", "descendants_drained", "stdout", "stderr", "status_observed_at", "command_timings")
TEXT_KEYS = {"stdout", "stderr", "tail", "tail_error", "wait_error", "error", "reason", "summary", "warnings"}
MAX_COMPACT_BYTES = 16000


def _role_attention(role):
    """When sampling roles, show failures and unfinished cleanup before quiet successes."""
    return (bool(role.get("error")) or role.get("state") in {"failed", "timeout", "cancelled", "inconclusive", "uncertain"},
            role.get("quiet") is False,
            role.get("lease_state") not in {None, "released", "cancelled", "expired"})


def execution_summary(value: dict, *, target=False) -> dict:
    result = {key: value[key] for key in EXECUTION_KEYS if value.get(key) is not None}
    if "id" in value and "execution_id" not in result:
        result["execution_id"] = value["id"]
        result["state"] = value.get("phase")
    roles = value.get("roles") or []
    if roles:
        selected = sorted(roles, key=_role_attention, reverse=True)[:8] if len(roles) > 8 else roles
        result["roles"] = [{key: role[key] for key in ROLE_KEYS if role.get(key) is not None}
                           for role in selected]
        if len(roles) > 8:
            result["roles_total"] = len(roles)
    if target and value.get("target"):
        result["target"] = value["target"]
    return result


def compact_data(value: dict, *, target=False) -> dict:
    if "session" in value:
        session = value["session"]
        result = {"session": {key: session[key] for key in ("id", "state", "sources") if key in session},
                  "context_file": value.get("context_file"),
                  "attachments_count": len(value.get("attachments") or [])}
        if "source_defaults" in value:
            result["source_defaults"] = value["source_defaults"]
    else:
        result = execution_summary(value, target=target)
    if "executions" in value:
        rows = value["executions"]
        result["executions"] = [execution_summary(row) for row in rows[-5:]]
        result["executions_total"] = len(rows)
    for key in ("notifications", "notification_status", "coordination_contacts", "coordination_peers", "message"):
        if key in value:
            result[key] = value[key]
    return result


def compact_runtime(runtime):
    """Collapse repeated healthy package identities, retaining diagnostic facts."""
    if not isinstance(runtime, dict) or not runtime or set(runtime) - {"client", "daemon"}:
        return runtime
    packages = {}
    identities = {}
    python = None
    for records in runtime.values():
        if not isinstance(records, list) or not records:
            return runtime
        for record in records:
            if not isinstance(record, dict):
                return runtime
            loaded = record.get("loaded") or {}
            if not isinstance(loaded, dict):
                return runtime
            package, version = loaded.get("package"), loaded.get("version")
            if record.get("status") != "current" or not package or not version or not loaded.get("python"):
                return runtime
            identity = (version, loaded.get("commit"), loaded.get("location"))
            if package in identities and identities[package] != identity:
                return runtime
            if python is not None and python != loaded["python"]:
                return runtime
            identities[package] = identity
            packages[package] = {"version": version, "commit": loaded.get("commit")}
            python = loaded["python"]
    return {"status": "current", "packages": packages, "python": python}


def present(result: dict, directory: Path, *, full=False, target=False) -> dict:
    """Save the actual response before projecting it. A write failure never undoes an operation."""
    path = Path(directory) / "results" / (result["invocation_id"] + ".json")
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
        result = {**result, "record_ref": str(path)}
    except OSError as exc:
        result = {**result, "warnings": [*result.get("warnings", []), f"Full record write failed: {exc}"]}
    if full:
        return result
    data = compact_data(result.get("data") or {}, target=target)
    # Shrink prose, never identifiers, state, release facts or usable references.
    # Coordination messages and an explicit target remain exact; their consumers
    # must not receive silently edited message text or executable shell payloads.
    def bound(value, limit, key=""):
        if isinstance(value, str) and key in TEXT_KEYS:
            return value[-limit:] if len(value) > limit else value
        if isinstance(value, dict):
            return {name: bound(item, limit, name) for name, item in value.items()}
        if isinstance(value, list):
            return [bound(item, limit, key) for item in value[:8]]
        return value
    protected = {key: data[key] for key in ("target", "notifications", "message") if key in data}
    projected = {key: value for key, value in data.items() if key not in protected}
    compact = {**result, "data": {**projected, **protected}}
    if not target and "runtime" in compact:
        compact["runtime"] = compact_runtime(compact["runtime"])
    for limit in (2000, 500, 100):
        bounded = bound(projected, limit)
        compact["data"] = {**bounded, **protected}
        compact["summary"] = bound(result["summary"], limit, "summary")
        if bounded != projected or compact["summary"] != result["summary"]:
            compact["detail_omitted"] = True
        if len(json.dumps(compact, ensure_ascii=False).encode()) <= MAX_COMPACT_BYTES:
            break
    if len(json.dumps(compact, ensure_ascii=False).encode()) > MAX_COMPACT_BYTES:
        # Optional context can be retrieved from the full record. Keep all
        # execution/role outcomes, cleanup facts, failure excerpts and log refs.
        for key in ("target", "sources", "source_defaults", "coordination_contacts", "coordination_peers"):
            compact["data"].pop(key, None)
        compact["detail_omitted"] = True
    return compact
