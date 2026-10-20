"""Coordinator-owned machine directory.

[Role] Consumers pass inventory *data* (a document) or the coordinator reads its own
store under ``coordinator_state_dir() / machines.json``. This module never
reads a consumer working-tree path.

[Timeouts & Locks] Inventory lock acquisition is bounded to 10 seconds.
Path-backed reads resolve the file instead of the write cache;
individual mutations and explicit replacement hold the adjacent
``machines.json.lock`` (the same protocol used by ``npu-top``'s inventory
bridge) across read-merge-atomic-replace.
[Invariants] An individual upsert merges into the latest on-disk record and
preserves C1 hardware/mount/tag fields the update omits. ``replace`` remains an
explicit whole-document replacement.
[Side-Effect Policy] Atomic inventory replacement and endpoint projection only.
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping

from mindie_coordinator.state_paths import coordinator_state_dir

MACHINES_FILENAME = "machines.json"
INVENTORY_LOCK_TIMEOUT_SECONDS = 10.0


class MachineDirectoryUnavailable(RuntimeError):
    """No machine directory document is available for this manager."""


@contextlib.contextmanager
def _inventory_lock(path: Path) -> Iterator[None]:
    """Serialize with every other writer of this inventory document.

    ``record_lock`` opens the adjacent ``machines.json.lock`` file and takes a
    process-wide plus OS file lock. Callers must not nest this context: a
    second flock on a separately opened descriptor would deadlock.
    """
    from remote_dev.core.locking import record_lock

    with record_lock(path, timeout_seconds=INVENTORY_LOCK_TIMEOUT_SECONDS):
        yield


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    """Write ``document`` through a same-directory temporary and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    previous_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if previous_mode is not None:
            os.chmod(temporary, previous_mode)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        # Windows has no portable directory-handle fsync; os.replace is the
        # durability boundary available there.
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _deep_merge(existing: Any, update: Any) -> Any:
    """Return ``update`` layered over ``existing`` without dropping nested keys."""
    if isinstance(existing, Mapping) and isinstance(update, Mapping):
        merged = {key: copy.deepcopy(value) for key, value in existing.items()}
        for key, value in update.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    return copy.deepcopy(update)


def _merge_mounts(existing: Any, update: Any) -> list[Any]:
    """Keep per-mount C1 metadata for targets the update still names."""
    if not isinstance(update, list):
        return copy.deepcopy(update)
    if not update or not isinstance(existing, list):
        return copy.deepcopy(update)
    known = {
        str(row.get("target")): row
        for row in existing
        if isinstance(row, Mapping) and row.get("target") is not None
    }
    merged = []
    for row in update:
        old = known.get(str(row.get("target"))) if isinstance(row, Mapping) else None
        merged.append(_deep_merge(old, row) if isinstance(old, Mapping) else copy.deepcopy(row))
    return merged


def _merge_machine_record(existing: Any, update: Mapping[str, Any]) -> dict[str, Any]:
    """Apply one individual update while preserving omitted C1-owned fields."""
    if not isinstance(existing, Mapping):
        return copy.deepcopy(dict(update))
    merged = copy.deepcopy(dict(update))
    for key, value in existing.items():
        # ``container`` is intentionally replace-on-upsert: the existing
        # unbind flow removes the key to clear a bound container. Every other
        # omitted field is C1 metadata and must survive a partial update.
        if key in update or key == "container":
            continue
        merged.setdefault(key, copy.deepcopy(value))
    for key in ("host", "hardware", "ssh"):
        old = existing.get(key)
        if key in update and isinstance(old, Mapping) and isinstance(update.get(key), Mapping):
            merged[key] = _deep_merge(old, update[key])
    if "mounts" in update:
        merged["mounts"] = _merge_mounts(existing.get("mounts"), update["mounts"])
    return merged


def _matching_indices(machines: list[Any], alias: Any, host_ip: Any) -> list[int]:
    indices: list[int] = []
    for index, row in enumerate(machines):
        if not isinstance(row, Mapping):
            continue
        if row.get("alias") == alias:
            indices.append(index)
        elif (
            alias == host_ip
            and row.get("alias") in {None, host_ip}
            and (row.get("host") or {}).get("ip") == host_ip
        ):
            indices.append(index)
    return indices


def _resolve_record(inventory: Mapping[str, Any], alias_or_host: str | None) -> dict[str, Any]:
    machines = [row for row in inventory["machines"] if isinstance(row, Mapping)]
    target = (alias_or_host or "").strip()
    if not target:
        default_alias = str(inventory.get("default_alias") or "").strip()
        if default_alias:
            target = default_alias
        elif len(machines) == 1:
            return dict(machines[0])
        else:
            raise ValueError("machine alias or host is required when multiple machines are registered")
    by_alias = [row for row in machines if row.get("alias") == target]
    if len(by_alias) == 1:
        return dict(by_alias[0])
    if len(by_alias) > 1:
        raise ValueError(f"machine alias {target!r} must resolve uniquely in the shared inventory")
    by_ip = [row for row in machines if (row.get("host") or {}).get("ip") == target]
    if len(by_ip) == 1:
        return dict(by_ip[0])
    if len(by_ip) > 1:
        raise ValueError(f"host IP {target!r} matches multiple machine records; specify alias")
    raise ValueError(f"unknown machine {target!r}; register it with mindie_hosts(action='upsert', ...)")


class MachineDirectory:
    def __init__(
        self,
        document: Mapping[str, Any] | None = None,
        *,
        path: Path | str | None = None,
    ):
        self._document = dict(document) if document is not None else None
        self._path = Path(path).expanduser() if path is not None else None
        # The legacy document-only constructor is an in-memory store. Once a
        # path exists (explicit or via replace/path()), reads use the file so
        # they cannot serve a stale snapshot after an external writer ran.
        self._memory_only = document is not None and path is None

    def path(self) -> Path:
        if self._path is None:
            self._path = coordinator_state_dir() / MACHINES_FILENAME
        return self._path

    def replace(self, document: Mapping[str, Any]) -> Path:
        """Persist a consumer-supplied inventory document into this store.

        This is the explicit whole-document replacement operation: no merge is
        performed. Individual mutations below use read-merge-atomic-replace.
        """
        if not isinstance(document, Mapping) or not isinstance(document.get("machines"), list):
            raise MachineDirectoryUnavailable("document is not a machine inventory")
        path = self.path().resolve()
        with _inventory_lock(path):
            return self._replace_unlocked(path, document)

    def _replace_unlocked(self, path: Path, document: Mapping[str, Any]) -> Path:
        if not isinstance(document, Mapping) or not isinstance(document.get("machines"), list):
            raise MachineDirectoryUnavailable("document is not a machine inventory")
        _atomic_write_json(path, document)
        self._document = dict(document)
        from mindie_coordinator.endpoint_bridge import sync_endpoints_file

        sync_endpoints_file(self._document, machines_path=path)
        return path

    def load(self) -> tuple[dict[str, Any], Path]:
        if self._memory_only and self._document is not None:
            inventory = self._document
            path = self._path if self._path is not None else Path("<memory>")
            if not isinstance(inventory.get("machines"), list):
                raise MachineDirectoryUnavailable("document is not a machine inventory")
            return dict(inventory), path
        path = self.path().resolve()
        if not path.is_file():
            if self._document is not None:
                inventory = self._document
                if not isinstance(inventory.get("machines"), list):
                    raise MachineDirectoryUnavailable("document is not a machine inventory")
                return dict(inventory), path
            raise MachineDirectoryUnavailable(
                f"machine directory is empty at {path}; pass a document to "
                "MachineDirectory.replace() or MachineDirectory(document=...)"
            )
        inventory = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(inventory, dict) or not isinstance(inventory.get("machines"), list):
            raise MachineDirectoryUnavailable(f"{path} is not a machine inventory document")
        return inventory, path

    def machines(self) -> list[dict[str, Any]]:
        inventory, _ = self.load()
        return list(inventory["machines"])

    def catalog(self) -> dict[str, Any]:
        inventory, path = self.load()
        return {
            "inventory_path": str(path),
            "machines": [
                {
                    "alias": row.get("alias"),
                    "host": row.get("host", {}).get("ip"),
                    "container_name": row.get("container", {}).get("name"),
                    "container_port": row.get("container", {}).get("ssh_port"),
                }
                for row in inventory["machines"]
            ],
        }

    def host(self, alias: str) -> dict[str, Any]:
        inventory, _ = self.load()
        matches = [row for row in inventory["machines"] if row.get("alias") == alias]
        if len(matches) != 1:
            raise ValueError("machine alias must resolve uniquely in the shared inventory")
        return matches[0]["host"]

    def record(self, alias_or_host: str | None = None) -> dict[str, Any]:
        """Resolve one machine record by alias, host IP, or configured default."""
        inventory, _ = self.load()
        return _resolve_record(inventory, alias_or_host)

    def upsert_machine(self, record: Mapping[str, Any], *, set_default: bool = False) -> Path:
        """Insert or merge one host record against the latest on-disk document.

        Never wipe distinct aliases sharing the same IP, and never drop C1
        hardware/mount/tag metadata that this partial update omits.
        """
        host_ip = (record.get("host") or {}).get("ip")
        alias = record.get("alias") or host_ip
        normalized = {**dict(record), "alias": alias}
        path = self.path().resolve()
        with _inventory_lock(path):
            try:
                inventory, _ = self.load()
            except MachineDirectoryUnavailable:
                inventory = {"machines": []}
            machines = list(inventory["machines"])
            matches = _matching_indices(machines, alias, host_ip)
            existing = machines[matches[0]] if matches else None
            merged = _merge_machine_record(existing, normalized)
            dropped = set(matches)
            machines = [row for index, row in enumerate(machines) if index not in dropped]
            machines.append(merged)
            document = {**inventory, "machines": machines}
            if set_default or (len(document["machines"]) == 1 and not document.get("default_alias")):
                document["default_alias"] = alias
            return self._replace_unlocked(path, document)

    def remove_machine(self, alias_or_host: str) -> tuple[Path, bool]:
        """Remove a machine by alias or host IP and re-sync endpoints.json."""
        target = alias_or_host.strip()
        if not target:
            raise ValueError("alias or host is required to remove a machine")
        path = self.path().resolve()
        with _inventory_lock(path):
            inventory, _ = self.load()
            before = list(inventory["machines"])
            remaining = [
                row
                for row in before
                if row.get("alias") != target and (row.get("host") or {}).get("ip") != target
            ]
            removed = len(remaining) < len(before)
            document = {**inventory, "machines": remaining}
            if document.get("default_alias") == target or (
                document.get("default_alias")
                and not any(r.get("alias") == document["default_alias"] for r in remaining)
            ):
                if len(remaining) == 1 and remaining[0].get("alias"):
                    document["default_alias"] = remaining[0]["alias"]
                else:
                    document.pop("default_alias", None)
            return self._replace_unlocked(path, document), removed

    def set_default(self, alias_or_host: str) -> Path:
        """Set `default_alias` to an existing machine and re-sync endpoints.json."""
        path = self.path().resolve()
        with _inventory_lock(path):
            inventory, _ = self.load()
            rec = _resolve_record(inventory, alias_or_host)
            alias = str(rec.get("alias") or (rec.get("host") or {}).get("ip") or "").strip()
            if not alias:
                raise ValueError("target machine record has no alias")
            document = {**inventory, "default_alias": alias}
            return self._replace_unlocked(path, document)

    def endpoints_document(self) -> dict[str, Any]:
        from mindie_coordinator.endpoint_bridge import build_endpoints_document

        try:
            inventory, _ = self.load()
        except MachineDirectoryUnavailable:
            return {"endpoints": {}}
        return build_endpoints_document(inventory)
