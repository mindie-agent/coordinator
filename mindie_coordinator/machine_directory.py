"""Coordinator-owned machine directory.

Consumers pass inventory *data* (a document) or the coordinator reads its own
store under ``coordinator_state_dir() / machines.json``. This module never
reads a consumer working-tree path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from mindie_coordinator.state_paths import coordinator_state_dir

MACHINES_FILENAME = "machines.json"


class MachineDirectoryUnavailable(RuntimeError):
    """No machine directory document is available for this manager."""


class MachineDirectory:
    def __init__(
        self,
        document: Mapping[str, Any] | None = None,
        *,
        path: Path | str | None = None,
    ):
        self._document = dict(document) if document is not None else None
        self._path = Path(path).expanduser() if path is not None else None

    def path(self) -> Path:
        if self._path is None:
            self._path = coordinator_state_dir() / MACHINES_FILENAME
        return self._path

    def replace(self, document: Mapping[str, Any]) -> Path:
        """Persist a consumer-supplied inventory document into this store."""
        if not isinstance(document, Mapping) or not isinstance(document.get("machines"), list):
            raise MachineDirectoryUnavailable("document is not a machine inventory")
        self._document = dict(document)
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        from mindie_coordinator.endpoint_bridge import sync_endpoints_file

        sync_endpoints_file(self._document, machines_path=path)
        return path

    def load(self) -> tuple[dict[str, Any], Path]:
        if self._document is not None:
            inventory = self._document
            path = self._path if self._path is not None else Path("<memory>")
            if not isinstance(inventory.get("machines"), list):
                raise MachineDirectoryUnavailable("document is not a machine inventory")
            return dict(inventory), path
        path = self.path().resolve()
        if not path.is_file():
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

    def upsert_machine(self, record: Mapping[str, Any], *, set_default: bool = False) -> Path:
        """Insert or replace one host record. Never wipe the rest of the directory."""
        host_ip = (record.get("host") or {}).get("ip")
        alias = record.get("alias") or host_ip
        normalized = {**dict(record), "alias": alias}
        try:
            inventory, _ = self.load()
            machines = [
                row
                for row in inventory["machines"]
                if (row.get("host") or {}).get("ip") != host_ip and row.get("alias") != alias
            ]
            machines.append(normalized)
            document = {**inventory, "machines": machines}
        except MachineDirectoryUnavailable:
            document = {"machines": [normalized]}
        if set_default or (len(document["machines"]) == 1 and not document.get("default_alias")):
            document["default_alias"] = alias
        return self.replace(document)

    def remove_machine(self, alias_or_host: str) -> tuple[Path, bool]:
        """Remove a machine by alias or host IP and re-sync endpoints.json."""
        target = alias_or_host.strip()
        if not target:
            raise ValueError("alias or host is required to remove a machine")
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
        path = self.replace(document)
        return path, removed

    def set_default(self, alias_or_host: str) -> Path:
        """Set `default_alias` to an existing machine and re-sync endpoints.json."""
        rec = self.record(alias_or_host)
        alias = str(rec.get("alias") or (rec.get("host") or {}).get("ip") or "").strip()
        if not alias:
            raise ValueError("target machine record has no alias")
        inventory, _ = self.load()
        document = {**inventory, "default_alias": alias}
        return self.replace(document)

    def endpoints_document(self) -> dict[str, Any]:
        from mindie_coordinator.endpoint_bridge import build_endpoints_document

        try:
            inventory, _ = self.load()
        except MachineDirectoryUnavailable:
            return {"endpoints": {}}
        return build_endpoints_document(inventory)
