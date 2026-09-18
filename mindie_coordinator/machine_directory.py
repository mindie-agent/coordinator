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

    def upsert_machine(self, record: Mapping[str, Any]) -> Path:
        """Insert or replace one host record. Never wipe the rest of the directory."""
        host_ip = (record.get("host") or {}).get("ip")
        try:
            inventory, _ = self.load()
            machines = [row for row in inventory["machines"]
                        if (row.get("host") or {}).get("ip") != host_ip and row.get("alias") != record.get("alias")]
            machines.append(dict(record))
            document = {**inventory, "machines": machines}
        except MachineDirectoryUnavailable:
            document = {"machines": [dict(record)]}
        return self.replace(document)
