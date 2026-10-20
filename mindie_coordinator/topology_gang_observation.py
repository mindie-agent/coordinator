"""Physical A2/A3 topology parsing for gang lease planning.

[Role]
Converts real `npu-smi info` captures into host/card/Die observations keyed by
physical `Phy-ID`, preserving process attribution so occupied Dies are excluded.

[Invariants]
- A2 has one Chip per NPU card; A3 exposes two sibling Chips (`2k`, `2k+1`).
- A Die is free only when health is OK, HBM is below 8192 MB, and no active
  process row remains. `VLLMWorker_*`, `VLLMEngine*`, and user processes all
  block the Die. A fleet snapshot that already classified a Die busy is kept
  busy even when its process attribution is incomplete. Phy-ID collisions fail
  the parse instead of guessing.
- Parsing is read-only and never infers free capacity from logical NPU IDs.

[Timeouts & Locks]
- Pure in-memory parsing; no network, subprocess, locks, or sleeps.

[Side-Effect Policy]
- Side-effect free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from mindie_coordinator.fleet_probe import BUSY_HBM_THRESHOLD_MB, parse_npu_smi_info
from mindie_coordinator.topology_gang_common import (
    GangLeaseError,
    SCALE_S0,
    SCALE_S1,
    SCALE_S2,
    SCALE_S3,
)


@dataclass(frozen=True)
class ProcessObservation:
    pid: int
    name: str
    memory_mb: int = 0
    owner: str = ""
    container_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "name": self.name,
            "memory_mb": self.memory_mb,
            "owner": self.owner,
            "container_name": self.container_name,
        }


@dataclass(frozen=True)
class DieObservation:
    host: str
    npu_id: int
    chip_id: int
    phy_id: int
    chip_name: str
    health: str
    hbm_used_mb: int
    hbm_total_mb: int
    processes: tuple[ProcessObservation, ...] = ()
    observed_busy: bool = False

    @property
    def exclusion_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.health.strip().upper() != "OK":
            reasons.append(f"health={self.health or 'unknown'}")
        if self.hbm_used_mb >= BUSY_HBM_THRESHOLD_MB:
            reasons.append(
                f"hbm={self.hbm_used_mb}/{self.hbm_total_mb}MB >= {BUSY_HBM_THRESHOLD_MB}MB"
            )
        for proc in self.processes:
            name = proc.name.strip() or "<unnamed>"
            if name.lower().startswith("vllmworker"):
                reasons.append(f"active VLLMWorker {name} pid={proc.pid}")
            else:
                owner = f" owner={proc.owner}" if proc.owner else ""
                reasons.append(f"active user process {name} pid={proc.pid}{owner}")
        if self.observed_busy and not reasons:
            # The authoritative npu-smi aggregate saw activity that this
            # observation cannot attribute (for example AICore-only use).
            # Never turn incomplete attribution into free capacity.
            reasons.append("npu-smi reported this Die busy")
        return tuple(dict.fromkeys(reasons))

    @property
    def free(self) -> bool:
        return not self.exclusion_reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "npu_id": self.npu_id,
            "chip_id": self.chip_id,
            "phy_id": self.phy_id,
            "chip_name": self.chip_name,
            "health": self.health,
            "hbm_used_mb": self.hbm_used_mb,
            "hbm_total_mb": self.hbm_total_mb,
            "observed_busy": self.observed_busy,
            "free": self.free,
            "exclusion_reasons": list(self.exclusion_reasons),
            "processes": [proc.to_dict() for proc in self.processes],
        }


@dataclass(frozen=True)
class CardObservation:
    host: str
    npu_id: int
    chip_name: str
    phy_ids: tuple[int, ...]
    dies: tuple[DieObservation, ...]

    @property
    def free_dies(self) -> tuple[DieObservation, ...]:
        return tuple(die for die in self.dies if die.free)

    @property
    def free_phy_ids(self) -> tuple[int, ...]:
        return tuple(sorted(die.phy_id for die in self.free_dies))

    @property
    def fully_free(self) -> bool:
        return bool(self.dies) and len(self.free_dies) == len(self.dies)

    @property
    def sibling_pair(self) -> bool:
        return (
            len(self.phy_ids) == 2
            and self.phy_ids[0] % 2 == 0
            and self.phy_ids[1] == self.phy_ids[0] + 1
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "npu_id": self.npu_id,
            "chip_name": self.chip_name,
            "phy_ids": list(self.phy_ids),
            "free_phy_ids": list(self.free_phy_ids),
            "fully_free": self.fully_free,
            "sibling_pair": self.sibling_pair,
            "dies": [die.to_dict() for die in self.dies],
        }


@dataclass(frozen=True)
class HostTopology:
    host: str
    cards: tuple[CardObservation, ...]
    driver_version: str = ""
    npu_smi_version: str = ""

    @property
    def dies(self) -> tuple[DieObservation, ...]:
        return tuple(die for card in self.cards for die in card.dies)

    @property
    def free_phy_ids(self) -> tuple[int, ...]:
        return tuple(sorted(die.phy_id for die in self.dies if die.free))

    @property
    def busy_phy_ids(self) -> tuple[int, ...]:
        return tuple(sorted(die.phy_id for die in self.dies if not die.free))

    @property
    def free_die_map(self) -> dict[int, DieObservation]:
        return {die.phy_id: die for die in self.dies if die.free}

    @property
    def free_dual_die_pairs(self) -> tuple[tuple[int, int], ...]:
        pairs = [
            card.phy_ids
            for card in self.cards
            if card.sibling_pair and card.fully_free
        ]
        return tuple(sorted(pairs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "driver_version": self.driver_version,
            "npu_smi_version": self.npu_smi_version,
            "free_phy_ids": list(self.free_phy_ids),
            "busy_phy_ids": list(self.busy_phy_ids),
            "free_dual_die_pairs": [list(pair) for pair in self.free_dual_die_pairs],
            "cards": [card.to_dict() for card in self.cards],
        }


def parse_npu_smi_capture(
    raw: str,
    *,
    host: str = "",
    pid_owners: Mapping[int, str] | None = None,
    pid_containers: Mapping[int, str] | None = None,
) -> HostTopology:
    """Parse a real A2/A3 `npu-smi info` capture into physical card topology."""
    parsed = parse_npu_smi_info(
        raw, pid_owners=pid_owners or {}, pid_containers=pid_containers or {}
    )
    npus = parsed.get("npus") or []
    if not npus:
        raise GangLeaseError("parse_failed", f"no NPU cards parsed for {host or '<unknown>'}")

    cards: list[CardObservation] = []
    seen_phy: dict[int, int] = {}
    for npu in npus:
        npu_id = int(npu["npu_id"])
        known_phy_ids = {int(die["phy_id"]) for die in npu.get("dies") or []}
        processes_by_phy: dict[int, list[ProcessObservation]] = {}
        unattributed: list[ProcessObservation] = []
        for proc in npu.get("processes") or []:
            phy = proc.get("phy_id")
            observation = ProcessObservation(
                pid=int(proc.get("pid") or 0),
                name=str(proc.get("process_name") or ""),
                memory_mb=int(proc.get("memory_mb") or 0),
                owner=str(proc.get("owner") or ""),
                container_name=str(proc.get("container_name") or ""),
            )
            if phy is None or int(phy) not in known_phy_ids:
                # Unknown attribution is unsafe: block every Die on this NPU.
                unattributed.append(observation)
            else:
                processes_by_phy.setdefault(int(phy), []).append(observation)
        dies: list[DieObservation] = []
        for die in npu.get("dies") or []:
            phy_id = int(die["phy_id"])
            if phy_id in seen_phy:
                raise GangLeaseError(
                    "parse_failed",
                    f"Phy-ID {phy_id} repeats on {host or '<unknown>'} (NPU {npu_id})",
                )
            seen_phy[phy_id] = npu_id
            procs = tuple(
                dict.fromkeys([*processes_by_phy.get(phy_id, []), *unattributed])
            )
            dies.append(
                DieObservation(
                    host=host,
                    npu_id=npu_id,
                    chip_id=int(die.get("chip_id", 0)),
                    phy_id=phy_id,
                    chip_name=str(npu.get("chip_name") or "Ascend"),
                    health=str(npu.get("health") or "Unknown"),
                    hbm_used_mb=int(die.get("hbm_used_mb") or 0),
                    hbm_total_mb=int(die.get("hbm_total_mb") or 0),
                    processes=procs,
                    observed_busy=bool(die.get("busy")),
                )
            )
        if not dies:
            raise GangLeaseError(
                "parse_failed", f"NPU {npu_id} has no parsed Dies on {host or '<unknown>'}"
            )
        cards.append(
            CardObservation(
                host=host,
                npu_id=npu_id,
                chip_name=str(npu.get("chip_name") or "Ascend"),
                phy_ids=tuple(sorted(die.phy_id for die in dies)),
                dies=tuple(dies),
            )
        )
    return HostTopology(
        host=host,
        cards=tuple(sorted(cards, key=lambda card: card.npu_id)),
        driver_version=str(parsed.get("driver_version") or ""),
        npu_smi_version=str(parsed.get("npu_smi_version") or parsed.get("smi_version") or ""),
    )


def topology_from_fleet_snapshot(
    snapshot: Mapping[str, Any], *, host: str = ""
) -> HostTopology:
    """Adapt one persisted `fleet_probe` snapshot to physical card topology.

    This is the production observation bridge: `fleet_probe` owns SSH and
    `npu-smi` parsing, while this module owns the physical Die/card contract
    consumed by the gang planner. It never re-derives a busy Die as free when
    the snapshot only has partial process attribution.
    """
    if not isinstance(snapshot, Mapping):
        raise GangLeaseError(
            "observation_invalid", "fleet topology snapshot must be an object"
        )
    resolved_host = str(
        host or snapshot.get("host") or snapshot.get("alias") or ""
    ).strip()
    if not snapshot.get("online"):
        detail = str(snapshot.get("error") or "host is offline")[:500]
        raise GangLeaseError(
            "observation_unavailable",
            f"fleet probe for {resolved_host or '<unknown>'} is unavailable: {detail}",
            host=resolved_host,
        )
    npus = snapshot.get("npus")
    if not isinstance(npus, (list, tuple)) or not npus:
        raise GangLeaseError(
            "observation_unavailable",
            f"fleet probe for {resolved_host or '<unknown>'} has no NPU topology",
            host=resolved_host,
        )

    cards: list[CardObservation] = []
    seen_phy: dict[int, int] = {}
    seen_npu: set[int] = set()
    for npu in npus:
        if not isinstance(npu, Mapping):
            raise GangLeaseError("observation_invalid", "fleet NPU row must be an object")
        try:
            npu_id = int(npu["npu_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GangLeaseError(
                "observation_invalid", "fleet NPU row has no integer npu_id"
            ) from exc
        if npu_id in seen_npu:
            raise GangLeaseError(
                "observation_invalid",
                f"NPU {npu_id} repeats on {resolved_host or '<unknown>'}",
            )
        seen_npu.add(npu_id)
        chip_name = str(npu.get("chip_name") or "Ascend")
        health = str(npu.get("health") or "Unknown")
        processes_by_phy: dict[int, list[ProcessObservation]] = {}
        unattributed: list[ProcessObservation] = []
        raw_processes = npu.get("processes")
        if raw_processes is not None and not isinstance(raw_processes, (list, tuple)):
            raise GangLeaseError(
                "observation_invalid", f"NPU {npu_id} processes must be an array"
            )
        for proc in raw_processes or []:
            if not isinstance(proc, Mapping):
                raise GangLeaseError(
                    "observation_invalid", f"NPU {npu_id} process row must be an object"
                )
            observation = ProcessObservation(
                pid=int(proc.get("pid") or 0),
                name=str(proc.get("process_name") or proc.get("name") or ""),
                memory_mb=int(proc.get("memory_mb") or 0),
                owner=str(proc.get("owner") or ""),
                container_name=str(proc.get("container_name") or ""),
            )
            phy = proc.get("phy_id")
            if phy is None:
                unattributed.append(observation)
            else:
                processes_by_phy.setdefault(int(phy), []).append(observation)

        raw_dies = npu.get("dies")
        if not isinstance(raw_dies, (list, tuple)) or not raw_dies:
            # A card without a parsed Die row is unobserved, not free.
            raise GangLeaseError(
                "observation_unavailable",
                f"NPU {npu_id} has no parsed Die rows on {resolved_host or '<unknown>'}",
                host=resolved_host,
            )
        dies: list[DieObservation] = []
        for die in raw_dies:
            if not isinstance(die, Mapping):
                raise GangLeaseError(
                    "observation_invalid", f"NPU {npu_id} Die row must be an object"
                )
            try:
                phy_id = int(die["phy_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise GangLeaseError(
                    "observation_invalid",
                    f"NPU {npu_id} Die row has no integer phy_id",
                ) from exc
            if phy_id in seen_phy:
                raise GangLeaseError(
                    "observation_invalid",
                    f"Phy-ID {phy_id} repeats on {resolved_host or '<unknown>'}",
                )
            seen_phy[phy_id] = npu_id
            dies.append(
                DieObservation(
                    host=resolved_host,
                    npu_id=npu_id,
                    chip_id=int(die.get("chip_id") or 0),
                    phy_id=phy_id,
                    chip_name=chip_name,
                    health=health,
                    hbm_used_mb=int(die.get("hbm_used_mb") or 0),
                    hbm_total_mb=int(die.get("hbm_total_mb") or 0),
                    processes=tuple(
                        dict.fromkeys(
                            [*processes_by_phy.get(phy_id, []), *unattributed]
                        )
                    ),
                    observed_busy=bool(die.get("busy")),
                )
            )
        cards.append(
            CardObservation(
                host=resolved_host,
                npu_id=npu_id,
                chip_name=chip_name,
                phy_ids=tuple(sorted(die.phy_id for die in dies)),
                dies=tuple(dies),
            )
        )
    return HostTopology(
        host=resolved_host,
        cards=tuple(sorted(cards, key=lambda card: card.npu_id)),
        driver_version=str(snapshot.get("driver_version") or ""),
        npu_smi_version=str(
            snapshot.get("npu_smi_version") or snapshot.get("smi_version") or ""
        ),
    )


def classify_topology_scale(die_count: int, *, node_count: int = 1) -> str:
    """Map a production Die/node count to S0-S3 without guessing 16-Die S2/S3."""
    if type(die_count) is not int or die_count < 1:
        raise GangLeaseError("invalid_spec", "die_count must be a positive integer")
    if type(node_count) is not int or node_count < 1:
        raise GangLeaseError("invalid_spec", "node_count must be a positive integer")
    if die_count == 1 and node_count == 1:
        return SCALE_S0
    if 2 <= die_count <= 4 and node_count == 1:
        return SCALE_S1
    if 8 <= die_count <= 16 and node_count == 1:
        return SCALE_S2
    if 16 <= die_count <= 32 and node_count >= 2:
        return SCALE_S3
    raise GangLeaseError(
        "invalid_spec",
        f"no production scale for {die_count} Dies across {node_count} node(s)",
    )


__all__ = [
    "BUSY_HBM_THRESHOLD_MB",
    "CardObservation",
    "DieObservation",
    "HostTopology",
    "ProcessObservation",
    "classify_topology_scale",
    "parse_npu_smi_capture",
    "topology_from_fleet_snapshot",
]
