"""Topology-aware multi-Die gang lease planning for Ascend A2/A3 clusters.

[Role]
Pure planner for production topology scales S0 (1 Die), S1 (2-4 intra-node
Dies), S2 (8-16 full-node / same-node PD Dies), and S3 (2+ node RoCE gangs).
It maps real A2/A3 `Phy-ID`s to non-overlapping `ASCEND_RT_VISIBLE_DEVICES`
groups for same-card dual-Die, intra-node HCCS, PD split, and multi-node gangs.

[Invariants]
- The host-side NPU queue remains the sole lease authority; a GangLease is a
  validated placement plan, never a reservation or process launch.
- A2 is one Die per card; A3 cards own sibling `(2k, 2k+1)` pairs. A pair is
  allocatable only when both Dies are free. Any active process (including
  `VLLMWorker_*`) or HBM >= 8192 MB excludes the Die.
- PD groups are disjoint; each emits its own visible-device string. Multi-node
  hosts keep per-host Phy-IDs instead of flattening colliding IDs.
- Stale `/dev/shm/hccl*` entries are read-only candidates. No command touches
  containers, signals PIDs, or deletes shared memory.

[Timeouts & Locks]
- Pure in-memory planning; callers own SSH/probe budgets. No locks or sleeps.

[Side-Effect Policy]
- Planning is side-effect free. Hygiene/cleanup helpers are command builders;
  cleanup removes only one validated `/tmp/mindie-npu-test-<uuid>` path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mindie_coordinator.topology_gang_common import (
    GangLeaseError,
    PLACEMENT_POLICIES,
    POLICY_DISAGGREGATED_PD_PAIR,
    POLICY_INTRA_NODE_HCCS,
    POLICY_MULTI_NODE_ROCE,
    POLICY_SAME_NPU_DUAL_DIE,
    POLICY_SCALES,
    SANDBOX_PREFIX,
    SCALE_DIE_BOUNDS,
    SCALES,
    SCALE_S0,
    SCALE_S1,
    SCALE_S2,
    SCALE_S3,
    SHM_STALE_AFTER_SECONDS,
)
from mindie_coordinator.topology_gang_hygiene import (
    HygieneReport,
    SANDBOX_PATTERN,
    SandboxEntry,
    ShmEntry,
    cleanup_sandbox_command,
    parse_sandbox_hygiene,
    sandbox_hygiene_command,
    validate_sandbox_path,
)
from mindie_coordinator.topology_gang_observation import (
    BUSY_HBM_THRESHOLD_MB,
    CardObservation,
    DieObservation,
    HostTopology,
    ProcessObservation,
    classify_topology_scale,
    parse_npu_smi_capture,
    topology_from_fleet_snapshot,
)


@dataclass(frozen=True)
class TopologySpec:
    """Validated production topology contract for one gang lease."""

    scale: str
    die_count: int
    placement_policy: str
    prefill_die_count: int | None = None
    decode_die_count: int | None = None
    node_count: int | None = None
    visible_phy_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scale, str) or self.scale not in SCALE_DIE_BOUNDS:
            raise GangLeaseError("invalid_spec", f"unsupported scale: {self.scale!r}")
        if type(self.die_count) is not int:
            raise GangLeaseError("invalid_spec", "die_count must be an integer")
        low, high = SCALE_DIE_BOUNDS[self.scale]
        if not low <= self.die_count <= high:
            raise GangLeaseError(
                "invalid_spec",
                f"{self.scale} requires {low}..{high} Dies, got {self.die_count}",
            )
        if (
            not isinstance(self.placement_policy, str)
            or self.placement_policy not in PLACEMENT_POLICIES
        ):
            raise GangLeaseError(
                "invalid_spec", f"unsupported placement_policy: {self.placement_policy!r}"
            )
        if self.scale not in POLICY_SCALES[self.placement_policy]:
            raise GangLeaseError(
                "unsupported_placement",
                f"{self.placement_policy} does not apply to {self.scale}",
            )

        if self.scale == SCALE_S3:
            nodes = 2 if self.node_count is None else self.node_count
            if type(nodes) is not int or nodes < 2:
                raise GangLeaseError("invalid_spec", "S3 node_count must be >= 2")
            if nodes > self.die_count:
                raise GangLeaseError("invalid_spec", "S3 node_count exceeds die_count")
            object.__setattr__(self, "node_count", nodes)
        else:
            if self.node_count not in (None, 1):
                raise GangLeaseError("invalid_spec", f"{self.scale} is single-node")
            object.__setattr__(self, "node_count", 1)

        if self.placement_policy == POLICY_SAME_NPU_DUAL_DIE and self.die_count % 2:
            raise GangLeaseError(
                "invalid_spec", "same_npu_dual_die requires an even Die count"
            )

        if self.placement_policy == POLICY_DISAGGREGATED_PD_PAIR:
            if self.die_count < 2 or self.die_count % 2:
                raise GangLeaseError(
                    "invalid_spec", "disaggregated_pd_pair requires an even Die count >= 2"
                )
            prefill, decode = self.prefill_die_count, self.decode_die_count
            if prefill is None and decode is None:
                prefill = decode = self.die_count // 2
                object.__setattr__(self, "prefill_die_count", prefill)
                object.__setattr__(self, "decode_die_count", decode)
            elif prefill is None or decode is None:
                raise GangLeaseError(
                    "invalid_spec", "both prefill_die_count and decode_die_count are required"
                )
            if (
                type(prefill) is not int
                or type(decode) is not int
                or prefill < 1
                or decode < 1
                or prefill + decode != self.die_count
            ):
                raise GangLeaseError(
                    "invalid_spec",
                    "prefill_die_count/decode_die_count must be positive and sum to die_count",
                )
        elif self.prefill_die_count is not None or self.decode_die_count is not None:
            raise GangLeaseError(
                "invalid_spec", "prefill/decode counts only apply to disaggregated_pd_pair"
            )

        if self.visible_phy_ids is not None:
            if isinstance(self.visible_phy_ids, (str, bytes)):
                raise GangLeaseError("invalid_spec", "visible_phy_ids must be an integer sequence")
            ids = tuple(self.visible_phy_ids)
            if any(type(phy) is not int or phy < 0 for phy in ids):
                raise GangLeaseError(
                    "invalid_spec", "visible_phy_ids must be nonnegative integers"
                )
            if len(set(ids)) != len(ids):
                raise GangLeaseError("invalid_spec", "visible_phy_ids must be distinct")
            if len(ids) != self.die_count:
                raise GangLeaseError(
                    "invalid_spec", "visible_phy_ids length must equal die_count"
                )
            if self.scale == SCALE_S3:
                raise GangLeaseError(
                    "invalid_spec", "visible_phy_ids is ambiguous across S3 hosts"
                )
            object.__setattr__(self, "visible_phy_ids", tuple(sorted(ids)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TopologySpec":
        if not isinstance(value, Mapping):
            raise GangLeaseError("invalid_spec", "topology_spec must be a mapping")
        supported = {
            "scale",
            "die_count",
            "placement_policy",
            "prefill_die_count",
            "decode_die_count",
            "node_count",
            "visible_phy_ids",
        }
        unknown = set(value) - supported
        if unknown:
            raise GangLeaseError(
                "invalid_spec",
                f"unsupported topology_spec fields: {', '.join(sorted(map(str, unknown)))}",
            )
        missing = {"die_count", "placement_policy"} - set(value)
        if missing:
            raise GangLeaseError(
                "invalid_spec", f"missing topology_spec fields: {', '.join(sorted(missing))}"
            )
        visible = value.get("visible_phy_ids")
        if visible is not None and not isinstance(visible, (list, tuple)):
            raise GangLeaseError("invalid_spec", "visible_phy_ids must be a list or tuple")
        die_count = value["die_count"]
        placement_policy = str(value["placement_policy"])
        scale = value.get("scale")
        node_count = value.get("node_count")
        if scale is None:
            if type(die_count) is not int:
                raise GangLeaseError("invalid_spec", "die_count must be an integer")
            inferred_nodes = node_count
            if inferred_nodes is None:
                inferred_nodes = 2 if placement_policy == POLICY_MULTI_NODE_ROCE else 1
            scale = classify_topology_scale(die_count, node_count=inferred_nodes)
        return cls(
            scale=str(scale),
            die_count=die_count,
            placement_policy=placement_policy,
            prefill_die_count=value.get("prefill_die_count"),
            decode_die_count=value.get("decode_die_count"),
            node_count=node_count,
            visible_phy_ids=tuple(visible) if visible is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "die_count": self.die_count,
            "placement_policy": self.placement_policy,
            "prefill_die_count": self.prefill_die_count,
            "decode_die_count": self.decode_die_count,
            "node_count": self.node_count,
            "visible_phy_ids": list(self.visible_phy_ids) if self.visible_phy_ids else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TopologySpec":
        return cls.from_mapping(value)


@dataclass(frozen=True)
class GangMember:
    host: str
    role: str
    phy_ids: tuple[int, ...]
    node_index: int = 0

    @property
    def visible_devices(self) -> str:
        return ",".join(str(phy) for phy in self.phy_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "role": self.role,
            "phy_ids": list(self.phy_ids),
            "node_index": self.node_index,
            "visible_devices": self.visible_devices,
        }


@dataclass(frozen=True)
class GangLease:
    scale: str
    placement_policy: str
    die_count: int
    node_count: int
    members: tuple[GangMember, ...]
    spec: TopologySpec

    @property
    def hosts(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(member.host for member in self.members))

    @property
    def groups(self) -> dict[str, tuple[int, ...]]:
        return {member.role: member.phy_ids for member in self.members}

    @property
    def visible_devices_by_role(self) -> dict[str, str]:
        return {member.role: member.visible_devices for member in self.members}

    @property
    def visible_phy_ids_by_host(self) -> dict[str, tuple[int, ...]]:
        grouped: dict[str, list[int]] = {}
        for member in self.members:
            grouped.setdefault(member.host, []).extend(member.phy_ids)
        return {host: tuple(sorted(phy_ids)) for host, phy_ids in grouped.items()}

    @property
    def total_die_count(self) -> int:
        return sum(len(member.phy_ids) for member in self.members)

    @property
    def ok(self) -> bool:
        return True

    def member(self, role: str) -> GangMember:
        for member in self.members:
            if member.role == role:
                return member
        raise KeyError(role)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "scale": self.scale,
            "placement_policy": self.placement_policy,
            "die_count": self.die_count,
            "node_count": self.node_count,
            "spec": self.spec.to_dict(),
            "hosts": list(self.hosts),
            "members": [member.to_dict() for member in self.members],
            "groups": {role: list(ids) for role, ids in self.groups.items()},
            "visible_devices_by_role": self.visible_devices_by_role,
            "visible_phy_ids_by_host": {
                host: list(ids) for host, ids in self.visible_phy_ids_by_host.items()
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)


def _host_label(host: HostTopology) -> str:
    return host.host or "<unnamed>"


def _select_dies(
    host: HostTopology,
    count: int,
    *,
    exclude: frozenset[int] = frozenset(),
    require_sibling_pair: bool = False,
) -> tuple[DieObservation, ...]:
    if require_sibling_pair:
        if count % 2:
            raise GangLeaseError(
                "invalid_spec", "same_npu_dual_die selection requires an even count"
            )
        pairs = [
            card
            for card in host.cards
            if card.sibling_pair
            and card.fully_free
            and not (set(card.phy_ids) & exclude)
        ]
        needed = count // 2
        if len(pairs) < needed:
            raise GangLeaseError(
                "insufficient_dual_die_pairs",
                f"{_host_label(host)} has {len(pairs)} free sibling pair(s), needs {needed}",
            )
        selected = [die for card in pairs[:needed] for die in card.dies]
        return tuple(sorted(selected, key=lambda die: die.phy_id))

    free_by_card = {
        card.npu_id: [die for die in card.dies if die.free and die.phy_id not in exclude]
        for card in host.cards
    }
    available = sum(len(dies) for dies in free_by_card.values())
    if available < count:
        raise GangLeaseError(
            "insufficient_free_dies",
            f"{_host_label(host)} has {available} free Die(s), needs {count}",
        )
    selected: list[DieObservation] = []
    remaining = count
    full_cards = [
        card
        for card in host.cards
        if card.fully_free and not (set(card.phy_ids) & exclude)
    ]
    for card in full_cards:
        if remaining <= 0:
            break
        take = min(remaining, len(card.dies))
        selected.extend(card.dies[:take])
        remaining -= take
    if remaining:
        for card in host.cards:
            if remaining <= 0:
                break
            if card in full_cards:
                continue
            for die in free_by_card[card.npu_id]:
                selected.append(die)
                remaining -= 1
                if remaining <= 0:
                    break
    if remaining:
        raise GangLeaseError(
            "insufficient_free_dies", f"{_host_label(host)} could not place {count} Dies"
        )
    return tuple(sorted(selected, key=lambda die: die.phy_id))


def _select_visible(
    host: HostTopology, visible_phy_ids: Sequence[int]
) -> tuple[DieObservation, ...]:
    free = host.free_die_map
    missing = [phy for phy in visible_phy_ids if phy not in free]
    if missing:
        raise GangLeaseError(
            "requested_dies_unavailable",
            f"{_host_label(host)} requested Phy-IDs are busy or absent: {missing}",
            host=host.host,
            missing=missing,
        )
    return tuple(free[phy] for phy in sorted(visible_phy_ids))


def _assert_same_card_pairs(
    host: HostTopology, selected: Sequence[DieObservation]
) -> None:
    selected_ids = {die.phy_id for die in selected}
    covered: set[int] = set()
    for card in host.cards:
        if set(card.phy_ids) <= selected_ids:
            if not card.sibling_pair:
                raise GangLeaseError(
                    "insufficient_dual_die_pairs",
                    f"{_host_label(host)} selected a non-sibling card {card.phy_ids}",
                )
            covered.update(card.phy_ids)
    if covered != selected_ids:
        missing = sorted(selected_ids - covered)
        raise GangLeaseError(
            "insufficient_dual_die_pairs",
            f"{_host_label(host)} selected Die(s) without a free sibling: {missing}",
        )


def _choose_single_host(hosts: Sequence[HostTopology], spec: TopologySpec) -> HostTopology:
    if spec.visible_phy_ids is not None:
        for host in hosts:
            if set(spec.visible_phy_ids) <= set(host.free_phy_ids):
                return host
        capacities = {_host_label(host): list(host.free_phy_ids) for host in hosts}
        raise GangLeaseError(
            "requested_dies_unavailable",
            f"no host has all requested Phy-IDs {list(spec.visible_phy_ids)} free",
            capacities=capacities,
        )

    if (
        spec.placement_policy == POLICY_SAME_NPU_DUAL_DIE
        and not any(card.sibling_pair for host in hosts for card in host.cards)
    ):
        raise GangLeaseError(
            "unsupported_topology",
            "same_npu_dual_die requires A3 dual-Die sibling cards; observed hosts expose none",
            hosts=[_host_label(host) for host in hosts],
        )

    candidates: list[tuple[int, int, int, HostTopology]] = []
    for index, host in enumerate(hosts):
        free_count = len(host.free_phy_ids)
        if spec.placement_policy == POLICY_SAME_NPU_DUAL_DIE:
            pairs = len(host.free_dual_die_pairs)
            if pairs * 2 >= spec.die_count:
                candidates.append((pairs, free_count, -index, host))
        elif free_count >= spec.die_count:
            candidates.append((0, free_count, -index, host))
    if not candidates:
        capacities = {_host_label(host): len(host.free_phy_ids) for host in hosts}
        code = (
            "insufficient_dual_die_pairs"
            if spec.placement_policy == POLICY_SAME_NPU_DUAL_DIE
            else "insufficient_free_dies"
        )
        reason = (
            f"no host has {spec.die_count // 2} free sibling pair(s) for same_npu_dual_die"
            if spec.placement_policy == POLICY_SAME_NPU_DUAL_DIE
            else f"no host can place {spec.die_count} {spec.placement_policy} Dies"
        )
        raise GangLeaseError(
            code,
            reason,
            capacities=capacities,
        )
    return max(candidates, key=lambda row: (row[0], row[1], row[2]))[3]


def _allocate_single_node(hosts: Sequence[HostTopology], spec: TopologySpec) -> GangLease:
    host = _choose_single_host(hosts, spec)
    if spec.placement_policy == POLICY_DISAGGREGATED_PD_PAIR:
        prefill_count = int(spec.prefill_die_count or spec.die_count // 2)
        decode_count = int(spec.decode_die_count or spec.die_count // 2)
        if spec.visible_phy_ids is not None:
            ordered = list(_select_visible(host, spec.visible_phy_ids))
        else:
            prefill_dies = _select_dies(host, prefill_count)
            prefill_ids = frozenset(die.phy_id for die in prefill_dies)
            decode_dies = _select_dies(host, decode_count, exclude=prefill_ids)
            ordered = list(prefill_dies) + list(decode_dies)
        prefill_dies = tuple(ordered[:prefill_count])
        decode_dies = tuple(ordered[prefill_count:])
        if len(prefill_dies) != prefill_count or len(decode_dies) != decode_count:
            raise GangLeaseError(
                "insufficient_free_dies",
                f"{_host_label(host)} cannot split {spec.die_count} Dies into "
                f"Prefill={prefill_count} and Decode={decode_count}",
            )
        return GangLease(
            scale=spec.scale,
            placement_policy=spec.placement_policy,
            die_count=spec.die_count,
            node_count=1,
            members=(
                GangMember(
                    host=host.host,
                    role="prefill",
                    phy_ids=tuple(sorted(die.phy_id for die in prefill_dies)),
                ),
                GangMember(
                    host=host.host,
                    role="decode",
                    phy_ids=tuple(sorted(die.phy_id for die in decode_dies)),
                ),
            ),
            spec=spec,
        )

    if spec.visible_phy_ids is not None:
        selected = _select_visible(host, spec.visible_phy_ids)
    elif spec.placement_policy == POLICY_SAME_NPU_DUAL_DIE:
        selected = _select_dies(host, spec.die_count, require_sibling_pair=True)
    else:
        selected = _select_dies(host, spec.die_count)
    if spec.placement_policy == POLICY_SAME_NPU_DUAL_DIE:
        _assert_same_card_pairs(host, selected)
    return GangLease(
        scale=spec.scale,
        placement_policy=spec.placement_policy,
        die_count=spec.die_count,
        node_count=1,
        members=(
            GangMember(
                host=host.host,
                role="default",
                phy_ids=tuple(sorted(die.phy_id for die in selected)),
            ),
        ),
        spec=spec,
    )


def _distribute_across_nodes(total: int, capacities: Sequence[int]) -> tuple[int, ...]:
    """Water-fill a gang across nodes, respecting each node's free-Die ceiling."""
    allocation = [0] * len(capacities)
    active = [index for index, capacity in enumerate(capacities) if capacity > 0]
    remaining = total
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        progressed = False
        for index in list(active):
            if remaining <= 0:
                break
            add = min(capacities[index] - allocation[index], share, remaining)
            if add:
                allocation[index] += add
                remaining -= add
                progressed = True
            if allocation[index] >= capacities[index]:
                active.remove(index)
        if not progressed:
            break
    if remaining:
        raise GangLeaseError(
            "insufficient_free_dies",
            f"cannot distribute {total} Dies over node capacities {list(capacities)}",
        )
    return tuple(allocation)


def _allocate_multi_node(hosts: Sequence[HostTopology], spec: TopologySpec) -> GangLease:
    node_count = int(spec.node_count or 2)
    capable = [host for host in hosts if host.free_phy_ids]
    if len(capable) < node_count:
        raise GangLeaseError(
            "insufficient_nodes",
            f"S3 needs {node_count} nodes with free Dies, found {len(capable)}",
            hosts=[_host_label(host) for host in capable],
        )
    ranked = sorted(
        enumerate(capable), key=lambda row: (-len(row[1].free_phy_ids), row[0])
    )[:node_count]
    selected_hosts = [host for _index, host in sorted(ranked, key=lambda row: row[0])]
    capacities = [len(host.free_phy_ids) for host in selected_hosts]
    if sum(capacities) < spec.die_count:
        raise GangLeaseError(
            "insufficient_free_dies",
            f"S3 needs {spec.die_count} Dies, selected nodes have {sum(capacities)}",
            capacities=dict(zip((_host_label(host) for host in selected_hosts), capacities)),
        )
    shares = _distribute_across_nodes(spec.die_count, capacities)
    members: list[GangMember] = []
    for index, (host, share) in enumerate(zip(selected_hosts, shares)):
        if share <= 0:
            continue
        selected = _select_dies(host, share)
        members.append(
            GangMember(
                host=host.host,
                role=f"node-{index}",
                phy_ids=tuple(sorted(die.phy_id for die in selected)),
                node_index=index,
            )
        )
    return GangLease(
        scale=spec.scale,
        placement_policy=spec.placement_policy,
        die_count=spec.die_count,
        node_count=node_count,
        members=tuple(members),
        spec=spec,
    )


def allocate_gang_lease(
    hosts: Sequence[HostTopology], spec: TopologySpec | Mapping[str, Any]
) -> GangLease:
    """Compute one topology-aware gang placement; never reserve or mutate devices."""
    normalized_spec = spec if isinstance(spec, TopologySpec) else TopologySpec.from_mapping(spec)
    if isinstance(hosts, HostTopology):
        hosts = [hosts]
    unique: list[HostTopology] = []
    seen_hosts: set[str] = set()
    for host in hosts:
        if not isinstance(host, HostTopology):
            raise GangLeaseError("invalid_spec", "hosts must contain HostTopology rows")
        if host.host in seen_hosts:
            raise GangLeaseError("invalid_spec", f"duplicate host in gang: {host.host}")
        seen_hosts.add(host.host)
        unique.append(host)
    if not unique:
        raise GangLeaseError("insufficient_nodes", "no host topology supplied")
    if normalized_spec.placement_policy == POLICY_MULTI_NODE_ROCE:
        return _allocate_multi_node(unique, normalized_spec)
    return _allocate_single_node(unique, normalized_spec)


plan_gang_lease = allocate_gang_lease


def allocate_s1_gang(
    hosts: Sequence[HostTopology],
    *,
    placement_policy: str = POLICY_INTRA_NODE_HCCS,
    die_count: int = 4,
    **kwargs: Any,
) -> GangLease:
    return allocate_gang_lease(
        hosts, TopologySpec(SCALE_S1, die_count, placement_policy, **kwargs)
    )


def allocate_s2_gang(
    hosts: Sequence[HostTopology],
    *,
    placement_policy: str = POLICY_INTRA_NODE_HCCS,
    die_count: int = 16,
    **kwargs: Any,
) -> GangLease:
    return allocate_gang_lease(
        hosts, TopologySpec(SCALE_S2, die_count, placement_policy, **kwargs)
    )


def allocate_s3_gang(
    hosts: Sequence[HostTopology],
    *,
    die_count: int = 16,
    node_count: int = 2,
    **kwargs: Any,
) -> GangLease:
    return allocate_gang_lease(
        hosts,
        TopologySpec(
            SCALE_S3, die_count, POLICY_MULTI_NODE_ROCE, node_count=node_count, **kwargs
        ),
    )


__all__ = [
    "BUSY_HBM_THRESHOLD_MB",
    "CardObservation",
    "DieObservation",
    "GangLease",
    "GangLeaseError",
    "GangMember",
    "HostTopology",
    "HygieneReport",
    "PLACEMENT_POLICIES",
    "POLICY_DISAGGREGATED_PD_PAIR",
    "POLICY_INTRA_NODE_HCCS",
    "POLICY_MULTI_NODE_ROCE",
    "POLICY_SAME_NPU_DUAL_DIE",
    "POLICY_SCALES",
    "ProcessObservation",
    "SANDBOX_PATTERN",
    "SANDBOX_PREFIX",
    "SCALES",
    "SCALE_DIE_BOUNDS",
    "SCALE_S0",
    "SCALE_S1",
    "SCALE_S2",
    "SCALE_S3",
    "SHM_STALE_AFTER_SECONDS",
    "SandboxEntry",
    "ShmEntry",
    "TopologySpec",
    "allocate_gang_lease",
    "allocate_s1_gang",
    "allocate_s2_gang",
    "allocate_s3_gang",
    "classify_topology_scale",
    "cleanup_sandbox_command",
    "parse_npu_smi_capture",
    "parse_sandbox_hygiene",
    "plan_gang_lease",
    "sandbox_hygiene_command",
    "topology_from_fleet_snapshot",
    "validate_sandbox_path",
]
