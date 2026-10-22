"""Local MindIE coordinator: task stdio MCP and host NPU allocation.

[Role] Re-exports the topology-gang public API and package version.
[Invariants] Importing the package starts no daemon and performs no host I/O.
[Timeouts & Locks] No waits or locks at import time.
[Side-Effect Policy] Read-only version lookup; no files are written.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mindie-coordinator")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.6.2"

from mindie_coordinator.topology_gang import (  # noqa: E402
    PLACEMENT_POLICIES,
    POLICY_DISAGGREGATED_PD_PAIR,
    POLICY_INTRA_NODE_HCCS,
    POLICY_MULTI_NODE_ROCE,
    POLICY_SAME_NPU_DUAL_DIE,
    SANDBOX_PREFIX,
    SCALES,
    SCALE_S0,
    SCALE_S1,
    SCALE_S2,
    SCALE_S3,
    SHM_STALE_AFTER_SECONDS,
    GangLease,
    GangLeaseError,
    GangMember,
    HostTopology,
    TopologySpec,
    allocate_gang_lease,
    allocate_s1_gang,
    allocate_s2_gang,
    allocate_s3_gang,
    classify_topology_scale,
    cleanup_sandbox_command,
    parse_npu_smi_capture,
    parse_sandbox_hygiene,
    plan_gang_lease,
    sandbox_hygiene_command,
    validate_sandbox_path,
)

__all__ = [
    "PLACEMENT_POLICIES",
    "POLICY_DISAGGREGATED_PD_PAIR",
    "POLICY_INTRA_NODE_HCCS",
    "POLICY_MULTI_NODE_ROCE",
    "POLICY_SAME_NPU_DUAL_DIE",
    "SANDBOX_PREFIX",
    "SCALES",
    "SCALE_S0",
    "SCALE_S1",
    "SCALE_S2",
    "SCALE_S3",
    "SHM_STALE_AFTER_SECONDS",
    "GangLease",
    "GangLeaseError",
    "GangMember",
    "HostTopology",
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
    "validate_sandbox_path",
]
