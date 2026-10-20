"""Shared constants and refusal type for topology-aware gang planning.

[Role]
Defines production S0-S3 bounds, placement-policy names, sandbox/shm safety
limits, and the deterministic `GangLeaseError` used by observation, planner, and
hygiene helpers.

[Invariants]
- Scale/policy names are the stable protocol consumed by coordinator callers.
- A refusal always carries a machine-readable `code`; no silent empty success.

[Timeouts & Locks]
- Pure constants and exception type; zero I/O, locks, or waits.

[Side-Effect Policy]
- Side-effect free.
"""

from __future__ import annotations

from typing import Any

SCALE_S0 = "S0"
SCALE_S1 = "S1"
SCALE_S2 = "S2"
SCALE_S3 = "S3"
SCALES = (SCALE_S0, SCALE_S1, SCALE_S2, SCALE_S3)
SCALE_DIE_BOUNDS = {SCALE_S0: (1, 1), SCALE_S1: (2, 4), SCALE_S2: (8, 16), SCALE_S3: (16, 32)}

POLICY_SAME_NPU_DUAL_DIE = "same_npu_dual_die"
POLICY_INTRA_NODE_HCCS = "intra_node_hccs"
POLICY_DISAGGREGATED_PD_PAIR = "disaggregated_pd_pair"
POLICY_MULTI_NODE_ROCE = "multi_node_roce"
PLACEMENT_POLICIES = (
    POLICY_SAME_NPU_DUAL_DIE,
    POLICY_INTRA_NODE_HCCS,
    POLICY_DISAGGREGATED_PD_PAIR,
    POLICY_MULTI_NODE_ROCE,
)
POLICY_SCALES = {
    POLICY_SAME_NPU_DUAL_DIE: frozenset({SCALE_S1, SCALE_S2}),
    POLICY_INTRA_NODE_HCCS: frozenset({SCALE_S0, SCALE_S1, SCALE_S2}),
    POLICY_DISAGGREGATED_PD_PAIR: frozenset({SCALE_S1, SCALE_S2}),
    POLICY_MULTI_NODE_ROCE: frozenset({SCALE_S3}),
}

SANDBOX_PREFIX = "/tmp/mindie-npu-test-"
SHM_STALE_AFTER_SECONDS = 900.0


class GangLeaseError(ValueError):
    """A deterministic placement refusal with an inspectable reason code."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


__all__ = [
    "GangLeaseError",
    "PLACEMENT_POLICIES",
    "POLICY_DISAGGREGATED_PD_PAIR",
    "POLICY_INTRA_NODE_HCCS",
    "POLICY_MULTI_NODE_ROCE",
    "POLICY_SAME_NPU_DUAL_DIE",
    "POLICY_SCALES",
    "SANDBOX_PREFIX",
    "SCALES",
    "SCALE_DIE_BOUNDS",
    "SCALE_S0",
    "SCALE_S1",
    "SCALE_S2",
    "SCALE_S3",
    "SHM_STALE_AFTER_SECONDS",
]
