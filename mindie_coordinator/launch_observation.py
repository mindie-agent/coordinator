"""Immutable facts from the managed launch's successful runtime attestation."""
from __future__ import annotations

from mindie_coordinator.runtime_profile import digest

ENV_NAME = "MINDIE_EXECUTION_OBSERVATION"


def launch_observation(binding, request, spec, environment):
    return {
        "workspace_snapshot": {
            ("workspace_commit" if name == "." else name.replace("-", "_") + "_commit"): commit
            for name, commit in (request.get("snapshots") or {}).items()
        },
        "environment": {"profile_key": binding.get("profile_key"),
                        "launch_env_digest": digest(spec.get("env") or {})},
        "native_digest": {"build_key": request.get("expected_build_key")},
        "machine": (binding.get("host_endpoint") or {}).get("host"),
        "command": spec["command"],
        "npu_devices": [int(value) for value in environment.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",") if value],
        **({"allow_external_busy": True} if request.get("allow_external_busy") else {}),
        "scope": "managed launch attestation; does not inspect later runtime mutations",
    }
