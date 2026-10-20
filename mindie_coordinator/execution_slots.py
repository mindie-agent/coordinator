"""Per-execution compiler caches inside a persistent user container.

[Role] Derive cache paths and shell setup from the existing execution identity.
[Invariants] Slots never share writable compiler caches; source roots are retained.
[Timeouts & Locks] Pure construction only; callers bound remote setup and cleanup.
[Side-Effect Policy] Cleanup is restricted to this slot's temporary cache root.
"""
from __future__ import annotations

import re
import shlex


SLOT_ENVIRONMENT = frozenset({
    "MINDIE_SLOT_ID", "ASCEND_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
})


def slot_environment(slot_id: str) -> dict[str, str]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", slot_id):
        raise ValueError("invalid execution slot identity")
    root = "/tmp/mindie-slot-" + slot_id
    return {
        "MINDIE_SLOT_ID": slot_id,
        "ASCEND_CACHE_PATH": root + "/ascend_cache",
        "TORCHINDUCTOR_CACHE_DIR": root + "/inductor_cache",
        "TRITON_CACHE_DIR": root + "/triton_cache",
    }


def slot_preamble(slot_id: str) -> str:
    environment = slot_environment(slot_id)
    exports = [f"export {name}={shlex.quote(value)}" for name, value in environment.items()]
    paths = [shlex.quote(value) for name, value in environment.items() if name != "MINDIE_SLOT_ID"]
    return "\n".join([*exports, "mkdir -p -- " + " ".join(paths)])


def slot_cleanup_command(slot_id: str) -> str:
    slot_environment(slot_id)
    return "rm -rf -- " + shlex.quote("/tmp/mindie-slot-" + slot_id)
