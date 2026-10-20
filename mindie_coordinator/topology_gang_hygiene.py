"""Read-only sandbox and stale HCCL shared-memory hygiene for gang preflight.

[Role]
Builds bounded, non-destructive shell probes for one
`/tmp/mindie-npu-test-<uuid>` sandbox and `/dev/shm/hccl*` artifacts, and parses
their output into explicit evidence. Also builds the single permitted cleanup
command, which removes only the validated sandbox path.

[Invariants]
- `/dev/shm/hccl*` is never opened for writing, renamed, or deleted. Age-based
  entries are `stale_candidate` evidence; deleting them requires a separate
  owner/workload check outside this module.
- No command runs `docker`, signals a PID, or touches a business container.
- Sandbox paths must match exactly `/tmp/mindie-npu-test-<name>`; symlinks are
  reported/refused rather than followed.

[Timeouts & Locks]
- Command builders are pure. Callers bound SSH; the emitted `stat`/`find`
  probes are finite and row-capped. No locks or sleeps.

[Side-Effect Policy]
- Building commands has no side effects. Executing `cleanup_sandbox_command`
  removes only the validated sandbox directory. `/dev/shm` stays untouched.
"""

from __future__ import annotations

import math
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from mindie_coordinator.topology_gang_common import (
    GangLeaseError,
    SANDBOX_PREFIX,
    SHM_STALE_AFTER_SECONDS,
)

SANDBOX_PATTERN = re.compile(r"^/tmp/mindie-npu-test-[A-Za-z0-9][A-Za-z0-9._-]*$")
_HYGIENE_ROW_LIMIT = 200


def validate_sandbox_path(path: str) -> str:
    """Accept exactly one generated `/tmp/mindie-npu-test-<uuid>` path."""
    if not isinstance(path, str) or not SANDBOX_PATTERN.fullmatch(path):
        raise GangLeaseError(
            "unsafe_sandbox",
            f"sandbox must match {SANDBOX_PREFIX}<uuid>: {path!r}",
        )
    parts = PurePosixPath(path).parts
    if len(parts) != 3 or parts[:2] != ("/", "tmp") or parts[2] in (".", ".."):
        raise GangLeaseError("unsafe_sandbox", f"unsafe sandbox path: {path!r}")
    return path


def sandbox_hygiene_command(
    sandbox_path: str, *, stale_after_seconds: float = SHM_STALE_AFTER_SECONDS
) -> str:
    """Build a read-only hygiene script for one sandbox and `/dev/shm/hccl*`."""
    sandbox = validate_sandbox_path(sandbox_path)
    if (
        not isinstance(stale_after_seconds, (int, float))
        or not math.isfinite(stale_after_seconds)
        or stale_after_seconds <= 0
    ):
        raise GangLeaseError("invalid_spec", "stale_after_seconds must be positive")
    quoted = shlex.quote(sandbox)
    return f"""set -u
sandbox={quoted}
echo "SANDBOX|$sandbox"
if [ -L "$sandbox" ]; then
  echo "SANDBOX_STATE|symlink"
elif [ -d "$sandbox" ]; then
  echo "SANDBOX_STATE|present"
elif [ -e "$sandbox" ]; then
  echo "SANDBOX_STATE|not_a_directory"
else
  echo "SANDBOX_STATE|absent"
fi
if [ -d "$sandbox" ] && [ ! -L "$sandbox" ]; then
  find "$sandbox" -mindepth 1 -maxdepth 2 -print 2>/dev/null | head -n {_HYGIENE_ROW_LIMIT} |
  while IFS= read -r entry; do
    kind=file
    [ -d "$entry" ] && kind=directory
    [ -L "$entry" ] && kind=symlink
    if [ -L "$entry" ]; then
      size=0
    else
      size=$(stat -c %s "$entry" 2>/dev/null || stat -f %z "$entry" 2>/dev/null || echo 0)
    fi
    echo "SANDBOX_ENTRY|$kind|$size|$entry"
  done
fi
echo "SHM_STALE_SECONDS|{stale_after_seconds:g}"
echo "SHM_BEGIN"
for entry in /dev/shm/hccl* /dev/shm/HCCL*; do
  [ -e "$entry" ] || continue
  size=$(stat -c %s "$entry" 2>/dev/null || stat -f %z "$entry" 2>/dev/null || echo 0)
  mtime=$(stat -c %Y "$entry" 2>/dev/null || stat -f %m "$entry" 2>/dev/null || echo 0)
  echo "SHM_ENTRY|$size|$mtime|$entry"
done
echo "SHM_END"
"""


def cleanup_sandbox_command(sandbox_path: str) -> str:
    """Build the only destructive command this module permits: sandbox removal."""
    sandbox = validate_sandbox_path(sandbox_path)
    quoted = shlex.quote(sandbox)
    return f"""set -u
sandbox={quoted}
if [ -L "$sandbox" ]; then
  echo "CLEANUP|refused_symlink"
  exit 0
fi
if [ ! -e "$sandbox" ]; then
  echo "CLEANUP|absent"
  exit 0
fi
rm -rf -- "$sandbox"
if [ -e "$sandbox" ]; then
  echo "CLEANUP|failed"
  exit 1
fi
echo "CLEANUP|removed"
"""


@dataclass(frozen=True)
class SandboxEntry:
    path: str
    kind: str
    size_bytes: int


@dataclass(frozen=True)
class ShmEntry:
    path: str
    size_bytes: int
    mtime: float
    age_seconds: float | None
    stale_candidate: bool


@dataclass(frozen=True)
class HygieneReport:
    sandbox_path: str
    sandbox_state: str
    sandbox_entries: tuple[SandboxEntry, ...]
    shm_entries: tuple[ShmEntry, ...]
    stale_after_seconds: float
    warnings: tuple[str, ...] = ()

    @property
    def sandbox_present(self) -> bool:
        return self.sandbox_state == "present"

    @property
    def stale_shm_paths(self) -> tuple[str, ...]:
        return tuple(entry.path for entry in self.shm_entries if entry.stale_candidate)

    @property
    def deletion_authorized(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sandbox_path": self.sandbox_path,
            "sandbox_state": self.sandbox_state,
            "sandbox_present": self.sandbox_present,
            "sandbox_entries": [
                {"path": e.path, "kind": e.kind, "size_bytes": e.size_bytes}
                for e in self.sandbox_entries
            ],
            "shm_entries": [
                {
                    "path": e.path,
                    "size_bytes": e.size_bytes,
                    "mtime": e.mtime,
                    "age_seconds": e.age_seconds,
                    "stale_candidate": e.stale_candidate,
                }
                for e in self.shm_entries
            ],
            "stale_shm_paths": list(self.stale_shm_paths),
            "deletion_authorized": self.deletion_authorized,
            "warnings": list(self.warnings),
        }


def parse_sandbox_hygiene(
    raw: str,
    *,
    now: float | None = None,
    stale_after_seconds: float | None = None,
) -> HygieneReport:
    """Parse the read-only hygiene command output; stale entries are candidates."""
    current_time = time.time() if now is None else now
    threshold = SHM_STALE_AFTER_SECONDS if stale_after_seconds is None else stale_after_seconds
    sandbox_path = ""
    sandbox_state = "unknown"
    entries: list[SandboxEntry] = []
    shm_entries: list[ShmEntry] = []
    warnings: list[str] = []
    for line in raw.splitlines():
        if line.startswith("SANDBOX|"):
            sandbox_path = line.split("|", 1)[1].strip()
        elif line.startswith("SANDBOX_STATE|"):
            sandbox_state = line.split("|", 1)[1].strip() or "unknown"
        elif line.startswith("SANDBOX_ENTRY|"):
            parts = line.split("|", 3)
            if len(parts) != 4:
                warnings.append(f"malformed sandbox row: {line}")
                continue
            _tag, kind, size, path = parts
            try:
                size_bytes = int(size)
            except ValueError:
                size_bytes = 0
            entries.append(SandboxEntry(path=path, kind=kind, size_bytes=size_bytes))
        elif line.startswith("SHM_STALE_SECONDS|"):
            if stale_after_seconds is None:
                try:
                    threshold = float(line.split("|", 1)[1])
                except ValueError:
                    warnings.append(f"malformed stale threshold: {line}")
        elif line.startswith("SHM_ENTRY|"):
            parts = line.split("|", 3)
            if len(parts) != 4:
                warnings.append(f"malformed hccl row: {line}")
                continue
            _tag, size, mtime, path = parts
            try:
                size_bytes = int(size)
            except ValueError:
                size_bytes = 0
            try:
                mtime_value = float(mtime)
            except ValueError:
                mtime_value = 0.0
            age = max(0.0, current_time - mtime_value) if mtime_value > 0 else None
            shm_entries.append(
                ShmEntry(
                    path=path,
                    size_bytes=size_bytes,
                    mtime=mtime_value,
                    age_seconds=age,
                    stale_candidate=age is not None and age >= threshold,
                )
            )
    return HygieneReport(
        sandbox_path=sandbox_path,
        sandbox_state=sandbox_state,
        sandbox_entries=tuple(entries),
        shm_entries=tuple(shm_entries),
        stale_after_seconds=threshold,
        warnings=tuple(warnings),
    )


__all__ = [
    "HygieneReport",
    "SANDBOX_PATTERN",
    "SandboxEntry",
    "ShmEntry",
    "cleanup_sandbox_command",
    "parse_sandbox_hygiene",
    "sandbox_hygiene_command",
    "validate_sandbox_path",
]
