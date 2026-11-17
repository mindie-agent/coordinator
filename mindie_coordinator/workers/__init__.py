"""Standalone pure-stdlib remote worker scripts for `mindie_coordinator`.

[Role]
Houses standalone Python scripts (`python3 -I` safe, zero `mindie_coordinator`
runtime imports) embedded or concatenated for remote SSH host/container execution
per Iron Rule 4 (no giant inline Python `-c` or heredoc string blobs in callers).

[Invariants]
- Every worker module in this package uses Python stdlib only so remote hosts and
  containers never require `mindie_coordinator` to be installed.
- Callers load worker source text via `functools.lru_cache` helpers.

[Timeouts & Locks]
- Subprocess invocations inside worker scripts enforce explicit bounded timeouts.

[Side-Effect Policy]
- Loading this package has zero side effects.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=8)
def load_worker_source(name: str) -> str:
    return (Path(__file__).resolve().parent / name).read_text(encoding="utf-8")
