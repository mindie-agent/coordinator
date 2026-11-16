"""Bounded retention for local execution caches and terminal run logs.

[Role]
Prunes rebuildable execution-source capture caches, fixed-input transport
descriptors, and per-run log directories once they exceed bounded retention.

[Invariants]
- Only rebuildable capture/transport files and terminal run directories are
  removed; fixed source snapshots, active runs, and `inconclusive`/`unknown`
  outcomes are never touched.
- Every deletion stays inside the supplied directory and never follows a
  symlink.

[Timeouts & Locks]
- Callers own the state directory; helpers take no locks and perform only
  bounded local directory scans, atomic replaces, and unlinks.

[Side-Effect Policy]
- Writes or deletes only inside the supplied cache, fixed-input, or runs
  directory.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import time

CAPTURE_RETENTION_SECONDS = 14 * 24 * 60 * 60
CAPTURE_RETENTION_LIMIT = 128
RUN_RETENTION_SECONDS = 14 * 24 * 60 * 60
RUN_RETENTION_LIMIT = 128
FIXED_INPUT_RETENTION_SECONDS = 24 * 60 * 60
PRUNABLE_RUN_PHASES = frozenset({"succeeded", "failed", "timeout", "cancelled"})


def _age(path: Path, now: float) -> float:
    try:
        return now - path.stat().st_mtime
    except TimeoutError:
        raise
    except OSError:
        return 0.0


def prune_capture_caches(
    directory: Path | str,
    *,
    keep: Path | str | None = None,
    max_age_seconds: float = CAPTURE_RETENTION_SECONDS,
    max_files: int = CAPTURE_RETENTION_LIMIT,
) -> list[str]:
    """Keep the newest bounded set of `capture-*.json` acceleration caches."""
    directory = Path(directory)
    keep_path = Path(keep) if keep is not None else None
    now = time.time()
    try:
        entries = sorted(
            (path for path in directory.glob("capture-*.json")
             if path.is_file() and not path.is_symlink()),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
    except TimeoutError:
        raise
    except OSError:
        return []
    retained = 1 if keep_path is not None and any(path == keep_path for path in entries) else 0
    removed = []
    for path in entries:
        if keep_path is not None and path == keep_path:
            continue
        if retained < max_files and _age(path, now) <= max_age_seconds:
            retained += 1
            continue
        try:
            path.unlink()
        except TimeoutError:
            raise
        except OSError:
            continue
        removed.append(str(path))
    return removed


def prune_fixed_inputs(
    directory: Path | str,
    *,
    max_age_seconds: float = FIXED_INPUT_RETENTION_SECONDS,
) -> list[str]:
    """Delete stale `.mindie-local/fixed-inputs/*.json` transport descriptors."""
    directory = Path(directory)
    now = time.time()
    try:
        entries = [path for path in directory.glob("*.json")
                   if path.is_file() and not path.is_symlink()]
    except TimeoutError:
        raise
    except OSError:
        return []
    removed = []
    for path in entries:
        if max_age_seconds > 0 and _age(path, now) <= max_age_seconds:
            continue
        try:
            path.unlink()
        except TimeoutError:
            raise
        except OSError:
            continue
        removed.append(str(path))
    return removed


def write_fixed_input_descriptor(directory: Path | str, filename: str, value: dict) -> Path:
    """Publish one transport descriptor atomically; failed writes leave no temp file."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    prune_fixed_inputs(directory)
    descriptor = directory / filename
    if descriptor.is_file():
        return descriptor
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         delete=False) as output:
            temporary = Path(output.name)
            json.dump(value, output)
            output.flush()
        os.replace(temporary, descriptor)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except TimeoutError:
                raise
            except OSError:
                pass
    return descriptor


def prune_terminal_run_directories(
    runs_dir: Path | str,
    terminal_ids,
    *,
    live_ids=(),
    max_age_seconds: float = RUN_RETENTION_SECONDS,
    max_runs: int = RUN_RETENTION_LIMIT,
    now: float | None = None,
) -> list[str]:
    """Remove only old terminal `<runs_dir>/<run_id>/` log directories.

    Callers must label only definitively terminal executions as `terminal_ids`;
    absent, active, and `inconclusive`/`unknown` runs are never candidates.
    """
    runs = Path(runs_dir)
    if not runs.is_dir():
        return []
    terminal = {value for value in terminal_ids if isinstance(value, str) and value}
    live = {value for value in live_ids if isinstance(value, str) and value}
    candidates = []
    try:
        entries = list(runs.iterdir())
    except TimeoutError:
        raise
    except OSError:
        return []
    for entry in entries:
        if entry.is_symlink() or not entry.is_dir():
            continue
        if entry.name not in terminal or entry.name in live:
            continue
        try:
            candidates.append((entry.stat().st_mtime, entry.name, entry))
        except TimeoutError:
            raise
        except OSError:
            continue
    candidates.sort(key=lambda row: row[0], reverse=True)
    current = time.time() if now is None else now
    removed = []
    for index, (mtime, name, entry) in enumerate(candidates):
        if index < max_runs and current - mtime <= max_age_seconds:
            continue
        try:
            shutil.rmtree(entry)
        except TimeoutError:
            raise
        except OSError:
            continue
        removed.append(name)
    return removed
