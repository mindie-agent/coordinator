"""Bounded retention for local execution caches and terminal run logs.

[Role]
Prunes rebuildable execution-source capture caches, fixed-input transport
descriptors, and per-run log directories once they exceed bounded retention.

[Invariants]
- Only rebuildable capture/transport files and terminal run directories are
  removed; fixed source snapshots, active runs, and `inconclusive`/`unknown`
  outcomes are never touched.
- A run directory is terminal only when the top-level phase is definitive and
  every role, managed job, lease, and preparation job is confirmed released.
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
RUN_PRUNE_COOLDOWN_SECONDS = 300
RUN_PRUNE_BUDGET_SECONDS = 1.0
FIXED_INPUT_RETENTION_SECONDS = 24 * 60 * 60
PRUNABLE_RUN_PHASES = frozenset({"succeeded", "failed", "cancelled"})
NONTERMINAL_RUN_STATES = frozenset({
    "active", "running", "preparing", "starting", "stopping", "orphaned_busy",
    "inconclusive", "uncertain", "unknown", "waiting", "waiting_for_runtime",
    "pending", "queued", "granted", "submitting", "releasing", "bound",
    "launch_pending", "finishing",
})
TERMINAL_JOB_STATES = frozenset({
    "succeeded", "completed", "failed", "timeout", "timed_out", "cancelled", "absent",
})
TERMINAL_LEASE_STATES = frozenset({"released", "cancelled", "expired"})
TERMINAL_COMPILE_STATES = TERMINAL_LEASE_STATES | {"not_started"}
_STATE_KEYS = frozenset({
    "state", "states", "phase", "status", "outcome", "lease_state", "job_state",
    "remote_state", "preparation_state", "submission_state", "resource_state",
    "task_state", "compile_state", "remote_outcome",
})


class SnapshotCommits(dict):
    """Parity snapshot mapping carrying a non-fatal post-sync cleanup warning."""

    def __init__(self, value, *, cleanup_warning=None):
        super().__init__(value)
        self.cleanup_warning = cleanup_warning


def _state_strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            state_key = isinstance(key, str) and (
                key in _STATE_KEYS or key.endswith("_state"))
            if state_key and isinstance(item, str):
                yield item
            yield from _state_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _state_strings(item)


def terminal_run_eligible(row: dict) -> bool:
    """True only when a run is terminal and every child resource is released.

    A top-level `failed` phase is not proof that retained preparation jobs,
    managed leases, or `inconclusive`/`unknown` role outcomes are gone.
    """
    if not isinstance(row, dict) or row.get("phase") not in PRUNABLE_RUN_PHASES:
        return False
    if any(state in NONTERMINAL_RUN_STATES for state in _state_strings(row)):
        return False
    for role in row.get("roles") or []:
        if not isinstance(role, dict) or not role.get("managed_job"):
            continue
        observation = role.get("observation")
        if not isinstance(observation, dict) or observation.get("state") not in TERMINAL_JOB_STATES:
            return False
        if observation.get("lease_state") not in TERMINAL_LEASE_STATES:
            return False
    if not row.get("roles") and row.get("managed_job"):
        observation = row.get("observation")
        if not isinstance(observation, dict) or observation.get("state") not in TERMINAL_JOB_STATES:
            return False
        if observation.get("lease_state") not in TERMINAL_LEASE_STATES:
            return False
    for records in (row.get("preparation_jobs") or {}).values():
        for job in records.values():
            if not isinstance(job, dict) or job.get("state") not in TERMINAL_JOB_STATES:
                return False
            if job.get("quiet") is not True:
                return False
            lease = job.get("compile_lease")
            if isinstance(lease, dict) and lease.get("state") not in TERMINAL_COMPILE_STATES:
                return False
    return True


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
