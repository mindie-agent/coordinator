"""Bounded retention for host coordination SQLite rows.

[Role]
Prunes old terminal task/hold rows, bounded event history, and expired peer
messages from `coordinator.sqlite3` during ordinary housekeeping.

[Invariants]
- Active leases (`tasks.state` outside the terminal set) and active holds are
  never deleted, regardless of age.
- At most `EVENT_RETENTION_LIMIT` newest events survive; old events and expired
  messages older than seven days are removed.
- A terminal task's port reservations are deleted before its task row.

[Timeouts & Locks]
- Runs inside the caller's `BEGIN IMMEDIATE` housekeeping transaction; no
  additional waits or locks are introduced.

[Side-Effect Policy]
- Deletes only rows from the coordinator database's own retention tables.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

TERMINAL_TASK_STATES = ("released", "expired", "cancelled", "failed", "completed")
TERMINAL_HOLD_STATES = ("released", "expired", "cancelled", "failed", "completed")
RETENTION_SECONDS = 7 * 24 * 60 * 60
EVENT_RETENTION_LIMIT = 4096
MESSAGE_RETENTION_SECONDS = 7 * 24 * 60 * 60


def _epoch_from_iso(value) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def backfill_message_timestamps(connection) -> None:
    """Assign real insertion times to messages created before the column existed."""
    rows = connection.execute(
        "SELECT cursor, data FROM messages WHERE created_at IS NULL").fetchall()
    for row in rows:
        try:
            value = json.loads(row["data"])
        except (TypeError, ValueError):
            value = {}
        created = _epoch_from_iso(value.get("at") if isinstance(value, dict) else None)
        connection.execute(
            "UPDATE messages SET created_at=? WHERE cursor=?",
            (0.0 if created is None else created, row["cursor"]),
        )


def prune_coordination_rows(connection, now: float) -> dict[str, int]:
    """Delete terminal/expired rows beyond retention and bound event history."""
    cutoff = now - RETENTION_SECONDS
    removed: dict[str, int] = {}
    task_states = ",".join("?" for _ in TERMINAL_TASK_STATES)
    cursor = connection.execute(
        f"DELETE FROM ports WHERE task_id IN ("
        f"SELECT task_id FROM tasks WHERE state IN ({task_states}) AND updated_at < ?)",
        (*TERMINAL_TASK_STATES, cutoff))
    removed["ports"] = cursor.rowcount
    cursor = connection.execute(
        f"DELETE FROM tasks WHERE state IN ({task_states}) AND updated_at < ?",
        (*TERMINAL_TASK_STATES, cutoff))
    removed["tasks"] = cursor.rowcount
    hold_states = ",".join("?" for _ in TERMINAL_HOLD_STATES)
    cursor = connection.execute(
        f"DELETE FROM holds WHERE state IN ({hold_states}) AND updated_at < ?",
        (*TERMINAL_HOLD_STATES, cutoff))
    removed["holds"] = cursor.rowcount
    cursor = connection.execute("DELETE FROM events WHERE at < ?", (cutoff,))
    removed["events"] = cursor.rowcount
    threshold = connection.execute(
        "SELECT event_id FROM events ORDER BY event_id DESC LIMIT 1 OFFSET ?",
        (EVENT_RETENTION_LIMIT - 1,),
    ).fetchone()
    if threshold is not None:
        cursor = connection.execute(
            "DELETE FROM events WHERE event_id < ?", (threshold["event_id"],))
        removed["events"] += cursor.rowcount
    cursor = connection.execute(
        "DELETE FROM messages WHERE created_at < ?",
        (now - MESSAGE_RETENTION_SECONDS,))
    removed["messages"] = cursor.rowcount
    return removed
