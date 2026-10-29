"""SQLite state, schema, serialization, and selection mixin.

[Role]
Owns the host coordination database: schema, WAL transactions, event/fence
serialization, housekeeping, peer messages, and device/port selection.

[Invariants]
- `PRAGMA journal_mode=WAL`, `busy_timeout=5000`, `synchronous=FULL` and
  `timeout=5` match the original single-file authority.
- Mutations run under `BEGIN IMMEDIATE`; selection never reserves another
  task's device or port, and timeout/unknown observations retain leases.

[Timeouts & Locks]
- SQLite timeout is 5s with a 5000ms busy timeout; transactions are exclusive.

[Side-Effect Policy]
- Writes only the coordinator SQLite database and its WAL sidecars.
"""

from __future__ import annotations

import contextlib
import json
import re
import sqlite3
import time
import uuid
from typing import Any, Iterator

from .mindie_npu_coordination_probe import (
    PORT_KINDS,
    SCHEMA_VERSION,
    CoordinationError,
    _authority_attribute,
    _diagnostic_elapsed,
    _load_devices,
    parse_port_range,
    require_safe_id,
    unguarded_cpu_process,
    unguarded_shared_process,
    utc_now_iso,
)


class CoordinationStateMixin:
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=5, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    agent_alias TEXT,
                    session_id TEXT,
                    container_name TEXT,
                    requested_count INTEGER NOT NULL,
                    requested_devices TEXT,
                    allow_external_busy INTEGER NOT NULL DEFAULT 0,
                    not_before REAL NOT NULL,
                    latest_start REAL NOT NULL,
                    estimated_duration_seconds INTEGER NOT NULL,
                    preemptible INTEGER NOT NULL DEFAULT 0,
                    priority INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL,
                    submitted_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    granted_devices TEXT,
                    activation_deadline REAL,
                    fence_token INTEGER,
                    started_at REAL,
                    expected_end REAL,
                    heartbeat_at REAL,
                    heartbeat_deadline REAL,
                    pid INTEGER,
                    message TEXT,
                    requested_service_port INTEGER,
                    service_port_choices TEXT,
                    granted_service_port INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_queue
                    ON tasks(state, priority DESC, submitted_at ASC);
                CREATE TABLE IF NOT EXISTS holds (
                    hold_id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    devices TEXT NOT NULL,
                    not_before REAL NOT NULL,
                    end_at REAL NOT NULL,
                    reason TEXT,
                    state TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_holds_window
                    ON holds(state, not_before, end_at);
                CREATE TABLE IF NOT EXISTS ports (
                    port INTEGER PRIMARY KEY,
                    kind TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_ports_task
                    ON ports(task_id);
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    at REAL NOT NULL,
                    kind TEXT NOT NULL,
                    task_id TEXT,
                    data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS message_peers (
                    user TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    seen_at REAL NOT NULL,
                    PRIMARY KEY(user, session_id)
                );
                CREATE TABLE IF NOT EXISTS messages (
                    cursor INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT NOT NULL UNIQUE,
                    sender TEXT NOT NULL,
                    sender_session TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    recipient_session TEXT NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_recipient
                    ON messages(recipient, recipient_session, cursor);
                """
            )
            task_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(tasks)")
            }
            if "agent_alias" not in task_columns:
                try:
                    connection.execute("ALTER TABLE tasks ADD COLUMN agent_alias TEXT")
                except sqlite3.OperationalError as exc:
                    if "duplicate column" not in str(exc).lower():
                        raise
            if "process_guard" not in task_columns:
                try:
                    connection.execute("ALTER TABLE tasks ADD COLUMN process_guard TEXT")
                except sqlite3.OperationalError as exc:
                    if "duplicate column" not in str(exc).lower():
                        raise
            for column, decl in (
                ("allow_external_busy", "INTEGER NOT NULL DEFAULT 0"),
                ("per_die_hbm_mb", "INTEGER NOT NULL DEFAULT 0"),
                ("per_worker_memory_mb", "INTEGER NOT NULL DEFAULT 2048"),
                ("compile_budget", "TEXT"),
                ("requested_service_port", "INTEGER"),
                ("service_port_choices", "TEXT"),
                ("granted_service_port", "INTEGER"),
            ):
                if column not in task_columns:
                    try:
                        connection.execute(f"ALTER TABLE tasks ADD COLUMN {column} {decl}")
                    except sqlite3.OperationalError as exc:
                        if "duplicate column" not in str(exc).lower():
                            raise
            # Older clients send the host protocol source with each request.
            # They do not know about process guards. Do not let their generic
            # "hardware free" UPDATE release a guarded CPU-initializing job.
            connection.execute("""
                CREATE TRIGGER IF NOT EXISTS protect_managed_process_release
                BEFORE UPDATE OF state ON tasks
                WHEN OLD.process_guard IS NOT NULL AND NEW.process_guard IS NOT NULL
                     AND NEW.state IN ('released', 'cancelled', 'expired')
                BEGIN
                    SELECT RAISE(ABORT, 'managed process guard must be verified before release');
                END
            """)
            connection.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            connection.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'",
                (str(SCHEMA_VERSION),),
            )
            connection.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('coordination_epoch', ?)",
                (str(uuid.uuid4()),),
            )
            connection.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('next_fence', '0')")
        finally:
            connection.close()

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        started = time.monotonic()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            _diagnostic_elapsed("state_lock_wait_ms", started)
            if self.expected_epoch is not None:
                epoch = connection.execute("SELECT value FROM meta WHERE key='coordination_epoch'").fetchone()
                if epoch is None or epoch["value"] != self.expected_epoch:
                    raise CoordinationError("coordination epoch changed; reconcile ownership before retrying")
            yield connection
            connection.execute("COMMIT")
        except Exception:
            with contextlib.suppress(sqlite3.Error):
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _event(
        self,
        connection: sqlite3.Connection,
        kind: str,
        *,
        task_id: str | None = None,
        data: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> None:
        connection.execute(
            "INSERT INTO events(at, kind, task_id, data) VALUES(?, ?, ?, ?)",
            (self.clock() if now is None else now, kind, task_id, json.dumps(data or {}, sort_keys=True)),
        )

    def _next_fence(self, connection: sqlite3.Connection) -> int:
        row = connection.execute("SELECT value FROM meta WHERE key='next_fence'").fetchone()
        value = int(row["value"]) + 1
        connection.execute("UPDATE meta SET value=? WHERE key='next_fence'", (str(value),))
        return value

    @staticmethod
    def _busy_set(observed: dict[str, Any] | None) -> set[int] | None:
        if observed is None or observed.get("status") != "ok" or observed.get("busy", {}) is None:
            return None
        return {int(key) for key in observed.get("busy", {})}

    @staticmethod
    def _serialize_task(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        payload = dict(row)
        payload["compile_budget"] = json.loads(payload["compile_budget"]) if payload.get("compile_budget") else None
        payload["requested_devices"] = _load_devices(payload.get("requested_devices")) or None
        payload["granted_devices"] = _load_devices(payload.get("granted_devices"))
        payload["preemptible"] = bool(payload.get("preemptible"))
        payload["allow_external_busy"] = bool(payload.get("allow_external_busy"))
        payload["process_guarded"] = bool(payload.pop("process_guard", None))
        for key in (
            "not_before",
            "latest_start",
            "submitted_at",
            "updated_at",
            "activation_deadline",
            "started_at",
            "expected_end",
            "heartbeat_at",
            "heartbeat_deadline",
        ):
            if payload.get(key) is not None:
                payload[key] = utc_now_iso(float(payload[key]))
        return payload

    @staticmethod
    def _serialize_hold(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        payload = dict(row)
        payload["devices"] = _load_devices(payload.get("devices"))
        for key in ("not_before", "end_at", "created_at", "updated_at"):
            payload[key] = utc_now_iso(float(payload[key]))
        return payload

    def _task_row(self, connection: sqlite3.Connection, task_id: str) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            raise CoordinationError(f"task not found: {task_id}")
        return row

    def _check_token(self, row: sqlite3.Row, token: int) -> None:
        if row["fence_token"] is None or int(row["fence_token"]) != int(token):
            raise CoordinationError(f"stale or invalid fencing token for task {row['task_id']}")

    @staticmethod
    def _message_actor(request: dict[str, Any]) -> tuple[str, str]:
        user = request.get("user")
        if not isinstance(user, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", user):
            raise CoordinationError("invalid message user")
        return user, require_safe_id(request.get("session_id"), label="message session")

    def message(self, request: dict[str, Any]) -> dict[str, Any]:
        """Store plain coordination text. This never changes a task or lease.

        Identity is supplied by the cooperating client, as with the host queue;
        shared root access is not an adversarial authentication boundary.
        """
        user, session_id = self._message_actor(request)
        text = request.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise CoordinationError("message must contain 1..4000 characters")
        message_id = require_safe_id(request.get("message_id"), label="message id")
        with self._transaction() as db:
            previous = db.execute("SELECT * FROM messages WHERE message_id=?", (message_id,)).fetchone()
            if previous is not None:
                value = json.loads(previous["data"])
                expected = (user, session_id, text, request.get("reply_to"),
                            request.get("recipient"), request.get("recipient_session"))
                original = (value["sender"], value["sender_session"], value["text"], value.get("reply_to"),
                            value.get("requested_recipient"), value.get("requested_session"))
                if expected != original:
                    raise CoordinationError("message id was already used for different content")
                return {"status": "sent", "message": {"cursor": previous["cursor"], **value}}
            reply_to = request.get("reply_to")
            if reply_to:
                original = db.execute("SELECT * FROM messages WHERE message_id=?", (reply_to,)).fetchone()
                if original is None or (original["recipient"], original["recipient_session"]) != (user, session_id):
                    raise CoordinationError("reply does not belong to this task")
                recipient, recipient_session = original["sender"], original["sender_session"]
                thread_id = json.loads(original["data"])["thread_id"]
            else:
                recipient, recipient_session = self._message_actor({
                    "user": request.get("recipient"), "session_id": request.get("recipient_session")})
                if db.execute("SELECT 1 FROM message_peers WHERE user=? AND session_id=?",
                              (recipient, recipient_session)).fetchone() is None:
                    raise CoordinationError("recipient has not used this host; use a current coordination reference")
                thread_id = message_id
            value = {"message_id": message_id, "thread_id": thread_id,
                     "kind": "coordination-reply" if reply_to else "coordination-message",
                     "sender": user, "sender_session": session_id,
                     "recipient": recipient, "recipient_session": recipient_session,
                     "requested_recipient": request.get("recipient"),
                     "requested_session": request.get("recipient_session"),
                     "reply_to": reply_to, "text": text, "at": utc_now_iso(self.clock())}
            cursor = db.execute("INSERT INTO messages(message_id,sender,sender_session,recipient,recipient_session,data) "
                                "VALUES(?,?,?,?,?,?)", (message_id, user, session_id, recipient,
                                                       recipient_session, json.dumps(value))).lastrowid
            return {"status": "sent", "message": {"cursor": cursor, **value}}

    def message_events(self, request: dict[str, Any]) -> dict[str, Any]:
        """Register this task's return address and read its persistent inbox.

        Cursor belongs to this mailbox epoch. Peer timestamps are observations,
        never evidence that a process ended or that devices can be reclaimed.
        """
        user, session_id = self._message_actor(request)
        after = request.get("after", 0)
        limit = request.get("limit", 2)
        if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 20:
            raise CoordinationError("invalid message cursor or limit")
        with self._transaction() as db:
            epoch = db.execute("SELECT value FROM meta WHERE key='coordination_epoch'").fetchone()[0]
            if request.get("mailbox_epoch") != epoch:
                after = 0
            db.execute("INSERT INTO message_peers(user,session_id,seen_at) VALUES(?,?,?) "
                       "ON CONFLICT(user,session_id) DO UPDATE SET seen_at=excluded.seen_at",
                       (user, session_id, self.clock()))
            rows = db.execute("SELECT cursor,data FROM messages WHERE recipient=? AND recipient_session=? "
                              "AND cursor>? ORDER BY cursor LIMIT ?", (user, session_id, after, limit)).fetchall()
            peers = []
            if request.get("include_peers"):
                peers = [dict(row) for row in db.execute(
                    "SELECT user,session_id,seen_at FROM message_peers WHERE NOT (user=? AND session_id=?) "
                    "ORDER BY seen_at DESC LIMIT 5", (user, session_id))]
            return {"status": "ok", "mailbox_epoch": epoch,
                    "cursor": rows[-1]["cursor"] if rows else after,
                    "events": [{"cursor": row["cursor"], **json.loads(row["data"])} for row in rows],
                    "peers": peers}

    def _housekeep(
        self,
        connection: sqlite3.Connection,
        observed: dict[str, Any] | None,
        *,
        now: float,
    ) -> list[dict[str, Any]]:
        changes: list[dict[str, Any]] = []
        busy = self._busy_set(observed)
        visible = (
            {int(device) for device in observed.get("devices", [])}
            if observed is not None and observed.get("status") == "ok"
            else None
        )

        for row in connection.execute("SELECT * FROM holds WHERE state='active' AND end_at<=?", (now,)).fetchall():
            connection.execute("UPDATE holds SET state='expired', updated_at=? WHERE hold_id=?", (now, row["hold_id"]))
            changes.append({"hold_id": row["hold_id"], "from": "active", "to": "expired"})
            self._event(connection, "hold-expired", data={"hold_id": row["hold_id"]}, now=now)

        for row in connection.execute(
            "SELECT * FROM tasks WHERE state='queued' AND latest_start<=?", (now,)
        ).fetchall():
            connection.execute(
                "UPDATE tasks SET state='expired', updated_at=?, message=? WHERE task_id=?",
                (now, "latest_start elapsed before grant", row["task_id"]),
            )
            changes.append({"task_id": row["task_id"], "from": "queued", "to": "expired"})
            self._event(connection, "task-expired-in-queue", task_id=row["task_id"], now=now)

        deadline_rows = connection.execute(
            """
            SELECT * FROM tasks
            WHERE state IN ('granted','starting')
              AND activation_deadline IS NOT NULL
              AND activation_deadline<=?
            """,
            (now,),
        ).fetchall()
        for row in deadline_rows:
            devices = set(_load_devices(row["granted_devices"]))
            still_busy = (
                (bool(devices) and (busy is None or visible is None
                    or not devices.issubset(visible)
                    or (bool(devices & busy) and not row["allow_external_busy"])))
                or _authority_attribute("process_guard_busy")(row["process_guard"])
            )
            next_state = "orphaned_busy" if still_busy else "expired"
            message = (
                "activation deadline elapsed; occupancy still busy or unknown"
                if still_busy
                else "activation deadline elapsed before task became active"
            )
            connection.execute(
                "UPDATE tasks SET state=?, updated_at=?, message=? WHERE task_id=?",
                (next_state, now, message, row["task_id"]),
            )
            changes.append({"task_id": row["task_id"], "from": row["state"], "to": next_state})
            self._event(
                connection,
                "activation-deadline-elapsed",
                task_id=row["task_id"],
                data={"to": next_state},
                now=now,
            )

        heartbeat_rows = connection.execute(
            """
            SELECT * FROM tasks
            WHERE state='active'
              AND heartbeat_deadline IS NOT NULL
              AND heartbeat_deadline<=?
            """,
            (now,),
        ).fetchall()
        for row in heartbeat_rows:
            devices = set(_load_devices(row["granted_devices"]))
            still_busy = (
                (bool(devices) and (busy is None or visible is None
                    or not devices.issubset(visible)
                    or (bool(devices & busy) and not row["allow_external_busy"])))
                or unguarded_cpu_process(row)
                or unguarded_shared_process(row)
                or _authority_attribute("process_guard_busy")(row["process_guard"])
            )
            next_state = "orphaned_busy" if still_busy else "released"
            message = 'CPU task has no process guard; process completion is unknown' if unguarded_cpu_process(row) else (
                "heartbeat expired while hardware remained busy or unknown"
                if still_busy
                else "heartbeat expired and hardware was observed free"
            )
            connection.execute(
                "UPDATE tasks SET state=?, process_guard=CASE WHEN ?='released' THEN NULL ELSE process_guard END, updated_at=?, message=? WHERE task_id=?",
                (next_state, next_state, now, message, row["task_id"]),
            )
            changes.append({"task_id": row["task_id"], "from": "active", "to": next_state})
            self._event(connection, "heartbeat-expired", task_id=row["task_id"], data={"to": next_state}, now=now)

        for row in connection.execute("SELECT * FROM tasks WHERE state='orphaned_busy'").fetchall():
            devices = set(_load_devices(row["granted_devices"]))
            # CPU reclamation needs process evidence, not unrelated hardware
            # visibility. Port owners still need an explicit listener check.
            if self._task_service_ports(connection, row["task_id"]):
                continue
            hardware_free = not devices or (busy is not None and visible is not None
                and devices.issubset(visible) and (not devices.intersection(busy) or row["allow_external_busy"]))
            if (hardware_free and not unguarded_cpu_process(row) and not unguarded_shared_process(row)
                    and not _authority_attribute("process_guard_busy")(row["process_guard"])):
                connection.execute(
                    "UPDATE tasks SET state='released', process_guard=NULL, updated_at=?, message=? WHERE task_id=?",
                    (now, "orphaned task has no pending owned process or device", row["task_id"]),
                )
                changes.append({"task_id": row["task_id"], "from": "orphaned_busy", "to": "released"})
                self._event(connection, "orphaned-task-released", task_id=row["task_id"], now=now)

        for row in connection.execute(
            "SELECT * FROM tasks WHERE state='active' AND expected_end IS NOT NULL AND expected_end<=?",
            (now,),
        ).fetchall():
            if row["message"] != "estimated duration exceeded; lease remains protected":
                connection.execute(
                    "UPDATE tasks SET updated_at=?, message=? WHERE task_id=?",
                    (now, "estimated duration exceeded; lease remains protected", row["task_id"]),
                )
                changes.append({"task_id": row["task_id"], "state": "active", "overdue": True})
                self._event(connection, "task-overdue", task_id=row["task_id"], now=now)
        return changes

    @staticmethod
    def _reserved_devices(connection: sqlite3.Connection, *, exclude_task: str | None = None) -> set[int]:
        reserved: set[int] = set()
        query = (
            "SELECT task_id, granted_devices FROM tasks "
            "WHERE state IN ('granted','starting','active','orphaned_busy')"
        )
        for row in connection.execute(query).fetchall():
            if exclude_task and row["task_id"] == exclude_task:
                continue
            reserved.update(_load_devices(row["granted_devices"]))
        return reserved

    @staticmethod
    def _hold_conflicts(
        connection: sqlite3.Connection,
        *,
        device: int,
        start_at: float,
        end_at: float,
    ) -> bool:
        for row in connection.execute(
            """
            SELECT devices FROM holds
            WHERE state='active' AND not_before<? AND end_at>?
            """,
            (end_at, start_at),
        ).fetchall():
            if device in _load_devices(row["devices"]):
                return True
        return False

    def _coordination_epoch(self, connection: sqlite3.Connection) -> str:
        row = connection.execute("SELECT value FROM meta WHERE key='coordination_epoch'").fetchone()
        if row is None:
            raise CoordinationError("coordination epoch is missing")
        return str(row["value"])

    @staticmethod
    def _serialize_port(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        payload = dict(row)
        payload["created_at"] = utc_now_iso(float(payload["created_at"]))
        return payload

    def _task_ports(self, connection: sqlite3.Connection, task_id: str) -> list[dict[str, Any]]:
        rows = connection.execute(
            "SELECT * FROM ports WHERE task_id=? ORDER BY port ASC", (task_id,)
        ).fetchall()
        return [item for item in (self._serialize_port(row) for row in rows) if item is not None]

    def _available_devices(
        self,
        connection: sqlite3.Connection,
        observed: dict[str, Any],
        *,
        exclude_task: str | None,
        start_at: float,
        end_at: float,
        allow_external_busy: bool = False,
    ) -> list[int]:
        busy = self._busy_set(observed) or set()
        reserved = self._reserved_devices(connection, exclude_task=exclude_task)
        available: list[int] = []
        for device in observed.get("devices", []):
            device = int(device)
            if (device in busy and not allow_external_busy) or device in reserved:
                continue
            if self._hold_conflicts(connection, device=device, start_at=start_at, end_at=end_at):
                continue
            available.append(device)
        return available

    def _select_granted_devices(
        self,
        *,
        requested: list[int],
        count: int,
        available: list[int],
    ) -> tuple[list[int] | None, list[int]]:
        if requested:
            missing = sorted(set(requested) - set(available))
            if missing:
                return None, missing
            return requested, []
        if len(available) < count:
            return None, []
        return sorted(available)[:count], []

    def _allocated_ports(self, connection: sqlite3.Connection, *, exclude_task: str | None = None) -> set[int]:
        allocated: set[int] = set()
        for row in connection.execute("SELECT port, task_id FROM ports").fetchall():
            if exclude_task and row["task_id"] == exclude_task:
                continue
            allocated.add(int(row["port"]))
        return allocated

    def _task_service_ports(self, connection: sqlite3.Connection, task_id: str) -> list[int]:
        return [
            int(row["port"])
            for row in connection.execute(
                "SELECT port FROM ports WHERE task_id=? AND kind='service' ORDER BY port",
                (task_id,),
            ).fetchall()
        ]

    def _claim_port(
        self,
        connection: sqlite3.Connection,
        *,
        port: int,
        kind: str,
        task_id: str,
        owner: str,
        now: float,
        listening: dict[str, Any] | None = None,
        allow_listening: bool = False,
    ) -> int:
        if kind not in PORT_KINDS:
            raise CoordinationError(f"unsupported port kind: {kind!r}")
        if type(port) is not int or not 0 < port < 65536:
            raise CoordinationError(f"invalid TCP port: {port!r}")
        existing = connection.execute("SELECT kind, task_id FROM ports WHERE port=?", (port,)).fetchone()
        if existing is not None:
            if existing["task_id"] == task_id and existing["kind"] == kind:
                return port
            raise CoordinationError(f"port {port} is already reserved as {existing['kind']}",
                                    error_code="port_reserved", port=port)
        if not allow_listening:
            if listening is None or listening.get("status") != "ok":
                raise CoordinationError(
                    (listening or {}).get("error") or "host listening ports are unavailable"
                )
            live = {int(item) for item in listening.get("ports", [])}
            if port in live:
                raise CoordinationError(f"port {port} is already bound on the host")
        connection.execute(
            "INSERT INTO ports(port, kind, task_id, owner, created_at) VALUES(?, ?, ?, ?, ?)",
            (port, kind, task_id, owner, now),
        )
        return port

    def _select_service_port(
        self,
        connection: sqlite3.Connection,
        *,
        choices: list[int],
        requested: int,
        task_id: str,
        owner: str,
        now: float,
        listening: dict[str, Any],
    ) -> int:
        if listening.get("status") != "ok":
            raise CoordinationError(
                listening.get("error") or "host listening ports are unavailable"
            )
        if not choices:
            if requested != 0:
                raise CoordinationError("service port requested but no declared runtime service ports")
            # Prepared task roots can share a host-managed automatic port pool.
            # The port is claimed below only after checking live and leased ports.
            first, last = parse_port_range(_authority_attribute("DEFAULT_SERVING_PORT_RANGE"))
            choices = list(range(first, last + 1))
        live = {int(item) for item in listening.get("ports", [])}
        allocated = self._allocated_ports(connection, exclude_task=task_id)
        preferred = None if requested == 0 else requested
        if preferred is not None and preferred not in choices:
            raise CoordinationError(f"service port {preferred} is not a declared runtime service port")
        candidates = [preferred] if preferred is not None else list(choices)
        owned = {
            int(row["port"]): str(row["kind"])
            for row in connection.execute(
                "SELECT port, kind FROM ports WHERE task_id=?", (task_id,)
            ).fetchall()
        }
        for port in candidates:
            existing_kind = owned.get(port)
            if existing_kind is not None:
                if existing_kind != "service":
                    if preferred is not None:
                        raise CoordinationError(f"port {port} is already reserved as {existing_kind}")
                    continue
                return port
            if port in allocated or port in live:
                if preferred is not None:
                    raise CoordinationError(f"port {port} is already reserved or bound on the host")
                continue
            connection.execute(
                "INSERT INTO ports(port, kind, task_id, owner, created_at) VALUES(?, ?, ?, ?, ?)",
                (port, "service", task_id, owner, now),
            )
            return port
        raise CoordinationError("no free declared service port")

    def _service_ports_busy(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        listening: dict[str, Any] | None,
    ) -> list[int]:
        ports = self._task_service_ports(connection, task_id)
        if not ports:
            return []
        if listening is None or listening.get("status") != "ok":
            return ports
        live = {int(item) for item in listening.get("ports", [])}
        return sorted(port for port in ports if port in live)

    def _clear_service_ports(self, connection: sqlite3.Connection, task_id: str) -> None:
        connection.execute("DELETE FROM ports WHERE task_id=? AND kind='service'", (task_id,))
        connection.execute(
            "UPDATE tasks SET granted_service_port=NULL WHERE task_id=?",
            (task_id,),
        )
