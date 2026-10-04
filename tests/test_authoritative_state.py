"""Loss of native identity or lease authority never becomes an empty first use."""
import contextlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from mindie_coordinator.agent_session import AgentSessions
from mindie_coordinator.ready_runtime import RuntimePool
from mindie_coordinator.owned_sqlite import AuthorityStateError
from mindie_coordinator.host import mindie_npu_coordination as host

KINDS = ("sessions", "pool", "host")
ERRORS = (AuthorityStateError, host.CoordinationError, sqlite3.DatabaseError)


def open_store(kind, root, **kwargs):
    if kind == "sessions":
        return AgentSessions(root)
    if kind == "pool":
        return RuntimePool(root, None)
    return host.NpuCoordinator(root, **kwargs)


def transaction(store):
    return store._transaction() if isinstance(store, host.NpuCoordinator) else store.transaction()


def seed(store):
    if isinstance(store, AgentSessions):
        context = store.attach("codex", "native-session", str(store.state_dir))
        with store.transaction() as db:
            store.put(db, "execution", {"id": "unknown-job", "state": "inconclusive", "remote_id": "same-remote-job"})
        return context
    if isinstance(store, RuntimePool):
        with store.transaction() as db:
            store.put(db, "run", {"id": "unknown-job", "state": "unknown", "remote_id": "same-remote-job", "epoch": "original-epoch"})
        return None
    store.submit({"task_id": "retained-task", "agent_id": "alice", "devices": [0], "queue_ttl_seconds": 1000})
    with store._transaction() as db:
        db.execute("UPDATE tasks SET state='active', fence_token=41, process_guard=?", ('{"marker":"retained"}',))
        db.execute("UPDATE meta SET value='41' WHERE key='next_fence'")
    return None


def rows(path):
    with contextlib.closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {name: db.execute('SELECT * FROM "' + name + '" ORDER BY rowid').fetchall() for name in tables}


def checkpoint(path):
    with contextlib.closing(sqlite3.connect(path)) as db:
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")


@pytest.mark.parametrize("kind", KINDS)
def test_valid_reopen_preserves_all_authority(kind, tmp_path):
    tmp_path.mkdir(exist_ok=True)  # A caller may create the parent before use.
    store = open_store(kind, tmp_path)
    context = seed(store)
    before = rows(store.db_path)
    again = open_store(kind, tmp_path)
    assert rows(store.db_path) == before
    if kind == "sessions":
        assert again.native_context("codex", "native-session")["attachment"]["id"] == context["attachment"]["id"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("live", [False, True], ids=["reopen", "running-object"])
@pytest.mark.parametrize("damage", ["missing", "empty", "invalid-marker", "incomplete-table", "bad-version"])
def test_damaged_authority_fails_without_repair(kind, live, damage, tmp_path):
    store = open_store(kind, tmp_path)
    seed(store)
    checkpoint(store.db_path)
    marker = store.db_path.with_name(store.db_path.name + ".owner")
    if damage == "missing":
        store.db_path.unlink()
    elif damage == "empty":
        store.db_path.write_bytes(b"")
    elif damage == "invalid-marker":
        marker.write_text("broken\n", encoding="utf-8")
    elif damage == "incomplete-table":
        with contextlib.closing(sqlite3.connect(store.db_path)) as db, db:
            if kind == "host":
                db.execute("DROP TABLE messages")
            elif kind == "pool":
                db.execute("DROP TABLE events")
            else:
                db.execute("ALTER TABLE records RENAME COLUMN data TO missing_data")
        checkpoint(store.db_path)
    else:
        with contextlib.closing(sqlite3.connect(store.db_path)) as db, db:
            if kind == "host":
                db.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
            else:
                db.execute("PRAGMA user_version=999")
        checkpoint(store.db_path)
    before = store.db_path.read_bytes() if store.db_path.exists() else None
    with pytest.raises(ERRORS):
        if live:
            with transaction(store):
                pytest.fail("damaged authority was accepted")
        else:
            open_store(kind, tmp_path)
    assert (store.db_path.read_bytes() if store.db_path.exists() else None) == before
    assert marker.exists()
    if kind == "sessions":
        assert list((tmp_path / "contexts").glob("*.json"))


@pytest.mark.parametrize("kind", KINDS)
def test_sqlite_open_cannot_recreate_db_lost_after_existence_check(kind, tmp_path):
    store = open_store(kind, tmp_path)
    checkpoint(store.db_path)
    original = sqlite3.connect

    def disappeared(path, *args, **kwargs):
        assert "mode=rw" in str(path)
        assert kwargs.get("uri") is True
        store.db_path.unlink()
        return original(path, *args, **kwargs)

    with patch("sqlite3.connect", side_effect=disappeared), pytest.raises(sqlite3.OperationalError):
        with transaction(store):
            pass
    assert not store.db_path.exists()


@pytest.mark.parametrize("kind", KINDS)
def test_creation_intent_is_durable_before_sqlite_create(kind, tmp_path):
    original = sqlite3.connect

    def interrupted(path, *args, **kwargs):
        if path != ":memory:":
            target = Path(path)
            assert target.with_name(target.name + ".owner").is_file()
            assert not target.exists()
            raise OSError("injected crash before SQLite creation")
        return original(path, *args, **kwargs)

    with patch("sqlite3.connect", side_effect=interrupted), pytest.raises(OSError, match="injected crash"):
        open_store(kind, tmp_path)
    with pytest.raises(ERRORS, match="missing"):
        open_store(kind, tmp_path)
    assert not list(tmp_path.glob("*.sqlite3"))


@pytest.mark.parametrize("kind", KINDS)
def test_interrupted_schema_creation_is_not_completed_on_reopen(kind, tmp_path):
    path = tmp_path / ("sessions.sqlite3" if kind == "sessions" else "coordinator.sqlite3")
    marker = path.with_name(path.name + ".owner")
    marker.write_text({"sessions": "mindie.agent-sessions.v1\n", "pool": "mindie.runtime-pool.v1\n", "host": host._HOST_OWNER}[kind])
    with contextlib.closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE partial_creation(id TEXT)")
    before = path.read_bytes()
    with pytest.raises(ERRORS):
        open_store(kind, tmp_path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("kind", KINDS)
def test_valid_legacy_store_only_acquires_marker(kind, tmp_path):
    store = open_store(kind, tmp_path)
    seed(store)
    marker = store.db_path.with_name(store.db_path.name + ".owner")
    marker.unlink()
    if kind != "host":
        with contextlib.closing(sqlite3.connect(store.db_path)) as db:
            db.execute("PRAGMA user_version=0")
    before = rows(store.db_path)
    open_store(kind, tmp_path)
    assert rows(store.db_path) == before
    assert marker.is_file()
    if kind != "host":
        with contextlib.closing(sqlite3.connect(store.db_path)) as db:
            assert db.execute("PRAGMA user_version").fetchone()[0] == 0


@pytest.mark.parametrize("kind,residue", [("sessions", "contexts"), ("sessions", "source-inputs"), ("pool", "runs"),
                                         ("pool", "results"), ("pool", "coordinator.ipc"), ("host", "coordinator.sqlite3-wal")])
def test_owned_residue_prevents_fresh_authority(kind, residue, tmp_path):
    (tmp_path / residue).write_text("existing authority residue", encoding="utf-8")
    with pytest.raises(ERRORS, match="missing"):
        open_store(kind, tmp_path)
    assert not list(tmp_path.glob("*.sqlite3"))
    assert (tmp_path / residue).read_text() == "existing authority residue"


@pytest.mark.parametrize("kind", KINDS)
def test_corrupt_bytes_are_preserved(kind, tmp_path):
    path = tmp_path / ("sessions.sqlite3" if kind == "sessions" else "coordinator.sqlite3")
    path.write_bytes(b"not a sqlite database")
    with pytest.raises(ERRORS):
        open_store(kind, tmp_path)
    assert path.read_bytes() == b"not a sqlite database"
    assert not path.with_name(path.name + ".owner").exists()


@pytest.mark.parametrize("kind", KINDS)
def test_concurrent_true_first_use_has_one_authority(kind, tmp_path):
    program = '''
import json, sys
from pathlib import Path
from mindie_coordinator.agent_session import AgentSessions
from mindie_coordinator.ready_runtime import RuntimePool
from mindie_coordinator.host.mindie_npu_coordination import NpuCoordinator
kind, root, number = sys.argv[1:]
root = Path(root)
if kind == 'host':
    store = NpuCoordinator(root)
    store.submit({'task_id':'task-' + number, 'agent_id':'alice', 'devices':[0]})
    print(store._epoch)
else:
    store = AgentSessions(root) if kind == 'sessions' else RuntimePool(root, None)
    with store.transaction() as db:
        store.put(db, 'inconclusive', {'id':number, 'state':'unknown'})
    print('saved')
'''
    processes = [subprocess.Popen([sys.executable, "-c", program, kind, str(tmp_path), str(i)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for i in range(6)]
    output = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr
        output.append(stdout.strip())
    assert len(set(output)) == 1
    store = open_store(kind, tmp_path)
    with transaction(store) as db:
        assert db.execute("SELECT COUNT(*) FROM " + ("tasks" if kind == "host" else "records")).fetchone()[0] == 6


def legacy_host(tmp_path):
    path = tmp_path / "coordinator.sqlite3"
    with contextlib.closing(sqlite3.connect(path)) as db, db:
        db.executescript((Path(__file__).parent / "fixtures" / "host-schema-v6.sql").read_text(encoding="utf-8"))
        db.executemany("INSERT INTO meta VALUES(?,?)", [("schema_version", "6"), ("coordination_epoch", "old-epoch"), ("next_fence", "41")])
        db.execute("INSERT INTO tasks(task_id,agent_id,requested_count,not_before,latest_start,estimated_duration_seconds,state,submitted_at,updated_at,process_guard,fence_token) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                   ("retained-task", "alice", 1, 1, 10000000000, 3600, "active", 1, 1, '{"marker":"retained"}', 41))
        db.execute("INSERT INTO ports VALUES(30000,'service','retained-task','alice',1)")
        db.execute("INSERT INTO messages VALUES(7,'same-message','alice','task-a','bob','task-b','{}')")
    return path


def test_complete_v6_migration_preserves_epoch_guard_fence_and_rows(tmp_path):
    path = legacy_host(tmp_path)
    before = rows(path)
    store = host.NpuCoordinator(tmp_path, expected_epoch="old-epoch")
    after = rows(path)
    for name in before.keys() - {"events", "meta"}:
        assert after[name] == before[name]
    assert dict(after["meta"]) == {"schema_version": "7", "coordination_epoch": "old-epoch", "next_fence": "41"}
    assert json.loads(after["events"][-1][-1])["from"] == 6
    with store._transaction() as db:
        assert next(row for row in db.execute("PRAGMA table_info(tasks)") if row["name"] == "latest_start")["notnull"] == 0
        with pytest.raises(sqlite3.IntegrityError, match="process guard"):
            db.execute("UPDATE tasks SET state='released'")
    with store._transaction() as db:
        db.execute("UPDATE tasks SET latest_start=NULL WHERE task_id='retained-task'")
        assert db.execute("SELECT latest_start FROM tasks WHERE task_id='retained-task'").fetchone()[0] is None
    reopened = host.NpuCoordinator(tmp_path, expected_epoch="old-epoch")
    assert len([r for r in rows(path)["events"] if r[2] == "schema-migrated"]) == 1
    assert reopened._epoch == "old-epoch"


@pytest.mark.parametrize("damage", ["missing-table", "missing-epoch", "invalid-fence", "wrong-version", "missing-guard", "wrong-epoch"])
def test_invalid_v6_never_migrates_or_changes_leases(damage, tmp_path):
    path = legacy_host(tmp_path)
    with contextlib.closing(sqlite3.connect(path)) as db, db:
        sql = {"missing-table": "DROP TABLE messages", "missing-epoch": "DELETE FROM meta WHERE key='coordination_epoch'",
               "invalid-fence": "UPDATE meta SET value='invalid' WHERE key='next_fence'", "wrong-version": "UPDATE meta SET value='5' WHERE key='schema_version'",
               "missing-guard": "DROP TRIGGER protect_managed_process_release"}
        if damage in sql:
            db.execute(sql[damage])
    before = path.read_bytes()
    with pytest.raises(ERRORS):
        host.NpuCoordinator(tmp_path, expected_epoch="different-epoch" if damage == "wrong-epoch" else None)
    assert path.read_bytes() == before
    assert not path.with_name(path.name + ".owner").exists()


def test_failed_migration_rolls_back_all_schema_and_lease_changes(tmp_path):
    path = legacy_host(tmp_path)
    before = rows(path)
    with patch.object(host.NpuCoordinator, "_event", side_effect=RuntimeError("cannot record migration")):
        with pytest.raises(RuntimeError, match="cannot record migration"):
            host.NpuCoordinator(tmp_path)
    assert rows(path) == before
    # The retry resumes only this validated transaction, never an external job.
    host.NpuCoordinator(tmp_path)
    assert dict(rows(path)["meta"])["coordination_epoch"] == "old-epoch"


def test_expected_old_epoch_never_creates_missing_host_authority(tmp_path):
    with pytest.raises(host.CoordinationError, match="missing"):
        host.NpuCoordinator(tmp_path, expected_epoch="old-epoch")
    assert not (tmp_path / "coordinator.sqlite3").exists()
    assert not (tmp_path / "coordinator.sqlite3.owner").exists()


def test_running_host_detects_epoch_change_before_write(tmp_path):
    store = host.NpuCoordinator(tmp_path)
    seed(store)
    with contextlib.closing(sqlite3.connect(store.db_path)) as db, db:
        db.execute("UPDATE meta SET value='replacement-epoch' WHERE key='coordination_epoch'")
    before = rows(store.db_path)
    with pytest.raises(host.CoordinationError, match="epoch changed"):
        store.submit({"task_id": "must-not-submit", "agent_id": "alice", "devices": [0]})
    assert rows(store.db_path) == before


def test_readonly_diagnostics_cannot_bypass_authority_marker(tmp_path):
    store = RuntimePool(tmp_path, None)
    with store.transaction() as db:
        store.put(db, "run", {"id": "owned-run", "owner": "alice"})
        store.event(db, "alice", "observation", run="owned-run")
    assert store.diagnostic_events("alice", ["owned-run"])["operations"]
    store.db_path.with_name(store.db_path.name + ".owner").write_text("invalid")
    with pytest.raises(AuthorityStateError, match="marker is invalid"):
        store.diagnostic_events("alice", ["owned-run"])


@pytest.mark.parametrize("kind", ["pool", "host"])
def test_event_cursor_schema_cannot_silently_lose_monotonicity(kind, tmp_path):
    store = open_store(kind, tmp_path)
    seed(store)
    with contextlib.closing(sqlite3.connect(store.db_path)) as db, db:
        sql = db.execute("SELECT sql FROM sqlite_master WHERE name='events'").fetchone()[0]
        db.execute("DROP TABLE events")
        db.execute(sql.replace("AUTOINCREMENT", ""))
    before = rows(store.db_path)
    with pytest.raises(ERRORS, match="schema"):
        open_store(kind, tmp_path)
    assert rows(store.db_path) == before


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("replaced", ["database", "marker"])
def test_live_authority_rejects_even_valid_replacement(kind, replaced, tmp_path):
    store = open_store(kind, tmp_path)
    seed(store)
    checkpoint(store.db_path)
    path = store.db_path if replaced == "database" else store.db_path.with_name(store.db_path.name + ".owner")
    before = rows(store.db_path)
    replacement = path.with_name(path.name + ".replacement")
    replacement.write_bytes(path.read_bytes())
    replacement.replace(path)
    with pytest.raises(ERRORS, match="replaced"):
        with transaction(store) as db:
            pytest.fail("replaced authority reached the business transaction")
    assert rows(store.db_path) == before
