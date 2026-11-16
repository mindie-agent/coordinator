"""Regression coverage for Wave 1 storage lifecycle bounds."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time
from unittest import mock

import pytest

from mindie_coordinator import native_cache_pool, runtime_profile
from mindie_coordinator.execution_capture import capture_records, save_cache
from mindie_coordinator.host.mindie_npu_coordination import NpuCoordinator
from mindie_coordinator.local_exec import no_window_kwargs
from mindie_coordinator.service import CoordinatorService
from mindie_coordinator.shared_source_objects import copy_fixed_objects
from mindie_coordinator.storage_retention import (
    prune_capture_caches,
    prune_fixed_inputs,
    prune_terminal_run_directories,
    write_fixed_input_descriptor,
)
from test_execution_inputs import git, repo

NOW = 1_800_000_000.0


def _parity_refs(source: Path) -> list[str]:
    output = git(source, "for-each-ref", "--format=%(refname)", "refs/parity/execution-inputs")
    return output.splitlines() if output else []


def test_save_cache_deletes_superseded_parity_refs_and_keeps_current(tmp_path):
    source = repo(tmp_path / "repo")
    roots = {"app": source}
    state = tmp_path / "state"

    first, _, (location, cache) = capture_records(roots, state, attempts=3, build_env={})
    first_ref = first[0].ref
    assert first_ref.startswith("refs/parity/execution-inputs/")
    save_cache(location, cache, first)
    # `retained()` also needs the companion SCM ref for a dirty source record.
    git(source, "update-ref", first_ref + "-scm", first[0].source_head)
    assert git(source, "rev-parse", "--verify", first_ref)

    (source / "value.txt").write_text("first edit", encoding="utf-8")
    second, _, (location, cache) = capture_records(roots, state, attempts=3, build_env={})
    second_ref = second[0].ref
    save_cache(location, cache, second)
    assert second_ref != first_ref
    assert git(source, "show", second_ref + ":value.txt") == "first edit"
    assert _parity_refs(source) == [second_ref]
    git(source, "update-ref", second_ref + "-scm", second[0].source_head)

    # Reusing a retained capture must keep its current parity refs intact.
    third, _, (location, cache) = capture_records(roots, state, attempts=3, build_env={})
    assert third[0].ref == second_ref
    save_cache(location, cache, third)
    assert set(_parity_refs(source)) == {second_ref, second_ref + "-scm"}

    (source / "value.txt").write_text("second edit", encoding="utf-8")
    fourth, _, (location, cache) = capture_records(roots, state, attempts=3, build_env={})
    save_cache(location, cache, fourth)
    assert fourth[0].ref != second_ref
    assert git(source, "show", fourth[0].ref + ":value.txt") == "second edit"
    assert _parity_refs(source) == [fourth[0].ref]


def test_capture_cache_retention_keeps_current_file_and_bounds_count(tmp_path):
    directory = tmp_path / "source-inputs"
    directory.mkdir()
    now = time.time()
    for index in range(4):
        path = directory / ("capture-%d.json" % index)
        path.write_text("{}", encoding="utf-8")
        os.utime(path, (now - index * 86400, now - index * 86400))
    current = directory / "capture-current.json"
    current.write_text("{}", encoding="utf-8")
    os.utime(current, (now - 10 * 86400, now - 10 * 86400))

    prune_capture_caches(directory, keep=current, max_age_seconds=2 * 86400, max_files=1)
    assert current.is_file()
    assert not (directory / "capture-0.json").exists()
    assert not (directory / "capture-3.json").exists()
    assert len(list(directory.glob("capture-*.json"))) == 1


def test_failed_fixed_input_write_leaves_no_temporary_file(tmp_path, monkeypatch):
    from mindie_coordinator import parity_install

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(parity_install.json, "dump", broken)
    with pytest.raises(OSError, match="disk full"):
        parity_install.materialize_command(
            workspace_id="ws", runtime_id="rt",
            endpoint={"host": "h", "port": 22, "user": "root", "root": "/runtime"},
            sources={"app": "/tmp/app"}, workspace_root=tmp_path,
            source_snapshot={"id": "snapshot", "records": [{"relpath": "app"}]},
        )
    directory = tmp_path / ".mindie-local" / "fixed-inputs"
    assert directory.is_dir()
    assert list(directory.iterdir()) == []


def test_stale_fixed_input_descriptor_is_pruned_before_publish(tmp_path):
    directory = tmp_path / "fixed-inputs"
    directory.mkdir()
    stale = directory / "stale.json"
    stale.write_text("{}", encoding="utf-8")
    old = time.time() - 25 * 3600
    os.utime(stale, (old, old))
    descriptor = write_fixed_input_descriptor(directory, "current.json", {"id": "current"})
    assert descriptor.is_file()
    assert not stale.exists()
    assert json.loads(descriptor.read_text(encoding="utf-8")) == {"id": "current"}


def test_fixed_input_descriptor_is_reclaimed_after_sync(tmp_path):
    directory = tmp_path / "fixed-inputs"
    descriptor = write_fixed_input_descriptor(directory, "current.json", {"id": "current"})
    assert descriptor.is_file()
    assert prune_fixed_inputs(directory, max_age_seconds=0) == [str(descriptor)]
    assert not descriptor.exists()


def _cache_entry(path: Path, size: int, age: float, *, bundle: str | None = None) -> None:
    path.mkdir(parents=True)
    (path / "blob").write_bytes(b"x" * size)
    if bundle:
        (path / "native-content.json").write_text(
            json.dumps({"bundle": bundle}), encoding="utf-8")
    stamp = time.time() - age
    os.utime(path, (stamp, stamp))


def test_native_cache_lru_enforces_entry_caps_and_preserves_pinned(tmp_path):
    cache = tmp_path / "cache"
    stale_bundle, live_bundle = "a" * 64, "b" * 64
    stale_content, live_content, kernel = "c" * 64, "d" * 64, "e" * 64
    _cache_entry(cache / "bundles" / stale_bundle, 10, 900)
    _cache_entry(cache / stale_content, 10, 800, bundle=stale_bundle)
    _cache_entry(cache / "bundles" / live_bundle, 10, 200)
    _cache_entry(cache / live_content, 10, 100, bundle=live_bundle)
    _cache_entry(cache / "kernel" / kernel, 10, 1000)

    result = native_cache_pool.evict_native_cache(
        cache, keep={live_bundle, live_content}, max_entries=3, max_total_bytes=10 ** 9)
    assert result["status"] == "evicted"
    assert (cache / live_content).is_dir()
    assert (cache / "bundles" / live_bundle).is_dir()
    assert not (cache / "kernel" / kernel).exists()
    assert not list(cache.glob(".trash-*"))

    # The stale bundle only becomes unreferenced after its content entry goes.
    second = native_cache_pool.evict_native_cache(
        cache, keep={live_bundle, live_content}, max_entries=2, max_total_bytes=10 ** 9)
    assert "bundle:" + stale_bundle in second["removed"]
    assert not (cache / "bundles" / stale_bundle).exists()
    assert not (cache / stale_content).exists()


def test_native_cache_byte_cap_and_staging_sweep(tmp_path):
    cache = tmp_path / "cache"
    first, second = "1" * 64, "2" * 64
    _cache_entry(cache / first, 40, 700)
    _cache_entry(cache / second, 40, 600)
    stale = cache / ".publish-killed"
    stale.mkdir()
    (stale / "partial").write_bytes(b"x")
    old = time.time() - 7200
    os.utime(stale, (old, old))
    fresh = cache / ".native-content-fresh"
    fresh.mkdir()

    result = native_cache_pool.evict_native_cache(
        cache, keep=(), max_entries=16, max_total_bytes=1)
    assert result["bytes"] == 0
    assert result["entries"] == 0
    assert not stale.exists()
    assert fresh.exists()


def test_locked_native_cache_entry_is_never_evicted(tmp_path):
    cache = tmp_path / "cache"
    old, fresh = "a" * 64, "b" * 64
    _cache_entry(cache / old, 10, 900)
    _cache_entry(cache / fresh, 10, 10)
    with native_cache_pool._cache_entry_lock(cache, "content", old, exclusive=True,
                                             blocking=False) as acquired:
        assert acquired is True
        native_cache_pool.evict_native_cache(cache, keep={fresh}, max_entries=1)
        assert (cache / old).is_dir()
    native_cache_pool.evict_native_cache(cache, keep={fresh}, max_entries=1)
    assert not (cache / old).exists()


def test_failed_bundle_publish_cleans_publish_staging(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "vllm-ascend").mkdir(parents=True)
    (root / "vllm-ascend" / "a.so").write_bytes(b"library")
    (root / "vllm-ascend" / "smoke.json").write_text("{}", encoding="utf-8")
    manifest = {
        "schema_version": 1, "profile": {}, "profile_key": "p", "build_inputs": {},
        "build_key": "b" * 64, "runtime_root": str(root),
        "files": {"vllm-ascend/a.so": {"sha256": "0" * 64, "role": "library"}},
        "evidence": {"smoke": {"path": "vllm-ascend/smoke.json", "sha256": "0" * 64}},
    }
    monkeypatch.setattr(runtime_profile, "verify", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad bundle")))
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="bad bundle"):
        runtime_profile._publish_captured_bundle(root, cache, manifest, timings={})
    assert cache.is_dir()
    assert not list(cache.glob(".publish-*"))


def test_housekeep_prunes_old_terminal_rows_but_keeps_active_leases(tmp_path):
    coordinator = NpuCoordinator(tmp_path / "state", clock=lambda: NOW)

    def submission(task_id, state, updated):
        return (
            "INSERT INTO tasks(task_id,agent_id,requested_count,allow_external_busy,"
            "not_before,latest_start,estimated_duration_seconds,preemptible,priority,state,"
            "submitted_at,updated_at) VALUES(?,?,0,0,?,?,60,0,0,?,?,?)",
            (task_id, "agent-" + task_id, NOW, NOW, state, updated, updated),
        )

    with coordinator._transaction() as db:
        db.execute(*submission("old-terminal", "released", NOW - 8 * 86400))
        db.execute(*submission("live-active", "active", NOW - 30 * 86400))
        db.execute(
            "INSERT INTO holds(hold_id,owner,devices,not_before,end_at,reason,state,"
            "created_at,updated_at) VALUES('old-hold','owner','[]',?,?,'r','expired',?,?)",
            (NOW, NOW, NOW - 8 * 86400, NOW - 8 * 86400))
        db.execute(
            "INSERT INTO holds(hold_id,owner,devices,not_before,end_at,reason,state,"
            "created_at,updated_at) VALUES('live-hold','owner','[]',?,?,'r','active',?,?)",
            (NOW, NOW + 3600, NOW, NOW))
        db.execute("INSERT INTO ports(port,kind,task_id,owner,created_at) "
                   "VALUES(30001,'service','old-terminal','owner',?)", (NOW - 8 * 86400,))
        db.execute("INSERT INTO ports(port,kind,task_id,owner,created_at) "
                   "VALUES(30002,'service','live-active','owner',?)", (NOW,))
        db.execute("INSERT INTO events(at,kind,task_id,data) VALUES(?, 'old', NULL, '{}')",
                   (NOW - 8 * 86400,))
        db.executemany("INSERT INTO events(at,kind,task_id,data) VALUES(?, 'recent', NULL, '{}')",
                       [(NOW - 1,)] * 4200)
        old_message = {"at": "2020-01-01T00:00:00Z", "text": "old"}
        new_message = {"at": "2027-01-01T00:00:00Z", "text": "new"}
        db.execute("INSERT INTO messages(message_id,sender,sender_session,recipient,"
                   "recipient_session,data,created_at) VALUES('old','a','s','b','s',?,?)",
                   (json.dumps(old_message), NOW - 8 * 86400))
        db.execute("INSERT INTO messages(message_id,sender,sender_session,recipient,"
                   "recipient_session,data,created_at) VALUES('new','a','s','b','s',?,?)",
                   (json.dumps(new_message), NOW))

    coordinator.snapshot({"status": "ok", "devices": [0, 1, 2, 3],
                          "busy": {}, "free": [0, 1, 2, 3]})

    with sqlite3.connect(tmp_path / "state" / "coordinator.sqlite3") as db:
        assert db.execute("SELECT task_id FROM tasks").fetchall() == [("live-active",)]
        assert db.execute("SELECT hold_id FROM holds").fetchall() == [("live-hold",)]
        assert db.execute("SELECT port FROM ports").fetchall() == [(30002,)]
        assert db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 4096
        assert db.execute("SELECT COUNT(*) FROM events WHERE kind='old'").fetchone()[0] == 0
        assert db.execute("SELECT message_id FROM messages").fetchall() == [("new",)]


def test_shared_snapshot_refs_keep_only_the_newest_window(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    shared = tmp_path / "shared.git"

    def run(*args, env=None):
        return subprocess.check_output(
            ["git", "-C", str(source), *args], text=True, encoding="utf-8",
            env={**os.environ, **(env or {})}, **no_window_kwargs()).strip()

    run("init")
    run("config", "user.name", "Retention Test")
    run("config", "user.email", "retention@example.invalid")
    commits = []
    for index in range(66):
        (source / "value.txt").write_text(str(index), encoding="utf-8")
        stamp = "@%d +0000" % (1577836800 + index)
        env = {"GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp}
        run("add", "value.txt")
        run("commit", "-m", str(index), env=env)
        commit = run("rev-parse", "HEAD")
        tree = run("rev-parse", "HEAD^{tree}")
        commits.append(commit)
        assert copy_fixed_objects(source, shared, {"commit": commit, "tree": tree})

    refs = subprocess.check_output(
        ["git", "-C", str(shared), "for-each-ref", "--format=%(refname)",
         "refs/mindie/snapshots/"], text=True, encoding="utf-8",
        **no_window_kwargs()).splitlines()
    assert len(refs) == 64
    assert "refs/mindie/snapshots/" + commits[-1] in refs
    assert "refs/mindie/snapshots/" + commits[0] not in refs


def test_terminal_run_retention_skips_active_and_unknown_runs(tmp_path):
    runs = tmp_path / "runs"
    for name in ("terminal", "active", "unknown"):
        (runs / name).mkdir(parents=True)
    old = time.time() - 20 * 86400
    for path in runs.iterdir():
        os.utime(path, (old, old))
    removed = prune_terminal_run_directories(
        runs, {"terminal", "active"}, live_ids={"active"},
        max_age_seconds=14 * 86400, max_runs=128)
    assert removed == ["terminal"]
    assert (runs / "active").is_dir()
    assert (runs / "unknown").is_dir()


def test_message_inserted_without_created_at_survives_housekeep(tmp_path):
    coordinator = NpuCoordinator(tmp_path / "state")
    fresh = {"text": "fresh legacy insert"}
    with coordinator._transaction() as db:
        db.execute(
            "INSERT INTO messages(message_id,sender,sender_session,recipient,"
            "recipient_session,data) VALUES('fresh','a','s','b','s',?)",
            (json.dumps(fresh),))
        db.execute(
            "INSERT INTO messages(message_id,sender,sender_session,recipient,"
            "recipient_session,data,created_at) VALUES('zero','a','s','b','s',?,0)",
            (json.dumps({"at": "2027-01-01T00:00:00Z", "text": "legacy zero"}),))

    with sqlite3.connect(tmp_path / "state" / "coordinator.sqlite3") as db:
        inserted = db.execute(
            "SELECT created_at FROM messages WHERE message_id='fresh'").fetchone()[0]
        assert inserted > 1_000_000_000

    coordinator.snapshot({"status": "ok", "devices": [0, 1, 2, 3],
                          "busy": {}, "free": [0, 1, 2, 3]})

    with sqlite3.connect(tmp_path / "state" / "coordinator.sqlite3") as db:
        rows = db.execute(
            "SELECT message_id, created_at FROM messages ORDER BY message_id").fetchall()
    assert [row[0] for row in rows] == ["fresh", "zero"]
    assert all(row[1] > 1_000_000_000 for row in rows)


def test_failed_run_with_unreleased_preparation_is_never_pruned(tmp_path):
    from mindie_coordinator.agent_session import AgentSessions

    sessions = AgentSessions(tmp_path / "sessions")
    service = CoordinatorService(tmp_path / "coordinator", pool=mock.MagicMock(),
                                 sessions=sessions)
    runs = service.state_dir / "runs"
    for run_id in ("failed-inconclusive", "failed-active-lease", "failed-clean"):
        (runs / run_id).mkdir(parents=True)
    old = time.time() - 20 * 86400
    for path in runs.iterdir():
        os.utime(path, (old, old))
    retained = {
        "id": "failed-inconclusive", "session_id": "s", "phase": "failed",
        "preparation_jobs": {"role": {"prepare-1": {
            "state": "inconclusive", "quiet": False,
        }}},
    }
    active_lease = {
        "id": "failed-active-lease", "session_id": "s", "phase": "failed",
        "roles": [{"name": "role", "managed_job": "job-1",
                   "observation": {"state": "failed", "lease_state": "active"}}],
    }
    clean = {"id": "failed-clean", "session_id": "s", "phase": "failed"}
    sessions.save_execution(retained)
    sessions.save_execution(active_lease)
    sessions.save_execution(clean)

    service._prune_terminal_runs()

    assert (runs / "failed-inconclusive").is_dir()
    assert (runs / "failed-active-lease").is_dir()
    assert not (runs / "failed-clean").exists()


def test_daemon_progress_path_prunes_terminal_runs_with_cooldown(tmp_path):
    from mindie_coordinator.agent_session import AgentSessions
    from mindie_coordinator.storage_retention import RUN_PRUNE_COOLDOWN_SECONDS

    sessions = AgentSessions(tmp_path / "sessions")
    pool = mock.MagicMock()
    service = CoordinatorService(tmp_path / "coordinator", pool=pool, sessions=sessions)
    sessions.save_execution({"id": "terminal", "session_id": "s", "phase": "succeeded"})
    runs = service.state_dir / "runs"
    (runs / "terminal").mkdir(parents=True)
    old = time.time() - 20 * 86400
    os.utime(runs / "terminal", (old, old))

    service._dispatch_progress()

    assert not (runs / "terminal").exists()
    pool.tick.assert_called_once()

    # A second clean run cannot be pruned again until the cooldown elapses.
    sessions.save_execution({"id": "terminal-2", "session_id": "s", "phase": "succeeded"})
    (runs / "terminal-2").mkdir(parents=True)
    os.utime(runs / "terminal-2", (old, old))
    service._dispatch_progress()
    assert (runs / "terminal-2").is_dir()

    service._last_run_prune = time.time() - RUN_PRUNE_COOLDOWN_SECONDS - 1
    service._dispatch_progress()
    assert not (runs / "terminal-2").exists()


def test_parity_cleanup_timeout_never_discards_snapshot_receipt(tmp_path, monkeypatch):
    service = CoordinatorService(tmp_path / "coordinator", pool=mock.MagicMock())
    commits = {"vllm": "a" * 40, "vllm-ascend": "b" * 40}
    binding = {"runtime_id": "runtime-1", "endpoint": {},
               "intent": {"session": "workspace"}, "environment": {}, "build_env": {}}

    monkeypatch.setattr("mindie_coordinator.service.materialize_command",
                        lambda **kwargs: ["parity", "sync"])
    monkeypatch.setattr("mindie_coordinator.service.source_paths", lambda snapshot: {})

    def fake_run(args, **kwargs):
        kwargs["stdout"].write(json.dumps({"status": "ready", "snapshot_commits": commits}))
        kwargs["stdout"].flush()
        return subprocess.CompletedProcess(args, 0)

    def cleanup_timeout(*args, **kwargs):
        raise TimeoutError("fixed-input cleanup timed out")

    monkeypatch.setattr("mindie_coordinator.service.subprocess.run", fake_run)
    monkeypatch.setattr("mindie_coordinator.service.prune_fixed_inputs", cleanup_timeout)
    result = service.sync_binding(binding, {"id": "snapshot", "records": []}, "run-1")
    assert result == commits
    assert result.cleanup_warning["timeout"] is True
    assert (service.state_dir / "runs" / "run-1" / "runtime-1" / "parity.json").is_file()


def test_staging_sweep_requires_an_unheld_builder_lock(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    active = cache / ".publish-live"
    active.mkdir()
    (active / "partial").write_bytes(b"x")
    old = time.time() - 7200
    os.utime(active, (old, old))

    with native_cache_pool.hold_staging_lock(active):
        assert native_cache_pool.sweep_native_cache_staging(cache) == []
        assert active.is_dir()

    assert native_cache_pool.sweep_native_cache_staging(cache) == [active.name]
    assert not active.exists()


def test_schema_seven_messages_table_migrates_created_at(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    database = state / "coordinator.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute(
            "CREATE TABLE messages (cursor INTEGER PRIMARY KEY AUTOINCREMENT,"
            "message_id TEXT NOT NULL UNIQUE,sender TEXT NOT NULL,"
            "sender_session TEXT NOT NULL,recipient TEXT NOT NULL,"
            "recipient_session TEXT NOT NULL,data TEXT NOT NULL)")
        db.execute(
            "INSERT INTO messages(message_id,sender,sender_session,recipient,"
            "recipient_session,data) VALUES('legacy','a','s','b','s',?)",
            (json.dumps({"at": "2027-01-01T00:00:00Z", "text": "legacy"}),))

    NpuCoordinator(state)

    with sqlite3.connect(database) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(messages)")}
        created = db.execute(
            "SELECT created_at FROM messages WHERE message_id='legacy'").fetchone()[0]
    assert "created_at" in columns
    assert created > 1_000_000_000
