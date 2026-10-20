"""Managed slots isolate writable caches and preserve the completed job result."""
import json
import sqlite3

import pytest

from mindie_coordinator.execution_slots import slot_environment
from mindie_coordinator.ready_runtime import RuntimePool
from test_coordinator import Backend, runtime_spec


class SlotBackend(Backend):
    def __init__(self, state):
        super().__init__(state)
        self.commands = {}
        self.cleaned = []
        self.cleanup_failure = False

    def job(self, runtime, job_id, action, **parameters):
        if action == "prepare":
            self.commands[job_id] = parameters["spec"]
        return super().job(runtime, job_id, action, **parameters)

    def cleanup_slot(self, runtime, slot_id):
        if self.cleanup_failure:
            raise OSError("cache cleanup unavailable")
        assert self.jobs["mindie-" + slot_id]["quiet"]
        self.cleaned.append(slot_id)


def start(pool, number):
    session = pool.session_open("alice", "session-" + str(number), {})
    pool.register("runtime-" + str(number), runtime_spec(number))
    binding = pool.checkout("alice", session["id"], "profile-a", "checkout-" + str(number), "runtime-" + str(number))
    return pool.managed_start("alice", binding["id"], "run-" + str(number), {}, "native-a", [], 0,
                              "compile-and-execute", {}, 60)


def test_same_container_has_distinct_slots_and_completion_keeps_sibling_caches(tmp_path):
    backend = SlotBackend(tmp_path / "host")
    pool = RuntimePool(tmp_path / "pool", backend)
    first, second = start(pool, 1), start(pool, 2)
    for job in (first, second):
        spec = backend.commands[job["job_id"]]
        for name, value in slot_environment(job["id"]).items():
            assert spec["env"][name] == value
            assert "export " + name + "=" + value in spec["command"]
    assert first["slot_id"] != second["slot_id"]
    assert not backend.cleaned
    backend.jobs[first["job_id"]].update(state="succeeded", quiet=True, result={"exit_code": 0})
    ended = pool.managed_control("alice", first["id"])
    assert ended["state"] == "succeeded" and ended["slot_cleanup"]["status"] == "completed"
    assert backend.cleaned == [first["slot_id"]]
    assert pool.managed_control("alice", second["id"])["state"] == "running"


def test_cache_cleanup_failure_does_not_replace_success_and_is_retained_in_receipt(tmp_path):
    backend = SlotBackend(tmp_path / "host")
    pool = RuntimePool(tmp_path / "pool", backend)
    job = start(pool, 1)
    backend.jobs[job["job_id"]].update(state="succeeded", quiet=True, result={"exit_code": 0})
    backend.cleanup_failure = True
    ended = pool.managed_control("alice", job["id"])
    assert (ended["state"], ended["lease_state"]) == ("succeeded", "released")
    assert ended["slot_cleanup"] == {"status": "failed", "error": "cache cleanup unavailable"}
    record = json.loads((tmp_path / "pool" / "runs" / (job["id"] + ".json")).read_text())
    assert record["process"]["result"]["exit_code"] == 0
    assert record["slot_cleanup"] == ended["slot_cleanup"]


def test_export_failure_preserves_completed_execution_and_never_relaunches(tmp_path, monkeypatch):
    backend = SlotBackend(tmp_path / "host")
    pool = RuntimePool(tmp_path / "pool", backend)
    job = start(pool, 1)
    backend.jobs[job["job_id"]].update(state="succeeded", quiet=True, result={"exit_code": 0})
    def unavailable(*args, **kwargs):
        raise PermissionError("receipt directory read-only")
    monkeypatch.setattr(pool, "export_execution_record", unavailable)
    ended = pool.managed_control("alice", job["id"])
    assert (ended["state"], ended["lease_state"]) == ("succeeded", "released")
    assert ended["recording_error"] == {"artifact": "managed-run", "error": "receipt directory read-only"}
    assert pool.managed_control("alice", job["id"])["state"] == "succeeded"
    assert [action for kind, action in backend.calls if kind == "job"].count("go") == 1


@pytest.mark.parametrize("failure", [sqlite3.OperationalError("database is locked"), TimeoutError("local recording timeout")])
def test_final_database_failure_preserves_remote_result_and_recovers_same_job(tmp_path, monkeypatch, failure):
    backend = SlotBackend(tmp_path / "host")
    pool = RuntimePool(tmp_path / "pool", backend)
    job = start(pool, 1)
    backend.jobs[job["job_id"]].update(state="succeeded", quiet=True,
        stdout="one real result", result={"exit_code": 0, "timings": {"wall_seconds": 0.25}})
    put = pool.put
    def fail_completed(db, kind, value):
        if kind == "job" and value.get("state") == "succeeded":
            raise failure
        return put(db, kind, value)
    monkeypatch.setattr(pool, "put", fail_completed)
    ended = pool.managed_control("alice", job["id"])
    assert (ended["state"], ended["lease_state"]) == ("succeeded", "released")
    assert ended["remote"]["stdout"] == "one real result"
    assert ended["remote"]["result"]["timings"]["wall_seconds"] == 0.25
    assert ended["recording_error"]["artifact"] == "job-state"
    assert ended["recording_error"]["timeout"] is isinstance(failure, TimeoutError)
    recovered = RuntimePool(tmp_path / "pool", backend).managed_control("alice", job["id"])
    assert recovered["state"] == "succeeded" and recovered["job_id"] == job["job_id"]
    assert [action for kind, action in backend.calls if kind == "job"].count("go") == 1
