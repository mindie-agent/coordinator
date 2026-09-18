"""Fresh discovery combines exact task and container facts without housekeeping."""
import json
import time
import uuid

import pytest

from test_fresh_preflight import case
from test_preflight_parallel import launch
from mindie_coordinator.backend import RemoteBackend
from mindie_coordinator.host import mindie_npu_coordination as protocol


INFO = {"Id": "container-a", "State": {"Running": True, "Paused": False, "Restarting": False}}


def test_context_reads_only_exact_task_and_does_not_run_housekeeping(tmp_path, monkeypatch):
    host = protocol.NpuCoordinator(tmp_path)
    host.submit({"task_id": "other", "agent_id": "other-owner", "npu_count": 0})
    before = host.snapshot(None)
    monkeypatch.setattr(protocol.NpuCoordinator, "_housekeep", lambda *a, **k: pytest.fail("discovery cannot reclaim other tasks"))
    commands = []
    monkeypatch.setattr(protocol.subprocess, "check_output", lambda args, **kwargs: commands.append(args) or json.dumps(INFO))
    absent = host.startup_context("new", "mindie-owner")
    assert absent == {"status": "ok", "coordination_epoch": before["coordination_epoch"], "tasks": [], "container": INFO}
    existing = host.startup_context("other", "mindie-owner")
    assert existing["tasks"] == before["tasks"]
    assert commands[0][-1] == "mindie-owner" and commands[0][:3] == ["docker", "inspect", "--format"]


@pytest.mark.parametrize("field,value", [("Running", False), ("Paused", True), ("Restarting", True), ("Id", "replacement")])
def test_compact_verification_still_rejects_invalid_context_container(launch, monkeypatch, field, value):
    backend, runtime, reply = launch
    info = {"Id": INFO["Id"], "State": dict(INFO["State"])}
    (info if field == "Id" else info["State"])[field] = value
    monkeypatch.setattr(backend, "_inspect_container", lambda *a: pytest.fail("must use the same-round fact"))
    monkeypatch.setattr(backend, "_inspect_manifest", lambda *a, **k: reply)
    with pytest.raises((RuntimeError, ValueError), match="container"):
        backend.verify_preflight(runtime, _container_info=info)


def test_compact_verification_reuses_the_same_round_container_fact(launch, monkeypatch):
    backend, runtime, reply = launch
    monkeypatch.setattr(backend, "_inspect_container", lambda *a: pytest.fail("duplicate container RPC"))
    monkeypatch.setattr(backend, "_inspect_manifest", lambda *a, **k: reply)
    assert backend.verify_preflight(runtime, _container_info=INFO) is True


def test_managed_context_is_used_once_and_lifecycle_spans_are_bounded(case):
    started = time.monotonic_ns()
    binding = case.bind("alice", case.root / "a")
    contexts, verifications = [], []

    def context(runtime, task_id):
        contexts.append(task_id)
        return {**case.backend.host(runtime, {"action": "status", "no_probe": True}), "container": INFO}

    case.backend.startup_context = context
    case.backend.verify_preflight = lambda *a, **k: verifications.append(k) or True
    case.backend.busy = [0]
    job = case.managed("alice", binding)
    assert job["state"] == "queued" and verifications[0]["_container_info"] == INFO
    case.backend.busy = []
    assert case.pool.managed_advance(job["id"])["state"] == "running"
    assert len(contexts) == 1 and "_container_info" not in verifications[1]
    events = [row for row in case.pool.events("alice")["events"] if row["kind"] == "run-operation"]
    assert {row["operation"] for row in events} == {"startup-context", "verify-preflight", "admission", "preflight", "prepare", "activate", "go"}
    assert all(row["elapsed_seconds"] >= 0 for row in events)
    finished = time.monotonic_ns()
    assert all(set(row) <= {"cursor", "kind", "at", "run", "operation", "elapsed_seconds",
                           "monotonic_ns", "process_instance_id", "diagnostics_context"} for row in events)
    assert len({row["process_instance_id"] for row in events}) == 1
    for row in events:
        assert uuid.UUID(row["process_instance_id"]).version == 4
        assert type(row["monotonic_ns"]) is int and started <= row["monotonic_ns"] <= finished
        assert set(row["diagnostics_context"]) <= {"trace_id", "operation_id", "phase_id"}
        assert all(len(value) == 32 and int(value, 16) >= 0 for value in row["diagnostics_context"].values())
        assert len(json.dumps(row).encode()) < 2048
    assert [row["monotonic_ns"] for row in events] == sorted(row["monotonic_ns"] for row in events)
    case.pool.managed_advance(job["id"])
    assert len([row for row in case.pool.events("alice")["events"] if row["kind"] == "run-operation"]) == len(events)


def test_backend_context_uses_the_host_authority_with_exact_task():
    backend = RemoteBackend()
    calls = []
    backend.host = lambda runtime, request: calls.append(request) or {"container": INFO}
    runtime = {"container_name": "mindie-owner"}
    assert backend.startup_context(runtime, "new-task") == {"container": INFO}
    assert calls == [{"action": "startup-context", "task_id": "new-task", "container_name": "mindie-owner"}]
