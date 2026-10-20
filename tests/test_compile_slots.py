"""Compile capacity and NPU capacity share fencing but never block each other."""
from mindie_coordinator.host.mindie_npu_coordination import NpuCoordinator


FREE = {"status": "ok", "devices": [0], "busy": {}, "free": [0]}


def submit(host, name, count=0):
    host.submit({"task_id": name, "agent_id": "owner-" + name, "npu_count": count})


def test_cpu_capacity_keeps_npu_fifo_independent_and_releases_only_one_slot(tmp_path):
    host = NpuCoordinator(tmp_path, max_compile_slots=2)
    for name in ("compile-a", "compile-b", "compile-c"):
        submit(host, name)
    first = host.acquire("compile-a", None)["task"]
    host.acquire("compile-b", None)
    assert host.acquire("compile-c", None)["reason"] == "compile_slots_full"
    submit(host, "execute", 1)
    assert host.acquire("execute", FREE)["task"]["granted_devices"] == [0]
    host.cancel("compile-a", None)
    assert host.acquire("compile-c", None)["task"]["granted_devices"] == []
    states = {row["task_id"]: row["state"] for row in host.snapshot(None)["tasks"]}
    assert states == {"compile-a": "cancelled", "compile-b": "granted", "compile-c": "granted", "execute": "granted"}
    assert first["requested_count"] == 0


def test_unstarted_cpu_grant_expiry_recovers_capacity_without_npu_probe(tmp_path):
    clock = [100.0]
    host = NpuCoordinator(tmp_path, max_compile_slots=1, clock=lambda: clock[0])
    submit(host, "compile-a")
    host.acquire("compile-a", None, grant_ttl_seconds=1)
    submit(host, "compile-b")
    clock[0] += 2
    assert host.acquire("compile-b", None)["status"] == "granted"
    assert host.snapshot(None, task_id="compile-a")["tasks"][0]["state"] == "expired"


def test_live_compiler_retains_slot_after_heartbeat_and_needs_completion(tmp_path, monkeypatch):
    clock = [100.0]
    host = NpuCoordinator(tmp_path, max_compile_slots=1, clock=lambda: clock[0])
    submit(host, "compile-a")
    token = host.acquire("compile-a", None)["task"]["fence_token"]
    host.preflight("compile-a", token, None)
    alive = [True]
    monkeypatch.setattr("mindie_coordinator.host.mindie_npu_coordination.process_guard_busy",
                        lambda value, *, completion_confirmed=False: bool(value) and (alive[0] or not completion_confirmed))
    guard = {"marker": "a" * 32, "boot_id": "test", "retain_until_release": True}
    host.activate("compile-a", token, pid=123, process_guard=guard, heartbeat_ttl_seconds=1)
    submit(host, "compile-b")
    clock[0] += 2
    assert host.acquire("compile-b", None)["reason"] == "compile_slots_full"
    alive[0] = False
    assert host.release("compile-a", token, None)["status"] == "orphaned_busy"
    assert host.release("compile-a", token, None, completion_confirmed=True)["status"] == "released"
    assert host.acquire("compile-b", None)["status"] == "granted"
