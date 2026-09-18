import threading
import time
from unittest.mock import patch

from mindie_coordinator.agent_session import AgentSessions
from mindie_coordinator.ready_runtime import RuntimePool
from mindie_coordinator.service import CoordinatorService
from mindie_coordinator.task_client import TaskClient
from test_coordinator import Backend, runtime_spec


def pool_fixture(tmp_path, hosts=1):
    backend = Backend(tmp_path / "host")
    pool = RuntimePool(tmp_path / "pool", backend)
    for index in range(hosts):
        spec = runtime_spec(index + 1, host=f"192.0.2.{index + 1}", recipe="rc")
        backend.mark_prepared(spec)
        pool.register(f"donor-{index}", spec)
    return pool, backend


def test_group_wait_uses_renewable_active_lease_past_old_grant_deadline(tmp_path):
    pool, backend = pool_fixture(tmp_path)
    now = [time.time()]
    backend.clock = pool.clock = lambda: now[0]
    session = pool.session_open("alice", "one-execution", {})
    binding = pool.checkout("alice", session["id"], "profile-a", "one", "donor-0")
    with patch.object(backend, "job", wraps=backend.job) as remote:
        job = pool.managed_start("alice", binding["id"], "one", {}, "native-a", [], 0, "true", {}, hold_go=True, queue_seconds=7200)
    prepared = next(call.kwargs["spec"] for call in remote.call_args_list if call.args[2] == "prepare")
    assert prepared["prepared_timeout_seconds"] == 7200
    assert prepared["timeout_seconds"] == 1800
    assert (job["state"], job["lease_state"], job["remote"]["state"]) == ("waiting", "active", "prepared")
    assert ("job", "go") not in backend.calls
    for _ in range(3):
        now[0] += 70  # each group wait exceeds the former 60-second grant window
        pool.managed_tick()
        job = pool.managed_control("alice", job["id"])
        assert (job["state"], job["lease_state"]) == ("waiting", "active")
    started = pool.managed_release_gate("alice", job["id"])
    assert started["state"] == "running"
    assert backend.calls.count(("job", "prepare")) == 1
    assert backend.calls.count(("job", "go")) == 1


def test_releasing_group_gate_survives_an_older_ticker_snapshot(tmp_path):
    pool, backend = pool_fixture(tmp_path)
    session = pool.session_open("alice", "one-execution", {})
    binding = pool.checkout("alice", session["id"], "profile-a", "one", "donor-0")
    stale = pool.managed_start("alice", binding["id"], "one", {}, "native-a", [], 0, "true", {}, hold_go=True)
    assert pool.managed_release_gate("alice", stale["id"])["state"] == "running"
    pool._save_managed(stale)
    with pool.transaction() as db:
        assert pool.get(db, "job", stale["id"])["hold_go"] is False
    assert pool.managed_control("alice", stale["id"])["state"] == "running"
    assert backend.calls.count(("job", "go")) == 1


def test_cancelled_role_closes_the_entire_group_before_go(tmp_path):
    pool, backend = pool_fixture(tmp_path, hosts=2)
    store = AgentSessions(tmp_path / "sessions")
    context = store.attach("codex", "cancelled-group", str(tmp_path))
    service = CoordinatorService(tmp_path / "coordinator", pool=pool, backend=backend, sessions=store)
    client = TaskClient(context["context_file"], service=service, user="alice")
    original = pool.managed_start

    def cancel_first(*args, **kwargs):
        job = original(*args, **kwargs)
        if args[2].endswith("role-0"):
            pool.managed_control("alice", job["id"], "stop")
            return pool.managed_control("alice", job["id"])
        return job

    roles = [{"name": f"role-{index}", "host": f"192.0.2.{index + 1}", "npu_count": 0}
             for index in range(2)]
    with patch.object(pool, "managed_start", side_effect=cancel_first):
        result = client.run("true", sources={}, topology={"roles": roles})
    assert result["state"] == "failed", result
    assert ("job", "go") not in backend.calls
    pool.managed_tick()
    with pool.transaction() as db:
        assert all(job["state"] == "cancelled" for job in pool.rows(db, "job"))


def test_four_hosts_start_gates_concurrently_after_all_leases_are_active(tmp_path):
    pool, backend = pool_fixture(tmp_path, hosts=4)
    store = AgentSessions(tmp_path / "sessions")
    context = store.attach("codex", "four-host-group", str(tmp_path))
    service = CoordinatorService(tmp_path / "coordinator", pool=pool, backend=backend, sessions=store)
    client = TaskClient(context["context_file"], service=service, user="alice")
    barrier = threading.Barrier(4)
    original = backend.job
    seen = []

    def job(runtime, job_id, action, **kwargs):
        if action == "go":
            with pool.transaction() as db:
                runs = pool.rows(db, "run")
            assert len(runs) == 4 and all(run["state"] == "active" for run in runs)
            seen.append(job_id)
            barrier.wait(timeout=5)
        return original(runtime, job_id, action, **kwargs)

    roles = [{"name": f"role-{index}", "host": f"192.0.2.{index + 1}", "npu_count": 0}
             for index in range(4)]
    with patch.object(backend, "job", side_effect=job):
        result = client.run("true", sources={}, topology={"roles": roles})
    assert result["state"] == "running", result
    assert len(set(seen)) == 4
    assert store.executions(context["session"]["id"])[0]["group_start_authorized"] is True


def test_busy_job_does_not_block_other_host_heartbeat_tick(tmp_path):
    pool, backend = pool_fixture(tmp_path, hosts=2)
    jobs = []
    for index in range(2):
        session = pool.session_open("alice", f"execution-{index}", {})
        binding = pool.checkout("alice", session["id"], "profile-a", f"checkout-{index}", f"donor-{index}")
        jobs.append(pool.managed_start("alice", binding["id"], f"run-{index}", {}, "native-a", [], 0, "true", {}))
    busy = pool._entity_lock("job", jobs[0]["id"])
    busy.acquire()
    backend.calls.clear()
    try:
        pool.managed_tick(limit=1)
    finally:
        busy.release()
    assert ("host", "heartbeat") in backend.calls


def test_single_host_constraint_does_not_fall_back_to_first_catalog_host(tmp_path):
    pool, backend = pool_fixture(tmp_path, hosts=2)
    store = AgentSessions(tmp_path / "sessions")
    context = store.attach("codex", "single-host-constraint", str(tmp_path))
    service = CoordinatorService(tmp_path / "coordinator", pool=pool, backend=backend, sessions=store)
    client = TaskClient(context["context_file"], service=service, user="alice")
    result = client.run("true", sources={}, topology={"host": "192.0.2.2"})
    assert result["state"] == "running", result
    assert result["target"]["endpoint"]["host"] == "192.0.2.2"
    assert result["roles"][0]["host"] == "192.0.2.2"
