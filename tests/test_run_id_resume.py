"""Admission identity is queryable after owner reconstruction without submit replay."""
import json
from mindie_coordinator.ready_runtime import RuntimePool
from mindie_coordinator.task_client import TaskClient
import test_coordinator


def test_run_id_survives_client_reconstruction_and_reads_original_process():
    task = test_coordinator.TaskClientTests()
    task.setUp()
    try:
        first = task.client.run("one-command", sources={}, resources={"npu_count": 0})
        assert first["run_id"] == first["execution_id"]
        job_id = next(iter(task.backend.jobs))
        calls = [action for kind, action in task.backend.calls if kind == "host"]
        submissions = calls.count("submit") + calls.count("submit-acquire") + calls.count("submit-acquire-preflight")
        resumed = TaskClient(task.context["context_file"],
            pool=RuntimePool(task.root / "manager", task.backend), user="alice")
        second = resumed.observe(run_id=first["run_id"])
        assert second["run_id"] == first["run_id"] and second["state"] == "running"
        assert list(task.backend.jobs) == [job_id]
        later = [action for kind, action in task.backend.calls if kind == "host"]
        assert later.count("submit") + later.count("submit-acquire") + later.count("submit-acquire-preflight") == submissions
        task.backend.jobs[job_id].update(state="succeeded", quiet=True, stdout="original-output", stderr="",
            result={"exit_code": 0, "descendants_drained": True, "timings": {"wall_seconds": 0.2}})
        assert resumed.observe(run_id=first["run_id"], refresh=True)["state"] == "succeeded"
        completed = resumed.observe(run_id=first["run_id"], action="wait", timeout_seconds=2)
        assert completed["run_id"] == first["run_id"] and completed["stdout"] == "original-output"
        assert completed["state"] == "succeeded" and completed["resources_released"]
        records = list((task.root / "manager" / "runs").glob("*.json"))
        assert len(records) == 1
        record = json.loads(records[0].read_text())
        assert record["run_id"] == first["run_id"]
        assert record["process"]["job_id"] == job_id
        with task.pool.transaction() as db:
            job = next(row for row in task.pool.rows(db, "job") if row["job_id"] == job_id)
        assert job["environment"]["MINDIE_RUN_ID"] == first["run_id"]
    finally:
        task.tearDown()
