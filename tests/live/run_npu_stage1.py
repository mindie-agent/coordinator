"""Durable real-NPU acceptance through the existing host and remote-dev APIs.

[Role] Admit, observe and release one existing benchmark command on an idle NPU.
[Invariants] A saved run_id precedes submit; resume only queries the original job.
[Timeouts & Locks] 120s start gate; 30s control RPC; 120s renewable host lease.
[Side-Effect Policy] Uses the shared host authority, one owned /tmp task directory,
and its slot caches. It never replaces another task or retries uncertain writes.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
import uuid
from pathlib import Path

from mindie_coordinator.backend import RemoteBackend
from mindie_coordinator.execution_slots import slot_environment, slot_preamble
from mindie_coordinator.host_queue import HostQueue
from remote_dev.core.container_endpoint import pin_container_endpoint
from remote_dev.core.endpoint import Endpoint
from remote_dev.processes import control


TERMINAL = {"released", "cancelled", "expired"}


def save(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(path)


def journal(path, event, **facts):
    with path.with_name("experiment-ledger.jsonl").open("a") as stream:
        stream.write(json.dumps({"event": event, "at": time.time(), **facts}) + "\n")


def host_call(path, record, action, **kwargs):
    request = {"action": action, **kwargs}
    if record.get("coordination_epoch"):
        request["coordination_epoch"] = record["coordination_epoch"]
    journal(path, "host_request", run_id=record["run_id"], request=request)
    result = HostQueue().request(record["host_endpoint"], request)
    journal(path, "host_response", run_id=record["run_id"], action=action, result=result)
    return result


def launch(args):
    path = args.record
    if path.exists():
        raise RuntimeError("record already exists; observe its run_id instead of submitting again")
    path.parent.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or "stage1-" + uuid.uuid4().hex
    host = {"host": args.host, "port": 22, "user": "root"}
    pinned = pin_container_endpoint(Endpoint(**host, container=args.container,
        root=args.root, cwd=args.root, runtime_env=False), timeout_ms=15000)
    endpoint = {**host, "container": pinned.container, "root": args.root,
                "cwd": args.root, "runtime_env": False}
    record = {"run_id": run_id, "job_id": run_id, "host_endpoint": host,
        "endpoint": endpoint, "container_name": args.container, "devices": args.devices,
        "submitted_at": time.time(), "command": args.command, "state": "pending"}
    save(path, record)
    path.with_name("PROGRESS.md").write_text(
        "# Stage 1 real NPU task\n\nin_flight_runs:\n- run_id: " + run_id +
        "\n  task: fused-rmsnorm-gate-dispatch-performance\n  record: " + str(path) +
        "\n  resume: observe this record; never launch it again\n")
    journal(path, "identity_registered", run_id=run_id, record=str(path))
    startup = host_call(path, record, "startup-context", task_id=run_id,
                        container_name=args.container)
    record.update(coordination_epoch=startup["coordination_epoch"],
                  container_id=startup["container"]["Id"])
    save(path, record)
    admitted = host_call(path, record, "submit-acquire-preflight", task_id=run_id,
        agent_id="mindie-stage1", session_id=run_id, container_name=args.container,
        devices=args.devices, allow_external_busy=False, queue_ttl_seconds=120,
        start_ttl_seconds=120, estimated_duration_seconds=args.timeout)
    record["task"] = admitted["task"]
    record["state"] = admitted["task"]["state"]
    save(path, record)
    if record["state"] != "starting":
        raise RuntimeError("requested physical devices were not admitted; saved task requires observation")
    environment = {**slot_environment(run_id), "MINDIE_RUN_ID": run_id,
                   "ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, admitted["task"]["granted_devices"]))}
    spec = {"command": slot_preamble(run_id) + "\n" + args.command,
        "cwd": args.root, "env": environment, "timeout_seconds": args.timeout,
        "prepared_timeout_seconds": 120}
    observed = control(endpoint, run_id, "prepare", spec=spec)
    record["remote"] = observed
    save(path, record)
    if observed["state"] != "prepared":
        raise RuntimeError("remote supervisor did not confirm the closed start gate")
    if args.ownership_marker_file:
        marker = observed["receipt"]["process_guard"]["marker"]
        source = "from pathlib import Path; Path(" + repr(args.ownership_marker_file) + ").write_text(" + repr(marker) + ")"
        published = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "root@" + args.host,
             shlex.join(["python3", "-c", source])],
            capture_output=True, text=True, timeout=15, check=True,
        )
        journal(path, "ownership_marker_published", run_id=run_id,
                marker_path=args.ownership_marker_file, exit_code=published.returncode)
    active = host_call(path, record, "activate", task_id=run_id,
        fence_token=record["task"]["fence_token"], heartbeat_ttl_seconds=120,
        process_guard=observed["receipt"]["process_guard"], prepared_supervisor={
            "container_name": args.container, "container_id": record["container_id"],
            "receipt": observed["receipt"]})
    record["task"] = active["task"]
    save(path, record)
    if record["task"]["state"] != "active":
        raise RuntimeError("host lease is not active; remote start gate remains closed")
    record["remote"] = control(endpoint, run_id, "go", authorization={"run_id": run_id,
        "epoch": record["coordination_epoch"], "fence": record["task"]["fence_token"]})
    record["state"] = record["remote"]["state"]
    save(path, record)
    journal(path, "launched", run_id=run_id, job_id=run_id, state=record["state"])
    return record


def observe(args):
    path = args.record
    record = json.loads(path.read_text())
    run_id = record["run_id"]
    remote = control(record["endpoint"], run_id, "tail")
    snapshot = host_call(path, record, "status", task_id=run_id, no_probe=True, event_limit=0)
    task = next((item for item in snapshot["tasks"] if item["task_id"] == run_id), None)
    if task is None:
        raise RuntimeError("original acknowledged host task is missing; never resubmit")
    if not remote.get("quiet") and task["state"] in {"active", "orphaned_busy"}:
        task = host_call(path, record, "heartbeat", task_id=run_id,
            fence_token=task["fence_token"], heartbeat_ttl_seconds=120)["task"]
    record.update(state=remote["state"], remote=remote, task=task)
    save(path, record)
    journal(path, "resumed_observation", run_id=run_id, job_id=remote.get("receipt", {}).get("job_id"),
        state=remote["state"], quiet=remote["quiet"], lease_state=task["state"])
    if args.finish:
        if not remote.get("quiet"):
            raise RuntimeError("remote process is still live; lease release is forbidden")
        if task["state"] not in TERMINAL:
            action = "cancel" if task["state"] in {"queued", "granted", "starting"} else "release"
            record["task"] = host_call(path, record, action, task_id=run_id,
                fence_token=task.get("fence_token"), completion_confirmed=True)["task"]
        save(path, record)
        if record["task"]["state"] not in TERMINAL:
            raise RuntimeError("host lease did not release")
        try:
            RemoteBackend().cleanup_slot({"endpoint": record["endpoint"]}, run_id)
            record["slot_cleanup"] = {"state": "completed"}
        except Exception as exc:
            record["slot_cleanup"] = {"state": "failed", "error": str(exc)}
        save(path, record)
        journal(path, "finished", run_id=run_id, state=record["state"],
            lease_state=record["task"]["state"], slot_cleanup=record["slot_cleanup"])
        path.with_name("PROGRESS.md").write_text(
            "# Stage 1 real NPU task\n\nin_flight_runs: []\n\nCompleted run_id: " + run_id +
            "\nRemote outcome: " + record["state"] + "\nLease: " + record["task"]["state"] + "\n")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["launch", "observe"])
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--run-id", help="predeclared identity for accompanying read-only samplers")
    parser.add_argument("--host", default="173.125.1.2")
    parser.add_argument("--container", default="mindie-bench-a2-core-20261017")
    parser.add_argument("--devices", nargs="+", type=int, default=[0])
    parser.add_argument("--root")
    parser.add_argument("--command")
    parser.add_argument("--ownership-marker-file", help="owned host sampler path, written before opening the start gate")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--finish", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    if args.action == "launch" and (not args.root or not args.command):
        parser.error("launch requires --root and --command")
    if args.wait and args.action != "observe":
        parser.error("--wait is observation only")
    finish = args.finish
    if args.wait:
        args.finish = False
        deadline = time.monotonic() + args.timeout + 45
        while True:
            result = observe(args)
            if result["remote"].get("quiet"):
                if finish:
                    args.finish = True
                    result = observe(args)
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("observation budget expired; saved run_id remains queryable")
            time.sleep(10)
    else:
        result = launch(args) if args.action == "launch" else observe(args)
    print(json.dumps({"run_id": result["run_id"], "state": result["state"],
        "lease_state": (result.get("task") or {}).get("state"),
        "quiet": (result.get("remote") or {}).get("quiet"), "record": str(args.record)}))


if __name__ == "__main__":
    main()
