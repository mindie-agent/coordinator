"""Non-destructive A3 CPU lease and same-process resume acceptance.

[Role] Run an actual Python call through host admission and remote-dev supervision.
[Invariants] Zero NPU devices; one submit and one supervisor; sandbox-only writes.
[Timeouts & Locks] 20s admission, 30s command, 15s observation and cleanup budgets.
[Side-Effect Policy] Creates and removes one /tmp/mindie-npu-test-* tree; never
reconfigures the existing container or touches another task's process/cache.
"""
from __future__ import annotations

import argparse
import base64
import copy
import difflib
import hashlib
import json
import shlex
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mindie_coordinator.backend import RemoteBackend
from mindie_coordinator.compile_lease import CompileLease, finish_compile_lease
from mindie_coordinator.host_queue import HostQueue
from mindie_coordinator.preparation_process import PreparationProcess, stop_preparation_process
from remote_dev.core.container_endpoint import pin_container_endpoint
from remote_dev.core.endpoint import Endpoint
from remote_dev.core.ssh_transport import run_script
from remote_dev.processes import control


CANARY_SOURCE = '''import hashlib, json, os, time
from pathlib import Path

def observed_call(input_id):
    start = time.time()
    module = Path(__file__).resolve()
    digest = hashlib.sha256(module.read_bytes()).hexdigest()
    with Path("invocations.txt").open("a") as out:
        out.write(input_id + "\\n")
    print("CANARY_CALL_STARTED", flush=True)
    time.sleep(3)
    proof = {
        "run_id": os.environ["MINDIE_SLOT_ID"],
        "patch_digest": os.environ["MINDIE_CANARY_PATCH_DIGEST"],
        "loaded_binary_sha256": digest,
        "loaded_module_path": str(module),
        "calls": [{"input_id": input_id, "call_path": "canary_module.observed_call", "timestamp": start}],
        "candidate_build_key": "python-canary:" + digest,
        "measurement_window": {"start_ts": start, "end_ts": time.time()},
        "evidence_ref": str(Path("proof.json").resolve()),
    }
    Path("proof.json").write_text(json.dumps(proof))
    print(json.dumps({"active_execution_proof": proof}), flush=True)
    print("CANARY_CALL_COMPLETED", flush=True)

observed_call("compile-resume-canary")
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.13.165")
    parser.add_argument("--container", default="vaws-offload-pd165-20261016")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--observe", type=Path)
    args = parser.parse_args()
    if args.observe:
        record = json.loads(args.observe.read_text())
        print(json.dumps(control(record["endpoint"], record["job_id"], "status")))
        return
    if args.output is None:
        parser.error("--output is required")
    args.output.mkdir(parents=True, exist_ok=True)
    remote_root = "/tmp/mindie-npu-test-compile-resume-" + uuid.uuid4().hex[:12]
    host = {"host": args.host, "port": 22, "user": "root"}
    pinned = pin_container_endpoint(Endpoint(**host, container=args.container,
        root=remote_root, cwd=remote_root, runtime_env=False), timeout_ms=15000)
    endpoint = {**host, "container": pinned.container, "root": remote_root, "cwd": remote_root}
    backend = RemoteBackend()
    queue = HostQueue()
    actions, records, output = [], [], []
    record_file = args.output / "preparation-record.json"
    running = threading.Event()
    cancelled = threading.Event()
    started = time.monotonic()

    def request(target, payload):
        actions.append(payload["action"])
        return queue.request(target, {**payload, "state_dir": remote_root + "/host-queue"})

    def save(record):
        records.append(copy.deepcopy(record))
        temporary = record_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(record_file)
        if record.get("state") == "running":
            running.set()

    def cleanup(target, slot):
        backend.cleanup_slot({"endpoint": target}, slot)

    patch = "".join(difflib.unified_diff([], CANARY_SOURCE.splitlines(keepends=True),
        fromfile="/dev/null", tofile="canary_module.py"))
    patch_digest = hashlib.sha256(patch.encode()).hexdigest()
    (args.output / "candidate.patch").write_text(patch)
    encoded = base64.b64encode(CANARY_SOURCE.encode()).decode()
    create = "import base64,pathlib; pathlib.Path('canary_module.py').write_bytes(base64.b64decode(" + repr(encoded) + "))"
    command = "mkdir -p -- " + shlex.quote(remote_root) + "\ncd -- " + shlex.quote(remote_root) + "\npython3 -c " + shlex.quote(create)
    prepared = run_script(pinned, command, timeout_ms=15000)
    if prepared.returncode != 0:
        raise RuntimeError("canary file creation failed: " + prepared.stderr[-500:])
    process = PreparationProcess(endpoint, "install-vllm-ascend", save, cancelled.is_set,
        timeout_seconds=30, compile_lease=CompileLease(host, args.container, request,
            queue_seconds=20, cleanup=cleanup))
    command = "export MINDIE_CANARY_PATCH_DIGEST=" + shlex.quote(patch_digest) + "\npython3 canary_module.py"
    try:
        with ThreadPoolExecutor(max_workers=1) as workers:
            future = workers.submit(process.run, command, on_output=lambda channel, text: output.append((channel, text)))
            if not running.wait(timeout=20):
                cancelled.set()
                future.result(timeout=15)
                raise AssertionError("no running observation")
            admission = json.loads(record_file.read_text())
            resumed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--observe", str(record_file)],
                capture_output=True, text=True, timeout=15, check=True)
            observed = json.loads(resumed.stdout)
            result = future.result(timeout=35)
        final = json.loads(record_file.read_text())
        tail = control(endpoint, final["job_id"], "tail")
        snapshot = request(host, {"action": "status", "no_probe": True})
        tasks = snapshot["tasks"]
        count = run_script(pinned, "wc -l < " + shlex.quote(remote_root + "/invocations.txt"), timeout_ms=15000)
        slot = shlex.quote("/tmp/mindie-slot-" + final["job_id"])
        cleaned = run_script(pinned, "test ! -e " + slot, timeout_ms=15000)
        assert result.returncode == 0 and final["state"] == "succeeded"
        assert final["compile_lease"]["state"] == "released"
        assert final["slot_cleanup"]["state"] == "completed" and cleaned.returncode == 0
        assert len(tasks) == 1 and tasks[0]["requested_count"] == 0
        assert actions.count("submit-acquire-preflight") == 1
        assert count.returncode == 0 and int(count.stdout.strip()) == 1
        assert admission["job_id"] == final["job_id"] == observed["receipt"]["job_id"]
        assert "CANARY_CALL_COMPLETED" in tail["stdout"]
        proof = next(json.loads(line)["active_execution_proof"] for channel, line in output
                     if channel == "stdout" and line.startswith('{"active_execution_proof":'))
        assert proof["run_id"] == final["job_id"] and proof["patch_digest"] == patch_digest
        assert proof["loaded_binary_sha256"] == hashlib.sha256(CANARY_SOURCE.encode()).hexdigest()
        assert tail["result"]["started_at"] <= tail["result"]["finished_at"]
        evidence = {"scope": "real A3 CPU admission and remote supervision; no NPU workload or performance claim",
            "host": args.host, "container": args.container, "run_id": final["job_id"],
            "admission": admission, "resumed_observation": observed, "final": final,
            "stdout": tail["stdout"], "stderr": tail["stderr"], "result": tail["result"],
            "host_tasks": tasks, "host_actions": actions, "active_execution_proof": proof,
            "invocation_count": 1, "elapsed_seconds": time.monotonic() - started}
        (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2))
        print(json.dumps({"run_id": final["job_id"], "state": final["state"],
            "lease_state": final["compile_lease"]["state"], "same_id_observation": observed["state"],
            "invocation_count": 1, "elapsed_seconds": evidence["elapsed_seconds"],
            "evidence": str(args.output / "evidence.json")}))
    finally:
        cancelled.set()
        if records:
            latest = records[-1]
            if not latest.get("quiet"):
                if not stop_preparation_process(latest, save, force=True, lease_request=request, cleanup=cleanup):
                    raise RuntimeError("canary stop is unresolved; sandbox retained for same-id observation: " + str(record_file))
            else:
                finish_compile_lease(latest, save, request=request, cleanup=cleanup)
        for target in (pinned, Endpoint(**host, root="/tmp", cwd="/tmp", runtime_env=False)):
            deleted = run_script(target, "rm -rf -- " + shlex.quote(remote_root), timeout_ms=15000)
            if deleted.returncode != 0:
                raise RuntimeError("canary sandbox cleanup failed: " + deleted.stderr[-500:])


if __name__ == "__main__":
    main()
