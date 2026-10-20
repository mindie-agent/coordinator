"""Prepare one isolated native donor root through the production preparation path.

This is the real `RemoteBackend.prepare_task_root` flow: fixed-source capture,
materialization, task-owned venv, install steps under the shared HostQueue compile
slots, finalize capture, and RuntimePool registration from the observed profile.
It does not create a container, image, global registry, or NPU workload.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = Path(__file__).resolve().parent
for directory in (REPO_ROOT, LIVE_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def _pin(host: str, container: str):
    from remote_dev.core.container_endpoint import pin_container_endpoint
    from remote_dev.core.endpoint import Endpoint
    return pin_container_endpoint(Endpoint(host=host, port=22, user="root", container=container,
                                           root="/", cwd="/", runtime_env=False), timeout_ms=15000)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="173.125.1.2")
    parser.add_argument("--container", default="mindie-bench-a2-core-20261017")
    parser.add_argument("--container-id", default="b2bb20466bf75cb252a421dcdb1c72433f3d6a67f03bccd04e4d887a4ac6405f")
    parser.add_argument("--user", default="bench-a2-core-20261017")
    parser.add_argument("--soc", default=os.environ.get("SOC_VERSION") or "ascend910b1")
    parser.add_argument("--state-root", type=Path,
                        default=Path.cwd() / ".mindie-003-wave4/native-live")
    parser.add_argument("--root", help="default /tmp/mindie-npu-test-native-<uuid>")
    parser.add_argument("--output", type=Path, help="default <state-root>/donor-preparation.json")
    args = parser.parse_args()
    args.state_root.mkdir(parents=True, exist_ok=True)
    coordinator = args.state_root / "coordinator"
    sessions = args.state_root / "agent-sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    os.environ["MINDIE_COORDINATOR_STATE_DIR"] = str(coordinator)
    os.environ["REMOTE_DEV_STATE_DIR"] = str(args.state_root / "remote-dev")
    root = args.root or "/tmp/mindie-npu-test-native-" + uuid.uuid4().hex[:12]
    os.environ.setdefault("CXX", "g++")
    os.environ.setdefault("SOC_VERSION", args.soc)
    output = args.output or args.state_root / "donor-preparation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    fixture = args.state_root / "fixtures" / ("base-" + uuid.uuid4().hex[:8])

    from mindie_coordinator.backend import RemoteBackend
    from mindie_coordinator.execution_sources import capture_sources
    from mindie_coordinator.ready_runtime import RuntimePool
    from source_native_cpp_probe import write_cpp_probe_repos
    from remote_dev.core.endpoint import Endpoint
    from remote_dev.core.ssh_transport import run_script

    sources = write_cpp_probe_repos(fixture, offset=1)
    snapshot = capture_sources(sources, coordinator)
    endpoint = _pin(args.host, args.container)
    container_id = run_script(
        Endpoint(host=args.host, port=22, user="root", root="/", cwd="/", runtime_env=False),
        "docker inspect --format '{{.Id}}' " + args.container, timeout_ms=20000).stdout.strip()
    if container_id != args.container_id:
        raise RuntimeError("pinned container identity differs from --container-id")
    python = root + "/.venv/bin/python"
    spec = {
        "user": args.user,
        "python": python,
        "recipe": None,
        "host_endpoint": {"host": args.host, "port": 22, "user": "root"},
        "endpoint": {"host": args.host, "port": int(endpoint.port), "user": "root",
                     "root": root, "cwd": root, "container": endpoint.container},
        "container_name": args.container,
        "service_ports": [],
        "machine_type": None,
        "source_snapshot": snapshot,
    }
    pool = RuntimePool(coordinator, RemoteBackend())
    image = pool.backend._preparation_container(spec)["Image"]
    records = {row["relpath"]: row for row in snapshot.get("records", [])}
    spec["preparation"] = {
        "source_id": snapshot["id"],
        "environment": {"image_digest": image, "soc": args.soc,
                        "build_env": snapshot.get("build_env", {})},
        "dependencies": {name: records[name].get("build_inputs", {}).get("dependencies")
                         for name in ("vllm", "vllm-ascend")},
        "native": {name: records[name].get("build_inputs", {}).get("native")
                   for name in ("vllm", "vllm-ascend")},
    }
    progress_path = args.state_root / "donor-progress.jsonl"
    job_path = args.state_root / "donor-jobs.jsonl"

    def job_update(job):
        with job_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(job, default=str, sort_keys=True) + "\n")

    def progress(event):
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, default=str, sort_keys=True) + "\n")

    try:
        prepared = pool.backend.prepare_task_root(
            spec, sources=sources, environment={"soc": args.soc}, source_snapshot=snapshot,
            on_progress=progress, log_dir=args.state_root / "donor-logs",
            on_preparation_job=job_update, cancel_requested=None,
        )
        if prepared is None:
            raise RuntimeError("native preparation returned no completed profile")
        runtime = pool.register("donor-" + uuid.uuid4().hex[:12], spec)
        result = {
            "status": "prepared",
            "root": root,
            "container": args.container,
            "container_id": container_id,
            "soc": args.soc,
            "source_snapshot": snapshot["id"],
            "source_records": [{key: record.get(key) for key in
                                ("relpath", "commit", "tree", "source_head", "scm_version")}
                               for record in snapshot.get("records", [])],
            "runtime_id": runtime["id"],
            "profile_key": (runtime.get("attestation") or {}).get("profile_key"),
            "build_key": (runtime.get("attestation") or {}).get("build_key"),
            "attestation_files": len((runtime.get("attestation") or {}).get("files") or {}),
            "native_receipt": getattr(prepared, "native_cache_receipt", None),
            "donor_jobs": str(job_path),
            "donor_progress": str(progress_path),
        }
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception:
        import traceback
        failure = {"status": "failed", "root": root, "traceback": traceback.format_exc(),
                   "donor_jobs": str(job_path), "donor_progress": str(progress_path)}
        output.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
