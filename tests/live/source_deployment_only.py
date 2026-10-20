"""Run only the real Git source deployment case through TaskClient."""
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

import run_source_native_acceptance as driver  # noqa: E402
from source_native_cpp_probe import source_file_hashes  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="173.125.1.2")
    p.add_argument("--container", default="mindie-bench-a2-core-20261017")
    p.add_argument("--container-id", default="b2bb20466bf75cb252a421dcdb1c72433f3d6a67f03bccd04e4d887a4ac6405f")
    p.add_argument("--user", default="bench-a2-core-20261017")
    p.add_argument("--soc", default="ascend910b1")
    p.add_argument("--cxx", default=os.environ.get("CXX") or "g++")
    p.add_argument("--agent-sessions-dir", default=str(Path.cwd() / ".mindie-003-wave4/native-live/agent-sessions"))
    p.add_argument("--output", type=Path, default=Path.cwd() / ".mindie-003-wave4/native-live/source-only")
    p.add_argument("--wait-timeout", type=int, default=600)
    p.add_argument("--business-timeout", type=int, default=300)
    p.add_argument("--real-vllm")
    p.add_argument("--real-vllm-ascend", help="actual existing vllm-ascend Git checkout (read-only)")
    p.add_argument("--source-archive", help="explicit source archive to deploy read-only")
    p.add_argument("--source-archive-commit", help="verified source commit represented by the archive")
    ns = p.parse_args()
    driver.use_local_state(ns.agent_sessions_dir)
    ns.output = ns.output.resolve()
    ns.output.mkdir(parents=True, exist_ok=True)
    service = driver._local_service(ns)
    driver._preflight_donor(ns, service)
    sandbox = Path("/tmp/mindie-npu-test-native-source-" + uuid.uuid4().hex[:10])
    sandbox.mkdir(parents=True, exist_ok=True)
    clients = []
    try:
        real = driver._real_sources(ns, sandbox)
        if real is None:
            raise RuntimeError("actual Git source pair/frozen archive is unavailable")
        client = driver._slot(ns, "real-source", service)
        clients.append(client)
        execution_id = driver._admit(
            ns, client, real["sources"], label="real-source", nonce=0, flags="",
            expected=0, command_override=driver._source_command(real["archive_commit"]))
        reply = driver._wait(ns, client, execution_id)
        proof = driver._source_proof(reply.get("stdout", ""))
        if proof.get("file_sha256") != source_file_hashes(real["sources"]):
            raise RuntimeError("real source file hashes differ from origin")
        snapshot = (driver._execution_row(client, execution_id).get("spec") or {}).get("source_snapshot") or {}
        records = {row["relpath"]: row for row in snapshot.get("records", [])}
        for relpath, expected in (("vllm-source", real["vllm_head"]),
                                  ("vllm-ascend-source", real["vllm_ascend_head"])):
            if relpath not in records or records[relpath].get("source_head") != expected:
                raise RuntimeError("source snapshot source_head differs from origin")
            if records[relpath].get("commit") != proof[
                    "vllm_head" if relpath == "vllm-source" else "vllm_ascend_head"]:
                raise RuntimeError("materialized source commit differs from snapshot")
        cleanup = driver._assert_slot_cleanup(
            {"host": ns.host, "port": 22, "user": "root",
             "container": ns.container_id, "root": "/", "cwd": "/"},
            ns, driver._execution_row(client, execution_id), driver._wait(ns, client, execution_id))
        case = {"schema": "mindie-real-source-deployment/1", "status": "succeeded",
                "run_id": execution_id, "context_file": client.context["context_file"],
                "mode": "source-deployment-only", "proof": proof, "snapshot": snapshot,
                "slot_cleanup": cleanup}
        (ns.output / "source-deployment.json").write_text(
            json.dumps(case, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "succeeded", "run_id": execution_id,
                          "proof_sha256": proof.get("file_sha256", {}),
                          "report": str(ns.output / "source-deployment.json")}, indent=2))
        return 0
    finally:
        service._stopped.set()
        import shutil
        shutil.rmtree(sandbox, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
