"""Finish only the two CPU-slot and real-source cases after accepted cache cases.

The accepted cold/hit/changed cases are not re-submitted. This helper uses the
same production TaskClient/local CoordinatorService and HostQueue path.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = Path(__file__).resolve().parent
for directory in (REPO_ROOT, LIVE_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_source_native_acceptance as driver  # noqa: E402
from source_native_cpp_probe import (  # noqa: E402
    set_scm_tag, set_source_offset, set_vllm_probe_marker, source_file_hashes,
    write_cpp_probe_repos,
)


def _args() -> SimpleNamespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="173.125.1.2")
    p.add_argument("--container", default="mindie-bench-a2-core-20261017")
    p.add_argument("--container-id", default="b2bb20466bf75cb252a421dcdb1c72433f3d6a67f03bccd04e4d887a4ac6405f")
    p.add_argument("--user", default="bench-a2-core-20261017")
    p.add_argument("--soc", default="ascend910b1")
    p.add_argument("--cxx", default=os.environ.get("CXX") or "g++")
    p.add_argument("--agent-sessions-dir", default=str(Path.cwd() / ".mindie-003-wave4/native-live/agent-sessions"))
    p.add_argument("--output", type=Path, default=Path.cwd() / ".mindie-003-wave4/native-live/parallel-source")
    p.add_argument("--wait-timeout", type=int, default=600)
    p.add_argument("--business-timeout", type=int, default=300)
    p.add_argument("--cold-key", default="711130a68bf6696dc42619bbbd0a2698e984c28551f00aeeb700f997581a2188")
    p.add_argument("--cold-relative", default="vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so")
    p.add_argument("--real-vllm")
    p.add_argument("--real-vllm-ascend")
    p.add_argument("--source-archive", help="explicit source archive to deploy read-only")
    p.add_argument("--source-archive-commit", help="verified source commit represented by the archive")
    p.add_argument("--keep-sandbox", action="store_true")
    p.add_argument("--skip-source", action="store_true")
    p.add_argument("--concurrent", action="store_true")
    return p.parse_args()


def main() -> int:
    ns = _args()
    driver.use_local_state(ns.agent_sessions_dir)
    ns.output = ns.output.resolve()
    ns.output.mkdir(parents=True, exist_ok=True)
    service = driver._local_service(ns)
    driver._preflight_donor(ns, service)
    from remote_dev.core.container_endpoint import pin_container_endpoint
    from remote_dev.core.endpoint import Endpoint
    host = {"host": ns.host, "port": 22, "user": "root"}
    pinned = pin_container_endpoint(Endpoint(**host, container=ns.container, root="/", cwd="/",
                                             runtime_env=False), timeout_ms=15000)
    endpoint = {**host, "container": pinned.container, "root": "/", "cwd": "/"}
    before = driver._cache_names(endpoint)
    sandbox = Path("/tmp/mindie-npu-test-native-parallel-" + uuid.uuid4().hex[:10])
    sandbox.mkdir(parents=True, exist_ok=True)
    sources = write_cpp_probe_repos(sandbox / "base", offset=1)
    set_vllm_probe_marker(sources["vllm"])
    set_source_offset(sources["vllm-ascend"], 9)
    set_scm_tag(sources["vllm-ascend"], "v0.0.2")
    status, active = driver._compile_slots(ns)
    max_slots = int(status["max_compile_slots"])
    if max_slots < 2:
        raise RuntimeError("HostQueue max_compile_slots is below two")
    nonce = int(uuid.uuid4().hex[:8], 16)
    flags = f"-DMINDIE_ACCEPTANCE_NONCE={nonce} -O2"
    clients, cases = [], []
    try:
        test_keys, test_bundles = set(), set()
        slots = []
        for index in range(2):
            client = driver._slot(ns, f"parallel-{index}", service)
            clients.append(client)
            slots.append((index, client))
        submitted = []

        def submit(item):
            index, client = item
            started = time.time()
            execution_id = driver._admit(
                ns, client, sources, label=f"parallel-{index}", nonce=nonce, flags=flags,
                expected=57 + nonce, cache_key=ns.cold_key, cache_relative=ns.cold_relative)
            return {"index": index, "client": client, "execution_id": execution_id,
                    "admission_started_at": started, "admission_returned_at": time.time()}

        if ns.concurrent:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="parallel-admit") as workers:
                submitted = list(workers.map(submit, slots))
        else:
            submitted = [submit(item) for item in slots]
        for item in submitted:
            index, client, execution_id = item["index"], item["client"], item["execution_id"]
            reply = driver._wait(ns, client, execution_id)
            case = driver._write_evidence(
                ns, client, label=f"parallel-{index}", execution_id=execution_id, reply=reply,
                proof=driver._proof(reply.get("stdout", "")),
                cache=driver.cache_facts(client.observe(execution_id, action="evidence", section="build")),
                source_repo=sources["vllm-ascend"])
            receipt = case["cache"]["receipt"]
            if receipt.get("hit") not in {True, False}:
                raise RuntimeError("parallel slot has no cache outcome")
            test_keys.add(receipt["cache_key"])
            if (case["cache"]["store"] or {}).get("bundle"):
                test_bundles.add(case["cache"]["store"]["bundle"])
            case["admission_window"] = {"started_at": item["admission_started_at"],
                                        "returned_at": item["admission_returned_at"]}
            case["proof_window"] = case["proof"]["measurement_window"]
            cases.append(case)
        active_after = [task for task in driver._compile_slots(ns)[1]
                        if task.get("task_id") not in {row["slot_id"] for row in cases}]
        cleanup = [driver._assert_slot_cleanup(
            endpoint, ns, driver._execution_row(client, case["run_id"]),
            driver._wait(ns, client, case["run_id"])) for client, case in zip(clients, cases)]

        if ns.skip_source:
            cleanup_result = driver._cleanup_cache(endpoint, test_keys, test_bundles)
            after = driver._cache_names(endpoint)
            lost = before - after - test_keys - test_bundles
            if lost:
                raise RuntimeError("cleanup removed unrelated cache entries")
            report = {"schema": "mindie-native-parallel-only/1", "status": "succeeded",
                      "max_compile_slots": max_slots, "active_before": active,
                      "active_other_after": active_after, "parallel": cases,
                      "parallel_slot_cleanup": cleanup,
                      "cache_before": sorted(before), "cache_after": sorted(after),
                      "cache_cleanup": cleanup_result}
            (ns.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                                  encoding="utf-8")
            print(json.dumps({"status": "succeeded", "report": str(ns.output / "report.json"),
                              "parallel_run_ids": [row["run_id"] for row in cases],
                              "max_compile_slots": max_slots}, indent=2))
            return 0
        real = driver._real_sources(ns, sandbox)
        if real is None:
            raise RuntimeError("actual Git source pair/frozen archive is unavailable")
        source_client = driver._slot(ns, "real-source", service)
        clients.append(source_client)
        source_id = driver._admit(
            ns, source_client, real["sources"], label="real-source", nonce=0, flags="",
            expected=0, command_override=driver._source_command(real["archive_commit"]))
        source_reply = driver._wait(ns, source_client, source_id)
        source_proof = driver._source_proof(source_reply.get("stdout", ""))
        if source_proof.get("file_sha256") != source_file_hashes(real["sources"]):
            raise RuntimeError("real source file hashes differ from origin")
        snapshot = (driver._execution_row(source_client, source_id).get("spec") or {}).get("source_snapshot") or {}
        records = {row["relpath"]: row for row in snapshot.get("records", [])}
        for relpath, expected in (("vllm-source", real["vllm_head"]),
                                  ("vllm-ascend-source", real["vllm_ascend_head"])):
            if relpath not in records or records[relpath].get("source_head") != expected:
                raise RuntimeError("source snapshot source_head differs from origin")
            if records[relpath].get("commit") != source_proof[
                    "vllm_head" if relpath == "vllm-source" else "vllm_ascend_head"]:
                raise RuntimeError("materialized source commit differs from snapshot")
        source_case = {"label": "real-source", "run_id": source_id,
                       "context_file": source_client.context["context_file"],
                       "mode": "source-deployment-only", "proof": source_proof,
                       "snapshot": snapshot}
        out = ns.output / "real-source"
        out.mkdir(parents=True, exist_ok=True)
        (out / "source-deployment.json").write_text(
            json.dumps(source_case, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        source_cleanup = driver._assert_slot_cleanup(
            endpoint, ns, driver._execution_row(source_client, source_id),
            driver._wait(ns, source_client, source_id))
        cleanup_result = driver._cleanup_cache(endpoint, test_keys, test_bundles)
        after = driver._cache_names(endpoint)
        lost = before - after - test_keys - test_bundles
        if lost:
            raise RuntimeError("cleanup removed unrelated cache entries")
        if test_keys & after or test_bundles & after:
            raise RuntimeError("test-owned cache entries survived cleanup")
        report = {"schema": "mindie-native-parallel-source/1",
                  "status": "succeeded", "max_compile_slots": max_slots,
                  "active_before": active, "active_other_after": active_after,
                  "parallel": cases, "parallel_slot_cleanup": cleanup,
                  "source": source_case, "source_slot_cleanup": source_cleanup,
                  "cache_before": sorted(before), "cache_after": sorted(after),
                  "cache_cleanup": cleanup_result}
        (ns.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                              encoding="utf-8")
        print(json.dumps({"status": "succeeded", "report": str(ns.output / "report.json"),
                          "parallel_run_ids": [row["run_id"] for row in cases],
                          "source_run_id": source_id, "max_compile_slots": max_slots}, indent=2))
        return 0
    finally:
        service._stopped.set()
        if not ns.keep_sandbox:
            import shutil
            shutil.rmtree(sandbox, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
