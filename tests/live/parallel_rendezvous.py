"""Two genuine CPU-native slots with an in-container rendezvous.

Each process publishes readiness in one fresh shared container directory, waits
for exactly two participants, then loads and calls its own extension for a real
measured 2-second interval. Both receive canonical manifests/bench evidence;
a native-view reuse without a content-cache receipt is recorded as such.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import traceback
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = Path(__file__).resolve().parent
for directory in (REPO_ROOT, LIVE_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_source_native_acceptance as driver  # noqa: E402
from source_native_cpp_probe import (  # noqa: E402
    set_scm_tag, set_source_offset, set_vllm_probe_marker, write_cpp_probe_repos,
)
from source_native_evidence import cache_facts, execution_times  # noqa: E402
from mindie_coordinator.run_manifest import (  # noqa: E402
    bench_execution_evidence, project_run_manifest, write_manifest,
)


def _command() -> str:
    return '"$MINDIE_PYTHON" vllm/parallel_native_probe.py\n'


def _write_case(args, client, reply, execution_id: str, proof: dict, out: Path, receipt):
    row = driver._execution_row(client, execution_id)
    snapshot = (row.get("spec") or {}).get("source_snapshot") or {}
    source = next(item for item in snapshot["records"] if item["relpath"] == "vllm-ascend")
    created, started, ended = execution_times(row, reply, "succeeded")
    manifest = project_run_manifest(
        run_id=execution_id,
        code={"source_head": source["source_head"], "snapshot_commit": source["commit"],
              "dirty": bool(source.get("changed_paths"))},
        created_at=created, started_at=started, ended_at=ended, terminal_status="passed",
        topology={"host": args.host, "npu_count": 0},
        native_cache_receipt=receipt, active_execution_proof=proof,
        profile_summary_ref=proof.get("ready_profile"),
    )
    bench = bench_execution_evidence(manifest["active_execution_proof"], manifest_run_id=execution_id)
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out / "run-manifest.json", manifest)
    (out / "proof.json").write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    (out / "bench-execution-evidence.json").write_text(
        json.dumps(bench, indent=2, sort_keys=True) + "\n")
    (out / "execution.json").write_text(json.dumps({"row": row, "reply": reply}, indent=2,
                                                   sort_keys=True) + "\n")
    return {"manifest": str(out / "run-manifest.json"), "proof": proof,
            "bench": str(out / "bench-execution-evidence.json"),
            "execution_id": execution_id, "cache_receipt": receipt,
            "slot_id": ((reply.get("roles") or [{}])[0]).get("slot_id"),
            "slot_cleanup": ((reply.get("roles") or [{}])[0]).get("slot_cleanup")}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="173.125.1.2")
    p.add_argument("--container", default="mindie-bench-a2-core-20261017")
    p.add_argument("--container-id", default="b2bb20466bf75cb252a421dcdb1c72433f3d6a67f03bccd04e4d887a4ac6405f")
    p.add_argument("--user", default="bench-a2-core-20261017")
    p.add_argument("--soc", default="ascend910b1")
    p.add_argument("--cxx", default=os.environ.get("CXX") or "g++")
    p.add_argument("--agent-sessions-dir", default=str(Path.cwd() / ".mindie-003-wave4/native-live/agent-sessions"))
    p.add_argument("--output", type=Path, default=Path.cwd() / ".mindie-003-wave4/native-live/parallel-rendezvous")
    p.add_argument("--wait-timeout", type=int, default=600)
    p.add_argument("--business-timeout", type=int, default=600)
    ns = p.parse_args()
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
    sandbox = Path("/tmp/mindie-npu-test-native-parallel-" + uuid.uuid4().hex[:12])
    sandbox.mkdir(parents=True, exist_ok=True)
    sources = write_cpp_probe_repos(sandbox / "source", offset=1)
    set_vllm_probe_marker(sources["vllm"])
    probe_source = LIVE_DIR / "parallel_native_probe.py"
    (sources["vllm"] / probe_source.name).write_bytes(probe_source.read_bytes())
    driver._git(sources["vllm"], "add", probe_source.name)
    driver._git(sources["vllm"], "commit", "-qm", "add native rendezvous probe")
    set_source_offset(sources["vllm-ascend"], 12)
    set_scm_tag(sources["vllm-ascend"], "v0.0.3")
    nonce = int(uuid.uuid4().hex[:8], 16)
    expected = 60 + nonce
    flags = f"-DMINDIE_ACCEPTANCE_NONCE={nonce} -O2"
    barrier = "/tmp/mindie-npu-test-native-barrier-" + uuid.uuid4().hex[:12]
    env = {"MINDIE_PARALLEL_DIR": barrier, "MINDIE_PARALLEL_EXPECTED": "2",
           "MINDIE_PARALLEL_TIMEOUT": "300"}
    clients, cases, admitted = [], [], {}
    pending = False
    try:
        for index in range(2):
            client = driver._slot(ns, f"parallel-{index}", service)
            clients.append(client)
            execution_id = driver._admit(
                ns, client, sources, label=f"parallel-{index}", nonce=nonce, flags=flags,
                expected=expected, command_override=_command(), extra_env=env)
            admitted[f"parallel-{index}"] = execution_id
            (ns.output / "admitted.json").write_text(json.dumps(admitted, indent=2, sort_keys=True) + "\n")
        for index, client in enumerate(clients):
            execution_id = admitted[f"parallel-{index}"]
            reply = driver._wait(ns, client, execution_id)
            stdout = reply.get("stdout", "")
            if "MINDIE_ACCEPTANCE_PROOF " not in (stdout or ""):
                tail = client.observe(execution_id, action="tail")
                stdout = tail.get("stdout") or (tail.get("roles") or [{}])[0].get("stdout", "")
            if "MINDIE_ACCEPTANCE_PROOF " not in (stdout or ""):
                result_path = f"{barrier}/{execution_id}.result.json"
                stdout = "MINDIE_ACCEPTANCE_PROOF " + driver._remote(
                    endpoint, "cat " + shlex.quote(result_path))
            proof = driver._proof(stdout)
            evidence = client.observe(execution_id, action="evidence", section="build")
            cache = cache_facts(evidence, required=False)
            receipt = cache["receipt"]
            (ns.output / f"parallel-{index}-build-evidence.json").write_text(
                json.dumps(evidence, indent=2, sort_keys=True) + "\n")
            cases.append(_write_case(ns, client, reply, execution_id, proof,
                                     ns.output / f"parallel-{index}", receipt))
        windows = [(case["proof"]["measurement_window"]["start_ts"],
                    case["proof"]["measurement_window"]["end_ts"]) for case in cases]
        overlap_start, overlap_end = max(w[0] for w in windows), min(w[1] for w in windows)
        if overlap_end <= overlap_start:
            raise RuntimeError("business proof windows do not overlap")
        paths = {case["proof"]["loaded_module_path"] for case in cases}
        if len(paths) != 2:
            raise RuntimeError("both slots did not load their own extension path")
        for case in cases:
            proof = case["proof"]
            if proof.get("values") != [expected] or proof.get("call_count", 0) < 2:
                raise RuntimeError("repeated C++ calls did not remain correct")
            row = driver._execution_row(clients[cases.index(case)], case["execution_id"])
            root = row["roles"][0]["binding"]["endpoint"]["root"]
            if not proof.get("loaded_module_path", "").startswith(root):
                raise RuntimeError("loaded extension escaped its execution view")
        cleanup = [driver._assert_slot_cleanup(
            endpoint, ns, driver._execution_row(client, case["execution_id"]),
            driver._wait(ns, client, case["execution_id"]))
            for client, case in zip(clients, cases)]
        test_keys = {case["cache_receipt"]["cache_key"] for case in cases if case["cache_receipt"]}
        test_bundles = set()
        for case in cases:
            store = case["cache_receipt"] or {}
            if store.get("bundle"):
                test_bundles.add(store["bundle"])
        cleanup_result = driver._cleanup_cache(endpoint, test_keys, test_bundles)
        driver._remote(endpoint, "rm -rf " + shlex.quote(barrier))
        after = driver._cache_names(endpoint)
        lost = before - after - test_keys - test_bundles
        if lost:
            raise RuntimeError("unrelated cache entries disappeared")
        report = {"schema": "mindie-native-parallel-rendezvous/1", "status": "succeeded",
                  "scope": "CPU-only C++ native cache/process fixture; not full vllm-ascend/AscendC/NPU performance",
                  "run_ids": admitted, "barrier": barrier, "expected": expected,
                  "overlap": {"start": overlap_start, "end": overlap_end,
                              "seconds": overlap_end - overlap_start},
                  "cases": cases, "slot_cleanup": cleanup,
                  "cache_before": sorted(before), "cache_after": sorted(after),
                  "cache_cleanup": cleanup_result}
        (ns.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "succeeded", "report": str(ns.output / "report.json"),
                          "run_ids": admitted, "overlap_seconds": overlap_end - overlap_start}, indent=2))
        return 0
    except driver.AcceptancePending as exc:
        pending = True
        receipt = {"status": "pending", "admitted": admitted, "barrier": barrier,
                   "sandbox": str(sandbox), "reply": exc.reply,
                   "agent_sessions_dir": ns.agent_sessions_dir}
        (ns.output / "pending.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "pending", "admitted": admitted}))
        return 2
    except Exception:
        failure = {"status": "failed", "admitted": admitted, "barrier": barrier,
                   "traceback": traceback.format_exc()}
        (ns.output / "failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    finally:
        service._stopped.set()
        if not pending:
            try:
                driver._remote(endpoint, "rm -rf " + shlex.quote(barrier))
            except Exception as exc:
                (ns.output / "cleanup-error.json").write_text(json.dumps(
                    {"barrier": barrier, "error": str(exc)}, indent=2) + "\n")
                raise
            import shutil
            shutil.rmtree(sandbox)


if __name__ == "__main__":
    raise SystemExit(main())
