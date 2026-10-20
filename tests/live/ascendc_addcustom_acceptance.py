"""Genuine AscendC AddCustom: CPU build/publish, fresh restore, NPU correctness.

Uses the public CANN KernelLaunch AddCustom sample, actual CANN CMake/bisheng
toolchain, production TaskClient/HostQueue, and production native content pool
modules staged as sample source. Build/publish and NPU phases are separate
executions; NPU runs under npu_count=1 only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = Path(__file__).resolve().parent
for directory in (REPO_ROOT, LIVE_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_source_native_acceptance as driver  # noqa: E402
from source_native_evidence import execution_times  # noqa: E402
from mindie_coordinator.run_manifest import (  # noqa: E402
    bench_execution_evidence, project_run_manifest, write_manifest,
)


def _marker(text: str, marker: str) -> dict:
    for line in reversed((text or "").splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise RuntimeError(marker.strip() + " missing")


def _remote_root(row: dict, source: str) -> str:
    return row["roles"][0]["binding"]["endpoint"]["root"] + "/" + source


def _read_json(endpoint, path: str):
    return json.loads(driver._remote(endpoint, "cat " + shlex.quote(path)))


def canonical_proof(raw_proof: dict, build_proof: dict) -> dict:
    """Bind the runtime receipt to its actual immutable build preparation.

    The patch digest is a derived source/build identity. Runtime timestamps,
    paths, digests, and calls remain the original remote receipt values.
    """
    if raw_proof["cache_key"] != build_proof["cache_key"]:
        raise RuntimeError("NPU receipt refers to a different native cache entry")
    if raw_proof["source_tree"] != build_proof["source_tree"]:
        raise RuntimeError("NPU receipt refers to a different source tree")
    preparation = build_proof["preparation"]
    payload = json.dumps(preparation, sort_keys=True, separators=(",", ":")).encode()
    return {**raw_proof, "patch_digest": hashlib.sha256(payload).hexdigest()}


def _build(ns, service, source, *, flags: str):
    client = driver._slot(ns, "ascendc-build", service)
    execution_id = driver._admit(
        ns, client, {"addcustom": source}, label="ascendc-build", nonce=0, flags="",
        expected=0, command_override="bash addcustom/build_sample.sh",
        extra_env={"MINDIE_ASCENDC_CXXFLAGS": flags}, timeout_seconds=600)
    reply = driver._wait(ns, client, execution_id)
    row = driver._execution_row(client, execution_id)
    root = _remote_root(row, "addcustom")
    # Read the actual build receipt through the execution's own root/endpoint.
    endpoint = row["roles"][0]["binding"]["endpoint"]
    proof = _read_json(endpoint, root + "/build-proof.json")
    return client, execution_id, reply, row, proof


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="173.125.1.2")
    p.add_argument("--container", default="mindie-bench-a2-core-20261017")
    p.add_argument("--container-id", default="b2bb20466bf75cb252a421dcdb1c72433f3d6a67f03bccd04e4d887a4ac6405f")
    p.add_argument("--user", default="bench-a2-core-20261017")
    p.add_argument("--soc", default="ascend910b4")
    p.add_argument("--cxx", default=os.environ.get("CXX") or "g++")
    p.add_argument("--agent-sessions-dir", default=str(Path.cwd() / ".mindie-003-wave4/native-live/agent-sessions"))
    p.add_argument("--output", type=Path, default=Path.cwd() / ".mindie-003-wave4/native-live/ascendc-complete")
    p.add_argument("--source", type=Path, default=Path.cwd() / ".mindie-003-wave4/native-live/ascendc-add-source")
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
    builds, clients, test_keys, test_bundles = [], [], set(), set()
    pending = False
    try:
        for index, flags in enumerate(("", "-DADD_CUSTOM_ACCEPTANCE=1")):
            client, execution_id, reply, row, proof = _build(
                ns, service, ns.source, flags=flags)
            clients.append(client)
            build = {"execution_id": execution_id, "flags": flags,
                     "proof": proof, "row": row, "root": _remote_root(row, "addcustom")}
            builds.append(build)
            test_keys.add(proof["cache_key"])
            if (proof.get("publish") or {}).get("bundle"):
                test_bundles.add(proof["publish"]["bundle"])
            (ns.output / f"build-{index}.json").write_text(
                json.dumps(build, indent=2, sort_keys=True) + "\n")
        if builds[0]["proof"]["cache_key"] == builds[1]["proof"]["cache_key"]:
            raise RuntimeError("compiler flag did not change the native content key")
        key = builds[0]["proof"]["cache_key"]
        # Fresh TaskClient root and NPU execution; restore is the only source
        # of the binary/library used on device.
        npu_client = driver._slot(ns, "ascendc-npu", service)
        clients.append(npu_client)
        npu_id = driver._admit(
            ns, npu_client, {"addcustom": ns.source}, label="ascendc-npu", nonce=0, flags="",
            expected=0, command_override="python3 addcustom/restore_npu.py",
            extra_env={"MINDIE_ASCENDC_CACHE_KEY": key}, resources={"npu_count": 1},
            timeout_seconds=600)
        npu_reply = driver._wait(ns, npu_client, npu_id)
        npu_row = driver._execution_row(npu_client, npu_id)
        npu_root = _remote_root(npu_row, "addcustom")
        raw_proof = _read_json(npu_row["roles"][0]["binding"]["endpoint"], npu_root + "/npu-proof.json")
        npu_proof = canonical_proof(raw_proof, builds[0]["proof"])
        snapshot = (npu_row.get("spec") or {}).get("source_snapshot") or {}
        source = next(item for item in snapshot["records"] if item["relpath"] == "addcustom")
        receipt = {"hit": True, "cache_key": key, "linked_files": int(npu_proof["linked_files"])}
        created, started, ended = execution_times(npu_row, npu_reply, "succeeded")
        manifest = project_run_manifest(
            run_id=npu_id, code={"source_head": source["source_head"],
                                  "snapshot_commit": source["commit"],
                                  "dirty": bool(source.get("changed_paths"))},
            created_at=created, started_at=started, ended_at=ended,
            terminal_status="passed", topology={"host": ns.host, "npu_count": 1},
            native_cache_receipt=receipt, active_execution_proof=npu_proof)
        out = ns.output / "npu"
        out.mkdir(parents=True, exist_ok=True)
        (out / "raw-proof.json").write_text(json.dumps(raw_proof, indent=2, sort_keys=True) + "\n")
        write_manifest(out / "run-manifest.json", manifest)
        (out / "proof.json").write_text(json.dumps(npu_proof, indent=2, sort_keys=True) + "\n")
        bench = bench_execution_evidence(manifest["active_execution_proof"], manifest_run_id=npu_id)
        (out / "bench-execution-evidence.json").write_text(
            json.dumps(bench, indent=2, sort_keys=True) + "\n")
        cleanup = driver._cleanup_cache(endpoint, test_keys, test_bundles)
        after = driver._cache_names(endpoint)
        lost = before - after - test_keys - test_bundles
        if lost:
            raise RuntimeError("unrelated cache entries disappeared")
        report = {"schema": "mindie-ascendc-addcustom/1", "status": "succeeded",
                  "scope": "real public CANN AddCustom AscendC sample, actual NPU binary; not full vLLM",
                  "source": {"repository": "https://gitee.com/ascend/samples.git",
                             "sample_commit": "6511a5f4a45a1f68bd5e617989e68560f2f35cd6",
                             "path": "operator_contrib/AddCustomSample/KernelLaunch/AddCustomKernel",
                             "soc": "Ascend910B4", "npu_count_build": 0, "npu_count_run": 1},
                  "builds": builds, "cache_key": key, "npu": npu_proof,
                  "canonical_manifest": str(out / "run-manifest.json"),
                  "bench": str(out / "bench-execution-evidence.json"),
                  "cache_receipt": receipt, "cache_cleanup": cleanup,
                  "proof_derivation": {"runtime_receipt": str(out / "raw-proof.json"),
                                       "patch_digest_basis": "sha256 of canonical immutable build preparation"},
                  "cache_after_owned_removed": sorted(after), "run_id": npu_id,
                  "root": npu_root}
        (ns.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "succeeded", "report": str(ns.output / "report.json"),
                          "run_id": npu_id, "cache_key": key,
                          "npu_sha": npu_proof["loaded_binary_sha256"],
                          "linked_files": npu_proof["linked_files"]}, indent=2))
        return 0
    except driver.AcceptancePending as exc:
        pending = True
        client, execution_id = driver._PENDING
        receipt = {"status": "pending", "run_id": execution_id,
                   "context_file": client.context["context_file"],
                   "agent_sessions_dir": ns.agent_sessions_dir,
                   "reply": exc.reply, "builds": builds}
        (ns.output / "pending.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "pending", "run_id": execution_id}))
        return 2
    except Exception:
        failure = {"status": "failed", "traceback": traceback.format_exc(),
                   "builds": builds, "cache_before": sorted(before)}
        (ns.output / "failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    finally:
        service._stopped.set()
        if not pending:
            errors = []
            for client in clients:
                try:
                    client.finish(force=True)
                except Exception as exc:
                    errors.append(str(exc))
            if errors:
                (ns.output / "cleanup-error.json").write_text(json.dumps({"errors": errors}, indent=2) + "\n")
                raise RuntimeError("accepted-task cleanup failed: " + "; ".join(errors))


if __name__ == "__main__":
    raise SystemExit(main())
