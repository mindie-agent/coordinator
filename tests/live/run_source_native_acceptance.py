"""Live CPU-only acceptance for source deployment and native content caching.

[Role]
Runs real isolated source builds through the existing TaskClient/HostQueue path
and records actual loaded-extension, call, cache and slot evidence.

[Invariants]
- Only npu_count=0 slots are requested; no NPU workload or performance claim.
- Sources are isolated temporary Git repositories, never existing external
  repositories. The C++ probe reuses the real vllm-ascend math_utils header but
  is explicitly a C++ cache probe, not an AscendC/NPU kernel compile.
- Cold/missed keys are unique to this run. Cleanup removes only recorded test
  keys/bundles; every pre-existing cache entry must survive.
- A cache hit is accepted only when the loaded extension is the same inode as
  the recorded content-pool file; logs and proof are kept immutable.

[Timeouts & Locks]
- Per-slot wait is bounded by --wait-timeout; HostQueue polling uses a short
  read-only status call. Build admission remains queue-owned.

[Side-Effect Policy]
- Writes only a per-run /tmp/mindie-npu-test-* sandbox, the recorded unique
  cache keys/bundles, and its output directory. No model, NPU, external repo,
  container, image, or registry writes. The cache is the existing container
  /tmp cache; no host /tmp bind is required.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Direct script execution must use this checkout, not an older installed wheel.
REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = Path(__file__).resolve().parent
for directory in (REPO_ROOT, LIVE_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from mindie_coordinator.agent_session import AgentSessions  # noqa: E402
from mindie_coordinator.backend import RemoteBackend  # noqa: E402
from mindie_coordinator.host_queue import HostQueue  # noqa: E402
from mindie_coordinator.ready_runtime import RuntimePool  # noqa: E402
from mindie_coordinator.service import CoordinatorService  # noqa: E402
from mindie_coordinator.run_manifest import write_manifest  # noqa: E402
from mindie_coordinator.task_client import TaskClient  # noqa: E402

TERMINAL_LEASE = {"released", "cancelled", "expired", "not_started"}
_ENV_LOCK = threading.Lock()
_PENDING = None


def use_local_state(agent_sessions_dir: str) -> None:
    """Force the coordinator state beside the supplied isolated sessions dir."""
    parent = Path(agent_sessions_dir).expanduser().resolve().parent
    os.environ["MINDIE_COORDINATOR_STATE_DIR"] = str(parent / "coordinator")
    os.environ["REMOTE_DEV_STATE_DIR"] = str(parent / "remote-dev")

# Copied verbatim from the real vllm-ascend source tree (csrc/kernels/math_utils.h)
# so the C++ probe exercises an authentic source file without mutating any repo.
from source_native_cpp_probe import (  # noqa: E402
    _commit, _git, local_candidate_build_key, patch_digest, proof_command,
    set_scm_tag, set_source_offset, set_vllm_probe_marker, source_file_hashes,
    write_cpp_probe_repos,
)
from source_native_evidence import (  # noqa: E402
    _archive_repo, _cache_names, _cleanup_cache, _clone_readonly, _real_sources,
    _remote, _source_command, _source_proof,
    cache_facts, epoch_iso, execution_times, normalize_receipt, project_manifest,
)


def _local_service(args):
    """Deliberate in-process production service; no daemon/pool mock."""
    store = AgentSessions(Path(args.agent_sessions_dir).expanduser().resolve())
    coordinator = Path(os.environ["MINDIE_COORDINATOR_STATE_DIR"])
    pool = RuntimePool(coordinator, RemoteBackend())
    service = CoordinatorService(coordinator, pool=pool, backend=pool.backend, sessions=store)
    # Admission runs synchronously; start the supervisor ticker only after the
    # first admission has completed so it cannot race preparation re-entry.
    service._async_progress = False
    service._acceptance_ticker = None
    return service


def _ensure_ticker(service):
    if getattr(service, "_acceptance_ticker", None) is None:
        service._acceptance_ticker = threading.Thread(
            target=service._tick_loop, name="acceptance-local-tick", daemon=True)
        service._acceptance_ticker.start()


def _slot(args, label: str, service):
    state_dir = Path(args.agent_sessions_dir).expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    store = AgentSessions(state_dir)
    context = store.attach("codex", f"source-native-{label}-{uuid.uuid4().hex}", str(args.output))
    return TaskClient(context["context_file"], user=args.user, service=service)


def _admit(args, client, sources: dict[str, Path], *, label: str, nonce: int, flags: str,
           expected: int, cache_key: str = "", cache_relative: str = "",
           command_override: str | None = None, extra_env: dict | None = None,
           resources: dict | None = None, timeout_seconds: int | None = None) -> str:
    env = {
        "MINDIE_ACCEPTANCE_INPUT": label,
        "MINDIE_ACCEPTANCE_EXPECTED": str(expected),
        "MINDIE_ACCEPTANCE_CACHE_KEY": cache_key,
        "MINDIE_ACCEPTANCE_CACHE_RELATIVE": cache_relative,
        **(extra_env or {}),
    }
    with build_environment(flags=flags, soc=args.soc, cxx=args.cxx):
        reply = client.run(
            command_override or proof_command(expected),
            sources={name: str(path) for name, path in sources.items()},
            env=env, environment={},
            resources=resources or {"npu_count": 0}, topology={"host": args.host},
            timeout_seconds=timeout_seconds or args.business_timeout, wait_until=None,
        )
    _ensure_ticker(client.coordinator)
    if not reply.get("execution_id"):
        raise RuntimeError("admission returned no execution id: " + json.dumps(reply, sort_keys=True)[:1000])
    if reply.get("state") in {"failed", "cancelled", "inconclusive"}:
        raise RuntimeError("admission did not start: " + json.dumps(reply, sort_keys=True)[:1000])
    return reply["execution_id"]


class AcceptancePending(RuntimeError):
    def __init__(self, reply: dict):
        super().__init__("acceptance wait is still in flight")
        self.reply = reply


def _wait(args, client, execution_id: str) -> dict:
    reply = client.wait(execution_id, until="released", timeout_seconds=args.wait_timeout)
    if reply.get("wait_timed_out"):
        # Preserve the accepted run id and active remote work. Never stop,
        # finish, clean a slot, or remove cache keys while it is busy.
        global _PENDING
        _PENDING = (client, execution_id)
        raise AcceptancePending(reply)
    if reply.get("state") != "succeeded" or reply.get("resources_released") is not True:
        raise RuntimeError("execution did not succeed/release: " + json.dumps(reply, sort_keys=True)[:2000])
    return reply


def _proof(stdout: str) -> dict:
    for line in reversed((stdout or "").splitlines()):
        marker = "MINDIE_ACCEPTANCE_PROOF "
        if line.startswith(marker):
            value = json.loads(line[len(marker):])
            if not isinstance(value, dict):
                raise RuntimeError("acceptance proof is not an object")
            return value
    raise RuntimeError("acceptance proof marker is missing from execution stdout")


def _execution_row(client, execution_id: str) -> dict:
    rows = client.status().get("executions", [])
    row = next((item for item in rows if item["id"] == execution_id), None)
    if row is None:
        raise RuntimeError("execution row disappeared from the task registry")
    return row


def _record_path(client, execution_id: str) -> Path:
    return Path(client.pool.state_dir) / "runs" / (execution_id + ".json")


def _write_evidence(args, client, *, label: str, execution_id: str, reply: dict, proof: dict,
                    cache: dict, source_repo: Path | None) -> dict:
    if proof.get("run_id") != execution_id:
        raise RuntimeError("proof run_id does not match the accepted execution")
    if source_repo is not None:
        if proof.get("patch_digest") != patch_digest(source_repo):
            raise RuntimeError("proof patch_digest does not match the deployed source")
    row = _execution_row(client, execution_id)
    snapshot = (row.get("spec") or {}).get("source_snapshot") or {}
    source = next((item for item in snapshot.get("records", [])
                   if item.get("relpath") == "vllm-ascend"), None)
    if source is None or not source.get("source_head") or not source.get("commit"):
        raise RuntimeError("accepted source snapshot has no vllm-ascend Git identity")
    if proof.get("source_commit") != source["commit"]:
        raise RuntimeError("proof source_commit does not match the accepted snapshot")
    if source_repo is not None and source["source_head"] != _git(source_repo, "rev-parse", "HEAD"):
        raise RuntimeError("accepted source_head does not match the deployed source")
    receipt = normalize_receipt(cache["receipt"])
    if not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("cache_key") or "")):
        raise RuntimeError("native cache receipt has no content key")
    state = reply.get("state")
    terminal_status = {"succeeded": "passed", "failed": "failed", "cancelled": "cancelled",
                       "inconclusive": "inconclusive", "timeout": "inconclusive"}.get(state)
    if terminal_status is None:
        raise RuntimeError("execution has no terminal status")
    created_at, started_at, ended_at = execution_times(row, reply, state)
    manifest, proof_obj, bench = project_manifest(
        run_id=execution_id,
        code={"source_head": source["source_head"], "snapshot_commit": source["commit"],
              "dirty": bool(source.get("changed_paths"))},
        proof=proof, created_at=created_at, started_at=started_at, ended_at=ended_at,
        terminal_status=terminal_status, topology={"host": args.host, "npu_count": 0},
        receipt=receipt, profile_summary_ref=proof.get("ready_profile") or None,
    )
    out = Path(args.output) / label
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out / "run-manifest.json", manifest)
    (out / "proof.json").write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "execution.json").write_text(json.dumps(
        {"row": row, "reply": reply, "cache": cache, "record_path": str(_record_path(client, execution_id))},
        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "bench-execution-evidence.json").write_text(
        json.dumps(bench, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"label": label, "run_id": execution_id, "cache": cache, "proof": proof,
            "context_file": client.context["context_file"],
            "manifest": str(out / "run-manifest.json"), "source_head": source["source_head"],
            "snapshot_commit": source["commit"], "bench_evidence": bench,
            "runtime_id": ((reply.get("roles") or [{}])[0]).get("runtime_id")
                          or ((row.get("roles") or [{}])[0].get("runtime_id")),
            "slot_id": ((reply.get("roles") or [{}])[0]).get("slot_id"),
            "slot_cleanup": ((reply.get("roles") or [{}])[0]).get("slot_cleanup")}


def _assert_slot_cleanup(endpoint, args, row: dict, reply: dict) -> dict:
    roles = reply.get("roles") or []
    slot_id = roles[0].get("slot_id") if roles else None
    cleanup = roles[0].get("slot_cleanup") if roles else None
    if not slot_id or (cleanup or {}).get("status") != "completed":
        raise RuntimeError("managed slot cleanup is not completed for " + row["id"])
    remote = _remote(endpoint, "test ! -e " + shlex.quote("/tmp/mindie-slot-" + slot_id))
    if remote.strip():
        raise RuntimeError("managed slot directory survived cleanup")
    compile_slots = []
    for records in (row.get("preparation_jobs") or {}).values():
        for job in records.values():
            lease = job.get("compile_lease")
            if not lease:
                continue
            if lease.get("state") not in TERMINAL_LEASE:
                raise RuntimeError("compile lease did not reach a terminal release state")
            if (job.get("slot_cleanup") or {}).get("state") != "completed":
                raise RuntimeError("compile slot cleanup is not completed")
            compile_slots.append({"task_id": lease.get("task_id"), "state": lease.get("state")})
    return {"slot_id": slot_id, "cleanup": cleanup, "directory_absent": True,
            "compile_slots": compile_slots}


def _compile_slots(args) -> tuple[dict, list[dict]]:
    host = {"host": args.host, "port": 22, "user": "root"}
    status = HostQueue().request(host, {"action": "status", "no_probe": True})
    if status.get("status") != "ok":
        raise RuntimeError("host queue status is unavailable")
    return status, [task for task in status.get("tasks") or []
                    if task.get("requested_count") == 0 and task.get("state") in
                    {"granted", "starting", "active", "orphaned_busy"}]


@contextlib.contextmanager
def build_environment(*, flags: str | None, soc: str | None, cxx: str):
    with _ENV_LOCK:
        previous = {key: os.environ.get(key) for key in ("CXX", "CXXFLAGS", "SOC_VERSION")}
        try:
            os.environ["CXX"] = cxx
            if flags is not None:
                os.environ["CXXFLAGS"] = flags
            if soc is not None:
                os.environ["SOC_VERSION"] = soc
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _client_for_execution(args, service, execution_id: str) -> TaskClient:
    store = AgentSessions(Path(args.agent_sessions_dir).expanduser().resolve())
    with store.transaction() as db:
        row = store.get(db, "execution", execution_id)
        attachments = [item for item in store.rows(db, "attachment")
                       if item.get("session_id") == row.get("session_id")]
    if not attachments:
        raise RuntimeError("accepted execution has no task attachment context")
    context = store.context(attachments[-1]["id"])
    return TaskClient(context["context_file"], user=args.user, service=service)


def _preflight_donor(args, service) -> list[dict]:
    """Require an existing registered runtime; never provision a container/image."""
    state_dir = Path(args.agent_sessions_dir).expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    store = AgentSessions(state_dir)
    context = store.attach("codex", "source-native-preflight-" + uuid.uuid4().hex, str(args.output))
    client = TaskClient(context["context_file"], user=args.user, service=service)
    matches = []
    for row in client.pool.catalog():
        if row.get("user") != args.user or row.get("container_name") != args.container:
            continue
        host = row.get("host") or (row.get("endpoint") or {}).get("host")
        if host != args.host or row.get("state") not in {"ready", "bound"}:
            continue
        container_id = row.get("container_id") or (row.get("attestation") or {}).get("container_id")
        if args.container_id and container_id != args.container_id:
            continue
        matches.append(row)
    if not matches:
        raise RuntimeError(
            "no existing prepared runtime for this user/host/container in the isolated "
            "agent-sessions state; register/point at the existing container state. "
            "This driver never provisions a mindie-user container or image.")
    client.finish(force=True)
    return matches


def _resume(args, pending_path: Path, service) -> int:
    data = json.loads(pending_path.read_text())
    use_local_state(data.get("agent_sessions_dir") or args.agent_sessions_dir)
    entries = list(data.get("runs") or [])
    pending, results = [], []
    for entry in entries:
        client = TaskClient(entry["context_file"], user=args.user, service=service)
        try:
            reply = _wait(args, client, entry["run_id"])
        except AcceptancePending:
            pending.append(entry)
            continue
        except Exception as exc:
            results.append({"label": entry.get("label"), "run_id": entry["run_id"],
                            "status": "failed", "error": str(exc)[:1000]})
            continue
        results.append({"label": entry.get("label"), "run_id": entry["run_id"],
                        "status": reply.get("state"), "proof": _proof(reply.get("stdout", ""))})
    report = {"schema": "mindie-source-native-resume/1",
              "status": "pending" if pending else "observed",
              "pending": pending, "results": results}
    (args.output / "resume.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                              encoding="utf-8")
    if pending:
        (args.output / "pending.json").write_text(json.dumps(
            {"schema": "mindie-source-native-pending/1", "status": "pending",
             "agent_sessions_dir": data.get("agent_sessions_dir"), "runs": pending},
            indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "pending", "run_ids": [entry["run_id"] for entry in pending],
                          "pending": str(args.output / "pending.json")}, indent=2))
        return 2
    failed = [row for row in results if row["status"] != "succeeded"]
    print(json.dumps({"status": "failed" if failed else "observed",
                      "run_ids": [row["run_id"] for row in results],
                      "resume": str(args.output / "resume.json")}, indent=2))
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", nargs="?", const="", help="reuse pending run ids; path optional")
    parser.add_argument("--host")
    parser.add_argument("--container")
    parser.add_argument("--container-id", help="expected existing container ID (optional strict check)")
    parser.add_argument("--user")
    parser.add_argument("--soc", default=os.environ.get("SOC_VERSION"))
    parser.add_argument("--cxx", default=os.environ.get("CXX") or "g++")
    parser.add_argument("--agent-sessions-dir", default="/tmp/mindie-npu-test-wave4/state/agent-sessions")
    parser.add_argument("--output", type=Path,
                        default=Path("/tmp/mindie-npu-test-wave4") / ("source-native-" + uuid.uuid4().hex[:12]))
    parser.add_argument("--wait-timeout", type=int, default=600)
    parser.add_argument("--business-timeout", type=int, default=300)
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--keep-sandbox", action="store_true")
    parser.add_argument("--real-vllm", help="actual existing vllm Git checkout (read-only)")
    parser.add_argument("--real-vllm-ascend", help="actual existing vllm-ascend Git checkout (read-only)")
    parser.add_argument("--source-archive", help="explicit source archive to deploy read-only")
    parser.add_argument("--source-archive-commit", help="verified source commit represented by the archive")
    parser.add_argument("--require-real-source", action="store_true")
    parser.add_argument("--reuse-cold", help="reuse an accepted, already-succeeded cold run id")
    parser.add_argument("--reuse-hit", help="reuse an accepted, already-succeeded cache-hit run id")
    args = parser.parse_args()
    if not 1 <= args.wait_timeout <= 600:
        parser.error("--wait-timeout must be 1..600; TaskClient wait is bounded at 600")
    if args.business_timeout < 30:
        parser.error("--business-timeout must be at least 30")
    if not args.dry_run and (not args.host or not args.container or not args.user or not args.soc):
        parser.error("live run requires --host, --container, --user and --soc (or SOC_VERSION)")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    use_local_state(args.agent_sessions_dir)
    if args.resume is not None:
        if not args.user:
            parser.error("--resume requires --user")
        pending_path = Path(args.resume) if args.resume else args.output / "pending.json"
        return _resume(args, pending_path, _local_service(args))
    sandbox = Path("/tmp/mindie-npu-test-wave4-native-" + uuid.uuid4().hex)
    sandbox.mkdir(parents=True, exist_ok=True)
    sources = write_cpp_probe_repos(sandbox / "base", offset=1)
    if not args.dry_run:
        set_vllm_probe_marker(sources["vllm"])
        set_source_offset(sources["vllm-ascend"], 4)
        set_scm_tag(sources["vllm-ascend"], "v0.0.2")
    if args.dry_run:
        check = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=sources["vllm-ascend"],
                               capture_output=True, text=True, timeout=300)
        if check.returncode:
            raise RuntimeError("dry-run C++ compile failed:\n" + check.stdout[-2000:] + check.stderr[-2000:])
        extension = next((sources["vllm-ascend"] / "vllm_ascend").glob("vllm_ascend_C*.so"), None)
        if extension is None or not extension.is_file():
            raise RuntimeError("dry-run did not produce a real extension")
        local_key = local_candidate_build_key(sources["vllm-ascend"])
        env = {**os.environ, "MINDIE_PYTHON": sys.executable, "MINDIE_RUN_ID": "debug-local-dry-run",
               "MINDIE_ACCEPTANCE_INPUT": "local-cpp-proof", "MINDIE_ACCEPTANCE_EXPECTED": "49",
               "MINDIE_ACCEPTANCE_BUILD_KEY": local_key,
               "PYTHONPATH": os.pathsep.join([str(sources["vllm"]), str(sources["vllm-ascend"])])}
        executed = subprocess.run(["bash", "-c", proof_command(49)], cwd=sources["vllm-ascend"].parent,
                                  env=env, capture_output=True, text=True, timeout=60)
        if executed.returncode:
            raise RuntimeError("dry-run proof execution failed:\n" + executed.stdout[-2000:] + executed.stderr[-2000:])
        proof = _proof(executed.stdout)
        dry = args.output / "dry-run"
        dry.mkdir(parents=True, exist_ok=True)
        shutil.copy2(extension, dry / extension.name)
        (dry / "proof.json").write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        head = _git(sources["vllm-ascend"], "rev-parse", "HEAD")
        started = epoch_iso(proof["measurement_window"]["start_ts"])
        ended = epoch_iso(proof["measurement_window"]["end_ts"])
        manifest, _proof_obj, bench = project_manifest(
            run_id=proof["run_id"],
            code={"source_head": head, "snapshot_commit": head, "dirty": False},
            proof=proof, created_at=started, started_at=started, ended_at=ended,
            terminal_status="passed", topology={"local": True, "npu_count": 0},
            profile_summary_ref=str(dry / "proof.json"),
        )
        write_manifest(dry / "run-manifest.json", manifest)
        (dry / "bench-execution-evidence.json").write_text(
            json.dumps(bench, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = {"status": "dry-run-local-cpp-proof-executed",
                  "scope": "local C++ shared-library proof only; no TaskClient, source deployment, cache, AscendC, NPU, or performance claim",
                  "extension": str(dry / extension.name), "proof": str(dry / "proof.json"),
                  "run_manifest": str(dry / "run-manifest.json"),
                  "bench_evidence": str(dry / "bench-execution-evidence.json"),
                  "proof_run_id": proof["run_id"]}
        (dry / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not args.keep_sandbox:
            shutil.rmtree(sandbox, ignore_errors=True)
        print(json.dumps(result, indent=2))
        return 0
    service = _local_service(args)
    donor = _preflight_donor(args, service)
    from remote_dev.core.container_endpoint import pin_container_endpoint
    from remote_dev.core.endpoint import Endpoint
    host = {"host": args.host, "port": 22, "user": "root"}
    pinned = pin_container_endpoint(Endpoint(**host, container=args.container, root="/", cwd="/",
                                             runtime_env=False), timeout_ms=15000)
    endpoint = {**host, "container": pinned.container, "root": "/", "cwd": "/"}
    before = _cache_names(endpoint)
    real = _real_sources(args, sandbox)
    if args.require_real_source and real is None:
        raise RuntimeError("--require-real-source was set but no actual Git source pair/archive was available")
    base_nonce = int(uuid.uuid4().hex[:8], 16)
    base_flags = f"-DMINDIE_ACCEPTANCE_NONCE={base_nonce} -O2"
    cases: list[dict] = []
    test_keys: set[str] = set()
    test_bundles: set[str] = set()
    clients: list[TaskClient] = []
    cleanup_result = None
    all_released = False
    pending = False
    source_case = None
    try:
        if args.reuse_cold:
            cold_id = args.reuse_cold
            cold_client = _client_for_execution(args, service, cold_id)
            clients.append(cold_client)
            cold_reply = _wait(args, cold_client, cold_id)
            cold_proof = _proof(cold_reply.get("stdout", ""))
            base_nonce = int(cold_proof["probe_value"]) - 52
            base_flags = f"-DMINDIE_ACCEPTANCE_NONCE={base_nonce} -O2"
            cold = _write_evidence(args, cold_client, label="cold", execution_id=cold_id,
                                   reply=cold_reply, proof=cold_proof,
                                   cache=cache_facts(cold_client.observe(cold_id, action="evidence", section="build")),
                                   source_repo=None)
            cold_key = cold["cache"]["receipt"]["cache_key"]
            cold_store = cold["cache"]["store"] or {}
            if cold["cache"]["receipt"].get("hit") is not False or cold_store.get("status") != "stored":
                raise RuntimeError("reused cold slot is not a real cache miss + store")
            if not isinstance(cold_key, str) or not re.fullmatch(r"[0-9a-f]{64}", cold_key):
                raise RuntimeError("store receipt has no valid content cache key")
            test_keys.add(cold_key)
            if cold_store.get("bundle"):
                test_bundles.add(cold_store["bundle"])
            cases.append(cold)
        else:
            cold_client = _slot(args, "cold", service)
            clients.append(cold_client)
            cold_id = _admit(args, cold_client, sources, label="cold", nonce=base_nonce,
                             flags=base_flags, expected=52 + base_nonce)
            cold_reply = _wait(args, cold_client, cold_id)
            cold_proof = _proof(cold_reply.get("stdout", ""))
            cold = _write_evidence(args, cold_client, label="cold", execution_id=cold_id,
                                   reply=cold_reply, proof=cold_proof,
                                   cache=cache_facts(cold_client.observe(cold_id, action="evidence", section="build")),
                                   source_repo=sources["vllm-ascend"])
            cold_key = cold["cache"]["receipt"]["cache_key"]
            cold_store = cold["cache"]["store"] or {}
            if cold["cache"]["receipt"].get("hit") is not False or cold_store.get("status") != "stored":
                raise RuntimeError("first slot was not a real cache miss + store")
            if not isinstance(cold_key, str) or not re.fullmatch(r"[0-9a-f]{64}", cold_key):
                raise RuntimeError("store receipt has no valid content cache key")
            test_keys.add(cold_key)
            if cold_store.get("bundle"):
                test_bundles.add(cold_store["bundle"])
            cases.append(cold)
        if cold.get("runtime_id"):
            cold_client.finish(force=True)

        if args.reuse_hit:
            hit_id = args.reuse_hit
            hit_client = _client_for_execution(args, service, hit_id)
            clients.append(hit_client)
            hit_reply = _wait(args, hit_client, hit_id)
            hit = _write_evidence(args, hit_client, label="hit", execution_id=hit_id,
                                  reply=hit_reply, proof=_proof(hit_reply.get("stdout", "")),
                                  cache=cache_facts(hit_client.observe(hit_id, action="evidence", section="build")),
                                  source_repo=None)
            receipt = hit["cache"]["receipt"]
            if receipt.get("hit") is not True or int(receipt.get("linked_files") or 0) <= 0:
                raise RuntimeError("reused hit slot is not a content-pool hit with linked files")
            if hit["proof"].get("cache_link", {}).get("same_inode") is not True:
                raise RuntimeError("reused hit slot loaded a file that was not the linked pool artifact")
            cases.append(hit)
        else:
            hit_client = _slot(args, "hit", service)
            clients.append(hit_client)
            hit_id = _admit(args, hit_client, sources, label="hit", nonce=base_nonce,
                            flags=base_flags, expected=52 + base_nonce, cache_key=cold_key,
                            cache_relative=cold_proof["loaded_module_relative"])
            hit_reply = _wait(args, hit_client, hit_id)
            hit = _write_evidence(args, hit_client, label="hit", execution_id=hit_id,
                                   reply=hit_reply, proof=_proof(hit_reply.get("stdout", "")),
                                   cache=cache_facts(hit_client.observe(hit_id, action="evidence", section="build")),
                                   source_repo=sources["vllm-ascend"])
            receipt = hit["cache"]["receipt"]
            if receipt.get("hit") is not True or int(receipt.get("linked_files") or 0) <= 0:
                raise RuntimeError("second slot did not hit the content pool with linked files")
            if hit["proof"].get("cache_link", {}).get("same_inode") is not True:
                raise RuntimeError("hit slot loaded a file that was not the linked pool artifact")
            cases.append(hit)
        if hit.get("runtime_id"):
            hit_client.finish(force=True)
        changed = write_cpp_probe_repos(sandbox / "changed", offset=1)
        set_vllm_probe_marker(changed["vllm"])
        set_source_offset(changed["vllm-ascend"], 5)
        set_scm_tag(changed["vllm-ascend"], "v0.0.2")
        changed_client = _slot(args, "changed-source", service)
        clients.append(changed_client)
        changed_id = _admit(args, changed_client, changed, label="changed-source",
                            nonce=base_nonce, flags=base_flags, expected=53 + base_nonce)
        changed_reply = _wait(args, changed_client, changed_id)
        changed_case = _write_evidence(args, changed_client, label="changed-source",
                                       execution_id=changed_id, reply=changed_reply,
                                       proof=_proof(changed_reply.get("stdout", "")),
                                       cache=cache_facts(changed_client.observe(changed_id, action="evidence", section="build")),
                                       source_repo=changed["vllm-ascend"])
        if changed_case["cache"]["receipt"].get("hit") is not False:
            raise RuntimeError("changed source incorrectly reused a native cache entry")
        if changed_case["proof"]["patch_digest"] == cold_proof["patch_digest"]:
            raise RuntimeError("changed source did not change the patch digest")
        test_keys.add(changed_case["cache"]["receipt"]["cache_key"])
        if (changed_case["cache"]["store"] or {}).get("bundle"):
            test_bundles.add(changed_case["cache"]["store"]["bundle"])
        cases.append(changed_case)

        compiler_nonce = int(uuid.uuid4().hex[:8], 16)
        compiler_flags = f"-DMINDIE_ACCEPTANCE_NONCE={compiler_nonce} -O2"
        compiler_client = _slot(args, "changed-compiler", service)
        clients.append(compiler_client)
        compiler_id = _admit(args, compiler_client, sources, label="changed-compiler",
                             nonce=compiler_nonce, flags=compiler_flags,
                             expected=52 + compiler_nonce)
        compiler_reply = _wait(args, compiler_client, compiler_id)
        compiler_case = _write_evidence(args, compiler_client, label="changed-compiler",
                                        execution_id=compiler_id, reply=compiler_reply,
                                        proof=_proof(compiler_reply.get("stdout", "")),
                                        cache=cache_facts(compiler_client.observe(compiler_id, action="evidence", section="build")),
                                        source_repo=sources["vllm-ascend"])
        if compiler_case["cache"]["receipt"].get("hit") is not False:
            raise RuntimeError("changed compiler condition incorrectly reused a native cache entry")
        if compiler_case["cache"]["receipt"]["cache_key"] == cold_key:
            raise RuntimeError("compiler condition did not change the content key")
        if compiler_case["proof"]["patch_digest"] != cold_proof["patch_digest"]:
            raise RuntimeError("compiler-only change was mislabeled as a source patch")
        test_keys.add(compiler_case["cache"]["receipt"]["cache_key"])
        if (compiler_case["cache"]["store"] or {}).get("bundle"):
            test_bundles.add(compiler_case["cache"]["store"]["bundle"])
        cases.append(compiler_case)

        # Two independent zero-NPU slots compiled concurrently. HostQueue remains
        # the authority; max_compile_slots is read from its own status.
        status_before, active_before = _compile_slots(args)
        max_slots = int(status_before["max_compile_slots"])
        concurrent_nonce = [int(uuid.uuid4().hex[:8], 16) for _ in range(2)]
        concurrent = []
        for index, nonce in enumerate(concurrent_nonce):
            client = _slot(args, f"parallel-{index}", service)
            clients.append(client)
            execution_id = _admit(args, client, sources, label=f"parallel-{index}", nonce=nonce,
                                  flags=f"-DMINDIE_ACCEPTANCE_NONCE={nonce} -O2",
                                  expected=49 + nonce)
            concurrent.append((client, execution_id, nonce))
        observed_max = len(active_before)
        with ThreadPoolExecutor(max_workers=2) as workers:
            futures = [workers.submit(_wait, args, client, execution_id)
                       for client, execution_id, _ in concurrent]
            time.sleep(0.1)
            while any(not future.done() for future in futures):
                _status, active = _compile_slots(args)
                observed_max = max(observed_max, len(active))
                time.sleep(0.5)
            replies = [future.result() for future in futures]
        for index, ((client, execution_id, nonce), reply) in enumerate(zip(concurrent, replies)):
            case = _write_evidence(args, client, label=f"parallel-{index}", execution_id=execution_id,
                                   reply=reply, proof=_proof(reply.get("stdout", "")),
                                   cache=cache_facts(client.observe(execution_id, action="evidence", section="build")),
                                   source_repo=sources["vllm-ascend"])
            if case["cache"]["receipt"].get("hit") is not False:
                raise RuntimeError("parallel slot was not an independent cold build")
            test_keys.add(case["cache"]["receipt"]["cache_key"])
            if (case["cache"]["store"] or {}).get("bundle"):
                test_bundles.add(case["cache"]["store"]["bundle"])
            cases.append(case)
        if observed_max > max_slots:
            raise RuntimeError(f"observed {observed_max} active compile slots > max_compile_slots={max_slots}")

        if real is not None:
            source_client = _slot(args, "real-source", service)
            clients.append(source_client)
            source_id = _admit(args, source_client, real["sources"], label="real-source",
                               nonce=base_nonce, flags=base_flags, expected=0,
                               command_override=_source_command(real["archive_commit"]))
            source_reply = _wait(args, source_client, source_id)
            source_proof = _source_proof(source_reply.get("stdout", ""))
            if source_proof.get("file_sha256") != source_file_hashes(real["sources"]):
                raise RuntimeError("real source deployment file hashes do not match origin")
            snapshot = (_execution_row(source_client, source_id).get("spec") or {}).get("source_snapshot") or {}
            records = {row.get("relpath"): row for row in snapshot.get("records") or []}
            for name, key in (("vllm-source", "vllm_head"), ("vllm-ascend-source", "vllm_ascend_head")):
                record = records.get(name)
                if record is None:
                    raise RuntimeError("real source snapshot is missing " + name)
                if source_proof.get(key) != record.get("commit"):
                    raise RuntimeError(name + " materialized HEAD mismatch")
                if record.get("source_head") != real[key]:
                    raise RuntimeError(name + " source_head mismatch")
            if source_proof.get("archive_commit") != real["archive_commit"]:
                raise RuntimeError("real source archive provenance mismatch")
            source_case = {"label": "real-source", "run_id": source_id,
                           "context_file": source_client.context["context_file"],
                           "mode": "source-deployment-only", "proof": source_proof,
                           "snapshot": snapshot}
            out = args.output / "real-source"
            out.mkdir(parents=True, exist_ok=True)
            (out / "source-deployment.json").write_text(
                json.dumps(source_case, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            cases.append(source_case)

        slot_cleanup = [_assert_slot_cleanup(endpoint, args, _execution_row(client, case["run_id"]),
                                             _wait(args, client, case["run_id"]))
                        for client, case in ((c, x) for c, x in zip(clients, cases))]
        all_released = True
        if not args.keep_cache:
            cleanup_result = _cleanup_cache(endpoint, test_keys, test_bundles)
        after = _cache_names(endpoint)
        lost = before - after - test_keys - test_bundles
        if lost:
            raise RuntimeError("cleanup removed an unrelated cache entry: " + ", ".join(sorted(lost)))
        if not args.keep_cache and (test_keys & after or test_bundles & after):
            raise RuntimeError("test-owned cache entries survived lifecycle cleanup")
        evidence = {
            "schema": "mindie-source-native-acceptance/1",
            "status": "passed",
            "scope": "minimal production-shaped C++ cache contract through TaskClient; real-Git source deployment is a separate case; CPU-only, no AscendC/NPU/performance claim",
            "host": args.host, "container": args.container, "user": args.user,
            "donor_runtime_ids": [row.get("id") or row.get("runtime_id") for row in donor],
            "max_compile_slots": max_slots, "observed_max_active_compile_slots": observed_max,
            "pre_existing_cache_entries": sorted(before),
            "test_cache_keys": sorted(test_keys), "test_cache_bundles": sorted(test_bundles),
            "cleanup": cleanup_result, "slot_cleanup": slot_cleanup,
            "real_source_deployment": source_case,
            "cases": cases,
        }
        (args.output / "acceptance.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                                                      encoding="utf-8")
        print(json.dumps({"status": "passed", "run_ids": [case["run_id"] for case in cases],
                          "max_compile_slots": max_slots, "observed_max": observed_max,
                          "evidence": str(args.output / "acceptance.json")}, indent=2))
        return 0
    except AcceptancePending:
        pending = True
        runs = [{"label": case["label"], "run_id": case["run_id"],
                 "context_file": case.get("context_file")} for case in cases]
        if _PENDING is not None:
            client, execution_id = _PENDING
            if not any(entry["run_id"] == execution_id for entry in runs):
                runs.append({"label": "in-flight", "run_id": execution_id,
                             "context_file": client.context["context_file"]})
        (args.output / "pending.json").write_text(json.dumps(
            {"schema": "mindie-source-native-pending/1", "status": "pending",
             "agent_sessions_dir": str(Path(args.agent_sessions_dir).resolve()), "runs": runs},
            indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "pending", "run_ids": [entry["run_id"] for entry in runs],
                          "pending": str(args.output / "pending.json"),
                          "resume": "reinvoke with --resume " + str(args.output / "pending.json")},
                         indent=2))
        return 2
    except Exception as exc:
        (args.output / "failure.json").write_text(json.dumps(
            {"schema": "mindie-source-native-failure/1", "status": "failed",
             "error": f"{type(exc).__name__}: {exc}"[:2000], "runs": cases},
            indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    finally:
        if all_released and not pending:
            for client in clients:
                try:
                    client.finish(force=True)
                except Exception as exc:
                    print("tasks.finish failed: " + str(exc)[:500], file=sys.stderr)
        if not args.keep_cache and cleanup_result is None and all_released and not pending:
            try:
                if test_keys or test_bundles:
                    _cleanup_cache(endpoint, test_keys, test_bundles)
            except Exception as exc:
                print("cache cleanup failed: " + str(exc)[:500], file=sys.stderr)
        if not args.keep_sandbox and not pending:
            shutil.rmtree(sandbox, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
