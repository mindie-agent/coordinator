"""Live proof/cache evidence helpers and Run Manifest projection.

[Role]
Normalizes TaskClient evidence/cache payloads, projects an observed proof into a
terminal Run Manifest, and gathers read-only live source-deployment and cache
evidence without inventing timestamps or cache receipts.

[Invariants]
- `TaskClient.observe(action="evidence")` wraps role data under `evidence`.
- A cache receipt is accepted from a normalized result or a store/miss result;
  the keyword-only manifest builder is never called with a result dict.
- Manifest status uses the actual terminal execution state and actual execution
  timestamps, never a caller-supplied success label.
- The bench projection is the same helper used by live and local proof paths.
- Live helpers keep their original signatures, defaults, and read-only behavior.

[Timeouts & Locks]
- Remote evidence probes use a bounded 20s timeout; source clone/archive work
  uses 120s/300s subprocess timeouts. No repository or coordinator locks are held.

[Side-Effect Policy]
- Manifest/cache construction is pure; callers write files.
- Source helpers read external repositories and write only inside the caller's
  private sandbox. Cache cleanup removes only explicitly recorded test keys,
  bundles, and matching indexes; all pre-existing entries survive.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIVE_DIR = Path(__file__).resolve().parent
for _directory in (_REPO_ROOT, _LIVE_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from mindie_coordinator.native_cache_pool import native_cache_receipt as cache_result_receipt  # noqa: E402
from mindie_coordinator.run_manifest import (  # noqa: E402
    bench_execution_evidence,
    execution_proof,
    project_run_manifest,
)
from source_native_cpp_probe import _commit, _git  # noqa: E402

TERMINAL = {
    "succeeded": "passed",
    "failed": "failed",
    "cancelled": "cancelled",
    "inconclusive": "inconclusive",
    "timeout": "inconclusive",
}


def epoch_iso(value: Any) -> str:
    timestamp = float(value)
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


def execution_times(row: Mapping[str, Any], reply: Mapping[str, Any], state: str) -> tuple[str, str, str]:
    """Actual created/running/terminal timestamps from the persisted row."""
    created_ts = row.get("created_at") or reply.get("observed_at")
    observed = row.get("phase_observed_at") or {}
    running_ts = observed.get("running") or created_ts
    terminal_ts = observed.get(state) or observed.get(row.get("phase")) or reply.get("observed_at") or running_ts
    created = epoch_iso(created_ts)
    started = epoch_iso(max(float(created_ts), float(running_ts)))
    ended = epoch_iso(max(float(running_ts), float(terminal_ts)))
    return created, started, ended


def normalize_receipt(value: Any) -> dict:
    if not isinstance(value, Mapping):
        raise RuntimeError("native cache receipt is not an object")
    if "hit" in value:
        receipt = {"hit": bool(value.get("hit")), "cache_key": str(value.get("cache_key") or ""),
                   "linked_files": int(value.get("linked_files") or 0)}
    else:
        receipt = cache_result_receipt(dict(value))
    return receipt


def cache_facts(payload: Any, *, required: bool = True) -> dict:
    """Read the real `evidence` envelope, not a mock top-level shape."""
    if not isinstance(payload, Mapping):
        raise RuntimeError("TaskClient evidence is not an object")
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), Mapping) else payload
    receipt = None
    store = None
    for role in evidence.get("roles") or []:
        receipt = receipt or role.get("native_cache_receipt")
        for log in role.get("logs") or []:
            if isinstance(log.get("native_cache"), dict):
                store = log["native_cache"]
    if receipt is None and store is not None:
        receipt = cache_result_receipt(store)
    if receipt is None and not required:
        return {"receipt": None, "store": store}
    if not isinstance(receipt, Mapping):
        raise RuntimeError("no native cache receipt was recorded")
    return {"receipt": normalize_receipt(receipt), "store": store}


def project_manifest(*, run_id: str, code: Mapping[str, Any], proof: Mapping[str, Any],
                     created_at: str, started_at: str, ended_at: str, terminal_status: str,
                     topology: Mapping[str, Any], receipt: Mapping[str, Any] | None = None,
                     profile_summary_ref: str | None = None) -> tuple[dict, dict, dict]:
    if terminal_status not in {"passed", "failed", "inconclusive", "cancelled"}:
        raise RuntimeError("unsupported terminal status")
    proof_obj = execution_proof(
        run_id=run_id, patch_digest=proof["patch_digest"],
        loaded_binary_sha256=proof["loaded_binary_sha256"],
        loaded_module_path=proof["loaded_module_path"], calls=proof["calls"],
        measurement_window=proof["measurement_window"],
        candidate_build_key=proof["candidate_build_key"],
        evidence_ref=proof["evidence_ref"],
    )
    manifest = project_run_manifest(
        run_id=run_id, code=code, created_at=created_at, started_at=started_at,
        ended_at=ended_at, terminal_status=terminal_status,
        command=[proof["calls"][0]["call_path"]], topology=dict(topology),
        native_cache_receipt=receipt, active_execution_proof=proof_obj,
        profile_summary_ref=profile_summary_ref,
    )
    bench = bench_execution_evidence(proof_obj, manifest_run_id=run_id)
    if set(bench) != {"run_id", "patch_digest", "loaded_binary_sha256", "loaded_module_path",
                      "calls", "evidence_ref"}:
        raise RuntimeError("bench adapter shape changed")
    return manifest, proof_obj, bench


def _remote(endpoint, command: str, *, timeout_ms: int = 20000) -> str:
    from remote_dev.core.endpoint import Endpoint
    from remote_dev.core.ssh_transport import run_script
    target = (Endpoint(**endpoint, runtime_env=False) if isinstance(endpoint, dict) else endpoint)
    result = run_script(target, command, timeout_ms=timeout_ms)
    if result.returncode:
        raise RuntimeError("remote command failed: " + result.stderr[-1000:])
    return result.stdout


def _cache_names(endpoint) -> set[str]:
    code = """import json, pathlib
root = pathlib.Path('/tmp/mindie-native-cache')
names = []
if root.is_dir():
    names = [p.name for p in root.iterdir()]
    bundles = root / 'bundles'
    if bundles.is_dir():
        names += ['bundles/' + p.name for p in bundles.iterdir()]
print(json.dumps(sorted(names)))
"""
    return set(json.loads(_remote(endpoint, "python3 -c " + shlex.quote(code))))


def _cleanup_cache(endpoint, keys: set[str], bundles: set[str]) -> dict:
    payload = json.dumps({"keys": sorted(keys), "bundles": sorted(bundles)})
    code = r'''import json, pathlib, re, shutil, sys
payload = json.loads(sys.argv[1])
root = pathlib.Path("/tmp/mindie-native-cache")
removed = {"keys": [], "bundles": [], "indexes": []}
if not root.is_dir():
    print(json.dumps(removed))
    raise SystemExit(0)
for key in payload["keys"]:
    if re.fullmatch(r"[0-9a-f]{64}", key) and (root / key).is_dir():
        shutil.rmtree(root / key)
        removed["keys"].append(key)
for bundle in payload["bundles"]:
    path = root / "bundles" / bundle
    if not re.fullmatch(r"[0-9a-f]{64}", bundle) or not path.is_dir():
        continue
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("build_key") == bundle:
        shutil.rmtree(path)
        removed["bundles"].append(bundle)
for path in list(root.glob("*.json")):
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        continue
    if data.get("bundle") in payload["bundles"]:
        path.unlink()
        removed["indexes"].append(path.name)
print(json.dumps(removed))
'''
    return json.loads(_remote(endpoint, "python3 - " + shlex.quote(payload) + " <<'PY'\n" + code + "\nPY"))


def _clone_readonly(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", "--shared", str(source), str(destination)],
                   check=True, capture_output=True, text=True, timeout=120)
    # Fixed source capture requires populated gitlinks. Populate only this
    # private clone; the external checkout remains read-only.
    subprocess.run(["git", "-C", str(destination), "submodule", "update", "--init", "--recursive"],
                   check=True, capture_output=True, text=True, timeout=300)
    return destination


def _archive_repo(archive: Path, destination: Path) -> tuple[Path, str]:
    import tarfile
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(destination)
    _git(destination, "init")
    _git(destination, "config", "user.name", "MindIE Acceptance")
    _git(destination, "config", "user.email", "acceptance@example.invalid")
    commit = _commit(destination, "frozen vllm-ascend source archive")
    _git(destination, "tag", "v0.0.1")
    return destination, commit


def _real_sources(args, sandbox: Path) -> dict | None:
    """Deploy actual existing Git sources read-only, or the frozen archive."""
    vllm_source = Path(args.real_vllm).expanduser().resolve() if args.real_vllm else None
    if vllm_source is None or not (vllm_source / ".git").exists():
        return None
    base = sandbox / "real-source"
    vllm = _clone_readonly(vllm_source, base / "vllm-source")
    if args.real_vllm_ascend:
        ascend_source = Path(args.real_vllm_ascend).expanduser().resolve()
        if not (ascend_source / ".git").exists():
            return None
        ascend = _clone_readonly(ascend_source, base / "vllm-ascend-source")
        archive_commit = None
    elif args.source_archive and Path(args.source_archive).is_file():
        ascend, created = _archive_repo(Path(args.source_archive), base / "vllm-ascend-source")
        archive_commit = args.source_archive_commit or None
    else:
        return None
    return {"sources": {"vllm-source": vllm, "vllm-ascend-source": ascend},
            "vllm_head": _git(vllm, "rev-parse", "HEAD"),
            "vllm_ascend_head": _git(ascend, "rev-parse", "HEAD"),
            "archive_commit": archive_commit}


def _source_command(archive_commit: str | None) -> str:
    code = f'''
import hashlib, json, os, pathlib, subprocess
root = pathlib.Path.cwd()
def head(name):
    return subprocess.check_output(["git", "-C", name, "rev-parse", "HEAD"], text=True).strip()
files = {{}}
for name in ("vllm-source", "vllm-ascend-source"):
    base = root / name
    for relative in ("README.md", "CMakeLists.txt", "csrc"):
        path = base / relative
        if path.is_file():
            files[name + "/" + relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif path.is_dir():
            for child in sorted(p for p in path.rglob("*") if p.is_file())[:64]:
                files[name + "/" + child.relative_to(base).as_posix()] = hashlib.sha256(child.read_bytes()).hexdigest()
proof = {{
    "run_id": os.environ["MINDIE_RUN_ID"],
    "mode": "source-deployment-only",
    "vllm_head": head("vllm-source"),
    "vllm_ascend_head": head("vllm-ascend-source"),
    "archive_commit": {archive_commit!r},
    "file_sha256": files,
}}
(root / "source-deployment-proof.json").write_text(json.dumps(proof, sort_keys=True) + "\\n")
print("MINDIE_SOURCE_DEPLOYMENT_PROOF " + json.dumps(proof, sort_keys=True))
'''
    return '"$MINDIE_PYTHON" - <<\'MINDIE_SOURCE_DEPLOYMENT\'\n' + code + '\nMINDIE_SOURCE_DEPLOYMENT\n'


def _source_proof(stdout: str) -> dict:
    marker = "MINDIE_SOURCE_DEPLOYMENT_PROOF "
    for line in reversed((stdout or "").splitlines()):
        if line.startswith(marker):
            value = json.loads(line[len(marker):])
            if not isinstance(value, dict):
                raise RuntimeError("source deployment proof is not an object")
            return value
    raise RuntimeError("source deployment proof marker is missing")
