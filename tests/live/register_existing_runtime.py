"""Register an existing prepared container root from its actual ready profile.

[Role]
Adds one already-running container/root to a dedicated local wave4 RuntimePool
state so the acceptance driver can use it as a donor. Never creates a container,
image, source checkout or global registry entry.

[Invariants]
- Container name must follow `mindie-<user>`; the user is derived, not invented.
- The root must already contain `.mindie-runtime/ready-profile.json`; the
  registration probe verifies and captures it from the container.
- `--container-id` is checked against the live container before registration.
- No ready profile is synthesized when the root has none.

[Timeouts & Locks]
- Pinned endpoint lookup 15s; remote profile read 20s; registration probe is
  owned by RuntimePool/RemoteBackend.

[Side-Effect Policy]
- Writes only the supplied local agent-sessions/coordinator state and output
  receipt. Existing container contents are read-only.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
import uuid
from pathlib import Path

# Direct script execution must use this checkout, not an older installed wheel.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mindie_coordinator.backend import RemoteBackend  # noqa: E402
from mindie_coordinator.ready_runtime import RuntimePool, safe_id  # noqa: E402
from remote_dev.core.container_endpoint import pin_container_endpoint  # noqa: E402
from remote_dev.core.endpoint import Endpoint  # noqa: E402
from remote_dev.core.ssh_transport import run_script  # noqa: E402


def _pin(host: str, container: str):
    return pin_container_endpoint(Endpoint(host=host, port=22, user="root", container=container,
                                           root="/", cwd="/", runtime_env=False), timeout_ms=15000)


def _run(endpoint, command: str, timeout_ms: int = 20000) -> str:
    result = run_script(endpoint, command, timeout_ms=timeout_ms)
    if result.returncode:
        raise RuntimeError("remote command failed: " + result.stderr[-1000:])
    return result.stdout


def discover_roots(endpoint) -> list[str]:
    output = _run(endpoint, "find /vllm-workspace/executions /tmp/mindie-npu-test-* "
                            "-maxdepth 8 -path '*/.mindie-runtime/ready-profile.json' "
                            "-type f -print 2>/dev/null || true")
    return sorted({str(Path(line).parent.parent) for line in output.splitlines() if line.strip()})


def read_profile(endpoint, root: str) -> dict:
    code = """import json, pathlib, sys
root = pathlib.Path(json.loads(sys.argv[1]))
manifest = json.loads((root / ".mindie-runtime/ready-profile.json").read_text())
print(json.dumps({"profile": manifest.get("profile") or {},
                  "execution_view": manifest.get("execution_view") or {},
                  "build_key": manifest.get("build_key"),
                  "profile_key": manifest.get("profile_key")}))
"""
    return json.loads(_run(endpoint, "python3 - " + json.dumps(root) + " <<'PY'\n" + code + "\nPY"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--container-id")
    parser.add_argument("--root", help="existing root containing .mindie-runtime/ready-profile.json")
    parser.add_argument("--discover", action="store_true", help="list actual ready-profile roots only")
    parser.add_argument("--agent-sessions-dir", default=str(Path.cwd() / ".mindie-003-wave4/state/agent-sessions"))
    parser.add_argument("--output", type=Path, default=Path(".mindie-003-wave4/existing-runtime-registration.json"))
    args = parser.parse_args()
    if not args.container.startswith("mindie-"):
        parser.error("existing container must follow mindie-<user>")
    user = safe_id(args.container[len("mindie-"):])
    endpoint = _pin(args.host, args.container)
    if args.container_id:
        actual = _run(Endpoint(host=args.host, port=22, user="root", root="/", cwd="/", runtime_env=False),
                      "docker inspect --format '{{.Id}}' " + shlex.quote(args.container)).strip()
        if actual != args.container_id:
            raise RuntimeError("existing container ID differs from --container-id")
    if args.discover or not args.root:
        roots = discover_roots(endpoint)
        print(json.dumps({"container": args.container, "container_id": args.container_id,
                          "ready_profile_roots": roots,
                          "msg": "pass --root with one of these actual roots"}, indent=2))
        return 0 if roots else 2
    root = str(Path(args.root))
    profile = read_profile(endpoint, root)
    if not profile.get("profile") or not profile.get("execution_view", {}).get("python"):
        raise RuntimeError("root has no actual ready profile with execution_view.python")
    python = profile["execution_view"]["python"]
    if not python.startswith("/") or ".." in python:
        raise RuntimeError("ready profile does not contain an absolute python path")
    state_dir = Path(args.agent_sessions_dir).expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    coordinator_dir = state_dir.parent / "coordinator"
    pool = RuntimePool(coordinator_dir, RemoteBackend())
    runtime_id = "donor-wave4-" + uuid.uuid4().hex[:12]
    spec = {
        "user": user,
        "python": python,
        "recipe": profile["profile"].get("recipe"),
        "endpoint": {"host": args.host, "port": int(endpoint.port), "user": "root",
                     "root": root, "cwd": root},
        "host_endpoint": {"host": args.host, "port": 22, "user": "root"},
        "container_name": args.container,
        "service_ports": [],
    }
    registered = pool.register(runtime_id, spec)
    attestation = registered.get("attestation") or {}
    receipt = {
        "status": "registered",
        "runtime_id": runtime_id,
        "user": user,
        "container": args.container,
        "container_id": attestation.get("container_id"),
        "root": root,
        "profile_key": attestation.get("profile_key"),
        "build_key": attestation.get("build_key"),
        "profile": attestation.get("profile"),
        "agent_sessions_dir": str(state_dir),
        "coordinator_state_dir": str(coordinator_dir),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "registered", "runtime_id": runtime_id, "user": user,
                      "root": root, "receipt": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
