"""Standalone remote manifest verification & donor qualification suffixes.

[Role]
Concatenated after `runtime_profile.py` and `build_inputs.py` by `backend.py`
to verify `.mindie-runtime/ready-profile.json` and pinned Git snapshots inside
remote containers without requiring `mindie_coordinator` to be installed remotely.

[Invariants]
- Pure Python stdlib; expects `Path`, `json`, `verify`, `verify_execution_view`,
  `digest`, and `_build_namespace` in module globals from concatenated preamble.
- All `git` subprocess checks enforce `timeout=30` and `stdin=subprocess.DEVNULL`.

[Timeouts & Locks]
- Bounded `30s` per `git` check inside caller's `45s` `RemoteBackend.bash` timeout.

[Side-Effect Policy]
- Strictly read-only inspection of `ready-profile.json` and Git `HEAD` / worktree status.
"""
from __future__ import annotations

READY_PROBE_SUFFIX = """
import subprocess
import sys

args = json.loads(sys.argv[1])
root = Path(args["root"])
manifest = json.loads((root / ".mindie-runtime/ready-profile.json").read_text(encoding="utf-8"))
if args.get("prepared_view") and "expected_manifest_digest" in args and digest(manifest) != args["expected_manifest_digest"]:
    raise ValueError("runtime changed before launch")
if args.get("prepared_view"):
    verify_execution_view(root, manifest)
else:
    verify(root, manifest)
if not args.get("prepared_view") and manifest["profile"].get("kind") != "command" and _build_namespace["runtime_build_inputs"](root, manifest["profile"], manifest["profile_key"]) != manifest["build_inputs"]:
    raise ValueError("cache miss: installed native artifacts do not match current source inputs")
for name, expected in args["snapshots"].items():
    repo = root / name
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, encoding="utf-8", timeout=30, stdin=subprocess.DEVNULL).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--name-only", "--ignore-submodules=dirty"], text=True, encoding="utf-8", timeout=30, stdin=subprocess.DEVNULL)
    if head != expected or dirty.strip():
        raise ValueError("runtime source differs from pinned snapshot: " + name)
if "expected_manifest_digest" in args:
    actual_digest = digest(manifest)
    if actual_digest != args["expected_manifest_digest"]:
        raise ValueError("runtime changed before launch")
    print(json.dumps({"manifest_digest": actual_digest}))
else:
    print(json.dumps(manifest))
"""

QUALIFY_DONOR_SUFFIX = """
import subprocess
import sys

args = json.loads(sys.argv[1])
root = Path(args["root"])
manifest = json.loads((root / ".mindie-runtime/ready-profile.json").read_text(encoding="utf-8"))
verify(root, manifest)
if _build_namespace["runtime_build_inputs"](root, manifest["profile"], manifest["profile_key"]) != manifest["build_inputs"]:
    print(json.dumps({"qualified": False, "reason": "installed native artifacts do not match current source inputs"}))
    raise SystemExit(0)
build_info = "vllm-ascend/vllm_ascend/_build_info.py"
if (root / build_info).is_file() and build_info not in manifest["files"]:
    print(json.dumps({"qualified": False, "reason": "generated build metadata has no verified donor hash"}))
    raise SystemExit(0)
for row in args["records"]:
    repo = root / row["relpath"]
    tree = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"], text=True, timeout=30, stdin=subprocess.DEVNULL).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--name-only", "--ignore-submodules=dirty"], text=True, timeout=30, stdin=subprocess.DEVNULL).strip()
    if tree != row["tree"] or dirty:
        print(json.dumps({"qualified": False, "reason": "source tree differs", "source": row["relpath"]}))
        raise SystemExit(0)
print(json.dumps({"qualified": True, "build_key": manifest["build_key"], **({"manifest": manifest} if args.get("include_manifest") else {})}))
"""
