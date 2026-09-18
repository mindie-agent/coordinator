#!/usr/bin/env python3
"""Attest/cache an already built runtime inside a prepared work root.

Use existing machine-management/parity installers BEFORE this command. No
package installation, container creation, card allocation or model loading.
The user container `mindie-<user>` is not created or deleted here.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from mindie_coordinator.build_inputs import runtime_build_inputs
from mindie_coordinator.git_sources import discover_repo_tree, iter_postorder
from mindie_coordinator.runtime_profile import capture, capture_launch_environment, file_digest, native_vendor_launch_environment, profile_key, publish, restore, verify


CANN_VERSION_CANDIDATES = (
    "/usr/local/Ascend/ascend-toolkit/latest/version.cfg",
    "/usr/local/Ascend/cann/version.cfg",
    "/usr/local/Ascend/ascend-toolkit/latest/arm64-linux/ascend_toolkit_install.info",
)
DRIVER_VERSION_CANDIDATES = (
    "/usr/local/Ascend/driver/version.info",
    "/usr/local/Ascend/driver/version.cfg",
)


REMOTE_CAPTURE_SUFFIX = r'''
import importlib.metadata
import os
import subprocess
import sys
import sysconfig
import time

args = json.loads(sys.argv[1])
finalize_started = time.monotonic()
timings = {}
root = Path(args["root"])
if args.get('installation_marker'):
    import datetime
    marker = Path(args['installation_marker'])
    if not marker.is_relative_to(root) or marker.is_symlink() or marker.parent.is_symlink():
        raise ValueError('installation marker escaped owned runtime')
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({'container_identity': args['container_identity'],
                                 'runtime_root': str(root), 'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}) + '\n')
recipe = args.get("recipe")
image_digest = args.get("image_digest")
if not image_digest:
    raise ValueError("cannot attest image digest")

def first_existing(paths):
    for path in paths:
        candidate = Path(path)
        if candidate.is_file():
            return candidate
    return None

def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError(f"cannot attest {name}: not installed") from exc

cann_file = first_existing(args.get("cann_files") or [])
driver_file = first_existing(args.get("driver_files") or [])
if cann_file is None or driver_file is None:
    raise ValueError("cannot attest CANN/driver version files")

soc = os.environ.get("SOC_VERSION") or os.environ.get("MINDIE_SOC_VERSION")
compiler = os.environ.get("CXX") or os.environ.get("C_COMPILER") or os.environ.get("CXX_COMPILER")
if not compiler:
    # Use the actual custom-op build selection when no compiler was exported.
    cache = root / "vllm-ascend/csrc/build/CMakeCache.txt"
    if cache.is_file():
        for line in cache.read_text().splitlines():
            if line.startswith("CMAKE_CXX_COMPILER:FILEPATH="):
                compiler = line.partition("=")[2].strip()
                break
toolchain = build_toolchain_from_logs(root) if not soc or not compiler else {}
reuse_file = root / '.mindie-runtime/reuse.json'
reuse = json.loads(reuse_file.read_text()) if reuse_file.is_file() else {}
soc = soc or toolchain.get("soc")
compiler = compiler or "; ".join(toolchain.get("compilers") or [])
soc = soc or reuse.get('soc')
compiler = compiler or reuse.get('compiler')
if not soc:
    raise ValueError("cannot attest soc from environment or completed build evidence")
if not compiler:
    raise ValueError("cannot attest compiler from environment or completed build evidence")
python_abi = sysconfig.get_config_var("SOABI")
if not python_abi:
    raise ValueError("cannot attest python_abi")

profile = {
    "image_digest": image_digest,
    "soc": soc,
    "driver": driver_file.read_text(errors="replace").strip()[:200] or "present",
    "cann": cann_file.read_text(errors="replace").strip()[:200] or "present",
    "python_abi": python_abi,
    "torch": package_version("torch"),
    "torch_npu": package_version("torch-npu"),
    "vllm": package_version("vllm"),
    "vllm_ascend": package_version("vllm-ascend"),
    "compiler": compiler,
    "build_env": dict(args.get('build_env') or {}),
    "launch_env": {},
    **installed_dependency_identity(),
    "compatibility_evidence": ".mindie-runtime/profile-evidence/smoke.json",
    "system_files": {
        "cann": {"path": str(cann_file), "sha256": file_digest(cann_file)},
        "driver": {"path": str(driver_file), "sha256": file_digest(driver_file)},
    },
}
if recipe:
    profile["recipe"] = recipe
if args.get("machine_type"):
    profile["machine_type"] = args["machine_type"]
profile["launch_env"] = capture_launch_environment(dict(os.environ))
profile['launch_env']['PYTHONPATH'] = ':'.join([str(root / '.mindie-runtime/metadata'), str(root / 'vllm'), str(root / 'vllm-ascend'), profile['launch_env'].get('PYTHONPATH', '')]).rstrip(':')
if args.get('source_versions'):
    profile['source_versions'] = args['source_versions']

metadata_started = time.monotonic()
capture_kernel_compile_recipe(root)
files = installed_native_files(root)
previous_launch_env = profile['launch_env']
profile['launch_env'] = native_vendor_launch_environment(root, files, profile['launch_env'])
upgraded_loader = profile['launch_env'] != previous_launch_env

evidence_dir = root / ".mindie-runtime/profile-evidence"
evidence_dir.mkdir(parents=True, exist_ok=True)
inputs = _build_namespace["runtime_build_inputs"](root, profile, profile_key(profile))
timings['metadata'] = time.monotonic() - metadata_started
smoke_started = time.monotonic()
if args.get('managed_finalize'):
    print('finalize-runtime: native smoke/proof', file=sys.stderr, flush=True)
if reuse.get('kind') == 'native' and reuse.get('compatibility_evidence') and not upgraded_loader:
    compatibility = json.loads(checked_file(root, reuse['compatibility_evidence']).read_text())
    smoke = {'kind': 'native-compatibility-reuse', 'python_import_executed': False,
             'profile_key': profile_key(profile), 'build_inputs': inputs,
             'compatibility': compatibility, 'source_mapping': native_source_mapping(root)}
else:
    compatibility = (json.loads(checked_file(root, reuse['import_evidence']).read_text())
                     if reuse.get('import_evidence') else None)
    candidate = {'runtime_root': str(root), 'profile': profile, 'build_inputs': inputs,
                 'files': {name: {'sha256': file_digest(checked_file(root, name)), 'role': role}
                           for name, role in files.items()}} if compatibility else None
    if compatibility and compatibility['key'] == native_import_closure_key(candidate):
        smoke = {'kind': 'native-import-closure-reuse', 'python_import_executed': False,
                 'profile_key': profile_key(profile), 'build_inputs': inputs,
                 'compatibility': compatibility, 'source_mapping': native_source_mapping(root)}
    else:
        smoke = native_import_smoke(root, profile, inputs)
    if upgraded_loader and reuse.get('compatibility_evidence'):
        smoke['reason'] = 'native-loader-environment-upgrade'
    if smoke.get('python_import_executed') is not False and not smoke['passed']:
        (evidence_dir / 'smoke.json').write_text(json.dumps(smoke, indent=2) + '\n')
        raise ValueError("installed runtime import smoke failed; inspect profile-evidence/smoke.json")
smoke['source_mapping'] = native_source_mapping(root)
timings['native_smoke'] = time.monotonic() - smoke_started
(evidence_dir / "smoke.json").write_text(json.dumps(smoke, indent=2) + "\n")
(evidence_dir / "cann.json").write_text(json.dumps(profile["system_files"]["cann"], sort_keys=True) + "\n")
(evidence_dir / "driver.json").write_text(json.dumps(profile["system_files"]["driver"], sort_keys=True) + "\n")
evidence = {name: ".mindie-runtime/profile-evidence/" + name + ".json" for name in ("cann", "driver", "smoke")}
if toolchain:
    evidence["toolchain_log"] = toolchain["path"]
if reuse:
    evidence['reuse'] = '.mindie-runtime/reuse.json'
hash_started = time.monotonic()
if args.get('managed_finalize'):
    print('finalize-runtime: capture native hashes', file=sys.stderr, flush=True)
manifest = capture(root, profile, inputs, files, evidence)
timings['capture_hash'] = time.monotonic() - hash_started
manifest['preparation'] = verified_preparation(args.get('preparation', {}), profile)
manifest['execution_view'] = {'source_id': args.get('source_id'), 'python': sys.executable}
if args.get('managed_finalize'):
    # The only source hashes used here were computed just above, in this owned
    # operation after compilation and smoke. Public attest/publish still verify
    # independently supplied native bytes. Copy destinations are always hashed.
    proof_started = time.monotonic()
    _verify_manifest_proof(root, manifest)
    timings['proof_verify'] = time.monotonic() - proof_started
else:
    verify(root, manifest)
cache_result = None
if args.get('publish_native_cache'):
    if not args.get('managed_finalize'):
        raise ValueError('cache handoff requires the current managed capture')
    store_started = time.monotonic()
    print('finalize-runtime: publish native cache', file=sys.stderr, flush=True)
    try:
        cache_result = _store_captured_native(root, Path(SHARED_NATIVE_CACHE), manifest, timings)
    except OSError as exc:
        # A completed filesystem failure in this optional cache does not undo
        # the owned runtime. Integrity/proof failures still abort finalization.
        cache_result = {'status': 'miss', 'reason': str(exc)}
        print('native cache publication failed: ' + str(exc), file=sys.stderr)
    timings['cache_store'] = time.monotonic() - store_started
marker = root / ".mindie-runtime/ready-profile.json"
temp = marker.with_suffix(".tmp")
try:
    temp.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    os.replace(temp, marker)
finally:
    temp.unlink(missing_ok=True)
timings['finalize_total'] = time.monotonic() - finalize_started
# Preserve the complete verified bytes while avoiding repetitive path/JSON
# overhead in stdout collection. Publication and its proof have one reply.
import base64, zlib
encoded_manifest = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
print(json.dumps({'manifest_zlib_base64': base64.b64encode(zlib.compress(encoded_manifest)).decode('ascii'),
                  'manifest_bytes': len(encoded_manifest), 'manifest_digest': digest(manifest),
                  'preparation_timings': timings, 'native_cache': cache_result,
                  'native_smoke_executed': smoke.get('python_import_executed', True)}))
'''


REMOTE_COMMAND_CAPTURE_SUFFIX = r'''
import sys
args = json.loads(sys.argv[1])
root = Path(args['root'])
profile = {'kind': 'command', 'image_digest': args['image_digest'],
           'python_abi': sysconfig.get_config_var('SOABI'),
           'build_env': {}, 'launch_env': command_launch_environment(dict(os.environ), roots=args.get('excluded_roots', []))}
for key in ('recipe', 'machine_type'):
    if args.get(key):
        profile[key] = args[key]
manifest = {'schema_version': 2, 'profile': profile, 'profile_key': profile_key(profile),
            'build_key': digest({'profile': profile, 'sources': args.get('source_id')}),
            'source_id': args.get('source_id'),
            'runtime_root': str(root.resolve()), 'build_inputs': {}, 'files': {}, 'evidence': {},
            'preparation': args.get('preparation', {})}
verify(root, manifest)
marker = root / '.mindie-runtime/ready-profile.json'
marker.parent.mkdir(parents=True, exist_ok=True)
temporary = marker.with_suffix('.tmp')
temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
os.replace(temporary, marker)
print(json.dumps(manifest))
'''


def require_clean_sources(root: Path):
    for name in ("vllm", "vllm-ascend"):
        for node in iter_postorder(discover_repo_tree(root / name, name)):
            dirty = subprocess.check_output(["git", "-C", str(node.repo_path), "status", "--porcelain", "--untracked-files=all"], text=True, encoding="utf-8")
            if dirty.strip():
                raise ValueError("attest a clean materialized parity snapshot, including child submodules")
            for child in node.children:
                path = child.repo_path.relative_to(node.repo_path).as_posix()
                head = subprocess.check_output(["git", "-C", str(child.repo_path), "rev-parse", "HEAD"], text=True, encoding="utf-8").strip()
                entry = subprocess.check_output(["git", "-C", str(node.repo_path), "ls-tree", "HEAD", "--", path], text=True, encoding="utf-8").strip()
                if entry != f"160000 commit {head}\t{path}":
                    raise ValueError("native submodule must be tracked at its pinned commit: " + child.relpath)


def attest(root: Path, spec: dict):
    profile = dict(spec["profile"])
    require_clean_sources(root)
    evidence_dir = root / ".mindie-runtime/profile-evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    # Preserve the prepared image environment when overlaying launch settings.
    environment = os.environ.copy()
    for name, value in profile["launch_env"].items():
        if name in {"PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "ASCEND_CUSTOM_OPP_PATH"} and not value:
            continue
        environment[name] = value + (":" + environment[name] if name in {"PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "ASCEND_CUSTOM_OPP_PATH"} and environment.get(name) else "")
    environment = native_vendor_launch_environment(root, spec['files'], environment)
    profile['launch_env'] = capture_launch_environment(environment)
    key = profile_key(profile)
    inputs = runtime_build_inputs(root, profile, key)
    try:
        smoke = subprocess.run([sys.executable, "-c", "import torch_npu, vllm, vllm_ascend, acl; import vllm_ascend.vllm_ascend_C"], env=environment,
                               capture_output=True, text=True, encoding="utf-8", timeout=60)
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        (evidence_dir / "smoke.json").write_text(json.dumps({"passed": False,
                                                             "build_inputs": inputs, "profile_key": key,
                                                             "error": "import smoke timed out after 60s",
                                                             "stderr": stderr[-8000:]}, indent=2))
        raise ValueError("import smoke timed out; inspect .mindie-runtime/profile-evidence/smoke.json") from exc
    (evidence_dir / "smoke.json").write_text(json.dumps({"passed": smoke.returncode == 0,
                                                       "build_inputs": inputs, "profile_key": key,
                                                       "stderr": smoke.stderr[-8000:]}, indent=2))
    if smoke.returncode:
        raise ValueError("import smoke failed; inspect .mindie-runtime/profile-evidence/smoke.json")
    for name in ("cann", "driver"):
        row = profile["system_files"][name]
        if file_digest(Path(row["path"])) != row["sha256"]:
            raise ValueError("actual environment differs from requested profile")
        (evidence_dir / (name + ".json")).write_text(json.dumps(row, sort_keys=True))
    evidence = {name: ".mindie-runtime/profile-evidence/" + name + ".json" for name in ("cann", "driver", "smoke")}
    manifest = capture(root, profile, inputs, spec["files"], evidence)
    verify(root, manifest)
    # New attestations replace the marker only after all checks have passed.
    marker = root / ".mindie-runtime/ready-profile.json"
    temp = marker.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    os.replace(temp, marker)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("attest", "publish", "restore"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--build-key")
    parser.add_argument("--owned-workers-stopped", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("preparation runs inside the owned Linux container")
    if not args.owned_workers_stopped:
        parser.error("first stop and verify only your workers, then pass --owned-workers-stopped")
    root = args.root.resolve()
    if args.action == "attest":
        if args.spec is None:
            parser.error("attest requires --spec")
        result = attest(root, json.loads(args.spec.read_text()))
    elif args.action == "publish":
        if args.cache is None:
            parser.error("publish requires --cache")
        manifest = json.loads((root / ".mindie-runtime/ready-profile.json").read_text())
        result = {"bundle": str(publish(root, args.cache, manifest)), "build_key": manifest["build_key"]}
    else:
        if args.cache is None or not args.build_key:
            parser.error("restore requires --cache and --build-key")
        if len(args.build_key) != 64 or any(c not in "0123456789abcdef" for c in args.build_key):
            parser.error("build key must be a SHA256 digest")
        bundle = args.cache / args.build_key
        manifest = json.loads((bundle / "manifest.json").read_text())
        if runtime_build_inputs(root, manifest["profile"], manifest["profile_key"]) != manifest["build_inputs"]:
            raise ValueError("cache miss: current source inputs differ from the requested bundle")
        restore(root, bundle, args.build_key)
        marker = root / ".mindie-runtime/ready-profile.json"
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        os.replace(temporary, marker)
        result = {"status": "restored", "build_key": args.build_key}
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
