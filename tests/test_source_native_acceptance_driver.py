"""The live driver's C++ probe and proof are executable without a remote host."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

import pytest

DRIVER_PATH = Path(__file__).parent / "live/run_source_native_acceptance.py"
SPEC = importlib.util.spec_from_file_location("source_native_acceptance", DRIVER_PATH)
driver = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(driver)


@pytest.mark.skipif(os.name != "posix", reason="The live Linux C++ probe requires a POSIX extension toolchain")
def test_real_cpp_probe_compiles_and_emits_actual_loaded_binary_proof(tmp_path):
    sources = driver.write_cpp_probe_repos(tmp_path / "source", offset=1)
    ascend = sources["vllm-ascend"]
    build = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=ascend,
                           capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, build.stdout[-3000:] + build.stderr[-3000:]
    extension = next((ascend / "vllm_ascend").glob("vllm_ascend_C*.so"))
    assert extension.is_file()

    root = tmp_path / "source"
    (root / ".mindie-runtime").mkdir()
    (root / ".mindie-runtime/ready-profile.json").write_text(
        json.dumps({"build_key": "b" * 64}), encoding="utf-8")
    environment = {
        **os.environ,
        "MINDIE_PYTHON": sys.executable,
        "MINDIE_RUN_ID": "debug-driver-proof",
        "MINDIE_ACCEPTANCE_INPUT": "driver-probe",
        "MINDIE_ACCEPTANCE_EXPECTED": "49",
        "MINDIE_ACCEPTANCE_CACHE_KEY": "",
        "MINDIE_ACCEPTANCE_CACHE_RELATIVE": "",
        "PYTHONPATH": os.pathsep.join([str(root / "vllm"), str(root / "vllm-ascend")]),
    }
    result = subprocess.run(["bash", "-c", driver.proof_command(49)], cwd=root, env=environment,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    line = next(line for line in result.stdout.splitlines()
                if line.startswith("MINDIE_ACCEPTANCE_PROOF "))
    proof = json.loads(line.split(" ", 1)[1])
    loaded = Path(proof["loaded_module_path"])
    assert loaded.is_file()
    assert proof["run_id"] == "debug-driver-proof"
    assert proof["probe_value"] == 49
    assert proof["loaded_binary_sha256"] == hashlib.sha256(loaded.read_bytes()).hexdigest()
    assert proof["patch_digest"] == driver.patch_digest(ascend)
    assert proof["calls"][0]["input_id"] == "driver-probe"
    assert proof["measurement_window"]["start_ts"] <= proof["calls"][0]["timestamp"]
    assert proof["calls"][0]["timestamp"] <= proof["measurement_window"]["end_ts"]
    assert (root / "acceptance-proof.json").is_file()


def test_changed_source_changes_patch_digest_without_touching_external_repos(tmp_path):
    sources = driver.write_cpp_probe_repos(tmp_path / "source", offset=1)
    before = driver.patch_digest(sources["vllm-ascend"])
    commit = driver.set_source_offset(sources["vllm-ascend"], 2)
    after = driver.patch_digest(sources["vllm-ascend"])
    assert after != before
    assert commit != driver._git(sources["vllm-ascend"], "rev-parse", "HEAD~1")


def test_source_deployment_proof_reads_actual_git_heads_and_files(tmp_path):
    root = tmp_path / "deployed"
    for name in ("vllm-source", "vllm-ascend-source"):
        repo = root / name
        repo.mkdir(parents=True)
        (repo / "README.md").write_text(name + "\n", encoding="utf-8")
        (repo / "csrc").mkdir()
        (repo / "csrc/kernel.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "Fixture"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], cwd=repo, check=True)
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    environment = {**os.environ, "MINDIE_PYTHON": sys.executable,
                   "MINDIE_RUN_ID": "debug-source-deploy"}
    result = subprocess.run(["bash", "-c", driver._source_command(None)], cwd=root, env=environment,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(line for line in result.stdout.splitlines()
                if line.startswith("MINDIE_SOURCE_DEPLOYMENT_PROOF "))
    proof = json.loads(line.split(" ", 1)[1])
    assert proof["mode"] == "source-deployment-only"
    assert proof["run_id"] == "debug-source-deploy"
    assert proof["vllm_head"] == subprocess.check_output(
        ["git", "-C", str(root / "vllm-source"), "rev-parse", "HEAD"], text=True).strip()
    assert proof["vllm_ascend_head"] == subprocess.check_output(
        ["git", "-C", str(root / "vllm-ascend-source"), "rev-parse", "HEAD"], text=True).strip()
    assert proof["file_sha256"]["vllm-source/README.md"]
    assert proof["file_sha256"]["vllm-ascend-source/csrc/kernel.cpp"]


@pytest.mark.skipif(os.name != "posix", reason="The live Linux C++ probe requires a POSIX extension toolchain")
def test_dry_run_persists_full_proof_manifest_and_bench_projection(tmp_path):
    output = tmp_path / "dry-run-output"
    result = subprocess.run([sys.executable, str(DRIVER_PATH), "--dry-run", "--output", str(output)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    dry = output / "dry-run"
    proof = json.loads((dry / "proof.json").read_text())
    manifest = json.loads((dry / "run-manifest.json").read_text())
    bench = json.loads((dry / "bench-execution-evidence.json").read_text())
    assert manifest["status"] == "passed"
    assert manifest["run_id"] == proof["run_id"]
    assert manifest["active_execution_proof"]["candidate_build_key"] == proof["candidate_build_key"]
    assert set(bench) == {"run_id", "patch_digest", "loaded_binary_sha256", "loaded_module_path",
                          "calls", "evidence_ref"}
    assert Path(proof["loaded_module_path"]).is_file() is False  # sandbox is removed; copy remains
    assert (dry / Path(proof["loaded_module_path"]).name).is_file()


@pytest.mark.skipif(os.name != "posix", reason="The live Linux C++ probe requires a POSIX extension toolchain")
def test_parallel_probe_writes_real_proofs_for_overlapping_cpp_processes(tmp_path):
    sources = driver.write_cpp_probe_repos(tmp_path / "source", offset=1)
    ascend = sources["vllm-ascend"]
    build = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=ascend,
                           capture_output=True, text=True, timeout=10)
    assert build.returncode == 0, build.stdout[-3000:] + build.stderr[-3000:]
    probe = DRIVER_PATH.with_name("parallel_native_probe.py")
    barrier = tmp_path / "barrier"
    processes = []
    roots = []
    for index in range(2):
        root = tmp_path / f"slot-{index}"
        shutil.copytree(tmp_path / "source", root)
        (root / ".mindie-runtime").mkdir()
        (root / ".mindie-runtime/ready-profile.json").write_text(
            json.dumps({"build_key": "b" * 64}))
        environment = {
            **os.environ, "MINDIE_RUN_ID": f"parallel-local-{index}",
            "MINDIE_PARALLEL_DIR": str(barrier), "MINDIE_PARALLEL_EXPECTED": "2",
            "MINDIE_PARALLEL_TIMEOUT": "5", "MINDIE_PARALLEL_SECONDS": "0.2",
            "MINDIE_ACCEPTANCE_EXPECTED": "49",
            "PYTHONPATH": str(root / "vllm-ascend"),
        }
        processes.append(subprocess.Popen([sys.executable, str(probe)], cwd=root, env=environment,
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        roots.append(root)
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, stdout[-1000:] + stderr[-1000:]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)
    proofs = [json.loads((root / "parallel-proof.json").read_text()) for root in roots]
    for proof in proofs:
        assert json.loads(Path(proof["evidence_ref"]).read_text()) == proof
        assert proof["probe_value"] == 49
        assert proof["call_count"] > 2
        assert all(proof["measurement_window"]["start_ts"] <= call["timestamp"]
                   <= proof["measurement_window"]["end_ts"] for call in proof["calls"])
    windows = [proof["measurement_window"] for proof in proofs]
    assert min(window["end_ts"] for window in windows) > max(window["start_ts"] for window in windows)
    assert proofs[0]["loaded_binary_sha256"] == proofs[1]["loaded_binary_sha256"]
    assert proofs[0]["loaded_inode"] != proofs[1]["loaded_inode"]
