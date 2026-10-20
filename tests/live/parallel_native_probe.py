"""Run two real native extensions after both execution slots are ready.

[Role] Remote CPU-only live fixture; records actual imported binary calls.
[Invariants] Every proof refers to a written file in its own execution view.
[Timeouts & Locks] Monotonic bounded rendezvous/call windows; Git read <= 10s.
[Side-Effect Policy] Writes only the explicit owned barrier and execution root.
"""
import hashlib
import json
import os
import pathlib
import subprocess
import time


def main() -> None:
    root = pathlib.Path.cwd()
    run_id = os.environ["MINDIE_RUN_ID"]
    barrier = pathlib.Path(os.environ["MINDIE_PARALLEL_DIR"])
    participants_expected = int(os.environ["MINDIE_PARALLEL_EXPECTED"])
    probe_expected = int(os.environ["MINDIE_ACCEPTANCE_EXPECTED"])
    timeout = float(os.environ["MINDIE_PARALLEL_TIMEOUT"])
    barrier.mkdir(parents=True, exist_ok=True)
    ready_path = barrier / (run_id + ".ready.json")
    fd = os.open(ready_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps({"run_id": run_id, "pid": os.getpid(),
                                  "ready_at": time.time()}) + "\n")
    deadline = time.monotonic() + timeout
    while True:
        ready = sorted(barrier.glob("*.ready.json"))
        if len(ready) == participants_expected:
            break
        if len(ready) > participants_expected:
            raise SystemExit("rendezvous saw more than expected participants")
        if time.monotonic() >= deadline:
            raise SystemExit("rendezvous timeout after %d/%d participants" % (len(ready), participants_expected))
        time.sleep(0.05)
    participants = sorted(json.loads(path.read_text())["run_id"] for path in ready)
    if len(set(participants)) != participants_expected or run_id not in participants:
        raise SystemExit("rendezvous participant identity mismatch")
    rendezvous_at = time.time()
    source = root / "vllm-ascend/csrc/binding.cpp"
    header = root / "vllm-ascend/csrc/kernels/math_utils.h"
    patch = hashlib.sha256(b"vllm-ascend-cpp-probe\0" + source.read_bytes()
                          + b"\0" + header.read_bytes()).hexdigest()
    import vllm_ascend.vllm_ascend_C as extension
    loaded = pathlib.Path(extension.__file__).resolve()
    profile_path = root / ".mindie-runtime/ready-profile.json"
    profile = json.loads(profile_path.read_text())
    started = time.time()
    deadline = time.monotonic() + float(os.environ.get("MINDIE_PARALLEL_SECONDS", "2.0"))
    calls = []
    values = set()
    while time.monotonic() < deadline:
        value = extension.probe(41)
        if value != probe_expected:
            raise SystemExit("compiled probe returned %r, expected %r" % (value, probe_expected))
        values.add(value)
        calls.append({"input_id": "parallel", "call_path": "vllm_ascend.vllm_ascend_C.probe",
                       "timestamp": time.time()})
        time.sleep(0.001)
    ended = time.time()
    if len(calls) < 2:
        raise SystemExit("bounded interval did not produce repeated calls")
    proof = {
        "run_id": run_id,
        "patch_digest": patch,
        "loaded_binary_sha256": hashlib.sha256(loaded.read_bytes()).hexdigest(),
        "loaded_module_path": str(loaded),
        "loaded_module_relative": str(loaded.relative_to(root)),
        "calls": calls,
        "measurement_window": {"start_ts": started, "end_ts": ended},
        "candidate_build_key": profile["build_key"],
        "evidence_ref": str((root / "parallel-proof.json").resolve()),
        "ready_profile": str(profile_path.resolve()),
        "source_commit": subprocess.check_output(["git", "-C", "vllm-ascend", "rev-parse", "HEAD"],
                                                 text=True, timeout=10).strip(),
        "probe_value": probe_expected,
        "participants": participants,
        "rendezvous_at": rendezvous_at,
        "loaded_inode": os.stat(loaded).st_ino,
        "loaded_nlink": os.stat(loaded).st_nlink,
        "call_count": len(calls),
        "values": sorted(values),
    }
    proof_path = root / "parallel-proof.json"
    proof_temp = proof_path.with_suffix(".tmp")
    proof_temp.write_text(json.dumps(proof, sort_keys=True) + "\n")
    os.replace(proof_temp, proof_path)
    result = {**proof, "pid": os.getpid(), "source_path": str(root)}
    result_path = barrier / (run_id + ".result.json")
    tmp = result_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, sort_keys=True) + "\n")
    os.replace(tmp, result_path)
    while True:
        if len(list(barrier.glob("*.result.json"))) == participants_expected:
            break
        if time.monotonic() >= deadline + timeout:
            raise SystemExit("result rendezvous timeout")
        time.sleep(0.05)
    print("MINDIE_ACCEPTANCE_PROOF " + json.dumps(proof, sort_keys=True))



if __name__ == "__main__":
    main()
