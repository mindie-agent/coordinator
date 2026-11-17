"""Orthogonal clean-architecture behavioral contracts for mindie-coordinator (AD-04, AD-05, P14, P2.7)."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest

from mindie_coordinator import env_manager, ops
from mindie_coordinator.machine_directory import MachineDirectory, MachineDirectoryUnavailable
from mindie_coordinator.parity_install import runtime_install_step_script
from mindie_coordinator.parity_snapshot import DEFAULT_ENV_PREAMBLE, PYTHON_METADATA_PREAMBLE
from mindie_coordinator.ready_runtime import RuntimePool


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_coordinator_workers_are_pure_stdlib_and_dead_hccl_preflight_deleted():
    assert not (REPO_ROOT / "mindie_coordinator" / "hccl_preflight.py").exists()
    stdlib_names = set(sys.stdlib_module_names)
    workers_dir = REPO_ROOT / "mindie_coordinator" / "workers"
    worker_files = [
        p for p in sorted(workers_dir.glob("*.py")) if p.name != "__init__.py"
    ]
    assert len(worker_files) >= 3
    for path in worker_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] in stdlib_names, f"{path.name} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, f"{path.name} uses relative import"
                assert (node.module or "").split(".")[0] in stdlib_names, f"{path.name} imports {node.module}"


def test_deterministic_python_selector_prefers_torch_npu_over_empty_newer_python(tmp_path):
    usr_local = tmp_path / "usr_local"
    # Python 3.11.10 with torch + torch_npu installed
    py311_bin = usr_local / "python3.11.10" / "bin" / "python3"
    py311_sp = usr_local / "python3.11.10" / "lib" / "python3.11" / "site-packages"
    py311_bin.parent.mkdir(parents=True)
    (py311_sp / "torch-2.6.0.dist-info").mkdir(parents=True)
    (py311_sp / "torch_npu-2.6.0.dist-info").mkdir(parents=True)
    py311_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    py311_bin.chmod(0o755)

    # Empty Python 3.12.2 WITHOUT torch_npu (would wrongly win under `sort -V | tail -n 1`)
    py312_bin = usr_local / "python3.12.2" / "bin" / "python3"
    (usr_local / "python3.12.2" / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
    py312_bin.parent.mkdir(parents=True)
    py312_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    py312_bin.chmod(0o755)

    selector_path = REPO_ROOT / "mindie_coordinator" / "workers" / "python_selector.py"
    harness = (
        "import glob as _g, importlib.util\n"
        f"_orig = _g.glob\n_g.glob = lambda pat: _orig({str(usr_local)!r} + '/python*/bin/python3') if pat.startswith('/usr/local/python') else _orig(pat)\n"
        f"spec = importlib.util.spec_from_file_location('sel', {str(selector_path)!r})\n"
        "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
        "print(mod.select_python())\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in {"MINDIE_PYTHON", "PYTHON", "VIRTUAL_ENV", "CONDA_PREFIX", "MINDIE_RUNTIME_ROOT"}}
    proc = subprocess.run([sys.executable, "-I", "-c", harness], env=env, capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == str(py311_bin)

    # Explicit MINDIE_PYTHON always wins first
    env_explicit = {**env, "MINDIE_PYTHON": str(py312_bin)}
    proc2 = subprocess.run([sys.executable, "-I", "-c", harness], env=env_explicit, capture_output=True, text=True, check=True)
    assert proc2.stdout.strip() == str(py312_bin)


def test_parity_scripts_preserve_intranet_pip_config_and_avoid_sort_v(monkeypatch):
    joined_meta = "\n".join(PYTHON_METADATA_PREAMBLE)
    joined_default = "\n".join(DEFAULT_ENV_PREAMBLE)
    for text in (joined_meta, joined_default):
        assert "sort -V" not in text
        assert "PIP_CONFIG_FILE=/dev/null" not in text
        assert "unset PIP_EXTRA_INDEX_URL" not in text

    monkeypatch.setenv("PIP_CONFIG_FILE", "/etc/intranet/pip.conf")
    monkeypatch.setenv("PIP_INDEX_URL", "https://pypi.intranet.local/simple")
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://ascend.intranet.local/simple")
    step_script = runtime_install_step_script(
        runtime_root="/vllm-workspace/executions/s1/worker",
        marker_dirname=".remote-code-parity",
        container_identity="mindie-alice",
        step="install-vllm",
        python="/vllm-workspace/executions/s1/worker/.venv/bin/python",
    )
    assert "PIP_CONFIG_FILE=/dev/null" not in step_script
    assert "unset PIP_EXTRA_INDEX_URL" not in step_script
    assert "export PIP_CONFIG_FILE=/etc/intranet/pip.conf" in step_script
    assert "export PIP_INDEX_URL=https://pypi.intranet.local/simple" in step_script
    assert "export PIP_EXTRA_INDEX_URL=https://ascend.intranet.local/simple" in step_script


def test_pre_submit_error_never_wedges_epoch_while_post_submit_becomes_uncertain(tmp_path):
    attestation = {
        "profile_key": "pk-1",
        "build_key": "bk-1",
        "container_id": "cid-1",
        "profile": {"launch_env": {}, "build_env": {}},
        "launch_preamble": "",
    }
    state = {"epoch": "epoch-1", "fail_before_submit": True, "fail_during_submit": False, "tasks": []}

    class ControlledBackend:
        def inspect(self, _runtime, **_kwargs):
            return dict(attestation)

        def host(self, _runtime, req):
            action = req["action"]
            if action == "container-ssh-reserve":
                return {"port": req["port"]}
            if action == "status":
                # First status read sets epoch; second status check fails BEFORE submit!
                if state["fail_before_submit"] and req.get("coordination_epoch") is not None:
                    raise RuntimeError("transient SSH mux drop before submit")
                return {"coordination_epoch": state["epoch"], "tasks": list(state["tasks"])}
            if action == "submit":
                if state["fail_during_submit"]:
                    raise RuntimeError("SSH dropped mid-submit")
                task = {"task_id": req["task_id"], "state": "granted", "granted_devices": [0], "fence_token": 7}
                state["tasks"] = [task]
                return {"task": task}
            if action == "acquire":
                return {"task": state["tasks"][0]}
            return {"task": {"state": "released"}}

    backend = ControlledBackend()
    pool = RuntimePool(tmp_path / "pool", backend)
    sess = pool.session_open("alice", "sess-1", {})
    spec = {
        "user": "alice",
        "python": "/usr/bin/python3",
        "endpoint": {"host": "192.0.2.10", "port": 46001, "user": "root", "root": "/work/r1", "cwd": "/work/r1"},
        "host_endpoint": {"host": "192.0.2.10", "port": 22, "user": "root"},
        "container_name": "mindie-alice",
        "service_ports": [],
    }
    pool.register("rt-1", spec)
    # Force legacy needs_repair state in DB and verify checkout self-heals it!
    with pool.lock, pool.transaction() as db:
        rt_row = pool.get(db, "runtime", "rt-1")
        rt_row["state"] = "needs_repair"
        pool.put(db, "runtime", rt_row)
    binding = pool.checkout("alice", sess["id"], "pk-1", "chk-1", "rt-1")
    assert binding["state"] == "bound"

    # Trigger pre-submit error after initial status sets epoch-1 -> run MUST remain "pending", NOT "uncertain"!
    # Note: _request_run reuses pending_status on first poll, so test resumed poll with pre-submit failure:
    state["fail_before_submit"] = False
    state["fail_during_submit"] = False
    # First test pre-submit failure via startup-context / verify-preflight path by simulating status failure after epoch discovery
    class PreSubmitFailOnceBackend(ControlledBackend):
        def __init__(self):
            self.calls = 0

        def submit_and_acquire(self, runtime, request):
            return self.host(runtime, {**request, "action": "submit"})

        def startup_context(self, _runtime, _task_id):
            return {"coordination_epoch": "epoch-before-reboot", "container": {"Id": "cid-1", "State": {"Running": True}}}

        def submit_and_preflight(self, runtime, request):
            return self.host(runtime, {**request, "action": "submit"})

        def verify_preflight(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient container inspect timeout before submit")
            return True

    backend2 = PreSubmitFailOnceBackend()
    pool2 = RuntimePool(tmp_path / "pool2", backend2)
    sess2 = pool2.session_open("alice", "sess-2", {})
    pool2.register("rt-2", spec)
    bind2 = pool2.checkout("alice", sess2["id"], "pk-1", "chk-2", "rt-2")
    run_pre = pool2._request_run("alice", bind2["id"], "req-pre", {}, "bk-1", [0], 0, _managed=True)
    # Because verify_preflight failed BEFORE submit_dispatched, state stays "pending" with epoch cleared!
    assert run_pre["state"] == "pending"
    assert run_pre["epoch"] is None
    assert run_pre["submit_dispatched"] is False

    # Host reboots to a brand-new epoch ("epoch-after-reboot"): next poll succeeds cleanly without wedging!
    state["epoch"] = "epoch-after-reboot"
    recovered = pool2.control("alice", run_pre["id"], "poll")
    assert recovered["state"] == "granted"
    assert recovered["task"]["fence_token"] == 7

    # Now verify that an error DURING submit (submit_dispatched=True) DOES transition to "uncertain"!
    pool2.control("alice", run_pre["id"], "release")
    state["tasks"] = []
    state["fail_during_submit"] = True
    run_mid = pool2.request_run("alice", bind2["id"], "req-mid", {}, "bk-1", [0], 0)
    assert run_mid["submit_dispatched"] is True
    assert run_mid["state"] == "uncertain"


def test_op_preflight_cuda_guard_and_clone_emit_warnings_without_blocking(tmp_path):
    from mindie_coordinator.op_preflight import check_snapshot_op_contracts, format_op_violations
    from mindie_coordinator.backend import RemoteBackend

    repo = tmp_path / "vllm-ascend"
    (repo / "vllm_ascend" / "ops").mkdir(parents=True)
    target = repo / "vllm_ascend" / "ops" / "rotary.py"
    target.write_text(
        "import torch, torch_npu\n"
        "def run_op(x, w):\n"
        "    if torch.cuda.is_available():\n"
        "        x = x.contiguous().clone()\n"
        "    out = torch_npu.npu_rms_norm(x, w, 1e-6)\n"
        "    return out.shape\n",
        encoding="utf-8",
    )
    snapshot = {"records": [{"relpath": "vllm-ascend", "source_path": str(repo), "changed_paths": ["vllm_ascend/ops/rotary.py"]}]}
    violations = check_snapshot_op_contracts(snapshot)
    assert violations, "Expected static op heuristic match"

    executed = []

    class RecordingShell:
        def run(self, _ep, cmd, **_kw):
            executed.append(cmd)
            return {"outcome": "success", "refs": {}}

    backend = RemoteBackend(shell=RecordingShell())
    binding = {"endpoint": {"cwd": "/vllm-workspace"}, "source_snapshot": snapshot}
    res = backend.preflight(binding, "python3 -c 'print(42)'", {})
    assert len(executed) == 1
    assert res["preflight_warnings"]
    assert format_op_violations(violations) in binding["preflight_warnings"]


def test_receipt_honesty_preserves_success_after_remote_provision_or_remove(tmp_path, monkeypatch):
    class FailingLocalSaveDirectory(MachineDirectory):
        def upsert_machine(self, record, *, set_default=False):
            raise MachineDirectoryUnavailable("ENOSPC writing machines.json")

        def endpoints_document(self):
            raise OSError(28, "No space left on device")

    store = FailingLocalSaveDirectory(path=tmp_path / "machines.json")
    monkeypatch.setattr(
        env_manager,
        "provision_user_container",
        lambda **_kw: {
            "user": "alice",
            "container_name": "mindie-alice",
            "host": "192.0.2.99",
            "ssh_port": 46010,
            "image": "rc",
            "workdir": "/vllm-workspace",
            "privileged": True,
            "state": "ready",
            "receipt_warning": "OSError: [Errno 28] No space left on device",
            "receipt_unconfirmed": True,
        },
    )
    prov = env_manager.handle_container(
        {"action": "provision", "host": "192.0.2.99", "image": "rc", "user": "root"},
        directory=store,
        state_dir=tmp_path,
        queue=mock.Mock(),
    )
    assert prov["state"] == "ready"
    assert prov["receipt_unconfirmed"] is True
    assert prov["receipt_warning"]

    # Also verify ops.mindie_call hoists receipt_warning + receipt_unconfirmed with outcome="success"
    monkeypatch.setattr(env_manager, "handle_container", lambda *_a, **_kw: prov)
    mcp_res = ops.mindie_call(
        "mindie.container",
        {"action": "provision", "host": "192.0.2.99", "image": "rc"},
        allow_native_context=False,
    )["result"]
    assert mcp_res["outcome"] == "success"
    assert mcp_res["receipt_unconfirmed"] is True
    assert mcp_res["receipt_warning"]


def test_control_plane_pr_checklist_validator():
    script_path = REPO_ROOT / ".github" / "scripts" / "check_pr_checklist.py"
    spec = importlib.util.spec_from_file_location("check_pr_checklist", script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    valid_body = """## 变更摘要 (Summary)
- Remove op_preflight hard blocker and enforce pre-submit epoch honesty in ready_runtime.py.

## 设计原则 Review (Pre-PR Checklist)
- [x] **CTRL-LEASE-OR-AUTH**: HostQueue lease authority preserved.
- [x] **CTRL-HANDLE**: Zero-_meta MCP and explicit execution_id reconnect verified.
- [x] **CTRL-STREAM-IO**: Remote workers verified pure stdlib under python -I.
- [x] **CTRL-RECEIPT**: Post-success local receipt failures keep outcome=success.

- **CTRL-LEASE-OR-AUTH**: Verified in test_container_reservation_lifecycle.py.
- **CTRL-HANDLE**: Verified in test_task_server.py and test_clean_arch_contracts.py.
- **CTRL-STREAM-IO**: Verified pure-stdlib AST check on mindie_coordinator/workers/*.py.
- **CTRL-RECEIPT**: Verified ENOSPC local receipt failure keeps outcome=success.

## 验证结果 (Verification)
- Ran isolated pytest -q -m "not live_npu" (all tests passed).
"""
    assert mod.validate_pr_body(valid_body, profile="control-plane") == []

    # Docs-only PR with `not-applicable: docs-only` is accepted
    docs_body = valid_body.replace(
        "Verified in test_container_reservation_lifecycle.py.",
        "not-applicable: docs-only update to configuration-and-tools.md",
    )
    assert mod.validate_pr_body(docs_body, profile="control-plane") == []

    # Unchecked box or bare `N/A` is rejected
    unchecked_body = valid_body.replace("- [x] **CTRL-RECEIPT**", "- [ ] **CTRL-RECEIPT**")
    assert any("Unchecked" in err or "Missing checked" in err for err in mod.validate_pr_body(unchecked_body, profile="control-plane"))

    placeholder_body = valid_body.replace("Verified in test_container_reservation_lifecycle.py.", "N/A")
    assert any("CTRL-LEASE-OR-AUTH" in err for err in mod.validate_pr_body(placeholder_body, profile="control-plane"))
