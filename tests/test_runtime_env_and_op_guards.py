"""Verification for Coordinator runtime env bootstrap, slot Python/lease guards, and Ascend op preflight."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from mindie_coordinator.endpoint_bridge import build_endpoints_document
from mindie_coordinator.env_manager import handle_container, handle_lease
from mindie_coordinator.execution_slots import (
    ASCEND_ENV_BOOTSTRAP_SH,
    slot_cleanup_command,
    slot_environment,
    slot_preamble,
)
from mindie_coordinator.machine_directory import MachineDirectory
from mindie_coordinator.managed_execution import task_preamble
from mindie_coordinator.op_preflight import (
    check_python_op_contracts,
    check_snapshot_op_contracts,
    format_op_violations,
    load_runtime_op_catalog,
    write_runtime_op_catalog,
)


@pytest.mark.skipif(os.name == "nt", reason="executes Linux container imports and launch payload locally")
def test_ascend_env_bootstrap_sources_unbound_var_script_safely_under_set_eu(tmp_path):
    """Huawei CANN set_env.sh references unbound variables; bootstrap must succeed and preserve set -eu."""
    fake_cann = tmp_path / "set_env.sh"
    fake_cann.write_text(
        "# Simulated CANN set_env.sh with unbound ZSH_VERSION and LD_LIBRARY_PATH\n"
        "if [ -n \"$ZSH_VERSION\" ]; then :; fi\n"
        "export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest\n"
        "export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp\n"
        "export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH\n"
        "false\n",
        encoding="utf-8",
    )
    bootstrap = ASCEND_ENV_BOOTSTRAP_SH.replace(
        "/usr/local/Ascend/ascend-toolkit/set_env.sh",
        shlex.quote(fake_cann.as_posix()),
    )
    script = (
        "set -eu\n"
        "unset ASCEND_HOME_PATH ASCEND_OPP_PATH LD_LIBRARY_PATH ZSH_VERSION\n"
        f"{bootstrap}\n"
        "printf 'HOME=%s\\nOPP=%s\\nOPTS=%s\\n' \"$ASCEND_HOME_PATH\" \"$ASCEND_OPP_PATH\" \"$-\"\n"
    )
    completed = subprocess.run(
        ["sh", "-c", script],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    lines = dict(line.split("=", 1) for line in completed.stdout.strip().splitlines())
    assert lines["HOME"] == "/usr/local/Ascend/ascend-toolkit/latest"
    assert lines["OPP"] == "/usr/local/Ascend/ascend-toolkit/latest/opp"
    assert "e" in lines["OPTS"] and "u" in lines["OPTS"]


def test_endpoint_bridge_propagates_runtime_env_file_for_container_endpoints(tmp_path):
    store = MachineDirectory(path=tmp_path / "machines.json")
    store.upsert_machine(
        {
            "alias": "atlas-a3",
            "host": {"ip": "192.168.13.165", "port": 22, "user": "root"},
        },
        set_default=True,
    )
    res = handle_container(
        {
            "action": "bind",
            "alias": "atlas-a3",
            "container_name": "mindie-dev",
            "ssh_port": 46012,
            "workdir": "/vllm-workspace",
            "runtime_env_file": "/etc/profile.d/mindie-ascend-env.sh",
        },
        directory=store,
        state_dir=tmp_path,
    )
    ep = res["remote_dev_endpoint"]
    assert ep["runtime_env_file"] == "/etc/profile.d/mindie-ascend-env.sh"

    doc = build_endpoints_document(store.load()[0])
    assert doc["endpoints"]["atlas-a3"]["runtime_env_file"] == "/etc/profile.d/mindie-ascend-env.sh"
    assert doc["endpoints"]["atlas-a3-docker"]["runtime_env_file"] == "/etc/profile.d/mindie-ascend-env.sh"
    assert doc["endpoints"]["default"]["runtime_env_file"] == "/etc/profile.d/mindie-ascend-env.sh"
    assert "runtime_env_file" not in doc["endpoints"]["atlas-a3-host"]


@pytest.mark.skipif(os.name == "nt", reason="executes Linux container imports and launch payload locally")
def test_slot_guards_block_unleased_device_override_and_preserve_worktree_under_python_isolated_flag(tmp_path):
    slot_id = f"test-guard-{os.getpid()}"
    # Create two competing vllm_ascend packages: base image vs active task worktree
    base_pkg_root = tmp_path / "base_image" / "vllm-ascend"
    (base_pkg_root / "vllm_ascend").mkdir(parents=True)
    (base_pkg_root / "vllm_ascend" / "__init__.py").write_text("ORIGIN = 'base-image'\n", encoding="utf-8")

    task_root = tmp_path / "task_worktree"
    (task_root / ".mindie-runtime" / "metadata").mkdir(parents=True)
    (task_root / "vllm-ascend" / "vllm_ascend").mkdir(parents=True)
    (task_root / "vllm-ascend" / "vllm_ascend" / "__init__.py").write_text("ORIGIN = 'active-worktree'\n", encoding="utf-8")

    binding = {
        "python": Path(sys.executable).as_posix(),
        "endpoint": {"cwd": task_root.as_posix()},
        "source_names": ["vllm-ascend"],
        "launch_preamble": f"export PYTHONPATH={shlex.quote(base_pkg_root.as_posix())}",
    }
    preamble = task_preamble(binding, slot_id=slot_id)
    try:
        # 1. Inline ASCEND_RT_VISIBLE_DEVICES=4,5 override on lease 0,1 must be restored to 0,1
        #    AND python3 -I must still load active-worktree instead of base-image!
        cmd_override = (
            f"{preamble}\n"
            "export MINDIE_LEASED_DEVICES=0,1\n"
            "export ASCEND_RT_VISIBLE_DEVICES=0,1\n"
            "ASCEND_RT_VISIBLE_DEVICES=4,5 python3 -I -c "
            "'import json, os, vllm_ascend; "
            "print(json.dumps({\"devices\": os.environ.get(\"ASCEND_RT_VISIBLE_DEVICES\"), "
            "\"origin\": vllm_ascend.ORIGIN, \"file\": vllm_ascend.__file__}))'\n"
        )
        res = subprocess.run(["sh", "-c", cmd_override], capture_output=True, text=True, timeout=10, check=True)
        payload = json.loads(res.stdout.strip().splitlines()[-1])
        assert payload["devices"] == "0,1"
        assert payload["origin"] == "active-worktree"
        assert Path(payload["file"]).resolve() == (task_root / "vllm-ascend" / "vllm_ascend" / "__init__.py").resolve()
        assert "[mindie-slot-guard] restored ASCEND_RT_VISIBLE_DEVICES='0,1'" in res.stderr

        # 2. Valid single-rank subset or relative index on lease 10,11 is mapped/preserved,
        #    while full unleased override 0,1 is blocked and restored to 10,11.
        cmd_subset = (
            f"{preamble}\n"
            "export MINDIE_LEASED_DEVICES=10,11\n"
            "export ASCEND_RT_VISIBLE_DEVICES=10,11\n"
            "r_phys=$(ASCEND_RT_VISIBLE_DEVICES=11 python3 -c 'import os; print(os.environ[\"ASCEND_RT_VISIBLE_DEVICES\"])')\n"
            "r_rel=$(ASCEND_RT_VISIBLE_DEVICES=1 python3 -c 'import os; print(os.environ[\"ASCEND_RT_VISIBLE_DEVICES\"])')\n"
            "r_bad=$(ASCEND_RT_VISIBLE_DEVICES=0,1 python3 -c 'import os; print(os.environ[\"ASCEND_RT_VISIBLE_DEVICES\"])' 2>/dev/null)\n"
            "printf '%s|%s|%s\\n' \"$r_phys\" \"$r_rel\" \"$r_bad\"\n"
        )
        res_sub = subprocess.run(["sh", "-c", cmd_subset], capture_output=True, text=True, timeout=10, check=True)
        assert res_sub.stdout.strip() == "11|11|10,11"
    finally:
        subprocess.run(["sh", "-c", slot_cleanup_command(slot_id)], check=False, timeout=5)


def test_op_preflight_catches_real_phase4_operator_bugs():
    # 1. Scenario 1 & 2 bug: torch_npu.npu_rms_norm returns (y, rstd) tuple used directly as Tensor,
    #    or torch.ops._C_ascend.rms_norm unpacked as a 2-tuple, or hallucinated op name.
    bad_rmsnorm_code = """
import torch
import torch_npu

def bad_forward(x, residual, weight, eps):
    out = torch_npu.npu_rms_norm(x, weight, eps)
    y, rstd = torch.ops._C_ascend.rms_norm(x, weight, eps)
    h, res = torch_npu.npu_add_rms_norm(x, residual, weight, eps)
    z = torch.ops._C_ascend.npu_fused_rmsnorm_gated(x, weight, eps)
    return out.view(-1), y, h, z
"""
    violations = check_python_op_contracts(bad_rmsnorm_code, "vllm_ascend/ops/layernorm.py")
    rules = [(v["rule"], v["operator"]) for v in violations]
    assert ("return_arity_mismatch", "torch.ops._C_ascend.rms_norm") in rules
    assert ("return_arity_mismatch", "torch_npu.npu_add_rms_norm") in rules
    assert ("unknown_ascend_op", "torch.ops._C_ascend.npu_fused_rmsnorm_gated") in rules
    assert ("return_arity_mismatch", "torch_npu.npu_rms_norm") in rules

    # 2. Scenario 3 bug: passing non-contiguous .transpose() view or temporary .clone() copy
    #    into Tensor(a!) in-place state argument of npu_recurrent_gated_delta_rule / npu_scatter_nd_update_.
    bad_gdn_code = """
import torch_npu

def gdn_decode_step(q, k, v, beta, ssm_state, idx, updates):
    transposed_state = ssm_state.transpose(-1, -2)
    out = torch_npu.npu_recurrent_gated_delta_rule(q, k, v, beta, initial_state=transposed_state)
    torch_npu.npu_scatter_nd_update_(ssm_state.permute(1, 0, 2), idx, updates)
    return out
"""
    gdn_violations = check_python_op_contracts(bad_gdn_code, "vllm_ascend/ops/linear_op.py")
    gdn_rules = [(v["rule"], v["operator"]) for v in gdn_violations]
    assert ("inplace_noncontiguous_or_copy_arg", "torch_npu.npu_recurrent_gated_delta_rule") in gdn_rules
    assert ("inplace_noncontiguous_or_copy_arg", "torch_npu.npu_scatter_nd_update_") in gdn_rules

    # 3. Unmigrated torch.cuda call in non-test Ascend module
    cuda_code = """
import torch

def sync_device():
    torch.cuda.synchronize()
"""
    cuda_violations = check_python_op_contracts(cuda_code, "vllm_ascend/worker/model_runner.py")
    assert len(cuda_violations) == 1
    assert cuda_violations[0]["rule"] == "unmigrated_cuda_call"
    assert "torch.npu.synchronize" in cuda_violations[0]["hint"]

    # 4. Valid Ascend operator code produces zero violations
    good_code = """
import torch
import torch_npu

def valid_forward(x, residual, weight, eps, q, k, v, beta, ssm_state):
    y, _ = torch_npu.npu_rms_norm(x, weight, eps)
    y2 = torch.ops._C_ascend.rms_norm(y, weight, eps)
    y3, _, residual = torch_npu.npu_add_rms_norm(y2, residual, weight, eps)
    out = torch_npu.npu_recurrent_gated_delta_rule(q, k, v, beta, initial_state=ssm_state)
    return y3, residual, out
"""
    assert check_python_op_contracts(good_code, "vllm_ascend/models/qwen3_5.py") == []


def test_snapshot_op_preflight_and_catalog_persistence(tmp_path):
    repo_dir = tmp_path / "vllm-ascend"
    (repo_dir / "vllm_ascend" / "models").mkdir(parents=True)
    bad_file = repo_dir / "vllm_ascend" / "models" / "qwen3_5.py"
    bad_file.write_text(
        "import torch_npu\n"
        "def run(x, w):\n"
        "    out = torch_npu.npu_rms_norm(x, w, 1e-6)\n"
        "    return out.shape\n",
        encoding="utf-8",
    )
    catalog_path = write_runtime_op_catalog(tmp_path)
    assert catalog_path.is_file()
    loaded = load_runtime_op_catalog(tmp_path)
    assert loaded["torch_npu.npu_rms_norm"]["return_arity"] == 2

    snapshot = {
        "records": [
            {
                "relpath": "vllm-ascend",
                "source_path": str(repo_dir),
                "changed_paths": ["vllm_ascend/models/qwen3_5.py"],
            }
        ]
    }
    violations = check_snapshot_op_contracts(snapshot, catalog=loaded)
    assert len(violations) == 1
    assert violations[0]["file"] == "vllm-ascend/vllm_ascend/models/qwen3_5.py"
    formatted = format_op_violations(violations)
    assert "Ascend operator contract preflight failed" in formatted
    assert "torch_npu.npu_rms_norm" in formatted

    from mindie_coordinator.backend import RemoteBackend
    from mindie_coordinator.managed_execution import ExecutionRequestError

    class FailingShellIfCalled:
        def run(self, *_args, **_kwargs):
            raise AssertionError("remote shell should not be called when static op preflight fails")

    backend = RemoteBackend(shell=FailingShellIfCalled())
    with pytest.raises(ExecutionRequestError, match="Ascend operator contract preflight failed"):
        backend.preflight(
            {"endpoint": {"cwd": "/vllm-workspace"}, "source_snapshot": snapshot},
            "python3 -c 'print(1)'",
            {},
        )

