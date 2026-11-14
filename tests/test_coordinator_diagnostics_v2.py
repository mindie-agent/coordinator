"""Tier 1 tests for `mindie-coordinator` internal fault and inventory `contract_violation` diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindie_coordinator import ops
from mindie_coordinator.machine_directory import MachineDirectory
from mindie_diagnostics.ingestion import ingest
from mindie_diagnostics.integration import configure_reporting
from mindie_diagnostics.outbox import Outbox


def _isolate_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    base = tmp_path.resolve()
    diag_root = base / "diag"
    cfg_file = base / "reporting.json"
    monkeypatch.setenv("HOME", str(base / "home"))
    monkeypatch.setenv("MINDIE_CONFIG_HOME", str(base / "config"))
    monkeypatch.setenv("MINDIE_DATA_HOME", str(base / "data"))
    monkeypatch.setenv("MINDIE_COORDINATOR_STATE_DIR", str(base / "coord"))
    monkeypatch.setenv("MINDIE_DIAGNOSTICS_ROOT", str(diag_root))
    monkeypatch.setenv("MINDIE_DIAGNOSTICS_CONFIG", str(cfg_file))
    (base / "home").mkdir(parents=True, exist_ok=True)
    diag_root.mkdir(parents=True, exist_ok=True)
    configure_reporting(True, roots=[str(diag_root)], config=cfg_file)
    return diag_root, cfg_file


def test_corrupt_machines_inventory_records_contract_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    diag_root, cfg_file = _isolate_env(tmp_path, monkeypatch)
    inv_path = tmp_path.resolve() / "coord" / "machines.json"
    inv_path.parent.mkdir(parents=True, exist_ok=True)
    inv_path.write_text('{"machines": "not-a-list"}', encoding="utf-8")

    with pytest.raises(ValueError, match="not a machine inventory"):
        MachineDirectory(path=inv_path).load()

    queue = Outbox(tmp_path.resolve() / "queue.db")
    assert ingest(diag_root, queue, config=cfg_file)["enqueued"] == 1
    ev = queue.claim()["payload"]["events"][-1]
    assert ev["component"] == "mindie-coordinator"
    assert ev["attributes"]["category"] == "contract_violation"
    assert ev["attributes"]["contract_name"] == "machines_inventory_json"
    assert ev["attributes"]["reason_code"] == "invalid_machines_schema"
    assert any("machine_directory" in f["module"] for f in ev["attributes"]["stack_frames"])


def test_mindie_call_records_internal_bug_but_ignores_caller_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    diag_root, cfg_file = _isolate_env(tmp_path, monkeypatch)
    out = ops.mindie_call("mindie.hosts", {"unsupported_arg": 1})
    assert out["result"]["outcome"] == "blocked"
    queue = Outbox(tmp_path.resolve() / "queue.db")
    assert ingest(diag_root, queue, config=cfg_file)["enqueued"] == 0

    import mindie_coordinator.env_manager as env_manager
    monkeypatch.setattr(env_manager, "handle_hosts", lambda *a, **k: (_ for _ in ()).throw(KeyError("secret_internal_key")))
    out2 = ops.mindie_call("mindie.hosts", {})
    assert out2["result"]["outcome"] == "blocked"
    assert ingest(diag_root, queue, config=cfg_file)["enqueued"] == 1
    ev = queue.claim()["payload"]["events"][-1]
    assert ev["component"] == "mindie-coordinator"
    assert ev["operation"] == "mindie.hosts"
    assert ev["attributes"]["category"] == "internal_exception"
    assert "secret_internal_key" not in json.dumps(ev)
    assert all(not f["module"].startswith("mindie_coordinator.mindie_coordinator.") for f in ev["attributes"]["stack_frames"])


def test_cli_non_dict_json_rejected_as_caller_and_crash_emits_incident_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mindie_coordinator import cli

    diag_root, cfg_file = _isolate_env(tmp_path, monkeypatch)
    queue = Outbox(tmp_path.resolve() / "queue.db")

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["hosts", "--json", "[]"])
    assert excinfo.value.code == 2
    capsys.readouterr()
    assert ingest(diag_root, queue, config=cfg_file)["enqueued"] == 0

    monkeypatch.setattr(ops, "mindie_call", lambda *a, **k: (_ for _ in ()).throw(KeyError("cli_secret")))
    with pytest.raises(KeyError):
        cli.main(["hosts"])
    err = capsys.readouterr().err
    assert "mindie-diagnostic-incident: " in err
    assert ingest(diag_root, queue, config=cfg_file)["enqueued"] == 1

    from mindie_coordinator.diagnostics import is_internal_coordinator_exception
    cur: BaseException = PermissionError("root")
    for idx in range(40):
        nxt = RuntimeError(f"w{idx}")
        nxt.__cause__ = cur
        cur = nxt
    assert is_internal_coordinator_exception(cur) is False


def test_round4_f5_observed_tool_and_cli_boundary_emit_single_shared_incident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """F5: cli.main(['hosts']) failing inside @observed_tool runtime_status() records ONE incident."""
    from mindie_coordinator import cli, env_manager, ops

    diag_root, cfg_file = _isolate_env(tmp_path, monkeypatch)
    queue = Outbox(tmp_path.resolve() / "f5_queue.db")

    monkeypatch.setattr(env_manager, "handle_hosts", lambda *_a, **_k: {"outcome": "success", "hosts": []})
    monkeypatch.setattr(ops, "runtime_status", lambda *_a, **_k: (_ for _ in ()).throw(KeyError("f5_secret")))

    with pytest.raises(KeyError) as excinfo:
        cli.main(["hosts"])
    stderr = capsys.readouterr().err
    assert "mindie-diagnostic-incident: " in stderr
    marker_id = stderr.strip().split("mindie-diagnostic-incident: ")[-1].splitlines()[0].strip()
    assert len(marker_id) == 32
    assert getattr(excinfo.value, "mindie_diagnostic", {}).get("incident_id") == marker_id

    assert ingest(diag_root, queue, config=cfg_file)["enqueued"] == 1
    ev = queue.claim()["payload"]["events"][-1]
    assert ev["component"] == "mindie-coordinator"
    assert ev["attributes"]["error_type"] == "KeyError"
    assert queue.claim() is None

    # Causal-chain wrapper (`raise ... from inner`) must reuse the inner incident instead of recording twice!
    from mindie_coordinator.diagnostics import record_coordinator_failure

    inner = KeyError("inner_key")
    r_in = record_coordinator_failure("mindie.hosts", exception=inner)
    assert r_in is not None and len(r_in["incident_id"]) == 32
    try:
        raise AssertionError("outer_wrap") from inner
    except AssertionError as outer:
        r_out = record_coordinator_failure("mindie.hosts", exception=outer)
    assert r_out["incident_id"] == r_in["incident_id"]
    assert ingest(diag_root, queue, config=cfg_file)["enqueued"] == 1
    assert queue.claim() is not None and queue.claim() is None
