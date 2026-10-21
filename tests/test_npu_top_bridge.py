"""Real loopback HTTP behavior and observation-only ranking for npu-top."""
from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

from mindie_coordinator import fleet_probe
from mindie_coordinator.npu_top_bridge import (
    NpuTopObservationClient,
    observation_host_order,
    observed_process_interference,
    rank_capacity_candidates,
)
from mindie_coordinator.placement import select_runtimes


def test_real_a2_process_ownership_distinguishes_own_foreign_and_pid_races():
    from mindie_coordinator.host.mindie_npu_coordination import parse_npu_smi_info

    capture = json.loads((Path(__file__).parent / "fixtures/a2-stage1-owned-process.json").read_text())
    occupancy = parse_npu_smi_info(capture["info"])
    identities = capture["process_ownership"]
    assert observed_process_interference(occupancy, 0, identities, marker_known=True) is False
    foreign = {pid: {**identity, "owned": False} for pid, identity in identities.items()}
    assert observed_process_interference(occupancy, 0, foreign, marker_known=True) is True
    raced = {pid: {**identity, "identity_stable": False} for pid, identity in identities.items()}
    assert observed_process_interference(occupancy, 0, raced, marker_known=True) is None
    assert observed_process_interference(occupancy, 0, {}, marker_known=True) is None
    assert observed_process_interference(occupancy, 0, identities, marker_known=False) is None


@pytest.fixture
def local_top():
    state = {"requests": [], "responses": {}, "statuses": {}}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            state["requests"].append(parsed)
            payload = state["responses"].get(parsed.path, {"error": "unknown path"})
            status = state["statuses"].get(parsed.path, 200)
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


CAPACITY = {
    "source": "cache",
    "observation": {"kind": "observed_availability", "allocation_authority": False,
                    "observed_at": 100, "age_seconds": 4},
    "candidates": [
        {"host": "10.0.0.1", "name": "a", "idle_npu_count": 4, "age_seconds": 3,
         "ThermalHealth": "healthy", "MaxWeightMountAvailGb": 120.0,
         "IdlePackagePairs": [[0, 1]]},
        {"host": "10.0.0.2", "name": "b", "idle_npu_count": 8, "age_seconds": 1,
         "ThermalHealth": "throttled", "MaxWeightMountAvailGb": 120.0,
         "IdlePackagePairs": []},
        {"host": "10.0.0.3", "name": "c", "idle_npu_count": 2, "age_seconds": 2,
         "MaxWeightMountAvailGb": 50.0, "IdlePackagePairs": []},
    ],
}


def test_capacity_sends_c2_kwargs_and_ranks_eligible_hosts(local_top):
    url, state = local_top
    state["responses"]["/api/agent/capacity"] = CAPACITY
    client = NpuTopObservationClient(url, timeout=1.0)
    result = client.capacity(min_idle_npus=2, max_age_seconds=240, tags=["A3"],
                             min_weight_mount_avail_gb=20.0, prefer_same_package=True,
                             exclude_throttled=True)
    query = parse_qs(state["requests"][0].query)
    assert query["min_weight_mount_avail_gb"] == ["20.0"]
    assert query["prefer_same_package"] == ["1"]
    assert query["exclude_throttled"] == ["1"]
    assert query["min_idle_npus"] == ["2"]
    assert query["tags"] == ["A3"]
    assert result["allocation_authority"] is False
    assert result["status"] == "partial"
    eligible = [candidate["host"] for candidate in result["eligible_candidates"]]
    assert eligible == ["10.0.0.1", "10.0.0.3"]
    throttled = next(row for row in result["candidates"] if row["host"] == "10.0.0.2")
    assert throttled["npu_top_eligible"] is False
    assert "ThermalHealth=throttled" in throttled["npu_top_reasons"]
    assert any("ThermalHealth not reported" in row for row in result["measurement_limitations"])


def test_monitor_unavailable_is_visible_and_not_an_empty_success(local_top):
    url, _state = local_top
    # Reserve and close a port so the connection is refused immediately.
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    client = NpuTopObservationClient(f"http://127.0.0.1:{port}", timeout=0.2)
    result = client.capacity()
    assert result["status"] == "unavailable"
    assert result["allocation_authority"] is False
    assert result["candidates"] == []
    assert result["measurement_limitations"]
    assert "unavailable" in result["measurement_limitations"][0]


def test_telemetry_window_keeps_incomplete_coverage_and_missing_registers(local_top):
    url, state = local_top
    state["responses"]["/api/agent/telemetry-window"] = {
        "coverage_ratio": 0.5,
        "throttling_detected": None,
        "aicore_p50": 71.0,
        "aicore_p95": 93.0,
        "hbm_peak_mb": 4096,
        "temp_max_c": 68,
        "measurement_limitations": ["power register not exposed"],
        "devices": [{"id": 0, "aicore_p50": 71.0}],
    }
    client = NpuTopObservationClient(url, timeout=1.0)
    evidence = client.telemetry_window("host-a", [0, 1], 100, 200)
    query = parse_qs(state["requests"][0].query)
    assert query["devices"] == ["0,1"]
    assert evidence["status"] == "partial"
    assert evidence["coverage_ratio"] == 0.5
    assert evidence["devices"] == [0, 1]
    assert evidence["throttling_detected"] is None
    joined = " ".join(evidence["measurement_limitations"])
    assert "coverage is incomplete" in joined
    assert "throttling_detected not reported" in joined
    assert "power register not exposed" in joined


def test_missing_telemetry_api_is_reported_as_unavailable(local_top):
    url, state = local_top
    state["statuses"]["/api/agent/telemetry-window"] = 404
    state["responses"]["/api/agent/telemetry-window"] = {"error": "not found"}
    client = NpuTopObservationClient(url, timeout=1.0)
    evidence = client.telemetry_window("host-a", [0], 100, 200)
    assert evidence["status"] == "unavailable"
    assert any("not installed (HTTP 404)" in row for row in evidence["measurement_limitations"])


def test_completed_run_projects_remote_clock_to_collector_window(local_top, tmp_path, monkeypatch):
    from mindie_coordinator import backend as backend_module
    from mindie_coordinator.machine_directory import MachineDirectory

    url, state = local_top
    monkeypatch.setenv("MINDIE_TOP_URL", url)
    # Same response binding as npu_top.telemetry_window.build_window, with the
    # currently unmeasured registers still absent rather than claimed healthy.
    state["responses"]["/api/agent/telemetry-window"] = {
        "server": {"host": "host-a"},
        "window": {"start_ts": 260.125, "end_ts": 270.5, "requested_devices": [0],
                   "start_ts_ns": 260125000000, "end_ts_ns": 270500000000},
        "coverage_ratio": 1.0, "throttling_detected": None,
        "resource_interference_detected": None,
        "measurement_limitations": ["Resource interference was not measured explicitly"],
    }
    clock_stdout = tmp_path / "clock.txt"
    clock_stdout.write_text("30.0\n")

    class ClockShell:
        def run(self, target, command, *, timeout_ms):
            assert target["host"] == "host-a" and timeout_ms == 2000
            assert "print(time.time())" in command
            return {"outcome": "success", "refs": {"stdout": str(clock_stdout)}}

    local_clock = iter([280.0, 280.125])
    monkeypatch.setattr(backend_module, "time", SimpleNamespace(time=lambda: next(local_clock)))
    backend = backend_module.RemoteBackend(shell=ClockShell(),
        machines=MachineDirectory(path=tmp_path / "machines.json"))
    runtime = {"endpoint": {"host": "host-a"}, "host_endpoint": {"host": "host-a"},
               "python": "/usr/bin/python3"}
    job = {"id": "inner", "execution_id": "public",
           "remote": {"result": {"started_at": 10.125, "finished_at": 20.375}}}
    evidence = backend.run_evidence(runtime, job, {"task": {"granted_devices": [0]}})["hardware_telemetry"]
    query = parse_qs(state["requests"][0].query)
    assert query["start_ts"] == ["260.125"] and query["end_ts"] == ["270.5"]
    assert evidence["start_ts"] == 260.125 and evidence["end_ts"] == 270.5
    assert evidence["run_id"] == "public" and evidence["status"] == "partial"
    assert evidence["command_window"] == {"started_at": 10.125, "finished_at": 20.375, "basis": "remote_supervisor"}
    assert evidence["clock_alignment"]["offset_min_seconds"] == 250.0
    assert any("offset stability" in message for message in evidence["measurement_limitations"])
    assert evidence["resource_interference_detected"] is None
    from mindie_coordinator.run_manifest import add_evidence, load_manifest, new_manifest, write_manifest
    from jsonschema import Draft202012Validator

    manifest = new_manifest(run_type="performance", run_id="public",
                            workspace_root=Path(__file__).parents[1])
    bound = add_evidence(manifest, hardware_telemetry=evidence)
    destination = tmp_path / "run-manifest.json"
    write_manifest(destination, bound)
    restored = load_manifest(destination)
    schema = json.loads((Path(__file__).parents[1] /
                         "mindie_coordinator/schemas/run-manifest-v1.schema.json").read_text())
    Draft202012Validator(schema).validate(restored)
    assert restored["hardware_telemetry"] == evidence
    assert restored["hardware_telemetry"]["start_ts"] == 260.125
    assert restored["hardware_telemetry"]["start_ts_ns"] == 260125000000


@pytest.mark.parametrize("changed", ["host", "devices", "window"])
def test_telemetry_from_another_host_device_or_window_is_unavailable(local_top, changed):
    url, state = local_top
    payload = {"server": {"host": "host-a"},
               "window": {"start_ts": 100, "end_ts": 200, "requested_devices": [0]},
               "coverage_ratio": 1.0, "throttling_detected": None}
    if changed == "host":
        payload["server"]["host"] = "host-b"
    elif changed == "devices":
        payload["window"]["requested_devices"] = [1]
    else:
        payload["window"]["end_ts"] = 201
    state["responses"]["/api/agent/telemetry-window"] = payload
    evidence = NpuTopObservationClient(url).telemetry_window("host-a", [0], 100, 200)
    assert evidence["status"] == "unavailable"
    assert any("do not match" in message or "does not match" in message
               for message in evidence["measurement_limitations"])


def test_local_client_bypasses_environment_proxies(local_top, monkeypatch):
    url, state = local_top
    state["responses"]["/api/agent/capacity"] = CAPACITY
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    client = NpuTopObservationClient(url, timeout=1.0)
    result = client.capacity(exclude_throttled=False)
    assert result["status"] in {"observed", "partial"}
    assert state["requests"]


def test_remote_url_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("MINDIE_TOP_ALLOW_REMOTE", raising=False)
    with pytest.raises(ValueError, match="non-loopback"):
        NpuTopObservationClient("http://10.0.0.5:8788")


def _snapshot(alias, host, free_ids):
    return {
        "alias": alias, "host": host, "port": 22, "user": "root", "online": True,
        "total_npus": 8, "free_npus": len(free_ids), "busy_npus": 8 - len(free_ids),
        "free_npu_ids": list(free_ids), "busy_npu_ids": [],
        "free_phy_ids": list(free_ids), "busy_phy_ids": [],
        "chip_summary": "Ascend910B", "npus": [], "containers": [], "mounts": [],
        "bound_container": None,
    }


def _observation(order):
    return {
        "source": "npu-top", "status": "observed", "allocation_authority": False,
        "candidates": [{"host": host, "npu_top_rank": rank, "npu_top_eligible": True}
                       for rank, host in enumerate(order)],
        "measurement_limitations": [], "observation": {"allocation_authority": False},
    }


def test_fleet_capacity_orders_probes_but_keeps_host_queue_authority(tmp_path, monkeypatch):
    from mindie_coordinator.machine_directory import MachineDirectory

    store = MachineDirectory(path=tmp_path / "machines.json")
    store.upsert_machine({"alias": "a", "host": {"ip": "10.0.0.1", "port": 22}})
    store.upsert_machine({"alias": "b", "host": {"ip": "10.0.0.2", "port": 22}})
    snapshots = {"a": _snapshot("a", "10.0.0.1", [0, 1]), "b": _snapshot("b", "10.0.0.2", [2, 3])}
    monkeypatch.setattr(fleet_probe, "probe_machine_record",
                        lambda record, **kwargs: snapshots[record["alias"]])

    class Client:
        def __init__(self):
            self.calls = []

        def capacity(self, **kwargs):
            self.calls.append(kwargs)
            return _observation(["10.0.0.2", "10.0.0.1"])

    client = Client()
    result = fleet_probe.find_fleet_capacity(count=2, directory=store, state_dir=tmp_path,
                                             npu_top_client=client)
    assert [row["alias"] for row in result["candidates"]] == ["b", "a"]
    assert result["candidates"][0]["npu_top_rank"] == 0
    assert result["npu_top_observation"]["allocation_authority"] is False
    assert client.calls and client.calls[0]["min_weight_mount_avail_gb"] == 0.0

    explicit = fleet_probe.find_fleet_capacity(count=2, host="a", directory=store,
                                               state_dir=tmp_path, npu_top_client=client)
    assert [row["alias"] for row in explicit["candidates"]] == ["a"]
    assert explicit["npu_top_observation"]["status"] == "skipped"
    assert len(client.calls) == 1


def test_placement_observation_reorders_only_matching_roots():
    catalog = [
        {"runtime_id": "ra", "state": "ready", "user": "u", "host": "10.0.0.1", "profile": {}},
        {"runtime_id": "rb", "state": "ready", "user": "u", "host": "10.0.0.2", "profile": {}},
    ]
    roles = [{"name": "default", "npu_count": 0}, {"name": "second", "npu_count": 0}]
    observed = {"candidates": [{"host": "10.0.0.2", "npu_top_rank": 0},
                               {"host": "10.0.0.1", "npu_top_rank": 1}]}
    chosen = select_runtimes(catalog, user="u", roles=roles,
                             observation=observed)
    assert chosen["runtime_ids"] == ["rb", "ra"]
    explicit = select_runtimes(catalog, user="u",
                               roles=[{"name": "default", "npu_count": 0, "host": "10.0.0.1"}],
                               observation=observed)
    assert explicit["runtime_ids"] == ["ra"]


def test_rank_helper_marks_unknown_mount_as_ineligible_not_zero_capacity():
    payload = {"candidates": [{"host": "unknown-mount", "idle_npu_count": 4,
                               "ThermalHealth": "healthy"}]}
    ranked = rank_capacity_candidates(payload, min_weight_mount_avail_gb=10.0)
    row = ranked["candidates"][0]
    assert row["npu_top_eligible"] is False
    assert any("unproven" in reason for reason in row["npu_top_reasons"])
    assert observation_host_order(ranked)["unknown-mount"] == 0


def test_fleet_probe_distinguishes_disk_io_wait_d_state_from_npu_hang():
    raw_probe = "\n".join([
        "===SECTION:NPU_SMI===",
        "| NPU     Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|",
        "| 0       910B4               | OK            | 92.1        42                0    / 0             |",
        "| 0       0                   | 0000:C1:00.0  | 0           1024 / 65536      18432/ 65536         |",
        "| NPU     Chip                | Process id    | Process name             | Process memory(MB) |",
        "| 0       0                   | 44120         | VLLM::EngineCore         | 18432              |",
        "===SECTION:CONTAINERS===",
        "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2|vllm-dev|ascend-vllm:latest|Up 2 hours|0.0.0.0:46001->22/tcp",
        "===SECTION:PID_MAP===",
        "44120|root|0::/docker/a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2|D|lock_page_killable",
    ])
    snap = fleet_probe.parse_fleet_probe_output(raw_probe, alias="a2-node", host="173.125.1.2")
    card0 = snap["npus"][0]
    assert card0["aicore_pct"] == 0.0
    assert card0["disk_io_wait_pids"] == [44120]
    proc = card0["processes"][0]
    assert proc["proc_state"] == "D"
    assert proc["wchan"] == "lock_page_killable"
    assert proc["disk_io_wait"] is True
    assert proc["container_name"] == "vllm-dev"


def test_checkpoint_preflight_detects_unquantized_experts_truncation_and_unaligned_head_dim(tmp_path):
    import struct
    from mindie_coordinator.checkpoint_preflight import inspect_safetensors_checkpoint

    (tmp_path / "config.json").write_text(json.dumps({
        "head_dim": 35,
        "quantization_config": {"quant_method": "compressed-tensors", "format": "w8a8"},
    }), encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "model.layers.0.mlp.experts.0.down_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
        }
    }), encoding="utf-8")
    header = {
        "model.layers.0.mlp.experts.0.down_proj.weight": {
            "dtype": "BF16",
            "shape": [1024, 1024],
            "data_offsets": [0, 2097152],
        }
    }
    raw_hdr = json.dumps(header).encode("utf-8")
    # Write truncated payload (only 64 bytes instead of 2097152)
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(
        struct.pack("<Q", len(raw_hdr)) + raw_hdr + (b"\x00" * 64)
    )
    report = inspect_safetensors_checkpoint(tmp_path, tp_size=2, ep_size=4)
    assert report["valid"] is False
    assert report["head_dim"] == 35
    assert report["head_dim_32b_aligned"] is False
    assert set(report["issue_codes"]) == {
        "unaligned_head_dim_32b",
        "missing_safetensors_shards",
        "truncated_or_corrupt_shard",
        "unquantized_routed_experts_in_quant_checkpoint",
    }
