"""Live SSH integration tests against real Ascend NPU hosts (A2 and A3).

Run explicitly with:
    pytest -m live_npu -v
"""

from __future__ import annotations

import subprocess
import tempfile
import uuid
from pathlib import Path

import pytest

from mindie_coordinator.env_manager import handle_lease, handle_probe
from mindie_coordinator.fleet_probe import find_fleet_capacity, probe_machine_record
from mindie_coordinator.machine_directory import MachineDirectory


def _cleanup_remote_dir(host: str, remote_dir: str) -> None:
    if not remote_dir.startswith("/tmp/mindie-npu-test-"):
        raise ValueError(f"Refusing to remove non-test remote dir: {remote_dir}")
    subprocess.run(
        [
            "ssh",
            "-n",
            "-o",
            "ControlMaster=no",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            f"root@{host}",
            f"rm -rf {remote_dir}",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


@pytest.mark.live_npu
def test_live_fleet_probe_and_capacity_across_a2_and_a3():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = MachineDirectory(path=root / "machines.json")
        store.upsert_machine({"alias": "a2-173", "host": {"ip": "173.125.1.2", "port": 22, "user": "root"}})
        store.upsert_machine({"alias": "a3-165-idle", "host": {"ip": "192.168.13.165", "port": 22, "user": "root"}})
        store.upsert_machine({"alias": "a3-154-busy", "host": {"ip": "192.168.13.154", "port": 22, "user": "root"}})

        snap_a2 = probe_machine_record(store.record("a2-173"), mode="live", state_dir=root)
        assert snap_a2["online"] is True, snap_a2.get("error")
        assert snap_a2["total_npus"] == 8
        assert all(n["die_count"] == 1 for n in snap_a2["npus"])
        assert [n["phy_ids"] for n in snap_a2["npus"]] == [[i] for i in range(8)]

        snap_a3_idle = probe_machine_record(store.record("a3-165-idle"), mode="live", state_dir=root)
        assert snap_a3_idle["online"] is True, snap_a3_idle.get("error")
        assert snap_a3_idle["total_npus"] == 8
        assert all(n["die_count"] == 2 for n in snap_a3_idle["npus"])
        assert [n["phy_ids"] for n in snap_a3_idle["npus"]] == [[i * 2, i * 2 + 1] for i in range(8)]

        snap_a3_busy = probe_machine_record(store.record("a3-154-busy"), mode="live", state_dir=root)
        assert snap_a3_busy["online"] is True, snap_a3_busy.get("error")
        assert snap_a3_busy["total_npus"] == 8
        assert all(n["die_count"] == 2 for n in snap_a3_busy["npus"])

        cap = find_fleet_capacity(count=2, directory=store, state_dir=root, mode="cache")
        assert cap["matched_count"] >= 2
        by_alias = {c["alias"]: c for c in cap["candidates"]}
        if "a3-165-idle" in by_alias:
            a3_cand = by_alias["a3-165-idle"]
            assert len(a3_cand["recommended_npu_ids"]) == 2
            assert len(a3_cand["recommended_devices"]) == 4
            assert a3_cand["recommended_devices"] == a3_cand["recommended_phy_ids"]


@pytest.mark.live_npu
@pytest.mark.parametrize(
    ("alias", "host", "npu_count"),
    [
        ("a2-173", "173.125.1.2", 1),
        ("a3-165-idle", "192.168.13.165", 2),
    ],
)
def test_live_lease_lifecycle_in_isolated_remote_state_dir(alias: str, host: str, npu_count: int):
    remote_state_dir = f"/tmp/mindie-npu-test-coord-{uuid.uuid4().hex[:10]}"
    lease_id = f"test-lease-{uuid.uuid4().hex[:8]}"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = MachineDirectory(path=root / "machines.json")
            store.upsert_machine({"alias": alias, "host": {"ip": host, "port": 22, "user": "root"}})

            acquired = handle_lease(
                {
                    "action": "acquire",
                    "alias": alias,
                    "lease_id": lease_id,
                    "npu_count": npu_count,
                    "remote_state_dir": remote_state_dir,
                },
                directory=store,
                state_dir=root,
            )
            assert acquired["state"] in ("granted", "active"), acquired
            assert len(acquired["devices"]) == npu_count
            assert acquired["fence_token"] is not None

            hb = handle_lease(
                {
                    "action": "heartbeat",
                    "alias": alias,
                    "lease_id": lease_id,
                    "remote_state_dir": remote_state_dir,
                },
                directory=store,
                state_dir=root,
            )
            assert hb["state"] in ("granted", "active"), hb

            status = handle_lease(
                {
                    "action": "status",
                    "alias": alias,
                    "lease_id": lease_id,
                    "remote_state_dir": remote_state_dir,
                },
                directory=store,
                state_dir=root,
            )
            assert status["count"] == 1
            assert status["leases"][0]["lease_id"] == lease_id

            released = handle_lease(
                {
                    "action": "release",
                    "alias": alias,
                    "lease_id": lease_id,
                    "remote_state_dir": remote_state_dir,
                },
                directory=store,
                state_dir=root,
            )
            assert released["state"] == "released"
    finally:
        _cleanup_remote_dir(host, remote_state_dir)


@pytest.mark.live_npu
def test_live_linux_proc_process_guard_busy_on_a3_host():
    from mindie_coordinator.host import mindie_npu_coordination

    coord_src = Path(mindie_npu_coordination.__file__).read_text(encoding="utf-8")
    coord_src = coord_src.split('if __name__ == "__main__":')[0]
    py_code = f"""{coord_src}

boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
marker = os.urandom(16).hex()
guard = {{"marker": marker, "boot_id": boot_id}}
env = dict(os.environ)
env[JOB_TOKEN_ENV] = marker
proc = subprocess.Popen(["python3", "-c", "import time; time.sleep(30)"], env=env)
try:
    time.sleep(0.2)
    busy_while_alive = process_guard_busy(guard)
finally:
    proc.terminate()
    proc.wait(timeout=5)
busy_after_exit = process_guard_busy(guard)
print(json.dumps({{"busy_while_alive": busy_while_alive, "busy_after_exit": busy_after_exit}}))
"""
    res = subprocess.run(
        [
            "ssh",
            "-o",
            "ControlMaster=no",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            "root@192.168.13.165",
            "python3 -",
        ],
        input=py_code,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    import json

    payload = json.loads(res.stdout.strip().splitlines()[-1])
    assert payload["busy_while_alive"] is True
    assert payload["busy_after_exit"] is False


