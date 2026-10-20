"""Cross-component concurrency contracts for the shared machine inventory.

These tests use the real coordinator ``MachineDirectory`` and the real
``npu-top`` inventory bridge. Both writers must serialize on the adjacent
``machines.json.lock`` file and merge against the latest on-disk document.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from mindie_coordinator.machine_directory import MachineDirectory

COORDINATOR_ROOT = Path(__file__).resolve().parents[1]
NPU_TOP_ROOT = COORDINATOR_ROOT.parent / "npu-top"


def _write_inventory(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_inventory(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _wait_for(path: Path, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path}")


def _terminate(*processes: subprocess.Popen) -> None:
    for process in processes:
        if process.poll() is None:
            process.kill()
    for process in processes:
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive cleanup
            pass


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def test_upsert_preserves_omitted_c1_fields_and_replace_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "machines.json"
    _write_inventory(path, {
        "version": 7,
        "default_alias": "a3-165",
        "machines": [{
            "alias": "a3-165",
            "host": {
                "ip": "192.168.13.165", "port": 22, "user": "root",
                "identity_file": "/home/operator/.ssh/id_ed25519",
            },
            "container": {"name": "mindie-dev", "ssh_port": 32222, "user": "root"},
            "ssh": {"extra": "coordinator-owned"},
            "tags": ["coordinator", "A3"],
            "hardware": {
                "soc": "Ascend910C", "dies_per_package": 2, "npu_count": 16,
                "vendor": {"rack": "r1"},
            },
            "mounts": [{
                "target": "/data/models", "weight_candidate": False,
                "note": "keep", "nfs": {"server": "nfs-a"},
            }],
            "arbitrary_c1": {"preserve": True},
        }],
        "unrelated": {"preserve": True},
    })

    store = MachineDirectory(path=path)
    store.upsert_machine({"alias": "a3-165", "host": {"ip": "192.168.13.165", "port": 2222}})

    document = _read_inventory(path)
    row = document["machines"][0]
    assert row["host"] == {
        "ip": "192.168.13.165", "port": 2222, "user": "root",
        "identity_file": "/home/operator/.ssh/id_ed25519",
    }
    assert row["hardware"] == {
        "soc": "Ascend910C", "dies_per_package": 2, "npu_count": 16,
        "vendor": {"rack": "r1"},
    }
    assert row["tags"] == ["coordinator", "A3"]
    assert row["mounts"][0]["note"] == "keep"
    assert row["mounts"][0]["nfs"] == {"server": "nfs-a"}
    assert row["arbitrary_c1"] == {"preserve": True}
    assert row["ssh"] == {"extra": "coordinator-owned"}
    assert document["unrelated"] == {"preserve": True}
    assert document["default_alias"] == "a3-165"

    # A partial nested update preserves the nested keys it omits.
    store.upsert_machine({
        "alias": "a3-165",
        "host": {"ip": "192.168.13.165", "port": 2222},
        "hardware": {"soc": "Ascend910C"},
        "tags": ["A3", "910C"],
        "mounts": [{"target": "/data/models", "weight_candidate": True}],
    })
    row = _read_inventory(path)["machines"][0]
    assert row["hardware"]["dies_per_package"] == 2
    assert row["hardware"]["vendor"] == {"rack": "r1"}
    assert row["mounts"][0] == {
        "target": "/data/models", "weight_candidate": True,
        "note": "keep", "nfs": {"server": "nfs-a"},
    }
    assert row["tags"] == ["A3", "910C"]

    # replace() is the explicit full-document operation: no merge.
    store.replace({"machines": [{"alias": "a3-165", "host": {"ip": "192.168.13.165"}}]})
    assert _read_inventory(path) == {
        "machines": [{"alias": "a3-165", "host": {"ip": "192.168.13.165"}}],
    }


def test_path_backed_reads_do_not_use_a_stale_document_cache(tmp_path: Path) -> None:
    path = tmp_path / "machines.json"
    store = MachineDirectory(path=path)
    store.upsert_machine({"alias": "alpha", "host": {"ip": "10.0.0.1", "port": 22}})
    assert store._document is not None  # the write populated the legacy cache

    document = _read_inventory(path)
    document["machines"].append({"alias": "observed", "host": {"ip": "10.0.0.2", "port": 22}})
    document["default_alias"] = "observed"
    _write_inventory(path, document)

    assert {row["alias"] for row in store.machines()} == {"alpha", "observed"}
    assert store.record("observed")["host"]["ip"] == "10.0.0.2"
    assert store.catalog()["inventory_path"] == str(path.resolve())


@pytest.mark.skipif(not NPU_TOP_ROOT.is_dir(), reason="sibling npu-top checkout is required")
def test_coordinator_and_npu_top_writers_interleave_without_losing_updates(tmp_path: Path) -> None:
    path = tmp_path / "machines.json"
    _write_inventory(path, {
        "version": 7,
        "default_alias": "a3-165",
        "machines": [{
            "alias": "a3-165",
            "host": {"ip": "192.168.13.165", "port": 22, "user": "root"},
            "tags": ["coordinator"],
            "hardware": {"chip_summary": "910C", "total_npus": 16},
            "mounts": [{"target": "/data/models", "weight_candidate": False, "note": "keep"}],
        }],
    })

    ready = tmp_path / "coordinator-read.marker"
    go = tmp_path / "coordinator-go.marker"
    attempted = tmp_path / "npu-top-attempted.marker"
    done = tmp_path / "npu-top-done.marker"

    coordinator_script = textwrap.dedent(
        f"""
        import sys
        import time
        from pathlib import Path

        sys.path.insert(0, {str(COORDINATOR_ROOT)!r})
        from mindie_coordinator import machine_directory as md

        path = Path({str(path)!r})
        ready = Path({str(ready)!r})
        go = Path({str(go)!r})
        original = md.MachineDirectory._replace_unlocked

        def paused(self, target, document):
            # This runs under the inventory lock, after the read/merge and
            # before the atomic replacement.
            ready.write_text("read", encoding="utf-8")
            deadline = time.monotonic() + 10.0
            while not go.exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("timed out waiting for the npu-top writer")
                time.sleep(0.01)
            return original(self, target, document)

        md.MachineDirectory._replace_unlocked = paused
        md.MachineDirectory(path=path).upsert_machine({{
            "alias": "a3-165",
            "host": {{"ip": "192.168.13.165", "port": 2222, "user": "operator"}},
        }})
        """
    )

    servers = [
        {
            "id": "s1", "name": "a3-165", "host": "192.168.13.165",
            "port": 22, "username": "root", "tags": ["A3"],
        },
        {
            "id": "s2", "name": "observed-new", "host": "10.0.0.77",
            "port": 22, "username": "root", "tags": ["A3"],
        },
    ]
    snapshots = {
        "s1": {
            "status": "online", "collected_at": 990,
            "summary": {"npu_count": 8, "busy_npu_count": 0},
            "devices": [{"npu_id": index, "name": "Ascend910"} for index in range(8)],
            "mounts": [{"target": "/data/weights", "fstype": "nfs4"}],
            "disks": [{"mount": "/data/weights", "available_bytes": 100 * 1024 ** 3}],
        },
        "s2": {
            "status": "online", "collected_at": 990,
            "summary": {"npu_count": 8, "busy_npu_count": 0},
            "devices": [{"npu_id": index, "name": "Ascend910"} for index in range(8)],
        },
    }
    npu_top_script = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path

        sys.path.insert(0, {str(NPU_TOP_ROOT)!r})
        from npu_top.inventory_bridge import sync_to_coordinator

        attempted = Path({str(attempted)!r})
        done = Path({str(done)!r})
        servers = {servers!r}
        snapshots = {snapshots!r}
        attempted.write_text("attempting", encoding="utf-8")
        sync_to_coordinator(
            Path({str(path)!r}), servers, {{server["id"]: snapshots[server["id"]] for server in servers}}
        )
        done.write_text("done", encoding="utf-8")
        """
    )

    coordinator = subprocess.Popen(
        [sys.executable, "-c", coordinator_script],
        cwd=COORDINATOR_ROOT, env=_child_env(), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    npu_top = None
    try:
        _wait_for(ready)
        npu_top = subprocess.Popen(
            [sys.executable, "-c", npu_top_script],
            cwd=NPU_TOP_ROOT, env=_child_env(), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        _wait_for(attempted)
        time.sleep(0.5)
        assert npu_top.poll() is None, (
            "npu-top sync completed while the coordinator held the inventory lock; "
            "the adjacent lock protocol is not shared"
        )
        assert not done.exists()

        go.write_text("go", encoding="utf-8")
        coordinator_stdout, coordinator_stderr = coordinator.communicate(timeout=15)
        npu_stdout, npu_stderr = npu_top.communicate(timeout=15)
        assert coordinator.returncode == 0, (coordinator_stdout, coordinator_stderr)
        assert npu_top.returncode == 0, (npu_stdout, npu_stderr)
        assert done.is_file()
    finally:
        _terminate(*(process for process in (coordinator, npu_top) if process is not None))

    assert path.with_name("machines.json.lock").is_file()
    document = _read_inventory(path)
    by_alias = {row["alias"]: row for row in document["machines"]}
    assert set(by_alias) == {"a3-165", "observed-new"}

    base = by_alias["a3-165"]
    assert base["host"] == {"ip": "192.168.13.165", "port": 2222, "user": "operator"}
    assert base["hardware"]["chip_summary"] == "910C"
    assert base["hardware"]["soc"] == "Ascend910C"
    assert base["hardware"]["npu_count"] == 16
    assert base["tags"] == ["coordinator", "A3", "910C", "dual-die"]
    mounts = {mount["target"]: mount for mount in base["mounts"]}
    assert mounts["/data/models"]["note"] == "keep"
    assert mounts["/data/weights"]["weight_candidate"] is True

    observed = by_alias["observed-new"]
    assert observed["host"] == {"ip": "10.0.0.77", "port": 22, "user": "root"}
    assert observed["hardware"]["npu_count"] == 16
    assert observed["tags"] == ["A3", "910C", "dual-die"]


@pytest.mark.skipif(not NPU_TOP_ROOT.is_dir(), reason="sibling npu-top checkout is required")
@pytest.mark.parametrize("operation", ["upsert", "remove", "set_default"])
def test_individual_mutations_wait_for_the_npu_top_inventory_lock(
    tmp_path: Path, operation: str,
) -> None:
    sys.path.insert(0, str(NPU_TOP_ROOT))
    try:
        from npu_top.inventory_bridge import inventory_lock
    finally:
        sys.path.remove(str(NPU_TOP_ROOT))

    path = tmp_path / "machines.json"
    store = MachineDirectory(path=path)
    store.upsert_machine({"alias": "alpha", "host": {"ip": "10.0.0.1", "port": 22}})
    store.upsert_machine({"alias": "beta", "host": {"ip": "10.0.0.2", "port": 22}})

    errors: list[BaseException] = []

    def mutate() -> None:
        try:
            if operation == "upsert":
                store.upsert_machine({"alias": "gamma", "host": {"ip": "10.0.0.3", "port": 22}})
            elif operation == "remove":
                store.remove_machine("beta")
            else:
                store.set_default("beta")
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    with inventory_lock(path):
        thread = threading.Thread(target=mutate)
        thread.start()
        time.sleep(0.2)
        assert thread.is_alive(), f"{operation} bypassed the adjacent inventory lock"
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert errors == []


@pytest.mark.skipif(not NPU_TOP_ROOT.is_dir(), reason="sibling npu-top checkout is required")
def test_lock_timeout_leaves_the_original_inventory_unchanged(tmp_path, monkeypatch):
    sys.path.insert(0, str(NPU_TOP_ROOT))
    try:
        from npu_top.inventory_bridge import inventory_lock
    finally:
        sys.path.remove(str(NPU_TOP_ROOT))
    path = tmp_path / "machines.json"
    store = MachineDirectory(path=path)
    store.upsert_machine({"alias": "alpha", "host": {"ip": "10.0.0.1", "port": 22}})
    before = path.read_bytes()
    monkeypatch.setattr("mindie_coordinator.machine_directory.INVENTORY_LOCK_TIMEOUT_SECONDS", 0.02)
    with inventory_lock(path), pytest.raises(TimeoutError, match="record lock"):
        store.upsert_machine({"alias": "beta", "host": {"ip": "10.0.0.2", "port": 22}})
    assert path.read_bytes() == before
