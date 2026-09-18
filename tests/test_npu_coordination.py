#!/usr/bin/env python3
"""Tests for the optional host-shared NPU coordination protocol."""

from __future__ import annotations

import os
import select
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from mindie_coordinator.host.mindie_npu_coordination import (
    CoordinationError,
    DEFAULT_SERVING_PORT_RANGE,
    JOB_TOKEN_ENV,
    NpuCoordinator,
    _confirmed_free_probe,
    parse_npu_smi_info,
    parse_port_range,
    process_guard_busy,
)


class FakeClock:
    def __init__(self, value: float = 1_800_000_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def occupancy(*, devices: list[int] | None = None, busy: list[int] | None = None) -> dict:
    visible = devices or [0, 1, 2, 3]
    busy_set = set(busy or [])
    return {
        "status": "ok",
        "devices": visible,
        "busy": {str(item): [{"kind": "test"}] for item in sorted(busy_set)},
        "free": [item for item in visible if item not in busy_set],
    }


class CoordinationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.clock = FakeClock()
        self.coordinator = NpuCoordinator(self.temp.name, clock=self.clock)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def submit(self, task_id: str, **extra):
        request = {
            "task_id": task_id,
            "agent_id": f"agent-{task_id}",
            "agent_alias": "team42",
            "npu_count": 1,
            "queue_ttl_seconds": 600,
            "estimated_duration_seconds": 60,
            **extra,
        }
        if "devices" in extra:
            request.pop("npu_count", None)
        return self.coordinator.submit(request)

    def test_task_snapshot_carries_uuid_owner_alias(self) -> None:
        result = self.coordinator.submit(
            {
                "task_id": "task-identity",
                "agent_id": "dce488a7-1af2-44b1-bb91-c3984743d33e",
                "agent_alias": "team42",
                "npu_count": 1,
                "queue_ttl_seconds": 600,
                "estimated_duration_seconds": 60,
            }
        )
        self.assertEqual(result["task"]["agent_id"], "dce488a7-1af2-44b1-bb91-c3984743d33e")
        self.assertEqual(result["task"]["agent_alias"], "team42")

    def test_managed_cpu_initialization_is_not_reallocated_after_heartbeat_expiry(self):
        self.submit("guarded-task", devices=[0])
        granted = self.coordinator.acquire("guarded-task", occupancy())
        token = granted["task"]["fence_token"]
        self.coordinator.preflight("guarded-task", token, occupancy())
        guard = {"marker": "a" * 32, "boot_id": "test"}
        with mock.patch("mindie_coordinator.host.mindie_npu_coordination.process_guard_busy", return_value=True):
            self.coordinator.activate("guarded-task", token, pid=1234, process_guard=guard, heartbeat_ttl_seconds=1)
            import sqlite3
            from contextlib import closing
            with closing(sqlite3.connect(Path(self.temp.name) / "coordinator.sqlite3")) as old_client, old_client:
                with self.assertRaisesRegex(sqlite3.IntegrityError, "process guard"):
                    old_client.execute("UPDATE tasks SET state='released' WHERE task_id='guarded-task'")
            self.clock.advance(2)
            self.assertEqual(self.coordinator.snapshot(occupancy())["tasks"][0]["state"], "orphaned_busy")
            self.assertEqual(self.coordinator.release("guarded-task", token, occupancy())["status"], "orphaned_busy")
            self.submit("waiting-task", devices=[0])
            self.assertNotEqual(self.coordinator.acquire("waiting-task", occupancy())["status"], "granted")
        with mock.patch("mindie_coordinator.host.mindie_npu_coordination.process_guard_busy", return_value=False):
            self.assertEqual(self.coordinator.acquire("waiting-task", occupancy())["status"], "granted")

    def test_subreaper_lease_requires_completion_even_after_supervisor_disappears(self):
        self.submit("retained-task", devices=[0])
        token = self.coordinator.acquire("retained-task", occupancy())["task"]["fence_token"]
        self.coordinator.preflight("retained-task", token, occupancy())
        guard = {"marker": "b" * 32, "boot_id": "test", "retain_until_release": True}
        with mock.patch("mindie_coordinator.host.mindie_npu_coordination.process_guard_busy", return_value=True):
            self.coordinator.activate("retained-task", token, pid=1234, process_guard=guard, heartbeat_ttl_seconds=1)
        # No marked process is left, but GC lacks a descendant completion receipt.
        def retained(value, *, completion_confirmed=False):
            return bool(value) and not completion_confirmed
        with mock.patch("mindie_coordinator.host.mindie_npu_coordination.process_guard_busy", side_effect=retained):
            self.clock.advance(2)
            self.assertEqual(self.coordinator.snapshot(occupancy())["tasks"][0]["state"], "orphaned_busy")
            self.assertEqual(self.coordinator.release("retained-task", token, occupancy())["status"], "orphaned_busy")
            self.submit("retained-waiter", devices=[0])
            self.assertNotEqual(self.coordinator.acquire("retained-waiter", occupancy())["status"], "granted")
            self.assertEqual(self.coordinator.release("retained-task", token, occupancy(), completion_confirmed=True)["status"], "released")
            self.assertEqual(self.coordinator.acquire("retained-waiter", occupancy())["status"], "granted")

    def test_process_guard_source_scans_public_remote_dev_marker(self):
        from mindie_coordinator.host import mindie_npu_coordination as module
        source = Path(module.__file__).read_text(encoding="utf-8")
        self.assertIn('JOB_TOKEN_ENV = "REMOTE_DEV_JOB_TOKEN"', source)
        self.assertNotIn("MINDIE_REMOTE_JOB_TOKEN", source)

    def _job_child_env(self, *, public: str | None, legacy: str | None) -> dict[str, str]:
        env = os.environ.copy()
        if public is None:
            env.pop(JOB_TOKEN_ENV, None)
        else:
            env[JOB_TOKEN_ENV] = public
        if legacy is None:
            env.pop("MINDIE_REMOTE_JOB_TOKEN", None)
        else:
            env["MINDIE_REMOTE_JOB_TOKEN"] = legacy
        return env

    def _spawn_ready_child(self, env: dict[str, str]) -> subprocess.Popen:
        script = (
            "import sys, time\n"
            "sys.stdout.write('ready\\n')\n"
            "sys.stdout.flush()\n"
            "time.sleep(30)\n"
        )
        return subprocess.Popen(
            [sys.executable, "-u", "-c", script],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _stop_child(self, proc: subprocess.Popen | None) -> None:
        if proc is None:
            return
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()

    def _proc_diag(self, proc: subprocess.Popen) -> str:
        root = Path("/proc") / str(proc.pid)
        try:
            stat = (root / "stat").read_text()
            state = stat.rsplit(") ", 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, IndexError) as exc:
            stat, state = f"<unreadable {type(exc).__name__}: {exc}>", "?"
        try:
            environ = (root / "environ").read_bytes()
            env_note = f"{len(environ)} bytes"
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError) as exc:
            environ, env_note = b"", f"unreadable {type(exc).__name__}: {exc}"
        stderr = b""
        if proc.stderr is not None:
            try:
                if proc.poll() is None:
                    os.set_blocking(proc.stderr.fileno(), False)
                stderr = proc.stderr.read() or b""
            except (OSError, ValueError, BlockingIOError, TypeError):
                pass
        return (
            f"pid={proc.pid} returncode={proc.poll()} stat_state={state} "
            f"environ={env_note} stat={stat[:200]!r} stderr={stderr[:500]!r}"
        )

    def _await_live_environ(self, proc: subprocess.Popen, expected: bytes) -> Path:
        stdout = proc.stdout
        if stdout is None:
            self.fail(f"child {self._proc_diag(proc)} has no stdout ready pipe")
        deadline = time.time() + 5
        buf = b""
        while time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            readable, _, _ = select.select([stdout], [], [], remaining)
            if readable:
                chunk = os.read(stdout.fileno(), 64)
                if not chunk:
                    break
                buf += chunk
                if b"ready\n" in buf:
                    break
            if proc.poll() is not None:
                self.fail(f"child died before ready signal {self._proc_diag(proc)}")
        else:
            self.fail(f"child never emitted ready signal stdout={buf!r} {self._proc_diag(proc)}")
        if b"ready\n" not in buf:
            self.fail(
                f"child closed stdout without ready signal stdout={buf!r} {self._proc_diag(proc)}"
            )
        deadline = time.time() + 2
        last = "no /proc snapshot"
        while time.time() < deadline:
            root = Path("/proc") / str(proc.pid)
            try:
                stat = (root / "stat").read_text()
                state = stat.rsplit(") ", 1)[1].split()[0]
                environ = (root / "environ").read_bytes()
                last = f"stat_state={state} environ_bytes={len(environ)} returncode={proc.poll()}"
                if state != "Z" and expected in environ.split(b"\0"):
                    return root
            except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, IndexError) as exc:
                last = f"{type(exc).__name__}: {exc} returncode={proc.poll()}"
            if proc.poll() is not None:
                self.fail(
                    f"child died after ready before /proc showed expected token "
                    f"{last} {self._proc_diag(proc)}"
                )
            time.sleep(0.01)
        self.fail(
            f"child ready but /proc never showed expected token {last} {self._proc_diag(proc)}"
        )

    def _iter_only_pids(self, *pids: int):
        entries = [Path("/proc") / str(pid) for pid in pids]
        original = Path.iterdir

        def iterdir(self_path):
            if os.fspath(self_path) == "/proc":
                yield from entries
                return
            yield from original(self_path)

        return mock.patch.object(Path, "iterdir", iterdir)

    @unittest.skipUnless(sys.platform.startswith("linux") and Path("/proc").is_dir(), "process environ scan is Linux /proc")
    def test_process_guard_sees_remote_dev_job_token_not_legacy_name(self):
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        marker = os.urandom(16).hex()
        public_token = f"{JOB_TOKEN_ENV}={marker}".encode()
        legacy_token = f"MINDIE_REMOTE_JOB_TOKEN={marker}".encode()
        guard = {"marker": marker, "boot_id": boot_id}
        proc = None
        leftover = None
        try:
            proc = self._spawn_ready_child(self._job_child_env(public=marker, legacy=None))
            root = self._await_live_environ(proc, public_token)
            pairs = (root / "environ").read_bytes().split(b"\0")
            self.assertIn(public_token, pairs)
            self.assertNotIn(legacy_token, pairs)
            self.assertNotEqual((root / "stat").read_text().rsplit(") ", 1)[1].split()[0], "Z")
            with self._iter_only_pids(proc.pid):
                self.assertTrue(process_guard_busy(guard))
        finally:
            self._stop_child(proc)
        try:
            leftover = self._spawn_ready_child(self._job_child_env(public=None, legacy=marker))
            root = self._await_live_environ(leftover, legacy_token)
            pairs = (root / "environ").read_bytes().split(b"\0")
            self.assertNotIn(public_token, pairs)
            self.assertIn(legacy_token, pairs)
            self.assertNotEqual((root / "stat").read_text().rsplit(") ", 1)[1].split()[0], "Z")
            with self._iter_only_pids(leftover.pid):
                self.assertFalse(process_guard_busy(guard))
        finally:
            self._stop_child(leftover)

    @unittest.skipUnless(sys.platform.startswith("linux") and Path("/proc").is_dir(), "process environ scan is Linux /proc")
    def test_process_guard_permission_error_keeps_busy(self):
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        marker = os.urandom(16).hex()
        blocked = Path("/proc") / "1"
        original_iterdir = Path.iterdir
        original_read_text = Path.read_text
        original_read_bytes = Path.read_bytes

        def iterdir(self_path):
            if os.fspath(self_path) == "/proc":
                yield blocked
                return
            yield from original_iterdir(self_path)

        def read_text(self_path, *args, **kwargs):
            if self_path.name == "stat" and self_path.parent == blocked:
                raise PermissionError
            return original_read_text(self_path, *args, **kwargs)

        def read_bytes(self_path, *args, **kwargs):
            if self_path.name == "environ" and self_path.parent == blocked:
                raise PermissionError
            return original_read_bytes(self_path, *args, **kwargs)

        with mock.patch.object(Path, "iterdir", iterdir), \
                mock.patch.object(Path, "read_text", read_text), \
                mock.patch.object(Path, "read_bytes", read_bytes):
            self.assertTrue(process_guard_busy({"marker": marker, "boot_id": boot_id}))

    def activate(self, task_id: str, *, heartbeat_ttl: int = 10) -> int:
        granted = self.coordinator.acquire(task_id, occupancy(), grant_ttl_seconds=10)
        token = granted["task"]["fence_token"]
        self.assertEqual(granted["status"], "granted")
        self.assertEqual(
            self.coordinator.preflight(task_id, token, occupancy(), start_ttl_seconds=10)["status"],
            "starting",
        )
        self.assertEqual(
            self.coordinator.activate(
                task_id,
                token,
                pid=1234,
                heartbeat_ttl_seconds=heartbeat_ttl,
            )["status"],
            "active",
        )
        return token

    def test_strict_fifo_and_atomic_multi_device_grants(self) -> None:
        self.submit("task-a", npu_count=2)
        self.submit("task-b", npu_count=2)

        waiting = self.coordinator.acquire("task-b", occupancy())
        self.assertEqual(waiting["status"], "waiting")
        self.assertEqual(waiting["reason"], "strict_fifo")
        self.assertEqual(waiting["ahead_task_id"], "task-a")

        first = self.coordinator.acquire("task-a", occupancy())
        second = self.coordinator.acquire("task-b", occupancy())
        self.assertEqual(first["task"]["granted_devices"], [0, 1])
        self.assertEqual(second["task"]["granted_devices"], [2, 3])
        self.assertNotEqual(first["task"]["fence_token"], second["task"]["fence_token"])

    def test_concurrent_agents_receive_disjoint_atomic_grants(self) -> None:
        self.submit("task-concurrent-a", npu_count=2)
        self.submit("task-concurrent-b", npu_count=2)
        barrier = threading.Barrier(2)

        def worker(task_id: str) -> list[int]:
            coordinator = NpuCoordinator(self.temp.name, clock=self.clock)
            barrier.wait()
            for _ in range(100):
                result = coordinator.acquire(task_id, occupancy())
                if result["status"] == "granted":
                    return result["task"]["granted_devices"]
                time.sleep(0.001)
            raise AssertionError(f"task did not receive a grant: {task_id}")

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(worker, "task-concurrent-a")
            second = pool.submit(worker, "task-concurrent-b")
            first_devices = first.result()
            second_devices = second.result()

        self.assertEqual(set(first_devices) & set(second_devices), set())
        self.assertEqual(set(first_devices) | set(second_devices), {0, 1, 2, 3})

    def test_real_occupancy_wins_over_declarations(self) -> None:
        self.submit("task-busy", devices=[0])
        result = self.coordinator.acquire("task-busy", occupancy(busy=[0]))
        self.assertEqual(result["status"], "waiting")
        self.assertEqual(result["reason"], "requested_devices_unavailable")
        self.assertEqual(result["unavailable_devices"], [0])

    def test_probe_failure_blocks_only_the_optional_grant(self) -> None:
        self.submit("task-probe")
        result = self.coordinator.acquire(
            "task-probe",
            {"status": "failed", "error": "npu-smi unavailable"},
        )
        self.assertEqual(result["status"], "probe_failed")
        snapshot = self.coordinator.snapshot(None, task_id="task-probe")
        self.assertEqual(snapshot["tasks"][0]["state"], "queued")

    def test_preflight_conflict_returns_grant_to_queue(self) -> None:
        self.submit("task-preflight", devices=[0])
        granted = self.coordinator.acquire("task-preflight", occupancy())
        token = granted["task"]["fence_token"]
        result = self.coordinator.preflight(
            "task-preflight",
            token,
            occupancy(busy=[0]),
        )
        self.assertEqual(result["status"], "waiting")
        self.assertEqual(result["reason"], "external_occupancy")
        self.assertEqual(result["task"]["state"], "queued")
        self.assertIsNone(result["task"]["fence_token"])

    def test_future_manual_hold_blocks_overlapping_estimated_window(self) -> None:
        self.coordinator.add_hold(
            {
                "hold_id": "hold-human",
                "owner": "human-alice",
                "devices": [0],
                "not_before": self.clock() + 30,
                "duration_seconds": 120,
                "reason": "manual run",
            },
            occupancy(),
        )
        self.submit("task-held", devices=[0], estimated_duration_seconds=60)
        result = self.coordinator.acquire("task-held", occupancy())
        self.assertEqual(result["status"], "waiting")
        self.assertEqual(result["unavailable_devices"], [0])

    def test_queue_and_unactivated_grant_expire(self) -> None:
        self.submit("task-queue", queue_ttl_seconds=5)
        self.clock.advance(6)
        snapshot = self.coordinator.snapshot(occupancy())
        task = next(item for item in snapshot["tasks"] if item["task_id"] == "task-queue")
        self.assertEqual(task["state"], "expired")

        self.submit("task-grant", queue_ttl_seconds=60)
        self.assertEqual(
            self.coordinator.acquire("task-grant", occupancy(), grant_ttl_seconds=5)["status"],
            "granted",
        )
        self.clock.advance(6)
        snapshot = self.coordinator.snapshot(occupancy())
        task = next(item for item in snapshot["tasks"] if item["task_id"] == "task-grant")
        self.assertEqual(task["state"], "expired")

    def test_heartbeat_timeout_keeps_busy_hardware_quarantined(self) -> None:
        self.submit("task-active", devices=[0])
        self.activate("task-active", heartbeat_ttl=5)
        self.clock.advance(6)

        busy_snapshot = self.coordinator.snapshot(occupancy(busy=[0]), task_id="task-active")
        self.assertEqual(busy_snapshot["tasks"][0]["state"], "orphaned_busy")

        free_snapshot = self.coordinator.snapshot(occupancy(), task_id="task-active")
        self.assertEqual(free_snapshot["tasks"][0]["state"], "released")

    def test_release_defers_while_hardware_is_busy(self) -> None:
        self.submit("task-release", devices=[0])
        token = self.activate("task-release")
        deferred = self.coordinator.release("task-release", token, occupancy(busy=[0]))
        self.assertEqual(deferred["status"], "orphaned_busy")
        released = self.coordinator.release("task-release", token, occupancy())
        self.assertEqual(released["status"], "released")

    def test_missing_device_is_unknown_not_free(self) -> None:
        self.submit("task-missing", devices=[0])
        token = self.activate("task-missing")
        result = self.coordinator.release(
            "task-missing",
            token,
            occupancy(devices=[1, 2, 3]),
        )
        self.assertEqual(result["status"], "orphaned_busy")
        self.assertEqual(result["conflicting_devices"], [0])

    def test_confirmation_probe_unions_transient_busy_samples(self) -> None:
        samples = iter([occupancy(busy=[0]), occupancy()])
        observed = _confirmed_free_probe(
            samples=2,
            interval_seconds=0,
            probe=lambda: next(samples),
        )
        self.assertIn("0", observed["busy"])
        self.assertNotIn(0, observed["free"])

    def test_npu_smi_parser_marks_process_and_hbm_occupancy(self) -> None:
        output = """
| 0     910B4      | OK              41.8        0                0 / 0 |
| 0     0          | 0000:C1:00.0     0            0 / 0       5000 / 65536 |
| 1     910B4      | OK              41.8        0                0 / 0 |
| 1     1          | 0000:81:00.0     0            0 / 0        100 / 65536 |
| NPU   Chip       | Process id      Process name             Process memory(MB) |
| 1     0            4321              root                     python             |
"""
        parsed = parse_npu_smi_info(output)
        self.assertEqual(parsed["devices"], [0, 1])
        self.assertEqual(sorted(parsed["busy"]), ["0", "1"])
        self.assertEqual(parsed["busy"]["0"][0]["kind"], "hbm_threshold")
        self.assertEqual(parsed["busy"]["1"][0]["pid"], 4321)

    def test_a3_process_columns_map_chip_to_physical_device_below_hbm_threshold(self):
        devices = """
| 0     Ascend910  | OK | 170 | 48 | 0 / 0 |
| 0     0         | 0000:9D:00.0 | 0 | 0 / 0 | 3364 / 65536 |
| 0     Ascend910  | OK | -   | 50 | 0 / 0 |
| 1     1         | 0000:9F:00.0 | 0 | 0 / 0 | 2951 / 65536 |
| 2     Ascend910  | OK | 170 | 48 | 0 / 0 |
| 0     8         | 0000:89:00.0 | 0 | 0 / 0 | 3000 / 65536 |
"""
        header = '| NPU Chip | Process id | Process name | Process memory(MB) |\n'
        rows = '| 0 0 | 4321 | python3 | 123 |\n| 0 1 | 4322 | python3 | 122 |\n| 2 0 | 4323 | worker | 120 |\n'
        parsed = parse_npu_smi_info(devices + header + rows)
        self.assertEqual(parsed['status'], 'ok')
        self.assertEqual({key: value[0]['pid'] for key, value in parsed['busy'].items()},
                         {'0': 4321, '1': 4322, '8': 4323})
        self.assertEqual(parsed['free'], [])
        self.submit('task-small-worker', devices=[1])
        self.assertEqual(self.coordinator.acquire('task-small-worker', parsed)['status'], 'waiting')
        for bad in [devices, devices + header + '| 0 9 | 4321 | python3 | 123 |\n',
                    devices + header + '| 0 1 | unknown | python3 | 123 |\n']:
            with self.subTest(output=bad):
                unknown = parse_npu_smi_info(bad)
                self.assertEqual(unknown['status'], 'failed')
                self.assertEqual(unknown['free'], [])

    def test_process_identity_columns_follow_the_device_table_layout(self):
        devices = """
| 0     Ascend910  | OK | 170 | 48 | 0 / 0 |
| 0     0         | 0000:9D:00.0 | 0 | 0 / 0 | 3364 / 65536 |
"""
        header = '| NPU Chip | Process id | Process name | Process memory(MB) |\n'
        # A multi-chip (NPU, Chip) -> Phy-ID table makes a lone first column an
        # ambiguous NPU index. Fail closed instead of misattributing the process.
        single = parse_npu_smi_info(devices + header + '| 0 | 4321 | python3 | 123 |\n')
        self.assertEqual(single['status'], 'failed')
        self.assertEqual(single['free'], [])
        unknown = parse_npu_smi_info(devices + header + '| 0 9 | 4321 | python3 | 123 |\n')
        self.assertEqual(unknown['status'], 'failed')
        self.assertEqual(unknown['free'], [])
        accepted = parse_npu_smi_info(devices + header + '| 0 0 | 4321 | python3 | 123 |\n')
        self.assertEqual(accepted['status'], 'ok')
        self.assertEqual(accepted['busy']['0'][0]['pid'], 4321)
        # Without any (NPU, Chip) table the layout is flat and the lone column
        # is the device identity itself.
        flat = """
| 0     910B4      | OK              41.8        0                0 / 0 |
| 1     910B4      | OK              41.8        0                0 / 0 |
| NPU   Chip       | Process id      Process name             Process memory(MB) |
| 1 | 4321 | python | 100 |
"""
        parsed = parse_npu_smi_info(flat)
        self.assertEqual(parsed['status'], 'ok')
        self.assertEqual(parsed['busy']['1'][0]['pid'], 4321)
        self.assertEqual(parsed['free'], [0])


class ContainerPortOwnershipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.clock = FakeClock()
        self.coordinator = NpuCoordinator(self.temp.name, clock=self.clock)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_automatic_service_ports_without_runtime_declarations(self) -> None:
        first_port, _ = parse_port_range(DEFAULT_SERVING_PORT_RANGE)
        listening = {"status": "ok", "ports": [first_port]}
        grants = []
        for index in range(2):
            task_id = f"automatic-{index}"
            self.coordinator.submit({
                "task_id": task_id, "agent_id": "test-agent", "npu_count": 1,
                "service_port": 0, "service_ports": [],
            })
            grant = self.coordinator.acquire(task_id, occupancy(), listening=listening)
            self.assertEqual(grant["status"], "granted")
            self.assertEqual(grant["task"]["granted_service_port"], first_port + index + 1)
            self.assertEqual(grant["environment"]["MINDIE_SERVICE_PORT"], str(first_port + index + 1))
            grants.append(grant)
        released = self.coordinator.release(
            "automatic-0", grants[0]["task"]["fence_token"], occupancy(),
            completion_confirmed=True, listening=listening,
        )
        self.assertEqual(released["status"], "released")
        self.coordinator.submit({
            "task_id": "automatic-reuse", "agent_id": "test-agent", "npu_count": 1,
            "service_port": 0,
        })
        reused = self.coordinator.acquire("automatic-reuse", occupancy(), listening=listening)
        self.assertEqual(reused["task"]["granted_service_port"], first_port + 1)

    def test_explicit_service_port_still_requires_runtime_declaration(self) -> None:
        with self.assertRaisesRegex(CoordinationError, "not in declared"):
            self.coordinator.submit({
                "task_id": "undeclared", "agent_id": "test-agent", "npu_count": 1,
                "service_port": 30000, "service_ports": [],
            })

    def test_user_ssh_reservation_blocks_service_collision_and_survives_release(self) -> None:
        reserved = self.coordinator.reserve_container_ssh(
            {"user": "maoxx241", "container_name": "mindie-maoxx241", "port": 46001}
        )
        self.assertEqual(reserved["status"], "reserved")
        self.assertEqual(reserved["port"], 46001)
        again = self.coordinator.reserve_container_ssh(
            {"user": "maoxx241", "container_name": "mindie-maoxx241", "port": 46001}
        )
        self.assertTrue(again["reused"])
        with self.assertRaisesRegex(Exception, "already has SSH port"):
            self.coordinator.reserve_container_ssh(
                {"user": "maoxx241", "container_name": "mindie-maoxx241", "port": 46002}
            )
        with self.assertRaisesRegex(Exception, "must be mindie-maoxx241"):
            self.coordinator.reserve_container_ssh(
                {"user": "maoxx241", "container_name": "mindie-other", "port": 46001}
            )
        self.coordinator.submit(
            {
                "task_id": "exec-1",
                "agent_id": "maoxx241",
                "npu_count": 1,
                "queue_ttl_seconds": 600,
                "estimated_duration_seconds": 60,
                "service_port": 0,
                "service_ports": [46001, 48001],
            }
        )
        listening = {"status": "ok", "ports": [46001]}
        granted = self.coordinator.acquire("exec-1", occupancy(), listening=listening)
        self.assertEqual(granted["status"], "granted")
        self.assertEqual(granted["task"]["granted_service_port"], 48001)
        self.assertEqual(granted["environment"]["MINDIE_SERVICE_PORT"], "48001")
        token = granted["task"]["fence_token"]
        busy_port = self.coordinator.release(
            "exec-1", token, occupancy(), completion_confirmed=True,
            listening={"status": "ok", "ports": [46001, 48001]},
        )
        self.assertEqual(busy_port["status"], "orphaned_busy")
        released = self.coordinator.release(
            "exec-1", token, occupancy(), completion_confirmed=True,
            listening={"status": "ok", "ports": [46001]},
        )
        self.assertEqual(released["status"], "released")
        snapshot = self.coordinator.snapshot(occupancy())
        self.assertEqual(
            [(row["port"], row["kind"]) for row in snapshot["ports"]],
            [(46001, "container_ssh")],
        )


if __name__ == "__main__":
    unittest.main()
