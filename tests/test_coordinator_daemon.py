"""Daemon, ensure-daemon, and service-client tests (split from test_coordinator.py)."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from mindie_coordinator.execution_sources import capture_sources
from mindie_coordinator.ready_runtime import RuntimePool

from test_coordinator import (
    Backend,
    TaskClientTests as _TaskClientTests,
    _init_git_workspace,
    runtime_spec,
)


class DaemonProcessTests(unittest.TestCase):
    def test_short_socket_ping_start_exit_restart_and_session_dirs_persist(self):
        from mindie_coordinator.agent_session import AgentSessions
        from mindie_coordinator.service import CoordinatorClient, socket_path

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "vllm-ascend-workspace" / ".mindie-local" / "closeout" / "daemon-path-check" / "coordinator"
            state.mkdir(parents=True)
            sessions = Path(tmp) / "overridden-sessions"
            AgentSessions(sessions)
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(
                [sys.executable, "-m", "mindie_coordinator", "daemon", "--state-dir", str(state)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, env=env,
            )
            client = CoordinatorClient(state)
            try:
                deadline = time.time() + 5
                while time.time() < deadline:
                    try:
                        self.assertEqual(client.call("ping")["runtime"][0]["loaded"]["package"], "mindie-coordinator")
                        break
                    except TimeoutError:
                        raise
                    except (RuntimeError, FileNotFoundError, ConnectionError, OSError):
                        time.sleep(0.05)
                else:
                    self.fail("daemon did not start")
                sock = socket_path(state)
                if os.name == "nt":
                    self.assertEqual(sock, state / "coordinator.ipc")
                else:
                    self.assertLessEqual(len(str(sock)), 80)
                self.assertTrue(sock.exists())
                with self.assertRaises(Exception):
                    client.advance(str(sessions), "alice", "0" * 64)
                import sqlite3
                with closing(sqlite3.connect(state / "coordinator.sqlite3")) as db:
                    row = db.execute("SELECT data FROM records WHERE kind='meta' AND id='session_dirs'").fetchone()
                self.assertIsNotNone(row)
                self.assertIn(str(sessions), json.loads(row[0])["paths"])
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)
                else:
                    proc.terminate()
                proc.wait(timeout=5)
                proc.stdout.close()
                proc = subprocess.Popen(
                    [sys.executable, "-m", "mindie_coordinator", "daemon", "--state-dir", str(state)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, env=env,
                )
                deadline = time.time() + 5
                while time.time() < deadline:
                    if proc.poll() is not None:
                        self.fail("daemon exited before restart: " + proc.stdout.read().decode(errors="replace"))
                    try:
                        client.call("ping")
                        break
                    except TimeoutError:
                        raise
                    except (RuntimeError, FileNotFoundError, ConnectionError, OSError):
                        time.sleep(0.05)
                else:
                    self.fail("daemon did not restart")
                with closing(sqlite3.connect(state / "coordinator.sqlite3")) as db:
                    row = db.execute("SELECT data FROM records WHERE kind='meta' AND id='session_dirs'").fetchone()
                self.assertIn(str(sessions), json.loads(row[0])["paths"])
            finally:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)
                else:
                    proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                proc.stdout.close()
                sock = socket_path(state)
                if sock.exists():
                    sock.unlink()

    def test_admit_returns_queued_while_preparation_runs(self):
        from mindie_coordinator.agent_session import AgentSessions
        from mindie_coordinator.ready_runtime import RuntimePool
        from mindie_coordinator.service import CoordinatorClient, CoordinatorService, socket_path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_git_workspace(root)
            for name in ("vllm", "vllm-ascend"):
                _init_git_workspace(root / name)
            backend = Backend(root / "host")
            backend.require_prepared = True
            pool = RuntimePool(root / "manager", backend)
            spec = runtime_spec(1, user="alice", recipe="rc")
            backend.mark_prepared(spec)
            pool.register("runtime-a", spec)
            sessions = AgentSessions(root / "sessions")
            context = sessions.attach("codex", "native-async", str(root))
            sessions.bind_sources(context, {"vllm": str(root / "vllm"), "vllm-ascend": str(root / "vllm-ascend")})
            service = CoordinatorService(root / "coordinator", pool=pool, backend=backend, sessions=sessions)
            started = threading.Event()
            release = threading.Event()

            def delayed(*args, **kwargs):
                started.set()
                self.assertTrue(release.wait(5))
                return {"vllm": "a" * 40, "vllm-ascend": "b" * 40}

            service.sync_binding = delayed
            worker = threading.Thread(target=service.serve, daemon=True)
            worker.start()
            client = CoordinatorClient(root / "coordinator")
            try:
                deadline = time.time() + 5
                while time.time() < deadline:
                    try:
                        client.call("ping")
                        break
                    except TimeoutError:
                        raise
                    except (RuntimeError, FileNotFoundError, ConnectionError, OSError):
                        time.sleep(0.05)
                else:
                    self.fail("in-process daemon did not listen")
                spec = {
                    "command": "true", "env": {}, "environment": {"recipe": "rc"},
                    "resources": {"npu_count": 1}, "topology": {},
                    "roles": [{"name": "default", "command": "true", "npu_count": 1}],
                    "timeout_seconds": 1800, "service": None,
                }
                spec["source_snapshot"] = capture_sources(
                    {name: str(root / name) for name in ("vllm", "vllm-ascend")}, sessions.state_dir)
                t0 = time.time()
                admitted = client.admit(str(sessions.state_dir), "alice", context["session"]["id"], spec)
                self.assertLess(time.time() - t0, 0.8)
                self.assertIn(admitted["state"], {"queued", "preparing", "bound", "waiting", "launch_pending"})
                self.assertEqual(len(admitted["execution_id"]), 64)
                t1 = time.time()
                client.call("ping")
                self.assertLess(time.time() - t1, 0.5)
                status = client.advance(str(sessions.state_dir), "alice", admitted["execution_id"], "status")
                self.assertEqual(status["execution_id"], admitted["execution_id"])
                self.assertNotEqual(status["state"], "timeout")
                self.assertTrue(started.wait(3))
                status = client.advance(str(sessions.state_dir), "alice", admitted["execution_id"], "status")
                self.assertEqual(status["progress"]["step"], "sync-sources")
                self.assertIn("parity.log", status["progress"]["log_ref"])
                self.assertTrue(Path(status["progress"]["log_ref"]).is_file())
                self.assertEqual(client.call("restart_if_idle")["status"], "busy")
                t2 = time.time()
                client.call("ping")
                self.assertLess(time.time() - t2, 0.5)
                release.set()
                lock = service._lock_for("execution", admitted["execution_id"])
                deadline = time.time() + 20
                while lock.locked() and time.time() < deadline:
                    time.sleep(0.02)
                self.assertFalse(lock.locked(), "fixture launch did not finish")
                completed = client.advance(str(sessions.state_dir), "alice", admitted["execution_id"], "status")
                self.assertEqual(completed["execution_id"], admitted["execution_id"])
                self.assertEqual(completed["state"], "running", completed)
            finally:
                release.set()
                service._stopped.set()
                worker.join(timeout=2.5)
                self.assertFalse(worker.is_alive())
                # This fixture forces the private stop event while a simulated
                # execution is live; drain its already scheduled workers too.
                deadline = time.monotonic() + 10
                while (service._active_requests or any(item.locked() for item in service._lock_registry.values())) and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertFalse(service._active_requests)
                self.assertFalse(any(item.locked() for item in service._lock_registry.values()))
                sock = socket_path(root / "coordinator")
                if sock.exists():
                    sock.unlink()

    def test_startup_with_pending_prepare_stays_responsive(self):
        from mindie_coordinator.agent_session import AgentSessions
        from mindie_coordinator.ready_runtime import RuntimePool
        from mindie_coordinator.service import CoordinatorClient, CoordinatorService, socket_path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_git_workspace(root)
            for name in ("vllm", "vllm-ascend"):
                _init_git_workspace(root / name)
            backend = Backend(root / "host")
            backend.require_prepared = True
            pool = RuntimePool(root / "manager", backend)
            spec = runtime_spec(1, user="alice", recipe="rc")
            backend.mark_prepared(spec)
            pool.register("runtime-a", spec)
            sessions = AgentSessions(root / "sessions")
            context = sessions.attach("codex", "native-pending", str(root))
            sessions.bind_sources(context, {"vllm": str(root / "vllm"), "vllm-ascend": str(root / "vllm-ascend")})
            run_spec = {
                "command": "true", "env": {}, "environment": {"recipe": "rc"},
                "resources": {"npu_count": 1}, "topology": {},
                "roles": [{"name": "default", "command": "true", "npu_count": 1}],
                "timeout_seconds": 1800, "service": None,
            }
            run_spec["source_snapshot"] = capture_sources(
                {name: str(root / name) for name in ("vllm", "vllm-ascend")}, sessions.state_dir)
            row = sessions.execution(context, "pending-prep", run_spec)
            row.update(phase="queued", admitted=True, user="alice")
            sessions.save_execution(row)
            service = CoordinatorService(root / "coordinator", pool=pool, backend=backend, sessions=sessions)
            started = threading.Event()
            release = threading.Event()

            def delayed(*args, **kwargs):
                started.set()
                self.assertTrue(release.wait(10))
                return {"vllm": "a" * 40, "vllm-ascend": "b" * 40}

            service.sync_binding = delayed
            worker = threading.Thread(target=service.serve, daemon=True)
            worker.start()
            client = CoordinatorClient(root / "coordinator")
            try:
                deadline = time.time() + 5
                while time.time() < deadline:
                    try:
                        t0 = time.time()
                        client.call("ping")
                        self.assertLess(time.time() - t0, 0.5)
                        break
                    except TimeoutError:
                        raise
                    except (RuntimeError, FileNotFoundError, ConnectionError, OSError):
                        time.sleep(0.05)
                else:
                    self.fail("daemon did not listen")
                self.assertTrue(started.wait(8))
                t1 = time.time()
                client.call("ping")
                self.assertLess(time.time() - t1, 0.5)
                status = client.advance(str(sessions.state_dir), "alice", row["id"], "status")
                self.assertEqual(status["execution_id"], row["id"])
                self.assertNotEqual(status["state"], "timeout")
            finally:
                release.set()
                lock = service._lock_for("execution", row["id"])
                deadline = time.monotonic() + 10
                while lock.locked() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertFalse(lock.locked(), "fixture preparation did not finish")
                service._stopped.set()
                worker.join(timeout=2.5)
                self.assertFalse(worker.is_alive())
                # This fixture forces the private stop event while a simulated
                # execution is live; drain its already scheduled workers too.
                deadline = time.monotonic() + 10
                while (service._active_requests or any(item.locked() for item in service._lock_registry.values())) and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertFalse(service._active_requests)
                self.assertFalse(any(item.locked() for item in service._lock_registry.values()))
                sock = socket_path(root / "coordinator")
                if sock.exists():
                    sock.unlink()


class ServiceClientTests(_TaskClientTests):
    """CoordinatorService/CoordinatorClient lifecycle and cleanup behavior."""

    def test_daemon_restart_refuses_active_work_and_lease_then_allows_idle(self):
        service = self.client.coordinator
        reply = self.client.run("serve")
        self.assertEqual(service.handle({"op": "restart_if_idle"})["value"]["status"], "busy")
        self.assertFalse(service._stopped.is_set())
        stopped = self.client.observe(reply["execution_id"], "stop")
        self.assertTrue(stopped["resources_released"])
        self.assertEqual(stopped["roles"][0]["lease_state"], "released")
        self.assertTrue(stopped["roles"][0]["quiet"])
        self.assertEqual(service.handle({"op": "restart_if_idle"})["value"]["status"], "stopping")
        self.assertTrue(service._stopped.is_set())

    def test_restart_does_not_overlap_while_stop_is_stopping(self):
        first = self.client.run("serve-vllm", service="vllm", timeout_seconds=None)
        original = self.backend.job

        def sticky(runtime, job_id, action, **params):
            if action == "stop":
                self.backend.jobs[job_id].update(state="stopping", quiet=False)
                return copy.deepcopy(self.backend.jobs[job_id])
            return original(runtime, job_id, action, **params)

        self.backend.job = sticky
        from mindie_coordinator import service as service_mod
        with mock.patch.object(service_mod, "STOP_WAIT_SECONDS", 0.2):
            reply = self.client.run("serve-vllm", service="vllm", timeout_seconds=None, restart=True)
        self.assertEqual(reply["execution_id"], first["execution_id"])
        self.assertEqual(reply["state"], "stopping")
        live = [row for row in self.store.executions(self.context["session"]["id"]) if row.get("phase") not in {"succeeded", "failed", "timeout", "cancelled", "inconclusive"}]
        self.assertEqual(len(live), 1)

    def test_stop_during_delayed_preparation_does_not_launch(self):
        self.client.coordinator._async_progress = True
        started = threading.Event()
        release = threading.Event()
        original = self.client.coordinator._place_or_prepare

        def delayed(*args, **kwargs):
            started.set()
            self.assertTrue(release.wait(5))
            return original(*args, **kwargs)

        self.client.coordinator._place_or_prepare = delayed
        admitted = self.client.run("true")
        self.assertIn(admitted["state"], {"queued", "preparing", "bound", "waiting", "launch_pending"})
        self.assertEqual(len(admitted["execution_id"]), 64)
        self.assertTrue(started.wait(3))
        self.assertEqual(len(self.pool.status("alice")["jobs"]), 0)
        stopped = self.client.observe(admitted["execution_id"], "stop")
        self.assertNotEqual(stopped["state"], "running")
        self.assertNotEqual(stopped["state"], "cancelled")
        self.assertTrue(stopped.get("cancel_requested"))
        self.assertEqual(len(self.pool.status("alice")["jobs"]), 0)
        release.set()
        lock = self.client.coordinator._lock_for("execution", admitted["execution_id"])
        deadline = time.time() + 5
        while lock.locked() and time.time() < deadline:
            time.sleep(0.02)
        self.assertFalse(lock.locked())
        final = self.client.observe(admitted["execution_id"])
        self.assertEqual(final["state"], "cancelled")
        self.assertEqual(len(self.pool.status("alice")["jobs"]), 0)
        self.assertFalse(any(job.get("state") not in {"succeeded", "failed", "timeout", "cancelled", "inconclusive"}
                             for job in self.pool.status("alice")["jobs"]))

    def test_group_stop_during_delayed_preparation_keeps_gates_closed(self):
        self._seed(self.pool, "runtime-a2", runtime_spec(2, user="alice", recipe="rc"))
        self.client.coordinator._async_progress = True
        started = threading.Event()
        release = threading.Event()
        original = self.client.coordinator._place_or_prepare

        def delayed(*args, **kwargs):
            started.set()
            self.assertTrue(release.wait(5))
            return original(*args, **kwargs)

        self.client.coordinator._place_or_prepare = delayed
        admitted = self.client.run(
            "unused",
            topology={"roles": [{"name": "prefill", "npu_count": 1, "host": "192.0.2.1"},
                                {"name": "decode", "npu_count": 1, "host": "192.0.2.1"}]},
        )
        self.assertIn(admitted["state"], {"queued", "preparing", "bound", "waiting", "launch_pending"})
        self.assertEqual(len(admitted["execution_id"]), 64)
        self.assertTrue(started.wait(3))
        self.assertEqual(len(self.pool.status("alice")["jobs"]), 0)
        self.client.observe(admitted["execution_id"], "stop")
        release.set()
        lock = self.client.coordinator._lock_for("execution", admitted["execution_id"])
        deadline = time.time() + 5
        while lock.locked() and time.time() < deadline:
            time.sleep(0.02)
        final = self.client.observe(admitted["execution_id"])
        self.assertEqual(final["state"], "cancelled")
        self.assertEqual(len(self.pool.status("alice")["jobs"]), 0)

    def test_finish_during_delayed_preparation_completes_on_ticker_without_retry(self):
        from mindie_coordinator import service as service_mod

        self.client.coordinator._async_progress = True
        started = threading.Event()
        release = threading.Event()
        original = self.client.coordinator._place_or_prepare

        def delayed(*args, **kwargs):
            started.set()
            self.assertTrue(release.wait(5))
            return original(*args, **kwargs)

        self.client.coordinator._place_or_prepare = delayed
        with mock.patch.object(service_mod, "TICK_SECONDS", 0.05):
            ticker = threading.Thread(target=self.client.coordinator._tick_loop, daemon=True)
            ticker.start()
            try:
                admitted = self.client.run("true")
                self.assertIn(admitted["state"], {"queued", "preparing", "bound", "waiting", "launch_pending"})
                self.assertEqual(len(admitted["execution_id"]), 64)
                self.assertTrue(started.wait(3))
                self.assertEqual(len(self.pool.status("alice")["jobs"]), 0)
                reply = self.client.finish()
                self.assertEqual(reply["state"], "finishing")
                session_id = self.context["session"]["id"]
                with self.store.transaction() as db:
                    session = self.store.get(db, "session", session_id)
                self.assertEqual(session["state"], "finishing")
                self.assertEqual(session["finish"]["user"], "alice")
                self.assertFalse(session["finish"]["force"])
                release.set()
                lock = self.client.coordinator._lock_for("execution", admitted["execution_id"])
                deadline = time.time() + 5
                while lock.locked() and time.time() < deadline:
                    time.sleep(0.02)
                self.assertFalse(lock.locked())
                state = None
                deadline = time.time() + 3
                while time.time() < deadline:
                    with self.store.transaction() as db:
                        state = self.store.get(db, "session", session_id)["state"]
                    if state == "finished":
                        break
                    time.sleep(0.02)
                self.assertEqual(state, "finished")
                rows = self.store.executions(session_id)
                self.assertEqual([row["phase"] for row in rows], ["cancelled"])
                remote = rows[0]["remote_session"]["id"]
                self.assertEqual(self.pool.session_bindings("alice", remote), [])
                self.assertEqual(len(self.pool.status("alice")["jobs"]), 0)
            finally:
                release.set()
                self.client.coordinator._stopped.set()
                ticker.join(timeout=2.5)

    def test_persisted_finishing_task_releases_bound_preflight_after_coordinator_restart(self):
        from mindie_coordinator.service import CoordinatorService

        self.client.coordinator._async_progress = True
        started = threading.Event()
        release = threading.Event()
        def delayed(*args, **kwargs):
            started.set()
            self.assertTrue(release.wait(5))

        self.backend.preflight = delayed
        admitted = self.client.run("true", preflight="check-before-launch")
        self.assertTrue(started.wait(3))
        reply = self.client.finish()
        self.assertEqual(reply["state"], "finishing")
        release.set()
        lock = self.client.coordinator._lock_for("execution", admitted["execution_id"])
        deadline = time.time() + 5
        while lock.locked() and time.time() < deadline:
            time.sleep(0.02)
        self.assertFalse(lock.locked())
        session_id = self.context["session"]["id"]
        with self.store.transaction() as db:
            session = self.store.get(db, "session", session_id)
        self.assertEqual(session["state"], "finishing")
        self.assertEqual(session["finish"]["user"], "alice")
        rows = self.store.executions(session_id)
        self.assertEqual(rows[0]["phase"], "cancelled")
        remote = rows[0]["remote_session"]["id"]
        self.assertTrue(self.pool.session_bindings("alice", remote))
        self.assertEqual(len(self.pool.status("alice")["jobs"]), 0)
        restarted = CoordinatorService(
            self.client.coordinator.state_dir, pool=self.pool, backend=self.backend,
        )
        self.assertIn(str(self.store.state_dir), restarted._session_dirs)
        restarted._dispatch_progress()
        with self.store.transaction() as db:
            session = self.store.get(db, "session", session_id)
        self.assertEqual(session["state"], "finished")
        self.assertEqual(self.pool.session_bindings("alice", remote), [])
        self.assertEqual(len(self.pool.status("alice")["jobs"]), 0)

    def test_cancelled_preparation_persists_first_binding_and_finish_recovers_write_gap(self):
        service = self.client.coordinator
        original = service._prepare_role
        prepared = []
        def prepare(store, user, row, role, environment, donor):
            result = original(store, user, row, role, environment, donor)
            persisted = store.executions(row["session_id"])[0]
            self.assertEqual(persisted["prepared_bindings"][role["name"]], result["binding"])
            prepared.append(role["name"])
            persisted["cancel_requested"] = True
            store.save_execution(persisted)
            return result
        with mock.patch.object(service, "_prepare_role", side_effect=prepare):
            reply = self.client.run("true", sources={}, topology={"roles": [
                {"name": "first", "npu_count": 0}, {"name": "second", "npu_count": 0}]})
        self.assertEqual(reply["state"], "cancelled")
        self.assertGreaterEqual(len(prepared), 1)
        self.assertLessEqual(len(prepared), 2)
        row = self.store.executions(self.context["session"]["id"])[0]
        remote = row["remote_session"]["id"]
        self.assertEqual(len(self.pool.session_bindings("alice", remote)), len(prepared))
        self.assertEqual(self.pool.status("alice")["jobs"], [])
        # Simulate losing the local role save after the pool committed its
        # atomic binding. The already persisted execution session owns it.
        row.pop("prepared_bindings")
        self.store.save_execution(row)
        from mindie_coordinator.service import CoordinatorService
        restarted = CoordinatorService(service.state_dir, pool=self.pool, backend=self.backend)
        result = restarted.finish(str(self.store.state_dir), "alice", self.context["session"]["id"])
        self.assertEqual(result["state"], "finished")
        self.assertEqual(self.pool.session_bindings("alice", remote), [])

    def test_recovered_preparation_reuses_its_binding_without_replacing_sources(self):
        from mindie_coordinator.provision import prepare_task_environment
        with mock.patch("mindie_coordinator.provision.prepare_task_environment", wraps=prepare_task_environment) as prepare:
            reply = self.client.run("true", sources={}, resources={"npu_count": 0})
        self.assertEqual(reply["state"], "running")
        row = self.store.executions(self.context["session"]["id"])[0]
        with mock.patch.object(self.backend, "prepare_task_root", side_effect=AssertionError("must not prepare again")), \
             mock.patch.object(self.backend, "inspect", side_effect=AssertionError("bound replay must not reprobe")):
            recovered = prepare_task_environment(*prepare.call_args.args, **prepare.call_args.kwargs)
        self.assertEqual(recovered["binding"], row["roles"][0]["binding"])
