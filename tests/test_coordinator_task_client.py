"""Task-client, session, and client-attachment tests (split from test_coordinator.py)."""
from __future__ import annotations

import copy
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from mindie_coordinator.execution_sources import capture_sources
from mindie_coordinator.ready_runtime import RuntimePool

from test_coordinator import (
    Backend,
    FakeShell,
    PoolTests as _PoolTests,
    TaskClientTests as _TaskClientTests,
    _init_git_workspace,
    runtime_spec,
)


class PoolSessionTests(_PoolTests):
    """Session, checkout, and task-root attachment behavior exercised through RuntimePool."""

    def test_managed_registration_binds_atomically_and_replays_one_probe(self):
        session = self.pool.session_open("alice", "prepared-session", {})
        spec = runtime_spec(3)
        self.backend.calls.clear()
        with ThreadPoolExecutor(2) as workers:
            results = list(workers.map(lambda _: self.pool._register_checkout(
                "prepared", spec, "alice", session["id"], "prepared-request"), range(2)))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0]["state"], "bound")
        self.assertEqual(sum(kind == "inspect" for kind, _ in self.backend.calls), 1)
        self.assertEqual(self.pool.session_bindings("alice", session["id"]), [results[0]["binding"]])
        restarted = RuntimePool(self.root / "manager", self.backend)
        self.assertEqual(restarted._register_checkout(
            "prepared", spec, "alice", session["id"], "prepared-request"), results[0])
        with self.assertRaisesRegex(ValueError, "different checkout"):
            self.pool._register_checkout("prepared", {**spec, "python": "/different/python"},
                                         "alice", session["id"], "prepared-request")
        self.pool.return_runtime("alice", results[0]["binding"]["id"])
        with self.assertRaisesRegex(ValueError, "returned binding"):
            self.pool._register_checkout("prepared", spec, "alice", session["id"], "prepared-request")

    def test_public_checkout_auto_selection_preserves_null_and_default_intents(self):
        for index, selection in enumerate(({}, {"runtime_id": None})):
            with self.subTest(selection=selection):
                self.pool.register("runtime-a", runtime_spec(1))
                session = self.pool.session_open("alice", f"auto-session-{index}", {})
                with mock.patch.object(self.backend, "inspect", wraps=self.backend.inspect) as probe:
                    first = self.pool.checkout("alice", session["id"], "profile-a", f"auto-{index}", **selection)
                    again = self.pool.checkout("alice", session["id"], "profile-a", f"auto-{index}", **selection)
                self.assertEqual(first, again)
                self.assertEqual(first["intent"]["runtime_id"], selection.get("runtime_id", ""))
                self.assertEqual(probe.call_count, 1)
                self.pool.return_runtime("alice", first["id"])

    def test_managed_registration_does_not_publish_unverified_or_half_bound_root(self):
        session = self.pool.session_open("alice", "prepared-session", {})
        original = self.backend.inspect
        def probe(*args, **kwargs):
            self.assertFalse(any(row["runtime_id"] == "prepared" for row in self.pool.catalog()))
            self.assertEqual(self.pool.session_bindings("alice", session["id"]), [])
            return original(*args, **kwargs)
        with mock.patch.object(self.backend, "inspect", side_effect=probe), \
             mock.patch.object(self.backend, "host", side_effect=TimeoutError("reservation failed")):
            with self.assertRaisesRegex(TimeoutError, "reservation failed"):
                self.pool._register_checkout("prepared", runtime_spec(3), "alice", session["id"], "prepared-request")
        self.assertFalse(any(row["runtime_id"] == "prepared" for row in self.pool.catalog()))
        self.assertEqual(self.pool.session_bindings("alice", session["id"]), [])

    def test_checkout_idempotency_authorization_and_no_silent_provision(self):
        binding = self.bind("alice", self.root / "a")
        self.assertEqual(self.bind("alice", self.root / "a")["id"], binding["id"])
        with self.assertRaises(PermissionError):
            self.pool.return_runtime("bob", binding["id"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.pool.register("duplicate", runtime_spec(1, user="alice"))
        session = self.pool.session_open("bob", "other", {"va": str(self.root / "b")})
        miss = self.pool.checkout("bob", session["id"], "missing", "missing")
        self.assertEqual(miss["status"], "cache_miss")
        self.assertFalse(miss["provisioning_started"])
        conflict = runtime_spec(2, user="bob", service_ports=[46001])
        with self.assertRaisesRegex(ValueError, "ports overlap"):
            self.pool.register("port-conflict", conflict)

    def test_return_requires_fresh_admin_verification(self):
        binding = self.bind("alice", self.root / "a")
        self.assertEqual(self.pool.return_runtime("alice", binding["id"])["runtime_state"], "needs_repair")
        session = self.pool.session_open("bob", "bob-session", {"va": str(self.root / "b")})
        self.assertEqual(self.pool.checkout("bob", session["id"], "profile-a", "bob", binding["runtime_id"])["status"], "cache_miss")
        self.pool.register(binding["runtime_id"], runtime_spec(1, user="alice"))
        self.assertEqual(self.pool.checkout("bob", session["id"], "profile-a", "bob", binding["runtime_id"])["status"], "cache_miss")
        self.assertEqual(self.pool.checkout("alice", binding["intent"]["session"], "profile-a", "alice-again", binding["runtime_id"])["state"], "bound")

    def test_task_worktree_rebind_returns_idle_roots_but_preserves_live_work(self):
        binding = self.bind("alice", self.root / "a")
        job = self.managed("alice", binding)
        with self.assertRaisesRegex(ValueError, "resolve/release"):
            self.pool.session_open("alice", "same-session-name", {"va": str(self.root / "new/worktree")})
        self.pool.managed_control("alice", job["id"], "stop")
        self.pool.managed_control("alice", job["id"])
        result = self.pool.session_open("alice", "same-session-name", {"va": str(self.root / "new/worktree")})
        self.assertEqual(result["id"], binding["intent"]["session"])
        self.assertEqual(self.pool.status("alice")["bindings"][0]["state"], "returned")

    def test_checkout_replay_of_a_returned_binding_fails_explicitly(self):
        binding = self.bind("alice", self.root / "a")
        self.assertEqual(self.bind("alice", self.root / "a")["id"], binding["id"])
        self.pool.return_runtime("alice", binding["id"])
        with self.assertRaisesRegex(ValueError, "returned binding"):
            self.bind("alice", self.root / "a")
        self.pool.register("runtime-a2", runtime_spec(2, user="alice"))
        fresh = self.pool.checkout("alice", binding["intent"]["session"], "profile-a", "checkout-2")
        self.assertEqual(fresh["state"], "bound")
        self.assertNotEqual(fresh["id"], binding["id"])
        # The returned runtime stays quarantined; the fresh request binds the
        # other ready root in the same user container.
        self.assertNotEqual(fresh["runtime_id"], binding["runtime_id"])
        self.assertEqual(fresh["container_name"], binding["container_name"])
        quarantined = next(row for row in self.pool.catalog() if row["runtime_id"] == binding["runtime_id"])
        self.assertEqual(quarantined["state"], "needs_repair")

    def test_checkout_marks_a_drifted_runtime_needs_repair_and_binds_the_next_candidate(self):
        original = self.backend.inspect

        def inspect(runtime, **kwargs):
            observed = original(runtime, **kwargs)
            if runtime.get("container_name") == "mindie-alice" and runtime["endpoint"]["cwd"].endswith("/1"):
                observed["build_key"] = "drifted"
            return observed

        self.backend.inspect = inspect
        self.pool.register("runtime-a2", runtime_spec(2, user="alice"))
        binding = self.bind("alice", self.root / "a")
        self.assertEqual(binding["runtime_id"], "runtime-a2")
        drifted = next(row for row in self.pool.catalog() if row["runtime_id"] == "runtime-a")
        self.assertEqual(drifted["state"], "needs_repair")
        self.assertIn("prepared environment changed", drifted["error"])
        session = self.pool.session_open("bob", "bob-session", {"va": str(self.root / "b")})
        self.assertEqual(self.pool.checkout("bob", session["id"], "profile-a", "bob", "runtime-a")["status"], "cache_miss")

    @unittest.skipIf(os.name == "nt", "executes Linux container imports and launch payload locally")
    def test_managed_launch_imports_bound_sources_from_task_root(self):
        import shlex

        root = self.root / "prepared role 空格"
        for repo, package, body in (
            ("vllm", "vllm", "SamplingParams = 'bound-vllm'\n"),
            ("vllm-ascend", "vllm_ascend", "marker = 'bound-ascend'\n"),
            ("support", "runtime_support", "marker = 'kept-support'\n"),
        ):
            package_dir = root / repo / package
            package_dir.mkdir(parents=True)
            (package_dir / "__init__.py").write_text(body)
        self.backend.attestation["launch_preamble"] = "export PYTHONPATH=" + shlex.quote(str(root / "support"))
        self.pool.register("runtime-a", runtime_spec(1, root=str(root), python=sys.executable))
        binding = self.bind("alice", self.root / "a")
        prepared = {}
        original_job = self.backend.job

        def record_spec(runtime, job_id, action, **params):
            if action == "prepare":
                prepared.update(params["spec"])
            return original_job(runtime, job_id, action, **params)

        code = ("import json; from vllm import SamplingParams; import vllm_ascend, runtime_support; "
                "print(json.dumps([SamplingParams, vllm_ascend.marker, runtime_support.marker]))")
        with mock.patch.object(self.backend, "job", side_effect=record_spec):
            job = self.pool.managed_start(
                "alice", binding["id"], "source-import", {"vllm": "a" * 40, "vllm-ascend": "b" * 40},
                "native-a", [0], 0, shlex.join([sys.executable, "-c", code]), {}, 60,
            )
        self.assertEqual(job["state"], "running")
        output = subprocess.check_output(
            ["bash", "-c", prepared["command"]], cwd=prepared["cwd"],
            env={**os.environ, **prepared["env"]}, text=True,
        )
        self.assertEqual(json.loads(output), ["bound-vllm", "bound-ascend", "kept-support"])
        from mindie_coordinator.backend import RemoteBackend

        shell = FakeShell({"outcome": "success"})
        RemoteBackend(shell=shell).preflight(binding, shlex.join([sys.executable, "-c", code]), {})
        target, command, _ = shell.calls[-1]
        output = subprocess.check_output(["bash", "-c", command], cwd=target["cwd"], text=True)
        self.assertEqual(json.loads(output), ["bound-vllm", "bound-ascend", "kept-support"])

    def test_two_roots_in_one_container_run_concurrently_and_stop_is_scoped(self):
        second_root = self.pool.register("runtime-a2", runtime_spec(2, user="alice", python="/opt/alice/venvs/root-2/bin/python"))
        alice = self.bind("alice", self.root / "a")
        other_session = self.pool.session_open("alice", "second-task", {"vllm": str(self.root / "vllm"), "vllm-ascend": str(self.root / "va")})
        other = self.pool.checkout("alice", other_session["id"], "profile-a", "other-root", "runtime-a2")
        first = self.managed("alice", alice, device=0, request_id="root-1")
        second = self.managed("alice", other, device=1, request_id="root-2")
        self.assertEqual((first["state"], second["state"]), ("running", "running"))
        self.assertEqual(alice["container_name"], other["container_name"])
        self.assertEqual(alice["endpoint"]["port"], other["endpoint"]["port"])
        self.assertNotEqual(alice["endpoint"]["cwd"], other["endpoint"]["cwd"])
        self.assertEqual(other["python"], "/opt/alice/venvs/root-2/bin/python")
        self.pool.managed_control("alice", first["id"], "stop")
        ended = self.pool.managed_control("alice", first["id"])
        self.assertEqual(ended["state"], "cancelled")
        peer = self.pool.status("alice")["jobs"]
        live = next(row for row in peer if row["id"] == second["id"])
        self.assertEqual(live["state"], "running")
        self.assertFalse(self.backend.jobs[second["job_id"]]["quiet"])
        self.assertEqual(second_root["container_name"], "mindie-alice")
        self.assertIn((46001, "container_ssh", "ssh.alice"), self._host_ports())


class TaskClientOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        from mindie_coordinator.agent_session import AgentSessions
        from mindie_coordinator.task_client import TaskClient
        self.TaskClient = TaskClient
        self.store = AgentSessions(self.root / "sessions")
        self.task_a = self.store.attach("codex", "native-task-a", str(self.root))
        self.task_b = self.store.attach("codex", "native-task-b", str(self.root))
        self.assertNotEqual(self.task_a["session"]["id"], self.task_b["session"]["id"])
        self.row = self.store.execution(self.task_a, "owned-fixture", {"command": "true"})
        self.row.update(phase="cancelled", user="alice", roles=[])
        self.store.save_execution(self.row)
        self.service = mock.Mock()
        self.owner = TaskClient(self.task_a["context_file"], user="alice", service=self.service)
        self.foreign = TaskClient(self.task_b["context_file"], user="alice", service=self.service)

    def tearDown(self):
        self.temp.cleanup()

    def test_foreign_read_stop_and_target_are_rejected_before_the_service(self):
        for action in ("status", "tail", "stop", "target"):
            with self.subTest(action=action):
                self.service.reset_mock()
                with self.assertRaisesRegex(ValueError, "another MindIE task"):
                    self.foreign.observe(self.row["id"], action)
                self.service.advance.assert_not_called()
                self.service.finish.assert_not_called()
        self.service.reset_mock()
        with self.assertRaisesRegex(ValueError, "another MindIE task"):
            self.foreign.target(self.row["id"])
        self.service.advance.assert_not_called()
        with self.store.transaction() as db:
            latest = self.store.get(db, "execution", self.row["id"])
        self.assertEqual(latest["phase"], "cancelled")
        self.assertFalse(latest.get("cancel_requested"))

    def test_owner_and_same_task_parent_child_may_observe(self):
        self.service.advance.return_value = {"execution_id": self.row["id"], "state": "cancelled"}
        own = self.owner.observe(self.row["id"])
        self.assertEqual(own["state"], "cancelled")
        self.service.advance.assert_called_once()
        child_ctx = self.store.attach(
            "codex", "native-task-a-child", str(self.root),
            parent_context=self.task_a["context_file"],
        )
        self.assertEqual(child_ctx["session"]["id"], self.task_a["session"]["id"])
        child = self.TaskClient(child_ctx["context_file"], user="alice", service=self.service)
        self.service.reset_mock()
        child_reply = child.observe(self.row["id"])
        self.assertEqual(child_reply["state"], "cancelled")
        self.service.advance.assert_called_once()
        associated = self.store.attach(
            "codex", "native-task-a-explicit", str(self.root),
            association=self.task_a["context_file"],
        )
        self.assertEqual(associated["session"]["id"], self.task_a["session"]["id"])
        peer = self.TaskClient(associated["context_file"], user="alice", service=self.service)
        self.service.reset_mock()
        peer_reply = peer.observe(self.row["id"], "stop")
        self.assertEqual(peer_reply["state"], "cancelled")
        self.service.advance.assert_called_once()

    def test_unknown_and_malformed_ids_fail_without_the_service(self):
        with self.assertRaisesRegex(ValueError, "invalid local execution id"):
            self.owner.observe("not-an-id")
        self.service.advance.assert_not_called()
        with self.assertRaisesRegex(ValueError, "unknown MindIE execution"):
            self.owner.observe("0" * 64)
        self.service.advance.assert_not_called()
        with self.assertRaisesRegex(ValueError, "invalid local execution id"):
            self.owner.target("not-an-id")
        self.service.advance.assert_not_called()


class TaskClientTests(_TaskClientTests):
    """End-to-end TaskClient admission, observation, and execution-root behavior."""

    def test_explicit_shared_device_reaches_host_and_stop_preserves_external_and_peer_work(self):
        self.backend.busy = [0]
        shared = self.client.run("serve-shared", sources={}, topology={"host": "192.0.2.1"},
                                 resources={"devices": [0], "allow_external_busy": True, "service_port": 0})
        self.assertEqual(shared["state"], "running", shared)
        self.assertEqual(shared["target"]["environment"]["ASCEND_RT_VISIBLE_DEVICES"], "0")
        self.assertTrue(shared["target"]["allow_external_busy"])
        self.assertTrue(shared["target"]["launch_observation"]["allow_external_busy"])
        self.assertTrue(shared["assignment"]["roles"][0]["allow_external_busy"])
        with self.pool.transaction() as db:
            run = self.pool.rows(db, "run")[0]
        self.assertTrue(run["intent"]["allow_external_busy"])
        self.assertEqual(run["task"]["requested_count"], 1)
        peer = self.client.run("serve-peer", sources={}, resources={"devices": [1]})
        self.assertEqual(peer["state"], "running", peer)
        stopped = self.client.observe(shared["execution_id"], "stop")
        self.assertEqual(stopped["state"], "cancelled")
        self.assertTrue(stopped["resources_released"])
        self.assertEqual(self.backend.busy, [0])
        self.assertEqual(self.client.observe(peer["execution_id"])["state"], "running")
        self.assertEqual(self.backend.calls.count(("job", "stop")), 1)

    def test_completed_preparation_failure_is_terminal_without_repeating_work(self):
        from mindie_coordinator.parity_support import RemoteCommandError
        service = self.client.coordinator
        prepare = mock.Mock(side_effect=RemoteCommandError(1, "incompatible torch build"))
        with mock.patch.object(service, "_place_or_prepare", prepare):
            result = self.client.run("serve-mismatched-source")
            self.assertEqual(result["state"], "failed")
            self.assertTrue(result["resources_released"])
            self.assertIn("incompatible torch build", Path(result["error_ref"]).read_text())
            service._dispatch_progress()
            observed = self.client.observe(result["execution_id"])
        self.assertEqual(observed["state"], "failed")
        prepare.assert_called_once()

    def test_lost_preparation_transport_keeps_unknown_state(self):
        from mindie_coordinator.parity_support import RemoteCommandError
        with mock.patch.object(self.client.coordinator, "_place_or_prepare",
                               side_effect=RemoteCommandError(255, "connection lost")):
            result = self.client.run("serve-over-unavailable-transport")
        self.assertEqual(result["state"], "uncertain")
        self.assertFalse(result["resources_released"])

    def test_preflight_failure_keeps_full_error_and_never_allocates(self):
        from mindie_coordinator.managed_execution import ExecutionRequestError
        message = "unknown CLI option " + "detail " * 200
        self.backend.preflight = mock.Mock(side_effect=ExecutionRequestError(message))
        result = self.client.run("serve", preflight="parse-only")
        self.assertEqual(result["state"], "failed")
        self.assertIn(message, Path(result["error_ref"]).read_text())
        self.assertTrue(result["resources_released"])
        self.assertEqual(result["progress"]["step"], "preflight")
        with self.pool.transaction() as db:
            self.assertEqual(self.pool.rows(db, "job"), [])
            self.assertEqual(self.pool.rows(db, "run"), [])

    def test_business_preflight_changed_view_is_rejected_before_payload(self):
        def preflight(binding, command, env):
            self.backend.prepared[binding['endpoint']['cwd']]['build_key'] = 'changed-by-user-preflight'
        self.backend.preflight = preflight
        result = self.client.run("must-not-execute", sources={}, preflight="change-view", resources={"npu_count": 0})
        self.assertEqual(result["state"], "failed")
        self.assertTrue(result["resources_released"])
        role = self.store.executions(self.context["session"]["id"])[0]["roles"][0]
        self.assertIn("runtime changed before launch", role["observation"]["error"])
        self.assertEqual(role["observation"]["lease_state"], "cancelled")
        self.assertEqual(self.backend.jobs, {})

    def test_run_observe_and_target_expose_user_container_and_selected_python(self):
        reply = self.client.run("true")
        self.assertEqual(reply["state"], "running")
        self.assertTrue(reply["target"]["live"])
        self.assertEqual(reply["target"]["user"], "alice")
        self.assertEqual(reply["target"]["container_name"], "mindie-alice")
        prepared = self.backend.prepared[reply["target"]["endpoint"]["cwd"]]
        self.assertEqual(reply["target"]["python"], prepared["python"])
        self.assertNotEqual(reply["target"]["endpoint"]["cwd"], "/vllm-workspace/alice/1")
        observed = self.client.observe(reply["execution_id"], "target")
        self.assertEqual(observed["target"]["runtime_id"], reply["target"]["runtime_id"])
        self.assertEqual(self.client.target(reply["execution_id"])["container_id"], "cid-mindie-alice")

    def test_running_execution_stays_running_during_background_probe(self):
        started = self.client.run("true")
        self.assertEqual(started["state"], "running")
        states = []
        managed_control = self.pool.managed_control

        def observe_during_probe(user, job_id, action, **kwargs):
            if action == "status":
                states.append(self.client.observe(started["execution_id"])["state"])
            return managed_control(user, job_id, action, **kwargs)

        with mock.patch.object(self.pool, "managed_control", side_effect=observe_during_probe):
            advanced = self.client.coordinator.advance(
                str(self.store.state_dir), "alice", started["execution_id"], action="progress")

        self.assertTrue(states)
        self.assertEqual(set(states), {"running"})
        self.assertEqual(advanced["state"], "running")

    def test_finish_closes_admission_before_a_new_run(self):
        first = self.client.run("true")
        self.assertEqual(first["state"], "running")
        finished = self.client.finish()
        self.assertEqual(finished["state"], "finished")
        with self.assertRaisesRegex(ValueError, "resume the task"):
            self.client.run("true")

    def test_unknown_observation_retains_ownership_then_retry_completes(self):
        reply = self.client.run("true")
        original = self.backend.job

        def once_unknown(runtime, identifier, action, **params):
            if action == "status":
                return {"state": "uncertain", "quiet": False, "reason": "probe failed"}
            return original(runtime, identifier, action, **params)

        self.backend.job = once_unknown
        held = self.client.observe(reply["execution_id"])
        self.assertEqual(held["state"], "uncertain")
        self.backend.job = original
        recovered = self.client.observe(reply["execution_id"])
        self.assertEqual(recovered["state"], "running")
        self.assertEqual(recovered["execution_id"], reply["execution_id"])

    def test_same_task_gets_a_new_binding_and_root_for_each_execution(self):
        first = self.client.run("true")
        self.client.observe(first["execution_id"], "stop")
        second = self.client.run("sleep 1")
        self.assertEqual(second["state"], "running")
        self.assertNotEqual(second["target"]["endpoint"]["cwd"], first["target"]["endpoint"]["cwd"])
        self.assertNotEqual(second["target"]["binding_id"], first["target"]["binding_id"])
        self.assertNotEqual(second["target"]["runtime_id"], first["target"]["runtime_id"])
        other_context = self.store.attach("codex", "native-alice-2", str(self.root))
        from mindie_coordinator.task_client import TaskClient
        other = TaskClient(other_context["context_file"], pool=self.pool, user="alice")
        stolen = other.run("true")
        stolen_cwd = (stolen.get("target") or {}).get("endpoint", {}).get("cwd")
        self.assertNotEqual(stolen_cwd, second["target"]["endpoint"]["cwd"])

    def test_queued_execution_advances_after_frontend_is_gone(self):
        from mindie_coordinator.task_client import TaskClient
        empty = RuntimePool(self.root / "manager-empty", self.backend)
        client = TaskClient(self.context["context_file"], pool=empty, user="alice")
        queued = client.run("true")
        self.assertEqual(queued["state"], "waiting_for_runtime")
        execution_id = queued["execution_id"]
        self._seed(empty, "runtime-late", runtime_spec(1, user="alice", recipe="rc"))
        client.coordinator.reconcile()
        advanced = client.observe(execution_id)
        self.assertEqual(advanced["state"], "running")
        self.assertTrue(advanced["target"]["live"])

    def test_two_role_request_runs_both_or_neither(self):
        self._seed(self.pool, "runtime-b-host", runtime_spec(3, user="alice", host="192.0.2.8", recipe="rc"))
        topology = {"distinct_hosts": True, "roles": [{"name": "prefill", "npu_count": 1, "command": "run-prefill"},
                              {"name": "decode", "npu_count": 1, "command": "run-decode"}]}
        both = self.client.run("unused", topology=topology)
        self.assertEqual(both["state"], "running")
        self.assertEqual(len(both["roles"]), 2)
        self.assertEqual({row["name"] for row in both["roles"]}, {"prefill", "decode"})
        self.assertEqual({row["state"] for row in both["roles"]}, {"running"})
        hosts = {row["host"] for row in both["assignment"]["roles"]}
        self.assertEqual(len(hosts), 2)
        lone = RuntimePool(self.root / "manager-one-host", self.backend)
        self._seed(lone, "only-a", runtime_spec(1, user="alice", recipe="rc"))
        from mindie_coordinator.task_client import TaskClient
        other_ctx = self.store.attach("codex", "native-alice-group", str(self.root))
        client = TaskClient(other_ctx["context_file"], pool=lone, user="alice")
        neither = client.run("unused", topology={**topology, "distinct_hosts": True})
        self.assertNotEqual(neither["state"], "running")
        self.assertTrue(neither["state"] in {"waiting_for_runtime", "waiting", "queued"})
        self.assertIsNone(neither.get("roles"))

    def test_incompatible_environment_is_not_silently_used(self):
        missed = self.client.run("true", environment={"recipe": "stable"})
        self.assertEqual(missed["state"], "waiting_for_runtime")
        self.assertIn("environment", missed.get("reason") or "")

    def test_named_service_reconnects_same_spec_and_restart_replaces(self):
        first = self.client.run("serve-vllm", service="vllm", timeout_seconds=None)
        self.assertEqual(first["state"], "running")
        again = self.client.run("serve-vllm", service="vllm", timeout_seconds=None)
        self.assertEqual(again["execution_id"], first["execution_id"])
        with self.assertRaisesRegex(ValueError, "different inputs: command"):
            self.client.run("serve-other", service="vllm", timeout_seconds=None)
        replaced = self.client.run("serve-other", service="vllm", timeout_seconds=None, restart=True)
        self.assertNotEqual(replaced["execution_id"], first["execution_id"])
        self.assertEqual(replaced["state"], "running")
        same = self.client.run("serve-other", service="vllm", timeout_seconds=None, restart=True)
        self.assertNotEqual(same["execution_id"], replaced["execution_id"])
        self.assertEqual(same["state"], "running")

    def test_target_refreshes_owned_job_after_pool_finishes(self):
        reply = self.client.run("true")
        self.assertTrue(reply["target"]["live"])
        job_id = self.store.executions(self.context["session"]["id"])[0]["roles"][0]["observation"]["job_id"]
        self.backend.jobs[job_id].update(state="succeeded", quiet=True, result={"state": "succeeded", "exit_code": 0})
        target = self.client.target(reply["execution_id"])
        self.assertEqual(target["state"], "succeeded")
        self.assertFalse(target["live"])

    def test_finishing_attach_does_not_reopen_admission(self):
        first = self.client.run("true")
        self.assertEqual(first["state"], "running")
        with self.store.transaction() as db:
            session = self.store.get(db, "session", self.context["session"]["id"])
            session["state"] = "finishing"
            self.store.put(db, "session", session)
        with self.assertRaisesRegex(ValueError, "wait for the coordinator to complete cleanup"):
            self.store.attach("codex", "native-alice", str(self.root))
        retry = self.client.finish()
        self.assertIn(retry["state"], {"finished", "finishing"})
        if retry["state"] == "finished":
            reopened = self.store.attach("codex", "native-alice", str(self.root))
            self.assertEqual(reopened["session"]["state"], "open")

    def test_register_empty_root_is_rejected_when_prepared_boundary_enforced(self):
        with self.assertRaisesRegex(ValueError, "ready-profile"):
            self.pool.register("empty-root", runtime_spec(9, user="alice", root="/empty/unprepared"))

    def test_prepare_creates_isolated_interpreter_not_donor_python(self):
        self.bind_native_sources()
        first = self.client.run("true")
        donor_python = first["target"]["python"]
        other_context = self.store.attach("codex", "native-alice-prep", str(self.root))
        self.store.bind_sources(other_context, {"vllm": str(self.root / "vllm"), "vllm-ascend": str(self.root / "vllm-ascend")})
        from mindie_coordinator.task_client import TaskClient
        other = TaskClient(other_context["context_file"], pool=self.pool, user="alice")
        other.coordinator.sync_binding = lambda *args, **kwargs: {"vllm": "a" * 40, "vllm-ascend": "b" * 40}
        prepared = other.run("true")
        self.assertEqual(prepared["state"], "running")
        self.assertNotEqual(prepared["target"]["python"], donor_python)
        self.assertTrue(str(prepared["target"]["python"]).endswith("/.venv/bin/python"))
        self.assertIn("/executions/", prepared["target"]["endpoint"]["cwd"])
        self.assertNotEqual(prepared["target"]["endpoint"]["cwd"], first["target"]["endpoint"]["cwd"])

    def test_resumed_task_checks_out_returned_runtime_with_fresh_identity(self):
        previous = self.client.run("true")
        spec = runtime_spec(1, user="alice", recipe="rc")
        for profile in ("profile-a", "profile-updated"):
            with self.subTest(profile=profile):
                self.assertEqual(self.client.finish()["state"], "finished")
                self.backend.mark_prepared(spec, profile_key=profile)
                self.pool.register("runtime-a", spec)
                self.store.attach("codex", "native-alice", str(self.root))
                current = self.client.run("true")
                self.assertEqual(current["state"], "running", current.get("error"))
                self.assertNotEqual(current["target"]["runtime_id"], previous["target"]["runtime_id"])
                self.assertNotEqual(current["target"]["binding_id"], previous["target"]["binding_id"])
                previous = current

    def test_role_host_constrains_auto_preparation(self):
        first = self.client.run("true")
        self.assertEqual(first["target"]["endpoint"]["host"], "192.0.2.1")
        missed = self.client.run(
            "true",
            topology={"roles": [{"name": "decode", "npu_count": 1, "host": "192.0.2.8"}]},
        )
        self.assertEqual(missed["state"], "waiting_for_runtime")
        self.assertIsNone(missed.get("target"))

    def test_missing_hardware_facts_are_not_a_match(self):
        missed = self.client.run("true", environment={"cann": "8.0.0"})
        self.assertEqual(missed["state"], "waiting_for_runtime")

    def test_two_role_success_and_mixed_failure_aggregate(self):
        self._seed(self.pool, "runtime-b-host", runtime_spec(3, user="alice", host="192.0.2.8", recipe="rc"))
        topology = {"roles": [{"name": "prefill", "npu_count": 1, "command": "run-prefill"},
                              {"name": "decode", "npu_count": 1, "command": "run-decode"}]}
        both = self.client.run("unused", topology=topology)
        self.assertEqual(both["state"], "running")
        row = self.store.executions(self.context["session"]["id"])[-1]
        for role in row["roles"]:
            self.backend.jobs[role["observation"]["job_id"]].update(
                state="succeeded", quiet=True, result={"state": "succeeded", "exit_code": 0})
        succeeded = self.client.observe(both["execution_id"])
        self.assertEqual(succeeded["state"], "succeeded")
        again = self.client.run("unused", topology=topology)
        self.assertEqual(again["state"], "running")
        row = self.store.executions(self.context["session"]["id"])[-1]
        jobs = [role["observation"]["job_id"] for role in row["roles"]]
        self.backend.jobs[jobs[0]].update(state="succeeded", quiet=True, result={"state": "succeeded", "exit_code": 0})
        self.backend.jobs[jobs[1]].update(state="failed", quiet=True, result={"state": "failed", "exit_code": 1})
        mixed = self.client.observe(again["execution_id"])
        self.assertEqual(mixed["state"], "failed")

    def test_same_task_materializes_independent_execution_roots_concurrently(self):
        self.bind_native_sources()
        started = threading.Event()
        release = threading.Event()
        syncs = []
        original = self.client.coordinator.sync_binding

        def blocked(*args, **kwargs):
            syncs.append(threading.current_thread().name)
            if len(syncs) == 1:
                started.set()
                self.assertTrue(release.wait(30))
            return original(*args, **kwargs)

        self.client.coordinator.sync_binding = blocked
        first = {}

        def run_first():
            try:
                first["reply"] = self.client.run("true")
            except BaseException as exc:
                first["error"] = exc
                started.set()

        worker = threading.Thread(target=run_first)
        worker.start()
        try:
            self.assertTrue(started.wait(30))
            if "error" in first:
                raise first["error"]
            second = self.client.run("sleep 1")
            self.assertEqual(second["state"], "running")
            self.assertEqual(len(syncs), 2)
        finally:
            release.set()
            worker.join(60)
        self.assertFalse(worker.is_alive(), "materialization worker must finish before fixture cleanup")
        if "error" in first:
            raise first["error"]
        self.assertEqual(first["reply"]["state"], "running")

    def test_same_host_roles_keep_separate_roots_and_literal_env(self):
        self._seed(self.pool, "runtime-a2", runtime_spec(2, user="alice", recipe="rc"))
        topology = {"roles": [
            {"name": "prefill", "npu_count": 1, "host": "192.0.2.1", "command": "run-prefill",
             "env": {"ROLE": "prefill"}},
            {"name": "decode", "npu_count": 1, "host": "192.0.2.1", "command": "run-decode",
             "env": {"ROLE": "decode"}},
        ]}
        both = self.client.run("unused", topology=topology)
        self.assertEqual(both["state"], "running")
        self.assertEqual({row["host"] for row in both["roles"]}, {"192.0.2.1"})
        self.assertEqual(len({row["root"] for row in both["roles"]}), 2)
        by_name = {row["name"]: row for row in both["roles"]}
        self.assertEqual(by_name["prefill"]["env"]["ROLE"], "prefill")
        self.assertEqual(by_name["decode"]["env"]["ROLE"], "decode")
        self.assertTrue(by_name["prefill"]["target"]["live"])
        self.assertNotEqual(by_name["prefill"]["endpoint"]["cwd"], by_name["decode"]["endpoint"]["cwd"])
        tailed = self.client.observe(both["execution_id"], "tail")
        self.assertEqual(len(tailed["roles"]), 2)
        self.assertTrue(all(item.get("tail") or item.get("stdout") for item in tailed["roles"]))
        decode = self.client.observe(both["execution_id"], "tail", role="decode")
        self.assertIn("tail", decode)
        self.assertEqual(decode["roles"][0]["name"], "decode")

    def test_explicit_distinct_hosts_still_require_two_hosts(self):
        missed = self.client.run(
            "unused",
            topology={"distinct_hosts": True,
                      "roles": [{"name": "prefill", "npu_count": 1}, {"name": "decode", "npu_count": 1}]},
        )
        self.assertEqual(missed["state"], "waiting_for_runtime")

    def test_tail_returns_logs_after_the_job_is_terminal(self):
        reply = self.client.run("true")
        job_id = self.store.executions(self.context["session"]["id"])[0]["roles"][0]["observation"]["job_id"]
        self.backend.jobs[job_id]["stdout"] = "finished-log-line"
        self.backend.jobs[job_id].update(state="succeeded", quiet=True, result={"state": "succeeded", "exit_code": 0})
        tailed = self.client.observe(reply["execution_id"], "tail")
        self.assertEqual(tailed["state"], "succeeded")
        self.assertIn("finished-log-line", tailed.get("tail") or "")

    def test_role_env_cannot_override_reserved_keys(self):
        with self.assertRaisesRegex(ValueError, "managed"):
            self.client.run("true", topology={"roles": [{"name": "prefill", "env": {"MINDIE_PYTHON": "/x"}}]})
