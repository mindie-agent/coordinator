"""Runtime-pool, ready-runtime, and lease tests (split from test_coordinator.py)."""
from __future__ import annotations

import copy
import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from unittest import mock

from mindie_coordinator.ready_runtime import RuntimePool

from test_coordinator import (
    Backend,
    CoordinationError,
    PoolTests as _PoolTests,
    _confirmed_free_probe,
    _init_git_workspace,
    runtime_spec,
)


class PoolTests(_PoolTests):
    def test_managed_binding_keeps_request_and_post_queue_drift_checks(self):
        session = self.pool.session_open("alice", "prepared-session", {})
        binding = self.pool._register_checkout(
            "prepared", runtime_spec(3), "alice", session["id"], "prepared-request")["binding"]
        original = copy.deepcopy(self.backend.attestation)
        self.backend.attestation["build_key"] = "drifted"
        with self.assertRaisesRegex(ValueError, "changed"):
            self.pool.request_run("alice", binding["id"], "request", {}, "native-a", [], 0)
        self.backend.attestation = original
        run = self.pool.request_run("alice", binding["id"], "request", {}, "native-a", [], 0)
        self.backend.attestation["build_key"] = "changed-after-queue"
        failed = self.pool.control("alice", run["id"], "preflight")
        self.assertEqual(failed["state"], "uncertain")
        self.assertIn("changed before launch", failed["error"])

    def test_managed_checks_changed_source_after_queue_without_preparing_payload(self):
        busy_binding = self.bind("bob", self.root / "bob")
        busy = self.request("bob", busy_binding)
        binding = self.bind("alice", self.root / "alice")
        source = self.root / "candidate.py"
        source.write_text("fixed input\n")
        original = self.backend.inspect
        def inspect(runtime, **kwargs):
            if source.read_text() != "fixed input\n":
                from mindie_coordinator.parity_support import RemoteCommandError
                raise RemoteCommandError(1, "runtime source differs from pinned snapshot: vllm")
            return original(runtime, **kwargs)
        with mock.patch.object(self.backend, "inspect", side_effect=inspect) as probe:
            job = self.pool.managed_start("alice", binding["id"], "managed", {"vllm": "a" * 40},
                                          "native-a", [0], 0, "must-not-execute", {})
            self.assertEqual(job["state"], "queued")
            probe.assert_not_called()
            source.write_text("changed while waiting\n")
            self.pool.control("bob", busy["id"], "cancel")
            rejected = self.pool.managed_control("alice", job["id"])
        self.assertEqual(probe.call_count, 1)
        self.assertEqual(rejected["state"], "failed")
        self.assertEqual(rejected["lease_state"], "cancelled")
        self.assertIn("differs from pinned snapshot", rejected["error"])
        self.assertEqual(self.backend.jobs, {})
        self.assertNotIn(("job", "prepare"), self.backend.calls)
        self.assertNotIn(("job", "go"), self.backend.calls)

    def test_managed_validation_transport_failure_releases_only_after_confirmed_cleanup(self):
        binding = self.bind("alice", self.root / "alice")
        original_host = self.backend.host
        def host(runtime, request):
            if request["action"] == "cancel":
                raise TimeoutError("host cancel unavailable")
            return original_host(runtime, request)
        with mock.patch.object(self.backend, "inspect", side_effect=TimeoutError("validation transport unavailable")), \
             mock.patch.object(self.backend, "host", side_effect=host):
            job = self.pool.managed_start("alice", binding["id"], "managed", {}, "native-a",
                                          [0], 0, "must-not-execute", {})
        self.assertEqual(job["state"], "releasing")
        self.assertEqual(job["lease_state"], "uncertain")
        self.assertEqual(job["preflight_error"], "validation transport unavailable")
        self.assertTrue(self.pool.peers())
        # A restart must retain the rejection; an unknown owned-process
        # observation cannot authorize releasing the host's grant.
        restarted = RuntimePool(self.root / "manager", self.backend)
        unknown = {"state": "uncertain", "quiet": False, "unknown": ["transport unavailable"]}
        with mock.patch.object(self.backend, "job", return_value=unknown) as observe:
            held = restarted.managed_control("alice", job["id"])
        self.assertEqual(held["state"], "stopping")
        self.assertNotIn(held["lease_state"], {"cancelled", "released", "expired"})
        self.assertEqual(observe.call_args.args[2], "stop")
        self.assertTrue(restarted.peers())
        failed = restarted.managed_control("alice", job["id"])
        self.assertEqual(failed["state"], "failed")
        self.assertEqual(failed["lease_state"], "cancelled")
        self.assertEqual(failed["error"], "validation transport unavailable")
        self.assertEqual(restarted.peers(), [])
        self.assertEqual(self.backend.jobs, {})
        self.assertNotIn(("job", "prepare"), self.backend.calls)
        self.assertNotIn(("job", "go"), self.backend.calls)

    def test_rejected_cpu_job_recovers_temporary_port_quarantine_without_activating(self):
        binding = self.bind("alice", self.root / "alice")
        original_host = self.backend.host
        def host(runtime, request):
            if request["action"] == "cancel":
                self.backend.listening = [48001]
            return original_host(runtime, request)
        with mock.patch.object(self.backend, "inspect", side_effect=ValueError("source differs")), \
             mock.patch.object(self.backend, "host", side_effect=host):
            job = self.pool.managed_start("alice", binding["id"], "managed", {}, "native-a",
                                          [], 0, "must-not-execute", {}, service_port=0)
        self.assertEqual(job["state"], "releasing")
        self.assertEqual(job["lease_state"], "orphaned_busy")
        self.backend.listening = []
        self.backend.calls.clear()
        rejected = self.pool.managed_control("alice", job["id"])
        self.assertEqual(rejected["state"], "failed")
        self.assertEqual(rejected["lease_state"], "released")
        self.assertEqual(rejected["error"], "source differs")
        self.assertEqual(self.backend.jobs, {})
        self.assertNotIn(("host", "heartbeat"), self.backend.calls)

    def test_two_management_roots_share_one_runtime_and_card_authority(self):
        with ThreadPoolExecutor(2) as workers:
            a, b = list(workers.map(lambda args: self.bind(*args), [("alice", self.root / "clone-a"), ("bob", self.root / "linked-b")]))
        self.assertNotEqual(a["runtime_id"], b["runtime_id"])
        arun, brun = self.request("alice", a), self.request("bob", b)
        self.assertEqual(arun["state"], "granted")
        self.assertEqual(brun["state"], "queued")
        self.assertEqual(self.pool.control("alice", arun["id"], "release")["state"], "released")
        restarted = RuntimePool(self.root / "manager", self.backend)
        restarted.tick()
        self.assertEqual(restarted.status("bob")["runs"][0]["state"], "granted")
        self.assertEqual(set(kind for kind, _ in self.backend.calls), {"inspect", "host"})

    def test_yield_acceptance_and_timeout_never_release_or_reassign(self):
        binding = self.bind("alice", self.root / "a")
        run = self.request("alice", binding)
        event = self.pool.message("bob", run["id"], "Could you release card 0 after this run?")
        self.pool.reply("alice", event["cursor"], "Yes, after completion")
        self.assertEqual(self.pool.status("alice")["runs"][0]["state"], "granted")
        self.assertEqual(len(self.pool.events("bob")["events"]), 1)
        with self.assertRaises(PermissionError):
            self.pool.reply("bob", event["cursor"], "unauthorized")
        self.backend.fail = True
        self.assertEqual(self.pool.control("alice", run["id"], "release")["state"], "uncertain")
        with self.assertRaises(ValueError):
            self.pool.return_runtime("alice", binding["id"])
        self.backend.fail = False
        self.backend.busy = [0]
        self.pool.control("alice", run["id"], "poll")
        self.assertEqual(self.pool.control("alice", run["id"], "release")["state"], "orphaned_busy")

    def test_lost_grant_reply_reconciles_same_task_after_restart(self):
        binding = self.bind("alice", self.root / "a")
        self.backend.fail_after = "acquire"
        run = self.request("alice", binding)
        self.assertEqual(run["state"], "uncertain")
        pool = RuntimePool(self.root / "manager", self.backend)
        recovered = pool.control("alice", run["id"], "poll")
        self.assertEqual(recovered["state"], "granted")
        self.assertEqual(recovered["task_id"], run["task_id"])

    def test_lost_submit_reply_and_initial_connection_failure_recover(self):
        binding = self.bind("alice", self.root / "a")
        self.backend.fail_after = "submit"
        run = self.request("alice", binding)
        self.assertEqual(run["state"], "uncertain")
        recovered = self.pool.control("alice", run["id"], "poll")
        self.assertEqual(recovered["state"], "granted")
        self.assertEqual(recovered["task_id"], run["task_id"])
        self.assertEqual(self.backend.calls.count(("host", "submit")), 1)
        record = json.loads((self.root / "manager/runs" / (run["id"] + ".json")).read_text())
        self.assertEqual(record["resources"]["state"], "granted")
        self.assertFalse(record["resources"]["released"])
        other = self.bind("bob", self.root / "b")
        self.backend.fail_after = "status"
        pending = self.request("bob", other)
        self.assertEqual(pending["state"], "pending")
        self.assertIsNone(pending["epoch"])
        recovered = self.pool.control("bob", pending["id"], "poll")
        self.assertEqual(recovered["state"], "queued")

    def test_first_submission_reuses_discovery_status(self):
        binding = self.bind("alice", self.root / "a")
        self.backend.calls.clear()
        run = self.request("alice", binding)
        self.assertEqual(run["state"], "granted")
        self.assertEqual([action for kind, action in self.backend.calls if kind == "host"],
                         ["status", "submit", "acquire"])

    def test_epoch_change_after_discovery_rejects_submit_without_creating_host_task(self):
        binding = self.bind("alice", self.root / "a")
        original = self.backend.host
        def host(runtime, request):
            reply = original(runtime, request)
            if request["action"] == "status" and "coordination_epoch" not in request:
                import sqlite3
                with closing(sqlite3.connect(self.backend.state / "192_0_2_1" / "coordinator.sqlite3")) as db, db:
                    db.execute("UPDATE meta SET value='new-epoch' WHERE key='coordination_epoch'")
            return reply
        with mock.patch.object(self.backend, "host", side_effect=host):
            run = self.request("alice", binding)
        self.assertEqual(run["state"], "uncertain")
        self.assertFalse(run["submitted"])
        self.assertIn("epoch changed", run["error"])
        tasks = original(runtime_spec(1), {"action": "status", "no_probe": True})["tasks"]
        self.assertFalse(any(task["task_id"] == run["task_id"] for task in tasks))

    def test_unsubmitted_request_expires_without_allocating_after_outage(self):
        import time
        binding = self.bind("alice", self.root / "a")
        self.backend.fail_after = "status"
        pending = self.request("alice", binding, queue_seconds=1)
        self.pool.clock = lambda: time.time() + 60
        expired = self.pool.control("alice", pending["id"], "poll")
        self.assertEqual(expired["state"], "expired")
        self.assertNotIn(("host", "submit"), self.backend.calls)

    def test_unsubmitted_pending_run_rejects_non_poll_actions_without_host_mutation(self):
        binding = self.bind("alice", self.root / "a")
        self.backend.fail_after = "status"
        pending = self.request("alice", binding)
        self.assertEqual(pending["state"], "pending")
        self.assertIsNone(pending["epoch"])

        calls_before = list(self.backend.calls)
        for action in ("preflight", "activate", "heartbeat", "release"):
            with self.subTest(action=action), self.assertRaisesRegex(
                ValueError, "unsubmitted pending execution; poll it first"
            ):
                self.pool.control("alice", pending["id"], action)
            self.assertEqual(self.backend.calls, calls_before)
            observed = self.pool.status("alice")["runs"][0]
            self.assertEqual(observed["state"], "pending")
            self.assertFalse(observed["submitted"])

    def test_unsubmitted_pending_run_can_cancel_without_submit(self):
        binding = self.bind("alice", self.root / "a")
        self.backend.fail_after = "status"
        pending = self.request("alice", binding)
        calls_before = len(self.backend.calls)

        cancelled = self.pool.control("alice", pending["id"], "cancel")

        self.assertEqual(cancelled["state"], "cancelled")
        self.assertNotIn(("host", "submit"), self.backend.calls[calls_before:])

    def test_native_refresh_is_forbidden_while_execution_is_unresolved(self):
        binding = self.bind("alice", self.root / "a")
        run = self.request("alice", binding)
        with self.assertRaises(ValueError):
            self.pool.refresh("alice", binding["id"])
        self.pool.control("alice", run["id"], "release")
        self.backend.attestation["build_key"] = "native-new"
        self.assertEqual(self.pool.refresh("alice", binding["id"])["build_key"], "native-new")

    def test_epoch_change_fails_closed_and_host_enforces_fence_epoch(self):
        binding = self.bind("alice", self.root / "a")
        run = self.request("alice", binding)
        import sqlite3
        with closing(sqlite3.connect(self.backend.state / "192_0_2_1" / "coordinator.sqlite3")) as db, db:
            db.execute("UPDATE meta SET value='new-epoch' WHERE key='coordination_epoch'")
        result = self.pool.control("alice", run["id"], "release")
        self.assertEqual(result["state"], "uncertain")
        with self.assertRaises(CoordinationError):
            self.backend.host(runtime_spec(1), {"action": "cancel", "task_id": run["task_id"], "coordination_epoch": run["epoch"]})

    def test_repeated_free_probe_requires_visibility_in_every_sample(self):
        samples = iter([{"status": "ok", "devices": [1], "busy": {}}, {"status": "ok", "devices": [0, 1], "busy": {}}])
        result = _confirmed_free_probe(samples=2, interval_seconds=0, probe=lambda: next(samples))
        self.assertEqual(result["free"], [1])

    def test_managed_launch_injects_and_retains_verified_observation(self):
        binding = self.bind("alice", self.root / "a")
        prepared = []
        original = self.backend.job
        def capture(runtime, job_id, action, **params):
            if action == "prepare":
                prepared.append(params["spec"])
            return original(runtime, job_id, action, **params)
        with mock.patch.object(self.backend, "job", side_effect=capture):
            job = self.managed("alice", binding)
        receipt = json.loads(prepared[0]["env"]["MINDIE_EXECUTION_OBSERVATION"])
        self.assertEqual(receipt, job["launch_observation"])
        self.assertEqual(receipt["workspace_snapshot"]["vllm_commit"], "a" * 40)
        self.assertEqual(receipt["native_digest"]["build_key"], "native-a")
        self.pool.managed_control("alice", job["id"], "stop")
        ended = self.pool.managed_control("alice", job["id"])
        self.assertEqual(ended["launch_observation"], receipt)

    def test_managed_gate_renew_restart_and_stop_one_preserves_peer(self):
        a, b = self.bind("alice", self.root / "a"), self.bind("bob", self.root / "b")
        first, second = self.managed("alice", a), self.managed("bob", b, 1)
        self.assertEqual((first["state"], second["state"]), ("running", "running"))
        self.assertLess(self.backend.calls.index(("host", "activate")), self.backend.calls.index(("job", "go")))
        with self.assertRaisesRegex(ValueError, "managed execution owns"):
            self.pool.control("alice", first["id"], "release")
        self.pool = RuntimePool(self.root / "manager", self.backend)
        self.pool.tick()
        self.assertIn(("host", "heartbeat"), self.backend.calls)
        self.pool.managed_control("alice", first["id"], "stop")
        ended = self.pool.managed_control("alice", first["id"])
        self.assertEqual(ended["state"], "cancelled")
        self.assertFalse(ended.get("runtime_returned"))
        self.assertEqual(self.pool.status("bob")["jobs"][0]["state"], "running")
        self.assertFalse(self.backend.jobs[second["job_id"]]["quiet"])
        self.assertEqual(next(row for row in self.pool.catalog() if row["runtime_id"] == a["runtime_id"])["state"], "bound")

    def test_managed_renewal_uses_one_host_request_and_completion_releases_directly(self):
        binding = self.bind("alice", self.root / "one-renewal")
        self.backend.calls.clear()
        job = self.managed("alice", binding)
        self.assertNotIn(("host", "heartbeat"), self.backend.calls)
        self.backend.calls.clear()
        observed = self.pool.managed_control("alice", job["id"])
        self.assertEqual(observed["state"], "running")
        self.assertEqual([action for kind, action in self.backend.calls if kind == "host"], ["heartbeat"])
        self.backend.jobs[job["job_id"]].update(state="succeeded", quiet=True)
        self.backend.calls.clear()
        ended = self.pool.managed_control("alice", job["id"])
        self.assertEqual((ended["state"], ended["lease_state"]), ("succeeded", "released"))
        self.assertEqual([action for kind, action in self.backend.calls if kind == "host"], ["release"])

    def test_managed_stop_is_not_overwritten_by_an_inflight_advance(self):
        import threading

        binding = self.bind("alice", self.root / "a")
        job = self.managed("alice", binding)
        advance_at_save = threading.Event()
        release_advance = threading.Event()
        stop_persisted = threading.Event()
        advance_thread: dict[str, int] = {}
        stop_thread: dict[str, int] = {}
        original_save = self.pool._save_managed
        original_put = self.pool.put
        delayed_once = {"armed": True}

        def delayed_save(row):
            if (delayed_once["armed"] and threading.get_ident() == advance_thread.get("ident")
                    and not row["cancel_requested"]):
                delayed_once["armed"] = False
                advance_at_save.set()
                if not release_advance.wait(5):
                    raise TimeoutError("test did not release the in-flight advance")
            return original_save(row)

        def tracked_put(db, kind, row):
            result = original_put(db, kind, row)
            if (kind == "job" and row["id"] == job["id"] and row["cancel_requested"]
                    and threading.get_ident() == stop_thread.get("ident")):
                stop_persisted.set()
            return result

        def advance():
            advance_thread["ident"] = threading.get_ident()
            return self.pool.managed_advance(job["id"])

        def stop():
            stop_thread["ident"] = threading.get_ident()
            return self.pool.managed_control("alice", job["id"], "stop", force=True)

        with mock.patch.object(self.pool, "_save_managed", side_effect=delayed_save), \
                mock.patch.object(self.pool, "put", side_effect=tracked_put), \
                ThreadPoolExecutor(2) as workers:
            advance_future = workers.submit(advance)
            self.assertTrue(advance_at_save.wait(5))
            stop_future = workers.submit(stop)
            try:
                self.assertTrue(stop_persisted.wait(5))
            finally:
                release_advance.set()
            advance_future.result(timeout=5)
            stopped = stop_future.result(timeout=5)

        self.assertTrue(stopped["cancel_requested"])
        self.assertTrue(stopped["force"])
        self.assertEqual(stopped["state"], "stopping")
        self.assertIn(("job", "stop"), self.backend.calls)
        self.assertTrue(self.backend.jobs[job["job_id"]]["quiet"])

    def test_malformed_remote_job_reply_keeps_live_job_retryable(self):
        binding = self.bind("alice", self.root / "a")
        job = self.managed("alice", binding)
        original_job = self.backend.job
        malformed_once = {"armed": True}

        def malformed_reply(runtime, job_id, action, **params):
            if action == "status" and malformed_once["armed"]:
                malformed_once["armed"] = False
                raise json.JSONDecodeError("truncated remote JSON", "{", 1)
            return original_job(runtime, job_id, action, **params)

        self.backend.job = malformed_reply
        uncertain = self.pool.managed_control("alice", job["id"])
        self.assertEqual((uncertain["state"], uncertain["lease_state"]), ("uncertain", "active"))
        self.assertEqual(self.backend.jobs[job["job_id"]]["state"], "running")

        recovered = self.pool.managed_tick()
        self.assertIsNone(recovered)
        persisted = self.pool.status("alice")["jobs"][0]
        self.assertEqual((persisted["state"], persisted["lease_state"]), ("running", "active"))

    def test_lost_go_reply_reuses_the_same_job_and_does_not_prepare_again(self):
        binding = self.bind("alice", self.root / "a")
        self.backend.fail_job_after = "go"
        job = self.managed("alice", binding)
        self.assertEqual(job["state"], "uncertain")
        restarted = RuntimePool(self.root / "manager", self.backend)
        recovered = restarted.managed_control("alice", job["id"])
        self.assertEqual(recovered["state"], "running")
        self.assertEqual(self.backend.calls.count(("job", "prepare")), 1)
        self.assertEqual(self.backend.calls.count(("job", "go")), 1)
        with self.assertRaises(PermissionError):
            restarted.managed_control("bob", job["id"], "stop")

    def test_managed_finish_releases_owned_lease_while_foreign_occupancy_remains(self):
        binding = self.bind("alice", self.root / "a")
        job = self.managed("alice", binding)
        self.backend.jobs[job["job_id"]].update(state="succeeded", quiet=True, result={"state": "succeeded", "exit_code": 0})
        self.backend.busy = [0]
        result = self.pool.managed_control("alice", job["id"])
        self.assertEqual((result["state"], result["lease_state"]), ("succeeded", "released"))
        self.assertEqual(self.backend.busy, [0])
        self.assertNotIn(("job", "stop"), self.backend.calls)
        manifest = json.loads((self.root / "manager/runs" / (job["id"] + ".json")).read_text())
        self.assertEqual(manifest["process"]["state"], "succeeded")
        self.assertTrue(manifest["resources"]["released"])
        self.assertEqual(manifest["process"]["result"]["exit_code"], 0)
        waiting = self.managed("alice", binding, request_id="next-exclusive")
        self.assertEqual(waiting["lease_state"], "queued")

    def test_explicit_shared_job_finishes_while_external_worker_remains_busy(self):
        binding = self.bind("alice", self.root / "a")
        self.backend.busy = [0]
        job = self.pool.managed_start("alice", binding["id"], "shared",
                                      {"vllm": "a" * 40, "vllm-ascend": "b" * 40},
                                      "native-a", [0], 0, "exec python task.py", {}, 60,
                                      allow_external_busy=True)
        self.assertEqual((job["state"], job["lease_state"]), ("running", "active"))
        self.assertEqual(job["launch_observation"]["npu_devices"], [0])
        self.assertTrue(job["launch_observation"]["allow_external_busy"])
        self.backend.jobs[job["job_id"]].update(state="succeeded", quiet=True, result={"state": "succeeded", "exit_code": 0})
        result = self.pool.managed_control("alice", job["id"])
        self.assertEqual((result["state"], result["lease_state"]), ("succeeded", "released"))
        self.assertEqual(self.backend.busy, [0])
        self.assertNotIn(("job", "stop"), self.backend.calls)
        record = json.loads((self.root / "manager/runs" / (job["id"] + ".json")).read_text())
        self.assertTrue(record["resources"]["released"])
        self.assertTrue(record["resources"]["allow_external_busy"])

    def test_orphaned_busy_recovers_via_heartbeat_without_stopping_live_family(self):
        binding = self.bind("alice", self.root / "a")
        job = self.managed("alice", binding)
        self.assertEqual(job["state"], "running")
        import sqlite3
        with closing(sqlite3.connect(self.backend.state / "192_0_2_1" / "coordinator.sqlite3")) as db, db:
            db.execute("UPDATE tasks SET state='orphaned_busy'")
        self.backend.calls.clear()
        recovered = self.pool.managed_control("alice", job["id"])
        self.assertEqual((recovered["state"], recovered["lease_state"]), ("running", "active"))
        self.assertIn(("host", "heartbeat"), self.backend.calls)
        self.assertNotIn(("job", "stop"), self.backend.calls)
        self.assertFalse(self.backend.jobs[job["job_id"]]["quiet"])

    def test_disappeared_job_directory_never_confirms_completion(self):
        binding = self.bind("alice", self.root / "a")
        job = self.managed("alice", binding)
        self.backend.jobs.clear()  # the whole job directory vanished remotely
        self.pool.managed_control("alice", job["id"], "stop")
        result = self.pool.managed_control("alice", job["id"])
        self.assertEqual(result["state"], "stopping")
        self.assertNotIn(("host", "release"), self.backend.calls)

    def test_deterministic_validation_failure_fails_terminally_without_retry(self):
        binding = self.bind("alice", self.root / "a")
        job = self.pool.managed_start("alice", binding["id"], "managed",
                                      {"vllm": "a" * 40, "vllm-ascend": "b" * 40},
                                      "wrong-build-key", [0], 0, "exec python task.py", {}, 60)
        self.assertEqual(job["state"], "failed")
        self.assertIn("cache miss", job["error"])
        calls = list(self.backend.calls)
        self.pool.managed_tick()
        self.assertEqual(self.backend.calls, calls)

    def test_operator_reconcile_unwedges_an_uncertain_run_and_records_event(self):
        binding = self.bind("alice", self.root / "a")
        job = self.managed("alice", binding)
        import sqlite3
        with closing(sqlite3.connect(self.backend.state / "192_0_2_1" / "coordinator.sqlite3")) as db, db:
            db.execute("UPDATE meta SET value='new-epoch' WHERE key='coordination_epoch'")
        wedged = self.pool.managed_control("alice", job["id"])
        self.assertEqual(wedged["state"], "uncertain")
        with self.assertRaises(ValueError):
            self.pool.reconcile("admin", job["id"], "")
        result = self.pool.reconcile("admin", job["id"], "host rebooted; epoch and tasks verified gone")
        self.assertEqual(result["run"]["state"], "cancelled")
        self.assertEqual(result["jobs"][0]["state"], "inconclusive")
        self.assertEqual(self.pool.events("admin")["events"][-1]["kind"], "run-reconciled")
        with self.assertRaises(ValueError):
            self.pool.reconcile("admin", job["id"], "already terminal")
        self.assertEqual(self.pool.return_runtime("alice", binding["id"])["status"], "returned")

    def test_drain_waits_for_existing_managed_job_and_disables_automatic_reuse(self):
        binding = self.bind("alice", self.root / "a")
        job = self.managed("alice", binding)
        self.assertEqual(self.pool.drain(binding["runtime_id"])["state"], "bound")
        self.assertEqual(self.pool.managed_control("alice", job["id"])["state"], "running")
        self.pool.managed_control("alice", job["id"], "stop")
        self.pool.managed_control("alice", job["id"])
        runtime = next(row for row in self.pool.catalog() if row["runtime_id"] == binding["runtime_id"])
        self.assertEqual(runtime["state"], "bound")
        self.assertTrue(runtime["draining"])
        self.pool.return_runtime("alice", binding["id"])
        runtime = next(row for row in self.pool.catalog() if row["runtime_id"] == binding["runtime_id"])
        self.assertEqual(runtime["state"], "draining")
        self.pool.register(binding["runtime_id"], runtime_spec(1, user="alice"))
        self.assertFalse(next(row for row in self.pool.catalog() if row["runtime_id"] == binding["runtime_id"])["draining"])

    def test_a_hung_host_probe_blocks_only_its_own_runtime(self):
        import threading
        entered, release = threading.Event(), threading.Event()
        original = self.backend.inspect

        def inspect(runtime, **kwargs):
            if runtime.get("container_name") == "mindie-alice":
                entered.set()
                release.wait(10)
            return original(runtime, **kwargs)

        self.backend.inspect = inspect
        try:
            alice = self.pool.session_open("alice", "hang-a", {"va": str(self.root / "ha")})
            bob = self.pool.session_open("bob", "hang-b", {"va": str(self.root / "hb")})
            with ThreadPoolExecutor(2) as workers:
                stuck = workers.submit(self.pool.checkout, "alice", alice["id"], "profile-a", "hang-a", "runtime-a")
                self.assertTrue(entered.wait(10))
                other = workers.submit(self.pool.checkout, "bob", bob["id"], "profile-a", "hang-b", "runtime-b")
                self.assertEqual(other.result(timeout=10)["state"], "bound")
                release.set()
                self.assertEqual(stuck.result(timeout=10)["state"], "bound")
        finally:
            release.set()

    def test_concurrent_checkout_races_never_double_bind_a_runtime(self):
        pool = RuntimePool(self.root / "single", self.backend)
        pool.register("runtime-c", runtime_spec(3, user="alice"))
        session = pool.session_open("alice", "race-session", {"va": str(self.root / "r")})
        with ThreadPoolExecutor(4) as workers:
            results = list(workers.map(lambda index: pool.checkout("alice", session["id"], "profile-a", "race-" + str(index)), range(4)))
        bound = [row for row in results if row.get("state") == "bound"]
        self.assertEqual(len(bound), 1)
        self.assertEqual(bound[0]["runtime_id"], "runtime-c")
        self.assertTrue(all(row.get("status") == "cache_miss" for row in results if row is not bound[0]))
        self.assertEqual(next(row for row in pool.catalog() if row["runtime_id"] == "runtime-c")["state"], "bound")
        winning_request = "race-" + str(results.index(bound[0]))
        with ThreadPoolExecutor(2) as workers:
            replays = list(workers.map(lambda _: pool.checkout("alice", session["id"], "profile-a", winning_request), range(2)))
        self.assertEqual({row["id"] for row in replays}, {bound[0]["id"]})

    def test_concurrent_conflicting_registration_has_exactly_one_winner(self):
        def attempt(index):
            try:
                return self.pool.register("duplicate-" + str(index), runtime_spec(9, user="alice"))
            except ValueError as exc:
                return str(exc)

        with ThreadPoolExecutor(2) as workers:
            results = list(workers.map(attempt, range(2)))
        self.assertEqual(sum(isinstance(row, dict) for row in results), 1)
        self.assertTrue(any(isinstance(row, str) and "overlap" in row for row in results))

    def test_never_submitted_pending_run_cancels_and_expires_while_host_stays_down(self):
        import time
        binding = self.bind("alice", self.root / "a")
        self.backend.fail_after = "status"
        pending = self.request("alice", binding)
        self.assertEqual((pending["state"], pending["epoch"]), ("pending", None))
        self.backend.fail = True  # the host never comes back
        calls = list(self.backend.calls)
        cancelled = self.pool.control("alice", pending["id"], "cancel")
        self.assertEqual(cancelled["state"], "cancelled")
        self.assertEqual(self.backend.calls, calls)  # no host contact at all

        self.backend.fail = False
        self.backend.fail_after = "status"
        second = self.pool.request_run("alice", binding["id"], "request-2",
                                       {"vllm": "a" * 40, "vllm-ascend": "b" * 40}, "native-a", [0], 0, queue_seconds=1)
        self.assertEqual(second["state"], "pending")
        self.backend.fail = True
        self.pool.clock = lambda: time.time() + 60
        calls = list(self.backend.calls)
        expired = self.pool.control("alice", second["id"], "poll")
        self.assertEqual(expired["state"], "expired")
        self.assertEqual(self.backend.calls, calls)
        self.assertNotIn(("host", "submit"), self.backend.calls)

    def test_transaction_closes_connections_and_rolls_back(self):
        import sqlite3
        with self.assertRaises(RuntimeError):
            with self.pool.transaction() as db:
                db.execute("INSERT INTO records VALUES('kind','ident','{}')")
                raise RuntimeError("boom")
        with self.pool.transaction() as db:
            self.assertIsNone(db.execute("SELECT data FROM records WHERE kind='kind'").fetchone())
            handle = db
        with self.assertRaises(sqlite3.ProgrammingError):
            handle.execute("SELECT 1")

    def test_reconcile_defends_identifiers_and_binding_ownership_consistency(self):
        binding = self.bind("alice", self.root / "a")
        run = self.request("alice", binding)
        with self.assertRaisesRegex(ValueError, "invalid identifier"):
            self.pool.reconcile("admin", "bad id!", "host inspected")
        with self.assertRaisesRegex(ValueError, "invalid identifier"):
            self.pool.reconcile("bad admin!", run["id"], "host inspected")
        with self.assertRaisesRegex(ValueError, "uncertain"):
            self.pool.reconcile("admin", run["id"], "host inspected", evidence="looked")
        with self.pool.transaction() as db:
            corrupted = self.pool.get(db, "binding", binding["id"])
            corrupted["owner"] = "bob"
            self.pool.put(db, "binding", corrupted)
        with self.assertRaises(PermissionError):
            self.pool.reconcile("admin", run["id"], "host inspected")

    def test_reconcile_terminates_a_vanished_job_wedge_only_with_evidence(self):
        binding = self.bind("alice", self.root / "a")
        job = self.managed("alice", binding)
        self.assertEqual(job["state"], "running")
        self.backend.jobs.clear()  # the whole job directory vanished remotely
        self.pool.managed_control("alice", job["id"], "stop")
        self.assertEqual(self.pool.managed_control("alice", job["id"])["state"], "stopping")
        reason = "job directory vanished while the lease stayed active"
        with self.assertRaisesRegex(ValueError, "evidence"):
            self.pool.reconcile("admin", job["id"], reason, force_release=True)
        with self.assertRaisesRegex(ValueError, "uncertain"):
            self.pool.reconcile("admin", job["id"], reason)
        self.backend.calls.clear()
        result = self.pool.reconcile("admin", job["id"], reason,
                                     evidence="host ls shows no job directory; docker top shows no family",
                                     force_release=True)
        self.assertEqual(result["run"]["state"], "cancelled")
        self.assertEqual(result["jobs"][0]["state"], "inconclusive")
        self.assertEqual(self.backend.calls, [("host", "release")])
        self.assertTrue(result["event"]["force_release"])
        self.assertIn("no job directory", result["event"]["evidence"])
        self.assertEqual(self.pool.return_runtime("alice", binding["id"])["status"], "returned")

    def test_two_successive_managed_jobs_reuse_the_user_container(self):
        binding = self.bind("alice", self.root / "a")
        first = self.managed("alice", binding, request_id="run-1")
        self.pool.managed_control("alice", first["id"], "stop")
        ended = self.pool.managed_control("alice", first["id"])
        self.assertEqual(ended["state"], "cancelled")
        runtime = next(row for row in self.pool.catalog() if row["runtime_id"] == binding["runtime_id"])
        self.assertEqual(runtime["state"], "bound")
        self.assertEqual(runtime["container_name"], "mindie-alice")
        second = self.managed("alice", binding, request_id="run-2")
        self.assertEqual(second["state"], "running")
        self.assertEqual(binding["endpoint"]["port"], binding["endpoint"]["port"])
        self.assertEqual(binding["container_name"], "mindie-alice")
        self.assertIn((46001, "container_ssh", "ssh.alice"), self._host_ports())

    def test_service_stop_releases_port_and_keeps_container_ssh(self):
        binding = self.bind("alice", self.root / "a")
        job = self.pool.managed_start("alice", binding["id"], "serve",
                                      {"vllm": "a" * 40, "vllm-ascend": "b" * 40},
                                      "native-a", [0], 0, "exec python serve.py", {}, None, service_port=0)
        self.assertEqual(job["state"], "running")
        self.assertEqual(job["service_port"], 48001)
        self.assertEqual(job["environment"]["MINDIE_SERVICE_PORT"], "48001")
        self.assertIn((48001, "service", job["request"] and self.pool.status("alice")["runs"][0]["task_id"]), self._host_ports())
        self.assertIn((46001, "container_ssh", "ssh.alice"), self._host_ports())
        self.backend.listening = [48001]
        stopping = self.pool.managed_control("alice", job["id"], "stop")
        self.assertIn(stopping["state"], {"stopping", "releasing", "cancelled"})
        still = self.pool.managed_control("alice", job["id"])
        self.assertEqual(still["lease_state"], "orphaned_busy")
        self.assertIn((48001, "service", self.pool.status("alice")["runs"][0]["task_id"]), self._host_ports())
        self.backend.listening = []
        ended = self.pool.managed_control("alice", job["id"])
        self.assertEqual(ended["state"], "cancelled")
        ports = self._host_ports()
        self.assertIn((46001, "container_ssh", "ssh.alice"), ports)
        self.assertFalse(any(kind == "service" for _, kind, _ in ports))

    def test_wrong_owner_and_stale_fence_cannot_mutate_another_run(self):
        binding = self.bind("alice", self.root / "a")
        job = self.managed("alice", binding)
        with self.assertRaises(PermissionError):
            self.pool.managed_control("bob", job["id"], "stop")
        run = self.pool.status("alice")["runs"][0]
        with self.assertRaises(CoordinationError):
            self.backend.host(runtime_spec(1, user="alice"), {
                "action": "release", "task_id": run["task_id"],
                "fence_token": run["task"]["fence_token"] + 1,
                "coordination_epoch": run["epoch"], "completion_confirmed": True,
            })
        self.assertEqual(self.pool.managed_control("alice", job["id"])["state"], "running")
