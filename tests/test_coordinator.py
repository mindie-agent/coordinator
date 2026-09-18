from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from unittest import mock


def _init_git_workspace(path: Path) -> None:
    """Session sources in this suite live under ``path``; export_manifest
    resolves code identity from the containing worktree."""
    path.mkdir(parents=True, exist_ok=True)
    if (path / ".git").exists():
        return
    subprocess.run(["git", "-C", str(path), "init"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "pool@example.invalid"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Pool Test"],
        check=True,
        capture_output=True,
    )
    (path / "README").write_text("pool\n", encoding="utf-8")
    # Match the consumer contract: runtime state is not a source snapshot.
    (path / ".git" / "info" / "exclude").write_text(
        "manager/\ncoordinator/\nsessions/\nhost/\n.mindie-local/\n", encoding="utf-8"
    )
    subprocess.run(["git", "-C", str(path), "add", "README"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "init"], check=True, capture_output=True)

from mindie_coordinator.host_queue import HOST_QUEUE_MODULE_ENV, HostQueueUnavailable, load_host_protocol

try:
    # The suite runs against the bundled host authority, or an explicit
    # MINDIE_HOST_QUEUE_MODULE override. One host still has exactly one
    # allocator implementation.
    host_protocol = load_host_protocol()
except HostQueueUnavailable as exc:  # pragma: no cover - configuration guard
    raise unittest.SkipTest(
        f"{exc}. The bundled host/mindie_npu_coordination.py is missing or "
        f"{HOST_QUEUE_MODULE_ENV} points at a path that does not exist."
    ) from exc

sys.modules.setdefault("mindie_npu_coordination", host_protocol)
handle_request = host_protocol.handle_request
_confirmed_free_probe = host_protocol._confirmed_free_probe
CoordinationError = host_protocol.CoordinationError
from mindie_coordinator.ready_runtime import RuntimePool
from mindie_coordinator.execution_sources import capture_sources
from mindie_coordinator.runtime_profile import capture, digest, publish, restore, verify


class Backend:
    """Real SQLite host protocol, simulated occupancy and prepared containers."""
    def __init__(self, state):
        self.state = state
        self.busy = []
        self.fail = False
        self.fail_after = None
        self.calls = []
        self.jobs = {}
        self.fail_job_after = None
        self.prepared = {}
        self.require_prepared = False
        self.machines = None
        self.clock = time.time
        self.protocol_lock = threading.RLock()
        self.attestation = {"profile_key": "profile-a", "build_key": "native-a",
                            "profile": {"launch_env": {"VLLM_VERSION": "test"}}}

    def mark_prepared(self, spec, profile_key="profile-a", build_key="native-a"):
        root = spec["endpoint"]["root"]
        recipe = spec.get("recipe")
        self.prepared[root] = {
            "python": spec["python"],
            "recipe": recipe,
            "profile": {"launch_env": {"VLLM_VERSION": "test"}, "recipe": recipe},
            "profile_key": profile_key,
            "build_key": build_key,
        }

    def prepare_task_root(self, spec, *, sources, environment, donor_python=None, **kwargs):
        root = spec["endpoint"]["root"]
        python = spec["python"]
        if {"vllm", "vllm-ascend"}.issubset(sources) and donor_python and python == donor_python:
            raise ValueError("task-owned interpreter must not be the donor interpreter")
        recipe = (environment or {}).get("recipe") or (environment or {}).get("image") or spec.get("recipe")
        profile = {"launch_env": {"VLLM_VERSION": "test"}, "recipe": recipe}
        for key in ("python_abi", "cann", "soc", "machine_type"):
            if (environment or {}).get(key):
                profile[key] = environment[key]
        self.prepared[root] = {
            "python": python,
            "recipe": recipe,
            "profile": profile,
            "profile_key": "profile-" + (recipe or "prepared"),
            "build_key": "native-" + (recipe or "prepared"),
        }
        return self.inspect(spec)

    def command_environment(self, donor):
        return {"python": "/usr/local/bin/python3", "launch_env": {"PATH": "/usr/local/bin:/usr/bin:/bin"}}

    def inspect(self, runtime, **kwargs):
        if self.fail:
            raise TimeoutError("probe unavailable")
        self.calls.append(("inspect", kwargs))
        root = (runtime.get("endpoint") or {}).get("cwd") or (runtime.get("endpoint") or {}).get("root")
        python = runtime.get("python")
        prepared = self.prepared.get(root)
        if prepared is None:
            if self.require_prepared:
                raise ValueError("missing ready-profile.json")
            observed = copy.deepcopy(self.attestation)
            observed["container_id"] = "cid-" + str(runtime.get("container_name") or "unknown")
            return observed
        if prepared["python"] != python:
            raise ValueError("interpreter does not match prepared root")
        return {
            "profile_key": prepared["profile_key"],
            "build_key": prepared["build_key"],
            "profile": copy.deepcopy(prepared["profile"]),
            "container_id": "cid-" + str(runtime.get("container_name") or "unknown"),
        }

    def host(self, runtime, request):
        if self.fail:
            raise TimeoutError("host unavailable")
        self.calls.append(("host", request["action"]))
        def guarded(value, *, completion_confirmed=False):
            guard = json.loads(value) if isinstance(value, str) else value
            return bool(guard) and (any(not job["quiet"] and job["receipt"]["process_guard"] == guard for job in self.jobs.values())
                                   or (guard.get("retain_until_release") and not completion_confirmed))
        listening = {"status": "ok", "ports": list(getattr(self, "listening", []))}
        host = runtime["host_endpoint"]["host"]
        state = Path(self.state) / host.replace(".", "_")
        state.mkdir(parents=True, exist_ok=True)
        with self.protocol_lock, mock.patch("mindie_npu_coordination.process_guard_busy", side_effect=guarded):
            result = handle_request({**request, "state_dir": str(state), "interval_seconds": 0.001},
                                    clock=self.clock,
                                    probe=lambda: {"status": "ok", "devices": [0, 1],
                                                   "busy": {str(d): ["test worker"] for d in self.busy}},
                                    device_probe=lambda device: {"status": "ok", "devices": [device],
                                                                 "busy": None, "free": []},
                                    listening_ports=lambda: listening)
        if self.fail_after == request["action"]:
            self.fail_after = None
            raise TimeoutError("reply lost after host mutation")
        return result

    def job(self, runtime, job_id, action, **params):
        self.calls.append(("job", action))
        if action == "prepare" and job_id not in self.jobs:
            self.jobs[job_id] = {"state": "prepared", "quiet": False,
                                 "receipt": {"pid": 4242, "process_guard": {"marker": job_id[-32:], "boot_id": "test-boot",
                                                                          "retain_until_release": True}}}
        if action == "go":
            self.jobs[job_id]["state"] = "running"
        if action == "stop" and job_id in self.jobs:
            self.jobs[job_id].update(state="cancelled", quiet=True)
        if action == "tail":
            job = self.jobs.get(job_id, {"state": "absent", "quiet": True})
            text = job.get("stdout") or f"stdout-{job_id[-8:]}"
            return {**copy.deepcopy(job), "stdout": text, "stderr": job.get("stderr") or "", "tail": text}
        if self.fail_job_after == action:
            self.fail_job_after = None
            raise TimeoutError("lost job reply")
        return copy.deepcopy(self.jobs.get(job_id, {"state": "absent", "quiet": True}))

    def job_host_pid(self, runtime, receipt):
        return receipt["pid"]


def runtime_spec(number, user="alice", python=None, root=None, service_ports=None, host=None, recipe=None):
    host = host or "192.0.2.1"
    ssh = {"alice": 46001, "bob": 46002}.get(user, 46100)
    if host != "192.0.2.1":
        ssh = 46100 + number
    spec = {
        "user": user,
        "python": python or f"/opt/{user}/venvs/root-{number}/bin/python",
        "endpoint": {"host": host, "port": ssh, "root": root or f"/vllm-workspace/{user}/{number}",
                     "user": "root"},
        "host_endpoint": {"host": host, "port": 22, "user": "root"},
        "container_name": "mindie-" + user,
        "service_ports": service_ports if service_ports is not None else [48000 + number],
    }
    if recipe:
        spec["recipe"] = recipe
    return spec


class FakeShell:
    """Stand-in for the injected remote-dev shell adapter."""

    def __init__(self, result=None):
        self.result = result or {}
        self.calls = []

    def run(self, target, command, *, timeout_ms=45000):
        self.calls.append((target, command, timeout_ms))
        return self.result


class BackendTests(unittest.TestCase):
    def test_inspect_verifies_selected_python_and_allows_sibling_workers(self):
        from mindie_coordinator.backend import RemoteBackend

        backend = RemoteBackend()
        spec = runtime_spec(1, user="alice")
        commands = []

        def bash(target, command):
            commands.append(command)
            if "docker inspect" in command:
                return json.dumps({"Id": "container-alice", "State": {"Running": True}})
            return json.dumps({"profile": {"launch_env": {}}, "profile_key": "profile-a", "build_key": "native-a"})

        with mock.patch.object(backend, "bash", side_effect=bash), mock.patch("mindie_coordinator.backend.launch_preamble", return_value=""):
            observed = backend.inspect(spec, idle=True)
        self.assertEqual(observed["container_id"], "container-alice")
        self.assertTrue(any(spec["python"] in command for command in commands))
        self.assertFalse(any("docker top" in command for command in commands))
        self.assertFalse(any(command.startswith("ss ") for command in commands))

        def stopped(target, command):
            if "docker inspect" in command:
                return json.dumps({"Id": "container-alice", "State": {"Running": False}})
            raise AssertionError("profile probe must not run for a stopped container")

        with mock.patch.object(backend, "bash", side_effect=stopped):
            with self.assertRaisesRegex(RuntimeError, "not running"):
                backend.inspect(spec, idle=True)

    def test_bash_failure_carries_outcome_and_bounded_stderr_without_command(self):
        from mindie_coordinator.backend import RemoteBackend

        target = {"host": "192.0.2.1", "port": 22, "user": "root"}
        with tempfile.TemporaryDirectory() as tmp:
            stdout = Path(tmp) / "stdout.log"
            stderr = Path(tmp) / "stderr.log"
            stdout.write_text("")
            stderr.write_text("padding line\n" * 100 + "final: device probe permission denied")
            result = {"outcome": "failed", "status": "nonzero_exit", "exit_code": 17,
                      "refs": {"stdout": str(stdout), "stderr": str(stderr)}}
            backend = RemoteBackend(shell=FakeShell(result))
            with self.assertRaises(RuntimeError) as caught:
                backend.bash(target, "echo SECRET-TOKEN-VALUE")
            message = str(caught.exception)
            self.assertIn("failed/nonzero_exit", message)
            self.assertIn("exit 17", message)
            self.assertIn("permission denied", message)
            self.assertNotIn("SECRET-TOKEN-VALUE", message)
            self.assertLessEqual(len(message), 400)  # bounded tail only
            blocked = {"outcome": "blocked", "status": "cwd_outside_root"}
            with self.assertRaisesRegex(RuntimeError, "blocked/cwd_outside_root"):
                RemoteBackend(shell=FakeShell(blocked)).bash(target, "true")

    def test_remote_prepare_refuses_donor_python_and_requires_sources(self):
        from mindie_coordinator.backend import RemoteBackend

        backend = RemoteBackend()
        spec = runtime_spec(1, user="alice")
        with self.assertRaisesRegex(ValueError, "donor"):
            backend.prepare_task_root(spec, sources={"vllm": "/a", "vllm-ascend": "/b"},
                                      environment={"recipe": "rc"}, donor_python=spec["python"])
        spec = runtime_spec(1, user="alice", python="/vllm-workspace/tasks/s/h/r/.venv/bin/python",
                            root="/vllm-workspace/tasks/s/h/r")
        with self.assertRaisesRegex(ValueError, "fixed execution source inputs"):
            backend.prepare_task_root(spec, sources={}, environment={"recipe": "rc"},
                                      donor_python="/opt/alice/venvs/root-1/bin/python")


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        _init_git_workspace(self.root)
        self.backend = Backend(self.root / "host")
        self.pool = RuntimePool(self.root / "manager", self.backend)
        self.pool.register("runtime-a", runtime_spec(1, user="alice"))
        self.pool.register("runtime-b", runtime_spec(1, user="bob"))

    def tearDown(self):
        self.temp.cleanup()

    def bind(self, owner, root):
        session = self.pool.session_open(owner, "same-session-name", {"vllm": str(root / "vllm"), "vllm-ascend": str(root / "va")})
        return self.pool.checkout(owner, session["id"], "profile-a", "checkout")

    def request(self, owner, binding, **extra):
        return self.pool.request_run(owner, binding["id"], "request", {"vllm": "a" * 40, "vllm-ascend": "b" * 40}, "native-a", [0], 0, **extra)

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

    def test_return_requires_fresh_admin_verification(self):
        binding = self.bind("alice", self.root / "a")
        self.assertEqual(self.pool.return_runtime("alice", binding["id"])["runtime_state"], "needs_repair")
        session = self.pool.session_open("bob", "bob-session", {"va": str(self.root / "b")})
        self.assertEqual(self.pool.checkout("bob", session["id"], "profile-a", "bob", binding["runtime_id"])["status"], "cache_miss")
        self.pool.register(binding["runtime_id"], runtime_spec(1, user="alice"))
        self.assertEqual(self.pool.checkout("bob", session["id"], "profile-a", "bob", binding["runtime_id"])["status"], "cache_miss")
        self.assertEqual(self.pool.checkout("alice", binding["intent"]["session"], "profile-a", "alice-again", binding["runtime_id"])["state"], "bound")

    def test_repeated_free_probe_requires_visibility_in_every_sample(self):
        samples = iter([{"status": "ok", "devices": [1], "busy": {}}, {"status": "ok", "devices": [0, 1], "busy": {}}])
        result = _confirmed_free_probe(samples=2, interval_seconds=0, probe=lambda: next(samples))
        self.assertEqual(result["free"], [1])

    def managed(self, owner, binding, device=0, request_id="managed"):
        return self.pool.managed_start(owner, binding["id"], request_id,
                                       {"vllm": "a" * 40, "vllm-ascend": "b" * 40},
                                       "native-a", [device], 0, "exec python task.py", {}, 60)

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

    def test_managed_finish_defers_release_for_real_device_occupancy(self):
        binding = self.bind("alice", self.root / "a")
        job = self.managed("alice", binding)
        self.backend.jobs[job["job_id"]].update(state="succeeded", quiet=True, result={"state": "succeeded", "exit_code": 0})
        self.backend.busy = [0]
        result = self.pool.managed_control("alice", job["id"])
        self.assertEqual((result["state"], result["lease_state"]), ("releasing", "orphaned_busy"))
        self.backend.busy = []
        result = self.pool.managed_control("alice", job["id"])
        self.assertEqual(result["state"], "succeeded")
        manifest = json.loads((self.root / "manager/runs" / (job["id"] + ".json")).read_text())
        self.assertEqual(manifest["process"]["state"], "succeeded")
        self.assertTrue(manifest["resources"]["released"])
        self.assertEqual(manifest["process"]["result"]["exit_code"], 0)

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
        with ThreadPoolExecutor(2) as workers:
            replays = list(workers.map(lambda _: pool.checkout("alice", session["id"], "profile-a", "race-0"), range(2)))
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

    def _host_ports(self):
        import sqlite3
        with closing(sqlite3.connect(self.backend.state / "192_0_2_1" / "coordinator.sqlite3")) as db, db:
            return [(row[0], row[1], row[2]) for row in db.execute("SELECT port, kind, task_id FROM ports ORDER BY port")]

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


class TaskClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.backend = Backend(self.root / "host")
        self.backend.require_prepared = True
        self.pool = RuntimePool(self.root / "manager", self.backend)
        self._seed(self.pool, "runtime-a", runtime_spec(1, user="alice", recipe="rc"))
        from mindie_coordinator.agent_session import AgentSessions
        from mindie_coordinator.task_client import TaskClient
        self.store = AgentSessions(self.root / "sessions")
        self.context = self.store.attach("codex", "native-alice", str(self.root))
        self.client = TaskClient(self.context["context_file"], pool=self.pool, user="alice")

    def bind_native_sources(self, context=None):
        """Only source-specific tests need Git capture and native preparation."""
        sources = {name: str(self.root / name) for name in ("vllm", "vllm-ascend")}
        for source in sources.values():
            _init_git_workspace(Path(source))
        self.store.bind_sources(context or self.context, sources)
        self.client.coordinator.sync_binding = lambda *args, **kwargs: {"vllm": "a" * 40, "vllm-ascend": "b" * 40}

    def _seed(self, pool, runtime_id, spec):
        self.backend.mark_prepared(spec)
        return pool.register(runtime_id, spec)

    def tearDown(self):
        self.temp.cleanup()

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
            first["reply"] = self.client.run("true")

        worker = threading.Thread(target=run_first)
        worker.start()
        try:
            self.assertTrue(started.wait(30))
            second = self.client.run("sleep 1")
            self.assertEqual(second["state"], "running")
            self.assertEqual(len(syncs), 2)
        finally:
            release.set()
            worker.join(60)
        self.assertFalse(worker.is_alive(), "materialization worker must finish before fixture cleanup")
        self.assertEqual(first["reply"]["state"], "running")

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


class AggregateStateTests(unittest.TestCase):
    def test_mixed_terminal_is_failed_and_all_succeeded_is_succeeded(self):
        from mindie_coordinator.service import aggregate_job_states
        self.assertEqual(aggregate_job_states(["succeeded", "failed"]), "failed")
        self.assertEqual(aggregate_job_states(["succeeded", "succeeded"]), "succeeded")
        self.assertEqual(aggregate_job_states(["running", "succeeded"]), "running")
        self.assertEqual(aggregate_job_states(["uncertain", "running"]), "uncertain")


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
                        t0 = time.time()
                        client.call("ping")
                        self.assertLess(time.time() - t0, 0.5)
                        break
                    except (RuntimeError, FileNotFoundError, ConnectionError, OSError):
                        time.sleep(0.05)
                else:
                    self.fail("daemon did not listen")
                self.assertTrue(started.wait(3))
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


class ProfileTests(unittest.TestCase):
    def test_attestation_requires_populated_pinned_native_submodules(self):
        import subprocess
        from mindie_coordinator.prepare_runtime import require_clean_sources

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def git(repo, *args):
                return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()

            def init(repo):
                repo.mkdir(parents=True)
                git(repo, 'init')
                git(repo, 'config', 'user.name', 'Test')
                git(repo, 'config', 'user.email', 'test@example.invalid')
                (repo / 'kernel.cpp').write_text('native input\n')
                git(repo, 'add', '.')
                git(repo, 'commit', '-m', 'base')

            for name in ('vllm', 'vllm-ascend'):
                init(root / name)
            va = root / 'vllm-ascend'
            (va / '.gitmodules').write_text('[submodule "catlass"]\n path = csrc/catlass\n url = ./catlass\n')
            (va / '.gitignore').write_text('csrc/catlass\n')
            git(va, 'add', '.')
            git(va, 'commit', '-m', 'declare native dependency')
            with self.assertRaisesRegex(RuntimeError, 'missing'):
                require_clean_sources(root)
            child = va / 'csrc/catlass'
            init(child)
            with self.assertRaisesRegex(ValueError, 'tracked at its pinned commit'):
                require_clean_sources(root)
            git(va, 'update-index', '--add', '--cacheinfo', '160000,' + git(child, 'rev-parse', 'HEAD') + ',csrc/catlass')
            git(va, 'commit', '-m', 'pin dependency')
            require_clean_sources(root)
            (child / 'kernel.cpp').write_text('uncommitted native change\n')
            with self.assertRaisesRegex(ValueError, 'clean materialized'):
                require_clean_sources(root)

    def test_complete_bundle_hashes_missing_metadata_env_and_cache_reuse(self):
        from mindie_coordinator.runtime_profile import PROFILE_FIELDS
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runtime"
            root.mkdir()
            for name in ["kernels.so", "binary_info_config.json", "cann.txt", "driver.txt", "smoke.txt"]:
                (root / name).write_text(name)
            (root / 'smoke.txt').write_text(json.dumps({'passed': True}))
            profile = {key: "test-version" for key in PROFILE_FIELDS}
            profile.update(build_env={}, launch_env={"VLLM_VERSION": "test"}, compatibility_evidence="smoke.txt")
            profile["system_files"] = {name: {"path": str(root / (name + ".txt")), "sha256": hashlib.sha256((name + ".txt").encode()).hexdigest()} for name in ["cann", "driver"]}
            inputs = {"vllm": "native-a", "vllm-ascend": "native-b"}
            manifest = capture(root, profile, inputs, {"kernels.so": "library", "binary_info_config.json": "metadata"}, {"cann": "cann.txt", "driver": "driver.txt", "smoke": "smoke.txt"})
            verify(root, manifest, check_environment=False)
            with mock.patch("mindie_coordinator.runtime_profile.importlib.metadata.version", return_value="test-version"), mock.patch("mindie_coordinator.runtime_profile.sysconfig.get_config_var", return_value="test-version"):
                bundle = publish(root, Path(tmp) / "bundles", manifest)
                self.assertEqual(publish(root, Path(tmp) / "bundles", manifest), bundle)
                (root / "smoke.txt").write_text(json.dumps({'passed': True, 'timestamp': 'new'}))
                refreshed = copy.deepcopy(manifest)
                refreshed["evidence"]["smoke"]["sha256"] = hashlib.sha256((root / "smoke.txt").read_bytes()).hexdigest()
                self.assertEqual(publish(root, Path(tmp) / "bundles", refreshed), bundle)
                # Restoration also restores the original complete evidence,
                # rather than mixing its manifest with a newer smoke receipt.
                (root / "kernels.so").unlink()
                with self.assertRaisesRegex(ValueError, "missing required"):
                    verify(root, manifest)
                restore(root, bundle, manifest["build_key"])
                with mock.patch("mindie_coordinator.runtime_profile.sysconfig.get_config_var", return_value="different-abi"):
                    with self.assertRaisesRegex(ValueError, "Python ABI changed"):
                        verify(root, manifest)
                (root / "binary_info_config.json").write_text("corrupt")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    verify(root, manifest)
            with self.assertRaises(ValueError):
                capture(root, profile, inputs, {"kernels.so": "library"}, {})
            from mindie_coordinator.runtime_profile import launch_preamble
            import os, subprocess
            profile["launch_env"]["PYTHONPATH"] = "/scoped/source"
            script = "export PYTHONPATH=/base/acl:/base/native-compat\n" + launch_preamble(profile) + '\nprintf "%s" "$PYTHONPATH"'
            result = subprocess.run([shutil.which("bash") or "bash", "-s"], input=script.encode("utf-8"), check=True,
                                    capture_output=True,
                                    env={**os.environ, "PYTHONPATH": "/base/acl:/base/native-compat"}).stdout.decode("utf-8")
            self.assertEqual(result, "/scoped/source:/base/acl:/base/native-compat")

    def test_attest_records_smoke_timeout_as_evidence(self):
        import subprocess
        from mindie_coordinator import prepare_runtime
        from mindie_coordinator.runtime_profile import PROFILE_FIELDS

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = {key: "test-version" for key in PROFILE_FIELDS}
            profile.update(build_env={}, launch_env={}, compatibility_evidence="smoke-ref",
                           system_files={name: {"path": str(root / (name + ".txt")), "sha256": "0" * 64} for name in ("cann", "driver")})
            with mock.patch("mindie_coordinator.prepare_runtime.require_clean_sources"), \
                    mock.patch("mindie_coordinator.prepare_runtime.runtime_build_inputs", return_value={"vllm": "native-a"}), \
                    mock.patch("mindie_coordinator.prepare_runtime.subprocess.run",
                               side_effect=subprocess.TimeoutExpired(cmd="smoke", timeout=60)):
                with self.assertRaisesRegex(ValueError, "timed out"):
                    prepare_runtime.attest(root, {"profile": profile, "files": {}})
            evidence = json.loads((root / ".mindie-runtime/profile-evidence/smoke.json").read_text())
            self.assertFalse(evidence["passed"])
            self.assertIn("timed out", evidence["error"])
            self.assertEqual(evidence["build_inputs"], {"vllm": "native-a"})


if __name__ == "__main__":
    unittest.main()
