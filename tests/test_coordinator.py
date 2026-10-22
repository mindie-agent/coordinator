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
    """Fixture surface for runtime-pool tests; concrete tests live in ``test_coordinator_pool.py`` and ``test_coordinator_task_client.py``."""

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

    def managed(self, owner, binding, device=0, request_id="managed"):
        return self.pool.managed_start(owner, binding["id"], request_id,
                                       {"vllm": "a" * 40, "vllm-ascend": "b" * 40},
                                       "native-a", [device], 0, "exec python task.py", {}, 60)

    def _host_ports(self):
        import sqlite3
        with closing(sqlite3.connect(self.backend.state / "192_0_2_1" / "coordinator.sqlite3")) as db, db:
            return [(row[0], row[1], row[2]) for row in db.execute("SELECT port, kind, task_id FROM ports ORDER BY port")]


class TaskClientTests(unittest.TestCase):
    """Fixture surface for task-client tests; concrete tests live in ``test_coordinator_task_client.py`` and ``test_coordinator_daemon.py``."""

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


class AggregateStateTests(unittest.TestCase):
    def test_mixed_terminal_is_failed_and_all_succeeded_is_succeeded(self):
        from mindie_coordinator.service import aggregate_job_states
        self.assertEqual(aggregate_job_states(["succeeded", "failed"]), "failed")
        self.assertEqual(aggregate_job_states(["succeeded", "succeeded"]), "succeeded")
        self.assertEqual(aggregate_job_states(["running", "succeeded"]), "running")
        self.assertEqual(aggregate_job_states(["uncertain", "running"]), "uncertain")


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
