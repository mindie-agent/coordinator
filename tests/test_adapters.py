"""Contracts of the injected dependencies this repository does not own."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from remote_dev.result import RESULT_SCHEMA_VERSION, make_result

from mindie_coordinator.backend import RemoteDev
from mindie_coordinator.git_sources import discover_repo_tree, iter_postorder
from mindie_coordinator.host_queue import (
    HostQueue,
    HostQueueUnavailable,
    bundled_host_module_path,
    host_queue_module_path,
    load_host_protocol,
)
from mindie_coordinator.machine_directory import MachineDirectory, MachineDirectoryUnavailable
from mindie_coordinator.ops import mindie_call

HOST_MODULE = '''
import json


class CoordinationError(ValueError):
    pass


def handle_request(request):
    return {"status": "ok", "echo": request}
'''


class RemoteDevAdapterTests(unittest.TestCase):
    def test_calls_require_an_explicit_host_and_port(self):
        shell = RemoteDev()
        for target in ({"alias": "runtime-a"}, {"host": "runtime.invalid"}, {"port": 22}):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "explicit host and port"):
                shell.run(target, "true")

    def test_successful_shell_uses_the_installed_remote_dev_package(self):
        from remote_dev.core.endpoint import resolve_endpoint
        from remote_dev.core.shell_ops import remote_bash
        from mindie_coordinator import backend as backend_mod

        self.assertIs(backend_mod.resolve_endpoint, resolve_endpoint)
        self.assertIs(backend_mod.remote_bash, remote_bash)

        target = {"host": "runtime.invalid", "port": 46010, "user": "root",
                  "root": "/vllm-workspace", "cwd": "/vllm-workspace"}
        envelope = {"outcome": "success", "status": "ok", "exit_code": 0,
                    "refs": {"stdout": "", "stderr": ""}}
        with mock.patch.object(
            backend_mod, "remote_bash", return_value={"result": envelope},
        ) as bash:
            result = RemoteDev().run(target, "true")
        self.assertEqual(result["outcome"], "success")
        endpoint = bash.call_args.args[0]
        self.assertEqual(endpoint.host, "runtime.invalid")
        self.assertEqual(endpoint.port, 46010)
        self.assertEqual(bash.call_args.kwargs["command"], "true")
        self.assertFalse(bash.call_args.kwargs["runtime_env"])
        self.assertTrue(bash.call_args.kwargs["wait"])


class ProcessControlBoundaryTests(unittest.TestCase):
    """Generic process control belongs to remote-dev, not this package."""

    def test_coordinator_does_not_ship_a_container_worker(self):
        from mindie_coordinator import backend as backend_mod
        self.assertFalse(hasattr(backend_mod, "worker_source"))
        self.assertFalse(hasattr(backend_mod, "WORKERS"))


class HostQueueAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.module = Path(self.temp.name) / "mindie_npu_coordination.py"
        self.module.write_text(HOST_MODULE)
        self.addCleanup(self.temp.cleanup)

    def test_unconfigured_authority_uses_the_bundled_module(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(host_queue_module_path(), bundled_host_module_path())
            seen = {}

            def run(target, command):
                seen.update(command=command)
                return json.dumps({"status": "ok", "tasks": []})

            self.assertEqual(HostQueue(run).request({"host": "h", "port": 22}, {"action": "status"})["status"], "ok")
            self.assertIn("def handle_request(", seen["command"])
            self.assertIn("class CoordinationError", seen["command"])

    def test_explicit_missing_module_fails_closed(self):
        missing = Path(self.temp.name) / "absent.py"
        with self.assertRaisesRegex(HostQueueUnavailable, "not found"):
            HostQueue(lambda target, command: "", module_path=missing).request(
                {"host": "h", "port": 22}, {"action": "status"})
        with mock.patch.dict("os.environ", {"MINDIE_HOST_QUEUE_MODULE": str(missing)}, clear=True):
            with self.assertRaisesRegex(HostQueueUnavailable, "not found"):
                HostQueue(lambda target, command: "").request({"host": "h", "port": 22}, {"action": "status"})

    def test_request_ships_the_configured_module_and_pins_the_host_root(self):
        seen = {}

        def run(target, command):
            seen.update(target=target, command=command)
            return json.dumps({"status": "ok", "tasks": []})

        queue = HostQueue(run, module_path=self.module)
        self.assertEqual(queue.request({"host": "host.invalid", "port": 22, "user": "root"},
                                       {"action": "status"})["status"], "ok")
        self.assertEqual(seen["target"], {"host": "host.invalid", "port": 22, "user": "root",
                                          "root": "/", "cwd": "/"})
        self.assertIn("def handle_request(request)", seen["command"])
        self.assertIn('{"action":"status"}', seen["command"])

    def test_unresolved_host_answers_and_transport_failures_never_look_granted(self):
        for status in ("failed", "needs_input", "probe_failed"):
            queue = HostQueue(lambda target, command: json.dumps({"status": status, "error": "no"}),
                              module_path=self.module)
            with self.subTest(status=status), self.assertRaisesRegex(RuntimeError, "no"):
                queue.request({"host": "host.invalid", "port": 22}, {"action": "acquire"})

        def broken(target, command):
            raise RuntimeError("runtime probe failed (failed/nonzero_exit, exit 255): ssh closed")

        with self.assertRaisesRegex(RuntimeError, "host coordination failed"):
            HostQueue(broken, module_path=self.module).request({"host": "host.invalid", "port": 22},
                                                              {"action": "release"})

    def test_the_host_protocol_loads_under_its_canonical_module_name(self):
        module = load_host_protocol(self.module)
        self.assertEqual(module.handle_request({"action": "status"})["status"], "ok")


class MachineDirectoryTests(unittest.TestCase):
    def test_unconfigured_directory_fails_closed_instead_of_an_empty_fleet(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(MachineDirectoryUnavailable, "machine directory is empty"):
                MachineDirectory(path=Path(tmp) / "machines.json").catalog()

    def test_catalog_and_unique_alias_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "machines.json"
            path.write_text(json.dumps({"machines": [
                {"alias": "alpha", "host": {"ip": "host-a.invalid", "port": 22, "user": "root"},
                 "container": {"name": "prepared-a", "ssh_port": 46010}},
                {"alias": "beta", "host": {"ip": "host-b.invalid"}, "container": {}},
                {"alias": "beta", "host": {"ip": "host-c.invalid"}, "container": {}}]}))
            directory = MachineDirectory(path=path)
            catalog = directory.catalog()
            self.assertEqual(catalog["inventory_path"], str(path.resolve()))
            self.assertEqual(catalog["machines"][0], {"alias": "alpha", "host": "host-a.invalid",
                                                      "container_name": "prepared-a", "container_port": 46010})
            self.assertEqual(directory.host("alpha")["port"], 22)
            with self.assertRaisesRegex(ValueError, "resolve uniquely"):
                directory.host("beta")
            with self.assertRaisesRegex(ValueError, "resolve uniquely"):
                directory.host("missing")

    def test_consumer_passes_a_document_not_a_consumer_path(self):
        document = {"machines": [
            {"alias": "alpha", "host": {"ip": "host-a.invalid", "port": 22, "user": "root"},
             "container": {"name": "prepared-a", "ssh_port": 46010}},
        ]}
        directory = MachineDirectory(document)
        self.assertEqual(directory.host("alpha")["ip"], "host-a.invalid")
        with tempfile.TemporaryDirectory() as tmp:
            store = MachineDirectory(path=Path(tmp) / "machines.json")
            written = store.replace(document)
            self.assertTrue(written.is_file())
            self.assertEqual(MachineDirectory(path=written).host("alpha")["port"], 22)


class GitSourceTests(unittest.TestCase):
    def git(self, repo, *args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    def init(self, repo):
        repo.mkdir(parents=True)
        self.git(repo, "init")
        self.git(repo, "config", "user.name", "Test")
        self.git(repo, "config", "user.email", "test@example.invalid")
        (repo / "file.txt").write_text("content\n")
        self.git(repo, "add", ".")
        self.git(repo, "commit", "-m", "base")

    def test_missing_and_unpopulated_submodules_are_reported_before_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            self.init(root)
            (root / ".gitmodules").write_text('[submodule "child"]\n path = nested/child\n url = ./child\n')
            self.git(root, "add", ".gitmodules")
            self.git(root, "commit", "-m", "declare")
            with self.assertRaisesRegex(RuntimeError, "missing"):
                discover_repo_tree(root, ".")
            (root / "nested/child").mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "not a populated Git worktree"):
                discover_repo_tree(root, ".")

    def test_children_are_visited_before_their_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            self.init(root)
            (root / ".gitmodules").write_text('[submodule "child"]\n path = nested/child\n url = ./child\n')
            self.init(root / "nested/child")
            order = [node.relpath for node in iter_postorder(discover_repo_tree(root, "."))]
            self.assertEqual(order, ["nested/child", "."])


class ResultAndToolContractTests(unittest.TestCase):
    def test_result_envelope_comes_from_remote_dev(self):
        result = make_result(tool="mindie.session", target={"kind": "mindie-task"}, outcome="success",
                             status="open", summary="MindIE open")
        self.assertEqual(result["schema_version"], RESULT_SCHEMA_VERSION)
        self.assertLessEqual({"tool", "invocation_id", "target", "outcome", "status", "summary",
                              "started_at", "duration_ms", "preview", "refs", "artifacts",
                              "changed_files", "warnings", "next"}, set(result))


    def test_an_unavailable_task_registry_is_blocked_and_never_a_remote_success(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            payload = mindie_call("mindie.session", {})
        self.assertEqual(payload["result"]["outcome"], "blocked")
        self.assertEqual(payload["result"]["status"], "unavailable")



if __name__ == "__main__":
    unittest.main()
