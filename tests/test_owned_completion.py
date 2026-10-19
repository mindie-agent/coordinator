"""Real process/port completion stays independent of foreign NPU occupancy."""

import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import tempfile
import unittest
import uuid

from mindie_coordinator.host.mindie_npu_coordination import (
    CoordinationError,
    JOB_TOKEN_ENV,
    NpuCoordinator,
    handle_request,
    probe_listening_ports,
)


@unittest.skipUnless(sys.platform == "linux", "process guards use Linux /proc")
class OwnedCompletionTests(unittest.TestCase):
    def test_exclusive_completion_releases_only_drained_owner_and_port(self):
        children = []

        def child(marker=None, *, listen_port=None):
            environment = dict(os.environ)
            environment.pop(JOB_TOKEN_ENV, None)
            if marker:
                environment[JOB_TOKEN_ENV] = marker
            script = (
                "import json,socket,sys\n"
                "sock=socket.socket()\n"
                "if sys.argv[1]!='idle':\n"
                " sock.bind(('127.0.0.1',int(sys.argv[1])));sock.listen()\n"
                "print(json.dumps({'port':sock.getsockname()[1]}),flush=True)\n"
                "for line in sys.stdin:\n"
                " if line.strip()=='quit':break\n"
                " print('alive',flush=True)\n"
            )
            process = subprocess.Popen(
                [sys.executable, "-u", "-c", script, str(listen_port) if listen_port else "idle"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=environment,
            )
            children.append(process)
            self.assertTrue(select.select([process.stdout], [], [], 3)[0], "child did not start")
            return process, json.loads(process.stdout.readline())["port"]

        def stop(process):
            process.stdin.write("quit\n")
            process.stdin.flush()
            self.assertEqual(process.wait(timeout=3), 0)

        try:
            foreign, _ = child()
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            # This captured occupancy shape is the only simulated boundary;
            # guards, epoch/fence, SQLite leases and listening ports are real.
            busy = {
                "status": "ok", "devices": [12, 13], "free": [13],
                "busy": {"12": [{"kind": "process", "name": "VLLMWorker", "pid": foreign.pid}]},
            }
            free = {"status": "ok", "devices": [12, 13], "free": [12, 13], "busy": {}}
            with tempfile.TemporaryDirectory() as state:
                queue = NpuCoordinator(state)
                epoch = queue.snapshot(None)["coordination_epoch"]
                queue.submit({
                    "task_id": "owned-task", "agent_id": "owner", "devices": [12, 13],
                    "allow_external_busy": False, "service_port": port, "service_ports": [port],
                })
                grant = queue.acquire(
                    "owned-task", free, listening=probe_listening_ports(),
                )["task"]
                token = grant["fence_token"]
                queue.preflight("owned-task", token, free)
                listener, _ = child(listen_port=port)
                marker = uuid.uuid4().hex
                owner, _ = child(marker)
                guard = {
                    "marker": marker,
                    "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    "retain_until_release": True,
                }
                queue.activate("owned-task", token, pid=owner.pid, process_guard=guard)
                request = {
                    "action": "release", "state_dir": state, "task_id": "owned-task",
                    "coordination_epoch": epoch, "fence_token": token,
                    "completion_confirmed": True,
                }

                def release(**changes):
                    return handle_request({**request, **changes}, probe=lambda: busy)

                with self.assertRaisesRegex(CoordinationError, "epoch changed"):
                    release(coordination_epoch=str(uuid.uuid4()))
                with self.assertRaises(CoordinationError):
                    release(fence_token=token + 1)
                self.assertEqual(release()["status"], "orphaned_busy")
                stop(owner)
                observed_ports = probe_listening_ports()
                self.assertEqual(observed_ports["status"], "ok")
                self.assertIn(port, observed_ports["ports"])
                held = release()
                self.assertEqual(held["status"], "orphaned_busy")
                self.assertEqual(held["conflicting_service_ports"], [port])
                stop(listener)
                self.assertEqual(
                    release(completion_confirmed=False, free_samples=1)["status"], "orphaned_busy",
                )
                released = release()
                self.assertEqual(released["status"], "released")
                self.assertIsNone(released["task"]["granted_service_port"])
                self.assertFalse(released["task"]["allow_external_busy"])
                foreign.stdin.write("ping\n")
                foreign.stdin.flush()
                self.assertTrue(select.select([foreign.stdout], [], [], 3)[0])
                self.assertEqual(foreign.stdout.readline().strip(), "alive")
                queue.submit({
                    "task_id": "next-task", "agent_id": "next-owner", "devices": [12, 13],
                    "allow_external_busy": False,
                })
                self.assertEqual(queue.acquire("next-task", busy)["status"], "waiting")
        finally:
            for process in children:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=3)
                for stream in (process.stdin, process.stdout, process.stderr):
                    stream.close()


if __name__ == "__main__":
    unittest.main()
