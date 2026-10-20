"""Container SSH reservation release is owner-scoped and removal-gated."""
from unittest.mock import Mock

import pytest

from mindie_coordinator.env_manager import handle_container
from mindie_coordinator.host import mindie_npu_coordination as host
from mindie_coordinator.machine_directory import MachineDirectory


def request(user, port=0):
    return {"action": "container-ssh-reserve", "user": user,
            "container_name": "mindie-" + user, "port": port}


def remove_result(removed=True):
    return {"success": True, "container": "mindie-alice", "removed": removed,
            "previous_state": "running" if removed else None}


def machine_store(tmp_path):
    store = MachineDirectory(path=tmp_path / "machines.json")
    store.upsert_machine({
        "alias": "atlas-a3", "user": "alice",
        "host": {"ip": "192.0.2.1", "port": 22, "user": "root"},
        "container": {"name": "mindie-alice", "ssh_port": 46000, "user": "root"},
    })
    return store


class RecordingQueue:
    def __init__(self, failure=None):
        self.requests = []
        self.failure = failure

    def request(self, target, payload):
        self.requests.append((target, payload))
        if self.failure is not None:
            raise self.failure
        return {"status": "released", "released": True, "user": "alice",
                "container_name": "mindie-alice", "port": 46000}


def test_release_is_owner_scoped_idempotent_and_preserves_unrelated_work(tmp_path):
    coordinator = host.NpuCoordinator(tmp_path)
    coordinator.reserve_container_ssh(request("alice", 46000))
    coordinator.reserve_container_ssh(request("bob", 46001))
    # Seed a genuine unrelated task with a service-port reservation through
    # the public host protocol, not by writing SQL.
    coordinator.submit({"task_id": "other-task", "agent_id": "bob", "npu_count": 0,
                        "service_port": 48001, "service_ports": [48001]})
    granted = coordinator.acquire("other-task", None,
                                 listening={"status": "ok", "ports": []})
    assert granted["status"] == "granted"
    assert granted["task"]["granted_service_port"] == 48001
    # A non-owner release is idempotent absent; Bob's reservation stays.
    other = coordinator.release_container_ssh({"action": "container-ssh-release", "user": "carol",
                                                "container_name": "mindie-carol", "port": 0})
    assert other == {"status": "released", "released": False, "reason": "absent",
                     "user": "carol", "container_name": "mindie-carol", "port": 0, "task_id": "ssh.carol"}
    with pytest.raises(host.CoordinationError, match="SSH port 46000"):
        coordinator.release_container_ssh({"action": "container-ssh-release", "user": "alice",
                                            "container_name": "mindie-alice", "port": 46001})
    released = coordinator.release_container_ssh({"action": "container-ssh-release", "user": "alice",
                                                   "container_name": "mindie-alice", "port": 46000})
    assert released["released"] is True and released["port"] == 46000
    again = coordinator.release_container_ssh({"action": "container-ssh-release", "user": "alice",
                                                "container_name": "mindie-alice", "port": 0})
    assert again["released"] is False and again["reason"] == "absent"
    import sqlite3
    with sqlite3.connect(coordinator.db_path) as connection:
        ports = {(int(port), kind, task_id) for port, kind, task_id in connection.execute(
            "SELECT port, kind, task_id FROM ports")}
        tasks = {(task_id, state) for task_id, state in connection.execute(
            "SELECT task_id, state FROM tasks")}
    assert ports == {(46001, "container_ssh", "ssh.bob"), (48001, "service", "other-task")}
    assert tasks == {("other-task", "granted")}
    # The public host protocol action uses the same owner-scoped lifecycle.
    public = host.handle_request({"action": "container-ssh-release", "state_dir": str(tmp_path),
                                  "user": "alice", "container_name": "mindie-alice", "port": 0},
                                 probe=Mock(side_effect=AssertionError("release must not probe NPUs")),
                                 listening_ports=Mock(side_effect=AssertionError("release must not probe ports")))
    assert public["released"] is False and public["reason"] == "absent"


@pytest.mark.parametrize("removed", [True, False])
def test_confirmed_removal_or_absence_releases_owner_reservation(tmp_path, monkeypatch, removed):
    store = machine_store(tmp_path)
    queue = RecordingQueue()
    monkeypatch.setattr("mindie_coordinator.env_manager.host_ops.remove_container_result",
                        lambda ns: remove_result(removed))
    result = handle_container({"action": "remove", "alias": "atlas-a3"}, directory=store,
                              state_dir=tmp_path, queue=queue)
    assert result["state"] == "ready"
    assert result["removal"]["removed"] is removed
    assert queue.requests == [({"host": "192.0.2.1", "port": 22, "user": "root"},
                               {"action": "container-ssh-release", "user": "alice",
                                "container_name": "mindie-alice", "port": 46000})]
    assert "container" not in store.record("atlas-a3")


def test_failed_or_unknown_removal_keeps_reservation_and_binding(tmp_path, monkeypatch):
    store = machine_store(tmp_path)
    queue = RecordingQueue()
    def fail(ns):
        raise RuntimeError("remote removal outcome unknown")
    monkeypatch.setattr("mindie_coordinator.env_manager.host_ops.remove_container_result", fail)
    with pytest.raises(RuntimeError, match="outcome unknown"):
        handle_container({"action": "remove", "alias": "atlas-a3"}, directory=store,
                         state_dir=tmp_path, queue=queue)
    assert queue.requests == []
    assert store.record("atlas-a3")["container"]["name"] == "mindie-alice"


def test_release_failure_after_removal_is_partial_and_keeps_binding(tmp_path, monkeypatch):
    store = machine_store(tmp_path)
    queue = RecordingQueue(failure=RuntimeError("host authority unavailable"))
    monkeypatch.setattr("mindie_coordinator.env_manager.host_ops.remove_container_result",
                        lambda ns: remove_result(True))
    result = handle_container({"action": "remove", "alias": "atlas-a3"}, directory=store,
                              state_dir=tmp_path, queue=queue)
    assert result["state"] == "partial"
    assert result["removal"]["removed"] is True
    assert result["reservation_release"]["status"] == "failed"
    assert "host authority unavailable" in result["reservation_release"]["error"]
    assert store.record("atlas-a3")["container"]["name"] == "mindie-alice"


def test_provision_default_callback_uses_real_host_authority(tmp_path, monkeypatch):
    import sqlite3
    from unittest.mock import Mock
    from mindie_coordinator import provision
    from mindie_coordinator.host_queue import HostQueue
    from mindie_coordinator.provision.host_ops import RemoteResult

    state = tmp_path / "authority"

    class RealAuthorityQueue(HostQueue):
        def __init__(self):
            super().__init__(lambda _target, _command: "{}")

        def request(self, target, payload):
            return host.handle_request(
                {**dict(payload), "state_dir": str(state)},
                probe=Mock(side_effect=AssertionError("container reservation must not probe NPUs")),
                listening_ports=lambda: {"status": "ok", "ports": [], "source": "test"},
            )

    monkeypatch.setattr("mindie_coordinator.host_queue.HostQueue", lambda: RealAuthorityQueue())
    monkeypatch.setattr(provision.host_ops, "run_remote_script",
                        lambda *a, **k: RemoteResult(None, 0, "", "", {"success": True, "free_port": 46000}))
    monkeypatch.setattr(provision.host_ops, "assert_remote_success",
                        Mock(side_effect=[{"free_port": 46000}, {}, {}]))
    monkeypatch.setattr(provision.host_ops, "find_public_key", lambda _: "fixture-key")
    monkeypatch.setattr(provision.host_ops, "load_public_key", lambda _: "ssh-ed25519 fixture")
    result = provision.provision_user_container(host="fixture", image="main", user="alice", machines=Mock())
    assert result["ssh_port"] == 46000
    with sqlite3.connect(state / "coordinator.sqlite3") as connection:
        row = connection.execute("SELECT port, owner FROM ports WHERE kind='container_ssh'").fetchone()
    assert row == (46000, "alice")


def test_public_container_remove_partial_is_non_success(monkeypatch, capsys):
    from mindie_coordinator import cli, ops
    partial = {
        "state": "partial", "action": "remove", "alias": "atlas-a3",
        "container_name": "mindie-alice",
        "removal": {"success": True, "removed": True, "previous_state": "running"},
        "reservation_release": {"status": "failed", "error": "host authority unavailable"},
        "remote_dev_endpoint": None,
    }
    monkeypatch.setattr("mindie_coordinator.env_manager.handle_container",
                        lambda args, state_dir: partial)
    reply = ops.mindie_call("mindie.container", {"action": "remove", "alias": "atlas-a3"},
                            allow_native_context=False)["result"]
    assert reply["status"] == "partial"
    assert reply["outcome"] != "success"
    assert reply["data"]["removal"]["removed"] is True
    assert reply["data"]["reservation_release"]["status"] == "failed"
    assert cli.main(["container", "--action", "remove", "--alias", "atlas-a3"]) == 1
    assert "partial" in capsys.readouterr().out


def _run_remove_script(tmp_path, docker_body):
    import json
    from pathlib import Path
    import shlex
    import shutil
    import subprocess
    import sys
    from mindie_coordinator.provision import host_ops
    bindir = tmp_path / "bin"
    bindir.mkdir()
    docker = bindir / "docker"
    docker.write_text("#!/bin/sh\n" + docker_body, encoding="utf-8", newline="\n")
    docker.chmod(0o755)
    python = bindir / "python3"
    python.write_text(
        "#!/bin/sh\nexec " + shlex.quote(Path(sys.executable).as_posix()) + ' "$@"\n',
        encoding="utf-8", newline="\n",
    )
    python.chmod(0o755)
    bash = shutil.which("bash")
    assert bash is not None, "Bash is required for the local Linux-peer fixture"
    completed = subprocess.run(
        [bash, "-c", 'export PATH="$PWD/bin:$PATH"\n'
         + host_ops.render_remove_container_host_script(), "remove", "mindie-alice"],
        capture_output=True, text=True, cwd=tmp_path, timeout=20,
    )
    sentinel = next((line for line in completed.stdout.splitlines()
                     if line.startswith(host_ops.SENTINEL)), None)
    assert sentinel is not None, (completed.returncode, completed.stdout, completed.stderr)
    return completed.returncode, json.loads(sentinel[len(host_ops.SENTINEL):])


@pytest.mark.parametrize("docker_body,expected_rc,expected", [
    ('''case "$1" in
  inspect) echo "Error: No such object: mindie-alice" >&2; exit 1 ;;
esac
exit 0
''', 0, {"success": True, "removed": False, "previous_state": None}),
    ('''case "$1" in
  inspect) echo "Cannot connect to the Docker daemon" >&2; exit 1 ;;
esac
exit 0
''', 12, {"success": False, "error": "docker inspect failed"}),
    ('''case "$1" in
  inspect)
    case "$2" in -f) echo running ;; esac
    exit 0 ;;
  rm) exit 0 ;;
esac
exit 0
''', 0, {"success": True, "removed": True, "previous_state": "running"}),
])
def test_remove_script_distinguishes_absence_from_docker_failure(tmp_path, docker_body, expected_rc, expected):
    rc, payload = _run_remove_script(tmp_path, docker_body)
    assert rc == expected_rc
    for key, value in expected.items():
        assert payload[key] == value


def test_local_known_hosts_cleanup_only_after_true_removal_and_isolated_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from mindie_coordinator.provision import host_ops
    from mindie_coordinator.provision.host_ops import RemoteResult
    args = SimpleNamespace(host="192.0.2.1", user="root", host_port=22,
                           container_name="mindie-alice", container_ssh_port=46000,
                           known_hosts=str(tmp_path / "known_hosts"),
                           clean_local_known_hosts=True)
    monkeypatch.setattr(host_ops, "run_remote_script",
                        lambda *a, **k: RemoteResult(None, 0, "", "", {
                            "success": True, "container": "mindie-alice",
                            "removed": True, "previous_state": "running"}))
    monkeypatch.setattr(host_ops, "remove_known_host_entry",
                        Mock(side_effect=TimeoutError("local cleanup timed out")))
    payload = host_ops.remove_container_result(args)
    assert payload["success"] is True and payload["removed"] is True
    assert payload["local_known_hosts_cleanup"]["success"] is False
    assert "TimeoutError" in payload["local_known_hosts_cleanup"]["error"]
    # Confirmed absence does not touch a host key that may now belong to another container.
    monkeypatch.setattr(host_ops, "run_remote_script",
                        lambda *a, **k: RemoteResult(None, 0, "", "", {
                            "success": True, "container": "mindie-alice",
                            "removed": False, "previous_state": None}))
    cleanup = Mock(side_effect=AssertionError("absent container must not clean known_hosts"))
    monkeypatch.setattr(host_ops, "remove_known_host_entry", cleanup)
    absent = host_ops.remove_container_result(args)
    assert absent["removed"] is False and absent["local_known_hosts_cleanup"]["skipped"] is True
    cleanup.assert_not_called()
