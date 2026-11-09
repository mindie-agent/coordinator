"""Business interpreter identity survives selection, recording and isolation."""
from pathlib import Path
from types import SimpleNamespace
import shlex
import subprocess
import sys
import venv

from mindie_coordinator.backend import RemoteBackend
from mindie_coordinator.machine_directory import MachineDirectory
from mindie_coordinator.provision import _record_ready_container
from mindie_coordinator.provision.task_environment import create_venv_script
from mindie_coordinator.service_gang import ServiceGangMixin


def make_environment(root):
    venv.EnvBuilder(with_pip=False).create(root)
    return root / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def test_command_environment_preserves_actual_virtualenv_interpreter(tmp_path, monkeypatch):
    python = make_environment(tmp_path / "selected")
    client = RemoteBackend()

    def local_bash(endpoint, script):
        argv = shlex.split(script.splitlines()[-1])
        assert argv[0] == "/selected/venv/bin/python"
        proc = subprocess.run([str(python), *argv[1:]], capture_output=True, text=True,
                              check=True, timeout=5,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        # The transport seam maps local fixture paths to the remote namespace.
        # Unwrapping _base_executable would fail this actual virtualenv check.
        assert Path(proc.stdout.strip()).absolute() == python.absolute()
        return "/selected/venv/bin/python\n"

    monkeypatch.setattr(client, "bash", local_bash)
    result = client.command_environment({"python": "/selected/venv/bin/python", "endpoint": {"root": "/sources/new"}})
    assert result["python"] == "/selected/venv/bin/python"


def test_selected_readiness_python_is_recorded_and_returned(tmp_path):
    directory = MachineDirectory(path=tmp_path / "machines.json")
    result = _record_ready_container("example.invalid", "fixture-image", "alice", "root", 22,
                                     46000, "A2", directory, python_path="/selected/venv/bin/python")
    assert result["python"] == "/selected/venv/bin/python"
    records, _ = directory.load()
    assert records["machines"][0]["container"]["python"] == result["python"]


def test_new_venv_does_not_claim_private_donor_dependencies(tmp_path):
    donor = make_environment(tmp_path / "donor")
    task = tmp_path / "task/.venv"
    # Create the task with the actual donor interpreter. This does not inherit
    # another virtualenv's private site-packages, even with system-site-packages.
    subprocess.run([str(donor), "-m", "venv", "--without-pip", "--system-site-packages", str(task)],
                   capture_output=True, check=True, timeout=5,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    python = task / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    script = create_venv_script("/task", "/task/.venv/bin/python", "/donor/bin/python")
    validation = script.split("<<'MINDIE_VENV'\n", 1)[1].split("\nMINDIE_VENV", 1)[0]
    # Hide real machine packages; package absence must not depend on the host.
    prelude = "import sys; sys.path[:] = [p for p in sys.path if 'site-packages' not in p]\n"
    proc = subprocess.run([str(python), "-I", "-c", prelude + validation, str(task), str(donor)],
                          capture_output=True, text=True, timeout=5,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert proc.returncode != 0
    assert "cannot inherit donor dependencies: torch, torch-npu" in proc.stderr
    assert "donor was not modified" in proc.stderr


def test_container_interpreter_reaches_preparation_without_crossing_containers(tmp_path):
    directory = MachineDirectory(path=tmp_path / "machines.json")
    directory.upsert_machine({"alias": "fixture", "host": {"ip": "example.invalid", "port": 22},
                              "container": {"name": "user-dev", "mode": "docker_exec",
                                            "python": "/selected/venv/bin/python"}})
    client = ServiceGangMixin()
    client.backend = SimpleNamespace(machines=directory)
    donor = client._ensure_user_container("alice", {"container_name": "user-dev"}, {}, set())
    assert donor["python"] == "/selected/venv/bin/python"
    other = client._ensure_user_container("alice", {"container_name": "other-dev"}, {}, set())
    assert "python" not in other


def test_readiness_transport_stays_within_windows_ssh_command_limit():
    from remote_dev.core.endpoint import Endpoint
    from remote_dev.core.python_discovery import compact_shell_script
    from remote_dev.core.ssh_transport import stream_ssh_command
    from mindie_coordinator.provision.host_probe_scripts import render_smoke_script
    script = 'set -- ""\n' + compact_shell_script(render_smoke_script(device_test=False))
    argv = stream_ssh_command(Endpoint(host="example.invalid", port=22, ssh_mux=False), script, timeout_ms=240000)
    assert len(subprocess.list2cmdline(argv)) < 8191


def test_repeated_provision_preserves_verified_container_interpreter_and_key(tmp_path, monkeypatch):
    from mindie_coordinator import provision
    from mindie_coordinator.provision import existing_container, host_ops

    directory = MachineDirectory(path=tmp_path / 'machines.json')
    replies = iter([{'machine_type': 'A2', 'free_port': 46000}, {},
                    {'python_path': '/business/venv/bin/python'}])
    monkeypatch.setattr(host_ops, 'run_remote_script', lambda *a, **k: next(replies))
    monkeypatch.setattr(host_ops, 'assert_remote_success', lambda result, **k: result)
    monkeypatch.setattr(host_ops, 'ensure_key', lambda _: ('/keys/container', 'ssh-ed25519 test'))
    options = dict(host='example.invalid', image='registry.example/image:fixed', user='alice',
                   ssh_port=46000, machines=directory, reserve_port=lambda **_: {'port': 46000})
    monkeypatch.setattr(existing_container, 'observe_existing', lambda *a, **k: {'status': 'unusable'})
    first = provision.provision_user_container(**options)
    monkeypatch.setattr(existing_container, 'observe_existing', lambda *a, **k: {'status': 'match'})
    second = provision.provision_user_container(**options)
    assert first['python'] == second['python'] == '/business/venv/bin/python'
    stored = directory.get('example.invalid')['container']
    assert stored['python'] == second['python']
    assert stored['identity_file'] == '/keys/container'


def test_fast_reuse_does_not_take_interpreter_from_another_endpoint(tmp_path, monkeypatch):
    from mindie_coordinator import provision
    from mindie_coordinator.provision import existing_container

    directory = MachineDirectory(path=tmp_path / 'machines.json')
    _record_ready_container('example.invalid', 'registry.example/image:fixed', 'alice', 'root', 22,
                            46000, 'A2', directory, python_path='/business/venv/bin/python')
    monkeypatch.setattr(existing_container, 'observe_existing', lambda *a, **k: {'status': 'match'})
    result = provision.provision_user_container(
        host='example.invalid', image='registry.example/image:fixed', user='alice', ssh_port=46001,
        machines=directory, reserve_port=lambda **_: {'port': 46001})
    assert 'python' not in result
    assert 'python' not in directory.get('example.invalid')['container']
