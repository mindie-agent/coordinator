"""Coordinator persists shared auth references without owning SSH credentials."""
from __future__ import annotations

import io
import json
import os
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mindie_coordinator.endpoint_bridge import build_endpoints_document
from mindie_coordinator.env_manager import handle_hosts
from mindie_coordinator.machine_directory import MachineDirectory
from mindie_coordinator.provision import host_ops


def test_password_access_remains_ready_when_key_installation_is_unavailable(tmp_path, monkeypatch):
    calls = []

    def setup(endpoint, **arguments):
        calls.append((endpoint, arguments))
        return {"auth_ref": "host-profile", "auth_method": "password", "identity_file": None,
                "bootstrap": {"status": "failed", "error": "authorized_keys is not writable"}}

    monkeypatch.setattr(host_ops, "setup_access", setup)
    monkeypatch.setattr("mindie_coordinator.env_manager.NpuTopObservationClient.register_host",
                        lambda _self, **arguments: {"status": "registered", **arguments})
    result = handle_hosts({
        "action": "bootstrap_key", "alias": "new-builder", "host": "host.invalid", "user": "developer",
        "password_env": "TEST_SSH_PASSWORD",
    }, state_dir=tmp_path)

    assert result["state"] == "ready"
    assert result["authentication"]["auth_method"] == "password"
    assert result["authentication"]["bootstrap"]["error"] == "authorized_keys is not writable"
    stored = MachineDirectory(path=tmp_path / "machines.json").record("new-builder")
    assert stored["host"]["auth_ref"] == "host-profile"
    assert result["npu_top_sync"]["auth_ref"] == "host-profile"
    assert "password" not in result["npu_top_sync"]
    assert calls[0][1]["password_env"] == "TEST_SSH_PASSWORD"
    assert calls[0][0].host == "host.invalid"
    assert calls[0][0].user == "developer"


def test_bootstrap_reuses_shared_profile_and_persists_the_verified_key(tmp_path, monkeypatch):
    store = MachineDirectory(path=tmp_path / "machines.json")
    store.upsert_machine({"alias": "builder", "host": {
        "ip": "host.invalid", "user": "developer", "port": 2200, "auth_ref": "existing-profile",
    }})
    observed = []

    def setup(endpoint, **arguments):
        observed.append((endpoint, arguments))
        return {"auth_ref": "existing-profile", "auth_method": "publickey",
                "identity_file": "/managed/verified-key", "bootstrap": {"status": "completed"}}

    monkeypatch.setattr(host_ops, "setup_access", setup)
    monkeypatch.setattr("mindie_coordinator.env_manager.NpuTopObservationClient.register_host",
                        lambda _self, **arguments: {"status": "registered"})
    handle_hosts({"action": "bootstrap_key", "alias": "builder"}, directory=store)

    assert observed[0][0].auth_ref == "existing-profile"
    assert observed[0][1]["password"] is None
    endpoint = store.endpoints_document()["endpoints"]["builder"]
    assert endpoint["identity_file"] == "/managed/verified-key"
    assert endpoint["auth_ref"] == "existing-profile"


def test_authentication_reference_follows_the_actual_ssh_endpoint():
    from mindie_coordinator.ready_runtime import endpoint as runtime_endpoint

    row = {"alias": "builder", "host": {
        "ip": "host.invalid", "port": 22, "user": "developer",
        "identity_file": "/keys/host", "auth_ref": "host-profile",
    }, "container": {
        "name": "work", "ssh_port": 2200, "user": "root",
        "identity_file": "/keys/container", "auth_ref": "container-profile",
    }}
    endpoints = build_endpoints_document({"machines": [row]})["endpoints"]
    assert endpoints["builder"]["auth_ref"] == "container-profile"
    assert endpoints["builder"]["identity_file"] == "/keys/container"
    assert endpoints["builder-host"]["auth_ref"] == "host-profile"
    assert endpoints["builder-docker"]["auth_ref"] == "host-profile"
    prepared = runtime_endpoint(endpoints["builder"], container=True)
    assert prepared["auth_ref"] == "container-profile"
    assert prepared["identity_file"] == "/keys/container"

    row["container"].pop("auth_ref")
    row["container"].pop("identity_file")
    endpoints = build_endpoints_document({"machines": [row]})["endpoints"]
    assert "auth_ref" not in endpoints["builder"]
    assert "identity_file" not in endpoints["builder"]


def test_legacy_bootstrap_stdin_delegates_to_shared_auth(monkeypatch, capsys):
    observed = []

    def setup(endpoint, **arguments):
        observed.append((endpoint, arguments))
        return {"auth_ref": "shared-profile", "auth_method": "password",
                "identity_file": None, "bootstrap": {"status": "skipped"}}

    monkeypatch.setattr(host_ops, "setup_access", setup)
    monkeypatch.setattr(host_ops.sys, "stdin", io.StringIO("fixture-password\n"))
    result = host_ops.cmd_bootstrap_host_key(SimpleNamespace(
        host="host.invalid", host_port=22, user="developer",
        password_stdin=True, public_key_file="/keys/custom.pub",
    ))
    output = capsys.readouterr().out
    assert result == 0
    assert json.loads(output)["auth_ref"] == "shared-profile"
    assert "fixture-password" not in output
    assert observed[0][1]["password"] == "fixture-password"
    assert observed[0][1]["identity_file"] == "/keys/custom.pub"


def test_shared_auth_failure_does_not_register_a_ready_host(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("host key changed")

    monkeypatch.setattr(host_ops, "setup_access", fail)
    with pytest.raises(RuntimeError, match="host key changed"):
        handle_hosts({"action": "upsert", "host": "host.invalid", "password_env": "SECRET"},
                     state_dir=tmp_path)
    assert not (tmp_path / "machines.json").exists()


def test_git_transport_preserves_shared_askpass_environment(tmp_path, monkeypatch):
    from mindie_coordinator.parity import git_push_url, git_ssh_environment
    from mindie_coordinator.parity_support import SshEndpoint
    from remote_dev.core import ssh_transport

    calls = []
    helper = tmp_path / "SSH fixture.py"
    received = tmp_path / "received.json"
    helper.write_text(
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['TEST_SSH_RECEIVED']).write_text(json.dumps(sys.argv[1:]))\n"
        "sys.stdout.write('0000')\n", encoding="utf-8",
    )

    def launch(endpoint, command=None, **arguments):
        calls.append((endpoint, command))
        argv = ["ssh", "-o", "BatchMode=no", "-p", str(endpoint.port), "--", endpoint.host]
        if command:
            argv.append(command)
            argv = [sys.executable, str(helper), *argv]
        return ssh_transport.SshLaunch(
            argv, {**os.environ, "SSH_ASKPASS": "/shared/askpass", "REMOTE_DEV_AUTH_REF": "shared-profile"}, (),
        )

    monkeypatch.setattr(ssh_transport, "ssh_launch", launch)
    endpoint = SshEndpoint("host.invalid", 22, "developer", container="work", auth_ref="shared-profile")
    environment = git_ssh_environment(endpoint)
    assert environment["SSH_ASKPASS"] == "/shared/askpass"
    assert environment["REMOTE_DEV_AUTH_REF"] == "shared-profile"
    assert shlex.split(environment["GIT_SSH_COMMAND"]) == [
        "ssh", "-T", "-o", "BatchMode=no", "-p", "22",
    ]
    url = git_push_url(endpoint, "/mirror/repository % name.git")
    subprocess.run(["git", "-c", "protocol.ext.allow=always", "ls-remote", url],
                   env={**environment, "TEST_SSH_RECEIVED": str(received)}, check=True,
                   capture_output=True, timeout=5)
    assert json.loads(received.read_text())[-1] == (
        "docker exec -i work git receive-pack '/mirror/repository % name.git'"
    )
    assert all(call[0].auth_ref == "shared-profile" for call in calls)
