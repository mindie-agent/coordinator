"""Execute the actual profile-probe script with stdlib command profiles."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig

import pytest

from mindie_coordinator.backend import RemoteBackend
from mindie_coordinator.parity_support import RemoteCommandError
from mindie_coordinator.runtime_profile import digest, launch_preamble, profile_key


class LocalProbeBackend(RemoteBackend):
    """Replace transport/Docker, execute the unmodified remote Python script."""

    def __init__(self):
        super().__init__()
        self.container = {"Id": "registered-container", "State": {"Running": True}}
        self.replies = []
        self.requests = []

    def bash(self, target, command):
        if command.startswith("docker inspect"):
            return json.dumps(self.container)
        header, body = command.split(" <<'MINDIE_READY_PROBE'\n", 1)
        script, trailer = body.rsplit("\nMINDIE_READY_PROBE\n", 1)
        assert not trailer
        arguments = shlex.split(header.splitlines()[-1])
        request = arguments[2]
        self.requests.append(json.loads(request))
        result = subprocess.run(
            [sys.executable, "-", request], input=script, text=True,
            capture_output=True, cwd=target["root"], timeout=30,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            encoding="utf-8",
        )
        if result.returncode:
            raise RemoteCommandError(result.returncode, result.stderr)
        self.replies.append(result.stdout)
        return result.stdout


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8",
        stderr=subprocess.PIPE,
    ).strip()


@pytest.fixture
def prepared_view(tmp_path):
    root = tmp_path / "view"
    root.mkdir()
    snapshots = {}
    for name in ("vllm", "vllm-ascend"):
        repo = root / name
        repo.mkdir()
        git(repo, "init")
        git(repo, "config", "user.name", "Probe test")
        git(repo, "config", "user.email", "probe@example.invalid")
        (repo / "source.py").write_text("value = 1\n", encoding="utf-8")
        git(repo, "add", "source.py")
        git(repo, "commit", "-m", "source")
        snapshots[name] = git(repo, "rev-parse", "HEAD")
    profile = {
        "kind": "command", "image_digest": "sha256:test-image",
        "python_abi": sysconfig.get_config_var("SOABI"),
        "build_env": {}, "launch_env": {},
    }
    manifest = {
        "schema_version": 2, "profile": profile, "profile_key": profile_key(profile),
        "build_key": digest({"profile": profile, "sources": "accepted-inputs"}),
        "source_id": "accepted-inputs", "runtime_root": str(root.resolve()),
        "files": {}, "evidence": {}, "build_inputs": {},
        # This field is outside profile/build keys; equality must still cover it.
        "preparation": {"notes": "retained-metadata" * 22000, "generation": 1},
    }
    marker = root / ".mindie-runtime/ready-profile.json"
    marker.parent.mkdir()
    marker.write_text(json.dumps(manifest), encoding="utf-8")
    backend = LocalProbeBackend()
    runtime = {
        "host_endpoint": {"host": "example.invalid", "port": 22, "user": "root"},
        "endpoint": {"host": "example.invalid", "port": 46001, "user": "root",
                     "root": str(root), "cwd": str(root)},
        "container_name": "owned-container", "python": sys.executable,
        "attestation": {**copy.deepcopy(manifest), "container_id": backend.container["Id"],
                        "launch_preamble": launch_preamble(profile, python=sys.executable)},
    }
    return backend, runtime, snapshots, marker, manifest


def test_compact_probe_matches_full_inspection_and_returns_only_digest(prepared_view):
    backend, runtime, snapshots, _marker, manifest = prepared_view
    assert backend.inspect(runtime, snapshots=snapshots) == runtime["attestation"]
    assert len(backend.replies[-1]) > 300000
    assert backend.verify_preflight(runtime, snapshots=snapshots) is True
    assert json.loads(backend.replies[-1]) == {"manifest_digest": digest(manifest)}
    assert len(backend.replies[-1]) < 100
    assert backend.requests[-1]["snapshots"] == snapshots
    assert backend.requests[-1]["expected_manifest_digest"] == digest(manifest)
    assert "expected_manifest_digest" not in backend.requests[0]


def test_metadata_outside_profile_and_build_keys_still_rejects(prepared_view):
    backend, runtime, snapshots, marker, manifest = prepared_view
    manifest["preparation"]["generation"] += 1
    marker.write_text(json.dumps(manifest), encoding="utf-8")
    assert backend.inspect(runtime, snapshots=snapshots) != runtime["attestation"]
    with pytest.raises(RemoteCommandError, match="runtime changed before launch"):
        backend.verify_preflight(runtime, snapshots=snapshots)


@pytest.mark.parametrize("source", ["vllm", "vllm-ascend"])
@pytest.mark.parametrize("change", ["tracked", "untracked", "commit"])
def test_every_pinned_git_source_is_checked(prepared_view, source, change):
    backend, runtime, snapshots, _marker, _manifest = prepared_view
    repo = Path(runtime["endpoint"]["root"]) / source
    if change == "untracked":
        (repo / "extra.py").write_text("unaccepted = True\n", encoding="utf-8")
    else:
        (repo / "source.py").write_text("value = 2\n", encoding="utf-8")
        if change == "commit":
            git(repo, "add", "source.py")
            git(repo, "commit", "-m", "different input")
    with pytest.raises(RemoteCommandError, match="runtime source differs from pinned snapshot: " + source):
        backend.verify_preflight(runtime, snapshots=snapshots)


def test_complete_profile_verification_precedes_digest_confirmation(prepared_view):
    backend, runtime, snapshots, marker, manifest = prepared_view
    manifest["profile"]["python_abi"] = "different-abi"
    manifest["profile_key"] = profile_key(manifest["profile"])
    manifest["build_key"] = digest({"profile": manifest["profile"], "sources": manifest["source_id"]})
    marker.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RemoteCommandError, match="command interpreter or execution root changed"):
        backend.verify_preflight(runtime, snapshots=snapshots)


def test_container_replacement_rejects_even_when_manifest_probe_succeeds(prepared_view):
    backend, runtime, snapshots, _marker, _manifest = prepared_view
    backend.container["Id"] = "replacement-container"
    with pytest.raises(ValueError, match="runtime container changed"):
        backend.verify_preflight(runtime, snapshots=snapshots)
    assert len(backend.requests) == 1


def test_derived_launch_preamble_must_equal_registered_value(prepared_view):
    backend, runtime, snapshots, _marker, _manifest = prepared_view
    runtime["attestation"]["launch_preamble"] += "\nexport UNREGISTERED=1"
    with pytest.raises(ValueError, match="runtime launch environment changed"):
        backend.verify_preflight(runtime, snapshots=snapshots)
    assert backend.requests == []


@pytest.mark.parametrize("response", ["", "{}", "null", "[]", '{"manifest_digest":',
                                     json.dumps({"manifest_digest": "0" * 64})])
def test_incomplete_or_false_remote_confirmation_is_not_success(prepared_view, monkeypatch, response):
    backend, runtime, snapshots, _marker, _manifest = prepared_view
    original = backend.bash
    monkeypatch.setattr(backend, "bash", lambda target, command:
                        original(target, command) if command.startswith("docker inspect") else response)
    with pytest.raises(ValueError):
        backend.verify_preflight(runtime, snapshots=snapshots)


@pytest.fixture
def adapter_pool(tmp_path):
    # Existing adapters exercise the real host SQLite protocol without devices.
    from test_coordinator import Backend, RuntimePool, runtime_spec

    backend = Backend(tmp_path / "host")
    pool = RuntimePool(tmp_path / "pool", backend)
    pool.register("runtime", runtime_spec(1))
    session = pool.session_open("alice", "compact-test", {})
    binding = pool.checkout("alice", session["id"], "profile-a", "checkout")
    backend.calls.clear()
    return backend, pool, binding


def test_managed_preflight_uses_compact_check_and_legacy_adapter_falls_back(adapter_pool):
    backend, pool, binding = adapter_pool
    checked = []
    backend.verify_preflight = lambda runtime, **kwargs: checked.append(kwargs) or True
    job = pool.managed_start("alice", binding["id"], "compact", {}, "native-a", [], 0, "true", {})
    assert job["state"] == "running"
    assert checked == [{"snapshots": {}}]
    assert not any(call[0] == "inspect" for call in backend.calls)
    pool.managed_control("alice", job["id"], "stop")
    pool.managed_control("alice", job["id"])
    del backend.verify_preflight
    backend.calls.clear()
    fallback = pool.managed_start("alice", binding["id"], "legacy", {}, "native-a", [], 0, "true", {})
    assert fallback["state"] == "running"
    assert any(call[0] == "inspect" for call in backend.calls)


def test_public_control_keeps_full_inspection(adapter_pool):
    backend, pool, binding = adapter_pool

    def unexpected_compact(*args, **kwargs):
        raise AssertionError("public control must retain its full probe")

    backend.verify_preflight = unexpected_compact
    run = pool.request_run("alice", binding["id"], "public", {}, "native-a", [], 0)
    assert run["state"] == "granted"
    backend.calls.clear()
    assert pool.control("alice", run["id"], "preflight")["state"] == "starting"
    assert any(call[0] == "inspect" for call in backend.calls)


@pytest.mark.parametrize("result", [None, False, {}])
def test_managed_hook_requires_explicit_confirmation_before_payload(adapter_pool, result):
    backend, pool, binding = adapter_pool
    backend.verify_preflight = lambda *args, **kwargs: result
    job = pool.managed_start("alice", binding["id"], "rejected", {}, "native-a", [], 0, "must-not-run", {})
    assert (job["state"], job["lease_state"]) == ("failed", "cancelled")
    assert "did not confirm" in job["error"]
    assert backend.jobs == {}
    assert not any(call in backend.calls for call in (("job", "prepare"), ("job", "go")))
