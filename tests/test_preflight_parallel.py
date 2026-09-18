"""Both launch observations finish before preflight succeeds or fails."""
from concurrent.futures import ThreadPoolExecutor
import threading
from unittest.mock import Mock

import pytest
from remote_dev.core.cancellation import current_event, request_context

from mindie_coordinator import backend as module
from mindie_coordinator.runtime_profile import digest


@pytest.fixture
def launch(monkeypatch):
    monkeypatch.setattr(module, "launch_preamble", lambda *args, **kwargs: "registered launch")
    manifest = {"profile": {"kind": "command"}, "runtime_root": "/owned"}
    runtime = {"python": "/owned/python", "prepared_native_view": True,
               "attestation": {**manifest, "container_id": "container-a",
                               "launch_preamble": "registered launch"}}
    return module.RemoteBackend(), runtime, {"manifest_digest": digest(manifest)}


def test_container_and_source_observations_overlap_in_one_preflight(launch, monkeypatch):
    backend, runtime, reply = launch
    entered = threading.Barrier(2)
    snapshots = {"project": "a" * 40}
    calls = []
    cancelled = threading.Event()

    def container(value):
        assert value is runtime
        assert current_event() is cancelled
        entered.wait(timeout=3)
        calls.append("container")
        return {"Id": "container-a"}

    def manifest(value, **kwargs):
        assert value is runtime
        assert current_event() is cancelled
        assert kwargs == {"snapshots": snapshots, "expected_digest": reply["manifest_digest"],
                          "prepared_view": True}
        entered.wait(timeout=3)
        calls.append("manifest")
        return reply

    monkeypatch.setattr(backend, "_inspect_container", container)
    monkeypatch.setattr(backend, "_inspect_manifest", manifest)
    with request_context(cancelled):
        assert backend.verify_preflight(runtime, snapshots=snapshots) is True
    assert sorted(calls) == ["container", "manifest"]


@pytest.mark.parametrize("failed_check", ["container", "manifest"])
def test_failure_waits_for_the_other_observation_and_preserves_exception(launch, monkeypatch, failed_check):
    backend, runtime, reply = launch
    failed, waiting, release, finished = (threading.Event() for _ in range(4))
    error = ValueError(f"{failed_check} changed")

    def inspect(name):
        if name == failed_check:
            failed.set()
            raise error
        waiting.set()
        assert release.wait(timeout=3)
        finished.set()
        return {"Id": "container-a"} if name == "container" else reply

    monkeypatch.setattr(backend, "_inspect_container", lambda *args, **kwargs: inspect("container"))
    monkeypatch.setattr(backend, "_inspect_manifest", lambda *args, **kwargs: inspect("manifest"))
    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(backend.verify_preflight, runtime)
        try:
            assert failed.wait(timeout=3)
            assert waiting.wait(timeout=3)
            assert not future.done(), "failure must not leave a remote observation running"
        finally:
            release.set()
        assert future.exception(timeout=3) is error
    assert finished.is_set()


def test_both_observation_errors_are_reported_after_both_complete(launch, monkeypatch):
    backend, runtime, _ = launch
    entered = threading.Barrier(2)
    container_error = OSError("container transport unavailable")
    source_error = ValueError("fixed source changed")

    def fail(error):
        entered.wait(timeout=3)
        raise error

    monkeypatch.setattr(backend, "_inspect_container", lambda *args, **kwargs: fail(container_error))
    monkeypatch.setattr(backend, "_inspect_manifest", lambda *args, **kwargs: fail(source_error))
    with pytest.raises(ExceptionGroup) as captured:
        backend.verify_preflight(runtime)
    assert captured.value.exceptions == (container_error, source_error)
    assert "container transport unavailable" in str(captured.value)
    assert "fixed source changed" in str(captured.value)


def test_invalid_local_launch_description_starts_no_remote_observation(launch, monkeypatch):
    backend, runtime, _ = launch
    runtime["attestation"]["launch_preamble"] = "unregistered launch"
    container, manifest = Mock(), Mock()
    monkeypatch.setattr(backend, "_inspect_container", container)
    monkeypatch.setattr(backend, "_inspect_manifest", manifest)
    with pytest.raises(ValueError, match="launch environment changed"):
        backend.verify_preflight(runtime)
    container.assert_not_called()
    manifest.assert_not_called()


def test_historical_donor_keeps_complete_adoption_check(launch, monkeypatch):
    backend, runtime, _ = launch
    runtime["reuse_only"] = True
    full = Mock(return_value=runtime["attestation"])
    monkeypatch.setattr(backend, "inspect", full)
    monkeypatch.setattr(backend, "_inspect_container", Mock(side_effect=AssertionError("separate check")))
    monkeypatch.setattr(backend, "_inspect_manifest", Mock(side_effect=AssertionError("separate check")))
    snapshots = {"project": "a" * 40}
    assert backend.verify_preflight(runtime, snapshots=snapshots) is True
    full.assert_called_once_with(runtime, snapshots=snapshots)
