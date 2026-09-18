"""Stable native attachment references survive competing Windows publishers."""
from concurrent.futures import ThreadPoolExecutor
import errno
import json
from pathlib import Path
import subprocess
import sys

import pytest

from mindie_coordinator import agent_session
from mindie_coordinator.agent_session import AgentSessions, load_context


@pytest.fixture
def attachment(tmp_path):
    store = AgentSessions(tmp_path / "state")
    context = store.attach("codex", "native", str(tmp_path))
    path = Path(context["context_file"])
    return store, context, path, json.loads(path.read_text())


def assert_no_temporary_files(path):
    assert list(path.parent.glob("*.tmp")) == []


def test_resume_reuses_complete_reference_while_reader_keeps_it_open(attachment, monkeypatch):
    store, context, path, reference = attachment
    store.detach(context)

    def unexpected_replace(*args):
        pytest.fail("resuming the same attachment must not replace its stable reference")

    monkeypatch.setattr(agent_session.os, "replace", unexpected_replace)
    with path.open() as reader:
        resumed = store.attach("codex", "native", context["attachment"]["cwd"])
        assert json.load(reader) == reference
        assert load_context(str(path))["attachment"]["state"] == "attached"
    assert resumed["session"]["id"] == context["session"]["id"]
    assert_no_temporary_files(path)


@pytest.mark.parametrize("winerror", [5, 32])
def test_competing_complete_publication_satisfies_denied_replace(attachment, monkeypatch, winerror):
    store, context, path, reference = attachment
    path.unlink()
    denied = PermissionError(errno.EACCES, "competing publisher holds the destination")
    denied.winerror = winerror
    attempts = []

    def competing_publish(source, destination):
        attempts.append(source)
        # The competitor wins after the initial absent-file check. Its result
        # is complete, just as a separate process's atomic publication would be.
        Path(destination).write_text(json.dumps(reference), encoding="utf-8")
        raise denied

    monkeypatch.setattr(agent_session.os, "replace", competing_publish)
    assert store._publish(context["attachment"]["id"])["session"]["id"] == context["session"]["id"]
    assert load_context(str(path))["session"]["id"] == context["session"]["id"]
    assert len(attempts) == 1
    assert_no_temporary_files(path)


@pytest.mark.parametrize("replacement", [None, "{", json.dumps({"attachment_id": "another-task"})])
def test_denied_replace_without_matching_complete_reference_still_fails(attachment, monkeypatch, replacement):
    store, context, path, _reference = attachment
    path.unlink()
    denied = PermissionError(errno.EACCES, "access denied")

    def denied_replace(source, destination):
        if replacement is not None:
            Path(destination).write_text(replacement, encoding="utf-8")
        raise denied

    monkeypatch.setattr(agent_session.os, "replace", denied_replace)
    with pytest.raises(PermissionError) as failure:
        store._publish(context["attachment"]["id"])
    assert failure.value is denied
    assert_no_temporary_files(path)


def test_incomplete_existing_reference_is_repaired_atomically(attachment):
    store, context, path, reference = attachment
    path.write_text("{", encoding="utf-8")
    store._publish(context["attachment"]["id"])
    assert json.loads(path.read_text()) == reference
    assert_no_temporary_files(path)


def test_write_failure_removes_temporary_reference(attachment, monkeypatch):
    store, context, path, _reference = attachment
    path.unlink()

    def failed_sync(*args):
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(agent_session.os, "fsync", failed_sync)
    with pytest.raises(OSError, match="disk full"):
        store._publish(context["attachment"]["id"])
    assert not path.exists()
    assert_no_temporary_files(path)


def test_concurrent_instances_and_processes_preserve_one_reference(attachment):
    store, context, path, reference = attachment

    def attach(_):
        return AgentSessions(store.state_dir).attach("codex", "native", context["attachment"]["cwd"])

    with path.open() as reader:
        with ThreadPoolExecutor(3) as workers:
            contexts = list(workers.map(attach, range(9)))
        assert {item["session"]["id"] for item in contexts} == {context["session"]["id"]}
        script = """
import json, sys
from pathlib import Path
from mindie_coordinator.agent_session import AgentSessions
context = AgentSessions(Path(sys.argv[1])).attach('codex', 'native', sys.argv[2])
print(json.dumps(context['session']['id']))
"""
        children = [subprocess.Popen([sys.executable, "-c", script, str(store.state_dir), context["attachment"]["cwd"]],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    for _ in range(3)]
        try:
            for child in children:
                stdout, stderr = child.communicate(timeout=20)
                assert child.returncode == 0, stderr
                assert json.loads(stdout) == context["session"]["id"]
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.communicate()
        assert json.load(reader) == reference
    assert len(store.sessions()) == 1
    assert_no_temporary_files(path)
