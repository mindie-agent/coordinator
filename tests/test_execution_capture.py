import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from mindie_coordinator.execution_sources import capture_sources
from mindie_coordinator import execution_capture, parity
from test_execution_inputs import git, repo


@pytest.fixture
def source(tmp_path):
    return repo(tmp_path / "repo")


def capture(source, tmp_path, name="app"):
    return capture_sources({name: str(source)}, tmp_path / "state")


def tree(source, snapshot):
    return git(source, "ls-tree", "-r", snapshot["records"][-1]["commit"])


def test_warm_capture_reuses_retained_objects_without_rebuilding(source, tmp_path):
    (source / "value.txt").write_text("dirty")
    first = capture(source, tmp_path)
    with patch.object(parity, "build_synthetic_snapshot", side_effect=AssertionError("unchanged tree")), \
         patch.object(execution_capture, "incremental_snapshot", side_effect=AssertionError("unchanged tree")):
        assert capture(source, tmp_path)["id"] == first["id"]


@pytest.mark.parametrize("change", ["delete-untracked", "rename-untracked", "rename-tracked", "dirty-to-clean", "second-dirty-edit"])
def test_incremental_paths_match_a_fresh_complete_capture(source, tmp_path, change):
    path = source / "-extra file.txt" if "untracked" in change else source / "value.txt"
    path.write_text("initial dirty")
    previous = capture(source, tmp_path)
    head, index = git(source, "rev-parse", "HEAD"), git(source, "write-tree")
    if change.startswith("delete"):
        path.unlink()
    elif change.startswith("rename"):
        path.rename(source / "renamed file.txt")
    elif change == "dirty-to-clean":
        git(source, "restore", "--worktree", "value.txt")
    else:
        path.write_text("another dirty")
    current = capture(source, tmp_path)
    complete = capture_sources({"app": str(source)}, tmp_path / "fresh-state")
    assert current["id"] != previous["id"]
    assert current["id"] == complete["id"]
    assert tree(source, current) == tree(source, complete)
    assert (git(source, "rev-parse", "HEAD"), git(source, "write-tree")) == (head, index)


def test_new_ignore_rule_drops_previously_captured_untracked_file(source, tmp_path):
    (source / "extra.dat").write_text("local")
    capture(source, tmp_path)
    ignore = source / ".git/info/exclude"
    ignore.write_text("extra.dat\n")
    current = capture(source, tmp_path)
    assert "extra.dat" not in tree(source, current)
    assert current["id"] == capture_sources({"app": str(source)}, tmp_path / "fresh")["id"]


@pytest.mark.parametrize("attributes", ["local", "external"])
def test_attribute_file_changes_invalidate_capture(source, tmp_path, attributes):
    (source / "line.txt").write_bytes(b"line\r\n")
    if attributes == "local":
        attribute = source / ".git/info/attributes"
    else:
        attribute = tmp_path / "attributes"
        git(source, "config", "core.attributesFile", str(attribute))
    attribute.write_text("line.txt -text\n")
    first = capture(source, tmp_path)
    attribute.write_text("line.txt text eol=lf\n")
    current = capture(source, tmp_path)
    assert current["id"] != first["id"]
    assert current["id"] == capture_sources({"app": str(source)}, tmp_path / "fresh")["id"]


def test_python_only_change_reuses_native_fingerprint_and_native_change_recomputes(source, tmp_path):
    (source / "module.py").write_text("x=1")
    (source / "native.cpp").write_text("int x=1;")
    first = capture(source, tmp_path, "vllm")
    (source / "module.py").write_text("x=2")
    with patch.object(parity, "build_input_fingerprints", side_effect=AssertionError("native files unchanged")):
        python_change = capture(source, tmp_path, "vllm")
    assert first["records"][0]["build_inputs"] == python_change["records"][0]["build_inputs"]
    (source / "native.cpp").write_text("int x=2;")
    native_change = capture(source, tmp_path, "vllm")
    assert native_change["records"][0]["build_inputs"]["native"] != first["records"][0]["build_inputs"]["native"]


def test_build_environment_is_fixed_and_invalidates_reuse(source, tmp_path, monkeypatch):
    monkeypatch.setenv("CFLAGS", "-DA")
    first = capture(source, tmp_path, "vllm")
    monkeypatch.setenv("CFLAGS", "-DB")
    second = capture(source, tmp_path, "vllm")
    assert first["id"] != second["id"]
    assert first["records"][0]["build_inputs"]["native"] == second["records"][0]["build_inputs"]["native"]
    assert second["records"][0]["build_inputs"]["build_env"] == hashlib.sha256(
        json.dumps(second["build_env"], sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("damage", ["cache", "snapshot-ref", "scm-ref"])
def test_missing_retention_or_corrupt_cache_falls_back(source, tmp_path, damage):
    first = capture(source, tmp_path)
    if damage == "cache":
        next((tmp_path / "state/source-inputs").glob("capture-*.json")).write_text("broken")
    else:
        git(source, "update-ref", "-d", first["records"][0]["ref"] + ("-scm" if damage == "scm-ref" else ""))
    with patch.object(parity, "build_synthetic_snapshot", wraps=parity.build_synthetic_snapshot) as build:
        assert capture(source, tmp_path)["id"] == first["id"]
    assert build.call_count == 1


def test_incremental_edit_during_capture_retries_and_keeps_the_last_stable_tree(source, tmp_path):
    (source / "value.txt").write_text("B")
    capture(source, tmp_path)
    (source / "value.txt").write_text("C")
    original = execution_capture.incremental_snapshot
    attempts = []

    def changing(*args):
        result = original(*args)
        attempts.append(result)
        if len(attempts) == 1:
            (source / "value.txt").write_text("D")
        return result

    with patch.object(execution_capture, "incremental_snapshot", side_effect=changing):
        current = capture(source, tmp_path)
    assert len(attempts) == 2
    assert git(source, "show", current["records"][0]["commit"] + ":value.txt") == "D"
    assert git(source, "for-each-ref", "refs/parity/execution-inputs") == ""


def test_scm_shallow_boundary_and_dirty_state_invalidate_version_cache(source, tmp_path):
    (source / "setup.py").write_text("# project")
    with patch("setuptools_scm.get_version", side_effect=["0.1", "0.2", "0.3"]) as scm:
        capture(source, tmp_path, "vllm")
        assert capture(source, tmp_path, "vllm")["records"][0]["scm_version"] == "0.1"
        (source / ".git/shallow").write_text(git(source, "rev-parse", "HEAD") + "\n")
        assert capture(source, tmp_path, "vllm")["records"][0]["scm_version"] == "0.2"
        git(source, "add", "setup.py")
        git(source, "commit", "-m", "project")
        assert capture(source, tmp_path, "vllm")["records"][0]["scm_version"] == "0.3"
    assert scm.call_count == 3


@pytest.mark.parametrize("rule", ["denylist", "native-pattern", "format"])
def test_capture_rule_changes_invalidate_persisted_cache(source, tmp_path, monkeypatch, rule):
    (source / "extra.txt").write_text("extra")
    first = capture(source, tmp_path, "vllm")
    if rule == "denylist":
        monkeypatch.setattr(parity, "DEFAULT_DENYLIST", (*parity.DEFAULT_DENYLIST, "extra.txt"))
    elif rule == "native-pattern":
        monkeypatch.setattr(parity, "VLLM_REINSTALL_PATTERNS", (*parity.VLLM_REINSTALL_PATTERNS, "extra.txt"))
    else:
        monkeypatch.setattr(execution_capture, "CACHE_FORMAT", execution_capture.CACHE_FORMAT + 1)
    with patch.object(parity, "build_synthetic_snapshot", wraps=parity.build_synthetic_snapshot) as build:
        current = capture(source, tmp_path, "vllm")
    assert build.call_count == 1
    if rule == "denylist":
        assert "extra.txt" not in tree(source, current)
    elif rule == "native-pattern":
        assert current["records"][0]["build_inputs"]["native"] != first["records"][0]["build_inputs"]["native"]


@pytest.mark.parametrize("filename", ["算子.cpp", pytest.param("line\nbreak.cpp", marks=pytest.mark.skipif(os.name == "nt", reason="Windows disallows newline filenames"))])
def test_native_fingerprint_uses_unquoted_git_paths(source, tmp_path, filename):
    path = source / filename
    path.write_text("int x=1;", encoding="utf-8")
    first = capture(source, tmp_path, "vllm")
    path.write_text("int x=2;", encoding="utf-8")
    current = capture(source, tmp_path, "vllm")
    assert current["records"][0]["build_inputs"]["native"] != first["records"][0]["build_inputs"]["native"]
    assert filename in current["records"][0]["changed_paths"]


@pytest.mark.parametrize("content", ["null", "[]", "{}"])
def test_incomplete_cache_is_a_miss(source, tmp_path, content):
    first = capture(source, tmp_path)
    next((tmp_path / "state/source-inputs").glob("capture-*.json")).write_text(content)
    assert capture(source, tmp_path)["id"] == first["id"]


def test_cache_write_failure_does_not_block_fixed_inputs(source, tmp_path):
    with patch.object(execution_capture, "save_cache", side_effect=PermissionError("cache busy")):
        snapshot = capture(source, tmp_path)
    assert git(source, "show", snapshot["records"][0]["ref"] + ":value.txt") == "A"
    assert (tmp_path / "state/source-inputs" / (snapshot["id"] + ".json")).exists()


def test_topology_edit_during_capture_rediscovers_before_retry(source, tmp_path):
    original = parity.build_synthetic_snapshot
    attempts = []

    def changing(*args, **kwargs):
        record = original(*args, **kwargs)
        attempts.append(record)
        if len(attempts) == 1:
            (source / ".gitmodules").write_text("# topology changed\n")
        return record

    with patch.object(parity, "build_synthetic_snapshot", side_effect=changing):
        snapshot = capture(source, tmp_path)
    assert len(attempts) == 2
    assert git(source, "show", snapshot["records"][0]["commit"] + ":.gitmodules") == "# topology changed"
