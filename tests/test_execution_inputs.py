import json
import subprocess
import threading
import time
from unittest.mock import Mock, patch

import pytest

from mindie_coordinator.agent_session import AgentSessions
from mindie_coordinator.execution_sources import capture_sources, validate_source_snapshot
from mindie_coordinator.placement import normalize_resources, role_plan
from mindie_coordinator.ready_runtime import RuntimePool
from mindie_coordinator.service import CoordinatorService
from mindie_coordinator.task_client import TaskClient


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, encoding="utf-8").strip()


def repo(path, content="A"):
    path.mkdir()
    git(path, "init")
    git(path, "config", "user.name", "Inputs Test")
    git(path, "config", "user.email", "inputs@example.invalid")
    (path / "value.txt").write_text(content, encoding="utf-8")
    git(path, "add", ".")
    git(path, "commit", "-m", "initial")
    return path


@pytest.fixture
def client(tmp_path):
    store = AgentSessions(tmp_path / "sessions")
    context = store.attach("codex", "fixed-inputs", str(tmp_path))
    backend = Mock()
    pool = RuntimePool(tmp_path / "pool", backend)
    service = CoordinatorService(tmp_path / "coordinator", pool=pool, backend=backend, sessions=store)
    service._async_progress = True
    service._schedule_progress = Mock()
    return TaskClient(context["context_file"], service=service, user="alice")


def test_active_a_new_defaults_b_later_edit_c_are_separate_inputs(client, tmp_path):
    first = repo(tmp_path / "first", "A")
    second = repo(tmp_path / "second", "B")
    client.sources({"app": str(first)})
    a = client.run("python app/main.py")
    client.sources({"app": str(second)})
    b = client.run("python app/main.py")
    (second / "value.txt").write_text("C", encoding="utf-8")

    rows = {row["id"]: row for row in client.store.executions(client.context["session"]["id"])}
    inputs_a = rows[a["execution_id"]]["spec"]["source_snapshot"]
    inputs_b = rows[b["execution_id"]]["spec"]["source_snapshot"]
    assert git(first, "show", inputs_a["records"][0]["commit"] + ":value.txt") == "A"
    assert git(second, "show", inputs_b["records"][0]["commit"] + ":value.txt") == "B"
    assert client.status()["session"]["sources"]["app"]["path"] == str(second.resolve())
    assert a["source_snapshot_id"] != b["source_snapshot_id"]
    assert client.run("python app/main.py")["source_snapshot_id"] != b["source_snapshot_id"]


def test_dirty_capture_preserves_head_index_and_ignored_files(tmp_path):
    source = repo(tmp_path / "repo")
    (source / ".gitignore").write_text("ignored.dat\n", encoding="utf-8")
    (source / "ignored.dat").write_text("large-data", encoding="utf-8")
    (source / "value.txt").write_text("dirty", encoding="utf-8")
    original_head, original_index = git(source, "rev-parse", "HEAD"), git(source, "write-tree")
    fixed = capture_sources({"app": str(source)}, tmp_path / "state")
    record = fixed["records"][0]
    assert record["source_head"] == original_head
    assert git(source, "rev-parse", "HEAD") == original_head
    assert git(source, "write-tree") == original_index
    assert "ignored.dat" not in git(source, "ls-tree", "--name-only", record["commit"]).splitlines()
    assert git(source, "show", record["ref"] + ":value.txt") == "dirty"
    assert git(source, "rev-parse", record["ref"] + "-scm") == original_head


def test_capture_retries_an_edit_during_capture_then_pins_the_stable_tree(tmp_path):
    source = repo(tmp_path / "repo")
    from mindie_coordinator.parity import build_synthetic_snapshot
    calls = []

    def changing(*args, **kwargs):
        result = build_synthetic_snapshot(*args, **kwargs)
        calls.append(result)
        if len(calls) == 1:
            (source / "value.txt").write_text("B", encoding="utf-8")
        return result

    with patch("mindie_coordinator.parity.build_synthetic_snapshot", side_effect=changing):
        fixed = capture_sources({"app": str(source)}, tmp_path / "state")
    assert len(calls) == 2
    assert git(source, "show", fixed["records"][0]["commit"] + ":value.txt") == "B"


def test_unstable_capture_is_not_admitted_and_cleans_temporary_refs(client, tmp_path):
    source = repo(tmp_path / "repo")
    from mindie_coordinator.parity import build_synthetic_snapshot
    count = 0

    def changing(*args, **kwargs):
        nonlocal count
        result = build_synthetic_snapshot(*args, **kwargs)
        count += 1
        (source / "value.txt").write_text(str(count), encoding="utf-8")
        return result

    with patch("mindie_coordinator.parity.build_synthetic_snapshot", side_effect=changing):
        with pytest.raises(ValueError, match="execution was not admitted"):
            client.run("true", sources={"app": str(source)})
    assert count == 3
    assert client.store.executions(client.context["session"]["id"]) == []
    assert git(source, "for-each-ref", "refs/parity/execution-inputs") == ""


def test_service_ensure_compares_dirty_source_and_connect_does_not_capture(client, tmp_path):
    source = repo(tmp_path / "repo")
    client.sources({"app": str(source)})
    first = client.run("serve", service="api")
    same = client.run("serve", service="api")
    assert first["execution_id"] == same["execution_id"]
    (source / "value.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="different inputs: sources"):
        client.run("serve", service="api")
    with patch("mindie_coordinator.execution_sources.capture_sources", side_effect=AssertionError("must not capture")):
        assert client.observe(service="api")["execution_id"] == first["execution_id"]


def test_service_replacement_requires_confirmed_resource_release(client):
    first = client.run("serve", service="api", sources={})
    incomplete = {"execution_id": first["execution_id"], "state": "failed", "resources_released": False}
    with patch.object(client.coordinator, "_stop_and_wait", return_value=incomplete):
        assert client.run("serve", service="api", sources={}, restart=True) == incomplete
    assert len(client.store.executions(client.context["session"]["id"])) == 1


def test_source_free_overrides_defaults_without_git_or_devices(client, tmp_path):
    source = repo(tmp_path / "repo")
    client.sources({"app": str(source)})
    with patch("mindie_coordinator.agent_session.worktree_reference", side_effect=AssertionError("must not read Git")):
        reply = client.run("printf ready", sources={})
    row = client.store.executions(client.context["session"]["id"])[0]
    assert reply["sources"] == {}
    assert row["spec"]["resources"] == {"npu_count": 0}
    assert row["spec"]["source_snapshot"]["records"] == []
    client.sources({})
    assert client.status()["session"]["sources"] == {}


def test_role_group_captures_once_and_progress_uses_accepted_paths(client, tmp_path):
    source = repo(tmp_path / "repo")
    with patch("mindie_coordinator.execution_sources.capture_sources", wraps=capture_sources) as capture:
        reply = client.run("true", sources={"app": str(source)}, topology={
            "roles": [{"name": "one", "npu_count": 0}, {"name": "two", "npu_count": 1}]})
    capture.assert_called_once()
    client.sources({})
    with patch.object(client.coordinator, "_place_or_prepare", return_value={"status": "waiting", "reason": "test"}) as prepare:
        client.coordinator.advance(str(client.store.state_dir), "alice", reply["execution_id"], action="progress")
    accepted = prepare.call_args.args[2]
    assert accepted["sources"] == {"app": str(source.resolve())}
    assert accepted["remote_session"]["session_id"] == reply["execution_id"]
    assert [role["npu_count"] for role in prepare.call_args.args[3]] == [0, 1]


def test_snapshot_identity_detects_mutation_but_ignores_transport_refs(tmp_path):
    source = repo(tmp_path / "repo")
    first = capture_sources({"app": str(source)}, tmp_path / "state")
    second = capture_sources({"app": str(source)}, tmp_path / "state")
    assert first["id"] == second["id"]
    second["records"][0]["commit"] = "f" * 40
    with pytest.raises(ValueError, match="does not match"):
        validate_source_snapshot(second)


@pytest.mark.parametrize("count", [-1, 0.5, "1", True])
def test_invalid_device_count_is_rejected_without_coercion(count):
    with pytest.raises(ValueError, match="nonnegative integer"):
        normalize_resources({"npu_count": count})


def test_zero_device_role_is_not_promoted_to_one():
    assert normalize_resources(None) == {"npu_count": 0}
    assert role_plan({"roles": [{"name": "cpu", "npu_count": 0}]}, {"npu_count": 2}, "true")[0]["npu_count"] == 0


@pytest.mark.parametrize("devices", [[], [1, 3]])
def test_consistent_device_count_is_accepted_and_canonicalized(client, devices):
    from jsonschema import validate
    from mindie_coordinator.ops import TOOL_SCHEMAS

    resources = {"devices": devices, "npu_count": len(devices)}
    validate({"command": "true", "sources": {}, "resources": resources}, TOOL_SCHEMAS["mindie.run"])
    assert normalize_resources(resources) == {"devices": devices}
    assert role_plan({"roles": [{"name": "one", **resources}]}, {}, "true")[0] == {
        "name": "one", "command": "true", "devices": devices}
    reply = client.run("true", sources={}, resources=resources)
    row = client.store.executions(client.context["session"]["id"])[0]
    assert row["id"] == reply["execution_id"]
    assert row["spec"]["resources"] == {"devices": devices}


def test_conflicting_device_count_is_rejected_before_capture(client):
    with patch("mindie_coordinator.execution_sources.capture_sources", side_effect=AssertionError("invalid request must not capture")):
        for request in ({"resources": {"devices": [], "npu_count": 1}},
                        {"topology": {"roles": [{"name": "one", "devices": [0], "npu_count": 2}]}}):
            with pytest.raises(ValueError, match="must equal the number of devices"):
                client.run("true", sources={}, **request)
    assert client.store.executions(client.context["session"]["id"]) == []


@pytest.mark.parametrize("arguments", [
    {"topology": {"hosts": ["192.0.2.2"]}},
    {"topology": {"host": "192.0.2.2", "roles": [{"name": "one"}]}},
    {"topology": {"roles": [{"name": "one", "hostname": "192.0.2.2"}]}},
    {"resources": {"npu_counts": 2}},
    {"environment": {"recpie": "rc"}},
])
def test_unknown_constraints_are_rejected_before_capture_or_admission(client, arguments):
    with patch("mindie_coordinator.execution_sources.capture_sources", side_effect=AssertionError("invalid request must not capture")):
        with pytest.raises(ValueError, match="unsupported|do not combine"):
            client.run("true", sources={}, **arguments)
    assert client.store.executions(client.context["session"]["id"]) == []


def test_explicit_devices_are_preserved_for_roles_without_an_override():
    roles = role_plan({"roles": [{"name": "one"}, {"name": "two", "npu_count": 1}]}, {"devices": [3]}, "true")
    assert roles[0]["devices"] == [3]
    assert "npu_count" not in roles[0]
    assert roles[1]["npu_count"] == 1
    assert "devices" not in roles[1]


def test_four_allowed_hosts_prepare_concurrently_and_preserve_role_progress(client):
    roles = [{"name": f"role-{index}", "host": f"host-{index}", "command": "true", "npu_count": 0}
             for index in range(4)]
    client.run("true", sources={}, topology={"roles": roles, "distinct_hosts": True})
    row = client.store.executions(client.context["session"]["id"])[0]
    catalog = [{"runtime_id": role["name"], "host": role["host"], "user": "alice"} for role in roles]
    barrier = threading.Barrier(4)

    def prepare(store, user, execution, role, environment, donor):
        barrier.wait(timeout=5)
        client.coordinator._save_progress(store, execution, role["name"], {"step": "built"})
        return {"id": role["name"]}

    with patch.object(client.pool, "catalog", return_value=catalog), \
         patch.object(client.coordinator, "_configured_machines", return_value=[]), \
         patch.object(client.coordinator, "_donor_for_role", side_effect=lambda user, env, role, *args: {"host": role["host"]}), \
         patch.object(client.coordinator, "_prepare_role", side_effect=prepare):
        result = client.coordinator._place_or_prepare(client.store, "alice", row, roles, {})
    assert result["runtime_ids"] == [role["name"] for role in roles]
    persisted = client.store.executions(client.context["session"]["id"])[0]
    assert set(persisted["role_progress"]) == {role["name"] for role in roles}


@pytest.mark.parametrize("phase", ["private", "compile"])
def test_cancelled_group_stops_before_launch_and_skips_waiting_compiler(client, phase):
    roles = [{"name": name, "host": "same-host", "command": "true", "npu_count": 0} for name in ("one", "two")]
    client.run("true", sources={}, topology={"roles": roles})
    row = client.store.executions(client.context["session"]["id"])[0]
    catalog = [{"runtime_id": role["name"], "host": "same-host", "user": "alice"} for role in roles]
    prepared = []
    barrier = threading.Barrier(2)

    def prepare_environment(*args, **kwargs):
        # Both private preparations are in flight before cancellation. Only
        # the actual compile_scope serializes same-host compiler work.
        barrier.wait(timeout=5)

        def complete_then_cancel():
            prepared.append(kwargs["role_name"])
            stored = client.store.executions(row["session_id"])[0]
            stored["cancel_requested"] = True
            client.store.save_execution(stored)
            return {"id": kwargs["role_name"]}

        if phase == "compile":
            with kwargs["compile_scope"]("install-vllm-ascend"):
                return complete_then_cancel()
        return complete_then_cancel()

    with patch.object(client.pool, "catalog", return_value=catalog), \
         patch.object(client.coordinator, "_donor_for_role", return_value={"host": "same-host"}), \
         patch("mindie_coordinator.provision.prepare_task_environment", side_effect=prepare_environment), \
         patch.object(client.pool, "checkout", side_effect=AssertionError("cancelled preparation must not checkout")) as checkout, \
         patch.object(client.pool, "managed_start", side_effect=AssertionError("cancelled preparation must not launch")) as launch:
        result = client.coordinator._progress(client.store, "alice", row)
    assert result["state"] == "cancelled" and result["resources_released"] is True
    assert len(prepared) == (1 if phase == "compile" else 2)
    checkout.assert_not_called()
    launch.assert_not_called()
