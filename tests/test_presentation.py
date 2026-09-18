import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from remote_dev.result import make_result
from mindie_coordinator.presentation import compact_runtime, present


def test_large_history_and_duplicate_envs_are_bounded_and_readable(tmp_path):
    value = {"session": {"id": "task-1", "state": "open"}, "executions": [
        {"id": str(i), "phase": "failed", "error": "ERROR " * 2000,
         "binding": {"launch_env": {"PATH": "a" * 5000}}, "roles": []} for i in range(100)]}
    result = make_result(tool="mindie.session", target={}, outcome="success", status="open", summary="MindIE open", extra={"data": value})
    compact = present(result, tmp_path)
    assert len(json.dumps(compact).encode()) < 16000
    assert compact["data"]["executions_total"] == 100
    assert len(compact["data"]["executions"]) == 5
    assert json.loads(Path(compact["record_ref"]).read_text(encoding="utf-8"))["data"] == value
    assert "binding" not in json.dumps(compact)


def test_explicit_target_is_exact_or_omitted_never_silently_truncated(tmp_path):
    target = {"launch_preamble": "x" * 3000}
    result = make_result(tool="mindie.execution", target={}, outcome="success", status="running", summary="MindIE running", extra={"data": {"execution_id": "e", "state": "running", "target": target}})
    compact = present(result, tmp_path, target=True)
    assert compact["data"]["target"] == target


def test_full_mode_and_write_failure_preserve_operation_result(tmp_path):
    value = {"state": "running", "execution_id": "e", "private_field": "full" * 10000}
    result = make_result(tool="mindie.run", target={}, outcome="success", status="running", summary="MindIE running", extra={"data": value})
    assert present(result, tmp_path, full=True)["data"] == value
    with patch.object(Path, "open", side_effect=OSError("disk full")):
        reply = present(result, tmp_path)
    assert reply["outcome"] == "success"
    assert reply["data"] == {"state": "running", "execution_id": "e"}
    assert len(json.dumps(reply).encode()) < 16000
    assert "record_ref" not in reply
    assert "disk full" in reply["warnings"][-1]
    with patch.object(Path, "open", side_effect=OSError("disk full")):
        assert present(result, tmp_path, full=True)["data"] == value


def test_preparation_tail_keeps_bounded_log_text_and_exact_record(tmp_path):
    log = "compiler output\n" * 1000 + "CompileError: affected operator failed"
    value = {"execution_id": "e", "state": "preparing", "resources_released": False,
             "progress": {"step": "install-native", "log_ref": "/logs/install.log"},
             "preparation_logs": [{"name": "default", "step": "install-native", "tail": log}],
             "tail": log}
    result = make_result(tool="mindie.execution", target={}, outcome="success", status="preparing",
                         summary="MindIE preparing", extra={"data": value})
    compact = present(result, tmp_path)
    assert compact["data"]["tail"].endswith("CompileError: affected operator failed")
    assert compact["data"]["preparation_logs"][0]["tail"].endswith("CompileError: affected operator failed")
    assert compact["data"]["resources_released"] is False
    assert len(json.dumps(compact).encode()) < 16000
    assert json.loads(Path(compact["record_ref"]).read_text(encoding="utf-8"))["data"] == value


@pytest.mark.parametrize("state,released,quiet", [("succeeded", True, True), ("failed", False, False)])
def test_large_role_logs_preserve_completion_failure_and_release_facts(tmp_path, state, released, quiet):
    log = "中文日志 " * 4000 + "CompileError: native build failed"
    value = {"execution_id": "e", "state": state, "resources_released": released,
             "error": log, "error_ref": "/logs/error.txt", "reason": "native build",
             "observation_freshness": {"fresh": False, "age_seconds": 3.5},
             "progress": {"step": "install-native", "log_ref": "/logs/install.log"},
             "roles": [{"name": str(i), "state": state, "quiet": quiet,
                        "lease_state": "released" if released else "held",
                        "error": log, "stdout": log, "stderr": log} for i in range(8)]}
    result = make_result(tool="mindie.execution", target={}, outcome="failed" if state == "failed" else "success",
                         status=state, summary="MindIE " + state, extra={"data": value})
    compact = present(result, tmp_path)
    data = compact["data"]
    assert len(json.dumps(compact, ensure_ascii=False).encode()) < 16000
    assert data["state"] == state and data["resources_released"] is released
    assert data["observation_freshness"] == value["observation_freshness"]
    assert data["progress"] == value["progress"]
    assert data["error_ref"] == value["error_ref"]
    assert data["error"].endswith("CompileError: native build failed")
    assert len(data["roles"]) == 8
    for role in data["roles"]:
        assert role["state"] == state and role["quiet"] is quiet
        assert role["lease_state"] == ("released" if released else "held")
        assert role["stderr"].endswith("CompileError: native build failed")
    assert json.loads(Path(compact["record_ref"]).read_text(encoding="utf-8"))["data"] == value


def test_oversized_target_is_omitted_but_cleanup_and_error_references_remain_exact(tmp_path):
    reference = "/logs/" + "long-directory/" * 20 + "failure.log"
    value = {"execution_id": "e", "state": "failed", "resources_released": False,
             "error": "CompileError: operator failed", "error_ref": reference,
             "progress": {"step": "install-native", "log_ref": reference},
             "target": {"launch_preamble": "x" * 20000},
             "roles": [{"name": "default", "state": "failed", "quiet": False, "lease_state": "held"}]}
    result = make_result(tool="mindie.execution", target={}, outcome="failed", status="failed",
                         summary="MindIE failed", extra={"data": value})
    compact = present(result, tmp_path, target=True)
    assert "target" not in compact["data"] and compact["detail_omitted"]
    assert compact["data"]["resources_released"] is False
    assert compact["data"]["error"] == value["error"]
    assert compact["data"]["error_ref"] == reference
    assert compact["data"]["progress"] == value["progress"]
    assert compact["data"]["roles"] == value["roles"]
    assert len(json.dumps(compact).encode()) < 16000
    assert json.loads(Path(compact["record_ref"]).read_text(encoding="utf-8"))["data"] == value


def test_role_limit_prioritizes_failure_and_unreleased_work_without_changing_facts(tmp_path):
    roles = [{"name": str(i), "state": "succeeded", "quiet": True, "lease_state": "released"} for i in range(8)]
    roles.extend([{"name": "failed", "state": "failed", "quiet": True, "lease_state": "released", "error": "bad input"},
                  {"name": "draining", "state": "succeeded", "quiet": False, "lease_state": "held"}])
    value = {"execution_id": "e", "state": "failed", "resources_released": False, "roles": roles}
    result = make_result(tool="mindie.execution", target={}, outcome="failed", status="failed",
                         summary="MindIE failed", extra={"data": value})
    compact = present(result, tmp_path)
    data = compact["data"]
    assert data["roles_total"] == 10 and len(data["roles"]) == 8
    for name in ("failed", "draining"):
        assert next(role for role in data["roles"] if role["name"] == name) == next(role for role in roles if role["name"] == name)
    assert data["state"] == "failed" and data["resources_released"] is False


def test_healthy_runtime_is_small_but_full_and_unhealthy_facts_remain_available(tmp_path):
    packages = {"mindie-coordinator": {"version": "0.4.0", "commit": "a" * 40},
                "remote-dev": {"version": "0.7.0", "commit": None}}
    runtime = {scope: [{"status": "current", "loaded": {"package": name, **identity,
                        "location": "/workspace/" * 40, "python": "/workspace/python", "pid": index},
                       "installed": {"package": name, **identity, "location": "/workspace/" * 40}}
                      for index, (name, identity) in enumerate(packages.items())]
               for scope in ("client", "daemon")}

    def result(facts):
        return make_result(tool="mindie.execution", target={}, outcome="success", status="preparing", summary="MindIE preparing",
                           extra={"runtime": facts, "data": {"execution_id": "e", "state": "preparing", "target": {"launch_preamble": "exact"}}})

    compact = present(result(runtime), tmp_path)
    assert compact["runtime"] == {"status": "current", "packages": packages, "python": "/workspace/python"}
    assert len(json.dumps(compact["runtime"])) < len(json.dumps(runtime)) / 10
    assert json.loads(Path(compact["record_ref"]).read_text())["runtime"] == runtime
    assert present(result(runtime), tmp_path, full=True)["runtime"] == runtime
    assert present(result(runtime), tmp_path, target=True)["runtime"] == runtime
    unhealthy = deepcopy(runtime)
    unhealthy["daemon"][0].update(status="restart_required", error="daemon still has the old package loaded")
    preserved = present(result(unhealthy), tmp_path)
    assert preserved["runtime"] == unhealthy


@pytest.mark.parametrize("changed,value", [("commit", "different-revision"), ("python", "/other/python"),
                                         ("location", "/other/site-packages")])
def test_same_version_in_different_runtimes_keeps_scope_evidence(changed, value):
    loaded = {"package": "mindie-coordinator", "version": "0.4.1.dev1", "commit": "revision-a",
              "python": "/env-a/python", "location": "/env-a/site-packages"}
    other = {**loaded, changed: value}
    runtime = {"client": [{"status": "current", "loaded": loaded, "installed": loaded}],
               "daemon": [{"status": "current", "loaded": other, "installed": other}]}
    assert compact_runtime(runtime) == runtime
