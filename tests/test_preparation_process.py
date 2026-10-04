"""Long preparation owns a durable process, including stop after daemon loss."""
import copy
import json
import sys
import threading
import time
from unittest.mock import MagicMock, Mock

import pytest

from mindie_coordinator.agent_session import AgentSessions
from mindie_coordinator.parity_support import PROGRESS_SENTINEL, RemoteCommandError, SshEndpoint, ssh_exec_stream
from mindie_coordinator.preparation_process import (
    PreparationCancelled, PreparationProcess, PreparationUncertain, stop_preparation_process,
)
from mindie_coordinator.service import CoordinatorService


ENDPOINT = {"host": "example.invalid", "port": 22, "user": "user", "root": "/tmp/task", "cwd": "/tmp/task"}


def test_receipt_saved_before_launch_and_progress_survives_split_chunks(monkeypatch, tmp_path):
    saved, actions = [], []
    chunks = iter([
        {"state": "running", "quiet": False, "stdout": "out\n", "stderr": PROGRESS_SENTINEL + '{"step": "bui',
         "receipt": {"pid": 123, "boot_id": "test"},
         "stdout_offset": 4, "stderr_offset": 30},
        {"state": "succeeded", "quiet": True, "stderr": 'ld"}\nwarn', "stderr_offset": 40,
         "result": {"exit_code": 0}},
    ])
    def control(endpoint, job_id, action, **kwargs):
        assert saved[0]["job_id"] == job_id
        actions.append(action)
        if action == "launch":
            assert saved[-1]["state"] == "pending"
            assert kwargs["spec"] == {"command": "build", "cwd": ENDPOINT["cwd"], "env": {},
                                      "timeout_seconds": None, "interactive": False}
            assert kwargs["authorization"] == {}
            assert kwargs["stdout_offset"] == kwargs["stderr_offset"] == 0
            assert kwargs["max_bytes"] == 32768 and kwargs["yield_time_ms"] == 1000
            assert kwargs["wait_for_exit"] is True
        else:
            assert action == "exchange"
            assert saved[-1]["receipt"]["pid"] == 123
            assert kwargs["stdout_offset"] == 4 and kwargs["stderr_offset"] == 30
            assert kwargs["wait_for_exit"] is True and kwargs["yield_time_ms"] == 1000
        return next(chunks)
    monkeypatch.setattr("mindie_coordinator.preparation_process.control", control)
    process = PreparationProcess(ENDPOINT, "build", lambda row: saved.append(copy.deepcopy(row)), lambda: False)
    events = []
    result = ssh_exec_stream(SshEndpoint("example.invalid", 22, "user"), "build", stream_progress=False,
                             on_progress=events.append, log_path=tmp_path / "build.log", process=process)
    assert result.stdout == "out\n" and result.stderr == "warn"
    assert events == [{"step": "build"}]
    assert saved[-1]["quiet"] is True and "stdout" not in saved[-1]
    assert actions == ["launch", "exchange"]


@pytest.mark.parametrize("exit_code", [0, 17])
def test_short_preparation_returns_initial_output_and_exit_receipt_without_another_rpc(monkeypatch, exit_code):
    saved, output = [], []
    control = Mock(return_value={"state": "succeeded" if exit_code == 0 else "failed", "quiet": True,
                                 "stdout": "complete", "stderr": "warning",
                                 "stdout_offset": 8, "stderr_offset": 7,
                                 "result": {"exit_code": exit_code}, "timings": {"prepare_ms": 10}})
    monkeypatch.setattr("mindie_coordinator.preparation_process.control", control)
    result = PreparationProcess(ENDPOINT, "build", lambda row: saved.append(copy.deepcopy(row)), lambda: False).run(
        "build", on_output=lambda *args: output.append(args))
    assert result.returncode == exit_code
    assert output == [("stdout", "complete"), ("stderr", "warning")]
    assert control.call_count == 1 and control.call_args.args[2] == "launch"
    assert saved[-1]["quiet"] is True and saved[-1]["stdout_offset"] == 8
    assert not ({"stdout", "stderr"} & saved[-1].keys())
    assert saved[-1]["timings"] == {"prepare_ms": 10}


@pytest.mark.parametrize('initial_quiet', [False, True])
@pytest.mark.parametrize('newline', [False, True])
def test_stage_timing_is_saved_from_final_output_before_callback(monkeypatch, initial_quiet, newline):
    progress = PROGRESS_SENTINEL + json.dumps({'phase': 'prepare-root', 'elapsed_seconds': .125})
    text = progress + ('\n' if newline else '')
    complete = {'state': 'succeeded', 'quiet': True, 'result': {'exit_code': 0}}
    chunks = ([dict(complete, stderr=text)] if initial_quiet else [
        {'state': 'running', 'quiet': False, 'stderr': text[:20], 'stderr_offset': 20},
        dict(complete, stderr=text[20:], stderr_offset=len(text)),
    ])
    remote = Mock(side_effect=chunks)
    monkeypatch.setattr('mindie_coordinator.preparation_process.control', remote)
    saved, output = [], []
    def on_output(channel, line):
        assert saved[-1]['quiet'] and saved[-1]['result']['exit_code'] == 0
        assert saved[-1]['stage_timings'] == {'prepare-root': .125}
        output.append((channel, line))
    process = PreparationProcess(ENDPOINT, 'build', lambda row: saved.append(copy.deepcopy(row)), lambda: False,
                                 setup=[('prepare-root', 'true')], bootstrap_root='/tmp')
    assert process.run('build', on_output=on_output).returncode == 0
    assert output == [('stderr', text)]
    assert [call.args[2] for call in remote.call_args_list] == (['launch'] if initial_quiet else ['launch', 'exchange'])
    assert not ({'stdout', 'stderr'} & saved[-1].keys())


def test_stage_timing_only_saves_new_or_changed_values(monkeypatch):
    lines = [PROGRESS_SENTINEL + json.dumps({'phase': 'prepare-root', 'elapsed_seconds': value}) + '\n'
             for value in (.125, .125, .25)]
    remote = Mock(return_value={'state': 'succeeded', 'quiet': True, 'result': {'exit_code': 0},
                                'stderr': ''.join(lines)})
    monkeypatch.setattr('mindie_coordinator.preparation_process.control', remote)
    saved = []
    process = PreparationProcess(ENDPOINT, 'build', lambda row: saved.append(copy.deepcopy(row)), lambda: False,
                                 setup=[('prepare-root', 'true')], bootstrap_root='/tmp')
    process.run('build', on_output=lambda *args: None)
    assert [row['stage_timings'] for row in saved if 'stage_timings' in row] == [
        {'prepare-root': .125}, {'prepare-root': .25}]
    assert remote.call_count == 1


@pytest.mark.parametrize('stage', ['persistence', 'output'])
@pytest.mark.parametrize('exit_code', [0, 17])
def test_stage_timing_postprocess_failure_preserves_completion(monkeypatch, stage, exit_code):
    from mindie_coordinator.preparation_process import PreparationPostprocessError
    remote = Mock(return_value={'state': 'succeeded' if exit_code == 0 else 'failed', 'quiet': True,
                                'result': {'exit_code': exit_code}, 'stderr': PROGRESS_SENTINEL +
                                json.dumps({'phase': 'prepare-root', 'elapsed_seconds': .125}) + '\n'})
    monkeypatch.setattr('mindie_coordinator.preparation_process.control', remote)
    saved = []
    persist_error, output_error = OSError('timing receipt unavailable'), RuntimeError('output callback unavailable')
    def save(row):
        if stage == 'persistence' and 'stage_timings' in row:
            raise persist_error
        saved.append(copy.deepcopy(row))
    output = Mock(side_effect=output_error)
    process = PreparationProcess(ENDPOINT, 'build', save, lambda: False,
                                 setup=[('prepare-root', 'true')], bootstrap_root='/tmp')
    with pytest.raises(PreparationPostprocessError) as caught:
        process.run('build', on_output=output)
    failure = caught.value
    assert failure.stage == stage
    assert failure.original_error is (persist_error if stage == 'persistence' else output_error)
    assert failure.completed.returncode == exit_code and failure.automatic_retry is False
    assert failure.preparation_record['stage_timings'] == {'prepare-root': .125}
    assert saved[-1]['quiet'] and saved[-1]['result']['exit_code'] == exit_code
    if stage == 'persistence':
        assert 'stage_timings' not in saved[-1]
        output.assert_not_called()  # The second configured failure cannot hide the first.
    else:
        assert saved[-1]['stage_timings'] == {'prepare-root': .125}
        output.assert_called_once()
    assert remote.call_count == 1


def test_cancel_drain_saves_split_stage_timing_without_final_newline(monkeypatch):
    progress = PROGRESS_SENTINEL + json.dumps({'phase': 'prepare-root', 'elapsed_seconds': .125})
    saved, actions, cancelled = [], [], [False]
    def control(endpoint, job_id, action, **kwargs):
        actions.append(action)
        if action == 'launch':
            cancelled[0] = True
            return {'state': 'running', 'quiet': False, 'stderr': progress[:20], 'stderr_offset': 20}
        if action == 'stop':
            return {'state': 'cancelled', 'quiet': True, 'result': {'exit_code': -15}}
        assert action == 'exchange' and kwargs['stderr_offset'] == 20
        return {'state': 'cancelled', 'quiet': True, 'stderr': progress[20:], 'stderr_offset': len(progress)}
    monkeypatch.setattr('mindie_coordinator.preparation_process.control', control)
    process = PreparationProcess(ENDPOINT, 'build', lambda row: saved.append(copy.deepcopy(row)), lambda: cancelled[0],
                                 setup=[('prepare-root', 'true')], bootstrap_root='/tmp')
    with pytest.raises(PreparationCancelled):
        process.run('build', on_output=lambda *args: None)
    assert saved[-1]['stage_timings'] == {'prepare-root': .125}
    assert saved[-1]['quiet'] and saved[-1]['result']['exit_code'] == -15
    assert actions == ['launch', 'stop', 'exchange']


def test_quiet_launch_drains_remaining_output_from_initial_offsets(monkeypatch):
    saved, output = [], []
    def control(endpoint, job_id, action, **kwargs):
        assert saved[0]["job_id"] == job_id
        if action == "launch":
            return {"state": "succeeded", "quiet": True, "stdout": "first",
                    "stdout_offset": 5, "stdout_bytes_remaining": 5, "result": {"exit_code": 0}}
        assert action == "exchange" and kwargs["stdout_offset"] == 5
        return {"state": "succeeded", "quiet": True, "stdout": "last\n",
                "stdout_offset": 10, "stdout_bytes_remaining": 0, "result": {"exit_code": 0}}
    monkeypatch.setattr("mindie_coordinator.preparation_process.control", control)
    result = PreparationProcess(ENDPOINT, "build", lambda row: saved.append(copy.deepcopy(row)), lambda: False).run(
        "build", on_output=lambda *args: output.append(args))
    assert result.returncode == 0 and output == [("stdout", "firstlast\n")]
    assert saved[-1]["stdout_offset"] == 10


@pytest.mark.parametrize("observation", [
    {"state": "absent", "quiet": True}, {"state": "uncertain", "quiet": False},
    {"state": "lost_outcome", "quiet": True}, {"state": "running", "quiet": False, "unknown": ["owner missing"]},
])
def test_initial_unknown_launch_observation_is_retained_without_relaunch(monkeypatch, observation):
    saved = []
    control = Mock(return_value=observation)
    monkeypatch.setattr("mindie_coordinator.preparation_process.control", control)
    with pytest.raises(PreparationUncertain, match="outcome is unknown"):
        PreparationProcess(ENDPOINT, "build", lambda row: saved.append(copy.deepcopy(row)), lambda: False).run(
            "build", on_output=lambda *args: None)
    assert control.call_count == 1 and control.call_args.args[2] == "launch"
    assert saved[-1]["job_id"] == saved[0]["job_id"] and saved[-1]["state"] == observation["state"]


def test_lost_launch_reply_retains_job_and_can_stop_without_an_initial_receipt(monkeypatch):
    saved, actions = [], []
    def control(endpoint, job_id, action, **kwargs):
        assert saved[0]["job_id"] == job_id
        actions.append(action)
        if action == "launch":
            raise OSError("lost launch reply")
        assert action == "stop"
        return {"state": "cancelled", "quiet": True, "receipt": {"pid": 456}}
    monkeypatch.setattr("mindie_coordinator.preparation_process.control", control)
    with pytest.raises(PreparationUncertain, match="was not replayed"):
        PreparationProcess(ENDPOINT, "build", lambda row: saved.append(copy.deepcopy(row)), lambda: False).run("build", on_output=lambda *a: None)
    assert actions == ["launch"]
    assert saved[-1]["job_id"] == saved[0]["job_id"]
    assert "receipt" not in saved[-1] and saved[-1]["quiet"] is False
    assert stop_preparation_process(saved[-1], lambda row: saved.append(copy.deepcopy(row)))
    assert actions == ["launch", "stop"]
    assert saved[-1]["receipt"]["pid"] == 456 and saved[-1]["quiet"] is True


@pytest.mark.parametrize("unknown", [False, True])
def test_cancel_stops_owned_job_and_requires_quiet(monkeypatch, unknown):
    saved, actions, cancelled = [], [], [False]
    def control(endpoint, job_id, action, **kwargs):
        actions.append(action)
        if action == "launch":
            cancelled[0] = True
            return {"state": "running", "quiet": False, "receipt": {"pid": 1}}
        if action == "stop":
            return {"state": "uncertain" if unknown else "cancelled", "quiet": not unknown,
                    "unknown": ["lost supervisor"] if unknown else []}
        return {"state": "cancelled", "quiet": True, "stdout": "last output\n", "stdout_offset": 12}
    monkeypatch.setattr("mindie_coordinator.preparation_process.control", control)
    output = []
    with pytest.raises(PreparationUncertain if unknown else PreparationCancelled):
        PreparationProcess(ENDPOINT, "build", lambda row: saved.append(copy.deepcopy(row)), lambda: cancelled[0]).run("build", on_output=lambda *a: output.append(a))
    assert actions.count("stop") == 1
    assert saved[-1]["quiet"] is not unknown
    assert output == ([] if unknown else [("stdout", "last output\n")])


def test_cancel_before_launch_has_no_remote_side_effect(monkeypatch):
    control, save = Mock(), Mock()
    monkeypatch.setattr("mindie_coordinator.preparation_process.control", control)
    with pytest.raises(PreparationCancelled, match="before command launch"):
        PreparationProcess(ENDPOINT, "build", save, lambda: True).run("build", on_output=lambda *a: None)
    control.assert_not_called()
    save.assert_not_called()


def test_stop_transport_failure_is_unknown(monkeypatch):
    monkeypatch.setattr("mindie_coordinator.preparation_process.control", Mock(side_effect=OSError("offline")))
    record = {"endpoint": ENDPOINT, "job_id": "prepare-retained", "quiet": False}
    assert stop_preparation_process(record, lambda _: None) is False
    assert record["state"] == "uncertain" and record["quiet"] is False


def test_diagnostic_failure_preserves_completed_remote_command(monkeypatch):
    from mindie_coordinator.preparation_process import PreparationPostprocessError
    control = Mock(return_value={"state": "succeeded", "quiet": True,
                                 "result": {"exit_code": 0}, "timings": {"prepare_ms": 10}})
    monkeypatch.setattr("mindie_coordinator.preparation_process.control", control)
    recorder = Mock()
    recorder.event.side_effect = OSError("diagnostic disk unavailable")
    monkeypatch.setattr("mindie_coordinator.preparation_process.get_recorder", lambda _: recorder)
    with pytest.raises(PreparationPostprocessError) as caught:
        PreparationProcess(ENDPOINT, "build", lambda _: None, lambda: False).run("build", on_output=lambda *a: None)
    assert caught.value.stage == "diagnostics"
    assert caught.value.completed.returncode == 0
    assert caught.value.preparation_record["quiet"] is True
    assert control.call_count == 1


@pytest.mark.parametrize("quiet", [False, True])
def test_restarted_service_stops_persisted_preparation_before_marking_cancelled(tmp_path, monkeypatch, quiet):
    store = AgentSessions(tmp_path / "sessions")
    context = store.attach("codex", "task", str(tmp_path))
    row = store.execution(context, "build", {"command": "build"})
    row.update(user="user", phase="preparing", preparation_jobs={"worker": {"prepare-retained": {
        "endpoint": ENDPOINT, "job_id": "prepare-retained", "step": "install", "quiet": False,
        "receipt": {"pid": 123, "boot_id": "verified"}}}})
    store.save_execution(row)
    # No original thread, local process handle or runtime binding remains.
    service = CoordinatorService(tmp_path / "coordinator", pool=MagicMock())
    observed = {"state": "cancelled" if quiet else "uncertain", "quiet": quiet,
                "unknown": [] if quiet else ["ownership uncertain"]}
    control = Mock(return_value=observed)
    monkeypatch.setattr("mindie_coordinator.preparation_process.control", control)
    reply = service.advance(str(tmp_path / "sessions"), "user", row["id"], action="stop")
    assert reply["state"] == ("cancelled" if quiet else "uncertain")
    assert reply["resources_released"] is quiet
    control.assert_called_once_with(ENDPOINT, "prepare-retained", "stop", force=False)
    with store.transaction() as db:
        latest = store.get(db, "execution", row["id"])
    assert latest["preparation_jobs"]["worker"]["prepare-retained"]["quiet"] is quiet


def test_retained_preparation_never_replaces_sources(tmp_path):
    service = CoordinatorService(tmp_path / "coordinator", pool=MagicMock())
    with pytest.raises(PreparationUncertain, match="retained preparation"):
        service._place_or_prepare(Mock(), "user", {"preparation_jobs": {"worker": {}}}, [], {})
    service.pool.catalog.assert_not_called()


def test_one_failed_build_cannot_make_other_unquiet_preparation_terminal(tmp_path):
    service = CoordinatorService(tmp_path / "coordinator", pool=MagicMock())
    row = {"id": "execution", "phase": "preparing", "preparation_jobs": {
        "rank0": {"one": {"quiet": True}}, "rank1": {"two": {"quiet": False}},
    }}
    reply = service._record_execution_error(Mock(), row, RemoteCommandError(1, "compiler failed"))
    assert reply["state"] == "uncertain" and reply["resources_released"] is False


def test_cancel_while_waiting_for_other_executions_compiler(tmp_path, monkeypatch):
    store = AgentSessions(tmp_path / "sessions")
    context = store.attach("codex", "waiting-task", str(tmp_path))
    row = store.execution(context, "queued-build", {"command": "build", "source_snapshot": {}})
    row.update(user="user", phase="preparing", sources={}, remote_session={"id": "remote"})
    store.save_execution(row)
    service = CoordinatorService(tmp_path / "coordinator", pool=MagicMock())
    monkeypatch.setattr(service, "_donor_for_role", lambda *a: {"host": "host-a"})
    prepared = Mock()
    def prepare_environment(*args, **kwargs):
        with kwargs['compile_scope']('install-vllm-ascend'):
            prepared()
    monkeypatch.setattr('mindie_coordinator.provision.prepare_task_environment', prepare_environment)
    host_lock = service._lock_for("compile-host", "host-a")
    host_lock.acquire()
    result = []
    waiter = threading.Thread(target=lambda: result.append(service._place_or_prepare(
        store, "user", row, [{"name": "worker"}], {})))
    try:
        waiter.start()
        time.sleep(0.05)
        with store.transaction() as db:
            cancelled = store.get(db, "execution", row["id"])
        cancelled["cancel_requested"] = True
        store.save_execution(cancelled)
        waiter.join(2)
        assert not waiter.is_alive(), "queued cancellation must not wait for the other compilation"
        assert host_lock.locked(), "the other execution must retain its preparation lock"
        prepared.assert_not_called()
        assert result[0]["status"] == "waiting" and row["cancel_requested"] is True
    finally:
        host_lock.release()
        waiter.join(2)


@pytest.mark.skipif(sys.platform != "linux", reason="remote-dev supervisor requires Linux /proc")
def test_actual_supervisor_stop_drains_child_and_preserves_output(tmp_path, monkeypatch):
    from remote_dev.processes.client import worker_source
    from remote_dev.processes.worker import control_job
    endpoint = {**ENDPOINT, "root": str(tmp_path), "cwd": str(tmp_path)}
    saved, output = [], []
    def control(endpoint, job_id, action, **kwargs):
        return control_job({"root": endpoint["root"], "job_id": job_id, "action": action, **kwargs}, worker_source())
    monkeypatch.setattr("mindie_coordinator.preparation_process.control", control)
    def cancelled():
        return (tmp_path / "ready").exists()
    process = PreparationProcess(endpoint, "compiler", lambda row: saved.append(copy.deepcopy(row)), cancelled)
    try:
        with pytest.raises(PreparationCancelled):
            process.run("sleep 300 &\nprintf 'compiler output\\n'; touch ready; wait", on_output=lambda *a: output.append(a))
        assert ("stdout", "compiler output\n") in output
        # Deserialize the package's facts as a restarted daemon would do.
        retained = json.loads(json.dumps(saved[-1]))
        final = control(endpoint, retained["job_id"], "status")
        assert final["quiet"] is True and final["processes"] == []
        assert final["result"]["descendants_drained"] is True
    finally:
        if saved:
            stop_preparation_process(saved[-1], lambda _: None, force=True)


@pytest.mark.skipif(sys.platform != "linux", reason="remote-dev supervisor requires Linux /proc")
def test_actual_supervisor_can_stop_after_losing_the_launch_reply(tmp_path, monkeypatch):
    from remote_dev.processes.client import worker_source
    from remote_dev.processes.worker import control_job

    endpoint = {**ENDPOINT, "root": str(tmp_path), "cwd": str(tmp_path)}
    saved, actions = [], []
    def control(endpoint, job_id, action, **kwargs):
        actions.append(action)
        result = control_job({"root": endpoint["root"], "job_id": job_id, "action": action, **kwargs}, worker_source())
        if action == "launch":
            raise OSError("launch completed remotely but its reply was lost")
        return result
    monkeypatch.setattr("mindie_coordinator.preparation_process.control", control)
    try:
        with pytest.raises(PreparationUncertain, match="was not replayed"):
            PreparationProcess(endpoint, "compiler", lambda row: saved.append(copy.deepcopy(row)), lambda: False).run(
                "sleep 300 &\nprintf 'compiler started\\n'; wait", on_output=lambda *a: None)
        retained = json.loads(json.dumps(saved[-1]))
        assert retained["state"] == "uncertain" and "receipt" not in retained
        assert stop_preparation_process(retained, lambda _: None)
        final = control(endpoint, retained["job_id"], "status")
        assert final["quiet"] is True and final["processes"] == []
        assert final["result"]["descendants_drained"] is True
        assert actions.count("launch") == 1 and "prepare" not in actions and "go" not in actions
    finally:
        if saved:
            stop_preparation_process(saved[-1], lambda _: None, force=True)


@pytest.mark.parametrize('stage', ['persistence', 'output'])
@pytest.mark.parametrize('exit_code', [0, 17])
def test_completed_remote_fact_survives_local_postprocessing_failure(monkeypatch, stage, exit_code):
    from mindie_coordinator.preparation_process import PreparationPostprocessError
    observed = {"state": "succeeded" if exit_code == 0 else "failed", "quiet": True,
                "stdout": "terminal output\n", "result": {"exit_code": exit_code}}
    remote = Mock(return_value=observed)
    monkeypatch.setattr('mindie_coordinator.preparation_process.control', remote)
    saved = []
    error = OSError('local recording unavailable')
    def save(row):
        saved.append(copy.deepcopy(row))
        if stage == 'persistence' and len(saved) > 1:
            raise error
    def output(*args):
        if stage == 'output':
            raise error
    with pytest.raises(PreparationPostprocessError) as caught:
        PreparationProcess(ENDPOINT, 'build', save, lambda: False).run('build', on_output=output)
    failure = caught.value
    assert failure.original_error is error and failure.stage == stage
    assert failure.completed.returncode == exit_code
    assert failure.preparation_record['quiet'] is True
    assert failure.preparation_record['result']['exit_code'] == exit_code
    assert len(saved) == 2 and remote.call_count == 1


def test_transport_failure_is_not_hidden_by_second_persistence_failure(monkeypatch):
    transport = OSError('original SSH failure')
    monkeypatch.setattr('mindie_coordinator.preparation_process.control', Mock(side_effect=transport))
    saved = []
    def save(row):
        saved.append(copy.deepcopy(row))
        if len(saved) > 1:
            raise OSError('disk failure')
    with pytest.raises(PreparationUncertain) as caught:
        PreparationProcess(ENDPOINT, 'build', save, lambda: False).run('build', on_output=lambda *a: None)
    assert caught.value.__cause__ is transport
    assert caught.value.preparation_record['job_id'] == saved[0]['job_id']
    assert len(saved) == 2


def test_verified_stop_cleanup_failure_preserves_quiet_receipt(monkeypatch):
    from mindie_coordinator.preparation_process import PreparationPostprocessError
    monkeypatch.setattr('mindie_coordinator.preparation_process.control', Mock(return_value={
        'state': 'cancelled', 'quiet': True, 'result': {'exit_code': -15}}))
    record = {'endpoint': ENDPOINT, 'job_id': 'prepare-owned', 'quiet': False}
    save = Mock(side_effect=OSError('disk failure'))
    with pytest.raises(PreparationPostprocessError) as caught:
        stop_preparation_process(record, save)
    assert caught.value.preparation_record['quiet'] is True
    assert caught.value.completed.returncode == -15 and save.call_count == 1


def test_explicit_preparation_timeout_is_reported(monkeypatch):
    remote = Mock(return_value={'state': 'timeout', 'quiet': True, 'result': {'exit_code': -15, 'timed_out': True}})
    monkeypatch.setattr('mindie_coordinator.preparation_process.control', remote)
    result = PreparationProcess(ENDPOINT, 'build', lambda _: None, lambda: False,
                                timeout_seconds=7).run('build', on_output=lambda *a: None)
    assert result.timed_out is True and remote.call_args.kwargs['spec']['timeout_seconds'] == 7
