"""Completion waits never hold control locks or delay unrelated host work."""
import copy
import threading
import time
import sys

import pytest

from test_coordinator import Backend, runtime_spec
from mindie_coordinator.ready_runtime import RuntimePool


@pytest.fixture
def running(tmp_path):
    backend = Backend(tmp_path / 'host')
    pool = RuntimePool(tmp_path / 'pool', backend)
    pool.register('runtime', runtime_spec(1))
    session = pool.session_open('alice', 'task', {})
    binding = pool.checkout('alice', session['id'], 'profile-a', 'checkout')
    job = pool.managed_start('alice', binding['id'], 'run', {}, 'native-a', [], 0, 'true', {}, 60)
    assert job['state'] == 'running'
    return pool, backend, job


def test_public_run_wait_wakes_on_completion_without_a_ticker(monkeypatch):
    from test_coordinator import TaskClientTests
    task = TaskClientTests()
    task.setUp()
    service = task.client.coordinator
    service._async_progress = True
    def wait(runtime, key, **kwargs):
        task.backend.jobs[key].update(state='succeeded', quiet=True, stdout='final business output', stderr='',
                                      result={'descendants_drained': True},
                                      receipt={**task.backend.jobs[key].get('receipt', {}), 'job_id': key})
        return copy.deepcopy(task.backend.jobs[key])
    monkeypatch.setattr(task.backend, 'wait_job', wait, raising=False)
    try:
        result = task.client.run('true', sources={}, resources={'npu_count': 0},
                                 wait_until='released', wait_timeout_seconds=5)
        assert result['state'] == 'succeeded' and result['resources_released'] is True
        assert 'tail' in result
        assert result['stdout'] == 'final business output'
        assert ('job', 'tail') not in task.backend.calls
    finally:
        service._stopped.set()
        task.tearDown()


@pytest.mark.parametrize('change', [None, 'running', 'active-lease', 'status-only', 'missing-stderr',
                                  'draining', 'wrong-job', 'unquiet', 'unknown'])
def test_terminal_tail_reuse_requires_same_drained_job_and_explicit_output(running, change):
    pool, backend, job = running
    remote = {'state': 'succeeded', 'quiet': True, 'receipt': {'job_id': job['job_id']},
              'result': {'descendants_drained': True}, 'stdout': 'final', 'stderr': ''}
    job.update(state='succeeded', lease_state='released', remote=remote)
    if change == 'running': job['state'] = 'running'
    elif change == 'active-lease': job['lease_state'] = 'active'
    elif change == 'status-only': remote.pop('stdout'); remote.pop('stderr')
    elif change == 'missing-stderr': remote.pop('stderr')
    elif change == 'draining': remote['result']['descendants_drained'] = False
    elif change == 'wrong-job': remote['receipt']['job_id'] = 'earlier-job'
    elif change == 'unquiet': remote['quiet'] = False
    elif change == 'unknown': remote['unknown'] = True
    with pool.transaction() as db:
        pool.put(db, 'job', job)
    backend.calls.clear()
    result = pool.managed_control('alice', job['id'], 'tail')
    assert (('job', 'tail') in backend.calls) is bool(change)
    if change is None:
        assert result['remote']['stdout'] == 'final' and result['remote']['stderr'] == ''


def test_completion_event_releases_without_a_second_status_poll(running, monkeypatch):
    pool, backend, job = running
    entered, complete, delivered, stopped = (threading.Event() for _ in range(4))
    waits = []
    def wait(runtime, key, **kwargs):
        waits.append(key)
        entered.set()
        assert complete.wait(3)
        backend.jobs[key].update(state='succeeded', quiet=True)
        return copy.deepcopy(backend.jobs[key])
    monkeypatch.setattr(backend, 'wait_job', wait, raising=False)
    observed = []
    def done(row):
        observed.append(row)
        delivered.set()
    try:
        pool.watch_completion(job['id'], done, stopped)
        assert entered.wait(3)
        pool.watch_completion(job['id'], done, stopped)
        assert pool._entity_lock('job', job['id']).acquire(blocking=False)
        pool._entity_lock('job', job['id']).release()
        backend.calls.clear()
        complete.set()
        assert delivered.wait(3)
        assert waits == [job['job_id']]
        assert observed[0]['state'] == 'succeeded' and observed[0]['lease_state'] == 'released'
        assert ('job', 'status') not in backend.calls
        assert ('host', 'release') in backend.calls
    finally:
        complete.set()
        stopped.set()


def test_wait_does_not_block_stop_or_overwrite_cancellation(running, monkeypatch):
    pool, backend, job = running
    entered, release, finished, stopped = (threading.Event() for _ in range(4))
    def wait(runtime, key, **kwargs):
        entered.set()
        assert release.wait(3)
        return {'state': 'succeeded', 'quiet': True}
    monkeypatch.setattr(backend, 'wait_job', wait, raising=False)
    try:
        pool.watch_completion(job['id'], lambda row: finished.set(), stopped)
        assert entered.wait(3)
        stopped_job = pool.managed_control('alice', job['id'], 'stop')
        assert stopped_job['state'] == 'stopping'
        stopped_job = pool.managed_control('alice', job['id'], 'status')
        assert stopped_job['state'] == 'cancelled'
        release.set()
        assert finished.wait(3)
        assert pool.managed_control('alice', job['id'])['state'] == 'cancelled'
    finally:
        release.set()
        stopped.set()


@pytest.mark.parametrize('state', ['succeeded', 'absent', 'lost_outcome'])
def test_early_result_or_lost_receipt_never_releases_before_quiet(running, monkeypatch, state):
    pool, backend, job = running
    entered, release, delivered, stopped = (threading.Event() for _ in range(4))
    calls = []
    def wait(runtime, key, **kwargs):
        calls.append(time.monotonic())
        if len(calls) == 1:
            entered.set()
            return {'state': state, 'quiet': False}
        assert release.wait(3)
        backend.jobs[key].update(state='succeeded', quiet=True)
        return copy.deepcopy(backend.jobs[key])
    monkeypatch.setattr(backend, 'wait_job', wait, raising=False)
    backend.calls.clear()
    try:
        pool.watch_completion(job['id'], lambda row: delivered.set(), stopped)
        assert entered.wait(3)
        assert ('host', 'release') not in backend.calls and not delivered.is_set()
        release.set()
        if state == 'succeeded':
            assert delivered.wait(3)
            assert calls[1] - calls[0] >= .08
        else:
            with pool._entity_lock('completion-watch', job['id']):
                pass
            assert len(calls) == 1 and not delivered.is_set()
    finally:
        release.set()
        stopped.set()


def test_slow_manual_pool_job_does_not_block_other_dispatch(tmp_path, monkeypatch):
    pool = RuntimePool(tmp_path, object())
    with pool.transaction() as db:
        for name in ('slow', 'healthy'):
            pool.put(db, 'run', {'id': name, 'owner': 'alice', 'state': 'queued', 'last_poll': 0})
    slow, healthy, release = (threading.Event() for _ in range(3))
    calls = []
    def control(owner, key, action):
        calls.append(key)
        if key == 'slow':
            slow.set()
            assert release.wait(3)
        else:
            healthy.set()
    monkeypatch.setattr(pool, 'control', control)
    try:
        pool.tick(background=True)
        assert slow.wait(3) and healthy.wait(3)
        pool.tick(background=True)
        assert calls.count('slow') == 1
    finally:
        release.set()
        with pool._entity_lock('background-tick', 'slow'):
            pass


@pytest.mark.skipif(sys.platform != 'linux', reason='real owned supervisor uses Linux process identities')
def test_real_owned_wait_returns_completion_and_allows_concurrent_stop(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from remote_dev.processes.client import worker_source
    from remote_dev.processes.worker import control_job
    from mindie_coordinator.backend import RemoteBackend
    worker = worker_source()
    backend = RemoteBackend()
    def control(runtime, key, action, **kwargs):
        return control_job({'root': str(tmp_path), 'job_id': key, 'action': action, **kwargs}, worker)
    monkeypatch.setattr(backend, 'job', control)
    for key, command, stop in [('natural', 'sleep .2; exit 17', False), ('cancelled', 'sleep 30', True)]:
        control({}, key, 'launch', spec={'command': command, 'cwd': str(tmp_path), 'env': {},
                'timeout_seconds': 60, 'interactive': False}, authorization={}, yield_time_ms=0)
        try:
            with ThreadPoolExecutor(1) as executor:
                pending = executor.submit(backend.wait_job, {}, key, timeout_seconds=3)
                if stop:
                    # The remote wait releases its job lock between observations.
                    control({}, key, 'stop', force=True)
                observed = pending.result(timeout=5)
            # A bounded exchange may wake on result publication before the
            # supervisor's final drain. Observe that same job until quiet.
            deadline = time.monotonic() + 3
            while not observed['quiet'] and time.monotonic() < deadline:
                time.sleep(.1)
                observed = control({}, key, 'status')
            assert observed['quiet'] is True
            assert observed['result']['descendants_drained'] is True
            if not stop:
                assert observed['result']['exit_code'] == 17
        finally:
            control({}, key, 'stop', force=True)
