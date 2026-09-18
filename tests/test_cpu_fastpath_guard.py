"""CPU observation skips hardware, without turning unknown process state free."""
import os

import pytest

from mindie_coordinator.host.mindie_npu_coordination import CoordinationError, NpuCoordinator, handle_request


FREE = {'status': 'ok', 'devices': [0], 'busy': {}, 'free': [0]}


def starting(host, task='cpu', count=0):
    host.submit({'task_id': task, 'agent_id': 'owner', 'npu_count': count})
    token = host.acquire(task, FREE if count else None)['task']['fence_token']
    host.preflight(task, token, FREE if count else None)
    return token


def test_cpu_activation_requires_guard_and_preserves_npu_pid_only_contract(tmp_path):
    host = NpuCoordinator(tmp_path)
    token = starting(host)
    with pytest.raises(CoordinationError, match='CPU task activation requires a valid process_guard'):
        host.activate('cpu', token, pid=os.getpid())
    npu_token = starting(host, 'npu', 1)
    assert host.activate('npu', npu_token, pid=os.getpid())['status'] == 'active'


@pytest.mark.parametrize('state', ['active', 'orphaned_busy'])
@pytest.mark.parametrize('action', ['release', 'cancel', 'gc'])
@pytest.mark.parametrize('observed', [None, FREE])
def test_legacy_cpu_pid_only_stays_unknown_even_when_hardware_is_free(tmp_path, state, action, observed):
    host = NpuCoordinator(tmp_path)
    token = starting(host)
    # Reproduce rows created before CPU activation required a process guard.
    with host._transaction() as db:
        db.execute("UPDATE tasks SET state=?, pid=?, heartbeat_deadline=0, activation_deadline=NULL WHERE task_id='cpu'",
                   (state, os.getpid()))
    if action == 'release':
        result = host.release('cpu', token, observed, completion_confirmed=True)['task']
    elif action == 'cancel':
        result = host.cancel('cpu', observed)['task']
    else:
        result = host.snapshot(observed, task_id='cpu')['tasks'][0]
    assert result['state'] == 'orphaned_busy'
    assert result['pid'] == os.getpid()


@pytest.mark.parametrize('preflight', [False, True])
def test_cpu_grant_that_never_activated_can_still_be_cancelled(tmp_path, preflight):
    host = NpuCoordinator(tmp_path)
    host.submit({'task_id': 'cpu', 'agent_id': 'owner', 'npu_count': 0})
    token = host.acquire('cpu', None)['task']['fence_token']
    if preflight:
        host.preflight('cpu', token, None)
    assert host.cancel('cpu', None)['status'] == 'cancelled'


def test_cpu_observation_does_not_free_an_expired_unobserved_npu_grant(tmp_path):
    now = [1000.0]
    host = NpuCoordinator(tmp_path, clock=lambda: now[0])
    host.submit({'task_id': 'npu', 'agent_id': 'owner', 'npu_count': 1})
    host.acquire('npu', FREE, grant_ttl_seconds=1)
    now[0] += 2
    host.submit({'task_id': 'cpu', 'agent_id': 'owner', 'npu_count': 0})
    assert host.acquire('cpu', None)['status'] == 'granted'
    npu = host.snapshot(None, task_id='npu')['tasks'][0]
    assert npu['state'] == 'orphaned_busy' and npu['granted_devices'] == [0]


def test_cpu_service_unknown_listener_preserves_the_reserved_port(tmp_path):
    host = NpuCoordinator(tmp_path)
    host.submit({'task_id': 'cpu', 'agent_id': 'owner', 'npu_count': 0, 'service_port': 0})
    grant = host.acquire('cpu', None, listening={'status': 'ok', 'ports': []})['task']
    def forbidden():
        raise AssertionError('CPU task must not probe NPU occupancy')
    for action in ('release', 'cancel'):
        result = handle_request({'action': action, 'task_id': 'cpu', 'state_dir': str(tmp_path),
                                 'fence_token': grant['fence_token'], 'completion_confirmed': True},
                                probe=forbidden, listening_ports=lambda: {'status': 'failed', 'error': 'unavailable'})
        assert result['status'] == 'orphaned_busy'
        assert result['task']['granted_service_port'] == grant['granted_service_port']
