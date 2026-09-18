"""One admission exchange retains epoch fencing and uncertain reconciliation."""
import sqlite3
from contextlib import closing
from types import MethodType

import pytest

import test_coordinator as fixtures
from mindie_coordinator.backend import RemoteBackend
from mindie_coordinator.host.mindie_npu_coordination import CoordinationError, NpuCoordinator, handle_request


@pytest.fixture
def case():
    value = fixtures.PoolTests(methodName='runTest')
    value.setUp()
    value.backend.submit_and_acquire = MethodType(RemoteBackend.submit_and_acquire, value.backend)
    try:
        yield value
    finally:
        value.tearDown()


def test_native_admission_persists_epoch_then_uses_one_mutating_exchange(case):
    binding = case.bind('alice', case.root / 'a')
    original = case.backend.host

    def host(runtime, request):
        if request['action'] == 'submit-acquire':
            with case.pool.transaction() as db:
                row = next(row for row in case.pool.rows(db, 'run') if row['task_id'] == request['task_id'])
            assert row['epoch'] == request['coordination_epoch'] and row['epoch']
            assert row['state'] == 'pending' and not row['submitted']
        return original(runtime, request)

    case.backend.host = host
    case.backend.calls.clear()
    run = case.request('alice', binding)
    assert run['state'] == 'granted'
    assert [action for kind, action in case.backend.calls if kind == 'host'] == ['status', 'submit-acquire']
    events = [item for item in case.pool.events('alice')['events']
              if item.get('kind') == 'run-state' and item.get('run') == run['id']]
    assert [item['state'] for item in events] == ['granted']


def test_lost_combined_reply_reconciles_same_grant_without_repeating_submission(case):
    binding = case.bind('alice', case.root / 'a')
    case.backend.fail_after = 'submit-acquire'
    run = case.request('alice', binding)
    assert run['state'] == 'uncertain' and run['epoch']
    restarted = fixtures.RuntimePool(case.root / 'manager', case.backend)
    recovered = restarted.control('alice', run['id'], 'poll')
    assert recovered['state'] == 'granted' and recovered['task_id'] == run['task_id']
    assert case.backend.calls.count(('host', 'submit-acquire')) == 1


def test_recovered_initial_discovery_emits_one_clean_granted_event(case):
    binding = case.bind('alice', case.root / 'a')
    case.backend.fail_after = 'status'
    run = case.request('alice', binding)
    assert run['state'] == 'pending' and run['error']
    recovered = case.pool.control('alice', run['id'], 'poll')
    assert recovered['state'] == 'granted' and 'error' not in recovered
    events = [item for item in case.pool.events('alice')['events']
              if item.get('kind') == 'run-state' and item.get('run') == run['id']
              and item.get('state') == 'granted']
    assert len(events) == 1 and events[0]['error'] is None


def test_epoch_changed_after_discovery_creates_no_task(case):
    binding = case.bind('alice', case.root / 'a')
    original = case.backend.host

    def host(runtime, request):
        result = original(runtime, request)
        if request['action'] == 'status' and 'coordination_epoch' not in request:
            with closing(sqlite3.connect(case.backend.state / '192_0_2_1/coordinator.sqlite3')) as db, db:
                db.execute("UPDATE meta SET value='restarted' WHERE key='coordination_epoch'")
        return result

    case.backend.host = host
    run = case.request('alice', binding)
    assert run['state'] == 'uncertain' and 'epoch changed' in run['error']
    tasks = original(fixtures.runtime_spec(1), {'action': 'status', 'no_probe': True})['tasks']
    assert not tasks


@pytest.mark.parametrize('shared,expected', [(False, 'waiting'), (True, 'granted')])
def test_combined_admission_keeps_authoritative_occupancy_and_conflicting_leases(tmp_path, shared, expected):
    coordinator = NpuCoordinator(tmp_path)
    epoch = coordinator.snapshot(None)['coordination_epoch']
    request = {'action': 'submit-acquire', 'state_dir': str(tmp_path), 'coordination_epoch': epoch,
               'task_id': 'one', 'agent_id': 'owner', 'devices': [0], 'allow_external_busy': shared}
    probes = []

    def probe():
        probes.append(True)
        return {'status': 'ok', 'devices': [0], 'busy': {'0': ['external worker']}}

    reply = handle_request(request, probe=probe)
    assert reply['status'] == expected and len(probes) == 1
    if shared:
        replay = handle_request(request, probe=lambda: pytest.fail('existing grant is not acquired again'))
        assert replay['task']['fence_token'] == reply['task']['fence_token']
        conflict = handle_request({**request, 'task_id': 'two'}, probe=probe)
        assert conflict['status'] == 'waiting'
    with pytest.raises(CoordinationError, match='different ownership'):
        handle_request({**request, 'agent_id': 'other'}, probe=probe)


def test_partial_submit_survives_failed_probe_without_new_task(tmp_path):
    coordinator = NpuCoordinator(tmp_path)
    request = {'action': 'submit-acquire', 'state_dir': str(tmp_path),
               'coordination_epoch': coordinator.snapshot(None)['coordination_epoch'],
               'task_id': 'one', 'agent_id': 'owner', 'devices': [0]}
    reply = handle_request(request, probe=lambda: {'status': 'failed', 'error': 'unknown visibility'})
    assert reply['status'] == 'probe_failed'
    assert [(row['task_id'], row['state']) for row in coordinator.snapshot(None)['tasks']] == [('one', 'queued')]


def test_combined_cpu_admission_skips_unneeded_npu_and_port_probes(tmp_path):
    coordinator = NpuCoordinator(tmp_path)
    request = {'action': 'submit-acquire', 'state_dir': str(tmp_path),
               'coordination_epoch': coordinator.snapshot(None)['coordination_epoch'],
               'task_id': 'cpu', 'agent_id': 'owner', 'npu_count': 0}
    reply = handle_request(request, probe=lambda: pytest.fail('CPU NPU probe'),
                           listening_ports=lambda: pytest.fail('no service port requested'))
    assert reply['status'] == 'granted' and reply['task']['granted_devices'] == []


def test_combined_admission_requires_a_previously_observed_epoch(tmp_path):
    with pytest.raises(CoordinationError, match='previously observed'):
        handle_request({'action': 'submit-acquire', 'state_dir': str(tmp_path), 'task_id': 'one', 'agent_id': 'owner'})
    assert not list(tmp_path.iterdir())
