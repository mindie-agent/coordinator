"""Container phases use existing execution progress without another observation."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from mindie_coordinator import provision
from mindie_coordinator.provision import existing_container
from mindie_coordinator.provision.host_ops import MachineManagementError, RemoteResult, SshTarget
from mindie_coordinator.service import CoordinatorService

IMAGE = 'registry.example/ascend@sha256:' + 'a' * 64
TARGET = SshTarget(host='fixture', user='root', port=22)


def setup(monkeypatch, *, match=False, failure=None):
    clock = [100.0]
    monkeypatch.setattr(provision, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    def observe(*args, **kwargs):
        clock[0] += 10.0
        return {'status': 'match' if match else 'unknown', 'reason': 'observation timed out',
                'stdout': 'PRIVATE FULL PAYLOAD', 'credential': 'SECRET'}
    observation = Mock(side_effect=observe)
    monkeypatch.setattr(existing_container, 'observe_existing', observation)
    durations = iter([2.0, 3.0, 4.0])
    def remote(*args, **kwargs):
        clock[0] += next(durations)
        if failure is not None:
            raise failure
        return RemoteResult(TARGET, 0, '', '', {'success': True, 'free_port': 2201})
    calls = Mock(side_effect=remote)
    monkeypatch.setattr(provision.host_ops, 'run_remote_script', calls)
    monkeypatch.setattr(provision.host_ops, 'find_public_key', lambda _: 'fixture-key')
    monkeypatch.setattr(provision.host_ops, 'load_public_key', lambda _: 'ssh-ed25519 SECRET')
    return observation, calls


@pytest.mark.parametrize('match', [False, True])
def test_progress_retains_initial_observation_and_actual_fallback_phases(monkeypatch, match):
    observe, calls = setup(monkeypatch, match=match)
    events = []
    result = provision.provision_user_container(host=TARGET.host, image=IMAGE, user='alice',
        ssh_port=2201, machines=Mock(), on_progress=events.append)
    assert result['state'] == 'ready'
    assert calls.call_count == (0 if match else 3) and observe.call_count == 1
    completed = [event for event in events if event['status'] == 'complete']
    phases = ['existing-container'] + ([] if match else ['host-probe', 'container-bootstrap', 'metadata-readiness'])
    assert [event['phase'] for event in completed] == phases
    assert completed[-1]['phase_seconds'] == dict(zip(phases, [10.0, 2.0, 3.0, 4.0]))
    assert completed[-1]['existing_observation'] == {'status': 'match' if match else 'unknown',
        'reason': 'observation timed out', 'elapsed_seconds': 10.0}
    assert 'SECRET' not in json.dumps(events) and 'PRIVATE FULL PAYLOAD' not in json.dumps(events)
    # Each start precedes the actual operation; progress never adds a probe.
    assert [(event['phase'], event['status']) for event in events] == [
        (phase, status) for phase in phases for status in ['running', 'complete']]


def test_failed_fallback_records_elapsed_without_changing_the_exception(monkeypatch):
    failure = MachineManagementError('failed with PRIVATE FULL PAYLOAD')
    observe, calls = setup(monkeypatch, failure=failure)
    events = []
    with pytest.raises(MachineManagementError) as error:
        provision.provision_user_container(host=TARGET.host, image=IMAGE, user='alice',
            ssh_port=2201, machines=Mock(), on_progress=events.append)
    assert error.value is failure and observe.call_count == calls.call_count == 1
    assert events[-1]['phase'] == 'host-probe' and events[-1]['status'] == 'failed'
    assert events[-1]['phase_seconds']['host-probe'] == 2.0
    assert events[-1]['existing_observation']['status'] == 'unknown'
    assert events[-1]['error_type'] == 'MachineManagementError'
    assert 'PRIVATE FULL PAYLOAD' not in json.dumps(events)


def test_placement_routes_owner_phases_to_persisted_run_log_before_root_preparation(tmp_path, monkeypatch):
    observe, calls = setup(monkeypatch)
    pool = MagicMock()
    pool.catalog.return_value = []
    pool.backend.host.return_value = {'port': 2201}
    service = CoordinatorService(tmp_path / 'coordinator', pool=pool)
    service._configured_machines = lambda: [{'host': {'ip': 'fixture'}, 'user': 'alice',
        'container': {'name': 'mindie-alice', 'ssh_port': 2201}}]
    monkeypatch.setattr(service, '_adopt_cancel', lambda *a: False)
    monkeypatch.setattr(service, '_prepare_role', Mock(side_effect=RuntimeError('reached root preparation')))
    row, store = {'id': 'owned-execution', 'spec': {}}, Mock()
    with pytest.raises(RuntimeError, match='reached root preparation'):
        service._place_or_prepare(store, 'alice', row, [{'name': 'worker'}], {'image': IMAGE})
    assert calls.call_count == 3 and observe.call_count == 1
    assert row['progress']['step'] == 'prepare-container'
    assert row['progress']['existing_observation']['status'] == 'unknown'
    log = tmp_path / 'coordinator/runs/owned-execution/worker/prepare-container.log'
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert records[-1]['phase'] == 'metadata-readiness'
    assert records[-1]['phase_seconds'] == {'existing-container': 10.0, 'host-probe': 2.0,
        'ssh-port-reservation': 0.0, 'container-bootstrap': 3.0, 'metadata-readiness': 4.0}
    service._save_progress(store, row, 'worker', {'step': 'prepare-root'})
    assert row['progress']['step'] == 'prepare-root'
    assert json.loads(log.read_text().splitlines()[-1])['existing_observation']['reason'] == 'observation timed out'
    assert 'SECRET' not in log.read_text() and 'PRIVATE FULL PAYLOAD' not in log.read_text()
