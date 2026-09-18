"""Automatic container ports are selected by the host's atomic authority."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from unittest.mock import Mock

import pytest

from mindie_coordinator import provision
from mindie_coordinator.host import mindie_npu_coordination as host
from mindie_coordinator.provision.host_ops import RemoteResult


def request(user, port=0):
    return {'action': 'container-ssh-reserve', 'user': user,
            'container_name': 'mindie-' + user, 'port': port}


@pytest.mark.parametrize('automatic', [True, False])
def test_independent_processes_compete_for_ports_in_one_authority(tmp_path, automatic):
    state = tmp_path / 'authority'
    coordinator = host.NpuCoordinator(state)
    coordinator.reserve_container_ssh(request('reserved-user', 46000))
    program = tmp_path / 'reserve.py'
    program.write_text('''import json, pathlib, sys, time
from mindie_coordinator.host.mindie_npu_coordination import handle_request, CoordinationError
payload = json.loads(sys.argv[1])
gate, ready = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
ready.touch()
deadline = time.monotonic() + 15
while not gate.exists():
    if time.monotonic() > deadline:
        raise RuntimeError('competition gate timed out')
    time.sleep(.005)
def no_npu():
    raise AssertionError('container port allocation must not probe NPUs')
try:
    # Both callers observe the same free range, before either Docker listener
    # exists. The persistent reservation must exclude the other winner.
    result = handle_request(payload, probe=no_npu,
        listening_ports=lambda: {'status': 'ok', 'ports': [46001]})
except CoordinationError as exc:
    result = {'status': 'rejected', 'error': str(exc)}
print(json.dumps(result))
''', encoding='utf-8')
    gate = tmp_path / 'go'
    processes = []
    environment = {**os.environ, 'PYTHONPATH': str(Path(host.__file__).resolve().parents[2]),
                   'PYTHONNOUSERSITE': '1'}
    try:
        for index in range(2):
            payload = {**request('racer-' + str(index), 0 if automatic else 46002),
                       'state_dir': str(state)}
            processes.append(subprocess.Popen(
                [sys.executable, str(program), json.dumps(payload), str(gate), str(tmp_path / f'ready-{index}')],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', env=environment))
        deadline = time.monotonic() + 10
        while not all((tmp_path / f'ready-{index}').exists() for index in range(2)):
            assert time.monotonic() < deadline, 'workers did not reach the competition gate'
            assert all(process.poll() is None for process in processes)
            time.sleep(.005)
        gate.touch()
        results = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=20)
            assert process.returncode == 0, stderr
            results.append(json.loads(stdout))
        if automatic:
            assert {row['port'] for row in results} == {46002, 46003}
            assert all(row['status'] == 'reserved' and row['reused'] is False for row in results)
        else:
            assert sorted(row['status'] for row in results) == ['rejected', 'reserved']
            assert 'port 46002 is already reserved' in next(row['error'] for row in results if row['status'] == 'rejected')
        with sqlite3.connect(coordinator.db_path) as connection:
            ports = {row[0] for row in connection.execute('SELECT port FROM ports')}
        assert ports == ({46000, 46002, 46003} if automatic else {46000, 46002})
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)


def test_automatic_reuses_own_reservation_and_fixed_request_never_moves_it(tmp_path):
    coordinator = host.NpuCoordinator(tmp_path)
    first = coordinator.reserve_container_ssh(request('alice', 46005))
    reused = coordinator.reserve_container_ssh(request('alice'), listening={'status': 'failed'})
    assert reused['port'] == first['port'] and reused['reused'] is True
    with pytest.raises(host.CoordinationError, match='already has SSH port 46005'):
        coordinator.reserve_container_ssh(request('alice', 46006))
    with pytest.raises(host.CoordinationError, match='already reserved'):
        coordinator.reserve_container_ssh(request('bob', 46005))


@pytest.mark.parametrize('listening', [None, {'status': 'failed'}, {'status': 'ok', 'ports': [46000, 46001]}])
def test_unknown_or_exhausted_automatic_pool_does_not_reserve(tmp_path, monkeypatch, listening):
    monkeypatch.setattr(host, 'DEFAULT_CONTAINER_SSH_PORT_RANGE', '46000:46001')
    coordinator = host.NpuCoordinator(tmp_path)
    with pytest.raises(host.CoordinationError, match='unavailable|no free'):
        coordinator.reserve_container_ssh(request('alice'), listening=listening)
    with sqlite3.connect(coordinator.db_path) as connection:
        assert connection.execute('SELECT count(*) FROM ports').fetchone()[0] == 0


def test_fixed_existing_listener_does_not_add_a_probe(tmp_path):
    result = host.handle_request({**request('alice', 2201), 'state_dir': str(tmp_path)},
        probe=Mock(side_effect=AssertionError('no NPU probe')),
        listening_ports=Mock(side_effect=AssertionError('fixed existing port needs no selection probe')))
    assert result['port'] == 2201


def provision_fixture(monkeypatch):
    calls = []
    def remote(target, script, **kwargs):
        calls.append((target, script, kwargs))
        return RemoteResult(target, 0, '', '', {'success': True, 'free_port': 46000})
    monkeypatch.setattr(provision.host_ops, 'run_remote_script', remote)
    monkeypatch.setattr(provision.host_ops, 'find_public_key', lambda _: 'fixture-key')
    monkeypatch.setattr(provision.host_ops, 'load_public_key', lambda _: 'ssh-ed25519 fixture')
    return calls


@pytest.mark.parametrize('fixed', [None, 2201])
def test_provision_uses_atomic_result_in_bootstrap_and_metadata_endpoint(monkeypatch, fixed):
    calls = provision_fixture(monkeypatch)
    reserve = Mock(return_value={'status': 'reserved', 'port': fixed or 46002})
    result = provision.provision_user_container(host='fixture', user='alice', image='main',
        ssh_port=fixed, reserve_port=reserve, machines=Mock())
    reserve.assert_called_once_with(user='alice', container_name='mindie-alice', port=fixed or 0)
    assert result['ssh_port'] == (fixed or 46002)
    assert calls[1][2]['args'][1] == str(fixed or 46002)
    assert calls[2][0].port == (fixed or 46002)


@pytest.mark.parametrize('failure', [RuntimeError('remote outcome unknown'), TimeoutError('deadline'),
                                     RuntimeError('cancelled')])
def test_provision_never_retries_uncertain_reservation_or_bootstraps(monkeypatch, failure):
    calls = provision_fixture(monkeypatch)
    reserve = Mock(side_effect=failure)
    with pytest.raises(type(failure)) as caught:
        provision.provision_user_container(host='fixture', user='alice', image='main',
            reserve_port=reserve, machines=Mock())
    assert caught.value is failure and len(calls) == 1
    reserve.assert_called_once()


def test_provision_rejects_changed_fixed_port(monkeypatch):
    calls = provision_fixture(monkeypatch)
    with pytest.raises(provision.host_ops.MachineManagementError, match='requested fixed port'):
        provision.provision_user_container(host='fixture', user='alice', image='main', ssh_port=2201,
            reserve_port=Mock(return_value={'port': 2202}), machines=Mock())
    assert len(calls) == 1
