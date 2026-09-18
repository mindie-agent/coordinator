"""Fixed-image reuse is a bounded observation, never an implicit replacement."""
from contextlib import redirect_stdout
import io
import json
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mindie_coordinator import provision
from mindie_coordinator.provision import existing_container as existing
from mindie_coordinator.provision.host_ops import MachineManagementError, RemoteResult, SshTarget


TARGET = SshTarget(host='192.0.2.10', user='root', port=22)
IMAGE = 'registry.example/ascend:v1.2.3'


def request(image=IMAGE):
    return {'container': 'mindie-alice', 'user': 'alice', 'ssh_port': 2201,
            'workdir': provision.DEFAULT_WORKDIR,
            'image_request': provision.host_ops.image_request_payload(image)}


def container(image=IMAGE):
    return {'Name': '/mindie-alice', 'Id': 'a' * 64, 'Image': 'sha256:' + 'b' * 64,
            'State': {'Running': True, 'Paused': False, 'Restarting': False, 'Pid': 101},
            'HostConfig': {'NetworkMode': 'host'},
            'Config': {'Image': image, 'Labels': {'com.mindie.managed': 'true',
                'com.mindie.namespace': 'alice', 'com.mindie.workdir': provision.DEFAULT_WORKDIR,
                'com.mindie.container_ssh_port': '2201', 'com.mindie.base_image': image,
                'com.mindie.base_image_id': 'sha256:' + 'b' * 64}}}


def command_results(info=None, listener=None, processes='PID\n101\n102\n'):
    return Mock(side_effect=[
        SimpleNamespace(stdout=json.dumps([info if info is not None else container()])),
        SimpleNamespace(stdout=processes),
        SimpleNamespace(stdout=listener if listener is not None else
                        'LISTEN 0 128 0.0.0.0:2201 0.0.0.0:* users:(("sshd",pid=102,fd=3))\n'),
    ])


@pytest.mark.parametrize('image', [IMAGE, 'registry.example/ascend@sha256:' + 'c' * 64])
def test_matching_fixed_image_requires_real_container_owned_ssh_listener(image):
    run = command_results(container(image))
    result = existing.inspect_existing(request(image), run=run)
    assert result['status'] == 'match' and result['image'] == image
    assert result['container_id'] == 'a' * 64 and result['image_id'] == 'sha256:' + 'b' * 64
    assert result['ssh_port'] == 2201 and result['listener_pids'] == [102]
    assert run.call_count == 3
    assert run.call_args_list[0].args[0] == ['docker', 'inspect', 'mindie-alice']
    assert run.call_args_list[1].args[0] == ['docker', 'top', 'a' * 64, '-eo', 'pid']
    assert run.call_args_list[2].args[0] == ['ss', '-ltnpH', 'sport = :2201']
    assert all(0 < call.kwargs['timeout'] <= 3 for call in run.call_args_list)


def test_different_image_is_rejected_even_if_base_label_claims_the_requested_image():
    info = container('registry.example/ascend:other')
    info['Config']['Labels']['com.mindie.base_image'] = IMAGE
    run = command_results(info)
    assert existing.inspect_existing(request(), run=run)['status'] == 'mismatch'
    assert run.call_count == 1
    # provision_user_container uses bootstrap's default prepared-cache=false.


@pytest.mark.parametrize('damage', ['name', 'id', 'image-id', 'stopped', 'paused', 'restarting',
                                  'pid', 'label', 'user', 'port', 'workdir', 'network'])
def test_incomplete_container_facts_cannot_skip_bootstrap(damage):
    info = container()
    if damage == 'name':
        info['Name'] = '/mindie-bob'
    elif damage == 'id':
        info['Id'] = ''
    elif damage == 'image-id':
        info['Image'] = ''
    elif damage == 'stopped':
        info['State']['Running'] = False
    elif damage in {'paused', 'restarting'}:
        info['State'][damage.title()] = True
    elif damage == 'pid':
        info['State']['Pid'] = 0
    elif damage == 'network':
        info['HostConfig']['NetworkMode'] = 'bridge'
    else:
        key = {'label': 'managed', 'user': 'namespace', 'port': 'container_ssh_port', 'workdir': 'workdir'}[damage]
        info['Config']['Labels']['com.mindie.' + key] = 'different'
    assert existing.inspect_existing(request(), run=command_results(info))['status'] == 'unknown'


@pytest.mark.parametrize('listener', ['', 'LISTEN 0 128 0.0.0.0:2201 *:*',
    'LISTEN 0 128 0.0.0.0:2201 *:* users:(("sshd",pid=999,fd=3))',
    'LISTEN 0 128 0.0.0.0:2201 *:* users:(("python",pid=102,fd=3))'])
def test_unowned_or_unobservable_listener_does_not_establish_reuse(listener):
    assert existing.inspect_existing(request(), run=command_results(listener=listener))['status'] == 'unknown'


@pytest.mark.parametrize('failure', [FileNotFoundError('docker'), subprocess.TimeoutExpired('docker', 3),
                                   subprocess.CalledProcessError(1, 'docker')])
def test_missing_or_slow_host_facts_are_bounded_unknowns(failure):
    assert existing.inspect_existing(request(), run=Mock(side_effect=failure))['status'] == 'unknown'


def test_transported_program_executes_the_same_self_contained_observation(monkeypatch):
    from remote_dev.core import ssh_transport
    remote = Mock()
    actual = command_results()

    def execute(endpoint, source, payload, *, timeout_ms):
        assert endpoint.root == '/' and endpoint.port == 22 and timeout_ms == 10000
        remote(source, payload)
        output = io.StringIO()
        with monkeypatch.context() as context:
            context.setattr(subprocess, 'run', actual)
            context.setattr('sys.stdin', io.StringIO(json.dumps(payload)))
            with redirect_stdout(output):
                exec(compile(source, '<transported-container-observation>', 'exec'), {})
        return json.loads(output.getvalue())

    monkeypatch.setattr(ssh_transport, 'run_remote_python', execute)
    first = existing.observe_existing(TARGET, **request())
    assert first['status'] == 'match' and remote.call_count == 1
    assert actual.call_count == 3


def test_fast_match_skips_bootstrap_and_smoke_but_keeps_port_reservation(monkeypatch):
    facts = existing.inspect_existing(request(), run=command_results())
    observe = Mock(return_value=facts)
    monkeypatch.setattr(existing, 'observe_existing', observe)
    remote = Mock(side_effect=AssertionError('fast match must not probe/bootstrap/smoke'))
    monkeypatch.setattr(provision.host_ops, 'run_remote_script', remote)
    directory, reserve = Mock(), Mock(return_value={'port': 2201})
    result = provision.provision_user_container(host=TARGET.host, image=IMAGE, user='alice',
        ssh_port=2201, machines=directory, reserve_port=reserve)
    assert result['image_verification'] == facts and result['state'] == 'ready'
    assert result['ssh_port'] == 2201
    reserve.assert_called_once_with(user='alice', container_name='mindie-alice', port=2201)
    assert directory.upsert_machine.call_args.args[0]['image']['requested'] == IMAGE
    observe.assert_called_once()
    remote.assert_not_called()


@pytest.mark.parametrize('status', ['mismatch', 'cancelled'])
def test_mismatch_and_cancellation_never_enter_mutating_fallback(status, monkeypatch):
    monkeypatch.setattr(existing, 'observe_existing', lambda *a, **k: {'status': status, 'reason': 'image mismatch'})
    remote, directory, reserve = Mock(), Mock(), Mock()
    monkeypatch.setattr(provision.host_ops, 'run_remote_script', remote)
    with pytest.raises(MachineManagementError, match='mismatch|cancelled'):
        provision.provision_user_container(host=TARGET.host, image=IMAGE, user='alice',
            ssh_port=2201, machines=directory, reserve_port=reserve)
    remote.assert_not_called()
    reserve.assert_not_called()
    directory.upsert_machine.assert_not_called()


def test_port_reservation_cannot_silently_change_the_verified_listener(monkeypatch):
    monkeypatch.setattr(existing, 'observe_existing', lambda *a, **k: {'status': 'match'})
    directory = Mock()
    with pytest.raises(MachineManagementError, match='reserved SSH port differs'):
        provision.provision_user_container(host=TARGET.host, image=IMAGE, user='alice',
            ssh_port=2201, machines=directory, reserve_port=lambda **k: {'port': 2299})
    directory.upsert_machine.assert_not_called()


def test_uncertain_read_returns_to_original_non_replacing_provision(monkeypatch):
    monkeypatch.setattr(existing, 'observe_existing', lambda *a, **k: {'status': 'unknown'})
    remote = Mock(side_effect=[RemoteResult(TARGET, 0, '', '', {'success': True, 'suggested_port': 2299}),
                              RemoteResult(TARGET, 0, '', '', {'success': True}),
                              RemoteResult(TARGET, 0, '', '', {'success': True})])
    monkeypatch.setattr(provision.host_ops, 'run_remote_script', remote)
    monkeypatch.setattr(provision.host_ops, 'find_public_key', lambda _: 'unused')
    monkeypatch.setattr(provision.host_ops, 'load_public_key', lambda _: 'ssh-ed25519 test')
    result = provision.provision_user_container(host=TARGET.host, image=IMAGE, user='alice',
                                               ssh_port=2201, machines=Mock())
    assert result['state'] == 'ready' and 'image_verification' not in result
    assert remote.call_count == 3
    assert remote.call_args_list[1].kwargs['args'][1] == '2201'
    assert remote.call_args_list[1].kwargs['args'][6] == 'false'


def test_transport_cancellation_is_preserved_instead_of_becoming_unknown(monkeypatch):
    from remote_dev.core import ssh_transport
    from remote_dev.core.cancellation import request_context
    cancel = threading.Event()
    cancel.set()
    monkeypatch.setattr(ssh_transport, 'run_remote_python', lambda *a, **k: {'status': 'failed'})
    with request_context(cancel):
        assert existing.observe_existing(TARGET, **request())['status'] == 'cancelled'
