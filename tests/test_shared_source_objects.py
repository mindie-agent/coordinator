import json
import subprocess
from types import SimpleNamespace

import pytest

from mindie_coordinator import parity
from mindie_coordinator.preparation_process import PreparationCancelled, PreparationUncertain
from mindie_coordinator.shared_source_objects import export_existing_objects


def container(identifier='a'):
    return {'Id': identifier * 64, 'Name': '/mindie-example', 'Image': 'sha256:' + 'b' * 64,
            'Config': {'Labels': {'com.mindie.managed': 'true'}}, 'State': {'Running': True},
            'Mounts': [{'Type': 'bind', 'Source': '/tmp', 'Destination': '/tmp', 'RW': True}]}


def request():
    return {'records': [{'commit': 'c' * 40, 'tree': 'd' * 40, 'repo_id': 'project',
                         'shared_mirror': '/tmp/shared/project.git'}],
            'export_source': 'bounded export program', 'legacy_cache': '/private-cache'}


def test_export_uses_fixed_docker_id_and_only_requested_objects():
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        if args[1] == 'ps':
            return SimpleNamespace(stdout='aaaa\n')
        if args[1] == 'inspect':
            return SimpleNamespace(stdout=json.dumps([container()]))
        assert args[:5] == ['docker', 'exec', '-i', 'a' * 64, 'python3']
        assert json.loads(kwargs['input']) == {'records': request()['records'], 'legacy_cache': '/private-cache'}
        return SimpleNamespace(returncode=0, stdout=json.dumps({'copied': ['c' * 40]}), stderr='')
    assert export_existing_objects(request(), run) == {'status': 'copied', 'copied': ['c' * 40]}
    assert len(calls) == 3


@pytest.mark.parametrize('damage', ['label', 'mount', 'id', 'stopped', 'image'])
def test_unqualified_container_is_not_read(damage):
    info = container()
    if damage == 'label':
        info['Config']['Labels'] = {}
    elif damage == 'mount':
        info['Mounts'] = []
    elif damage == 'id':
        info['Id'] = 'reused-name'
    elif damage == 'stopped':
        info['State']['Running'] = False
    else:
        info['Image'] = 'mutable-tag'
    def run(args, **kwargs):
        assert args[1] != 'exec'
        return SimpleNamespace(stdout='aaaa\n' if args[1] == 'ps' else json.dumps([info]))
    assert export_existing_objects(request(), run) == {'status': 'miss', 'copied': []}


@pytest.mark.parametrize('outcome', ['timeout', 'signal'])
def test_unknown_export_is_not_replayed_on_another_donor(outcome):
    calls = []
    def run(args, **kwargs):
        if args[1] == 'ps':
            return SimpleNamespace(stdout='aaaa\nbbbb\n')
        if args[1] == 'inspect':
            return SimpleNamespace(stdout=json.dumps([container(), container('e')]))
        calls.append(args)
        if outcome == 'timeout':
            raise subprocess.TimeoutExpired(args, kwargs['timeout'])
        return SimpleNamespace(returncode=137, stdout='', stderr='copy interrupted')
    assert export_existing_objects(request(), run)['status'] == 'uncertain'
    assert len(calls) == 1


def test_known_unavailable_donor_uses_another_candidate():
    calls = []
    def run(args, **kwargs):
        if args[1] == 'ps':
            return SimpleNamespace(stdout='aaaa\nbbbb\n')
        if args[1] == 'inspect':
            return SimpleNamespace(stdout=json.dumps([container(), container('e')]))
        calls.append(args[3])
        if len(calls) == 1:
            return SimpleNamespace(returncode=1, stdout='', stderr='container disappeared')
        return SimpleNamespace(returncode=0, stdout=json.dumps({'copied': ['c' * 40]}), stderr='')
    result = export_existing_objects(request(), run)
    assert result['status'] == 'copied' and result['copied'] == ['c' * 40]
    assert calls == ['a' * 64, 'e' * 64] and result['diagnostics']


@pytest.mark.parametrize('stdout', ['startup notice\n{}', '[]', '{}', '{"copied":null}',
    '{"copied":[{}]}', '{"copied":["unexpected"]}', '{"copied":[],"diagnostics":{}}'])
@pytest.mark.parametrize('second_valid', [False, True])
def test_completed_invalid_donor_reply_keeps_other_candidates_and_git_fallback(stdout, second_valid):
    calls = []
    def run(args, **kwargs):
        if args[1] == 'ps':
            return SimpleNamespace(stdout='aaaa\nbbbb\n')
        if args[1] == 'inspect':
            return SimpleNamespace(stdout=json.dumps([container(), container('e')]))
        calls.append(args[3])
        output = json.dumps({'copied': ['c' * 40]}) if second_valid and len(calls) == 2 else stdout
        return SimpleNamespace(returncode=0, stdout=output, stderr='')
    result = export_existing_objects(request(), run)
    assert calls == ['a' * 64, 'e' * 64]
    assert result['status'] == ('copied' if second_valid else 'miss')
    assert result['copied'] == (['c' * 40] if second_valid else [])
    assert result['diagnostics']


@pytest.mark.parametrize('reply', [{'status': 'cancelled'}, {'status': 'uncertain'},
                                 {'status': 'failed', 'remote_outcome': 'unknown'}])
def test_uncertain_reply_is_not_downgraded_by_missing_copied_data(reply):
    calls = []
    def run(args, **kwargs):
        if args[1] == 'ps':
            return SimpleNamespace(stdout='aaaa\nbbbb\n')
        if args[1] == 'inspect':
            return SimpleNamespace(stdout=json.dumps([container(), container('e')]))
        calls.append(args[3])
        return SimpleNamespace(returncode=0, stdout=json.dumps(reply), stderr='')
    assert export_existing_objects(request(), run) == reply
    assert calls == ['a' * 64]


def test_container_timeout_is_an_unknown_host_reply():
    def run(args, **kwargs):
        if args[1] == 'ps':
            return SimpleNamespace(stdout='aaaa\n')
        if args[1] == 'inspect':
            return SimpleNamespace(stdout=json.dumps([container()]))
        return SimpleNamespace(returncode=0, stdout=json.dumps({'status': 'uncertain', 'reason': 'git timeout'}))
    assert export_existing_objects(request(), run)['status'] == 'uncertain'


def test_local_snapshot_hints_are_bounded_and_exclude_other_repo_refs(monkeypatch):
    lines = ['f' * 40 + ' refs/mindie/inputs/current/project',
             'e' * 40 + ' refs/mindie/inputs/current/project-scm']
    for index in range(100):
        oid = f'{index:040x}'
        lines.extend([oid + f' refs/mindie/inputs/source-{index}/project',
                      oid + f' refs/parity-transport/owner/{index}/project'])
    monkeypatch.setattr(parity, 'git', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout='\n'.join(lines)))
    hints = parity._local_snapshot_bases(SimpleNamespace(source_path='unused', repo_id='project', commit='f' * 40))
    assert len(hints) == len(set(hints)) == 64
    assert 'f' * 40 not in hints and 'e' * 40 not in hints


@pytest.mark.parametrize('failure', [PermissionError('unreadable local refs'),
                                     subprocess.TimeoutExpired('for-each-ref', 10)])
def test_optional_local_hints_do_not_gate_fixed_inputs(monkeypatch, failure):
    def unavailable(*args, **kwargs):
        raise failure
    monkeypatch.setattr(parity, 'git', unavailable)
    assert parity._local_snapshot_bases(SimpleNamespace(source_path='unused', repo_id='project', commit='f' * 40)) == []


@pytest.mark.parametrize('reply, error', [
    ({'status': 'cancelled'}, PreparationCancelled),
    ({'status': 'failed', 'remote_outcome': 'unknown'}, PreparationUncertain),
    ({'status': 'failed', 'exit_code': 1, 'stderr_tail': 'bad object'}, parity.ParityUnavailable),
    ({'status': 'timeout'}, PreparationUncertain),
    ({'status': 'copied', 'remote_outcome': 'unknown'}, PreparationUncertain),
])
def test_transport_failure_never_becomes_cache_miss(monkeypatch, reply, error):
    import remote_dev.core.ssh_transport as transport
    monkeypatch.setattr(transport, 'run_remote_python', lambda *args, **kwargs: reply)
    with pytest.raises(error):
        parity._export_existing_source_objects({'host': 'fixture', 'user': 'test', 'port': 22},
                                               request()['records'], '/private-cache')
