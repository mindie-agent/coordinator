import json
import shutil
from types import SimpleNamespace

import pytest

from mindie_coordinator import provision


@pytest.fixture
def bash_executable():
    executable = shutil.which('bash')
    if not executable:
        pytest.skip('Bash is required for bootstrap behavior tests')
    return executable


def test_docker_exec_provision_uses_host_transport_without_ssh(monkeypatch):
    calls, records = [], []
    def run(target, script, **options):
        calls.append((target, script, options))
        return {'success': True, 'free_port': 2022, 'machine_type': 'A5'}
    monkeypatch.setattr(provision.host_ops, 'run_remote_script', run)
    monkeypatch.setattr(provision.host_ops, 'assert_remote_success', lambda result, **kw: result)
    monkeypatch.setattr(provision.host_ops, 'ensure_key', lambda *a: pytest.fail('container key must not be installed'))
    row = provision.provision_user_container(
        host='example', image='example/runtime:fixed', user='tester', mode='docker_exec',
        machines=SimpleNamespace(upsert_machine=records.append),
        reserve_port=lambda **kw: pytest.fail('SSH port must not be reserved'))
    assert len(calls) == 3
    assert all(target.port == 22 for target, _, _ in calls)
    assert calls[1][2]['args'][1] == '0'
    assert calls[1][2]['args'][-1] == 'docker_exec'
    assert 'docker exec -i mindie-tester bash -s --' in calls[2][1]
    assert provision.host_ops.render_smoke_script(device_test=False) in calls[2][1]
    assert records[0]['container']['mode'] == 'docker_exec'
    assert 'ssh_port' not in records[0]['container']
    assert 'ssh_port' not in row
    assert row['mode'] == 'docker_exec'


@pytest.mark.parametrize('options', [{'mode': 'fallback'}, {'mode': 'docker_exec', 'ssh_port': 2022}])
def test_invalid_mode_fails_before_remote_side_effects(monkeypatch, options):
    monkeypatch.setattr(provision.host_ops, 'run_remote_script', lambda *a, **kw: pytest.fail('unexpected SSH'))
    with pytest.raises(ValueError):
        provision.provision_user_container(host='example', image='example/runtime:fixed', user='tester', **options)


def test_docker_exec_cleanup_does_not_release_an_ssh_reservation(tmp_path, monkeypatch):
    from mindie_coordinator.env_manager import handle_container
    from mindie_coordinator.machine_directory import MachineDirectory
    store = MachineDirectory(path=tmp_path / 'machines.json')
    store.upsert_machine({'alias': 'builder', 'host': {'ip': 'example', 'user': 'root', 'port': 22},
                          'container': {'name': 'mindie-tester', 'mode': 'docker_exec', 'workdir': '/workspace'},
                          'user': 'tester'})
    monkeypatch.setattr(provision.host_ops, 'remove_container_result', lambda ns: {'success': True, 'removed': True})
    queue = SimpleNamespace(request=lambda *a: pytest.fail('no SSH reservation exists'))
    result = handle_container({'action': 'remove', 'alias': 'builder'}, directory=store, queue=queue)
    assert result['state'] == 'ready'
    assert result['reservation_release'] is None
    assert not store.get('builder').get('container')


def test_bootstrap_docker_exec_skips_package_and_firewall_commands(bash_executable):
    import subprocess
    script = provision.host_ops.render_bootstrap_host_script()
    syntax = subprocess.run([bash_executable, '-n'], input=script, text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr
    start = script.index('if [ "$mode" = "docker_exec" ]; then\n  package_bootstrap=')
    end = script.index('\npayload=$(python3', start)
    fragment = script[start:end]
    result = subprocess.run([bash_executable, '-s'], input='''set -euo pipefail
mode=docker_exec
container=mindie-tester
actions=()
docker() { echo unexpected-docker >&2; return 99; }
ufw() { echo unexpected-firewall >&2; return 99; }
firewall-cmd() { echo unexpected-firewall >&2; return 99; }
install_container_ssh_packages() { echo unexpected-install >&2; return 99; }
emit_progress() { :; }
emit_json() { echo "$1"; }
run_with_progress() { shift 3; "$@"; }
configure_container_state() { echo metadata-configured; }
''' + fragment + '\nprintf "%s\\n" "${actions[*]}" "$package_bootstrap"\n', text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ''
    assert result.stdout.splitlines() == [
        'metadata-configured', 'configured-container-env',
        '{"install_skipped": true, "reason": "docker_exec"}']


def test_cli_passes_explicit_docker_exec_mode(monkeypatch):
    from mindie_coordinator.cli import main
    captured = {}
    def prepare(**kwargs):
        captured.update(kwargs)
        return {'state': 'ready'}
    monkeypatch.setattr(provision, 'provision_user_container', prepare)
    assert main(['provision', '--host', 'example', '--image', 'example/runtime:fixed', '--mode', 'docker_exec']) == 0
    assert captured['mode'] == 'docker_exec'


def test_docker_exec_metadata_failure_does_not_record_ready_endpoint(monkeypatch):
    records, calls = [], []
    def run(target, script, **kwargs):
        calls.append(script)
        if len(calls) == 3:
            raise RuntimeError('metadata missing')
        return {'success': True}
    monkeypatch.setattr(provision.host_ops, 'run_remote_script', run)
    monkeypatch.setattr(provision.host_ops, 'assert_remote_success', lambda result, **kw: result)
    with pytest.raises(RuntimeError, match='metadata missing'):
        provision.provision_user_container(host='example', image='example/runtime:fixed', user='tester',
                                          mode='docker_exec', machines=SimpleNamespace(upsert_machine=records.append))
    assert not records
    assert len(calls) == 3


@pytest.mark.parametrize('existing_port', ['2022', '<no value>', '0'])
def test_docker_exec_reuse_rejects_ssh_mode_before_mutation(existing_port, bash_executable):
    import subprocess
    script = provision.host_ops.render_bootstrap_host_script()
    start = script.index('container_exists=false\nif docker inspect "$container"')
    end = script.index('  current_image=', start)
    guard = script[start:end] + '\nfi\nprintf "reuse-allowed\\n"\n'
    prelude = '''set -euo pipefail
mode=docker_exec
container=mindie-tester
namespace=tester
workdir=/workspace
existing_port="$1"
emit_json() { echo "$1"; }
docker() {
  if [ "$1" != inspect ]; then echo unexpected-mutation >&2; return 99; fi
  if [ "${2:-}" != -f ]; then return 0; fi
  case "$3" in
    *com.mindie.managed*) echo true ;;
    *com.mindie.namespace*) echo tester ;;
    *com.mindie.workdir*) echo /workspace ;;
    *com.mindie.container_ssh_port*) echo "$existing_port" ;;
    *) echo unexpected-inspect >&2; return 99 ;;
  esac
}
'''
    result = subprocess.run([bash_executable, '-s', '--', existing_port], input=prelude + guard,
                            text=True, capture_output=True)
    assert result.stderr == ''
    if existing_port == '0':
        assert result.returncode == 0
        assert result.stdout.strip() == 'reuse-allowed'
    else:
        assert result.returncode == 21, result.stdout
        assert 'reuse-allowed' not in result.stdout
        assert 'container mode' in json.loads(result.stdout)['error']
