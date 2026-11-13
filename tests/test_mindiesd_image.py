"""MindIE-SD images have their own machine and readiness contracts."""

import base64
import json
import os
import shlex
import subprocess
import sys
import zlib
from unittest.mock import Mock

import pytest

from mindie_coordinator.provision import host_ops
from mindie_coordinator import provision
from mindie_coordinator.provision.image_resolver import (
    MachineManagementError,
    image_family,
    infer_machine_type_from_image,
    resolve_image_request,
)


@pytest.mark.parametrize(('tag', 'machine'), [
    ('v3.1.0-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12', 'A2'),
    ('v3.1.0-A2-ubuntu22.04-py3.11', 'A2'),
    ('v3.1.0-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12', 'A3'),
    ('v3.1.0-cann9.1.0-torch_npu2.10.0.post4-950-ubuntu22.04-py3.12', 'A5'),
    ('v3.1.0-A5-ubuntu22.04-py3.11', 'A5'),
])
def test_mindiesd_explicit_image_keeps_its_tag_and_validates_machine(tag, machine):
    ref = 'swr.cn-south-1.myhuaweicloud.com/ascendhub/mindiesd:' + tag
    assert image_family(ref) == 'mindiesd'
    assert infer_machine_type_from_image(ref) == machine
    assert resolve_image_request(ref, machine_type=machine).candidates == (ref,)
    other = 'A3' if machine != 'A3' else 'A2'
    with pytest.raises(MachineManagementError, match='appears to target'):
        resolve_image_request(ref, machine_type=other)


def test_mindiesd_unknown_tag_requires_machine_override():
    for ref in (
        'registry.example/mindiesd:custom-build',
        'registry.example/mindiesd:git-a3b4c5d',
        'registry.example/mindie-sd:git-a5f0123',
    ):
        assert image_family(ref) == 'mindiesd'
        assert infer_machine_type_from_image(ref) is None
        with pytest.raises(MachineManagementError, match='machine type'):
            resolve_image_request(ref)
        assert resolve_image_request(ref, machine_type='A2').candidates == (ref,)


def test_mindiesd_tag_plus_digest_retains_inferred_machine_type():
    ref = (
        'swr.cn-south-1.myhuaweicloud.com/ascendhub/mindiesd:'
        'v3.1.0-A5-ubuntu22.04-py3.11@sha256:' + 'a' * 64
    )
    assert host_ops.image_family(ref) == 'mindiesd'
    assert infer_machine_type_from_image(ref) == 'A5'
    with pytest.raises(MachineManagementError, match='appears to target A5'):
        resolve_image_request(ref, machine_type='A3')


def test_unrelated_explicit_image_keeps_existing_behavior():
    ref = 'registry.example/custom-image:v1'
    assert image_family(ref) is None
    assert resolve_image_request(ref, machine_type='A3').candidates == (ref,)


@pytest.mark.parametrize('family', ['mindiesd', 'mindie'])
def test_mindiesd_metadata_readiness_checks_package_without_native_import(tmp_path, family):
    payload = host_ops.render_smoke_script(device_test=False, image_family=family)
    code = payload.split('"$PYTHON_BIN" - <<\'PY\' "$PYTHON_BIN" "$sourced_file_log"\n', 1)[1].split('\nPY\n', 1)[0]
    (tmp_path / 'mindiesd.py').write_text("raise RuntimeError('native import before lease')\n")
    for name in ('torch', 'torch-npu', 'mindiesd'):
        directory = tmp_path / (name.replace('-', '_') + '-1.0.dist-info')
        directory.mkdir()
        (directory / 'METADATA').write_text(f'Name: {name}\nVersion: 1.0\n')
    prelude = "import sys; sys.path[:] = [p for p in sys.path if 'site-packages' not in p]; sys.path.insert(0, " + repr(str(tmp_path)) + ")\n"
    result = subprocess.run([sys.executable, '-c', prelude + code, sys.executable, ''],
                            capture_output=True, text=True, env={**os.environ, 'PYTHONNOUSERSITE': '1'})
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.split(host_ops.SENTINEL, 1)[1])
    assert report['mindiesd_version'] == '1.0'


def test_mindiesd_readiness_rejects_missing_package(tmp_path):
    payload = host_ops.render_smoke_script(device_test=False, image_family='mindiesd')
    code = payload.split('"$PYTHON_BIN" - <<\'PY\' "$PYTHON_BIN" "$sourced_file_log"\n', 1)[1].split('\nPY\n', 1)[0]
    for name in ('torch', 'torch-npu'):
        directory = tmp_path / (name.replace('-', '_') + '-1.0.dist-info')
        directory.mkdir()
        (directory / 'METADATA').write_text(f'Name: {name}\nVersion: 1.0\n')
    prelude = "import sys; sys.path[:] = [p for p in sys.path if 'site-packages' not in p]; sys.path.insert(0, " + repr(str(tmp_path)) + ")\n"
    result = subprocess.run([sys.executable, '-c', prelude + code, sys.executable, ''],
                            capture_output=True, text=True, env={**os.environ, 'PYTHONNOUSERSITE': '1'})
    assert result.returncode == 3
    report = json.loads(result.stdout.split(host_ops.SENTINEL, 1)[1])
    assert report['success'] is False
    assert 'mindiesd' in report['error']


@pytest.mark.parametrize('repository', ['mindiesd', 'mindie'])
def test_mindiesd_host_machine_mismatch_stops_before_container_creation(monkeypatch, repository):
    ref = f'swr.cn-south-1.myhuaweicloud.com/ascendhub/{repository}:v3.1.0-A5-ubuntu22.04-py3.11'
    remote = Mock(return_value=host_ops.RemoteResult(
        host_ops.SshTarget('192.0.2.5'), 0, '', '', {'success': True, 'machine_type': 'A3', 'suggested_port': 46001}))
    monkeypatch.setattr(host_ops, 'run_remote_script', remote)
    with pytest.raises(MachineManagementError, match='A5.*A3'):
        provision.provision_user_container(host='192.0.2.5', image=ref, user='alice',
                                           reserve_port=Mock())
    assert remote.call_count == 1


@pytest.mark.parametrize('repository', ['mindiesd', 'mindie'])
def test_mindiesd_provision_uses_package_readiness(monkeypatch, tmp_path, repository):
    ref = f'swr.cn-south-1.myhuaweicloud.com/ascendhub/{repository}:v3.1.0-A5-ubuntu22.04-py3.11'
    target = host_ops.SshTarget('192.0.2.5')
    remote = Mock(side_effect=[
        host_ops.RemoteResult(target, 0, '', '', {'success': True, 'machine_type': 'A5'}),
        host_ops.RemoteResult(target, 0, '', '', {'success': True}),
        host_ops.RemoteResult(target, 0, '', '', {'success': True}),
    ])
    monkeypatch.setattr(host_ops, 'run_remote_script', remote)
    monkeypatch.setattr(host_ops, 'ensure_key', lambda _: ('test-key', 'ssh-ed25519 test'))
    directory = Mock()
    result = provision.provision_user_container(
        host=target.host, image=ref, user='alice', machines=directory,
        reserve_port=lambda **_: {'port': 46001})
    assert result['state'] == 'ready'
    assert directory.upsert_machine.call_args.args[0]['host']['machine_type'] == 'A5'
    # Exercise the dispatched readiness payload after transport decoding.
    # Torch metadata alone cannot satisfy a MindIE-SD image's readiness.
    script = remote.call_args_list[2].args[1]
    encoded = shlex.split(script.rsplit(' "$@"', 1)[0])[-1]
    decoded = zlib.decompress(base64.b85decode(encoded)).decode('utf-8')
    code = decoded.split('"$PYTHON_BIN" - <<\'PY\' "$PYTHON_BIN" "$sourced_file_log"\n', 1)[1].split('\nPY\n', 1)[0]
    for name in ('torch', 'torch-npu'):
        metadata = tmp_path / (name.replace('-', '_') + '-1.0.dist-info')
        metadata.mkdir()
        (metadata / 'METADATA').write_text(f'Name: {name}\nVersion: 1.0\n', encoding='utf-8', newline='\n')
    prelude = "import sys; sys.path[:] = [p for p in sys.path if 'site-packages' not in p]; sys.path.insert(0, " + repr(str(tmp_path)) + ")\n"
    observed = subprocess.run([sys.executable, '-I', '-c', prelude + code, sys.executable, ''],
                              capture_output=True, text=True, timeout=5,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    assert observed.returncode == (3 if repository == 'mindiesd' else 0)
    report = json.loads(observed.stdout.split(host_ops.SENTINEL, 1)[1])
    assert report['success'] is (repository == 'mindie')
    if repository == 'mindiesd':
        assert 'mindiesd' in report['error']


@pytest.mark.parametrize('repository', ['mindiesd', 'mindie'])
def test_mindiesd_fast_path_reuse_preserves_inferred_machine_and_rejects_label_mismatch(monkeypatch, repository):
    from mindie_coordinator.provision import existing_container

    ref = f'swr.cn-south-1.myhuaweicloud.com/ascendhub/{repository}:v3.1.0-A5-ubuntu22.04-py3.11'
    monkeypatch.setattr(existing_container, 'observe_existing', lambda *a, **k: {'status': 'match'})
    monkeypatch.setattr(host_ops, 'run_remote_script', Mock(side_effect=AssertionError('fast match must not probe')))
    directory = Mock()
    result = provision.provision_user_container(
        host='192.0.2.5', image=ref, user='alice', ssh_port=2201,
        machines=directory, reserve_port=lambda **_: {'port': 2201},
    )
    assert result['state'] == 'ready'
    assert directory.upsert_machine.call_args.args[0]['host']['machine_type'] == 'A5'

    # Verify inspect_existing rejects an existing container whose com.mindie.machine_type mismatches
    req = {
        'container': 'mindie-alice',
        'user': 'alice',
        'ssh_port': 2201,
        'workdir': provision.DEFAULT_WORKDIR,
        'image_request': {**host_ops.image_request_payload(ref), 'machine_type': 'A5'},
    }
    info = {
        'Name': '/mindie-alice',
        'Id': 'a' * 64,
        'Image': 'sha256:' + 'b' * 64,
        'State': {'Running': True, 'Paused': False, 'Restarting': False, 'Pid': 101},
        'HostConfig': {'NetworkMode': 'host', 'Privileged': True},
        'Config': {
            'Image': ref,
            'Labels': {
                'com.mindie.managed': 'true',
                'com.mindie.namespace': 'alice',
                'com.mindie.workdir': provision.DEFAULT_WORKDIR,
                'com.mindie.container_ssh_port': '2201',
                'com.mindie.machine_type': 'A3',
            },
        },
    }
    inspected = existing_container.inspect_existing(
        req,
        run=Mock(return_value=Mock(stdout=json.dumps([info]))),
    )
    assert inspected['status'] == 'mismatch'
    assert 'A3' in inspected['reason'] and 'A5' in inspected['reason']


def test_bootstrap_script_handles_mindiesd_tags_and_optional_a5_devmm_svm():
    script = host_ops.render_bootstrap_host_script()
    assert '*-950-*|*-950)' in script
    assert '*-910b-*|*-910b|*-a2-*|*-a2)' in script
    assert 'if [ -e /dev/devmm_svm ]; then' in script
    assert 'device_args+=(--device=/dev/devmm_svm)' in script


@pytest.mark.parametrize('prefix', ['', 'registry.example:5000/ascendhub/'])
@pytest.mark.parametrize('digest', ['', '@sha256:' + 'a' * 64])
def test_legacy_mindie_image_validates_machine_and_preserves_reference(prefix, digest):
    ref = prefix + 'mindie:3.1.0_torch2.9.0-A5-py3.11-ubuntu24.04-x86_64' + digest
    assert image_family(ref) == 'mindie'
    assert infer_machine_type_from_image(ref) == 'A5'
    assert resolve_image_request(ref).candidates == (ref,)
    with pytest.raises(MachineManagementError, match='appears to target A5'):
        resolve_image_request(ref, machine_type='A3')


@pytest.mark.parametrize('repository', ['mindie-tools', 'not-mindie', 'custom-image'])
def test_mindie_candidate_matching_does_not_match_substrings(repository):
    assert image_family(f'registry.example/{repository}:mindie-A5') is None


def test_legacy_mindie_unknown_tag_requires_machine_override():
    ref = 'registry.example/mindie:custom-build'
    with pytest.raises(MachineManagementError, match='machine type'):
        resolve_image_request(ref)
    assert resolve_image_request(ref, machine_type='A5').candidates == (ref,)


def test_legacy_mindie_without_sd_keeps_generic_metadata_readiness(tmp_path):
    payload = host_ops.render_smoke_script(device_test=False, image_family='mindie')
    code = payload.split('"$PYTHON_BIN" - <<\'PY\' "$PYTHON_BIN" "$sourced_file_log"\n', 1)[1].split('\nPY\n', 1)[0]
    for name in ('torch', 'torch-npu'):
        directory = tmp_path / (name.replace('-', '_') + '-1.0.dist-info')
        directory.mkdir()
        (directory / 'METADATA').write_text(f'Name: {name}\nVersion: 1.0\n', newline='\n')
    # A native module alone is not package metadata and must never be imported here.
    (tmp_path / 'mindiesd.py').write_text("raise RuntimeError('native import before lease')\n", newline='\n')
    prelude = "import sys; sys.path[:] = [p for p in sys.path if 'site-packages' not in p]; sys.path.insert(0, " + repr(str(tmp_path)) + ")\n"
    result = subprocess.run([sys.executable, '-c', prelude + code, sys.executable, ''],
                            capture_output=True, text=True, env={**os.environ, 'PYTHONNOUSERSITE': '1'})
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.split(host_ops.SENTINEL, 1)[1])
    assert report['success'] is True
    assert 'mindiesd_version' not in report


@pytest.mark.parametrize(('family', 'sd_metadata', 'broken_import'), [
    ('mindie', False, True),
    ('mindie', True, False),
    ('mindie', True, True),
    ('mindiesd', False, False),
])
def test_device_smoke_imports_only_confirmed_sd_and_reports_import_failures(
    tmp_path, family, sd_metadata, broken_import,
):
    payload = host_ops.render_smoke_script(device_test=True, image_family=family)
    code = payload.split('"$PYTHON_BIN" - <<\'PY\' "$PYTHON_BIN" "$sourced_file_log"\n', 1)[1].split('\nPY\n', 1)[0]
    (tmp_path / 'torch.py').write_text(
        "__version__ = '1.0'\n"
        "class Tensor:\n"
        "    device = 'npu:0'\n"
        "    shape = (1, 2)\n"
        "    def npu(self): return self\n"
        "def zeros(*args): return Tensor()\n"
        "class npu:\n"
        "    @staticmethod\n"
        "    def device_count(): return 1\n", newline='\n')
    (tmp_path / 'torch_npu.py').write_text('', newline='\n')
    sd_code = "raise RuntimeError('broken SD native import')\n" if broken_import else "__version__ = 'source'\n"
    (tmp_path / 'mindiesd.py').write_text(sd_code, newline='\n')
    if sd_metadata:
        directory = tmp_path / 'mindiesd-1.0.dist-info'
        directory.mkdir()
        (directory / 'METADATA').write_text('Name: mindiesd\nVersion: 1.0\n', newline='\n')
    prelude = "import sys; sys.path[:] = [p for p in sys.path if 'site-packages' not in p]; sys.path.insert(0, " + repr(str(tmp_path)) + ")\n"
    result = subprocess.run([sys.executable, '-c', prelude + code, sys.executable, ''],
                            capture_output=True, text=True, env={**os.environ, 'PYTHONNOUSERSITE': '1'})
    report = json.loads(result.stdout.split(host_ops.SENTINEL, 1)[1])
    should_fail = sd_metadata and broken_import
    assert result.returncode == (3 if should_fail else 0), result.stderr
    assert report['success'] is not should_fail
    if should_fail:
        assert report['error'] == 'broken SD native import'
    elif family == 'mindiesd' or sd_metadata:
        assert report['mindiesd_version'] == 'source'
    else:
        assert 'mindiesd_version' not in report
