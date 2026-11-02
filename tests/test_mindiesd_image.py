"""MindIE-SD images have their own machine and readiness contracts."""

import json
import os
import subprocess
import sys
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
    ref = 'registry.example/mindiesd:custom-build'
    with pytest.raises(MachineManagementError, match='machine type'):
        resolve_image_request(ref)
    assert resolve_image_request(ref, machine_type='A2').candidates == (ref,)


def test_unrelated_explicit_image_keeps_existing_behavior():
    ref = 'registry.example/custom-image:v1'
    assert image_family(ref) is None
    assert resolve_image_request(ref, machine_type='A3').candidates == (ref,)


def test_mindiesd_metadata_readiness_checks_package_without_native_import(tmp_path):
    payload = host_ops.render_smoke_script(device_test=False, image_family='mindiesd')
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


def test_mindiesd_host_machine_mismatch_stops_before_container_creation(monkeypatch):
    ref = 'swr.cn-south-1.myhuaweicloud.com/ascendhub/mindiesd:v3.1.0-A5-ubuntu22.04-py3.11'
    remote = Mock(return_value=host_ops.RemoteResult(
        host_ops.SshTarget('192.0.2.5'), 0, '', '', {'success': True, 'machine_type': 'A3', 'suggested_port': 46001}))
    monkeypatch.setattr(host_ops, 'run_remote_script', remote)
    with pytest.raises(MachineManagementError, match='A5.*A3'):
        provision.provision_user_container(host='192.0.2.5', image=ref, user='alice',
                                           reserve_port=Mock())
    assert remote.call_count == 1


def test_mindiesd_provision_uses_package_readiness(monkeypatch):
    ref = 'swr.cn-south-1.myhuaweicloud.com/ascendhub/mindiesd:v3.1.0-A5-ubuntu22.04-py3.11'
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
    assert "importlib.metadata.version('mindiesd')" in remote.call_args_list[2].args[1]
