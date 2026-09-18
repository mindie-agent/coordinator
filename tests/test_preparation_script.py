import copy
import os
import shlex
import subprocess
import sys

import pytest

from mindie_coordinator.preparation_script import COMMAND_BYTES, preparation_command, preparation_command_fits


def test_large_composite_program_fits_without_splitting_jobs():
    script = '# native module\n' * 15000 + 'printf complete'
    assert len(script.encode()) > COMMAND_BYTES
    assert preparation_command_fits(script)
    assert len(preparation_command(script).encode()) < COMMAND_BYTES
    assert preparation_command('true') == 'true'
    assert not preparation_command_fits('#' * (8 * 1024 * 1024))
    assert not preparation_command_fits(os.urandom(100000).hex())


@pytest.mark.skipif(sys.platform not in ('linux', 'darwin'), reason='requires POSIX descriptor paths')
@pytest.mark.parametrize('code', [0, 17])
def test_large_program_preserves_cwd_stdin_output_and_exit(tmp_path, code):
    script = '# package program\n' * 16000 + '\nprintf "%s\\n" "$PWD"\ncat\nexit ' + str(code)
    root = tmp_path / 'fresh root'
    command = preparation_command(script, setup=[('prepare-root', 'mkdir -p ' + shlex.quote(str(root)))], cwd=str(root))
    result = subprocess.run(['bash', '-c', command], capture_output=True, text=True, timeout=10)
    assert result.returncode == code and result.stdout == str(root) + '\n'
    assert '"phase": "prepare-root"' in result.stderr and '"elapsed_seconds":' in result.stderr


@pytest.mark.skipif(sys.platform != 'linux', reason='actual owned supervisor requires Linux')
def test_first_job_bootstraps_from_existing_parent_once(tmp_path, monkeypatch):
    from remote_dev.processes.client import worker_source
    from remote_dev.processes.worker import control_job
    from mindie_coordinator.preparation_process import PreparationProcess
    root = tmp_path / 'execution'
    endpoint = {'host': 'local.invalid', 'port': 22, 'user': 'test', 'root': str(root), 'cwd': str(root)}
    saved, output = [], []
    def control(endpoint, job_id, action, **kwargs):
        assert saved and saved[-1]['job_id'] == job_id
        return control_job({'root': endpoint['root'], 'job_id': job_id, 'action': action, **kwargs}, worker_source())
    monkeypatch.setattr('mindie_coordinator.preparation_process.control', control)
    process = PreparationProcess(endpoint, 'materialize', lambda row: saved.append(copy.deepcopy(row)), lambda: False,
        setup=[('prepare-root', 'mkdir -p ' + shlex.quote(str(root)) + '; echo setup >> ' + shlex.quote(str(root / 'count')))],
        bootstrap_root=str(tmp_path))
    for _ in range(2):
        result = process.run('printf materialized', on_output=lambda *args: output.append(args))
        assert result.returncode == 0 and saved[-1]['quiet']
    assert (root / 'count').read_text().splitlines() == ['setup']
    assert saved[0]['endpoint']['root'] == str(tmp_path)
    assert saved[-1]['endpoint']['root'] == str(root)
    assert any('prepare-root' in row.get('stage_timings', {}) for row in saved)
