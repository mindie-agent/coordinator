import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig

import pytest

from mindie_coordinator.parity import ATB_CXX_ABI_PROBE, DEFAULT_ENV_PREAMBLE


@pytest.fixture
def recorded_abi(tmp_path):
    metadata = tmp_path / 'image/torch-1.0.dist-info'
    metadata.mkdir(parents=True)
    (metadata / 'METADATA').write_text('Name: torch\nVersion: 1.0\n')
    (metadata.parent / 'torch.py').write_text('raise RuntimeError("must not import torch to source ATB")\n')
    receipt = tmp_path / '.mindie-runtime/reuse.json'
    receipt.parent.mkdir()
    receipt.write_text(json.dumps({'atb_abi': {'cxx_abi': '1', 'torch': '1.0',
                                             'python_abi': sysconfig.get_config_var('SOABI')}}))
    env = {**os.environ, 'PYTHONPATH': str(metadata.parent)}
    return receipt, env


@pytest.mark.parametrize('change', [None, 'torch', 'python_abi', 'cxx_abi', 'missing', 'broken'])
def test_atb_abi_requires_verified_matching_runtime_metadata(recorded_abi, change):
    receipt, env = recorded_abi
    if change in ('torch', 'python_abi', 'cxx_abi'):
        row = json.loads(receipt.read_text())
        row['atb_abi'][change] = 'unknown'
        receipt.write_text(json.dumps(row))
    elif change == 'missing':
        receipt.unlink()
    elif change == 'broken':
        receipt.write_text('partial')
    result = subprocess.run([sys.executable, '-c', ATB_CXX_ABI_PROBE, str(receipt)],
                            env=env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == ('1' if change is None else '')


@pytest.mark.skipif(sys.platform != 'linux', reason='ATB activation is a Linux shell contract')
@pytest.mark.parametrize('known', [True, False])
def test_atb_setup_receives_supported_argument_or_keeps_native_detection(recorded_abi, tmp_path, known):
    receipt, env = recorded_abi
    if not known:
        receipt.write_text('{}')
    setup = tmp_path / 'atb/set_env.sh'
    setup.parent.mkdir()
    recorded = tmp_path / 'arguments'
    setup.write_text('printf "%s" "$*" > ' + shlex.quote(str(recorded)) + '\n')
    # Both the fixed allowlisted ATB path and loop member map to the fixture.
    script = '\n'.join(DEFAULT_ENV_PREAMBLE).replace('/usr/local/Ascend/nnal/atb/set_env.sh', str(setup))
    subprocess.run(['bash', '-eu', '-c', script], capture_output=True, text=True, check=True,
                   env={**env, 'MINDIE_RUNTIME_ROOT': str(tmp_path),
                        'PATH': str(Path(sys.executable).parent) + ':' + env['PATH']}, timeout=20)
    # Bash source without arguments keeps safe_source's original $1 (the
    # filename); ATB ignores it and performs its original ABI detection.
    assert recorded.read_text() == ('--cxx_abi=1' if known else str(setup))
