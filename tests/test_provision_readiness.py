"""Container preparation cannot exercise devices before any resource grant."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mindie_coordinator.provision import host_ops


def payload(device_test):
    script = host_ops.render_smoke_script(device_test=device_test)
    return script.split('"$PYTHON_BIN" - <<\'PY\' "$PYTHON_BIN" "$sourced_file_log"\n', 1)[1].split('\nPY\n', 1)[0]


@pytest.mark.parametrize('available', [True, False])
def test_metadata_readiness_never_imports_torch_or_touches_a_device(tmp_path, available):
    # Isolate metadata too: a developer machine with torch installed must not
    # hide the missing-package case in this actual remote Python payload.
    prelude = "import sys; sys.path[:] = [p for p in sys.path if 'site-packages' not in p]; sys.path.insert(0, " + repr(str(tmp_path)) + ")\n"
    (tmp_path / 'torch.py').write_text("raise RuntimeError('device library must not be imported')\n")
    (tmp_path / 'torch_npu.py').write_text("raise RuntimeError('device library must not be imported')\n")
    if available:
        for name in ('torch', 'torch-npu'):
            directory = tmp_path / (name.replace('-', '_') + '-2.10.0.dist-info')
            directory.mkdir()
            (directory / 'METADATA').write_text('Name: ' + name + '\nVersion: 2.10.0\n')
    result = subprocess.run([sys.executable, '-c', prelude + payload(False), sys.executable, ''],
                            capture_output=True, text=True, env={**os.environ, 'PYTHONNOUSERSITE': '1'})
    assert result.returncode == (0 if available else 3), result.stderr
    report = json.loads(result.stdout.split(host_ops.SENTINEL, 1)[1])
    assert report['success'] is available
    assert report['device_test'] is False
    assert 'device' not in report and 'device_count' not in report


def test_explicit_device_smoke_retains_device_test_by_default():
    assert host_ops.render_smoke_script() == host_ops.render_smoke_script(device_test=True)
    assert 'x = torch.zeros(1, 2).npu()' in payload(True)
