"""Tier 0 arithmetic/kernel captures and Tier 1 grant/build-environment contracts."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mindie_coordinator.compile_environment import BUILD_JOB_VARIABLES, compile_environment
from mindie_coordinator.host import mindie_npu_coordination as authority
from mindie_coordinator.host import mindie_npu_coordination_resources as resources
from mindie_coordinator.placement import normalize_resources, role_plan, hardware_shortfall


@pytest.mark.parametrize('slots,cpu,memory,peak,jobs', [
    (4, 64, 32768, 2048, 4), (4, 12, 131072, 2048, 3),
    (1, 16, 2047, 2048, 0), (4, 16, 4096, 2048, 0),
    (4, 3, 32768, 2048, 0), (4, 16, None, 2048, 0),
])
def test_product_budget_divides_both_observed_resources(slots, cpu, memory, peak, jobs):
    budget = resources.compile_slot_budget(slots, cpu, memory, peak)
    assert budget['jobs_per_slot'] == jobs
    assert slots * jobs <= cpu
    if memory is not None:
        assert slots * jobs <= memory // peak


def kernel_capture(monkeypatch, *, available='MemAvailable: 10240000 kB\n', quota='250000 100000'):
    files = {
        '/proc/self/cgroup': '0::/tenant/compiler\n',
        '/proc/self/mountinfo': '29 23 0:26 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n',
        '/proc/meminfo': available,
        '/sys/fs/cgroup/cpu.max': 'max 100000',
        '/sys/fs/cgroup/tenant/cpu.max': '800000 100000',
        '/sys/fs/cgroup/tenant/compiler/cpu.max': quota,
        '/sys/fs/cgroup/memory.max': 'max',
        '/sys/fs/cgroup/tenant/memory.max': str(16 * 1024**3),
        '/sys/fs/cgroup/tenant/memory.current': str(13 * 1024**3),
        '/sys/fs/cgroup/tenant/compiler/memory.max': str(6 * 1024**3),
        '/sys/fs/cgroup/tenant/compiler/memory.current': str(2 * 1024**3),
    }
    def read(path, *args, **kwargs):
        key = path.as_posix()
        if key not in files:
            raise FileNotFoundError(path)
        return files[key]
    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(os, 'sched_getaffinity', lambda _: set(range(32)), raising=False)
    return files


def test_nested_cgroups_bound_host_memory_and_affinity(monkeypatch):
    files = kernel_capture(monkeypatch)
    assert resources._cgroup_cpu_cores() == 2
    assert resources._available_memory_mb() == 3072
    files['/proc/meminfo'] = 'MemAvailable: 1048576 kB\n'
    assert resources._available_memory_mb() == 1024
    files['/sys/fs/cgroup/tenant/compiler/memory.current'] = str(7 * 1024**3)
    assert resources._available_memory_mb() == 0


def test_subcore_cpu_and_missing_host_memory_never_force_one_worker(monkeypatch):
    kernel_capture(monkeypatch, available='MemTotal: 99999999 kB\n', quota='50000 100000')
    assert resources._cgroup_cpu_cores() == 0
    assert resources._available_memory_mb() is None
    assert resources.host_worker_cap() == 0


def test_kernel_probe_timeout_remains_visible(monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutError('kernel read deadline')
    monkeypatch.setattr(Path, 'read_text', timeout)
    with pytest.raises(TimeoutError, match='kernel read deadline'):
        resources._available_memory_mb()
    with pytest.raises(TimeoutError, match='kernel read deadline'):
        resources._cgroup_cpu_cores()


def test_grant_persists_share_and_distinguishes_budget_shortfall_from_full_slots(tmp_path, monkeypatch):
    memory = [32768]
    monkeypatch.setattr(authority, '_available_memory_mb', lambda: memory[0])
    monkeypatch.setattr(authority, '_cgroup_cpu_cores', lambda: 64)
    queue = authority.NpuCoordinator(tmp_path, max_compile_slots=4)
    for index in range(5):
        queue.submit({'task_id': f'compile-{index}', 'agent_id': 'owner', 'npu_count': 0,
                      'per_worker_memory_mb': 2048})
    first = queue.acquire('compile-0', None)['task']
    assert first['compile_budget']['jobs_per_slot'] == 4
    memory[0] = 8191  # Four configured shares cannot each fit one compiler.
    waiting = queue.acquire('compile-1', None)
    assert waiting['reason'] == 'compile_budget_insufficient'
    assert waiting['task']['fence_token'] is None
    assert queue.snapshot(None, task_id='compile-0')['tasks'][0]['compile_budget'] == first['compile_budget']
    memory[0] = 32768
    for index in range(1, 4):
        assert queue.acquire(f'compile-{index}', None)['task']['compile_budget'] == first['compile_budget']
    assert queue.acquire('compile-4', None)['reason'] == 'compile_slots_full'


def a3_capture():
    return authority.parse_npu_smi_info((Path(__file__).parent / 'fixtures/a3-192-168-13-165-idle.txt').read_text())


def test_per_die_gate_uses_actual_a3_phy_id_measurements_and_rechecks_preflight(tmp_path):
    observed = a3_capture()
    assert observed['hbm']['0'] == {'used_mb': 3127, 'total_mb': 65536}
    assert observed['hbm']['1'] == {'used_mb': 2891, 'total_mb': 65536}
    queue = authority.NpuCoordinator(tmp_path)
    queue.submit({'task_id': 'die-request', 'agent_id': 'owner', 'npu_count': 1, 'per_die_hbm_mb': 62500})
    grant = queue.acquire('die-request', observed)
    assert grant['task']['granted_devices'] == [1]
    assert grant['task']['per_die_hbm_mb'] == 62500
    stale = copy.deepcopy(observed)
    stale['hbm']['1']['used_mb'] = 3037  # The same idle Die now has 62499 MiB.
    result = queue.preflight('die-request', grant['task']['fence_token'], stale)
    assert result['status'] == 'waiting' and result['task']['state'] == 'queued'
    assert result['task']['granted_devices'] == []


def test_aggregate_hbm_and_missing_observation_cannot_authorize_one_die(tmp_path):
    observed = a3_capture()
    assert sum(row['total_mb'] - row['used_mb'] for row in observed['hbm'].values()) > 65536
    queue = authority.NpuCoordinator(tmp_path)
    queue.submit({'task_id': 'too-large', 'agent_id': 'owner', 'npu_count': 1, 'per_die_hbm_mb': 65536})
    assert queue.acquire('too-large', observed)['status'] == 'waiting'
    queue.cancel('too-large', observed)
    queue.submit({'task_id': 'unobserved', 'agent_id': 'owner', 'devices': [1], 'per_die_hbm_mb': 1})
    observed['hbm']['1']['vram_free_observed'] = False
    assert queue.acquire('unobserved', observed)['status'] == 'waiting'
    observed['hbm']['1'].pop('vram_free_observed')
    observed['hbm']['1'].pop('used_mb')
    assert queue.acquire('unobserved', observed)['status'] == 'waiting'
    assert queue.snapshot(None, task_id='unobserved')['tasks'][0]['fence_token'] is None


def test_per_die_constraint_survives_role_normalization_and_candidate_screening():
    normalized = normalize_resources({'npu_count': 2, 'per_die_hbm_mb': 62500})
    roles = role_plan(None, normalized, 'run')
    assert roles[0]['per_die_hbm_mb'] == 62500
    gap = hardware_shortfall({'host': 'host'}, roles[0], {'host': {'vram_free_mb': 100000}})
    assert gap['gaps']['hbm_mb'] == {'required': 125000, 'available': 100000}


def test_compiler_environment_retains_smaller_values_and_explicitly_handles_legacy():
    budget = resources.compile_slot_budget(4, 64, 65536, 2048)
    env = compile_environment(budget, {'MAX_JOBS': '2', 'MINDIE_BUILD_JOBS': '99'})
    assert all(env[key] == '2' for key in BUILD_JOB_VARIABLES)
    assert env['MINDIE_COMPILE_JOBS_LIMIT'] == '2'
    assert compile_environment({'status': 'unavailable'}, {'MAX_JOBS': '3'}) == {'MAX_JOBS': '3'}


def test_captured_build_environment_cannot_override_granted_or_smaller_jobs(tmp_path, monkeypatch):
    from mindie_coordinator import parity_install
    runtime = tmp_path / 'runtime'
    (runtime / '.mindie-runtime').mkdir(parents=True)
    (runtime / 'vllm-ascend').mkdir()
    (runtime / 'vllm-ascend/CMakeLists.txt').write_text('VERSION_EQUAL "1.2.3"\n')
    (runtime / 'vllm-ascend/torch.py').write_text('__version__ = "1.2.3"\n')
    (runtime / 'vllm-ascend/sitecustomize.py').write_text(
        'import sys\nsys.stdout.reconfigure(newline="\\n")\n')
    (runtime / '.mindie-runtime/build-source.json').write_text(json.dumps({
        'versions': {'vllm-ascend': {'version': '1.0'}},
        'build_env': {name: '99' for name in BUILD_JOB_VARIABLES}}))
    monkeypatch.setattr(parity_install, 'DEFAULT_ENV_PREAMBLE', [
        f'export PYTHON={parity_install.quoted(Path(sys.executable).as_posix())}',
        f'export PYTHONPATH={parity_install.quoted((runtime / "vllm-ascend").as_posix())}',
    ])
    for name in BUILD_JOB_VARIABLES:
        monkeypatch.setenv(name, '100')  # Captured author inputs are also bounded.
    script = parity_install.runtime_install_step_script(runtime_root=runtime.as_posix(), marker_dirname='.markers',
        container_identity='test-container', step='check-build-compat')
    script += '\nprintf "%s,%s,%s\\n" "$MAX_JOBS" "$MINDIE_BUILD_JOBS" "$CMAKE_BUILD_PARALLEL_LEVEL"\n'
    env = compile_environment({'status': 'granted', 'jobs_per_slot': 8}, {'MAX_JOBS': '2'})
    bash = (str(Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Git/bin/bash.exe')
            if os.name == 'nt' else 'bash')
    result = subprocess.run([bash, '-c', script], text=True, capture_output=True,
                            env={**os.environ, **env}, timeout=5)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == '2,2,2'


def test_windows_memory_probe_uses_observed_physical_headroom(monkeypatch):
    import ctypes
    from types import SimpleNamespace
    def missing(*args, **kwargs):
        raise FileNotFoundError('no proc files')
    def no_sysconf(*args):
        raise AttributeError('no sysconf')
    def memory_status(pointer):
        status = pointer._obj
        assert status.length == 64
        status.total_phys = 64 * 1024**3
        status.available_phys = 3 * 1024**3
        return 1
    monkeypatch.setattr(Path, 'read_text', missing)
    monkeypatch.setattr(os, 'sysconf', no_sysconf, raising=False)
    monkeypatch.setattr(ctypes, 'windll', SimpleNamespace(kernel32=SimpleNamespace(
        GlobalMemoryStatusEx=memory_status)), raising=False)
    assert resources._available_memory_mb() == 3072
