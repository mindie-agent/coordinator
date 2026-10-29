"""Observed execution headroom and immutable per-slot grants.

[Role] Read kernel capacity and calculate CPU/RAM shares and per-Die HBM gates.
[Invariants] Missing memory is not capacity; slots divide one host budget once.
[Timeouts & Locks] Local bounded kernel files only; admission holds the existing DB lock.
[Side-Effect Policy] Read-only observations and pure arithmetic; no allocation ledger.
"""
from __future__ import annotations

import os
from pathlib import Path

COMPILE_WORKER_MEMORY_MB = 2048


def _cgroup_directories():
    """Resolve this process's v2 group and ancestors, including delegated mounts."""
    try:
        membership = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines()
                          if line.startswith('0::'))
        mounts = Path('/proc/self/mountinfo').read_text().splitlines()
    except TimeoutError:
        raise
    except (FileNotFoundError, StopIteration):
        return []
    for line in mounts:
        before, _, after = line.partition(' - ')
        if after.split()[0] != 'cgroup2':
            continue
        fields = before.split()
        root, mount = Path(fields[3]), Path(fields[4])
        try:
            relative = Path(membership).relative_to(root)
        except ValueError:
            continue
        current = mount / relative
        return [current, *[parent for parent in current.parents if parent == mount or mount in parent.parents]]
    return []


def _cgroup_cpu_cores() -> int:
    affinity = getattr(os, 'sched_getaffinity', None)
    cores = len(affinity(0)) if affinity else (os.cpu_count() or 1)
    for directory in _cgroup_directories():
        try:
            quota, period = (directory / 'cpu.max').read_text().split()
        except FileNotFoundError:
            continue
        if quota != 'max':
            cores = min(cores, int(quota) // int(period))
    return cores


def _available_memory_mb() -> int | None:
    """Observed host headroom, bounded by every finite cgroup memory limit."""
    available = None
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                available = int(line.split()[1]) // 1024
                break
    except FileNotFoundError:
        # Other kernels expose page counters through sysconf or sysctl. Never
        # substitute physical total or a guessed percentage for headroom.
        try:
            available = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') // (1024 * 1024)
        except (ValueError, AttributeError):
            import ctypes
            windows = getattr(ctypes, 'windll', None)
            if windows is not None:
                class MemoryStatus(ctypes.Structure):
                    _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32),
                        *[(name, ctypes.c_uint64) for name in ('total_phys', 'available_phys',
                          'total_pagefile', 'available_pagefile', 'total_virtual',
                          'available_virtual', 'available_extended_virtual')]]
                status = MemoryStatus(length=ctypes.sizeof(MemoryStatus))
                if not windows.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                    raise OSError('GlobalMemoryStatusEx failed')
                available = status.available_phys // (1024 * 1024)
            else:
                libc = ctypes.CDLL(None)
                if hasattr(libc, 'host_statistics') and hasattr(libc, 'mach_host_self'):
                    # HOST_VM_INFO's first counters are free/active/inactive/wired;
                    # free already includes speculative pages (Mach vm_statistics.h).
                    counters = (ctypes.c_uint32 * 15)()
                    count = ctypes.c_uint32(15)
                    host = libc.mach_host_self()
                    try:
                        if libc.host_statistics(host, 2, counters, ctypes.byref(count)) != 0:
                            return None
                        available = (counters[0] + counters[2]) * os.sysconf('SC_PAGE_SIZE') // (1024 * 1024)
                    finally:
                        libc.mach_port_deallocate(ctypes.c_uint32.in_dll(libc, 'mach_task_self_').value, host)
    for directory in _cgroup_directories():
        try:
            maximum = (directory / 'memory.max').read_text().strip()
        except FileNotFoundError:
            continue
        if maximum != 'max':
            current = int((directory / 'memory.current').read_text())
            remaining = max(0, int(maximum) - current) // (1024 * 1024)
            if available is not None:
                available = min(available, remaining)
    return available


def compile_slot_budget(slots, cpu_cores, usable_memory_mb, per_worker_memory_mb):
    """Static shares preserve capacity for every configured concurrent slot."""
    memory_per_slot = None if usable_memory_mb is None else usable_memory_mb // slots
    cpu_per_slot = cpu_cores // slots
    jobs = 0 if memory_per_slot is None else min(cpu_per_slot, memory_per_slot // per_worker_memory_mb)
    return {'slots': slots, 'jobs_per_slot': jobs, 'cpu_cores': cpu_per_slot,
            'memory_mb': memory_per_slot, 'per_worker_memory_mb': per_worker_memory_mb,
            'usable_cpu_cores': cpu_cores, 'usable_memory_mb': usable_memory_mb,
            'status': 'granted' if jobs > 0 else 'insufficient'}


def host_worker_cap() -> int:
    memory = _available_memory_mb()
    return 0 if memory is None else min(_cgroup_cpu_cores(), memory // COMPILE_WORKER_MEMORY_MB)


def hbm_eligible_devices(observed, devices, per_die_hbm_mb):
    """Only each Phy-ID's measured total minus used can satisfy its request."""
    if not per_die_hbm_mb:
        return set(devices)
    measurements = (observed or {}).get('hbm', {})
    eligible = set()
    for device in devices:
        row = measurements.get(str(device), measurements.get(device, {}))
        total = row.get('hbm_total_mb', row.get('total_mb'))
        used = row.get('hbm_used_mb', row.get('used_mb'))
        if ((observed or {}).get('vram_free_observed') is False or row.get('vram_free_observed') is False
                or type(total) is not int or type(used) is not int):
            continue
        if max(0, total - used) >= per_die_hbm_mb:
            eligible.add(device)
    return eligible
