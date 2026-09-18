"""Isolated task work roots: sources, interpreter, installs, verified profile.

A donor container/SSH endpoint may be reused. Donor site-packages, editable
references and build artifacts are never mutated. Immutable image packages
may be visible through ``venv --system-site-packages``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from mindie_coordinator.placement import SUPPORTED_RECIPES
from mindie_coordinator.provision.host_ops import DEFAULT_WORKDIR
from mindie_coordinator.ready_runtime import safe_id
from mindie_coordinator.runtime_profile import digest

INSTALL_STEPS = (
    "check-build-compat",
    "install-vllm",
    "install-vllm-ascend-requirements",
    "install-vllm-ascend",
    "verify-deps",
)


def _safe_segment(value: str, limit: int) -> str:
    text = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in str(value))
    return text[:limit] or "x"


def isolated_root(session_id: str, role_name: str, host: str = "") -> str:
    session_safe = _safe_segment(session_id, 48)
    role_safe = safe_id(role_name)
    if host:
        host_safe = _safe_segment(str(host).replace(":", "-"), 40)
        return f"{DEFAULT_WORKDIR}/executions/{session_safe}/{host_safe}/{role_safe}"
    return f"{DEFAULT_WORKDIR}/executions/{session_safe}/{role_safe}"


def isolated_python(root: str) -> str:
    return f"{root}/.venv/bin/python"


def task_runtime_id(session_id: str, host: str, role_name: str) -> str:
    session_safe = _safe_segment(session_id, 24)
    host_safe = _safe_segment(str(host).replace(":", "-"), 40)
    role_safe = _safe_segment(role_name, 24)
    identity = digest([session_id, host, role_name])[:16]
    return safe_id(f"t{session_safe}-h{host_safe}-{role_safe}-{identity}")


def create_venv_script(root: str, python: str, donor_python: str | None = None) -> str:
    """Create a task-owned venv. Image packages may be reused; donor venv is not."""
    from mindie_coordinator.parity import PYTHON_METADATA_PREAMBLE
    from mindie_coordinator.parity_support import quoted

    donor = quoted(donor_python or "")
    return "\n".join(
        [
            "set -euo pipefail",
            f"mkdir -p {quoted(root)}",
            *PYTHON_METADATA_PREAMBLE,
            'IMAGE_PYTHON="$PYTHON"',
            'if [ -z "$IMAGE_PYTHON" ]; then echo "image python3 not found" >&2; exit 1; fi',
            f"DONOR={donor}",
            'if [ -n "$DONOR" ] && [ "$IMAGE_PYTHON" = "$DONOR" ]; then',
            '  IMAGE_PYTHON="$(ls -1d /usr/local/python*/bin/python3 2>/dev/null | sort -V | tail -n 1 || true)"',
            "fi",
            'if [ -n "$DONOR" ] && [ "$IMAGE_PYTHON" = "$DONOR" ]; then',
            '  echo "cannot create a task interpreter from the donor python" >&2',
            "  exit 1",
            "fi",
            f'"$IMAGE_PYTHON" -m venv --without-pip --system-site-packages {quoted(str(PurePosixPath(root) / ".venv"))}',
            f"test -x {quoted(python)}",
            f'test {quoted(python)} != "$DONOR"',
            # Image pip is importable through system-site-packages; its
            # installation scheme is still this new venv. Seed only if absent.
            f'if ! {quoted(python)} -c "import pip" >/dev/null 2>&1; then',
            f'  "$IMAGE_PYTHON" -m venv --system-site-packages {quoted(str(PurePosixPath(root) / ".venv"))}',
            'fi',
            f'{quoted(python)} - {quoted(str(PurePosixPath(root) / ".venv"))} <<\'MINDIE_VENV\'',
            'import pathlib, sys, sysconfig',
            'expected = pathlib.Path(sys.argv[1]).resolve()',
            'if pathlib.Path(sys.prefix).resolve() != expected or not pathlib.Path(sysconfig.get_paths()["purelib"]).resolve().is_relative_to(expected):',
            '    raise ValueError("pip installation scheme escaped the owned interpreter")',
            'MINDIE_VENV',
        ]
    )


class TaskRootBusy(RuntimeError):
    """The deterministic task root is still bound; wait instead of overwriting."""


def reusable_preparation(snapshot: dict, environment: dict, donor: dict) -> dict:
    """Known recipe inputs, independent of task, role and materialization paths."""
    records = {row['relpath']: row for row in snapshot.get('records', [])}
    profile = donor.get('profile') or (donor.get('attestation') or {}).get('profile') or {}
    base = {name: profile.get(name) for name in (
        'image_digest', 'soc', 'driver', 'cann', 'python_abi', 'torch', 'torch_npu', 'compiler', 'system_files')}
    base.update(environment=environment, build_env=snapshot.get('build_env', {}))
    dependencies = {name: row.get('build_inputs', {}).get('dependencies') for name, row in records.items()
                    if name in ('vllm', 'vllm-ascend')}
    native = {name: row.get('build_inputs', {}).get('native') for name, row in records.items()
              if name in ('vllm', 'vllm-ascend')}
    return {'source_id': snapshot['id'], 'environment': base,
            'dependencies': dependencies, 'native': native,
            'dependency_key': digest({'environment': base, 'dependencies': dependencies}),
            'native_key': digest({'environment': base, 'dependencies': dependencies, 'native': native})}


def prepare_task_environment(
    pool,
    *,
    user: str,
    session_id: str,
    role_name: str,
    environment: dict[str, Any] | None,
    donor: dict[str, Any],
    sources: dict[str, str] | None = None,
    source_snapshot: dict | None = None,
    on_progress=None,
    log_dir=None,
    on_preparation_job=None,
    cancel_requested=None,
    checkout_session=None,
    compile_scope=None,
) -> dict[str, Any]:
    """Create an isolated task root, install into a task-owned interpreter, register.

    ``sources`` must be the actual local vllm / vllm-ascend worktrees. Registration
    happens only after the backend has a verified ready-profile for this root.
    """
    environment = dict(environment or {})
    sources = dict(sources or {})
    if source_snapshot is None:
        raise ValueError("preparation requires submission's fixed source descriptor")
    native_recipe = {'vllm', 'vllm-ascend'}.issubset(sources)
    host = str(
        donor.get("host")
        or (donor.get("host_endpoint") or {}).get("host")
        or (donor.get("endpoint") or {}).get("host")
        or ""
    )
    if not host:
        raise ValueError("preparation needs a host identity")
    root = isolated_root(session_id, role_name, host)
    python = isolated_python(root)
    runtime_id = task_runtime_id(session_id, host, role_name)
    checkout_request = hashlib.sha256(f"{session_id}:{runtime_id}:{role_name}".encode()).hexdigest()
    donor_python = donor.get("python")
    if donor_python and python == donor_python:
        raise ValueError("task-owned interpreter must not be the donor interpreter")

    existing = next((item for item in pool.catalog()
                     if item.get("runtime_id") == runtime_id or item.get("root") == root), None)
    if existing:
        if existing.get('root') != root:
            raise ValueError('runtime identity resolves to a different execution root')
        if existing.get("state") in {"ready", "bound"}:
            runtime = next(row for row in _runtime_rows(pool) if row["id"] == existing["runtime_id"])
            if checkout_session:
                binding = pool.checkout(user, checkout_session, runtime['attestation']['profile_key'],
                                        checkout_request, runtime['id'])
                if binding.get('status') == 'cache_miss':
                    raise TaskRootBusy('execution root is not available for its managed binding')
                return {**runtime, 'binding': binding}
            return runtime
        raise TaskRootBusy(f"execution root {root} exists but is not verified; inspect its retained evidence")

    raw_host = donor.get("host_endpoint") or {}
    host_port = raw_host.get("port")
    if not host_port and isinstance(donor.get("host"), dict):
        host_port = donor["host"].get("port")
    host_endpoint = {
        "host": host,
        "port": int(host_port or 22),
        "user": raw_host.get("user") or "root",
    }
    ssh_port = donor.get("ssh_port") or (donor.get("endpoint") or {}).get("port")
    if not ssh_port:
        raise ValueError("user container has no reserved SSH endpoint to reuse")
    recipe = environment.get("recipe") or environment.get("image") or donor.get("recipe")
    if recipe and recipe not in SUPPORTED_RECIPES and not donor.get("recipe"):
        raise ValueError(f"unsupported environment recipe {recipe!r}")
    container_endpoint = {
        "host": host_endpoint["host"],
        "port": int(ssh_port),
        "user": (donor.get("endpoint") or {}).get("user") or "root",
        "root": root,
        "cwd": root,
    }
    spec = {
        "user": user,
        "python": python,
        "recipe": recipe,
        "host_endpoint": {
            "host": host_endpoint["host"],
            "port": int(host_endpoint.get("port") or 22),
            "user": host_endpoint.get("user") or "root",
        },
        "endpoint": container_endpoint,
        "service_ports": list(donor.get("service_ports") or []),
        "container_name": donor.get("container_name") or ("mindie-" + user),
        "machine_type": environment.get("machine_type") or donor.get("machine_type"),
        "source_snapshot": source_snapshot,
        "preparation": reusable_preparation(source_snapshot, environment, donor),
        "donor_launch_env": (donor.get('profile') or (donor.get('attestation') or {}).get('profile') or {}).get('launch_env', {}),
    }
    if not native_recipe:
        command_environment = pool.backend.command_environment(donor)
        spec['python'] = donor_python = command_environment['python']
        spec['donor_launch_env'] = command_environment['launch_env']
        spec['excluded_launch_roots'] = [donor.get('root'), (donor.get('endpoint') or {}).get('root'), (donor.get('endpoint') or {}).get('cwd')]
    reuse = None
    if native_recipe:
        candidates = []
        for candidate in _runtime_rows(pool):
            previous = (candidate.get('attestation') or {}).get('preparation') or {}
            if candidate.get('user') != user or candidate.get('container_name') != spec['container_name']:
                continue
            if candidate.get('endpoint', {}).get('host') != host or candidate.get('endpoint', {}).get('port') != int(ssh_port):
                continue
            if candidate.get('state') not in {'ready', 'bound'}:
                continue
            expected = reusable_preparation(source_snapshot, environment, candidate)
            # The native recipe's generated import-time SoC data belongs to
            # its output proof. Older incomplete bundles can donate verified
            # dependencies but must rebuild their native outputs.
            generated_metadata = 'vllm-ascend/vllm_ascend/_build_info.py'
            complete_metadata = generated_metadata in (candidate.get('attestation') or {}).get('files', {})
            if previous.get('native_key') == expected['native_key'] and complete_metadata:
                candidates.insert(0, ('native', candidate, expected))
            elif previous.get('dependency_key') == expected['dependency_key']:
                candidates.append(('dependencies', candidate, expected))
        if candidates:
            kind, candidate, expected = candidates[0]
            spec['preparation'] = expected
            reuse = {'kind': kind, 'runtime': candidate}
            if kind == 'native':
                spec['python'] = candidate['python']
        else:
            # Historical roots retain their original owner/format. Qualify an
            # exact source only by current remote evidence, without restamping
            # its profile or mutating its interpreter and outputs.
            for candidate in _runtime_rows(pool):
                if candidate.get('user') != user or candidate.get('container_name') != spec['container_name']:
                    continue
                if candidate.get('endpoint', {}).get('host') != host or candidate.get('endpoint', {}).get('port') != int(ssh_port):
                    continue
                if candidate.get('state') not in {'ready', 'bound'} or (candidate.get('attestation') or {}).get('preparation'):
                    continue
                if not candidate.get('attestation', {}).get('build_inputs') or source_snapshot.get('build_env'):
                    continue
                result = pool.backend.qualify_prepared_inputs(candidate, source_snapshot)
                if result.get('qualified'):
                    spec['preparation'] = reusable_preparation(source_snapshot, environment, candidate)
                    reuse = {'kind': 'native', 'runtime': candidate, 'qualification': result}
                    spec['python'] = candidate['python']
                    break
    backend = pool.backend
    prepared = backend.prepare_task_root(
        spec, sources=sources, environment=environment, donor_python=donor_python,
        source_snapshot=source_snapshot, reuse=reuse,
        workspace_root=str(Path(next(iter(sources.values()))).resolve().parent) if sources else None,
        on_progress=on_progress, log_dir=log_dir,
        on_preparation_job=on_preparation_job, cancel_requested=cancel_requested,
        **({"compile_scope": compile_scope} if compile_scope is not None else {}),
    )
    if checkout_session:
        from mindie_coordinator.backend import PreparedNativeView
        if isinstance(prepared, PreparedNativeView):
            return pool._bind_prepared(runtime_id, spec, user, checkout_session, checkout_request, prepared.attestation)
        return pool._register_checkout(runtime_id, spec, user, checkout_session, checkout_request)
    return pool.register(runtime_id, spec)


def _runtime_rows(pool):
    with pool.transaction() as db:
        return pool.rows(db, "runtime")
