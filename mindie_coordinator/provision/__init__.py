"""Prepare the per-user container and a task work root from a named recipe.

This package owns bootstrap/image/smoke. Consumers pass host/image/user;
they do not import workspace machine-management scripts.
"""

from __future__ import annotations

import getpass
import json
import time
from typing import Any

from mindie_coordinator.machine_directory import MachineDirectory
from mindie_coordinator.provision import host_ops
from mindie_coordinator.provision.task_environment import (  # noqa: F401
    TaskRootBusy,
    isolated_python,
    isolated_root,
    prepare_task_environment,
    task_runtime_id,
)
from mindie_coordinator.ready_runtime import safe_id, user_container_name
from mindie_coordinator.user_identity import load_github_identity

DEFAULT_WORKDIR = host_ops.DEFAULT_WORKDIR


def task_work_root(session_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in session_id)[:80]
    return f"{DEFAULT_WORKDIR}/tasks/{safe}"


def provision_user_container(
    *,
    host: str,
    image: str,
    user: str | None = None,
    host_user: str = host_ops.DEFAULT_HOST_USER,
    host_port: int = host_ops.DEFAULT_HOST_PORT,
    ssh_port: int | None = None,
    machine_type: str | None = None,
    password_env: str | None = None,
    machines: MachineDirectory | None = None,
    reserve_port=None,
    on_progress=None,
) -> dict[str, Any]:
    """Create or reuse `mindie-<user>` on `host` from a named image/recipe."""
    timings, observation = {}, {}

    def progress(phase, status, **details):
        if on_progress is not None:
            on_progress({'step': 'prepare-container', 'phase': phase, 'status': status,
                         'phase_seconds': dict(timings), **details,
                         **({'existing_observation': dict(observation)} if observation else {})})

    def measured(phase, operation):
        progress(phase, 'running')
        started = time.monotonic()
        try:
            result = operation()
        except Exception as exc:
            timings[phase] = round(time.monotonic() - started, 6)
            progress(phase, 'failed', error_type=type(exc).__name__)
            raise
        timings[phase] = round(time.monotonic() - started, 6)
        if phase == 'existing-container':
            observation.update(status=str(result.get('status', 'unknown'))[:40],
                               elapsed_seconds=timings[phase])
            reason = result.get('reason')
            if isinstance(reason, str):
                observation['reason'] = reason[:300]
            if result.get('remote_outcome') == 'unknown':
                observation['remote_outcome'] = 'unknown'
        progress(phase, 'complete')
        return result

    def remote(phase, target, script, *, require_payload=True, **options):
        return measured(phase, lambda: host_ops.assert_remote_success(
            host_ops.run_remote_script(target, script, **options), require_payload=require_payload))

    identity = load_github_identity() if not user else None
    user = safe_id(user or (identity["login"] if identity else getpass.getuser()))
    container = user_container_name(user)
    if not image or image in getattr(host_ops, "LEGACY_IMAGE_SELECTORS", {"auto"}):
        raise ValueError(
            "provision requires an explicit image selector: local-latest, rc, main, stable, "
            "or a full image reference"
        )
    host_target = host_ops.SshTarget(host=host, user=host_user, port=host_port)
    if password_env:
        args = _namespace(
            host=host, user=host_user, host_port=host_port, public_key_file=None,
            password=None, password_env=password_env, password_stdin=False, print_command=False,
        )
        if host_ops.cmd_bootstrap_host_key(args) != 0:
            raise host_ops.MachineManagementError("host key bootstrap failed")
    image_request = host_ops.image_request_payload(image, machine_type=machine_type)
    if ssh_port and image_request['policy'] == 'explicit' and len(image_request['candidates']) == 1:
        from mindie_coordinator.provision.existing_container import observe_existing
        existing = measured('existing-container', lambda: observe_existing(host_target, container=container, user=user,
                                    ssh_port=int(ssh_port), workdir=DEFAULT_WORKDIR,
                                    image_request=image_request))
        if existing.get('status') == 'cancelled':
            raise host_ops.MachineManagementError('existing container verification cancelled')
        if existing.get('status') == 'mismatch':
            raise host_ops.MachineManagementError(existing['reason'])
        if existing.get('status') == 'match':
            chosen_port = int(ssh_port)
            if reserve_port is not None:
                reserved = measured('ssh-port-reservation', lambda: reserve_port(user=user, container_name=container, port=chosen_port))
                if int(reserved.get('port') or chosen_port) != chosen_port:
                    raise host_ops.MachineManagementError('reserved SSH port differs from the verified existing listener')
            return {**_record_ready_container(host, image, user, host_user, host_port,
                                             chosen_port, machine_type, machines),
                    'image_verification': existing}
    probe_payload = remote('host-probe',
        host_target,
        host_ops.render_host_probe_script(),
        args=[json.dumps(image_request, ensure_ascii=False), host_ops.DEFAULT_PORT_RANGE, "mindie-"],
        timeout_seconds=host_ops.DEFAULT_PROBE_TIMEOUT_SECONDS,
        stream_progress=False,
        reuse_connection=True,
        require_payload=True,
    )
    chosen_port = int(ssh_port or probe_payload.get("free_port") or probe_payload.get("suggested_port") or probe_payload.get("ssh_port")
                      or probe_payload.get("container_ssh_port") or 0)
    if reserve_port is not None:
        # A probe's free port is only a live observation, not a reservation.
        # Let the host owner select and claim an automatic port atomically.
        requested_port = int(ssh_port or 0)
        reserved = measured('ssh-port-reservation', lambda: reserve_port(user=user, container_name=container, port=requested_port))
        chosen_port = int(reserved.get("port") or 0)
        if requested_port and chosen_port != requested_port:
            raise host_ops.MachineManagementError("reserved SSH port differs from the requested fixed port")
    if chosen_port <= 0:
        raise host_ops.MachineManagementError("host did not reserve or offer a container SSH port")
    key_path = host_ops.find_public_key(None)
    public_key = host_ops.load_public_key(key_path)
    remote('container-bootstrap',
        host_target,
        host_ops.render_bootstrap_host_script(),
        args=[
            container,
            str(chosen_port),
            json.dumps(image_request, ensure_ascii=False),
            DEFAULT_WORKDIR,
            public_key,
            user,
            "false",
            machine_type or "",
            "",
        ],
        timeout_seconds=host_ops.DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS,
        stream_progress=False,
    )
    container_target = host_ops.SshTarget(host=host, user="root", port=chosen_port)
    remote('metadata-readiness',
        container_target,
        host_ops.render_smoke_script(device_test=False),
        args=[""],
        timeout_seconds=host_ops.DEFAULT_SMOKE_TIMEOUT_SECONDS,
        stream_progress=False,
        reuse_connection=True,
        require_payload=True,
    )
    return _record_ready_container(host, image, user, host_user, host_port,
                                   chosen_port, machine_type, machines)


def _record_ready_container(host, image, user, host_user, host_port, chosen_port, machine_type, machines):
    container = user_container_name(user)
    record = {
        "alias": host,
        "host": {"ip": host, "port": host_port, "user": host_user, "machine_type": machine_type},
        "container": {"name": container, "ssh_port": chosen_port, "user": "root", "workdir": DEFAULT_WORKDIR},
        "image": {"requested": image},
        "user": user,
    }
    directory = machines or MachineDirectory()
    directory.upsert_machine(record)
    return {
        "user": user,
        "container_name": container,
        "host": host,
        "ssh_port": chosen_port,
        "image": image,
        "workdir": DEFAULT_WORKDIR,
        "state": "ready",
    }


def _namespace(**values):
    return type("Args", (), values)()
