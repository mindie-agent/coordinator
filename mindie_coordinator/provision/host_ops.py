#!/usr/bin/env python3
"""Host and container bootstrap operations owned by mindie-coordinator.

[Role]
Owns bounded remote script execution, progress parsing, host targeting, and
container removal. SSH authentication and key bootstrap belong to remote-dev. Image
resolution and script rendering live in dedicated sibling modules and are
re-exported here for existing callers.

[Invariants]
- Never allows bare image repositories or moving `:latest` tags; the selector
  rules are enforced by `image_resolver` and re-exported here.
- `run_remote_script` uses `remote-dev` transport (`run_rpc_script` for reusable
  probes, `run_stream` for long-running bootstrap/install flows).
- Host bootstrap returns the shared authentication profile used by later calls.

[Timeouts & Locks]
- Probe timeout: `DEFAULT_PROBE_TIMEOUT_SECONDS` (60s).
- Container bootstrap timeout: `DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS` (1800s).
- Smoke timeout: `DEFAULT_SMOKE_TIMEOUT_SECONDS` (240s).

[Side-Effect Policy]
- Credentials are handled only by remote-dev's shared authentication provider.
- Unknown or cancelled remote probe outcomes raise `MachineManagementError`
  immediately without silent fallback or blind replay.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import sys
from dataclasses import dataclass
from typing import Any, Sequence

from mindie_coordinator.provision.host_bootstrap_script import (
    render_bootstrap_host_script,
)
from mindie_coordinator.provision.host_probe_scripts import (
    render_host_probe_script,
    render_mesh_add_peer_script,
    render_mesh_export_key_script,
    render_remove_container_host_script,
    render_smoke_script,
)
from mindie_coordinator.provision.image_resolver import (
    DEFAULT_HOST_PORT,
    DEFAULT_HOST_USER,
    DEFAULT_IMAGE,
    DEFAULT_IMAGE_CANDIDATES,
    FORBIDDEN_IMAGE_TAGS,
    IMAGE_POLICY_LOCAL_LATEST,
    IMAGE_REGISTRY_MIRRORS,
    IMAGE_REGISTRY_NJU,
    IMAGE_REGISTRY_OFFICIAL,
    IMAGE_RESOLVER_USER_AGENT,
    IMAGE_SELECTOR_ALIASES,
    IMAGE_SELECTOR_LOCAL_LATEST,
    IMAGE_SELECTOR_MAIN,
    IMAGE_SELECTOR_RC,
    IMAGE_SELECTOR_STABLE,
    IMAGE_SUFFIX_BY_MACHINE_TYPE,
    LATEST_RELEASE_PAGE,
    LATEST_RELEASE_TIMEOUT_SECONDS,
    LEGACY_IMAGE_SELECTORS,
    LOCAL_IMAGE_DISCOVERY_PY,
    MACHINE_TYPE_CHOICES,
    PROGRESS_SENTINEL,
    RELEASES_API,
    RELEASES_PAGE,
    SEMVER_TAG_PATTERN,
    SENTINEL,
    SOC_TO_MACHINE_TYPE,
    ImageResolution,
    MachineManagementError,
    SshTarget,
    choose_latest_tag,
    choose_release_tag_from_catalog,
    choose_release_tag_from_page,
    docker_ref_repo,
    docker_ref_tag,
    fetch_latest_prerelease_tag,
    fetch_latest_release_tag,
    fetch_release_catalog,
    image_candidates_for_tag,
    image_request_payload as _image_request_payload,
    image_tag_for_machine,
    infer_machine_type_from_image,
    normalize_machine_type,
    parse_release_tag,
    print_json,
    require_explicit_image_ref,
    resolve_image_request as _resolve_image_request,
    run_local,
    shell_join,
    validate_explicit_image_for_machine,
)
from remote_dev.core.endpoint import Endpoint, resolve_endpoint
from remote_dev.core.ssh_transport import run_stream
from remote_dev.auth import ensure_key, setup_access

DEFAULT_WORKDIR = "/vllm-workspace"
DEFAULT_PORT_RANGE = "46000:46999"
DEFAULT_KNOWN_HOSTS = pathlib.Path.home() / ".ssh" / "known_hosts"
DEFAULT_PROBE_TIMEOUT_SECONDS = 60
DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS = 1800
DEFAULT_SMOKE_TIMEOUT_SECONDS = 240


def resolve_image_request(
    spec: str, *, machine_type: str | None = None
) -> ImageResolution:
    """Resolve through the module-global catalog fetcher for patchability."""
    return _resolve_image_request(
        spec, machine_type=machine_type, _catalog_fetcher=fetch_release_catalog
    )


def image_request_payload(
    spec: str, *, machine_type: str | None = None
) -> dict[str, Any]:
    """Build the image request payload through the re-exported fetcher seam."""
    return _image_request_payload(
        spec, machine_type=machine_type, _catalog_fetcher=fetch_release_catalog
    )


def _script_with_positional_args(script: str, args: Sequence[str]) -> str:
    """Keep ``bash -s --`` positional parameters when the stream uses ``bash -c``.

    ``run_stream`` quotes the whole script as one remote command. Prefixing
    ``set --`` with POSIX-quoted literals makes ``$1``… the same values the
    old stdin ``bash -s -- args`` path passed through.
    """
    if not args:
        return script
    return "set -- " + " ".join(shlex.quote(str(item)) for item in args) + "\n" + script


@dataclass
class RemoteResult:
    target: SshTarget
    returncode: int
    stdout: str
    stderr: str
    payload: dict[str, Any] | None
    timed_out: bool = False
    timeout_seconds: int | None = None
    progress_events: list[dict[str, Any]] | None = None


def parse_sentinel(stdout: str) -> dict[str, Any] | None:
    payload: dict[str, Any] | None = None
    for line in stdout.splitlines():
        if line.startswith(SENTINEL):
            try:
                payload = json.loads(line[len(SENTINEL) :])
            except json.JSONDecodeError as exc:
                payload = {"success": False, "error": f"invalid remote JSON: {exc}"}
    return payload


def parse_progress_event(line: str) -> dict[str, Any] | None:
    if not line.startswith(PROGRESS_SENTINEL):
        return None
    try:
        event = json.loads(line[len(PROGRESS_SENTINEL) :])
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict):
        return None
    return event


def format_progress_event(
    event: dict[str, Any], *, target: SshTarget | None = None
) -> str:
    phase = str(event.get("phase") or "remote")
    message = str(event.get("message") or event.get("status") or "running")
    expected = event.get("expected_seconds")
    if isinstance(expected, int) and expected > 0:
        message = f"{message} (~{expected}s)"
    host = f" {target.host}" if target is not None else ""
    return f"[mindie{host}][{phase}] {message}"


def emit_progress_event(
    event: dict[str, Any], *, target: SshTarget | None = None
) -> None:
    sys.stderr.write(format_progress_event(event, target=target) + "\n")
    sys.stderr.flush()


def run_remote_script(
    target: SshTarget,
    script: str,
    *,
    args: Sequence[str] = (),
    batch_mode: bool = True,
    timeout_seconds: int | None = None,
    stream_progress: bool = True,
    reuse_connection: bool = False,
    identity_file: str | None = None,
    auth_ref: str | None = None,
) -> RemoteResult:
    if not batch_mode:
        raise MachineManagementError(
            "coordinator provision uses non-interactive SSH only; pass password-env for first host-key bootstrap"
        )
    endpoint = Endpoint.for_long_stream(host=target.host, port=int(target.port), user=target.user,
                                       identity_file=identity_file, auth_ref=auth_ref)
    remote_script = _script_with_positional_args(script, args)
    timeout_ms = None if not timeout_seconds else int(timeout_seconds * 1000)
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    progress_events: list[dict[str, Any]] = []

    def on_output(channel: str, text: str) -> None:
        if channel == "stdout":
            stdout_parts.append(text)
        else:
            stderr_parts.append(text)
        event = parse_progress_event(text)
        if event is not None:
            progress_events.append(event)
            if stream_progress:
                emit_progress_event(event, target=target)

    try:
        if reuse_connection:
            # Bounded package-owned probes use the same connection as managed
            # preparation. Long bootstrap/install commands retain streaming.
            from remote_dev.core.ssh_transport import run_rpc_script
            endpoint = resolve_endpoint({'host': target.host, 'port': int(target.port),
                                         'user': target.user, 'root': '/', 'cwd': '/',
                                         'identity_file': identity_file, 'auth_ref': auth_ref})
            completed = run_rpc_script(endpoint, remote_script, timeout_ms=timeout_ms)
            if completed.cancelled:
                raise MachineManagementError('provision probe cancelled; no fallback command was sent')
            if completed.returncode is None and not completed.timed_out:
                raise MachineManagementError('provision probe outcome is unknown; no fallback command was sent')
            for channel in ('stdout', 'stderr'):
                for line in (getattr(completed, channel) or '').splitlines(keepends=True):
                    on_output(channel, line)
        else:
            completed = run_stream(
                endpoint,
                remote_script,
                timeout_ms=timeout_ms,
                merge_stderr=False,
                on_output=on_output,
            )
    except FileNotFoundError as exc:
        raise MachineManagementError("required local command not found: ssh") from exc

    stdout = "".join(stdout_parts)
    stderr = "".join(stderr_parts)
    if completed.timed_out and completed.stderr:
        note = completed.stderr
        if note not in stderr:
            stderr = f"{stderr}{note}" if stderr else note
    returncode = 0 if completed.returncode is None else int(completed.returncode)
    return RemoteResult(
        target=target,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        payload=parse_sentinel(stdout),
        timed_out=completed.timed_out,
        timeout_seconds=timeout_seconds,
        progress_events=progress_events,
    )


def compact_failure_tail(
    text: str, *, max_lines: int = 20, max_chars: int = 1600
) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    lines = stripped.splitlines()
    tail = "\n".join(lines[-max_lines:])
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


def assert_remote_success(
    result: RemoteResult, *, require_payload: bool = True
) -> dict[str, Any]:
    payload = dict(result.payload or {})
    if result.progress_events:
        payload.setdefault("progress_events", result.progress_events)
    if result.timeout_seconds is not None:
        transport = payload.setdefault("transport", {})
        transport.setdefault("timeout_seconds", result.timeout_seconds)
        transport.setdefault("timed_out", result.timed_out)

    if result.timed_out:
        if result.payload is not None and result.payload.get("success") is True:
            transport = payload.setdefault("transport", {})
            transport.update(
                {
                    "timeout_seconds": result.timeout_seconds,
                    "timed_out": True,
                    "timeout_recovered_after_payload": True,
                    "returncode": result.returncode,
                }
            )
            return payload
        detail = f"remote command timed out after {result.timeout_seconds}s"
        if result.progress_events:
            detail += f"; last progress: {format_progress_event(result.progress_events[-1], target=result.target)}"
        stderr_tail = compact_failure_tail(result.stderr)
        if stderr_tail:
            detail += f"\n--- stderr tail ---\n{stderr_tail}"
        raise MachineManagementError(detail)

    if require_payload and result.payload is None:
        detail = (
            result.stderr.strip()
            or result.stdout.strip()
            or "remote command produced no JSON payload"
        )
        raise MachineManagementError(detail)
    if result.returncode != 0:
        stderr_tail = compact_failure_tail(result.stderr)
        if result.payload and result.payload.get("error"):
            base = str(result.payload["error"])
            if stderr_tail:
                raise MachineManagementError(
                    f"{base}\n--- stderr tail ---\n{stderr_tail}"
                )
            raise MachineManagementError(base)
        detail = (
            result.stderr.strip() or result.stdout.strip() or "remote command failed"
        )
        raise MachineManagementError(detail)
    return payload


def _namespace(**values: Any) -> argparse.Namespace:
    return argparse.Namespace(**values)


def host_target(args: argparse.Namespace) -> SshTarget:
    return SshTarget(host=args.host, user=args.user, port=args.host_port)


def cmd_bootstrap_host_key(args: argparse.Namespace) -> int:
    """Keep the legacy CLI input boundary while remote-dev owns authentication."""
    endpoint = resolve_endpoint({
        "host": args.host, "port": int(args.host_port), "user": args.user,
        "root": "/", "cwd": "/", "runtime_env": False,
    })
    identity_file = getattr(args, "identity_file", None) or getattr(args, "public_key_file", None)
    if getattr(args, "print_command", False):
        command = ["remote-dev", "auth", "setup", "--host", endpoint.host,
                   "--port", str(endpoint.port), "--user", endpoint.user]
        if identity_file:
            command.extend(["--identity-file", str(identity_file)])
        print_json({"success": True, "executed": False, "result": "command-preview",
                    "bootstrap_command": shell_join(command)})
        return 0
    password = getattr(args, "password", None)
    if getattr(args, "password_stdin", False):
        password = sys.stdin.readline().rstrip("\r\n")
        if not password:
            raise MachineManagementError("no password was received on stdin")
    result = setup_access(endpoint, password=password,
                          password_env=getattr(args, "password_env", None),
                          identity_file=identity_file,
                          install_key=getattr(args, "install_key", True),
                          timeout_ms=getattr(args, "timeout_ms", 30000))
    print_json({"success": True, **result})
    return 0


def remove_known_host_entry(
    host: str, port: int, known_hosts: pathlib.Path
) -> dict[str, Any]:
    if not known_hosts.exists():
        return {
            "success": True,
            "known_hosts": str(known_hosts),
            "removed": False,
            "reason": "known_hosts file does not exist",
        }
    proc = run_local(["ssh-keygen", "-R", f"[{host}]:{port}", "-f", str(known_hosts)])
    combined = (proc.stdout + "\n" + proc.stderr).lower()
    return {
        "success": proc.returncode == 0 or "not found" in combined,
        "known_hosts": str(known_hosts),
        "removed": proc.returncode == 0,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def remove_container_result(args: argparse.Namespace) -> dict[str, Any]:
    """Remove the named container and return the confirmed host payload."""
    target = host_target(args)
    result = run_remote_script(
        target,
        render_remove_container_host_script(),
        args=[args.container_name],
        batch_mode=True,
    )
    payload = assert_remote_success(result)
    payload["target"] = {
        "host": target.host,
        "user": target.user,
        "host_port": target.port,
    }
    if (payload.get("removed") is True and args.clean_local_known_hosts
            and args.container_ssh_port):
        try:
            payload["local_known_hosts_cleanup"] = remove_known_host_entry(
                args.host,
                args.container_ssh_port,
                pathlib.Path(args.known_hosts).expanduser().resolve(),
            )
        except Exception as exc:
            # Remote removal is already confirmed; a local key cleanup failure
            # is represented beside it, never converted into remote failure.
            payload["local_known_hosts_cleanup"] = {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}"[:800],
            }
    elif args.clean_local_known_hosts and args.container_ssh_port:
        payload["local_known_hosts_cleanup"] = {
            "success": False,
            "skipped": True,
            "reason": "container was already absent; no host key belongs to this removal",
        }
    return payload


def cmd_remove_container(args: argparse.Namespace) -> int:
    payload = remove_container_result(args)
    print_json(payload)
    return 0
