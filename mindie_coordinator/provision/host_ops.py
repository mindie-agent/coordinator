#!/usr/bin/env python3
"""Host and container bootstrap operations owned by mindie-coordinator.

[Role]
Owns non-interactive SSH transport, bounded remote script execution, progress
parsing, askpass/key bootstrap, host targeting, and container removal. Image
resolution and script rendering live in dedicated sibling modules and are
re-exported here for existing callers.

[Invariants]
- Never allows bare image repositories or moving `:latest` tags; the selector
  rules are enforced by `image_resolver` and re-exported here.
- `run_remote_script` uses `remote-dev` transport (`run_rpc_script` for reusable
  probes, `run_stream` for long-running bootstrap/install flows).
- Every moved public name remains importable from this module.

[Timeouts & Locks]
- Probe timeout: `DEFAULT_PROBE_TIMEOUT_SECONDS` (60s).
- Container bootstrap timeout: `DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS` (1800s).
- Smoke timeout: `DEFAULT_SMOKE_TIMEOUT_SECONDS` (240s).

[Side-Effect Policy]
- Non-interactive SSH only (`BatchMode=yes` or `SSH_ASKPASS` via `--password-env`).
- Unknown or cancelled remote probe outcomes raise `MachineManagementError`
  immediately without silent fallback or blind replay.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
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
from remote_dev.core.ssh_transport import interactive_ssh_command, run_stream, ssh_base_cmd

DEFAULT_WORKDIR = "/vllm-workspace"
DEFAULT_PORT_RANGE = "46000:46999"
DEFAULT_KNOWN_HOSTS = pathlib.Path.home() / ".ssh" / "known_hosts"
DEFAULT_PASSWORD_ENV = "MINDIE_SSH_PASSWORD"
DEFAULT_PROBE_TIMEOUT_SECONDS = 60
DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS = 1800
DEFAULT_SMOKE_TIMEOUT_SECONDS = 240
ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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


def ssh_command(
    target: SshTarget,
    *,
    batch_mode: bool = True,
    extra_options: Sequence[str] = (),
    identity_file: pathlib.Path | None = None,
) -> list[str]:
    if not batch_mode:
        raise MachineManagementError(
            "coordinator provision uses non-interactive SSH only; pass password-env for first host-key bootstrap"
        )
    payload = {
        "host": target.host,
        "port": int(target.port),
        "user": target.user,
        "root": "/",
        "cwd": "/",
        "ssh_mux": False,
        "runtime_env": False,
    }
    if identity_file is not None:
        payload["identity_file"] = str(identity_file)
    command = ssh_base_cmd(resolve_endpoint(payload))
    for option in extra_options:
        command.extend(["-o", option])
    return command


def validate_env_name(name: str) -> str:
    if not ENV_NAME_PATTERN.fullmatch(name):
        raise MachineManagementError(f"invalid environment variable name: {name!r}")
    return name


def private_key_for_public_key(path: pathlib.Path | None) -> pathlib.Path | None:
    if path is None:
        return None
    candidate = path.with_suffix("") if path.suffix == ".pub" else path
    if candidate.exists():
        return candidate.resolve()
    return None


def read_password_value(args: argparse.Namespace) -> tuple[str | None, str | None, str]:
    if getattr(args, "password", None) is not None:
        return args.password, "literal", DEFAULT_PASSWORD_ENV
    if getattr(args, "password_env", None):
        env_name = validate_env_name(args.password_env)
        value = os.environ.get(env_name)
        if value is None:
            raise MachineManagementError(
                f"environment variable {env_name!r} is not set"
            )
        return value, f"env:{env_name}", env_name
    if getattr(args, "password_stdin", False):
        value = sys.stdin.read()
        if value is None:
            value = ""
        value = value.rstrip("\r\n")
        if not value:
            raise MachineManagementError("no password was received on stdin")
        return value, "stdin", DEFAULT_PASSWORD_ENV
    return None, None, DEFAULT_PASSWORD_ENV


def build_authorized_keys_remote_command(public_key: str) -> str:
    quoted_key = shlex.quote(public_key)
    return (
        "umask 077; "
        "mkdir -p ~/.ssh && touch ~/.ssh/authorized_keys; "
        "chmod 700 ~/.ssh; "
        "chmod 600 ~/.ssh/authorized_keys; "
        f"grep -qxF {quoted_key} ~/.ssh/authorized_keys 2>/dev/null || "
        f"printf '%s\n' {quoted_key} >> ~/.ssh/authorized_keys"
    )


def write_askpass_helper(temp_dir: pathlib.Path, env_name: str) -> pathlib.Path:
    env_name = validate_env_name(env_name)
    if os.name == "nt":
        helper_py = temp_dir / "askpass.py"
        helper_py.write_text(
            "import os, sys\n"
            f"sys.stdout.write(os.environ.get({env_name!r}, '') + '\n')\n",
            encoding="utf-8",
        )
        helper = temp_dir / "askpass.cmd"
        helper.write_text(
            f'@echo off\r\nsetlocal\r\n"{sys.executable}" "{helper_py}"\r\n',
            encoding="utf-8",
        )
    else:
        helper = temp_dir / "askpass.sh"
        helper.write_text(
            f"#!/bin/sh\nprintf '%s\\n' \"${{{env_name}}}\"\n",
            encoding="utf-8",
        )
        helper.chmod(0o700)
    return helper


def run_with_askpass(
    cmd: Sequence[str],
    *,
    password: str,
    env_name: str = DEFAULT_PASSWORD_ENV,
) -> subprocess.CompletedProcess[str]:
    env_name = validate_env_name(env_name)
    temp_dir = pathlib.Path(tempfile.mkdtemp(prefix="mindie-askpass-"))
    try:
        helper = write_askpass_helper(temp_dir, env_name)
        env = os.environ.copy()
        env[env_name] = password
        env["SSH_ASKPASS"] = str(helper)
        env["SSH_ASKPASS_REQUIRE"] = "force"
        env.setdefault("DISPLAY", "mindie:0")
        return run_local(cmd, env=env, stdin_source=subprocess.DEVNULL)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


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
) -> RemoteResult:
    if not batch_mode:
        raise MachineManagementError(
            "coordinator provision uses non-interactive SSH only; pass password-env for first host-key bootstrap"
        )
    endpoint = Endpoint.for_long_stream(host=target.host, port=int(target.port), user=target.user)
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
                                         'user': target.user, 'root': '/', 'cwd': '/'})
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


def find_public_key(explicit: str | None = None) -> pathlib.Path:
    if explicit:
        path = pathlib.Path(explicit).expanduser().resolve()
        if not path.exists():
            raise MachineManagementError(f"public key file not found: {path}")
        return path

    candidates = [
        pathlib.Path.home() / ".ssh" / "id_ed25519.pub",
        pathlib.Path.home() / ".ssh" / "id_rsa.pub",
        pathlib.Path.home() / ".ssh" / "id_ecdsa.pub",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen:
        default_pub = candidates[0]
        default_priv = default_pub.with_suffix("")
        try:
            default_pub.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(default_priv)],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if proc.returncode == 0 and default_pub.exists():
                return default_pub.resolve()
        except (OSError, subprocess.SubprocessError):
            pass
    raise MachineManagementError(
        "no local public key found; pass --public-key-file or create ~/.ssh/id_ed25519.pub"
    )


def load_public_key(path: pathlib.Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text.startswith("ssh-"):
        raise MachineManagementError(f"not a valid SSH public key: {path}")
    return text


def _namespace(**values: Any) -> argparse.Namespace:
    return argparse.Namespace(**values)


def host_target(args: argparse.Namespace) -> SshTarget:
    return SshTarget(host=args.host, user=args.user, port=args.host_port)


def check_direct_ssh(
    target: SshTarget,
    *,
    identity_file: pathlib.Path | None = None,
) -> dict[str, Any]:
    try:
        result = run_local(
            ssh_command(target, batch_mode=True, identity_file=identity_file)
            + ["printf", "ok"]
        )
    except MachineManagementError as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "ok": result.returncode == 0 and result.stdout.endswith("ok"),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "identity_file": str(identity_file) if identity_file is not None else None,
    }


def build_bootstrap_host_key_command(
    target: SshTarget,
    *,
    key_path: pathlib.Path,
    public_key: str,
) -> tuple[str, list[str]]:
    del key_path
    remote_cmd = build_authorized_keys_remote_command(public_key)
    endpoint = resolve_endpoint(
        {
            "host": target.host,
            "port": int(target.port),
            "user": target.user,
            "root": "/",
            "cwd": "/",
            "ssh_mux": False,
            "runtime_env": False,
        }
    )
    command = interactive_ssh_command(endpoint, [remote_cmd])
    return "ssh", command


def cmd_bootstrap_host_key(args: argparse.Namespace) -> int:
    target = host_target(args)
    key_path = find_public_key(args.public_key_file)
    private_key = private_key_for_public_key(key_path)
    public_key = load_public_key(key_path)
    password, password_source, password_env_name = read_password_value(args)
    before = check_direct_ssh(target, identity_file=private_key)
    tool_name, command = build_bootstrap_host_key_command(
        target,
        key_path=key_path,
        public_key=public_key,
    )
    payload: dict[str, Any] = {
        "target": {"host": target.host, "user": target.user, "host_port": target.port},
        "public_key_file": str(key_path),
        "private_key_file": str(private_key) if private_key is not None else None,
        "precheck": before,
        "bootstrap_tool": tool_name,
        "bootstrap_command": shell_join(command),
        "password_mode": password_source,
        "needs_interactive_terminal": password is None,
        "password_automation": "ssh-askpass" if password is not None else None,
    }
    if before["ok"]:
        payload.update(
            {"success": True, "executed": False, "result": "already-configured"}
        )
        print_json(payload)
        return 0

    if args.print_command:
        payload.update(
            {
                "success": True,
                "executed": False,
                "result": "command-preview",
                "message": "run the bootstrap command once if you want to handle the password manually",
            }
        )
        print_json(payload)
        return 0

    if password is not None:
        proc = run_with_askpass(command, password=password, env_name=password_env_name)
        after = check_direct_ssh(target, identity_file=private_key)
        payload.update(
            {
                "executed": True,
                "command_returncode": proc.returncode,
                "postcheck": after,
                "success": after["ok"],
                "result": "bootstrapped" if after["ok"] else "failed",
                "mode": "noninteractive-password",
                "stdout_tail": compact_failure_tail(proc.stdout),
                "stderr_tail": compact_failure_tail(proc.stderr),
            }
        )
        if not after["ok"]:
            payload["error"] = (
                "password-based host bootstrap did not establish key-based SSH; verify the supplied password, username, and host password-login policy"
            )
            print_json(payload)
            return 2
        print_json(payload)
        return 0

    payload.update(
        {
            "success": False,
            "executed": False,
            "result": "needs_input",
            "error": (
                "host key bootstrap requires --password-env or --password-stdin; "
                "coordinator provision does not open an interactive SSH password prompt"
            ),
        }
    )
    print_json(payload)
    return 2


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
