"""Public host NPU queue API.

The scaffold imports this module as ``mindie_coordinator.host_queue``. It
re-exports the host allocation protocol and the client that ships that
protocol to a physical host.

The host module itself stays stdlib-only: it is read as source text and
executed on the host. ``MINDIE_HOST_QUEUE_MODULE`` overrides the bundled file.
"""

from __future__ import annotations
from remote_dev.observability import observed_operation

import importlib.util
import json
import os
import shlex
from pathlib import Path
from typing import Any, Callable

from mindie_coordinator.host.mindie_npu_coordination import (  # noqa: F401
    DEFAULT_CONTAINER_SSH_PORT_RANGE,
    DEFAULT_SERVING_PORT_RANGE,
    DEFAULT_STATE_DIR,
    SCHEMA_VERSION,
    CoordinationError,
    NpuCoordinator,
    container_ssh_task_id,
    handle_request,
    parse_npu_smi_hbm,
    parse_npu_smi_info,
    probe_listening_ports,
    probe_named_container,
    JOB_TOKEN_ENV,
    process_guard_busy,
    probe_npu_occupancy,
    resolve_host_state_dir,
    user_container_name,
)

HOST_QUEUE_MODULE_ENV = "MINDIE_HOST_QUEUE_MODULE"
SOURCE_DELIMITER = "__MINDIE_NPU_COORDINATION_SOURCE__"
UNRESOLVED = {"failed", "needs_input", "probe_failed", "timeout"}
RUNNER = """
import sys as _sys
try:
    _request = json.loads(_sys.argv[1]) if len(_sys.argv) > 1 else json.load(_sys.stdin)
    _result = handle_request(_request)
    print(json.dumps(_result, indent=2, ensure_ascii=False, sort_keys=True))
except CoordinationError as _exc:
    _failure = {"status": "needs_input", "error": str(_exc)}
    if getattr(_exc, 'error_code', None):
        _failure['error_code'] = _exc.error_code
        if getattr(_exc, 'port', None) is not None:
            _failure['port'] = _exc.port
    print(json.dumps(_failure, indent=2, ensure_ascii=False, sort_keys=True))
    raise SystemExit(2)
except Exception as _exc:
    print(json.dumps({"status": "failed", "error": str(_exc)}, indent=2, ensure_ascii=False, sort_keys=True))
    raise SystemExit(2)
"""


def _reservation_failure(payload: dict[str, Any]) -> str | None:
    """Decode only known host failure fields; never expose an output tail."""
    if (payload.get('status') != 'failed' or payload.get('exit_code') != 2
            or payload.get('remote_outcome') == 'unknown'):
        return None
    tail = payload.get('stdout_tail', '')
    if not isinstance(tail, str) or len(tail) > 4000:
        return None
    try:
        failure = json.loads(tail)
    except (TypeError, ValueError):
        return None
    if not isinstance(failure, dict) or failure.get('status') != 'needs_input':
        return None
    messages = {
        'port_reserved': 'host port is already reserved',
        'container_ssh_port_mismatch': 'user already has a different SSH port reserved',
        'listening_unavailable': 'host listening ports are unavailable',
        'container_ssh_port_exhausted': 'no free container SSH port',
    }
    code = failure.get('error_code')
    if not isinstance(code, str) or code not in messages:
        return None
    port = failure.get('port')
    suffix = f' (port {port})' if type(port) is int and 0 < port < 65536 else ''
    return messages[code] + suffix + f' [{code}]'


class HostQueueUnavailable(RuntimeError):
    """An explicitly configured host-queue module path does not exist."""


def bundled_host_module_path() -> Path:
    from mindie_coordinator.host import mindie_npu_coordination as bundled

    return Path(bundled.__file__).resolve()


def host_queue_module_path(path: Path | str | None = None) -> Path:
    configured = str(path or os.environ.get(HOST_QUEUE_MODULE_ENV, ""))
    if not configured:
        return bundled_host_module_path()
    resolved = Path(configured).expanduser()
    if not resolved.is_file():
        raise HostQueueUnavailable(f"host coordination module not found: {resolved}")
    return resolved


def load_host_protocol(path: Path | str | None = None):
    """Import the host protocol (bundled package module, or an override file)."""
    configured = str(path or os.environ.get(HOST_QUEUE_MODULE_ENV, ""))
    if not configured:
        from mindie_coordinator.host import mindie_npu_coordination as bundled

        return bundled
    resolved = host_queue_module_path(path)
    spec = importlib.util.spec_from_file_location("mindie_npu_coordination", resolved)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HostQueue:
    """Send fixed protocol code with a small per-request JSON payload.

    The native transport caches code on its existing SSH connection. An
    explicitly supplied shell callable retains the embedding adapter contract.
    """

    def __init__(self, run: Callable[[dict[str, Any], str], str] | None = None, *, module_path=None):
        self._run = run
        self._module_path = module_path
        self._source: str | None = None

    def source(self) -> str:
        if self._source is None:
            self._source = host_queue_module_path(self._module_path).read_text(encoding="utf-8")
        return self._source

    def command(self, request: dict[str, Any]) -> str:
        payload = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
        return (
            "set -euo pipefail\n"
            "if command -v python3 >/dev/null 2>&1; then _py=python3; "
            "elif command -v python >/dev/null 2>&1; then _py=python; "
            "else echo '{\"status\":\"failed\",\"error\":\"python not found on host\"}'; exit 127; fi\n"
            f'"$_py" - {shlex.quote(payload)} <<\'{SOURCE_DELIMITER}\'\n'
            f"{self.source()}\n{RUNNER}\n{SOURCE_DELIMITER}\n"
        )

    @observed_operation(lambda self, host_endpoint, request: "host." + str(request.get("action", "unknown")), component="mindie-coordinator", level="DEBUG")
    def request(self, host_endpoint: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        target = {**host_endpoint, "root": "/", "cwd": "/"}
        # Resolve the configured authority before sending anything. Requests
        # never alter the code key, and uncertain mutations are never replayed
        # through the shell adapter after a transport failure.
        source = self.source()
        try:
            if self._run is None:
                from remote_dev.core.endpoint import resolve_endpoint
                from remote_dev.core.ssh_transport import run_remote_python

                payload = run_remote_python(resolve_endpoint(target), source + "\n" + RUNNER,
                                            request, timeout_ms=45000)
            else:
                payload = json.loads(self._run(target, self.command(request)))
        except Exception as exc:
            raise RuntimeError(
                "host coordination failed; inspect endpoint logs and reconcile before retry: "
                f"{exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError("host coordination returned a non-object response")
        if isinstance(payload.get("diagnostics"), dict):
            from mindie_diagnostics import get_recorder
            get_recorder("mindie-coordinator").event("DEBUG", "host.completed", **payload["diagnostics"])
        if payload.get("status") in UNRESOLVED or (payload.get("status") == "cancelled" and "task" not in payload):
            from remote_dev.core.errors import RemoteExecutionError
            details = payload.get("error_details") or {}
            raise RemoteExecutionError(_reservation_failure(payload) or payload.get("error", "host state unknown"),
                category=details.get("category", "host_state"), submission_state=details.get("submission_state"),
                retryable=details.get("retryable", False))
        return payload
