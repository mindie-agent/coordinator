"""IPC client, daemon bootstrap, and socket loop for the coordinator.

[Role]
Owns `CoordinatorClient`, socket/lock paths and native-owner checks, daemon
identity comparison/startup, `ensure_daemon`, `main`, and the daemon accept/tick
loop bound onto `CoordinatorService`.

[Invariants]
- A client marks IPC submission uncertain as soon as connect succeeds; a lost
  admission reply is never automatically replayed.
- Daemon identity compares loaded runtime packages before reuse or restart.
- The daemon publishes only a complete IPC marker after its listener is ready.

[Timeouts & Locks]
- IPC socket timeout defaults to `CLIENT_TIMEOUT_SECONDS`; waits use
  `max(120, timeout_seconds + 10)`.
- The startup lock uses a 6s deadline and the daemon lock is non-blocking.
- The tick loop reads `TICK_SECONDS` lazily from `service.py`; terminal IPC
  responses keep the daemon-side `CLIENT_TIMEOUT_SECONDS` bound.

[Side-Effect Policy]
- Mutates only coordinator IPC/startup markers and daemon process state; the
  daemon tick loop persists through the service/pool/store owners.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from mindie_diagnostics import bind_context, current_context, get_recorder, wrap_context
from remote_dev.core.errors import RemoteExecutionError, error_details
from remote_dev.observability import detached_tool

from mindie_coordinator.client_paths import client_path
from mindie_coordinator.placement import validate_wait
from mindie_coordinator.state_paths import coordinator_state_dir

LOCK_NAME = "coordinator.lock"
IPC_NAME = "coordinator.ipc"
CLIENT_TIMEOUT_SECONDS = 60.0


def _service_module():
    """Resolve mutable public service globals only at call time."""
    from mindie_coordinator import service

    return service


def socket_path(state_dir: Path) -> Path:
    """Short per-user/state socket. macOS AF_UNIX paths cap near 104 bytes."""
    if os.name == "nt":
        return Path(state_dir) / IPC_NAME
    resolved = str(Path(state_dir).expanduser().resolve())
    digest = hashlib.sha256(resolved.encode()).hexdigest()[:16]
    user = "".join(ch if ch.isalnum() else "-" for ch in getpass.getuser())[:12] or "user"
    return Path("/tmp") / f"vc-{user}-{digest}.sock"


def _lock_daemon(handle, *, release=False):
    if os.name == "nt":
        import msvcrt

        if not release and os.fstat(handle.fileno()).st_size == 0:
            handle.write(b" ")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK if release else msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle, fcntl.LOCK_UN if release else fcntl.LOCK_EX | fcntl.LOCK_NB)


def lock_path(state_dir: Path) -> Path:
    return Path(state_dir) / LOCK_NAME


def require_native_owner(state_dir: Path, *, platform: str | None = None) -> None:
    if (platform or os.name) == "nt":
        return
    try:
        address = json.loads((Path(state_dir) / IPC_NAME).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    except TimeoutError:
        raise
    except (OSError, ValueError) as exc:
        raise RuntimeError("cannot determine coordinator ownership from its IPC marker") from exc
    if isinstance(address, dict) and "port" in address:
        raise RuntimeError(
            "this shared state directory belongs to a Windows coordinator; configure the WSL "
            "coordinator MCP and session hooks to invoke the managed Windows Python executable "
            "on /mnt/<drive>. A second Linux coordinator cannot manage the same state."
        )


class CoordinatorClient:
    def __init__(self, state_dir: Path, *, timeout=120):
        self.state_dir = Path(client_path(state_dir)).expanduser().resolve()
        self.timeout = timeout

    def call(self, op: str, **payload) -> Any:
        payload["diagnostics_context"] = current_context()
        service_module = _service_module()
        path = service_module.socket_path(self.state_dir)
        if not path.exists():
            raise RuntimeError(f"coordinator daemon is not running at {path}")
        if os.name == "nt":
            try:
                address = json.loads(path.read_text(encoding="utf-8"))
                port, token = int(address["port"]), address["token"]
                if not 0 < port < 65536 or not isinstance(token, str) or not token:
                    raise ValueError("invalid port or token")
            except (ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(f"invalid coordinator IPC marker at {path}") from exc
            payload["_ipc_token"] = token
            target = ("127.0.0.1", port)
        else:
            target = str(path)
        conn = service_module.socket.socket(
            socket.AF_INET if os.name == "nt" else socket.AF_UNIX, socket.SOCK_STREAM
        )
        conn.settimeout(max(self.timeout, payload.get("timeout_seconds", 0) + 10) if op == "wait" else self.timeout)
        submission_state = "not_sent"
        try:
            conn.connect(target)
            submission_state = "uncertain"
            conn.sendall((json.dumps({"op": op, **payload}) + "\n").encode())
            data = b""
            while b"\n" not in data:
                chunk = conn.recv(1 << 16)
                if not chunk:
                    break
                data += chunk
        except TimeoutError:
            raise
        except OSError as exc:
            try:
                conn.close()
            except TimeoutError:
                raise
            except OSError:
                pass
            raise RemoteExecutionError("coordinator IPC response unavailable; do not resubmit an uncertain admission",
                category="ipc_transport", submission_state=submission_state,
                retryable=submission_state == "not_sent") from exc
        try:
            if not data or b"\n" not in data:
                raise RemoteExecutionError("coordinator IPC reply was lost; original request outcome is uncertain",
                    category="ipc_response", submission_state="uncertain")
            try:
                reply = json.loads(data.decode())
            except (ValueError, UnicodeError) as exc:
                raise RemoteExecutionError("coordinator IPC reply was invalid; original request outcome is uncertain",
                    category="ipc_protocol", submission_state="uncertain") from exc
        finally:
            try:
                conn.close()
            except TimeoutError:
                raise
            except OSError:
                pass
        if not reply.get("ok"):
            detail = reply.get("error_details") or {}
            raise RemoteExecutionError(reply.get("error") or "coordinator request failed",
                                       category=detail.get("category", "coordinator"),
                                       submission_state=detail.get("submission_state"), retryable=detail.get("retryable", False))
        return reply.get("value")

    def admit(self, sessions_dir, user, session_id, spec, restart=False):
        # A long-lived Python client must also retry an earlier busy upgrade.
        # Existing execution controls bypass this admission-only boundary.
        service_module = _service_module()
        owner = service_module.ensure_daemon(self.state_dir)
        self.runtime = owner.runtime
        if not _daemon_runtime_matches(self):
            return _runtime_update(service_module.LOADED_RUNTIMES, self.runtime, owner.runtime_update)
        return self.call("admit", sessions_dir=str(sessions_dir), user=user,
                         session_id=session_id, spec=spec, restart=restart,
                         client_runtime=service_module.LOADED_RUNTIMES)

    def advance(self, sessions_dir, user, execution_id, action="status", force=False, role=None, refresh=True):
        payload = dict(sessions_dir=str(sessions_dir), user=user,
                       execution_id=execution_id, action=action, force=force)
        if role:
            payload["role"] = role
        if not refresh:
            payload["refresh"] = False
        return self.call("advance", **payload)

    def wait(self, sessions_dir, user, execution_id, *, until="released", timeout_seconds=30, role=None):
        validate_wait(until, timeout_seconds)
        return self.call("wait", sessions_dir=str(sessions_dir), user=user, execution_id=execution_id,
                         until=until, timeout_seconds=timeout_seconds, role=role)

    def evidence(self, sessions_dir, user, execution_id, *, role=None, section="all", path=None):
        return self.call("evidence", sessions_dir=str(sessions_dir), user=user, execution_id=execution_id,
                         role=role, section=section, path=path)

    def finish(self, sessions_dir, user, session_id, force=False):
        return self.call("finish", sessions_dir=str(sessions_dir), user=user,
                         session_id=session_id, force=force)

    def runtime_register(self, runtime_id, spec):
        return self.call("runtime_register", runtime_id=runtime_id, spec=spec)

    def notifications(self, sessions_dir, user, session_id):
        return self.call("notifications", sessions_dir=str(sessions_dir), user=user, session_id=session_id)

    def message(self, sessions_dir, user, session_id, recipient, text):
        return self.call("message", sessions_dir=str(sessions_dir), user=user, session_id=session_id,
                         recipient=recipient, text=text)


def _runtime_identity(rows):
    return {row["package"]: {key: row.get(key) for key in ("version", "commit", "python", "location")}
            for row in rows if row.get("package")}


def _daemon_runtime_matches(client: CoordinatorClient) -> bool:
    service_module = _service_module()
    return _runtime_identity(service_module.LOADED_RUNTIMES) == _runtime_identity(
        [row.get("loaded") or {} for row in client.runtime or []])


def _runtime_update(selected, daemon, facts):
    return {"state": "needs_runtime_update", "reason": facts.get("reason", "daemon update is pending"),
            "runtime_update": {"selected": selected, "daemon": daemon},
            **({"active_executions": facts["active_executions"]} if "active_executions" in facts else {})}


def ensure_daemon(state_dir: Path) -> CoordinatorClient:
    service_module = _service_module()
    state_dir = Path(client_path(state_dir)).expanduser().resolve()
    service_module.require_native_owner(state_dir)
    client = service_module.CoordinatorClient(state_dir)
    try:
        client.runtime = (client.call("ping") or {}).get("runtime")
        if _daemon_runtime_matches(client):
            return client
    except TimeoutError:
        raise
    except (RuntimeError, FileNotFoundError, ConnectionError, OSError):
        pass
    Path(state_dir).mkdir(parents=True, exist_ok=True, mode=0o700)
    # A separate short-lived startup lock avoids spawning losing daemons that
    # are still importing when the caller has already received another's ping.
    with (Path(state_dir) / "coordinator.start.lock").open("a+b") as guard:
        deadline = time.monotonic() + 6
        while True:
            try:
                service_module._lock_daemon(guard)
                break
            except TimeoutError:
                raise
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"coordinator startup is still in progress at {state_dir}")
                time.sleep(0.05)
        try:
            return _ensure_daemon_locked(state_dir, client)
        finally:
            service_module._lock_daemon(guard, release=True)


def _ensure_daemon_locked(state_dir: Path, client: CoordinatorClient) -> CoordinatorClient:
    service_module = _service_module()
    service_module.require_native_owner(state_dir)
    try:
        client.runtime = (client.call("ping") or {}).get("runtime")
    except TimeoutError:
        raise
    except (RuntimeError, FileNotFoundError, ConnectionError, OSError):
        pass
    else:
        if _daemon_runtime_matches(client):
            return client
        try:
            reply = client.call("restart_if_idle")
        except TimeoutError:
            raise
        except (RuntimeError, FileNotFoundError, ConnectionError, OSError) as exc:
            client.runtime_update = {"reason": f"daemon idle restart unavailable: {exc}"}
            return client
        if not isinstance(reply, dict) or reply.get("status") != "stopping":
            client.runtime_update = reply if isinstance(reply, dict) else {"reason": "daemon idle restart returned no status"}
            return client
        deadline = time.monotonic() + 5
        while service_module.socket_path(state_dir).exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        if service_module.socket_path(state_dir).exists():
            raise RuntimeError("daemon shutdown is still pending")
    log_path = Path(state_dir) / "daemon.log"
    environment = dict(os.environ)
    environment.setdefault("REMOTE_DEV_STATE_DIR", str(Path(state_dir).resolve().parent / "remote-dev-state"))
    options = ({"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
               if os.name == "nt" else {"start_new_session": True})
    with log_path.open("ab") as log:
        process = service_module.subprocess.Popen(
            [sys.executable, "-m", "mindie_coordinator", "daemon", "--state-dir", str(state_dir)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=environment, **options,
        )
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            client.runtime = (client.call("ping") or {}).get("runtime")
            return client
        except TimeoutError:
            raise
        except (RuntimeError, FileNotFoundError, ConnectionError, OSError):
            if process.poll() is not None:
                detail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
                if "coordinator already running for" not in detail:
                    raise RuntimeError(f"coordinator daemon exited ({process.returncode}): {detail}")
            time.sleep(0.05)
    raise RuntimeError(f"could not start coordinator daemon at {state_dir}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    service_module = _service_module()
    parser = argparse.ArgumentParser(description="Run the persistent local MindIE coordinator")
    parser.add_argument("--state-dir", default="", help="Coordinator state directory")
    parser.add_argument("--action", choices=("serve", "status", "restart-if-idle"), default="serve")
    args = parser.parse_args(argv)
    state = Path(client_path(args.state_dir)).expanduser().resolve() if args.state_dir else coordinator_state_dir()
    if args.action == "status":
        print(json.dumps(service_module.CoordinatorClient(state).call("ping")))
        return 0
    if args.action == "restart-if-idle":
        reply = service_module.CoordinatorClient(state).call("restart_if_idle")
        if reply["status"] == "busy":
            print(json.dumps(reply))
            return 1
        deadline = time.monotonic() + 5
        while service_module.socket_path(state).exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        if service_module.socket_path(state).exists():
            raise RuntimeError("daemon shutdown is still pending")
        print(json.dumps({"status": "restarted", "runtime": service_module.ensure_daemon(state).runtime}))
        return 0
    return service_module.CoordinatorService(state).serve()


def serve(service_instance) -> int:
    lock = lock_path(service_instance.state_dir).open("a+b")
    try:
        _lock_daemon(lock)
    except TimeoutError:
        raise
    except OSError:
        lock.close()
        raise RuntimeError(f"coordinator already running for {service_instance.state_dir}")
    path = socket_path(service_instance.state_dir)
    server = None
    marker = service_instance.state_dir / IPC_NAME
    temporary = marker.with_name(marker.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        path.unlink(missing_ok=True)
        windows = os.name == "nt"
        server = socket.socket(socket.AF_INET if windows else socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(("127.0.0.1", 0) if windows else str(path))
        except TimeoutError:
            raise
        except OSError as exc:
            raise RuntimeError(f"coordinator socket bind failed at {path} ({len(str(path))} bytes): {exc}") from exc
        if not windows:
            os.chmod(path, 0o700)
        service_instance._ipc_token = secrets.token_hex(32) if windows else None
        address = {"host": "127.0.0.1", "port": server.getsockname()[1],
                   "token": service_instance._ipc_token} if windows else {"socket": str(path)}
        server.listen(16)
        server.settimeout(1.000_000)
        # Publish only a complete record after the listener is ready.
        temporary.write_text(
            json.dumps({**address, "state_dir": str(service_instance.state_dir.resolve())}) + "\n", encoding="utf-8"
        )
        temporary.chmod(0o600)
        os.replace(temporary, marker)
        print(f"mindie-coordinator listening socket={path} state={service_instance.state_dir} lock={lock_path(service_instance.state_dir)}",
              flush=True)
        service_instance._async_progress = True
        ticker = threading.Thread(target=service_instance._tick_loop, name="mindie-coordinator-tick", daemon=True)
        ticker.start()
        while not service_instance._stopped.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            threading.Thread(target=service_instance._serve_conn, args=(conn,), daemon=True).start()
    finally:
        service_instance._stopped.set()
        service_instance.close_messages()
        if server is not None:
            server.close()
        try:
            path.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)
        finally:
            try:
                _lock_daemon(lock, release=True)
            finally:
                lock.close()
    return 0

def _tick_loop(service_instance) -> None:
    service_module = _service_module()
    while not service_instance._stopped.wait(service_module.TICK_SECONDS):
        try:
            service_instance._dispatch_progress()
        except Exception as exc:
            service_instance._record_daemon_error(f"tick: {type(exc).__name__}: {exc}")

def _dispatch_progress(service_instance) -> None:
    with service_instance._lifecycle_lock:
        if service_instance._stopped.is_set():
            return
        service_instance._active_requests += 1
    try:
        service_instance._dispatch_progress_active()
    finally:
        with service_instance._lifecycle_lock:
            service_instance._active_requests -= 1

def _dispatch_progress_active(service_instance) -> None:
    service_module = _service_module()
    sessions = []
    execution_jobs = set()
    for directory in list(service_instance._session_dirs):
        try:
            store = service_instance.store(directory)
            executions = store.all_executions()
            sessions.append((directory, store, executions))
            execution_jobs.update(role["managed_job"] for row in executions
                                  if row.get("admitted") and row.get("phase") not in service_module.DONE
                                  and not service_instance._lock_for("execution", row["id"]).locked()
                                  for role in row.get("roles", []) if role.get("managed_job"))
        except Exception as exc:
            service_instance._record_daemon_error(f"sessions {directory}: {exc}")
            continue
    # Each execution worker supervises its own roles. The pool services
    # standalone jobs and free roles of busy executions. One stalled host
    # must not consume healthy siblings' renewal budget while their shared
    # execution worker remains busy. Per-job locks prevent overlap.
    service_instance.pool.tick(exclude_managed=execution_jobs, background=True)
    for directory, store, executions in sessions:
        finishing = {session["id"] for session in store.sessions() if session.get("state") == "finishing"}
        for row in executions:
            if row.get("session_id") in finishing:
                continue
            if not row.get("admitted") or row.get("phase") in service_module.DONE:
                continue
            lock = service_instance._lock_for("execution", row["id"])
            if lock.locked():
                continue
            thread = threading.Thread(
                target=wrap_context(service_instance._tick_one), args=(directory, row, lock),
                name=f"mindie-progress-{row['id'][:12]}", daemon=True,
            )
            thread.start()
        service_instance._resume_finishing_sessions(directory, store)

def _tick_one(service_instance, directory, row, lock: threading.Lock) -> None:
    with bind_context(row.get("diagnostics_context") or {}), detached_tool():
        return service_instance._tick_one_correlated(directory, row, lock)

def _tick_one_correlated(service_instance, directory, row, lock: threading.Lock) -> None:
    if not lock.acquire(blocking=False):
        return
    try:
        service_instance._advance_locked(directory, row.get("user") or "", row["id"])
    except Exception as exc:
        try:
            store = service_instance.store(directory)
            service_instance._record_execution_error(store, row, exc)
        except Exception as record_exc:
            service_instance._record_daemon_error(f"execution {row.get('id')}: {exc}; persist {record_exc}")
    finally:
        lock.release()
    if service_instance._async_progress:
        try:
            store = service_instance.store(directory)
            with store.transaction() as db:
                current = store.get(db, 'execution', row['id'])
            for role in current.get('roles', []):
                if role.get('managed_job') and (role.get('observation') or {}).get('state') == 'running':
                    service_instance.pool.watch_completion(role['managed_job'],
                        lambda _job: service_instance.advance(directory, row.get('user') or '', row['id'], action='progress'), service_instance._stopped)
        except Exception as exc:
            service_instance._record_daemon_error('completion observer: ' + str(exc))

def _serve_conn(service_instance, conn: socket.socket) -> None:
    conn.settimeout(CLIENT_TIMEOUT_SECONDS)
    try:
        data = b""
        while True:
            chunk = conn.recv(1 << 16)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        if not data.strip():
            return
        try:
            request = json.loads(data.decode())
            if service_instance._ipc_token is not None and not secrets.compare_digest(
                str(request.pop("_ipc_token", "")), service_instance._ipc_token
            ):
                raise PermissionError("coordinator IPC authentication failed")
            reply = service_instance.handle(request)
        except Exception as exc:
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "error_details": error_details(exc)}
        conn.sendall((json.dumps(reply, default=str) + "\n").encode())
    except TimeoutError:
        raise
    except (OSError, socket.timeout, json.JSONDecodeError) as exc:
        get_recorder("mindie-coordinator").event("WARNING", "ipc.response_unavailable", error_type=type(exc).__name__)
        return
    finally:
        try:
            conn.close()
        except TimeoutError:
            raise
        except OSError:
            pass
