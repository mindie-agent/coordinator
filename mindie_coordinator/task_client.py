"""Local task facade. Hosted work is admitted to the persistent coordinator."""

from __future__ import annotations

from remote_dev.observability import observed_tool
from mindie_diagnostics import get_recorder

import getpass
import hashlib
from pathlib import Path

from mindie_coordinator.agent_session import AgentSessions, load_context
from mindie_coordinator.client_paths import client_path
from mindie_coordinator.placement import normalize_environment, normalize_resources, role_plan, validate_user_env, validate_wait
from mindie_coordinator.ready_runtime import safe_id
from mindie_coordinator.state_paths import coordinator_state_dir
from mindie_coordinator.user_identity import load_github_identity

DONE = {"succeeded", "failed", "timeout", "cancelled", "inconclusive"}


def _user_identity(explicit: str | None = None, *, identity_file=None) -> tuple[str, dict | None]:
    if explicit:
        return safe_id(explicit), None
    identity = load_github_identity(identity_file)
    return (safe_id(identity["login"]), identity) if identity else (safe_id(getpass.getuser()), None)


def coordinator_user(explicit: str | None = None, *, identity_file=None) -> str:
    return _user_identity(explicit, identity_file=identity_file)[0]


class TaskClient:
    """Run and observe managed work from an existing native task context.

    Prefer the existing mindie_run / mindie_execution MCP tools when available.
    For Python callers, this is the same task API:

        from mindie_coordinator.task_client import TaskClient

        client = TaskClient("/local/task-context.json")
        run = client.run(
            '"$MINDIE_PYTHON" -c "import torch_npu; print(1)"',
            sources={"vllm": "/local/vllm", "vllm-ascend": "/local/vllm-ascend"},
            resources={"devices": [0]},
            topology={"host": "npu-host"},
            wait_until="released", wait_timeout_seconds=180,
        )
        execution_id = run["execution_id"]
        print(run["state"], run["resources_released"], run.get("stdout", ""))

    Use the native attachment's supplied context_file path. TaskClient() can
    instead resolve MINDIE_CONTEXT_FILE or the actual native task context; no
    session or attach call is needed before run. Explicit task association
    requires authorization; a directory alone does not identify another task.

    The command runs in the remote execution root. MINDIE_PYTHON is its prepared
    interpreter; source names become subdirectories there and are on PYTHONPATH.
    sources maps names to local Git worktrees and captures current edits at run.
    Omit sources to use task defaults; sources={} has no source dependencies.
    Resources default to no NPU. Use npu_count for any available devices, or
    devices plus topology.host for specific physical devices on a chosen host.
    Only for explicitly intended sharing, add allow_external_busy=True with
    exactly one devices entry; other managed leases still conflict.

    run returns after admission unless wait_until is supplied. Alternatively,
    script_file="/local/business.sh" captures a UTF-8 shell file (up to 1 MiB)
    instead of command. UTF-8 BOM and CRLF are normalized for the remote shell;
    raw and submitted-command digests are recorded. Its local path is not
    part of service identity. CLI: run --script-file business.sh --wait released
    --wait-timeout-seconds 180. Business timeout_seconds is a separate limit.
    wait(until="released") confirms termination and resource release, including
    failed executions: success also requires state == "succeeded". A bounded
    wait may return wait_timed_out=True; inspect it and wait on the same id again
    as needed. This observation timeout does not stop or resubmit the execution.
    Wait budgets are 0-600 seconds. The owner waits internally; no client status
    loop is needed. Terminal wait includes cached stdout/stderr/tail, or an
    explicit tail_error/logs_pending; logs never change lifecycle facts.
    Remote completion still follows the owner's two-second supervision cadence.
    observe(action="evidence", section="build", path="operator") reads recorded
    source/preparation/build receipts and artifact hashes, with raw refs. It
    does not revalidate mutable remote files or prove business correctness.
    observe(action="tail") returns logs (tail and separate stdout/stderr).
    observe(action="status", refresh=False) reads cached progress;
    observe(action="stop") stops that execution. finish() closes the whole task
    and stops its owned executions; it is not needed after each completed run.
    """

    def __init__(self, context_file="", *, pool=None, user=None, service=None, allow_native_context=True,
                 identity_file=None):
        self.context = load_context(context_file, allow_native_context=allow_native_context)
        self.store = AgentSessions(Path(self.context["state_dir"]))
        if self.context["session"].get("user") and not user:
            selected, identity = self.context["session"]["user"], None
        else:
            selected, identity = _user_identity(user, identity_file=identity_file)
        self.context = self.store.bind_user(self.context, selected, github_identity=identity, explicit=bool(user))
        self.user = safe_id(self.context["session"]["user"])
        self._pool = pool
        self._service = service

    @property
    def coordinator(self):
        if self._service is not None:
            return self._service
        if self._pool is not None:
            from mindie_coordinator.service import CoordinatorService
            self._service = CoordinatorService(
                coordinator_state_dir(self.store.state_dir),
                pool=self._pool, backend=self._pool.backend, sessions=self.store,
            )
            return self._service
        from mindie_coordinator.service import ensure_daemon
        self._service = ensure_daemon(coordinator_state_dir(self.store.state_dir))
        return self._service

    @property
    def pool(self):
        owner = self.coordinator
        return owner.pool if hasattr(owner, "pool") else self._pool

    @observed_tool("mindie.session", component="mindie-coordinator")
    def status(self):
        context = self.store.context(self.context["attachment"]["id"])
        with self.store.transaction() as db:
            attachments = [row for row in self.store.rows(db, "attachment") if row["session_id"] == context["session"]["id"]]
        return self._with_notifications({**context, "attachments": attachments,
                                         "executions": self.store.executions(context["session"]["id"])})

    def _with_notifications(self, value):
        session_id = self.context["session"]["id"]
        rows = self.store.executions(session_id)
        has_host = any((binding or {}).get("host_endpoint") for row in rows
                       for binding in [row.get("binding"), *(role.get("binding") for role in row.get("roles") or [])])
        if not has_host:
            with self.store.transaction() as db:
                has_host = any(row.get("session_id") == session_id and row.get("user") == self.user
                               for row in self.store.rows(db, "mailbox"))
        if not has_host:
            return value  # A local session does not start a daemon for mail.
        try:
            return {**value, **self.coordinator.notifications(str(self.store.state_dir), self.user, session_id)}
        except Exception as exc:
            return {**value, "notification_status": {"state": "unavailable", "error": str(exc)[:300]}}

    @observed_tool("mindie.message", component="mindie-coordinator")
    def message(self, recipient, text):
        """Send text to an existing coordination or reply reference.

        Sender identity, host routing, thread and retry bookkeeping are internal.
        This action cannot stop an execution or transfer resource ownership.
        """
        return self.coordinator.message(str(self.store.state_dir), self.user,
                                        self.context["session"]["id"], recipient, text)

    def reply(self, reply_reference, text):
        return self.message(reply_reference, text)

    @observed_tool("mindie.session", component="mindie-coordinator")
    def sources(self, sources):
        self.context = self.store.bind_sources(self.context, sources)
        return self.context

    @observed_tool("mindie.run", component="mindie-coordinator")
    def run(self, command=None, *, script_file=None, sources=None, env=None, environment=None, resources=None, topology=None,
            timeout_seconds=1800, service=None, restart=False, preflight=None,
            wait_until=None, wait_timeout_seconds=30):
        """Admit fixed inputs and resources for one supervised execution.

        ``command`` is remote shell code; use ``"$MINDIE_PYTHON"`` for the prepared
        interpreter. ``sources`` maps source names to local Git worktrees;
        their fixed snapshots become remote subdirectories on Python's path.
        Omit sources for task defaults, or pass ``{}`` for no source inputs.
        ``resources`` defaults to no NPU; ``topology={"host": "npu-host"}``
        selects a host. The returned ``execution_id`` identifies admitted work,
        not a completed command. Read logs with ``observe(action="tail")`` and
        confirm completion with ``wait(until="released")`` on that id.
        Supply ``wait_until`` and ``wait_timeout_seconds`` to combine admission
        and that bounded wait. ``script_file`` replaces ``command`` with the
        UTF-8 text read once from a local file, at most 1 MiB. A BOM is removed
        and CRLF becomes LF; original and submitted command digests are kept.

        ``resources={"devices": [id], "allow_external_busy": True}`` explicitly
        shares one physical NPU with external processes. Other managed leases
        remain exclusive; stopping this execution only stops its own family.
        """
        if wait_until is not None:
            validate_wait(wait_until, wait_timeout_seconds)
        script = None
        if script_file is not None:
            if command is not None:
                raise ValueError("provide exactly one of command or script_file")
            path = Path(client_path(script_file)).expanduser().resolve()
            with path.open("rb") as stream:
                raw = stream.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024 or b'\x00' in raw:
                raise ValueError("script_file must be UTF-8 shell text of at most 1 MiB without NUL bytes")
            command = raw.decode("utf-8-sig").replace("\r\n", "\n")
            script = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                      "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest()}
        if not command or not isinstance(command, str) or not command.strip():
            raise ValueError("command is required")
        env = validate_user_env(env)
        environment = normalize_environment(environment)
        resources = normalize_resources(resources)
        roles = role_plan(topology, resources, command)
        if preflight is not None and (not isinstance(preflight, str) or not preflight.strip()):
            raise ValueError("preflight must be a nonempty shell command")
        from mindie_coordinator.execution_sources import capture_sources
        if sources is None:
            # Resolve this attachment's automatic sources or explicit task
            # override once. Accepted work never consults either mapping again.
            context = self.store.context(self.context["attachment"]["id"])
            defaults = context["source_defaults"]
            if defaults["origin"] == "unknown":
                raise ValueError(defaults["reason"])
            sources = {name: source["path"] for name, source in defaults["sources"].items()}
        with get_recorder("mindie-coordinator").operation("sources.capture"):
            source_snapshot = capture_sources(sources, self.store.state_dir)
        spec = {
            "command": command, "env": env, "environment": environment or {},
            "resources": resources, "topology": topology or {}, "roles": roles,
            "source_snapshot": source_snapshot,
            "timeout_seconds": timeout_seconds, "service": service,
            "preflight": preflight,
        }
        if script is not None:
            spec["script"] = script
        reply = self.coordinator.admit(str(self.store.state_dir), self.user,
                                      self.context["session"]["id"], spec, restart=restart)
        if wait_until is not None and reply.get("execution_id"):
            try:
                return self.wait(reply["execution_id"], until=wait_until, timeout_seconds=wait_timeout_seconds)
            except Exception as exc:
                # Admission has happened. A failed observation must retain its
                # durable reference rather than invite another submission.
                return {**reply, "wait_error": f"{type(exc).__name__}: {exc}"[:500]}
        return self._with_notifications(reply)

    def _require_execution_id(self, execution_id):
        if not isinstance(execution_id, str) or len(execution_id) != 64 or any(
                char not in "0123456789abcdef" for char in execution_id):
            raise ValueError("invalid local execution id")
        with self.store.transaction() as db:
            row = self.store.get(db, "execution", execution_id)
        if row.get("session_id") != self.context["session"]["id"]:
            raise ValueError("execution belongs to another MindIE task")

    @observed_tool("mindie.execution", component="mindie-coordinator")
    def target(self, execution_id):
        self._require_execution_id(execution_id)
        reply = self.coordinator.advance(str(self.store.state_dir), self.user, execution_id, action="target")
        if "target" in reply:
            return reply["target"]
        raise ValueError("execution has no runtime binding; no guessed target")

    def resolve_execution(self, execution_id=None, *, service=None):
        """Resolve one owned reference without starting a daemon or allocating resources."""
        if bool(execution_id) == bool(service):
            raise ValueError("provide exactly one of execution_id or service")
        if execution_id:
            self._require_execution_id(execution_id)
            return execution_id
        if not isinstance(service, str) or not service.strip():
            raise ValueError("service must be a nonempty string")
        rows = [row for row in self.store.executions(self.context["session"]["id"])
                if (row.get("spec") or {}).get("service") == service]
        live = [row for row in rows if row.get("phase") not in DONE]
        if len(live) > 1:
            raise ValueError("service has multiple live executions; use an execution_id")
        selected = live or sorted(rows, key=lambda row: (row.get("created_at", 0), row["id"]))[-1:]
        return selected[0]["id"] if selected else None

    @observed_tool("mindie.execution", component="mindie-coordinator")
    def observe(self, execution_id=None, action="status", force=False, role=None, refresh=True, *, service=None,
                until="released", timeout_seconds=30, section="all", path=None):
        """Read status/logs/target/evidence, wait for, or stop one owned execution.

        Pass the run reply's execution_id, or a task-scoped service name.
        ``action="tail"`` returns ``tail`` and separate ``stdout``/``stderr``
        (per role for multiple roles). Status is fresh by default; use
        ``refresh=False`` for cached progress. Stop requests termination;
        ``wait(until="released")`` confirms resource release afterward.
        ``action="wait"`` uses ``until`` and ``timeout_seconds`` (0-600).
        ``action="evidence"`` reads existing local receipts, optionally using
        ``section`` (all/sources/preparation/build) and artifact ``path`` filter.
        ``section="diagnostics"`` exports a redacted support attachment from
        this execution's retained local events; its file reference is returned.
        """
        if action not in {"status", "tail", "stop", "target", "wait", "evidence"}:
            raise ValueError("unsupported execution action")
        execution_id = self.resolve_execution(execution_id, service=service)
        if execution_id is None:
            return {"state": "not_found", "service": service}
        self._require_execution_id(execution_id)
        if action == "wait":
            return self.wait(execution_id, until=until, timeout_seconds=timeout_seconds, role=role)
        if action == "evidence":
            return self.coordinator.evidence(str(self.store.state_dir), self.user, execution_id,
                                             role=role, section=section, path=path)
        if action == "target":
            target = self.target(execution_id)
            reply = {"execution_id": execution_id, "state": target["state"], "target": target,
                     "service_port": target.get("service_port"), "live": target.get("live")}
            if role:
                reply = self.coordinator.advance(str(self.store.state_dir), self.user, execution_id,
                                                 action="target", force=force, role=role)
            return reply
        reply = self.coordinator.advance(str(self.store.state_dir), self.user, execution_id,
                                        action=action, force=force, role=role,
                                        **({"refresh": False} if action == "status" and not refresh else {}))
        return self._with_notifications(reply) if action == "status" else reply

    @observed_tool("mindie.finish", component="mindie-coordinator")
    def finish(self, force=False):
        local = self.store.close_if_unmanaged(self.context["session"]["id"], user=self.user, force=force)
        if local is not None:
            return local
        return self.coordinator.finish(str(self.store.state_dir), self.user,
                                       self.context["session"]["id"], force=force)

    @observed_tool("mindie.execution", component="mindie-coordinator")
    def wait(self, execution_id, *, until="running", timeout_seconds=30, role=None):
        """Wait on one owned execution; return the last facts on bounded timeout.

        A terminal failure ends a running wait. Release waits end only when
        the coordinator confirms both termination and resource release;
        check ``state == "succeeded"`` separately for business success.
        A timeout returns the last facts plus ``wait_timed_out=True``; inspect
        them and wait on the same id again as needed. This observation timeout
        does not cancel or resubmit the work.
        The owner uses a condition notification and its existing supervision
        cadence; terminal logs are fetched once and retained in the execution.
        """
        validate_wait(until, timeout_seconds)
        self._require_execution_id(execution_id)
        return self._with_notifications(self.coordinator.wait(str(self.store.state_dir), self.user, execution_id,
                                      until=until, timeout_seconds=timeout_seconds, role=role))
