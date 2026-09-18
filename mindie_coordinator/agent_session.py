"""Local development-task identity, independent of the remote fleet.

Native sessions are attachments, not task ids. A new native root session creates
a new task; resuming that same native session retains its task. Only an explicit
association or a child attachment joins another existing task. No operation here
allocates resources, reads transcripts, resets sources or removes worktrees.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

from mindie_coordinator.state_paths import agent_sessions_root
from mindie_coordinator.client_paths import client_path

CLIENTS = {"claude", "grok", "kimi", "codex", "cursor"}


def git_common_directory(path: str | Path) -> Path:
    """Resolve repository identity with Git, including linked-worktree subdirs."""
    directory = Path(client_path(path)).expanduser().resolve(strict=True)
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", timeout=5, check=True,
    )
    return Path(client_path(result.stdout.strip())).resolve(strict=True)


def source_defaults(session: dict, attachment: dict) -> dict:
    """Select proven explicit task defaults or this native attachment's sources."""
    if session.get("source_mode") == "explicit":
        return {"origin": "explicit", "sources": session.get("sources", {})}
    if session.get("sources"):
        # Previous releases mixed automatic and explicit mappings. Neither a
        # matching path nor a one-entry map establishes which one was intended.
        return {"origin": "unknown", "sources": {},
                "reason": "Saved source defaults have no provenance; set sources explicitly before submitting, "
                          "or pass sources on this run. No legacy mapping was assumed to follow the native cwd."}
    if attachment.get("source_error"):
        return {"origin": "unknown", "sources": {}, "reason": attachment["source_error"]}
    return {"origin": attachment.get("source_mode", "none"), "sources": attachment.get("sources", {})}


def worktree_reference(path: str) -> dict:
    """Inspect an actual repository; never materialize a second source copy."""
    # Git must not inherit the MCP input pipe while its transport reads it.
    # On Windows that can block Git startup and even timeout cleanup.
    source = Path(client_path(path)).expanduser().resolve(strict=True)
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "--show-toplevel"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", timeout=5, check=True,
    )
    root = Path(result.stdout.strip()).resolve()
    info = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--git-common-dir", "HEAD"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", timeout=5, check=True,
    ).stdout.splitlines()
    return {"path": str(root), "git_common_dir": str((root / info[0]).resolve()), "head_at_bind": info[1]}


class AgentSessions:
    def __init__(self, state_dir: Path | None = None):
        self.state_dir = Path(client_path(state_dir or agent_sessions_root())).expanduser().resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db_path = self.state_dir / "sessions.sqlite3"
        with self.transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS records(kind TEXT, id TEXT, data TEXT NOT NULL, PRIMARY KEY(kind,id))")

    @contextlib.contextmanager
    def transaction(self):
        # `with sqlite3.connect(...)` commits or rolls back but never closes.
        with contextlib.closing(sqlite3.connect(self.db_path, timeout=10)) as db:
            with db:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("PRAGMA synchronous=FULL")
                db.execute("BEGIN IMMEDIATE")
                yield db

    @staticmethod
    def get(db, kind, key):
        row = db.execute("SELECT data FROM records WHERE kind=? AND id=?", (kind, key)).fetchone()
        if row is None:
            raise ValueError(f"unknown MindIE {kind}")
        return json.loads(row[0])

    @staticmethod
    def put(db, kind, value):
        db.execute("INSERT OR REPLACE INTO records VALUES(?,?,?)", (kind, value["id"], json.dumps(value, sort_keys=True)))

    @staticmethod
    def rows(db, kind):
        return [json.loads(row[0]) for row in db.execute("SELECT data FROM records WHERE kind=? ORDER BY rowid", (kind,))]

    def context(self, attachment_id: str) -> dict:
        with self.transaction() as db:
            attachment = self.get(db, "attachment", attachment_id)
            session = self.get(db, "session", attachment["session_id"])
        return {"schema_version": "mindie.agent-context.v1", "state_dir": str(self.state_dir),
                "session": session, "attachment": attachment,
                "source_defaults": source_defaults(session, attachment),
                "context_file": str(self.state_dir / "contexts" / (attachment_id + ".json"))}

    def _publish(self, attachment_id):
        context = self.context(attachment_id)
        path = Path(context["context_file"])
        path.parent.mkdir(exist_ok=True, mode=0o700)
        reference = {key: context[key] for key in ("schema_version", "state_dir")} | {"attachment_id": attachment_id}

        def already_published():
            try:
                return json.loads(path.read_text(encoding="utf-8")) == reference
            except (OSError, ValueError):
                return False

        # The file is an immutable reference; current state comes from SQLite.
        # Reusing it also permits Windows readers to keep the file open.
        if already_published():
            return context
        temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                os.chmod(temporary, 0o600)
                json.dump(reference, stream)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.replace(temporary, path)
            except PermissionError:
                # Another process may have published the same reference since
                # our check. Only that complete result makes a denied replace
                # successful; unrelated access failures must still propagate.
                if not already_published():
                    raise
        finally:
            temporary.unlink(missing_ok=True)
        return context

    def attach(self, client: str, native_session_id: str, cwd: str, *, parent_context: str = "",
               association: str = "", agent_id: str = "") -> dict:
        if client not in CLIENTS or not native_session_id or len(native_session_id) > 512:
            raise ValueError("a supported client and its actual native session id are required")
        if parent_context and association:
            raise ValueError("choose child inheritance or an explicit task association")
        parent = load_context(parent_context or association) if parent_context or association else None
        if parent and Path(parent["state_dir"]) != self.state_dir:
            raise ValueError("use the associated task's local registry; do not duplicate its identity")
        now = time.time()
        # Native identity, not cwd/window/PID/history recency, distinguishes a
        # new task from resume. Subagent ids supplement clients that reuse the
        # parent's native session id for child conversations.
        identity = [client, native_session_id, agent_id]
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        actual_cwd = str(Path(client_path(cwd)).expanduser().resolve())
        with self.transaction() as db:
            existing = [row for row in self.rows(db, "attachment") if row["id"] == key]
            if not existing:
                session_id = parent["session"]["id"] if parent else "mindie-" + uuid.uuid4().hex
                if parent:
                    session = self.get(db, "session", session_id)
                    if session["state"] != "open":
                        raise ValueError("task is finished; explicitly reopen it before attaching")
                else:
                    self.put(db, "session", {"id": session_id, "state": "open", "created_at": now,
                                             "sources": {}, "source_mode": "automatic"})
                self.put(db, "attachment", {
                    "id": key, "session_id": session_id, "client": client,
                    "native_session_id": native_session_id, "agent_id": agent_id or None,
                    "parent_id": parent["attachment"]["id"] if parent_context else None,
                    "association": "child" if parent_context else "explicit" if association else "new-task",
                    "cwd": actual_cwd, "state": "attached", "created_at": now,
                })
            else:
                old = existing[0]
                if parent and old["session_id"] != parent["session"]["id"]:
                    raise ValueError("native session already belongs to another task")
                session = self.get(db, "session", old["session_id"])
                if parent_context:
                    # A child resume keeps the fresh-attach invariant: never
                    # join a task whose finish has already started.
                    if session["state"] != "open":
                        raise ValueError("task is finished; explicitly reopen it before attaching")
                elif session["state"] == "finishing":
                    raise ValueError("task is finishing; wait for the coordinator to complete cleanup")
                elif session["state"] == "finished":
                    session["state"] = "open"
                    self.put(db, "session", session)
                if old.get("cwd") != actual_cwd:
                    # A native handoff can move this attachment. Never retain
                    # the previous mutable automatic source after that move.
                    old.pop("sources", None)
                    old.pop("source_mode", None)
                    old.pop("source_error", None)
                old.update(state="attached", resumed_at=now, cwd=actual_cwd)
                self.put(db, "attachment", old)
        return self._publish(key)

    def native_context(self, client: str, native_session_id: str, agent_id: str = "") -> dict:
        with self.transaction() as db:
            matches = [row for row in self.rows(db, "attachment")
                       if row["client"] == client and row["native_session_id"] == native_session_id
                       and (row.get("agent_id") or "") == agent_id and row["state"] == "attached"]
        if len(matches) != 1:
            raise ValueError("native session association is missing or ambiguous; recover by passing the "
                             "task context file explicitly (context_file argument or MINDIE_CONTEXT_FILE)")
        return self.context(matches[0]["id"])

    def bind_user(self, context: dict, user: str, *, github_identity: dict | None = None,
                  explicit: bool = False) -> dict:
        """Bind task attribution once; resumed or associated attachments inherit it."""
        with self.transaction() as db:
            session = self.get(db, "session", context["session"]["id"])
            existing = session.get("user")
            if not existing:
                # Older tasks stored attribution on executions only. Preserve
                # that proven history rather than relabeling accepted work.
                prior = {row["user"] for row in self.rows(db, "execution")
                         if row["session_id"] == session["id"] and row.get("user")}
                if len(prior) > 1:
                    raise ValueError("MindIE task has inconsistent historical user attribution")
                existing = next(iter(prior), None)
            if existing and explicit and existing != user:
                raise ValueError(f"MindIE task is already bound to user {existing!r}; it cannot resume as {user!r}")
            if not session.get("user"):
                session.update(user=existing or user, user_bound_at=time.time())
                if github_identity is not None and (not existing or existing == user):
                    session["github_identity"] = dict(github_identity)
                self.put(db, "session", session)
        return self.context(context["attachment"]["id"])

    def bind_configured_user(self, context: dict, *, identity_file=None) -> dict:
        if context["session"].get("user"):
            return context
        from mindie_coordinator.user_identity import load_github_identity
        identity = load_github_identity(identity_file)
        if identity is None:
            return context
        return self.bind_user(context, identity["login"], github_identity=identity)

    def bind_sources(self, context: dict, sources: dict[str, str]) -> dict:
        references = {}
        for name, path in sources.items():
            if not name or name in {".", ".."} or "/" in name or "\\" in name:
                raise ValueError("source names must be single repository names")
            references[name] = worktree_reference(path)
        with self.transaction() as db:
            session = self.get(db, "session", context["session"]["id"])
            if session["state"] != "open":
                raise ValueError("resume the task before binding sources")
            # Defaults apply only to future submissions. Replace the mapping
            # so {} can deliberately clear it and removed repos do not linger.
            session["sources"] = references
            session["source_mode"] = "explicit"
            self.put(db, "session", session)
        return self.context(context["attachment"]["id"])

    def bind_native_sources(self, context: dict, *, sources: dict[str, str] | None = None) -> dict:
        """Bind consumer-selected roots or native cwd, without changing task defaults."""
        attachment = self.context(context["attachment"]["id"])["attachment"]
        if sources is None and attachment.get("source_error"):
            raise ValueError(attachment["source_error"])
        if sources is None and attachment.get("source_mode") == "native-prepared":
            # A native CLI may resume in its original checkout after the
            # consumer selected an independent editing workspace. attach()
            # already discards automatic sources when the actual cwd changes.
            # Revalidate the selected repositories (including their current
            # HEADs); absence of a new map does not revoke that selection.
            sources = {name: reference["path"] for name, reference in attachment["sources"].items()}
        if sources is None:
            reference = worktree_reference(attachment["cwd"])
            common = Path(reference["git_common_dir"])
            name = common.parent.name if common.name == ".git" else common.stem
            references, source_mode = {name: reference}, "native-cwd"
        else:
            references = {}
            try:
                for name, path in sources.items():
                    if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name or "\\" in name:
                        raise ValueError("source names must be single repository names")
                    references[name] = worktree_reference(path)
            except (OSError, ValueError, subprocess.SubprocessError):
                with self.transaction() as db:
                    current = self.get(db, "attachment", attachment["id"])
                    if current["cwd"] == attachment["cwd"]:
                        current.update(sources={}, source_mode="unknown", source_error=
                                       "Prepared native source roots are unavailable; restore the selected repositories "
                                       "or pass sources explicitly. No empty source set was assumed.")
                        self.put(db, "attachment", current)
                raise
            source_mode = "native-prepared"
        with self.transaction() as db:
            current = self.get(db, "attachment", attachment["id"])
            if current["cwd"] != attachment["cwd"]:
                raise ValueError("native working directory changed while binding its source")
            current.update(sources=references, source_mode=source_mode)
            current.pop("source_error", None)
            self.put(db, "attachment", current)
        return self.context(attachment["id"])

    def detach(self, context: dict) -> dict:
        with self.transaction() as db:
            attachment = self.get(db, "attachment", context["attachment"]["id"])
            attachment.update(state="detached", detached_at=time.time())
            self.put(db, "attachment", attachment)
        # Detaching a frontend never stops a job or releases a lease.
        return self.context(attachment["id"])

    def sessions(self) -> list[dict]:
        with self.transaction() as db:
            return self.rows(db, "session")

    def executions(self, session_id: str) -> list[dict]:
        with self.transaction() as db:
            return [row for row in self.rows(db, "execution") if row["session_id"] == session_id]

    def all_executions(self) -> list[dict]:
        with self.transaction() as db:
            return self.rows(db, "execution")

    def close_if_unmanaged(self, session_id: str, *, user: str, force=False) -> dict | None:
        """Close a task locally only when it has never admitted remote work.

        The same write transaction guards admission's open-state check, so a
        concurrent submit either becomes managed first or sees a closed task.
        """
        with self.transaction() as db:
            session = self.get(db, "session", session_id)
            rows = [row for row in self.rows(db, "execution") if row["session_id"] == session_id]
            remote_facts = ("remote_session", "roles", "managed_job", "binding", "preparation_jobs")
            if any(row.get("admitted") or any(row.get(key) for key in remote_facts) for row in rows):
                return None
            if session["state"] != "finished":
                session["state"] = "finished"
                session["finish"] = {"user": user, "force": bool(force), "at": time.time()}
                self.put(db, "session", session)
            observations = []
            for row in rows:
                row.update(phase="cancelled", cancel_requested=True)
                self.put(db, "execution", row)
                observations.append({"execution_id": row["id"], "state": "cancelled", "resources_released": True})
            return {"state": "finished", "executions": observations, "worktrees_preserved": True}

    def admit_execution(self, session_id: str, request_id: str, spec: dict, *, user: str) -> dict:
        """Publish accepted execution facts atomically with the open-task check."""
        key = hashlib.sha256(json.dumps([session_id, request_id]).encode()).hexdigest()
        with self.transaction() as db:
            if self.get(db, "session", session_id)["state"] != "open":
                raise ValueError("task admission is closed; resume the task before starting another execution")
            existing = [row for row in self.rows(db, "execution") if row["id"] == key]
            if existing:
                row = existing[0]
                if row["spec"] != spec:
                    raise ValueError("execution request id reused with different arguments")
                if row.get("admitted"):
                    if row.get("user") != user:
                        raise PermissionError("execution belongs to another principal")
                    return row
            else:
                row = {"id": key, "session_id": session_id, "request_id": request_id,
                       "spec": spec, "created_at": time.time()}
            from mindie_diagnostics import current_context
            row.update(phase="queued", admitted=True, user=user, diagnostics_context=current_context())
            self.put(db, "execution", row)
            return row

    def execution(self, context: dict, request_id: str, spec: dict) -> dict:
        key = hashlib.sha256(json.dumps([context["session"]["id"], request_id]).encode()).hexdigest()
        with self.transaction() as db:
            matches = [row for row in self.rows(db, "execution") if row["id"] == key]
            if matches:
                if matches[0]["spec"] != spec:
                    raise ValueError("execution request id reused with different arguments")
                return matches[0]
            if self.get(db, "session", context["session"]["id"])["state"] != "open":
                raise ValueError("resume the task before starting another execution")
            row = {"id": key, "session_id": context["session"]["id"], "request_id": request_id,
                   "spec": spec, "phase": "planned", "created_at": time.time()}
            self.put(db, "execution", row)
            return row

    def save_execution(self, row):
        with self.transaction() as db:
            try:
                current = self.get(db, "execution", row["id"])
            except ValueError:
                current = {}
            if current.get("cancel_requested"):
                row["cancel_requested"] = True
            if current.get("force"):
                row["force"] = True
            self.put(db, "execution", row)


def load_context(context_file: str = "", *, allow_native_context: bool = True) -> dict:
    filename = context_file or os.environ.get("MINDIE_CONTEXT_FILE", "")
    if not filename:
        callers = [(client, os.environ.get(key, "").strip()) for client, key in
                   (("codex", "CODEX_THREAD_ID"), ("cursor", "CURSOR_CONVERSATION_ID"),
                    ("grok", "GROK_SESSION_ID"), ("kimi", "KIMI_SESSION_ID"))]
        callers = [(client, native) for client, native in callers if native]
        if len(callers) > 1:
            raise ValueError("conflicting native client identities; pass context_file explicitly")
        if callers and callers[0][0] in {"grok", "kimi"} and allow_native_context:
            client, native = callers[0]
            agent = os.environ.get("KIMI_AGENT_ID", "").strip() if client == "kimi" else ""
            if client == "kimi" and agent == "main":
                agent = ""
            # These clients attach before tools run. A later shell cd does not
            # create a task or change its sources; an unknown child stays unknown.
            return AgentSessions().native_context(client, native, agent)
        # Codex and Cursor expose native ids to local commands even when the
        # session hook cannot export MINDIE_CONTEXT_FILE to their environment.
        # Resolve only that identity; cwd is attachment metadata, never a key.
        client, native = callers[0] if callers else ("codex", "")
        session = os.environ.get("CODEX_SESSION_ID", "").strip() if client == "codex" else ""
        if not native or not allow_native_context:
            raise ValueError("MindIE context is required; use the native session hook or explicit task association")
        if client == "cursor" and str(uuid.UUID(native)) != native:
            # Cursor encodes/truncates arbitrary ids before shell export.
            # Its native UUID is lossless; never guess an encoded identity.
            raise ValueError("unsupported native Cursor shell identity; pass context_file explicitly")
        if session and session != native:
            raise ValueError("conflicting native Codex identities; pass context_file explicitly")
        parent = os.environ.get("MINDIE_PARENT_CONTEXT", "")
        association = os.environ.get("MINDIE_ATTACH_CONTEXT", "")
        if parent or association:
            inherited = load_context(parent or association)
            store = AgentSessions(Path(inherited["state_dir"]))
        else:
            store = AgentSessions()
        try:
            existing = store.native_context(client, native)
        except ValueError:
            context = store.attach(client, native, str(Path.cwd()),
                                   parent_context=parent, association=association)
            try:
                return store.bind_native_sources(context)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                # A local task remains usable before its directory has a Git
                # commit. Match the normal session hook's source boundary.
                print(f"MindIE: source reference not yet bound: {type(exc).__name__}", file=sys.stderr)
                return context
        if parent or association:
            # Preserve association validation without treating a later shell
            # cd as a native worktree handoff or replacing its source defaults.
            return store.attach(client, native, existing["attachment"]["cwd"],
                                parent_context=parent, association=association)
        return existing
    path = Path(client_path(filename)).expanduser().resolve(strict=True)
    reference = json.loads(path.read_text())
    if reference.get("schema_version") != "mindie.agent-context.v1":
        raise ValueError("not a MindIE agent context")
    store = AgentSessions(Path(reference["state_dir"]))
    context = store.context(reference["attachment_id"])
    if Path(context["context_file"]) != path:
        raise ValueError("context path does not match its registered attachment")
    return context
