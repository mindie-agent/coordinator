"""Reporting and observation mixin for the shared ready-runtime registry.

[Role]
Houses the read/report surface of `RuntimePool`: run-manifest evidence export,
native-cache receipt recovery, diagnostic events, owner status/catalog queries,
coordination messages, busy checks, and the background observation tick. The
concrete `RuntimePool` in `ready_runtime` supplies persistence and lifecycle.

[Invariants]
- Writes only files under `<state_dir>/runs/` and events/records via the pool's
  existing transactions; never mutates remote host or container state.
- Reads are owner-scoped: another principal's run/session/message is rejected
  before any data is returned.
- Evidence projection is best-effort after terminal facts; local evidence
  failure is isolated and never rewrites lifecycle state.
- `TERMINAL` is defined here and re-exported by `ready_runtime` unchanged.

[Timeouts & Locks]
- Reuses the pool's SQLite `WAL`/`busy_timeout=5000`/`BEGIN IMMEDIATE`
  transactions and per-entity locks; no new lock or timeout is introduced.

[Side-Effect Policy]
- Writes only local report artifacts/events. The background tick delegates all
  host calls to `RuntimePool.control`/`ManagedExecution` without new effects.
"""

from __future__ import annotations

import contextlib
import json
import re
import sqlite3
import threading
import time
import uuid

from mindie_diagnostics import get_recorder

from mindie_coordinator.managed_execution import JOB_TERMINAL
from mindie_coordinator.run_manifest import (
    bench_execution_evidence, project_run_manifest, utc_from_epoch, utc_now, write_manifest,
)

TERMINAL = {"released", "cancelled", "expired"}


class RuntimePoolReportingMixin:
    """Observation/report methods shared by `RuntimePool`."""

    def owned_source_names(self, owner, session):
        with self.transaction() as db:
            return tuple(self.owned(db, "session", session, owner)["sources"])

    def export_execution_record(self, run, binding, job=None):
        """Persist observed execution facts without reading mutable local worktrees.

        Business Run Manifests remain an explicit report format. A CPU command
        has no fabricated Git identity and a shell exit is not model validation.
        """
        with self.transaction() as db:
            if job is None:
                job = self.find(db, "job", run["id"])
        record = {
            "schema_version": "mindie.managed-run.v2",
            "run_id": (job or {}).get("execution_id") or run["id"],
            "managed_run_id": run["id"],
            "created_at": run["created_at"],
            "updated_at": utc_now(),
            "snapshots": dict(run["intent"]["snapshots"]),
            "environment": {"profile_key": binding["profile_key"], "build_key": binding["build_key"],
                            "endpoint": binding["endpoint"]},
            "resources": {"state": run["state"], "task_id": run["task_id"], "epoch": run["epoch"],
                          "released": run["state"] in TERMINAL,
                          "devices": run.get("task", {}).get("granted_devices", []),
                          "service_port": run.get("service_port"),
                          **({"allow_external_busy": True} if run["intent"].get("allow_external_busy") else {})},
            "process": None,
        }
        if job:
            record["process"] = {"job_id": job["job_id"], "state": job["state"],
                                 "runtime_id": binding["runtime_id"],
                                 "command": (job.get("spec") or {}).get("command"),
                                 "remote_dir": job.get("remote", {}).get("remote_dir"),
                                 "result": job.get("remote", {}).get("result")}
            record["slot_id"] = job.get("slot_id")
            record["slot_cleanup"] = job.get("slot_cleanup")
            for field in ("hardware_telemetry", "native_cache_receipt", "recording_error", "proof_error"):
                if field in job:
                    record[field] = job[field]
        self._attach_run_manifest(record, run, binding, job)
        path = self.state_dir / "runs" / (run["id"] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _native_cache_receipt_from_logs(self, run, job):
        receipt = (job or {}).get("native_cache_receipt")
        if isinstance(receipt, dict) and receipt.get("hit") is True and receipt.get("cache_key"):
            return receipt, None
        execution_id = (job or {}).get("execution_id") or run["id"]
        root = self.state_dir / "runs" / execution_id
        found = {}
        try:
            logs = sorted(root.glob("*/*.log"))[:64]
        except TimeoutError:
            raise
        except OSError as exc:
            return None, f"role log scan failed: {exc}"
        for log in logs:
            try:
                lines = log.read_text(encoding="utf-8", errors="replace").splitlines()[-2048:]
            except TimeoutError:
                raise
            except OSError as exc:
                return None, f"role log read failed: {exc}"
            for line in reversed(lines):
                if not line.startswith("{"):
                    continue
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                store = value.get("native_cache") if isinstance(value, dict) else None
                if not isinstance(store, dict):
                    continue
                from mindie_coordinator.preparation_cache import native_cache_receipt
                candidate = native_cache_receipt(store)
                if re.fullmatch(r"[0-9a-f]{64}", str(candidate.get("cache_key") or "")):
                    found[json.dumps(candidate, sort_keys=True)] = candidate
        if len(found) > 1:
            return None, "conflicting cold native-cache receipts in role logs"
        if found:
            return next(iter(found.values())), None
        return (receipt, None) if isinstance(receipt, dict) and receipt.get("cache_key") else (None, None)

    def _attach_run_manifest(self, record, run, binding, job):
        """Project actual terminal facts; local evidence failure is isolated."""
        if not job or job.get("state") not in JOB_TERMINAL:
            return
        status = {"succeeded": "passed", "failed": "failed", "cancelled": "cancelled",
                  "inconclusive": "inconclusive", "timeout": "inconclusive"}.get(job["state"])
        snapshots = run["intent"].get("snapshots") or {}
        if status is None or not snapshots:
            record["run_manifest_unavailable"] = "terminal state or fixed Git source identity is unavailable"
            return
        result = (job.get("remote") or {}).get("result") or {}
        start, end = result.get("started_at"), result.get("finished_at")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or end <= start:
            record["run_manifest_unavailable"] = "remote command start/end timestamps are unavailable"
            return
        source_name = "vllm-ascend" if "vllm-ascend" in snapshots else next(iter(snapshots))
        source_snapshot = (job.get("spec") or {}).get("source_snapshot") or {}
        source_record = next((item for item in source_snapshot.get("records", [])
                              if item.get("relpath") == source_name
                              and item.get("commit") == snapshots[source_name]), None)
        if source_record is None or not source_record.get("source_head"):
            record["run_manifest_unavailable"] = "accepted original Git identity is unavailable; fixed snapshot SHA alone does not prove source_head or dirty state"
            return
        receipt, receipt_error = self._native_cache_receipt_from_logs(run, job)
        if receipt_error:
            record["run_manifest_error"] = {"artifact": "native-cache-receipt", "error": receipt_error}
        proof = job.get("active_execution_proof") if isinstance(job.get("active_execution_proof"), dict) else None
        root = (binding.get("endpoint") or {}).get("root", "")
        profile_ref = root + "/.mindie-runtime/ready-profile.json" if root else None
        try:
            manifest = project_run_manifest(
                run_id=job.get("execution_id") or run["id"],
                code={"source_head": source_record["source_head"],
                      "snapshot_commit": source_record["commit"],
                      "dirty": bool(source_record["changed_paths"])},
                created_at=run["created_at"], started_at=utc_from_epoch(start),
                ended_at=utc_from_epoch(end), terminal_status=status,
                environment=record["environment"],
                workspace_snapshot={"snapshots": snapshots, "source_snapshot": source_snapshot},
                command=[(job.get("spec") or {}).get("command") or ""],
                topology={"host": (binding.get("endpoint") or {}).get("host"),
                          "devices": record["resources"].get("devices") or []},
                hardware_telemetry=job.get("hardware_telemetry"),
                native_cache_receipt=receipt,
                active_execution_proof=proof,
                profile_summary_ref=profile_ref,
            )
            path = self.state_dir / "runs" / (run["id"] + ".run-manifest.json")
            write_manifest(path, manifest)
            record["run_manifest"] = {"path": str(path), "status": manifest["status"],
                                      "run_id": manifest["run_id"]}
            if proof is not None:
                bench = bench_execution_evidence(manifest["active_execution_proof"],
                                                 manifest_run_id=manifest["run_id"])
                bench_path = self.state_dir / "runs" / (run["id"] + ".bench-execution-evidence.json")
                bench_path.write_text(json.dumps(bench, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                record["bench_execution_evidence"] = str(bench_path)
        except TimeoutError as exc:
            record["run_manifest_error"] = {"artifact": "run-manifest", "error": str(exc)[:500], "timeout": True}
        except (OSError, ValueError) as exc:
            record["run_manifest_error"] = {"artifact": "run-manifest", "error": str(exc)[:500]}

    @contextlib.contextmanager
    def _run_operation(self, run, operation):
        """Bounded lifecycle evidence; never record command text or heartbeat polls."""
        started = time.monotonic()
        try:
            with get_recorder("mindie-coordinator").operation("run." + operation, run_id=run["id"]):
                yield
        finally:
            elapsed = time.monotonic() - started
            # This event is a diagnostic projection, not lifecycle authority.
            # A full log disk must not turn a completed release into a failure.
            try:
                with self.transaction() as db:
                    self.event(db, run["owner"], "run-operation", run=run["id"], operation=operation,
                               elapsed_seconds=round(elapsed, 6))
            except Exception as exc:
                get_recorder("mindie-coordinator").event("WARNING", "diagnostics.persistence_failed",
                    error_type=type(exc).__name__, run_id=run["id"])

    def diagnostic_events(self, owner, run_ids, *, limit=128):
        """Read only events for exact owned runs; never query or housekeep a host."""
        keys = list(dict.fromkeys(run_ids))[:64]
        if not keys:
            return {"operations": [], "truncated": False}
        limit = max(1, min(512, int(limit)))
        uri = self.db_path.resolve().as_uri() + "?mode=ro"
        with contextlib.closing(sqlite3.connect(uri, uri=True)) as db:
            for key in keys:
                self.owned(db, "run", key, owner)
            marks = ",".join("?" for _ in keys)
            rows = db.execute("SELECT id,data FROM events WHERE owner=? AND json_extract(data,'$.run') IN (" + marks + ") ORDER BY id DESC LIMIT ?",
                              (owner, *keys, limit + 1)).fetchall()
        return {"operations": [{"cursor": identifier, **json.loads(data)} for identifier, data in reversed(rows[:limit])],
                "truncated": len(rows) > limit, "basis": "recorded local owner events", "clock_domain": "coordinator-owner"}

    def _verify_run_preflight(self, run, runtime, *, managed, container_info=None):
        try:
            compact_verify = getattr(self.backend, "verify_preflight", None) if managed else None
            if callable(compact_verify):
                options = {"_container_info": container_info} if container_info is not None else {}
                if compact_verify(runtime, snapshots=run["intent"]["snapshots"], **options) is not True:
                    raise ValueError("runtime verification did not confirm the registered view")
            else:
                observed = self.backend.inspect(runtime, idle=True, snapshots=run["intent"]["snapshots"])
                if observed != runtime["attestation"]:
                    raise ValueError("runtime changed before launch")
        except Exception as exc:
            if managed:
                # Validation precedes any prepare; its failure survives restart.
                run["preflight_error"] = str(exc)[:500]
            raise

    def tick(self, limit: int = 4, *, exclude_managed=(), background=False):
        """Observe manual leases and supervise explicitly registered jobs."""
        with self.transaction() as db:
            managed = {row["id"] for row in self.rows(db, "job")}
            rows = [row for row in self.rows(db, "run") if row["state"] not in TERMINAL and row["id"] not in managed]
        if background:
            with self.transaction() as db:
                jobs = [row for row in self.rows(db, 'job')
                        if row['state'] not in JOB_TERMINAL and row['id'] not in exclude_managed]
            candidates = [(row, False) for row in rows] + [(row, True) for row in jobs]
            started = 0
            for row, managed in sorted(candidates, key=lambda item: item[0]['last_poll']):
                lock = self._entity_lock('background-tick', row['id'])
                if not lock.acquire(blocking=False):
                    continue
                def advance(row=row, managed=managed, lock=lock):
                    try:
                        if managed:
                            self.managed_advance(row['id'], wait=False)
                        else:
                            self.control(row['owner'], row['id'], 'poll')
                    finally:
                        lock.release()
                threading.Thread(target=advance, name='mindie-pool-' + row['id'][:12], daemon=True).start()
                started += 1
                if started >= limit:
                    break
            return
        for row in sorted(rows, key=lambda row: row["last_poll"])[:limit]:
            self.control(row["owner"], row["id"], "poll")
        self.managed_tick(limit, exclude=exclude_managed)

    def status(self, owner: str):
        with self.transaction() as db:
            return {kind + "s": [row for row in self.rows(db, kind) if row["owner"] == owner]
                    for kind in ("session", "binding", "run", "job")}

    def session_bindings(self, owner: str, session: str):
        with self.transaction() as db:
            self.owned(db, "session", session, owner)
            return [row for row in self.rows(db, "binding")
                    if row["owner"] == owner and row["intent"]["session"] == session and row["state"] == "bound"]

    def peers(self):
        with self.transaction() as db:
            return [{"run": row["id"], "owner": row["owner"], "state": row["state"],
                     "devices": row.get("task", {}).get("granted_devices", [])}
                    for row in self.rows(db, "run") if row["state"] not in TERMINAL]

    def catalog(self):
        with self.transaction() as db:
            rows = []
            for row in self.rows(db, "runtime"):
                profile = (row.get("attestation") or {}).get("profile") or {}
                rows.append({
                    "runtime_id": row["id"], "state": row["state"],
                    "draining": row.get("draining", False),
                    'reuse_only': row.get('reuse_only', False),
                    "user": row.get("user"), "python": row.get("python"),
                    "recipe": row.get("recipe") or profile.get("recipe"),
                    "python_abi": profile.get("python_abi"),
                    "cann": profile.get("cann"), "soc": profile.get("soc"),
                    "machine_type": profile.get("machine_type") or row.get("machine_type"),
                    "profile": profile,
                    "host": row["host_endpoint"]["host"],
                    "host_endpoint": row["host_endpoint"],
                    "endpoint": row["endpoint"],
                    "ssh_port": row["endpoint"]["port"],
                    "container_name": row.get("container_name"),
                    "container_id": (row.get("attestation") or {}).get("container_id"),
                    "root": row["endpoint"]["cwd"],
                    "profile_key": row["attestation"]["profile_key"],
                    "build_key": row["attestation"]["build_key"],
                    "service_ports": row.get("service_ports", []),
                    "error": row.get("error"),
                })
            return rows

    def runtime_busy(self, runtime_id: str, *, except_job: str | None = None) -> bool:
        with self.transaction() as db:
            from mindie_coordinator.managed_execution import JOB_TERMINAL
            bindings = [row for row in self.rows(db, "binding")
                        if row["runtime_id"] == runtime_id and row["state"] == "bound"]
            for binding in bindings:
                for job in self.rows(db, "job"):
                    if job["binding_id"] != binding["id"] or job["state"] in JOB_TERMINAL:
                        continue
                    if except_job and job["id"] == except_job:
                        continue
                    return True
            return False

    def message(self, owner: str, target_run: str, text: str):
        if not text.strip() or len(text) > 4000:
            raise ValueError("message must contain 1..4000 characters")
        with self.transaction() as db:
            run = self.get(db, "run", target_run)
            return self.event(db, run["owner"], "coordination-message", sender=owner, run=target_run, text=text)

    def reply(self, owner: str, cursor: int, text: str):
        if not text.strip() or len(text) > 4000:
            raise ValueError("reply must contain 1..4000 characters")
        with self.transaction() as db:
            row = db.execute("SELECT owner,data FROM events WHERE id=?", (cursor,)).fetchone()
            if row is None or row[0] != owner:
                raise PermissionError("message belongs to another principal")
            message = json.loads(row[1])
            if "sender" not in message:
                raise ValueError("event is not a message")
            return self.event(db, message["sender"], "coordination-reply", sender=owner, reply_to=cursor, text=text)

    def events(self, owner: str, after: int = 0, limit: int = 100):
        with self.transaction() as db:
            rows = db.execute("SELECT id,data FROM events WHERE owner=? AND id>? ORDER BY id LIMIT ?",
                              (owner, after, min(100, max(1, limit)))).fetchall()
            return {"events": [{"cursor": row[0], **json.loads(row[1])} for row in rows],
                    "cursor": rows[-1][0] if rows else after}
