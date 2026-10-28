"""Evidence, status projection, and observation support for the coordinator.

[Role]
`ServiceEvidenceMixin` supplies bounded wait/observe/tail reads, evidence and
status projection, explicit placement capacity/topology observations, error and
finish-session reporting to
`CoordinatorService`.

[Invariants]
- Observation reads persisted facts first; a stale cache requests asynchronous
  refresh instead of blocking status reads on remote I/O.
- Evidence is built only from recorded local inputs and receipts, never by
  re-probing a mutable remote root.
- A terminal wait may enrich logs but never replaces newer lifecycle facts.

[Timeouts & Locks]
- Wait deadlines are bounded by `STATUS_CACHE_SECONDS` wakeups and the caller's
  timeout; execution locks are non-blocking for reads.
- Placement alone probes at most eight hosts serially under `_topology_probe_lock`;
  per-probe budgets apply, without an aggregate sampling deadline.
- `DONE`, `LIVE`, `STATUS_CACHE_SECONDS`, and `PERMANENT_ERRORS` are read
  lazily from `service.py` to keep module import acyclic.

[Side-Effect Policy]
- Persists terminal log caches, error logs, and finish-session state through
  `AgentSessions`; evidence reads never validate or mutate remote artifacts.
"""

from __future__ import annotations

import json
import re
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mindie_diagnostics import get_recorder
from remote_dev.core.errors import error_details

from mindie_coordinator.parity_support import RemoteCommandError
from mindie_coordinator.placement import requested_hardware, validate_wait
from mindie_coordinator.topology_gang_common import GangLeaseError


def _service_module():
    """Resolve mutable public service globals only at call time."""
    from mindie_coordinator import service

    return service


class ServiceEvidenceMixin:
    def _observe_host_topologies(self, user, topology):
        """Probe real fleet snapshots for the bounded gang candidate set.

        Fleet probe output is the production schema; its `npus` rows are
        adapted by `topology_from_fleet_snapshot`. Probe failures are reported
        per host and never replaced with an assumed topology.
        """
        from mindie_coordinator.topology_gang_observation import (
            topology_from_fleet_snapshot,
        )

        limit = _service_module().MAX_GANG_PROBE_HOSTS
        records = self._topology_host_records(user, topology)
        wanted = str((topology or {}).get("host") or "").strip()
        limit = len(records) if wanted else limit
        observations = []
        reports: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            host = str((record.get("host") or {}).get("ip") or "").strip()
            if index >= limit:
                reports.append({
                    "host": host, "state": "skipped",
                    "error": f"not probed: bounded observation window of {limit} hosts",
                })
                continue
            try:
                with self._topology_probe_lock:
                    snapshot = self._probe_host_record(record)
            except Exception as exc:
                reports.append({
                    "host": host, "state": "unavailable",
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                })
                continue
            try:
                observation = topology_from_fleet_snapshot(snapshot, host=host)
            except GangLeaseError as exc:
                reports.append({
                    "host": host, "state": "unavailable",
                    "code": exc.code, "error": exc.message[:500],
                })
                continue
            observations.append(observation)
            reports.append({
                "host": host,
                "state": "observed",
                **({"cache_hit": bool(snapshot.get("cache_hit"))}
                   if isinstance(snapshot, dict) else {}),
                **({"cache_age_seconds": snapshot.get("cache_age_seconds")}
                   if isinstance(snapshot, dict) else {}),
                "free_phy_ids": list(observation.free_phy_ids),
                "free_dual_die_pairs": [list(pair) for pair in observation.free_dual_die_pairs],
            })
        return observations, reports

    def _observe_host_specs(self, user, roles, topology):
        """Snapshot candidate host capacity once for explicit hardware floors.

        Reuses the gang candidate ordering and the same `_probe_host_record`
        seam, so one placement decision probes each host at most once and the
        result is carried downstream instead of re-proved. Returns
        `({host: HostHardwareSpec}, reports)`; a host whose probe fails is
        absent from the map and keeps its error in `reports`.

        These are point-in-time feasibility checks, not reservations: nothing
        here holds CPU, RAM or HBM, and per-Die allocation stays with the host
        queue. Probes are serial and bounded only by their own per-call budgets
        (see the module timeout note), so this runs only for roles that
        actually declared a hardware floor.
        """
        from remote_dev.core.host_spec import normalize_remote_probe_spec

        if not any(requested_hardware(role) for role in roles):
            return {}, []
        specs: dict[str, dict[str, Any]] = {}
        reports: list[dict[str, Any]] = []
        probe_limit = _service_module().MAX_GANG_PROBE_HOSTS
        records = self._topology_host_records(user, topology)
        for record in records[:probe_limit]:
            host = str((record.get("host") or {}).get("ip") or "").strip()
            try:
                with self._topology_probe_lock:
                    snapshot = self._probe_host_record(record)
            except Exception as exc:
                reports.append({"host": host, "state": "unavailable", "error": f"{type(exc).__name__}: {exc}"[:500]})
                continue
            if snapshot.get("online") is False:
                reports.append({"host": host, "state": "unavailable",
                                "error": str(snapshot.get("error") or "host is offline")[:500]})
                continue
            specs[host] = normalize_remote_probe_spec(snapshot)
            reports.append({"host": host, "state": "observed",
                            "cache_hit": bool(snapshot.get("cache_hit"))})
        if len(records) > probe_limit:
            reports.extend({"host": str((record.get("host") or {}).get("ip") or "").strip(), "state": "skipped",
                            "error": f"not probed: bounded observation window of {probe_limit} hosts"}
                           for record in records[probe_limit:])
        return specs, reports

    def wait(self, sessions_dir, user, execution_id, *, until="released", timeout_seconds=30, role=None):
        service_module = _service_module()
        validate_wait(until, timeout_seconds)
        deadline = time.monotonic() + timeout_seconds
        while True:
            # Holding the condition across the cached read prevents a save/notify
            # between inspection and sleep. The execution worker owns all remote IO.
            with self._changed:
                reply = self._observe(sessions_dir, user, execution_id, "status", role=role, refresh=False)
                terminal = reply.get("state") in service_module.DONE
                reached = ((until == "running" and (terminal or reply.get("state") == "running"))
                           or (until == "released" and terminal and reply.get("resources_released") is True))
                if reached:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stopped.is_set():
                    break
                # The bounded wake also resumes supervision after a restart or
                # a missed external update. No client-side status RPC loop.
                self._changed.wait(min(remaining, service_module.STATUS_CACHE_SECONDS))
        if terminal:
            if reply.get("resources_released") is True:
                reply = self._wait_terminal_logs(sessions_dir, user, execution_id, reply, deadline, role)
            else:
                # Terminal business state can precede descendant/lease cleanup.
                # Its log is still growing; only the final released view may
                # populate the reusable completion log cache.
                reply = {**reply, "logs_pending": True}
        return {**reply, "wait_until": until, **({"wait_timed_out": True} if not reached else {})}

    def _owned_row(self, sessions_dir, user, execution_id):
        store = self.store(sessions_dir)
        with store.transaction() as db:
            row = store.get(db, "execution", execution_id)
        if user and row.get("user") and row["user"] != user:
            raise PermissionError("execution belongs to another principal")
        return store, row

    def _wait_terminal_logs(self, sessions_dir, user, execution_id, reply, deadline, role):
        store, row = self._owned_row(sessions_dir, user, execution_id)
        lock = self._lock_for("terminal-logs", execution_id)
        if "terminal_logs" not in row and lock.acquire(blocking=False):
            def collect():
                try:
                    _, current = self._owned_row(sessions_dir, user, execution_id)
                    # Another completed waiter may have populated this between reads.
                    if "terminal_logs" in current:
                        return
                    try:
                        value = self._tail(store, user, current)
                        logs = {key: value[key] for key in ("stdout", "stderr", "tail", "preparation_logs") if key in value}
                        logs["roles"] = [{key: item[key] for key in ("name", "stdout", "stderr", "tail") if key in item}
                                         for item in value.get("roles", [])]
                    except Exception as exc:
                        logs = {"tail_error": f"{type(exc).__name__}: {exc}"[:500]}
                    # The progress worker owns this lock for every lifecycle
                    # read/modify/save. Tail IO is outside it; only publication
                    # joins that serialization to preserve newer release facts.
                    with self._lock_for("execution", execution_id):
                        _, current = self._owned_row(sessions_dir, user, execution_id)
                        current["terminal_logs"] = logs
                        self._save_execution(store, current)
                finally:
                    lock.release()
            threading.Thread(target=collect, name=f"mindie-logs-{execution_id[:12]}", daemon=True).start()
        with self._changed:
            while True:
                _, row = self._owned_row(sessions_dir, user, execution_id)
                if "terminal_logs" in row:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {**reply, "logs_pending": True}
                self._changed.wait(remaining)
        # Enrich logs only; the collector cannot replace fresh lifecycle facts.
        reply = self._reply(row, role=role)
        logs = row["terminal_logs"]
        reply.update({key: value for key, value in logs.items() if key != "roles"})
        by_name = {item["name"]: item for item in logs.get("roles", [])}
        for view in reply.get("roles", []):
            view.update(by_name.get(view.get("name"), {}))
        return reply

    def _observe(self, sessions_dir, user, execution_id, action, role=None, refresh=True) -> dict[str, Any]:
        store = self.store(sessions_dir)
        with store.transaction() as db:
            row = store.get(db, "execution", execution_id)
        if user and row.get("user") and row["user"] != user:
            raise PermissionError("execution belongs to another principal")
        user = user or row.get("user")
        if action == "status" and not refresh:
            # Observation reads persisted facts. A stale sample asks the
            # existing execution worker to refresh asynchronously, so neither
            # a slow link nor a dead host blocks status or a bounded wait.
            stale = not self._observation_freshness(row)["fresh"]
            deferred = stale and bool(row.get("admitted")) and not self._reply(row).get("resources_released")
            if deferred:
                self._schedule_progress(sessions_dir, user, execution_id)
            reply = self._reply(row, role=role)
            reply["observation_freshness"].update(
                source="cache" if row.get("jobs_observed_at") else "local",
                refresh_requested=False, refresh_deferred=deferred,
            )
            return reply
        lock = self._lock_for("execution", execution_id)
        acquired = lock.acquire(blocking=False)
        refreshed = False
        try:
            if acquired:
                # Another observer may have refreshed between the first read
                # and lock acquisition. Do not overwrite its newer snapshot.
                with store.transaction() as db:
                    row = store.get(db, "execution", execution_id)
                current = self._observation_freshness(row)["fresh"]
                if row.get("roles") and (action != "status" or refresh or not current):
                    row = self._refresh_jobs(store, user, row)
                    refreshed = bool(row.get("jobs_observed_at"))
                elif row.get("preparation_jobs") and not row.get("roles") and (action != "status" or refresh):
                    self._refresh_preparation_jobs(store, row)
            if action == "tail":
                return self._tail(store, user, row, role=role)
            reply = self._reply(row, role=role)
            reply["observation_freshness"].update(
                source="refreshed" if refreshed else "cache" if row.get("jobs_observed_at") else "local",
                refresh_requested=bool(refresh), refresh_deferred=not acquired,
            )
            return reply
        finally:
            if acquired:
                lock.release()

    def _request_stop(self, sessions_dir, user, execution_id, force) -> dict[str, Any]:
        store = self.store(sessions_dir)
        with store.transaction() as db:
            row = store.get(db, "execution", execution_id)
        if user and row.get("user") and row["user"] != user:
            raise PermissionError("execution belongs to another principal")
        user = user or row.get("user")
        row["cancel_requested"] = True
        if force:
            row["force"] = True
        self._save_execution(store, row)
        lock = self._lock_for("execution", execution_id)
        if lock.acquire(blocking=False):
            try:
                with store.transaction() as db:
                    row = store.get(db, "execution", execution_id)
                return self._stop(store, user, row, force)
            finally:
                lock.release()
        with store.transaction() as db:
            latest = store.get(db, "execution", execution_id)
        latest["cancel_requested"] = True
        if force:
            latest["force"] = True
        return self._reply(latest)

    def _record_execution_error(self, store, row, exc) -> dict[str, Any]:
        permanent = isinstance(exc, _service_module().PERMANENT_ERRORS)
        # A build/preparation command that exited nonzero has a known result.
        # Repeating the same source preparation cannot resolve that failure.
        # Lost SSH transport (255) and already-bound/live work remain recoverable.
        if (isinstance(exc, RemoteCommandError) and exc.returncode != 255
                and row.get("phase") == "preparing" and not row.get("roles")):
            permanent = True
        if any(not job.get("quiet") for records in (row.get("preparation_jobs") or {}).values()
               for job in records.values()):
            permanent = False
        path = self.state_dir / "runs" / row["id"] / "error.log"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)) + "\n")
            row["error_ref"] = str(path)
        except TimeoutError:
            raise
        except OSError as log_exc:
            get_recorder("mindie-coordinator").event("WARNING", "diagnostics.persistence_failed", error_type=type(log_exc).__name__)
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
        details = error_details(exc)
        if isinstance(exc, GangLeaseError):
            details.setdefault("code", exc.code)
            for key, value in exc.details.items():
                if key not in details:
                    details[key] = value
        row["error_details"] = details
        row["phase"] = "failed" if permanent else "uncertain"
        self._save_execution(store, row)
        return self._reply(row)

    def _record_daemon_error(self, message: str) -> None:
        get_recorder("mindie-coordinator").event("ERROR", "daemon.error", message=message)
        path = self.state_dir / "daemon.log"
        try:
            with path.open("a") as stream:
                stream.write(datetime.now(timezone.utc).isoformat() + " " + message[:500] + "\n")
        except TimeoutError:
            raise
        except OSError:
            return

    def _role_view(self, row, role) -> dict[str, Any]:
        job = role.get("observation") or {}
        binding = role.get("binding") or {}
        endpoint = dict(binding.get("endpoint") or {})
        view = {
            "name": role.get("name"),
            "state": job.get("state") or row.get("phase"),
            "runtime_id": role.get("runtime_id"),
            "service_port": job.get("service_port") if job.get("service_port") is not None else role.get("service_port"),
            **({"allow_external_busy": True} if role.get("allow_external_busy") else {}),
            "host": endpoint.get("host"),
            "root": endpoint.get("cwd") or endpoint.get("root"),
            "endpoint": endpoint or None,
            "env": dict(role.get("env") or {}),
            "error": job.get("error"),
            "error_details": job.get("error_details"),
            "lease_state": job.get("lease_state"),
            "quiet": (job.get("remote") or {}).get("quiet"),
            "descendants_drained": (job.get("remote") or {}).get("descendants_drained"),
            "status_observed_at": role.get("status_observed_at"),
            "command_timings": ((job.get("remote") or {}).get("result") or {}).get("timings"),
            "slot_id": job.get("slot_id"),
            "slot_cleanup": job.get("slot_cleanup"),
            "recording_error": job.get("recording_error"),
        }
        if binding.get("endpoint"):
            target = self._target(row, binding, job)
            role_state = job.get("state") or row.get("phase")
            target = {**target, "state": role_state, "live": role_state in _service_module().LIVE}
            view["target"] = target
            view["environment"] = target.get("environment") or {}
            if view["service_port"] is None:
                view["service_port"] = target.get("service_port")
        return view

    @staticmethod
    def _observation_freshness(row):
        service_module = _service_module()
        sampled = row.get("jobs_observed_at")
        age = time.time() - sampled if isinstance(sampled, (int, float)) else None
        return {"snapshot_completed_at": sampled, "age_seconds": round(age, 3) if age is not None else None,
                "max_age_seconds": service_module.STATUS_CACHE_SECONDS,
                "fresh": age is not None and 0 <= age <= service_module.STATUS_CACHE_SECONDS}

    def _reply(self, row, role=None) -> dict[str, Any]:
        service_module = _service_module()
        job = row.get("observation") or {}
        state = row.get("phase") or job.get("state")
        payload = {"execution_id": row["id"], "run_id": row["id"], "state": state, "service": (row.get("spec") or {}).get("service"),
                   "assignment": row.get("assignment"), "observed_at": time.time(),
                   "progress": row.get("progress"), "observation_freshness": self._observation_freshness(row)}
        if row.get("role_progress"):
            payload["role_progress"] = row["role_progress"]
        snapshot = (row.get("spec") or {}).get("source_snapshot")
        if snapshot:
            payload["source_snapshot_id"] = snapshot["id"]
            payload["sources"] = snapshot["sources"]
        if row.get("cancel_requested"):
            payload["cancel_requested"] = True
        if row.get("error"):
            payload["error"] = str(row["error"])[:500]
        if row.get("recording_error"):
            payload["recording_error"] = row["recording_error"]
        if row.get("error_details"):
            payload["error_details"] = row["error_details"]
            payload["error_ref"] = row.get("error_ref")
        if row.get("topology_plan"):
            payload["topology"] = row["topology_plan"]
        if row.get("placement") and state == "waiting_for_runtime":
            payload.update({k: row["placement"].get(k)
                            for k in ("reason", "provisioning_started", "host_observations")
                            if k in (row["placement"] or {})})
        roles = row.get("roles") or []
        from mindie_coordinator.ready_runtime import TERMINAL
        payload["resources_released"] = state in service_module.DONE and all(
            not item.get("managed_job") or (item.get("observation") or {}).get("lease_state") in TERMINAL
            for item in roles) and all(job.get("quiet") and (not job.get("compile_lease")
                or job["compile_lease"].get("state") in {"released", "cancelled", "expired", "not_started"})
                for records in (row.get("preparation_jobs") or {}).values() for job in records.values())
        role_views = [self._role_view(row, item) for item in roles]
        if role:
            role_views = [item for item in role_views if item.get("name") == role]
        if len(roles) > 1 or role_views:
            payload["roles"] = role_views
        binding = row.get("binding") or (roles[0]["binding"] if roles else None)
        first_job = (roles[0].get("observation") if roles else None) or job
        if binding and state in service_module.LIVE | service_module.DONE:
            payload["target"] = self._target(row, binding, first_job)
            payload["service_port"] = payload["target"]["service_port"]
        return payload

    def _tail(self, store, user, row, role=None) -> dict[str, Any]:
        payload = self._reply(row, role=role)
        selected = row.get("roles") or []
        if role:
            selected = [item for item in selected if item.get("name") == role]
        collected = []
        for item in selected:
            if not item.get("managed_job"):
                continue
            observed = self.pool.managed_control(user, item["managed_job"], "tail")
            remote = observed.get("remote") or {}
            stdout = remote.get("stdout") or ""
            stderr = remote.get("stderr") or ""
            text = stdout or stderr or remote.get("tail") or ""
            collected.append({"name": item.get("name"), "stdout": stdout, "stderr": stderr, "tail": text})
        if len(collected) == 1:
            payload["stdout"] = collected[0]["stdout"]
            payload["stderr"] = collected[0]["stderr"]
            payload["tail"] = collected[0]["tail"] or collected[0]["stdout"]
        if payload.get("roles"):
            by_name = {item["name"]: item for item in collected}
            for view in payload["roles"]:
                extra = by_name.get(view.get("name"))
                if extra:
                    view.update(extra)
        if not selected and row.get("role_progress"):
            preparation_logs = []
            for name, progress in row["role_progress"].items():
                if role and name != role:
                    continue
                path = progress.get("log_ref")
                if path and Path(path).is_file():
                    with Path(path).open("rb") as stream:
                        stream.seek(max(0, Path(path).stat().st_size - 32768))
                        preparation_logs.append({"name": name, "step": progress.get("step"),
                                                 "tail": stream.read().decode("utf-8", errors="replace")})
            payload["preparation_logs"] = preparation_logs
            if len(preparation_logs) == 1:
                payload["tail"] = preparation_logs[0]["tail"]
        return payload

    @staticmethod
    def _manifest_summary(manifest):
        return {**{key: manifest[key] for key in ("runtime_root", "profile_key", "build_key", "build_inputs",
                    "preparation", "execution_view", "source_id") if key in manifest},
                "file_count": len(manifest.get("files", {}))}

    def evidence(self, sessions_dir, user, execution_id, *, role=None, section="all", path=None):
        """Read this execution's existing local source/build receipts; never probe a mutable remote root."""
        if section not in {"all", "sources", "preparation", "build", "diagnostics"}:
            raise ValueError("section must be all, sources, preparation, build or diagnostics")
        if path is not None and (not isinstance(path, str) or not path or len(path) > 1000):
            raise ValueError("path must be a nonempty artifact path substring of at most 1000 characters")
        store, row = self._owned_row(sessions_dir, user, execution_id)
        payload = self._reply(row, role=role)
        snapshot = row.get("spec", {}).get("source_snapshot") or {}
        evidence = {"basis": "recorded execution inputs and preparation receipts; no live revalidation"}
        runs = [item["managed_job"] for item in row.get("roles", []) if item.get("managed_job") and (role is None or item.get("name") == role)]
        try:
            diagnostic_events = self.pool.diagnostic_events(user, runs)
        except PermissionError:
            raise
        except Exception as exc:
            diagnostic_events = {"operations": [], "unavailable": type(exc).__name__}
        evidence["diagnostics"] = {"context": row.get("diagnostics_context") or {}, **diagnostic_events}
        if section == "diagnostics":
            from mindie_coordinator.diagnostics import execution_bundle
            try:
                evidence["support_bundle"] = execution_bundle(row, payload, diagnostic_events,
                    self.state_dir / "results" / ("support-" + uuid.uuid4().hex + ".json"))
            except Exception as exc:
                evidence["support_bundle"] = {"unavailable": type(exc).__name__}
        if section in {"all", "sources"}:
            evidence["sources"] = {
                "snapshot_id": snapshot.get("id"), "captured_at": snapshot.get("captured_at"),
                "record_ref": str(store.state_dir / "source-inputs" / (snapshot.get("id", "unknown") + ".json")),
                "repositories": [{key: record[key] for key in ("relpath", "commit", "tree", "source_head", "ref", "build_inputs")
                                  if key in record} for record in snapshot.get("records", [])],
            }
            if row.get("spec", {}).get("script"):
                evidence["script"] = row["spec"]["script"]
        if section in {"all", "preparation"}:
            evidence["preparation"] = {"stages": row.get("stage_history", []),
                "current": row.get("role_progress") or row.get("progress"),
                "phase_observed_at": row.get("phase_observed_at", {}),
                "jobs": [{"name": name, **{key: job[key] for key in ("step", "state", "quiet", "job_id", "observed_at", "stage_timings", "elapsed_seconds", "timings", "command_timings", "transport_totals", "diagnostics_context") if key in job}}
                         for name, jobs in row.get("preparation_jobs", {}).items() if role is None or role == name
                         for job in jobs.values()]}
        if section in {"all", "preparation", "build"}:
            names = list(dict.fromkeys([*(item["name"] for item in row.get("roles", [])),
                          *row.get("role_progress", {}), *row.get("preparation_results", {})]))
            if role:
                names = [name for name in names if name == role]
            evidence["roles"] = [self._role_evidence(row, name, path) for name in names]
        payload["evidence"] = evidence
        return payload

    def _role_evidence(self, row, name, path):
        directory = (self.state_dir / "runs" / row["id"] / name).resolve()
        run_root = (self.state_dir / "runs" / row["id"]).resolve()
        if not directory.is_relative_to(run_root):
            raise ValueError("invalid recorded role log directory")
        result = {"name": name, "logs": []}
        observed = next((role.get("observation") or {} for role in row.get("roles", [])
                         if role.get("name") == name), {})
        for field in ("hardware_telemetry", "native_cache_receipt", "recording_error"):
            if field in observed:
                result[field] = observed[field]
        if name in row.get("native_cache_receipts", {}):
            result["native_cache_receipt"] = row["native_cache_receipts"][name]
        summary = row.get("preparation_results", {}).get(name)
        if summary:
            result["build"] = summary
        # Only this owned execution's direct role logs are considered. Raw
        # receipts are decoded internally, with the backend's existing bound
        # and digest checks; their base64 text never goes to the Agent.
        from mindie_coordinator.backend import _captured_manifest
        manifest = None
        selected = next((item for item in row.get("roles", []) if item.get("name") == name), {})
        binding = selected.get("binding") or row.get("prepared_bindings", {}).get(name) or {}
        runtime_id = selected.get("runtime_id") or binding.get("runtime_id")
        if runtime_id:
            try:
                with self.pool.transaction() as db:
                    runtime = self.pool.get(db, "runtime", runtime_id)
                retained = runtime.get("attestation") or {}
                if self._matching_receipt(row, binding, retained):
                    manifest = retained
                    result["build"] = {**self._manifest_summary(manifest), "runtime_id": runtime_id,
                                       "record_ref": str(self.pool.db_path)}
                else:
                    result["receipt_error"] = "Retained runtime profile differs from this execution's fixed binding"
            except TimeoutError:
                raise
            except (ValueError, OSError) as exc:
                result["receipt_error"] = f"{type(exc).__name__}: {exc}"[:500]
        for log in sorted(directory.glob("*.log"))[:32]:
            if not log.is_file():
                continue
            entry = {"step": log.stem, "log_ref": str(log), "bytes": log.stat().st_size}
            try:
                with log.open("rb") as stream:
                    size = log.stat().st_size
                    limit = 32 * 1024 * 1024
                    stream.seek(max(0, size - limit))
                    text = stream.read(limit).decode("utf-8", errors="replace")
                captured = None
                for line in reversed(text.splitlines()):
                    if not line.startswith('{'):
                        continue
                    try:
                        value = json.loads(line)
                    except ValueError:
                        continue
                    for key in ('preparation_timings', 'native_cache', 'native_smoke_executed'):
                        if key in value:
                            entry.setdefault(key, value[key])
                    store = value.get('native_cache')
                    if isinstance(store, dict):
                        from mindie_coordinator.preparation_cache import native_cache_receipt
                        receipt = native_cache_receipt(store)
                        if re.fullmatch(r'[0-9a-f]{64}', str(receipt.get('cache_key') or '')):
                            current = result.get('native_cache_receipt')
                            if current is None or (not current.get('hit') and not receipt.get('hit')):
                                result['native_cache_receipt'] = receipt
                            elif current != receipt:
                                result['native_cache_receipt_conflict'] = True
                    if "manifest_zlib_base64" in value:
                        captured = _captured_manifest(line)
                    elif value.get("profile_key") and value.get("build_key") and "files" in value:
                        captured = value
                    if captured is not None:
                        break
                if captured is not None:
                    if not self._matching_receipt(row, binding, captured):
                        raise ValueError("profile receipt does not match this execution's source snapshot and binding")
                    manifest = captured
                    result["build"] = {**self._manifest_summary(manifest), "record_ref": str(log)}
                    entry["summary"] = "Captured profile decoded; source snapshot matches"
                else:
                    entry["tail"] = text[-2000:]
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"[:500]
            result["logs"].append(entry)
        if path is not None:
            if manifest is None:
                result["artifact_status"] = "No retained complete profile receipt; artifact hashes unavailable"
            else:
                matches = [(key, value) for key, value in manifest.get("files", {}).items() if path in key]
                result["artifacts"] = [{"path": key, **value} for key, value in matches[:32]]
                result["artifacts_total"] = len(matches)
        return result

    @staticmethod
    def _matching_receipt(row, binding, manifest):
        source_id = (manifest.get("execution_view") or {}).get("source_id") or manifest.get("source_id") or (manifest.get("preparation") or {}).get("source_id")
        endpoint = binding.get("endpoint") or {}
        root = endpoint.get("root") or endpoint.get("cwd")
        return (bool(manifest) and source_id == row.get("spec", {}).get("source_snapshot", {}).get("id")
                and (not binding.get("build_key") or manifest.get("build_key") == binding["build_key"])
                and (not root or manifest.get("runtime_root") == root))

    def _target(self, row, binding, job) -> dict[str, Any]:
        service_module = _service_module()
        state = row.get("phase") or (job or {}).get("state")
        environment = {}
        lease_env = (job or {}).get("environment") or {}
        if lease_env.get("ASCEND_RT_VISIBLE_DEVICES"):
            environment["ASCEND_RT_VISIBLE_DEVICES"] = lease_env["ASCEND_RT_VISIBLE_DEVICES"]
        service_port = (job or {}).get("service_port")
        if service_port is None and lease_env.get("MINDIE_SERVICE_PORT"):
            service_port = int(lease_env["MINDIE_SERVICE_PORT"])
        if service_port is not None:
            environment["MINDIE_SERVICE_PORT"] = str(service_port)
        # A completed job keeps its original attestation even if its binding
        # is subsequently refreshed for another source version.
        launch_observation = dict((job or {}).get("launch_observation") or {})
        return {
            "execution_id": row["id"], "session_id": row["session_id"],
            "runtime_id": binding["runtime_id"], "binding_id": binding["id"],
            "user": binding.get("user"), "container_name": binding.get("container_name"),
            "endpoint": dict(binding["endpoint"]), "host_endpoint": binding.get("host_endpoint"),
            "container_id": binding.get("container_id"), "python": binding.get("python"),
            "profile_key": binding.get("profile_key"), "build_key": binding.get("build_key"),
            "launch_env": binding.get("launch_env") or {},
            "launch_preamble": binding.get("launch_preamble") or "",
            "launch_observation": launch_observation,
            **({"allow_external_busy": True} if (job or {}).get("request", {}).get("allow_external_busy") else {}),
            "environment": environment, "service_port": service_port,
            "state": state, "live": state in service_module.LIVE,
            "assignment": row.get("assignment"),
        }

    def finish(self, sessions_dir, user, session_id, force=False) -> dict[str, Any]:
        service_module = _service_module()
        store = self.store(sessions_dir)
        with store.transaction() as db:
            session = store.get(db, "session", session_id)
            session["state"] = "finishing"
            session["finish"] = {"user": user, "force": bool(force), "at": time.time()}
            store.put(db, "session", session)
        states = []
        for row in store.executions(session_id):
            if row.get("phase") in service_module.DONE:
                continue
            states.append(self.advance(sessions_dir, user, row["id"], action="stop", force=force))
        completed = self._complete_finishing_session(sessions_dir, store, session_id)
        if completed["state"] == "finished":
            return {**completed, "executions": states or completed.get("executions") or []}
        return {"state": "finishing", "executions": states, "worktrees_preserved": True}

    def _resume_finishing_sessions(self, sessions_dir, store) -> None:
        service_module = _service_module()
        for session in store.sessions():
            if session.get("state") != "finishing":
                continue
            intent = session.get("finish") or {}
            user = intent.get("user") or ""
            force = bool(intent.get("force"))
            for row in store.executions(session["id"]):
                if not row.get("admitted") or row.get("phase") in service_module.DONE:
                    continue
                if self._lock_for("execution", row["id"]).locked():
                    continue
                try:
                    self.advance(sessions_dir, user or row.get("user") or "", row["id"],
                                 action="stop", force=force)
                except Exception as exc:
                    self._record_daemon_error(f"finish stop {row.get('id')}: {exc}")
            try:
                self._complete_finishing_session(sessions_dir, store, session["id"])
            except Exception as exc:
                self._record_daemon_error(f"finish {session.get('id')}: {exc}")

    def _complete_finishing_session(self, sessions_dir, store, session_id) -> dict[str, Any]:
        service_module = _service_module()
        with store.transaction() as db:
            session = store.get(db, "session", session_id)
        if session["state"] == "finished":
            return {"state": "finished", "executions": [self._reply(row) for row in store.executions(session_id)],
                    "worktrees_preserved": True}
        if session["state"] != "finishing":
            return {"state": session["state"], "executions": [], "worktrees_preserved": True}
        rows = store.executions(session_id)
        finishing = {"state": "finishing", "executions": [self._reply(row) for row in rows],
                     "worktrees_preserved": True}
        if any(row.get("admitted") and row.get("phase") not in service_module.DONE for row in rows):
            return finishing
        acquired = []
        try:
            for row in sorted(rows, key=lambda item: item["id"]):
                lock = self._lock_for("execution", row["id"])
                if not lock.acquire(blocking=False):
                    return finishing
                acquired.append(lock)
            rows = store.executions(session_id)
            finishing = {"state": "finishing", "executions": [self._reply(row) for row in rows],
                         "worktrees_preserved": True}
            if any(row.get("admitted") and row.get("phase") not in service_module.DONE for row in rows):
                return finishing
            user = (session.get("finish") or {}).get("user") or next(
                (row.get("user") for row in rows if row.get("user")), "")
            seen = set()
            for row in rows:
                bindings = [role.get("binding") for role in row.get("roles") or []]
                bindings.extend((row.get("prepared_bindings") or {}).values())
                if row.get("remote_session"):
                    bindings.extend(self.pool.session_bindings(user, row["remote_session"]["id"]))
                for binding in bindings:
                    if not binding or binding["id"] in seen:
                        continue
                    seen.add(binding["id"])
                    try:
                        self.pool.return_runtime(user, binding["id"])
                    except ValueError:
                        return finishing
                    except PermissionError:
                        continue
            with store.transaction() as db:
                session = store.get(db, "session", session_id)
                session["state"] = "finished"
                store.put(db, "session", session)
            return {"state": "finished", "executions": [self._reply(row) for row in rows],
                    "worktrees_preserved": True}
        finally:
            for lock in acquired:
                lock.release()
