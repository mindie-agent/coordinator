"""Persisted execution supervision on the existing runtime/host authorities.

This is a bounded job state machine, not a workflow engine. Remote-dev owns
process receipts; RuntimePool owns checkouts; the host owns physical NPU leases.
"""
from __future__ import annotations

import re
import json
import shlex
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath
from mindie_diagnostics import get_recorder, wrap_context, current_context, bind_context

from mindie_coordinator.runtime_profile import digest
from remote_dev.core.errors import error_details
from remote_dev.observability import detached_tool
from mindie_coordinator.launch_observation import ENV_NAME, launch_observation

JOB_TERMINAL = {"succeeded", "failed", "timeout", "cancelled", "inconclusive"}
LEASE_TERMINAL = {"released", "cancelled", "expired"}


class ExecutionRequestError(ValueError):
    """A permanent request-validation failure that cannot succeed on retry."""


def task_preamble(binding):
    """Use the bound interpreter and sources for both preflight and launch."""
    command = binding.get("launch_preamble", "")
    if binding.get("python"):
        command += "\nexport MINDIE_PYTHON=" + shlex.quote(binding["python"])
    # Repository directories under the task root otherwise shadow editable packages.
    source_names = binding.get("source_names", ())
    if not source_names:
        return command
    root = PurePosixPath(binding["endpoint"]["cwd"])
    sources = ":".join(str(root / name) for name in (".mindie-runtime/metadata", *source_names))
    return command + "\nexport PYTHONPATH=" + shlex.quote(sources) + '"${PYTHONPATH:+:$PYTHONPATH}"'


class ManagedExecution:
    def managed_start(self, owner, binding_id, request_id, snapshots, expected_build_key,
                      devices, npu_count, command, env, timeout_seconds=1800,
                      priority=0, queue_seconds=1800, service_port=None, hold_go=False,
                      allow_external_busy=False):
        if not isinstance(command, str) or not command.strip() or len(command) > 200000:
            raise ValueError("a bounded nonempty shell command is required")
        if not isinstance(env, dict) or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
                                           or key.startswith("REMOTE_DEV_JOB_")
                                           or key in {"ASCEND_RT_VISIBLE_DEVICES", "MINDIE_SERVICE_PORT", "MINDIE_PYTHON", ENV_NAME}
                                           or not isinstance(value, str) for key, value in env.items()):
            raise ValueError("invalid environment or attempted override of managed device or service-port ownership")
        if timeout_seconds is not None and (type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 86400):
            raise ValueError("timeout_seconds must be None or 1..86400")
        if type(queue_seconds) is not int or not 1 <= queue_seconds <= 86400:
            raise ValueError("queue_seconds must be 1..86400")
        if service_port is not None and (type(service_port) is not int or service_port < 0):
            raise ValueError("service_port must be None, 0, or a positive declared runtime service port")
        if type(allow_external_busy) is not bool or (allow_external_busy and (
                not isinstance(devices, list) or len(devices) != 1 or npu_count
                or any(type(device) is not int or device < 0 for device in devices))):
            raise ValueError("allow_external_busy requires a boolean and exactly one explicit physical device")
        key = digest([owner, binding_id, request_id])
        request = {"binding_id": binding_id, "request_id": request_id, "snapshots": snapshots,
                   "expected_build_key": expected_build_key, "devices": devices, "npu_count": npu_count,
                   "priority": priority, "queue_seconds": queue_seconds, "service_port": service_port}
        if allow_external_busy:
            request["allow_external_busy"] = True
        specification = {"command": command, "env": env, "timeout_seconds": timeout_seconds,
                         "requested_service_port": service_port}
        with self.lock, self.transaction() as db:
            binding = self.owned(db, "binding", binding_id, owner)
            existing = [row for row in self.rows(db, "job") if row["id"] == key]
            if existing:
                job = existing[0]
                if job["request"] != request or job["spec"] != specification:
                    raise ValueError("managed execution request id reused with different arguments")
                if bool(job.get("hold_go")) != bool(hold_go):
                    raise ValueError("managed execution request id reused with different arguments")
                return job
            if binding["state"] != "bound":
                raise ValueError("cannot start a managed job on a returned runtime")
            job = {"id": key, "owner": owner, "binding_id": binding_id,
                   "diagnostics_context": current_context(),
                   "session": binding["intent"]["session"], "request": request, "spec": specification,
                   "job_id": "mindie-" + key, "state": "pending", "last_poll": 0,
                   "cancel_requested": False, "force": False, "hold_go": bool(hold_go)}
            self.put(db, "job", job)  # intent survives a crash before lease creation
            self.event(db, owner, "managed-job-created", run=key, job_id=job["job_id"])
        return self.managed_advance(key)

    def managed_control(self, owner, job_id, action="status", force=False):
        with self.lock, self.transaction() as db:
            job = self.owned(db, "job", job_id, owner)
            binding = self.owned(db, "binding", job["binding_id"], owner)
            runtime = self.get(db, "runtime", binding["runtime_id"])
            if action == "stop":
                job.update(cancel_requested=True, force=bool(job.get("force") or force))
                self.put(db, "job", job)
            elif action not in {"status", "tail"}:
                raise ValueError("managed execution action must be status, tail or stop")
        if action == "tail":
            remote = job.get('remote') or {}
            # Completion exchange already returns a bounded tail. Reuse only
            # this exact drained job, never an earlier/status-only snapshot.
            if (job['state'] in JOB_TERMINAL and job.get('lease_state') in LEASE_TERMINAL
                    and remote.get('state') in JOB_TERMINAL and remote.get('quiet') is True
                    and not remote.get('unknown')
                    and (remote.get('receipt') or {}).get('job_id') == job['job_id']
                    and (remote.get('result') or {}).get('descendants_drained') is True
                    and all(isinstance(remote.get(key), str) for key in ('stdout', 'stderr'))):
                return job
            return {**job, "remote": self.backend.job(runtime, job["job_id"], "tail")}
        return self.managed_advance(job_id) if job["state"] not in JOB_TERMINAL else job

    def managed_release_gate(self, owner, job_id):
        with self.lock, self.transaction() as db:
            job = self.owned(db, "job", job_id, owner)
            job["hold_go"] = False
            self.put(db, "job", job)
        return self.managed_advance(job_id)

    def managed_advance(self, key, *, wait=True, _completed_observation=None):
        # Per-job lock: remote probes/supervision for one job never block
        # another job's advancement; the global lock guards only DB sections.
        lock = self._entity_lock("job", key)
        if not lock.acquire(blocking=wait):
            with self.transaction() as db:
                return self.get(db, "job", key)
        try:
            with self.lock, self.transaction() as db:
                job = self.get(db, "job", key)
                binding = self.owned(db, "binding", job["binding_id"], job["owner"])
                runtime = self.get(db, "runtime", binding["runtime_id"])
                runs = [row for row in self.rows(db, "run") if row["id"] == key]
            correlation = bind_context(job.get("diagnostics_context") or {})
            correlation.__enter__()
            if job["state"] in JOB_TERMINAL:
                return job
            try:
                if not runs:
                    if job["cancel_requested"]:
                        job["state"] = "cancelled"
                        job["runtime_returned"] = False
                        return self._save_managed(job)
                    # Fixed inputs and binding/lease parameters are admitted
                    # locally. A new run can verify inputs immediately before
                    # combined admission/preflight; queue recovery rechecks them.
                    run = self._request_run(job["owner"], **job["request"], check_remote=False, _managed=True)
                elif runs[0]["state"] in {"active", "orphaned_busy"}:
                    # A running job needs one authoritative renewal, below,
                    # after observing its supervisor. Polling the same lease
                    # first repeats a host round trip and an occupancy scan.
                    # Completion goes directly to fenced host release; failed
                    # renewals retain uncertainty and reconcile on the next turn.
                    run = runs[0]
                else:
                    run = self.control(job["owner"], key, "poll", _managed=True)
                if run.get("preflight_error"):
                    job["preflight_error"] = run["preflight_error"]
                job["lease_state"] = run["state"]
                if run.get("environment"):
                    job["environment"] = run["environment"]
                if run.get("service_port") is not None:
                    job["service_port"] = run["service_port"]
                # This owner persists a run before its only prepare path. With
                # no run on entry, no supervisor can have been sent yet. Any
                # resumed run (even without a saved remote receipt) is observed.
                known_absent = (not runs and run["state"] in {"pending", "queued", "granted", "starting", "cancelled"}
                                and not job.get("had_receipt") and not (job.get("remote") or {}).get("receipt"))
                observed = ({"state": "absent", "quiet": True} if known_absent
                            else _completed_observation if _completed_observation is not None
                            else self.backend.job(runtime, job["job_id"], "status"))
                job["had_receipt"] = bool(job.get("had_receipt") or observed.get("receipt")
                                          or (job.get("remote") or {}).get("receipt"))
                if observed.get("state") == "absent" and job["had_receipt"]:
                    # A job directory that existed and then disappeared is lost
                    # supervision, not a drained family. It must never satisfy
                    # completion_confirmed or disable retain_until_release.
                    observed = {**observed, "quiet": False}
                job["remote"] = observed
                self._refresh_managed_cancel(job)
                if run["state"] in {"uncertain", "pending"} and not job.get("preflight_error"):
                    job.update(state="waiting" if run["state"] == "pending" else "uncertain", error=run.get("error"))
                    return self._save_managed(job)

                timed_out = (observed.get("result") or {}).get("state") == "timeout"
                renewed = False
                if timed_out:
                    job["timed_out"] = True
                if (run["state"] == "orphaned_busy" and not job["cancel_requested"]
                        and not job.get("preflight_error") and not timed_out):
                    # Host quarantine is not proof of a dead family. Heartbeat
                    # is the only recovery that moves orphaned_busy back to
                    # active; a live marked family is never stopped here.
                    run = self.control(job["owner"], key, "heartbeat", _managed=True)
                    renewed = run["state"] == "active"
                    job["lease_state"] = run["state"]
                    if run["state"] == "orphaned_busy":
                        job.update(state="uncertain",
                                   error="host quarantines the lease; the live family is preserved")
                        return self._save_managed(job)
                lost_lease = run["state"] in LEASE_TERMINAL
                if job["cancel_requested"] or job.get("preflight_error") or lost_lease or timed_out:
                    if not observed["quiet"]:
                        job["remote"] = self.backend.job(runtime, job["job_id"], "stop", force=job["force"])
                        job["state"] = "stopping"
                        return self._save_managed(job)
                    return self._finish_managed(job, run, binding, runtime, observed)

                if observed["state"] not in {"absent", "prepared", "running"}:
                    if observed["quiet"]:
                        return self._finish_managed(job, run, binding, runtime, observed)
                    job.update(state="uncertain", error="remote process ownership is unresolved")
                    return self._save_managed(job)

                if run["state"] == "queued":
                    job["state"] = "queued"
                    return self._save_managed(job)
                if run["state"] == "granted":
                    run = self.control(job["owner"], key, "preflight", _managed=True)
                    if run.get("preflight_error"):
                        job["preflight_error"] = run["preflight_error"]
                        if not observed["quiet"]:
                            job["remote"] = self.backend.job(runtime, job["job_id"], "stop", force=job["force"])
                            job["state"] = "stopping"
                            return self._save_managed(job)
                        return self._finish_managed(job, run, binding, runtime, observed)
                if run["state"] == "starting":
                    if self._refresh_managed_cancel(job):
                        return self._cancel_managed_launch(job, run, binding, runtime, observed)
                    receipt = launch_observation(binding, job["request"], job["spec"], run["environment"])
                    job["launch_observation"] = receipt
                    command = task_preamble(binding)
                    command += ("\nexport ASCEND_RT_VISIBLE_DEVICES="
                                + shlex.quote(run["environment"]["ASCEND_RT_VISIBLE_DEVICES"]))
                    if run["environment"].get("MINDIE_SERVICE_PORT"):
                        command += "\nexport MINDIE_SERVICE_PORT=" + shlex.quote(run["environment"]["MINDIE_SERVICE_PORT"])
                    command += "\n" + job["spec"]["command"]
                    specification = {**job["spec"], "command": command, "cwd": binding["endpoint"]["cwd"],
                                     "env": {**job["spec"]["env"], **run["environment"],
                                             ENV_NAME: json.dumps(receipt, sort_keys=True)},
                                     "prepared_timeout_seconds": max(120, job["request"]["queue_seconds"])
                                     if job.get("hold_go") else 120}
                    job["service_port"] = run.get("service_port")
                    with self._run_operation(run, "prepare"):
                        observed = self.backend.job(runtime, job["job_id"], "prepare", spec=specification)
                    job["remote"] = observed
                    if observed["state"] != "prepared":
                        raise RuntimeError("start gate has no verified waiting supervisor")
                    run = self.control(job["owner"], key, "activate", _managed=True,
                                       process_guard=observed["receipt"]["process_guard"],
                                       _prepared_receipt=observed["receipt"])
                    renewed = run["state"] == "active"
                if run["state"] == "active":
                    # This renewal belongs to a persisted job, not an idle AI
                    # connection. A manager restart reconciles the same job id.
                    if not renewed:
                        run = self.control(job["owner"], key, "heartbeat", _managed=True)
                    if run["state"] != "active":
                        raise RuntimeError("lease is not active; the start gate remains closed")
                    if observed["state"] == "prepared":
                        if job.get("hold_go"):
                            # A role waiting for its group has a supervised
                            # closed start gate and a renewable active lease.
                            # Do not hold a short-lived unactivated grant while
                            # another host prepares or waits for its devices.
                            job.update(state="waiting", remote=observed, lease_state="active")
                            return self._save_managed(job)
                        else:
                            if self._refresh_managed_cancel(job):
                                return self._cancel_managed_launch(job, run, binding, runtime, observed)
                            authorization = {"run_id": key, "epoch": run["epoch"], "fence": run["task"]["fence_token"]}
                            with self._run_operation(run, "go"):
                                observed = self.backend.job(runtime, job["job_id"], "go", authorization=authorization)
                    elif observed["state"] == "absent":
                        raise RuntimeError("active lease lost its job receipt; do not relaunch")
                    job.update(state="running", remote=observed, lease_state=run["state"])
                else:
                    job.update(state=run["state"], lease_state=run["state"])
                job.pop("error", None)
            except ExecutionRequestError as exc:
                # Deterministic validation failures (bad snapshot, build_key
                # mismatch, an unresolved execution on the binding) can never
                # succeed on retry. Fail terminally instead of wedging the
                # binding in an uncertain poll loop.
                job.update(state="failed", error=str(exc)[:500], error_details=error_details(exc))
            except Exception as exc:
                job.update(state="uncertain", error=str(exc)[:500], error_details=error_details(exc))
            return self._save_managed(job)
        finally:
            if "correlation" in locals():
                correlation.__exit__(None, None, None)
            lock.release()

    def watch_completion(self, key, on_complete, stopped):
        """One owned wait per job, outside all control/lease locks.

        The regular supervision budget still renews leases. This observer only
        accelerates terminal completion; it cannot launch or replay a job.
        """
        if not callable(getattr(self.backend, 'wait_job', None)):
            return
        lock = self._entity_lock('completion-watch', key)
        if not lock.acquire(blocking=False):
            return
        def watch():
            drain_delay = 0.1
            try:
                while not stopped.is_set():
                    with self.transaction() as db:
                        job = self.get(db, 'job', key)
                        binding = self.get(db, 'binding', job['binding_id'])
                        runtime = self.get(db, 'runtime', binding['runtime_id'])
                    if job['state'] in JOB_TERMINAL or job['state'] != 'running':
                        return
                    with bind_context(job.get("diagnostics_context") or {}):
                        observed = self.backend.wait_job(runtime, job['job_id'], timeout_seconds=10)
                    if observed.get('quiet') is True and observed.get('state') not in {'absent', 'prepared', 'running'}:
                        completed = self.managed_advance(key, _completed_observation=observed)
                        on_complete(completed)
                        return
                    if observed.get('state') != 'running':
                        if observed.get('state') in {'absent', 'prepared', 'uncertain', 'lost_outcome'} or observed.get('unknown'):
                            return
                        # Result publication may precede descendant quiet by a
                        # short drain interval. Exchange already wakes on the
                        # result file; back off locally instead of rapid RPCs.
                        if stopped.wait(drain_delay):
                            return
                        drain_delay = min(1.0, drain_delay * 2)
            except Exception as exc:
                # Ordinary status/recovery remains authoritative after a lost
                # wait reply. Never manufacture quiet or resend the command.
                get_recorder("mindie-coordinator").event("WARNING", "completion.observation_failed",
                    run_id=key, error_type=type(exc).__name__, category="observation_unavailable")
                return
            finally:
                lock.release()
        def detached_watch():
            with detached_tool():
                watch()
        threading.Thread(target=wrap_context(detached_watch), name='mindie-completion-' + key[:12], daemon=True).start()

    def _refresh_managed_cancel(self, job):
        with self.transaction() as db:
            current = self.get(db, "job", job["id"])
        if current.get("cancel_requested"):
            job["cancel_requested"] = True
            job["force"] = bool(job.get("force") or current.get("force"))
        return job.get("cancel_requested", False)

    def _cancel_managed_launch(self, job, run, binding, runtime, observed):
        if not observed["quiet"]:
            job["remote"] = self.backend.job(runtime, job["job_id"], "stop", force=job["force"])
            job["state"] = "stopping"
            return self._save_managed(job)
        return self._finish_managed(job, run, binding, runtime, observed)

    def _save_managed(self, job):
        job["last_poll"] = self.clock()
        with self.transaction() as db:
            # Stop intent is monotonic. A concurrent control request may have
            # persisted cancellation while this advancement was doing remote
            # work from an older snapshot; never overwrite it on save.
            current = self.get(db, "job", job["id"])
            if current.get("cancel_requested"):
                job["cancel_requested"] = True
                job["force"] = bool(job.get("force") or current.get("force"))
            if not current.get("hold_go"):
                job["hold_go"] = False
            self.put(db, "job", job)
            runs = [row for row in self.rows(db, "run") if row["id"] == job["id"]]
            binding = self.get(db, "binding", job["binding_id"])
        if runs:
            self.export_execution_record(runs[0], binding, job=job)
        return job

    def _finish_managed(self, job, run, binding, runtime, observed):
        # The remote subreaper reports quiet only after all descendants drain.
        # Host GC cannot substitute an empty marker scan for this observation.
        if not observed["quiet"]:
            raise RuntimeError("cannot release a job without confirmed process completion")
        if run["state"] not in LEASE_TERMINAL:
            action = "cancel" if run["state"] in {"pending", "queued", "granted"} else "release"
            run = self.control(job["owner"], job["id"], action, _managed=True, completion_confirmed=True)
        job["lease_state"] = run["state"]
        if run["state"] not in LEASE_TERMINAL:
            job.update(state="releasing", error=run.get("error"))
            return self._save_managed(job)
        # Execution owns the process family, NPU lease and service port.
        # The task keeps its mutable work root until mindie_finish.
        state = ("failed" if job.get("preflight_error") else "cancelled" if job["cancel_requested"] else "timeout" if job.get("timed_out")
                 else observed.get("state", "lost_outcome"))
        job["state"] = state if state in JOB_TERMINAL else "inconclusive"
        job["remote"] = observed
        job["runtime_returned"] = False
        if job.get("preflight_error"):
            job["error"] = job["preflight_error"]
        elif job["state"] == "inconclusive" and run["state"] == "expired" and not job.get("had_receipt"):
            job["error"] = "lease expired before the command started: " + (run.get("task", {}).get("message") or "activation deadline elapsed")
        else:
            job.pop("error", None)
        return self._save_managed(job)

    def managed_tick(self, limit=4, *, exclude=()):
        with self.transaction() as db:
            jobs = [row for row in self.rows(db, "job") if row["state"] not in JOB_TERMINAL and row["id"] not in exclude]
        pending = [row for row in sorted(jobs, key=lambda item: item["last_poll"])
                   if not self._entity_lock("job", row["id"]).locked()][:limit]
        if not pending:
            return
        # A slow role or an advancement already in progress must not consume
        # another host's lease heartbeat budget.
        with ThreadPoolExecutor(max_workers=min(4, len(pending)), thread_name_prefix="mindie-supervise") as workers:
            list(workers.map(lambda row: self.managed_advance(row["id"], wait=False), pending))
