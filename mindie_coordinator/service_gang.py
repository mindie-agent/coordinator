"""Gang placement, preparation, and lease orchestration for the coordinator.

[Role]
`ServiceGangMixin` supplies topology observation/planning, role placement and
preparation, and managed-lease start/stop orchestration to `CoordinatorService`.

[Invariants]
- Topology-gang roles are planned only from observed host/card snapshots and
  must bind exactly those hosts/devices; count-only plans never launch.
- Group start authorization is persisted only after every role holds an active
  lease and a prepared/running job.
- Preparation job references are retained and never silently replayed.

[Timeouts & Locks]
- `GANG_PLAN_RETRY_SECONDS` gates retryable topology wait retries.
- `_topology_probe_lock` serializes fleet probes; `_map_roles` is bounded to
  four workers. Preparation runs in at most four workers.
- `STOP_WAIT_SECONDS` is read lazily from `service.py` so callers patching the
  public service module retain control of stop waits.

[Side-Effect Policy]
- Persists execution progress/preparation/lease facts through the coordinator
  store and `RuntimePool`; fleet probes are read-only `npu-smi` observations.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from mindie_diagnostics import get_recorder

from mindie_coordinator.machine_directory import MachineDirectoryUnavailable
from mindie_coordinator.placement import (
    can_prepare,
    distinct_hosts_required,
    host_key,
    plan_gang_placement,
    provisionable_recipe,
    runtime_matches,
)
from mindie_coordinator.preparation_process import (
    PreparationCancelled,
    PreparationUncertain,
    stop_preparation_process,
)
from mindie_coordinator.provision.task_environment import TaskRootBusy
from mindie_coordinator.ready_runtime import user_container_name
from mindie_coordinator.topology_gang_common import GangLeaseError

# A gang refusal that can become satisfiable when another lease releases or a
# host comes back is retried on the ordinary progress cadence, but not faster
# than this window, so a waiting execution never becomes an SSH probe loop.
GANG_PLAN_RETRY_SECONDS = 15.0
GANG_RETRYABLE_CODES = frozenset({
    "insufficient_free_dies",
    "insufficient_dual_die_pairs",
    "insufficient_nodes",
    "requested_dies_unavailable",
    "requested_host_unobserved",
    "host_observation_required",
    "host_observation_unavailable",
    "observation_unavailable",
    "observation_invalid",
})
MAX_GANG_PROBE_HOSTS = 8


def _service_module():
    """Resolve mutable public service globals only at call time."""
    from mindie_coordinator import service

    return service


class ServiceGangMixin:
    def _advance_locked(self, sessions_dir, user, execution_id, action="progress", force=False) -> dict[str, Any]:
        store = self.store(sessions_dir)
        with store.transaction() as db:
            row = store.get(db, "execution", execution_id)
        if user and row.get("user") and row["user"] != user:
            raise PermissionError("execution belongs to another principal")
        user = user or row.get("user")
        if row.get("phase") in _service_module().DONE:
            if not self._reply(row).get("resources_released"):
                # Pool cleanup can complete after the business state becomes
                # terminal. Refresh its owned facts without restarting work.
                row = self._refresh_jobs(store, user, row)
                self._refresh_preparation_jobs(store, row)
            return self._reply(row)
        with store.transaction() as db:
            session = store.get(db, "session", row["session_id"])
        if session.get("state") == "finishing" or row.get("cancel_requested"):
            force = force or bool(row.get("force")) or bool((session.get("finish") or {}).get("force"))
            return self._stop(store, user, row, force)
        try:
            return self._progress(store, user, row)
        except Exception as exc:
            return self._record_execution_error(store, row, exc)

    def _adopt_cancel(self, store, row) -> bool:
        with store.transaction() as db:
            latest = store.get(db, "execution", row["id"])
            session = store.get(db, "session", row["session_id"])
        if latest.get("cancel_requested"):
            row["cancel_requested"] = True
        if latest.get("force"):
            row["force"] = True
        if session.get("state") == "finishing":
            row["cancel_requested"] = True
            if (session.get("finish") or {}).get("force"):
                row["force"] = True
        return bool(row.get("cancel_requested"))

    def _halt_if_cancelled(self, store, user, row):
        if self._adopt_cancel(store, row):
            return self._stop(store, user, row, bool(row.get("force")))
        return None

    def _probe_host_record(self, record):
        """Return one fleet-probe snapshot; tests replace this exact seam."""
        from mindie_coordinator.fleet_probe import probe_machine_record
        return probe_machine_record(record, mode="auto", state_dir=self.state_dir)

    def _topology_host_records(self, user, topology):
        """Order candidate hosts: user roots, configured machines, explicit host.

        The planner prefers earlier rows when capacity ties, so a ready user
        root is considered before a configured machine that still needs a
        container. Configured records carry the real host SSH port/user.
        """
        wanted = str((topology or {}).get("host") or "").strip()
        configured: dict[str, dict[str, Any]] = {}
        for record in self._configured_machines():
            host = str((record.get("host") or {}).get("ip") or "").strip()
            if host and (not wanted or host == wanted):
                configured.setdefault(host, record)
        records: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(record):
            host = str((record.get("host") or {}).get("ip") or "").strip()
            if not host or host in seen or (wanted and host != wanted):
                return
            seen.add(host)
            records.append(record)

        for item in self.pool.catalog():
            if item.get("user") != user:
                continue
            host = str(item.get("host") or "").strip()
            if not host:
                continue
            record = configured.get(host)
            if record is None:
                endpoint = item.get("host_endpoint") or {}
                record = {
                    "alias": item.get("runtime_id") or host,
                    "host": {
                        "ip": host,
                        "port": int(endpoint.get("port") or 22),
                        "user": str(endpoint.get("user") or "root"),
                    },
                }
            add(record)
        for record in configured.values():
            add(record)
        if wanted and wanted not in seen:
            # Explicit hosts are probed even before they are configured; the
            # observation failure remains visible if SSH cannot reach them.
            add({"alias": wanted, "host": {"ip": wanted, "port": 22, "user": "root"}})
        return records

    def _observe_host_topologies(self, user, topology):
        """Probe real fleet snapshots for the bounded gang candidate set.

        Fleet probe output is the production schema; its `npus` rows are
        adapted by `topology_from_fleet_snapshot`. Probe failures are reported
        per host and never replaced with an assumed topology.
        """
        from mindie_coordinator.topology_gang_observation import (
            topology_from_fleet_snapshot,
        )

        records = self._topology_host_records(user, topology)
        wanted = str((topology or {}).get("host") or "").strip()
        limit = len(records) if wanted else MAX_GANG_PROBE_HOSTS
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

    def _record_topology_wait(self, store, row, planned):
        row["phase"] = "waiting"
        row["error"] = str(planned.get("error") or "topology-gang placement is waiting")[:500]
        details = planned.get("details") or {}
        row["error_details"] = {
            "code": planned.get("code"),
            "retryable": True,
            **({key: value for key, value in details.items() if key not in {"hosts"}}
               if isinstance(details, dict) else {}),
        }
        row["topology_retry_at"] = time.time() + GANG_PLAN_RETRY_SECONDS
        self._save_execution(store, row)
        return self._reply(row)

    def _plan_gang_roles(self, store, user, row, spec):
        """Observe hosts, then plan concrete sibling/pair/group device bindings."""
        topology = spec.get("topology") or {}
        resources = spec.get("resources") or {}
        observations, reports = self._observe_host_topologies(user, topology)
        plan = {
            "source": "fleet_probe",
            "observed_at": time.time(),
            "requested": {"topology": dict(topology), "resources": dict(resources)},
            "hosts": reports,
            "lease": None,
        }
        row["topology_plan"] = plan
        if not observations:
            errors = [item["error"] for item in reports if item.get("error")]
            detail = ("; ".join(errors[:3]) if errors else "no candidate host")
            return {
                "status": "waiting",
                "code": "host_observation_unavailable",
                "error": f"topology-gang placement has no usable host observation: {detail}"[:500],
            }
        try:
            lease, roles = plan_gang_placement(
                topology,
                resources,
                spec["command"],
                host_topologies=observations,
                require_observed=True,
            )
        except GangLeaseError as exc:
            if exc.code in GANG_RETRYABLE_CODES:
                return {
                    "status": "waiting",
                    "code": exc.code,
                    "error": f"topology-gang placement refused: {exc.message}"[:500],
                    "details": dict(exc.details),
                }
            raise
        if lease is None or not roles or any(
            not role.get("devices") or not role.get("host") for role in roles
        ):
            raise GangLeaseError(
                "host_observation_required",
                "topology-gang placement did not produce concrete host/device bindings",
            )
        plan.update({
            "spec": lease.spec.to_dict(),
            "lease": lease.to_dict(),
            "roles": [
                {
                    "name": role["name"],
                    "host": role["host"],
                    "devices": list(role["devices"]),
                    **({"service_port": role["service_port"]}
                       if role.get("service_port") is not None else {}),
                }
                for role in roles
            ],
        })
        return {"status": "ready", "roles": roles, "lease": lease}

    @staticmethod
    def _gang_device_mismatches(roles, jobs):
        """Compare host-granted devices with the observed plan, never count-only."""
        mismatches = []
        for role, job in zip(roles, jobs):
            planned = sorted(int(device) for device in (role.get("devices") or []))
            if not planned:
                continue
            raw = (job.get("environment") or {}).get("ASCEND_RT_VISIBLE_DEVICES")
            if not raw:
                # The host has not granted devices yet. A running command with
                # no visible-device binding is broken; a queued/granted job is
                # simply still waiting for its exact requested set.
                if job.get("state") == "running":
                    mismatches.append({
                        "role": role.get("name"), "planned": planned, "granted": None,
                    })
                continue
            try:
                granted = sorted(
                    int(part) for part in str(raw).split(",") if part.strip()
                )
            except ValueError:
                granted = []
            if granted != planned:
                mismatches.append({
                    "role": role.get("name"), "planned": planned, "granted": granted,
                })
        return mismatches

    def _gang_grant_problem(self, row, jobs):
        binding_mismatches = [
            {
                "role": role["name"],
                "planned": role.get("host"),
                "bound": ((role.get("binding") or {}).get("endpoint") or {}).get("host"),
            }
            for role in row["roles"]
            if role.get("host")
            and ((role.get("binding") or {}).get("endpoint") or {}).get("host") != role["host"]
        ]
        device_mismatches = self._gang_device_mismatches(row["roles"], jobs)
        if not binding_mismatches and not device_mismatches:
            return None
        return {
            "code": "granted_devices_mismatch",
            "binding_mismatches": binding_mismatches,
            "device_mismatches": device_mismatches,
        }

    def _place_or_prepare(self, store, user, row, roles, environment) -> dict[str, Any]:
        if row.get("preparation_jobs"):
            # A restarted/failed preparation retains durable job references.
            # Re-materializing here could replace sources under a live compiler.
            # Keep this execution observable/stoppable without replaying work.
            raise PreparationUncertain("retained preparation jobs require stop/quiet cleanup before a new execution")
        catalog = self.pool.catalog()
        # Catalog entries supply compatible environments and artifact donors,
        # never an arbitrary writable cwd to repurpose for the current sources.
        # Preparation resolves the execution's own root and reusable artifacts.
        topology = (row.get("spec") or {}).get("topology") or {}
        need_distinct = distinct_hosts_required(roles, topology)

        if need_distinct:
            known = {host_key(item) for item in catalog if item.get("user") == user and host_key(item)}
            for record in self._configured_machines():
                ip = (record.get("host") or {}).get("ip")
                if ip:
                    known.add(ip)
            if len(known) < len(roles):
                return {"status": "cache_miss", "reason":
                        "not enough distinct hosts with a matching prepared environment for this topology",
                        "provisioning_started": False}
        placements = []
        used_hosts: set[str] = set()
        for role in roles:
            donor = self._donor_for_role(user, environment, role, used_hosts, need_distinct)
            if donor is None:
                donor = self._ensure_user_container(user, environment, role, used_hosts, need_distinct,
                    on_progress=lambda event, role=role: self._save_container_progress(store, row, role['name'], event))
            if donor is None:
                return {"status": "cache_miss", "reason": "no allowed host provides the requested environment",
                        "provisioning_started": False}
            host = host_key(donor)
            if need_distinct and host:
                used_hosts.add(host)
            placements.append((role, donor))

        def prepare(placement):
            role, donor = placement
            # Private roots and immutable cache reads can prepare concurrently.
            # Only actual compiler stages share the host CPU budget below.
            if self._adopt_cancel(store, row):
                return None
            try:
                return self._prepare_role(store, user, row, role, environment, donor)
            except PreparationCancelled:
                return None

        try:
            if len(placements) == 1:
                prepared_roles = [prepare(placements[0])]
            else:
                with ThreadPoolExecutor(max_workers=min(4, len(placements)),
                                        thread_name_prefix="mindie-prepare") as workers:
                    prepared_roles = list(workers.map(prepare, placements))
        except TaskRootBusy as exc:
            return {"status": "waiting", "reason": str(exc)}
        ids = []
        bindings = []
        snapshots = {}
        for (role, _donor), prepared in zip(placements, prepared_roles):
            if prepared is None:
                return {"status": "waiting", "reason": "execution cancellation requested during preparation"}
            catalog = {item["runtime_id"]: item for item in self.pool.catalog()}
            item = catalog.get(prepared["id"])
            if item is None or not runtime_matches(item, environment, role):
                return {"status": "cache_miss",
                        "reason": "prepared environment does not match requested constraints",
                        "provisioning_started": True}
            ids.append(prepared["id"])
            bindings.append(prepared.get("binding"))
            # Completed preparation already published and checked this exact
            # input. A second materialization would remove generated version
            # metadata/native copies and repeat successful preparation work.
            if (prepared.get("attestation", {}).get("preparation") or {}).get("source_id") == row["spec"]["source_snapshot"]["id"]:
                snapshots[prepared["id"]] = {record["relpath"]: record["commit"]
                                             for record in row["spec"]["source_snapshot"]["records"]}
        return {"runtime_ids": ids, "bindings": bindings, "snapshots": snapshots,
                "reason": None, "provisioning_started": True}

    def _donor_for_role(self, user, environment, role, used_hosts, require_distinct=False):
        catalog = self.pool.catalog()
        for item in catalog:
            if item.get("user") != user or not item.get("host"):
                continue
            if require_distinct and item["host"] in used_hosts:
                continue
            if not can_prepare(environment, item, role):
                continue
            return item
        return None

    def _configured_machines(self):
        machines = getattr(self.backend, "machines", None)
        if machines is None:
            return []
        try:
            return list(machines.machines())
        except TimeoutError:
            raise
        except (OSError, ValueError) as exc:
            # Configured inventory failures are recoverable external state,
            # not an empty fleet or a permanently invalid execution request.
            raise MachineDirectoryUnavailable(
                f"cannot read configured machine inventory at {machines.path()}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _ensure_user_container(self, user, environment, role, used_hosts, require_distinct=False, *, on_progress=None):
        recipe = environment.get("recipe") or environment.get("image")
        if recipe and not provisionable_recipe(recipe):
            return None
        wanted_host = str(role["host"]) if role.get("host") else None
        for record in self._configured_machines():
            host_info = record.get("host") or {}
            host_ip = host_info.get("ip")
            if not host_ip or (require_distinct and host_ip in used_hosts):
                continue
            if wanted_host and host_ip != wanted_host:
                continue
            machine_type = host_info.get("machine_type")
            if environment.get("machine_type") and machine_type != environment["machine_type"]:
                continue
            configured = record.get("container") or {}
            owned = record.get("user") in (None, user) and configured.get("name") == user_container_name(user)
            ssh_port = configured.get("ssh_port") if owned else None
            host_endpoint = {
                "host": host_ip,
                "port": int(host_info.get("port") or 22),
                "user": host_info.get("user") or "root",
            }
            if not ssh_port or recipe:
                if not recipe:
                    # An existing configured container needs no image choice.
                    # Creating a container still requires an explicit recipe.
                    continue
                from mindie_coordinator.provision import provision_user_container

                def reserve_port(*, user, container_name, port):
                    stub = {
                        "host_endpoint": host_endpoint,
                        "container_name": container_name,
                        "endpoint": {"host": host_ip, "port": port, "user": "root"},
                    }
                    return self.backend.host(stub, {
                        "action": "container-ssh-reserve", "user": user,
                        "container_name": container_name, "port": port,
                    })

                result = provision_user_container(
                    host=host_ip, image=recipe, user=user,
                    host_user=host_endpoint["user"], host_port=host_endpoint["port"],
                    # A configured name/port does not prove the requested
                    # image. The provision owner verifies existing containers
                    # and rejects mismatches without replacing them.
                    ssh_port=int(ssh_port) if ssh_port else None,
                    machine_type=machine_type or environment.get("machine_type"),
                    machines=getattr(self.backend, "machines", None),
                    reserve_port=reserve_port,
                    **({'on_progress': on_progress} if on_progress is not None else {}),
                )
                ssh_port = result["ssh_port"]
            else:
                stub = {
                    "host_endpoint": host_endpoint,
                    "container_name": user_container_name(user),
                    "endpoint": {"host": host_ip, "port": int(ssh_port), "user": "root"},
                }
                self.backend.host(stub, {
                    "action": "container-ssh-reserve", "user": user,
                    "container_name": user_container_name(user), "port": int(ssh_port),
                })
            return {
                "user": user,
                "host": host_ip,
                "host_endpoint": host_endpoint,
                "ssh_port": int(ssh_port),
                "endpoint": {"host": host_ip, "port": int(ssh_port), "user": "root"},
                "container_name": user_container_name(user),
                "recipe": recipe,
                "machine_type": machine_type,
                "service_ports": [],
            }
        return None

    def _save_container_progress(self, store, row, role, event):
        # Keep bounded phase facts in the existing execution log even after
        # prepare-root replaces the current progress. No remote payloads/keys.
        log = self.state_dir / 'runs' / row['id'] / role / 'prepare-container.log'
        self._save_progress(store, row, role, {**event, 'log_ref': str(log)})
        try:
            with log.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
        except TimeoutError:
            raise
        except OSError as exc:
            get_recorder("mindie-coordinator").event("WARNING", "diagnostics.persistence_failed", error_type=type(exc).__name__)

    def _save_progress(self, store, row, role, event):
        service_module = _service_module()
        now = service_module.time.time()
        monotonic = service_module.time.monotonic()
        clock_domain = service_module._PROCESS_CLOCK_DOMAIN
        if event.get("log_ref"):
            path = Path(event["log_ref"])
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch(exist_ok=True)
            except TimeoutError:
                raise
            except OSError as exc:
                get_recorder("mindie-coordinator").event("WARNING", "diagnostics.persistence_failed", error_type=type(exc).__name__)
        with self._lock_for("progress", row["id"]):
            if role and event.get("step") == "shared-native-cache":
                from mindie_coordinator.preparation_cache import native_cache_receipt
                row.setdefault("native_cache_receipts", {})[role] = native_cache_receipt(event)
            previous = (row.get("role_progress") or {}).get(role, {}) if role else row.get("progress") or {}
            same_step = (previous.get("step"), previous.get("role")) == (event.get("step"), role)
            if previous.get("started_at") and not same_step:
                row.setdefault("stage_history", []).append({
                    **previous, "ended_at": now,
                    **({"elapsed_seconds": round(max(0, monotonic - previous["started_monotonic"]), 3)}
                       if previous.get("clock_domain") == clock_domain and "started_monotonic" in previous else
                       {"elapsed_seconds": None, "timing_incomplete": "owner clock domain changed"})})
                row["stage_history"] = row["stage_history"][-64:]
            progress = {**(previous if same_step else {}), **event, "role": role,
                        "started_at": previous.get("started_at", now) if same_step else now,
                        "updated_at": now,
                        "started_monotonic": previous.get("started_monotonic", monotonic) if same_step and previous.get("clock_domain") == clock_domain else monotonic,
                        "clock_domain": clock_domain}
            if not same_step:
                get_recorder("mindie-coordinator").event("INFO", "execution.phase", execution_id=row["id"], role=role, phase=event.get("step"))
            row["progress"] = progress
            if role:
                row.setdefault("role_progress", {})[role] = progress
            self._save_execution(store, row)

    def _prepare_role(self, store, user, row, role, environment, donor):
        from mindie_coordinator.provision import prepare_task_environment
        prepared = prepare_task_environment(
            self.pool, user=user, session_id=row["id"], role_name=role["name"],
            environment=environment, donor=donor, sources=row.get("sources") or {},
            source_snapshot=row["spec"]["source_snapshot"],
            on_progress=lambda event: self._save_progress(store, row, role["name"], event),
            log_dir=self.state_dir / "runs" / row["id"] / role["name"],
            on_preparation_job=lambda job: self._save_preparation_job(store, row, role["name"], job),
            cancel_requested=lambda: self._adopt_cancel(store, row),
            checkout_session=row["remote_session"]["id"],
        )
        # Each finished role remains recoverable while its siblings prepare.
        # A crash before this save is also covered by the pool's durable
        # execution-session binding, which finish reads below.
        if prepared.get("binding"):
            with self._lock_for("progress", row["id"]):
                row.setdefault("prepared_bindings", {})[role["name"]] = prepared["binding"]
                if prepared.get("attestation"):
                    attestation = prepared["attestation"]
                    row.setdefault("preparation_results", {})[role["name"]] = self._manifest_summary(attestation)
                    receipt = prepared.get("native_cache_receipt") or attestation.get("native_cache_receipt")
                    if isinstance(receipt, dict) and receipt.get("cache_key"):
                        row.setdefault("native_cache_receipts", {})[role["name"]] = receipt
                self._save_execution(store, row)
        return prepared

    def _save_preparation_job(self, store, row, role, job):
        with self._lock_for("progress", row["id"]):
            row.setdefault("preparation_jobs", {}).setdefault(role, {})[job["job_id"]] = dict(job)
            progress = (row.get("role_progress") or {}).get(role)
            if progress and progress.get("step") == job.get("step"):
                progress["process"] = {key: job.get(key) for key in ("job_id", "state", "quiet", "observed_at")}
                current = row.get("progress") or {}
                if current.get("role") == role and current.get("step") == job.get("step"):
                    current["process"] = progress["process"]
            self._save_execution(store, row)

    def _refresh_preparation_jobs(self, store, row):
        for role, records in (row.get("preparation_jobs") or {}).items():
            for record in records.values():
                if record.get("quiet"):
                    from mindie_coordinator.compile_lease import finish_compile_lease
                    finish_compile_lease(record,
                        lambda value: self._save_preparation_job(store, row, role, value))
                    continue
                try:
                    observed = preparation_control(record["endpoint"], record["job_id"], "status")
                except Exception as exc:
                    observed = {"state": "uncertain", "quiet": False, "error": str(exc)[:500]}
                record.update({key: value for key, value in observed.items() if key != "processes"})
                record["observed_at"] = time.time()
                self._save_preparation_job(store, row, role, record)

    def _refresh_jobs(self, store, user, row):
        if not row.get("roles"):
            return row
        def observe_role(role):
            if not role.get("managed_job"):
                return
            job = self.pool.managed_control(user, role["managed_job"], "status")
            role["observation"] = job
            role["status_observed_at"] = time.time()
        _service_module()._map_roles(observe_role, row["roles"])
        if row["roles"]:
            if any(role.get("managed_job") for role in row["roles"]):
                row["jobs_observed_at"] = time.time()
            row["observation"] = row["roles"][0].get("observation")
            row["managed_job"] = row["roles"][0].get("managed_job")
            states = [role.get("observation", {}).get("state") for role in row["roles"] if role.get("observation")]
            if states:
                row["phase"] = _service_module().aggregate_job_states(states)
        self._save_execution(store, row)
        return row

    def _stop(self, store, user, row, force):
        row["cancel_requested"] = True
        if force:
            row["force"] = True
        self._stop_jobs(user, row, force)
        preparation_quiet = self._stop_preparation_jobs(store, row, force)
        row = self._refresh_jobs(store, user, row)
        job_states = [role.get("observation", {}).get("state")
                      for role in row.get("roles") or [] if role.get("managed_job")]
        if not preparation_quiet:
            row["phase"] = "uncertain"
            row["error"] = "preparation stop has not verified quiet; retained jobs remain owned"
            self._save_execution(store, row)
        elif not job_states:
            row["phase"] = "cancelled"
            self._save_execution(store, row)
        return self._reply(row)

    def _stop_preparation_jobs(self, store, row, force):
        jobs = [(role, job) for role, records in (row.get("preparation_jobs") or {}).items()
                for job in records.values()]
        def stop_job(item):
            role, job = item
            if job.get("quiet"):
                from mindie_coordinator.compile_lease import finish_compile_lease
                return finish_compile_lease(job, lambda value: self._save_preparation_job(store, row, role, value))
            return stop_preparation_process(job,
                lambda value: self._save_preparation_job(store, row, role, value), force=force)
        if len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as workers:
                return all(list(workers.map(stop_job, jobs)))
        return all(stop_job(item) for item in jobs)

    def _stop_and_wait(self, store, sessions_dir, user, row, timeout=None):
        service_module = _service_module()
        if timeout is None:
            timeout = service_module.STOP_WAIT_SECONDS
        deadline = time.time() + timeout
        reply = self._request_stop(sessions_dir, user, row["id"], False)
        while reply.get("state") not in service_module.DONE and time.time() < deadline:
            time.sleep(0.05)
            reply = self._request_stop(sessions_dir, user, row["id"], False)
        return reply

    def _stop_jobs(self, user, row, force):
        failures = []
        def stop_role(role):
            if role.get("managed_job"):
                try:
                    role["observation"] = self.pool.managed_control(user, role["managed_job"], "stop", force)
                except Exception as exc:
                    failures.append({"role": role.get("name"), "error": f"{type(exc).__name__}: {exc}"[:500]})
        _service_module()._map_roles(stop_role, row.get("roles") or [])
        if failures:
            row["stop_errors"] = failures
        return failures

    def _record_assignment(self, row, jobs):
        assignment = row.get("assignment") or {"roles": []}
        for index, (role, job) in enumerate(zip(row["roles"], jobs)):
            devices = (job.get("environment") or {}).get("ASCEND_RT_VISIBLE_DEVICES")
            entry = (assignment.get("roles") or [{}] * len(jobs))
            if index < len(entry):
                entry[index].update({
                    "name": role["name"],
                    "runtime_id": role["runtime_id"],
                    "host": role["binding"]["endpoint"]["host"],
                    "root": role["binding"]["endpoint"]["cwd"],
                    "rank": index,
                    "devices": devices,
                    **({"allow_external_busy": True} if role.get("allow_external_busy") else {}),
                    "service_port": job.get("service_port"),
                    "state": job["state"],
                })
        row["assignment"] = assignment
