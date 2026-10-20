"""Topology-gang production boundary: observed planning, admission, host grant.

Tier 0/1 only. These tests use the real `CoordinatorService` and the real
`mindie_npu_coordination` SQLite queue in isolated temporary state directories.
Fleet snapshots are produced by the production `parse_fleet_probe_output`
parser from checked-in A3 `npu-smi` captures; no compressed or synthetic
success payload replaces the host queue's grant.
"""

from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

from mindie_coordinator.agent_session import AgentSessions
from mindie_coordinator.fleet_probe import parse_fleet_probe_output
from mindie_coordinator.host import mindie_npu_coordination as host_protocol
from mindie_coordinator.host.mindie_npu_coordination import handle_request
from mindie_coordinator.placement import (
    plan_gang_placement,
    plan_gang_roles,
    role_plan,
)
from mindie_coordinator.ready_runtime import RuntimePool
from mindie_coordinator.task_client import TaskClient
from mindie_coordinator.topology_gang import (
    CardObservation,
    DieObservation,
    HostTopology,
    parse_npu_smi_capture,
    topology_from_fleet_snapshot,
)
from mindie_coordinator.topology_gang_common import GangLeaseError

FIXTURES = Path(__file__).parent / "fixtures"
A3_PARTIAL = FIXTURES / "a3-192-168-13-156-partial.txt"
A3_BUSY = FIXTURES / "a3-192-168-13-154-busy.txt"
A3_IDLE = FIXTURES / "a3-192-168-13-165-idle.txt"
A2 = FIXTURES / "a2-173-125-1-2.txt"
HOST = "192.0.2.1"


def _fleet_snapshot(path: Path, host: str = HOST) -> dict:
    raw = "===SECTION:NPU_SMI===\n" + path.read_text(encoding="utf-8")
    return parse_fleet_probe_output(raw, alias=host, host=host)


def _topology(free_phy_ids: tuple[int, ...], host: str = HOST) -> HostTopology:
    """Build a two-card A3 observation where only ``free_phy_ids`` are free."""
    cards = []
    for npu_id in range(2):
        phy_ids = (2 * npu_id, 2 * npu_id + 1)
        dies = tuple(
            DieObservation(
                host=host,
                npu_id=npu_id,
                chip_id=index,
                phy_id=phy_id,
                chip_name="Ascend910",
                health="OK",
                hbm_used_mb=0,
                hbm_total_mb=65536,
                observed_busy=phy_id not in free_phy_ids,
            )
            for index, phy_id in enumerate(phy_ids)
        )
        cards.append(
            CardObservation(
                host=host,
                npu_id=npu_id,
                chip_name="Ascend910",
                phy_ids=phy_ids,
                dies=dies,
            )
        )
    return HostTopology(host=host, cards=tuple(cards))


def test_tier0_fleet_snapshot_adapter_preserves_real_a3_sibling_pairs():
    snapshot = _fleet_snapshot(A3_PARTIAL)
    topology = topology_from_fleet_snapshot(snapshot, host=HOST)

    assert topology.host == HOST
    assert [card.phy_ids for card in topology.cards] == [
        (2 * index, 2 * index + 1) for index in range(8)
    ]
    assert topology.free_phy_ids == (4, 5, 6, 7)
    assert topology.free_dual_die_pairs == ((4, 5), (6, 7))
    assert topology.free_phy_ids == tuple(
        parse_npu_smi_capture(A3_PARTIAL.read_text(encoding="utf-8"), host=HOST).free_phy_ids
    )

    # A parsed busy Die without attributable process/HBM rows must stay busy.
    for card in snapshot["npus"]:
        card["processes"] = []
        for die in card["dies"]:
            die["busy"] = True
            die["hbm_used_mb"] = 0
    all_busy = topology_from_fleet_snapshot(snapshot, host=HOST)
    assert all_busy.free_phy_ids == ()
    assert all(
        any("npu-smi reported" in reason for reason in die.exclusion_reasons)
        for die in all_busy.dies
    )


def test_tier0_role_plan_preserves_explicit_host_devices_and_port():
    """The reviewed defect-4 request must not lose its explicit binding."""
    roles = role_plan(
        {
            "scale": "S1",
            "die_count": 2,
            "placement_policy": "same_npu_dual_die",
            "host": "required-a3",
        },
        {"devices": [4, 5], "service_port": 8000},
        "true",
    )
    assert roles == [
        {
            "name": "default",
            "command": "true",
            "devices": [4, 5],
            "host": "required-a3",
            "service_port": 8000,
        }
    ]


def test_tier0_role_plan_rejects_genuine_explicit_conflicts():
    with pytest.raises(GangLeaseError) as straddling:
        role_plan(
            {"scale": "S1", "die_count": 2, "placement_policy": "same_npu_dual_die"},
            {"devices": [1, 2]},
            "true",
        )
    assert straddling.value.code == "conflicting_devices"

    with pytest.raises(GangLeaseError) as count:
        role_plan(
            {"scale": "S2", "die_count": 16, "placement_policy": "intra_node_hccs"},
            {"npu_count": 4},
            "true",
        )
    assert count.value.code == "conflicting_npu_count"

    with pytest.raises(ValueError, match="cannot be combined"):
        role_plan(
            {
                "scale": "S1",
                "die_count": 2,
                "placement_policy": "same_npu_dual_die",
                "roles": [{"name": "one"}],
            },
            {},
            "true",
        )

    with pytest.raises(GangLeaseError) as multi_node:
        plan_gang_roles(
            {
                "scale": "S3",
                "die_count": 16,
                "placement_policy": "multi_node_roce",
                "node_count": 2,
                "host": HOST,
            },
            {},
            "true",
        )
    assert multi_node.value.code == "conflicting_host"

    a2 = parse_npu_smi_capture(A2.read_text(encoding="utf-8"), host="a2-host")
    with pytest.raises(GangLeaseError) as unsupported:
        plan_gang_placement(
            {"scale": "S1", "die_count": 2, "placement_policy": "same_npu_dual_die"},
            {},
            "true",
            host_topologies=[a2],
            require_observed=True,
        )
    assert unsupported.value.code == "unsupported_topology"


def test_tier0_plan_gang_requires_observations_for_executable_roles():
    topology = {
        "scale": "S1",
        "die_count": 2,
        "placement_policy": "same_npu_dual_die",
    }
    structural = plan_gang_roles(topology, {}, "serve")
    assert structural == [{"name": "default", "command": "serve", "npu_count": 2}]

    with pytest.raises(GangLeaseError) as refusal:
        plan_gang_roles(topology, {}, "serve", require_observed=True)
    assert refusal.value.code == "host_observation_required"


def test_tier0_observed_plan_carries_requested_devices_host_and_port():
    topology = {
        "scale": "S1",
        "die_count": 2,
        "placement_policy": "same_npu_dual_die",
        "host": HOST,
    }
    observation = parse_npu_smi_capture(A3_PARTIAL.read_text(encoding="utf-8"), host=HOST)
    lease, roles = plan_gang_placement(
        topology,
        {"devices": [4, 5], "service_port": 8000},
        "serve",
        host_topologies=[observation],
        require_observed=True,
    )
    assert lease is not None and lease.hosts == (HOST,)
    assert roles == [
        {
            "name": "default",
            "command": "serve",
            "devices": [4, 5],
            "host": HOST,
            "service_port": 8000,
        }
    ]
    assert lease.groups == {"default": (4, 5)}

    blocked = _topology(free_phy_ids=(6, 7))
    with pytest.raises(GangLeaseError) as unavailable:
        plan_gang_placement(
            topology,
            {"devices": [4, 5]},
            "serve",
            host_topologies=[blocked],
            require_observed=True,
        )
    assert unavailable.value.code == "requested_dies_unavailable"


class GangBackend:
    """Real host protocol and SQLite queue with a configurable occupancy sample."""

    def __init__(self, state: Path, *, devices: list[int], busy: set[int]):
        self.state = Path(state)
        self.devices = list(devices)
        self.busy = set(busy)
        self.calls: list[tuple[str, str]] = []
        self.jobs: dict[str, dict] = {}
        self.prepared: dict[str, dict] = {}
        self.require_prepared = False
        self.machines = None
        self.protocol_lock = threading.RLock()
        self.attestation = {
            "profile_key": "profile-a",
            "build_key": "native-a",
            "profile": {"launch_env": {"VLLM_VERSION": "test"}},
        }

    def mark_prepared(self, spec: dict, profile_key="profile-a", build_key="native-a"):
        root = spec["endpoint"]["root"]
        recipe = spec.get("recipe")
        self.prepared[root] = {
            "python": spec["python"],
            "recipe": recipe,
            "profile": {"launch_env": {"VLLM_VERSION": "test"}, "recipe": recipe},
            "profile_key": profile_key,
            "build_key": build_key,
        }

    def prepare_task_root(self, spec, *, sources, environment, donor_python=None, **kwargs):
        root = spec["endpoint"]["root"]
        python = spec["python"]
        if {"vllm", "vllm-ascend"}.issubset(sources) and donor_python and python == donor_python:
            raise ValueError("task-owned interpreter must not be the donor interpreter")
        recipe = (environment or {}).get("recipe") or (environment or {}).get("image") or spec.get("recipe")
        profile = {"launch_env": {"VLLM_VERSION": "test"}, "recipe": recipe}
        for key in ("python_abi", "cann", "soc", "machine_type"):
            if (environment or {}).get(key):
                profile[key] = environment[key]
        self.prepared[root] = {
            "python": python,
            "recipe": recipe,
            "profile": profile,
            "profile_key": "profile-" + (recipe or "prepared"),
            "build_key": "native-" + (recipe or "prepared"),
        }
        return self.inspect(spec)

    def command_environment(self, donor):
        return {
            "python": "/usr/local/bin/python3",
            "launch_env": {"PATH": "/usr/local/bin:/usr/bin:/bin"},
        }

    def inspect(self, runtime, **kwargs):
        root = (runtime.get("endpoint") or {}).get("cwd") or (runtime.get("endpoint") or {}).get("root")
        python = runtime.get("python")
        prepared = self.prepared.get(root)
        if prepared is None:
            if self.require_prepared:
                raise ValueError("missing ready-profile.json")
            observed = copy.deepcopy(self.attestation)
            observed["container_id"] = "cid-" + str(runtime.get("container_name") or "unknown")
            return observed
        if prepared["python"] != python:
            raise ValueError("interpreter does not match prepared root")
        return {
            "profile_key": prepared["profile_key"],
            "build_key": prepared["build_key"],
            "profile": copy.deepcopy(prepared["profile"]),
            "container_id": "cid-" + str(runtime.get("container_name") or "unknown"),
        }

    def host(self, runtime, request):
        self.calls.append(("host", request["action"]))
        def guarded(value, *, completion_confirmed=False):
            guard = json.loads(value) if isinstance(value, str) else value
            return bool(guard) and (
                any(
                    not job["quiet"] and job["receipt"]["process_guard"] == guard
                    for job in self.jobs.values()
                )
                or (guard.get("retain_until_release") and not completion_confirmed)
            )
        listening = {"status": "ok", "ports": []}
        host = runtime["host_endpoint"]["host"]
        state = self.state / host.replace(".", "_")
        state.mkdir(parents=True, exist_ok=True)
        with self.protocol_lock, mock.patch.object(
            host_protocol, "process_guard_busy", side_effect=guarded
        ):
            return handle_request(
                {
                    **request,
                    "state_dir": str(state),
                    "interval_seconds": 0.001,
                },
                clock=time.time,
                probe=lambda: {
                    "status": "ok",
                    "devices": list(self.devices),
                    "busy": {str(device): ["test worker"] for device in self.busy},
                },
                device_probe=lambda device: {
                    "status": "ok",
                    "devices": [device],
                    "busy": None,
                    "free": [],
                },
                listening_ports=lambda: listening,
            )

    def job(self, runtime, job_id, action, **params):
        self.calls.append(("job", action))
        if action == "prepare" and job_id not in self.jobs:
            self.jobs[job_id] = {
                "state": "prepared",
                "quiet": False,
                "receipt": {
                    "pid": 4242,
                    "process_guard": {
                        "marker": job_id[-32:],
                        "boot_id": "test-boot",
                        "retain_until_release": True,
                    },
                },
            }
        if action == "go":
            self.jobs[job_id]["state"] = "running"
        if action == "stop" and job_id in self.jobs:
            self.jobs[job_id].update(state="cancelled", quiet=True)
        if action == "tail":
            job = self.jobs.get(job_id, {"state": "absent", "quiet": True})
            text = job.get("stdout") or f"stdout-{job_id[-8:]}"
            return {
                **copy.deepcopy(job),
                "stdout": text,
                "stderr": job.get("stderr") or "",
                "tail": text,
            }
        return copy.deepcopy(self.jobs.get(job_id, {"state": "absent", "quiet": True}))

    def job_host_pid(self, runtime, receipt):
        return receipt["pid"]


def _runtime_spec(number=1, *, user="alice", host=HOST, service_ports=None):
    return {
        "user": user,
        "python": f"/opt/{user}/venvs/root-{number}/bin/python",
        "endpoint": {
            "host": host,
            "port": 46001,
            "root": f"/vllm-workspace/{user}/{number}",
            "user": "root",
        },
        "host_endpoint": {"host": host, "port": 22, "user": "root"},
        "container_name": "mindie-" + user,
        "service_ports": service_ports if service_ports is not None else [48000 + number],
    }


def _client(tmp_path: Path, *, devices=None, busy=None):
    backend = GangBackend(
        tmp_path / "host",
        devices=list(devices or range(16)),
        busy=set(busy or ()),
    )
    pool = RuntimePool(tmp_path / "manager", backend)
    spec = _runtime_spec(service_ports=[48000])
    backend.mark_prepared(spec)
    pool.register("runtime-a", spec)
    store = AgentSessions(tmp_path / "sessions")
    context = store.attach("codex", "native-gang", str(tmp_path))
    return TaskClient(context["context_file"], pool=pool, user="alice"), backend, pool


def _run_gang(client, topology, resources):
    return client.run(
        "true",
        sources={},
        topology=topology,
        resources=resources,
    )


def test_tier1_service_observes_topology_and_grants_only_planned_siblings(tmp_path):
    client, backend, pool = _client(
        tmp_path,
        busy=set(range(16)) - {4, 5},
    )
    service = client.coordinator
    snapshot = _fleet_snapshot(A3_PARTIAL)
    with mock.patch.object(service, "_probe_host_record", return_value=snapshot):
        reply = _run_gang(
            client,
            {
                "scale": "S1",
                "die_count": 2,
                "placement_policy": "same_npu_dual_die",
                "host": HOST,
            },
            {"devices": [4, 5], "service_port": 48000},
        )

    assert reply["state"] == "running", reply
    assignment = reply["assignment"]["roles"][0]
    assert assignment["host"] == HOST
    assert assignment["devices"] == "4,5"
    assert assignment["service_port"] == 48000
    assert reply["target"]["environment"]["ASCEND_RT_VISIBLE_DEVICES"] == "4,5"
    assert reply["target"]["environment"]["MINDIE_SERVICE_PORT"] == "48000"
    assert ("host", "submit") in backend.calls
    assert reply["topology"]["lease"]["groups"] == {"default": [4, 5]}

    # Read the real host SQLite task back: the grant is not reconstructed from
    # the coordinator's own reply.
    with pool.transaction() as db:
        run = pool.rows(db, "run")[0]
    status = handle_request(
        {
            "action": "status",
            "state_dir": str(Path(backend.state) / HOST.replace(".", "_")),
            "task_id": run["task_id"],
            "no_probe": True,
        },
        clock=time.time,
        listening_ports=lambda: {"status": "ok", "ports": []},
    )
    task = next(item for item in status["tasks"] if item["task_id"] == run["task_id"])
    assert task["requested_devices"] == [4, 5]
    assert task["granted_devices"] == [4, 5]


def test_tier1_non_sibling_free_set_is_refused_before_submit(tmp_path):
    client, backend, pool = _client(tmp_path, devices=list(range(4)), busy={0, 3})
    service = client.coordinator
    # Free devices are [1, 2] but neither forms a sibling pair: (0,1) and
    # (2,3) each have a busy Die. The queue must never grant [1, 2].
    observation = _topology(free_phy_ids=(1, 2))
    with mock.patch.object(
        service,
        "_observe_host_topologies",
        return_value=([observation], [{"host": HOST, "state": "observed"}]),
    ):
        reply = _run_gang(
            client,
            {
                "scale": "S1",
                "die_count": 2,
                "placement_policy": "same_npu_dual_die",
            },
            {},
        )

    assert reply["state"] == "waiting", reply
    assert reply["error_details"]["code"] == "insufficient_dual_die_pairs"
    assert "sibling" in reply["error"] or "pair" in reply["error"]
    assert ("host", "submit") not in backend.calls
    with pool.transaction() as db:
        assert pool.rows(db, "job") == []
        assert pool.rows(db, "run") == []


def test_tier1_missing_topology_observation_waits_without_submit(tmp_path):
    client, backend, pool = _client(tmp_path)
    service = client.coordinator
    with mock.patch.object(
        service,
        "_probe_host_record",
        return_value={"online": False, "error": "ssh probe timed out"},
    ):
        reply = _run_gang(
            client,
            {
                "scale": "S1",
                "die_count": 2,
                "placement_policy": "same_npu_dual_die",
            },
            {},
        )

    assert reply["state"] == "waiting", reply
    assert reply["error_details"]["code"] == "host_observation_unavailable"
    assert "ssh probe timed out" in reply["error"]
    assert ("host", "submit") not in backend.calls
    with pool.transaction() as db:
        assert pool.rows(db, "job") == []
        assert pool.rows(db, "run") == []


def test_tier1_a2_observation_is_a_terminal_same_card_conflict(tmp_path):
    client, backend, pool = _client(tmp_path)
    service = client.coordinator
    a2_snapshot = _fleet_snapshot(A2)
    with mock.patch.object(service, "_probe_host_record", return_value=a2_snapshot):
        reply = _run_gang(
            client,
            {
                "scale": "S1",
                "die_count": 2,
                "placement_policy": "same_npu_dual_die",
            },
            {},
        )

    assert reply["state"] == "failed", reply
    assert reply["error_details"]["code"] == "unsupported_topology"
    assert ("host", "submit") not in backend.calls
    with pool.transaction() as db:
        assert pool.rows(db, "job") == []


def test_tier1_explicit_host_is_filtered_not_substituted(tmp_path):
    client, backend, _pool = _client(tmp_path)
    service = client.coordinator
    probed = []

    def probe(record):
        host = record["host"]["ip"]
        probed.append(host)
        return {"online": False, "error": f"cannot reach {host}", "host": host}

    with mock.patch.object(service, "_probe_host_record", side_effect=probe):
        reply = _run_gang(
            client,
            {
                "scale": "S1",
                "die_count": 2,
                "placement_policy": "same_npu_dual_die",
                "host": "required-a3",
            },
            {},
        )

    assert reply["state"] == "waiting", reply
    assert probed == ["required-a3"]
    assert reply["topology"]["hosts"][0]["host"] == "required-a3"
    assert "required-a3" in reply["error"]
    assert ("host", "submit") not in backend.calls


def test_tier1_busy_requested_devices_wait_instead_of_substituting(tmp_path):
    client, backend, pool = _client(
        tmp_path,
        busy=set(range(16)) - {6, 7},
    )
    service = client.coordinator
    snapshot = _fleet_snapshot(A3_PARTIAL)
    # The capture says 4,5 were free; the queue now reports them busy. The
    # observation is stale by design here, so admission must wait for the
    # exact requested pair rather than let the queue pick 6,7.
    with mock.patch.object(service, "_probe_host_record", return_value=snapshot):
        reply = _run_gang(
            client,
            {
                "scale": "S1",
                "die_count": 2,
                "placement_policy": "same_npu_dual_die",
                "host": HOST,
            },
            {"devices": [4, 5]},
        )

    # The stale observation still allows a plan, but the host queue refuses to
    # grant the unavailable pair. No actual job may be submitted with 6,7.
    assert reply["state"] in {"queued", "waiting"}, reply
    assert ("host", "submit") in backend.calls
    with pool.transaction() as db:
        runs = pool.rows(db, "run")
    assert runs and all(run["state"] not in {"active"} for run in runs)
    assert all(
        (run.get("environment") or {}).get("ASCEND_RT_VISIBLE_DEVICES")
        in (None, "", "4,5")
        for run in runs
    )
    # The host task is queued on the exact requested devices, never granted a
    # different free pair.
    task = next(item for item in _host_tasks(backend) if item["task_id"] == runs[0]["task_id"])
    assert task["state"] == "queued"
    assert task["requested_devices"] == [4, 5]
    assert task["granted_devices"] in (None, [])


def _host_tasks(backend):
    status = handle_request(
        {
            "action": "status",
            "state_dir": str(Path(backend.state) / HOST.replace(".", "_")),
            "no_probe": True,
        },
        clock=time.time,
        listening_ports=lambda: {"status": "ok", "ports": []},
    )
    return status["tasks"]
