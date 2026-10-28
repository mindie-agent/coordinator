"""Tests for coordinator's SSH host, container, NPU probe, and lease management."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from remote_dev.core.endpoint import resolve_endpoint

from mindie_coordinator.endpoint_bridge import (
    build_endpoints_document,
    resolve_coordinator_endpoint,
)
from mindie_coordinator.env_manager import (
    handle_container,
    handle_hosts,
    handle_lease,
    handle_probe,
)
from mindie_coordinator.fleet_probe import (
    find_fleet_capacity,
    parse_fleet_probe_output,
    parse_npu_smi_info,
    probe_machine_record,
)
from mindie_coordinator.host_queue import HostQueue
from mindie_coordinator.machine_directory import MachineDirectory
from mindie_coordinator.ops import mindie_call
from mindie_coordinator.task_server import call_tool

SAMPLE_910C_PROBE_OUTPUT = """===SECTION:NPU_SMI===
+------------------------------------------------------------------------------------------------+
| npu-smi 24.1.rc2                 Version: 24.1.rc2                                             |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B                | OK            | 92.4        42                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           0    / 0          1280 / 32768         |
| 1                         | 0000:C2:00.0  | 0           0    / 0          1280 / 32768         |
+---------------------------+---------------+----------------------------------------------------+
| 1     910B                | OK            | 180.0       55                0    / 0             |
| 0                         | 0000:81:00.0  | 40          0    / 0          24000/ 32768         |
| 1                         | 0000:82:00.0  | 50          0    / 0          24000/ 32768         |
+---------------------------+---------------+----------------------------------------------------+
| 2     910B                | OK            | 88.0        40                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          1024 / 32768         |
| 1                         | 0000:42:00.0  | 0           0    / 0          1024 / 32768         |
+---------------------------+---------------+----------------------------------------------------+
| 3     910B                | OK            | 89.0        41                0    / 0             |
| 0                         | 0000:01:00.0  | 0           0    / 0          1024 / 32768         |
| 1                         | 0000:02:00.0  | 0           0    / 0          1024 / 32768         |
+===========================+===============+====================================================+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| 1       0                 | 12345         | vllm_worker              | 22000                   |
+===========================+===============+====================================================+
===SECTION:CONTAINERS===
0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef|mindie-alice|quay.nju.edu.cn/ascend/vllm-ascend:main-a3|Up 3 hours|0.0.0.0:46012->22/tcp
===SECTION:PID_MAP===
12345|alice|0::/system.slice/docker-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.scope
===SECTION:SYSTEM===
HOSTNAME:atlas-node-01
UPTIME:up 14 days, 3 hours
LOADAVG:2.50 1.80 1.20 2/1024 9999
CORES:192
CPU_MODEL:Kunpeng-920
MEM:1048576:262144:786432
===SECTION:MOUNTS===
/dev/nvme0n1p1 ext4 536870912000 107374182400 429496729600 20% /
10.10.0.1:/models nfs4 10995116277760 2199023255552 8796093022208 20% /data/weights
===SECTION:TOP_PROCS===
12345 alice 98.5 2.1 4194304 vllm_worker
"""


class EndpointBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = MachineDirectory(path=self.root / "machines.json")

    def test_upsert_syncs_endpoints_json_for_remote_dev(self):
        self.store.upsert_machine(
            {
                "alias": "atlas-a3",
                "host": {"ip": "10.0.0.12", "port": 22, "user": "root"},
                "container": {"name": "mindie-alice", "ssh_port": 46012, "user": "root", "workdir": "/vllm-workspace"},
            },
            set_default=True,
        )
        endpoints_path = self.root / "endpoints.json"
        self.assertTrue(endpoints_path.is_file())
        doc = json.loads(endpoints_path.read_text(encoding="utf-8"))
        self.assertEqual(doc["default_alias"], "atlas-a3")
        self.assertEqual(doc["endpoints"]["atlas-a3"]["port"], 46012)
        self.assertEqual(doc["endpoints"]["atlas-a3-host"]["port"], 22)
        self.assertEqual(doc["endpoints"]["atlas-a3-docker"]["container"]["name"], "mindie-alice")
        self.assertEqual(doc["endpoints"]["default"]["port"], 46012)

        with mock.patch.dict(
            os.environ,
            {
                "REMOTE_DEV_ENDPOINTS_FILE": str(endpoints_path),
                "MINDIE_COORDINATOR_STATE_DIR": str(self.root),
            },
            clear=False,
        ):
            ep_alias = resolve_endpoint({"alias": "atlas-a3"})
            self.assertEqual((ep_alias.host, ep_alias.port, ep_alias.cwd), ("10.0.0.12", 46012, "/vllm-workspace"))
            ep_default_alias = resolve_endpoint({"alias": "default"})
            self.assertEqual((ep_default_alias.host, ep_default_alias.port), ("10.0.0.12", 46012))
            ep_host = resolve_endpoint({"alias": "atlas-a3-host"})
            self.assertEqual((ep_host.host, ep_host.port), ("10.0.0.12", 22))
            from remote_dev.core.endpoint import unregister_resolver
            from mindie_coordinator.endpoint_bridge import setup_resolvers

            setup_resolvers()
            try:
                ep_zero_arg = resolve_endpoint({})
                self.assertEqual((ep_zero_arg.host, ep_zero_arg.port), ("10.0.0.12", 46012))
            finally:
                unregister_resolver("mindie_coordinator")

    def test_docker_exec_mode_creates_container_endpoint(self):
        doc = build_endpoints_document(
            {
                "machines": [
                    {
                        "alias": "dev-box",
                        "host": {"ip": "10.0.0.99", "port": 22, "user": "ubuntu"},
                        "container": {"name": "vllm-dev", "mode": "docker_exec", "workdir": "/workspace"},
                    }
                ]
            }
        )
        primary = doc["endpoints"]["dev-box"]
        self.assertEqual(primary["host"], "10.0.0.99")
        self.assertEqual(primary["port"], 22)
        self.assertEqual(primary["container"], {"name": "vllm-dev", "user": "root", "workdir": "/workspace"})

    def test_resolver_plugin_resolves_alias_and_host(self):
        self.store.upsert_machine(
            {
                "alias": "node-1",
                "host": {"ip": "192.168.1.50", "port": 2222, "user": "ascend"},
            }
        )
        by_alias = resolve_coordinator_endpoint({"alias": "node-1", "cwd": "/tmp"}, directory=self.store)
        self.assertIsNotNone(by_alias)
        self.assertEqual((by_alias["host"], by_alias["port"], by_alias["cwd"]), ("192.168.1.50", 2222, "/tmp"))
        by_ip = resolve_coordinator_endpoint({"host": "192.168.1.50"}, directory=self.store)
        self.assertEqual(by_ip["port"], 2222)


class FleetProbeTests(unittest.TestCase):
    def test_parse_dual_die_910c_and_container_attribution(self):
        snap = parse_fleet_probe_output(
            SAMPLE_910C_PROBE_OUTPUT,
            alias="atlas-a3",
            host="10.0.0.12",
            port=22,
            user="root",
        )
        self.assertTrue(snap["online"])
        self.assertEqual(snap["total_npus"], 4)
        self.assertEqual(snap["free_npu_ids"], [0, 2, 3])
        self.assertEqual(snap["busy_npu_ids"], [1])
        card0 = snap["npus"][0]
        self.assertEqual(card0["die_count"], 2)
        self.assertEqual(card0["hbm_used_mb"], 2560)
        self.assertEqual(card0["hbm_total_mb"], 65536)
        self.assertFalse(card0["busy"])
        card1 = snap["npus"][1]
        self.assertTrue(card1["busy"])
        self.assertEqual(card1["processes"][0]["owner"], "alice")
        self.assertEqual(card1["processes"][0]["container_name"], "mindie-alice")
        self.assertEqual(snap["containers"][0]["ssh_port"], 46012)
        weights = [m for m in snap["mounts"] if m["likely_weights_dir"]]
        self.assertEqual(len(weights), 1)
        self.assertEqual(weights[0]["mountpoint"], "/data/weights")

    def test_parse_real_golden_a2_and_a3_fixtures(self):
        fixtures_dir = Path(__file__).parent / "fixtures"
        a2_raw = (fixtures_dir / "a2-173-125-1-2.txt").read_text(encoding="utf-8")
        a3_idle_raw = (fixtures_dir / "a3-192-168-13-165-idle.txt").read_text(encoding="utf-8")
        a3_busy_raw = (fixtures_dir / "a3-192-168-13-154-busy.txt").read_text(encoding="utf-8")

        a2 = parse_npu_smi_info(a2_raw)
        self.assertEqual(a2["smi_version"], "25.2.2")
        self.assertEqual(a2["driver_version"], "25.2.2")
        self.assertEqual(len(a2["npus"]), 8)
        self.assertEqual([n["npu_id"] for n in a2["npus"]], list(range(8)))
        self.assertEqual([n["phy_ids"] for n in a2["npus"]], [[i] for i in range(8)])
        self.assertTrue(all(not n["busy"] for n in a2["npus"]))

        a3_idle = parse_npu_smi_info(a3_idle_raw)
        self.assertEqual(a3_idle["smi_version"], "26.0.rc1")
        self.assertEqual(a3_idle["driver_version"], "26.0.rc1")
        self.assertEqual(len(a3_idle["npus"]), 8)
        self.assertEqual([n["npu_id"] for n in a3_idle["npus"]], list(range(8)))
        self.assertEqual(
            [n["phy_ids"] for n in a3_idle["npus"]],
            [[i * 2, i * 2 + 1] for i in range(8)],
        )
        self.assertTrue(all(n["die_count"] == 2 and not n["busy"] for n in a3_idle["npus"]))
        self.assertEqual(a3_idle["npus"][0]["hbm_total_mb"], 131072)

        a3_busy = parse_npu_smi_info(a3_busy_raw)
        self.assertEqual(len(a3_busy["npus"]), 8)
        self.assertTrue(all(n["busy"] for n in a3_busy["npus"]))
        self.assertEqual(len(a3_busy["processes"]), 16)
        self.assertEqual(
            sorted(p["phy_id"] for p in a3_busy["processes"]),
            list(range(16)),
        )

    def test_capacity_finder_maps_a3_dual_die_to_physical_phy_ids(self):
        fixtures_dir = Path(__file__).parent / "fixtures"
        a3_idle_raw = (fixtures_dir / "a3-192-168-13-165-idle.txt").read_text(encoding="utf-8")
        probe_out = f"===SECTION:NPU_SMI===\n{a3_idle_raw}\n===SECTION:SYSTEM===\nHOSTNAME:a3-node\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = MachineDirectory(path=root / "machines.json")
            store.upsert_machine({"alias": "atlas-a3", "host": {"ip": "192.168.13.165", "port": 22}})
            cap = find_fleet_capacity(
                count=2,
                directory=store,
                state_dir=root,
                mode="live",
                runner=lambda _spec, _script: probe_out,
            )
            self.assertEqual(cap["matched_count"], 1)
            rec = cap["recommended"]
            self.assertEqual(rec["alias"], "atlas-a3")
            self.assertEqual(rec["recommended_npu_ids"], [0, 1])
            self.assertEqual(rec["recommended_phy_ids"], [0, 1, 2, 3])
            self.assertEqual(rec["recommended_devices"], [0, 1, 2, 3])
            self.assertEqual(rec["ascend_rt_visible_devices"], "0,1,2,3")
            self.assertEqual(rec["free_phy_ids"], list(range(16)))
            self.assertTrue(rec["contiguous"])

    def test_capacity_finder_prefers_contiguous_free_npus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = MachineDirectory(path=root / "machines.json")
            store.upsert_machine({"alias": "atlas-a3", "host": {"ip": "10.0.0.12", "port": 22}})
            cap = find_fleet_capacity(
                count=2,
                directory=store,
                state_dir=root,
                mode="live",
                runner=lambda _spec, _script: SAMPLE_910C_PROBE_OUTPUT,
            )
            self.assertEqual(cap["matched_count"], 1)
            rec = cap["recommended"]
            self.assertEqual(rec["alias"], "atlas-a3")
            self.assertEqual(rec["recommended_devices"], [2, 3])
            self.assertEqual(rec["ascend_rt_visible_devices"], "2,3")
            self.assertTrue(rec["contiguous"])


class EnvManagerAndMcpIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env_patcher = mock.patch.dict(
            os.environ,
            {"MINDIE_COORDINATOR_STATE_DIR": str(self.root)},
            clear=False,
        )
        self.env_patcher.start()
        self.addCleanup(self.env_patcher.stop)

    def test_hosts_and_container_lifecycle_via_mcp_call_tool(self):
        # 1. Upsert host without needing a task context_file
        upsert_res = call_tool(
            "mindie_hosts",
            {"action": "upsert", "alias": "atlas-a3", "host": "10.0.0.12", "port": 22, "machine_type": "A3", "set_default": True},
        )
        self.assertFalse(upsert_res["isError"], upsert_res)
        data = upsert_res["structuredContent"]["data"]
        self.assertEqual(data["alias"], "atlas-a3")
        self.assertEqual(data["default_alias"], "atlas-a3")

        # 2. Bind an existing container in docker_exec mode
        bind_res = call_tool(
            "mindie_container",
            {"action": "bind", "alias": "atlas-a3", "container_name": "vllm-dev", "mode": "docker_exec", "workdir": "/vllm-workspace"},
        )
        self.assertFalse(bind_res["isError"], bind_res)
        ep = bind_res["structuredContent"]["data"]["remote_dev_endpoint"]
        self.assertEqual(ep["container"]["name"], "vllm-dev")

        # 3. List hosts and verify endpoints.json is populated
        list_res = call_tool("mindie_hosts", {"action": "list"})
        self.assertFalse(list_res["isError"])
        self.assertEqual(len(list_res["structuredContent"]["data"]["machines"]), 1)
        self.assertIn("atlas-a3", list_res["structuredContent"]["data"]["remote_dev_endpoints"])

        # 4. Unbind container
        rem_res = call_tool("mindie_container", {"action": "remove", "alias": "atlas-a3", "unbind_only": True})
        self.assertFalse(rem_res["isError"], rem_res)
        self.assertNotIn("container", rem_res["structuredContent"]["data"]["remote_dev_endpoint"])

    def test_probe_actions_preserve_structured_fields_through_presentation(self):
        call_tool("mindie_hosts", {"action": "upsert", "alias": "atlas-a3", "host": "10.0.0.12"})
        with mock.patch(
            "mindie_coordinator.fleet_probe._default_ssh_runner",
            return_value=SAMPLE_910C_PROBE_OUTPUT,
        ):
            cap = mindie_call("mindie.probe", {"action": "capacity", "npu_count": 2, "mode": "live"})["result"]
            self.assertEqual(cap["outcome"], "success")
            self.assertEqual(cap["data"]["recommended"]["recommended_devices"], [2, 3])

            npu = mindie_call("mindie.probe", {"action": "npu", "alias": "atlas-a3", "mode": "cache"})["result"]
            self.assertEqual(npu["outcome"], "success")
            self.assertEqual(len(npu["data"]["hosts"][0]["npus"]), 4)

            mounts = mindie_call("mindie.probe", {"action": "mounts", "alias": "atlas-a3", "mode": "cache"})["result"]
            self.assertEqual(mounts["data"]["hosts"][0]["weights_mounts"][0]["mountpoint"], "/data/weights")

    def test_lease_acquire_heartbeat_and_release_with_real_host_coordinator(self):
        from mindie_coordinator.host import mindie_npu_coordination

        fixtures_dir = Path(__file__).parent / "fixtures"
        a2_raw = (fixtures_dir / "a2-173-125-1-2.txt").read_text(encoding="utf-8")
        host_state_dir = self.root / "remote_host_state"

        store = MachineDirectory(path=self.root / "machines.json")
        store.upsert_machine({"alias": "atlas-a2", "host": {"ip": "173.125.1.2", "port": 22}})

        class RealCoordinatorHostQueue(HostQueue):
            def __init__(self):
                super().__init__(lambda _target, _cmd: "{}")

            def request(self, target, payload):
                req = {**dict(payload), "state_dir": str(host_state_dir)}
                return mindie_npu_coordination.handle_request(
                    req,
                    probe=lambda: mindie_npu_coordination.parse_npu_smi_info(a2_raw),
                    listening_ports=lambda: {"status": "ok", "ports": [], "source": "test"},
                )

        queue = RealCoordinatorHostQueue()
        acquired = handle_lease(
            {"action": "acquire", "alias": "atlas-a2", "lease_id": "lease-alice-1", "npu_count": 2, "service_port": 0},
            directory=store,
            queue=queue,
            state_dir=self.root,
        )
        self.assertEqual(acquired["state"], "active")
        self.assertEqual(acquired["lease_state"], "granted")
        self.assertEqual(acquired["devices"], [0, 1])
        self.assertEqual(acquired["ascend_rt_visible_devices"], "0,1")
        self.assertGreaterEqual(acquired["service_port"], 18000)
        self.assertIsNotNone(acquired["fence_token"])
        self.assertEqual(acquired["env"]["ASCEND_RT_VISIBLE_DEVICES"], "0,1")

        hb = handle_lease(
            {"action": "heartbeat", "alias": "atlas-a2", "lease_id": "lease-alice-1"},
            directory=store,
            queue=queue,
            state_dir=self.root,
        )
        self.assertEqual(hb["state"], "active")
        self.assertEqual(hb["lease_state"], "granted")
        self.assertEqual(hb["fence_token"], acquired["fence_token"])

        status = handle_lease(
            {"action": "status", "alias": "atlas-a2", "lease_id": "lease-alice-1"},
            directory=store,
            queue=queue,
            state_dir=self.root,
        )
        self.assertEqual(status["count"], 1)
        self.assertEqual(status["leases"][0]["lease_id"], "lease-alice-1")

        rel = handle_lease(
            {"action": "release", "alias": "atlas-a2", "lease_id": "lease-alice-1"},
            directory=store,
            queue=queue,
            state_dir=self.root,
        )
        self.assertEqual(rel["state"], "released")

    def test_container_provision_reserves_port_and_preserves_multi_alias_same_host(self):
        from mindie_coordinator import provision
        from mindie_coordinator.host import mindie_npu_coordination

        fixtures_dir = Path(__file__).parent / "fixtures"
        a3_idle_raw = (fixtures_dir / "a3-192-168-13-165-idle.txt").read_text(encoding="utf-8")
        host_state_dir = self.root / "remote_host_state"
        store = MachineDirectory(path=self.root / "machines.json")
        store.upsert_machine({"alias": "atlas-a3-alice", "host": {"ip": "192.168.13.165", "port": 22}})
        store.upsert_machine({"alias": "atlas-a3-bob", "host": {"ip": "192.168.13.165", "port": 22}})
        self.assertEqual(len(store.machines()), 2)

        reserved_requests = []

        class RealPortReserveQueue(HostQueue):
            def __init__(self):
                super().__init__(lambda _target, _cmd: "{}")

            def request(self, target, payload):
                reserved_requests.append(payload)
                req = {**dict(payload), "state_dir": str(host_state_dir)}
                return mindie_npu_coordination.handle_request(
                    req,
                    probe=lambda: mindie_npu_coordination.parse_npu_smi_info(a3_idle_raw),
                    listening_ports=lambda: {"status": "ok", "ports": [], "source": "test"},
                )

        with mock.patch.object(
            provision.host_ops,
            "run_remote_script",
            side_effect=["probe", "bootstrap", "smoke"],
        ) as remote_calls, mock.patch.object(
            provision.host_ops,
            "assert_remote_success",
            side_effect=[{"free_port": 46000}, {}, {}],
        ), mock.patch.object(
            provision.host_ops,
            "ensure_key",
            return_value=(str(self.root / "key"), "ssh-ed25519 test"),
        ), mock.patch.object(
            provision.host_ops,
            "fetch_release_catalog",
            return_value=[{"tag_name": "v0.13.0", "draft": False, "prerelease": False}],
        ):
            res = handle_container(
                {
                    "action": "provision",
                    "alias": "atlas-a3-bob",
                    "user_tag": "bob",
                    "image": "stable",
                },
                directory=store,
                queue=RealPortReserveQueue(),
            )
        self.assertEqual(res["alias"], "atlas-a3-bob")
        image_request = json.loads(remote_calls.call_args_list[1].kwargs["args"][2])
        self.assertEqual(image_request["policy"], "latest-official-release")
        self.assertEqual(image_request["resolved_tag"], "v0.13.0")
        self.assertEqual(image_request, json.loads(remote_calls.call_args_list[0].kwargs["args"][0]))
        self.assertTrue(any(r.get("action") == "container-ssh-reserve" for r in reserved_requests))
        # Verify both aliases on 192.168.13.165 remain intact
        aliases = [m["alias"] for m in store.machines()]
        self.assertEqual(aliases, ["atlas-a3-alice", "atlas-a3-bob"])
        bob_rec = store.record("atlas-a3-bob")
        self.assertEqual(bob_rec["container"]["name"], "mindie-bob")
        self.assertGreaterEqual(bob_rec["container"]["ssh_port"], 2222)


    def test_coordinator_to_remote_dev_six_harness_bridge(self):
        import remote_dev.core.endpoint as rd_endpoint
        import remote_dev.mcp.tools as rd_tools
        from mindie_coordinator.endpoint_bridge import setup_resolvers

        call_tool(
            "mindie_hosts",
            {
                "action": "upsert",
                "alias": "atlas-a3",
                "host": "10.0.0.88",
                "port": 22,
                "user": "root",
                "machine_type": "A3",
                "workdir": "/vllm-workspace",
                "set_default": True,
            },
        )
        call_tool(
            "mindie_container",
            {
                "action": "bind",
                "alias": "atlas-a3",
                "container_name": "mindie-dev-88",
                "mode": "docker_exec",
                "workdir": "/vllm-workspace",
            },
        )

        # Register coordinator's setup_resolvers with remote_dev
        with mock.patch.object(rd_endpoint, "_RESOLVERS", []), mock.patch.object(
            rd_endpoint, "_ENV_RESOLVERS_LOADED", True
        ):
            setup_resolvers()
            # 1. Antigravity view_file style resolved via coordinator alias
            with mock.patch.object(rd_tools, "remote_read", return_value={"ok": True}) as mock_read:
                rd_tools.call_tool(
                    "remote_view_file",
                    {
                        "alias": "atlas-a3",
                        "AbsolutePath": "/vllm-workspace/model.py",
                        "StartLine": 10,
                        "EndLine": 25,
                    },
                )
                ep = mock_read.call_args.args[0]
                self.assertEqual((ep.host, ep.port, ep.container), ("10.0.0.88", 22, "mindie-dev-88"))
                self.assertEqual(mock_read.call_args.kwargs["file_path"], "/vllm-workspace/model.py")
                self.assertEqual(mock_read.call_args.kwargs["start_line"], 10)
                self.assertEqual(mock_read.call_args.kwargs["end_line"], 25)

            # 2. Codex exec_command style resolved via coordinator target_alias
            with mock.patch.object(rd_tools, "remote_bash", return_value={"ok": True}) as mock_bash:
                rd_tools.call_tool(
                    "remote_exec_command",
                    {
                        "target_alias": "atlas-a3-host",
                        "cmd": "npu-smi info",
                        "workdir": "/vllm-workspace",
                    },
                    profile="codex",
                )
                ep_host = mock_bash.call_args.args[0]
                self.assertEqual((ep_host.host, ep_host.port, ep_host.container), ("10.0.0.88", 22, None))
                self.assertEqual(mock_bash.call_args.kwargs["command"], "npu-smi info")

    def test_fleet_probe_script_never_runs_untimed_npu_smi_after_timeout(self):
        import subprocess
        from mindie_coordinator.fleet_probe import REMOTE_FLEET_PROBE_SCRIPT

        bin_dir = self.root / "fake_bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        calls_log = self.root / "npu_smi_calls.log"
        fake_npu_smi = bin_dir / "npu-smi"
        fake_npu_smi.write_text(
            "#!/bin/sh\necho \"direct:$*\" >> npu_smi_calls.log\nexit 0\n",
            encoding="utf-8",
            newline="\n",
        )
        fake_npu_smi.chmod(0o755)
        fake_timeout = bin_dir / "timeout"
        fake_timeout.write_text(
            "#!/bin/sh\necho \"timed:$*\" >> npu_smi_calls.log\nexit 124\n",
            encoding="utf-8",
            newline="\n",
        )
        fake_timeout.chmod(0o755)

        npu_section = REMOTE_FLEET_PROBE_SCRIPT.split('echo "===SECTION:CONTAINERS==="')[0] + 'echo "===SECTION:CONTAINERS==="\n'
        npu_section = npu_section.replace(
            'export PATH="/usr/local/Ascend/driver/tools:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"',
            'export PATH="$PWD/fake_bin:$PATH"',
        )
        # Git's sh on Windows also needs shell-native paths and LF scripts.
        res = subprocess.run(
            ["sh", "-c", npu_section], cwd=self.root, capture_output=True, text=True, timeout=5, check=True
        )
        self.assertIn("===SECTION:NPU_SMI===", res.stdout)
        recorded = calls_log.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(recorded, ["timed:10 npu-smi info"])

    def test_upsert_with_in_memory_password_bootstraps_key_syncs_npu_top_and_discovers_existing_container(self):
        from mindie_coordinator import provision

        identity_file = str(self.root / "managed-key")
        setup_calls = []
        npu_top_calls = []

        def shared_setup(endpoint, **kwargs):
            setup_calls.append((endpoint, kwargs))
            return {"auth_ref": "shared-host-profile", "auth_method": "publickey",
                    "identity_file": identity_file, "bootstrap": {"status": "completed"}}

        def fake_register_host(self_client, **kwargs):
            npu_top_calls.append(kwargs)
            return {"source": "npu-top", "status": "registered", "server": {"host": kwargs["host"]}, "auth": {"ready": 1}}

        probe_output = (
            SAMPLE_910C_PROBE_OUTPUT.split("===SECTION:CONTAINERS===")[0]
            + "===SECTION:CONTAINERS===\n"
            + "f6327a9012bc|vllm_debug_c0|swr.cn-south-1.myhuaweicloud.com/ascendhub/vllm-ascend:v0.13.0-a3|Up 3 days|0.0.0.0:32222->22/tcp\n"
            + "===SECTION:SYSTEM===\nHOSTNAME:atlas-a3-165\n"
        )

        with mock.patch.object(provision.host_ops, "setup_access", side_effect=shared_setup), \
             mock.patch("mindie_coordinator.npu_top_bridge.NpuTopObservationClient.register_host", fake_register_host):
            res = handle_hosts(
                {
                    "action": "upsert",
                    "alias": "a3-165",
                    "host": "192.168.13.165",
                    "port": 22,
                    "user": "root",
                    "password": "in-memory-secret-pw",
                    "machine_type": "A3",
                    "container_name": "f6327a90",
                    "probe_hardware": True,
                    "set_default": True,
                },
                state_dir=self.root,
                runner=lambda _target, _script: probe_output,
            )

        self.assertEqual(res["state"], "ready")
        self.assertEqual(len(setup_calls), 1)
        self.assertEqual(setup_calls[0][1]["password"], "in-memory-secret-pw")
        self.assertEqual(res["record"]["host"]["identity_file"], identity_file)
        self.assertEqual(res["record"]["host"]["auth_ref"], "shared-host-profile")
        self.assertEqual(len(npu_top_calls), 1)
        self.assertEqual(npu_top_calls[0]["host"], "192.168.13.165")
        self.assertEqual(npu_top_calls[0]["auth_ref"], "shared-host-profile")
        self.assertNotIn("password", npu_top_calls[0])
        self.assertEqual(res["npu_top_sync"]["status"], "registered")
        # Container matched by ID prefix and populated with full container name + published ssh_port
        self.assertEqual(res["record"]["container"]["name"], "vllm_debug_c0")
        self.assertEqual(res["record"]["container"]["id"], "f6327a9012bc")
        self.assertEqual(res["record"]["container"]["ssh_port"], 32222)
        # Password is never written to machines.json or remote_dev_endpoints.json
        machines_raw = (self.root / "machines.json").read_text(encoding="utf-8")
        endpoints_raw = Path(res["endpoints_file"]).read_text(encoding="utf-8")
        self.assertNotIn("in-memory-secret-pw", machines_raw)
        self.assertNotIn("in-memory-secret-pw", endpoints_raw)

    def test_existing_container_reuse_in_service_and_runtime_pool(self):
        from mindie_coordinator.service import CoordinatorService
        from mindie_coordinator.ready_runtime import RuntimePool

        handle_hosts(
            {
                "action": "upsert",
                "alias": "a2-194",
                "host": "173.125.1.2",
                "port": 22,
                "user": "root",
                "machine_type": "A2",
            },
            state_dir=self.root,
        )
        handle_container(
            {
                "action": "bind",
                "alias": "a2-194",
                "container_name": "existing_user_container",
                "mode": "docker_exec",
                "workdir": "/workspace",
            },
            state_dir=self.root,
        )

        backend = mock.MagicMock()
        backend.machines = MachineDirectory(path=self.root / "machines.json")
        backend.inspect.return_value = {"container_id": "cid-existing-123", "profile": {"soc": "Ascend910B2"}}
        pool = RuntimePool(self.root / "pool", backend=backend)
        service = CoordinatorService(self.root, pool=pool, backend=backend)

        # 1. Reuses bound docker_exec container without recipe or mindie-<user> port reservation
        donor = service._ensure_user_container("alice", {}, {"name": "worker"}, set())
        self.assertIsNotNone(donor)
        self.assertEqual(donor["container_name"], "existing_user_container")
        self.assertEqual(donor["endpoint"]["container"], "existing_user_container")
        self.assertEqual(donor["endpoint"]["port"], 22)
        backend.host.assert_not_called()

        # 2. Explicit environment={"container_name": "other_debug_box"} overrides bound container
        override = service._ensure_user_container(
            "alice",
            {"container_name": "other_debug_box"},
            {"name": "worker", "host": "a2-194"},
            set(),
        )
        self.assertIsNotNone(override)
        self.assertEqual(override["container_name"], "other_debug_box")
        self.assertEqual(override["endpoint"]["container"], "other_debug_box")

        # 3. RuntimePool.register accepts non-mindie-<user> container names
        registered = pool.register(
            "rt-existing",
            {
                "user": "alice",
                "python": "/usr/bin/python3",
                "container_name": "existing_user_container",
                "endpoint": {
                    "host": "173.125.1.2",
                    "port": 22,
                    "user": "root",
                    "container": "existing_user_container",
                    "root": "/workspace/tasks/t1",
                    "cwd": "/workspace/tasks/t1",
                },
                "host_endpoint": {"host": "173.125.1.2", "port": 22, "user": "root"},
            },
        )
        self.assertEqual(registered["container_name"], "existing_user_container")
        self.assertEqual(registered["state"], "ready")
        backend.host.assert_not_called()


if __name__ == "__main__":
    unittest.main()
