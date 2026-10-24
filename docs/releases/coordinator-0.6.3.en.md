# mindie-coordinator 0.6.3: Preparation Job Status Refresh, HBM Telemetry Consistency & Observed-Capacity Placement

[中文](https://github.com/mindie-agent/coordinator/blob/v0.6.3/docs/releases/coordinator-0.6.3.md)

`mindie-coordinator 0.6.3` restores remote preparation job status refresh and compile-lease teardown, keeps measured used HBM and card/Die supplement telemetry consistent across single-Die and dual-Die NPUs, and enforces declared CPU, memory, and aggregate free-HBM floors against observed host capacity throughout placement and container preparation.

---

## 1. Key Fixes & Behavior Changes

### 1. Remote Preparation Job Refresh & Compile Lease Teardown (`service_gang.py`, `service_evidence.py`, `service.py`, PR `#63`)

- Restored the missing `preparation_control` import in `service_gang.py` so refreshing remote preparation jobs queries real remote status instead of hitting an internal `NameError` and recording `state="uncertain"`.
- Wired preparation job refresh into status observation and terminal progression so container preparation and source build jobs update remote state and release compile leases cleanly.

### 2. Factual Used HBM Preservation & Card/Die Telemetry Consistency (`fleet_probe_parser.py`, PR `#64`)

- **Preserving Measured Used HBM on A5 Cards**: On Ascend 950DT (A5), device busy status continues to follow the process table (keeping idle cards leasable when no workload processes are attached), while the measured total used HBM (`hbm_used_mb`) from `npu-smi info` is preserved as-is for allocation headroom calculations rather than zeroed or reduced by an assumed baseline.
- **Consistent Card/Die `-t usages` Enrichment**: `parse_npu_smi_usages` applies card-wide `npu-smi info -t usages` HBM figures only to single-Die or card-only rows and synchronizes the single-Die entry, leaving multi-Die A3 cards untouched so card-wide numbers never overwrite per-Die telemetry. Card and physical-Die free/busy lists are derived consistently via `device_occupancy_inventory`.

### 3. Observed-Capacity Admission Across Placement & Preparation (`service_evidence.py`, `service_gang.py`, `service.py`, `placement.py`, PR `#64`)

- **Per-Decision Host Spec Snapshot**: When any role declares `cpu_cores`, `memory_mb`, or `hbm_mb`, `ServiceEvidenceMixin._observe_host_specs` probes up to `8` candidate hosts once via `_probe_host_record` (`probe_machine_record`) and normalizes the results into a `host_specs` snapshot.
- **Enforced Across the `CoordinatorService` Flow**: Within `CoordinatorService`, that `host_specs` snapshot is checked across:
  1. **Donor Selection**: checking environment compatibility via `can_prepare` first and host capacity via `hardware_shortfall` second;
  2. **User Container Provisioning**: skipping undersized hosts via `hardware_shortfall` before provisioning;
  3. **Post-Preparation** and **Final Role Binding Verification**: verifying `runtime_matches(..., host_specs)` before binding (the standalone `select_runtimes(..., host_specs=...)` helper also accepts `host_specs`).
- **Host-Level Feasibility Screening & Shortfall Reporting**: `hbm_mb`, `cpu_cores`, and `memory_mb` are evaluated as host-level feasibility floors against observed `cpu_cores`, usable memory, and aggregate free HBM (`vram_free_mb`), while per-Die allocation authority remains with the host queue. When a host is unobserved or undersized, `shortfall_reason` reports deduplicated per-host shortfall details.

---

## 2. Verification

- Local unit and contract suites covering preparation job refresh, factual A5 used HBM preservation, single/dual-Die `-t usages` consistency, host-capacity admission, and `shortfall_reason` reporting all passed.
- Verified on standing A2 (`173.125.1.2`) and A3 (`192.168.13.165`, `192.168.13.154`) hosts via `test_live_fleet_probe_and_capacity_across_a2_and_a3` (**`1 passed` in `8.01s`**).
