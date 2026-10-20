# ruff: noqa: E702  # deliberate one-line guards keep this module under 380 lines
"""HCCL / RoCE / HCCS preflight diagnostics distilled from CANNBot guidance.
[Role]
Pure parsers/validators for hccn_tool evidence, HCCL environment, and ranktables.
[Invariants]
- No subprocess, filesystem, network, environment mutation, or hidden state.
- Rank IDs are unique and contiguous 0..N-1; device IPs are unique and valid.
  A3 super-pod `super_device_id`/`Phy-ID` is unique per super pod, not globally.
- PCIE and ROCE intra flags are mutually exclusive; EXEC >= CONNECT timeout.
- HCCL_BUFFSIZE is in MiB and must be >= 1.
[Timeouts & Locks]
- No timeouts, sleeps, locks, or I/O; bounded by input length.
[Side-Effect Policy]
- Strictly read-only; never mutates inputs or the process environment.
"""
from __future__ import annotations
import ipaddress
import re
from collections.abc import Mapping
from typing import Any
_IFNAME = re.compile(r"[A-Za-z0-9_.:-]+")

def _soc_family(soc: str) -> str:
    text = str(soc or "").upper().replace("-", "_")
    if any(token in text for token in ("910_93", "91093", "910C", "A3")):
        return "A3"
    if any(token in text for token in ("910_95", "91095", "950", "A5")):
        return "A5"
    return "A2" if any(token in text for token in ("910B", "910A", "A2")) else "unknown"
def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return None
    token = value.strip().replace(",", "").replace("_", "")
    try:
        return int(token, 10)
    except ValueError:
        return int(token, 16) if token.lower().startswith("0x") else None
def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    match = re.search(r"0[xX][0-9a-fA-F]+|[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(value))
    if not match:
        return None
    token = match.group(0)
    return int(token, 16) if token.lower().startswith("0x") else float(token)
def _norm_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
def _pairs(output: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for line in output.splitlines():
        match = re.match(r"\s*([A-Za-z][A-Za-z0-9 _./()\-]*?)\s*[:=]\s*(\S.*?)\s*$", line)
        if match:
            pairs.setdefault(_norm_key(match.group(1)), match.group(2))
    return pairs
def _lookup(pairs: Mapping[str, str], aliases: tuple[str, ...]) -> str | None:
    return next((pairs[_norm_key(a)] for a in aliases if _norm_key(a) in pairs), None)
def _lookup_norm(mapping: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
    normalized = {_norm_key(key): value for key, value in mapping.items()}
    return next((normalized[_norm_key(a)] for a in aliases if _norm_key(a) in normalized), None)
def _first_int(pairs: Mapping[str, str], aliases: tuple[str, ...]) -> int:
    number = _number(_lookup(pairs, aliases))
    return int(number) if number is not None and number >= 0 else 0
def _present(pairs: Mapping[str, str], output: str) -> bool | None:
    value = _lookup(pairs, ("present", "optical_present", "module_present", "optical_module_present"))
    if value is None:
        return False if re.search(r"\boptical\b[^\n]*\babsent\b", output, re.I) else None
    token = value.strip().lower()
    if token in {"present", "yes", "true", "1", "on", "ok"}:
        return True
    return False if token in {"absent", "no", "false", "0", "off", "none"} else None
def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
def _ifname_issue(value: str, *, multi_node: bool) -> str | None:
    if re.search(r"\s", value):
        return "contains whitespace"
    parts = [part for part in value.split(",") if part]
    if not parts:
        return "is empty"
    for part in parts:
        name = part[2:] if part.startswith("^=") else part.lstrip("^=")
        if not name:
            return "contains an empty interface name"
        if not _IFNAME.fullmatch(name):
            return f"contains invalid interface name {name!r}"
    names = [(part[1:] if part.startswith("=") else part)
             for part in parts if multi_node and not part.startswith("^")]
    return "selects loopback only" if names and all(n == "lo" for n in names) else None
def parse_hccn_link_and_optical(
    device_id: int, *, link_output: str, optical_output: str = "",
    stat_output: str = "", ip_output: str = "",
) -> dict[str, Any]:
    """Parse `hccn_tool` link/optical/stat/ip output for one device."""
    if isinstance(device_id, bool) or not isinstance(device_id, int) or device_id < 0:
        raise ValueError("device_id must be a non-negative integer")
    for name, value in (("link_output", link_output), ("optical_output", optical_output),
                        ("stat_output", stat_output), ("ip_output", ip_output)):
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
    issues: list[str] = []
    match = re.search(r"\blink\s*(?:status|state)\s*[:=]\s*(UP|DOWN)\b", link_output, re.I)
    if not match:
        match = re.search(r"\blink\b[^\n]*?[:=]\s*(UP|DOWN)\b", link_output, re.I)
    if not match:
        match = re.search(r"\blink\b[^\n]{0,40}?\b(UP|DOWN)\b", link_output, re.I)
    link_up = bool(match and match.group(1).upper() == "UP")
    if not match:
        issues.append("link status not reported by hccn_tool -link -g")
    elif not link_up:
        issues.append("link status is DOWN")
    optical = _pairs(optical_output)
    optical_present = _present(optical, optical_output)
    tx_power = _number(_lookup(optical, ("tx_power", "tx_power_dbm", "txpower", "tx_optical_power", "transmit_power")))
    rx_power = _number(_lookup(optical, ("rx_power", "rx_power_dbm", "rxpower", "rx_optical_power", "receive_power")))
    temp = _number(_lookup(optical, ("optical_temp", "optical_temperature", "optical_temp_c", "temperature_c", "module_temp", "module_temp_c", "temperature", "temp")))
    if optical_present is False:
        issues.append("optical module absent")
    stat = _pairs(stat_output)
    crc = _first_int(stat, ("mac_rx_fcs_pkt_num", "mac_rx_fcs_err_pkt_num", "crc_err_cnt", "crc_errors", "fcs_err_cnt"))
    pcs = _first_int(stat, ("pcs_err_cnt", "pcs_err_num", "pcs_errors", "mac_rx_pcs_err_pkt_num"))
    symbol = _first_int(stat, ("mac_rx_symbol_err_pkt_num", "symbol_err_cnt", "symbol_errors", "symbol_error_count"))
    if crc:
        issues.append(f"CRC/FCS errors: {crc} (mac_rx_fcs_pkt_num)")
    if pcs:
        issues.append(f"PCS errors: {pcs} (pcs_err_cnt)")
    if symbol:
        issues.append(f"symbol errors: {symbol}")
    ip = _pairs(ip_output)
    ip_address = _lookup(ip, ("ipaddr", "ip_addr", "ip_address", "address", "device_ip"))
    netmask = _lookup(ip, ("netmask", "mask", "subnet_mask"))
    if ip_address:
        ip_address = ip_address.split()[0].split("/")[0]
        try:
            ipaddress.ip_address(ip_address)
        except ValueError:
            issues.append(f"invalid device IP: {ip_address}")
            ip_address = None
    elif ip_output.strip():
        issues.append("device IP not reported by hccn_tool -ip -g")
    if netmask:
        netmask = netmask.split()[0]
    return {"device_id": device_id, "status": "down" if not link_up else "degraded" if issues else "healthy",
            "link_up": link_up, "optical_present": optical_present, "crc_errors": crc,
            "pcs_errors": pcs, "symbol_errors": symbol, "tx_power_dbm": tx_power,
            "rx_power_dbm": rx_power, "optical_temperature_c": temp, "ip_address": ip_address,
            "netmask": netmask, "issues": issues}
def audit_hccl_environment(
    env: Mapping[str, str], *, multi_node: bool = False,
    pd_disaggregated: bool = False, soc: str = "Ascend910B",
) -> dict[str, Any]:
    """Audit HCCL/RoCE/Gloo environment variables against CANNBot invariants."""
    if not isinstance(env, Mapping):
        raise TypeError("env must be a mapping")
    values = {str(key): "" if value is None else str(value) for key, value in env.items()}
    warnings: list[str] = []
    errors: list[str] = []
    recommended: dict[str, str] = {}
    family = _soc_family(soc)
    moe = any(any(token in key.upper() for token in ("MOE", "EXPERT", "EP_SIZE", "EP_WORLD", "USE_EP", "ALL2ALL", "ALL_TO_ALL")) for key in values); network = multi_node or pd_disaggregated
    whitelist = values.get("HCCL_WHITELIST_DISABLE", "").strip()
    if whitelist:
        flag = _int_value(whitelist)
        if flag not in (0, 1):
            errors.append("HCCL_WHITELIST_DISABLE must be 0 or 1")
        elif flag == 0 and not values.get("HCCL_WHITELIST_FILE", "").strip():
            warnings.append("HCCL_WHITELIST_DISABLE=0 enables whitelist checking but HCCL_WHITELIST_FILE is unset")
            recommended["HCCL_WHITELIST_DISABLE"] = "1"
    socket_ifname = values.get("HCCL_SOCKET_IFNAME", "").strip()
    if_ip = values.get("HCCL_IF_IP", "").strip()
    gloo_ifname = values.get("GLOO_SOCKET_IFNAME", "").strip()
    if not socket_ifname and not if_ip and network:
        warnings.append("HCCL_SOCKET_IFNAME/HCCL_IF_IP unset; automatic NIC selection can bind the wrong interface")
        recommended["HCCL_SOCKET_IFNAME"] = "<roce-nic-name>"
    if socket_ifname and if_ip:
        warnings.append("both HCCL_IF_IP and HCCL_SOCKET_IFNAME are set; HCCL_IF_IP takes precedence")
    if if_ip:
        try:
            address = ipaddress.ip_address(if_ip)
        except ValueError:
            errors.append(f"HCCL_IF_IP is not a valid IP address: {if_ip!r}")
        else:
            if multi_node and address.is_loopback:
                errors.append("HCCL_IF_IP must not be a loopback address on multi-node jobs")
    if socket_ifname:
        issue = _ifname_issue(socket_ifname, multi_node=multi_node)
        if issue:
            errors.append(f"HCCL_SOCKET_IFNAME {issue}: {socket_ifname!r}")
    if network and not gloo_ifname:
        warnings.append("GLOO_SOCKET_IFNAME unset; Gloo CPU groups may bind the wrong NIC or IPv6")
        recommended["GLOO_SOCKET_IFNAME"] = "<host-nic-name>"
    elif gloo_ifname and re.search(r"\s", gloo_ifname):
        errors.append("GLOO_SOCKET_IFNAME must not contain whitespace")
    buffsize = values.get("HCCL_BUFFSIZE", "").strip()
    buffsize_mb = _int_value(buffsize) if buffsize else None
    if buffsize and (buffsize_mb is None or buffsize_mb < 1):
        errors.append("HCCL_BUFFSIZE must be an integer number of MiB >= 1")
        recommended["HCCL_BUFFSIZE"] = "1024"
    elif (network or moe) and buffsize_mb is not None and buffsize_mb < 200:
        warnings.append(f"HCCL_BUFFSIZE={buffsize_mb} MiB is below the 200 MiB default; large All2All/MoE traffic can stall")
        recommended["HCCL_BUFFSIZE"] = "1024"
    elif (network or moe) and not buffsize:
        warnings.append("HCCL_BUFFSIZE is unset; the 200 MiB default may be too small for multi-node All2All/MoE")
        recommended["HCCL_BUFFSIZE"] = "1024"
    expansion = values.get("HCCL_OP_EXPANSION_MODE", "").strip().upper().replace("-", "_")
    if expansion and expansion not in {"AIV", "AI_CPU", "AICPU", "HOST"}:
        errors.append(f"HCCL_OP_EXPANSION_MODE has unsupported value {expansion!r}")
        recommended["HCCL_OP_EXPANSION_MODE"] = "AIV" if family in {"A2", "A3"} else "AI_CPU"
    for key, low, high, multiple in (("HCCL_RDMA_TC", 0, 255, 4), ("HCCL_RDMA_SL", 0, 7, 1)):
        raw = values.get(key, "").strip()
        number = _int_value(raw) if raw else None
        if raw and (number is None or not low <= number <= high or number % multiple):
            errors.append(f"{key} must be an integer in [{low},{high}]" + (" and a multiple of 4" if multiple == 4 else ""))
    if multi_node and bool(values.get("HCCL_RDMA_TC")) != bool(values.get("HCCL_RDMA_SL")):
        warnings.append("HCCL_RDMA_TC and HCCL_RDMA_SL should be configured together to match switch QoS/PFC")
    timeouts: dict[str, int | None] = {}
    for key, minimum in (("HCCL_CONNECT_TIMEOUT", 1), ("HCCL_EXEC_TIMEOUT", 0)):
        raw = values.get(key, "").strip()
        number = _int_value(raw) if raw else None
        if raw and (number is None or number < minimum):
            errors.append(f"{key} must be an integer >= {minimum} seconds")
            number = None
        timeouts[key] = number
    connect, execute = timeouts["HCCL_CONNECT_TIMEOUT"], timeouts["HCCL_EXEC_TIMEOUT"]
    if connect is not None and execute not in (None, 0) and execute < connect:
        errors.append("HCCL_EXEC_TIMEOUT must be >= HCCL_CONNECT_TIMEOUT; otherwise link setup can exceed execution timeout")
    if network:
        recommended.setdefault("HCCL_CONNECT_TIMEOUT", "1200")
        recommended.setdefault("HCCL_EXEC_TIMEOUT", "1200")
    if _truthy(values.get("ASCEND_LAUNCH_BLOCKING")):
        (warnings if any("DEBUG" in key.upper() and _truthy(value) for key, value in values.items()) else errors).append("ASCEND_LAUNCH_BLOCKING=1 forces synchronous launch; unset it for production/throughput runs")
        recommended["ASCEND_LAUNCH_BLOCKING"] = "0"
    pcie = _int_value(values.get("HCCL_INTRA_PCIE_ENABLE")) if values.get("HCCL_INTRA_PCIE_ENABLE") else None
    roce = _int_value(values.get("HCCL_INTRA_ROCE_ENABLE")) if values.get("HCCL_INTRA_ROCE_ENABLE") else None
    for key, flag in (("HCCL_INTRA_PCIE_ENABLE", pcie), ("HCCL_INTRA_ROCE_ENABLE", roce)):
        if values.get(key) and flag not in (0, 1):
            errors.append(f"{key} must be 0 or 1")
    if pcie == 1 and roce == 1:
        errors.append("HCCL_INTRA_PCIE_ENABLE and HCCL_INTRA_ROCE_ENABLE cannot both be 1")
    if family == "A5" and roce == 1:
        errors.append("HCCL_INTRA_ROCE_ENABLE=1 is not supported on A5 (Ascend910_95) super-pods")
    elif family == "A3" and roce == 1:
        llm_datadist = any("DATADIST" in key.upper() and _truthy(value) for key, value in values.items())
        warnings.append("A3 intra-node RoCE requires LLM-DataDist, validated RoCE links, and matching HCCL_RDMA_TC/SL" if llm_datadist
                        else "HCCL_INTRA_ROCE_ENABLE=1 on A3 only affects LLM-DataDist; HCCL still uses HCCS")
    elif family == "A2" and roce == 1:
        warnings.append("HCCL_INTRA_ROCE_ENABLE=1 forces A2 intra-node RoCE instead of HCCS; verify RoCE health and bandwidth")
    if roce == 1:
        recommended["HCCL_INTRA_ROCE_ENABLE"] = "0"
    return {"status": "invalid" if errors else "warning" if warnings else "ok",
            "warnings": warnings, "errors": errors, "recommended_env": recommended,
            "soc": soc, "multi_node": multi_node, "pd_disaggregated": pd_disaggregated, "moe_inferred": moe}
def validate_hccl_ranktable(ranktable: Mapping[str, Any], *, soc: str = "Ascend910B") -> dict[str, Any]:
    """Validate a ranktable v1.0 AI-Server table or v1.2 A3 super-pod table."""
    if not isinstance(ranktable, Mapping):
        return {"valid": False, "status": "invalid", "errors": ["ranktable must be a mapping"],
                "warnings": [], "rank_count": 0, "server_count": 0, "super_pod_count": 0}
    errors: list[str] = []
    warnings: list[str] = []
    version = str(ranktable.get("version", "")).strip()
    table_status = str(ranktable.get("status", "")).strip()
    servers = ranktable.get("server_list")
    super_pods = ranktable.get("super_pod_list")
    if version not in {"1.0", "1.2"}:
        errors.append("ranktable version must be '1.0' (or '1.2' for A3 super-pod mode)")
    if table_status and table_status.lower() != "completed":
        errors.append(f"ranktable status={table_status!r} is not 'completed'")
    if not isinstance(servers, list) or not servers:
        errors.append("server_list must be a non-empty list")
        servers = []
    else:
        declared = _int_value(ranktable.get("server_count")) if "server_count" in ranktable else len(servers)
        if declared is None:
            errors.append("server_count must be an integer")
        elif declared != len(servers):
            errors.append(f"server_count={declared} does not match server_list length {len(servers)}")
    server_ids: list[str] = []
    flat_devices: list[tuple[str, Mapping[str, Any]]] = []
    seen_device_ids: set[tuple[str, int]] = set()
    for index, server in enumerate(servers):
        if not isinstance(server, Mapping):
            errors.append(f"server_list[{index}] must be a mapping"); continue
        server_id = str(server.get("server_id", "")).strip()
        if not server_id:
            errors.append(f"server_list[{index}] is missing server_id"); continue
        if server_id in server_ids:
            errors.append(f"duplicate server_id: {server_id!r}"); continue
        server_ids.append(server_id)
        devices = server.get("device")
        if not isinstance(devices, list) or not devices:
            errors.append(f"server {server_id!r} must contain a non-empty device list"); devices = []
        for device_index, device in enumerate(devices):
            if not isinstance(device, Mapping):
                errors.append(f"server {server_id!r} device[{device_index}] must be a mapping"); continue
            flat_devices.append((server_id, device))
            device_id = _int_value(device.get("device_id"))
            if device_id is None or device_id < 0:
                errors.append(f"server {server_id!r} device[{device_index}] has invalid device_id")
            elif (server_id, device_id) in seen_device_ids:
                errors.append(f"server {server_id!r} has duplicate device_id {device_id}")
            else:
                seen_device_ids.add((server_id, device_id))
    seen_ips: dict[str, str] = {}
    rank_ids: list[int] = []
    for server_id, device in flat_devices:
        device_ip = str(device.get("device_ip", "")).strip()
        if not device_ip:
            errors.append(f"server {server_id!r} has a device without device_ip")
        else:
            try:
                ipaddress.ip_address(device_ip)
            except ValueError:
                errors.append(f"server {server_id!r} has invalid device_ip {device_ip!r}")
            if device_ip in seen_ips:
                errors.append(f"duplicate device_ip {device_ip!r} on {seen_ips[device_ip]!r} and {server_id!r}")
            else:
                seen_ips[device_ip] = server_id
        rank_id = _int_value(device.get("rank_id"))
        if rank_id is None or rank_id < 0:
            errors.append(f"server {server_id!r} has an invalid rank_id")
        else:
            rank_ids.append(rank_id)
    if len(rank_ids) == len(flat_devices) and len(set(rank_ids)) == len(rank_ids):
        expected = set(range(len(flat_devices)))
        missing, extra = sorted(expected - set(rank_ids)), sorted(set(rank_ids) - expected)
        if missing or extra:
            errors.append(f"rank_id must be contiguous 0..{len(flat_devices) - 1}; missing={missing} extra={extra}")
    elif len(set(rank_ids)) != len(rank_ids):
        errors.append("rank_id values must be unique")
    super_pod_count = 0
    if super_pods is not None:
        if not isinstance(super_pods, list) or not super_pods:
            errors.append("super_pod_list must be a non-empty list")
        else:
            super_pod_count = len(super_pods)
            pod_ids: set[str] = set()
            assigned: dict[str, str] = {}
            devices_by_server = {sid: [device for owner, device in flat_devices if owner == sid] for sid in server_ids}
            for pod_index, pod in enumerate(super_pods):
                if not isinstance(pod, Mapping):
                    errors.append(f"super_pod_list[{pod_index}] must be a mapping"); continue
                pod_id = str(pod.get("super_pod_id", "")).strip()
                if not pod_id:
                    errors.append(f"super_pod_list[{pod_index}] is missing super_pod_id"); continue
                if pod_id in pod_ids:
                    errors.append(f"duplicate super_pod_id: {pod_id!r}"); continue
                pod_ids.add(pod_id)
                refs = pod.get("server_list")
                if not isinstance(refs, list) or not refs:
                    errors.append(f"super pod {pod_id!r} must contain a non-empty server_list"); continue
                phy_ids: list[int] = []
                for ref in refs:
                    ref_id = ref if isinstance(ref, str) else str(ref.get("server_id", "")).strip() if isinstance(ref, Mapping) else ""
                    if not ref_id:
                        errors.append(f"super pod {pod_id!r} has a server reference without server_id"); continue
                    if ref_id not in server_ids:
                        errors.append(f"super pod {pod_id!r} references unknown server {ref_id!r}"); continue
                    if ref_id in assigned:
                        errors.append(f"server {ref_id!r} is assigned to multiple super pods"); continue
                    assigned[ref_id] = pod_id
                    for device in devices_by_server[ref_id]:
                        phy = _int_value(_lookup_norm(device, ("super_device_id", "phy_id", "phyid")))
                        if phy is None or phy < 0:
                            errors.append(f"server {ref_id!r} is missing a valid super_device_id/Phy-ID")
                        elif phy in phy_ids:
                            errors.append(f"super pod {pod_id!r} has duplicate super_device_id/Phy-ID {phy}")
                        else:
                            phy_ids.append(phy)
            for server_id in server_ids:
                if server_id not in assigned:
                    errors.append(f"server {server_id!r} is not assigned to any super pod")
    elif _soc_family(soc) == "A3":
        warnings.append("A3 ranktable has no super_pod_list; AI Server mode skips super-pod Phy-ID checks")
    return {"valid": not errors, "status": "ok" if not errors else "invalid",
            "errors": errors, "warnings": warnings, "soc": soc, "version": version or None,
            "server_count": len(servers), "rank_count": len(flat_devices),
            "super_pod_count": super_pod_count}
