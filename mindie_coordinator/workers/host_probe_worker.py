"""Standalone remote Ascend host prerequisite, Docker, and NPU SoC probe worker.

[Role]
Executes on a remote host via `python3 -` (`python -I` pure stdlib compatible)
to gather Docker daemon status, required Ascend device/driver paths, `npu-smi`
board/SoC topology, local vllm-ascend image candidates, and an unoccupied TCP port.

[Invariants]
- Pure Python stdlib; placeholders (`__SOC_TO_MACHINE_TYPE__`,
  `__LOCAL_IMAGE_DISCOVERY_PY__`, `__LOCAL_IMAGE_DISCOVERY_POLICY__`,
  `__SENTINEL__`) are substituted byte-identically by `host_probe_scripts.py`.
- All `subprocess.run` invocations keep `timeout=15` and `stdin=subprocess.DEVNULL`.

[Timeouts & Locks]
- Subprocesses bounded to `15s` each under outer `host_ops` `60s` SSH budget.

[Side-Effect Policy]
- Read-only queries (`docker`, `npu-smi`, `ss`, filesystem `exists()`); emits a
  single `__SENTINEL__<json>` line to stdout.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import socket
import subprocess
import sys
from typing import Optional

image_request, port_range, prefix, sourced_scripts = sys.argv[1:5]
image = json.loads(image_request)
start_s, end_s = port_range.split(":", 1)
start, end = int(start_s), int(end_s)
SOC_TO_MACHINE_TYPE = __SOC_TO_MACHINE_TYPE__
SOC_MATCH_ORDER = sorted(SOC_TO_MACHINE_TYPE, key=len, reverse=True)


def run(cmd: list[str], *, env: Optional[dict[str, str]] = None) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=15,
        stdin=subprocess.DEVNULL,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


__LOCAL_IMAGE_DISCOVERY_PY__


candidates = list(image.get("candidates") or [])
local_image_inspects: list[dict[str, object]] = []
result: dict[str, object] = {
    "success": True,
    "hostname": socket.gethostname(),
    "python": shutil.which("python3") or shutil.which("python"),
    "docker": {
        "present": shutil.which("docker") is not None,
        "version": None,
        "info_ok": False,
    },
    "required_paths": {},
    "missing_required_paths": [],
    "optional_mounts": [],
    "npu_smi_present": pathlib.Path("/usr/local/bin/npu-smi").exists() or shutil.which("npu-smi") is not None,
    "npu_probe": {
        "present": pathlib.Path("/usr/local/bin/npu-smi").exists() or shutil.which("npu-smi") is not None,
        "commands": [],
        "detected_soc": None,
        "machine_type": None,
        "detection_source": None,
        "persisted_soc": None,
        "persisted_machine_type": None,
        "success": False,
        "sourced_scripts": [line for line in sourced_scripts.splitlines() if line],
        "env": {
            "PATH": os.environ.get("PATH"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
            "ASCEND_HOME_PATH": os.environ.get("ASCEND_HOME_PATH"),
            "SOC_VERSION": os.environ.get("SOC_VERSION"),
            "MINDIE_NPU_SOC": os.environ.get("MINDIE_NPU_SOC"),
        },
    },
    "image": {
        "name": image.get("requested"),
        "requested": image.get("requested"),
        "selector": image.get("selector"),
        "policy": image.get("policy"),
        "resolved_tag": image.get("resolved_tag"),
        "candidates": candidates,
        "mirror_order": list(image.get("mirror_order") or []),
        "machine_type": image.get("machine_type"),
        "present_local": False,
        "present_local_candidates": [],
    },
    "managed_containers": [],
    "free_port": None,
    "free_port_range": [start, end],
    "firewall": {
        "ufw": shutil.which("ufw") is not None,
        "firewalld": shutil.which("firewall-cmd") is not None,
    },
}

required = [
    "/dev/davinci_manager",
    "/dev/hisi_hdc",
    "/dev/devmm_svm",
    "/usr/local/Ascend/driver",
    "/usr/local/Ascend/driver/lib64/common",
    "/usr/local/Ascend/driver/lib64/driver",
    "/usr/local/dcmi",
    "/usr/local/bin/npu-smi",
    "/usr/local/sbin",
    "/usr/share/zoneinfo/Asia/Shanghai",
]
for item in required:
    exists = pathlib.Path(item).exists()
    result["required_paths"][item] = exists
    if not exists:
        result["missing_required_paths"].append(item)

for item in ["/home", "/tmp", "/weight", "/weights", "/models", "/data", "/mnt"]:
    if pathlib.Path(item).exists():
        result["optional_mounts"].append(item)

if result["docker"]["present"]:
    rc, out, _ = run(["docker", "--version"])
    if rc == 0:
        result["docker"]["version"] = out
    rc, _, _ = run(["docker", "info"])
    result["docker"]["info_ok"] = rc == 0
    if image.get("policy") == "__LOCAL_IMAGE_DISCOVERY_POLICY__":
        rc, out, _ = run(["docker", "image", "ls", "--no-trunc", "--quiet"])
        image_ids = sorted(set(out.splitlines())) if rc == 0 else []
        if image_ids:
            rc, out, _ = run(["docker", "image", "inspect", *image_ids])
            if rc == 0:
                try:
                    inspected = json.loads(out)
                    if isinstance(inspected, list):
                        local_image_inspects = [item for item in inspected if isinstance(item, dict)]
                except json.JSONDecodeError:
                    local_image_inspects = []
    present_local_candidates: list[str] = []
    for candidate in candidates:
        rc, _, _ = run(["docker", "image", "inspect", candidate])
        if rc == 0:
            present_local_candidates.append(candidate)
    result["image"]["present_local_candidates"] = present_local_candidates
    result["image"]["present_local"] = bool(present_local_candidates)
    rc, out, _ = run(["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}\t{{.Image}}"])
    if rc == 0:
        matched: list[tuple[str, str, str]] = []
        for line in out.splitlines():
            if not line:
                continue
            parts = line.split("\t", 2)
            if len(parts) < 3:
                continue
            name, status, actual_image = parts
            if name.startswith(prefix):
                matched.append((name, status, actual_image))
        priv_map: dict[str, Optional[bool]] = {}
        if matched:
            priv_rc, priv_out, _ = run(
                [
                    "docker",
                    "inspect",
                    "-f",
                    "{{.Name}}\t{{.HostConfig.Privileged}}",
                    *[item[0] for item in matched],
                ]
            )
            if priv_rc == 0:
                for pline in priv_out.splitlines():
                    pparts = pline.split("\t", 1)
                    if len(pparts) != 2:
                        continue
                    cname = pparts[0].strip().lstrip("/")
                    pval = pparts[1].strip()
                    if pval in ("true", "false"):
                        priv_map[cname] = pval == "true"
        result["managed_containers"] = [
            {
                "name": name,
                "status": status,
                "image": actual_image,
                "privileged": priv_map.get(name),
            }
            for name, status, actual_image in matched
        ]

fresh_probe_text: list[str] = []


def probe_command(label: str, cmd: list[str]) -> str:
    rc, out, err = run(cmd, env=os.environ.copy())
    entry = {
        "label": label,
        "command": cmd,
        "returncode": rc,
        "stdout_tail": "\n".join(out.splitlines()[-20:]),
        "stderr_tail": "\n".join(err.splitlines()[-20:]),
    }
    result["npu_probe"]["commands"].append(entry)
    if rc == 0:
        result["npu_probe"]["success"] = True
        if out:
            fresh_probe_text.append(out)
        if err:
            fresh_probe_text.append(err)
    return out if rc == 0 else ""


def npu_smi_value(text: str, field: str) -> Optional[str]:
    match = re.search(
        rf"^\s*{re.escape(field)}\s*:\s*(.*?)\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def first_npu_chip_ids(text: str) -> tuple[Optional[int], Optional[int]]:
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        if any(part.lower().startswith("ascend") for part in parts[2:]):
            return int(parts[0]), int(parts[1])
    return None, None


def canonical_soc_from_board_text(text: str) -> Optional[str]:
    chip_name = npu_smi_value(text, "Chip Name")
    if not chip_name:
        return None
    chip_type = npu_smi_value(text, "Chip Type")
    npu_name = npu_smi_value(text, "NPU Name")
    chip_name = re.sub(r"\s+", "", chip_name)
    chip_type = re.sub(r"\s+", "", chip_type or "")
    npu_name = re.sub(r"\s+", "", npu_name or "")
    lowered_chip_name = chip_name.lower()
    if "310" in lowered_chip_name and chip_type:
        return (chip_type + chip_name).lower()
    if "910" in lowered_chip_name:
        if chip_type:
            return (chip_type + chip_name).lower()
        if npu_name:
            return f"{chip_name}_{npu_name}".lower()
    if "950" in lowered_chip_name and npu_name:
        return f"{chip_name}_{npu_name}".lower()
    return chip_name.lower()


def machine_type_from_soc(soc: Optional[str]) -> Optional[str]:
    if not soc:
        return None
    normalized = soc.strip().lower()
    if normalized.startswith("ascend950"):
        return "A5"
    return SOC_TO_MACHINE_TYPE.get(normalized)


def detect_from_text(text: str) -> tuple[Optional[str], Optional[str]]:
    normalized = text.lower()
    ascend_950_match = re.search(r"\bascend950[a-z0-9_-]*", normalized)
    if ascend_950_match is not None:
        return ascend_950_match.group(0), "A5"
    for token in SOC_MATCH_ORDER:
        if token in normalized:
            return token, SOC_TO_MACHINE_TYPE[token]
    if re.search(r"\bascend910\b", normalized) is not None:
        return "ascend910", "A3"
    return None, None


probe_command("info", ["npu-smi", "info"])
info_list = probe_command("info-list", ["npu-smi", "info", "-l"])
info_map = probe_command("info-map", ["npu-smi", "info", "-m"])

npu_id_text = npu_smi_value(info_list, "NPU ID")
npu_id = int(npu_id_text) if npu_id_text and npu_id_text.isdigit() else 0
mapped_npu_id, mapped_chip_id = first_npu_chip_ids(info_map)
if mapped_npu_id is not None:
    npu_id = mapped_npu_id
chip_id = mapped_chip_id if mapped_chip_id is not None else 0
result["npu_probe"]["query_npu_id"] = npu_id
result["npu_probe"]["query_chip_id"] = chip_id

board_device = probe_command(
    "board-device",
    ["npu-smi", "info", "-t", "board", "-i", str(npu_id)],
)
board_chip = probe_command(
    "board-chip",
    ["npu-smi", "info", "-t", "board", "-i", str(npu_id), "-c", str(chip_id)],
)
probe_command(
    "common-device",
    ["npu-smi", "info", "-t", "common", "-i", str(npu_id)],
)

board_device_soc = canonical_soc_from_board_text(board_device)
board_chip_soc = canonical_soc_from_board_text(board_chip)
precise_candidates = [
    ("board-device", board_device_soc),
    ("board-chip", board_chip_soc),
]
for source, soc in precise_candidates:
    machine_type = machine_type_from_soc(soc)
    if machine_type is not None:
        result["npu_probe"]["detected_soc"] = soc
        result["npu_probe"]["machine_type"] = machine_type
        result["npu_probe"]["detection_source"] = source
        break

if result["npu_probe"]["machine_type"] is None:
    soc, machine_type = detect_from_text("\n".join(fresh_probe_text))
    if machine_type is not None:
        result["npu_probe"]["detected_soc"] = soc
        result["npu_probe"]["machine_type"] = machine_type
        result["npu_probe"]["detection_source"] = "npu-smi-text"

soc_from_env = os.environ.get("SOC_VERSION") or os.environ.get("MINDIE_NPU_SOC")
persisted_soc, persisted_machine_type = detect_from_text(soc_from_env or "")
result["npu_probe"]["persisted_soc"] = persisted_soc
result["npu_probe"]["persisted_machine_type"] = persisted_machine_type
if result["npu_probe"]["machine_type"] is None and persisted_machine_type is not None:
    result["npu_probe"]["detected_soc"] = persisted_soc
    result["npu_probe"]["machine_type"] = persisted_machine_type
    result["npu_probe"]["detection_source"] = "persisted-env-fallback"

result["detected_soc"] = result["npu_probe"]["detected_soc"]
result["machine_type"] = result["npu_probe"]["machine_type"]
result["prerequisites_ok"] = bool(
    result["docker"]["present"]
    and result["docker"]["info_ok"]
    and not result["missing_required_paths"]
)
result["warnings"] = []
if result["machine_type"] is None:
    result["warnings"].append("machine type could not be inferred from npu-smi output; pass an explicit override if needed")
if (
    result["npu_probe"]["detection_source"] != "persisted-env-fallback"
    and persisted_soc is not None
    and result["detected_soc"] is not None
    and persisted_soc != result["detected_soc"]
):
    result["warnings"].append(
        "persisted SOC metadata differs from the fresh npu-smi board query; using the fresh hardware result"
    )

if image.get("policy") == "__LOCAL_IMAGE_DISCOVERY_POLICY__":
    discovery_machine_type = image.get("machine_type") or result["machine_type"]
    local_discovery = select_latest_local_vllm_ascend_image(
        local_image_inspects,
        discovery_machine_type,
    )
    selected_reference = local_discovery.get("selected_reference")
    result["image"]["local_discovery"] = local_discovery
    result["image"]["candidates"] = [selected_reference] if selected_reference else []
    result["image"]["present_local_candidates"] = [selected_reference] if selected_reference else []
    result["image"]["present_local"] = bool(selected_reference)
    if not selected_reference:
        result["warnings"].append(
            "no machine-compatible local vllm-ascend image was discovered"
        )

rc, out, _ = run(["ss", "-ltnH"])
used: set[int] = set()
if rc == 0:
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        local = parts[3]
        if ":" not in local:
            continue
        try:
            used.add(int(local.rsplit(":", 1)[1]))
        except ValueError:
            continue
for port in range(start, end + 1):
    if port not in used:
        result["free_port"] = port
        break

print("__SENTINEL__" + json.dumps(result, ensure_ascii=False))
