"""Bounded host/container probe and mesh helper scripts.

[Role]
Renders the bash/Python probe, smoke, mesh key/peer, and container-removal
scripts executed by host_ops; rendering itself has no local effects.

[Invariants]
- Rendered script text is byte-stable for a given input; placeholders such as
  `__SENTINEL__`, `__PROGRESS__`, and `__LOCAL_IMAGE_DISCOVERY_PY__` are
  substituted literally.
- The embedded probe `run()` keeps `timeout=15` and `stdin=subprocess.DEVNULL`.
- Probe SOC detection order (`SOC_MATCH_ORDER`) derives from the shared
  `SOC_TO_MACHINE_TYPE` mapping.

[Timeouts & Locks]
- Embedded probe subprocesses use a 15s timeout; outer SSH timeouts are owned by
  `host_ops` (60s probe, 240s smoke).

[Side-Effect Policy]
- Rendering is side-effect free; emitted scripts only read probe data or manage
  the container/mesh resources explicitly passed by the caller.
"""

from __future__ import annotations

import json

from mindie_coordinator.provision.image_resolver import (
    IMAGE_POLICY_LOCAL_LATEST,
    LOCAL_IMAGE_DISCOVERY_PY,
    PROGRESS_SENTINEL,
    SENTINEL,
    SOC_TO_MACHINE_TYPE,
)

def render_host_probe_script() -> str:
    template = r"""#!/usr/bin/env bash
set -euo pipefail
image_request_json="$1"
port_range="$2"
prefix="$3"

export PATH="/usr/local/bin:/usr/local/sbin:${PATH:-}"
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}"
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
  if [ -d /usr/local/Ascend/ascend-toolkit/latest ]; then
    export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
  elif [ -d /usr/local/Ascend/ascend-toolkit ]; then
    export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
  fi
fi
sourced_scripts=""
safe_source() {
  file="$1"
  shift || true
  if [ -f "$file" ]; then
    set +u
    . "$file" "$@" >/dev/null 2>&1 || true
    set -u
    sourced_scripts="${sourced_scripts}${file}${*:+ $*}
"
  fi
}
safe_source /etc/profile.d/mindie-ascend-env.sh
safe_source /usr/local/Ascend/ascend-toolkit/set_env.sh
safe_source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
safe_source /usr/local/Ascend/nnal/atb/set_env.sh "--cxx_abi=${MINDIE_ATB_CXX_ABI:-1}"

py=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then
    py="$cand"
    break
  fi
done
if [ -z "$py" ]; then
  echo "__SENTINEL__{\"success\": false, \"error\": \"python not found on remote host\"}"
  exit 3
fi
"$py" - "$image_request_json" "$port_range" "$prefix" "$sourced_scripts" <<'PY'
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
    rc, out, _ = run(["docker", "ps", "-a", "--format", "{{.Names}}	{{.Status}}	{{.Image}}"])
    if rc == 0:
        rows = []
        for line in out.splitlines():
            if not line:
                continue
            parts = line.split("	", 2)
            if len(parts) < 3:
                continue
            name, status, actual_image = parts
            if name.startswith(prefix):
                rows.append({"name": name, "status": status, "image": actual_image})
        result["managed_containers"] = rows

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
    [
        "npu-smi",
        "info",
        "-t",
        "board",
        "-i",
        str(npu_id),
        "-c",
        str(chip_id),
    ],
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
PY
"""
    return (
        template.replace("__SENTINEL__", SENTINEL)
        .replace("__SOC_TO_MACHINE_TYPE__", json.dumps(SOC_TO_MACHINE_TYPE, ensure_ascii=False))
        .replace("__LOCAL_IMAGE_DISCOVERY_PY__", LOCAL_IMAGE_DISCOVERY_PY)
        .replace("__LOCAL_IMAGE_DISCOVERY_POLICY__", IMAGE_POLICY_LOCAL_LATEST)
    )


def render_smoke_script(*, device_test: bool = True) -> str:
    template = r"""#!/usr/bin/env bash
set -euo pipefail
requested_python="${1:-}"

emit_progress() {
  python3 - "$1" "$2" "$3" <<'PY' >&2
import json
import sys
print("__PROGRESS__" + json.dumps({
    "phase": sys.argv[1],
    "status": sys.argv[2],
    "message": sys.argv[3],
}, ensure_ascii=False))
PY
}

detect_python() {
  if [ -n "$requested_python" ] && [ -x "$requested_python" ]; then
    printf '%s\n' "$requested_python"
    return 0
  fi
  while IFS= read -r candidate; do
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done < <(ls -1d /usr/local/python*/bin/python3 2>/dev/null | sort -V -r)
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  return 1
}

emit_progress "python-discovery" "running" "discovering a usable python inside the container"
PYTHON_BIN="$(detect_python || true)"
if [ -z "$PYTHON_BIN" ]; then
  echo "__SENTINEL__{\"success\": false, \"error\": \"python not found in container\", \"phase\": \"python-discovery\"}"
  exit 40
fi

emit_progress "env" "running" "preparing Ascend and python environment"
export PATH="/usr/local/bin:/usr/local/sbin:${PATH:-}"
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

sourced_file_log=""
safe_source() {
  file="$1"
  shift || true
  if [ -f "$file" ]; then
    set +u
    . "$file" "$@" >/dev/null 2>&1 || true
    set -u
    sourced_file_log+="$file${*:+ $*}\n"
  fi
}

safe_source /etc/profile.d/mindie-ascend-env.sh
safe_source /usr/local/Ascend/ascend-toolkit/set_env.sh
safe_source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
safe_source /usr/local/Ascend/nnal/atb/set_env.sh "--cxx_abi=${MINDIE_ATB_CXX_ABI:-1}"
safe_source /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/bin/set_env.bash

"$PYTHON_BIN" - <<'PY' "$PYTHON_BIN" "$sourced_file_log"
import json
import os
import sys
import traceback

PROGRESS = "__PROGRESS__"
python_path = sys.argv[1]
sourced_text = sys.argv[2]

def progress(phase: str, message: str) -> None:
    sys.stderr.write(PROGRESS + json.dumps({"phase": phase, "status": "running", "message": message}, ensure_ascii=False) + "\n")
    sys.stderr.flush()

result = {
    "success": False,
    "python_path": python_path,
    "python_version": sys.version.split()[0],
    "driver_ld_library_path_prefix": [
        "/usr/local/Ascend/driver/lib64/common",
        "/usr/local/Ascend/driver/lib64/driver",
        "/usr/local/Ascend/driver/lib64",
    ],
    "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
    "sourced_scripts": [line for line in sourced_text.splitlines() if line],
}
if not __DEVICE_TEST__:
    # Provisioning has no device grant. Verify the interpreter's package
    # metadata only; native imports and business execution have later owners.
    import importlib.metadata
    try:
        result.update(success=True, device_test=False,
                      torch_version=importlib.metadata.version('torch'),
                      torch_npu_version=importlib.metadata.version('torch-npu'))
    except importlib.metadata.PackageNotFoundError as exc:
        result.update(success=False, device_test=False, error=str(exc))
    print("__SENTINEL__" + json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["success"] else 3)
try:
    progress("smoke", "importing torch and torch_npu")
    import torch
    import torch_npu  # noqa: F401

    progress("smoke", "allocating a small tensor on NPU")
    x = torch.zeros(1, 2).npu()
    progress("smoke", "container smoke test passed")
    result.update(
        {
            "success": True,
            "torch_version": getattr(torch, "__version__", None),
            "shape": list(x.shape),
            "device": str(x.device),
            "device_count": torch.npu.device_count(),
            "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        }
    )
    if not str(x.device).startswith("npu"):
        raise RuntimeError(f"unexpected device: {x.device}")
except Exception as exc:  # pragma: no cover - runtime dependent
    tb = traceback.format_exc().splitlines()
    result.update(
        {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": tb[-25:],
        }
    )
print("__SENTINEL__" + json.dumps(result, ensure_ascii=False))
raise SystemExit(0 if result["success"] else 3)
PY
"""
    return template.replace("__DEVICE_TEST__", repr(device_test)).replace("__SENTINEL__", SENTINEL).replace(
        "__PROGRESS__", PROGRESS_SENTINEL
    )


def render_mesh_export_key_script() -> str:
    template = r"""#!/usr/bin/env bash
set -euo pipefail
comment="$1"
install -d -m 700 /root/.ssh
if [ ! -f /root/.ssh/id_ed25519 ]; then
  ssh-keygen -t ed25519 -C "$comment" -f /root/.ssh/id_ed25519 -N "" >/dev/null
fi
pubkey="$(cat /root/.ssh/id_ed25519.pub)"
fingerprint="$(ssh-keygen -lf /root/.ssh/id_ed25519.pub | awk '{print $2}')"
python3 - <<'PY' "$comment" "$pubkey" "$fingerprint"
import json
import sys
print("__SENTINEL__" + json.dumps({
    "success": True,
    "comment": sys.argv[1],
    "public_key": sys.argv[2],
    "fingerprint": sys.argv[3],
}, ensure_ascii=False))
PY
"""
    return template.replace("__SENTINEL__", SENTINEL)


def render_mesh_add_peer_script() -> str:
    template = r"""#!/usr/bin/env bash
set -euo pipefail
peer_key="$1"
peer_host="$2"
peer_port="$3"
install -d -m 700 /root/.ssh
[ -f /root/.ssh/authorized_keys ] || touch /root/.ssh/authorized_keys
[ -f /root/.ssh/known_hosts ] || touch /root/.ssh/known_hosts
if ! grep -qxF "$peer_key" /root/.ssh/authorized_keys 2>/dev/null; then
  printf '%s\n' "$peer_key" >> /root/.ssh/authorized_keys
fi
chmod 600 /root/.ssh/authorized_keys
ssh-keyscan -p "$peer_port" "$peer_host" >> /root/.ssh/known_hosts 2>/dev/null || true
python3 - <<'PY' "$peer_host" "$peer_port" "$peer_key"
import json
import sys
parts = sys.argv[3].split()
print("__SENTINEL__" + json.dumps({
    "success": True,
    "peer_host": sys.argv[1],
    "peer_port": int(sys.argv[2]),
    "peer_key_added": True,
    "peer_key_comment": parts[-1] if parts else None,
}, ensure_ascii=False))
PY
"""
    return template.replace("__SENTINEL__", SENTINEL)


def render_remove_container_host_script() -> str:
    template = r"""#!/usr/bin/env bash
set -euo pipefail
container="$1"
if ! command -v docker >/dev/null 2>&1; then
  echo "__SENTINEL__{\"success\": false, \"error\": \"docker not found on host\"}"
  exit 10
fi

emit_success() {
  python3 - "$@" <<'PY'
import json
import sys
print("__SENTINEL__" + json.dumps({
    "success": True,
    "container": sys.argv[1],
    "removed": sys.argv[2] == "true",
    "previous_state": None if sys.argv[3] == "" else sys.argv[3],
}, ensure_ascii=False))
PY
}

emit_failure() {
  python3 - "$@" <<'PY'
import json
import sys
print("__SENTINEL__" + json.dumps({
    "success": False,
    "error": sys.argv[2],
    "detail": sys.argv[3] if len(sys.argv) > 3 else "",
}, ensure_ascii=False))
PY
}

if inspect_output="$(docker inspect "$container" 2>&1)"; then
  state="$(docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null)" || state=""
  if ! docker rm -f "$container" >/dev/null 2>&1; then
    emit_failure "$container" "docker removal failed" ""
    exit 13
  fi
  emit_success "$container" true "$state"
else
  case "$inspect_output" in
    *"No such object"*|*"No such container"*)
      emit_success "$container" false ""
      ;;
    *)
      emit_failure "$container" "docker inspect failed" "$inspect_output"
      exit 12
      ;;
  esac
fi
"""
    return template.replace("__SENTINEL__", SENTINEL)
