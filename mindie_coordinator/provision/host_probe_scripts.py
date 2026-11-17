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

from remote_dev.auth import INSTALL_PUBLIC_KEY_SH

from mindie_coordinator.provision.image_resolver import (
    IMAGE_POLICY_LOCAL_LATEST,
    LOCAL_IMAGE_DISCOVERY_PY,
    PROGRESS_SENTINEL,
    SENTINEL,
    SOC_TO_MACHINE_TYPE,
)
from mindie_coordinator.workers import load_worker_source


def render_host_probe_script() -> str:
    probe_worker = load_worker_source("host_probe_worker.py")
    template = (
        """#!/usr/bin/env bash
set -euo pipefail
image_request_json="$1"
port_range="$2"
prefix="$3"

export PATH="/usr/local/bin:/usr/local/sbin:${PATH:-}"
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}"
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
  if [ -d /usr/local/Ascend/cann ]; then
    export ASCEND_HOME_PATH=/usr/local/Ascend/cann
  elif [ -d /usr/local/Ascend/ascend-toolkit/latest ]; then
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
safe_source /usr/local/Ascend/cann/set_env.sh
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
  echo "__SENTINEL__{\\"success\\": false, \\"error\\": \\"python not found on remote host\\"}"
  exit 3
fi
"$py" - "$image_request_json" "$port_range" "$prefix" "$sourced_scripts" <<'PY'
"""
        + probe_worker
        + "PY\n"
    )
    return (
        template.replace("__SENTINEL__", SENTINEL)
        .replace("__SOC_TO_MACHINE_TYPE__", json.dumps(SOC_TO_MACHINE_TYPE, ensure_ascii=False))
        .replace("__LOCAL_IMAGE_DISCOVERY_PY__", LOCAL_IMAGE_DISCOVERY_PY)
        .replace("__LOCAL_IMAGE_DISCOVERY_POLICY__", IMAGE_POLICY_LOCAL_LATEST)
    )


def render_smoke_script(*, device_test: bool = True, image_family: str | None = None) -> str:
    if image_family not in (None, "mindie", "mindiesd"):
        raise ValueError(f"unsupported image family {image_family!r}")
    from remote_dev.core.python_discovery import selection_command
    required = ["torch", "torch-npu"] + (["mindiesd"] if image_family == "mindiesd" else [])
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
  __PYTHON_SELECTION__ "$requested_python"
}

emit_progress "python-discovery" "running" "discovering a usable python inside the container"
PYTHON_BIN="$(detect_python)"
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
import importlib.metadata
import json
import os
import sys

python_path = sys.argv[1]
sourced_text = sys.argv[2]

def mindiesd_metadata_version():
    if __IMAGE_FAMILY__ not in ('mindie', 'mindiesd'):
        return None
    try:
        return importlib.metadata.version('mindiesd')
    except importlib.metadata.PackageNotFoundError:
        if __IMAGE_FAMILY__ == 'mindiesd':
            raise
        return None

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
__SMOKE_BODY__
print("__SENTINEL__" + json.dumps(result, ensure_ascii=False))
raise SystemExit(0 if result["success"] else 3)
PY
"""
    if not device_test:
        smoke_body = """try:
    result.update(success=True, device_test=False,
                  torch_version=importlib.metadata.version('torch'),
                  torch_npu_version=importlib.metadata.version('torch-npu'))
    version = mindiesd_metadata_version()
    if version is not None:
        result['mindiesd_version'] = version
except importlib.metadata.PackageNotFoundError as exc:
    result.update(success=False, device_test=False, error=str(exc))"""
    else:
        smoke_body = """import traceback

PROGRESS = "__PROGRESS__"

def progress(phase: str, message: str) -> None:
    sys.stderr.write(PROGRESS + json.dumps({"phase": phase, "status": "running", "message": message}, ensure_ascii=False) + "\\n")
    sys.stderr.flush()

try:
    progress("smoke", "importing torch and torch_npu")
    import torch
    import torch_npu  # noqa: F401
    if __IMAGE_FAMILY__ == 'mindiesd' or mindiesd_metadata_version() is not None:
        import mindiesd  # noqa: F401
        result["mindiesd_version"] = getattr(mindiesd, "__version__", None) or importlib.metadata.version("mindiesd")

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
    )"""
    return (
        template.replace("__SMOKE_BODY__", smoke_body)
        .replace("__PYTHON_SELECTION__", selection_command(required, compact=False))
        .replace("__IMAGE_FAMILY__", repr(image_family))
        .replace("__SENTINEL__", SENTINEL)
        .replace("__PROGRESS__", PROGRESS_SENTINEL)
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
__INSTALL_PUBLIC_KEY_SH__
remote_dev_install_public_key "$peer_key" /root/.ssh
[ -f /root/.ssh/known_hosts ] || touch /root/.ssh/known_hosts
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
    return (template.replace("__SENTINEL__", SENTINEL)
            .replace("__INSTALL_PUBLIC_KEY_SH__", INSTALL_PUBLIC_KEY_SH))


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
