"""First half of the managed container bootstrap host script template.

[Role]
Holds the literal head of the bootstrap script (helper setup through the
container SSHD configuration); `host_bootstrap_script` concatenates it with the
tail module before placeholder substitution.

[Invariants]
- `_BOOTSTRAP_TEMPLATE_HEAD + _BOOTSTRAP_TEMPLATE_TAIL` is byte-identical to the
  original single raw-string template.
- The text is inert; no placeholder substitution happens here.

[Timeouts & Locks]
- None in this module. The rendered script's outer timeout is owned by
  `host_ops` (1800s).

[Side-Effect Policy]
- Importing this module only defines a string constant; no I/O or mutation.
"""

from __future__ import annotations

_BOOTSTRAP_TEMPLATE_HEAD = r"""#!/usr/bin/env bash
set -euo pipefail
container="$1"
port="$2"
image_request_json="$3"
workdir="$4"
pubkey="$5"
namespace="${6:-}"
replace_on_image_change="${7:-false}"
machine_type_input="${8:-}"
soc_input="${9:-}"
use_prepared_image_cache="${10:-false}"
visible_devices="${11:-}"

emit_json() {
  python3 - "$1" <<'PY'
import json
import sys
print("__SENTINEL__" + json.dumps(json.loads(sys.argv[1]), ensure_ascii=False))
PY
}

emit_progress() {
  python3 - "$1" "$2" "$3" "${4:-}" <<'PY' >&2
import json
import sys
payload = {
    "phase": sys.argv[1],
    "status": sys.argv[2],
    "message": sys.argv[3],
}
if len(sys.argv) > 4 and sys.argv[4]:
    try:
        payload["expected_seconds"] = int(sys.argv[4])
    except ValueError:
        pass
print("__PROGRESS__" + json.dumps(payload, ensure_ascii=False))
PY
}

image_field() {
  python3 - "$image_request_json" "$1" <<'PY'
import json
import sys
payload = json.loads(sys.argv[1])
field = sys.argv[2]
value = payload.get(field)
if value is None:
    raise SystemExit(1)
if isinstance(value, str):
    print(value)
else:
    print(json.dumps(value, ensure_ascii=False))
PY
}

resolve_image_candidates() {
  python3 - "$image_request_json" <<'PY'
import json
import sys
payload = json.loads(sys.argv[1])
for item in payload.get("candidates") or []:
    print(item)
PY
}

discover_local_vllm_ascend_images() {
  local requested_machine_type="$1"
  local inspect_file
  local discovery
  local -a image_ids=()
  mapfile -t image_ids < <(docker image ls --no-trunc --quiet | sort -u)
  if [ "${#image_ids[@]}" -eq 0 ]; then
    printf '%s\n' '{"machine_type": null, "matched_count": 0, "eligible_count": 0, "matched_images": [], "selected_reference": null, "selected_image_id": null, "selected_created": null}'
    return 0
  fi
  inspect_file="$(mktemp)"
  if ! docker image inspect "${image_ids[@]}" >"$inspect_file"; then
    rm -f "$inspect_file"
    return 1
  fi
  if ! discovery="$(python3 - "$requested_machine_type" "$inspect_file" <<'PY'
import json
import sys

machine_type = sys.argv[1] or None
with open(sys.argv[2], encoding="utf-8") as handle:
    image_inspects = json.load(handle)

__LOCAL_IMAGE_DISCOVERY_PY__

print(json.dumps(
    select_latest_local_vllm_ascend_image(image_inspects, machine_type),
    ensure_ascii=False,
))
PY
)"; then
    rm -f "$inspect_file"
    return 1
  fi
  rm -f "$inspect_file"
  printf '%s\n' "$discovery"
}

json_string_array() {
  python3 - "$@" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1:], ensure_ascii=False))
PY
}

run_with_progress() {
  phase="$1"
  message="$2"
  expected_seconds="$3"
  shift 3
  log_file="$(mktemp)"
  "$@" >"$log_file" 2>&1 &
  pid=$!
  start_ts=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    sleep 8
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    elapsed=$(( $(date +%s) - start_ts ))
    if [ -s "$log_file" ]; then
      last_line="$(tail -n 1 "$log_file" 2>/dev/null | tr -d '\r' | sed 's/[^[:print:]\t]//g' | cut -c1-180)"
      if [ -n "$last_line" ]; then
        emit_progress "$phase" "running" "$message - $last_line" "$expected_seconds"
      else
        emit_progress "$phase" "running" "$message - still working (elapsed ${elapsed}s)" "$expected_seconds"
      fi
    else
      emit_progress "$phase" "running" "$message - still working (elapsed ${elapsed}s)" "$expected_seconds"
    fi
  done
  wait "$pid"
  status=$?
  if [ "$status" -ne 0 ]; then
    tail -n 120 "$log_file" >&2 || true
  fi
  rm -f "$log_file"
  return "$status"
}

infer_type_from_image() {
  image_ref="$1"
  lowered="$(printf '%s' "$image_ref" | tr '[:upper:]' '[:lower:]')"
  case "$lowered" in
    *-a5*|*-950dt*|*-950pr*)
      printf 'A5\n'
      return 0
      ;;
    *-310p*)
      printf '310P\n'
      return 0
      ;;
    *-a3*)
      printf 'A3\n'
      return 0
      ;;
  esac
  repo_part="$image_ref"
  if [[ "$repo_part" == *"@"* ]]; then
    repo_part="${repo_part%@*}"
  fi
  if [[ "$repo_part" == *":"* && "$repo_part" != *"//"* ]]; then
    maybe_repo="${repo_part%:*}"
  else
    maybe_repo="$repo_part"
  fi
  if [[ "$maybe_repo" == *"ascend/vllm-ascend" ]]; then
    printf 'A2\n'
    return 0
  fi
  return 1
}

prepared_image_tag() {
  image_ref="$1"
  image_id="$(docker image inspect -f '{{.Id}}' "$image_ref" 2>/dev/null || true)"
  if [ -z "$image_id" ]; then
    return 1
  fi
  image_hash="${image_id#sha256:}"
  image_hash="$(printf '%s' "$image_hash" | tr '[:upper:]' '[:lower:]' | cut -c1-16)"
  if [ -z "$image_hash" ]; then
    return 1
  fi
  printf 'mindie-session-prepared:%s-ssh-v2\n' "$image_hash"
}

write_host_state() {
  host_machine_type="$1"
  host_container_type="$2"
  host_soc="$3"
  host_image="$4"
  install -d -m 0755 /etc/mindie /etc/profile.d
  cat > "$host_env_file" <<EOF_HOST_ENV
#!/bin/sh
export PATH="/usr/local/bin:/usr/local/sbin:\${PATH:-}"
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:\${LD_LIBRARY_PATH:-}"
if [ -z "\${ASCEND_HOME_PATH:-}" ]; then
  if [ -d /usr/local/Ascend/ascend-toolkit/latest ]; then
    export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
  elif [ -d /usr/local/Ascend/ascend-toolkit ]; then
    export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
  fi
fi
if [ -n "$host_machine_type" ]; then
  export MINDIE_MACHINE_TYPE="$host_machine_type"
fi
if [ -n "$host_container_type" ]; then
  export MINDIE_CONTAINER_TYPE="$host_container_type"
fi
if [ -n "$host_soc" ]; then
  export MINDIE_NPU_SOC="$host_soc"
  export SOC_VERSION="$host_soc"
fi
if [ -n "$host_image" ]; then
  export MINDIE_CONTAINER_IMAGE="$host_image"
fi
EOF_HOST_ENV
  chmod 0644 "$host_env_file"
  actions+=("configured-host-env")
  python3 - "$host_machine_type" "$host_container_type" "$host_soc" "$host_image" "$container" "$port" "$workdir" "$namespace" "$host_env_file" > "$host_metadata_file" <<'PY'
import json
import socket
import sys
machine_type, container_type, soc, image, container, port, workdir, namespace, env_file = sys.argv[1:]
print(json.dumps({
    "hostname": socket.gethostname(),
    "machine_type": machine_type or None,
    "container_type": container_type or None,
    "soc": soc or None,
    "container": container,
    "container_ssh_port": int(port),
    "selected_image": image or None,
    "workdir": workdir,
    "namespace": namespace or None,
    "env_file": env_file,
}, ensure_ascii=False, indent=2))
PY
  chmod 0644 "$host_metadata_file"
  actions+=("wrote-host-metadata")
}

install_container_ssh_packages() {
  docker exec -i "$container" bash -s -- <<'INNER_PKG'
set -euo pipefail
state_file=/tmp/mindie-package-bootstrap.json
rm -f "$state_file"
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  python3 - "$state_file" <<'PY'
import json
import os
import pathlib
import re
from urllib.parse import urlsplit, urlunsplit

state_file = pathlib.Path(__import__("sys").argv[1])
fixed_host = "mirrors.nju.edu.cn"
os_release = {}
os_release_path = pathlib.Path("/etc/os-release")
if os_release_path.exists():
    for line in os_release_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os_release[key] = value.strip().strip('"')

sources_dir = pathlib.Path("/etc/apt/sources.list.d")
source_files = []
for candidate in [pathlib.Path("/etc/apt/sources.list"), *sorted(sources_dir.glob("*.list")), *sorted(sources_dir.glob("*.sources"))]:
    if candidate.exists() and candidate.is_file():
        source_files.append(candidate)

list_pattern = re.compile(r"^\s*deb(?:-src)?\s+(?:\[[^\]]+\]\s+)?(\S+)\s+(\S+)")
entries = []
uri_set = {}
for source_file in source_files:
    content = source_file.read_text(encoding="utf-8", errors="replace")
    if source_file.suffix == ".list" or source_file.name == "sources.list":
        for line in content.splitlines():
            if line.strip().startswith("#"):
                continue
            match = list_pattern.match(line)
            if not match:
                continue
            uri, suite = match.groups()
            entries.append((uri, suite))
            uri_set[uri] = None
    else:
        uris = []
        suites = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("URIs:"):
                uris.extend(stripped.split(":", 1)[1].split())
            elif stripped.startswith("Suites:"):
                suites.extend(stripped.split(":", 1)[1].split())
        for uri in uris:
            uri_set[uri] = None
            for suite in suites:
                entries.append((uri, suite))

changed_files = []
changed = False
replacements = {}
for uri in uri_set:
    original = urlsplit(uri)
    if not original.scheme or not original.netloc:
        continue
    if original.netloc == fixed_host:
        continue
    replacements[uri] = urlunsplit((original.scheme, fixed_host, original.path, original.query, original.fragment))
for source_file in source_files:
    original_text = source_file.read_text(encoding="utf-8", errors="replace")
    updated_text = original_text
    for old_uri, new_uri in replacements.items():
        updated_text = updated_text.replace(old_uri, new_uri)
    if updated_text != original_text:
        source_file.write_text(updated_text, encoding="utf-8")
        changed = True
        changed_files.append(str(source_file))

state = {
    "package_manager": "apt",
    "selected_mirror": fixed_host,
    "source_strategy": "fixed-nju",
    "candidate_timings_ms": {},
    "changed_files": changed_files,
    "changed": changed,
    "install_skipped": False,
    "entries_detected": len(entries),
    "os_id": os_release.get("ID"),
}
state_file.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
PY
  echo "apt-get update"
  apt-get -o Acquire::Retries=1 -o Acquire::http::Timeout=10 -o Acquire::https::Timeout=10 -o Acquire::Check-Date=false -o Acquire::Check-Valid-Until=false update
  echo "apt-get install openssh-server openssh-client"
  apt-get install -y --no-install-recommends openssh-server openssh-client
elif command -v dnf >/dev/null 2>&1; then
  printf '%s' '{"package_manager":"dnf","selected_mirror":null,"candidate_timings_ms":{},"changed_files":[],"changed":false,"install_skipped":false}' > "$state_file"
  echo "dnf install openssh-server openssh-clients"
  dnf install -y openssh-server openssh-clients
elif command -v yum >/dev/null 2>&1; then
  printf '%s' '{"package_manager":"yum","selected_mirror":null,"candidate_timings_ms":{},"changed_files":[],"changed":false,"install_skipped":false}' > "$state_file"
  echo "yum install openssh-server openssh-clients"
  yum install -y openssh-server openssh-clients
  elif command -v microdnf >/dev/null 2>&1; then
    printf '%s' '{"package_manager":"microdnf","selected_mirror":null,"candidate_timings_ms":{},"changed_files":[],"changed":false,"install_skipped":false}' > "$state_file"
    echo "microdnf install openssh-server openssh-clients"
    microdnf install -y openssh-server openssh-clients
  else
    echo "no supported package manager found for ssh installation" >&2
    exit 97
  fi
  _mindie_runtime_python=""
  for _d in $(ls -1d /usr/local/python*/bin 2>/dev/null | sort -V -r); do
    if [ -x "$_d/python3" ]; then
      _mindie_runtime_python="$_d/python3"
      break
    fi
  done
  if [ -z "$_mindie_runtime_python" ] && command -v python3 >/dev/null 2>&1; then
    _mindie_runtime_python="$(command -v python3)"
  fi
  if [ -n "$_mindie_runtime_python" ]; then
    install -d -m 0755 /etc/pip
    cat > /etc/pip.conf <<'EOF_PIP'
[global]
index-url = https://repo.huaweicloud.com/repository/pypi/simple
trusted-host = repo.huaweicloud.com
EOF_PIP
    chmod 0644 /etc/pip.conf
    echo "[mindie-bootstrap] pip configured: /etc/pip.conf (index=huaweicloud)"
  else
    echo "[mindie-bootstrap] runtime python not found, skipping pip setup"
  fi
INNER_PKG
}

configure_container_state() {
  docker exec -i "$container" bash -s -- "$port" "$pubkey" "$machine_type" "$container_type" "$soc" "$selected_image" "$workdir" "$namespace" "$visible_devices" <<'INNER_CFG'
set -euo pipefail
port="$1"
pubkey="$2"
machine_type="$3"
container_type="$4"
soc="$5"
image="$6"
workdir="$7"
namespace="$8"
visible_devices="$9"
export PATH="/usr/local/bin:/usr/local/sbin:${PATH:-}"
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}"
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
  if [ -d /usr/local/Ascend/ascend-toolkit/latest ]; then
    export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
  elif [ -d /usr/local/Ascend/ascend-toolkit ]; then
    export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
  fi
fi
__INSTALL_PUBLIC_KEY_SH__
remote_dev_install_public_key "$pubkey" /root/.ssh
install -d -m 0755 /run/sshd /etc/profile.d /etc/mindie
_mindie_python_dir=""
for _d in $(ls -1d /usr/local/python*/bin 2>/dev/null | sort -V -r); do
  if [ -x "$_d/python3" ]; then
    _mindie_python_dir="$_d"
    break
  fi
done
if [ -n "$_mindie_python_dir" ]; then
  _path_prefix="$_mindie_python_dir:/usr/local/bin:/usr/local/sbin"
else
  _path_prefix="/usr/local/bin:/usr/local/sbin"
fi
_mindie_atb_cxx_abi="1"
if [ -n "$_mindie_python_dir" ] && [ -x "$_mindie_python_dir/python3" ]; then
  _torch_cxx_abi="$("$_mindie_python_dir/python3" - <<'PY' 2>/dev/null || true
import torch
print("1" if torch.compiled_with_cxx11_abi() else "0")
PY
)"
  case "$_torch_cxx_abi" in
    0|1)
      _mindie_atb_cxx_abi="$_torch_cxx_abi"
      ;;
  esac
fi
patch_atb_set_env_source() {
  file="$1"
  [ -f "$file" ] || return 0
  if [ ! -f "${file}.mindie-atb-abi.bak" ]; then
    cp -a "$file" "${file}.mindie-atb-abi.bak" 2>/dev/null || true
  fi
  tmp_file="$(mktemp)"
  if awk -v abi="$_mindie_atb_cxx_abi" '
    /^[[:space:]]*(source|\.)[[:space:]]+\/usr\/local\/Ascend\/nnal\/atb\/set_env\.sh[[:space:]]*$/ {
      sub(/[[:space:]]*$/, "")
      print $0 " --cxx_abi=" abi
      next
    }
    { print }
  ' "$file" > "$tmp_file"; then
    cat "$tmp_file" > "$file"
  fi
  rm -f "$tmp_file"
}
patch_atb_set_env_source /etc/profile
patch_atb_set_env_source /root/.bashrc
cat > /etc/profile.d/mindie-ascend-env.sh <<EOF_CONTAINER_ENV
#!/bin/sh
export MINDIE_ATB_CXX_ABI="\${MINDIE_ATB_CXX_ABI:-$_mindie_atb_cxx_abi}"
if [ -z "\${ASCEND_HOME_PATH:-}" ]; then
  if [ -d /usr/local/Ascend/ascend-toolkit/latest ]; then
    export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
  elif [ -d /usr/local/Ascend/ascend-toolkit ]; then
    export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
  fi
fi
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  . /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
fi
if [ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]; then
  . /usr/local/Ascend/ascend-toolkit/latest/set_env.sh >/dev/null 2>&1 || true
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  . /usr/local/Ascend/nnal/atb/set_env.sh "--cxx_abi=\${MINDIE_ATB_CXX_ABI:-1}" >/dev/null 2>&1 || true
fi
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:\${LD_LIBRARY_PATH:-}"
export PATH="$_path_prefix:\${PATH:-}"
if [ -n "$machine_type" ]; then
  export MINDIE_MACHINE_TYPE="$machine_type"
fi
if [ -n "$container_type" ]; then
  export MINDIE_CONTAINER_TYPE="$container_type"
fi
if [ -n "$soc" ]; then
  export MINDIE_NPU_SOC="$soc"
  export SOC_VERSION="$soc"
fi
if [ -n "$image" ]; then
  export MINDIE_CONTAINER_IMAGE="$image"
fi
if [ -n "$visible_devices" ]; then
  export ASCEND_RT_VISIBLE_DEVICES="$visible_devices"
fi
EOF_CONTAINER_ENV
chmod 0644 /etc/profile.d/mindie-ascend-env.sh
python3 - "$machine_type" "$container_type" "$soc" "$image" "$workdir" "$namespace" "$_mindie_atb_cxx_abi" "$visible_devices" > /etc/mindie/container-info.json <<'PY'
import json
import sys
machine_type, container_type, soc, image, workdir, namespace, atb_cxx_abi, visible_devices = sys.argv[1:]
print(json.dumps({
    "machine_type": machine_type or None,
    "container_type": container_type or None,
    "soc": soc or None,
    "image": image or None,
    "workdir": workdir,
    "namespace": namespace or None,
    "atb_cxx_abi": atb_cxx_abi or None,
    "visible_devices": [int(item) for item in visible_devices.split(",") if item] if visible_devices else [],
    "env_file": "/etc/profile.d/mindie-ascend-env.sh",
}, ensure_ascii=False, indent=2))
PY
chmod 0644 /etc/mindie/container-info.json
_mindie_runtime_python=""
if [ -n "$_mindie_python_dir" ] && [ -x "$_mindie_python_dir/python3" ]; then
  _mindie_runtime_python="$_mindie_python_dir/python3"
elif command -v python3 >/dev/null 2>&1; then
  _mindie_runtime_python="$(command -v python3)"
fi
if [ -n "$_mindie_runtime_python" ]; then
  install -d -m 0755 /etc/pip
  cat > /etc/pip.conf <<'EOF_PIP'
[global]
index-url = https://repo.huaweicloud.com/repository/pypi/simple
trusted-host = repo.huaweicloud.com
EOF_PIP
  chmod 0644 /etc/pip.conf
  echo "[mindie-bootstrap] pip configured: /etc/pip.conf (index=huaweicloud)"
else
  echo "[mindie-bootstrap] runtime python not found, skipping pip setup"
fi
ssh-keygen -A >/dev/null 2>&1 || true
cat > /etc/ssh/sshd_mindie_config <<EOF_SSH
Port ${port}
ListenAddress 0.0.0.0
Protocol 2
HostKey /etc/ssh/ssh_host_rsa_key
HostKey /etc/ssh/ssh_host_ecdsa_key
HostKey /etc/ssh/ssh_host_ed25519_key
PermitRootLogin prohibit-password
PubkeyAuthentication yes
PasswordAuthentication no
KbdInteractiveAuthentication no
ChallengeResponseAuthentication no
UsePAM no
PrintMotd no
AuthorizedKeysFile .ssh/authorized_keys
PidFile /run/sshd_mindie.pid
EOF_SSH
if [ -n "$visible_devices" ]; then
  printf 'SetEnv ASCEND_RT_VISIBLE_DEVICES=%s\n' "$visible_devices" >> /etc/ssh/sshd_mindie_config
fi
/usr/sbin/sshd -t -f /etc/ssh/sshd_mindie_config
if [ -f /run/sshd_mindie.pid ]; then
  kill "$(cat /run/sshd_mindie.pid)" 2>/dev/null || true
  rm -f /run/sshd_mindie.pid
fi
pkill -f '/etc/ssh/sshd_mindie_config' 2>/dev/null || true
/usr/sbin/sshd -f /etc/ssh/sshd_mindie_config
if command -v ss >/dev/null 2>&1; then
  ss -ltnH | awk '{print $4}' | grep -Eq "[:.]${port}$"
fi
INNER_CFG
}

"""
