"""Second half of the managed container bootstrap host script template.

[Role]
Holds the literal tail of the bootstrap script (preflight through final JSON
emission); `host_bootstrap_script` concatenates it after the head module.

[Invariants]
- Concatenated with `_host_bootstrap_template_a._BOOTSTRAP_TEMPLATE_HEAD`, the
  result is byte-identical to the original single raw-string template.
- The text is inert; no placeholder substitution happens here.

[Timeouts & Locks]
- None in this module. The rendered script's outer timeout is owned by
  `host_ops` (1800s).

[Side-Effect Policy]
- Importing this module only defines a string constant; no I/O or mutation.
"""

from __future__ import annotations

_BOOTSTRAP_TEMPLATE_TAIL = r"""emit_progress "preflight" "running" "validating host docker and Ascend prerequisites" 15
if ! command -v docker >/dev/null 2>&1; then
  emit_json '{"success": false, "error": "docker not found on host", "phase": "probe"}'
  exit 10
fi
if ! docker info >/dev/null 2>&1; then
  emit_json '{"success": false, "error": "docker info failed on host", "phase": "probe"}'
  exit 11
fi

missing=()
for p in \
  /dev/davinci_manager \
  /dev/hisi_hdc \
  /dev/devmm_svm \
  /usr/local/Ascend/driver \
  /usr/local/Ascend/driver/lib64/common \
  /usr/local/Ascend/driver/lib64/driver \
  /usr/local/dcmi \
  /usr/local/bin/npu-smi \
  /usr/local/sbin \
  /usr/share/zoneinfo/Asia/Shanghai
  do
  [ -e "$p" ] || missing+=("$p")
done
if [ "${#missing[@]}" -gt 0 ]; then
  missing_json=$(python3 - <<'PY' "${missing[@]}"
import json
import sys
print(json.dumps(list(sys.argv[1:]), ensure_ascii=False))
PY
)
  emit_json "{\"success\": false, \"error\": \"required host paths missing\", \"phase\": \"probe\", \"missing\": ${missing_json}}"
  exit 12
fi

actions=()
created=false
started=false
pulled=false
installed_ssh=false
firewall="none"
selected_image=""
run_image=""
prepared_image=""
used_prepared_image_cache=false
created_prepared_image_cache=false
image_resolution="unknown"
package_bootstrap='{"package_manager": null, "selected_mirror": null, "candidate_timings_ms": {}, "changed_files": [], "changed": false, "install_skipped": false}'
host_env_file="/etc/profile.d/mindie-ascend-env.sh"
host_metadata_file="/etc/mindie/host-info.json"
container_env_file="/etc/profile.d/mindie-ascend-env.sh"
container_metadata_file="/etc/mindie/container-info.json"
requested_image="$(image_field requested || true)"
requested_selector="$(image_field selector || true)"
image_policy="$(image_field policy || true)"
docker_pull_policy="${MINDIE_DOCKER_PULL_POLICY:-missing}"
resolved_tag="$(image_field resolved_tag || true)"
machine_type="${machine_type_input:-$(image_field machine_type || true)}"
soc="${soc_input:-}"
container_type=""
previous_image=""
selected_image_id=""
local_image_discovery='null'
mapfile -t image_candidates < <(resolve_image_candidates)
if [ "$image_policy" = "__LOCAL_IMAGE_DISCOVERY_POLICY__" ]; then
  emit_progress "image-discovery" "running" "scanning local Docker images for the newest machine-compatible vllm-ascend image" 30
  if ! local_image_discovery="$(discover_local_vllm_ascend_images "$machine_type")"; then
    emit_json '{"success": false, "error": "failed to inspect local Docker images", "phase": "image-discovery"}'
    exit 19
  fi
  selected_local_image="$(python3 - "$local_image_discovery" <<'PY'
import json
import sys
print(json.loads(sys.argv[1]).get("selected_reference") or "")
PY
)"
  if [ -n "$selected_local_image" ]; then
    image_candidates=("$selected_local_image")
  else
    image_candidates=()
  fi
fi
runtime_image_candidates_json="$(json_string_array "${image_candidates[@]}")"
if [ "${#image_candidates[@]}" -eq 0 ]; then
  empty_candidates_payload=$(python3 - "$image_request_json" "$local_image_discovery" <<'PY'
import json
import sys
request = json.loads(sys.argv[1])
discovery = json.loads(sys.argv[2]) if sys.argv[2] != "null" else None
print(json.dumps({
    "success": False,
    "error": "image selector resolved to zero candidates",
    "phase": "image-discovery" if discovery is not None else "image",
    "requested_image": request.get("requested"),
    "requested_selector": request.get("selector"),
    "image_policy": request.get("policy"),
    "local_image_discovery": discovery,
}, ensure_ascii=False))
PY
)
  emit_json "$empty_candidates_payload"
  exit 19
fi

container_exists=false
if docker inspect "$container" >/dev/null 2>&1; then
  container_exists=true
  current_image=$(docker inspect -f '{{.Config.Image}}' "$container")
  current_base_image=$(docker inspect -f '{{ index .Config.Labels "com.mindie.base_image" }}' "$container" 2>/dev/null || true)
  if [ "$current_base_image" = "<no value>" ]; then
    current_base_image=""
  fi
  previous_image="$current_image"
  image_matches=false
  if [ "$image_policy" = "__LOCAL_IMAGE_DISCOVERY_POLICY__" ]; then
    candidate_image_id="$(docker image inspect -f '{{.Id}}' "${image_candidates[0]}" 2>/dev/null || true)"
    current_source_image_id="$(docker inspect -f '{{.Image}}' "$container" 2>/dev/null || true)"
    recorded_base_image_id="$(docker inspect -f '{{ index .Config.Labels "com.mindie.base_image_id" }}' "$container" 2>/dev/null || true)"
    if [ "$recorded_base_image_id" != "<no value>" ] && [ -n "$recorded_base_image_id" ]; then
      current_source_image_id="$recorded_base_image_id"
    fi
    if [ -n "$candidate_image_id" ] && [ "$candidate_image_id" = "$current_source_image_id" ]; then
      image_matches=true
    fi
  else
    for candidate in "${image_candidates[@]}"; do
      if [ "$candidate" = "$current_image" ] || { [ "$use_prepared_image_cache" = "true" ] && [ "$candidate" = "$current_base_image" ]; }; then
        image_matches=true
        break
      fi
    done
  fi
  if [ "$image_matches" != "true" ]; then
    if [ "$replace_on_image_change" = "true" ]; then
      emit_progress "image" "running" "existing container image $current_image does not match requested selector; recreating container" 60
      docker rm -f "$container" >/dev/null
      actions+=("removed-container-for-image-change")
      container_exists=false
    else
      mismatch_payload=$(python3 - <<'PY' "$current_image" "$image_request_json" "$runtime_image_candidates_json" "$local_image_discovery"
import json
import sys
current_image, image_request_json, runtime_candidates_json, local_discovery_json = sys.argv[1:]
image_request = json.loads(image_request_json)
print(json.dumps({
    "success": False,
    "error": "existing container image does not match the requested selector",
    "phase": "image",
    "existing_container_image": current_image,
    "requested_image": image_request.get("requested"),
    "requested_selector": image_request.get("selector"),
    "image_policy": image_request.get("policy"),
    "resolved_tag": image_request.get("resolved_tag"),
    "image_candidates": json.loads(runtime_candidates_json),
    "local_image_discovery": json.loads(local_discovery_json) if local_discovery_json != "null" else None,
    "suggestion": "rerun with explicit container replacement consent or remove the managed container first",
}, ensure_ascii=False))
PY
)
      emit_json "$mismatch_payload"
      exit 21
    fi
  fi
fi

if [ "$container_exists" = "true" ]; then
  emit_progress "container" "running" "reusing an existing managed container" 15
  state=$(docker inspect -f '{{.State.Status}}' "$container")
  selected_image=$(docker inspect -f '{{.Config.Image}}' "$container")
  run_image="$selected_image"
  existing_base_image=$(docker inspect -f '{{ index .Config.Labels "com.mindie.base_image" }}' "$container" 2>/dev/null || true)
  if [ "$existing_base_image" = "<no value>" ]; then
    existing_base_image=""
  fi
  if [ "$use_prepared_image_cache" = "true" ] && [ -n "$existing_base_image" ]; then
    run_image="$selected_image"
    selected_image="$existing_base_image"
    prepared_image="$(docker inspect -f '{{ index .Config.Labels "com.mindie.prepared_image" }}' "$container" 2>/dev/null || true)"
    if [ "$prepared_image" = "<no value>" ]; then
      prepared_image=""
    fi
    used_prepared_image_cache=true
  fi
  selected_image_id="$(docker inspect -f '{{.Image}}' "$container" 2>/dev/null || true)"
  recorded_base_image_id="$(docker inspect -f '{{ index .Config.Labels "com.mindie.base_image_id" }}' "$container" 2>/dev/null || true)"
  if [ "$recorded_base_image_id" != "<no value>" ] && [ -n "$recorded_base_image_id" ]; then
    selected_image_id="$recorded_base_image_id"
  fi
  image_resolution="existing-container"
  if [ "$state" != "running" ]; then
    docker start "$container" >/dev/null
    started=true
    actions+=("started-existing-container")
  else
    actions+=("reused-existing-container")
  fi
else
  emit_progress "image" "running" "resolved image selector ${requested_image:-unknown} (${image_policy:-unknown})" 30
  for candidate in "${image_candidates[@]}"; do
    emit_progress "image" "running" "trying image candidate $candidate" 600
    if [ "$image_policy" = "__LOCAL_IMAGE_DISCOVERY_POLICY__" ]; then
      if docker image inspect "$candidate" >/dev/null 2>&1; then
        selected_image="$candidate"
        image_resolution="local-discovery"
        actions+=("selected-latest-local-vllm-ascend-image")
        break
      fi
      emit_progress "image-discovery" "running" "locally discovered image $candidate disappeared before deployment" 30
      continue
    fi
    if [ "$docker_pull_policy" != "always" ] && [ "$image_policy" != "main-branch" ] && docker image inspect "$candidate" >/dev/null 2>&1; then
      selected_image="$candidate"
      image_resolution="local-cache"
      actions+=("reused-local-image-cache")
      break
    fi
    if run_with_progress "image-pull" "docker pull $candidate" 600 docker pull "$candidate"; then
      selected_image="$candidate"
      image_resolution="pulled"
      pulled=true
      actions+=("pulled-image")
      break
    fi
    emit_progress "image" "running" "pull failed for $candidate; checking local cache" 30
  done
  if [ -z "$selected_image" ]; then
    emit_json '{"success": false, "error": "no usable image candidate was found", "phase": "image"}'
    exit 20
  fi
  selected_image_id="$(docker image inspect -f '{{.Id}}' "$selected_image" 2>/dev/null || true)"
  if [ -z "$machine_type" ]; then
    machine_type="$(infer_type_from_image "$selected_image" || true)"
  fi
  container_type="$machine_type"
  run_image="$selected_image"
  if [ "$use_prepared_image_cache" = "true" ]; then
    prepared_image="$(prepared_image_tag "$selected_image" || true)"
    if [ -n "$prepared_image" ] && docker image inspect "$prepared_image" >/dev/null 2>&1; then
      emit_progress "image-cache" "running" "using prepared session image $prepared_image" 10
      run_image="$prepared_image"
      used_prepared_image_cache=true
      actions+=("reused-prepared-image-cache")
    elif [ -n "$prepared_image" ]; then
      emit_progress "image-cache" "running" "prepared session image cache miss; will populate after ssh package install" 10
    fi
  fi

  mount_args=()
  for optional in /home /tmp /weight /weights /models /data /mnt; do
    if [ -e "$optional" ]; then
      mount_args+=("-v" "$optional:$optional")
    fi
  done

  visibility_args=()
  if [ -n "$visible_devices" ]; then
    visibility_args+=("--env" "ASCEND_RT_VISIBLE_DEVICES=$visible_devices")
    visibility_args+=("--label" "com.mindie.visible_devices=$visible_devices")
  fi

  emit_progress "container" "running" "creating managed container" 45
  docker run --name "$container" -it -d --network host --shm-size=500g \
    --privileged=true \
    --label com.mindie.managed=true \
    --label com.mindie.container_ssh_port="$port" \
    --label com.mindie.workdir="$workdir" \
    --label com.mindie.namespace="$namespace" \
    --label com.mindie.machine_type="$machine_type" \
    --label com.mindie.container_type="$machine_type" \
    --label com.mindie.soc="$soc" \
    -w "$workdir" \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    --entrypoint=bash \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime:ro \
    "${mount_args[@]}" \
    "${visibility_args[@]}" \
    --label com.mindie.base_image="$selected_image" \
    --label com.mindie.base_image_id="$selected_image_id" \
    --label com.mindie.run_image="$run_image" \
    --label com.mindie.prepared_image="$prepared_image" \
    "$run_image" >/dev/null

  created=true
  actions+=("created-container")
fi

if [ -z "$machine_type" ]; then
  machine_type="$(infer_type_from_image "$selected_image" || true)"
fi
container_type="$machine_type"
write_host_state "$machine_type" "$container_type" "$soc" "$selected_image"

if ! docker exec "$container" bash -lc 'command -v sshd >/dev/null 2>&1 && command -v ssh >/dev/null 2>&1'; then
  emit_progress "container-ssh" "running" "installing openssh inside the container" 900
  if ! run_with_progress "container-ssh-install" "installing openssh inside the container" 900 install_container_ssh_packages; then
    emit_json '{"success": false, "error": "installing openssh inside container failed", "phase": "container-ssh-install"}'
    exit 30
  fi
  package_bootstrap="$(docker exec "$container" bash -lc 'cat /tmp/mindie-package-bootstrap.json 2>/dev/null || true')"
  docker exec "$container" bash -lc 'rm -f /tmp/mindie-package-bootstrap.json' >/dev/null 2>&1 || true
  if [ -z "$package_bootstrap" ]; then
    package_bootstrap='{"package_manager": null, "selected_mirror": null, "candidate_timings_ms": {}, "changed_files": [], "changed": false, "install_skipped": false}'
  fi
  installed_ssh=true
  actions+=("installed-openssh")
  if [ "$use_prepared_image_cache" = "true" ] && [ -n "$prepared_image" ] && ! docker image inspect "$prepared_image" >/dev/null 2>&1; then
    emit_progress "image-cache" "running" "committing prepared session image $prepared_image" 120
    if run_with_progress "image-cache-commit" "committing prepared session image" 120 docker commit "$container" "$prepared_image"; then
      created_prepared_image_cache=true
      actions+=("created-prepared-image-cache")
    else
      emit_progress "image-cache" "running" "prepared session image commit failed; continuing without cache" 30
    fi
  fi
else
  package_bootstrap='{"package_manager": null, "selected_mirror": null, "candidate_timings_ms": {}, "changed_files": [], "changed": false, "install_skipped": true}'
  if [ "$use_prepared_image_cache" = "true" ] && [ "$run_image" = "$selected_image" ] && [ -n "$prepared_image" ] && ! docker image inspect "$prepared_image" >/dev/null 2>&1; then
    emit_progress "image-cache" "running" "tagging selected image as prepared session image $prepared_image" 30
    if docker tag "$selected_image" "$prepared_image" >/dev/null 2>&1; then
      created_prepared_image_cache=true
      actions+=("tagged-prepared-image-cache")
    fi
  fi
fi

emit_progress "container-ssh" "running" "configuring dedicated container sshd" 60
if ! run_with_progress "container-ssh-config" "configuring dedicated container sshd" 60 configure_container_state; then
  emit_json '{"success": false, "error": "configuring dedicated container sshd failed", "phase": "container-ssh"}'
  exit 31
fi
actions+=("configured-container-env")
actions+=("configured-dedicated-sshd")

emit_progress "firewall" "running" "updating host firewall for the container ssh port" 30
if command -v ufw >/dev/null 2>&1; then
  ufw allow "${port}/tcp" >/dev/null 2>&1 || true
  firewall="ufw"
elif command -v firewall-cmd >/dev/null 2>&1 && firewall-cmd --state >/dev/null 2>&1; then
  firewall-cmd --quiet --add-port="${port}/tcp" --permanent >/dev/null 2>&1 || true
  firewall-cmd --quiet --reload >/dev/null 2>&1 || true
  firewall="firewalld"
fi

payload=$(python3 - <<'PY' "$container" "$port" "$image_request_json" "$selected_image" "$image_resolution" "$workdir" "$namespace" "$created" "$started" "$pulled" "$installed_ssh" "$firewall" "${actions[*]}" "$previous_image" "$replace_on_image_change" "$machine_type" "$container_type" "$soc" "$host_env_file" "$host_metadata_file" "$container_env_file" "$container_metadata_file" "$package_bootstrap" "$run_image" "$prepared_image" "$use_prepared_image_cache" "$used_prepared_image_cache" "$created_prepared_image_cache" "$docker_pull_policy" "$visible_devices" "$selected_image_id" "$runtime_image_candidates_json" "$local_image_discovery"
import json
import sys
(
    container,
    port,
    image_request_json,
    selected_image,
    image_resolution,
    workdir,
    namespace,
    created,
    started,
    pulled,
    installed_ssh,
    firewall,
    actions,
    previous_image,
    replace_on_image_change,
    machine_type,
    container_type,
    soc,
    host_env_file,
    host_metadata_file,
    container_env_file,
    container_metadata_file,
    package_bootstrap,
    run_image,
    prepared_image,
    use_prepared_image_cache,
    used_prepared_image_cache,
    created_prepared_image_cache,
    docker_pull_policy,
    visible_devices,
    selected_image_id,
    runtime_image_candidates_json,
    local_image_discovery_json,
) = sys.argv[1:]
image_request = json.loads(image_request_json)
print(json.dumps({
    "success": True,
    "container": container,
    "container_ssh_port": int(port),
    "image": selected_image,
    "requested_image": image_request.get("requested"),
    "requested_selector": image_request.get("selector"),
    "image_policy": image_request.get("policy"),
    "resolved_tag": image_request.get("resolved_tag"),
    "selected_image": selected_image,
    "selected_image_id": selected_image_id or None,
    "run_image": run_image or selected_image,
    "prepared_image": prepared_image or None,
    "prepared_image_cache_requested": use_prepared_image_cache == "true",
    "used_prepared_image_cache": used_prepared_image_cache == "true",
    "created_prepared_image_cache": created_prepared_image_cache == "true",
    "image_resolution": image_resolution,
    "docker_pull_policy": docker_pull_policy,
    "visible_devices": [int(item) for item in visible_devices.split(",") if item] if visible_devices else [],
    "image_candidates": json.loads(runtime_image_candidates_json),
    "image_mirror_order": list(image_request.get("mirror_order") or []),
    "local_image_discovery": json.loads(local_image_discovery_json) if local_image_discovery_json != "null" else None,
    "workdir": workdir,
    "namespace": namespace or None,
    "previous_image": previous_image or None,
    "created": created == "true",
    "started_existing": started == "true",
    "pulled_image": pulled == "true",
    "installed_openssh": installed_ssh == "true",
    "replace_container_on_image_change": replace_on_image_change == "true",
    "firewall": firewall,
    "actions": [item for item in actions.split() if item],
    "machine_type": machine_type or None,
    "container_type": container_type or None,
    "soc": soc or None,
    "host_env_file": host_env_file,
    "host_metadata_file": host_metadata_file,
    "container_env_file": container_env_file,
    "container_metadata_file": container_metadata_file,
    "package_bootstrap": json.loads(package_bootstrap) if package_bootstrap else None,
}, ensure_ascii=False))
PY
)
emit_progress "complete" "done" "managed container bootstrap completed" 0
emit_json "$payload"
"""
