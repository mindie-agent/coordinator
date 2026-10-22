"""Remote runtime installation and materialization scripts for code parity.

[Role]
- Renders remote install steps (`prepare_isolated_root_script`,
  `runtime_install_step_script`), materializes fixed snapshots into a runtime
  root, and exports missing source objects between peers.

[Invariants]
- Install steps are pinned by exact Git commits and a task-owned interpreter;
  packages supplied by the base image are never downgraded.
- Manifest/summary payloads describe the admitted snapshot; uncertain remote
  outcomes are never reported as success.

[Timeouts & Locks]
- Callers hold the parity lock; remote install steps stream through
  `ssh_exec_stream` and preserve caller-supplied process/log deadlines.
  Export and materialization timeouts propagate instead of being replayed.

[Side-Effect Policy]
- Remote writes are confined to the isolated runtime root and carrier cache;
  no local worktree or source ref is modified here.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

from mindie_coordinator.parity_snapshot import (
    ASCEND_PIP_EXTRA_INDEX_URL, ASCEND_PIP_TRUSTED_HOST, DEFAULT_ENV_PREAMBLE,
    DEFAULT_ROOT_PRESERVE_PATHS, IMAGE_PROVIDED_REQUIREMENT_NAMES,
    PARITY_BRANCH_NAME, PIP_INDEX_NAME, PIP_INDEX_URL, PIP_TRUSTED_HOST,
    PROGRESS_SENTINEL, REMOTE_RUNTIME_ENV_PASSTHROUGH, ParityUnavailable,
    SnapshotRecord, emit_progress, mirror_path_for, validate_relative_posix_path,
)
from mindie_coordinator.parity_support import (
    SshEndpoint, ensure_local_git_identity, now_utc, quoted, ssh_exec, ssh_exec_stream,
)

def materialize_command(
    *, workspace_id: str, runtime_id: str, endpoint: dict, sources: dict[str, str],
    workspace_root: Path | str | None = None, source_snapshot: dict | None = None,
) -> list[str]:
    """Build the in-package parity CLI that materializes explicit source repos.

    Bound ``sources`` are the worktrees to snapshot. A Git parent of those
    trees is not required. ``workspace_root`` is only optional local state.
    """
    if not sources and not (source_snapshot or {}).get("records"):
        raise ValueError("materialization requires at least one fixed source")
    command = [
        sys.executable, "-m", "mindie_coordinator.parity", "sync",
        "--workspace-id", workspace_id, "--server-name", runtime_id,
        "--runtime-root", endpoint["root"], "--container-identity", runtime_id,
        "--container-host", endpoint["host"], "--container-port", str(endpoint["port"]),
        "--container-user", endpoint["user"], "--apply-mode", "materialize",
    ]
    if workspace_root:
        command.extend(["--workspace-root", str(Path(workspace_root).expanduser())])
    for name, path in sources.items():
        command.extend(["--source", name + "=" + path])
    if source_snapshot is not None:
        # The durable descriptor is already captured by submission. This file
        # transports that data; parity must not capture a newer working tree.
        state = Path(workspace_root or tempfile.gettempdir()) / ".mindie-local" / "fixed-inputs"
        state.mkdir(parents=True, exist_ok=True)
        transport_key = hashlib.sha256(json.dumps(source_snapshot, sort_keys=True).encode()).hexdigest()[:20]
        descriptor = state / (source_snapshot["id"] + '-' + transport_key + ".json")
        if not descriptor.exists():
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=state, delete=False) as out:
                json.dump(source_snapshot, out)
                temporary = Path(out.name)
            os.replace(temporary, descriptor)
        command.extend(["--snapshot-file", str(descriptor)])
    return command

def _write_materialized_build_source(root, data):
    """Publish fixed SCM/build metadata only after complete source materialization."""
    import json
    import os
    import tempfile
    import time
    from pathlib import Path
    started = time.monotonic()
    directory = Path(root) / '.mindie-runtime'
    directory.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.build-source-', dir=directory)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, sort_keys=True)
        os.replace(temporary, directory / 'build-source.json')
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return time.monotonic() - started

def _export_existing_source_objects(host_endpoint, records, legacy_cache):
    import inspect
    from remote_dev.core.endpoint import resolve_endpoint
    from remote_dev.core.ssh_transport import run_remote_python
    from mindie_coordinator.preparation_process import PreparationCancelled, PreparationUncertain
    from mindie_coordinator.shared_source_objects import (
        copy_fixed_objects, export_container_objects, export_existing_objects,
    )
    source = inspect.getsource(copy_fixed_objects) + '\n' + inspect.getsource(export_container_objects)
    source += '\nimport json,sys\nprint(json.dumps(export_container_objects(json.load(sys.stdin))))\n'
    script = inspect.getsource(export_existing_objects)
    script += '\nimport json,sys\nprint(json.dumps(export_existing_objects(json.load(sys.stdin))))\n'
    result = run_remote_python(resolve_endpoint({**host_endpoint, 'root': '/', 'cwd': '/'}), script,
        {'records': records, 'legacy_cache': legacy_cache, 'export_source': source}, timeout_ms=360000)
    if result.get('status') == 'cancelled':
        raise PreparationCancelled('fixed object export cancelled; it was not replayed')
    if (result.get('status') == 'failed' and result.get('exit_code') is not None
            and result.get('remote_outcome') != 'unknown'):
        raise ParityUnavailable('fixed object export failed: ' + str(result.get('stderr_tail') or result))
    if result.get('status') not in {'copied', 'miss'} or result.get('remote_outcome') == 'unknown':
        raise PreparationUncertain('fixed object export did not complete; it was not replayed: ' + str(result))
    return result

def remote_runtime_env_exports() -> list[str]:
    lines: list[str] = []
    for key in REMOTE_RUNTIME_ENV_PASSTHROUGH:
        if key in os.environ:
            lines.append(f'export {key}={quoted(os.environ[key])}')
    return lines

def resolved_root_preserve_paths(marker_dirname: str, extra_paths: list[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    for path in [*DEFAULT_ROOT_PRESERVE_PATHS, marker_dirname, *extra_paths]:
        normalized = validate_relative_posix_path(path, label='preserve path')
        if normalized not in ordered:
            ordered.append(normalized)
    return tuple(ordered)

def marker_path_for(runtime_root: str, marker_dirname: str) -> str:
    return str(PurePosixPath(runtime_root) / marker_dirname / 'runtime-install.json')

def container_repo_path(runtime_root: str, record: SnapshotRecord) -> str:
    if record.relpath in ('', '.'):
        return runtime_root
    return str(PurePosixPath(runtime_root) / record.relpath)

def prepare_isolated_root_script(runtime_root: str) -> str:
    """Create or reset an isolated task root. Never pip-uninstall image packages."""
    hostname_repair = (
        'if command -v hostname >/dev/null 2>&1; then '
        '( flock -x 9; h="$(hostname 2>/dev/null || true)"; '
        'if [ -n "$h" ] && ! grep -q -F "$h" /etc/hosts 2>/dev/null; then '
        'echo "127.0.0.1 $h" >> /etc/hosts; fi ) 9>/tmp/mindie-hostname.lock; fi'
    )
    return '\n'.join(
        [
            'set -eo pipefail',
            f'mkdir -p {quoted(runtime_root)}',
            hostname_repair,
            f'rm -rf {quoted(str(PurePosixPath(runtime_root) / "vllm"))} {quoted(str(PurePosixPath(runtime_root) / "vllm-ascend"))}',
            f'rm -rf {quoted(str(PurePosixPath(runtime_root) / ".git/modules/vllm"))} {quoted(str(PurePosixPath(runtime_root) / ".git/modules/vllm-ascend"))}',
        ]
    )

def task_python_exports(python: str) -> list[str]:
    """Pin every pip/build/executable selection to the task-owned interpreter."""
    q = quoted(python)
    return [
        f'export PYTHON={q}',
        'export PIP="$PYTHON -m pip"',
        'export HI_PYTHON="$PYTHON"',
        'export Python3_EXECUTABLE="$PYTHON"',
        'export Python_EXECUTABLE="$PYTHON"',
        'PYTHON_BIN_DIR="$(dirname "$PYTHON")"',
        'if [ -n "${MINDIE_PYTHON_SHIM_DIR:-}" ]; then',
        '  ln -sf "$PYTHON" "$MINDIE_PYTHON_SHIM_DIR/python"',
        '  ln -sf "$PYTHON" "$MINDIE_PYTHON_SHIM_DIR/python3"',
        'fi',
        'export PATH="${MINDIE_PYTHON_SHIM_DIR:+$MINDIE_PYTHON_SHIM_DIR:}$PYTHON_BIN_DIR:$PATH"',
        'hash -r',
        'export CMAKE_ARGS="-DPython3_EXECUTABLE=\\"$PYTHON\\" -DPython_EXECUTABLE=\\"$PYTHON\\""',
        f'test -x {q}',
    ]

def render_git_clean(repo_dir: str, preserve_paths: tuple[str, ...]) -> str:
    parts = ['git', '-C', quoted(repo_dir), 'clean', '-ffd']
    for path in preserve_paths:
        parts.extend(['-e', quoted(path)])
    parts.append('>/dev/null')
    return ' '.join(parts)

def materialize_runtime(
    *,
    container: SshEndpoint,
    runtime_root: str,
    container_cache_root: str,
    workspace_id: str,
    marker_dirname: str,
    root_preserve_paths: tuple[str, ...],
    records: list[SnapshotRecord],
    dry_run: bool,
) -> None:
    record_by_relpath = {record.relpath: record for record in records}
    if '.' in record_by_relpath:
        roots = [record_by_relpath['.']]
    else:
        roots = [record for record in records if '/' not in record.relpath]
    if not roots:
        raise ValueError('materialize requires explicit source records (vllm, vllm-ascend)')
    parity_tracking_ref = f'refs/remotes/parity/{PARITY_BRANCH_NAME}'

    def render_repo_step(record: SnapshotRecord) -> str:
        repo_dir = container_repo_path(runtime_root, record)
        mirror_path = mirror_path_for(container_cache_root, workspace_id, record)
        lines = ['set -eo pipefail', f'mkdir -p {quoted(str(PurePosixPath(repo_dir).parent))}']
        if record.relpath in ('', '.'):
            lines.append(f'if [ ! -e {quoted(str(PurePosixPath(repo_dir) / ".git"))} ]; then git init {quoted(repo_dir)} >/dev/null; fi')
        else:
            lines.append(
                f'if [ ! -e {quoted(str(PurePosixPath(repo_dir) / ".git"))} ]; then rm -rf {quoted(repo_dir)} && git clone --no-checkout {quoted(mirror_path)} {quoted(repo_dir)} >/dev/null; fi'
            )
        lines.extend(
            [
                f'git -C {quoted(repo_dir)} remote get-url parity >/dev/null 2>&1 || git -C {quoted(repo_dir)} remote add parity {quoted(mirror_path)}',
                f'git -C {quoted(repo_dir)} remote set-url parity {quoted(mirror_path)}',
                f'git -C {quoted(repo_dir)} fetch --force --no-recurse-submodules parity {quoted(PARITY_BRANCH_NAME + ":" + parity_tracking_ref)} >/dev/null',
                f'git -C {quoted(repo_dir)} checkout -B parity/current {quoted(parity_tracking_ref)} >/dev/null',
                f'git -C {quoted(repo_dir)} reset --hard {quoted(parity_tracking_ref)} >/dev/null',
            ]
        )
        lines.append(render_git_clean(repo_dir, root_preserve_paths))
        for child in record.submodules:
            child_relpath = child['path'] if record.relpath in ('', '.') else f"{record.relpath}/{child['path']}"
            child_record = record_by_relpath[child_relpath]
            child_mirror = mirror_path_for(container_cache_root, workspace_id, child_record)
            submodule_url_key = f"submodule.{child['name']}.url"
            lines.extend(
                [
                    f'git -C {quoted(repo_dir)} config {quoted(submodule_url_key)} {quoted(child_mirror)}',
                    f'git -C {quoted(repo_dir)} submodule sync -- {quoted(child["path"])} >/dev/null || true',
                ]
            )
        return '\n'.join(lines)

    def collect_scripts(record: SnapshotRecord, out: list[str]) -> None:
        emit_progress('materialize-repo', relpath=record.relpath)
        out.append(render_repo_step(record))
        for child in record.submodules:
            child_relpath = child['path'] if record.relpath in ('', '.') else f"{record.relpath}/{child['path']}"
            collect_scripts(record_by_relpath[child_relpath], out)

    if dry_run:
        return
    parts: list[str] = [
        'set -eo pipefail',
        f'mkdir -p {quoted(runtime_root)}',
        f'mkdir -p {quoted(str(PurePosixPath(runtime_root) / marker_dirname))}',
    ]
    repo_scripts: list[str] = []
    for root_record in roots:
        collect_scripts(root_record, repo_scripts)
    parts.extend(repo_scripts)
    # Resolve through the public module: existing tests and callers replace
    # parity.ssh_exec to observe the exact materialization script.
    from mindie_coordinator import parity as _parity
    _parity.ssh_exec(container, '\n'.join(parts))

def runtime_install_step_script(
    *,
    runtime_root: str,
    marker_dirname: str,
    container_identity: str,
    step: str,
    uninstall_packages: tuple[str, ...] = (),
    python: str | None = None,
) -> str:
    lines = ['set -euo pipefail', f'cd {quoted(runtime_root)}']
    lines.extend(remote_runtime_env_exports())
    lines.append(f'export MINDIE_RUNTIME_ROOT={quoted(runtime_root)}')
    if step != 'write-marker':
        lines.extend(DEFAULT_ENV_PREAMBLE)
    if python and step != 'write-marker':
        lines.extend(task_python_exports(python))
    if step in {'verify-imports', 'verify-deps'}:
        # Shared native reuse copies outputs instead of installing an editable
        # package. Verify this execution view, as the final profile probe does.
        lines.extend([
            'if [ -f "$MINDIE_RUNTIME_ROOT/.mindie-runtime/shared-native.json" ]; then',
            '  export PYTHONPATH="$MINDIE_RUNTIME_ROOT/.mindie-runtime/metadata:$MINDIE_RUNTIME_ROOT/vllm:$MINDIE_RUNTIME_ROOT/vllm-ascend${PYTHONPATH:+:$PYTHONPATH}"',
            'fi',
        ])
    if step in {'install-vllm', 'install-vllm-ascend', 'install-vllm-ascend-incremental', 'install-vllm-ascend-requirements', 'check-build-compat'}:
        source_name = 'vllm' if step == 'install-vllm' else 'vllm-ascend'
        # Only package-created fixed input data enters the build shell. Never
        # derive versions from the synthetic commit's truncated history.
        lines.extend([
            'if [ -f "$MINDIE_RUNTIME_ROOT/.mindie-runtime/build-source.json" ]; then',
            '  eval "$("$PYTHON" - ' + quoted(source_name) + ' "$MINDIE_RUNTIME_ROOT/.mindie-runtime/build-source.json" <<\'MINDIE_BUILD_INPUTS\'',
            'import json, shlex, sys',
            'spec = json.load(open(sys.argv[2]))',
            'for key, value in spec.get("build_env", {}).items():',
            '    if not __import__("re").fullmatch(r"[A-Z_][A-Z0-9_]*", key): raise ValueError("invalid build environment key")',
            '    print("export " + key + "=" + shlex.quote(value))',
            'version = spec["versions"][sys.argv[1]]["version"]',
            'print("export SETUPTOOLS_SCM_PRETEND_VERSION=" + shlex.quote(version))',
            'MINDIE_BUILD_INPUTS',
            ')"',
            'fi',
        ])
    if step in {'install-vllm', 'install-vllm-ascend', 'install-vllm-ascend-requirements'}:
        lines.extend(
            [
                'emit_progress() {',
                "  python3 - \"$1\" \"$2\" \"$3\" \"${4:-}\" <<'PY' >&2",
                'import json',
                'import sys',
                'payload = {"phase": sys.argv[1], "message": sys.argv[2]}',
                'if sys.argv[4]: payload["remote_log"] = sys.argv[4]',
                'if len(sys.argv) > 3 and sys.argv[3]:',
                '    try:',
                '        payload["expected_seconds"] = int(sys.argv[3])',
                '    except ValueError:',
                '        pass',
                f'print("{PROGRESS_SENTINEL}" + json.dumps(payload, ensure_ascii=False))',
                'PY',
                '}',
                'run_with_log_progress() {',
                '  phase="$1"',
                '  message="$2"',
                '  expected_seconds="$3"',
                '  log_file="$4"',
                '  shift 4',
                '  "$@" >"$log_file" 2>&1 &',
                '  pid=$!',
                '  emit_progress "$phase" "$message" "$expected_seconds" "$log_file"',
                '  start_ts=$(date +%s)',
                '  while kill -0 "$pid" 2>/dev/null; do',
                '    sleep 8',
                '    if ! kill -0 "$pid" 2>/dev/null; then',
                '      break',
                '    fi',
                '    elapsed=$(( $(date +%s) - start_ts ))',
                '    if [ -s "$log_file" ]; then',
                '      last_line="$(tail -n 1 "$log_file" 2>/dev/null | tr -d \"\\r\" | sed \"s/[^[:print:]\\t]//g\" | cut -c1-180)"',
                '      if [ -n "$last_line" ]; then',
                '        emit_progress "$phase" "$message - $last_line" "$expected_seconds"',
                '      else',
                '        emit_progress "$phase" "$message - still working (elapsed ${elapsed}s)" "$expected_seconds"',
                '      fi',
                '    else',
                '      emit_progress "$phase" "$message - still working (elapsed ${elapsed}s)" "$expected_seconds"',
                '    fi',
                '  done',
                '  set +e',
                '  wait "$pid"',
                '  status=$?',
                '  set -e',
                '  if [ "$status" -ne 0 ]; then',
                '    tail -n 160 "$log_file" >&2 || true',
                '  fi',
                '  return "$status"',
                '}',
                'run_with_progress() {',
                '  phase="$1"',
                '  message="$2"',
                '  expected_seconds="$3"',
                '  shift 3',
                '  mkdir -p "$MINDIE_RUNTIME_ROOT/.mindie-runtime/prepare-logs"',
                '  log_file="$(mktemp "$MINDIE_RUNTIME_ROOT/.mindie-runtime/prepare-logs/${phase}.XXXXXX")"',
                '  set +e',
                '  run_with_log_progress "$phase" "$message" "$expected_seconds" "$log_file" "$@"',
                '  status=$?',
                '  set -e',
                '  return "$status"',
                '}',
                'configure_pip_index() {',
                '  unset PIP_EXTRA_INDEX_URL',
                '  export PIP_CONFIG_FILE=/dev/null',
                f'  export PIP_INDEX_URL={quoted(PIP_INDEX_URL)}',
                f'  export PIP_TRUSTED_HOST={quoted(PIP_TRUSTED_HOST)}',
                f'  emit_progress "runtime-pip-index" "using pip index {PIP_INDEX_NAME}" 30',
                '}',
                'pip_install_fast() {',
                '  phase="$1"',
                '  message="$2"',
                '  expected_seconds="$3"',
                '  shift 3',
                '  configure_pip_index',
                f'  run_with_progress "$phase" "$message via {PIP_INDEX_NAME}" "$expected_seconds" "$PYTHON" -m pip "$@"',
                '}',
                'install_editable_fast() {',
                '  phase="$1"',
                '  message="$2"',
                '  target_dir="$3"',
                '  expected_seconds="$4"',
                '  shift 4',
                '  cd "$target_dir"',
                '  configure_pip_index',
                f'  run_with_progress "$phase" "$message via {PIP_INDEX_NAME}" "$expected_seconds" "$@"',
                '}',
            ]
        )

    if step == 'uninstall':
        pkg_args = ' '.join(uninstall_packages) if uninstall_packages else 'vllm vllm-ascend vllm_ascend'
        lines.append(f'"$PYTHON" -m pip uninstall -y {pkg_args} >/dev/null 2>&1 || true')
    elif step == 'install-vllm':
        lines.extend(
            [
                f'cd {quoted(str(PurePosixPath(runtime_root) / "vllm"))}',
                'export VLLM_TARGET_DEVICE=empty',
                'export TORCH_DEVICE_BACKEND_AUTOLOAD=0',
                'install_editable_fast "runtime-install-vllm" "building editable vllm" . 900 "$PYTHON" -m pip install --no-deps -e . --no-build-isolation',
            ]
        )
    elif step == 'install-vllm-ascend-requirements':
        # The hardware-coupled stack (torch / torch_npu / triton-ascend /
        # torchvision / torchaudio) is decided per package at install time:
        #   - already installed in the image -> drop the pinned line (avoid
        #     downgrading the image's tested runtime);
        #   - genuinely missing -> keep it and add the public Ascend
        #     extra-index so it resolves from mirrors.huaweicloud.com/ascend,
        #     exactly like vllm-ascend/Dockerfile pulls torch_npu/triton.
        # Regex matches the requirement name at line start, terminated by a
        # version op / marker / comment / EOL, so ``torch`` never swallows
        # ``torchvision``. Constant names are already PEP 503 canonical.
        _names = sorted(
            {name.strip().lower() for name in IMAGE_PROVIDED_REQUIREMENT_NAMES},
            key=len,
            reverse=True,
        )
        _pkg_group = '(' + '|'.join(_names) + ')'
        _line_re = r'^[[:space:]]*' + _pkg_group + r'[[:space:]]*($|[<>=!~;#[])'
        # awk uses the captured package name to test installability on the
        # image, so keep the group in a shell-friendly single expression.
        # The terminated line regex is the MATCH gate: without it a bare
        # prefix would misclassify e.g. ``torchao`` as image-provided ``torch``
        # and silently drop the line from the install list.
        lines.extend(
            [
                f'cd {quoted(str(PurePosixPath(runtime_root) / "vllm-ascend"))}',
                'filtered_req="$(mktemp -t mindie-req.XXXXXX.txt)"',
                'dropped=""',
                'kept_hw=""',
                'while IFS= read -r req_line || [ -n "$req_line" ]; do',
                f'  if printf "%s" "$req_line" | grep -qiE {quoted(_line_re)}; then',
                f'    pkg="$(printf "%s" "$req_line" | grep -oiE {quoted("^[[:space:]]*" + _pkg_group)} | tr -d "[:space:]" | tr "[:upper:]" "[:lower:]")"',
                '  else',
                '    pkg=""',
                '  fi',
                '  if [ -n "$pkg" ]; then',
                # Treat the package as image-provided if pip knows about it,
                # regardless of the pinned version (prevents downgrade).
                '    if "$PYTHON" -m pip show "$pkg" >/dev/null 2>&1; then',
                '      dropped="$dropped $pkg"',
                '      continue',
                '    else',
                '      kept_hw="$kept_hw $pkg"',
                '    fi',
                '  fi',
                '  printf "%s\\n" "$req_line" >> "$filtered_req"',
                'done < requirements.txt',
                '[ -n "$dropped" ] && emit_progress "runtime-req-filter" '
                '"image provides:$dropped" 5 || true',
                # If any hardware package is genuinely missing, allow the
                # public Ascend index for this install only.
                'req_index_args=""',
                '[ -n "$kept_hw" ] && emit_progress "runtime-req-ascend-index" '
                f'"pulling from ascend index:$kept_hw" 5 && req_index_args={quoted("--extra-index-url " + ASCEND_PIP_EXTRA_INDEX_URL + " --trusted-host " + ASCEND_PIP_TRUSTED_HOST)} || true',
                'pip_install_fast "runtime-install-vllm-ascend-requirements" '
                '"installing vllm-ascend requirements (image stack reconciled)" 900 '
                'install $req_index_args -r "$filtered_req"',
                'rm -f "$filtered_req"',
            ]
        )
    elif step == 'check-build-compat':
        # Fail fast BEFORE the multi-minute custom-ops build when the
        # checked-out vllm-ascend submodule is not version-matched to the base
        # image. vllm-ascend/CMakeLists.txt hard-pins the expected torch
        # version (``if(NOT TORCH_VERSION VERSION_EQUAL "X.Y.Z")
        # message(FATAL_ERROR ...)``); if it disagrees with the image's torch
        # the cmake configure aborts deep in the build (the confusing
        # "kineto_LIBRARY-NOTFOUND" + FATAL_ERROR we hit on qwen35-125). Detect
        # the mismatch up front and tell the user how to resolve it instead of
        # wasting the build time. This is a preflight, never a silent skip.
        lines.extend(
            [
                f'cd {quoted(str(PurePosixPath(runtime_root) / "vllm-ascend"))}',
                'required_torch="$(grep -oE \'VERSION_EQUAL[[:space:]]*"[0-9]+\\.[0-9]+\\.[0-9]+"\' CMakeLists.txt 2>/dev/null | grep -oE \'[0-9]+\\.[0-9]+\\.[0-9]+\' | head -1)"',
                'if [ -z "$required_torch" ]; then',
                '  echo "build-compat: no torch version pin in CMakeLists.txt; skipping preflight"',
                'else',
                '  installed_torch="$("$PYTHON" -c \'import torch; print(torch.__version__.split("+")[0])\' 2>/dev/null || true)"',
                '  if [ -z "$installed_torch" ]; then',
                '    echo "ERROR: cannot import torch in the image to check build compatibility" >&2',
                '    exit 1',
                '  fi',
                '  if [ "$required_torch" != "$installed_torch" ]; then',
                '    echo "ERROR: vllm-ascend custom-ops build requires torch==$required_torch but this image ships torch==$installed_torch." >&2',
                '    echo "The bound vllm-ascend sources are not version-matched to this environment." >&2',
                '    echo "Bind a vllm-ascend worktree whose CMakeLists.txt expects torch $installed_torch," >&2',
                '    echo "or request an environment recipe/image that provides torch $required_torch." >&2',
                '    echo "The coordinator will not skip the custom-ops build or retarget the image stack." >&2',
                '    exit 1',
                '  fi',
                '  echo "build-compat: image torch $installed_torch matches vllm-ascend requirement"',
                'fi',
            ]
        )
    elif step == 'install-vllm-ascend-incremental':
        import inspect
        from mindie_coordinator import native_incremental, native_kernel_recipe, runtime_profile
        script = Path(native_incremental.__file__).read_text(encoding='utf-8')
        compiler = Path(native_kernel_recipe.__file__).read_text(encoding='utf-8').replace('from __future__ import annotations\n', '')
        lines.extend([
            '"$PYTHON" - "$MINDIE_RUNTIME_ROOT" <<\'MINDIE_NATIVE_INCREMENTAL\'',
            script,
            compiler,
            inspect.getsource(runtime_profile.compiled_opc_recipe),
            'import sys',
            'build_incremental_kernel(Path(sys.argv[1]), compile_recipe=lambda root, plan, env: '
            'compile_kernel_recipe(root, plan, env, read_recipe=compiled_opc_recipe, '
            'content_metadata=(metadata := incremental_content_metadata(plan)), '
            'content_cache=(Path("/tmp/mindie-native-cache/kernel") if metadata else None)))',
            'MINDIE_NATIVE_INCREMENTAL',
        ])
    elif step == 'install-vllm-ascend':
        lines.extend(
            [
                f'cd {quoted(str(PurePosixPath(runtime_root) / "vllm-ascend"))}',
                'install_editable_fast "runtime-install-vllm-ascend" "building editable vllm-ascend custom ops" . 2400 "$PYTHON" -m pip install --no-deps -v -e . --no-build-isolation',
            ]
        )
    elif step == 'verify-imports':
        lines.extend(
            [
                '"$PYTHON" - <<\'PY\'',
                'import sys',
                'import torch',
                'import torch_npu  # noqa: F401',
                'import vllm',
                'import vllm_ascend',
                'print(f"editable-import-smoke=ok python={sys.executable} torch={torch.__version__} vllm={getattr(vllm, \'__version__\', \'unknown\')}")',
                'PY',
            ]
        )
    elif step == 'verify-deps':
        # Check vllm-ascend deps only.  vllm-ascend intentionally overrides
        # some vllm constraints (e.g. opencv-python-headless) to keep numpy
        # compatible with CANN, so checking vllm deps would false-positive.
        lines.extend(
            [
                '"$PYTHON" - <<\'PY\'',
                'import sys',
                'from importlib.metadata import requires, version as pkg_version',
                'from packaging.requirements import Requirement',
                'from packaging.utils import canonicalize_name',
                'from packaging.version import InvalidVersion, Version',
                '',
                'def importable(module):',
                '    try:',
                '        __import__(module)',
                '        return True',
                '    except Exception:',
                '        return False',
                '',
                'def public_version(value):',
                '    try:',
                '        return Version(value).public',
                '    except InvalidVersion:',
                '        return value.split("+", 1)[0]',
                '',
                'def requirement_satisfied(req, installed):',
                '    if not req.specifier:',
                '        return True',
                '    if req.specifier.contains(installed, prereleases=True):',
                '        return True',
                '    public = public_version(installed)',
                '    if public != installed and req.specifier.contains(public, prereleases=True):',
                '        return True',
                '    # Paired vLLM Ascend images provide torch_npu as runtime state.',
                '    if canonicalize_name(req.name) == "torch-npu" and importable("torch_npu"):',
                '        return True',
                '    return False',
                '',
                'errors = []',
                'try:',
                '    reqs = requires("vllm-ascend") or []',
                'except Exception:',
                '    reqs = []',
                'for raw in reqs:',
                '    try:',
                '        r = Requirement(raw)',
                '        if r.marker and not r.marker.evaluate():',
                '            continue',
                '        try:',
                '            installed = pkg_version(r.name)',
                '        except Exception:',
                '            if canonicalize_name(r.name) == "torch-npu" and importable("torch_npu"):',
                '                continue',
                '            raise',
                '        if not requirement_satisfied(r, installed):',
                '            errors.append(f"{r.name}{r.specifier} (installed {installed})")',
                '    except Exception as exc:',
                '        # Do NOT swallow: a missing package (raised above) or an',
                '        # unparseable requirement is a real verification failure.',
                '        errors.append(f"{raw!r} could not be verified: {exc!r}")',
                'if errors:',
                '    for e in errors:',
                '        print(f"MISMATCH: {e}", file=sys.stderr)',
                '    sys.exit(1)',
                'print("dependency-check=ok")',
                'PY',
            ]
        )
    elif step == 'write-marker':
        lines.extend(
            [
                f'mkdir -p {quoted(str(PurePosixPath(runtime_root) / marker_dirname))}',
                (
                    'cat > '
                    + quoted(marker_path_for(runtime_root, marker_dirname))
                    + " <<'JSON'\n"
                    + json.dumps(
                        {
                            'container_identity': container_identity,
                            'runtime_root': runtime_root,
                            'updated_at': now_utc(),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + '\nJSON'
                ),
            ]
        )
    else:
        raise ValueError(f'unknown runtime install step: {step}')
    return '\n'.join(lines)

def run_runtime_install_step(
    *,
    container: SshEndpoint,
    runtime_root: str,
    marker_dirname: str,
    container_identity: str,
    step: str,
    stream_progress: bool = False,
    uninstall_packages: tuple[str, ...] = (),
    python: str | None = None,
    on_progress=None,
    log_path=None,
    process=None,
) -> None:
    script = runtime_install_step_script(
        runtime_root=runtime_root,
        marker_dirname=marker_dirname,
        container_identity=container_identity,
        step=step,
        uninstall_packages=uninstall_packages,
        python=python,
    )
    # Route through the public module so monkeypatched parity SSH helpers keep
    # observing this install step.
    from mindie_coordinator import parity as _parity
    if stream_progress or on_progress is not None or log_path is not None or process is not None:
        _parity.ssh_exec_stream(container, script, stream_progress=stream_progress,
                                on_progress=on_progress, log_path=log_path, process=process)
    else:
        _parity.ssh_exec(container, script)

def verify_runtime_commits_map(
    *,
    container: SshEndpoint,
    runtime_root: str,
    expected: dict[str, str],
    include_untracked: bool = False,
) -> dict[str, str]:
    lines = ['set -eo pipefail']
    for relpath in expected:
        repo_dir = runtime_root if relpath in ('', '.') else str(PurePosixPath(runtime_root) / relpath)
        clean = f"git -C {quoted(repo_dir)} diff --quiet HEAD --"
        if include_untracked:
            clean += f' && test -z "$(git -C {quoted(repo_dir)} ls-files --others --exclude-standard)"'
        lines.append(
            f"if {clean}; then "
            f"printf '%s %s\\n' {quoted(relpath)} \"$(git -C {quoted(repo_dir)} rev-parse HEAD)\"; "
            f"else printf '%s %s\\n' {quoted(relpath)} dirty-runtime; fi"
        )
    # Resolve through the public module: tests and callers replace
    # parity.ssh_exec to observe the exact verification script.
    from mindie_coordinator import parity as _parity
    result = _parity.ssh_exec(container, '\n'.join(lines))
    observed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        relpath, commit = line.split(maxsplit=1)
        observed[relpath] = commit
    return observed

def verify_runtime_commits(
    *,
    container: SshEndpoint,
    runtime_root: str,
    records: list[SnapshotRecord],
    dry_run: bool,
) -> dict[str, str]:
    expected = {record.relpath: record.commit for record in records}
    if dry_run:
        return expected
    return verify_runtime_commits_map(container=container, runtime_root=runtime_root, expected=expected)

def make_manifest(
    *,
    workspace_root: Path,
    workspace_id: str,
    snapshot_id: str,
    server_name: str,
    container_identity: str,
    runtime_root: str,
    container_cache_root: str,
    marker_dirname: str,
    root_preserve_paths: tuple[str, ...],
    records: list[SnapshotRecord],
    runtime_install_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    git_name, git_email = ensure_local_git_identity(workspace_root)
    return {
        'schema_version': 2,
        'generated_at': now_utc(),
        'workspace_root': str(workspace_root),
        'workspace_id': workspace_id,
        'snapshot_id': snapshot_id,
        'server_name': server_name,
        'container_identity': container_identity,
        'runtime_root': runtime_root,
        'container_cache_root': container_cache_root,
        'marker_dirname': marker_dirname,
        'root_preserve_paths': list(root_preserve_paths),
        'git_identity': {'name': git_name, 'email': git_email},
        'repos': [asdict(record) for record in records],
        'runtime_install_env': runtime_install_env or {},
        'local_source_of_truth': 'tracked + staged + unstaged + untracked-nonignored',
    }

def summary_payload(
    *,
    status: str,
    server_name: str,
    container_identity: str,
    workspace_id: str,
    container_cache_root: str | None,
    records: list[SnapshotRecord],
    reinstall_status: str,
    reason: str | None,
    first_install: bool,
    runtime_install_env: dict[str, str] | None = None,
    observed_runtime_commits: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        'status': status,
        'server_name': server_name,
        'container_identity': container_identity,
        'workspace_id': workspace_id,
        'container_cache_root': container_cache_root,
        'first_install': first_install,
        'snapshot_commits': {record.relpath: record.commit for record in records},
        'build_inputs': {record.relpath: record.build_inputs for record in records if record.build_inputs},
        'runtime_commits': observed_runtime_commits,
        'reinstall': reinstall_status,
        'runtime_install_env': runtime_install_env or {},
        'reason': reason,
    }

def final_manifest(manifest: dict[str, Any], *, status: str, reinstall_status: str, runtime_commits: dict[str, str] | None) -> dict[str, Any]:
    enriched = dict(manifest)
    enriched['completed_at'] = now_utc()
    enriched['status'] = status
    enriched['reinstall'] = reinstall_status
    enriched['runtime_commits'] = runtime_commits
    return enriched

def resolve_state_root(workspace_root: str | None, sources: dict[str, Path]) -> Path:
    if workspace_root:
        return Path(workspace_root).expanduser().resolve()
    if sources:
        return next(iter(sources.values())).resolve().parent
    raise ValueError('bind the actual vllm and vllm-ascend worktrees')
