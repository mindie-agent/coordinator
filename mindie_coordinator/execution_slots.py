"""Per-execution compiler caches, Ascend env bootstrap, and Python runtime guards.

[Role]
Derives per-slot cache paths (`/tmp/mindie-slot-<slot_id>`) and renders the
shell setup preamble that:
1. Safely sources Ascend `set_env.sh` scripts under `set +eu` when `ASCEND_HOME_PATH`
   is missing in non-login container shells.
2. Installs a slot-local `sitecustomize.py` and `python`/`python3` wrapper that
   enforce `MINDIE_LEASED_DEVICES` (preventing unauthorized `ASCEND_RT_VISIBLE_DEVICES`
   overrides) and `MINDIE_WORKTREE_SOURCES` (preventing `python3 -I` or base-image
   `.pth` files from shadowing the active task worktree).

[Invariants]
- Slots never share writable compiler caches; source roots are retained.
- `slot_environment(slot_id)` returns the exact cache and slot identity variables
  expected by managed job receipts.

[Timeouts & Locks]
- Pure string/path construction only; callers bound remote setup and cleanup.

[Side-Effect Policy]
- All generated files live strictly under `/tmp/mindie-slot-<slot_id>` and are
  removed by `slot_cleanup_command(slot_id)`.
"""
from __future__ import annotations

import re
import shlex

SLOT_ENVIRONMENT = frozenset({
    "MINDIE_SLOT_ID", "ASCEND_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
})

ASCEND_ENV_BOOTSTRAP_SH = r"""_mindie_saved_opts="$-"
set +eu
for _mindie_drv_lib in /usr/local/Ascend/driver/lib64/common /usr/local/Ascend/driver/lib64/driver /usr/local/Ascend/driver/lib64; do
  if [ -d "$_mindie_drv_lib" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$_mindie_drv_lib:"*) ;;
      *) export LD_LIBRARY_PATH="$_mindie_drv_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
  fi
done
if [ -f /etc/profile.d/mindie-ascend-env.sh ] && [ -z "${ASCEND_HOME_PATH:-}" ]; then
  . /etc/profile.d/mindie-ascend-env.sh >/dev/null 2>&1 || true
fi
if [ -z "${ASCEND_HOME_PATH:-}" ] || [ -z "${ASCEND_OPP_PATH:-}" ]; then
  for _mindie_env_script in \
    /usr/local/Ascend/cann-*/set_env.sh \
    /usr/local/Ascend/cann/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/nnal/atb/set_env.sh \
    /usr/local/Ascend/mindie/set_env.sh; do
    if [ -f "$_mindie_env_script" ]; then
      . "$_mindie_env_script" >/dev/null 2>&1 || true
    fi
  done
fi
unset _mindie_drv_lib _mindie_env_script
case "$_mindie_saved_opts" in *e*) set -e ;; esac
case "$_mindie_saved_opts" in *u*) set -u ;; esac
unset _mindie_saved_opts"""

SLOT_SITECUSTOMIZE_PY = r"""import importlib.machinery
import importlib.util
import os
import sys

def _enforce_mindie_slot_guards():
    leased_raw = os.environ.get("MINDIE_LEASED_DEVICES")
    if leased_raw is not None:
        leased = [ item.strip() for item in leased_raw.split(",") if item.strip() ]
        current_raw = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
        current = [ item.strip() for item in current_raw.split(",") if item.strip() ]
        if not leased:
            if current:
                os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ""
                sys.stderr.write(
                    f"[mindie-slot-guard] restored ASCEND_RT_VISIBLE_DEVICES='' "
                    f"(blocked unleased override {current_raw!r} on CPU-only lease)\n"
                )
        else:
            leased_set = set(leased)
            if not current:
                os.environ["ASCEND_RT_VISIBLE_DEVICES"] = leased_raw
            elif all(item in leased_set for item in current):
                pass
            elif (
                0 < len(current) < len(leased)
                and all(item.isdigit() and 0 <= int(item) < len(leased) for item in current)
            ):
                mapped = ",".join(leased[int(item)] for item in current)
                os.environ["ASCEND_RT_VISIBLE_DEVICES"] = mapped
            else:
                os.environ["ASCEND_RT_VISIBLE_DEVICES"] = leased_raw
                sys.stderr.write(
                    f"[mindie-slot-guard] restored ASCEND_RT_VISIBLE_DEVICES={leased_raw!r} "
                    f"(blocked unleased override {current_raw!r})\n"
                )

    sources_raw = os.environ.get("MINDIE_WORKTREE_SOURCES", "")
    if sources_raw:
        worktree_dirs = [ os.path.abspath(p) for p in sources_raw.split(":") if p.strip() ]
        repo_basenames = {
            os.path.basename(p): p
            for p in worktree_dirs
            if os.path.basename(p) not in {"", "metadata"}
        }
        cleaned_path = []
        for entry in sys.path:
            if not entry:
                cleaned_path.append(entry)
                continue
            abs_entry = os.path.abspath(entry)
            base = os.path.basename(abs_entry)
            if base in repo_basenames and abs_entry != repo_basenames[base]:
                continue
            if abs_entry not in worktree_dirs:
                cleaned_path.append(entry)
        sys.path[:] = worktree_dirs + cleaned_path

    # Chain any downstream sitecustomize on sys.path outside this slot directory
    this_dir = os.path.abspath(os.path.dirname(__file__))
    search_paths = [ p for p in sys.path if p and os.path.abspath(p) != this_dir ]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", search_paths)
    if spec is not None and spec.origin and os.path.abspath(spec.origin) != os.path.abspath(__file__):
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

_enforce_mindie_slot_guards()
"""

SLOT_PYTHON_WRAPPER_SH = r"""#!/bin/sh
_slot_bin="$(cd "$(dirname "$0")" && pwd)"
_slot_site="$(dirname "$_slot_bin")/site"
if [ -n "${MINDIE_WORKTREE_SOURCES:-}" ]; then
  export PYTHONPATH="$_slot_site:$MINDIE_WORKTREE_SOURCES${PYTHONPATH:+:$PYTHONPATH}"
else
  export PYTHONPATH="$_slot_site${PYTHONPATH:+:$PYTHONPATH}"
fi
_real_py=""
if [ -n "${MINDIE_PYTHON:-}" ] && [ -x "$MINDIE_PYTHON" ] && [ "$(dirname "$MINDIE_PYTHON")" != "$_slot_bin" ]; then
  _real_py="$MINDIE_PYTHON"
else
  _old_ifs="$IFS"
  IFS=":"
  for _dir in $PATH; do
    if [ -n "$_dir" ] && [ "$_dir" != "$_slot_bin" ] && [ -x "$_dir/python3" ]; then
      _real_py="$_dir/python3"
      break
    fi
  done
  IFS="$_old_ifs"
fi
if [ -z "$_real_py" ]; then
  _real_py="/usr/bin/python3"
fi
# Strip -I (isolated mode) into -s (no user site) before script/-c/-m so
# PYTHONPATH and sitecustomize guards remain active for the task worktree.
_parsing_opts=1
set -- "$@"
_count=$#
_idx=0
while [ "$_idx" -lt "$_count" ]; do
  _arg="$1"
  shift
  _idx=$((_idx + 1))
  if [ "$_parsing_opts" -eq 1 ]; then
    case "$_arg" in
      --)
        _parsing_opts=0
        set -- "$@" "$_arg"
        ;;
      -I)
        set -- "$@" "-s"
        ;;
      -c|-m)
        _parsing_opts=0
        set -- "$@" "$_arg"
        ;;
      -[!-]*I*)
        _converted="$(printf '%s' "$_arg" | tr 'I' 's')"
        set -- "$@" "$_converted"
        ;;
      -*)
        set -- "$@" "$_arg"
        ;;
      *)
        _parsing_opts=0
        set -- "$@" "$_arg"
        ;;
    esac
  else
    set -- "$@" "$_arg"
  fi
done
exec "$_real_py" "$@"
"""


def slot_environment(slot_id: str) -> dict[str, str]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", slot_id):
        raise ValueError("invalid execution slot identity")
    root = "/tmp/mindie-slot-" + slot_id
    return {
        "MINDIE_SLOT_ID": slot_id,
        "ASCEND_CACHE_PATH": root + "/ascend_cache",
        "TORCHINDUCTOR_CACHE_DIR": root + "/inductor_cache",
        "TRITON_CACHE_DIR": root + "/triton_cache",
    }


def slot_preamble(slot_id: str) -> str:
    environment = slot_environment(slot_id)
    root = "/tmp/mindie-slot-" + slot_id
    site_dir = root + "/site"
    bin_dir = root + "/bin"
    exports = [f"export {name}={shlex.quote(value)}" for name, value in environment.items()]
    paths = [
        *(shlex.quote(value) for name, value in environment.items() if name != "MINDIE_SLOT_ID"),
        shlex.quote(site_dir),
        shlex.quote(bin_dir),
    ]
    q_site = shlex.quote(site_dir)
    q_bin = shlex.quote(bin_dir)
    return "\n".join([
        ASCEND_ENV_BOOTSTRAP_SH,
        *exports,
        "mkdir -p -- " + " ".join(paths),
        f"cat > {q_site}/sitecustomize.py <<'MINDIE_SLOT_SITECUSTOMIZE'\n{SLOT_SITECUSTOMIZE_PY}MINDIE_SLOT_SITECUSTOMIZE",
        f"cat > {q_bin}/python3 <<'MINDIE_SLOT_PYWRAP'\n{SLOT_PYTHON_WRAPPER_SH}MINDIE_SLOT_PYWRAP",
        f"chmod +x {q_bin}/python3",
        f"ln -sf python3 {q_bin}/python",
        f'export PYTHONPATH={q_site}"${{PYTHONPATH:+:$PYTHONPATH}}"',
        f'export PATH={q_bin}"${{PATH:+:$PATH}}"',
    ])


def slot_cleanup_command(slot_id: str) -> str:
    slot_environment(slot_id)
    return "rm -rf -- " + shlex.quote("/tmp/mindie-slot-" + slot_id)
