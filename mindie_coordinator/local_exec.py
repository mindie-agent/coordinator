"""Cross-platform local Git/Bash discovery, extended-path normalization, and UTF-8 helpers.

[Role]
  Centralize local subprocess discovery and execution kwargs across Windows (`CP936`,
  `CREATE_NO_WINDOW`, user-scoped Git for Windows, `\\\\?\\` extended-path stripping)
  and WSL2 (preventing `/mnt/c` DrvFS or PE `.exe` entries on `$PATH` from shadowing
  native POSIX `git` and `bash`).

[Invariants]
  * `resolve_bash` on Windows rejects `System32\\bash.exe`, `Sysnative\\bash.exe`,
    `SysWOW64\\bash.exe`, and `Microsoft\\WindowsApps\\bash.exe` WSL launcher stubs,
    deriving Git Bash (`..\\bin\\bash.exe` / `..\\usr\\bin\\bash.exe`) from `git.exe`
    or standard user/system Git installations.
  * `which_native`, `resolve_git`, and `resolve_bash` preserve explicit relative paths,
    preserve `..` until symlink resolution before `/proc/self/mountinfo` checks, and
    never swallow `TimeoutError` on POSIX or Windows.
  * `utf8_python_env` explicitly sets `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` so
    inherited `CP936` settings cannot override pipe encoding.

[Timeouts & Locks]
  Bounded local stat, 64-byte PE header, and `/proc/self/mountinfo` reads; no locks.

[Side-Effect Policy]
  Pure filesystem discovery and dictionary construction; never mutates `os.environ`.
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
from typing import Mapping

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
_IS_WINDOWS_OVERRIDE: bool | None = None
_MOUNTINFO_PATH = "/proc/self/mountinfo"
_WIN_SCRIPT_EXTS = (".exe", ".cmd", ".bat", ".com", ".ps1")
_WIN_DEFAULT_PATHEXT = ".COM;.EXE;.BAT;.CMD;.VBS;.JS;.WS;.MSC"
_WIN_RESERVED_STEMS = frozenset(
    {
        "CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$",
        *(f"COM{idx}" for idx in range(1, 10)),
        *(f"LPT{idx}" for idx in range(1, 10)),
        "COM\xb9", "COM\xb2", "COM\xb3", "LPT\xb9", "LPT\xb2", "LPT\xb3",
    }
)


def _is_windows() -> bool:
    return (os.name == "nt") if _IS_WINDOWS_OVERRIDE is None else _IS_WINDOWS_OVERRIDE


def _is_windows_fs() -> bool:
    return (sys.platform == "win32") if _IS_WINDOWS_OVERRIDE is None else _IS_WINDOWS_OVERRIDE


def no_window_kwargs() -> dict[str, int]:
    return {"creationflags": CREATE_NO_WINDOW} if _is_windows() else {}


def _env_get(env_map: Mapping[str, str], key: str, default: str | None = None) -> str | None:
    if key in env_map:
        return env_map[key]
    if _is_windows() or _is_windows_fs():
        folded = key.casefold()
        for existing_key, value in env_map.items():
            if existing_key.casefold() == folded:
                return value
    return default


def utf8_python_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if base is None else base
    if _is_windows() or _is_windows_fs():
        env = {
            k: v
            for k, v in source.items()
            if k.casefold() not in {"pythonutf8", "pythonioencoding"}
        }
    else:
        env = dict(source)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _valid_win32_components(parts: list[str]) -> bool:
    for part in parts:
        if not part or part in {".", ".."} or part.endswith((".", " ")) or ":" in part:
            return False
        stem = part.split(".", 1)[0].rstrip(" ").upper()
        if stem in _WIN_RESERVED_STEMS:
            return False
    return True


def strip_windows_extended_prefix(path: str | os.PathLike[str]) -> str:
    raw = os.fspath(path)
    if not _is_windows() or not raw.startswith("\\\\?\\") or "/" in raw:
        return raw
    if raw[:8].casefold() == "\\\\?\\unc\\":
        rest = raw[8:]
        body = rest[:-1] if rest.endswith("\\") else rest
        if not body or body.endswith("\\"):
            return raw
        parts = body.split("\\")
        unc = "\\" + raw[7:]
        if len(parts) >= 2 and len(unc) < 248 and _valid_win32_components(parts):
            return unc
        return raw
    if len(raw) >= 7 and raw[4].isascii() and raw[4].isalpha() and raw[5] == ":" and raw[6] == "\\":
        candidate = raw[4:]
        tail = candidate[3:]
        body = tail[:-1] if tail.endswith("\\") else tail
        if tail and not body:
            return raw
        parts = body.split("\\") if body else []
        if len(candidate) < 248 and _valid_win32_components(parts):
            return candidate
    return raw


def _is_regular_file(cand: str, *, require_exec: bool = False) -> bool:
    try:
        st = os.stat(cand)
        if not stat.S_ISREG(st.st_mode):
            return False
        return os.access(cand, os.X_OK) if require_exec else True
    except TimeoutError:
        raise
    except OSError:
        return False


def _join_uncollapsed(base: str, leaf: str) -> str:
    if base == "/":
        return f"/{leaf}"
    trimmed = base.rstrip("/")
    return f"{trimmed}/{leaf}"


def _strict_realpath(raw_text: str) -> str:
    probe = raw_text if raw_text.startswith("/") else _join_uncollapsed(os.getcwd(), raw_text)
    suffix: list[str] = []
    hops = 0
    while True:
        try:
            resolved = os.path.realpath(probe, strict=True)
            return os.path.normpath(os.path.join(resolved, *reversed(suffix))) if suffix else resolved
        except TimeoutError:
            raise
        except (FileNotFoundError, NotADirectoryError):
            try:
                if hops < 40 and stat.S_ISLNK(os.lstat(probe).st_mode):
                    hops += 1
                    target = os.readlink(probe)
                    parent_dir = os.path.dirname(probe) or "/"
                    expanded = target if target.startswith("/") else _join_uncollapsed(parent_dir, target)
                    probe = "/".join([expanded.rstrip("/"), *reversed(suffix)]) if suffix else expanded
                    suffix.clear()
                    continue
            except TimeoutError:
                raise
            except OSError:
                pass
            trimmed = probe.rstrip("/")
            if not trimmed:
                return os.path.normpath(os.path.join("/", *reversed(suffix))) if suffix else "/"
            parent, name = os.path.split(trimmed)
            if not name or parent == probe:
                return probe
            suffix.append(name)
            probe = parent or "/"
        except OSError:
            return probe


def is_wsl_drvfs(value: str | os.PathLike[str]) -> bool:
    if _is_windows_fs():
        return False
    try:
        resolved = _strict_realpath(os.fspath(value))
        with open(_MOUNTINFO_PATH, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
    except TimeoutError:
        raise
    except OSError:
        return False
    best_mp, best_fstype, best_opts = "", "", ""
    for line in lines:
        parts = line.split(" - ", 1)
        if len(parts) != 2:
            continue
        lhs, rhs = parts[0].split(), parts[1].split()
        if len(lhs) < 5 or len(rhs) < 3:
            continue
        mp = lhs[4].replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n").replace("\\134", "\\")
        if (resolved == mp or resolved.startswith(mp.rstrip("/") + "/") or mp == "/") and len(mp) >= len(best_mp):
            best_mp, best_fstype, best_opts = mp, rhs[0], rhs[2]
    return bool(best_mp) and best_fstype in {"drvfs", "9p"} and ("aname=drvfs" in best_opts or best_fstype == "drvfs")


def is_pe_executable(value: str | os.PathLike[str]) -> bool:
    try:
        with open(os.fspath(value), "rb") as handle:
            dos = handle.read(64)
            if len(dos) < 64 or dos[:2] != b"MZ":
                return False
            handle.seek(int.from_bytes(dos[0x3C:0x40], "little", signed=False))
            return handle.read(4) == b"PE\x00\x00"
    except TimeoutError:
        raise
    except OSError:
        return False


def _which_windows(command: str, *, path: str | None = None, pathext_override: str | None = None) -> str | None:
    if not command:
        return None
    pathext_raw = pathext_override or _env_get(os.environ, "PATHEXT") or _WIN_DEFAULT_PATHEXT
    pathext: list[str] = []
    for raw_ext in pathext_raw.split(";"):
        if not raw_ext:
            continue
        norm_ext = raw_ext if raw_ext.startswith(".") else f".{raw_ext}"
        for variant in (norm_ext.lower(), norm_ext):
            if variant not in pathext:
                pathext.append(variant)
    has_known_ext = any(command.lower().endswith(ext.lower()) for ext in pathext)
    if "\\" in command or "/" in command or (len(command) >= 2 and command[1] == ":"):
        direct_candidates = [command] if has_known_ext else [command, *(command + ext for ext in pathext)]
        for cand in direct_candidates:
            if _is_regular_file(cand):
                return cand if ("\\" in cand or "/" in cand) else os.path.abspath(cand)
        return None
    search_path = _env_get(os.environ, "PATH", None) if path is None else path
    if search_path is None:
        search_path = os.defpath
    if not search_path:
        return None
    sep = ";" if ";" in search_path else os.pathsep
    names = [command] if has_known_ext else [command + ext for ext in pathext]
    for raw_entry in search_path.split(sep):
        entry = raw_entry.strip().strip('"')
        if not entry:
            continue
        join_sep = "\\" if ("\\" in entry or (len(entry) >= 2 and entry[1] == ":")) else os.sep
        prefix = entry if entry in {"\\", "/"} else entry.rstrip("/\\")
        for name in names:
            cand = f"{prefix}{name}" if prefix in {"\\", "/"} else f"{prefix}{join_sep}{name}"
            if _is_regular_file(cand):
                return cand
    return None


def which_native(command: str, *, path: str | None = None) -> str | None:
    if not command:
        return None
    if _is_windows_fs():
        return _which_windows(command, path=path)
    if "/" in command:
        if _is_regular_file(command, require_exec=True) and not is_pe_executable(command) and not is_wsl_drvfs(command):
            return command
        return None
    if command.lower().endswith(_WIN_SCRIPT_EXTS):
        return None
    search_path = os.environ.get("PATH", None) if path is None else path
    if search_path is None:
        search_path = os.defpath
    if not search_path:
        return None
    for entry in search_path.split(os.pathsep):
        if not entry:
            continue
        if entry == ".":
            cand = _join_uncollapsed(os.getcwd(), command)
        elif entry.startswith("/"):
            cand = _join_uncollapsed(entry, command)
        else:
            cand = _join_uncollapsed(_join_uncollapsed(os.getcwd(), entry), command)
        if not _is_regular_file(cand, require_exec=True):
            continue
        if is_wsl_drvfs(cand) or is_pe_executable(cand):
            continue
        return cand
    return None


def _is_windows_wsl_bash_stub(candidate: str) -> bool:
    norm = candidate.replace("/", "\\").rstrip("\\").casefold()
    return (
        norm.endswith(("\\windows\\system32\\bash.exe", "\\windows\\sysnative\\bash.exe", "\\windows\\syswow64\\bash.exe"))
        or "\\microsoft\\windowsapps\\" in norm
    )


def _win_git_roots(env_map: Mapping[str, str]) -> list[str]:
    roots: list[str] = []
    seen: set[str] = set()
    for key, rel_parts in (
        ("LOCALAPPDATA", ("Programs", "Git")),
        ("ProgramFiles", ("Git",)),
        ("ProgramW6432", ("Git",)),
        ("ProgramFiles(x86)", ("Git",)),
        ("USERPROFILE", ("AppData", "Local", "Programs", "Git")),
        ("USERPROFILE", ("scoop", "apps", "git", "current")),
    ):
        base = _env_get(env_map, key)
        if base:
            sep_b = "/" if (base.startswith("/") and "\\" not in base) else "\\"
            trimmed = base.rstrip("/\\")
            cand = sep_b.join((trimmed, *rel_parts))
            folded = cand.casefold()
            if folded not in seen:
                seen.add(folded)
                roots.append(cand)
    return roots


def resolve_git(*, path: str | None = None, env: Mapping[str, str] | None = None) -> str:
    env_map = os.environ if env is None else env
    search_path = _env_get(env_map, "PATH", None) if path is None else path
    if not _is_windows_fs():
        found = which_native("git", path=search_path)
        if found:
            return found
        raise FileNotFoundError("Native POSIX git executable not found on PATH (Windows PE/DrvFS shims ignored)")
    effective_path = os.defpath if search_path is None else search_path
    found = _which_windows("git", path=effective_path, pathext_override=_env_get(env_map, "PATHEXT")) if effective_path else None
    if found:
        return found
    if path is None:
        for root in _win_git_roots(env_map):
            sep_r = "/" if (root.startswith("/") and "\\" not in root) else "\\"
            for sub_parts in (("cmd", "git.exe"), ("bin", "git.exe")):
                cand = sep_r.join((root, *sub_parts))
                if _is_regular_file(cand):
                    return cand
    raise FileNotFoundError("Git executable not found on Windows PATH or standard installations")


def resolve_bash(*, path: str | None = None, env: Mapping[str, str] | None = None) -> str:
    env_map = os.environ if env is None else env
    search_path = _env_get(env_map, "PATH", None) if path is None else path
    if not _is_windows_fs():
        found = which_native("bash", path=search_path)
        if found:
            return found
        raise FileNotFoundError("Native POSIX bash executable not found on PATH (Windows PE/DrvFS shims ignored)")
    raw_path = os.defpath if search_path is None else search_path
    sep = ";" if ";" in raw_path else os.pathsep
    for entry in raw_path.split(sep):
        clean = entry.strip().strip('"')
        if not clean:
            continue
        join_sep = "\\" if ("\\" in clean or (len(clean) >= 2 and clean[1] == ":")) else os.sep
        prefix = clean.rstrip("/\\")
        for name in ("bash.exe", "bash"):
            cand = f"{prefix}{join_sep}{name}"
            if not _is_windows_wsl_bash_stub(cand) and _is_regular_file(cand):
                return cand
    roots: list[str] = []
    try:
        git_exe = resolve_git(path=path, env=env_map)
        norm_git = git_exe.replace("/", "\\") if "\\" in git_exe else git_exe
        sep_git = "\\" if "\\" in norm_git else os.sep
        cur = norm_git.rsplit(sep_git, 1)[0] if sep_git in norm_git else ""
        for _ in range(4):
            if not cur:
                break
            parent = cur.rsplit(sep_git, 1)[0] if sep_git in cur else ""
            if not parent or parent == cur:
                break
            roots.append(parent)
            cur = parent
    except FileNotFoundError:
        pass
    if path is None:
        roots.extend(_win_git_roots(env_map))
    for root in roots:
        sep_root = "\\" if ("\\" in root or (len(root) >= 2 and root[1] == ":")) else os.sep
        trimmed_root = root.rstrip("/\\")
        for sub in (("bin", "bash.exe"), ("usr", "bin", "bash.exe")):
            cand = sep_root.join((trimmed_root, *sub))
            if not _is_windows_wsl_bash_stub(cand) and _is_regular_file(cand):
                return cand
    raise FileNotFoundError("Git Bash (bash.exe) not found; Windows WSL bash.exe stubs ignored")


def resolve_bash_or_none(*, path: str | None = None, env: Mapping[str, str] | None = None) -> str | None:
    try:
        return resolve_bash(path=path, env=env)
    except FileNotFoundError:
        return None
