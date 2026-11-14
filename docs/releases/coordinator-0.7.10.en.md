# mindie-coordinator 0.7.10

## Fixes & Cross-Platform Hardening

- **Windows User-Scoped Git Bash Discovery & WSL Stub Isolation (`local_exec.py`)**: Add `resolve_git` and `resolve_bash` to discover user-scoped `%LOCALAPPDATA%\Programs\Git` and `git.exe`-relative Git Bash (`..\bin\bash.exe` / `..\usr\bin\bash.exe`) while rejecting `System32\bash.exe`, `Sysnative\bash.exe`, and `WindowsApps\bash.exe` WSL launcher stubs.
- **WSL2 Native Binary Isolation & Symlink Path Fidelity**: Preserve explicit relative paths and symlink-plus-`..` semantics before checking `/proc/self/mountinfo` DrvFS mounts and PE headers when resolving `git` and `bash`, and guard `$PATH` splitting in `execution_slots.py` with `set -f`.
- **Windows `CP936` UTF-8 Pipes, `CREATE_NO_WINDOW` & `TimeoutError` Honesty**: Enforce explicit UTF-8 file/subprocess encoding (`utf8_python_env` overriding inherited `CP936` settings), `CREATE_NO_WINDOW` across local spawns, safe `\\?\` extended-path stripping, and `except TimeoutError: raise` before `except OSError:` in `npu_top_bridge` and local execution helpers.
- **Companion Dependency Alignment**: Upgrade pinned dependencies to `remote-dev v0.11.11` and `mindie-diagnostics v0.5.12`.
