# mindie-coordinator 0.7.14: Windows Persistent Daemon Win32 Job Breakaway and State-Path Hardening

[中文](coordinator-0.7.14.md)

`mindie-coordinator v0.7.14` keeps the local Windows coordinator daemon alive across per-call `mindie-harness` foreground CLI invocations, shortens temporary IPC staging filenames for deep Windows state directories, and advances dependency pins to `mindie-diagnostics v0.5.17` and `remote-dev v0.11.15`.

## Windows Persistent Daemon Win32 Job Breakaway & Path Hardening

- **Breakaway from Per-Call Foreground CLI Win32 Jobs**: On Windows, `mindie-harness` supervises each foreground CLI invocation inside a per-call Win32 Job Object configured with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_BREAKAWAY_OK`. Previously, spawning `mindie_coordinator daemon` on demand inherited that foreground Job and caused the persistent daemon to be terminated as soon as the foreground CLI command exited. `ensure_daemon` (`_ensure_daemon_locked` in `mindie_coordinator/service_client.py`) now spawns `mindie_coordinator daemon` on Windows (`os.name == "nt"`) with `CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB`, allowing the daemon to break out of the foreground CLI Job and remain alive across subsequent calls without console popups.
- **Clean Fallback Under Non-Breakaway Outer Host Jobs**: When an outer host or CI runner Job forbids breakaway and rejects `CreateProcess` with `ERROR_ACCESS_DENIED` (`winerror == 5`), `_ensure_daemon_locked` falls back once to `CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP`. `TimeoutError` is explicitly re-raised ahead of `OSError`, any `OSError` with `winerror != 5` propagates immediately without retry, and POSIX platforms keep `start_new_session=True` unchanged.
- **Compact IPC Staging Markers Near `MAX_PATH`**: `serve()` shortens pre-publish IPC staging filenames to `coordinator.ipc.<hex8>.tmp` prior to atomic `os.replace`, reducing path-length pressure inside deep Windows state directories.

## Dependency Pin Updates

- **`mindie-diagnostics`**: Updated to `v0.5.17` (`7464cbb4381e9dc996871d192eb3f05f61dfdb28`).
- **`remote-dev`**: Updated to `v0.11.15` (`784fc2ca1db172f683889a6d47543c37141b3278`).

## Verification

- **Cross-Platform Spawn & Error-Boundary Unit Coverage** (`tests/test_win_wsl_audit.py`): Verifies `CREATE_BREAKAWAY_FROM_JOB` flag composition, single-retry fallback strictly on `winerror == 5`, immediate propagation of `TimeoutError` and non-`winerror=5` `OSError`, and unchanged `start_new_session=True` on POSIX.
- **Native Windows Nested Win32 Job Integration Suite** (`tests/test_windows_service.py`): Exercises live nested `outer_job` + `foreground_job` hierarchies (`IsProcessInJob`, `TerminateJobObject`, and loopback IPC `ping`) confirming both daemon survival across foreground Job termination and clean fallback when breakaway is denied with zero leaked processes, alongside Windows extended-length path (`\\?\`) and unprivileged symlink test portability (`1,573 passed, 0 failed`).
