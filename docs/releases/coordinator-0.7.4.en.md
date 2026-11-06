# mindie-coordinator 0.7.4

[中文](coordinator-0.7.4.md)

## Windows Zero Console Popup Hardening for Local Git & SSH Subprocesses

- Explicitly passes `subprocess.CREATE_NO_WINDOW` to local `git`, `ssh`, and session-probe subprocesses across `mindie_coordinator` (`state_paths.py`, `code_identity.py`, `git_sources.py`, `build_inputs.py`, `parity_support.py`, `parity_snapshot.py`, `service.py`, `agent_session.py`) and `hooks/mindie_session.py` so Windows GUI hosts never flash console windows.
