# mindie-coordinator 0.7.4

[English](coordinator-0.7.4.en.md)

## Windows 零控制台弹窗加固与本地 Git / SSH 子进程窗口隐藏

- 为 `mindie_coordinator`（`state_paths.py`、`code_identity.py`、`git_sources.py`、`build_inputs.py`、`parity_support.py`、`parity_snapshot.py`、`service.py`、`agent_session.py`）与 `hooks/mindie_session.py` 中的本地 `git`、`ssh` 及会话探测子进程统一显式附加 `subprocess.CREATE_NO_WINDOW`，彻底消除 Windows GUI 宿主下的控制台窗口闪现。
