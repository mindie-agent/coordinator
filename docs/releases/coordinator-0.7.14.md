# mindie-coordinator 0.7.14：Windows 常驻守护进程 Win32 Job 脱离机制与长路径状态目录加固

[English](coordinator-0.7.14.en.md)

`mindie-coordinator v0.7.14` 修复了 Windows 环境下常驻协调器守护进程随单次前台 CLI 调用退出而被连带终止的问题，缩短深层状态目录下的 IPC 临时标记文件名，并同步升级 `mindie-diagnostics v0.5.17` 与 `remote-dev v0.11.15`。

## Windows 常驻守护进程脱离前台 Win32 Job 与长路径加固

- **支持脱离单次前台 CLI 的 Win32 Job Object**：在 Windows 上，`mindie-harness` 会将每次前台 CLI 调用放入带有 `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_BREAKAWAY_OK` 的临时 Win32 Job Object 中监管。此前首次按需拉起常驻守护进程时未申请脱离该 Job，导致前台 CLI 命令一结束、关闭 Job 句柄时就把后台常驻的 `mindie_coordinator daemon` 一并杀死。现在 `ensure_daemon`（`mindie_coordinator/service_client.py` 中的 `_ensure_daemon_locked`）在 Windows（`os.name == "nt"`）下默认以 `CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB` 启动 `mindie_coordinator daemon`，使常驻守护进程安全脱离单次前台 CLI Job，在后续命令与会话间持续存活且全程无弹窗。
- **外层宿主禁止脱离时的受控回退**：当外层宿主或 CI Runner 的顶层 Job 禁止子进程脱离、系统返回 `ERROR_ACCESS_DENIED`（`winerror == 5`）时，`_ensure_daemon_locked` 会干净回退至 `CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP` 完成启动；同时严格保持在 `OSError` 捕获前显式重抛 `TimeoutError`，其余非 `winerror == 5` 的启动错误原样透传、绝不盲目重试。POSIX 平台保持 `start_new_session=True` 不变。
- **精简 IPC 原子发布临时文件名**：`serve()` 将监听就绪前的临时标记文件名缩短为 `coordinator.ipc.<hex8>.tmp` 后再执行原子 `os.replace`，降低接近 Windows `MAX_PATH` 限制的深层状态目录下的路径长度开销。

## 配套依赖升级

- **`mindie-diagnostics`**：升级锁定至 `v0.5.17`（`7464cbb4381e9dc996871d192eb3f05f61dfdb28`）。
- **`remote-dev`**：升级锁定至 `v0.11.15`（`784fc2ca1db172f683889a6d47543c37141b3278`）。

## 验证覆盖

- **跨平台启动标志与错误边界单测**（`tests/test_win_wsl_audit.py`）：覆盖默认请求 `CREATE_BREAKAWAY_FROM_JOB`、`winerror == 5` 单次回退、`TimeoutError` 优先重抛、非 `5` 号 `OSError` 立即报错及 POSIX `start_new_session=True` 行为。
- **Windows 原生嵌套 Win32 Job 实机集成测试**（`tests/test_windows_service.py`）：构建 `outer_job` + `foreground_job` 双层真实 Win32 Job，实测验证允许脱离时前台 `TerminateJobObject` 后常驻守护进程继续存活并响应本地回环 IPC `ping`，以及禁止脱离时干净回退且外层测试结束零残留进程；同步加固 Windows 超长算子构建路径（`\\?\`）与无符号链接权限环境下的测试可移植性，全量隔离测试套件 `1,573 passed, 0 failed`。
