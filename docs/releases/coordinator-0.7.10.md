# mindie-coordinator 0.7.10

## 关键修复与跨平台增强

- **Windows 用户级 Git Bash 发现与 WSL 桩隔离（`local_exec.py`）**：新增 `resolve_git` / `resolve_bash`，自动识别 `%LOCALAPPDATA%\Programs\Git` 及由 `git.exe` 推导出的 Git Bash（`..\bin\bash.exe` / `..\usr\bin\bash.exe`），显式排除 `System32\bash.exe`、`Sysnative\bash.exe` 与 `WindowsApps\bash.exe` 等未安装发行版的 WSL 启动桩。
- **WSL2 原生工具链隔离与符号链接路径保真**：在 WSL2 下解析 `git` 与 `bash` 时保留显式相对路径与符号链接 `..` 语义，拒绝 `/mnt/c` DrvFS 挂载及 PE `.exe` 伪装二进制，并在 `execution_slots.py` 远端 `$PATH` 遍历中启用 `set -f` 防止括号/通配符路径展开异常。
- **Windows `CP936` UTF-8 管道与 `CREATE_NO_WINDOW` 加固**：所有本地子进程与源码/配置读写统一强制 UTF-8 编码（`utf8_python_env` 显式覆盖继承的 `CP936` 设置）与无窗口标志 `CREATE_NO_WINDOW`，安全归一化 `\\?\` 扩展前缀，并补齐 `npu_top_bridge` 等边界的 `except TimeoutError: raise` 状态诚实保证。
- **配套依赖升级**：同步升级至 `remote-dev v0.11.11` 与 `mindie-diagnostics v0.5.12`。
