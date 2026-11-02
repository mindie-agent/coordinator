# mindie-coordinator 0.7.3

[English](coordinator-0.7.3.en.md)

## xLLM 与 MindIE-SD 一等公民框架运行时预检与容器镜像支持

- 新增 `mindie_coordinator/xllm_env.py`（`validate_xllm_runtime_env`、`probe_xllm_runtime_readiness`）与 `mindie_coordinator/mindie_sd_env.py`（`validate_mindie_sd_runtime_env`、`probe_mindie_sd_runtime_readiness`），并在 `env_manager.py` 及顶层包入口统一导出 `xllm` 与 `mindie-sd` 运行时环境预检能力。
- 在容器镜像解析与宿主机探测（`provision/image_resolver.py`、`provision/host_probe_scripts.py`、`provision/__init__.py`）中支持 `mindie-sd` / `mindiesd` 与 `xllm` 镜像族识别、容器池开通、现有容器绑定及启动模板（`_host_bootstrap_template_a.py` / `_host_bootstrap_template_b.py`）适配，并同步更新中英文配置与工具文档。
