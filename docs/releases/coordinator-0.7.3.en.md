# mindie-coordinator 0.7.3

[中文](coordinator-0.7.3.md)

## First-Class xLLM & MindIE-SD Framework Environment Preflight and Container Provisioning

- Add `mindie_coordinator/xllm_env.py` (`validate_xllm_runtime_env`, `probe_xllm_runtime_readiness`) and `mindie_coordinator/mindie_sd_env.py` (`validate_mindie_sd_runtime_env`, `probe_mindie_sd_runtime_readiness`), and integrate `xllm` and `mindie-sd` framework preflight across `env_manager.py` and the top-level package API.
- Support `mindie-sd` / `mindiesd` and `xllm` image family resolution, container provisioning, existing-container attachment, and host bootstrap templates in `provision/image_resolver.py`, `provision/host_probe_scripts.py`, `provision/__init__.py`, and `_host_bootstrap_template_{a,b}.py`, with updated bilingual configuration and tool documentation.
