"""Measured-profile wiring for the incremental kernel content cache."""
from __future__ import annotations

import pytest

from mindie_coordinator import native_incremental as native
from mindie_coordinator import native_kernel_recipe as recipe
from mindie_coordinator.parity import runtime_install_step_script
from test_native_incremental import kernel as _kernel_fixture

kernel = _kernel_fixture


def test_incremental_plan_carries_only_measured_profile_identity(kernel):
    root, bundle, source, installed, entries, manifest, preparation = kernel
    manifest["profile"] = {
        "cann": "8.2.RC1",
        "soc": "Ascend910B",
        "python_abi": "cp312",
    }
    plan = native.kernel_rebuild_plan(root, bundle, manifest, preparation, entries)
    assert plan["cann_version"] == "8.2.RC1"
    assert plan["soc_version"] == "Ascend910B"
    assert plan["python_abi"] == "cp312"
    assert native.incremental_content_metadata(plan) == {
        "cann_version": "8.2.RC1",
        "soc_version": "Ascend910B",
        "python_abi": "cp312",
    }


def test_missing_measured_profile_metadata_never_becomes_cache_reuse(kernel):
    root, bundle, source, installed, entries, manifest, preparation = kernel
    plan = native.kernel_rebuild_plan(root, bundle, manifest, preparation, entries)
    assert "cann_version" not in plan
    assert native.incremental_content_metadata(plan) is None

    manifest["profile"] = {"cann": "8.2.RC1", "soc": "Ascend910B"}
    partial = native.kernel_rebuild_plan(root, bundle, manifest, preparation, entries)
    assert partial["cann_version"] == "8.2.RC1"
    assert "python_abi" not in partial
    assert native.incremental_content_metadata(partial) is None


def test_parity_incremental_script_wires_content_cache_and_measured_plan_fields():
    script = runtime_install_step_script(
        runtime_root="/vllm-workspace/executions/example",
        marker_dirname=".mindie-runtime",
        container_identity="mindie-example",
        step="install-vllm-ascend-incremental",
        python="/vllm-workspace/executions/example/.venv/bin/python",
    )
    assert 'content_metadata=(metadata := incremental_content_metadata(plan))' in script
    assert ('content_cache=(Path("/tmp/mindie-native-cache/kernel") if metadata else None)'
            in script)
    assert "read_recipe=compiled_opc_recipe" in script


def test_compile_skips_cache_when_measured_metadata_is_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(recipe, "_recipe_inputs", lambda *_args: ([], "dynamic.py", "ini"))
    monkeypatch.setattr(recipe, "kernel_content_key",
                        lambda *_args, **_kwargs: pytest.fail("must not guess cache metadata"))
    monkeypatch.setattr(recipe.tempfile, "mkdtemp",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            RuntimeError("compiler path reached")))

    with pytest.raises(RuntimeError, match="compiler path reached"):
        recipe.compile_kernel_recipe(
            tmp_path, {"operator": "fixture", "unit": "ascend910b"}, {},
            read_recipe=lambda _path: None,
            content_cache=tmp_path / "cache",
            content_metadata=None,
        )


def test_cache_lookup_failure_does_not_block_the_compiler(tmp_path, monkeypatch):
    monkeypatch.setattr(recipe, "_recipe_inputs", lambda *_args: ([], "dynamic.py", "ini"))
    monkeypatch.setattr(recipe, "kernel_content_key", lambda *_args, **_kwargs: "a" * 64)
    monkeypatch.setattr(recipe, "restore_kernel_recipe_content",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            ValueError("corrupt optional entry")))
    monkeypatch.setattr(recipe.tempfile, "mkdtemp",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            RuntimeError("compiler path reached")))

    with pytest.raises(RuntimeError, match="compiler path reached"):
        recipe.compile_kernel_recipe(
            tmp_path, {"operator": "fixture", "unit": "ascend910b"}, {},
            read_recipe=lambda _path: None,
            content_cache=tmp_path / "cache",
            content_metadata={"cann_version": "8.2.RC1", "soc_version": "Ascend910B",
                              "python_abi": "cp312"},
        )
