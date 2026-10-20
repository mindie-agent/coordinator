"""Static AST and binary schema preflight for Ascend NPU custom operators.

[Role]
Extracts ground-truth operator schemas (`npu::*`, `_C_ascend::*`) and validates
Python source files in sub-second AST preflight before NPU lease allocation or
model launch.

[Invariants]
- Stdlib-only so the module can run both locally on coordinator source snapshots
  and remotely inside container preflight probes without external dependencies.
- Never executes user code or initializes NPU devices during static AST checks.
- Every violation reports exact file path, line number, operator name, violated
  contract (`unknown_ascend_op`, `return_arity_mismatch`,
  `inplace_noncontiguous_or_copy_arg`, `unmigrated_cuda_call`), and a concrete
  remediation hint.

[Timeouts & Locks]
- Pure in-memory AST analysis (`< 50ms` per file); zero subprocesses or locks.

[Side-Effect Policy]
- `check_python_op_contracts` and `check_paths_op_contracts` are pure read-only
  functions. `write_runtime_op_catalog` writes `.mindie-runtime/ascend-ops-catalog.json`
  atomically via `.tmp` + `os.replace`.
"""
from __future__ import annotations

import ast
import difflib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

CATALOG_RELATIVE_PATH = ".mindie-runtime/ascend-ops-catalog.json"

# Ground-truth contracts for high-frequency Ascend operators in torch_npu and
# vllm_ascend._C_ascend.
# - return_arity: 1 for single Tensor, >=2 for tuple return, 0 for None/in-place.
# - in_place_args: list of [positional_index, keyword_name] with Tensor(a!) semantics.
DEFAULT_ASCEND_OP_CATALOG: dict[str, dict[str, Any]] = {
    # torch_npu / npu:: namespace
    "torch_npu.npu_rms_norm": {
        "namespace": "torch_npu",
        "op": "npu_rms_norm",
        "return_arity": 2,
        "returns": "(Tensor y, Tensor rstd)",
        "in_place_args": [],
        "hint": (
            "torch_npu.npu_rms_norm(x, gamma, epsilon) returns a 2-tuple (y, rstd); "
            "unpack `y, _ = torch_npu.npu_rms_norm(...)` or index `[0]`, or use "
            "`torch.ops._C_ascend.rms_norm(x, gamma, epsilon)` which returns a single Tensor."
        ),
    },
    "torch_npu.npu_add_rms_norm": {
        "namespace": "torch_npu",
        "op": "npu_add_rms_norm",
        "return_arity": 3,
        "returns": "(Tensor y, Tensor rstd, Tensor x_out)",
        "in_place_args": [],
        "hint": (
            "torch_npu.npu_add_rms_norm(x1, x2, gamma, epsilon) returns a 3-tuple "
            "(y, rstd, x_out); unpack `y, _, residual = torch_npu.npu_add_rms_norm(...)`."
        ),
    },
    "torch_npu.npu_swiglu": {
        "namespace": "torch_npu",
        "op": "npu_swiglu",
        "return_arity": 1,
        "returns": "Tensor",
        "in_place_args": [],
        "hint": "torch_npu.npu_swiglu(x, dim=-1) returns a single Tensor.",
    },
    "torch_npu.npu_rotary_mul": {
        "namespace": "torch_npu",
        "op": "npu_rotary_mul",
        "return_arity": 1,
        "returns": "Tensor",
        "in_place_args": [],
        "hint": "torch_npu.npu_rotary_mul(x, r1, r2) returns a single Tensor.",
    },
    "torch_npu.npu_recurrent_gated_delta_rule": {
        "namespace": "torch_npu",
        "op": "npu_recurrent_gated_delta_rule",
        "return_arity": 1,
        "returns": "Tensor",
        "in_place_args": [[4, "initial_state"], [4, "state"]],
        "hint": (
            "torch_npu.npu_recurrent_gated_delta_rule mutates `initial_state` (`Tensor(a!)`) "
            "in-place and requires a contiguous tensor buffer; never pass `.transpose(...)`, "
            "`.permute(...)`, or a temporary `.to(...)`/`.clone()` copy directly as `initial_state`."
        ),
    },
    "torch_npu.npu_chunk_gated_delta_rule": {
        "namespace": "torch_npu",
        "op": "npu_chunk_gated_delta_rule",
        "return_arity": 2,
        "returns": "(Tensor out, Tensor final_state)",
        "in_place_args": [],
        "hint": "torch_npu.npu_chunk_gated_delta_rule returns a 2-tuple (out, final_state).",
    },
    "torch_npu.npu_causal_conv1d_update": {
        "namespace": "torch_npu",
        "op": "npu_causal_conv1d_update",
        "return_arity": 1,
        "returns": "Tensor",
        "in_place_args": [[1, "conv_state"]],
        "hint": (
            "torch_npu.npu_causal_conv1d_update mutates `conv_state` (`Tensor(a!)`) in-place; "
            "pass a contiguous persistent state buffer rather than a non-contiguous `.transpose()` view."
        ),
    },
    "torch_npu.npu_scatter_nd_update_": {
        "namespace": "torch_npu",
        "op": "npu_scatter_nd_update_",
        "return_arity": 1,
        "returns": "Tensor(a!)",
        "in_place_args": [[0, "var"], [0, "self"]],
        "hint": (
            "torch_npu.npu_scatter_nd_update_ mutates its first argument (`var`, `Tensor(a!)`) "
            "in-place; pass a contiguous tensor rather than a non-contiguous `.transpose()`/`.permute()` view."
        ),
    },
    "torch_npu.npu_grouped_matmul": {
        "namespace": "torch_npu",
        "op": "npu_grouped_matmul",
        "return_arity": 1,
        "returns": "Tensor[]",
        "in_place_args": [],
        "hint": "torch_npu.npu_grouped_matmul returns a list of Tensors.",
    },
    "torch_npu.npu_quant_matmul": {
        "namespace": "torch_npu",
        "op": "npu_quant_matmul",
        "return_arity": 1,
        "returns": "Tensor",
        "in_place_args": [],
        "hint": "torch_npu.npu_quant_matmul returns a single Tensor.",
    },
    "torch_npu.npu_DynamicQuant": {
        "namespace": "torch_npu",
        "op": "npu_DynamicQuant",
        "return_arity": 2,
        "returns": "(Tensor y, Tensor scale)",
        "in_place_args": [],
        "hint": "torch_npu.npu_dynamic_quant returns (y, scale).",
    },
    "torch_npu.npu_dynamic_quant": {
        "namespace": "torch_npu",
        "op": "npu_dynamic_quant",
        "return_arity": 2,
        "returns": "(Tensor y, Tensor scale)",
        "in_place_args": [],
        "hint": "torch_npu.npu_dynamic_quant(x) returns a 2-tuple (y, scale).",
    },
    # _C_ascend custom op namespace
    "torch.ops._C_ascend.rms_norm": {
        "namespace": "torch.ops._C_ascend",
        "op": "rms_norm",
        "return_arity": 1,
        "returns": "Tensor",
        "in_place_args": [],
        "hint": (
            "torch.ops._C_ascend.rms_norm(x, weight, eps) returns a single Tensor "
            "(unlike torch_npu.npu_rms_norm which returns a 2-tuple (y, rstd))."
        ),
    },
    "torch.ops._C_ascend.fused_add_rms_norm": {
        "namespace": "torch.ops._C_ascend",
        "op": "fused_add_rms_norm",
        "return_arity": 2,
        "returns": "(Tensor x, Tensor residual)",
        "in_place_args": [[0, "input"], [1, "residual"]],
        "hint": "torch.ops._C_ascend.fused_add_rms_norm mutates input/residual and returns (x, residual).",
    },
    "torch.ops._C_ascend.fused_rmsnorm_gated": {
        "namespace": "torch.ops._C_ascend",
        "op": "fused_rmsnorm_gated",
        "return_arity": 1,
        "returns": "Tensor",
        "in_place_args": [],
        "hint": "torch.ops._C_ascend.fused_rmsnorm_gated returns a single Tensor.",
    },
    "torch.ops._C_ascend.weak_ref_tensor": {
        "namespace": "torch.ops._C_ascend",
        "op": "weak_ref_tensor",
        "return_arity": 1,
        "returns": "Tensor",
        "in_place_args": [],
        "hint": "torch.ops._C_ascend.weak_ref_tensor(tensor) returns a single Tensor.",
    },
    "torch.ops._C_ascend.reshape_and_cache": {
        "namespace": "torch.ops._C_ascend",
        "op": "reshape_and_cache",
        "return_arity": 0,
        "returns": "None",
        "in_place_args": [[2, "key_cache"], [3, "value_cache"]],
        "hint": "torch.ops._C_ascend.reshape_and_cache mutates key_cache/value_cache in-place and returns None.",
    },
}

# Mirror torch_npu.<op> entries onto torch.ops.npu.<op>
for _key, _spec in list(DEFAULT_ASCEND_OP_CATALOG.items()):
    if _key.startswith("torch_npu."):
        _op = _spec["op"]
        DEFAULT_ASCEND_OP_CATALOG.setdefault(
            f"torch.ops.npu.{_op}",
            {**_spec, "namespace": "torch.ops.npu"},
        )

# Known hallucinated / mistyped operator names frequently produced by agents
KNOWN_INVALID_ASCEND_OPS: dict[str, str] = {
    "torch_npu.rms_norm": "torch_npu.npu_rms_norm",
    "torch_npu.add_rms_norm": "torch_npu.npu_add_rms_norm",
    "torch_npu.npu_fused_rms_norm": "torch_npu.npu_rms_norm",
    "torch_npu.npu_rmsnorm": "torch_npu.npu_rms_norm",
    "torch_npu.npu_fused_rmsnorm_gated": "torch.ops._C_ascend.fused_rmsnorm_gated",
    "torch_npu.fused_rmsnorm_gated": "torch.ops._C_ascend.fused_rmsnorm_gated",
    "torch_npu.npu_gdn_recurrent": "torch_npu.npu_recurrent_gated_delta_rule",
    "torch_npu.recurrent_gated_delta_rule": "torch_npu.npu_recurrent_gated_delta_rule",
    "torch.ops._C_ascend.npu_rms_norm": "torch.ops._C_ascend.rms_norm",
    "torch.ops._C_ascend.rmsnorm": "torch.ops._C_ascend.rms_norm",
    "torch.ops._C_ascend.npu_fused_rmsnorm_gated": "torch.ops._C_ascend.fused_rmsnorm_gated",
    "torch.ops._C_ascend.fused_rms_norm_gated": "torch.ops._C_ascend.fused_rmsnorm_gated",
    "torch.ops.npu.rms_norm": "torch.ops.npu.npu_rms_norm",
    "torch.ops.npu.fused_rmsnorm_gated": "torch.ops._C_ascend.fused_rmsnorm_gated",
}

_TENSOR_ONLY_ATTRIBUTES = frozenset({
    "view", "reshape", "contiguous", "to", "type_as", "half", "float", "bfloat16",
    "int", "long", "unsqueeze", "squeeze", "transpose", "permute", "expand",
    "repeat", "clone", "detach", "cpu", "npu", "cuda", "copy_", "add_", "mul_",
    "masked_fill", "masked_fill_", "stride", "size", "dim", "ndim", "shape",
    "dtype", "device", "numel", "is_contiguous", "data_ptr", "flatten", "split",
    "chunk", "narrow", "index_select", "scatter_", "gather", "matmul", "mm", "bmm",
})

_NONCONTIGUOUS_VIEW_METHODS = frozenset({"transpose", "permute", "swapaxes", "swapdims", "t"})
_TEMP_COPY_METHODS = frozenset({"clone", "to", "contiguous", "cpu", "float", "half", "bfloat16", "detach"})
_UNMIGRATED_CUDA_APIS = frozenset({
    "synchronize", "current_stream", "Stream", "Event", "set_device", "get_device_name",
    "get_device_capability", "mem_get_info", "empty_cache", "memory_allocated",
    "max_memory_allocated", "reset_peak_memory_stats", "ipc_collect",
})


def extract_live_ascend_op_catalog() -> dict[str, dict[str, Any]]:
    """Extract live JIT operator schemas from torch/torch_npu/_C_ascend when loaded."""
    catalog = {k: dict(v) for k, v in DEFAULT_ASCEND_OP_CATALOG.items()}
    try:
        import torch  # type: ignore[import-not-found]
        schemas = torch._C._jit_get_all_schemas()
    except Exception:
        return catalog
    for schema in schemas:
        name = getattr(schema, "name", "")
        if not name.startswith(("npu::", "_C_ascend::")):
            continue
        ns_raw, _, op_name = name.partition("::")
        if not op_name or op_name.startswith("_"):
            continue
        returns = list(getattr(schema, "returns", ()) or ())
        arity = len(returns)
        ret_desc = ", ".join(str(getattr(r, "type", "Tensor")) for r in returns)
        if arity > 1:
            ret_desc = f"({ret_desc})"
        elif arity == 0:
            ret_desc = "None"
        in_place: list[list[Any]] = []
        for idx, arg in enumerate(getattr(schema, "arguments", ()) or ()):
            alias_info = getattr(arg, "alias_info", None)
            if alias_info is not None and getattr(alias_info, "is_write", False):
                in_place.append([idx, getattr(arg, "name", f"arg{idx}")])
        prefixes = ("torch_npu", "torch.ops.npu") if ns_raw == "npu" else ("torch.ops._C_ascend",)
        for prefix in prefixes:
            full_name = f"{prefix}.{op_name}"
            existing = catalog.get(full_name, {})
            catalog[full_name] = {
                "namespace": prefix,
                "op": op_name,
                "return_arity": existing.get("return_arity", arity),
                "returns": existing.get("returns") or ret_desc,
                "in_place_args": existing.get("in_place_args") or in_place,
                "hint": existing.get("hint") or f"{full_name} returns {ret_desc}.",
            }
    return catalog


def write_runtime_op_catalog(root: Path) -> Path:
    """Write `.mindie-runtime/ascend-ops-catalog.json` atomically."""
    target = Path(root).resolve() / CATALOG_RELATIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    catalog = extract_live_ascend_op_catalog()
    payload = {"schema_version": 1, "operators": catalog}
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return target


def load_runtime_op_catalog(root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load `.mindie-runtime/ascend-ops-catalog.json` merged with default contracts."""
    catalog = {k: dict(v) for k, v in DEFAULT_ASCEND_OP_CATALOG.items()}
    if root is None:
        return catalog
    candidate = Path(root) / CATALOG_RELATIVE_PATH
    if not candidate.is_file():
        return catalog
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
        ops = data.get("operators") if isinstance(data, dict) else None
        if isinstance(ops, dict):
            for k, v in ops.items():
                if isinstance(v, dict):
                    catalog[k] = {**catalog.get(k, {}), **v}
    except (OSError, ValueError):
        pass
    return catalog


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    curr = node
    while isinstance(curr, ast.Attribute):
        parts.append(curr.attr)
        curr = curr.value
    if isinstance(curr, ast.Name):
        parts.append(curr.id)
        return ".".join(reversed(parts))
    return None


def _classify_arg_view_or_copy(
    expr: ast.AST,
    local_views: Mapping[str, str],
) -> tuple[str | None, str | None]:
    """Return `(issue_kind, detail)` if `expr` is a non-contiguous view or temporary copy."""
    if isinstance(expr, ast.Name) and expr.id in local_views:
        return "noncontiguous_view", f"variable `{expr.id}` holds non-contiguous view `{local_views[expr.id]}`"
    if isinstance(expr, ast.Attribute) and expr.attr in {"T", "mT"}:
        src = _dotted_name(expr) or f"<expr>.{expr.attr}"
        return "noncontiguous_view", f"`{src}` is a non-contiguous transpose view"
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        method = expr.func.attr
        if method in _NONCONTIGUOUS_VIEW_METHODS:
            src = ast.unparse(expr) if hasattr(ast, "unparse") else f"<tensor>.{method}(...)"
            return "noncontiguous_view", f"`{src}` produces a non-contiguous view"
        if method in _TEMP_COPY_METHODS:
            src = ast.unparse(expr) if hasattr(ast, "unparse") else f"<tensor>.{method}(...)"
            return "temporary_copy", f"`{src}` passes a temporary tensor copy whose in-place mutation is discarded"
    return None, None


def _is_noncontiguous_expr(expr: ast.AST) -> str | None:
    if isinstance(expr, ast.Attribute) and expr.attr in {"T", "mT"}:
        return ast.unparse(expr) if hasattr(ast, "unparse") else f".{expr.attr}"
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        if expr.func.attr in _NONCONTIGUOUS_VIEW_METHODS:
            return ast.unparse(expr) if hasattr(ast, "unparse") else f".{expr.func.attr}(...)"
    return None


def check_python_op_contracts(
    source_text: str,
    filename: str = "<string>",
    *,
    catalog: Mapping[str, dict[str, Any]] | None = None,
    check_unmigrated_cuda: bool = True,
) -> list[dict[str, Any]]:
    """Run static AST operator contract checks on Python source code."""
    ops_catalog = catalog if catalog is not None else DEFAULT_ASCEND_OP_CATALOG
    try:
        tree = ast.parse(source_text, filename=filename)
    except SyntaxError as exc:
        return [{
            "file": filename,
            "line": exc.lineno or 1,
            "rule": "syntax_error",
            "operator": None,
            "message": f"SyntaxError: {exc.msg}",
            "hint": "Fix the Python syntax error before launching on NPU.",
        }]

    violations: list[dict[str, Any]] = []
    known_ascend_ops_by_ns: dict[str, set[str]] = {}
    for full_key, spec in ops_catalog.items():
        ns = spec.get("namespace") or full_key.rpartition(".")[0]
        op = spec.get("op") or full_key.rpartition(".")[2]
        known_ascend_ops_by_ns.setdefault(ns, set()).add(op)

    def resolve_op_call(call_node: ast.Call) -> tuple[str | None, dict[str, Any] | None]:
        dotted = _dotted_name(call_node.func)
        if not dotted:
            return None, None
        # Normalize vllm_ascend_C.<op> -> torch.ops._C_ascend.<op>
        if dotted.startswith("vllm_ascend.vllm_ascend_C.") or dotted.startswith("vllm_ascend_C."):
            op_name = dotted.rpartition(".")[2]
            dotted = f"torch.ops._C_ascend.{op_name}"
        if dotted in KNOWN_INVALID_ASCEND_OPS:
            replacement = KNOWN_INVALID_ASCEND_OPS[dotted]
            repl_spec = ops_catalog.get(replacement, {})
            violations.append({
                "file": filename,
                "line": getattr(call_node, "lineno", 1),
                "rule": "unknown_ascend_op",
                "operator": dotted,
                "message": f"`{dotted}` does not exist in the Ascend operator library.",
                "hint": f"Use `{replacement}` instead. {repl_spec.get('hint', '')}".strip(),
            })
            return dotted, None
        if dotted in ops_catalog:
            return dotted, ops_catalog[dotted]
        for ns_prefix in ("torch.ops._C_ascend.", "torch.ops.npu."):
            if dotted.startswith(ns_prefix):
                ns = ns_prefix[:-1]
                op_name = dotted[len(ns_prefix):]
                if "." not in op_name and ns in known_ascend_ops_by_ns:
                    candidates = difflib.get_close_matches(
                        op_name, sorted(known_ascend_ops_by_ns[ns]), n=2, cutoff=0.55
                    )
                    if candidates:
                        suggestion = ", ".join(f"`{ns}.{c}`" for c in candidates)
                        violations.append({
                            "file": filename,
                            "line": getattr(call_node, "lineno", 1),
                            "rule": "unknown_ascend_op",
                            "operator": dotted,
                            "message": f"`{dotted}` is not a registered `{ns}` operator.",
                            "hint": f"Did you mean {suggestion}?",
                        })
                return dotted, None
        return None, None

    def check_scope(stmts: list[ast.stmt]) -> None:
        tuple_vars: dict[str, tuple[int, str, dict[str, Any]]] = {}
        local_views: dict[str, str] = {}

        for stmt in stmts:
            # Recurse into nested control flow blocks while keeping function scopes distinct
            for child_block_attr in ("body", "orelse", "finalbody"):
                if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    block = getattr(stmt, child_block_attr, None)
                    if isinstance(block, list):
                        check_scope(block)
            if isinstance(stmt, ast.Try):
                for handler in stmt.handlers:
                    check_scope(handler.body)
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                check_scope(stmt.body)
                continue

            # Track local variable assignments
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Name):
                    view_desc = _is_noncontiguous_expr(stmt.value)
                    if view_desc:
                        local_views[target.id] = view_desc
                    else:
                        local_views.pop(target.id, None)

                if isinstance(stmt.value, ast.Call):
                    op_name, spec = resolve_op_call(stmt.value)
                    if op_name and spec:
                        arity = int(spec.get("return_arity", 1))
                        if isinstance(target, (ast.Tuple, ast.List)):
                            has_star = any(isinstance(elt, ast.Starred) for elt in target.elts)
                            if arity == 1 and len(target.elts) > 1:
                                violations.append({
                                    "file": filename,
                                    "line": stmt.lineno,
                                    "rule": "return_arity_mismatch",
                                    "operator": op_name,
                                    "message": (
                                        f"`{op_name}` returns a single {spec.get('returns', 'Tensor')}, "
                                        f"not a {len(target.elts)}-tuple."
                                    ),
                                    "hint": spec.get("hint", f"Assign `{op_name}(...)` to a single variable."),
                                })
                            elif arity >= 2 and not has_star and len(target.elts) != arity:
                                violations.append({
                                    "file": filename,
                                    "line": stmt.lineno,
                                    "rule": "return_arity_mismatch",
                                    "operator": op_name,
                                    "message": (
                                        f"`{op_name}` returns a {arity}-tuple {spec.get('returns', '')}, "
                                        f"but is unpacked into {len(target.elts)} targets."
                                    ),
                                    "hint": spec.get("hint", f"Unpack all {arity} elements or index `[0]`."),
                                })
                        elif isinstance(target, ast.Name):
                            if arity >= 2:
                                tuple_vars[target.id] = (stmt.lineno, op_name, spec)
                            else:
                                tuple_vars.pop(target.id, None)
                elif isinstance(target, ast.Name):
                    tuple_vars.pop(target.id, None)

            # Inspect all calls and expressions within this statement (not inside nested defs/classes)
            for node in ast.walk(stmt):
                if node is not stmt and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    continue

                if isinstance(node, ast.Call):
                    dotted = _dotted_name(node.func)
                    if check_unmigrated_cuda and dotted and dotted.startswith("torch.cuda."):
                        cuda_api = dotted.split(".", 2)[2]
                        if cuda_api in _UNMIGRATED_CUDA_APIS:
                            violations.append({
                                "file": filename,
                                "line": getattr(node, "lineno", 1),
                                "rule": "unmigrated_cuda_call",
                                "operator": dotted,
                                "message": f"Unmigrated `{dotted}` call in Ascend code path.",
                                "hint": f"Replace `{dotted}` with `torch.npu.{cuda_api}` or `torch_npu.npu.{cuda_api}`.",
                            })

                    # Avoid duplicate resolve_op_call on top-level Assign value already checked
                    if isinstance(stmt, ast.Assign) and node is stmt.value and len(stmt.targets) == 1:
                        op_name = _dotted_name(node.func)
                        if op_name and (op_name.startswith("vllm_ascend.vllm_ascend_C.") or op_name.startswith("vllm_ascend_C.")):
                            op_name = f"torch.ops._C_ascend.{op_name.rpartition('.')[2]}"
                        spec = ops_catalog.get(op_name or "")
                    else:
                        op_name, spec = resolve_op_call(node)

                    if op_name and spec:
                        # Check in-place arguments for non-contiguous views or temporary copies
                        for pos_idx, kw_name in spec.get("in_place_args") or ():
                            arg_expr: ast.AST | None = None
                            for kw in node.keywords:
                                if kw.arg == kw_name:
                                    arg_expr = kw.value
                                    break
                            if arg_expr is None and isinstance(pos_idx, int) and pos_idx < len(node.args):
                                arg_expr = node.args[pos_idx]
                            if arg_expr is not None:
                                issue, detail = _classify_arg_view_or_copy(arg_expr, local_views)
                                if issue is not None:
                                    violations.append({
                                        "file": filename,
                                        "line": getattr(arg_expr, "lineno", getattr(node, "lineno", 1)),
                                        "rule": "inplace_noncontiguous_or_copy_arg",
                                        "operator": op_name,
                                        "message": (
                                            f"In-place argument `{kw_name}` (`Tensor(a!)`) of `{op_name}` "
                                            f"received {detail}."
                                        ),
                                        "hint": spec.get("hint", f"Pass a contiguous persistent Tensor buffer to `{kw_name}`."),
                                    })

                # Direct attribute access on a tuple-returning op call: e.g. torch_npu.npu_rms_norm(...).view(...)
                if isinstance(node, ast.Attribute) and node.attr in _TENSOR_ONLY_ATTRIBUTES:
                    if isinstance(node.value, ast.Call):
                        op_name = _dotted_name(node.value.func)
                        spec = ops_catalog.get(op_name or "")
                        if op_name and spec and int(spec.get("return_arity", 1)) >= 2:
                            violations.append({
                                "file": filename,
                                "line": getattr(node, "lineno", 1),
                                "rule": "return_arity_mismatch",
                                "operator": op_name,
                                "message": (
                                    f"`{op_name}` returns a {spec['return_arity']}-tuple {spec.get('returns', '')}, "
                                    f"so `.{node.attr}` fails with AttributeError on tuple."
                                ),
                                "hint": spec.get("hint", f"Index `{op_name}(...)[0].{node.attr}` or unpack the tuple first."),
                            })
                    elif isinstance(node.value, ast.Name) and node.value.id in tuple_vars:
                        _, op_name, spec = tuple_vars[node.value.id]
                        violations.append({
                            "file": filename,
                            "line": getattr(node, "lineno", 1),
                            "rule": "return_arity_mismatch",
                            "operator": op_name,
                            "message": (
                                f"`{node.value.id}` holds the {spec['return_arity']}-tuple result of `{op_name}` "
                                f"{spec.get('returns', '')}, so `{node.value.id}.{node.attr}` fails on tuple."
                            ),
                            "hint": spec.get("hint", f"Unpack `{node.value.id}, _ = {op_name}(...)` or index `{node.value.id}[0]`."),
                        })

                # Binary arithmetic on a variable holding a tuple-returning op result
                if isinstance(node, ast.BinOp) and not isinstance(node.op, ast.Add):
                    for side in (node.left, node.right):
                        if isinstance(side, ast.Name) and side.id in tuple_vars:
                            _, op_name, spec = tuple_vars[side.id]
                            violations.append({
                                "file": filename,
                                "line": getattr(node, "lineno", 1),
                                "rule": "return_arity_mismatch",
                                "operator": op_name,
                                "message": (
                                    f"`{side.id}` holds the {spec['return_arity']}-tuple result of `{op_name}` "
                                    f"and cannot be used directly in tensor arithmetic."
                                ),
                                "hint": spec.get("hint", f"Unpack `{side.id}, _ = {op_name}(...)` or index `{side.id}[0]`."),
                            })

    check_scope(tree.body)
    # Deduplicate while preserving source order
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for item in violations:
        key = (item["file"], item["line"], item["rule"], item["operator"], item["message"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def check_paths_op_contracts(
    root: Path,
    relpaths: Iterable[str],
    *,
    catalog: Mapping[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Validate a list of relative `.py` file paths under `root`."""
    resolved_root = Path(root).resolve()
    ops_catalog = catalog if catalog is not None else load_runtime_op_catalog(resolved_root)
    violations: list[dict[str, Any]] = []
    for rel in relpaths:
        if not isinstance(rel, str) or not rel.endswith(".py"):
            continue
        candidate = (resolved_root / rel).resolve()
        if not candidate.is_relative_to(resolved_root) or not candidate.is_file():
            continue
        # Skip test files when checking unmigrated CUDA calls (tests may mock torch.cuda)
        is_test = any(part.startswith("test") or part == "tests" for part in Path(rel).parts)
        try:
            text = candidate.read_text(encoding="utf-8")
        except OSError:
            continue
        violations.extend(
            check_python_op_contracts(
                text,
                rel,
                catalog=ops_catalog,
                check_unmigrated_cuda=not is_test,
            )
        )
    return violations


def check_snapshot_op_contracts(
    source_snapshot: Mapping[str, Any] | None,
    *,
    catalog: Mapping[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Validate modified `.py` files recorded in a coordinator `source_snapshot`."""
    if not isinstance(source_snapshot, Mapping):
        return []
    violations: list[dict[str, Any]] = []
    for record in source_snapshot.get("records") or ():
        if not isinstance(record, Mapping):
            continue
        source_path = record.get("source_path")
        changed_paths = record.get("changed_paths") or ()
        if not source_path or not changed_paths:
            continue
        repo_root = Path(str(source_path))
        if not repo_root.is_dir():
            continue
        rel_prefix = str(record.get("relpath") or "")
        for item in check_paths_op_contracts(repo_root, changed_paths, catalog=catalog):
            qualified_file = (
                f"{rel_prefix}/{item['file']}"
                if rel_prefix and rel_prefix not in {".", ""}
                else item["file"]
            )
            violations.append({**item, "file": qualified_file})
    return violations


def format_op_violations(violations: Iterable[Mapping[str, Any]]) -> str:
    """Format AST operator contract violations into a concise diagnostic report."""
    items = list(violations)
    if not items:
        return ""
    lines = [f"Ascend operator contract preflight failed ({len(items)} violation(s)):"]
    for idx, v in enumerate(items[:10], start=1):
        lines.append(
            f"  [{idx}] {v.get('file')}:{v.get('line')} [{v.get('rule')}] {v.get('message')} -> {v.get('hint')}"
        )
    if len(items) > 10:
        lines.append(f"  ... and {len(items) - 10} more violation(s).")
    return "\n".join(lines)
