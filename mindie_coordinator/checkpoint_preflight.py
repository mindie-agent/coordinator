"""Zero-NPU Safetensors header & quantization contract preflight inspector.

[Role]
Reads only `config.json`, `model.safetensors.index.json`, and the 8-byte header
length + JSON metadata header of `.safetensors` shards on disk to verify weight
shard completeness, truncation, dtype breakdown, `quantization_config` vs.
routed-expert tensor dtype consistency, and Ascend 32B UB alignment before
acquiring NPU leases.

[Invariants]
- Zero NPU allocation, zero PyTorch/safetensors import, and zero payload read
  beyond each shard's UTF-8 JSON header (`8 + header_len` bytes).
- Deterministic classification of tensor roles (`routed_expert`, `shared_expert`,
  `attention`, `lm_head_or_embed`, `other`) and dtypes (`BF16`, `F16`, `F32`,
  `I8`, `F8_E4M3`, `I32`, `U8`).
- Flags `unquantized_routed_experts` when `config.json` declares W8A8/W4A8/int8/int4
  quantization while routed-expert projection tensors remain 16-bit (`BF16`/`F16`).

[Timeouts & Locks]
- Pure local/mounted filesystem header reads (`< 100ms` typical); no locks or subprocesses.

[Side-Effect Policy]
- Strictly read-only.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

_DTYPE_BYTES: dict[str, int] = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}
_MAX_HEADER_BYTES = 100 * 1024 * 1024


def _tensor_role(name: str) -> str:
    lowered = name.lower()
    if any(k in lowered for k in ("shared_expert", "shared_experts")):
        return "shared_expert"
    if any(k in lowered for k in ("experts.", "routed_expert", ".mlp.experts", "moe.experts", "w1", "w2", "w3")):
        return "routed_expert"
    if any(k in lowered for k in ("self_attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj", "kv_a_proj", "kv_b_proj", "q_a_proj", "q_b_proj")):
        return "attention"
    if any(k in lowered for k in ("embed_tokens", "lm_head", "wte", "word_embeddings")):
        return "lm_head_or_embed"
    return "other"


def _read_safetensors_header(shard_path: Path) -> tuple[dict[str, Any], int, int]:
    """Return `(header_dict, header_byte_len, file_size)` for one `.safetensors` file."""
    file_size = shard_path.stat().st_size
    if file_size < 8:
        raise ValueError(f"truncated safetensors shard (< 8 bytes): {shard_path.name}")
    with shard_path.open("rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        if header_len <= 0 or header_len > _MAX_HEADER_BYTES or 8 + header_len > file_size:
            raise ValueError(f"invalid or truncated safetensors header length ({header_len}) in {shard_path.name}")
        raw_header = fh.read(header_len)
    header = json.loads(raw_header.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not a JSON object in {shard_path.name}")
    return header, header_len, file_size


def inspect_safetensors_checkpoint(
    model_dir: Path | str,
    *,
    tp_size: int = 1,
    ep_size: int = 1,
) -> dict[str, Any]:
    """Inspect a HuggingFace / msmodelslim `.safetensors` directory without loading weights."""
    root = Path(model_dir)
    issues: list[dict[str, str]] = []
    if not root.is_dir():
        return {
            "valid": False, "model_dir": str(root),
            "issues": [{"code": "missing_model_dir", "message": f"Model directory does not exist: {root}"}],
        }

    config: dict[str, Any] = {}
    config_path = root / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append({"code": "invalid_config_json", "message": f"Failed to parse config.json: {exc}"})

    quant_cfg = config.get("quantization_config") if isinstance(config.get("quantization_config"), dict) else {}
    quant_desc_path = root / "quant_model_description.json"
    declared_quant = str(
        quant_cfg.get("quant_method") or quant_cfg.get("format") or config.get("quantization")
        or ("ascend_quant" if quant_desc_path.is_file() else "")
    ).strip().lower()
    is_declared_quant = bool(declared_quant and declared_quant not in {"none", "null", "false"})

    # Check Ascend 32B UB alignment on head_dim
    text_cfg = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    head_dim = text_cfg.get("head_dim") or text_cfg.get("qk_head_dim") or text_cfg.get("v_head_dim")
    if head_dim is None and text_cfg.get("hidden_size") and text_cfg.get("num_attention_heads"):
        try:
            head_dim = int(text_cfg["hidden_size"]) // int(text_cfg["num_attention_heads"])
        except (TypeError, ValueError, ZeroDivisionError):
            head_dim = None
    head_dim_int = int(head_dim) if isinstance(head_dim, int) and head_dim > 0 else None
    if head_dim_int is not None and (head_dim_int * 2) % 32 != 0:
        padded_head_dim = int(math.ceil((head_dim_int * 2) / 32.0) * 16)
        issues.append({
            "code": "unaligned_head_dim_32b",
            "message": f"head_dim={head_dim_int} (BF16/FP16 {head_dim_int * 2}B) is not 32-byte UB aligned; pad inner dimension to {padded_head_dim} before FIA/CustomOp launch and slice back to [:, :, :{head_dim_int}].",
        })

    index_path = root / "model.safetensors.index.json"
    expected_shards: list[str] = []
    if index_path.is_file():
        try:
            idx_doc = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = idx_doc.get("weight_map") if isinstance(idx_doc, dict) else {}
            if isinstance(weight_map, dict):
                expected_shards = sorted({str(v) for v in weight_map.values() if v})
        except (OSError, json.JSONDecodeError) as exc:
            issues.append({"code": "invalid_safetensors_index", "message": f"Failed to parse model.safetensors.index.json: {exc}"})

    shard_files = [root / name for name in expected_shards] if expected_shards else sorted(root.glob("*.safetensors"))
    missing_shards = [p.name for p in shard_files if not p.is_file()]
    if missing_shards:
        issues.append({"code": "missing_safetensors_shards", "message": f"Missing {len(missing_shards)} shard(s): {', '.join(missing_shards[:5])}"})

    dtype_bytes: dict[str, int] = {}
    role_bytes: dict[str, int] = {"routed_expert": 0, "shared_expert": 0, "attention": 0, "lm_head_or_embed": 0, "other": 0}
    routed_expert_dtype_bytes: dict[str, int] = {}
    tensor_count = 0
    inspected_shards = 0

    for shard in shard_files:
        if not shard.is_file():
            continue
        try:
            header, header_len, file_size = _read_safetensors_header(shard)
        except ValueError as exc:
            issues.append({"code": "truncated_or_corrupt_shard", "message": str(exc)})
            continue
        inspected_shards += 1
        max_end_offset = 0
        for tname, meta in header.items():
            if tname == "__metadata__" or not isinstance(meta, dict):
                continue
            tensor_count += 1
            dtype = str(meta.get("dtype") or "UNKNOWN").upper()
            offsets = meta.get("data_offsets")
            if isinstance(offsets, list) and len(offsets) == 2:
                start_off, end_off = int(offsets[0]), int(offsets[1])
                nbytes = max(0, end_off - start_off)
                if end_off > max_end_offset:
                    max_end_offset = end_off
            else:
                shape = meta.get("shape") or []
                elem_cnt = math.prod(int(d) for d in shape) if shape else 0
                nbytes = elem_cnt * _DTYPE_BYTES.get(dtype, 2)
            dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + nbytes
            role = _tensor_role(tname)
            role_bytes[role] = role_bytes.get(role, 0) + nbytes
            if role == "routed_expert" and not tname.endswith(("scale", "offset", "zero_point")):
                routed_expert_dtype_bytes[dtype] = routed_expert_dtype_bytes.get(dtype, 0) + nbytes

        expected_file_size = 8 + header_len + max_end_offset
        if max_end_offset > 0 and file_size < expected_file_size:
            issues.append({
                "code": "truncated_or_corrupt_shard",
                "message": f"Shard {shard.name} is truncated: actual {file_size} bytes < required {expected_file_size} bytes.",
            })

    total_bytes = sum(dtype_bytes.values())
    bf16_fp16_expert_bytes = routed_expert_dtype_bytes.get("BF16", 0) + routed_expert_dtype_bytes.get("F16", 0)
    int_expert_bytes = sum(v for k, v in routed_expert_dtype_bytes.items() if k in {"I8", "U8", "F8_E4M3", "F8_E5M2", "I32"})
    if is_declared_quant and bf16_fp16_expert_bytes > 0 and bf16_fp16_expert_bytes >= int_expert_bytes:
        extra_gib = (bf16_fp16_expert_bytes * 0.5) / (1024 ** 3)
        issues.append({
            "code": "unquantized_routed_experts_in_quant_checkpoint",
            "message": (
                f"config.json declares quantization ({declared_quant}), but routed-expert weights contain "
                f"{bf16_fp16_expert_bytes / (1024 ** 3):.2f} GiB in BF16/F16 (~{extra_gib:.2f} GiB unquantized bloat across the fleet)."
            ),
        })

    tp = max(1, int(tp_size or 1))
    ep = max(1, int(ep_size or 1))
    non_routed_bytes = total_bytes - role_bytes["routed_expert"]
    est_rank_weight_gib = round(((non_routed_bytes / tp) + (role_bytes["routed_expert"] / max(tp, ep))) / (1024 ** 3), 3)

    return {
        "valid": len(issues) == 0 and inspected_shards > 0,
        "model_dir": str(root),
        "declared_quantization": declared_quant or None,
        "head_dim": head_dim_int,
        "head_dim_32b_aligned": True if head_dim_int is None else ((head_dim_int * 2) % 32 == 0),
        "inspected_shards": inspected_shards,
        "missing_shards": missing_shards,
        "tensor_count": tensor_count,
        "total_weight_bytes": total_bytes,
        "total_weight_gib": round(total_bytes / (1024 ** 3), 3),
        "dtype_bytes": dtype_bytes,
        "role_bytes": role_bytes,
        "routed_expert_dtype_bytes": routed_expert_dtype_bytes,
        "estimated_weight_gib_per_rank": est_rank_weight_gib,
        "issues": issues,
        "issue_codes": [item["code"] for item in issues],
    }
