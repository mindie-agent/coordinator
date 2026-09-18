"""Board/SoC and npu-smi HBM parsing owned by the coordinator.

Occupancy HBM parsing is the host protocol's ``parse_npu_smi_hbm``. Board/SoC
token construction is the same labelled-field contract the machine probe uses.
npu-top is a separate display service and is not imported here.
"""
from __future__ import annotations

import re

SOC_TO_MACHINE_TYPE = {
    "910b": "A2",
    "910c": "A3",
    "310p": "310P",
    "ascend910b1": "A2",
    "ascend910b2": "A2",
    "ascend910b2c": "A2",
    "ascend910b3": "A2",
    "ascend910b4": "A2",
    "ascend910b4-1": "A2",
    "ascend910_9391": "A3",
    "ascend910_9381": "A3",
    "ascend910_9372": "A3",
    "ascend910_9392": "A3",
    "ascend910_9382": "A3",
    "ascend910_9362": "A3",
    "ascend310p1": "310P",
    "ascend310p3": "310P",
    "ascend310p5": "310P",
    "ascend310p7": "310P",
    "ascend310p3vir01": "310P",
    "ascend310p3vir02": "310P",
    "ascend310p3vir04": "310P",
    "ascend310p3vir08": "310P",
}
SOC_MATCH_ORDER = sorted(SOC_TO_MACHINE_TYPE, key=len, reverse=True)
ASCEND_950_SOC_PATTERN = re.compile(r"\bascend950[a-z0-9_-]*", re.IGNORECASE)
BARE_ASCEND_910_PATTERN = re.compile(r"\bascend910\b", re.IGNORECASE)

__all__ = [
    "SOC_TO_MACHINE_TYPE",
    "canonical_soc_from_board_text",
    "detect_from_text",
    "detect_machine_type_from_text",
    "first_npu_chip_ids",
    "machine_type_from_soc",
    "normalize_soc_token",
    "npu_smi_value",
    "parse_npu_smi_hbm",
]


def normalize_soc_token(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    return normalized or None


def npu_smi_value(text: str | None, field: str) -> str | None:
    """Return one labelled value from npu-smi's colon-delimited output."""
    if not text:
        return None
    pattern = re.compile(
        rf"^\s*{re.escape(field)}\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE
    )
    match = pattern.search(text)
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def canonical_soc_from_board_text(text: str | None) -> str | None:
    """Build the full SoC token from a detailed ``npu-smi -t board`` result.

    A2/310P report the architecture prefix in ``Chip Type``. A3 and A5 report
    the distinguishing part in ``NPU Name``.
    """
    chip_name = npu_smi_value(text, "Chip Name")
    if not chip_name:
        return None
    chip_type = npu_smi_value(text, "Chip Type")
    npu_name = npu_smi_value(text, "NPU Name")
    normalized_chip_name = re.sub(r"\s+", "", chip_name)
    normalized_chip_type = re.sub(r"\s+", "", chip_type or "")
    normalized_npu_name = re.sub(r"\s+", "", npu_name or "")
    lowered_chip_name = normalized_chip_name.lower()

    if "310" in lowered_chip_name and normalized_chip_type:
        return normalize_soc_token(normalized_chip_type + normalized_chip_name)
    if "910" in lowered_chip_name:
        if normalized_chip_type:
            return normalize_soc_token(normalized_chip_type + normalized_chip_name)
        if normalized_npu_name:
            return normalize_soc_token(f"{normalized_chip_name}_{normalized_npu_name}")
    if "950" in lowered_chip_name and normalized_npu_name:
        return normalize_soc_token(f"{normalized_chip_name}_{normalized_npu_name}")
    return normalize_soc_token(normalized_chip_name)


def machine_type_from_soc(soc: str | None) -> str | None:
    normalized = normalize_soc_token(soc)
    if normalized is None:
        return None
    if normalized.startswith("ascend950"):
        return "A5"
    return SOC_TO_MACHINE_TYPE.get(normalized)


def detect_machine_type_from_text(text: str | None) -> tuple[str | None, str | None]:
    if not text:
        return None, None
    normalized = text.lower()
    ascend_950_match = ASCEND_950_SOC_PATTERN.search(normalized)
    if ascend_950_match is not None:
        return ascend_950_match.group(0).lower(), "A5"
    for token in SOC_MATCH_ORDER:
        if token in normalized:
            return token, SOC_TO_MACHINE_TYPE[token]
    if BARE_ASCEND_910_PATTERN.search(normalized) is not None:
        return "ascend910", "A3"
    return None, None


detect_from_text = detect_machine_type_from_text


def first_npu_chip_ids(text: str | None) -> tuple[int | None, int | None]:
    if not text:
        return None, None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        if any(part.lower().startswith("ascend") for part in parts[2:]):
            return int(parts[0]), int(parts[1])
    return None, None


def parse_npu_smi_hbm(output: str) -> dict[int, dict[str, int]]:
    from mindie_coordinator.host.mindie_npu_coordination import parse_npu_smi_hbm as _parse_npu_smi_hbm

    return _parse_npu_smi_hbm(output)
