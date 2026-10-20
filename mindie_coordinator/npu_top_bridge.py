"""Observation-only bridge to the local `npu-top` agent API.

[Role]
Reads `npu-top` capacity observations and task-window hardware telemetry over
the loopback HTTP API, normalizes C2 capacity kwargs/fields, and ranks eligible
hosts. It never allocates, reserves, or leases a device.

[Invariants]
- `npu-top` observations rank eligible hosts only. The host-side NPU queue
  remains the sole allocation authority; `allocation_authority` is always false.
- A monitor outage or a telemetry gap is returned as explicit
  `status`/`measurement_limitations`; it never becomes an empty successful
  observation or masks an already successful remote execution.
- Explicit host requests are not rewritten by observation ranking.
- Capacity/telemetry queries use bounded HTTP timeouts and bypass environment
  proxies (`ProxyHandler({})`), matching the local-only npu-top client contract.

[Timeouts & Locks]
- Capacity read timeout defaults to 2.0s; telemetry to 3.0s. Both are clamped
  to `0.1..10.0s`; no unbounded network wait and no locks.

[Side-Effect Policy]
- Loopback HTTP GET requests only; no local or remote writes.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import ProxyHandler, Request, build_opener

DEFAULT_URL = "http://127.0.0.1:8788"
DEFAULT_CAPACITY_TIMEOUT = 2.0
DEFAULT_TELEMETRY_TIMEOUT = 3.0
MIN_TIMEOUT = 0.1
MAX_TIMEOUT = 10.0
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
CAPACITY_PATH = "/api/agent/capacity"
TELEMETRY_PATH = "/api/agent/telemetry-window"
CAPACITY_KWARGS = ("min_weight_mount_avail_gb", "prefer_same_package", "exclude_throttled")


class NpuTopUnavailable(RuntimeError):
    """The local observation API could not answer a request."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def observed_process_interference(
    occupancy: Mapping[str, Any], device: int, identities: Mapping[str, Any], *, marker_known: bool,
) -> bool | None:
    """Bind a complete NPU process table to the admitted job's process marker.

    This measures other processes on the selected NPU. It does not assert that
    the host's CPU, storage or network has no contention. Missing tables, PID
    reuse races and unattributed HBM keep the observation unknown.
    """
    if not marker_known or occupancy.get("status") != "ok" or device not in occupancy.get("devices", []):
        return None
    reasons = (occupancy.get("busy") or {}).get(str(device), [])
    processes = [identities.get(str(reason["pid"]), {}) for reason in reasons if reason.get("kind") == "process"]
    if any(item.get("identity_stable") is True and item.get("owned") is False for item in processes):
        return True
    if any(reason.get("kind") != "process" for reason in reasons):
        return None
    if all(item.get("identity_stable") is True and item.get("owned") is True for item in processes):
        return False
    return None


def _bounded_timeout(value: float | str | None, default: float) -> float:
    try:
        number = float(default if value is None else value)
    except (TypeError, ValueError):
        number = default
    return min(MAX_TIMEOUT, max(MIN_TIMEOUT, number))


def _environment_timeout(name: str, default: float) -> float:
    return _bounded_timeout(os.environ.get(name), default)


def _flag(value: bool) -> str:
    return "1" if value else "0"


class NpuTopObservationClient:
    """Small read-only client with explicit bounded/no-proxy behavior."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None, *, opener=None) -> None:
        self.base_url = (base_url or os.environ.get("MINDIE_TOP_URL") or DEFAULT_URL).rstrip("/")
        parsed = urlparse(self.base_url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("npu-top bridge requires an http URL")
        if (parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                and os.environ.get("MINDIE_TOP_ALLOW_REMOTE") != "1"):
            raise ValueError("non-loopback npu-top URL requires MINDIE_TOP_ALLOW_REMOTE=1")
        self.timeout = _bounded_timeout(timeout, _environment_timeout(
            "MINDIE_TOP_TIMEOUT_SECONDS", DEFAULT_CAPACITY_TIMEOUT))
        self.opener = opener or build_opener(ProxyHandler({}))

    def _get(self, path: str, query: Mapping[str, Any], *, timeout: float | None = None) -> dict:
        url = f"{self.base_url}{path}?{urlencode(query)}"
        request = Request(url, headers={"Accept": "application/json"}, method="GET")
        try:
            with self.opener.open(request, timeout=_bounded_timeout(timeout, self.timeout)) as response:
                body = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            message = None
            try:
                message = json.loads(exc.read()).get("error")
            except (json.JSONDecodeError, AttributeError):
                pass
            raise NpuTopUnavailable(message or f"npu-top API returned HTTP {exc.code}",
                                    status=exc.code) from exc
        except (OSError, URLError, json.JSONDecodeError, ValueError) as exc:
            raise NpuTopUnavailable(f"npu-top unavailable at {self.base_url}: {exc}") from exc
        if len(body) > MAX_RESPONSE_BYTES:
            raise NpuTopUnavailable("npu-top response exceeded the bounded observation size")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise NpuTopUnavailable("npu-top returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise NpuTopUnavailable("npu-top returned a non-object observation")
        return payload

    def capacity(self, *, min_idle_npus: int = 1, max_age_seconds: int = 300,
                 tags: list[str] | None = None, include_disabled: bool = False,
                 min_weight_mount_avail_gb: float = 0.0, prefer_same_package: bool = True,
                 exclude_throttled: bool = True, timeout: float | None = None) -> dict:
        """Read capacity candidates with the C2 selection kwargs.

        Older npu-top servers ignore the three new query parameters; callers
        still receive the original candidate list plus explicit gap notes.
        """
        if type(min_idle_npus) is not int or min_idle_npus < 0:
            raise ValueError("min_idle_npus must be a nonnegative integer")
        if type(max_age_seconds) is not int or max_age_seconds < 0:
            raise ValueError("max_age_seconds must be a nonnegative integer")
        if not isinstance(min_weight_mount_avail_gb, (int, float)) or min_weight_mount_avail_gb < 0:
            raise ValueError("min_weight_mount_avail_gb must be nonnegative")
        requirements = {
            "min_idle_npus": min_idle_npus,
            "max_age_seconds": max_age_seconds,
            "tags": list(tags or []),
            "include_disabled": bool(include_disabled),
            "min_weight_mount_avail_gb": float(min_weight_mount_avail_gb),
            "prefer_same_package": bool(prefer_same_package),
            "exclude_throttled": bool(exclude_throttled),
        }
        query = {
            "min_idle_npus": str(min_idle_npus),
            "max_age_seconds": str(max_age_seconds),
            "tags": ",".join(requirements["tags"]),
            "include_disabled": _flag(include_disabled),
            "min_weight_mount_avail_gb": str(float(min_weight_mount_avail_gb)),
            "prefer_same_package": _flag(prefer_same_package),
            "exclude_throttled": _flag(exclude_throttled),
        }
        try:
            payload = self._get(CAPACITY_PATH, query, timeout=timeout)
        except NpuTopUnavailable as exc:
            return unavailable_capacity(requirements, str(exc), status=exc.status)
        if payload.get("error"):
            return unavailable_capacity(requirements, str(payload["error"]))
        return rank_capacity_payload(payload, requirements)

    def telemetry_window(self, host: str, devices: list[int], start_ts: float, end_ts: float,
                         *, timeout: float | None = None) -> dict:
        """Read one explicit task measurement window; gaps stay visible."""
        if not isinstance(host, str) or not host.strip():
            raise ValueError("telemetry host must be a nonempty string")
        if (not isinstance(devices, list) or not devices
                or any(type(device) is not int or device < 0 for device in devices)):
            raise ValueError("telemetry devices must be distinct nonnegative integers")
        if len(set(devices)) != len(devices):
            raise ValueError("telemetry devices must be distinct nonnegative integers")
        if (any(type(value) not in (int, float) or not math.isfinite(value)
                for value in (start_ts, end_ts)) or start_ts < 0 or end_ts < start_ts):
            raise ValueError("telemetry window requires finite 0 <= start_ts <= end_ts")
        query = {"host": host.strip(), "devices": ",".join(str(device) for device in devices),
                 "start_ts": str(start_ts), "end_ts": str(end_ts)}
        try:
            payload = self._get(TELEMETRY_PATH, query, timeout=timeout or DEFAULT_TELEMETRY_TIMEOUT)
        except NpuTopUnavailable as exc:
            status = "unavailable"
            limitation = f"npu-top telemetry unavailable: {exc}"
            if exc.status == 404:
                limitation = "npu-top telemetry-window API is not installed (HTTP 404)"
            return telemetry_evidence({}, host=host, devices=devices, start_ts=start_ts,
                                      end_ts=end_ts, status=status,
                                      extra_limitations=[limitation], error=str(exc))
        if payload.get("error"):
            return telemetry_evidence({}, host=host, devices=devices, start_ts=start_ts,
                                      end_ts=end_ts, status="unavailable",
                                      extra_limitations=[f"npu-top telemetry unavailable: {payload['error']}"],
                                      error=str(payload["error"]))
        return telemetry_evidence(payload, host=host, devices=devices, start_ts=start_ts, end_ts=end_ts)


def unavailable_capacity(requirements: Mapping[str, Any], error: str, *, status: int | None = None) -> dict:
    """Visible monitor outage that cannot be mistaken for zero capacity."""
    limitation = f"npu-top monitor unavailable: {error}"
    return {
        "source": "npu-top",
        "status": "unavailable",
        "error": error,
        "http_status": status,
        "allocation_authority": False,
        "requirements": dict(requirements),
        "observation": None,
        "candidates": [],
        "eligible_candidates": [],
        "measurement_limitations": [limitation],
    }


def _candidate_reasons(candidate: dict, requirements: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    eligible = True
    thermal = candidate.get("ThermalHealth")
    if thermal is None:
        reasons.append("ThermalHealth not reported; throttling filter could not be applied")
    elif thermal == "throttled" and requirements.get("exclude_throttled", True):
        eligible = False
        reasons.append("ThermalHealth=throttled")
    elif thermal not in {"healthy", "throttled", "unknown"}:
        reasons.append("unrecognized ThermalHealth; throttling filter not applied")
    threshold = float(requirements.get("min_weight_mount_avail_gb") or 0.0)
    mount = candidate.get("MaxWeightMountAvailGb")
    if mount is None:
        if threshold > 0:
            eligible = False
            reasons.append("MaxWeightMountAvailGb not reported; required weight-mount space is unproven")
        else:
            reasons.append("MaxWeightMountAvailGb not reported; mount filter not requested")
    else:
        try:
            if float(mount) < threshold:
                eligible = False
                reasons.append(f"MaxWeightMountAvailGb={mount} < requested {threshold}")
        except (TypeError, ValueError):
            eligible = False
            reasons.append("MaxWeightMountAvailGb is not numeric")
    pairs = candidate.get("IdlePackagePairs")
    if pairs is None:
        reasons.append("IdlePackagePairs not reported; same-package preference unavailable")
    elif not isinstance(pairs, list) or any(not isinstance(pair, list) for pair in pairs):
        reasons.append("IdlePackagePairs has an unsupported shape")
    return eligible, reasons


def _pair_score(candidate: dict) -> int:
    pairs = candidate.get("IdlePackagePairs")
    if not isinstance(pairs, list):
        return 0
    return sum(1 for pair in pairs if isinstance(pair, list) and len(pair) >= 2)


def rank_capacity_payload(payload: Mapping[str, Any], requirements: Mapping[str, Any]) -> dict:
    """Rank candidates without allocating; explicit fields stay unchanged."""
    limitations: list[str] = []
    rows = payload.get("candidates")
    if not isinstance(rows, list):
        return unavailable_capacity(requirements, "npu-top capacity payload has no candidate list")
    ranked = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            limitations.append(f"candidate[{index}] is not an object; ignored for ranking")
            continue
        candidate = dict(raw)
        eligible, reasons = _candidate_reasons(candidate, requirements)
        candidate["npu_top_eligible"] = eligible
        candidate["npu_top_reasons"] = reasons
        candidate["npu_top_pair_score"] = _pair_score(candidate) if requirements.get("prefer_same_package") else 0
        ranked.append(candidate)
        limitations.extend(f"{candidate.get('host') or candidate.get('name')}: {reason}" for reason in reasons)
    ranked.sort(key=lambda item: (
        0 if item["npu_top_eligible"] else 1,
        0 if item["npu_top_pair_score"] else 1,
        -item["npu_top_pair_score"],
        -(int(item.get("idle_npu_count") or 0)),
        int(item.get("age_seconds") or 0),
        str(item.get("host") or item.get("name") or ""),
    ))
    for rank, candidate in enumerate(ranked):
        candidate["npu_top_rank"] = rank
    eligible = [candidate for candidate in ranked if candidate["npu_top_eligible"]]
    result_limitations = list(payload.get("measurement_limitations") or [])
    result_limitations.extend(limitations)
    return {
        "source": "npu-top",
        "status": "observed" if not result_limitations else "partial",
        "allocation_authority": False,
        "requirements": dict(requirements),
        "observation": payload.get("observation"),
        "candidates": ranked,
        "eligible_candidates": eligible,
        "measurement_limitations": list(dict.fromkeys(result_limitations)),
    }


def observation_host_order(observation: Mapping[str, Any] | None) -> dict[str, int]:
    """Map host/name identities to observation rank for stable sorting."""
    if not isinstance(observation, Mapping):
        return {}
    rows = observation.get("candidates")
    if not isinstance(rows, list):
        return {}
    order: dict[str, int] = {}
    for rank, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        value = row.get("npu_top_rank", rank)
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = rank
        for key in ("host", "name", "alias"):
            identity = str(row.get(key) or "").strip().casefold()
            if identity and identity not in order:
                order[identity] = value
    return order


def rank_capacity_candidates(payload: Mapping[str, Any], **requirements: Any) -> dict:
    """Convenience wrapper for already-fetched capacity payloads."""
    normalized = {
        "min_idle_npus": int(requirements.get("min_idle_npus", 1)),
        "max_age_seconds": int(requirements.get("max_age_seconds", 300)),
        "tags": list(requirements.get("tags") or []),
        "include_disabled": bool(requirements.get("include_disabled", False)),
        "min_weight_mount_avail_gb": float(requirements.get("min_weight_mount_avail_gb", 0.0)),
        "prefer_same_package": bool(requirements.get("prefer_same_package", True)),
        "exclude_throttled": bool(requirements.get("exclude_throttled", True)),
    }
    return rank_capacity_payload(payload, normalized)


def telemetry_evidence(payload: Mapping[str, Any], *, host: str, devices: list[int],
                       start_ts: float, end_ts: float, status: str | None = None,
                       extra_limitations: list[str] | None = None,
                       error: str | None = None) -> dict:
    """Build the C3 `hardware_telemetry` evidence object.

    Missing coverage or registers become `measurement_limitations` instead of
    being silently treated as healthy. `status` is `observed` only when the
    requested window is fully covered and every claimed signal is present.
    """
    limitations = list(extra_limitations or [])
    server = payload.get("server") or {}
    window = payload.get("window") or {}
    if server and host not in {server.get("host"), server.get("name")}:
        limitations.append("telemetry host does not match the requested host")
        status = "unavailable"
    if window and (window.get("start_ts") != start_ts or window.get("end_ts") != end_ts
                   or sorted(window.get("requested_devices") or []) != sorted(devices)):
        limitations.append("telemetry devices or window do not match the request")
        status = "unavailable"
    if payload and (not server or not window):
        limitations.append("telemetry response does not bind its host and device window")
    for field in ("coverage_ratio", "throttling_detected"):
        if payload.get(field) is None:
            limitations.append(f"{field} not reported for this window")
    try:
        coverage = float(payload.get("coverage_ratio")) if payload.get("coverage_ratio") is not None else 0.0
    except (TypeError, ValueError):
        coverage = 0.0
        limitations.append("coverage_ratio is not numeric")
    if payload.get("coverage_ratio") is not None and coverage < 1.0:
        limitations.append(f"telemetry window coverage is incomplete ({coverage:.3f})")
    for limitation in payload.get("measurement_limitations") or []:
        if isinstance(limitation, str) and limitation not in limitations:
            limitations.append(limitation)
    if status is None:
        status = "observed" if not limitations else "partial"
    result = {
        "source": "npu-top",
        "status": status,
        "host": host,
        "devices": list(devices),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "measured_at": int(payload.get("measured_at") or payload.get("observed_at") or time.time()),
        "coverage_ratio": coverage,
        "throttling_detected": payload.get("throttling_detected"),
        "resource_interference_detected": payload.get("resource_interference_detected"),
        "freq_mhz_min": payload.get("freq_mhz_min"),
        "power_max_w": payload.get("power_max_w"),
        "temp_max_c": payload.get("temp_max_c"),
        "measurement_limitations": list(dict.fromkeys(limitations)),
    }
    for field in ("aicore_p50", "aicore_p95", "hbm_peak_mb", "temp_max_c",
                  "power_max_w", "freq_mhz_min", "observation"):
        if payload.get(field) is not None:
            result[field] = payload[field]
    for field in ("start_ts_ns", "end_ts_ns"):
        if field in window:
            result[field] = window[field]
    per_device = payload.get("per_device")
    if per_device is None and isinstance(payload.get("devices"), list) and all(
            isinstance(row, dict) for row in payload["devices"]):
        per_device = payload["devices"]
    if per_device is not None:
        result["per_device"] = per_device
    if error is not None:
        result["error"] = error
    return result
