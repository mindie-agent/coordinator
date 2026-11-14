"""Public support projection of an already-authorized execution's local facts.

[Role] Projects one execution's recorded timing, correlation, and outcome
events into a diagnostics bundle for reporting.

[Invariants] No global log scan, host observation, business output, source path,
or raw record is read. Missing historical clock/correlation facts remain
explicit gaps rather than inferred values.

[Timeouts & Locks] Pure in-memory projection; no I/O, subprocesses, or locks.

[Side-Effect Policy] Returns a new dictionary and never mutates the source row.
"""
from datetime import datetime, timezone
from pathlib import Path
import time

from mindie_diagnostics import collect_bundle, current_context

_SOURCE_ROOT = Path(__file__).resolve().parent
_EXACT_INTERNAL_TYPES = (AssertionError, KeyError, AttributeError, IndexError, ZeroDivisionError)


class _CoordinatorFailureMarker:
    __slots__ = ("result",)

    def __init__(self, result):
        self.result = result


def _iter_causal_chain(exc: BaseException):
    seen: set[int] = set()
    pending: list[BaseException | None] = [exc]
    while pending and len(seen) < 32:
        cur = pending.pop(0)
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        yield cur
        if getattr(cur, "__cause__", None) is not None:
            pending.append(cur.__cause__)
        if getattr(cur, "__context__", None) is not None and not getattr(cur, "__suppress_context__", False):
            pending.append(cur.__context__)


def is_internal_coordinator_exception(exc: BaseException) -> bool:
    """True only for genuine internal coordinator bugs, never caller/SSH/config errors."""
    seen: set[int] = set()
    pending: list[BaseException | None] = [exc]
    while pending and len(seen) < 32:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if getattr(current, "category", None) is not None or getattr(current, "_mindie_caller_error", False):
            return False
        if isinstance(current, (OSError, TimeoutError, ValueError, KeyboardInterrupt, SystemExit, GeneratorExit)):
            return False
        for cls in type(current).__mro__:
            if cls.__name__ in {
                "RemoteDevError",
                "CoordinationError",
                "ManagedTaskContextRequired",
                "MachineDirectoryUnavailable",
                "HostQueueUnavailable",
            }:
                return False
        if getattr(current, "__cause__", None) is not None:
            pending.append(current.__cause__)
        if getattr(current, "__context__", None) is not None and not getattr(current, "__suppress_context__", False):
            pending.append(current.__context__)
    if pending or type(exc) not in _EXACT_INTERNAL_TYPES:
        return False
    tb = exc.__traceback__
    while tb is not None:
        mod = str(tb.tb_frame.f_globals.get("__name__") or "")
        if mod == "mindie_coordinator" or mod.startswith("mindie_coordinator."):
            return True
        tb = tb.tb_next
    return False


def record_coordinator_failure(
    operation: str,
    *,
    stage: str = "tool_call",
    category: str = "internal_exception",
    exception: BaseException | None = None,
    reason_code: str | None = None,
    contract_name: str | None = None,
) -> dict | None:
    """Record a confirmed internal coordinator fault or local state contract violation."""
    if exception is not None:
        for cur in _iter_causal_chain(exception):
            marker = getattr(cur, "_mindie_coordinator_diag", None)
            if type(marker) is _CoordinatorFailureMarker and isinstance(marker.result, dict):
                try:
                    exception._mindie_coordinator_diag = marker
                except Exception:
                    pass
                return marker.result
            anchor = getattr(cur, "_mindie_failure_anchor", None)
            if anchor is not None:
                try:
                    from remote_dev.observability import _FailureAnchor
                    if type(anchor) is _FailureAnchor and isinstance(anchor.diagnostic, dict):
                        exception._mindie_coordinator_diag = _CoordinatorFailureMarker(anchor.diagnostic)
                        return anchor.diagnostic
                except Exception:
                    pass
    try:
        from mindie_diagnostics.integration import record_failure
        base_kwargs = {
            "component": "mindie-coordinator",
            "operation": operation,
            "stage": stage,
            "category": category,
            "exception": exception,
        }
        v2_kwargs: dict[str, object] = {"source_root": _SOURCE_ROOT}
        if reason_code is not None:
            v2_kwargs["reason_code"] = reason_code
        if contract_name is not None:
            v2_kwargs["contract_name"] = contract_name
        try:
            result = record_failure(**base_kwargs, **v2_kwargs)
        except TypeError:
            result = record_failure(**base_kwargs)
    except Exception:
        return None
    if exception is not None and isinstance(result, dict):
        try:
            exception._mindie_coordinator_diag = _CoordinatorFailureMarker(result)
        except Exception:
            pass
    return result if isinstance(result, dict) else None


def execution_bundle(row, reply, diagnostic_events, output):
    records = []
    gaps = set()
    context = row.get("diagnostics_context") or current_context()

    def add(kind, stamp, monotonic, process, attributes, *, correlation=None):
        correlation = correlation or context
        if not correlation.get("trace_id") or not correlation.get("operation_id"):
            gaps.add("historical_correlation_missing")
            return
        if not monotonic or not process:
            gaps.add("historical_clock_missing")
        records.append({"schema": 1, "component": "mindie-coordinator", "event": kind,
            "timestamp": datetime.fromtimestamp(stamp, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "monotonic_ns": monotonic or 0, "process_instance_id": process,
            "clock_domain_unknown": not bool(monotonic and process), "severity": "INFO",
            **correlation, "attributes": attributes})

    # This timestamp describes this read-only snapshot, never the execution's
    # earlier completion. Durations below come from their original clock domain.
    from mindie_coordinator.service import _PROCESS_CLOCK_DOMAIN
    add("evidence.snapshot", time.time(), time.monotonic_ns(), _PROCESS_CLOCK_DOMAIN,
        {"execution_id": row["id"], "status": reply.get("state"),
         "resources_released": reply.get("resources_released")})
    for item in diagnostic_events.get("operations", [])[:128]:
        elapsed = item.get("elapsed_seconds")
        add("pool." + item["kind"], item["at"], item.get("monotonic_ns"), item.get("process_instance_id"),
            {"stage": item.get("operation"), "job_id": item.get("job_id"),
             **({"duration_ms": elapsed * 1000} if isinstance(elapsed, (int, float)) else {})},
            correlation=item.get("diagnostics_context"))
    for item in row.get("stage_history", [])[-64:]:
        elapsed = item.get("elapsed_seconds")
        add("execution.phase", item["started_at"], int(item.get("started_monotonic", 0) * 1e9), item.get("clock_domain"),
            {"stage": item.get("step"), **({"duration_ms": elapsed * 1000} if isinstance(elapsed, (int, float)) else {})})
    # Passing selected events is deliberate: a shared daemon's log contains
    # other users and tasks. The collector must not discover additional files.
    result = collect_bundle(".", records=records, include_logs=False, output=output)
    if diagnostic_events.get("unavailable"):
        gaps.add("pool_events_unavailable")
    if diagnostic_events.get("truncated"):
        gaps.add("pool_events_truncated")
    # Never let compact result rendering truncate an attachment's event list
    # while leaving its original digest. Return the completed local artifact.
    return {key: result[key] for key in ("bundle_id", "content_sha256", "summary", "redaction")} | {
        "record_ref": str(output), "scope": "owned execution records only", "gaps": sorted(gaps)}
