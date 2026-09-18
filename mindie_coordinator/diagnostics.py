"""Public support projection of an already-authorized execution's local facts.

No global log scan, host observation, business output, source path or raw record
is needed. Missing historical clock/correlation facts remain explicit gaps.
"""
from datetime import datetime, timezone
import time

from mindie_diagnostics import collect_bundle, current_context


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
