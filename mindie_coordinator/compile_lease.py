"""CPU preparation admission through the existing host queue.

[Role] Bind a remote-dev preparation supervisor to a zero-NPU host lease.
[Invariants] Queue state remains host-owned; a live compiler never releases capacity.
[Timeouts & Locks] Admission is bounded by queue_seconds; active leases renew every 30s.
[Side-Effect Policy] Persist the task identity before submit; lost writes are observed,
never replayed. Lease cleanup errors remain separate from the compiler outcome.
"""
from __future__ import annotations

import time


TERMINAL = {"released", "cancelled", "expired", "not_started"}
COMPILE_STEPS = {"install-vllm", "install-vllm-ascend", "install-vllm-ascend-incremental"}


def _clean_compile_slot(record, cleanup=None):
    if not record.get("slot_id") or (record.get("slot_cleanup") or {}).get("state") == "completed":
        return
    try:
        if cleanup is None:
            from mindie_coordinator.backend import RemoteBackend
            RemoteBackend().cleanup_slot({"endpoint": record["endpoint"]}, record["slot_id"])
        else:
            cleanup(record["endpoint"], record["slot_id"])
        record["slot_cleanup"] = {"state": "completed"}
    except Exception as exc:
        record["slot_cleanup"] = {"state": "failed", "error": str(exc)[:500]}


def finish_compile_lease(record, save, *, request=None, cleanup=None):
    """Reconcile and release the same CPU lease only after confirmed quiet."""
    lease = record.get("compile_lease")
    if not lease:
        return True
    if not record.get("quiet"):
        return False
    if lease.get("state") in TERMINAL:
        _clean_compile_slot(record, cleanup)
        save(record)
        return True
    if request is None:
        from mindie_coordinator.host_queue import HostQueue
        request = HostQueue().request
    try:
        snapshot = request(lease["host_endpoint"], {"action": "status", "no_probe": True,
            **({"coordination_epoch": lease["coordination_epoch"]} if lease.get("coordination_epoch") else {})})
        if lease.get("coordination_epoch") and snapshot["coordination_epoch"] != lease["coordination_epoch"]:
            raise RuntimeError("compile lease epoch changed; original ownership requires reconciliation")
        task = next((item for item in snapshot["tasks"] if item["task_id"] == lease["task_id"]), None)
        if task is None:
            if lease.get("state") not in {None, "pending"}:
                raise RuntimeError("previously acknowledged compile lease disappeared")
            lease["state"] = "not_started"
        elif task["state"] in TERMINAL:
            lease["state"] = task["state"]
        else:
            reply = request(lease["host_endpoint"], {
                "action": "cancel" if task["state"] in {"queued", "granted", "starting"} else "release",
                "task_id": lease["task_id"], "coordination_epoch": snapshot["coordination_epoch"],
                "fence_token": task.get("fence_token"), "completion_confirmed": True,
            })
            lease["state"] = reply["task"]["state"]
        lease.pop("cleanup_error", None)
    except Exception as exc:
        lease["cleanup_error"] = str(exc)[:500]
    if lease.get("state") in TERMINAL:
        _clean_compile_slot(record, cleanup)
    save(record)
    return lease.get("state") in TERMINAL


class CompileLease:
    def __init__(self, host_endpoint, container_name, request, *, queue_seconds=1800, cleanup=None, per_worker_memory_mb=2048):
        self.host_endpoint = dict(host_endpoint)
        self.container_name = container_name
        self.request = request
        self.queue_seconds = queue_seconds
        self.cleanup = cleanup
        self.renewed_at = 0.0
        self.per_worker_memory_mb = per_worker_memory_mb

    def acquire(self, record, save, cancelled):
        from mindie_coordinator.preparation_process import PreparationCancelled, PreparationUncertain
        lease = record["compile_lease"] = {
            "task_id": record["job_id"], "host_endpoint": self.host_endpoint,
            "container_name": self.container_name, "state": "pending",
        }
        save(record)
        try:
            startup = self.request(self.host_endpoint, {"action": "startup-context",
                "task_id": lease["task_id"], "container_name": self.container_name})
            lease.update(coordination_epoch=startup["coordination_epoch"], container_id=startup["container"]["Id"])
            save(record)
            deadline = time.monotonic() + self.queue_seconds
            reply = self.request(self.host_endpoint, {
                "action": "submit-acquire-preflight", "task_id": lease["task_id"],
                "coordination_epoch": lease["coordination_epoch"], "agent_id": "compiler-" + record["job_id"],
                "container_name": self.container_name, "npu_count": 0,
                "per_worker_memory_mb": self.per_worker_memory_mb,
                "queue_ttl_seconds": self.queue_seconds, "estimated_duration_seconds": 7200,
            })
            while True:
                task = reply["task"]
                lease.update(state=task["state"], fence_token=task.get("fence_token"))
                save(record)
                if cancelled() or time.monotonic() >= deadline:
                    # The compiler was never dispatched. Its queue reservation
                    # can be cancelled without manufacturing a process receipt.
                    record.update(state="cancelled", quiet=True)
                    finish_compile_lease(record, save, request=self.request, cleanup=self.cleanup)
                    raise PreparationCancelled("cancelled while waiting for CPU compilation capacity")
                if task["state"] == "starting":
                    lease["compile_budget"] = task.get("compile_budget") or {
                        "status": "unavailable", "reason": "host did not provide a per-slot compile budget"}
                    save(record)
                    return
                if task["state"] in TERMINAL:
                    record.update(state="cancelled", quiet=True)
                    save(record)
                    raise PreparationCancelled("compile lease ended before its supervisor started")
                if task["state"] == "granted":
                    reply = self.request(self.host_endpoint, {"action": "preflight",
                        "task_id": lease["task_id"], "coordination_epoch": lease["coordination_epoch"],
                        "fence_token": lease["fence_token"]})
                else:
                    time.sleep(0.1)
                    reply = self.request(self.host_endpoint, {"action": "acquire",
                        "task_id": lease["task_id"], "coordination_epoch": lease["coordination_epoch"]})
        except PreparationCancelled:
            raise
        except Exception as exc:
            lease["error"] = str(exc)[:500]
            save(record)
            raise PreparationUncertain("compile admission reply was lost; retained lease requires observation") from exc

    def activate(self, record, observation, save):
        lease = record["compile_lease"]
        reply = self.request(self.host_endpoint, {"action": "activate", "task_id": lease["task_id"],
            "coordination_epoch": lease["coordination_epoch"], "fence_token": lease["fence_token"],
            "process_guard": observation["receipt"]["process_guard"],
            "prepared_supervisor": {"container_name": self.container_name,
                "container_id": lease["container_id"], "receipt": observation["receipt"]}})
        lease["state"] = reply["task"]["state"]
        save(record)
        if lease["state"] != "active":
            raise RuntimeError("compile lease is not active; supervisor start gate stays closed")
        self.renewed_at = time.monotonic()
        return {"run_id": record["job_id"], "epoch": lease["coordination_epoch"], "fence": lease["fence_token"]}

    def heartbeat(self, record, save):
        if time.monotonic() - self.renewed_at < 30:
            return
        lease = record["compile_lease"]
        reply = self.request(self.host_endpoint, {"action": "heartbeat", "task_id": lease["task_id"],
            "coordination_epoch": lease["coordination_epoch"], "fence_token": lease["fence_token"]})
        lease["state"] = reply["task"]["state"]
        save(record)
        if lease["state"] != "active":
            raise RuntimeError("compile supervisor lost its active host lease")
        self.renewed_at = time.monotonic()
