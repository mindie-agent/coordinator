"""Durable remote-dev ownership for coordinator preparation commands."""
from __future__ import annotations

import json
import time
import uuid
from mindie_diagnostics import get_recorder, current_context
from remote_dev.observability import observed_operation

from remote_dev.core.ssh_transport import RemoteCompleted
from remote_dev.processes import control
from mindie_coordinator.preparation_script import preparation_command


class PreparationCancelled(RuntimeError):
    """Cancellation completed with a verified quiet process family."""


class PreparationUncertain(RuntimeError):
    """Retained process facts require observation or stop, never command replay."""


def _remember(record, observation, save):
    # Output belongs to the step log, not the execution database. Keep the
    # receipt, outcome and cursors needed to observe/stop after daemon restart.
    record.update({key: value for key, value in observation.items()
                   if key not in {"stdout", "stderr", "processes", "timings", "transport"}})
    if (observation.get("result") or {}).get("timings"):
        record["command_timings"] = observation["result"]["timings"]
    for key in ("timings", "transport"):
        if observation.get(key):
            # Preserve the launch phases and accumulate bounded exchange costs;
            # a later status-only observation cannot erase earlier evidence.
            if key == "timings":
                record.setdefault("timings", {}).update(observation[key])
            else:
                totals = record.setdefault("transport_totals", {"observations": 0})
                totals["observations"] += 1
                for name in ("connection_wait_ms", "rpc_ms", "pool_wait_ms"):
                    if isinstance(observation[key].get(name), (int, float)):
                        totals[name] = totals.get(name, 0) + observation[key][name]
            get_recorder("mindie-coordinator").event("DEBUG", "preparation.observation", job_id=record["job_id"],
                phase=record.get("step"), **observation[key])
    record["observed_at"] = time.time()
    save(record)


def stop_preparation_process(record, save, *, force=False, drain_seconds=10):
    """Stop only the persisted owned job; uncertainty never means completion."""
    try:
        observation = control(record["endpoint"], record["job_id"], "stop", force=force)
        _remember(record, observation, save)
        deadline = time.monotonic() + drain_seconds
        while not observation.get("quiet") and not observation.get("unknown") and time.monotonic() < deadline:
            time.sleep(0.1)
            observation = control(record["endpoint"], record["job_id"], "status")
            _remember(record, observation, save)
        return bool(observation.get("quiet"))
    except Exception as exc:
        _remember(record, {"state": "uncertain", "quiet": False, "error": str(exc)[:500]}, save)
        return False


class PreparationProcess:
    def __init__(self, endpoint, step, save, cancel_requested, *, timeout_seconds=7200,
                 setup=(), bootstrap_root=None):
        self.endpoint = dict(endpoint)
        self.step = step
        self.save = save
        self.cancel_requested = cancel_requested
        self.timeout_seconds = timeout_seconds
        self.setup = list(setup)
        self.bootstrap_root = bootstrap_root

    @observed_operation("preparation.command", component="mindie-coordinator")
    def run(self, script, *, on_output):
        started = time.monotonic()
        if self.cancel_requested():
            raise PreparationCancelled("preparation cancelled before command launch")
        endpoint = dict(self.endpoint)
        setup = self.setup
        if setup:
            endpoint.update(root=self.bootstrap_root, cwd=self.bootstrap_root)
        command = preparation_command(script, setup=setup, cwd=(self.endpoint.get('cwd') or self.endpoint['root']) if setup else None)
        record = {"endpoint": endpoint, "job_id": "prepare-" + uuid.uuid4().hex,
                  "step": self.step, "state": "pending", "quiet": False,
                  "stdout_offset": 0, "stderr_offset": 0, "diagnostics_context": current_context()}
        def save_record(value):
            value['elapsed_seconds'] = round(time.monotonic() - started, 6)
            self.save(value)
        if setup:
            record['setup_steps'] = [step for step, _ in setup]
        # Persist BEFORE the first remote side effect. A lost launch reply can
        # always be observed or stopped by this exact job id without replay.
        save_record(record)
        # A lost reply retains this exact job, never a second bootstrap/reset.
        self.setup = []
        pending = {"stdout": "", "stderr": ""}

        def emit(observation, final=False):
            for channel in pending:
                pending[channel] += observation.get(channel, "")
                lines = pending[channel].splitlines(keepends=True)
                pending[channel] = ""
                for line in lines:
                    if final or line.endswith(("\n", "\r")):
                        if channel == 'stderr' and line.startswith('__MINDIE_PARITY_PROGRESS__='):
                            try:
                                event = json.loads(line.split('=', 1)[1])
                                if event.get('phase') in record.get('setup_steps', []) and isinstance(event.get('elapsed_seconds'), (int, float)):
                                    record.setdefault('stage_timings', {})[event['phase']] = event['elapsed_seconds']
                            except (ValueError, TypeError):
                                pass
                        on_output(channel, line)
                    else:
                        pending[channel] = line

        def exchange(wait=1000):
            # Keep the bounded yield/cancel cadence, but do not add a round
            # trip just because final output arrived before the exit receipt.
            observation = control(endpoint, record["job_id"], "exchange",
                                  stdout_offset=record["stdout_offset"], stderr_offset=record["stderr_offset"],
                                  max_bytes=32768, yield_time_ms=wait, wait_for_exit=True)
            emit(observation)
            _remember(record, observation, save_record)
            return observation

        try:
            observation = control(endpoint, record["job_id"], "launch", spec={
                "command": command, "cwd": endpoint["cwd"], "env": {},
                "timeout_seconds": self.timeout_seconds, "interactive": False,
            }, authorization={}, stdout_offset=record["stdout_offset"],
                stderr_offset=record["stderr_offset"], max_bytes=32768, yield_time_ms=1000,
                wait_for_exit=True)
            emit(observation)
            _remember(record, observation, save_record)
            while True:
                if self.cancel_requested():
                    if not stop_preparation_process(record, save_record):
                        raise PreparationUncertain("preparation stop has not verified quiet; retained job can be stopped again")
                    while True:
                        observation = exchange(0)
                        if not any(observation.get(channel + "_bytes_remaining") for channel in pending):
                            break
                    emit({}, final=True)
                    raise PreparationCancelled("preparation command stopped with verified quiet")
                if observation.get("unknown") or observation.get("state") in {"uncertain", "lost_outcome", "absent"}:
                    raise PreparationUncertain("preparation process outcome is unknown; command was not replayed")
                if observation.get("quiet") and not any(observation.get(channel + "_bytes_remaining") for channel in pending):
                    emit({}, final=True)
                    result = observation.get("result") or {}
                    code = result.get("exit_code")
                    if code is None:
                        raise PreparationUncertain("quiet preparation process has no command exit receipt")
                    return RemoteCompleted(int(code), "", "", timed_out=False)
                observation = exchange()
        except (PreparationCancelled, PreparationUncertain):
            raise
        except Exception as exc:
            _remember(record, {"state": "uncertain", "quiet": False, "error": str(exc)[:500]}, save_record)
            raise PreparationUncertain("preparation transport failed; retained owned job was not replayed") from exc
