"""Adapters to the remote-dev package and host NPU device authority.

[Role]
Bridges `mindie_coordinator` to `remote_dev` (`remote_bash`, `run_rpc_script`,
`run_remote_python`, `remote_dev.processes.control`) and the remote host queue
(`HostQueue`) for container preparation, runtime inspection, and job activation.

[Invariants]
- `HostQueue` remains the single device-allocation authority; `RemoteBackend`
  never allocates cards locally.
- `_captured_manifest` verifies exact uncompressed byte length, zlib EOF, and
  SHA-256 digest before accepting a remote runtime capture receipt.
- `verify_preflight` runs container state and manifest digest probes in parallel
  and drains both before allowing host preflight or raising an error.

[Timeouts & Locks]
- Default `remote_shell` / `bash` timeout: `45000ms` (`45s`).
- `preflight` and `_export_shared_native` timeout: `120000ms` (`120s`).

[Side-Effect Policy]
- Never mutates `donor_python` site-packages or stops unrelated user containers.
"""

from __future__ import annotations

import json
import shlex
import time
from pathlib import Path

from remote_dev.core.endpoint import resolve_endpoint
from remote_dev.core.shell_ops import remote_bash

from mindie_coordinator.backend_preparation import (
    BackendPreparationMixin,
    PreparedNativeView,
    _captured_manifest,
    _package_file,
)
from mindie_coordinator.host_queue import HostQueue
from mindie_coordinator.machine_directory import MachineDirectory
from mindie_coordinator.runtime_profile import build_key, digest, launch_preamble, native_compatibility_key, profile_key


class RemoteDev:
    """Explicit-endpoint shell via the installed `remote_dev` package."""

    def run(self, target: dict, command: str, *, timeout_ms: int = 45000) -> dict:
        return remote_shell(target, command, timeout_ms=timeout_ms)


def remote_shell(target: dict, command: str, *, timeout_ms: int = 45000) -> dict:
    """Run one command through remote-dev's explicit-endpoint shell."""
    if not target.get("host") or not target.get("port"):
        raise ValueError("remote-dev calls require an explicit host and port")
    return remote_bash(
        resolve_endpoint(dict(target)),
        command=command,
        timeout_ms=timeout_ms,
        runtime_env=False,
        wait=True,
    )["result"]


class RemoteBackend(BackendPreparationMixin):
    supports_task_messages = True

    def __init__(self, *, shell=None, host_queue=None, machines=None, host_queue_module=None):
        self.shell = shell or RemoteDev()
        self.machines = machines or MachineDirectory()
        self.host_queue = host_queue or HostQueue(self.bash if shell is not None else None, module_path=host_queue_module)

    def job(self, runtime, job_id, action, **parameters):
        from remote_dev.processes import control

        endpoint = dict(runtime["endpoint"])
        return control(resolve_endpoint(endpoint), job_id, action, **parameters)

    def wait_job(self, runtime, job_id, *, timeout_seconds=10):
        """Wait inside the existing remote supervisor; no launch or lease change."""
        return self.job(runtime, job_id, 'exchange', wait_for_exit=True,
                        yield_time_ms=int(timeout_seconds * 1000), max_bytes=1)

    def cleanup_slot(self, runtime, slot_id):
        from mindie_coordinator.execution_slots import slot_cleanup_command
        result = self.shell.run(runtime["endpoint"], slot_cleanup_command(slot_id), timeout_ms=15000)
        if result["outcome"] != "success":
            raise RuntimeError("slot cache cleanup did not complete: " + str(result))

    def run_evidence(self, runtime, job, run):
        """Read one completed NPU window without changing execution state."""
        devices = run.get("task", {}).get("granted_devices") or []
        if not devices:
            return {}
        from mindie_coordinator.npu_top_bridge import NpuTopObservationClient
        result = (job.get("remote") or {}).get("result") or {}
        start, end = result.get("started_at"), result.get("finished_at")
        evidence = {"source": "npu-top", "status": "unavailable",
            "run_id": job.get("execution_id") or job["id"],
            "host": runtime["host_endpoint"]["host"], "devices": devices,
            "coverage_ratio": 0.0, "throttling_detected": None,
            "resource_interference_detected": None, "measurement_limitations": []}
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or end <= start:
            evidence["measurement_limitations"] = ["remote command timestamps are unavailable; telemetry was not inferred from client wall time"]
        else:
            try:
                # npu-top timestamps its samples on the collector, while the
                # supervisor reports the remote wall clock. Measure an offset
                # interval; never query a different 250s-old window as though
                # the two machines had synchronized clocks.
                before = time.time()
                clock = self.shell.run(runtime["endpoint"], shlex.quote(runtime["python"]) +
                    " -c 'import time; print(time.time())'", timeout_ms=2000)
                after = time.time()
                if clock["outcome"] != "success":
                    raise RuntimeError("remote clock sample unavailable")
                remote_now = float(Path(clock["refs"]["stdout"]).read_text().strip())
                offset_min, offset_max = before - remote_now, after - remote_now
                evidence.update(NpuTopObservationClient().telemetry_window(
                    evidence["host"], devices, start + offset_min, end + offset_max))
                evidence["command_window"] = {"started_at": start, "finished_at": end,
                                              "basis": "remote_supervisor"}
                evidence["clock_alignment"] = {"offset_min_seconds": offset_min,
                    "offset_max_seconds": offset_max, "basis": "bounded remote clock sample"}
                evidence["measurement_limitations"].append(
                    "command window is a bounded clock-offset projection; offset stability across the command is unverified")
                if evidence["status"] == "observed":
                    evidence["status"] = "partial"
            except Exception as exc:
                evidence["measurement_limitations"] = [f"telemetry read failed: {type(exc).__name__}: {exc}"[:500]]
        return {"hardware_telemetry": evidence}

    def submit_and_acquire(self, runtime, request):
        """Use the already persisted epoch for one host admission exchange."""
        return self.host(runtime, {**request, "action": "submit-acquire"})

    def submit_and_preflight(self, runtime, request):
        """The caller just verified this new run's fixed runtime inputs."""
        return self.host(runtime, {**request, "action": "submit-acquire-preflight"})

    def startup_context(self, runtime, task_id):
        """Read this new task's epoch and container identity in one host call."""
        return self.host(runtime, {"action": "startup-context", "task_id": task_id,
                                   "container_name": runtime["container_name"]})

    def preflight(self, binding, command, env):
        from mindie_coordinator.managed_execution import ExecutionRequestError, task_preamble
        if (env or {}).get("MINDIE_SKIP_OP_PREFLIGHT") != "1" and binding.get("source_snapshot"):
            from mindie_coordinator.op_preflight import check_snapshot_op_contracts, format_op_violations
            violations = check_snapshot_op_contracts(binding["source_snapshot"])
            if violations:
                raise ExecutionRequestError(format_op_violations(violations))
        script = "set -e\n" + "\n".join(f"export {key}={shlex.quote(value)}" for key, value in env.items())
        script += "\n" + task_preamble(binding) + "\nexport MINDIE_SERVICE_PORT=0\n" + command
        result = self.shell.run(binding["endpoint"], script, timeout_ms=120000)
        if result["outcome"] != "success":
            refs = result.get("refs") or {}
            detail = Path(refs["stderr"]).read_text(errors="replace") if refs.get("stderr") else str(result)
            cause = detail.strip().splitlines()[-1][-300:] if detail.strip() else result.get("summary", "no stderr")
            raise ExecutionRequestError(f"preflight failed before NPU allocation: {cause}\nlog refs: {refs}")

    def activate_prepared(self, runtime, request, receipt):
        """Map the supervisor identity and activate through one host authority call."""
        return self.host(runtime, {**request, "action": "activate", "prepared_supervisor": {
            "container_name": runtime["container_name"],
            "container_id": runtime["attestation"]["container_id"],
            "receipt": receipt,
        }})

    def catalog(self):
        return self.machines.catalog()

    def resolve_registration(self, spec):
        if "machine" not in spec:
            return spec
        host = self.machines.host(spec["machine"])
        user = spec["user"]
        return {"user": user, "python": spec["python"],
                "host_endpoint": {"host": host["ip"], "port": host.get("port", 22), "user": host.get("user", "root")},
                "endpoint": {"host": host["ip"], "port": spec["port"], "root": spec["root"],
                             "cwd": spec.get("cwd") or spec["root"], "user": spec.get("ssh_user", "root")},
                "container_name": spec.get("container_name") or ("mindie-" + user),
                "service_ports": spec.get("service_ports", [])}

    def host(self, runtime, request):
        return self.host_queue.request(runtime["host_endpoint"], request)

    def bash(self, target, command):
        result = self.shell.run(target, command, timeout_ms=45000)
        if result["outcome"] != "success":
            from mindie_coordinator.parity_support import RemoteCommandError
            refs = result.get("refs") or {}
            stderr = Path(refs["stderr"]).read_text(errors="replace")[-300:].strip() if refs.get("stderr") else ""
            code = result.get("exit_code")
            message = (f"runtime probe failed ({result['outcome']}/{result.get('status')}, "
                       f"exit {code}): {stderr or 'no stderr'}")
            if type(code) is int and code > 0:
                raise RemoteCommandError(code, message)
            raise RuntimeError(message)
        return Path(result["refs"]["stdout"]).read_text()

    def _inspect_container(self, runtime):
        host = {**runtime["host_endpoint"], "root": "/", "cwd": "/"}
        name = shlex.quote(runtime["container_name"])
        fields = shlex.quote('{"Id":{{json .Id}},"State":{{json .State}}}')
        info = json.loads(self.bash(host, f"docker inspect --format {fields} {name}"))
        if not info["State"]["Running"] or info["State"].get("Paused") or info["State"].get("Restarting"):
            raise RuntimeError("prepared container is not running normally")
        return info

    def inspect(self, runtime, *, idle=False, snapshots=None):
        # idle remains an inspect of the selected prepared root and container
        # identity. It must not require the whole user container to be empty:
        # sibling roots may have authorized executions.
        del idle
        info = self._inspect_container(runtime)
        if runtime.get('reuse_only'):
            observed = self.qualify_prepared_inputs(runtime, runtime['source_snapshot'], include_manifest=True)
            if not observed.get('qualified'):
                raise ValueError('artifact donor does not match fixed input: ' + observed.get('reason', 'unknown'))
            manifest = observed['manifest']
            return {**manifest, 'container_id': info['Id'],
                    'launch_preamble': launch_preamble(manifest['profile'], python=runtime['python'])}
        manifest = self._inspect_manifest(runtime, snapshots=snapshots)
        return {**manifest, "container_id": info["Id"],
                "launch_preamble": launch_preamble(manifest["profile"], python=runtime.get("python"))}

    def verify_preflight(self, runtime, *, snapshots=None, _container_info=None):
        """Check the registered launch view with a compact remote reply.

        Owned native publications reuse their completed output proof and check
        mutable environment/source facts. Other roots retain full inspection.
        A digest confirms the complete manifest; the live container and derived
        launch preamble are compared separately.
        """
        expected = runtime["attestation"]
        if runtime.get('reuse_only'):
            # Historical donor qualification has its own source-tree contract.
            # Such roots are not managed bindings; retain its complete probe.
            if self.inspect(runtime, snapshots=snapshots) != expected:
                raise ValueError("runtime changed before launch")
            return True
        manifest = {key: value for key, value in expected.items()
                    if key not in {"container_id", "launch_preamble"}}
        if launch_preamble(manifest["profile"], python=runtime.get("python")) != expected.get("launch_preamble"):
            raise ValueError("runtime launch environment changed before launch")
        expected_digest = digest(manifest)

        def check_container():
            info = self._inspect_container(runtime) if _container_info is None else _container_info
            if _container_info is not None and (not info["State"]["Running"]
                    or info["State"].get("Paused") or info["State"].get("Restarting")):
                raise RuntimeError("prepared container is not running normally")
            if info["Id"] != expected.get("container_id"):
                raise ValueError("runtime container changed before launch")

        def check_manifest():
            reply = self._inspect_manifest(runtime, snapshots=snapshots,
                                           expected_digest=expected_digest,
                                           prepared_view=runtime.get('prepared_native_view', False))
            if reply != {"manifest_digest": expected_digest}:
                raise ValueError("runtime verification returned no matching manifest digest")

        # These read-only observations have no data dependency. Keep both in
        # this preflight (never across queue waits), and drain both operations
        # before allowing host preflight or reporting any failure.
        from concurrent.futures import ThreadPoolExecutor
        from contextvars import copy_context
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="mindie-preflight") as workers:
            checks = [workers.submit(copy_context().run, check) for check in (check_container, check_manifest)]
        errors = [check.exception() for check in checks if check.exception() is not None]
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("runtime preflight checks failed: " + "; ".join(map(str, errors)), errors)
        return True

    def _inspect_manifest(self, runtime, *, snapshots=None, expected_digest=None, prepared_view=False):
        python = runtime.get("python") or "python3"
        module = _package_file("runtime_profile.py").read_text()
        request = json.dumps({"root": runtime["endpoint"].get("cwd") or runtime["endpoint"]["root"],
                              "snapshots": snapshots or {}, "prepared_view": prepared_view,
                              **({"expected_manifest_digest": expected_digest} if expected_digest is not None else {})})
        build_source = _package_file("build_inputs.py").read_text()
        runner = "\n_build_namespace = {}\nexec(" + repr(build_source) + ", _build_namespace)\n" + '''
import subprocess
import sys
args = json.loads(sys.argv[1])
root = Path(args["root"])
manifest = json.loads((root / ".mindie-runtime/ready-profile.json").read_text())
if args.get('prepared_view') and 'expected_manifest_digest' in args and digest(manifest) != args['expected_manifest_digest']:
    raise ValueError('runtime changed before launch')
if args.get('prepared_view'):
    verify_execution_view(root, manifest)
else:
    verify(root, manifest)
if not args.get('prepared_view') and manifest['profile'].get('kind') != 'command' and _build_namespace["runtime_build_inputs"](root, manifest["profile"], manifest["profile_key"]) != manifest["build_inputs"]:
    raise ValueError("cache miss: installed native artifacts do not match current source inputs")
for name, expected in args["snapshots"].items():
    repo = root / name
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, encoding="utf-8", timeout=30, stdin=subprocess.DEVNULL).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--name-only", "--ignore-submodules=dirty"], text=True, encoding="utf-8", timeout=30, stdin=subprocess.DEVNULL)
    if head != expected or dirty.strip():
        raise ValueError("runtime source differs from pinned snapshot: " + name)
if 'expected_manifest_digest' in args:
    actual_digest = digest(manifest)
    if actual_digest != args['expected_manifest_digest']:
        raise ValueError('runtime changed before launch')
    print(json.dumps({'manifest_digest': actual_digest}))
else:
    print(json.dumps(manifest))
'''
        view = runtime['endpoint'].get('cwd') or runtime['endpoint']['root']
        prefix = self._probe_preamble(runtime, view)
        command = (prefix + shlex.quote(python) + " - " + shlex.quote(request)
                   + " <<'MINDIE_READY_PROBE'\n" + module + runner + "\nMINDIE_READY_PROBE\n")
        return json.loads(self.bash(runtime["endpoint"], command))

    def qualify_prepared_inputs(self, runtime, source_snapshot, *, include_manifest=False):
        """Read-only qualification of a historical verified exact-source donor.

        Old registration is evidence, never a new execution. Its complete
        artifact hashes, environment and clean Git trees must still match.
        A differing source tree is not adopted by guessing build scope.
        """
        from mindie_coordinator.parity import validate_relative_posix_path
        records = source_snapshot.get('records', [])
        for record in records:
            validate_relative_posix_path(record['relpath'], label='artifact donor source')
        request = {'root': runtime['endpoint']['root'], 'records': records, 'include_manifest': include_manifest}
        module = _package_file('runtime_profile.py').read_text()
        build_source = _package_file('build_inputs.py').read_text()
        runner = "\n_build_namespace = {}\nexec(" + repr(build_source) + ", _build_namespace)\n" + r'''
import subprocess, sys
args = json.loads(sys.argv[1])
root = Path(args['root'])
manifest = json.loads((root / '.mindie-runtime/ready-profile.json').read_text())
verify(root, manifest)
if _build_namespace['runtime_build_inputs'](root, manifest['profile'], manifest['profile_key']) != manifest['build_inputs']:
    print(json.dumps({'qualified': False, 'reason': 'installed native artifacts do not match current source inputs'}))
    raise SystemExit(0)
build_info = 'vllm-ascend/vllm_ascend/_build_info.py'
if (root / build_info).is_file() and build_info not in manifest['files']:
    print(json.dumps({'qualified': False, 'reason': 'generated build metadata has no verified donor hash'}))
    raise SystemExit(0)
for row in args['records']:
    repo = root / row['relpath']
    tree = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD^{tree}'], text=True, timeout=30, stdin=subprocess.DEVNULL).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo), 'diff', 'HEAD', '--name-only', '--ignore-submodules=dirty'], text=True, timeout=30, stdin=subprocess.DEVNULL).strip()
    if tree != row['tree'] or dirty:
        print(json.dumps({'qualified': False, 'reason': 'source tree differs', 'source': row['relpath']}))
        raise SystemExit(0)
print(json.dumps({'qualified': True, 'build_key': manifest['build_key'], **({'manifest': manifest} if args.get('include_manifest') else {})}))
'''
        root = runtime['endpoint']['root']
        prefix = self._probe_preamble(runtime, root)
        script = prefix + '\n' + shlex.quote(runtime['python']) + ' - ' + shlex.quote(json.dumps(request))
        script += " <<'MINDIE_QUALIFY'\n" + module + runner + '\nMINDIE_QUALIFY\n'
        return json.loads(self.bash(runtime['endpoint'], script))

    @staticmethod
    def _probe_preamble(runtime, root):
        # Metadata in image/CANN paths must resolve in the same environment
        # used by the captured proof and actual launch, before Python starts.
        profile = (runtime.get('attestation') or {}).get('profile') or runtime.get('profile')
        prefix = launch_preamble(profile, python=runtime.get('python')) + '\n' if profile else ''
        return prefix + 'export PYTHONPATH=' + shlex.quote(':'.join([
            root + '/.mindie-runtime/metadata', root + '/vllm', root + '/vllm-ascend'])) + '"${PYTHONPATH:+:$PYTHONPATH}"\n'

    def command_environment(self, donor):
        """Resolve the real image interpreter without building an environment."""
        from mindie_coordinator.runtime_profile import command_launch_environment
        endpoint = donor.get('endpoint') or {}
        profile = donor.get('profile') or donor.get('attestation', {}).get('profile') or {}
        roots = [donor.get('root'), endpoint.get('root'), endpoint.get('cwd')]
        environment = command_launch_environment(profile.get('launch_env', {}), roots=roots)
        candidate = donor.get('python')
        preamble = '\n'.join('export ' + key + '=' + shlex.quote(value) for key, value in environment.items())
        if candidate:
            command = shlex.quote(candidate)
        else:
            preamble += '\nIMAGE_PYTHON="$(ls -1d /usr/local/python*/bin/python3 2>/dev/null | sort -V | tail -n 1)"\n'
            preamble += 'if [ -z "$IMAGE_PYTHON" ]; then IMAGE_PYTHON="$(command -v python3)"; fi\n'
            command = '"$IMAGE_PYTHON"'
        code = 'import os,sys; print(os.path.realpath(getattr(sys,"_base_executable",sys.executable)))'
        python = self.bash(endpoint, preamble + '\n' + command + ' -c ' + shlex.quote(code)).strip()
        if not python.startswith('/') or '\n' in python:
            raise ValueError('image interpreter discovery did not return an absolute path')
        return {'python': python, 'launch_env': environment}
