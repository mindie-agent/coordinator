"""Adapters to the remote-dev package and host NPU device authority.

remote-dev is imported as the `remote_dev` package. The host queue remains
the single device-allocation authority. Generic process control is
`remote_dev.processes.control`.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import zlib
from dataclasses import dataclass
from pathlib import Path

from remote_dev.core.endpoint import resolve_endpoint
from remote_dev.core.shell_ops import remote_bash

from mindie_coordinator.host_queue import HostQueue
from mindie_coordinator.machine_directory import MachineDirectory
from mindie_coordinator.runtime_profile import build_key, digest, launch_preamble, native_compatibility_key, profile_key


@dataclass(frozen=True)
class PreparedNativeView:
    """Internal completed native proof, never a registration argument."""
    attestation: dict


def _captured_manifest(output: str) -> dict:
    """Decode a bounded complete capture receipt, never a preview or a subset."""
    limit = 16 * 1024 * 1024
    reply = json.loads(output)
    size = reply.get('manifest_bytes')
    encoded = reply.get('manifest_zlib_base64')
    if (type(size) is not int or not 0 < size <= limit or not isinstance(encoded, str)
            or len(encoded) > limit * 2):
        raise ValueError('invalid captured manifest size or encoding')
    compressed = base64.b64decode(encoded, validate=True)
    decoder = zlib.decompressobj()
    raw = decoder.decompress(compressed, size + 1)
    if len(raw) != size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError('captured manifest is truncated or exceeds its declared size')
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or digest(manifest) != reply.get('manifest_digest'):
        raise ValueError('captured manifest digest differs')
    return manifest


def _package_file(relative: str) -> Path:
    return Path(__file__).resolve().parent / relative


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


class RemoteBackend:
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
        script = "set -e\n" + "\n".join(f"export {key}={shlex.quote(value)}" for key, value in env.items())
        script += "\n" + task_preamble(binding) + "\nexport MINDIE_SERVICE_PORT=0\n" + command
        result = self.shell.run(binding["endpoint"], script, timeout_ms=120000)
        if result["outcome"] != "success":
            refs = result.get("refs") or {}
            detail = Path(refs["stderr"]).read_text(errors="replace") if refs.get("stderr") else str(result)
            cause = detail.strip().splitlines()[-1][-300:] if detail.strip() else result.get("summary", "no stderr")
            raise ExecutionRequestError(f"preflight failed before NPU allocation: {cause}\nlog refs: {refs}")

    def job_host_pid(self, runtime, receipt):
        code = '''
import json, subprocess
from pathlib import Path
request = json.loads(__import__('sys').argv[1])
info = json.loads(subprocess.check_output(['docker','inspect','--format','{{json .}}',request['container_name']], text=True, encoding="utf-8"))
if info['Id'] != request['container_id']:
    raise RuntimeError('container identity changed before activation')
receipt = request['receipt']
if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != receipt['boot_id']:
    raise RuntimeError('host boot identity changed')
rows = subprocess.check_output(['docker','top',request['container_name'],'-eo','pid'], text=True, encoding="utf-8").splitlines()[1:]
matches = []
for row in rows:
    pid = int(row.strip())
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        status = Path(f'/proc/{pid}/status').read_text().splitlines()
        namespace = next(line.split()[1:] for line in status if line.startswith('NSpid:'))
        if int(namespace[-1]) == receipt['pid'] and fields[19] == receipt['start_ticks'] and fields[0] != 'Z':
            matches.append(pid)
    except FileNotFoundError:
        continue
if len(matches) != 1:
    raise RuntimeError('cannot identify a unique host PID for the waiting supervisor')
print(json.dumps({'pid':matches[0]}))
'''
        request = {"container_name": runtime["container_name"], "container_id": runtime["attestation"]["container_id"], "receipt": receipt}
        command = "python3 - " + shlex.quote(json.dumps(request)) + " <<'MINDIE_HOST_PID'\n" + code + "\nMINDIE_HOST_PID\n"
        return json.loads(self.bash({**runtime["host_endpoint"], "root": "/", "cwd": "/"}, command))["pid"]

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
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, encoding="utf-8").strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--name-only", "--ignore-submodules=dirty"], text=True, encoding="utf-8")
    untracked = subprocess.check_output(["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard", "-z"], text=True, encoding="utf-8")
    private = {".mindie-runtime", ".remote-code-parity", "Mooncake"} if name == "." else set()
    extras = [path for path in untracked.split("\\0") if path and path.split("/", 1)[0] not in private]
    if head != expected or dirty.strip() or extras:
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
    tree = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD^{tree}'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo), 'diff', 'HEAD', '--name-only', '--ignore-submodules=dirty'], text=True).strip()
    untracked = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '--others', '--exclude-standard'], text=True).strip()
    if tree != row['tree'] or dirty or untracked:
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

    def prepare_task_root(self, spec, *, sources, environment, donor_python=None, workspace_root=None,
                          source_snapshot=None, reuse=None,
                          on_progress=None, log_dir=None, on_preparation_job=None, cancel_requested=None, compile_scope=None):
        """Materialize fixed sources and prepare or reuse their native environment.

        Does not mutate ``donor_python`` site-packages. Image packages may be
        reused via ``venv --system-site-packages``.
        """
        from mindie_coordinator.parity import (
            DEFAULT_MARKER_DIRNAME,
            materialize_fixed_sources,
            prepare_isolated_root_script,
            run_runtime_install_step,
        )
        from mindie_coordinator.parity_support import SshEndpoint, ssh_exec_stream
        from mindie_coordinator.provision.task_environment import INSTALL_STEPS, create_venv_script
        from mindie_coordinator.preparation_process import PreparationCancelled, PreparationUncertain

        endpoint = spec["endpoint"]
        # This cache is scoped to this preparation call, never a prior ready
        # result supplied by a caller or a container name across generations.
        spec.pop('_prepared_container', None)
        root = endpoint["root"]
        python = spec["python"]
        native_recipe = {"vllm", "vllm-ascend"}.issubset(sources or {})
        if native_recipe and donor_python and python == donor_python and not reuse:
            raise ValueError("task-owned interpreter must not be the donor interpreter")
        if source_snapshot is None:
            raise ValueError('preparation requires fixed execution source inputs')
        versions = {record['relpath']: {'version': record.get('scm_version'), 'source_head': record.get('source_head')}
                    for record in source_snapshot.get('records', []) if record['relpath'] in ('vllm', 'vllm-ascend')}
        publication = None
        if native_recipe:
            if any(not row['version'] for row in versions.values()):
                raise ValueError('native preparation requires captured source SCM versions')
            if reuse and reuse['kind'] == 'native':
                try:
                    native_compatibility_key(reuse['runtime']['attestation'])
                except (KeyError, ValueError):
                    pass
                else:
                    from mindie_coordinator.native_publication import NativeViewPublication
                    publication = NativeViewPublication(spec, reuse['runtime'], versions)
        container = SshEndpoint(host=endpoint["host"], port=int(endpoint["port"]), user=endpoint["user"])
        pending_setup = []
        def owned_process(step):
            nonlocal pending_setup
            if on_preparation_job is None:
                return None
            from mindie_coordinator.preparation_process import PreparationProcess
            from mindie_coordinator.provision.host_ops import DEFAULT_WORKDIR
            setup, pending_setup = pending_setup, []
            return PreparationProcess(endpoint, step, on_preparation_job, cancel_requested or (lambda: False),
                                      setup=setup, bootstrap_root=DEFAULT_WORKDIR)

        def check_cancel():
            if cancel_requested is not None and cancel_requested():
                from mindie_coordinator.preparation_process import PreparationCancelled
                raise PreparationCancelled("preparation cancelled between steps")

        def progress(step, event=None):
            value = {"step": step, **(event or {})}
            if log_dir is not None:
                value["log_ref"] = str(Path(log_dir) / (step + ".log"))
            if on_progress is not None:
                on_progress(value)
            return value.get("log_ref")

        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        # The first owned job starts in the existing container workspace and
        # establishes this execution's root before its materialization body.
        scripts = [("prepare-root", prepare_isolated_root_script(root))]
        if (native_recipe and (not reuse or reuse['kind'] != 'native')) or (not native_recipe and not donor_python):
            scripts.append(("create-venv", create_venv_script(root, python, donor_python)))
        if on_preparation_job is not None:
            pending_setup = scripts
        for step, script in (() if on_preparation_job is not None else scripts):
            check_cancel()
            log = progress(step)
            if step == "prepare-root":
                from remote_dev.core.ssh_transport import run_rpc_script
                from mindie_coordinator.parity_support import RemoteCommandError

                # A short package-owned filesystem command can establish the
                # same RPC connection used by the subsequent owned job. The
                # Python RPC does not require its endpoint root to exist yet.
                try:
                    completed = run_rpc_script(resolve_endpoint(endpoint), script, timeout_ms=45000)
                except Exception as exc:
                    if log:
                        with Path(log).open('a', encoding='utf-8') as stream:
                            stream.write(str(exc) + '\n')
                    raise
                if log:
                    with Path(log).open('a', encoding='utf-8') as stream:
                        stream.write((completed.stdout or '') + (completed.stderr or ''))
                if completed.cancelled:
                    from mindie_coordinator.preparation_process import PreparationCancelled
                    raise PreparationCancelled('root preparation cancelled after its command stopped')
                code = 255 if completed.timed_out or completed.returncode is None else completed.returncode
                if code:
                    raise RemoteCommandError(code, f'command failed ({code}): prepare-root\n'
                        f'stdout:\n{completed.stdout or ""}\nstderr:\n{completed.stderr or ""}')
            else:
                ssh_exec_stream(container, script, stream_progress=False, log_path=log,
                                process=owned_process(step))
            check_cancel()
        identity = spec.get("container_name") or ("mindie-" + spec["user"])
        check_cancel()
        log = progress("materialize")
        if sources:
            materialized = materialize_fixed_sources(
                workspace_id=identity, endpoint=endpoint, source_snapshot=source_snapshot,
                host_endpoint=spec['host_endpoint'],
                log_path=log, process=owned_process("materialize"),
                on_progress=lambda event: progress("materialize", event),
                build_source=({'versions': versions, 'build_env': source_snapshot.get('build_env', {})}
                              if native_recipe else None),
                **({'native_publication': publication} if publication is not None else {}),
            )
        check_cancel()
        if native_recipe:
            if publication is not None:
                if 'native_view' in materialized:
                    prepared = publication.accept(materialized['native_view'])
                else:
                    # A composite program exceeding the existing command byte
                    # budget retains the original two-operation path.
                    log = progress('publish-native-view')
                    prepared = self._prepare_native_view(spec, reuse['runtime'], versions,
                        process=owned_process('publish-native-view'), log_path=log)
                check_cancel()
                return PreparedNativeView(prepared)
        def reuse_previous():
            from mindie_coordinator.preparation_cache import REMOTE_REUSE_SUFFIX
            previous = reuse['runtime']
            request = {'kind': reuse['kind'], 'root': root, 'source_root': previous['endpoint']['root'],
                       'python': python, 'versions': versions}
            module = _package_file('runtime_profile.py').read_text()
            cache_source = _package_file('preparation_cache.py').read_text()
            previous_env = launch_preamble(previous['attestation']['profile'], python=previous['python'])
            script = (previous_env + '\n' + shlex.quote(previous['python']) + ' - ' + shlex.quote(json.dumps(request))
                      + " <<'MINDIE_REUSE'\n" + module + '\nexec(' + repr(cache_source) + ', globals())\n' + REMOTE_REUSE_SUFFIX + '\nMINDIE_REUSE\n')
            log = progress('reuse-' + reuse['kind'])
            ssh_exec_stream(container, script, stream_progress=False, log_path=log,
                            process=owned_process('reuse-' + reuse['kind']))
        steps = () if not native_recipe or (reuse and reuse['kind'] == 'native') else INSTALL_STEPS
        if reuse and reuse['kind'] == 'dependencies':
            steps = tuple(step for step in steps if step != 'install-vllm-ascend-requirements')
        compiled_native = False

        def install(step):
            nonlocal compiled_native
            log = progress(step)
            from contextlib import nullcontext
            scope = (compile_scope(step) if compile_scope and step in
                     {'install-vllm', 'install-vllm-ascend', 'install-vllm-ascend-incremental'} else nullcontext())
            with scope:
                run_runtime_install_step(
                    container=container,
                    runtime_root=root,
                    marker_dirname=DEFAULT_MARKER_DIRNAME,
                    container_identity=identity,
                    step=step,
                    stream_progress=False,
                    python=python,
                    on_progress=lambda event, step=step: progress(step, event),
                    log_path=log,
                    process=owned_process(step),
                )
            if step == 'install-vllm-ascend':
                compiled_native = True

        cached_native = False
        incremental_native = False

        def rebuild(reason):
            nonlocal cached_native
            progress('shared-native-cache', {'status': 'miss', 'reason': str(reason)[:500]})
            discarded = self._shared_native(spec, 'discard', versions, process=owned_process('shared-native-discard'))
            if discarded.get('status') != 'discarded':
                raise RuntimeError('cannot discard incomplete cached outputs: ' + discarded.get('reason', 'unknown'))
            cached_native = False
            install('check-build-compat')
            install('install-vllm')
            install('install-vllm-ascend')
            install('verify-deps')

        def accept_cache(cache):
            nonlocal cached_native, compiled_native, incremental_native
            progress('shared-native-cache', cache)
            cached_native = cache.get('status') in {'hit', 'incremental'}
            if not cached_native:
                return False
            if not cache.get('dependencies', {}).get('satisfied'):
                if reuse and reuse['kind'] == 'dependencies':
                    reuse_previous()
                else:
                    install('install-vllm-ascend-requirements')
                install('verify-deps')
                validated = self._shared_native(spec, 'revalidate', versions,
                    process=owned_process('shared-native-revalidate-after-dependencies'))
                if validated.get('status') != 'validated':
                    # Imports or broad requirement ranges cannot establish the
                    # original bundle's ABI after a dependency overlay changed.
                    rebuild(validated.get('reason', 'cached native environment changed during dependency repair'))
                    return True
            if cache['status'] == 'incremental':
                install('install-vllm-ascend-incremental')
                compiled_native = True
                incremental_native = True
            return True

        if native_recipe and steps:
            # The recipient venv already sees the image packages. Restore and
            # check their real dependency metadata before any pip/copy work.
            cache = self._shared_native(spec, 'restore', versions, process=owned_process('shared-native-restore'))
            remaining_donors = None
            while cache.get('status') == 'miss':
                exported = self._export_shared_native(spec, **({'donors': remaining_donors}
                                                              if remaining_donors is not None else {}))
                remaining_donors = exported.get('remaining_donors', [])
                progress('shared-native-export', {key: value for key, value in exported.items()
                                                   if key != 'remaining_donors'})
                check_cancel()
                if exported.get('status') != 'stored':
                    break
                cache = self._shared_native(spec, 'restore', versions,
                    candidate={key: exported[key] for key in ('bundle', 'native_key')},
                    process=owned_process('shared-native-restore-after-export'))
                if not remaining_donors:
                    break
            if accept_cache(cache):
                # Profile capture below performs this execution's real import
                # once. No editable install or repeated native smoke is needed.
                steps = ()
            elif reuse:
                reuse_previous()
        elif reuse:
            reuse_previous()

        for step in steps:
            if step == 'install-vllm-ascend' and reuse and reuse['kind'] == 'dependencies':
                # Copying a same-container dependency overlay may have repaired
                # an ABI mismatch from the early image-only lookup.
                cache = self._shared_native(spec, 'restore', versions,
                    process=owned_process('shared-native-restore-after-dependencies'))
                if accept_cache(cache):
                    continue
            if cached_native and step == 'verify-deps':
                continue
            install(step)
        check_cancel()
        log = progress("finalize-runtime")
        try:
            captured = self._write_ready_profile(spec, environment, native_recipe=native_recipe, source_versions=versions,
                                                process=owned_process("finalize-runtime"), log_path=log,
                                                managed_finalize=True, publish_native_cache=compiled_native)
        except (PreparationCancelled, PreparationUncertain):
            raise
        except Exception as exc:
            if incremental_native or not cached_native:
                raise
            rebuild(exc)
            captured = self._write_ready_profile(spec, environment, native_recipe=native_recipe, source_versions=versions,
                                                process=owned_process("finalize-runtime"), log_path=log,
                                                managed_finalize=True, publish_native_cache=compiled_native)
        check_cancel()
        if native_recipe and captured is not None:
            return PreparedNativeView(captured)
        # Legacy adapters and command roots retain their registration probe.

    def _shared_native(self, spec, action, versions, *, process=None, candidate=None):
        """One bounded automatic cache lookup/copy; no other user's runtime."""
        from mindie_coordinator.parity import PYTHON_METADATA_PREAMBLE, task_python_exports
        from mindie_coordinator.preparation_cache import REMOTE_SHARED_SUFFIX
        from mindie_coordinator.preparation_process import PreparationCancelled, PreparationUncertain
        try:
            root, python = spec['endpoint']['root'], spec['python']
            request = {'root': root, 'action': action, 'preparation': spec.get('preparation', {}),
                       'versions': versions, 'machine_type': spec.get('machine_type')}
            if candidate is not None:
                request['candidate'] = candidate
            if action == 'restore':
                request['image_digest'] = self._preparation_container(spec)['Image']
            module = _package_file('runtime_profile.py').read_text()
            build_source = _package_file('build_inputs.py').read_text()
            incremental = _package_file('native_incremental.py').read_text()
            cache = _package_file('preparation_cache.py').read_text()
            preamble = '\n'.join(['set -euo pipefail', 'export MINDIE_RUNTIME_ROOT=' + shlex.quote(root),
                                   *PYTHON_METADATA_PREAMBLE, *task_python_exports(python),
                                   'export PYTHONPATH=' + shlex.quote(':'.join([root + '/.mindie-runtime/metadata',
                                                                            root + '/vllm', root + '/vllm-ascend']))])
            command = (preamble + '\n' + shlex.quote(python) + ' - ' + shlex.quote(json.dumps(request))
                       + " <<'MINDIE_SHARED_NATIVE'\n" + module + '\nexec(' + repr(build_source) + ', globals())\n'
                       + 'exec(' + repr(incremental) + ', globals())\nexec(' + repr(cache) + ', globals())\n'
                       + REMOTE_SHARED_SUFFIX + '\nMINDIE_SHARED_NATIVE\n')
            if process is None:
                return json.loads(self.bash(spec['endpoint'], command))
            from mindie_coordinator.parity_support import SshEndpoint, ssh_exec_stream
            endpoint = spec['endpoint']
            result = ssh_exec_stream(SshEndpoint(endpoint['host'], int(endpoint['port']), endpoint['user']),
                                     command, stream_progress=False, process=process)
            return json.loads(result.stdout)
        except (PreparationCancelled, PreparationUncertain):
            raise
        except Exception as exc:
            return {'status': 'miss', 'reason': str(exc)[:500]}

    def _export_shared_native(self, spec, *, donors=None):
        """An existing verified donor is exported only when the cache missed."""
        from remote_dev.core.ssh_transport import run_remote_python
        from mindie_coordinator.preparation_process import PreparationCancelled, PreparationUncertain
        host = {**spec['host_endpoint'], 'root': '/', 'cwd': '/'}
        image = self._preparation_container(spec)['Image']
        source = '\n'.join(_package_file(name).read_text(encoding='utf-8')
                           for name in ('runtime_profile.py',))
        # Separate compile units preserve module-level future imports.
        for name in ('build_inputs.py', 'native_incremental.py', 'preparation_cache.py'):
            source += '\nexec(' + repr(_package_file(name).read_text(encoding='utf-8')) + ', globals())\n'
        source += "\nimport signal\nsignal.alarm(45)\nargs=json.loads(sys.argv[1])\nprint(json.dumps(store_shared_native(Path(args['root']),Path(SHARED_NATIVE_CACHE))))\n"
        module = _package_file('shared_native_discovery.py').read_text(encoding='utf-8')
        script = module + '\nimport sys\nprint(json.dumps(export_verified_donor(json.load(sys.stdin))))\n'
        request = {'preparation': spec.get('preparation', {}), 'image_digest': image,
                   'machine_type': spec.get('machine_type'), 'export_source': source}
        if donors is not None:
            request['donors'] = donors
        reply = run_remote_python(resolve_endpoint(host), script, request, timeout_ms=120000)
        if reply.get('status') == 'cancelled':
            raise PreparationCancelled('shared native export cancelled')
        if reply.get('status') == 'uncertain' or reply.get('remote_outcome') == 'unknown':
            raise PreparationUncertain('shared native export outcome is uncertain')
        return reply

    def _prepare_native_view(self, spec, previous, versions, *, process=None, log_path=None):
        from mindie_coordinator.preparation_cache import REMOTE_NATIVE_VIEW_SUFFIX
        from mindie_coordinator.parity_support import SshEndpoint, ssh_exec_stream

        donor = {key: value for key, value in previous['attestation'].items()
                 if key not in {'container_id', 'launch_preamble'}}
        snapshot = spec['source_snapshot']
        request = {'root': spec['endpoint']['root'], 'source_root': previous['endpoint']['root'],
                   'versions': versions, 'source_id': snapshot['id'],
                   'build_env': snapshot.get('build_env', {}), 'preparation': spec['preparation'],
                   'build_inputs': {row['relpath']: row['build_inputs'] for row in snapshot['records']
                                    if row['relpath'] in ('vllm', 'vllm-ascend')},
                   'donor_manifest_digest': digest(donor)}
        module = _package_file('runtime_profile.py').read_text()
        build_source = _package_file('build_inputs.py').read_text()
        cache_source = _package_file('preparation_cache.py').read_text()
        script = (launch_preamble(donor['profile'], python=previous['python']) + '\n'
                  + shlex.quote(previous['python']) + ' - ' + shlex.quote(json.dumps(request))
                  + " <<'MINDIE_NATIVE_VIEW'\n" + module + '\nexec(' + repr(build_source)
                  + ', globals())\nexec(' + repr(cache_source) + ', globals())\n'
                  + REMOTE_NATIVE_VIEW_SUFFIX + '\nMINDIE_NATIVE_VIEW\n')
        endpoint = spec['endpoint']
        completed = ssh_exec_stream(SshEndpoint(endpoint['host'], int(endpoint['port']), endpoint['user']),
                                   script, stream_progress=False, process=process, log_path=log_path)
        reply = json.loads(completed.stdout)
        manifest = {**reply['manifest'], 'files': donor['files']}
        if (reply.get('manifest_digest') != digest(manifest)
                or manifest.get('execution_view', {}).get('source_id') != snapshot['id']
                or manifest.get('runtime_root') != endpoint['root']):
            raise ValueError('native publication did not return the fixed execution view')
        return {**manifest, 'container_id': previous['attestation']['container_id'],
                'launch_preamble': launch_preamble(manifest['profile'], python=spec['python'])}

    def _preparation_container(self, spec):
        """Immutable image identity from this preparation's exact generation."""
        if '_prepared_container' not in spec:
            host = {**spec['host_endpoint'], 'root': '/', 'cwd': '/'}
            fields = shlex.quote('{"Id":{{json .Id}},"Image":{{json .Image}},"State":{{json .State}}}')
            value = json.loads(self.bash(host, f"docker inspect --format {fields} {shlex.quote(spec['container_name'])}"))
            if not value.get('Id') or not value.get('Image') or not value.get('State', {}).get('Running'):
                raise ValueError('cannot select image from an absent or stopped container')
            spec['_prepared_container'] = {'Id': value['Id'], 'Image': value['Image']}
        return spec['_prepared_container']

    def _write_ready_profile(self, spec, environment, *, native_recipe=True, source_versions=None,
                             process=None, log_path=None, managed_finalize=False, publish_native_cache=False):
        from mindie_coordinator.prepare_runtime import (
            CANN_VERSION_CANDIDATES,
            DRIVER_VERSION_CANDIDATES,
            REMOTE_CAPTURE_SUFFIX,
            REMOTE_COMMAND_CAPTURE_SUFFIX,
        )

        python = spec["python"]
        root = spec["endpoint"]["root"]
        host = {**spec["host_endpoint"], "root": "/", "cwd": "/"}
        name = shlex.quote(spec["container_name"])
        fields = shlex.quote('{"Id":{{json .Id}},"Image":{{json .Image}},"State":{{json .State}}}')
        info = json.loads(self.bash(host, f"docker inspect --format {fields} {name}"))
        if not info.get('Id') or not info.get('Image'):
            raise ValueError("cannot attest image digest")
        if not info['State']['Running'] or info['State'].get('Paused') or info['State'].get('Restarting'):
            raise ValueError('capture container is not running normally')
        expected = spec.get('_prepared_container')
        if expected and any(info[key] != expected[key] for key in ('Id', 'Image')):
            raise ValueError('preparation container generation changed before finalization')
        recipe = (environment or {}).get("recipe") or (environment or {}).get("image") or spec.get("recipe")
        module = _package_file("runtime_profile.py").read_text()
        build_source = _package_file("build_inputs.py").read_text()
        request = json.dumps({
            "root": root,
            "recipe": recipe,
            "image_digest": info['Image'],
            "machine_type": (environment or {}).get("machine_type") or spec.get("machine_type"),
            "cann_files": list(CANN_VERSION_CANDIDATES),
            "driver_files": list(DRIVER_VERSION_CANDIDATES),
            'preparation': spec.get('preparation', {}),
            'source_id': spec.get('source_snapshot', {}).get('id'),
            'source_versions': source_versions or {},
            'build_env': spec.get('source_snapshot', {}).get('build_env', {}),
            'excluded_roots': spec.get('excluded_launch_roots', []),
            'managed_finalize': managed_finalize,
            'publish_native_cache': publish_native_cache,
            'container_identity': spec.get('container_name') or ('mindie-' + spec['user'] if spec.get('user') else None),
            'installation_marker': (root + '/.remote-code-parity/runtime-install.json'
                                    if managed_finalize and native_recipe else None),
        })
        from mindie_coordinator.parity import DEFAULT_ENV_PREAMBLE, task_python_exports

        suffix = REMOTE_CAPTURE_SUFFIX if native_recipe else REMOTE_COMMAND_CAPTURE_SUFFIX
        runner = "\n_build_namespace = {}\nexec(" + repr(build_source) + ", _build_namespace)\n"
        if publish_native_cache:
            runner += 'exec(' + repr(_package_file('preparation_cache.py').read_text()) + ', globals())\n'
        runner += suffix
        preamble_lines = ["set -euo pipefail"]
        if native_recipe:
            preamble_lines.extend(['export MINDIE_RUNTIME_ROOT=' + shlex.quote(root), *DEFAULT_ENV_PREAMBLE, *task_python_exports(python),
                'export PYTHONPATH=' + shlex.quote(':'.join([root + '/.mindie-runtime/metadata', root + '/vllm', root + '/vllm-ascend'])) + '"${PYTHONPATH:+:$PYTHONPATH}"'])
        else:
            # The image's recorded launch settings can be reused without
            # sourcing CANN/toolchain installers for an ordinary command.
            preamble_lines.append('unset PYTHONPATH ASCEND_CUSTOM_OPP_PATH')
            for key, value in spec.get('donor_launch_env', {}).items():
                preamble_lines.append('export ' + key + '=' + shlex.quote(value))
            preamble_lines.append('export PATH=' + shlex.quote(str(__import__('pathlib').PurePosixPath(python).parent)) + '"${PATH:+:$PATH}"')
        preamble = '\n'.join(preamble_lines)
        command = (preamble + "\n" + shlex.quote(python) + " - " + shlex.quote(request)
                   + " <<'MINDIE_CAPTURE_PROBE'\n" + module + runner + "\nMINDIE_CAPTURE_PROBE\n")
        if process is None:
            output = self.bash(spec["endpoint"], command)
        else:
            from mindie_coordinator.parity_support import SshEndpoint, ssh_exec_stream
            endpoint = spec["endpoint"]
            completed = ssh_exec_stream(SshEndpoint(endpoint["host"], int(endpoint["port"]), endpoint["user"]),
                                        command, stream_progress=False, process=process, log_path=log_path)
            output = completed.stdout
        if native_recipe:
            manifest = _captured_manifest(output)
            source_id = spec.get('source_snapshot', {}).get('id')
            files, evidence = manifest.get('files', {}), manifest.get('evidence', {})
            if (manifest.get('schema_version') != 1 or not files
                    or not {'library', 'metadata'}.issubset({row.get('role') for row in files.values()})
                    or not {'cann', 'driver', 'smoke'}.issubset(evidence)
                    or any(not re.fullmatch('[0-9a-f]{64}', str(row.get('sha256', '')))
                           for row in [*files.values(), *evidence.values()])
                    or manifest.get('profile_key') != profile_key(manifest['profile'])
                    or manifest.get('build_key') != build_key(manifest['profile'], manifest['build_inputs'])):
                raise ValueError('capture did not return a complete native proof')
            if (not source_id or manifest.get('execution_view', {}).get('source_id') != source_id
                    or manifest.get('execution_view', {}).get('python') != python
                    or manifest.get('runtime_root') != root
                    or manifest.get('profile', {}).get('image_digest') != info['Image']):
                raise ValueError('capture did not return this fixed execution view')
            return {**manifest, 'container_id': info['Id'],
                    'launch_preamble': launch_preamble(manifest['profile'], python=python)}
