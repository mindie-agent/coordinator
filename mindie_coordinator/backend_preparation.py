"""Preparation mixin for remote runtime materialization.

[Role]
Houses `RemoteBackend`'s task-root preparation, shared-native cache restore /
export, native-view publication, preparation container identity, and ready
profile capture. `backend.RemoteBackend` composes this mixin with inspection,
preflight, host queue, and remote shell adapters.

[Invariants]
- Fixed Git source inputs are mandatory; a task-owned interpreter is never the
  donor interpreter.
- Every installed/captured artifact is verified against the same fixed
  execution view before it is returned; uncertain/cancelled preparation is
  propagated, not converted into an ordinary cache miss.
- Remote script text and heredoc delimiters are preserved byte-for-byte from
  the former backend module; the facade keeps its explicit `timeout=30,
  stdin=subprocess.DEVNULL` git calls.
- `_captured_manifest` verifies exact uncompressed byte length, zlib EOF, and
  SHA-256 digest before accepting a remote capture receipt.
- `native_compatibility_key` is resolved through the `backend` facade at call
  time so existing monkeypatch/import surfaces keep working.

[Timeouts & Locks]
- `_export_shared_native` uses `120000ms` (`120s`); preparation scripts retain
  their own bounded `signal.alarm(45)`/step timeouts and owned-job processes.
- No local locks; remote jobs serialize through the host queue.

[Side-Effect Policy]
- Mutates only the task's prepared root, its transaction-owned containers, and
  the shared native cache. Never mutates `donor_python` site-packages or stops
  unrelated user containers.
"""

from __future__ import annotations

import base64
import json
import re
import shlex
import zlib
from dataclasses import dataclass
from pathlib import Path

from remote_dev.core.endpoint import resolve_endpoint

from mindie_coordinator.runtime_profile import (
    build_key, digest, launch_preamble, native_compatibility_key, profile_key,
)


@dataclass(frozen=True)
class PreparedNativeView:
    """Internal completed native proof, never a registration argument."""
    attestation: dict
    native_cache_receipt: dict | None = None


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


def _native_compatibility_key(manifest: dict) -> str:
    """Resolve through the `backend` facade so its re-export stays patchable."""
    from mindie_coordinator import backend as backend_facade
    return getattr(backend_facade, "native_compatibility_key", native_compatibility_key)(manifest)


class BackendPreparationMixin:
    """Preparation methods composed into `backend.RemoteBackend`."""

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
                    _native_compatibility_key(reuse['runtime']['attestation'])
                except (KeyError, ValueError):
                    pass
                else:
                    from mindie_coordinator.native_publication import NativeViewPublication
                    publication = NativeViewPublication(spec, reuse['runtime'], versions)
        container = SshEndpoint(host=endpoint["host"], port=int(endpoint["port"]), user=endpoint["user"],
                                container=endpoint.get("container"))
        pending_setup = []
        def owned_process(step):
            nonlocal pending_setup
            if on_preparation_job is None:
                return None
            from mindie_coordinator.preparation_process import PreparationProcess
            from mindie_coordinator.compile_lease import COMPILE_STEPS, CompileLease
            from mindie_coordinator.provision.host_ops import DEFAULT_WORKDIR
            setup, pending_setup = pending_setup, []
            return PreparationProcess(endpoint, step, on_preparation_job, cancel_requested or (lambda: False),
                                      setup=setup, bootstrap_root=DEFAULT_WORKDIR,
                                      compile_lease=CompileLease(spec["host_endpoint"], spec["container_name"],
                                          self.host_queue.request,
                                          cleanup=lambda endpoint, slot: self.cleanup_slot({"endpoint": endpoint}, slot))
                                      if step in COMPILE_STEPS else None)

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
            receipt = captured.pop('native_cache_receipt', None)
            return PreparedNativeView(captured, receipt)
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
            pool = _package_file('native_cache_pool.py').read_text()
            cache = _package_file('preparation_cache.py').read_text()
            preamble = '\n'.join(['set -euo pipefail', 'export MINDIE_RUNTIME_ROOT=' + shlex.quote(root),
                                   *PYTHON_METADATA_PREAMBLE, *task_python_exports(python),
                                   'export PYTHONPATH=' + shlex.quote(':'.join([root + '/.mindie-runtime/metadata',
                                                                            root + '/vllm', root + '/vllm-ascend']))])
            command = (preamble + '\n' + shlex.quote(python) + ' - ' + shlex.quote(json.dumps(request))
                       + " <<'MINDIE_SHARED_NATIVE'\n" + module + '\nexec(' + repr(build_source) + ', globals())\n'
                       + 'exec(' + repr(incremental) + ', globals())\nexec(' + repr(pool) + ', globals())\n'
                       + 'exec(' + repr(cache) + ', globals())\n'
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
        for name in ('build_inputs.py', 'native_incremental.py', 'native_cache_pool.py', 'preparation_cache.py'):
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
            runner += 'exec(' + repr(_package_file('native_cache_pool.py').read_text()) + ', globals())\n'
            runner += 'exec(' + repr(_package_file('preparation_cache.py').read_text()) + ', globals())\n'
        runner += suffix
        preamble_lines = ["set -euo pipefail"]
        if native_recipe:
            preamble_lines.extend(['export MINDIE_RUNTIME_ROOT=' + shlex.quote(root), *DEFAULT_ENV_PREAMBLE, *task_python_exports(python),
                'export PYTHONPATH=' + shlex.quote(':'.join([root + '/.mindie-runtime/metadata', root + '/vllm', root + '/vllm-ascend'])) + '"${PYTHONPATH:+:$PYTHONPATH}"'])
        else:
            # The image's recorded launch settings can be reused, falling back
            # to safe Ascend set_env.sh bootstrap when no donor launch_env was captured.
            from mindie_coordinator.execution_slots import ASCEND_ENV_BOOTSTRAP_SH
            preamble_lines.append('unset PYTHONPATH ASCEND_CUSTOM_OPP_PATH')
            donor_env = spec.get('donor_launch_env') or {}
            if not donor_env.get('ASCEND_HOME_PATH'):
                preamble_lines.append(ASCEND_ENV_BOOTSTRAP_SH)
            for key, value in donor_env.items():
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
            reply = json.loads(output)
            store = reply.get('native_cache')
            if isinstance(store, dict):
                from mindie_coordinator.native_cache_pool import native_cache_receipt
                receipt = native_cache_receipt(store)
                if re.fullmatch(r'[0-9a-f]{64}', str(receipt.get('cache_key') or '')):
                    manifest['native_cache_receipt'] = receipt
            return {**manifest, 'container_id': info['Id'],
                    'launch_preamble': launch_preamble(manifest['profile'], python=python)}
