"""Internal composition of fixed source materialization and native publication."""
from __future__ import annotations

from pathlib import Path
import shlex

from mindie_coordinator.runtime_profile import digest, launch_preamble


class NativeViewPublication:
    """One fixed donor/view pair, never a caller-selected remote callback."""

    def __init__(self, spec: dict, previous: dict, versions: dict):
        self.spec = spec
        self.previous = previous
        self.donor = {key: value for key, value in previous['attestation'].items()
                      if key not in {'container_id', 'launch_preamble'}}
        snapshot = spec['source_snapshot']
        self.request = {'root': spec['endpoint']['root'], 'source_root': previous['endpoint']['root'],
                        'versions': versions, 'source_id': snapshot['id'],
                        'build_env': snapshot.get('build_env', {}), 'preparation': spec['preparation'],
                        'build_inputs': {row['relpath']: row['build_inputs'] for row in snapshot['records']
                                         if row['relpath'] in ('vllm', 'vllm-ascend')},
                        'donor_manifest_digest': digest(self.donor)}
        package = Path(__file__).resolve().parent
        self.modules = '\n'.join('exec(' + repr((package / name).read_text(encoding='utf-8')) + ', globals())'
                                  for name in ('runtime_profile.py', 'build_inputs.py', 'preparation_cache.py'))

    def _publish_program(self) -> str:
        return (self.modules + '\n_native_args = ' + repr(self.request) + '\n'
                + "_native_source = Path(_native_args['source_root'])\n"
                + "_native_donor = json.loads((_native_source / '.mindie-runtime/ready-profile.json').read_text())\n"
                + "_native_reply = prepare_native_view(Path(_native_args['root']), _native_source, _native_donor, _native_args)\n")

    def _shell(self, program: str) -> str:
        return (launch_preamble(self.donor['profile'], python=self.previous['python']) + '\n'
                + shlex.quote(self.previous['python']) + " - <<'MINDIE_NATIVE_VIEW'\n"
                + program + '\nMINDIE_NATIVE_VIEW\n')

    def wrap_program(self, materialize_program: str) -> str:
        """The supplied package program sets result; missing never publishes."""
        publish = (self._publish_program() + "result['native_view'] = _native_reply\n"
                   + "result.setdefault('preparation_timings', {}).update(_native_reply['preparation_timings'])\n")
        body = (materialize_program + "\nif result.get('status') == 'materialized':\n"
                + '\n'.join('    ' + line for line in publish.splitlines())
                + '\nprint(json.dumps(result, separators=(",", ":")))\n')
        return self._shell(body)

    def accept(self, reply: dict) -> dict:
        manifest = {**reply['manifest'], 'files': self.donor['files']}
        if (reply.get('manifest_digest') != digest(manifest)
                or manifest.get('execution_view', {}).get('source_id') != self.request['source_id']
                or manifest.get('runtime_root') != self.request['root']):
            raise ValueError('native publication did not return the fixed execution view')
        return {**manifest, 'container_id': self.previous['attestation']['container_id'],
                'launch_preamble': launch_preamble(manifest['profile'], python=self.spec['python'])}
