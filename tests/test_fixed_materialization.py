"""Real local Git peers run the exact remote program without SSH or devices."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from mindie_coordinator import parity
from mindie_coordinator.execution_sources import source_identity


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def make_record(repo, name, *, submodules=()):
    commit = git(repo, 'rev-parse', 'HEAD')
    return parity.SnapshotRecord(name, name.replace('/', '__'), commit, None, commit,
        git(repo, 'rev-parse', 'HEAD^{tree}'), 'refs/inputs/' + commit, [], list(submodules),
        source_path=str(repo))


def snapshot(records):
    result = {'schema_version': 'mindie.execution-sources.v1', 'sources': {},
              'records': [asdict(row) for row in records], 'build_env': {}}
    result['id'] = source_identity(result)
    return result


@pytest.fixture
def peer(tmp_path, monkeypatch):
    if os.name == 'nt':
        pytest.skip('the remote Linux flock program runs in WSL, not native Windows')
    source = tmp_path / 'source'
    source.mkdir()
    git(source, 'init', '-q')
    git(source, 'config', 'user.name', 'Test')
    git(source, 'config', 'user.email', 'test@example.invalid')
    (source / 'model.py').write_text('value = 1\n')
    git(source, 'add', '.')
    git(source, 'commit', '-qm', 'initial')
    commands, transfers = [], []
    original_git = parity.git

    def stream(endpoint, script, **kwargs):
        commands.append(script)
        completed = subprocess.run(['bash', '-s'], input=script, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        return SimpleNamespace(stdout=completed.stdout)

    def transfer(repo, args, **kwargs):
        if args[0] == 'push':
            transfers.append(args)
        return original_git(repo, args, **kwargs)

    monkeypatch.setattr(parity, 'ssh_exec_stream', stream)
    monkeypatch.setattr(parity, 'git', transfer)
    monkeypatch.setattr(parity, 'git_remote_url', lambda endpoint, mirror: mirror)
    monkeypatch.setattr(parity, 'git_ssh_environment', lambda endpoint: os.environ.copy())
    def run(records=None, root='runtime', *, owner='test', shared_cache=None, host=None, build_source=None):
        return parity.materialize_fixed_sources(workspace_id=owner,
            endpoint={'host': 'fixture', 'port': 22, 'user': 'fixture', 'root': str(tmp_path / root)},
            source_snapshot=snapshot(records or [make_record(source, 'project')]),
            container_cache_root=str(tmp_path / 'cache'), shared_cache_root=shared_cache,
            host_endpoint=host, build_source=build_source)
    return SimpleNamespace(source=source, commands=commands, transfers=transfers, run=run, stream=stream,
                           cache=tmp_path / 'cache', root=tmp_path / 'runtime')


def test_build_metadata_is_written_only_by_completed_materialization(peer, monkeypatch):
    payload = {'versions': {'project': {'version': 'fixed', 'source_head': 'admitted'}},
               'build_env': {'FIXED_OPTION': 'yes'}}
    observed = []
    def stream(*args, **kwargs):
        result = peer.stream(*args, **kwargs)
        reply = json.loads(result.stdout)
        marker = peer.root / '.mindie-runtime/build-source.json'
        observed.append(reply['status'])
        if reply['status'] == 'missing':
            assert not marker.exists()
        else:
            assert json.loads(marker.read_text()) == payload
            assert reply['build_source_seconds'] >= 0
        return result
    monkeypatch.setattr(parity, 'ssh_exec_stream', stream)
    peer.run(build_source=payload)
    assert observed == ['missing', 'materialized']
    assert len(peer.commands) == 2
    assert not list((peer.root / '.mindie-runtime').glob('.build-source-*'))


def test_build_metadata_write_failure_does_not_return_success(peer):
    peer.run()
    marker = peer.root / '.mindie-runtime/build-source.json'
    marker.mkdir()
    with pytest.raises(RuntimeError):
        peer.run(build_source={'versions': {}})
    assert marker.is_dir()
    assert not list(marker.parent.glob('.build-source-*'))


def test_shared_hit_copies_only_exact_objects_to_private_owner(peer, tmp_path):
    record = make_record(peer.source, 'project')
    shared = str(tmp_path / 'shared')
    peer.run([record], shared_cache=shared)
    original_mirror = parity.mirror_path_for(str(peer.cache), 'test', record)
    original_refs = git(original_mirror, 'show-ref')
    peer.commands.clear()
    peer.transfers.clear()
    peer.run([record], root='other-root', owner='other-owner', shared_cache=shared)
    assert len(peer.commands) == 1 and not peer.transfers
    private = Path(parity.mirror_path_for(str(peer.cache), 'other-owner', record))
    assert git(original_mirror, 'show-ref') == original_refs
    assert set(git(private, 'for-each-ref', '--format=%(refname)').splitlines()) == {
        'refs/mindie/snapshots/' + record.commit, 'refs/parity/other-owner/transport-carrier'}
    assert git(Path(shared) / 'project.git', 'for-each-ref', '--format=%(refname)') == 'refs/mindie/snapshots/' + record.commit
    assert not (private / 'objects/info/alternates').exists()
    # Clearing a shared cache cannot change an already admitted private tree.
    import shutil
    shutil.rmtree(shared)
    assert git(private, 'cat-file', '-p', record.commit + ':model.py') == 'value = 1'
    assert git(peer.root.parent / 'other-root/project', 'show', 'HEAD:model.py') == 'value = 1'


def test_cold_owner_automatically_exports_legacy_exact_objects(peer, tmp_path, monkeypatch):
    from mindie_coordinator.shared_source_objects import export_container_objects
    record = make_record(peer.source, 'project')
    peer.run([record])  # Historical private mirror, no prior shared publication.
    shared = str(tmp_path / 'shared')
    calls = []
    def export(host, rows, cache):
        calls.append(rows)
        result = export_container_objects({'records': rows, 'legacy_cache': cache})
        return {'status': 'copied', **result}
    monkeypatch.setattr(parity, '_export_existing_source_objects', export)
    peer.commands.clear()
    peer.transfers.clear()
    peer.run([record], root='cold-owner', owner='cold', shared_cache=shared, host={'host': 'fixture'})
    assert len(peer.commands) == 2 and not peer.transfers
    assert len(calls) == 1 and [r['commit'] for r in calls[0]] == [record.commit]
    # Its next Python edit uses the imported exact snapshot as a delta base,
    # with no repeated host discovery and no full Git upload.
    (peer.source / 'model.py').write_text('value = 2\n')
    git(peer.source, 'commit', '-am', 'next Python edit')
    peer.commands.clear()
    peer.run([make_record(peer.source, 'project')], root='cold-edit', owner='cold',
             shared_cache=shared, host={'host': 'fixture'})
    assert len(calls) == 1 and len(peer.commands) == 2 and not peer.transfers


def test_cold_owner_small_edit_uses_shared_fixed_base_without_full_push(peer, tmp_path, monkeypatch):
    first = make_record(peer.source, 'project')
    shared = str(tmp_path / 'shared')
    peer.run([first], shared_cache=shared)
    (peer.source / 'model.py').write_text('value = 2\n')
    git(peer.source, 'commit', '-am', 'new recipient edit')
    second = make_record(peer.source, 'project')
    def unexpected_discovery(*args):
        pytest.fail('shared fixed base should avoid host discovery')
    monkeypatch.setattr(parity, '_export_existing_source_objects', unexpected_discovery)
    peer.commands.clear()
    peer.transfers.clear()
    result = peer.run([second], root='new-owner-edit', owner='other', shared_cache=shared,
                      host={'host': 'fixture'})
    assert len(peer.commands) == 2 and not peer.transfers
    assert result['commits'] == {'project': second.commit}
    assert (peer.root.parent / 'new-owner-edit/project/model.py').read_text() == 'value = 2\n'
    assert (peer.root / 'project/model.py').read_text() == 'value = 1\n'
    private = parity.mirror_path_for(str(peer.cache), 'other', second)
    assert git(private, 'rev-parse', 'refs/mindie/snapshots/' + first.commit) == first.commit


@pytest.mark.parametrize('foreign_snapshot', [False, True])
def test_disconnected_snapshots_pack_only_the_cpp_edit_for_a_new_owner(peer, tmp_path, monkeypatch,
                                                                     foreign_snapshot):
    import random
    # Unchanged data must exceed the inline limit. A tiny repository can send
    # its entire tree and accidentally make the delta test appear successful.
    (peer.source / 'unchanged.bin').write_bytes(random.Random(17).randbytes(256 * 1024))
    cpp = peer.source / 'operator.cpp'
    cpp.write_text('int Operator(int input) { return input + 1; }\n')
    git(peer.source, 'add', '.')
    git(peer.source, 'commit', '-qm', 'baseline input')
    def fixed_record():
        record = make_record(peer.source, 'project')
        commit = subprocess.check_output(['git', '-C', str(peer.source), 'commit-tree', record.tree,
            '-m', 'fixed tree snapshot'], text=True,
            env={**os.environ, 'GIT_AUTHOR_DATE': '1970-01-01T00:00:00Z',
                 'GIT_COMMITTER_DATE': '1970-01-01T00:00:00Z'}).strip()
        assert 'parent ' not in git(peer.source, 'cat-file', '-p', commit)
        return replace(record, commit=commit, ref='refs/inputs/' + commit)
    first = fixed_record()
    shared = str(tmp_path / 'shared')
    peer.run([first], shared_cache=shared)
    donor = parity.mirror_path_for(str(peer.cache), 'test', first)
    donor_refs = git(donor, 'show-ref')
    if foreign_snapshot:
        from mindie_coordinator.shared_source_objects import copy_fixed_objects
        foreign = tmp_path / 'independent-source'
        git(tmp_path, 'clone', '--local', str(peer.source), str(foreign))
        (foreign / 'marker.py').write_text("marker = 'another source copy'\n")
        git(foreign, 'add', '.')
        foreign_tree = git(foreign, 'write-tree')
        unknown = subprocess.check_output(['git', '-C', str(foreign), 'commit-tree', foreign_tree,
            '-m', 'independent parentless marker'], text=True,
            env={**os.environ, 'GIT_AUTHOR_NAME': 'Test', 'GIT_AUTHOR_EMAIL': 'test@example.invalid',
                 'GIT_COMMITTER_NAME': 'Test', 'GIT_COMMITTER_EMAIL': 'test@example.invalid',
                 'GIT_AUTHOR_DATE': '1970-01-01T00:00:01Z',
                 'GIT_COMMITTER_DATE': '1970-01-01T00:00:01Z'}).strip()
        copy_fixed_objects(foreign, Path(shared) / 'project.git', {'commit': unknown, 'tree': foreign_tree})
        assert 'parent ' not in git(foreign, 'cat-file', '-p', unknown)
        assert parity.git(peer.source, ['cat-file', '-e', unknown], check=False).returncode
        assert git(Path(shared) / 'project.git', 'for-each-ref', '--count=1', '--sort=-creatordate',
                   '--format=%(objectname)', 'refs/mindie/snapshots/') == unknown
        # Keep the older fixed input ref, without a local transport association
        # for this recipient. A different source copy must not hide that base.
        git(peer.source, 'update-ref', 'refs/mindie/inputs/old-source/project', first.commit)
        for ref in git(peer.source, 'for-each-ref', '--format=%(refname)', 'refs/parity-transport').splitlines():
            git(peer.source, 'update-ref', '-d', ref)
    cpp.write_text('int Operator(int recipient_op) { return recipient_op + 1; }\n')
    git(peer.source, 'commit', '-am', 'one operator variable edit')
    second = fixed_record()
    assert first.commit in parity._local_snapshot_bases(second)
    assert parity._fixed_inline_pack(peer.source, second.commit, second.commit, first.commit) is None
    packs = []
    original = parity._fixed_inline_pack
    def capture(*args, **kwargs):
        pack = original(*args, **kwargs)
        packs.append(pack)
        return pack
    monkeypatch.setattr(parity, '_fixed_inline_pack', capture)
    peer.commands.clear()
    peer.transfers.clear()
    result = peer.run([second], root='new-owner', owner='recipient', shared_cache=shared)
    assert len(peer.commands) == 2 and not peer.transfers
    assert len(packs) == 1 and packs[0]['bytes'] < 16 * 1024
    carrier = packs[0]['carrier']
    assert git(peer.source, 'rev-parse', carrier + '^') == first.commit
    assert git(peer.source, 'rev-parse', carrier + '^{tree}') == second.tree
    assert carrier != second.commit
    assert result['commits'] == {'project': second.commit}
    assert git(peer.root.parent / 'new-owner/project', 'rev-parse', 'HEAD') == second.commit
    assert git(donor, 'show-ref') == donor_refs


def test_unknown_local_shared_base_keeps_normal_git_fallback(peer, tmp_path):
    first = make_record(peer.source, 'project')
    shared = str(tmp_path / 'shared')
    peer.run([first], shared_cache=shared)
    # A different local history cannot construct an inline pack excluding the
    # shared SHA, even though its private remote now has that negotiation base.
    unrelated = tmp_path / 'unrelated'
    unrelated.mkdir()
    git(unrelated, 'init', '-q')
    git(unrelated, 'config', 'user.name', 'Test')
    git(unrelated, 'config', 'user.email', 'test@example.invalid')
    (unrelated / 'model.py').write_text('different admitted input\n')
    git(unrelated, 'add', '.')
    git(unrelated, 'commit', '-qm', 'unrelated source')
    record = make_record(unrelated, 'project')
    peer.commands.clear()
    peer.transfers.clear()
    result = peer.run([record], root='unrelated-owner', owner='other', shared_cache=shared)
    assert len(peer.commands) == 2 and len(peer.transfers) == 1
    assert result['commits'] == {'project': record.commit}
    assert (peer.root.parent / 'unrelated-owner/project/model.py').read_text() == 'different admitted input\n'


@pytest.mark.parametrize('different_author', [False, True])
def test_fresh_shared_clone_hint_preserves_queued_edit_and_uses_verified_base(
        peer, tmp_path, monkeypatch, different_author):
    import random
    from test_clean_snapshot_base import capture, clone, configure

    (peer.source / 'unchanged.bin').write_bytes(random.Random(31).randbytes(256 * 1024))
    (peer.source / 'operator.cpp').write_bytes(b'int value() { return 1; }\n')
    git(peer.source, 'add', '.')
    git(peer.source, 'commit', '-qm', 'baseline source')
    first = capture(peer.source)[-1]
    shared = str(tmp_path / 'shared')
    peer.run([first], shared_cache=shared)
    donor = parity.mirror_path_for(str(peer.cache), 'test', first)
    donor_refs = git(donor, 'show-ref')
    fresh = clone(peer.source, tmp_path / 'fresh-source')
    if different_author:
        configure(fresh, 'Recipient')
    (fresh / 'operator.cpp').write_bytes(b'int value() { return 2; }\n')
    admitted = capture(fresh)[-1]
    # The queue delay must not cause either hint generation or materialization
    # to incorporate a later HEAD or a second dirty edit.
    git(fresh, 'add', '.')
    git(fresh, 'commit', '-qm', 'later HEAD')
    (fresh / 'operator.cpp').write_bytes(b'int value() { return 3; }\n')
    packs = []
    original = parity._fixed_inline_pack
    def record_pack(*args, **kwargs):
        pack = original(*args, **kwargs)
        packs.append(pack)
        return pack
    monkeypatch.setattr(parity, '_fixed_inline_pack', record_pack)
    peer.commands.clear()
    peer.transfers.clear()
    result = peer.run([admitted], root='fresh-owned', owner='fresh', shared_cache=shared)
    assert len(peer.commands) == 2
    if different_author:
        # Equal file trees alone do not establish the old commit's identity.
        # No common immutable commit is found, so the existing Git path runs.
        assert len(peer.transfers) == 1 and packs == [None]
    else:
        assert not peer.transfers and len(packs) == 1
        assert packs[0]['previous'] == first.commit and packs[0]['bytes'] < 16 * 1024
    assert result['commits'] == {'project': admitted.commit}
    target = peer.root.parent / 'fresh-owned/project'
    assert git(target, 'rev-parse', 'HEAD') == admitted.commit
    assert (target / 'operator.cpp').read_bytes() == b'int value() { return 2; }\n'
    assert (fresh / 'operator.cpp').read_bytes() == b'int value() { return 3; }\n'
    assert git(donor, 'show-ref') == donor_refs


def test_shared_publications_are_serialized_and_keep_both_snapshots(peer, tmp_path):
    from mindie_coordinator.shared_source_objects import copy_fixed_objects
    first = make_record(peer.source, 'project')
    (peer.source / 'model.py').write_text('value = 2\n')
    git(peer.source, 'commit', '-am', 'second')
    second = make_record(peer.source, 'project')
    shared = tmp_path / 'shared.git'
    with ThreadPoolExecutor(2) as executor:
        results = list(executor.map(lambda record: copy_fixed_objects(peer.source, shared, asdict(record)),
                                    [first, second]))
    assert results == [True, True]
    assert set(git(shared, 'for-each-ref', '--format=%(objectname)').splitlines()) == {first.commit, second.commit}
    assert not (shared / 'objects/info/alternates').exists()


def test_shared_wrong_tree_and_missing_blob_never_publish_success(peer, tmp_path):
    from mindie_coordinator.shared_source_objects import copy_fixed_objects
    record = make_record(peer.source, 'project')
    shared = tmp_path / 'shared.git'
    with pytest.raises(ValueError, match='wrong tree'):
        copy_fixed_objects(peer.source, shared, {**asdict(record), 'tree': '0' * 40})
    assert not shared.exists()
    assert not copy_fixed_objects(tmp_path / 'absent', shared, asdict(record))
    blob = git(peer.source, 'rev-parse', 'HEAD:model.py')
    (peer.source / '.git/objects' / blob[:2] / blob[2:]).unlink()
    with pytest.raises(RuntimeError, match='fixed object copy failed'):
        copy_fixed_objects(peer.source, shared, asdict(record))
    assert not git(shared, 'for-each-ref', '--format=%(refname)')


def test_uncertain_legacy_export_does_not_fall_back_to_upload(peer, tmp_path, monkeypatch):
    from mindie_coordinator.preparation_process import PreparationUncertain
    def unknown(*args):
        raise PreparationUncertain('lost export reply')
    monkeypatch.setattr(parity, '_export_existing_source_objects', unknown)
    with pytest.raises(PreparationUncertain, match='lost export reply'):
        peer.run(shared_cache=str(tmp_path / 'shared'), host={'host': 'fixture'})
    assert len(peer.commands) == 1 and not peer.transfers
    assert not (peer.root / '.mindie-runtime/source-materialization.json').exists()


@pytest.mark.parametrize('failure', [PermissionError('read only'), OSError('no space'),
                                     ValueError('corrupt shared cache')])
def test_private_snapshot_does_not_require_shared_publication(peer, tmp_path, monkeypatch, capsys, failure):
    from mindie_coordinator import parity_support
    record = make_record(peer.source, 'project')
    peer.run([record])
    def fail(*args):
        raise failure
    monkeypatch.setattr(parity_support, 'copy_fixed_objects', fail)
    runtime = tmp_path / 'private-valid'
    result = parity_support._materialize_fixed({
        'root': str(runtime), 'source_id': snapshot([record])['id'],
        'carrier_ref': 'refs/parity/test/transport-carrier',
        'records': [{**asdict(record), 'mirror': parity.mirror_path_for(str(peer.cache), 'test', record),
                     'shared_mirror': str(tmp_path / 'shared.git')}],
    })
    assert result['status'] == 'materialized'
    assert (runtime / 'project/model.py').read_text() == 'value = 1\n'
    assert 'shared source publication unavailable' in capsys.readouterr().err


def test_bad_shared_candidate_falls_back_to_verified_git_upload(peer, tmp_path):
    record = make_record(peer.source, 'project')
    shared = tmp_path / 'shared'
    shared.mkdir()
    broken = shared / 'project.git'
    git(shared, 'clone', '--bare', '--local', str(peer.source), str(broken))
    blob = git(peer.source, 'rev-parse', 'HEAD:model.py')
    (broken / 'objects' / blob[:2] / blob[2:]).unlink()
    result = peer.run([record], owner='cold', shared_cache=str(shared))
    assert result['status'] == 'materialized' and len(peer.transfers) == 1
    assert (peer.root / 'project/model.py').read_text() == 'value = 1\n'


def test_private_snapshot_does_not_swallow_shared_copy_timeout(peer, tmp_path, monkeypatch):
    from mindie_coordinator import parity_support
    record = make_record(peer.source, 'project')
    peer.run([record])
    def timeout(*args):
        raise subprocess.TimeoutExpired('git fetch', 60)
    monkeypatch.setattr(parity_support, 'copy_fixed_objects', timeout)
    runtime = tmp_path / 'unknown-copy'
    with pytest.raises(subprocess.TimeoutExpired):
        parity_support._materialize_fixed({
            'root': str(runtime), 'source_id': snapshot([record])['id'],
            'carrier_ref': 'refs/parity/test/transport-carrier',
            'records': [{**asdict(record), 'mirror': parity.mirror_path_for(str(peer.cache), 'test', record),
                         'shared_mirror': str(tmp_path / 'shared.git')}],
        })
    assert not (runtime / '.mindie-runtime/source-materialization.json').exists()


def test_invalid_legacy_mirror_does_not_hide_valid_donor(peer, tmp_path, monkeypatch):
    from mindie_coordinator import shared_source_objects as shared
    record = make_record(peer.source, 'project')
    parent = tmp_path / 'legacy/workspaces'
    broken = parent / 'first/mirrors/nested/project.git'
    valid = parent / 'second/mirrors/nested/project.git'
    for mirror in (broken, valid):
        mirror.parent.mkdir(parents=True)
        git(mirror.parent, 'clone', '--bare', '--local', str(peer.source), str(mirror))
    blob = git(peer.source, 'rev-parse', 'HEAD:model.py')
    (broken / 'objects' / blob[:2] / blob[2:]).unlink()
    original = shared.copy_fixed_objects
    tried = []
    def copy(source, *args):
        tried.append(source)
        return original(source, *args)
    monkeypatch.setattr(shared, 'copy_fixed_objects', copy)
    # Force candidate order; filesystem directory enumeration is unspecified.
    original_glob = Path.glob
    monkeypatch.setattr(Path, 'glob', lambda path, pattern: iter([broken, valid])
                        if path == parent else original_glob(path, pattern))
    result = shared.export_container_objects({'legacy_cache': str(tmp_path / 'legacy'),
        'records': [{**asdict(record), 'shared_mirror': str(tmp_path / 'shared.git')}]})
    assert result['copied'] == [record.commit] and tried == [broken, valid]
    assert result['diagnostics']


def test_cold_two_operations_warm_one_without_recapture(peer):
    record = make_record(peer.source, 'project')
    result = peer.run([record])
    assert result['status'] == 'materialized'
    assert len(peer.commands) == 2 and len(peer.transfers) == 1
    assert peer.transfers[0][-2].startswith(record.commit + ':refs/mindie/snapshots/')
    peer.commands.clear()
    peer.transfers.clear()
    # Current user files and even local retained refs are irrelevant on a hit.
    (peer.source / 'model.py').write_text('new unsubmitted bytes\n')
    result = peer.run([record], root='second')
    assert len(peer.commands) == 1 and not peer.transfers
    assert (peer.root.parent / 'second/project/model.py').read_text() == 'value = 1\n'
    assert result['commits'] == {'project': record.commit}
    mirror = Path(parity.mirror_path_for(str(peer.cache), 'test', record))
    local_objects = peer.root.parent / 'second/project/.git/objects'
    objects = [path for path in (mirror / 'objects').rglob('*') if path.is_file()]
    assert objects and any((local_objects / path.relative_to(mirror / 'objects')).stat().st_ino
                           == path.stat().st_ino for path in objects)


def test_concurrent_roots_ignore_mutable_mirror_refs(peer):
    first = make_record(peer.source, 'project')
    peer.run([first])
    (peer.source / 'model.py').write_text('value = 2\n')
    git(peer.source, 'commit', '-am', 'second')
    second = make_record(peer.source, 'project')
    peer.run([second], root='second')
    mirror = parity.mirror_path_for(str(peer.cache), 'test', first)
    git(mirror, 'update-ref', 'refs/heads/parity-current', second.commit)
    git(mirror, 'update-ref', 'refs/parity/test/transport-carrier', second.commit)
    peer.commands.clear()
    with ThreadPoolExecutor(2) as executor:
        results = list(executor.map(lambda pair: peer.run([pair[0]], root=pair[1]),
                                    [(first, 'parallel-one'), (second, 'parallel-two')]))
    assert [row['commits']['project'] for row in results] == [first.commit, second.commit]
    assert len(peer.commands) == 2
    assert git(mirror, 'rev-parse', 'refs/mindie/snapshots/' + first.commit) == first.commit


def test_missing_reply_does_not_touch_root_and_transfer_failure_keeps_mirror(peer, monkeypatch):
    # Keep exercising the large/cold Git transport fallback rather than the
    # small edit path carried by the existing owned preparation operation.
    monkeypatch.setattr(parity, '_fixed_inline_pack', lambda *args, **kwargs: None)
    first = make_record(peer.source, 'project')
    peer.run([first])
    receipt = peer.root / '.mindie-runtime/source-materialization.json'
    before = receipt.read_bytes()
    (peer.source / 'model.py').write_text('value = 2\n')
    git(peer.source, 'commit', '-am', 'second')
    second = make_record(peer.source, 'project')
    original = parity.git
    def fail(repo, args, **kwargs):
        if args[0] == 'push':
            assert receipt.read_bytes() == before
            raise RuntimeError('injected failed transfer')
        return original(repo, args, **kwargs)
    monkeypatch.setattr(parity, 'git', fail)
    with pytest.raises(RuntimeError, match='injected failed transfer'):
        peer.run([second])
    assert receipt.read_bytes() == before
    mirror = parity.mirror_path_for(str(peer.cache), 'test', first)
    assert git(mirror, 'rev-parse', 'refs/mindie/snapshots/' + first.commit) == first.commit


def test_small_edit_uses_owned_rpc_pack_and_preserves_previous_root(peer):
    first = make_record(peer.source, 'project')
    peer.run([first])
    before = (peer.root / '.mindie-runtime/source-materialization.json').read_bytes()
    (peer.source / 'model.py').write_text('value = 2\n')
    git(peer.source, 'commit', '-am', 'second')
    second = make_record(peer.source, 'project')
    peer.commands.clear()
    peer.transfers.clear()
    result = peer.run([second], root='second')
    assert result['commits'] == {'project': second.commit}
    assert len(peer.commands) == 2 and not peer.transfers
    assert "'inline_packs':" in peer.commands[-1]
    assert (peer.root.parent / 'second/project/model.py').read_text() == 'value = 2\n'
    assert (peer.root / 'project/model.py').read_text() == 'value = 1\n'
    assert (peer.root / '.mindie-runtime/source-materialization.json').read_bytes() == before
    mirror = parity.mirror_path_for(str(peer.cache), 'test', first)
    for row in (first, second):
        assert git(mirror, 'rev-parse', 'refs/mindie/snapshots/' + row.commit) == row.commit


@pytest.mark.parametrize('damage, message', [('digest', 'size or digest differs'),
                                            ('base', 'prerequisite disappeared')])
def test_invalid_inline_pack_never_materializes_or_changes_pins(peer, monkeypatch, damage, message):
    first = make_record(peer.source, 'project')
    peer.run([first])
    (peer.source / 'model.py').write_text('value = 2\n')
    git(peer.source, 'commit', '-am', 'second')
    second = make_record(peer.source, 'project')
    original = parity._fixed_inline_pack
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is not None
        result['sha256' if damage == 'digest' else 'previous'] = '0' * (64 if damage == 'digest' else 40)
        return result
    monkeypatch.setattr(parity, '_fixed_inline_pack', corrupt)
    peer.commands.clear()
    peer.transfers.clear()
    with pytest.raises(RuntimeError, match=message):
        peer.run([second], root='second')
    assert len(peer.commands) == 2 and not peer.transfers
    assert not (peer.root.parent / 'second/.mindie-runtime/source-materialization.json').exists()
    assert not (peer.root.parent / 'second/project').exists()
    mirror = parity.mirror_path_for(str(peer.cache), 'test', first)
    assert git(mirror, 'rev-parse', 'refs/mindie/snapshots/' + first.commit) == first.commit
    assert subprocess.run(['git', '-C', mirror, 'show-ref', '--verify',
                           'refs/mindie/snapshots/' + second.commit], capture_output=True).returncode


def test_unknown_inline_materialization_is_not_replayed(peer, monkeypatch):
    peer.run()
    (peer.source / 'model.py').write_text('value = 2\n')
    git(peer.source, 'commit', '-am', 'second')
    calls = []
    def uncertain(endpoint, script, **kwargs):
        calls.append(script)
        if len(calls) == 2:
            raise RuntimeError('owned process outcome is unknown')
        return peer.stream(endpoint, script, **kwargs)
    monkeypatch.setattr(parity, 'ssh_exec_stream', uncertain)
    peer.transfers.clear()
    with pytest.raises(RuntimeError, match='outcome is unknown'):
        peer.run(root='second')
    assert len(calls) == 2 and not peer.transfers


@pytest.mark.parametrize('size', [100000, 300000])
def test_large_edit_keeps_git_transport_fallback(peer, size):
    peer.run()
    (peer.source / 'large.bin').write_bytes(os.urandom(size))
    git(peer.source, 'add', '.')
    git(peer.source, 'commit', '-qm', 'large new blob')
    peer.commands.clear()
    peer.transfers.clear()
    peer.run(root='second')
    assert len(peer.commands) == 2 and len(peer.transfers) == 1
    assert (peer.root.parent / 'second/project/large.bin').read_bytes() == (peer.source / 'large.bin').read_bytes()


def test_multiple_inline_packs_respect_total_command_argument_limit(peer):
    other = peer.source.parent / 'other-source'
    git(peer.source.parent, 'clone', '-q', str(peer.source), str(other))
    git(other, 'config', 'user.name', 'Test')
    git(other, 'config', 'user.email', 'test@example.invalid')
    names = [(peer.source, 'project'), (other, 'other')]
    peer.run([make_record(path, name) for path, name in names])
    for path, _ in names:
        (path / 'new.bin').write_bytes(os.urandom(50000))
        git(path, 'add', '.')
        git(path, 'commit', '-qm', 'medium edit')
    peer.commands.clear()
    peer.transfers.clear()
    peer.run([make_record(path, name) for path, name in names], root='second')
    assert len(peer.commands) == 2 and len(peer.transfers) == 1
    assert len(peer.commands[-1].encode()) <= 96 * 1024
    # Exercise the exact worker invocation: argv, not bash stdin.
    result = subprocess.run(['bash', '-c', peer.commands[-1]], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for path, name in names:
        assert (peer.root.parent / 'second' / name / 'new.bin').read_bytes() == (path / 'new.bin').read_bytes()


def test_retry_same_inputs_repairs_tracked_and_untracked_files(peer):
    record = make_record(peer.source, 'project')
    peer.run([record])
    (peer.root / 'project/model.py').write_text('damaged\n')
    (peer.root / 'project/extra.py').write_text('untracked\n')
    peer.run([record])
    assert (peer.root / 'project/model.py').read_text() == 'value = 1\n'
    assert not (peer.root / 'project/extra.py').exists()


def test_different_inputs_cannot_overwrite_owned_root(peer):
    first = make_record(peer.source, 'project')
    peer.run([first])
    (peer.source / 'model.py').write_text('value = 2\n')
    git(peer.source, 'commit', '-am', 'second')
    with pytest.raises(RuntimeError, match='belongs to different inputs'):
        peer.run()
    assert git(peer.root / 'project', 'rev-parse', 'HEAD') == first.commit


def test_submodule_checkout_is_complete_before_parent_validation(peer):
    child = peer.source.parent / 'child'
    git(peer.source.parent, 'clone', '-q', str(peer.source), str(child))
    git(peer.source, '-c', 'protocol.file.allow=always', 'submodule', 'add', str(child), 'nested')
    git(peer.source, 'commit', '-am', 'submodule')
    parent = make_record(peer.source, 'project', submodules=[{'name': 'nested', 'path': 'nested'}])
    nested = make_record(peer.source / 'nested', 'project/nested')
    peer.run([nested, parent])
    assert git(peer.root / 'project', 'status', '--porcelain') == ''
    assert git(peer.root / 'project/nested', 'rev-parse', 'HEAD') == nested.commit


def test_verification_failure_never_publishes_receipt(peer, monkeypatch):
    record = make_record(peer.source, 'project')
    peer.run([record], root='donor')
    def corrupt(endpoint, script, **kwargs):
        script = script.replace('observed = {}',
            "(root / 'project/model.py').write_text('corrupted')\n        observed = {}")
        return peer.stream(endpoint, script, **kwargs)
    monkeypatch.setattr(parity, 'ssh_exec_stream', corrupt)
    with pytest.raises(RuntimeError, match='git diff failed'):
        peer.run([record])
    assert not (peer.root / '.mindie-runtime/source-materialization.json').exists()


def test_owned_missing_and_materialize_use_distinct_durable_jobs(peer, monkeypatch):
    from mindie_coordinator import preparation_process
    from mindie_coordinator.parity_support import ssh_exec_stream
    jobs, saved = [], []
    def control(endpoint, job_id, action, **kwargs):
        assert action == 'launch'
        assert saved[-1]['job_id'] == job_id and saved[-1]['state'] == 'pending'
        jobs.append(job_id)
        # The real remote-dev worker persists its job before launching the
        # command, including the first missing-object probe.
        job_dir = Path(endpoint['root']) / '.remote-dev/jobs' / job_id
        job_dir.mkdir(parents=True)
        (job_dir / 'receipt.json').write_text(json.dumps({'job_id': job_id, 'owned': True}))
        script = kwargs['spec']['command']
        result = subprocess.run(['bash', '-s'], input=script, capture_output=True, text=True)
        return {'quiet': True, 'state': 'completed', 'result': {'exit_code': result.returncode},
                'stdout': result.stdout, 'stderr': result.stderr}
    monkeypatch.setattr(preparation_process, 'control', control)
    monkeypatch.setattr(parity, 'ssh_exec_stream', ssh_exec_stream)
    endpoint = {'host': 'fixture', 'port': 22, 'user': 'fixture',
                'root': str(peer.root), 'cwd': str(peer.root)}
    process = preparation_process.PreparationProcess(endpoint, 'materialize',
        lambda row: saved.append(dict(row)), lambda: False)
    result = parity.materialize_fixed_sources(workspace_id='test', endpoint=endpoint,
        source_snapshot=snapshot([make_record(peer.source, 'project')]),
        container_cache_root=str(peer.cache), process=process)
    assert result['status'] == 'materialized'
    assert len(jobs) == 2 and len(set(jobs)) == 2
    assert saved[-1]['quiet']
    for job_id in jobs:
        receipt = peer.root / '.remote-dev/jobs' / job_id / 'receipt.json'
        assert json.loads(receipt.read_text()) == {'job_id': job_id, 'owned': True}


@pytest.mark.parametrize('name', ['.remote-dev', '.remote-dev/nested', '.mindie-runtime', '.venv', '.git'])
def test_sources_cannot_overlap_owned_runtime_directories(tmp_path, monkeypatch, name):
    row = parity.SnapshotRecord(name, name.replace('/', '__'), 'a' * 40, None, 'a' * 40,
                               'b' * 40, 'refs/input', [], [], source_path=str(tmp_path))
    def unexpected(*args, **kwargs):
        raise AssertionError('reserved source paths must fail before remote work')
    monkeypatch.setattr(parity, 'ssh_exec_stream', unexpected)
    with pytest.raises(ValueError, match='overlaps coordinator-owned runtime files'):
        parity.materialize_fixed_sources(workspace_id='test',
            endpoint={'host': 'fixture', 'port': 22, 'user': 'fixture', 'root': '/isolated/run'},
            source_snapshot=snapshot([row]))


def test_preserves_environment_and_ignored_native_outputs(peer):
    (peer.source / '.gitignore').write_text('*.so\n')
    git(peer.source, 'add', '.')
    git(peer.source, 'commit', '-qm', 'native output ignore')
    record = make_record(peer.source, 'project')
    peer.run([record])
    native = peer.root / 'project/native.so'
    native.write_bytes(b'native-output')
    environment = peer.root / '.venv'
    environment.mkdir()
    (environment / 'interpreter').write_bytes(b'owned-environment')
    peer.run([record])
    assert native.read_bytes() == b'native-output'
    assert (environment / 'interpreter').read_bytes() == b'owned-environment'


@pytest.mark.parametrize('reply', [None, 'broken', {}, {'status': 'missing', 'missing': []},
                                  {'status': 'materialized', 'source_id': 'wrong'}])
def test_invalid_or_uncertain_reply_is_not_replayed(tmp_path, monkeypatch, reply):
    row = parity.SnapshotRecord('project', 'project', 'a' * 40, None, 'a' * 40, 'b' * 40,
                               'refs/input', [], [], source_path=str(tmp_path))
    calls = []
    def stream(*args, **kwargs):
        calls.append(args)
        if reply is None:
            raise RuntimeError('owned process outcome is unknown')
        return SimpleNamespace(stdout=reply if isinstance(reply, str) else json.dumps(reply))
    monkeypatch.setattr(parity, 'ssh_exec_stream', stream)
    with pytest.raises(RuntimeError):
        parity.materialize_fixed_sources(workspace_id='test',
            endpoint={'host': 'fixture', 'port': 22, 'user': 'fixture', 'root': '/isolated/run'},
            source_snapshot=snapshot([row]))
    assert len(calls) == 1
