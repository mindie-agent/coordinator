"""Cold-clone hints are optional transport inputs, not source recapture."""
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import subprocess

import pytest

from mindie_coordinator import parity


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args]).decode('utf-8').strip()


def configure(repo, name='Test'):
    git(repo, 'config', 'user.name', name)
    git(repo, 'config', 'user.email', 'test@example.invalid')
    git(repo, 'config', 'core.autocrlf', 'false')


def capture(repo):
    return parity.build_snapshot_records(repo, 'test', 'capture', parity.DEFAULT_DENYLIST,
        source_roots={'project': repo}, with_build_inputs=False)


@pytest.fixture
def source(tmp_path, monkeypatch):
    # Exercise Windows CI's global checkout policy without changing user config.
    global_config = tmp_path / 'gitconfig'
    global_config.write_bytes(b'[core]\n\tautocrlf = true\n')
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', str(global_config))
    for key in list(os.environ):
        if key.startswith(('GIT_AUTHOR_', 'GIT_COMMITTER_')):
            monkeypatch.delenv(key)
    path = tmp_path / 'source'
    path.mkdir()
    git(path, 'init', '-q')
    configure(path)
    (path / 'operator.cpp').write_bytes(b'int value() { return 1; }\n')
    git(path, 'add', '.')
    git(path, 'commit', '-qm', 'initial')
    return path


def clone(source, path, child=None):
    # Fix checkout semantics before clone writes .gitmodules or child sources.
    git(path.parent, '-c', 'core.autocrlf=false', 'clone', '--shared', '-q', str(source), str(path))
    configure(path)
    if child:
        git(path, '-c', 'core.autocrlf=false', 'clone', '--shared', '-q', str(source / child), str(path / child))
        configure(path / child)
    return path


def add_child(source, name):
    child = source / name
    child.mkdir()
    git(child, 'init', '-q')
    configure(child)
    (child / 'source.h').write_bytes(b'#define VERSION 1\n')
    git(child, 'add', '.')
    git(child, 'commit', '-qm', 'child')
    git(source, 'config', '-f', '.gitmodules', 'submodule.child.path', name)
    git(source, 'config', '-f', '.gitmodules', 'submodule.child.url', './child')
    git(source, 'add', '.')
    git(source, 'commit', '-qm', 'submodule')


def state(repo):
    index = Path(git(repo, 'rev-parse', '--git-path', 'index'))
    if not index.is_absolute():
        index = repo / index
    files = {}
    for directory, dirs, names in os.walk(repo):
        dirs[:] = [name for name in dirs if name != '.git']
        for name in names:
            if name != '.git':
                path = Path(directory) / name
                files[path.relative_to(repo).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return git(repo, 'rev-parse', 'HEAD'), git(repo, 'for-each-ref'), index.read_bytes(), files


@pytest.mark.parametrize('child', [None, 'dep', '依赖'])
def test_shared_clone_without_retained_base_recovers_fixed_clean_snapshot(source, tmp_path, child):
    if child:
        add_child(source, child)
    baseline = capture(source)[-1]
    fresh = clone(source, tmp_path / 'fresh', child)
    assert not git(fresh, 'for-each-ref', 'refs/parity', 'refs/mindie')
    (fresh / 'operator.cpp').write_bytes(b'int value() { int output = 1; return output; }\n')
    record = capture(fresh)[-1]
    # Only the just-captured snapshot has a retained ref; it is not a base.
    assert git(fresh, 'for-each-ref', '--format=%(objectname)', 'refs/parity') == record.commit
    before = state(fresh)
    assert parity._local_snapshot_bases(record) == [baseline.commit]
    assert state(fresh) == before
    if child:
        assert git(fresh, 'rev-parse', record.source_head + '^{tree}') != baseline.tree
        assert git(fresh, 'ls-tree', baseline.commit, '--', child).split()[2] == record.submodules[0]['commit']


def test_fixed_gitlink_with_spaces_is_not_split_into_path_tokens(source):
    # Full recursive capture currently embeds relpaths in ref names, which
    # cannot contain spaces. Exercise this helper with valid fixed Git objects
    # directly, without changing that unrelated existing capture limitation.
    add_child(source, 'dependency space')
    child_commit = git(source / 'dependency space', 'rev-parse', 'HEAD')
    def fixed_record(label):
        record = parity.build_synthetic_snapshot(parity.RepoNode('project', source, None),
            workspace_id='test', snapshot_id=label, denylist=parity.DEFAULT_DENYLIST,
            child_commits={})
        return replace(record, source_path=str(source),
                       submodules=[{'path': 'dependency space', 'commit': child_commit}])
    baseline = fixed_record('clean')
    (source / 'operator.cpp').write_bytes(b'int value() { return 2; }\n')
    record = fixed_record('dirty')
    before = state(source)
    assert parity._clean_head_snapshot_base(record) == baseline.commit
    assert state(source) == before


def test_queued_input_does_not_read_later_head_workfiles_or_author_environment(source, tmp_path, monkeypatch):
    baseline = capture(source)[-1]
    fresh = clone(source, tmp_path / 'fresh')
    (fresh / 'operator.cpp').write_bytes(b'int value() { return 2; }\n')
    record = capture(fresh)[-1]
    git(fresh, 'add', '.')
    git(fresh, 'commit', '-qm', 'HEAD changed while input queued')
    (fresh / 'operator.cpp').write_bytes(b'int value() { return 3; }\n')
    monkeypatch.setenv('GIT_AUTHOR_NAME', 'Another author')
    monkeypatch.setenv('GIT_COMMITTER_DATE', '2026-01-01T00:00:00Z')
    before = state(fresh)
    assert parity._local_snapshot_bases(record) == [baseline.commit]
    assert state(fresh) == before
    assert git(fresh, 'show', record.commit + ':operator.cpp') == 'int value() { return 2; }'


def test_dirty_submodule_uses_its_admitted_commit_not_clean_or_later_child(source, tmp_path):
    add_child(source, 'dep')
    baseline = capture(source)[-1]
    fresh = clone(source, tmp_path / 'fresh', 'dep')
    (fresh / 'dep/source.h').write_bytes(b'#define VERSION 2\n')
    (fresh / 'operator.cpp').write_bytes(b'int value() { return 2; }\n')
    records = capture(fresh)
    child, record = records
    (fresh / 'dep/source.h').write_bytes(b'#define VERSION 3\n')
    before = state(fresh)
    hint = parity._clean_head_snapshot_base(record)
    assert hint and hint != baseline.commit
    assert git(fresh, 'ls-tree', hint, '--', 'dep').split()[2] == child.commit
    assert git(fresh / 'dep', 'show', child.commit + ':source.h') == '#define VERSION 2'
    assert git(fresh, 'show', hint + ':operator.cpp') == 'int value() { return 1; }'
    assert state(fresh) == before


def test_author_difference_is_not_silently_normalized_to_donor_identity(source, tmp_path):
    baseline = capture(source)[-1]
    fresh = clone(source, tmp_path / 'fresh')
    configure(fresh, 'Recipient')
    (fresh / 'operator.cpp').write_bytes(b'int value() { return 2; }\n')
    record = capture(fresh)[-1]
    hint = parity._clean_head_snapshot_base(record)
    assert hint != baseline.commit
    assert git(fresh, 'rev-parse', hint + '^{tree}') == baseline.tree
    assert 'author Recipient ' in git(fresh, 'cat-file', '-p', hint)


@pytest.mark.parametrize('damage', ['parented', 'message', 'missing-commit', 'missing-head',
                                  'missing-tree', 'missing-child', 'duplicate-child', 'wrong-child',
                                  'changed-gitmodules'])
def test_unsupported_or_missing_objects_leave_no_hint(source, damage):
    add_child(source, 'dep')
    (source / 'operator.cpp').write_bytes(b'int value() { return 2; }\n')
    record = capture(source)[-1]
    if damage in ('parented', 'message'):
        args = ['commit-tree', record.tree]
        if damage == 'parented':
            args += ['-p', record.source_head]
        message = parity.commit_message(record.relpath) if damage == 'parented' else 'unknown snapshot format'
        record = replace(record, commit=git(source, *args, '-m', message))
    elif damage == 'missing-commit':
        record = replace(record, commit='0' * 40)
    elif damage == 'missing-head':
        record = replace(record, source_head='0' * 40)
    elif damage == 'missing-tree':
        record = replace(record, tree='0' * 40)
    elif damage == 'missing-child':
        record = replace(record, submodules=[])
    elif damage == 'duplicate-child':
        record = replace(record, submodules=record.submodules * 2)
    elif damage == 'wrong-child':
        record = replace(record, submodules=[{**record.submodules[0], 'commit': '0' * 40}])
    else:
        with (source / '.gitmodules').open('a') as stream:
            stream.write('\n# changed after HEAD\n')
        record = capture(source)[-1]
    before = state(source)
    assert parity._clean_head_snapshot_base(record) is None
    assert state(source) == before


@pytest.mark.parametrize('kind', ['retained', 'clean', 'zero-limit'])
def test_existing_hint_or_clean_capture_has_no_extra_tree_discovery(source, monkeypatch, kind):
    baseline = capture(source)[-1]
    if kind != 'clean':
        (source / 'operator.cpp').write_bytes(b'int value() { return 2; }\n')
    record = capture(source)[-1]
    if kind == 'retained':
        git(source, 'update-ref', 'refs/mindie/inputs/old/project', baseline.commit)
    monkeypatch.setattr(parity, '_clean_head_snapshot_base',
                        lambda *args: pytest.fail('unnecessary base reconstruction'))
    assert parity._local_snapshot_bases(record, limit=0 if kind == 'zero-limit' else 64) == (
        [baseline.commit] if kind == 'retained' else [])


@pytest.mark.parametrize('error', [PermissionError('unreadable object store'),
                                 subprocess.TimeoutExpired('git', 5)])
def test_failed_optional_object_read_is_a_hint_miss(source, monkeypatch, error):
    record = capture(source)[-1]
    def unavailable(*args, **kwargs):
        raise error
    monkeypatch.setattr(parity.subprocess, 'run', unavailable)
    assert parity._clean_head_snapshot_base(record) is None
