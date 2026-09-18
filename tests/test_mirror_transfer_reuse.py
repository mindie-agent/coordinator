"""Live local Git peers exercise the remote mirror protocol without SSH."""
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_coordinator import parity


def run_git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def snapshot(repo, name):
    head = run_git(repo, 'rev-parse', 'HEAD')
    ref = 'refs/inputs/' + head
    run_git(repo, 'update-ref', ref, head)
    return parity.SnapshotRecord(name, name, head, None, head,
                                 run_git(repo, 'rev-parse', 'HEAD^{tree}'), ref, [], [])


@pytest.fixture
def local_peer(monkeypatch):
    bash = str(Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Git/bin/bash.exe') if os.name == 'nt' else shutil.which('bash')
    if not bash or not Path(bash).is_file():
        pytest.skip('local remote-shell fixture requires Bash')
    commands, transfers = [], []

    def shell(endpoint, script):
        commands.append(script)
        result = subprocess.run([bash, '-s'], input=script.encode(), capture_output=True, check=True)
        return SimpleNamespace(stdout=result.stdout.decode())

    original_git = parity.git

    def git(repo, args, **kwargs):
        if args[0] in {'push', 'ls-remote'}:
            transfers.append(args)
        return original_git(repo, args, **kwargs)

    monkeypatch.setattr(parity, 'ssh_exec', shell)
    monkeypatch.setattr(parity, 'git', git)
    monkeypatch.setattr(parity, 'git_remote_url', lambda endpoint, mirror: mirror)
    monkeypatch.setattr(parity, 'git_ssh_environment', lambda endpoint: os.environ.copy())
    return commands, transfers


def test_batch_probe_skips_unchanged_repos_and_retries_partial_transfer(tmp_path, local_peer):
    commands, transfers = local_peer
    endpoint = parity.SshEndpoint('local-fixture', 22, 'fixture')
    repos, records, mirrors = {}, {}, {}
    for name in ('vllm', 'vllm-ascend'):
        repo = tmp_path / name
        repo.mkdir()
        run_git(repo, 'init', '-q')
        run_git(repo, 'config', 'user.name', 'Test')
        run_git(repo, 'config', 'user.email', 'test@example.invalid')
        (repo / 'model.py').write_text('value = 1\n')
        run_git(repo, 'add', '.')
        run_git(repo, 'commit', '-qm', 'first')
        repos[name], records[name] = repo, snapshot(repo, name)
        mirrors[name] = (tmp_path / ('mirror ' + name + '.git')).as_posix()

    def transfer():
        before = len(commands)
        observed = parity.ensure_remote_bare_repos(
            endpoint, list(mirrors.values()), False, refs=parity.snapshot_mirror_refs('test'))
        assert len(commands) == before + 1
        transfers.clear()
        reports = {name: parity.push_snapshot_via_git(
            repos[name], container=endpoint, mirror_path=mirrors[name], record=record,
            workspace_id='test', remote_refs=observed[mirrors[name]]) for name, record in records.items()}
        assert all(args[0] == 'push' for args in transfers), 'the batch probe replaces per-repo ls-remote'
        for name, record in records.items():
            assert run_git(mirrors[name], 'rev-parse', 'refs/heads/parity-current^{tree}') == record.tree
            assert run_git(mirrors[name], 'rev-parse', 'refs/parity/test/current') == record.commit
        return reports

    assert not any(row.get('skipped') for row in transfer().values())
    assert len(transfers) == 2  # The new mirror did not already contain either commit.
    assert all(row['skipped'] for row in transfer().values())
    assert transfers == []

    original = records['vllm-ascend']
    (repos['vllm-ascend'] / 'model.py').write_text('value = 2\n')
    run_git(repos['vllm-ascend'], 'commit', '-am', 'Python change')
    records['vllm-ascend'] = snapshot(repos['vllm-ascend'], 'vllm-ascend')
    reports = transfer()
    assert reports['vllm']['skipped'] and not reports['vllm-ascend'].get('skipped')
    assert len(transfers) == 1

    # A lost/interrupted publication may leave only two of the three refs.
    # A fresh remote observation must repair it, never remember local success.
    run_git(mirrors['vllm-ascend'], 'update-ref', '-d', 'refs/parity/test/current')
    assert not transfer()['vllm-ascend'].get('skipped')
    assert len(transfers) == 1

    records['vllm-ascend'] = original
    assert not transfer()['vllm-ascend'].get('skipped')
    assert len(transfers) == 1


def test_dangling_ref_is_not_a_verified_remote_hit(tmp_path, local_peer):
    endpoint = parity.SshEndpoint('local-fixture', 22, 'fixture')
    mirror = tmp_path / 'mirror.git'
    parity.ensure_remote_bare_repos(endpoint, [mirror.as_posix()], False)
    dangling = mirror / 'refs/heads/parity-current'
    dangling.parent.mkdir(parents=True, exist_ok=True)
    dangling.write_text('a' * 40 + '\n')
    observed = parity.ensure_remote_bare_repos(
        endpoint, [mirror.as_posix()], False, refs=parity.snapshot_mirror_refs('test'))
    assert observed[mirror.as_posix()] == {}
