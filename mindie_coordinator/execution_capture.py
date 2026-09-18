"""Incremental local capture; cached records always name retained Git objects."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

from mindie_coordinator import parity
from mindie_coordinator.build_inputs import BUILD_INPUT_ENV_KEYS, DEPENDENCY_INSTALL_PATTERNS
from mindie_coordinator.parity_support import git, glob_match_any

# Bump when capture semantics change beyond the rules fingerprint below.
CACHE_FORMAT = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_bytes(path):
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return b""


def topology_key(nodes):
    return digest([(node.relpath, str(node.repo_path.resolve()),
                    file_bytes(node.repo_path / ".git").hex() if (node.repo_path / ".git").is_file()
                    else str((node.repo_path / ".git").resolve()),
                    file_bytes(node.repo_path / ".gitmodules").hex()) for node in nodes])


def common_directory(repo):
    directory = repo / ".git"
    if directory.is_file():
        directory = (repo / directory.read_text(encoding="utf-8").strip().removeprefix("gitdir: ")).resolve()
    common = directory / "commondir"
    return (directory / common.read_text(encoding="utf-8").strip()).resolve() if common.exists() else directory.resolve()


def retained(record):
    """A retained ref prevents Git GC; missing/moved refs are cache misses."""
    common = common_directory(Path(record["source_path"]))
    expected = {record["ref"]: record["commit"]}
    if record.get("source_head"):
        expected[record["ref"] + "-scm"] = record["source_head"]
    packed = None
    for ref, commit in expected.items():
        value = file_bytes(common / ref).decode().strip()
        if not value:
            if packed is None:
                packed = {line.split(" ", 1)[1]: line.split(" ", 1)[0]
                          for line in file_bytes(common / "packed-refs").decode().splitlines()
                          if line and not line.startswith(("#", "^"))}
            value = packed.get(ref)
        if value != commit:
            return False
    return True


def observe(node):
    # Git performs its normal index/worktree comparison. Unlike status alone,
    # hashing each dirty/untracked file also detects a second edit of that file.
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    status = git(node.repo_path, ["status", "--porcelain=v2", "--branch", "--untracked-files=all", "-z"], env=env).stdout
    config = git(node.repo_path, ["config", "--null", "--list"], env=env).stdout
    settings = dict(item.split("\n", 1) for item in config.split("\0") if "\n" in item)
    common = common_directory(node.repo_path)
    hasher = hashlib.sha256((status + "\0" + config).encode())
    attributes = [common / "info/attributes"]
    if settings.get("core.attributesfile"):
        attributes.append(node.repo_path / Path(settings["core.attributesfile"]).expanduser())
    attribute_key = digest([file_bytes(path).hex() for path in attributes])
    hasher.update(attribute_key.encode())
    tokens = iter(status.split("\0"))
    head, dirty, paths, untracked = "", False, [], []
    for token in tokens:
        if token.startswith("# branch.oid "):
            head = token.removeprefix("# branch.oid ")
        if token.startswith("1 "):
            path, dirty = token.split(" ", 8)[8], True
        elif token.startswith("2 "):
            path, dirty = token.split(" ", 9)[9], True
            paths.append(next(tokens))
        elif token.startswith("u "):
            path, dirty = token.split(" ", 10)[10], True
        elif token.startswith("? "):
            path = token[2:]
            untracked.append(path)
        else:
            continue
        paths.append(path)
        candidate = node.repo_path / path
        if candidate.is_symlink():
            hasher.update(os.readlink(candidate).encode())
        elif candidate.is_file():
            with candidate.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
        hasher.update(b"\0")
    scm = None
    if node.relpath in {"vllm", "vllm-ascend"}:
        tags = git(node.repo_path, ["for-each-ref", "--format=%(refname) %(objectname)", "refs/tags"], env=env).stdout
        scm = digest([head, dirty, tags, config, file_bytes(common / "shallow").hex(),
                      datetime.now(timezone.utc).date().isoformat(), version("setuptools-scm"),
                      {key: value for key, value in os.environ.items()
                       if key.startswith("SETUPTOOLS_SCM_") or key == "SOURCE_DATE_EPOCH"}])
    return {"content": hasher.hexdigest(), "scm": scm, "head": head, "paths": paths, "untracked": untracked,
            "config": digest([config, attribute_key]), "identity": [settings.get("user.name"), settings.get("user.email")]}


def incremental_snapshot(node, previous, before, old_state):
    """Use Git's prior fixed tree, adding only paths that can differ from it."""
    paths = sorted(set(before["paths"]) | set(old_state.get("paths", [])))
    if (previous.get("source_head") != before["head"] or before["config"] != old_state.get("config")
            or any((node.repo_path / path).exists() for path in set(old_state.get("untracked", [])) - set(before["paths"]))
            or any(Path(path).name in {".gitignore", ".gitattributes", ".gitmodules"} for path in paths)
            or any(glob_match_any(path, parity.DEFAULT_DENYLIST) or
                   any(path.startswith(pattern) for pattern in parity.DEFAULT_DENYLIST if pattern.endswith("/"))
                   for path in paths)):
        return None
    # Child gitlinks in a synthetic tree differ for transport; changing a child
    # requires the ordinary recursive snapshot builder, never git add on it.
    child_paths = [child.relpath.removeprefix(node.relpath + "/") for child in node.children]
    paths = [path for path in paths if path not in child_paths]
    temporary = tempfile.NamedTemporaryFile(prefix="mindie-input-index-", delete=False)
    temporary.close()
    env = {**os.environ, "GIT_INDEX_FILE": temporary.name, "GIT_OPTIONAL_LOCKS": "0"}
    for role in ("AUTHOR", "COMMITTER"):
        env.setdefault("GIT_" + role + "_NAME", before["identity"][0] or "remote-code-parity")
        env.setdefault("GIT_" + role + "_EMAIL", before["identity"][1] or "remote-code-parity@example.invalid")
        env.setdefault("GIT_" + role + "_DATE", "1970-01-01T00:00:00Z")
    try:
        git(node.repo_path, ["read-tree", previous["tree"]], env=env)
        if paths:
            git(node.repo_path, ["add", "-A", "--", *[":(literal)" + path for path in paths]], env=env)
        tree = git(node.repo_path, ["write-tree"], env=env).stdout.strip()
        commit = git(node.repo_path, ["commit-tree", tree, "-m", parity.commit_message(node.relpath)], env=env).stdout.strip()
        changed = list(filter(None, git(node.repo_path, ["diff", "--name-only", "-z", before["head"], commit]).stdout.split("\0")))
        changed = parity.filter_transport_only_child_paths(changed, set(child_paths) - set(previous["changed_paths"]))
        ref = parity.synthetic_ref("execution-inputs", uuid.uuid4().hex, node.relpath)
        git(node.repo_path, ["update-ref", ref, commit])
        return parity.SnapshotRecord(node.relpath, previous["repo_id"], before["head"], before["head"],
                                     commit, tree, ref, changed, previous["submodules"], source_path=str(node.repo_path))
    finally:
        Path(temporary.name).unlink(missing_ok=True)


def cleanup(records):
    for record in records:
        git(Path(record.source_path), ["update-ref", "-d", record.ref], check=False)


def capture_records(roots, state_dir, *, attempts, build_env):
    rules = digest([CACHE_FORMAT, parity.DEFAULT_DENYLIST, parity.VLLM_REINSTALL_PATTERNS,
                    parity.VLLM_ASCEND_REINSTALL_PATTERNS, DEPENDENCY_INSTALL_PATTERNS, BUILD_INPUT_ENV_KEYS])
    location = Path(state_dir) / "source-inputs" / ("capture-" + digest({name: str(path) for name, path in roots.items()}) + ".json")
    try:
        cache = json.loads(location.read_text(encoding="utf-8"))
        if not isinstance(cache, dict):
            cache = {}
        if cache.pop("checksum") != digest(cache) or cache.get("rules") != rules:
            cache = {}
    except (OSError, ValueError, KeyError):
        cache = {}
    nodes = []
    try:
        by_name = {row["relpath"]: parity.RepoNode(row["relpath"], Path(row["path"]), row["name"])
                   for row in cache.get("nodes", [])}
        for row in cache.get("nodes", []):
            by_name[row["relpath"]].children = [by_name[name] for name in row["children"]]
        nodes = list(by_name.values())
        if not nodes or topology_key(nodes) != cache["topology"]:
            nodes = []
    except (OSError, ValueError, KeyError):
        nodes = []
    if not nodes:
        cache = {}
        for name, path in roots.items():
            nodes.extend(parity.iter_postorder(parity.discover_repo_tree(path, name)))
    topology = topology_key(nodes)
    changed = []
    with ThreadPoolExecutor(max_workers=min(8, len(nodes))) as workers:
        for attempt in range(attempts):
            environment = {**build_env, **{key: value for key, value in os.environ.items() if key.startswith("GIT_")}}
            before = dict(zip((node.relpath for node in nodes), workers.map(observe, nodes)))
            records, changed, entries = [], [], {}
            try:
                for node in nodes:
                    children = {record.relpath: record for record in records}
                    child_commits = [[child.relpath, children[child.relpath].commit] for child in node.children]
                    key = digest([before[node.relpath], environment, child_commits])
                    old = cache.get("records", {}).get(node.relpath, {})
                    if old.get("key") == key and retained(old["record"]):
                        record = parity.SnapshotRecord(**old["record"])
                    else:
                        record = None
                        if old.get("children") == child_commits and retained(old["record"]):
                            record = incremental_snapshot(node, old["record"], before[node.relpath], old["state"])
                        if record is None:
                            record = parity.build_synthetic_snapshot(node, workspace_id="execution-inputs",
                                snapshot_id=uuid.uuid4().hex, denylist=tuple(parity.DEFAULT_DENYLIST), child_commits=children)
                        record.source_path = str(node.repo_path.resolve())
                        changed.append(record)
                        if node.relpath in {"vllm", "vllm-ascend"}:
                            previous = old.get("record") or {}
                            patterns = (parity.VLLM_REINSTALL_PATTERNS if node.relpath == "vllm"
                                        else parity.VLLM_ASCEND_REINSTALL_PATTERNS)
                            reusable = previous.get("build_inputs") and retained(previous)
                            if reusable:
                                delta = filter(None, git(node.repo_path, ["diff", "--name-only", "-z", previous["commit"], record.commit]).stdout.split("\0"))
                                child_paths = [child.relpath.removeprefix(node.relpath + "/") for child in node.children]
                                reusable = not any(path in child_paths or glob_match_any(path, patterns) for path in delta)
                            if reusable:
                                record.build_inputs = {**previous["build_inputs"], "build_env": digest(build_env)}
                            else:
                                record.build_inputs = parity.build_input_fingerprints(node.repo_path, record.commit, patterns, build_env=build_env)
                            if old.get("scm") == before[node.relpath]["scm"] and previous.get("scm_version"):
                                record.scm_version = previous["scm_version"]
                            elif (node.repo_path / "pyproject.toml").exists() or (node.repo_path / "setup.py").exists():
                                from setuptools_scm import get_version
                                record.scm_version = get_version(root=str(node.repo_path))
                    records.append(record)
                    entries[node.relpath] = {"key": key, "scm": before[node.relpath]["scm"], "record": asdict(record),
                                            "state": before[node.relpath], "children": child_commits}
                after = dict(zip((node.relpath for node in nodes), workers.map(observe, nodes)))
                stable_topology = topology_key(nodes) == topology
                current_environment = {**build_env, **{key: value for key, value in os.environ.items() if key.startswith("GIT_")}}
                if before == after and stable_topology and current_environment == environment:
                    break
            except Exception:
                cleanup(changed)
                raise
            cleanup(changed)
            if not stable_topology:
                cache, nodes = {}, []
                for name, path in roots.items():
                    nodes.extend(parity.iter_postorder(parity.discover_repo_tree(path, name)))
                topology = topology_key(nodes)
        else:
            raise ValueError(f"sources changed during {attempts} capture attempts; execution was not admitted")
    cache = {"rules": rules, "nodes": [{"relpath": node.relpath, "path": str(node.repo_path), "name": node.submodule_name,
                         "children": [child.relpath for child in node.children]} for node in nodes],
             "topology": topology, "records": entries}
    return records, changed, (location, cache)


def save_cache(location, cache, records):
    for record in records:
        cache["records"][record.relpath]["record"] = asdict(record)
    cache["checksum"] = digest(cache)
    temporary = location.with_suffix("." + uuid.uuid4().hex + ".tmp")
    try:
        location.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(cache, sort_keys=True), encoding="utf-8")
        os.replace(temporary, location)
    finally:
        temporary.unlink(missing_ok=True)
