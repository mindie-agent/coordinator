"""Task-owned interpreter/root isolation for prepare and materialize."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mindie_coordinator.parity import (
    DEFAULT_MARKER_DIRNAME,
    DEFAULT_ROOT_PRESERVE_PATHS,
    build_snapshot_records,
    materialize_runtime,
    prepare_isolated_root_script,
    render_git_clean,
    resolved_root_preserve_paths,
    runtime_install_step_script,
    task_python_exports,
)
from mindie_coordinator.parity_support import SshEndpoint


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "-C", str(path), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "iso@example.invalid"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Iso"], check=True, capture_output=True)
    (path / "README").write_text("iso\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "README"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "init"], check=True, capture_output=True)


class IsolatedPrepareScriptTests(unittest.TestCase):
    def test_isolated_root_script_never_pip_uninstalls_image_packages(self):
        script = prepare_isolated_root_script("/vllm-workspace/tasks/s/h/r")
        self.assertIn("mkdir -p", script)
        self.assertNotIn("pip uninstall", script)
        self.assertNotIn("$PIP", script)
        self.assertNotIn("uninstall -y vllm", script)
        import mindie_coordinator.parity as parity
        self.assertFalse(hasattr(parity, "first_install_prepare_script"))

    def test_task_python_controls_pip_and_cmake_even_with_spaces(self):
        python = "/tmp/task root/.venv/bin/python"
        script = runtime_install_step_script(
            runtime_root="/vllm-workspace/tasks/s/h/r",
            marker_dirname=DEFAULT_MARKER_DIRNAME,
            container_identity="mindie-alice",
            step="uninstall",
            python=python,
        )
        pin = "\n".join(task_python_exports(python))
        self.assertIn(pin.splitlines()[0], script)
        after_preamble = script.split("export MKL_NUM_THREADS=1", 1)[-1]
        self.assertIn('export CMAKE_ARGS="-DPython3_EXECUTABLE=\\"$PYTHON\\" -DPython_EXECUTABLE=\\"$PYTHON\\""', after_preamble)
        self.assertIn("/tmp/task root/.venv/bin/python", after_preamble)
        self.assertNotIn("ls -1d /usr/local/python", after_preamble)
        self.assertIn('"$PYTHON" -m pip uninstall', after_preamble)

    @unittest.skipIf(os.name == "nt", "executes the Linux container venv payload locally")
    def test_venv_uses_prepared_image_python_instead_of_ssh_path(self):
        from mindie_coordinator.provision.task_environment import create_venv_script
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_bin = root / "system-bin"
            bad_bin.mkdir()
            bad = bad_bin / "python3"
            bad.write_text("#!/bin/sh\necho missing-ensurepip >&2\nexit 97\n")
            bad.chmod(0o755)
            task = root / "task root"
            python = task / ".venv/bin/python"
            image_python = getattr(sys, "_base_executable", sys.executable)
            preamble = ("export PYTHON=" + shlex.quote(image_python),)
            with mock.patch("mindie_coordinator.parity.PYTHON_METADATA_PREAMBLE", preamble):
                script = create_venv_script(str(task), str(python))
            result = subprocess.run(["bash", "-c", script], text=True,
                                    capture_output=True, timeout=30,
                                    env={**os.environ, "PATH": str(bad_bin) + os.pathsep + os.environ["PATH"]})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(python.is_file())
            actual = subprocess.check_output([str(python), "-c", "import sys; print(sys.base_prefix)"], text=True)
            expected = subprocess.check_output([image_python, "-c", "import sys; print(sys.base_prefix)"], text=True)
            self.assertEqual(actual, expected)
            self.assertIn("include-system-site-packages = true", (task / ".venv/pyvenv.cfg").read_text())

    def test_git_clean_preserves_venv_and_build_without_gitignore(self):
        preserve = resolved_root_preserve_paths(DEFAULT_MARKER_DIRNAME, [])
        self.assertIn(".venv", DEFAULT_ROOT_PRESERVE_PATHS)
        self.assertIn("build", DEFAULT_ROOT_PRESERVE_PATHS)
        rendered = render_git_clean("/vllm-workspace/tasks/s/h/r/vllm-ascend", preserve)
        self.assertIn(".venv", rendered)
        self.assertIn("build", rendered)
        self.assertIn("git -C", rendered)


class ExplicitSourceSnapshotTests(unittest.TestCase):
    def test_snapshot_and_materialize_plan_use_bound_repos_not_a_git_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "not-a-git-parent"
            parent.mkdir()
            vllm = parent / "vllm"
            ascend = parent / "vllm-ascend"
            _init_repo(vllm)
            _init_repo(ascend)
            elsewhere = Path(tmp) / "daemon-cwd"
            elsewhere.mkdir()
            previous = Path.cwd()
            try:
                os.chdir(elsewhere)
                records = build_snapshot_records(
                    parent, "ws", "snap1", (),
                    {"vllm": vllm, "vllm-ascend": ascend},
                )
            finally:
                os.chdir(previous)
            relpaths = {record.relpath for record in records}
            self.assertIn("vllm", relpaths)
            self.assertIn("vllm-ascend", relpaths)
            self.assertNotIn(".", relpaths)
            captured: list[str] = []

            def fake_ssh(container, script, **kwargs):
                captured.append(script)
                return mock.Mock(stdout="", returncode=0)

            endpoint = SshEndpoint(host="192.0.2.10", port=46000, user="root")
            with mock.patch("mindie_coordinator.parity.ssh_exec", fake_ssh):
                materialize_runtime(
                    container=endpoint,
                    runtime_root="/vllm-workspace/tasks/s/h/r",
                    container_cache_root="/root/.cache/mindie/remote-code-parity",
                    workspace_id="ws",
                    marker_dirname=DEFAULT_MARKER_DIRNAME,
                    root_preserve_paths=resolved_root_preserve_paths(DEFAULT_MARKER_DIRNAME, []),
                    records=records,
                    dry_run=False,
                )
            script = "\n".join(captured)
            self.assertIn("/vllm-workspace/tasks/s/h/r/vllm", script)
            self.assertIn("/vllm-workspace/tasks/s/h/r/vllm-ascend", script)
            self.assertIn(".venv", script)
            self.assertNotIn("git init /vllm-workspace/tasks/s/h/r >/dev/null", script)


class FakeTaskPythonInvocationTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "executes Linux shell fixtures and interpreters locally")
    def test_uninstall_verify_and_editable_invoke_task_python_with_spaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            runtime_root = base / "runtime"
            (runtime_root / "vllm").mkdir(parents=True)
            (runtime_root / "vllm-ascend").mkdir(parents=True)
            bindir = base / "task root" / ".venv" / "bin"
            bindir.mkdir(parents=True)
            logs = base / "argv"
            logs.mkdir()
            fake = bindir / "python"
            fake.write_text(
                "#!/bin/sh\n"
                f"record=$(mktemp {shlex.quote(str(logs / 'call.XXXXXX'))})\n"
                "printf '%s\\n' \"$0\" \"$@\" > \"$record\"\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            python = str(fake)
            for step in ("uninstall", "verify-imports", "verify-deps", "install-vllm"):
                script = runtime_install_step_script(
                    runtime_root=str(runtime_root),
                    marker_dirname=DEFAULT_MARKER_DIRNAME,
                    container_identity="mindie-alice",
                    step=step,
                    python=python,
                )
                result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            # Progress reporting and pip run concurrently. Separate files keep
            # argv boundaries intact even when shell printf uses several writes.
            recorded = [path.read_text(encoding="utf-8").splitlines() for path in logs.iterdir()]
            self.assertGreaterEqual(len(recorded), 4)
            task_calls = [call[1:] for call in recorded if call[0] == python]
            self.assertCountEqual(task_calls, [
                ["-m", "pip", "uninstall", "-y", "vllm", "vllm-ascend", "vllm_ascend"],
                ["-"],
                ["-"],
                ["-m", "pip", "install", "--no-deps", "-e", ".", "--no-build-isolation"],
            ])
            self.assertNotIn("install_consent.py", runtime_install_step_script(
                runtime_root=str(runtime_root),
                marker_dirname=DEFAULT_MARKER_DIRNAME,
                container_identity="mindie-alice",
                step="check-build-compat",
                python=python,
            ))
