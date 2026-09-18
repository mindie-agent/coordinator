"""Installed-artifact and environment boundaries for automatic attestation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mindie_coordinator.backend import RemoteBackend
from mindie_coordinator.runtime_profile import capture_launch_environment, installed_native_files


class PreparedProfileTests(unittest.TestCase):
    def test_preserves_cann_paths_without_device_assignment_or_temporary_shim(self):
        result = capture_launch_environment({
            "ASCEND_OPP_PATH": "/image/cann/opp", "ASCEND_AICPU_PATH": "/image/cann",
            "ASCEND_TOOLKIT_HOME": "/image/cann", "ATB_HOME_PATH": "/image/atb",
            "SOC_VERSION": "ascend910_9391", "ASCEND_RT_VISIBLE_DEVICES": "7",
            "MINDIE_PYTHON_SHIM_DIR": "/tmp/owned-shim",
            "PATH": "/tmp/owned-shim:/task/.venv/bin:/tmp/owned-shim:/usr/bin",
            "API_TOKEN": "must-not-be-captured",
        })
        self.assertEqual(result, {
            "ASCEND_OPP_PATH": "/image/cann/opp", "ASCEND_AICPU_PATH": "/image/cann",
            "ASCEND_TOOLKIT_HOME": "/image/cann", "ATB_HOME_PATH": "/image/atb",
            "SOC_VERSION": "ascend910_9391", "PATH": "/task/.venv/bin:/usr/bin",
        })

    def test_enumerates_complete_installed_tree_without_venv_or_build_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix = "vllm-ascend/vllm_ascend/"
            expected = {
                prefix + "vllm_ascend_C.so": "library",
                prefix + "vllm_ascend_kernels.so": "library",
                prefix + "_cann_ops_custom/vendors/test/op_api/lib/libcust_opapi.so": "library",
                prefix + "_cann_ops_custom/vendors/test/op_impl/kernel/second.o": "library",
                prefix + "_cann_ops_custom/vendors/test/op_impl/config/ops.json": "metadata",
                prefix + "_cann_ops_custom/vendors/test/version.info": "metadata",
            }
            unrelated = [".venv/lib/unrelated.so", "vllm-ascend/csrc/build/build.so",
                         "vllm-ascend/csrc/build/binary_info_config.json"]
            for name in [*expected, *unrelated]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            self.assertEqual(installed_native_files(root), expected)
            (root / (prefix + "_cann_ops_custom/vendors/test/op_impl/config/ops.json")).unlink()
            with self.assertRaisesRegex(ValueError, "complete installed"):
                installed_native_files(root)

    def test_unrelated_libraries_cannot_replace_missing_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "unrelated.so").write_text("library")
            (root / "version.txt").write_text("metadata")
            with self.assertRaisesRegex(ValueError, "installed vllm_ascend_C"):
                installed_native_files(root)

    def test_symlinked_vendor_artifact_is_not_silently_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "vllm-ascend/vllm_ascend"
            vendor = package / "_cann_ops_custom"
            vendor.mkdir(parents=True)
            (package / "vllm_ascend_C.so").write_text("extension")
            (vendor / "config.json").write_text("config")
            (root / "external.so").write_text("external")
            (vendor / "kernel.so").symlink_to(root / "external.so")
            with self.assertRaisesRegex(ValueError, "symlink"):
                installed_native_files(root)

    def test_internal_vendor_alias_is_materialized_into_complete_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "vllm-ascend/vllm_ascend"
            vendor = package / "_cann_ops_custom"
            vendor.mkdir(parents=True)
            (package / "vllm_ascend_C.so").write_text("extension")
            (vendor / "config.json").write_text("config")
            (vendor / "actual.so").write_text("kernel")
            alias = vendor / "alias.so"
            alias.symlink_to("actual.so")
            files = installed_native_files(root)
            self.assertFalse(alias.is_symlink())
            self.assertEqual(alias.read_text(), "kernel")
            self.assertEqual(files[alias.relative_to(root).as_posix()], "library")
            self.assertEqual(files[(vendor / "actual.so").relative_to(root).as_posix()], "library")

    def test_capture_sources_image_environment_before_task_python(self):
        backend = RemoteBackend()
        spec = {"python": "/owned root/.venv/bin/python", "endpoint": {"root": "/owned root"},
                "host_endpoint": {}, "container_name": "test"}
        info = {"Id": "container", "Image": "sha256:test", "State": {"Running": True}}
        with mock.patch.object(backend, "bash", side_effect=[json.dumps(info), "{}\n"]) as bash:
            with self.assertRaisesRegex(ValueError, 'invalid captured manifest'):
                backend._write_ready_profile(spec, {"recipe": "test"})
        command = bash.call_args_list[1].args[1]
        self.assertIn("/etc/profile.d/mindie-ascend-env.sh", command)
        self.assertLess(command.index("/etc/profile.d/mindie-ascend-env.sh"),
                        command.index("export MINDIE_PYTHON="))
        self.assertIn("'/owned root/.venv/bin/python' - ", command)


if __name__ == "__main__":
    unittest.main()
