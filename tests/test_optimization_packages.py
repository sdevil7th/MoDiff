import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modiff import optimization_packages as optimizations


class OptimizationPackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.path_patchers = [
            mock.patch.object(optimizations, "OPTIMIZATION_ROOT", root),
            mock.patch.object(optimizations, "ENVIRONMENTS_DIR", root / "environments"),
            mock.patch.object(optimizations, "STAGING_DIR", root / "staging"),
            mock.patch.object(optimizations, "STATE_PATH", root / "state.json"),
            mock.patch.object(optimizations, "RECEIPTS_PATH", root / "receipts.json"),
        ]
        for patcher in self.path_patchers:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.path_patchers):
            patcher.stop()
        self.temporary.cleanup()

    def create_environment(self, environment_id):
        root = optimizations.ENVIRONMENTS_DIR / environment_id
        site_packages = root / "site-packages"
        site_packages.mkdir(parents=True)
        (root / "manifest.json").write_text(
            json.dumps({"id": environment_id, "capabilities": ["torchao"]}),
            encoding="utf-8",
        )
        (root / "validation.json").write_text(
            json.dumps({"status": "passed"}),
            encoding="utf-8",
        )
        return root

    def test_catalog_is_profile_gated_and_disabled_by_default(self):
        catalog = optimizations.public_catalog(
            runtime_profile={"installed": "amd-rocm-linux"},
            hardware={"torch": {"version": "2.9.1+rocm7.2"}, "amd_architectures": ["gfx1151"]},
        )
        by_id = {item["id"]: item for item in catalog["capabilities"]}
        self.assertTrue(by_id["torchao"]["compatible"])
        self.assertFalse(by_id["hub_attention_kernels"]["compatible"])
        self.assertFalse(by_id["torchao"]["enabled"])

    def test_activation_and_rollback_only_use_validated_environments(self):
        self.create_environment("first")
        self.create_environment("second")
        first = optimizations.activate_environment("first")
        self.assertTrue(first["restartRequired"])
        second = optimizations.activate_environment("second")
        self.assertEqual(second["state"]["previousEnvironmentId"], "first")
        rolled_back = optimizations.rollback_environment()
        self.assertEqual(rolled_back["state"]["activeEnvironmentId"], "first")
        with self.assertRaises(ValueError):
            optimizations.activate_environment("missing")

    def test_failed_stage_never_changes_active_environment(self):
        self.create_environment("active")
        optimizations.activate_environment("active")
        failed_process = mock.Mock(returncode=1, stdout="", stderr="compiler failed")
        with (
            mock.patch.object(optimizations, "_uv_executable", return_value="/managed/uv"),
            mock.patch.object(optimizations.subprocess, "run", return_value=failed_process),
            self.assertRaisesRegex(RuntimeError, "compiler failed"),
        ):
            optimizations.install_capability(
                "torchao",
                runtime_profile={"installed": "amd-rocm-linux"},
                hardware={"torch": {"version": "2.9.1+rocm7.2"}},
            )
        self.assertEqual(optimizations.read_state()["activeEnvironmentId"], "active")

    def test_source_package_bootstraps_toolchain_before_abi_install(self):
        self.create_environment("active")
        optimizations.activate_environment("active")
        succeeded = mock.Mock(returncode=0, stdout="installed", stderr="")
        validation = {"status": "passed", "detail": {"torch": "2.9.1+rocm7.2"}}
        with (
            mock.patch.object(optimizations, "_uv_executable", return_value="/managed/uv"),
            mock.patch.object(optimizations.subprocess, "run", return_value=succeeded) as run,
            mock.patch.object(optimizations, "_run_validation", return_value=validation),
        ):
            result = optimizations.install_capability(
                "flash_attention_2",
                runtime_profile={"installed": "amd-rocm-linux"},
                hardware={
                    "torch": {"version": "2.9.1+rocm7.2"},
                    "amd_architectures": ["gfx1151"],
                },
            )

        self.assertEqual(run.call_count, 2)
        build_command, package_command = (call.args[0] for call in run.call_args_list)
        self.assertIn("ninja==1.13.0", build_command)
        self.assertIn("flash-attn==2.8.3.post1", package_command)
        self.assertIn("--no-build-isolation", package_command)
        self.assertIn("--no-deps", package_command)
        self.assertTrue(result["requiresActivation"])
        self.assertFalse(result["activeRuntimeChanged"])
        self.assertEqual(optimizations.read_state()["activeEnvironmentId"], "active")
        self.assertTrue((optimizations.ENVIRONMENTS_DIR / result["environmentId"] / "validation.json").is_file())

    def test_auto_requires_opt_in_baseline_review_and_exact_runtime(self):
        self.create_environment("active")
        optimizations.activate_environment("active")
        optimizations.set_capability_enabled("regional_compile", True)
        runtime = "runtime-a"
        common = {
            "runtime_fingerprint": runtime,
            "model_type": "Qwen-Image-2512",
            "mode": "text_to_image",
            "artifact": "Qwen/Qwen-Image-2512",
            "workload_key": "workload-a",
        }
        optimizations.record_workload_baseline(
            **common,
            measurement={"elapsedSeconds": 100.0, "peakAllocatedBytes": 1000},
        )
        observed = optimizations.record_workload_observation(
            capability_id="regional_compile",
            **common,
            selection={"regionalCompile": True},
            measurement={"elapsedSeconds": 80.0, "peakAllocatedBytes": 1000},
        )
        self.assertEqual(
            optimizations.qualified_auto_overrides(**common),
            {},
        )
        optimizations.qualify_receipt(observed["id"], output_reviewed=True)
        self.assertEqual(
            optimizations.qualified_auto_overrides(**common),
            {"regionalCompile": True},
        )
        self.assertEqual(
            optimizations.qualified_auto_overrides(**{**common, "runtime_fingerprint": "runtime-b"}),
            {},
        )

    def test_import_probe_receipt_never_authorizes_auto(self):
        receipt = optimizations.record_probe_receipt(
            capability_id="regional_compile",
            runtime_fingerprint="runtime",
            result={"status": "passed"},
        )
        self.assertEqual(receipt["status"], "probe_passed")
        self.assertFalse(receipt["autoEligible"])

    def test_auto_combines_independently_qualified_capabilities(self):
        self.create_environment("active")
        optimizations.activate_environment("active")
        for capability in ("regional_compile", "channels_last"):
            optimizations.set_capability_enabled(capability, True)
        common = {
            "runtime_fingerprint": "runtime-a",
            "model_type": "Qwen-Image-2512",
            "mode": "text_to_image",
            "artifact": "Qwen/Qwen-Image-2512",
            "workload_key": "workload-a",
        }
        optimizations.record_workload_baseline(
            **common,
            measurement={"elapsedSeconds": 100.0, "peakAllocatedBytes": 1000},
        )
        compile_receipt = optimizations.record_workload_observation(
            capability_id="regional_compile",
            **common,
            selection={"regionalCompile": True},
            measurement={"elapsedSeconds": 80.0, "peakAllocatedBytes": 1000},
        )
        layout_receipt = optimizations.record_workload_observation(
            capability_id="channels_last",
            **common,
            selection={"channelsLast": True},
            measurement={"elapsedSeconds": 95.0, "peakAllocatedBytes": 900},
        )
        optimizations.qualify_receipt(compile_receipt["id"], output_reviewed=True)
        optimizations.qualify_receipt(layout_receipt["id"], output_reviewed=True)
        self.assertEqual(
            optimizations.qualified_auto_overrides(**common),
            {"channelsLast": True, "regionalCompile": True},
        )


if __name__ == "__main__":
    unittest.main()
