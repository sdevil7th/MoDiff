"""No-GPU contracts for the separately managed MI300X preview runtime."""

import contextlib
import io
import json
import tempfile
import tomllib
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from modiff import install
from modiff import runtime_profile as runtime


PROFILE = "amd-instinct-rocm-linux"


def host(**overrides):
    return {
        "os": "linux", "os_id": "ubuntu", "os_version": "24.04",
        "architecture": "x86_64", "kernel": "6.8.0-100-generic", "wsl": False,
        "amd_candidate": True, "amd_usable": True, "amd_architectures": ["gfx942"],
        "nvidia_usable": False, "kfd_present": True, "kfd_accessible": True,
        "render_nodes": ["/dev/dri/renderD128"], "rocminfo_returncode": 0,
        **overrides,
    }


def hardware(**overrides):
    return {"torch": {
        "available": True, "version": "2.10.0+rocm7.14.0", "hip_version": "7.14.0",
        "cuda_available": True, **overrides,
    }}


class InstinctProfileTests(unittest.TestCase):
    def test_explicit_and_detected_instinct_do_not_select_ryzen(self):
        for selection in ("amd-instinct", "amd", "auto"):
            with self.subTest(selection=selection):
                self.assertEqual(install.resolve_profile(selection, host()), PROFILE)
        self.assertEqual(install.resolve_profile("amd", host(amd_architectures=["gfx1151"])), "amd-rocm-linux")
        self.assertEqual(install.resolve_profile("auto", host(wsl=True)), "cpu")

    def test_instinct_system_check_is_read_only_and_never_calls_ryzen_remediation(self):
        args = install.parser().parse_args(["--accelerator", "amd-instinct", "--system-check", "--json", "--backend-only"])
        with (
            patch.object(install, "detect_host", return_value=host()),
            patch.object(install, "_amd_qualification", side_effect=AssertionError("Ryzen qualification called")),
            patch.object(install, "_write_journal", side_effect=AssertionError("Journal mutated")),
            patch.object(install, "_run", side_effect=AssertionError("Install executed")),
        ):
            plan = install.install(args)
        self.assertEqual(plan["profile"], PROFILE)
        self.assertEqual(plan["support_tier"], "preview")
        self.assertTrue(plan["execution_ready"])
        self.assertFalse(plan["issues"])
        self.assertIn("--accelerator amd-instinct --resume", plan["resume_command"])
        self.assertTrue(plan["uv_config"].endswith("amd-instinct-rocm-linux.uv.toml"))

    def test_bad_instinct_hosts_are_blocked_without_any_administrator_action(self):
        cases = [
            ({"os_version": "26.04"}, "unsupported-os"),
            ({"os_id": "debian"}, "unsupported-os"),
            ({"kernel": "6.5.0"}, "kernel-too-old"),
            ({"kfd_present": False}, "gpu-device-nodes-missing"),
            ({"render_nodes": []}, "gpu-device-nodes-missing"),
            ({"kfd_accessible": False}, "kfd-permission-denied"),
            ({"rocminfo_returncode": 1}, "rocminfo-failed"),
            ({"amd_architectures": []}, "amd-architecture-unverified"),
            ({"amd_architectures": ["gfx1151"]}, "unsupported-amd-architecture"),
            ({"amd_architectures": ["gfx942", "gfx1151"]}, "unsupported-amd-architecture"),
            ({"wsl": True}, "unsupported-platform"),
        ]
        for changes, expected in cases:
            with self.subTest(changes=changes), patch.object(install, "detect_host", return_value=host(**changes)):
                plan = install.build_plan(install.parser().parse_args(["--accelerator", "amd-instinct", "--backend-only"]))
            self.assertFalse(plan["execution_ready"])
            self.assertIn(expected, [item["code"] for item in plan["issues"]])
            self.assertTrue(all(not item.get("action") for item in plan["issues"]))

    def test_profile_pins_device_sdk_and_preserves_existing_ryzen_contract(self):
        manifest = runtime.load_manifest()
        spec = manifest["profiles"][PROFILE]
        requirement = runtime.PROJECT_ROOT / spec["requirements"]
        lines = requirement.read_text().splitlines()
        self.assertEqual(spec["tier"], "preview")
        self.assertEqual(spec["device_families"], ["gfx942"])
        self.assertIn(f"torch[device-gfx942]=={spec['torch']}", lines)
        self.assertIn(f"torchvision[device-gfx942]=={spec['torchvision']}", lines)
        self.assertIn(f"torchaudio=={spec['torchaudio']}", lines)
        self.assertIn(f"triton=={spec['triton']}", lines)
        self.assertIn("rocm-sdk-device-gfx942==7.14.0", lines)
        self.assertIn("-e .", lines)
        ryzen = manifest["profiles"]["amd-rocm-linux"]
        self.assertEqual(ryzen["torch"], "2.9.1+rocm7.2.0")
        self.assertEqual(ryzen["device_families"], ["gfx1150", "gfx1151"])

    def test_public_index_fallback_is_scoped_and_covered_by_the_runtime_digest(self):
        spec = runtime.load_manifest()["profiles"][PROFILE]
        config = runtime.PROJECT_ROOT / spec["uv_config"]
        content = tomllib.loads(config.read_text())
        self.assertEqual(content["index"][0]["url"], spec["index"])
        self.assertEqual(content["index"][0]["ignore-error-codes"], [403])
        self.assertEqual(content["index"][1], {"name": "pypi", "url": "https://pypi.org/simple", "default": True})
        self.assertNotIn("index-strategy", content)
        requirement = runtime.PROJECT_ROOT / spec["requirements"]
        self.assertIn(config, runtime.runtime_contract_paths(requirement))
        ryzen_requirement = runtime.PROJECT_ROOT / runtime.load_manifest()["profiles"]["amd-rocm-linux"]["requirements"]
        self.assertNotIn(config, runtime.runtime_contract_paths(ryzen_requirement))

    def test_missing_index_config_blocks_install_and_existing_runtime(self):
        manifest = runtime.load_manifest()
        manifest["profiles"][PROFILE]["uv_config"] = "requirements/profiles/does-not-exist.uv.toml"
        with patch.object(install, "load_manifest", return_value=manifest), patch.object(install, "detect_host", return_value=host()):
            plan = install.build_plan(install.parser().parse_args(["--accelerator", "amd-instinct", "--system-check"]))
        self.assertFalse(plan["execution_ready"])
        self.assertIn("profile-lock-missing", [item["code"] for item in plan["issues"]])
        with tempfile.TemporaryDirectory() as temporary:
            root = self._saved_state(temporary)
            with (
                patch.object(runtime, "load_manifest", return_value=manifest),
                patch.object(runtime, "normalized_os", return_value="linux"),
                patch.object(runtime, "_device_tensor_probe", return_value={"ready": True}),
            ):
                report = runtime.runtime_profile(hardware(), venv=root)
        self.assertFalse(report["execution_ready"])
        self.assertEqual(report["runtime_contract"]["status"], "unavailable")

    def _saved_state(self, directory):
        spec = runtime.load_manifest()["profiles"][PROFILE]
        requirement = runtime.PROJECT_ROOT / spec["requirements"]
        root = Path(directory)
        (root / runtime.STATE_NAME).write_text(json.dumps({
            "profile": PROFILE, "support_tier": "preview",
            "runtime_contract_schema": runtime.RUNTIME_CONTRACT_SCHEMA,
            "lock_digest": runtime.lock_digest(requirement, profile=PROFILE),
        }))
        return root

    def test_runtime_keeps_instinct_identity_and_checks_version_and_device_failures(self):
        cases = [
            ({}, True, True),
            ({"version": "2.9.1+rocm7.2.0", "hip_version": "7.2.0"}, True, False),
            ({"hip_version": "7.140.0"}, True, False),
            ({"version": "2.10.1+rocm7.14.0"}, True, False),
            ({}, False, False),
            ({"cuda_available": False}, True, False),
            ({"hip_version": None, "cuda_version": "12.8"}, True, False),
        ]
        for changes, tensor_ready, expected in cases:
            with self.subTest(changes=changes, tensor_ready=tensor_ready), tempfile.TemporaryDirectory() as temporary:
                root = self._saved_state(temporary)
                with (
                    patch.object(runtime, "normalized_os", return_value="linux"),
                    patch.object(runtime, "normalized_arch", return_value="x86_64"),
                    patch.object(runtime, "_device_tensor_probe", return_value={"ready": tensor_ready, "device": "cuda:0", "message": "test failure"}),
                ):
                    report = runtime.runtime_profile(hardware(**changes), venv=root)
                self.assertEqual(report["execution_ready"], expected)
                self.assertEqual(report["support_tier"], "preview")
                self.assertIn("--accelerator amd-instinct --repair", report["repair_command"])
                if expected:
                    self.assertEqual(report["installed"], PROFILE)

    def test_live_tensor_probe_rejects_wrong_gpu_even_with_correct_wheel(self):
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            get_device_properties=lambda _: types.SimpleNamespace(gcnArchName="gfx1151"),
        ))
        runtime._device_tensor_probe.cache_clear()
        try:
            with patch.object(runtime.importlib, "import_module", return_value=fake_torch):
                result = runtime._device_tensor_probe(PROFILE, "2.10.0+rocm7.14.0")
            self.assertFalse(result["ready"])
            self.assertIn("requires gfx942", result["message"])
        finally:
            runtime._device_tensor_probe.cache_clear()

    def test_staged_smoke_checks_versions_and_gpu_architecture_before_tensor(self):
        fake = types.SimpleNamespace(
            __version__="2.10.0+rocm7.14.0",
            version=types.SimpleNamespace(hip="7.14.0", cuda=None),
            cuda=types.SimpleNamespace(is_available=lambda: True, get_device_properties=lambda _: types.SimpleNamespace(gcnArchName="gfx1151")),
        )
        with patch.dict("sys.modules", {"torch": fake}), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(AssertionError, "Expected gfx942"):
                exec(install._smoke_script(PROFILE), {})
        fake.__version__ = "2.10.0+cu128"
        with patch.dict("sys.modules", {"torch": fake}), self.assertRaisesRegex(AssertionError, "cu128"):
            exec(install._smoke_script(PROFILE), {})


if __name__ == "__main__":
    unittest.main()
