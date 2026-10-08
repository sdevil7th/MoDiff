"""No-GPU contracts for the separately managed MI300X preview runtime."""

import contextlib
import io
import json
from importlib import metadata
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


def native_hardware(**overrides):
    observed = hardware(cuda_device_count=1, **overrides)
    observed["devices"] = [{
        "type": "cuda", "device": "cuda:0", "architecture": "gfx942:sramecc+:xnack-",
        "memory_kind": "dedicated",
    }, {"type": "cpu", "device": "cpu:0"}]
    return observed


def reviewed_package_versions():
    from packaging.requirements import Requirement

    specification = runtime.load_manifest()["profiles"][PROFILE]
    versions = {}
    for line in (runtime.PROJECT_ROOT / specification["requirements"]).read_text().splitlines():
        if line and not line.startswith(("#", "-e ")):
            requirement = Requirement(line)
            versions[requirement.name] = next(iter(requirement.specifier)).version
    return versions


class InstinctProfileTests(unittest.TestCase):
    def setUp(self):
        self.base_runtime = patch.object(runtime, "base_runtime_status", return_value={
            "status": "verified", "verified": True, "matches": True,
            "issues": [], "current_digest": "f" * 64,
        })
        self.base_runtime.start()
        self.addCleanup(self.base_runtime.stop)
        self.package_versions = reviewed_package_versions()

        def installed_version(name):
            try:
                return self.package_versions[name]
            except KeyError:
                raise metadata.PackageNotFoundError(name) from None

        packages = patch.object(metadata, "version", side_effect=installed_version)
        packages.start()
        self.addCleanup(packages.stop)

    def test_receipt_free_instinct_uses_observed_device_and_reviewed_packages(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(runtime, "normalized_os", return_value="linux"),
            patch.object(runtime, "normalized_arch", return_value="x86_64"),
            patch.object(runtime, "_device_tensor_probe", return_value={"ready": True, "device": "cuda:0"}) as probe,
            patch.dict("os.environ", {"MODIFF_RUNTIME_PROFILE": "amd-rocm-linux"}),
        ):
            root = Path(temporary)
            report = runtime.runtime_profile(native_hardware(), venv=root)
            self.assertFalse((root / runtime.STATE_NAME).exists())

        self.assertEqual(report["installed"], PROFILE)
        self.assertTrue(report["execution_ready"])
        self.assertEqual(report["support_tier"], "preview")
        self.assertIsNone(report["requested"])
        self.assertIsNone(report["installation"])
        self.assertIn("uv pip install --python .venv/bin/python --config-file", report["repair_command"])
        probe.assert_called_once_with(PROFILE, "2.10.0+rocm7.14.0")

    def _native_report(self, observed=None, *, requested=None, saved_profile=None, saved_tier="supported", tensor_ready=True):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(runtime, "normalized_os", return_value="linux"),
            patch.object(runtime, "normalized_arch", return_value="x86_64"),
            patch.object(runtime, "_device_tensor_probe", return_value={"ready": tensor_ready, "message": "tensor failed"}),
        ):
            root = Path(temporary)
            if saved_profile == PROFILE:
                self._saved_state(temporary)
            elif saved_profile:
                (root / runtime.STATE_NAME).write_text(json.dumps({"profile": saved_profile, "support_tier": saved_tier}))
            return runtime.runtime_profile(observed or native_hardware(), requested=requested, venv=root)

    def test_receipt_free_instinct_enforces_actual_torch_and_hip_versions(self):
        for changes in (
            {"version": "2.10.1+rocm7.14.0"},
            {"hip_version": "7.140.0"},
            {"hip_version": "10.1.0"},
        ):
            with self.subTest(changes=changes):
                report = self._native_report(native_hardware(**changes))
                self.assertEqual(report["installed"], PROFILE)
                self.assertFalse(report["execution_ready"])
                self.assertIn("profile-version-mismatch", [issue["code"] for issue in report["issues"]])

    def test_receipt_free_instinct_requires_all_sdk_distributions_and_pins(self):
        originals = self.package_versions.copy()
        for name in originals:
            for version in (None, "0.0.1"):
                with self.subTest(name=name, version=version):
                    self.package_versions = originals.copy()
                    if version is None:
                        del self.package_versions[name]
                    else:
                        self.package_versions[name] = version
                    report = self._native_report()
                    self.assertFalse(report["execution_ready"])
                    self.assertTrue(report["repair_required"])
                    self.assertIn("profile-package-mismatch", [issue["code"] for issue in report["issues"]])
                    self.assertTrue(any(name in issue["message"] for issue in report["issues"]))

    def test_receipt_free_instinct_preserves_manifest_prohibitions(self):
        for name in runtime.load_manifest()["profiles"][PROFILE]["prohibited"]:
            with self.subTest(name=name):
                self.package_versions[name] = "1.0.0"
                report = self._native_report()
                self.assertFalse(report["execution_ready"])
                self.assertTrue(any(f"prohibits installed package: {name}" in issue["message"] for issue in report["issues"]))
                del self.package_versions[name]

    def test_receipt_free_instinct_rejects_malformed_sdk_version_metadata(self):
        name = runtime.load_manifest()["profiles"][PROFILE]["required"][0]
        for value in (None, "", "broken-version", 123):
            with self.subTest(value=value):
                self.package_versions[name] = value
                report = self._native_report()
                self.assertFalse(report["execution_ready"])
                self.assertIn("profile-package-mismatch", [issue["code"] for issue in report["issues"]])
                if not isinstance(value, str) or not value:
                    self.assertTrue(any(f"{name} has invalid installed version metadata" in issue["message"] for issue in report["issues"]))

    def test_native_device_detection_does_not_promote_ryzen_mixed_or_unknown_devices(self):
        for architecture, memory_kind, count in (
            ("gfx1151", "shared", 1), ("gfx908", "dedicated", 1),
            ("gfx942", "shared", 1), (None, "dedicated", 1),
            ("gfx942", "dedicated", 0), ("gfx942", "dedicated", 2),
        ):
            with self.subTest(architecture=architecture, memory_kind=memory_kind, count=count):
                observed = native_hardware()
                observed["torch"]["cuda_device_count"] = count
                observed["devices"][0].update(architecture=architecture, memory_kind=memory_kind)
                if count == 2:
                    observed["devices"].append({"type": "cuda", "device": "cuda:1", "architecture": "gfx1151", "memory_kind": "shared"})
                with patch.object(runtime, "_profile_package_issues", side_effect=AssertionError("Instinct SDK policy applied")):
                    report = self._native_report(observed)
                self.assertEqual(report["installed"], "amd-rocm-linux")

    def test_explicit_and_saved_sdk_selections_remain_authoritative(self):
        observed = native_hardware(version="2.9.1+rocm7.2.0", hip_version="7.2.0")
        report = self._native_report(observed, requested="amd-rocm-linux")
        self.assertEqual(report["installed"], "amd-rocm-linux")
        self.assertEqual(report["requested"], "amd-rocm-linux")
        report = self._native_report(saved_profile=PROFILE)
        self.assertEqual(report["installed"], PROFILE)
        self.assertEqual(report["requested"], PROFILE)
        self.assertTrue(report["execution_ready"])
        report = self._native_report(requested="cpu")
        self.assertFalse(report["execution_ready"])
        self.assertIn("profile-mismatch", [issue["code"] for issue in report["issues"]])

    def test_receipt_free_instinct_supersedes_stale_native_identity_and_requires_device_probe(self):
        for saved_profile in ("cpu", "nvidia-cuda"):
            with self.subTest(saved_profile=saved_profile):
                report = self._native_report(saved_profile=saved_profile)
                self.assertEqual(report["installed"], PROFILE)
                self.assertTrue(report["execution_ready"])
                self.assertEqual(report["support_tier"], "preview")
        report = self._native_report(tensor_ready=False)
        self.assertFalse(report["execution_ready"])
        self.assertIn("device-tensor-failed", [issue["code"] for issue in report["issues"]])

    def test_native_cuda_uses_its_own_tier_after_stale_cpu_receipt(self):
        observed = hardware(version="2.11.0+cu128", hip_version=None, cuda_version="12.8")
        report = self._native_report(observed, saved_profile="cpu", saved_tier="experimental")
        self.assertEqual(report["installed"], "nvidia-cuda")
        self.assertTrue(report["execution_ready"])
        self.assertEqual(report["support_tier"], runtime.load_manifest()["profiles"]["nvidia-cuda"]["tier"])

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
                patch.object(runtime, "base_runtime_status", return_value={
                    "status": "verified", "verified": True, "matches": True,
                    "issues": [], "current_digest": "f" * 64,
                }),
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

    def test_native_base_verification_does_not_mask_instinct_index_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._saved_state(temporary)
            spec = runtime.load_manifest()["profiles"][PROFILE]
            original = runtime.PROJECT_ROOT / spec["uv_config"]
            changed_config = Path(temporary) / "instinct.uv.toml"
            changed_config.write_text(original.read_text() + "\n# changed index contract\n")
            manifest = runtime.load_manifest()
            manifest["profiles"][PROFILE]["uv_config"] = str(changed_config)
            with (
                patch.object(runtime, "load_manifest", return_value=manifest),
                patch.object(runtime, "normalized_os", return_value="linux"),
                patch.object(runtime, "normalized_arch", return_value="x86_64"),
                patch.object(runtime, "base_runtime_status", return_value={
                    "status": "verified", "verified": True, "matches": True,
                    "issues": [], "current_digest": "f" * 64,
                }),
                patch.object(runtime, "_device_tensor_probe", return_value={"ready": True}),
            ):
                report = runtime.runtime_profile(hardware(), venv=root)

        self.assertFalse(report["execution_ready"])
        self.assertTrue(report["repair_required"])
        self.assertEqual(report["runtime_contract"]["status"], "drifted")
        self.assertIn("runtime-contract-drift", [issue["code"] for issue in report["issues"]])

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
