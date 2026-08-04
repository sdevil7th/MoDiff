import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modiff.runtime_profile import (
    MANIFEST_PATH,
    RUNTIME_CONTRACT_SCHEMA,
    load_manifest,
    lock_digest,
    runtime_contract_paths,
    runtime_profile,
)


def hardware(
    *,
    version,
    cuda_version=None,
    hip_version=None,
    cuda_available=False,
    xpu_available=False,
    mps_built=False,
    mps_available=False,
):
    return {
        "torch": {
            "available": True,
            "version": version,
            "cuda_version": cuda_version,
            "hip_version": hip_version,
            "cuda_available": cuda_available,
            "xpu_available": xpu_available,
            "mps_built": mps_built,
            "mps_available": mps_available,
        },
        "devices": [{"type": "cpu", "device": "cpu:0"}],
        "default_device": "cpu:0",
    }


class RuntimeProfileTests(unittest.TestCase):
    def test_managed_macos_cpu_profile_ignores_the_wheels_optional_mps_capability(self):
        with tempfile.TemporaryDirectory() as temporary:
            venv = self.write_profile(temporary, "cpu")
            with (
                patch("modiff.runtime_profile.normalized_os", return_value="macos"),
                patch("modiff.runtime_profile.normalized_arch", return_value="arm64"),
                patch(
                    "modiff.runtime_profile._device_tensor_probe",
                    return_value={"ready": True, "device": "cpu", "message": None},
                ),
            ):
                profile = runtime_profile(
                    hardware(
                        version="2.8.0",
                        mps_built=True,
                        mps_available=True,
                    ),
                    venv=venv,
                )

        self.assertTrue(profile["execution_ready"])
        self.assertEqual(profile["installed"], "cpu")
        self.assertEqual(profile["device_validation"]["device"], "cpu")
        self.assertNotIn("profile-mismatch", [issue["code"] for issue in profile["issues"]])

    def test_managed_apple_mps_profile_keeps_using_mps(self):
        with tempfile.TemporaryDirectory() as temporary:
            venv = self.write_profile(temporary, "apple-mps")
            with (
                patch("modiff.runtime_profile.normalized_os", return_value="macos"),
                patch("modiff.runtime_profile.normalized_arch", return_value="arm64"),
                patch(
                    "modiff.runtime_profile._device_tensor_probe",
                    return_value={"ready": True, "device": "mps:0", "message": None},
                ),
            ):
                profile = runtime_profile(
                    hardware(
                        version="2.8.0",
                        mps_built=True,
                        mps_available=True,
                    ),
                    venv=venv,
                )

        self.assertTrue(profile["execution_ready"])
        self.assertEqual(profile["installed"], "apple-mps")
        self.assertEqual(profile["device_validation"]["device"], "mps:0")

    def test_runtime_contract_digest_changes_with_every_contract_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = tuple(Path(temporary) / name for name in ("profile.txt", "pyproject.toml", "manifest.json"))
            for index, path in enumerate(paths):
                path.write_text(f"contract-{index}", encoding="utf-8")
            baseline = lock_digest(paths[0], contract_paths=paths)
            for index, path in enumerate(paths):
                original = path.read_text(encoding="utf-8")
                path.write_text(f"{original}-changed", encoding="utf-8")
                self.assertNotEqual(lock_digest(paths[0], contract_paths=paths), baseline)
                path.write_text(original, encoding="utf-8")

    def test_runtime_contract_digest_ignores_unrelated_accelerator_profiles(self):
        with tempfile.TemporaryDirectory() as temporary:
            requirement = Path(temporary) / "profile.txt"
            requirement.write_text("torch==unit", encoding="utf-8")
            manifest = {
                "schema_version": 1,
                "python": "3.12.*",
                "profiles": {
                    "cpu": {"requirements": "requirements/profiles/cpu.txt", "tier": "supported"},
                    "intel-xpu": {"requirements": "requirements/profiles/intel-xpu.txt", "tier": "preview"},
                },
            }
            with patch("modiff.runtime_profile.load_manifest", return_value=manifest):
                baseline = lock_digest(requirement, contract_paths=(requirement,), profile="cpu")

            manifest["profiles"]["intel-xpu"]["tier"] = "supported"
            with patch("modiff.runtime_profile.load_manifest", return_value=manifest):
                unrelated_change = lock_digest(requirement, contract_paths=(requirement,), profile="cpu")

            manifest["profiles"]["cpu"]["tier"] = "conditional"
            with patch("modiff.runtime_profile.load_manifest", return_value=manifest):
                selected_change = lock_digest(requirement, contract_paths=(requirement,), profile="cpu")

        self.assertEqual(unrelated_change, baseline)
        self.assertNotEqual(selected_change, baseline)

    def write_profile(self, directory: str, profile: str):
        root = Path(directory)
        manifest = load_manifest()
        requirement = MANIFEST_PATH.parents[2] / manifest["profiles"][profile]["requirements"]
        (root / "modiff-profile.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "profile": profile,
                    "support_tier": manifest["profiles"][profile]["tier"],
                    "manifest_revision": manifest["revision"],
                    "requirements": requirement.name,
                    "runtime_contract_schema": RUNTIME_CONTRACT_SCHEMA,
                    "lock_digest": lock_digest(
                        requirement,
                        contract_paths=runtime_contract_paths(requirement),
                        profile=profile,
                    ),
                }
            ),
            encoding="utf-8",
        )
        return root

    def test_managed_rocm_profile_rejects_a_cuda_torch_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            venv = self.write_profile(temporary, "amd-rocm-linux")
            profile = runtime_profile(
                hardware(version="2.11.0+cu128", cuda_version="12.8"),
                venv=venv,
            )

        self.assertFalse(profile["execution_ready"])
        self.assertEqual(profile["status"], "mismatch")
        self.assertEqual(profile["installed"], "nvidia-cuda")
        self.assertIn("profile-mismatch", [issue["code"] for issue in profile["issues"]])

    def test_rocm_profile_requires_a_successful_device_tensor(self):
        with tempfile.TemporaryDirectory() as temporary:
            venv = self.write_profile(temporary, "amd-rocm-linux")
            with patch(
                "modiff.runtime_profile._device_tensor_probe",
                return_value={"ready": False, "device": "cuda:0", "message": "tensor failed"},
            ):
                profile = runtime_profile(
                    hardware(
                        version="2.9.1+rocm7.2.0",
                        hip_version="7.2.0",
                        cuda_available=True,
                    ),
                    venv=venv,
                )

        self.assertFalse(profile["execution_ready"])
        self.assertIn("device-tensor-failed", [issue["code"] for issue in profile["issues"]])

    def test_matching_rocm_profile_is_ready_after_device_tensor(self):
        with tempfile.TemporaryDirectory() as temporary:
            venv = self.write_profile(temporary, "amd-rocm-linux")
            with patch(
                "modiff.runtime_profile._device_tensor_probe",
                return_value={"ready": True, "device": "cuda:0", "message": None},
            ):
                profile = runtime_profile(
                    hardware(
                        version="2.9.1+rocm7.2.0",
                        hip_version="7.2.0",
                        cuda_available=True,
                    ),
                    venv=venv,
                )

        self.assertTrue(profile["execution_ready"])
        self.assertEqual(profile["status"], "ready")
        self.assertEqual(profile["device_validation"]["device"], "cuda:0")
        self.assertEqual(profile["runtime_contract"]["status"], "verified")

    def test_matching_intel_xpu_profile_is_ready_after_device_tensor(self):
        with tempfile.TemporaryDirectory() as temporary:
            venv = self.write_profile(temporary, "intel-xpu")
            with patch(
                "modiff.runtime_profile._device_tensor_probe",
                return_value={"ready": True, "device": "xpu:0", "message": None},
            ):
                profile = runtime_profile(
                    hardware(version="2.12.1+xpu", xpu_available=True),
                    venv=venv,
                )

        self.assertTrue(profile["execution_ready"])
        self.assertEqual(profile["installed"], "intel-xpu")
        self.assertEqual(profile["device_validation"]["device"], "xpu:0")

    def test_changed_runtime_contract_requires_repair_before_preflight_is_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            venv = self.write_profile(temporary, "cpu")
            with (
                patch("modiff.runtime_profile.lock_digest", return_value="f" * 64),
                patch(
                    "modiff.runtime_profile._device_tensor_probe",
                    return_value={"ready": True, "device": "cpu", "message": None},
                ),
            ):
                profile = runtime_profile(hardware(version="2.8.0"), venv=venv)

        self.assertFalse(profile["execution_ready"])
        self.assertEqual(profile["status"], "repair-required")
        self.assertTrue(profile["repair_required"])
        self.assertEqual(profile["runtime_contract"]["status"], "drifted")
        self.assertIn("runtime-contract-drift", [issue["code"] for issue in profile["issues"]])
        self.assertEqual(profile["repair_command"], "./install.sh --accelerator cpu --repair")

    def test_legacy_contract_record_is_non_blocking_when_runtime_checks_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            venv = Path(temporary)
            (venv / "modiff-profile.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "profile": "cpu",
                        "support_tier": "supported",
                        "requirements": "cpu.txt",
                        "lock_digest": "a" * 64,
                        "runtime_contract_files": [
                            "requirements/profiles/cpu.txt",
                            "pyproject.toml",
                            "modiff/compatibility/accelerators.v1.json",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch(
                "modiff.runtime_profile._device_tensor_probe",
                return_value={"ready": True, "device": "cpu", "message": None},
            ):
                profile = runtime_profile(hardware(version="2.8.0"), venv=venv)

        self.assertTrue(profile["execution_ready"])
        self.assertFalse(profile["repair_required"])
        self.assertEqual(profile["runtime_contract"]["status"], "legacy")
        self.assertIn("runtime-contract-legacy", [issue["code"] for issue in profile["issues"]])

    def test_experimental_profile_repair_command_includes_required_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            venv = Path(temporary)
            (venv / "modiff-profile.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "profile": "cpu",
                        "support_tier": "experimental",
                        "lock_digest": "a" * 64,
                        "runtime_contract_files": ["requirements/profiles/cpu.txt"],
                    }
                ),
                encoding="utf-8",
            )
            with patch(
                "modiff.runtime_profile._device_tensor_probe",
                return_value={"ready": True, "device": "cpu", "message": None},
            ):
                profile = runtime_profile(hardware(version="2.8.0"), venv=venv)

        self.assertEqual(
            profile["repair_command"],
            "./install.sh --accelerator cpu --repair --allow-experimental",
        )

    def test_saved_profile_without_contract_digest_requires_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            venv = Path(temporary)
            (venv / "modiff-profile.json").write_text(
                json.dumps({"schema_version": 1, "profile": "cpu"}),
                encoding="utf-8",
            )
            with patch(
                "modiff.runtime_profile._device_tensor_probe",
                return_value={"ready": True, "device": "cpu", "message": None},
            ):
                profile = runtime_profile(hardware(version="2.8.0"), venv=venv)

        self.assertEqual(profile["status"], "repair-required")
        self.assertEqual(profile["runtime_contract"]["status"], "unverified")
        self.assertIn("runtime-contract-unverified", [issue["code"] for issue in profile["issues"]])


if __name__ == "__main__":
    unittest.main()
