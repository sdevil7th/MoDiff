"""CPU metadata/contract fixtures; no optional import, install, or device call."""
import importlib.util
import os
from pathlib import Path
from packaging.tags import Tag
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid
import tempfile

from modiff import optional_runtimes as runtimes
from modiff import optimization_packages as optimization

PROFILE_ID = runtimes.TRANSFORMERS_MAIN_PEFT_BITSANDBYTES_RUNTIME_PROFILE_ID
WINDOWS_LOCK = {
    "distribution": "bitsandbytes", "version": "0.50.0",
    "filename": "bitsandbytes-0.50.0-py3-none-win_amd64.whl",
    "url": "https://files.pythonhosted.org/packages/78/40/7b2253f94ec8c6235606aa393f315d941d02c44f141b7bccae14202b3fea/bitsandbytes-0.50.0-py3-none-win_amd64.whl",
    "sha256": "7f9ebf51d76a00341711cdd557af97555274b5aad3444975ff9927866a18bc11",
    "byteSize": 37961095, "platform": "windows", "pythonTag": "cp312", "machine": "x86_64",
}


class DeliveryTests(unittest.TestCase):
    def test_exact_windows_lock_addition(self):
        profile = runtimes.OPTIONAL_RUNTIME_PROFILES[PROFILE_ID]
        self.assertEqual(profile.artifact_locks, (*runtimes._bitsandbytes_linux_x86_64_artifact_locks(), WINDOWS_LOCK))
        self.assertEqual({package.distribution for package in profile.packages}, {"bitsandbytes"})
        self.assertEqual(profile.satisfies_profiles, ())
        self.assertEqual(profile.source_builds, ())

    def test_only_one_target_changes(self):
        profile = runtimes.OPTIONAL_RUNTIME_PROFILES[PROFILE_ID]
        for platform_name, _, machine in runtimes._OPTIONAL_RUNTIME_TARGETS:
            with self.subTest(platform=platform_name, machine=machine):
                target = profile.contract_for_target(platform_name=platform_name, machine=machine)
                qualified = (platform_name, machine) == ("windows", "x86_64")
                self.assertEqual(target.contract_state, "qualified" if qualified else "candidate_unqualified")
                self.assertEqual((target.cutover_ready, target.install_action_available, target.activation_available), (qualified,) * 3)
        self.assertEqual((profile.cutover_ready, profile.install_action_available, profile.activation_available), (False,) * 3)

    def test_spec_equals_actual_temporary_installation(self):
        self.assertEqual(runtimes.OPTIONAL_RUNTIME_PROFILES[PROFILE_ID].spec_digest,
                         "sha256:3905f3876edeb3a6cb79af897a3d04d8836d3a63573fec12ce3fccf4ddb67053")

    def test_stale_or_nonconsented_request_is_refused_before_base_or_installer(self):
        profile = runtimes.OPTIONAL_RUNTIME_PROFILES[PROFILE_ID]
        with (mock.patch.object(optimization, "current_base_binding") as base,
              mock.patch.object(optimization, "_verified_uv_executable") as installer):
            for digest, consent in [("sha256:" + "0" * 64, True), (profile.spec_digest, False)]:
                with self.subTest(digest=digest, consent=consent), self.assertRaises(ValueError):
                    optimization.validate_optional_runtime_install_request(PROFILE_ID, digest, consent=consent)
            base.assert_not_called()
            installer.assert_not_called()

    def test_windows_install_request_uses_only_exact_wheel_and_base_owned_libraries(self):
        profile = runtimes.OPTIONAL_RUNTIME_PROFILES[PROFILE_ID]
        with (mock.patch.object(runtimes, "optional_runtime_target", return_value=("windows", "x86_64")),
              mock.patch.object(optimization, "_platform_name", return_value="windows"),
              mock.patch.object(optimization, "_machine_name", return_value="x86_64"),
              mock.patch("packaging.tags.sys_tags", return_value=(Tag("py3", "none", "win_amd64"),)),
              mock.patch.object(optimization, "current_base_binding", return_value={}) as base,
              mock.patch.object(optimization, "_verified_uv_executable", return_value="reviewed-uv-no-execution")):
            request = optimization.validate_optional_runtime_install_request(PROFILE_ID, profile.spec_digest, consent=True)
        self.assertEqual(request["artifactLocks"], [WINDOWS_LOCK])
        self.assertEqual(request["sourceBuilds"], [])
        self.assertTrue({"torch", "transformers", "peft", "diffusers"} - {"diffusers"} <=
                        {row["distribution"] for row in request["baseContracts"]})
        base.assert_called_once()

    def test_linux_install_request_remains_refused_before_base_or_installer(self):
        profile = runtimes.OPTIONAL_RUNTIME_PROFILES[PROFILE_ID]
        with (mock.patch.object(runtimes, "optional_runtime_target", return_value=("linux", "x86_64")),
              mock.patch.object(optimization, "current_base_binding") as base,
              mock.patch.object(optimization, "_verified_uv_executable") as installer):
            with self.assertRaisesRegex(RuntimeError, "not qualified for installation"):
                optimization.validate_optional_runtime_install_request(PROFILE_ID, profile.spec_digest, consent=True)
        base.assert_not_called()
        installer.assert_not_called()


class CatalogTests(unittest.TestCase):
    def setUp(self):
        review_root = os.environ.get("MODIFF_METADATA_TEST_ROOT")
        if review_root is None:
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            review_root = temporary.name
        self.root = Path(review_root) / uuid.uuid4().hex
        self.root.mkdir()
        self.native = self.root / "native"
        self.foreign = self.root / "foreign"
        self.environments = self.root / "managed/optimizations/environments"
        self.environment_id = "runtime-9-cafebabe"
        self.site = self.environments / self.environment_id / "site-packages"
        for root in [self.native, self.foreign, self.site]:
            (root / "bitsandbytes").mkdir(parents=True)
            (root / "bitsandbytes/__init__.py").write_bytes(b"# metadata fixture only; never imported\n")
            (root / "bitsandbytes-0.50.0.dist-info").mkdir()
        self.profile = runtimes.OPTIONAL_RUNTIME_PROFILES[PROFILE_ID]

    def observe(self, *, origin="overlay", change=None, runtime="nvidia-cuda", loaded=True, platform_name="windows", catalog_mode="supplied"):
        root = {"overlay": self.site, "native": self.native, "foreign": self.foreign}[origin]
        profile = {
            "id": PROFILE_ID, "specDigest": self.profile.spec_digest, "status": "present_unqualified",
            "contractState": "qualified", "cutoverReady": True, "installActionAvailable": True,
            "activationAvailable": True, "overlayStatus": "active", "platform": platform_name, "machine": "x86_64",
        }
        state = {
            "activeEnvironmentId": self.environment_id, "activeTrustClass": "artifact_locked_optional",
            "enabledCapabilities": [], "_storageStatus": "ok",
        }
        record = {"kind": "optional_runtime", "id": PROFILE_ID, "specDigest": self.profile.spec_digest}
        inspection = {
            "status": "recorded", "sitePackages": self.site,
            "manifest": {"trustClass": "artifact_locked_optional", "specs": [record]},
            "validation": {"status": "passed"},
        }
        catalog = {
            "schemaVersion": 1, "profiles": [profile],
            "overlay": {"processLoadStatus": "active", "state": {
                "activeEnvironmentId": self.environment_id, "activeStatus": "active",
            }},
        }
        version = "0.50.0"
        module_root = root
        if change:
            kind, value = change
            if kind == "profile":
                field, val = value
                profile[field] = val
            elif kind == "state":
                field, val = value
                state[field] = val
            elif kind == "process":
                catalog["overlay"]["processLoadStatus"] = value
            elif kind == "inspection":
                inspection["status"] = value
            elif kind == "manifestDigest":
                record["specDigest"] = value
            elif kind == "version":
                version = value
            elif kind == "moduleRoot":
                module_root = value
        distribution = SimpleNamespace(_path=root / "bitsandbytes-0.50.0.dist-info", version=version)
        spec = SimpleNamespace(origin=str(module_root / "bitsandbytes/__init__.py"),
                               submodule_search_locations=[str(module_root / "bitsandbytes")])
        environment = {"MODIFF_RUNTIME_OVERLAY_STATUS": "active" if loaded else "base",
                       "MODIFF_OPTIMIZATION_ENVIRONMENT": self.environment_id if loaded else ""}
        with (
            mock.patch.object(optimization, "read_state", return_value=state),
            mock.patch.object(optimization, "ENVIRONMENTS_DIR", self.environments),
            mock.patch.object(optimization, "MANAGED_ROOT", self.root / "managed"),
            mock.patch.object(optimization, "_verified_existing_managed_directory", return_value=self.environments),
            mock.patch.object(optimization, "_environment_inspection", return_value=inspection),
            mock.patch.object(optimization, "_catalog_environment_ids", return_value=([self.environment_id], {})),
            mock.patch.object(optimization, "active_install", return_value=None),
            mock.patch.object(optimization, "_package_version", side_effect=lambda name: version if name == "bitsandbytes" else None),
            mock.patch.object(runtimes, "optional_runtime_target", return_value=(platform_name, "x86_64")),
            mock.patch.object(optimization, "_normalized_platform", return_value=platform_name),
            mock.patch.object(importlib.metadata, "distribution", return_value=distribution),
            mock.patch.object(importlib.util, "find_spec", return_value=spec),
            mock.patch("sysconfig.get_path", return_value=str(self.native)),
            mock.patch.dict(os.environ, environment),
        ):
            with mock.patch.object(optimization, "public_optional_runtime_catalog",
                                   wraps=optimization.public_optional_runtime_catalog) as publisher:
                kwargs = {"runtime_profile": {"installed": runtime},
                          "hardware": {"torch": {"version": "2.11.0+cu128"}},
                          "optional_runtime_catalog": catalog}
                if catalog_mode == "publisher-default":
                    kwargs.pop("optional_runtime_catalog")
                elif catalog_mode == "publisher-none":
                    kwargs["optional_runtime_catalog"] = None
                elif catalog_mode == "publisher-supplied":
                    kwargs["optional_runtime_catalog"] = publisher()
                elif catalog_mode == "malformed-supplied":
                    kwargs["optional_runtime_catalog"] = {}
                value = optimization.public_catalog(**kwargs)
                if catalog_mode in {"publisher-default", "publisher-none", "publisher-supplied"}:
                    publisher.assert_called_once_with()
                else:
                    publisher.assert_not_called()
        return next(row for row in value["capabilities"] if row["id"] == "bitsandbytes")

    def test_actual_active_overlay_is_not_mislabeled_as_base(self):
        row = self.observe()
        self.assertEqual(row["delivery"], "optional_overlay")
        self.assertTrue(row["deliveryQualified"] and row["availableForExecution"])
        self.assertEqual(row["optionalRuntimeProfileId"], PROFILE_ID)
        self.assertFalse(row["canInstall"] or row["canEnable"])

    def test_linux_overlay_metadata_does_not_grant_delivery(self):
        row = self.observe(platform_name="linux")
        self.assertFalse(row["deliveryQualified"] or row["availableForExecution"])
        self.assertEqual(row["delivery"], "unqualified")

    def test_existing_quanto_contract_is_preserved(self):
        profile_id = runtimes.TRANSFORMERS_MAIN_PEFT_QUANTO_RUNTIME_PROFILE_ID
        catalog = {"profiles": [{"id": profile_id, "contractState": "qualified",
                                 "cutoverReady": True, "overlayStatus": "active"}]}
        with (mock.patch.object(optimization, "read_state", return_value={"enabledCapabilities": []}),
              mock.patch.object(optimization, "_package_version", return_value="0.2.7"),
              mock.patch.object(optimization, "_normalized_platform", return_value="linux")):
            result = optimization.public_catalog(runtime_profile={"installed": "amd-rocm-linux"},
                                                  hardware={}, optional_runtime_catalog=catalog)
        row = next(item for item in result["capabilities"] if item["id"] == "optimum_quanto")
        self.assertTrue(row["deliveryQualified"] and row["availableForExecution"])
        self.assertEqual(row["delivery"], "optional_overlay")
        self.assertEqual(row["optionalRuntimeProfileId"], profile_id)

    def test_native_specialist_delivery_is_preserved(self):
        row = self.observe(origin="native", loaded=False)
        self.assertEqual(row["delivery"], "base_profile")
        self.assertTrue(row["deliveryQualified"] and row["availableForExecution"])
        self.assertIsNone(row["optionalRuntimeProfileId"])

    def test_foreign_package_origin_is_not_base_or_overlay_authority(self):
        row = self.observe(origin="foreign")
        self.assertEqual(row["delivery"], "unqualified")
        self.assertFalse(row["deliveryQualified"] or row["availableForExecution"])

    def test_module_and_metadata_origins_must_agree(self):
        row = self.observe(change=("moduleRoot", self.foreign))
        self.assertFalse(row["deliveryQualified"] or row["availableForExecution"])

    def test_active_overlay_requires_current_profile_state_and_digest(self):
        cases = [
            ("profile", ("specDigest", "sha256:" + "0" * 64)),
            ("profile", ("contractState", "candidate_unqualified")),
            ("profile", ("cutoverReady", False)),
            ("profile", ("installActionAvailable", False)),
            ("profile", ("activationAvailable", False)),
            ("profile", ("status", "wrong_version")),
            ("profile", ("overlayStatus", "staged")),
            ("profile", ("overlayStatus", "repair_required")),
            ("profile", ("platform", "linux")),
            ("profile", ("machine", "arm64")),
            ("state", ("activeEnvironmentId", "runtime-8-deadbeef")),
            ("state", ("activeTrustClass", "legacy_optimization")),
            ("state", ("_storageStatus", "repair_required")),
            ("process", "base"), ("process", "restart_required"),
            ("inspection", "repair_required"),
            ("manifestDigest", "sha256:" + "1" * 64),
            ("version", "0.49.2"),
        ]
        for change in cases:
            with self.subTest(change=change):
                row = self.observe(change=change)
                self.assertFalse(row["deliveryQualified"] or row["availableForExecution"])

    def test_old_worker_does_not_gain_an_active_overlay(self):
        self.assertFalse(self.observe(loaded=False)["deliveryQualified"])

    def test_package_delivery_does_not_imply_cpu_or_rocm_execution(self):
        for runtime in ["cpu", "amd-rocm-linux", "intel-xpu", "apple-mps"]:
            with self.subTest(runtime=runtime):
                row = self.observe(runtime=runtime)
                self.assertFalse(row["compatible"] or row["availableForExecution"])
                self.assertIn("Requires one of", row["disabledReason"])


    def test_default_catalog_uses_actual_metadata_publisher_once(self):
        for catalog_mode in ("publisher-default", "publisher-none"):
            with self.subTest(catalog_mode=catalog_mode):
                row = self.observe(catalog_mode=catalog_mode)
                self.assertEqual(row["delivery"], "optional_overlay")
                self.assertTrue(row["deliveryQualified"] and row["availableForExecution"])
                self.assertEqual(row["optionalRuntimeProfileId"], PROFILE_ID)

    def test_explicit_actual_publisher_snapshot_is_not_resolved_twice(self):
        row = self.observe(catalog_mode="publisher-supplied")
        self.assertEqual(row["delivery"], "optional_overlay")
        self.assertTrue(row["deliveryQualified"] and row["availableForExecution"])

    def test_explicit_malformed_snapshot_does_not_trigger_default_resolution(self):
        row = self.observe(catalog_mode="malformed-supplied")
        self.assertEqual(row["delivery"], "unqualified")
        self.assertFalse(row["deliveryQualified"] or row["availableForExecution"])

    def test_default_actual_publisher_keeps_current_delivery_guards(self):
        cases = [
            {"origin": "foreign"}, {"loaded": False}, {"platform_name": "linux"},
            {"runtime": "cpu"}, {"runtime": "amd-rocm-linux"},
            {"change": ("state", ("activeTrustClass", "legacy_optimization"))},
            {"change": ("state", ("_storageStatus", "repair_required"))},
            {"change": ("inspection", "repair_required")},
            {"change": ("manifestDigest", "sha256:" + "0" * 64)},
            {"change": ("version", "0.49.2")},
            {"change": ("moduleRoot", self.foreign)},
        ]
        for case in cases:
            with self.subTest(case=case):
                row = self.observe(catalog_mode="publisher-default", **case)
                self.assertFalse(row["availableForExecution"])
                if case.get("runtime") not in {"cpu", "amd-rocm-linux"}:
                    self.assertFalse(row["deliveryQualified"])


    def test_default_quanto_uses_genuine_catalog_with_existing_contract(self):
        profile_id = runtimes.TRANSFORMERS_MAIN_PEFT_QUANTO_RUNTIME_PROFILE_ID
        profile = runtimes.OPTIONAL_RUNTIME_PROFILES[profile_id]
        state = {"activeEnvironmentId": self.environment_id, "activeTrustClass": "artifact_locked_optional",
                 "enabledCapabilities": [], "_storageStatus": "ok"}
        inspection = {"status": "recorded", "sitePackages": self.site,
                      "manifest": {"trustClass": "artifact_locked_optional", "specs": [
                          {"kind": "optional_runtime", "id": profile_id, "specDigest": profile.spec_digest}]},
                      "validation": {"status": "passed"}}
        with (
            mock.patch.object(optimization, "read_state", return_value=state),
            mock.patch.object(optimization, "ENVIRONMENTS_DIR", self.environments),
            mock.patch.object(optimization, "MANAGED_ROOT", self.root / "managed"),
            mock.patch.object(optimization, "_verified_existing_managed_directory", return_value=self.environments),
            mock.patch.object(optimization, "_environment_inspection", return_value=inspection),
            mock.patch.object(optimization, "_catalog_environment_ids", return_value=([self.environment_id], {})),
            mock.patch.object(optimization, "active_install", return_value=None),
            mock.patch.object(optimization, "_package_version", side_effect=lambda name: "0.2.7" if name == "optimum-quanto" else None),
            mock.patch.object(runtimes, "optional_runtime_target", return_value=("linux", "x86_64")),
            mock.patch.object(optimization, "_normalized_platform", return_value="linux"),
            mock.patch.object(importlib.metadata, "version", side_effect=lambda name: "0.2.7" if name == "optimum-quanto" else None),
            mock.patch.object(importlib.util, "find_spec", return_value=SimpleNamespace(origin="metadata-fixture-only")),
            mock.patch.dict(os.environ, {"MODIFF_RUNTIME_OVERLAY_STATUS": "active",
                                       "MODIFF_OPTIMIZATION_ENVIRONMENT": self.environment_id}),
            mock.patch.object(optimization, "public_optional_runtime_catalog",
                              wraps=optimization.public_optional_runtime_catalog) as publisher,
        ):
            value = optimization.public_catalog(runtime_profile={"installed": "amd-rocm-linux"}, hardware={})
            publisher.assert_called_once_with()
            row = next(item for item in value["capabilities"] if item["id"] == "optimum_quanto")
        self.assertEqual(row["delivery"], "optional_overlay")
        self.assertTrue(row["deliveryQualified"] and row["availableForExecution"])
        self.assertEqual(row["optionalRuntimeProfileId"], profile_id)


if __name__ == "__main__":
    unittest.main()
