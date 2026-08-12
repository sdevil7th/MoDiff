import json
import os
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from modiff import optimization_packages as optimizations
from modiff import runtime_overlays


class OptimizationPackageTests(unittest.TestCase):
    ACTIVE_ENVIRONMENT_ID = "runtime-9-cafebabe"

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.path_patchers = [
            mock.patch.object(optimizations, "OPTIMIZATION_ROOT", root),
            mock.patch.object(optimizations, "ENVIRONMENTS_DIR", root / "environments"),
            mock.patch.object(optimizations, "STAGING_DIR", root / "staging"),
            mock.patch.object(optimizations, "ARTIFACTS_DIR", root / "artifacts"),
            mock.patch.object(optimizations, "MANAGED_ROOT", root),
            mock.patch.object(optimizations, "STATE_PATH", root / "state.json"),
            mock.patch.object(optimizations, "RECEIPTS_PATH", root / "receipts.json"),
            mock.patch.object(optimizations, "PROMOTION_PATH", root / "promotion.json"),
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

    def set_active_environment(self, environment_id):
        state = optimizations._default_state()
        state["activeEnvironmentId"] = environment_id
        state["activeTrustClass"] = "legacy_optimization"
        optimizations._write_state(state)

    @staticmethod
    def lease():
        return SimpleNamespace(cancel_event=threading.Event())

    def test_catalog_is_profile_gated_and_disabled_by_default(self):
        catalog = optimizations.public_catalog(
            runtime_profile={"installed": "amd-rocm-linux"},
            hardware={"torch": {"version": "2.9.1+rocm7.2"}, "amd_architectures": ["gfx1151"]},
        )
        by_id = {item["id"]: item for item in catalog["capabilities"]}
        self.assertTrue(by_id["torchao"]["compatible"])
        self.assertFalse(by_id["hub_attention_kernels"]["compatible"])
        self.assertFalse(by_id["torchao"]["enabled"])
        self.assertFalse(by_id["torchao"]["canInstall"])
        self.assertFalse(by_id["torchao"]["canEnable"])
        self.assertIn("immutable artifact lock", by_id["torchao"]["disabledReason"])

    def test_locked_requirements_keep_windows_file_hash_out_of_url_path(self):
        wheel = optimizations.OPTIMIZATION_ROOT / "demo_pkg-1.0.0-py3-none-any.whl"
        wheel.write_bytes(b"wheel")
        digest = "a" * 64
        body = optimizations._locked_requirements_body(
            [{"distribution": "demo-pkg", "sha256": digest}],
            [wheel],
        ).decode("utf-8")
        self.assertIn("demo-pkg @ file:///", body)
        self.assertIn(f" --hash=sha256:{digest}\n", body)
        self.assertNotIn(".whl#sha256=", body)

    def test_overlay_installer_never_links_staged_files_to_a_shared_cache(self):
        source = Path(optimizations.__file__).read_text(encoding="utf-8")
        command = source[source.index('command = [') : source.index('install_result =', source.index('command = ['))]
        self.assertIn('"--link-mode",\n            "copy",', command)

    @staticmethod
    def promotion_inspection(environment_id, *, environment_root, **_kwargs):
        candidate = environment_root / environment_id
        if not candidate.is_dir():
            return {"status": "repair_required"}
        return {
            "status": "ready",
            "manifest": {"schemaVersion": 2, "id": environment_id},
            "validation": {"schemaVersion": 2, "environmentId": environment_id},
        }

    def test_prepared_promotion_journal_recovers_exact_staged_environment(self):
        environment_id = "runtime-9-cafebabe"
        staged = optimizations.STAGING_DIR / environment_id
        destination = optimizations.ENVIRONMENTS_DIR / environment_id
        (staged / "site-packages").mkdir(parents=True)
        optimizations.ENVIRONMENTS_DIR.mkdir()
        (staged / "site-packages" / "proof.txt").write_text("reviewed", encoding="utf-8")
        inspection = self.promotion_inspection(
            environment_id,
            environment_root=optimizations.STAGING_DIR,
        )
        anchor = optimizations._promotion_anchor(inspection)
        optimizations._write_promotion_record(
            environment_id,
            phase="prepared",
            anchor=anchor,
        )
        lease = runtime_overlays.InstallLease(
            token="recovery",
            owner_kind="test",
            owner_id=environment_id,
            cancel_event=threading.Event(),
        )
        with (
            mock.patch.object(runtime_overlays, "MANAGED_ROOT", optimizations.MANAGED_ROOT),
            mock.patch.object(runtime_overlays, "_ACTIVE_INSTALL", lease),
            mock.patch.object(
                optimizations,
                "_environment_inspection",
                side_effect=self.promotion_inspection,
            ),
        ):
            recovered = optimizations._reconcile_promotion(lease)
        self.assertEqual(recovered, environment_id)
        self.assertTrue(lease.committed)
        self.assertFalse(staged.exists())
        self.assertEqual((destination / "site-packages" / "proof.txt").read_text(), "reviewed")
        self.assertFalse(optimizations.PROMOTION_PATH.exists())

    def test_prepared_journal_acknowledges_crash_after_exact_rename(self):
        environment_id = "runtime-9-cafebabe"
        destination = optimizations.ENVIRONMENTS_DIR / environment_id
        destination.mkdir(parents=True)
        inspection = self.promotion_inspection(
            environment_id,
            environment_root=optimizations.ENVIRONMENTS_DIR,
        )
        optimizations._write_promotion_record(
            environment_id,
            phase="prepared",
            anchor=optimizations._promotion_anchor(inspection),
        )
        lease = self.lease()
        with (
            mock.patch.object(
                optimizations,
                "_environment_inspection",
                side_effect=self.promotion_inspection,
            ),
            mock.patch.object(optimizations, "promote_staged_environment") as promote,
            mock.patch.object(runtime_overlays, "MANAGED_ROOT", optimizations.MANAGED_ROOT),
        ):
            self.assertEqual(optimizations._reconcile_promotion(lease), environment_id)
        promote.assert_not_called()
        self.assertTrue(destination.is_dir())
        self.assertFalse(optimizations.PROMOTION_PATH.exists())

    def test_ambiguous_promotion_journal_fails_closed_without_mutation(self):
        environment_id = "runtime-9-cafebabe"
        staged = optimizations.STAGING_DIR / environment_id
        destination = optimizations.ENVIRONMENTS_DIR / environment_id
        staged.mkdir(parents=True)
        destination.mkdir(parents=True)
        inspection = self.promotion_inspection(
            environment_id,
            environment_root=optimizations.STAGING_DIR,
        )
        optimizations._write_promotion_record(
            environment_id,
            phase="prepared",
            anchor=optimizations._promotion_anchor(inspection),
        )
        with self.assertRaisesRegex(RuntimeError, "ambiguous filesystem state"):
            optimizations._reconcile_promotion(self.lease())
        self.assertTrue(staged.is_dir())
        self.assertTrue(destination.is_dir())
        self.assertTrue(optimizations.PROMOTION_PATH.is_file())

    def test_malformed_promotion_journal_never_downgrades_to_no_record(self):
        optimizations.OPTIMIZATION_ROOT.mkdir(parents=True, exist_ok=True)
        optimizations.PROMOTION_PATH.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "environmentId": "../escape",
                    "phase": "prepared",
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "promotion journal requires repair"):
            optimizations._reconcile_promotion(self.lease())
        self.assertTrue(optimizations.PROMOTION_PATH.is_file())

    def test_legacy_activation_is_unqualified_and_rollback_deactivates_to_base(self):
        self.set_active_environment("runtime-1-deadbeef")
        with self.assertRaisesRegex(RuntimeError, "unqualified"):
            optimizations.activate_environment("runtime-2-feedface")
        with (
            mock.patch.object(
                optimizations, "reserve_install", side_effect=lambda *_args: self.lease()
            ),
            mock.patch.object(optimizations, "release_install"),
        ):
            rolled_back = optimizations.rollback_environment()
        self.assertIsNone(rolled_back["state"]["activeEnvironmentId"])
        self.assertTrue(rolled_back["restartRequired"])

    def test_startup_never_imports_or_inserts_a_legacy_overlay(self):
        self.set_active_environment("runtime-1-deadbeef")
        original_path = list(optimizations.sys.path)
        with (
            mock.patch.object(
                optimizations, "reserve_install", side_effect=lambda *_args: self.lease()
            ),
            mock.patch.object(optimizations, "release_install"),
            mock.patch.object(
                optimizations,
                "_safe_environment_path",
                side_effect=AssertionError("legacy overlay must not be inspected for import"),
            ) as safe_path,
            mock.patch.dict(os.environ, {}, clear=False),
        ):
            self.assertIsNone(optimizations.activate_runtime_overlay())
            self.assertEqual(
                os.environ.get("MODIFF_RUNTIME_OVERLAY_STATUS"), "repair_required"
            )
        safe_path.assert_not_called()
        self.assertEqual(optimizations.sys.path, original_path)

    def test_schema_one_environment_requires_repair(self):
        self.create_environment("legacy-v1")
        self.assertEqual(
            optimizations._environment_inspection("legacy-v1")["status"],
            "repair_required",
        )

    def test_existing_corrupt_state_is_recovery_only_and_explicit_rollback_repairs_it(self):
        corrupt_documents = {
            "malformed": b"{",
            "duplicate": b'{"schemaVersion":2,"schemaVersion":2}',
            "oversized": b" " * (32 * 1024 * 1024 + 1),
        }
        for label, body in corrupt_documents.items():
            with self.subTest(label=label):
                optimizations.STATE_PATH.write_bytes(body)
                self.assertEqual(
                    optimizations.read_state()["_storageStatus"], "repair_required"
                )
                with (
                    mock.patch.object(
                        optimizations,
                        "reserve_install",
                        side_effect=lambda *_args: self.lease(),
                    ),
                    mock.patch.object(optimizations, "release_install"),
                    mock.patch.dict(os.environ, {}, clear=False),
                ):
                    self.assertIsNone(optimizations.activate_runtime_overlay())
                    self.assertEqual(
                        os.environ.get("MODIFF_RUNTIME_OVERLAY_STATUS"),
                        "repair_required",
                    )
                    repaired = optimizations.rollback_environment()
                self.assertIsNone(repaired["state"]["activeEnvironmentId"])
                self.assertEqual(optimizations.read_state()["_storageStatus"], "ok")

    def test_semantically_noncanonical_state_is_always_recovery_only(self):
        active = "runtime-1-deadbeef"
        cases = {}
        wrong_schema = optimizations._default_state()
        wrong_schema["schemaVersion"] = 999
        cases["wrong schema"] = wrong_schema
        extra_key = optimizations._default_state()
        extra_key["privatePath"] = "C:/private"
        cases["extra key"] = extra_key
        invalid_id = optimizations._default_state()
        invalid_id.update(
            {"activeEnvironmentId": "../escape", "activeTrustClass": "legacy_optimization"}
        )
        cases["invalid id"] = invalid_id
        duplicate_capability = optimizations._default_state()
        duplicate_capability["enabledCapabilities"] = ["regional_compile", "regional_compile"]
        cases["duplicate capability"] = duplicate_capability
        unknown_capability = optimizations._default_state()
        unknown_capability["enabledCapabilities"] = ["unknown_capability"]
        cases["unknown capability"] = unknown_capability
        too_many = optimizations._default_state()
        too_many["enabledCapabilities"] = ["regional_compile"] * 65
        cases["too many capabilities"] = too_many
        invalid_time = optimizations._default_state()
        invalid_time["updatedAt"] = "C:/private/time"
        cases["invalid timestamp"] = invalid_time
        same_ids = optimizations._default_state()
        same_ids.update(
            {
                "activeEnvironmentId": active,
                "previousEnvironmentId": active,
                "activeTrustClass": "artifact_locked_optional",
                "previousTrustClass": "artifact_locked_optional",
            }
        )
        cases["same active and previous"] = same_ids

        for label, document in cases.items():
            with self.subTest(label=label):
                optimizations.STATE_PATH.write_text(json.dumps(document), encoding="utf-8")
                state = optimizations.read_state()
                self.assertEqual(state["_storageStatus"], "repair_required")
                with mock.patch.dict(os.environ, {}, clear=False):
                    catalog = optimizations.public_optional_runtime_catalog()
                self.assertEqual(
                    catalog["overlay"]["processLoadStatus"], "repair_required"
                )
                with self.assertRaisesRegex(RuntimeError, "corrupt runtime state"):
                    optimizations.set_capability_enabled("regional_compile", True)

    def test_state_symlink_reset_never_touches_external_target(self):
        external = Path(self.temporary.name) / "outside-state.json"
        external.write_text("external-sentinel", encoding="utf-8")
        try:
            optimizations.STATE_PATH.symlink_to(external)
        except OSError as exc:
            self.skipTest(f"state symlink unavailable: {exc}")
        self.assertEqual(optimizations.read_state()["_storageStatus"], "repair_required")
        with (
            mock.patch.object(
                optimizations, "reserve_install", side_effect=lambda *_args: self.lease()
            ),
            mock.patch.object(optimizations, "release_install"),
        ):
            optimizations.rollback_environment()
        self.assertEqual(external.read_text(encoding="utf-8"), "external-sentinel")
        self.assertFalse(optimizations.STATE_PATH.is_symlink())
        self.assertEqual(optimizations.read_state()["_storageStatus"], "ok")

    def test_state_hardlink_reset_never_overwrites_external_target(self):
        external = Path(self.temporary.name) / "outside-hardlink-state.json"
        sentinel = json.dumps(optimizations._default_state(), sort_keys=True)
        external.write_text(sentinel, encoding="utf-8")
        os.link(external, optimizations.STATE_PATH)
        self.assertEqual(optimizations.read_state()["_storageStatus"], "repair_required")
        with (
            mock.patch.object(
                optimizations, "reserve_install", side_effect=lambda *_args: self.lease()
            ),
            mock.patch.object(optimizations, "release_install"),
        ):
            optimizations.rollback_environment()
        self.assertEqual(external.read_text(encoding="utf-8"), sentinel)
        self.assertEqual(optimizations.read_state()["_storageStatus"], "ok")

    @unittest.skipUnless(os.name == "nt", "Windows junction regression")
    def test_optimization_root_junction_is_never_read_or_reset(self):
        base = Path(self.temporary.name) / "junction-case"
        managed = base / "managed"
        outside = base / "outside"
        linked = managed / "optimizations"
        managed.mkdir(parents=True)
        outside.mkdir()
        sentinel = json.dumps(optimizations._default_state(), sort_keys=True)
        (outside / "state.json").write_text(sentinel, encoding="utf-8")
        created = subprocess.run(
            ["cmd.exe", "/c", "mklink", "/J", str(linked), str(outside)],
            capture_output=True,
            text=True,
            check=False,
        )
        if created.returncode != 0:
            self.skipTest(f"junction creation unavailable: {created.stderr}")
        try:
            with (
                mock.patch.object(optimizations, "MANAGED_ROOT", managed),
                mock.patch.object(optimizations, "OPTIMIZATION_ROOT", linked),
                mock.patch.object(optimizations, "STATE_PATH", linked / "state.json"),
            ):
                self.assertEqual(
                    optimizations.read_state()["_storageStatus"], "repair_required"
                )
                with (
                    mock.patch.object(
                        optimizations,
                        "reserve_install",
                        side_effect=lambda *_args: self.lease(),
                    ),
                    mock.patch.object(optimizations, "release_install"),
                    self.assertRaises(OSError),
                ):
                    optimizations.rollback_environment()
            self.assertEqual((outside / "state.json").read_text(encoding="utf-8"), sentinel)
        finally:
            os.rmdir(linked)

    def test_catalog_prioritizes_active_and_previous_and_redacts_forged_fields(self):
        optimizations.ENVIRONMENTS_DIR.mkdir(parents=True)
        active = "runtime-1-deadbeef"
        previous = "runtime-2-feedface"
        state = optimizations._default_state()
        state.update(
            {
                "activeEnvironmentId": active,
                "previousEnvironmentId": previous,
                "activeTrustClass": "legacy_optimization",
                "previousTrustClass": "legacy_optimization",
            }
        )
        optimizations._write_state(state)
        for index in range(40):
            (optimizations.ENVIRONMENTS_DIR / f"runtime-{index + 10}-aaaaaaaa").mkdir()
        (optimizations.ENVIRONMENTS_DIR / active).mkdir()
        (optimizations.ENVIRONMENTS_DIR / previous).mkdir()

        inspection = {
            "status": "recorded",
            "manifest": {
                "trustClass": "legacy_optimization",
                "createdAt": "C:/private/path",
                "specs": [{"kind": "optimization", "id": "torchao"}],
            },
            "validation": {
                "status": "C:/private/status",
                "validatedAt": "C:/private/time",
                "bindingDigest": "C:/private/digest",
            },
        }
        with (
            mock.patch.object(
                optimizations, "_environment_inspection", return_value=inspection
            ) as inspect_environment,
            mock.patch.object(
                optimizations,
                "overlay_file_seal_matches",
                side_effect=AssertionError("status GET must not hash an overlay"),
            ),
            mock.patch.object(
                optimizations,
                "verify_artifact_anchored_overlay",
                side_effect=AssertionError("status GET must not hash wheel archives"),
            ),
        ):
            catalog = optimizations.public_catalog(
                runtime_profile={"installed": "amd-rocm-linux"},
                hardware={"torch": {"version": "2.9.1+rocm7.2"}},
            )
            optional_catalog = optimizations.public_optional_runtime_catalog()
        self.assertTrue(inspect_environment.call_args_list)
        self.assertTrue(
            all(
                call.kwargs.get("verify_integrity") is False
                for call in inspect_environment.call_args_list
            )
        )
        by_id = {item["id"]: item for item in catalog["environments"]}
        self.assertIn(active, by_id)
        self.assertIn(previous, by_id)
        self.assertTrue(catalog["environmentScan"]["truncated"])
        self.assertEqual(by_id[active]["status"], "legacy_unqualified")
        self.assertIsNone(by_id[active]["createdAt"])
        self.assertIsNone(by_id[active]["validation"]["status"])
        self.assertIsNone(by_id[active]["validation"]["validatedAt"])
        self.assertIsNone(by_id[active]["validation"]["bindingDigest"])
        optional_ids = {
            item["id"] for item in optional_catalog["overlay"]["environments"]
        }
        self.assertIn(active, optional_ids)
        self.assertIn(previous, optional_ids)

    def test_failed_stage_never_changes_active_environment(self):
        self.set_active_environment(self.ACTIVE_ENVIRONMENT_ID)
        with (
            mock.patch.object(optimizations, "reserve_install") as reserve,
            mock.patch.object(optimizations, "_install_reviewed_overlay") as install,
            self.assertRaisesRegex(RuntimeError, "immutable artifact lock"),
        ):
            optimizations.install_capability(
                "torchao",
                runtime_profile={"installed": "amd-rocm-linux"},
                hardware={"torch": {"version": "2.9.1+rocm7.2"}},
            )
        reserve.assert_not_called()
        install.assert_not_called()
        self.assertEqual(
            optimizations.read_state()["activeEnvironmentId"], self.ACTIVE_ENVIRONMENT_ID
        )

    def test_legacy_source_package_rejects_before_staging(self):
        self.set_active_environment(self.ACTIVE_ENVIRONMENT_ID)
        with (
            mock.patch.object(optimizations, "reserve_install") as reserve,
            mock.patch.object(optimizations, "_install_reviewed_overlay") as install,
            self.assertRaisesRegex(RuntimeError, "immutable artifact lock"),
        ):
            optimizations.install_capability(
                "flash_attention_2",
                runtime_profile={"installed": "amd-rocm-linux"},
                hardware={
                    "torch": {"version": "2.9.1+rocm7.2"},
                    "amd_architectures": ["gfx1151"],
                },
            )
        reserve.assert_not_called()
        install.assert_not_called()
        self.assertEqual(
            optimizations.read_state()["activeEnvironmentId"], self.ACTIVE_ENVIRONMENT_ID
        )

    def test_auto_requires_opt_in_baseline_review_and_exact_runtime(self):
        self.set_active_environment(self.ACTIVE_ENVIRONMENT_ID)
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

    def test_probe_receipt_never_persists_raw_stdout_stderr_or_paths(self):
        result = SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "supported": True,
                    "compileAvailable": True,
                    "cudaAvailable": False,
                    "deviceCount": 0,
                    "privatePath": "C:/private/stdout",
                }
            ),
            stderr="C:/private/stderr token=secret",
        )
        with mock.patch.object(optimizations.subprocess, "run", return_value=result):
            receipt = optimizations.probe_capability(
                "regional_compile", runtime_fingerprint={"private": "value"}
            )
        self.assertEqual(receipt["status"], "probe_passed")
        persisted = optimizations.RECEIPTS_PATH.read_text(encoding="utf-8")
        self.assertNotIn("C:/private", persisted)
        self.assertNotIn("stderr", persisted)
        self.assertNotIn("stdout", persisted)
        stored_result = optimizations.read_receipts()["receipts"][0]["result"]
        self.assertEqual(stored_result["status"], "passed")
        self.assertTrue(stored_result["supported"])
        self.assertRegex(stored_result["diagnosticDigest"], r"^[0-9a-f]{64}$")

    def test_unqualified_package_probe_rejects_without_subprocess(self):
        with (
            mock.patch.object(optimizations.subprocess, "run") as run,
            self.assertRaisesRegex(RuntimeError, "immutable artifact locks"),
        ):
            optimizations.probe_capability("torchao", runtime_fingerprint={})
        run.assert_not_called()

    def test_auto_combines_independently_qualified_capabilities(self):
        self.set_active_environment(self.ACTIVE_ENVIRONMENT_ID)
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
