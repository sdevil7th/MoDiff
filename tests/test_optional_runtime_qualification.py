from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qualify_optional_runtime.py"
SPEC = importlib.util.spec_from_file_location("modiff_optional_runtime_qualification", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


class OptionalRuntimeQualificationTests(unittest.TestCase):
    def test_preflight_preserves_profile_and_reports_exact_artifact_plan(self):
        import modiff.optional_runtimes as optional_runtimes

        before = optional_runtimes.OPTIONAL_RUNTIME_PROFILES[qualification.PROFILE_ID]
        result = qualification.qualification_preflight()
        after = optional_runtimes.OPTIONAL_RUNTIME_PROFILES[qualification.PROFILE_ID]

        self.assertIs(before, after)
        self.assertEqual(result["candidateSpecDigest"], before.spec_digest)
        self.assertEqual(result["qualificationSpecDigest"], before.spec_digest)
        self.assertFalse(result["sourceFlagsDormant"])
        self.assertTrue(result["sourceTargetQualified"])
        self.assertEqual(result["artifactCount"], len(before.packages))
        self.assertGreater(result["artifactBytes"], 0)
        self.assertRegex(result["artifactPlanDigest"], r"^sha256:[0-9a-f]{64}$")
        self.assertIsInstance(result["managedUvReceiptPresent"], bool)
        if result["status"] == "ready":
            self.assertTrue(result["managedUvReceiptPresent"])
        self.assertIn(result["status"], {"ready", "not_ready"})

    def test_full_qualification_requires_consent_before_preflight(self):
        with mock.patch.object(qualification, "qualification_preflight") as preflight:
            with self.assertRaisesRegex(RuntimeError, "explicit --consent"):
                qualification.run_qualification(consent=False)
        preflight.assert_not_called()

    def test_target_contract_refuses_incoherent_production_flags(self):
        from dataclasses import replace
        import modiff.optional_runtimes as optional_runtimes

        candidate = optional_runtimes.OPTIONAL_RUNTIME_PROFILES[qualification.PROFILE_ID]
        current = candidate.contract_for_target()
        with self.assertRaisesRegex(ValueError, "Invalid optional-runtime target contract"):
            replace(current, activation_available=False)

    def test_pending_macos_target_projects_only_that_target(self):
        import modiff.optional_runtimes as optional_runtimes

        candidate = optional_runtimes.OPTIONAL_RUNTIME_PROFILES[qualification.PROFILE_ID]
        with (
            mock.patch.object(qualification, "_platform_name", return_value="macos"),
            mock.patch.object(qualification, "_machine_name", return_value="arm64"),
        ):
            source, projected = qualification._future_profile()
        self.assertIs(source, candidate)
        self.assertFalse(
            source.contract_for_target(platform_name="macos", machine="arm64").cutover_ready
        )
        self.assertTrue(
            projected.contract_for_target(platform_name="macos", machine="arm64").cutover_ready
        )
        self.assertEqual(projected.contract_state, source.contract_state)
        self.assertNotEqual(projected.spec_digest, source.spec_digest)
        self.assertEqual(
            projected.contract_for_target(platform_name="linux", machine="x86_64"),
            source.contract_for_target(platform_name="linux", machine="x86_64"),
        )

    def test_exact_main_profile_has_a_linux_scoped_requalification_path(self):
        import modiff.optional_runtimes as optional_runtimes

        profile_id = optional_runtimes.TRANSFORMERS_MAIN_PEFT_RUNTIME_PROFILE_ID
        candidate = optional_runtimes.OPTIONAL_RUNTIME_PROFILES[profile_id]
        with mock.patch.object(qualification, "_platform_name", return_value="linux"), mock.patch.object(
            qualification, "_machine_name", return_value="x86_64"
        ):
            result = qualification.qualification_preflight(profile_id)

        self.assertEqual(result["profileId"], profile_id)
        self.assertEqual(result["candidateSpecDigest"], candidate.spec_digest)
        self.assertEqual(
            result["qualificationSpecDigest"],
            result["candidateSpecDigest"],
        )
        self.assertFalse(result["sourceFlagsDormant"])
        self.assertTrue(result["sourceTargetQualified"])
        self.assertEqual(result["sourceBuildCount"], 1)
        self.assertEqual(result["artifactCount"], len(candidate.packages))
        self.assertEqual(
            candidate.packages[0].version,
            "5.16.0.dev0",
        )

    def test_preflight_rejects_dirty_or_unavailable_source_revision(self):
        for source in (
            {"commit": "a" * 40, "dirty": True, "available": True},
            {"commit": "unavailable", "dirty": True, "available": False},
        ):
            with self.subTest(source=source), mock.patch.object(
                qualification,
                "_source_revision",
                return_value=source,
            ):
                result = qualification.qualification_preflight()
            self.assertEqual(result["status"], "not_ready")
            self.assertFalse(result["sourceRevisionReady"])
            self.assertEqual(result["source"], source)

    def test_source_revision_rejects_noncommit_output(self):
        completed = mock.Mock(stdout="main\n")
        with mock.patch.object(
            qualification.subprocess,
            "run",
            return_value=completed,
        ):
            source = qualification._source_revision()
        self.assertEqual(
            source,
            {"commit": "unavailable", "dirty": True, "available": False},
        )

    def test_not_ready_preflight_returns_a_failing_exit_status(self):
        with mock.patch.object(
            qualification,
            "qualification_preflight",
            return_value={"status": "not_ready"},
        ), redirect_stdout(io.StringIO()):
            status = qualification.main(["--preflight-only"])
        self.assertEqual(status, 1)

    def test_verified_uv_copy_rejects_a_forged_executable(self):
        from modiff.tool_locks import UV_TOOL_LOCKS

        lock = UV_TOOL_LOCKS[(qualification._platform_name(), qualification._machine_name())]
        executable_name = "uv.exe" if qualification._platform_name() == "windows" else "uv"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source" / "tools" / "uv"
            source.mkdir(parents=True)
            (source / executable_name).write_bytes(b"forged")
            (source / "receipt.json").write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "archiveSha256": lock["archiveSha256"],
                        "executableSha256": lock["executableSha256"],
                        "executable": executable_name,
                    }
                ),
                encoding="utf-8",
            )

            target = root / "target"
            with self.assertRaisesRegex(RuntimeError, "reviewed identity"):
                qualification.copy_verified_uv(root / "source", target)
            self.assertFalse(target.exists())

    def test_evidence_is_bounded_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence.json"
            qualification._write_evidence(evidence, {"status": "passed"})
            self.assertEqual(json.loads(evidence.read_text(encoding="utf-8")), {"status": "passed"})
            with self.assertRaises(FileExistsError):
                qualification._write_evidence(evidence, {"status": "changed"})

            oversized = Path(temporary) / "oversized.json"
            with self.assertRaisesRegex(RuntimeError, "safe bound"):
                qualification._write_evidence(
                    oversized,
                    {"value": "x" * qualification.MAX_EVIDENCE_BYTES},
                )
            self.assertFalse(oversized.exists())


if __name__ == "__main__":
    unittest.main()
