import hashlib
import json
import re
import unittest
from pathlib import Path

from modiff.model_artifact_catalog import catalog_repository_pin


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "visualcloze-artifact-review.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class VisualClozeArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_full_diffusers_snapshots_are_reviewed_but_not_admitted(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "review_complete_not_admitted")
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["liveQualified"])
        self.assertFalse(admission["autoEligible"])
        self.assertFalse(admission["galleryEligible"])
        self.assertEqual(admission["executableModes"], [])

        repositories = self.review["repositories"]
        self.assertEqual(
            {repository["repository"] for repository in repositories},
            {
                "VisualCloze/VisualClozePipeline-384",
                "VisualCloze/VisualClozePipeline-512",
            },
        )
        for repository in repositories:
            with self.subTest(repository=repository["repository"]):
                self.assertRegex(repository["revision"], SHA1)
                self.assertFalse(repository["gated"])
                self.assertFalse(repository["private"])
                self.assertFalse(repository["pythonFilesPresent"])
                self.assertFalse(repository["trustRemoteCodeRequired"])
                self.assertFalse(repository["licenseFilePresent"])
                self.assertIsNone(catalog_repository_pin(repository["repository"]))

    def test_each_seven_file_bfloat16_inventory_is_exact(self):
        for repository in self.review["repositories"]:
            with self.subTest(repository=repository["repository"]):
                files = sorted(repository["weightFiles"], key=lambda item: item["path"])
                self.assertEqual(repository["weightFileCount"], 7)
                self.assertEqual(repository["weightBytes"], 33743379958)
                self.assertEqual(sum(item["byteSize"] for item in files), 33743379958)
                canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
                self.assertEqual(
                    repository["weightInventorySha256"],
                    hashlib.sha256(canonical).hexdigest(),
                )
                self.assertEqual(
                    repository["safetensorsDtypeParameterCounts"],
                    {"BF16": 11902391360},
                )
                for item in files:
                    self.assertTrue(item["path"].endswith(".safetensors"))
                    self.assertRegex(item["sha256"], SHA256)

    def test_nested_matrix_contract_and_package_validation_gaps_are_explicit(self):
        contract = self.review["pipelineContract"]
        self.assertEqual(
            contract["inputShape"],
            "batch -> rows -> equal-width image-or-target-null cells",
        )
        self.assertEqual(contract["outputShape"], "batch -> one-or-more generated target images")
        self.assertEqual(contract["reviewedResolutions"], [384, 512])
        self.assertEqual(contract["maxSequenceLength"], 512)
        self.assertIn("no cumulative input-pixel ceiling", contract["packageValidationGaps"])
        self.assertIn("no maximum denoising-step bound", contract["packageValidationGaps"])
        self.assertIn("generic_visual_context_matrix_contract", self.review["admission"]["unresolvedGates"])

        pinned = self.review["pinnedDiffusers"]
        self.assertEqual(pinned["combinedPipelineClass"], "VisualClozePipeline")
        self.assertEqual(pinned["generationPipelineClass"], "VisualClozeGenerationPipeline")
        self.assertEqual(pinned["outputClass"], "FluxPipelineOutput")
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        for field in ("combinedPipelineSha256", "generationPipelineSha256", "processorSha256"):
            self.assertRegex(pinned[field], SHA256)

    def test_license_and_resource_evidence_are_not_overstated(self):
        license_review = self.review["license"]
        self.assertEqual(license_review["id"], "apache-2.0")
        self.assertFalse(license_review["licenseFilePresent"])
        envelope = self.review["remoteResourceEnvelope"]
        self.assertEqual(envelope["status"], "estimate_only_remote_qualification_pending")
        self.assertGreater(envelope["minimumSelectiveDiskBytes"], 33743379958)
        self.assertIn("physical_macos_execution", self.review["admission"]["unresolvedGates"])

    def test_legacy_pytorch_lora_artifacts_are_explicitly_excluded(self):
        legacy = self.review["excludedLegacyArtifacts"]
        self.assertEqual(legacy["repository"], "VisualCloze/VisualCloze")
        self.assertRegex(legacy["revision"], SHA1)
        self.assertEqual(len(legacy["files"]), 2)
        for item in legacy["files"]:
            self.assertTrue(item["path"].endswith(".pth"))
            self.assertEqual(item["byteSize"], 2482363148)
            self.assertRegex(item["sha256"], SHA256)


if __name__ == "__main__":
    unittest.main()
