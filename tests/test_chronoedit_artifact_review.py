import hashlib
import json
import re
import unittest
from pathlib import Path

from modiff.model_artifact_catalog import catalog_repository_pin


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "chronoedit-artifact-review.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ChronoEditArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_public_snapshot_is_reviewed_but_every_surface_remains_closed(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        repository = self.review["repository"]
        self.assertEqual(repository["repository"], "nvidia/ChronoEdit-14B-Diffusers")
        self.assertRegex(repository["revision"], SHA1)
        self.assertFalse(repository["gated"])
        self.assertFalse(repository["private"])
        self.assertFalse(repository["pythonFilesPresent"])
        self.assertFalse(repository["trustRemoteCodeRequired"])
        self.assertIsNone(catalog_repository_pin(repository["repository"]))

        admission = self.review["admission"]
        self.assertEqual(admission["status"], "review_complete_not_admitted_guardrail_and_license")
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["liveQualified"])
        self.assertFalse(admission["autoEligible"])
        self.assertFalse(admission["galleryEligible"])
        self.assertEqual(admission["executableModes"], [])

    def test_required_twenty_one_file_safe_core_inventory_is_exact(self):
        files = sorted(self.review["weightFiles"], key=lambda item: item["path"])
        repository = self.review["repository"]
        self.assertEqual(repository["weightFileCount"], 21)
        self.assertEqual(repository["weightBytes"], 90075130404)
        self.assertEqual(len(files), 21)
        self.assertEqual(sum(item["byteSize"] for item in files), 90075130404)
        canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(repository["weightInventorySha256"], hashlib.sha256(canonical).hexdigest())
        for item in files:
            self.assertTrue(item["path"].endswith(".safetensors"))
            self.assertRegex(item["sha256"], SHA256)

    def test_guardrail_dependency_and_unsafe_files_are_not_silently_omitted(self):
        guardrail = self.review["guardrailReview"]
        self.assertFalse(guardrail["diffusersPipelineSafetyCheckerPresent"])
        self.assertEqual(guardrail["bundledDirectoryFileCount"], 102)
        self.assertEqual(guardrail["bundledDirectoryBytes"], 7171449905)
        self.assertEqual(
            {item["path"] for item in guardrail["unsafeFiles"]},
            {
                "Cosmos-Guardrail1/face_blur_filter/Resnet50_Final.pth",
                "Cosmos-Guardrail1/video_content_safety_filter/safety_filter.pt",
            },
        )
        for item in guardrail["unsafeFiles"]:
            self.assertRegex(item["sha256"], SHA256)
        self.assertIn("safe_serialized_complete_guardrail_integration", self.review["admission"]["unresolvedGates"])

    def test_governing_license_is_external_and_not_overstated_as_apache_only(self):
        license_review = self.review["license"]
        self.assertEqual(
            license_review["id"],
            "nvidia-open-model-license-with-apache-2.0-additional-information",
        )
        self.assertTrue(license_review["commercialUseClaimedByModelCard"])
        self.assertFalse(license_review["licenseFilePresent"])
        self.assertTrue(license_review["mutableExternalTerms"])
        self.assertIn("NVIDIA", license_review["evidence"])
        self.assertIn("automatically terminate", self.review["guardrailReview"]["modelCardTerminationCondition"])

    def test_package_source_recipe_and_model_index_mismatch_are_explicit(self):
        pinned = self.review["pinnedDiffusers"]
        self.assertEqual(pinned["pipelineClass"], "ChronoEditPipeline")
        self.assertEqual(pinned["transformerClass"], "ChronoEditTransformer3DModel")
        self.assertEqual(pinned["outputClass"], "ChronoEditPipelineOutput")
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        for field in ("pipelineSha256", "transformerSha256", "outputSha256"):
            self.assertRegex(pinned[field], SHA256)

        compatibility = self.review["modelIndexCompatibility"]
        self.assertEqual(compatibility["artifactPipelineClass"], "WanImageToVideoPipeline")
        self.assertEqual(compatibility["artifactTransformerClass"], "WanTransformer3DModel")
        self.assertTrue(compatibility["packageRecipeRequiresExplicitComponentOverrides"])

    def test_edit_reasoning_and_optional_lora_contracts_are_sealed(self):
        contract = self.review["pipelineContract"]
        self.assertEqual(contract["input"], "one RGB image plus one edit instruction")
        self.assertEqual(contract["output"], "a frame sequence whose final frame is the edited image")
        self.assertEqual(contract["baseRecipe"]["numFrames"], 5)
        self.assertEqual(contract["baseRecipe"]["numInferenceSteps"], 50)
        self.assertEqual(contract["temporalReasoningRecipe"]["numFrames"], 29)
        self.assertEqual(contract["distilledRecipe"]["numInferenceSteps"], 8)
        self.assertIn("non-temporal mode silently forces five frames", contract["packageValidationGaps"])

        optional = self.review["optionalArtifacts"]
        self.assertEqual(len(optional), 3)
        for artifact in optional:
            self.assertRegex(artifact["revision"], SHA1)
            self.assertTrue(artifact["safetensorsPath"].endswith(".safetensors"))
            self.assertRegex(artifact["sha256"], SHA256)

    def test_resource_values_remain_qualification_pending(self):
        envelope = self.review["remoteResourceEnvelope"]
        self.assertEqual(envelope["status"], "upstream_measured_modiff_qualification_pending")
        self.assertEqual(envelope["upstreamReportedGpuMemoryGB"]["cpuOffload"], 34)
        self.assertEqual(envelope["upstreamReportedGpuMemoryGB"]["cpuOffloadWithTemporalReasoning"], 38)
        self.assertIn("physical_macos_execution", self.review["admission"]["unresolvedGates"])


if __name__ == "__main__":
    unittest.main()
