import hashlib
import json
import re
import unittest
from pathlib import Path

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.studio_execution_specs import STUDIO_EXECUTION_SPEC_DEFINITIONS
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data" / "hunyuan-image-artifact-review.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HunyuanImageArtifactReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_safe_diffusers_candidate_has_an_exact_uncataloged_weight_inventory(self):
        candidate = self.review["repositories"]["diffusersCandidate"]
        weights = self.review["weightFiles"]
        self.assertRegex(candidate["revision"], SHA1)
        self.assertFalse(candidate["private"])
        self.assertFalse(candidate["gated"])
        self.assertFalse(candidate["pythonFilesPresent"])
        self.assertFalse(candidate["trustRemoteCodeRequired"])
        self.assertEqual(candidate["modelIndexClass"], "HunyuanImagePipeline")
        self.assertEqual(len(weights), candidate["weightFileCount"])
        self.assertEqual(sum(item["byteSize"] for item in weights), candidate["weightBytes"])
        self.assertTrue(all(item["path"].endswith(".safetensors") for item in weights))
        self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in weights))
        canonical = json.dumps(
            sorted(weights, key=lambda item: item["path"]),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), candidate["weightInventorySha256"])
        self.assertIsNone(catalog_repository_pin(candidate["repository"]))

    def test_immutable_upstream_license_receipt_preserves_territory_restrictions(self):
        upstream = self.review["repositories"]["officialUpstream"]
        self.assertRegex(upstream["revision"], SHA1)
        self.assertTrue(upstream["licenseFilePresent"])
        self.assertTrue(upstream["noticeFilePresent"])
        self.assertFalse(upstream["usedForRuntime"])
        self.assertRegex(upstream["licenseSha256"], SHA256)
        self.assertRegex(upstream["noticeSha256"], SHA256)
        self.assertIsNone(catalog_repository_pin(upstream["repository"]))

        license_contract = self.review["license"]
        expected_territories = {"European Union", "South Korea", "United Kingdom"}
        self.assertEqual(set(license_contract["preambleExcludedTerritories"]), expected_territories)
        self.assertEqual(set(license_contract["definedTerritoryExcludes"]), expected_territories)
        self.assertTrue(license_contract["territoryTextRequiresLegalReview"])
        self.assertTrue(license_contract["outputUseOutsideTerritoryProhibited"])
        self.assertTrue(license_contract["acceptanceTriggeredByUse"])
        self.assertTrue(license_contract["hostedServiceAddressed"])
        self.assertEqual(license_contract["commercialLicenseThresholdMonthlyActiveUsers"], 100000000)
        self.assertTrue(license_contract["machineGeneratedContentDisclosureRequiredForPublicUse"])

    def test_package_contract_is_pinned_without_overstating_runtime_qualification(self):
        pinned = self.review["pinnedRuntime"]
        self.assertEqual(pinned["pipelineClass"], "HunyuanImagePipeline")
        self.assertEqual(pinned["refinerPipelineClass"], "HunyuanImageRefinerPipeline")
        self.assertEqual(pinned["transformerClass"], "HunyuanImageTransformer2DModel")
        self.assertEqual(pinned["autoencoderClass"], "AutoencoderKLHunyuanImage")
        self.assertEqual(pinned["guiderClass"], "AdaptiveProjectedMixGuidance")
        self.assertEqual(pinned["modelCpuOffloadSequence"], "text_encoder->text_encoder_2->transformer->vae")
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        self.assertTrue(pinned["interruptFlagAvailable"])
        self.assertFalse(pinned["safetyCheckerAvailable"])
        for key, value in pinned.items():
            if key.endswith("Sha256"):
                self.assertRegex(value, SHA256)

        contract = self.review["pipelineContract"]
        self.assertEqual(contract["modes"], ["text_to_image"])
        self.assertEqual((contract["reviewedWidth"], contract["reviewedHeight"]), (2048, 2048))
        self.assertEqual(contract["numInferenceSteps"], 50)
        self.assertEqual(contract["guidanceScale"], 3.5)
        self.assertTrue(contract["negativePromptAvailable"])
        self.assertTrue(contract["modelCpuOffload"])
        self.assertFalse(contract["refinerAdmitted"])

    def test_territory_gate_keeps_every_executable_and_user_facing_surface_closed(self):
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "contract_only_territory_restricted")
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["capabilityExposed"])
        self.assertEqual(admission["canonicalWorkflowsAdded"], 0)
        self.assertFalse(admission["clientSurfaceAdded"])
        self.assertFalse(admission["autoEligible"])
        self.assertFalse(admission["galleryEligible"])
        self.assertFalse(admission["liveQualified"])
        self.assertFalse(admission["fullWeightFilesDownloaded"])
        self.assertEqual(admission["remoteWeightBytesFetched"], 0)
        self.assertIn("legal_territory_and_distribution_review", admission["unresolvedGates"])
        self.assertIn("product_territory_enforcement", admission["unresolvedGates"])
        self.assertNotIn("HunyuanImagePipeline", IMAGE_PIPELINE_ADAPTERS)
        self.assertFalse(
            any(
                definition["modelType"] == "HunyuanImagePipeline"
                for definition in STUDIO_EXECUTION_SPEC_DEFINITIONS.values()
            )
        )

    def test_resource_claim_is_estimate_only_and_does_not_reuse_the_fp8_result(self):
        envelope = self.review["remoteResourceEnvelope"]
        self.assertEqual(envelope["status"], "modiff_estimate_only_qualification_pending")
        self.assertEqual(envelope["selectedWeightBytes"], self.review["repositories"]["diffusersCandidate"]["weightBytes"])
        self.assertFalse(envelope["publisherFp8FigureAppliesToSelectedBfloat16Candidate"])


if __name__ == "__main__":
    unittest.main()
