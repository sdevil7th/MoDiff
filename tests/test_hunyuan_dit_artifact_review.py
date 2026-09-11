import hashlib
import json
import re
import unittest
from pathlib import Path

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.studio_execution_specs import STUDIO_EXECUTION_SPEC_DEFINITIONS
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data" / "hunyuan-dit-artifact-review.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HunyuanDiTArtifactReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_exact_safe_base_inventory_is_cataloged_for_the_standalone_pipeline(self):
        repository = self.review["repository"]
        weights = self.review["weightFiles"]
        self.assertRegex(repository["revision"], SHA1)
        self.assertFalse(repository["private"])
        self.assertFalse(repository["gated"])
        self.assertFalse(repository["pythonFilesPresent"])
        self.assertFalse(repository["trustRemoteCodeRequired"])
        self.assertEqual(repository["modelIndexClass"], "HunyuanDiTPipeline")
        self.assertEqual(len(weights), repository["weightFileCount"])
        self.assertEqual(sum(item["byteSize"] for item in weights), repository["weightBytes"])
        self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in weights))
        canonical = json.dumps(
            sorted(weights, key=lambda item: item["path"]),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), repository["weightInventorySha256"])
        pin = catalog_repository_pin(repository["repository"], model_type="HunyuanDiTPipeline")
        self.assertIsNotNone(pin)
        self.assertEqual(pin["revision"], repository["revision"])
        self.assertEqual(pin["license"], "tencent-hunyuan-community")

    def test_bounded_standalone_adapter_and_execution_spec_share_one_exact_contract(self):
        adapter = IMAGE_PIPELINE_ADAPTERS["HunyuanDiTPipeline"]
        contract = self.review["pipelineContract"]
        self.assertEqual(adapter.modes, frozenset({"text_to_image"}))
        self.assertTrue(adapter.safe_serialization_required)
        self.assertEqual(adapter.max_inference_steps, contract["numInferenceSteps"])
        self.assertEqual((adapter.min_output_side, adapter.max_output_side), (1024, 1024))
        self.assertEqual(adapter.max_sequence_length, contract["t5TokenLimit"])

        definition = STUDIO_EXECUTION_SPEC_DEFINITIONS["hunyuan-dit-v1-2-distilled:text-to-image:v1"]
        self.assertEqual(definition["modelType"], "HunyuanDiTPipeline")
        self.assertEqual(definition["mode"], "text_to_image")
        capability = definition["capability"]
        self.assertEqual(capability["defaultRepo"], self.review["repository"]["repository"])
        self.assertEqual(capability["defaultSize"], {"width": 1024, "height": 1024, "aspectRatio": "1:1"})
        self.assertEqual(capability["recommendedSteps"], 25)
        self.assertEqual(capability["recommendedGuidance"], 5.0)
        self.assertEqual(capability["recommendedMaxSequenceLength"], 256)
        self.assertFalse(capability["autoEligible"])
        self.assertFalse(capability["galleryEligible"])

    def test_package_and_license_receipts_reuse_the_controlnet_review_without_conflation(self):
        pinned = self.review["pinnedRuntime"]
        self.assertEqual(pinned["pipelineClass"], "HunyuanDiTPipeline")
        self.assertEqual(pinned["transformerClass"], "HunyuanDiT2DModel")
        self.assertEqual(pinned["modelCpuOffloadSequence"], "text_encoder->text_encoder_2->transformer->vae")
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        self.assertTrue(pinned["interruptFlagAvailable"])
        for key, value in pinned.items():
            if key.endswith("Sha256"):
                self.assertRegex(value, SHA256)

        license_review = self.review["license"]
        self.assertRegex(license_review["sealedLicenseRevision"], SHA1)
        self.assertRegex(license_review["sealedLicenseSha256"], SHA256)
        self.assertTrue(license_review["acceptanceTriggeredByUse"])
        self.assertTrue(license_review["hostedServiceAddressed"])
        self.assertTrue(license_review["immutableTermsAcknowledgementRequired"])

    def test_source_admission_does_not_claim_live_or_local_heavy_execution(self):
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "source_complete_execution_pending")
        self.assertEqual(admission["executableModes"], ["HunyuanDiTPipeline:text_to_image"])
        self.assertTrue(admission["runtimeCatalogExposed"])
        self.assertTrue(admission["downloadCatalogExposed"])
        self.assertTrue(admission["remoteOnly"])
        self.assertFalse(admission["liveQualified"])
        self.assertFalse(admission["autoEligible"])
        self.assertFalse(admission["galleryEligible"])
        self.assertFalse(admission["fullWeightFilesDownloaded"])
        self.assertEqual(admission["remoteWeightBytesFetched"], 0)
        self.assertIn("physical_macos_execution", admission["unresolvedGates"])
        self.assertEqual(
            self.review["remoteResourceEnvelope"]["status"],
            "estimate_only_modiff_qualification_pending",
        )


if __name__ == "__main__":
    unittest.main()
