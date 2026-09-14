import hashlib
import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.studio_execution_specs import STUDIO_EXECUTION_SPEC_DEFINITIONS
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "latent-diffusion-artifact-review.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LatentDiffusionArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_exact_legacy_partition_is_recorded_without_fetching_weight_bytes(self):
        repository = self.review["repository"]
        weights = self.review["selectedWeightFiles"]
        self.assertRegex(repository["revision"], SHA1)
        self.assertFalse(repository["private"])
        self.assertFalse(repository["gated"])
        self.assertFalse(repository["pythonFilesPresent"])
        self.assertFalse(repository["trustRemoteCodeRequired"])
        self.assertEqual(repository["modelIndexClass"], "LDMTextToImagePipeline")
        self.assertEqual(len(weights), repository["selectedWeightFileCount"])
        self.assertEqual(sum(item["byteSize"] for item in weights), repository["selectedWeightBytes"])
        self.assertEqual(repository["safetensorsWeightFileCount"], 0)
        self.assertTrue(all(item["serialization"] == "pytorch_pickle_bin" for item in weights))
        self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in weights))
        canonical = json.dumps(
            sorted(weights, key=lambda item: item["path"]),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), repository["selectedWeightInventorySha256"])

    def test_package_contract_lacks_safe_cancellation_and_resource_bounds(self):
        pinned = self.review["pinnedDiffusers"]
        self.assertEqual(pinned["pipelineClass"], "LDMTextToImagePipeline")
        self.assertEqual(pinned["packageOwnedBertClass"], "LDMBertModel")
        self.assertRegex(pinned["pipelineSha256"], SHA256)
        self.assertFalse(pinned["callbackOnStepEndAvailable"])
        self.assertFalse(pinned["interruptFlagAvailable"])
        self.assertFalse(pinned["safetyCheckerAvailable"])

        contract = self.review["pipelineContract"]
        self.assertEqual(contract["modes"], ["text_to_image"])
        self.assertEqual((contract["defaultWidth"], contract["defaultHeight"]), (256, 256))
        self.assertEqual(contract["defaultNumInferenceSteps"], 50)
        self.assertEqual(contract["effectivePromptTokenLimit"], 77)
        self.assertFalse(contract["negativePromptAvailable"])
        self.assertIsNone(contract["packageUpperBounds"]["numInferenceSteps"])
        self.assertIsNone(contract["packageUpperBounds"]["outputPixels"])

    def test_pickle_scanner_observation_does_not_weaken_safe_serialization_policy(self):
        serialization = self.review["serializationReview"]
        self.assertFalse(serialization["safeSerializationAvailable"])
        self.assertTrue(serialization["allRequiredWeightsUsePickleContainers"])
        self.assertTrue(serialization["hubScannerObservedTypicalTorchPickleRebuildImports"])
        self.assertFalse(serialization["staticScannerResultMakesPickleSafeForMoDiff"])

    def test_fail_closed_review_adds_no_catalog_or_user_facing_surface(self):
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "review_complete_blocked_unsafe_serialization_and_cancellation")
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["capabilityExposed"])
        self.assertEqual(admission["canonicalWorkflowsAdded"], 0)
        self.assertFalse(admission["clientSurfaceAdded"])
        self.assertFalse(admission["modelWeightsDownloaded"])
        self.assertEqual(admission["remoteWeightBytesFetched"], 0)
        self.assertIn("safe_official_safetensors_artifacts", admission["unresolvedGates"])
        self.assertIsNone(catalog_repository_pin(self.review["repository"]["repository"]))
        self.assertNotIn("LDMTextToImagePipeline", IMAGE_PIPELINE_ADAPTERS)
        self.assertFalse(
            any(
                definition["modelType"] == "LDMTextToImagePipeline"
                for definition in STUDIO_EXECUTION_SPEC_DEFINITIONS.values()
            )
        )

    def test_license_and_resources_remain_explicitly_unqualified(self):
        license_review = self.review["licenseReview"]
        self.assertEqual(license_review["modelCardLicenseTag"], "apache-2.0")
        self.assertFalse(license_review["snapshotLicenseFilePresent"])
        self.assertFalse(license_review["immutableLicenseReceiptComplete"])
        envelope = self.review["remoteResourceEnvelope"]
        self.assertEqual(envelope["status"], "estimate_only_not_admitted")
        self.assertEqual(envelope["selectedWeightBytes"], self.review["repository"]["selectedWeightBytes"])
        self.assertGreater(envelope["minimumSelectiveDiskBytes"], envelope["selectedWeightBytes"])


if __name__ == "__main__":
    unittest.main()
