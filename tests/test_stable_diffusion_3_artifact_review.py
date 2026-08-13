import hashlib
import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.studio_execution_specs import STUDIO_EXECUTION_SPEC_DEFINITIONS
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data" / "stable-diffusion-3-artifact-review.json"
CONTROL_REVIEW_PATH = ROOT / "data" / "sd3-controlnet-artifact-review.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class StableDiffusion3ArtifactReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
        cls.control_review = json.loads(CONTROL_REVIEW_PATH.read_text(encoding="utf-8"))

    def test_exact_gated_repository_is_sealed_but_not_cataloged(self):
        repository = self.review["repository"]
        self.assertEqual(repository["repository"], "stabilityai/stable-diffusion-3-medium-diffusers")
        self.assertRegex(repository["revision"], SHA1)
        self.assertFalse(repository["private"])
        self.assertEqual(repository["gated"], "auto")
        self.assertFalse(repository["pythonFilesPresent"])
        self.assertFalse(repository["trustRemoteCodeRequired"])
        self.assertTrue(repository["licenseFileAccessibleWithoutAuthentication"])
        self.assertFalse(repository["modelIndexAccessibleWithoutAcceptedGate"])
        self.assertFalse(repository["componentConfigsAccessibleWithoutAcceptedGate"])
        self.assertEqual(repository["gatedArtifactHttpStatus"], 401)
        self.assertIsNone(catalog_repository_pin(repository["repository"]))

    def test_base_inventory_exactly_reuses_the_controlnet_review_without_fetching_weights(self):
        repository = self.review["repository"]
        weights = sorted(self.review["selectedWeightFiles"], key=lambda item: item["path"])
        canonical = json.dumps(weights, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(weights, sorted(self.control_review["selectedBaseWeightFiles"], key=lambda item: item["path"]))
        self.assertEqual(len(weights), repository["selectedWeightFileCount"])
        self.assertEqual(sum(item["byteSize"] for item in weights), repository["selectedWeightBytes"])
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), repository["selectedWeightInventorySha256"])
        self.assertTrue(all(item["path"].endswith(".safetensors") for item in weights))
        self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in weights))

        receipt = self.review["crossReviewReceipt"]
        self.assertTrue(receipt["sameRepositoryRevision"])
        self.assertTrue(receipt["sameSelectedBaseWeightInventory"])
        self.assertEqual(receipt["remoteSafetensorsHeaderBytesFetchedByPriorReview"], 0)
        self.assertEqual(self.review["admission"]["baseWeightBytesFetched"], 0)

    def test_official_package_routes_and_bounds_are_sealed(self):
        pinned = self.review["pinnedDiffusers"]
        self.assertRegex(pinned["revision"], SHA1)
        self.assertEqual(pinned["textToImagePipelineClass"], "StableDiffusion3Pipeline")
        self.assertEqual(pinned["imageToImagePipelineClass"], "StableDiffusion3Img2ImgPipeline")
        self.assertEqual(pinned["inpaintPipelineClass"], "StableDiffusion3InpaintPipeline")
        self.assertEqual(pinned["transformerClass"], "SD3Transformer2DModel")
        self.assertTrue(all(SHA256.fullmatch(value) for value in pinned["sourceSha256"].values()))
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        self.assertTrue(pinned["interruptFlagAvailable"])
        self.assertTrue(pinned["interruptFlagReadByDenoisingLoop"])
        self.assertFalse(pinned["safetyCheckerAvailable"])

        contract = self.review["pipelineContract"]
        self.assertEqual((contract["defaultWidth"], contract["defaultHeight"]), (1024, 1024))
        self.assertEqual(contract["dimensionMultiple"], 16)
        self.assertEqual(contract["defaultGuidanceScale"], 7.0)
        self.assertEqual(contract["defaultMaxSequenceLength"], 256)
        self.assertEqual(contract["packageMaxSequenceLength"], 512)
        self.assertTrue(contract["negativePromptAvailable"])
        self.assertEqual(set(contract["modes"]), {"text_to_image", "edit_image", "inpaint"})
        self.assertEqual(contract["modes"]["text_to_image"]["defaultNumInferenceSteps"], 28)
        for mode in ("edit_image", "inpaint"):
            self.assertEqual(contract["modes"][mode]["defaultNumInferenceSteps"], 50)
            self.assertEqual(contract["modes"][mode]["defaultStrength"], 0.6)
        self.assertEqual(contract["packageUpperBounds"]["maxSequenceLength"], 512)
        self.assertIsNone(contract["packageUpperBounds"]["numInferenceSteps"])
        self.assertIsNone(contract["packageUpperBounds"]["outputPixels"])

    def test_noncommercial_terms_remain_unaccepted_and_block_admission(self):
        license_review = self.review["licenseReview"]
        self.assertEqual(license_review["id"], "stabilityai-nc-research-community")
        self.assertRegex(license_review["licenseSha256"], SHA256)
        self.assertTrue(license_review["gateAcceptanceRequired"])
        self.assertFalse(license_review["taskScopedProductAcceptanceObtained"])
        self.assertTrue(license_review["acceptanceTriggeredByUse"])
        self.assertTrue(license_review["nonCommercialOnly"])
        self.assertTrue(license_review["productionUseProhibitedWithoutSeparateLicense"])
        self.assertTrue(license_review["hostedServiceOrApiUseProhibitedWithoutSeparateLicense"])
        self.assertFalse(license_review["legalAndProductApprovalComplete"])

    def test_app_preflight_and_runtime_surfaces_fail_closed_without_deletes(self):
        preflight = self.review["appDownloadPreflight"]
        self.assertEqual(preflight["transportPolicy"], "app_only")
        self.assertTrue(preflight["planRequestedThroughApp"])
        self.assertFalse(preflight["submissionAttempted"])
        self.assertFalse(preflight["termsAcceptedByMoDiff"])
        self.assertFalse(preflight["directWeightDownloadPerformed"])
        self.assertFalse(preflight["olderModelsDeleted"])
        self.assertTrue(preflight["standaloneFits"])
        self.assertFalse(preflight["fitsWithQueue"])
        self.assertEqual(
            preflight["preSnapshotHeadroomAfterReserveBytes"],
            preflight["preflightFreeBytes"]
            - preflight["preflightExistingQueuedReservationBytes"]
            - preflight["reserveBytes"],
        )
        self.assertEqual(
            preflight["additionalBytesRequired"],
            preflight["snapshotRemainingBytes"] - preflight["preSnapshotHeadroomAfterReserveBytes"],
        )

        admission = self.review["admission"]
        self.assertEqual(admission["status"], "review_complete_blocked_noncommercial_license_and_capacity")
        for field in (
            "runtimeCatalogExposed",
            "downloadCatalogExposed",
            "capabilityExposed",
            "clientSurfaceAdded",
            "autoEligible",
            "galleryEligible",
            "modelWeightsDownloaded",
        ):
            self.assertFalse(admission[field])
        self.assertEqual(admission["canonicalWorkflowsAdded"], 0)

        standard_classes = {
            "StableDiffusion3Pipeline",
            "StableDiffusion3Img2ImgPipeline",
            "StableDiffusion3InpaintPipeline",
        }
        self.assertTrue(standard_classes.isdisjoint(IMAGE_PIPELINE_ADAPTERS))
        self.assertFalse(
            any(
                definition["modelType"] in standard_classes
                for definition in STUDIO_EXECUTION_SPEC_DEFINITIONS.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
