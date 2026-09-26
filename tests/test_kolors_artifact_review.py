import hashlib
import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.studio_execution_specs import STUDIO_EXECUTION_SPEC_DEFINITIONS
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "kolors-artifact-review.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class KolorsArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_exact_public_fp16_safetensors_partition_is_sealed(self):
        repository = self.review["repository"]
        weights = self.review["weightFiles"]
        self.assertRegex(repository["revision"], SHA1)
        self.assertFalse(repository["private"])
        self.assertFalse(repository["gated"])
        self.assertFalse(repository["pythonFilesPresent"])
        self.assertFalse(repository["trustRemoteCodeRequired"])
        self.assertEqual(repository["modelIndexClass"], "KolorsPipeline")
        self.assertEqual(repository["modelIndexComponentLibrary"], "kolors")
        self.assertEqual(len(weights), repository["weightFileCount"])
        self.assertEqual(sum(item["byteSize"] for item in weights), repository["weightBytes"])
        self.assertTrue(all(item["path"].endswith(".safetensors") for item in weights))
        self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in weights))
        canonical = json.dumps(
            sorted(weights, key=lambda item: item["path"]),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), repository["weightInventorySha256"])

    def test_package_owned_text_and_image_routes_are_recorded_without_admission(self):
        pinned = self.review["pinnedDiffusers"]
        self.assertEqual(pinned["pipelineClass"], "KolorsPipeline")
        self.assertEqual(pinned["imageToImagePipelineClass"], "KolorsImg2ImgPipeline")
        self.assertEqual(pinned["packageOwnedTextEncoderClass"], "ChatGLMModel")
        self.assertEqual(pinned["packageOwnedTokenizerClass"], "ChatGLMTokenizer")
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        self.assertTrue(pinned["interruptFlagAvailable"])
        self.assertFalse(pinned["safetyCheckerAvailable"])
        self.assertTrue(all(SHA256.fullmatch(value) for value in pinned["sourceSha256"].values()))

        contract = self.review["pipelineContract"]
        self.assertEqual(contract["modes"], ["text_to_image", "edit_image"])
        self.assertEqual((contract["defaultWidth"], contract["defaultHeight"]), (1024, 1024))
        self.assertEqual(contract["defaultNumInferenceSteps"], 50)
        self.assertEqual(contract["defaultGuidanceScale"], 5.0)
        self.assertEqual(contract["defaultMaxSequenceLength"], 256)
        self.assertEqual(contract["defaultImageToImageStrength"], 0.3)
        self.assertIsNone(contract["packageUpperBounds"]["numInferenceSteps"])
        self.assertIsNone(contract["packageUpperBounds"]["outputPixels"])

    def test_custom_model_license_overrides_the_misleading_card_tag_for_admission(self):
        license_review = self.review["licenseReview"]
        self.assertEqual(license_review["modelCardSpdxTag"], "apache-2.0")
        self.assertTrue(license_review["governingModelLicenseFilePresent"])
        self.assertRegex(license_review["governingModelLicenseSha256"], SHA256)
        self.assertFalse(license_review["modelLicenseIsApache2"])
        self.assertTrue(license_review["acceptanceTriggeredByUseOrAccess"])
        self.assertFalse(license_review["taskScopedProductAcceptanceObtained"])
        self.assertTrue(license_review["commercialRegistrationStatementPresent"])
        self.assertTrue(license_review["cloudVendorOrOver100MMonthlyUsersRequiresSeparateAuthorization"])
        self.assertTrue(license_review["distributionCarriesSourceLicenseAndUseRestrictionDuties"])
        self.assertFalse(license_review["legalAndProductApprovalComplete"])

    def test_fail_closed_review_adds_no_download_or_user_facing_surface(self):
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "review_complete_blocked_custom_model_license")
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["capabilityExposed"])
        self.assertEqual(admission["canonicalWorkflowsAdded"], 0)
        self.assertFalse(admission["clientSurfaceAdded"])
        self.assertFalse(admission["modelWeightsDownloaded"])
        self.assertEqual(admission["remoteWeightBytesFetched"], 0)
        self.assertIn("task_scoped_model_license_acceptance", admission["unresolvedGates"])
        self.assertIsNone(catalog_repository_pin(self.review["repository"]["repository"]))
        self.assertNotIn("KolorsPipeline", IMAGE_PIPELINE_ADAPTERS)
        self.assertFalse(
            any(
                definition["modelType"] == "KolorsPipeline"
                for definition in STUDIO_EXECUTION_SPEC_DEFINITIONS.values()
            )
        )

    def test_resource_envelope_remains_estimate_only(self):
        envelope = self.review["remoteResourceEnvelope"]
        self.assertEqual(envelope["status"], "estimate_only_not_admitted")
        self.assertEqual(envelope["selectedWeightBytes"], self.review["repository"]["weightBytes"])
        self.assertGreater(envelope["minimumSelectiveDiskBytes"], envelope["selectedWeightBytes"])
        self.assertGreater(envelope["minimumSystemRamBytes"], envelope["minimumSelectiveDiskBytes"])


if __name__ == "__main__":
    unittest.main()
