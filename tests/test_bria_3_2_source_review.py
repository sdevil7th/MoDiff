import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "bria-3.2-source-review.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Bria32SourceReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_gated_noncommercial_artifact_remains_uncataloged(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        admission = self.review["admission"]
        self.assertEqual(
            admission["status"],
            "reviewed_not_admitted_gated_noncommercial_unresolved_artifact",
        )
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["liveQualified"])
        self.assertEqual(admission["executableModes"], [])

        license_review = self.review["licenseReview"]
        self.assertTrue(license_review["acceptanceRequired"])
        self.assertFalse(license_review["commercialUseAllowedWithoutSeparateAgreement"])
        self.assertEqual(license_review["linkedLicenseDeed"], "cc-by-nc-4.0")
        self.assertFalse(license_review["exactRepositoryLicensePayloadReviewed"])

        repository = self.review["repository"]
        self.assertTrue(repository["gated"])
        self.assertFalse(repository["private"])
        self.assertIsNone(repository["revision"])
        self.assertFalse(repository["weightInventoryExact"])
        self.assertFalse(repository["listedPythonFilesPresent"])
        self.assertFalse(repository["trustRemoteCodeRequiredByPackageRoute"])
        self.assertIsNone(catalog_repository_pin(repository["repository"]))
        self.assertNotIn("BriaPipeline", IMAGE_PIPELINE_ADAPTERS)

    def test_public_tree_observations_are_not_promoted_to_exact_inventory(self):
        repository = self.review["repository"]
        self.assertEqual(repository["weightFileCountObserved"], 5)
        self.assertEqual(len(repository["observedWeightFiles"]), 5)
        for item in repository["observedWeightFiles"]:
            self.assertTrue(item["path"].endswith(".safetensors"))
            self.assertRegex(item["publishedRoundedSize"], r"^[0-9.]+ (?:GB|MB)$")
            self.assertNotIn("byteSize", item)
            self.assertNotIn("sha256", item)
        self.assertTrue(
            all(SHA1.fullmatch(commit) for commit in repository["observedFileCommitCandidates"])
        )

        access = self.review["metadataAccess"]
        self.assertEqual(access["anonymousApiHttpStatus"], 401)
        self.assertIsNone(access["resolvedRepositoryRevision"])
        self.assertFalse(access["exactArtifactInventoryAvailable"])
        self.assertFalse(access["modelIndexPayloadReviewed"])
        self.assertFalse(access["componentConfigPayloadsReviewed"])

    def test_package_owned_api_and_bounds_are_recorded_without_admission(self):
        pinned = self.review["pinnedDiffusers"]
        self.assertEqual(
            pinned["revision"],
            "bb56997d4b7e87f0743f26a612f49ec4e7ce7213",
        )
        self.assertEqual(
            pinned["packageOwnedClasses"],
            ["BriaPipeline", "BriaTransformer2DModel"],
        )
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        self.assertTrue(all(SHA256.fullmatch(value) for value in pinned["sourceSha256"].values()))

        contract = self.review["pipelineContract"]
        self.assertEqual(
            (
                contract["defaultWidth"],
                contract["defaultHeight"],
                contract["defaultNumInferenceSteps"],
                contract["defaultGuidanceScale"],
                contract["defaultMaxSequenceLength"],
                contract["maximumPackageSequenceLength"],
            ),
            (1024, 1024, 30, 5.0, 128, 512),
        )
        self.assertIn("num_inference_steps", contract["unboundedByPackage"])
        self.assertIn("height", contract["unboundedByPackage"])
        self.assertIn("width", contract["unboundedByPackage"])

    def test_required_granular_precision_route_is_explicitly_unresolved(self):
        precision = self.review["precisionContract"]
        self.assertFalse(precision["genericAdapterCanRepresentExactRecipe"])
        self.assertEqual(precision["loadDtype"], "bfloat16")
        self.assertEqual(precision["requiredTextEncoderFinalProjectionDtype"], "float32")
        self.assertEqual(precision["requiredVaeDtypeWhenShiftFactorIsZero"], "float32")
        envelope = self.review["resourceEnvelope"]
        self.assertEqual(
            envelope["status"],
            "not_estimated_until_authenticated_inventory_review",
        )
        self.assertIsNone(envelope["minimumSelectiveDiskBytesEstimate"])


if __name__ == "__main__":
    unittest.main()
