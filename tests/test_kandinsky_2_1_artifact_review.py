import hashlib
import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.studio_execution_specs import STUDIO_EXECUTION_SPEC_DEFINITIONS


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data" / "kandinsky-2-1-artifact-review.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Kandinsky21ArtifactReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_three_exact_public_snapshots_are_safe_only_and_uncataloged(self):
        expected = {
            "decoder": (
                "kandinsky-community/kandinsky-2-1",
                "04b366659663bf60deacb96fd3e38f1cb39a3be4",
            ),
            "prior": (
                "kandinsky-community/kandinsky-2-1-prior",
                "b726af62d69111bbf62b5d71bb9d90d9867fbce4",
            ),
            "inpaintDecoder": (
                "kandinsky-community/kandinsky-2-1-inpaint",
                "1ecbdf9b40b35eb9106ece73976f92b13f475433",
            ),
        }
        self.assertEqual(set(self.review["repositories"]), set(expected))
        for role, (repository, revision) in expected.items():
            with self.subTest(role=role):
                reviewed = self.review["repositories"][role]
                self.assertEqual((reviewed["repository"], reviewed["revision"]), (repository, revision))
                self.assertFalse(reviewed["private"])
                self.assertFalse(reviewed["gated"])
                self.assertEqual(reviewed["declaredLicense"], "apache-2.0")
                self.assertFalse(reviewed["licenseFilePresent"])
                self.assertFalse(reviewed["pythonFilesPresent"])
                self.assertFalse(reviewed["trustRemoteCodeRequired"])
                self.assertIsNone(catalog_repository_pin(repository))

    def test_selected_weight_inventories_are_exact_and_header_only(self):
        expected = {
            "decoder": (3, 7428873150, "c89660a4870aa88352cdc970240a60e52e5e34e99b12e28e216a1e26a721baf9"),
            "prior": (3, 5798697734, "aec79dbda98a49640221db5230df3d3cc64dcb628f0bd44a0580b5fdce7c329e"),
            "inpaintDecoder": (3, 7428942270, "d85f6e595048f82c4ed662a7bf956e432f664e6341e083ffe7e1de6cf1568864"),
        }
        fetched_headers = {}
        for role, (file_count, byte_size, digest) in expected.items():
            with self.subTest(role=role):
                repository = self.review["repositories"][role]
                weights = repository["selectedWeightFiles"]
                entries = [
                    {field: weight[field] for field in ("path", "byteSize", "sha256")}
                    for weight in weights
                ]
                canonical = json.dumps(
                    sorted(entries, key=lambda item: item["path"]),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                self.assertEqual((len(weights), sum(item["byteSize"] for item in weights)), (file_count, byte_size))
                self.assertEqual(hashlib.sha256(canonical).hexdigest(), digest)
                self.assertEqual(repository["weightInventorySha256"], digest)
                self.assertTrue(all(item["path"].endswith(".safetensors") for item in weights))
                self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in weights))
                self.assertTrue(all(item["tensorCount"] > 0 for item in weights))
                for item in weights:
                    fetched_headers.setdefault(item["sha256"], item["headerBytesFetched"])

        serialization = self.review["serializationReview"]
        self.assertEqual(
            sum(fetched_headers.values()),
            serialization["safetensorsHeaderBytesFetched"],
        )
        self.assertEqual(
            len(fetched_headers),
            serialization["safetensorsHeadersFetched"],
        )
        self.assertFalse(serialization["fullWeightFilesDownloaded"])
        self.assertEqual(serialization["remoteWeightBytesFetched"], 0)
        self.assertTrue(serialization["safeSerializationAvailableForEveryRequiredComponent"])
        self.assertFalse(serialization["duplicateLegacyBinWeightsSelected"])

    def test_package_contract_seals_three_combined_modes_and_cancellation_gap(self):
        pinned = self.review["pinnedDiffusers"]
        self.assertEqual(pinned["revision"], "bb56997d4b7e87f0743f26a612f49ec4e7ce7213")
        self.assertEqual(len(pinned["pipelineClasses"]), 7)
        self.assertTrue(all(SHA256.fullmatch(digest) for digest in pinned["sourceSha256"].values()))

        contracts = self.review["combinedPipelineContracts"]
        self.assertEqual(contracts["textToImage"]["requiredInputs"], ["prompt"])
        self.assertEqual(contracts["imageToImage"]["requiredInputs"], ["prompt", "image"])
        self.assertEqual(contracts["inpaint"]["requiredInputs"], ["prompt", "image", "mask_image"])
        for mode in ("textToImage", "imageToImage", "inpaint"):
            contract = contracts[mode]
            self.assertEqual(
                (
                    contract["defaultNumInferenceSteps"],
                    contract["defaultPriorNumInferenceSteps"],
                    contract["defaultGuidanceScale"],
                    contract["defaultHeight"],
                    contract["defaultWidth"],
                ),
                (100, 25, 4.0, 512, 512),
            )
        self.assertEqual(contracts["maximumTextTokens"], 77)
        self.assertEqual(contracts["requestedResolutionRoundedUpToMultiple"], 64)
        self.assertTrue(contracts["legacyDecoderCallbackAvailable"])
        self.assertFalse(contracts["priorCallbackAvailable"])
        self.assertFalse(contracts["cooperativeCancellationCoversPriorAndDecoder"])
        self.assertFalse(contracts["interruptFlagAvailable"])
        self.assertFalse(contracts["safetyCheckerAvailable"])
        self.assertFalse(contracts["backendOwnedUpperBoundsPresent"])

    def test_connected_prior_cannot_be_loaded_with_both_exact_revisions(self):
        review = self.review["connectedPriorRevisionReview"]
        self.assertTrue(review["combinedPipelinesLoadConnectedPipes"])
        self.assertTrue(review["modelCardConnectionContainsRepositoryOnly"])
        self.assertFalse(review["modelCardConnectionContainsImmutableRevision"])
        self.assertTrue(review["decoderAndPriorRevisionsAreDistinct"])
        self.assertTrue(review["fromPretrainedCopiesDecoderRevisionToConnectedPrior"])
        self.assertFalse(review["decoderRevisionExistsOnPriorRepository"])
        self.assertFalse(review["immutableCombinedFromPretrainedLoadIsValid"])
        self.assertFalse(review["downloadPathForwardsRevisionToConnectedPrior"])
        self.assertTrue(review["downloadPathUsesMovingConnectedPrior"])
        self.assertFalse(review["backendCompositeRevisionBindingImplemented"])

    def test_license_receipt_is_upstream_only(self):
        license_review = self.review["licenseReview"]
        self.assertEqual(license_review["modelCardDeclaredLicense"], "apache-2.0")
        self.assertFalse(license_review["modelSnapshotLicenseFilePresent"])
        self.assertFalse(license_review["modelSnapshotNoticeFilePresent"])
        self.assertEqual(license_review["upstreamLicenseId"], "apache-2.0")
        self.assertEqual(license_review["upstreamLicenseByteSize"], 11357)
        self.assertRegex(license_review["upstreamLicenseSha256"], SHA256)
        self.assertTrue(license_review["exactModelSnapshotLicenseClarificationRequired"])

    def test_admission_remains_closed_on_every_user_facing_surface(self):
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "review_complete_blocked")
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["capabilityExposed"])
        self.assertEqual(admission["canonicalWorkflowsAdded"], 0)
        self.assertFalse(admission["clientSurfaceAdded"])
        self.assertFalse(admission["autoEligible"])
        self.assertFalse(admission["galleryEligible"])
        self.assertFalse(admission["liveQualified"])
        self.assertIn("exact_connected_prior_revision_binding", admission["unresolvedGates"])
        self.assertIn("cooperative_prior_and_decoder_cancellation", admission["unresolvedGates"])
        self.assertIn("physical_macos_execution", admission["unresolvedGates"])
        self.assertFalse(
            any(
                definition.get("modelType", "").startswith("kandinsky-2-1")
                for definition in STUDIO_EXECUTION_SPEC_DEFINITIONS.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
