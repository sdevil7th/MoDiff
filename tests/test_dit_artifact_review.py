import hashlib
import json
import re
import unittest
from pathlib import Path

from modiff.model_artifact_catalog import catalog_repository_pin


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data" / "dit-artifact-review.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DiTArtifactReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_only_exact_official_snapshots_are_reviewed_and_not_cataloged(self):
        discovery = self.review["officialRepositoryDiscovery"]
        self.assertEqual(discovery["exactFacebookResultCount"], 2)
        self.assertEqual(discovery["safetensorsSnapshotsFound"], 0)
        self.assertFalse(discovery["communityConversionsReviewedForAdmission"])
        expected = {
            "imagenet256": (
                "facebook/DiT-XL-2-256",
                "eab87f77abd5aef071a632f08807fbaab0b704d0",
                [256, 256],
            ),
            "imagenet512": (
                "facebook/DiT-XL-2-512",
                "101a3d462b22d64c4afdd4d0c8c59a2c0b961b99",
                [512, 512],
            ),
        }
        for variant, (repository, revision, resolution) in expected.items():
            with self.subTest(variant=variant):
                reviewed = self.review["repositories"][variant]
                self.assertEqual(reviewed["repository"], repository)
                self.assertEqual(reviewed["revision"], revision)
                self.assertEqual(reviewed["outputResolution"], resolution)
                self.assertFalse(reviewed["private"])
                self.assertFalse(reviewed["gated"])
                self.assertFalse(reviewed["pythonFilesPresent"])
                self.assertFalse(reviewed["trustRemoteCodeRequired"])
                self.assertIsNone(catalog_repository_pin(repository))

    def test_weight_inventory_is_exact_and_fails_the_safe_serialization_gate(self):
        flattened = []
        for variant in ("imagenet256", "imagenet512"):
            repository = self.review["repositories"][variant]
            weights = self.review["weightFiles"][variant]
            self.assertEqual(repository["weightFileCount"], 2)
            self.assertEqual(repository["weightBytes"], sum(item["byteSize"] for item in weights))
            self.assertEqual(repository["safetensorsFileCount"], 0)
            self.assertEqual(repository["legacyPytorchBinFileCount"], 2)
            self.assertEqual(repository["repositoryBytes"], 3334284997)
            for item in weights:
                self.assertTrue(item["path"].endswith(".bin"))
                self.assertEqual(item["serialization"], "legacy_pytorch_pickle_bin")
                self.assertRegex(item["sha256"], SHA256)
                flattened.append({"variant": variant, **item})

        canonical = json.dumps(
            sorted(flattened, key=lambda item: (item["variant"], item["path"])),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(canonical).hexdigest(),
            self.review["artifactInventoryCanonicalization"]["weightInventorySha256"],
        )

    def test_metadata_and_repository_contract_are_immutable(self):
        for variant in ("imagenet256", "imagenet512"):
            for digest in self.review["metadataSha256"][variant].values():
                self.assertRegex(digest, SHA256)
        contract = self.review["repositoryContract"]
        self.assertEqual(contract["modelIndexPipelineClass"], "DiTPipeline")
        self.assertEqual(contract["installedCompatibilityTransformerClass"], "DiTTransformer2DModel")
        self.assertEqual(contract["imageNetClassCount"], 1000)
        self.assertEqual((contract["minimumClassLabel"], contract["maximumClassLabel"]), (0, 999))
        self.assertEqual(contract["classifierFreeGuidanceNullClass"], 1000)
        self.assertEqual(contract["vaeScaleFactor"], 8)

    def test_pinned_package_api_is_exact_but_lacks_runtime_guardrails(self):
        pinned = self.review["pinnedDiffusers"]
        self.assertEqual(pinned["revision"], "bb56997d4b7e87f0743f26a612f49ec4e7ce7213")
        for field in (
            "pipelineSha256",
            "transformerSha256",
            "autoencoderSha256",
            "schedulerSha256",
            "pipelineUtilsSha256",
        ):
            self.assertRegex(pinned[field], SHA256)
        self.assertIn("class_labels: list[int]", pinned["callSignature"])
        self.assertEqual(pinned["modelCpuOffloadSequence"], "transformer->vae")
        self.assertFalse(pinned["callbackOnStepEndAvailable"])
        self.assertFalse(pinned["cooperativeCancellationAvailable"])
        self.assertFalse(pinned["safetyCheckerAvailable"])
        self.assertEqual(pinned["callDefaultInferenceSteps"], 50)
        self.assertEqual(pinned["docstringDefaultInferenceSteps"], 250)
        self.assertEqual(pinned["officialExampleInferenceSteps"], 25)

    def test_license_and_resource_claims_remain_conservative(self):
        license_review = self.review["licenseReview"]
        self.assertEqual(license_review["declaredLicense"], "cc-by-nc-4.0")
        self.assertFalse(license_review["bundledLicenseTextPresent"])
        self.assertFalse(license_review["commercialUsePermitted"])
        self.assertFalse(license_review["commercialProductAdmissionPermitted"])
        self.assertTrue(license_review["productLegalApprovalRequired"])
        for envelope in self.review["resourceEnvelopes"].values():
            self.assertEqual(envelope["status"], "estimate_only_modiff_qualification_pending")
            self.assertGreaterEqual(envelope["minimumSelectiveDiskBytes"], 2 * 3334245438)
            self.assertGreater(envelope["minimumAcceleratorMemoryBytes"], 3334245438)

    def test_admission_is_closed_without_download_or_surface_claims(self):
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "review_complete_blocked")
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["capabilityExposed"])
        self.assertEqual(admission["canonicalWorkflowsAdded"], 0)
        self.assertFalse(admission["clientSurfaceAdded"])
        self.assertFalse(admission["autoEligible"])
        self.assertFalse(admission["galleryEligible"])
        self.assertFalse(admission["fullWeightFilesDownloaded"])
        self.assertEqual(admission["remoteWeightBytesFetched"], 0)
        self.assertIn("official_safetensors_artifacts", admission["unresolvedGates"])
        self.assertIn("commercial_product_weight_rights", admission["unresolvedGates"])
        self.assertIn("physical_macos_execution", admission["unresolvedGates"])


if __name__ == "__main__":
    unittest.main()
