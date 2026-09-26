import hashlib
import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "dreamlite-artifact-review.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DreamLiteArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_exact_safe_repositories_are_source_complete_but_not_live_qualified(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        self.assertEqual(self.review["format"], "safetensors")
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "source_complete_execution_pending")
        self.assertTrue(admission["runtimeCatalogExposed"])
        self.assertTrue(admission["downloadCatalogExposed"])
        self.assertFalse(admission["liveQualified"])
        self.assertFalse(admission["autoEligible"])
        self.assertFalse(admission["galleryEligible"])
        self.assertIn("physical_macos_execution", admission["unresolvedGates"])

        repository_code = self.review["repositoryCode"]
        self.assertFalse(repository_code["pythonFilesPresent"])
        self.assertFalse(repository_code["trustRemoteCodeRequired"])
        self.assertTrue(repository_code["upstreamLibraryClassesOnly"])

        for repository in self.review["repositories"]:
            self.assertRegex(repository["revision"], SHA1)
            self.assertFalse(repository["gated"])
            self.assertFalse(repository["private"])
            pin = catalog_repository_pin(repository["repository"])
            self.assertIsNotNone(pin)
            self.assertEqual(pin["revision"], repository["revision"])
            self.assertEqual(pin["license"], "cc-by-nc-4.0")

    def test_two_three_file_inventories_share_encoder_and_vae_only(self):
        shared = self.review["sharedWeightFiles"]
        variants = self.review["variantWeightFiles"]
        self.assertEqual(len(shared), 2)
        self.assertNotEqual(variants["base"]["sha256"], variants["mobile"]["sha256"])

        repositories = {item["role"]: item for item in self.review["repositories"]}
        for role, variant in variants.items():
            entries = sorted([*shared, variant], key=lambda item: item["path"])
            canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
            repository = repositories[role]
            self.assertEqual(repository["weightFileCount"], len(entries))
            self.assertEqual(repository["weightBytes"], sum(item["byteSize"] for item in entries))
            self.assertEqual(repository["weightBytes"], 5040118270)
            self.assertEqual(
                repository["weightInventorySha256"],
                hashlib.sha256(canonical).hexdigest(),
            )
            for item in entries:
                self.assertTrue(item["path"].endswith(".safetensors"))
                self.assertRegex(item["sha256"], SHA256)

    def test_base_and_mobile_generation_controls_remain_distinct(self):
        contracts = {
            item["pipelineClass"]: item for item in self.review["pipelineContracts"]
        }
        base = contracts["DreamLitePipeline"]
        self.assertEqual(base["modes"], ["text_to_image", "edit_image"])
        self.assertEqual(
            (
                base["defaultWidth"],
                base["defaultHeight"],
                base["recommendedNumInferenceSteps"],
                base["defaultGuidanceScale"],
                base["defaultImageGuidanceScale"],
                base["maxSequenceLength"],
            ),
            (1024, 1024, 28, 3.5, 1.5, 200),
        )

        mobile = contracts["DreamLiteMobilePipeline"]
        self.assertTrue(mobile["guidanceInputsIgnored"])
        self.assertEqual(mobile["defaultNumInferenceSteps"], 4)
        self.assertEqual(mobile["maxNumInferenceSteps"], 8)
        self.assertIsNone(mobile["defaultGuidanceScale"])
        self.assertFalse(base["stepCallbackSupported"])
        self.assertFalse(mobile["stepCallbackSupported"])

    def test_pinned_library_sources_and_noncommercial_boundary_are_explicit(self):
        self.assertFalse(self.review["license"]["commercialUseAllowed"])
        self.assertEqual(self.review["license"]["id"], "cc-by-nc-4.0")
        self.assertTrue(
            all(
                SHA256.fullmatch(digest)
                for digest in self.review["pinnedDiffusersSources"].values()
            )
        )
        self.assertEqual(
            self.review["remoteResourceEnvelope"]["status"],
            "estimate_only_qualification_pending",
        )


if __name__ == "__main__":
    unittest.main()
