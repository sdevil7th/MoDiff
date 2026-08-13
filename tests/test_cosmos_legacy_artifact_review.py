import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin


REVIEW_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "cosmos-legacy-artifact-review.json"
)
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CosmosLegacyArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_gated_family_remains_metadata_only_and_uncataloged(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        self.assertEqual(self.review["format"], "safetensors")
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "metadata_only")
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["liveQualified"])
        self.assertEqual(admission["executableModes"], [])
        self.assertIn("nvidia_open_model_license_acceptance", admission["unresolvedGates"])

        for repository in self.review["repositories"]:
            self.assertEqual(repository["gated"], "auto")
            self.assertFalse(repository["private"])
            self.assertRegex(repository["revision"], SHA1)
            self.assertIsNone(catalog_repository_pin(repository["repository"]))
            if repository["modelIndex"] is not None:
                self.assertFalse(repository["modelIndex"]["contentAccessible"])
                self.assertRegex(repository["modelIndex"]["gitBlobSha1"], SHA1)

    def test_eight_immutable_safetensors_manifests_have_exact_component_totals(self):
        expected = {
            "cosmos1TextToWorld7B": (8, 24575681664),
            "cosmos1VideoToWorld7B": (12, 41564530104),
            "predict2TextToImage2B": (4, 14150076300),
            "predict2VideoToWorld2B": (4, 14150092684),
            "predict2VideoToWorld14B": (9, 38767856404),
            "predict2.5BasePostTrained2B": (6, 21210423156),
            "transfer2.5General2B": (6, 21919819988),
            "transfer2.5EdgeControlNet": (1, 942523208),
        }
        self.assertEqual(len(self.review["repositories"]), 8)
        for repository in self.review["repositories"]:
            component_count = sum(
                component["fileCount"] for component in repository["weightComponents"].values()
            )
            component_bytes = sum(
                component["byteSize"] for component in repository["weightComponents"].values()
            )
            self.assertEqual(component_count, repository["weightFileCount"])
            self.assertEqual(component_bytes, repository["weightBytes"])
            self.assertEqual(
                (repository["weightFileCount"], repository["weightBytes"]),
                expected[repository["role"]],
            )
            self.assertRegex(repository["weightInventorySha256"], SHA256)

    def test_three_accepted_license_versions_remain_explicit_external_gates(self):
        expected = {
            "2025-01-06": (8909, 2),
            "2025-04-30": (8908, 3),
            "2025-09-23": (5861, 3),
        }
        for license_contract in self.review["licenses"]:
            self.assertTrue(license_contract["acceptanceRequired"])
            self.assertTrue(license_contract["commercialUseAllowedByText"])
            self.assertTrue(license_contract["builtOnNvidiaCosmosNoticeRequired"])
            self.assertTrue(license_contract["licenseAndNoticeFileRequiredOnDistribution"])
            self.assertTrue(license_contract["trustworthyAiTermsLinked"])
            self.assertRegex(license_contract["promptSha256"], SHA256)
            self.assertEqual(
                (license_contract["promptByteSize"], len(license_contract["repositoryRoles"])),
                expected[license_contract["releaseDate"]],
            )

    def test_source_recipes_are_recorded_without_live_qualification(self):
        recipes = self.review["reviewedRecipes"]
        self.assertEqual(
            (
                recipes["cosmos1"]["width"],
                recipes["cosmos1"]["height"],
                recipes["cosmos1"]["numFrames"],
                recipes["cosmos1"]["numInferenceSteps"],
                recipes["cosmos1"]["fps"],
            ),
            (1280, 704, 121, 36, 30),
        )
        self.assertEqual(
            (
                recipes["predict2.5"]["numFrames"],
                recipes["predict2.5"]["numInferenceSteps"],
                recipes["predict2.5"]["fps"],
            ),
            (93, 36, 16),
        )
        self.assertEqual(recipes["transfer2.5Edge"]["numFramesPerChunk"], 93)
        self.assertEqual(recipes["transfer2.5Edge"]["controlsConditioningScale"], 1.0)
        self.assertTrue(
            all(
                envelope["status"] == "estimate_only_qualification_pending"
                for envelope in self.review["remoteResourceEnvelopes"].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
