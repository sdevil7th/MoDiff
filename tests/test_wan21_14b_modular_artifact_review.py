import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.modular_workflow_contracts import (
    PINNED_MODULAR_REPOSITORY_VARIANTS,
    WAN_FLF_REPOSITORY,
    WAN_I2V_720P_REPOSITORY,
    WAN_I2V_REPOSITORY,
    WAN_T2V_14B_REPOSITORY,
    WAN_T2V_REPOSITORY,
    WAN_WORKFLOW_REPOSITORIES,
)


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "wan21-14b-modular-artifact-review.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Wan21FourteenBModularArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
        self.repositories = {item["role"]: item for item in self.review["repositories"]}

    def test_admission_is_exact_and_does_not_claim_live_qualification(self):
        admission = self.review["admission"]
        self.assertEqual(self.review["schemaVersion"], 1)
        self.assertEqual(self.review["format"], "safetensors")
        self.assertEqual(admission["status"], "graph_qualified_execution_pending")
        self.assertTrue(admission["repositoryScopedLoaderAdmission"])
        self.assertFalse(admission["newHighLevelModeAdded"])
        self.assertFalse(admission["newModelNamedClientBranchAdded"])
        self.assertFalse(admission["liveQualified"])
        self.assertFalse(admission["remoteCodeAllowed"])
        self.assertIn("physical_macos_execution", admission["unresolvedGates"])

    def test_runtime_variants_and_catalog_pins_are_repository_scoped(self):
        self.assertEqual(
            PINNED_MODULAR_REPOSITORY_VARIANTS["WanModularPipeline"],
            (WAN_T2V_REPOSITORY, WAN_T2V_14B_REPOSITORY),
        )
        self.assertEqual(
            PINNED_MODULAR_REPOSITORY_VARIANTS["WanImage2VideoModularPipeline"],
            (WAN_I2V_REPOSITORY, WAN_I2V_720P_REPOSITORY, WAN_FLF_REPOSITORY),
        )
        self.assertEqual(
            WAN_WORKFLOW_REPOSITORIES,
            (
                ("image2video", WAN_I2V_REPOSITORY),
                ("image2video", WAN_I2V_720P_REPOSITORY),
                ("flf2v", WAN_FLF_REPOSITORY),
            ),
        )
        for repository in self.review["repositories"]:
            pin = catalog_repository_pin(repository["repository"])
            self.assertIsNotNone(pin)
            self.assertEqual(pin["revision"], repository["revision"])
            self.assertEqual(pin["license"], "apache-2.0")

    def test_exact_safe_weight_receipts_and_distinct_transformers(self):
        expected = {
            "textToVideo14B": (18, 80385341396, 12, 57154077760),
            "imageToVideo480P": (21, 90075953948, 14, 65580472552),
            "imageToVideo720P": (21, 90075953948, 14, 65580472552),
            "firstLastFrame720P": (21, 90077762204, 14, 65582280808),
        }
        for role, repository in self.repositories.items():
            transformer = repository["transformer"]
            self.assertEqual(
                (
                    repository["weightFileCount"],
                    repository["weightBytes"],
                    transformer["fileCount"],
                    transformer["weightBytes"],
                ),
                expected[role],
            )
            self.assertRegex(repository["canonicalInventorySha256"], SHA256)
            self.assertEqual(len(transformer["weightSha256"]), transformer["fileCount"])
            self.assertTrue(all(SHA256.fullmatch(value) for value in transformer["weightSha256"]))

        self.assertEqual(
            self.repositories["imageToVideo480P"]["transformer"]["configSha256"],
            self.repositories["imageToVideo720P"]["transformer"]["configSha256"],
        )
        self.assertNotEqual(
            self.repositories["imageToVideo480P"]["transformer"]["weightSha256"],
            self.repositories["imageToVideo720P"]["transformer"]["weightSha256"],
        )
        self.assertEqual(self.repositories["firstLastFrame720P"]["transformer"]["posEmbedSeqLen"], 514)

    def test_source_recipes_keep_resolution_and_workflow_semantics_distinct(self):
        t2v = self.repositories["textToVideo14B"]["sourceRecipe"]
        i2v_480 = self.repositories["imageToVideo480P"]["sourceRecipe"]
        i2v_720 = self.repositories["imageToVideo720P"]["sourceRecipe"]
        flf = self.repositories["firstLastFrame720P"]["sourceRecipe"]
        self.assertEqual(
            (t2v["height"], t2v["width"], t2v["numFrames"], t2v["guidanceScale"], t2v["fps"]),
            (480, 832, 81, 5.0, 15),
        )
        self.assertEqual((i2v_480["maxArea"], i2v_720["maxArea"]), (480 * 832, 720 * 1280))
        self.assertEqual((i2v_480["numFrames"], i2v_720["numFrames"]), (81, 81))
        self.assertEqual((i2v_480["fps"], i2v_720["fps"]), (16, 16))
        self.assertTrue(flf["requiresLastImage"])
        self.assertEqual((flf["maxArea"], flf["guidanceScale"], flf["fps"]), (720 * 1280, 5.5, 16))
        self.assertEqual(
            self.review["remoteResourceEnvelope"]["status"],
            "estimate_only_qualification_pending",
        )


if __name__ == "__main__":
    unittest.main()
