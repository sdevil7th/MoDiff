import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin


REVIEW_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "kandinsky5-video-artifact-review.json"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Kandinsky5VideoArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_heavy_family_remains_contract_only_and_uncataloged(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        self.assertEqual(self.review["format"], "safetensors")
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "contract_only")
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["liveQualified"])
        self.assertEqual(admission["executableModes"], [])
        self.assertIn(
            "corrected_pro_image_to_video_documentation",
            admission["unresolvedGates"],
        )
        for repository in self.review["repositories"]:
            self.assertRegex(repository["revision"], r"^[0-9a-f]{40}$")
            self.assertFalse(repository["gated"])
            self.assertFalse(repository["private"])
            self.assertIsNone(catalog_repository_pin(repository["repository"]))

    def test_ten_exact_inventories_share_encoders_and_vae_but_not_transformers(self):
        common = self.review["commonWeightFiles"]
        common_bytes = sum(item["byteSize"] for item in common)
        self.assertEqual((len(common), common_bytes), (7, 19280899008))
        self.assertTrue(all(item["path"].endswith(".safetensors") for item in common))
        self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in common))

        expected = {
            "proSft5s": (62666834400, 43385935392),
            "liteSft5s": (23854029536, 4573130528),
            "liteNoCfg5s": (23854029536, 4573130528),
            "liteDistilled5s": (23854029536, 4573130528),
            "litePretrain5s": (23854029536, 4573130528),
            "liteSft10s": (23854029536, 4573130528),
            "liteNoCfg10s": (23854029536, 4573130528),
            "liteDistilled10s": (23854029536, 4573130528),
            "litePretrain10s": (23854029536, 4573130528),
            "proI2vSft5s": (96526404752, 77245505744),
        }
        repositories = self.review["repositories"]
        self.assertEqual(len(repositories), 10)
        self.assertEqual(len({item["transformer"]["sha256"] for item in repositories}), 10)
        for repository in repositories:
            transformer = repository["transformer"]
            self.assertEqual(repository["weightFileCount"], len(common) + 1)
            self.assertEqual(repository["weightBytes"], common_bytes + transformer["byteSize"])
            self.assertEqual(
                (repository["weightBytes"], transformer["byteSize"]),
                expected[repository["role"]],
            )
            self.assertRegex(repository["weightInventorySha256"], SHA256)
            self.assertRegex(repository["modelIndexSha256"], SHA256)
            self.assertRegex(transformer["configSha256"], SHA256)
            self.assertRegex(transformer["sha256"], SHA256)

    def test_pipeline_contracts_keep_image_required(self):
        contracts = {
            contract["pipelineClass"]: contract for contract in self.review["pipelineContracts"]
        }
        self.assertEqual(contracts["Kandinsky5T2VPipeline"]["requiredInputs"], ["prompt"])
        self.assertEqual(contracts["Kandinsky5I2VPipeline"]["requiredInputs"], ["image"])
        for contract in contracts.values():
            self.assertEqual(
                (
                    contract["defaultWidth"],
                    contract["defaultHeight"],
                    contract["defaultNumFrames"],
                    contract["defaultNumInferenceSteps"],
                    contract["defaultGuidanceScale"],
                ),
                (768, 512, 121, 50, 5.0),
            )

    def test_source_recipes_reject_broken_i2v_docs_and_preserve_variant_rules(self):
        recipes = self.review["reviewedRecipes"]
        rejected = recipes["rejectedDocumentationImageToVideoExample"]
        self.assertFalse(rejected["validRecipeProof"])
        self.assertFalse(rejected["loadedImagePassedToPipeline"])
        self.assertEqual(rejected["callsPipelineClass"], "Kandinsky5T2VPipeline")
        self.assertEqual(rejected["correctPipelineClass"], "Kandinsky5I2VPipeline")
        self.assertTrue(recipes["sourceImageToVideo"]["imageRequired"])

        self.assertEqual(
            (
                recipes["liteFiveSecond"]["numFrames"],
                recipes["liteTenSecond"]["numFrames"],
                recipes["liteNoCfgAndDistilled"]["distilledNumInferenceSteps"],
                recipes["liteNoCfgAndDistilled"]["guidanceScale"],
            ),
            (121, 241, 16, 1.0),
        )
        self.assertEqual(recipes["liteTenSecond"]["attentionBackend"], "flex")
        self.assertTrue(recipes["liteTenSecond"]["compileDynamic"])
        self.assertTrue(recipes["proTextToVideo"]["cpuModelOffload"])
        self.assertTrue(
            all(
                envelope["status"] == "estimate_only_qualification_pending"
                for envelope in self.review["remoteResourceEnvelopes"].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
