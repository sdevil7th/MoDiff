import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "easyanimate-artifact-review.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EasyAnimateArtifactReviewTests(unittest.TestCase):
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
        self.assertIn("generic_media_preprocessing_contracts", admission["unresolvedGates"])
        for repository in self.review["repositories"]:
            self.assertRegex(repository["revision"], r"^[0-9a-f]{40}$")
            self.assertFalse(repository["gated"])
            self.assertFalse(repository["private"])
            self.assertIsNone(catalog_repository_pin(repository["repository"]))

    def test_exact_safe_weight_receipts_keep_three_transformers_distinct(self):
        common = self.review["commonWeightFiles"]
        common_bytes = sum(item["byteSize"] for item in common)
        self.assertEqual((len(common), common_bytes), (6, 17560358508))
        self.assertTrue(all(item["path"].endswith(".safetensors") for item in common))
        self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in common))

        expected = {
            "textToVideo7B": (7, 31189417564, 16, 36),
            "inpaint12B": (7, 41159117300, 33, 48),
            "control12B": (7, 41159485940, 48, 48),
        }
        for repository in self.review["repositories"]:
            transformer = repository["transformer"]
            self.assertEqual(
                (
                    repository["weightFileCount"],
                    repository["weightBytes"],
                    transformer["inputChannels"],
                    transformer["layers"],
                ),
                expected[repository["role"]],
            )
            self.assertEqual(repository["weightFileCount"], len(common) + 1)
            self.assertEqual(repository["weightBytes"], common_bytes + transformer["byteSize"])
            self.assertTrue(transformer["path"].endswith(".safetensors"))
            self.assertRegex(transformer["sha256"], SHA256)
            self.assertRegex(transformer["configSha256"], SHA256)
            self.assertRegex(repository["weightInventorySha256"], SHA256)
            self.assertRegex(repository["modelIndex"]["sha256"], SHA256)

    def test_source_contract_and_recipes_bind_exact_pipeline_roles(self):
        source = self.review["sourceContract"]
        self.assertRegex(source["diffusersRevision"], r"^[0-9a-f]{40}$")
        self.assertFalse(source["remoteCodeRequired"])
        documentation = source["documentation"]
        self.assertEqual(
            (
                documentation["minimumDimension"],
                documentation["maximumDimension"],
                documentation["minimumNumFrames"],
                documentation["maximumNumFrames"],
                documentation["bestNumFrames"],
                documentation["recommendedExportFps"],
            ),
            (256, 1024, 1, 49, 49, 8),
        )
        self.assertRegex(documentation["sha256"], SHA256)

        contracts = {item["repositoryRole"]: item for item in self.review["pipelineContracts"]}
        self.assertEqual(
            {
                role: (item["pipelineClass"], item["transformerInputChannels"])
                for role, item in contracts.items()
            },
            {
                "textToVideo7B": ("EasyAnimatePipeline", 16),
                "inpaint12B": ("EasyAnimateInpaintPipeline", 33),
                "control12B": ("EasyAnimateControlPipeline", 48),
            },
        )
        self.assertEqual(contracts["textToVideo7B"]["requiredInputs"], ["prompt"])
        self.assertEqual(
            contracts["inpaint12B"]["requiredPipelineTensorInputs"],
            ["video", "mask_video"],
        )
        self.assertEqual(
            contracts["control12B"]["requiredPipelineTensorInputs"],
            ["control_video"],
        )

    def test_reviewed_examples_are_not_reported_as_live_qualification(self):
        recipes = self.review["reviewedRecipes"]
        self.assertEqual(
            (
                recipes["textToVideo"]["width"],
                recipes["textToVideo"]["height"],
                recipes["textToVideo"]["numFrames"],
                recipes["textToVideo"]["numInferenceSteps"],
                recipes["textToVideo"]["guidanceScale"],
                recipes["textToVideo"]["fps"],
            ),
            (512, 512, 49, 50, 6.0, 8),
        )
        self.assertEqual(
            (recipes["imageToVideo"]["height"], recipes["imageToVideo"]["width"]),
            (448, 576),
        )
        self.assertEqual(
            (recipes["controlToVideo"]["height"], recipes["controlToVideo"]["width"]),
            (672, 384),
        )
        self.assertTrue(
            all(
                envelope["status"] == "estimate_only_qualification_pending"
                for envelope in self.review["remoteResourceEnvelopes"].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
