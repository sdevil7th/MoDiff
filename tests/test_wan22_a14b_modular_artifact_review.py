import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.modular_workflow_discovery import reviewed_modular_workflow_contract


REVIEW_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "wan22-a14b-modular-artifact-review.json"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Wan22A14BModularArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_modular_admission_remains_contract_only_without_changing_standard_adapters(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        self.assertEqual(self.review["format"], "safetensors")
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "contract_only")
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["newDownloadCatalogEntryAdded"])
        self.assertFalse(admission["liveQualified"])
        self.assertEqual(admission["executableModes"], [])
        self.assertEqual(
            set(admission["existingStandardAdaptersPreserved"]),
            {
                "Wan22Pipeline:text_to_video",
                "WanImageToVideoPipeline:image_to_video",
            },
        )
        for repository in self.review["repositories"]:
            pin = catalog_repository_pin(repository["repository"])
            self.assertIsNotNone(pin)
            self.assertEqual(pin["revision"], repository["revision"])
            self.assertEqual(pin["license"], "apache-2.0")

    def test_exact_contracts_retain_upstream_required_inputs_and_defaults(self):
        for reviewed in self.review["modularContracts"]:
            contract = reviewed_modular_workflow_contract(reviewed["pipelineClass"])
            self.assertEqual(contract["blocksClass"], reviewed["blocksClass"])
            self.assertEqual(len(contract["workflows"]), 1)
            workflow = contract["workflows"][0]
            self.assertEqual(
                {
                    "id": workflow["id"],
                    "taskId": workflow["taskId"],
                    "requiredInputs": workflow["requiredInputs"],
                },
                reviewed["workflow"],
            )
            defaults = {item["name"]: item["default"] for item in workflow["inputs"]}
            self.assertEqual(defaults["num_inference_steps"], reviewed["defaultNumInferenceSteps"])
            self.assertIn("videos", {item["name"] for item in workflow["outputs"]})
            if reviewed["pipelineClass"] == "Wan22Image2VideoModularPipeline":
                self.assertEqual(
                    (defaults["height"], defaults["width"], defaults["num_frames"]),
                    (
                        reviewed["defaultHeight"],
                        reviewed["defaultWidth"],
                        reviewed["defaultNumFrames"],
                    ),
                )
            else:
                self.assertEqual(
                    (defaults["height"], defaults["width"], defaults["num_frames"]),
                    (None, None, None),
                )

    def test_exact_dual_expert_receipts_cover_all_safe_weights(self):
        common = self.review["commonWeightFiles"]
        common_bytes = sum(item["byteSize"] for item in common)
        self.assertEqual((len(common), common_bytes), (4, 11869443100))
        self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in common))

        expected_totals = {
            "textToVideoA14B": (57154077760, 126177598620),
            "imageToVideoA14B": (57155716160, 126180875420),
        }
        for repository in self.review["repositories"]:
            shards = repository["expertShardPairs"]
            self.assertEqual([item["shard"] for item in shards], [f"{i:05d}" for i in range(1, 13)])
            self.assertEqual(repository["expertWeightBytesEach"], sum(item["byteSize"] for item in shards))
            self.assertEqual(repository["weightFileCount"], len(common) + 2 * len(shards))
            self.assertEqual(
                repository["weightBytes"],
                common_bytes + 2 * sum(item["byteSize"] for item in shards),
            )
            self.assertEqual(
                (repository["expertWeightBytesEach"], repository["weightBytes"]),
                expected_totals[repository["role"]],
            )
            for item in shards:
                self.assertRegex(item["highNoiseSha256"], SHA256)
                self.assertRegex(item["lowNoiseSha256"], SHA256)

    def test_standard_index_fallback_and_publisher_recipes_are_explicit(self):
        repositories = {item["role"]: item for item in self.review["repositories"]}
        text = repositories["textToVideoA14B"]
        image = repositories["imageToVideoA14B"]
        self.assertEqual((text["boundaryRatio"], image["boundaryRatio"]), (0.875, 0.9))
        self.assertEqual(
            (text["modelIndex"]["className"], text["modelIndex"]["resolvedModularPipelineClass"]),
            ("WanPipeline", "Wan22ModularPipeline"),
        )
        self.assertEqual(
            (image["modelIndex"]["className"], image["modelIndex"]["resolvedModularPipelineClass"]),
            ("WanImageToVideoPipeline", "Wan22Image2VideoModularPipeline"),
        )
        self.assertTrue(all(not item["modelIndex"]["modularIndexPresent"] for item in repositories.values()))

        recipes = self.review["reviewedRecipes"]
        self.assertEqual(
            (
                recipes["textToVideo"]["numFrames"],
                recipes["textToVideo"]["numInferenceSteps"],
                recipes["textToVideo"]["guidanceScale"],
                recipes["textToVideo"]["guidanceScale2"],
            ),
            (81, 40, 4.0, 3.0),
        )
        self.assertEqual(
            (
                recipes["imageToVideo"]["maxArea"],
                recipes["imageToVideo"]["guidanceScale"],
                recipes["imageToVideo"]["guidanceScale2"],
            ),
            (480 * 832, 3.5, "inherits_guidance_scale"),
        )
        self.assertEqual(
            self.review["remoteResourceEnvelope"]["status"],
            "estimate_only_qualification_pending",
        )


if __name__ == "__main__":
    unittest.main()
