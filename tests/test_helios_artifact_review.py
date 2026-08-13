import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.modular_workflow_discovery import reviewed_modular_workflow_contract


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "helios-artifact-review.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HeliosArtifactReviewTests(unittest.TestCase):
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
        self.assertIn("immutable_modular_component_descriptors", admission["unresolvedGates"])
        for repository in self.review["repositories"]:
            self.assertRegex(repository["revision"], r"^[0-9a-f]{40}$")
            self.assertFalse(repository["gated"])
            self.assertFalse(repository["private"])
            self.assertIsNone(catalog_repository_pin(repository["repository"]))

    def test_all_three_modular_contracts_keep_exact_generic_workflows(self):
        expected_workflows = {
            "text2video": {"taskId": "text_to_video", "requiredInputs": ["prompt"]},
            "image2video": {
                "taskId": "image_to_video",
                "requiredInputs": ["image", "prompt"],
            },
            "video2video": {
                "taskId": "video_to_video",
                "requiredInputs": ["prompt", "video"],
            },
        }
        for reviewed in self.review["modularContracts"]:
            contract = reviewed_modular_workflow_contract(reviewed["pipelineClass"])
            self.assertEqual(contract["blocksClass"], reviewed["blocksClass"])
            workflows = {workflow["id"]: workflow for workflow in contract["workflows"]}
            self.assertEqual(
                {
                    name: {
                        "taskId": workflow["taskId"],
                        "requiredInputs": workflow["requiredInputs"],
                    }
                    for name, workflow in workflows.items()
                },
                expected_workflows,
            )
            for workflow in workflows.values():
                defaults = {item["name"]: item["default"] for item in workflow["inputs"]}
                self.assertEqual((defaults["height"], defaults["width"]), (384, 640))
                self.assertEqual(defaults["num_frames"], reviewed["defaultNumFrames"])
                self.assertIn("videos", {item["name"] for item in workflow["outputs"]})
                if "defaultNumInferenceSteps" in reviewed:
                    self.assertEqual(
                        defaults["num_inference_steps"], reviewed["defaultNumInferenceSteps"]
                    )
                else:
                    self.assertEqual(
                        defaults["pyramid_num_inference_steps_list"],
                        reviewed["defaultPyramidNumInferenceSteps"],
                    )

    def test_exact_weight_receipts_separate_selected_and_unused_transformers(self):
        common = self.review["commonWeightFiles"]
        common_bytes = sum(item["byteSize"] for item in common)
        self.assertEqual((len(common), common_bytes), (6, 23231263636))
        for item in common:
            self.assertTrue(item["path"].endswith(".safetensors"))
            self.assertRegex(item["sha256"], SHA256)

        for repository in self.review["repositories"]:
            selected = repository["selectedTransformerFiles"]
            excluded = repository["excludedTransformerFiles"]
            all_files = common + selected + excluded
            self.assertEqual(repository["selectedWeightFileCount"], len(common) + len(selected))
            self.assertEqual(
                repository["selectedWeightBytes"],
                common_bytes + sum(item["byteSize"] for item in selected),
            )
            self.assertEqual(repository["fullWeightFileCount"], len(all_files))
            self.assertEqual(
                repository["fullWeightBytes"],
                sum(item["byteSize"] for item in all_files),
            )
            self.assertEqual(len(all_files), len({item["path"] for item in all_files}))
            self.assertEqual(repository["selectedWeightBytes"], 80481086028)
            self.assertEqual(repository["fullWeightBytes"], 137730908420)
            self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in all_files))

    def test_mutable_modular_descriptors_and_source_recipes_are_explicit(self):
        artifact_index = self.review["artifactIndex"]
        self.assertIsNone(artifact_index["componentDescriptorRevision"])
        self.assertEqual(
            artifact_index["componentDescriptorRevisionAppliesTo"],
            ["scheduler", "text_encoder", "tokenizer", "transformer", "vae"],
        )
        self.assertEqual(
            artifact_index["standardIndexSelectedWeightPrefixes"],
            ["text_encoder/", "transformer/", "vae/"],
        )

        base = self.review["reviewedRecipes"]["baseTextToVideo"]
        self.assertEqual(
            (base["numFrames"], base["numInferenceSteps"], base["guidanceScale"]),
            (99, 50, 5.0),
        )
        distilled = self.review["reviewedRecipes"]["distilledPyramid"]
        self.assertEqual(distilled["pyramidNumInferenceSteps"], [2, 2, 2])
        self.assertEqual((distilled["numFramesRequested"], distilled["numFramesRounded"]), (240, 264))
        self.assertEqual(distilled["guidanceScale"], 1.0)
        self.assertTrue(distilled["isAmplifyFirstChunk"])
        self.assertFalse(self.review["publisherClaims"]["acceptedAsLiveEvidence"])
        self.assertEqual(
            self.review["remoteResourceEnvelope"]["status"],
            "estimate_only_qualification_pending",
        )


if __name__ == "__main__":
    unittest.main()
