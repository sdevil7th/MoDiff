import hashlib
import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin


REVIEW_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "diffusiongemma-artifact-review.json"
)
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DiffusionGemmaArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_remote_heavy_family_remains_uncataloged_and_invisible(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        self.assertEqual(self.review["format"], "safetensors")
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "remote_only_reviewed_not_admitted")
        self.assertTrue(admission["remoteOnly"])
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["liveQualified"])
        self.assertFalse(admission["userVisible"])
        self.assertEqual(admission["executableModes"], [])
        self.assertIn(
            "backend_owned_bounded_diffusion_text_contract",
            admission["unresolvedGates"],
        )

        repository = self.review["repository"]
        self.assertRegex(repository["revision"], SHA1)
        self.assertFalse(repository["gated"])
        self.assertFalse(repository["private"])
        self.assertFalse(repository["pythonFilesPresent"])
        self.assertFalse(repository["trustRemoteCodeRequired"])
        self.assertIsNone(catalog_repository_pin(repository["repository"]))
        self.assertEqual(self.review["license"]["id"], "apache-2.0")
        self.assertTrue(self.review["license"]["commercialUseAllowed"])
        contract = self.review["genericContractReview"]
        self.assertFalse(contract["currentMoDiffNodePresent"])
        self.assertEqual(
            contract["candidateInputContracts"],
            ["text_to_text", "image_to_text"],
        )

    def test_exact_eleven_shard_bfloat16_inventory_is_sealed(self):
        repository = self.review["repository"]
        weights = self.review["weightFiles"]
        self.assertEqual((len(weights), sum(item["byteSize"] for item in weights)), (11, 51647701024))
        self.assertEqual(repository["weightFileCount"], len(weights))
        self.assertEqual(repository["weightBytes"], sum(item["byteSize"] for item in weights))
        self.assertEqual(repository["totalFileBytes"], 51680024015)
        self.assertEqual(
            (
                repository["weightIndex"]["shardCount"],
                repository["weightIndex"]["tensorCount"],
                repository["weightIndex"]["totalParameters"],
                repository["weightIndex"]["totalTensorBytes"],
            ),
            (11, 1047, 25823778864, 51647562456),
        )
        canonical = json.dumps(
            sorted(weights, key=lambda item: item["path"]),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(
            repository["weightInventorySha256"],
            hashlib.sha256(canonical).hexdigest(),
        )
        for item in weights:
            self.assertTrue(item["path"].endswith(".safetensors"))
            self.assertRegex(item["sha256"], SHA256)
        self.assertTrue(
            all(SHA256.fullmatch(digest) for digest in repository["metadataSha256"].values())
        )

    def test_package_owned_pipeline_and_model_contract_are_exact(self):
        runtime = self.review["pinnedRuntime"]
        self.assertEqual(
            runtime["diffusers"]["revision"],
            "bb56997d4b7e87f0743f26a612f49ec4e7ce7213",
        )
        self.assertEqual(runtime["transformers"]["version"], "5.14.1")
        self.assertTrue(
            all(
                SHA256.fullmatch(digest)
                for digest in runtime["diffusers"]["sourceSha256"].values()
            )
        )
        self.assertTrue(
            all(
                SHA256.fullmatch(digest)
                for digest in runtime["transformers"]["sourceSha256"].values()
            )
        )
        self.assertEqual(runtime["noWeightApiProbe"]["status"], "passed")

        repository = self.review["repository"]
        self.assertEqual(
            repository["modelIndex"],
            {
                "modelClass": "DiffusionGemmaForBlockDiffusion",
                "pipelineClass": "DiffusionGemmaPipeline",
                "processorClass": "Gemma4Processor",
                "schedulerClass": "BlockRefinementScheduler",
            },
        )
        self.assertEqual(
            (
                repository["config"]["canvasLength"],
                repository["config"]["numHiddenLayers"],
                repository["config"]["numExperts"],
                repository["config"]["activeExperts"],
                repository["config"]["maxPositionEmbeddings"],
            ),
            (256, 30, 128, 8, 262144),
        )

    def test_recipe_callback_and_missing_upper_bounds_are_explicit(self):
        pipeline = self.review["pipelineContract"]
        self.assertTrue(pipeline["callbackOnStepEndSupported"])
        self.assertEqual(pipeline["callbackTensorInputs"], ["canvas", "logits"])
        self.assertEqual(pipeline["requiredOneOf"], ["prompt", "messages"])
        self.assertEqual(pipeline["output"]["fields"], ["sequences", "texts"])
        self.assertEqual(
            (
                pipeline["defaults"]["generationLength"],
                pipeline["defaults"]["numInferenceStepsPerCanvas"],
                pipeline["defaults"]["temperature"],
                pipeline["defaults"]["confidenceThreshold"],
            ),
            (256, 48, 0.0, 0.005),
        )

        recipe = self.review["reviewedRecipe"]
        self.assertEqual(recipe["sampler"], "EntropyBoundScheduler")
        self.assertEqual(
            (recipe["entropyBound"], recipe["temperatureMaximum"], recipe["temperatureMinimum"]),
            (0.1, 0.8, 0.4),
        )
        bounds = self.review["upstreamBoundsReview"]
        self.assertFalse(bounds["upperBoundsPresent"])
        self.assertIn("maximum_generation_tokens", bounds["requiredFutureControls"])
        self.assertIn("cooperative_cancellation_via_step_callback", bounds["requiredFutureControls"])
        self.assertEqual(
            self.review["resourceEnvelope"]["status"],
            "estimate_only_qualification_pending",
        )


if __name__ == "__main__":
    unittest.main()
