import hashlib
import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin


REVIEW_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "llada2-remote-code-security-review.json"
)
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LLaDA2RemoteCodeSecurityReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_remote_code_family_remains_blocked_and_uncataloged(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "blocked_remote_code")
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["liveQualified"])
        self.assertEqual(admission["executableModes"], [])
        self.assertEqual(
            admission["unresolvedGates"],
            [
                "task_scoped_operator_authorization_for_exact_remote_code",
                "backend_owned_resource_caps_and_cancellation",
                "remote_heavy_hardware_execution",
                "physical_macos_execution",
            ],
        )
        repository = self.review["repository"]
        self.assertRegex(repository["revision"], SHA1)
        self.assertFalse(repository["gated"])
        self.assertFalse(repository["private"])
        self.assertTrue(repository["transformersAutoModelRequiresTrustRemoteCode"])
        self.assertIsNone(catalog_repository_pin(repository["repository"]))

    def test_immutable_remote_python_review_is_narrow_and_records_side_effect(self):
        review = self.review["remoteCodeReview"]
        self.assertEqual(
            review["result"],
            "no_prohibited_effect_found_but_not_safe_execution_proof",
        )
        self.assertEqual(
            review["autoMap"],
            {
                "AutoConfig": "configuration_llada2_moe.LLaDA2MoeConfig",
                "AutoModel": "modeling_llada2_moe.LLaDA2MoeModel",
                "AutoModelForCausalLM": "modeling_llada2_moe.LLaDA2MoeModelLM",
            },
        )
        self.assertTrue(
            all(count == 0 for count in review["forbiddenPrimitiveCounts"].values())
        )
        self.assertEqual(
            review["topLevelExecutableCallSideEffects"],
            ["ALL_LAYERNORM_LAYERS.append(LLaDA2MoeRMSNorm)"],
        )
        self.assertEqual(
            review["importTimeSideEffects"],
            [
                {
                    "effect": "ALL_LAYERNORM_LAYERS.append(LLaDA2MoeRMSNorm)",
                    "line": 89,
                    "scope": "process_global_transformers_registry",
                }
            ],
        )
        self.assertEqual(
            [
                (item["path"], item["byteSize"], item["lineCount"])
                for item in review["files"]
            ],
            [
                ("configuration_llada2_moe.py", 3157, 88),
                ("modeling_llada2_moe.py", 60421, 1439),
            ],
        )
        for item in review["files"]:
            self.assertRegex(item["gitBlobSha1"], SHA1)
            self.assertRegex(item["sha256"], SHA256)

    def test_exact_safetensors_inventory_and_model_shape_are_sealed(self):
        repository = self.review["repository"]
        weights = self.review["weightFiles"]
        self.assertEqual(len(weights), repository["weightFileCount"])
        self.assertEqual(sum(item["byteSize"] for item in weights), repository["weightBytes"])
        self.assertEqual(
            (repository["weightFileCount"], repository["weightBytes"]),
            (8, 32513130952),
        )
        self.assertEqual(repository["weightIndex"]["fileCount"], len(weights))
        self.assertEqual(repository["weightIndex"]["tensorCount"], 14813)
        self.assertEqual(repository["weightIndex"]["totalTensorBytes"], 32511296512)
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

        config = repository["config"]
        self.assertEqual(
            (
                config["hiddenSize"],
                config["numHiddenLayers"],
                config["numAttentionHeads"],
                config["numExperts"],
                config["expertsPerToken"],
                config["vocabSize"],
                config["maxPositionEmbeddings"],
            ),
            (2048, 20, 16, 256, 8, 157184, 32768),
        )

    def test_pinned_pipeline_still_needs_backend_owned_upper_bounds(self):
        pinned = self.review["pinnedDiffusers"]
        self.assertTrue(pinned["llada2PipelineExported"])
        self.assertEqual(pinned["pipelineClass"], "LLaDA2Pipeline")
        self.assertRegex(pinned["pipelineSourceSha256"], SHA256)
        self.assertRegex(pinned["schedulerSourceSha256"], SHA256)

        inputs = self.review["pipelineInputReview"]
        self.assertFalse(inputs["upperBoundsPresent"])
        self.assertIn("generation_length_limit", inputs["requiredFutureControls"])
        self.assertIn("cooperative_cancellation", inputs["requiredFutureControls"])


if __name__ == "__main__":
    unittest.main()
