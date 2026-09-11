import hashlib
import json
import re
import unittest
from pathlib import Path

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.modular_workflow_discovery import reviewed_modular_workflow_contract
from modiff.studio_execution_specs import MINIMAX_MUSIC3_DIFFUSERS_FILES


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "minimax-music3-artifact-review.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class MiniMaxMusic3ArtifactReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_exact_public_repository_and_bounded_diffusers_partition_are_sealed(self):
        repository = self.review["repository"]
        self.assertEqual(repository["repository"], "MiniMaxAI/MiniMax-Music3")
        self.assertRegex(repository["revision"], SHA1)
        self.assertFalse(repository["private"])
        self.assertFalse(repository["gated"])
        self.assertEqual(repository["fullRepositoryFileCount"], 88)
        self.assertEqual(repository["fullRepositoryBytes"], 57_353_379_600)
        self.assertEqual(repository["selectedFileCount"], len(repository["selectedFiles"]))
        self.assertEqual(repository["selectedFileCount"], 23)
        self.assertEqual(repository["selectedBytes"], 28_517_608_999)
        self.assertEqual(
            repository["selectedBytes"] + repository["excludedBytes"],
            repository["fullRepositoryBytes"],
        )
        self.assertTrue(all(not path.endswith(".py") for path in repository["selectedFiles"]))
        self.assertEqual(MINIMAX_MUSIC3_DIFFUSERS_FILES, repository["selectedFiles"])
        for excluded in ("qwen_7B/", "flowmatching_vae.pth", "dav.pth", "scripts/"):
            self.assertIn(excluded, repository["excludedRuntimeIrrelevantRoots"])
        self.assertEqual(
            catalog_repository_pin(repository["repository"], model_type="MiniMaxMusic3ModularPipeline"),
            {
                "repo": repository["repository"],
                "revision": repository["revision"],
                "license": "minimax-music3-community-license",
                "kind": "base",
                "modelType": "MiniMaxMusic3ModularPipeline",
            },
        )

    def test_safetensors_inventory_uses_hub_lfs_digests(self):
        files = self.review["weightFiles"]
        ordered = sorted(files, key=lambda item: item["path"])
        canonical = json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(files, ordered)
        self.assertEqual(len(files), self.review["weightFileCount"])
        self.assertEqual(sum(item["byteSize"] for item in files), self.review["weightBytes"])
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), self.review["weightInventorySha256"])
        self.assertTrue(all(item["path"].endswith(".safetensors") for item in files))
        self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in files))

    def test_modular_index_is_self_contained_and_requires_revision_normalization(self):
        index = self.review["modularIndex"]
        self.assertEqual(index["pipelineClass"], "MiniMaxMusic3ModularPipeline")
        self.assertEqual(index["blocksClass"], "MiniMaxMusic3Blocks")
        self.assertRegex(index["sha256"], SHA256)
        self.assertTrue(index["allRemoteComponentsSelfReferenceRepository"])
        self.assertEqual(
            set(index["componentRevisions"]),
            {
                "condition_encoder",
                "language_model",
                "rvq_depth_decoder",
                "scheduler",
                "tokenizer",
                "transformer",
                "vocoder",
            },
        )
        self.assertTrue(all(value is None for value in index["componentRevisions"].values()))

    def test_pinned_upstream_workflow_matches_reviewed_runtime_contract(self):
        workflow_review = self.review["reviewedWorkflow"]
        contract = reviewed_modular_workflow_contract("MiniMaxMusic3ModularPipeline")
        self.assertEqual(contract["blocksClass"], "MiniMaxMusic3Blocks")
        self.assertEqual(len(contract["workflows"]), 1)
        workflow = contract["workflows"][0]
        self.assertEqual(workflow["id"], workflow_review["workflowId"])
        self.assertEqual(workflow["taskId"], workflow_review["taskId"])
        self.assertEqual(workflow["requiredInputs"], workflow_review["requiredInputs"])
        self.assertEqual(
            [field["name"] for field in workflow["outputs"]],
            workflow_review["reviewedInspectableOutputs"],
        )
        self.assertEqual(workflow_review["publicOutputs"], ["audios"])
        self.assertEqual(workflow_review["upstreamBlockSequence"], ["semantic_generator", "denoise", "decode"])
        self.assertEqual(workflow_review["nativeSampleRate"], 44_100)
        self.assertEqual(workflow_review["nativeChannels"], 2)
        self.assertEqual(workflow_review["maximumAudioFrames"], 9_000)

    def test_custom_license_and_admission_are_not_overstated(self):
        license_review = self.review["licenseReview"]
        self.assertEqual(license_review["id"], "MiniMax-Music3 Community License")
        self.assertIsNone(license_review["modelCardLicenseTag"])
        self.assertTrue(license_review["embeddedLicenseFilePresent"])
        self.assertTrue(license_review["revisionBoundAcknowledgementRequired"])
        self.assertEqual(license_review["annualRevenueAuthorizationThresholdUsdExclusive"], 20_000_000)
        self.assertTrue(license_review["hostedGenerationSafeguardsRequired"])
        self.assertRegex(license_review["licenseFileSha256"], SHA256)

        admission = self.review["admission"]
        self.assertEqual(admission["status"], "artifact_reviewed_runner_pending")
        self.assertEqual(admission["executableModes"], [])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["liveQualified"])


if __name__ == "__main__":
    unittest.main()
