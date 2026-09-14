import hashlib
import json
import re
import unittest
from pathlib import Path

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.studio_execution_specs import STUDIO_EXECUTION_SPEC_DEFINITIONS
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data" / "hidream-image-artifact-review.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HiDreamImageArtifactReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_three_public_safe_snapshots_have_exact_distinct_inventories(self):
        shared = self.review["sharedWeightFiles"]
        self.assertEqual(len(shared), 5)
        self.assertEqual(sum(item["byteSize"] for item in shared), self.review["sharedWeightBytes"])
        canonical_shared = json.dumps(
            sorted(shared, key=lambda item: item["path"]),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(hashlib.sha256(canonical_shared).hexdigest(), self.review["sharedWeightInventorySha256"])

        repositories = {item["variant"]: item for item in self.review["repositories"]}
        self.assertEqual(set(repositories), {"full", "dev", "fast"})
        for variant, repository in repositories.items():
            with self.subTest(variant=variant):
                self.assertRegex(repository["revision"], SHA1)
                self.assertFalse(repository["private"])
                self.assertFalse(repository["gated"])
                self.assertFalse(repository["pythonFilesPresent"])
                self.assertFalse(repository["trustRemoteCodeRequired"])
                self.assertFalse(repository["licenseFilePresent"])
                self.assertEqual(repository["modelIndexClass"], "HiDreamImagePipeline")
                self.assertEqual(repository["missingRequiredSubfolders"], ["text_encoder_4", "tokenizer_4"])
                self.assertEqual(len(repository["transformerWeightFiles"]), 7)
                inventory = sorted([*shared, *repository["transformerWeightFiles"]], key=lambda item: item["path"])
                self.assertEqual(len(inventory), repository["weightFileCount"])
                self.assertEqual(sum(item["byteSize"] for item in inventory), repository["weightBytes"])
                canonical = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
                self.assertEqual(hashlib.sha256(canonical).hexdigest(), repository["weightInventorySha256"])
                self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in inventory))
                for key, value in repository.items():
                    if key.endswith("Sha256"):
                        self.assertRegex(value, SHA256)

    def test_required_llama_component_is_gated_and_not_exactly_reviewable_without_acceptance(self):
        llama = self.review["requiredExternalComponent"]
        self.assertEqual(llama["repository"], "meta-llama/Llama-3.1-8B-Instruct")
        self.assertRegex(llama["revision"], SHA1)
        self.assertEqual(llama["gating"], "manual")
        self.assertEqual(llama["license"], "llama3.1")
        self.assertEqual(llama["requiredClasses"], ["LlamaForCausalLM", "PreTrainedTokenizerFast"])
        self.assertEqual(llama["safetensorsFileCount"], 4)
        self.assertEqual(llama["safetensorsBytes"], 16060556376)
        self.assertFalse(llama["weightOidsVisibleBeforeAcceptance"])
        self.assertFalse(llama["configAccessibleBeforeAcceptance"])
        self.assertEqual(llama["configHttpStatusBeforeAcceptance"], 401)
        self.assertFalse(llama["exactArtifactReviewComplete"])

    def test_package_contract_preserves_each_variant_recipe_and_cancellation_surface(self):
        pinned = self.review["pinnedRuntime"]
        self.assertEqual(pinned["pipelineClass"], "HiDreamImagePipeline")
        self.assertEqual(pinned["transformerClass"], "HiDreamImageTransformer2DModel")
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        self.assertTrue(pinned["interruptFlagAvailable"])
        self.assertFalse(pinned["safetyCheckerAvailable"])
        self.assertEqual(
            pinned["modelCpuOffloadSequence"],
            "text_encoder->text_encoder_2->text_encoder_3->text_encoder_4->transformer->vae",
        )
        for key, value in pinned.items():
            if key.endswith("Sha256"):
                self.assertRegex(value, SHA256)

        contract = self.review["pipelineContract"]
        self.assertEqual(contract["modes"], ["text_to_image"])
        self.assertEqual(contract["maxSequenceLengthDefault"], 128)
        self.assertTrue(contract["negativePromptAvailable"])
        self.assertEqual(contract["recipes"]["full"], {
            "steps": 50,
            "guidanceScale": 5.0,
            "shift": 3.0,
            "scheduler": "UniPCMultistepScheduler",
        })
        self.assertEqual(contract["recipes"]["dev"]["steps"], 28)
        self.assertEqual(contract["recipes"]["dev"]["guidanceScale"], 0.0)
        self.assertEqual(contract["recipes"]["fast"]["steps"], 16)
        self.assertEqual(contract["recipes"]["fast"]["guidanceScale"], 0.0)

    def test_blocked_review_does_not_create_runtime_or_client_facing_truth(self):
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "review_complete_blocked_incomplete_gated_composite")
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["capabilityExposed"])
        self.assertEqual(admission["canonicalWorkflowsAdded"], 0)
        self.assertFalse(admission["clientSurfaceAdded"])
        self.assertFalse(admission["autoEligible"])
        self.assertFalse(admission["galleryEligible"])
        self.assertFalse(admission["fullWeightFilesDownloaded"])
        self.assertEqual(admission["remoteWeightBytesFetched"], 0)
        self.assertGreaterEqual(len(admission["blockingReasons"]), 5)
        self.assertIn(
            "task_scoped_llama_3_1_license_acceptance_and_authenticated_artifact_review",
            admission["unresolvedGates"],
        )
        for repository in self.review["repositories"]:
            self.assertIsNone(catalog_repository_pin(repository["repository"]))
        self.assertIsNone(catalog_repository_pin(self.review["requiredExternalComponent"]["repository"]))
        self.assertNotIn("HiDreamImagePipeline", IMAGE_PIPELINE_ADAPTERS)
        self.assertFalse(
            any(
                definition["modelType"] == "HiDreamImagePipeline"
                for definition in STUDIO_EXECUTION_SPEC_DEFINITIONS.values()
            )
        )

    def test_license_and_resource_claims_remain_conservative(self):
        license_review = self.review["license"]
        self.assertTrue(license_review["gatedLlamaAcceptanceRequired"])
        self.assertFalse(license_review["acceptancePerformedByThisReview"])
        self.assertFalse(license_review["modelSnapshotLicenseFilesPresent"])
        envelope = self.review["remoteResourceEnvelope"]
        self.assertEqual(envelope["status"], "modiff_estimate_only_qualification_pending")
        self.assertEqual(
            envelope["combinedSelectiveWeightBytes"],
            envelope["publicSnapshotWeightBytes"] + envelope["requiredExternalLlamaWeightBytes"],
        )
        self.assertGreater(envelope["minimumSelectiveDiskBytes"], envelope["combinedSelectiveWeightBytes"])
        self.assertGreater(envelope["minimumSystemRamBytes"], envelope["minimumSelectiveDiskBytes"])


if __name__ == "__main__":
    unittest.main()
