import json
import re
import unittest
from pathlib import Path

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.modular_whole_workflow_contracts import reviewed_whole_workflow_graph_adapter
from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "ltx-2.5-artifact-review.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LTX25ArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_gated_artifact_stays_out_of_runtime_and_download_catalogs(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        self.assertRegex(self.review["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(self.review["format"], "safetensors")
        self.assertEqual(self.review["hub"]["gated"], "auto")
        self.assertFalse(self.review["hub"]["fileAccessAuthorized"])
        self.assertEqual(self.review["admission"]["status"], "contract_only")
        self.assertTrue(self.review["admission"]["sourceRecipesReviewed"])
        self.assertFalse(self.review["admission"]["liveQualified"])
        self.assertFalse(self.review["admission"]["runtimeCatalogExposed"])
        self.assertFalse(self.review["admission"]["downloadCatalogExposed"])
        self.assertEqual(self.review["admission"]["executableModes"], [])
        self.assertIsNone(catalog_repository_pin(self.review["repository"]))

    def test_public_weight_metadata_is_exact_but_gated_partition_selection_remains_pending(self):
        inventory = self.review["artifactInventory"]
        files = inventory["weightFiles"]
        self.assertEqual(inventory["repositoryWeightFileCount"], 31)
        self.assertEqual(inventory["repositoryWeightBytes"], 163896920128)
        self.assertEqual(len(files), 31)
        self.assertEqual(len({item["path"] for item in files}), 31)
        self.assertTrue(all(item["path"].endswith(".safetensors") for item in files))
        self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in files))
        self.assertEqual(inventory["repositoryWeightBytes"], sum(item["byteSize"] for item in files))
        self.assertFalse(inventory["indexFilesAccessible"])

        transformer_files = [item["path"] for item in files if item["path"].startswith("transformer/")]
        self.assertEqual(len(transformer_files), 12)
        self.assertEqual(sum("-of-00004.safetensors" in path for path in transformer_files), 4)
        self.assertEqual(sum("-of-00008.safetensors" in path for path in transformer_files), 8)
        self.assertEqual(
            sum(item["path"].startswith("transformer_full/") for item in files),
            4,
        )
        self.assertEqual(
            self.review["remoteResourceEnvelope"]["status"],
            "not_established_until_partition_selection",
        )

    def test_three_source_recipes_bind_exact_schedules_and_audio_video_handoffs(self):
        reviewed_revision = self.review["sourceReview"]["diffusersRevision"]
        self.assertEqual(reviewed_revision, "90b4e34e79a86ec5e7f2437634fe95ecd2108796")
        self.assertNotEqual(reviewed_revision, PINNED_DIFFUSERS_REVISION)
        self.assertEqual(self.review["admission"]["status"], "contract_only")
        recipes = self.review["recipeContracts"]
        self.assertEqual(
            set(recipes),
            {"distilledSingleStage", "fullSftWithStage2Lora", "distilledTwoStage"},
        )
        distilled = recipes["distilledSingleStage"]
        self.assertEqual(
            distilled["sigmas"],
            [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875],
        )
        self.assertFalse(distilled["numInferenceStepsSubstitutionAllowed"])
        self.assertEqual(distilled["guidance"]["guidanceScale"], 1.0)
        self.assertEqual(distilled["guidance"]["stgScale"], 0.0)

        full = recipes["fullSftWithStage2Lora"]
        self.assertEqual(full["stage1TransformerPath"], "transformer_full")
        self.assertTrue(full["stage1Scheduler"]["useDynamicShifting"])
        self.assertEqual(full["stage1Scheduler"]["shiftTerminal"], 0.1)
        self.assertEqual(full["stage2Sigmas"], [0.909375, 0.725, 0.421875])
        self.assertEqual(full["audioVideoHandoff"], "retain_stage1_audio_and_stage2_video")

        two_stage = recipes["distilledTwoStage"]
        self.assertEqual(two_stage["stage1Sigmas"], distilled["sigmas"])
        self.assertEqual(two_stage["stage2Sigmas"], [0.909375, 0.725, 0.421875])
        self.assertTrue(two_stage["reuseGeneratorAcrossStages"])
        self.assertEqual(two_stage["audioVideoHandoff"], "carry_audio_latents_into_stage2")

    def test_enhancer_duration_and_diffusion_decode_paths_are_explicit(self):
        components = self.review["componentContracts"]
        enhancer = components["promptEnhancement"]
        self.assertFalse(enhancer["implicit"])
        self.assertFalse(enhancer["discoveryDownloadAllowed"])
        self.assertFalse(enhancer["artifactShipsComponent"])
        self.assertTrue(enhancer["requiredExecutionAction"])
        self.assertEqual(enhancer["exampleModel"], "google/gemma-4-E2B-it")
        self.assertEqual(enhancer["maxNewTokens"], 600)

        duration = components["durationHead"]
        self.assertTrue(duration["automaticDurationAvailable"])
        self.assertTrue(duration["inContextRequiresExplicitNumFrames"])

        decoder = components["diffusionDecoder"]
        self.assertEqual(decoder["attentionProcessor"], "LTX2VideoVaeNeighborhoodNattenProcessor")
        self.assertTrue(decoder["requiresKernelsPackage"])
        self.assertFalse(decoder["implicitKernelDownloadAllowed"])
        self.assertTrue(decoder["explicitRuntimeSetupRequired"])
        self.assertTrue(decoder["tilingRequiredForBoundedPeakMemory"])
        self.assertFalse(decoder["denormalizeLatents"])
        self.assertEqual(components["latentOutputAudioDecode"]["outputSampleRate"], 48000)

    def test_four_workflows_have_exact_native_diffusion_decoder_graph_adapters(self):
        expected = {
            "text2video": ["text_encoder", "duration", "denoise", "decode"],
            "image2video": ["text_encoder", "duration", "vae_encoder", "denoise", "decode"],
            "condition": ["text_encoder", "duration", "condition_encoder", "denoise", "decode"],
            "in_context": ["text_encoder", "condition_encoder", "reference_encoder", "denoise", "decode"],
        }
        for workflow_id, sequence in expected.items():
            adapter = reviewed_whole_workflow_graph_adapter("LTX25ModularPipeline", workflow_id)
            self.assertEqual(adapter["adapterId"], "official_top_level_blocks")
            self.assertEqual(adapter["upstreamBlockSequence"], sequence)
            self.assertEqual(len(adapter["stateEdges"]), len(sequence) - 1)
        self.assertFalse(self.review["admission"]["runtimeCatalogExposed"])
        self.assertFalse(self.review["admission"]["downloadCatalogExposed"])


if __name__ == "__main__":
    unittest.main()
