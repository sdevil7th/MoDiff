import hashlib
import json
import re
import unittest
from pathlib import Path

from modiff.model_artifact_catalog import catalog_repository_pin


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data" / "mochi-artifact-review.json"
GRAPH_PATH = ROOT / "data" / "graphs" / "studio" / "mochi-pipeline" / "text-to-video.json"
MANIFEST_PATH = ROOT / "data" / "workflow-library-manifest.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class MochiArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_exact_safe_repository_is_admitted_remote_only(self):
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "source_complete_execution_pending")
        self.assertTrue(admission["runtimeCatalogExposed"])
        self.assertTrue(admission["downloadCatalogExposed"])
        self.assertTrue(admission["remoteOnly"])
        self.assertFalse(admission["liveQualified"])
        self.assertFalse(admission["autoEligible"])
        self.assertFalse(admission["galleryEligible"])

        repository = self.review["repository"]
        self.assertRegex(repository["revision"], SHA1)
        self.assertFalse(repository["gated"])
        self.assertFalse(repository["private"])
        self.assertFalse(repository["pythonFilesPresent"])
        self.assertFalse(repository["trustRemoteCodeRequired"])
        pin = catalog_repository_pin(repository["repository"])
        self.assertIsNotNone(pin)
        self.assertEqual(pin["revision"], repository["revision"])
        self.assertEqual(pin["license"], "apache-2.0")

    def test_eight_file_selected_safetensors_inventory_is_exact(self):
        files = sorted(self.review["weightFiles"], key=lambda item: item["path"])
        repository = self.review["repository"]
        self.assertEqual(repository["weightFileCount"], 8)
        self.assertEqual(repository["weightBytes"], 40024303350)
        self.assertEqual(sum(item["byteSize"] for item in files), repository["weightBytes"])
        canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(repository["weightInventorySha256"], hashlib.sha256(canonical).hexdigest())
        for item in files:
            self.assertTrue(item["path"].endswith(".safetensors"))
            self.assertRegex(item["sha256"], SHA256)

    def test_duplicate_and_full_precision_partitions_are_explicitly_excluded(self):
        partitions = {item["id"]: item for item in self.review["excludedArtifactPartitions"]}
        self.assertEqual(
            set(partitions),
            {
                "original-flat-format",
                "unindexed-two-shard-text-encoder",
                "default-float32-transformer",
                "default-float32-vae",
            },
        )
        self.assertEqual(partitions["unindexed-two-shard-text-encoder"]["fileCount"], 2)
        self.assertEqual(partitions["default-float32-transformer"]["fileCount"], 5)
        for partition in partitions.values():
            self.assertEqual(partition["fileCount"], len(partition["files"]))
            self.assertEqual(partition["byteSize"], sum(item["byteSize"] for item in partition["files"]))
            self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in partition["files"]))

    def test_indexed_loader_and_native_bounded_recipe_are_explicit(self):
        loader = self.review["loaderPartitionContract"]
        self.assertEqual(loader["pipelineVariant"], "bf16")
        self.assertTrue(loader["preloadIndexedTextEncoder"])
        self.assertEqual(loader["textEncoderIndexFileCount"], 4)
        self.assertTrue(loader["useSafetensors"])

        contract = self.review["pipelineContract"]
        self.assertEqual(contract["modes"], ["text_to_video"])
        self.assertEqual(
            (
                contract["admittedWidth"],
                contract["admittedHeight"],
                contract["admittedNumFrames"],
                contract["admittedNumInferenceSteps"],
                contract["admittedGuidanceScale"],
                contract["maxSequenceLength"],
                contract["documentedFramesPerSecond"],
            ),
            (848, 480, 31, 64, 4.5, 256, 30),
        )
        self.assertEqual(contract["numFramesCongruence"], "num_frames = 6n + 1")
        self.assertTrue(contract["vaeTilingRequired"])
        self.assertFalse(contract["safetyCheckerPresent"])
        self.assertEqual(contract["modelCardDiffusersNumFrames"], 84)

    def test_package_sources_mps_workaround_and_resources_are_sealed(self):
        pinned = self.review["pinnedDiffusers"]
        self.assertEqual(pinned["pipelineClass"], "MochiPipeline")
        self.assertEqual(pinned["pipelineOutputClass"], "MochiPipelineOutput")
        self.assertEqual(pinned["autoencoderClass"], "AutoencoderKLMochi")
        self.assertEqual(pinned["transformerClass"], "MochiTransformer3DModel")
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        self.assertTrue(pinned["mpsLatentDtypeCorrectionPresent"])
        for field in ("pipelineSha256", "pipelineOutputSha256", "transformerSha256", "autoencoderSha256"):
            self.assertRegex(pinned[field], SHA256)
        self.assertTrue(all(SHA256.fullmatch(digest) for digest in self.review["metadataSha256"].values()))

        envelope = self.review["remoteResourceEnvelope"]
        self.assertEqual(envelope["status"], "upstream_reported_modiff_qualification_pending")
        self.assertEqual(envelope["upstreamReportedVramGB"]["diffusersBfloat16Variant"], 22)
        self.assertIn("physical_macos_execution", self.review["admission"]["unresolvedGates"])

    def test_license_and_portable_graph_are_not_overstated(self):
        license_review = self.review["license"]
        self.assertEqual(license_review["id"], "apache-2.0")
        self.assertFalse(license_review["licenseFilePresent"])

        graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        nodes = {node["data"].get("studioRole"): node for node in graph["nodes"]}
        loader_params = nodes["wanPipeline"]["data"]["params"]
        action_params = nodes["wanGenerate"]["data"]["params"]
        recipe_params = nodes["diffusersRecipe"]["data"]["params"]
        self.assertEqual(loader_params["pipeline_class"]["value"], "MochiPipeline")
        self.assertEqual(loader_params["model_id"]["value"], {"source": "hub", "value": "genmo/mochi-1-preview"})
        self.assertEqual(loader_params["revision"]["value"], self.review["repository"]["revision"])
        self.assertEqual(loader_params["dtype"]["value"], "bfloat16")
        self.assertEqual(recipe_params["offload_mode"]["value"], "sequential_cpu")
        self.assertTrue(recipe_params["vae_slicing"]["value"])
        self.assertFalse(recipe_params["vae_tiling"]["value"])
        self.assertEqual(
            {
                field: action_params[field]["value"]
                for field in (
                    "width",
                    "height",
                    "num_frames",
                    "num_inference_steps",
                    "guidance_scale",
                    "max_sequence_length",
                )
            },
            {
                "width": 848,
                "height": 480,
                "num_frames": 31,
                "num_inference_steps": 64,
                "guidance_scale": 4.5,
                "max_sequence_length": 256,
            },
        )

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        entry = next(workflow for workflow in manifest["workflows"] if workflow["id"] == "MochiPipeline:text_to_video")
        self.assertEqual(entry["requiredArtifacts"], ["genmo/mochi-1-preview"])
        self.assertEqual(entry["runtimeQualificationStatus"], "unqualified")


if __name__ == "__main__":
    unittest.main()
