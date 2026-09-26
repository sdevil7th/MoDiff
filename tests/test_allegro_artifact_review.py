import hashlib
import json
import re
import unittest
from pathlib import Path

from modiff.model_artifact_catalog import catalog_repository_pin


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "allegro-artifact-review.json"
GRAPH_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "graphs"
    / "studio"
    / "allegro-pipeline"
    / "text-to-video.json"
)
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "workflow-library-manifest.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AllegroArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_exact_safe_repository_is_admitted_remote_only(self):
        self.assertEqual(self.review["schemaVersion"], 1)
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

    def test_six_file_bfloat16_safetensors_inventory_is_exact(self):
        files = sorted(self.review["weightFiles"], key=lambda item: item["path"])
        repository = self.review["repository"]
        self.assertEqual(repository["weightFileCount"], 6)
        self.assertEqual(repository["weightBytes"], 25293069108)
        self.assertEqual(sum(item["byteSize"] for item in files), 25293069108)
        canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(repository["weightInventorySha256"], hashlib.sha256(canonical).hexdigest())
        self.assertEqual(repository["safetensorsDtypeParameterCounts"], {"BF16": 2771907856})
        for item in files:
            self.assertTrue(item["path"].endswith(".safetensors"))
            self.assertRegex(item["sha256"], SHA256)

    def test_duplicate_unsafe_pytorch_shards_are_excluded(self):
        excluded = self.review["excludedUnsafeWeightFiles"]
        self.assertEqual(len(excluded), 2)
        self.assertEqual(sum(item["byteSize"] for item in excluded), 19049317384)
        for item in excluded:
            self.assertTrue(item["path"].endswith(".bin"))
            self.assertIn("duplicate unsafe PyTorch serialization", item["reason"])
            self.assertRegex(item["sha256"], SHA256)

    def test_native_bounded_recipe_and_package_sources_are_explicit(self):
        contract = self.review["pipelineContract"]
        self.assertEqual(contract["modes"], ["text_to_video"])
        self.assertEqual(
            (
                contract["defaultWidth"],
                contract["defaultHeight"],
                contract["defaultNumFrames"],
                contract["defaultNumInferenceSteps"],
                contract["defaultGuidanceScale"],
                contract["maxSequenceLength"],
                contract["documentedFramesPerSecond"],
            ),
            (1280, 720, 88, 100, 7.5, 512, 15),
        )
        self.assertEqual(contract["textAndTransformerDtype"], "bfloat16")
        self.assertEqual(contract["vaeDtype"], "float32")
        self.assertTrue(contract["vaeTiling"])
        self.assertFalse(contract["safetyCheckerPresent"])

        pinned = self.review["pinnedDiffusers"]
        self.assertEqual(pinned["pipelineClass"], "AllegroPipeline")
        self.assertEqual(pinned["autoencoderClass"], "AutoencoderKLAllegro")
        self.assertEqual(pinned["transformerClass"], "AllegroTransformer3DModel")
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        for field in ("pipelineSha256", "pipelineOutputSha256", "transformerSha256", "autoencoderSha256"):
            self.assertRegex(pinned[field], SHA256)
        self.assertTrue(all(SHA256.fullmatch(digest) for digest in self.review["metadataSha256"].values()))

    def test_license_and_resource_evidence_are_not_overstated(self):
        license_review = self.review["license"]
        self.assertEqual(license_review["id"], "apache-2.0")
        self.assertFalse(license_review["licenseFilePresent"])
        envelope = self.review["remoteResourceEnvelope"]
        self.assertEqual(envelope["status"], "upstream_measured_modiff_qualification_pending")
        self.assertEqual(envelope["upstreamReportedGpuMemoryGB"]["sequentialCpuOffload"], 9.3)
        self.assertGreater(envelope["minimumSelectiveDiskBytes"], self.review["repository"]["weightBytes"])
        self.assertIn("physical_macos_execution", self.review["admission"]["unresolvedGates"])

    def test_portable_graph_uses_only_the_exact_allegro_contract(self):
        graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        nodes = {node["data"].get("studioRole"): node for node in graph["nodes"]}
        loader_params = nodes["wanPipeline"]["data"]["params"]
        action_params = nodes["wanGenerate"]["data"]["params"]
        recipe_params = nodes["diffusersRecipe"]["data"]["params"]
        self.assertEqual(loader_params["pipeline_class"]["value"], "AllegroPipeline")
        self.assertEqual(loader_params["model_id"]["value"], {"source": "hub", "value": "rhymes-ai/Allegro"})
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
                "width": 1280,
                "height": 720,
                "num_frames": 88,
                "num_inference_steps": 100,
                "guidance_scale": 7.5,
                "max_sequence_length": 512,
            },
        )

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        entry = next(
            workflow for workflow in manifest["workflows"] if workflow["id"] == "AllegroPipeline:text_to_video"
        )
        self.assertEqual(entry["requiredArtifacts"], ["rhymes-ai/Allegro"])
        self.assertEqual(entry["runtimeQualificationStatus"], "unqualified")


if __name__ == "__main__":
    unittest.main()
