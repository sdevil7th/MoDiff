import hashlib
import json
import re
import unittest
from pathlib import Path

from modiff.model_artifact_catalog import catalog_repository_pin


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data" / "latte-artifact-review.json"
GRAPH_PATH = ROOT / "data" / "graphs" / "studio" / "latte-pipeline" / "text-to-video.json"
MANIFEST_PATH = ROOT / "data" / "workflow-library-manifest.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LatteArtifactReviewTests(unittest.TestCase):
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

    def test_six_file_safetensors_inventory_is_exact(self):
        files = sorted(self.review["weightFiles"], key=lambda item: item["path"])
        repository = self.review["repository"]
        self.assertEqual(repository["weightFileCount"], 6)
        self.assertEqual(repository["weightBytes"], 23614979636)
        self.assertEqual(sum(item["byteSize"] for item in files), repository["weightBytes"])
        canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(repository["weightInventorySha256"], hashlib.sha256(canonical).hexdigest())
        for item in files:
            self.assertTrue(item["path"].endswith(".safetensors"))
            self.assertRegex(item["sha256"], SHA256)

    def test_unsafe_and_unreferenced_artifacts_are_excluded(self):
        excluded = {item["path"]: item for item in self.review["excludedArtifacts"]}
        self.assertEqual(set(excluded), {
            "t2v_v20240523.pt",
            "vae_temporal_decoder/diffusion_pytorch_model.safetensors",
        })
        self.assertIn("unsafe legacy PyTorch serialization", excluded["t2v_v20240523.pt"]["reason"])
        self.assertIn("not referenced", excluded["vae_temporal_decoder/diffusion_pytorch_model.safetensors"]["reason"])
        self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in excluded.values()))

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
            (512, 512, 16, 50, 7.5, 120, 8),
        )
        self.assertEqual(contract["torchDtype"], "float16")
        self.assertEqual(contract["decodeChunkSize"], 14)
        self.assertTrue(contract["maskFeature"])
        self.assertTrue(contract["enableTemporalAttentions"])
        self.assertFalse(contract["cleanCaption"])
        self.assertFalse(contract["safetyCheckerPresent"])
        self.assertEqual(contract["signatureAndDocstringDefaultDiscrepancies"]["numInferenceSteps"], {
            "docstring": 100,
            "signature": 50,
        })

        pinned = self.review["pinnedDiffusers"]
        self.assertEqual(pinned["pipelineClass"], "LattePipeline")
        self.assertEqual(pinned["pipelineOutputClass"], "LattePipelineOutput")
        self.assertEqual(pinned["autoencoderClass"], "AutoencoderKL")
        self.assertEqual(pinned["transformerClass"], "LatteTransformer3DModel")
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        for field in ("pipelineSha256", "transformerSha256", "autoencoderSha256"):
            self.assertRegex(pinned[field], SHA256)
        self.assertTrue(all(SHA256.fullmatch(digest) for digest in self.review["metadataSha256"].values()))

    def test_license_and_resource_evidence_are_not_overstated(self):
        license_review = self.review["license"]
        self.assertEqual(license_review["id"], "apache-2.0")
        self.assertFalse(license_review["licenseFilePresent"])
        envelope = self.review["remoteResourceEnvelope"]
        self.assertEqual(envelope["status"], "estimated_modiff_qualification_pending")
        self.assertGreater(envelope["minimumSelectiveDiskBytes"], self.review["repository"]["weightBytes"])
        self.assertIn("physical_macos_execution", self.review["admission"]["unresolvedGates"])

    def test_portable_graph_uses_only_the_exact_latte_contract(self):
        graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        nodes = {node["data"].get("studioRole"): node for node in graph["nodes"]}
        loader_params = nodes["wanPipeline"]["data"]["params"]
        action_params = nodes["wanGenerate"]["data"]["params"]
        recipe_params = nodes["diffusersRecipe"]["data"]["params"]
        self.assertEqual(loader_params["pipeline_class"]["value"], "LattePipeline")
        self.assertEqual(loader_params["model_id"]["value"], {"source": "hub", "value": "maxin-cn/Latte-1"})
        self.assertEqual(loader_params["revision"]["value"], self.review["repository"]["revision"])
        self.assertEqual(loader_params["dtype"]["value"], "float16")
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
                "width": 512,
                "height": 512,
                "num_frames": 16,
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
                "max_sequence_length": 120,
            },
        )

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        entry = next(workflow for workflow in manifest["workflows"] if workflow["id"] == "LattePipeline:text_to_video")
        self.assertEqual(entry["requiredArtifacts"], ["maxin-cn/Latte-1"])
        self.assertEqual(entry["runtimeQualificationStatus"], "unqualified")


if __name__ == "__main__":
    unittest.main()
