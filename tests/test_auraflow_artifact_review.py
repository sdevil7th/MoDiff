import hashlib
import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.studio_execution_specs import (
    AURAFLOW_V03_DIFFUSERS_FILES,
    studio_capability_definitions,
)


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "auraflow-v0.3-artifact-review.json"
GRAPH_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "graphs"
    / "studio"
    / "aura-flow-pipeline"
    / "text-to-image.json"
)
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "workflow-library-manifest.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AuraFlowArtifactReviewTests(unittest.TestCase):
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

    def test_four_file_fp16_safetensors_inventory_is_exact(self):
        files = sorted(self.review["weightFiles"], key=lambda item: item["path"])
        repository = self.review["repository"]
        self.assertEqual(repository["weightFileCount"], 4)
        self.assertEqual(repository["weightBytes"], 16835036374)
        self.assertEqual(sum(item["byteSize"] for item in files), 16835036374)
        canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(repository["weightInventorySha256"], hashlib.sha256(canonical).hexdigest())
        for item in files:
            self.assertTrue(item["path"].endswith(".safetensors"))
            self.assertIn("fp16", item["path"])
            self.assertRegex(item["sha256"], SHA256)

    def test_app_download_selection_is_runnable_and_excludes_duplicate_weights(self):
        capability = studio_capability_definitions()["AuraFlowPipeline"]
        self.assertEqual(capability["downloadFiles"], AURAFLOW_V03_DIFFUSERS_FILES)
        self.assertEqual(len(AURAFLOW_V03_DIFFUSERS_FILES), 18)
        selected = set(AURAFLOW_V03_DIFFUSERS_FILES)
        reviewed_weights = {item["path"] for item in self.review["weightFiles"]}
        self.assertTrue(reviewed_weights.issubset(selected))
        self.assertIn(
            "transformer/diffusion_pytorch_model.safetensors.fp16.index.json",
            selected,
        )
        self.assertNotIn("aura_flow_0.3.safetensors", selected)
        self.assertNotIn("text_encoder/model.safetensors", selected)
        self.assertNotIn("transformer/diffusion_pytorch_model.safetensors.index.json", selected)
        self.assertNotIn("vae/diffusion_pytorch_model.safetensors", selected)

    def test_bounded_package_owned_native_recipe_is_explicit(self):
        contract = self.review["pipelineContract"]
        self.assertEqual(contract["modes"], ["text_to_image"])
        self.assertEqual(
            (
                contract["defaultWidth"],
                contract["defaultHeight"],
                contract["defaultNumInferenceSteps"],
                contract["defaultGuidanceScale"],
                contract["maxSequenceLength"],
            ),
            (1536, 768, 50, 3.5, 256),
        )
        self.assertEqual(contract["maximumMoDiffOutputSide"], 1536)
        self.assertEqual(contract["maximumMoDiffInferenceSteps"], 50)
        pinned = self.review["pinnedDiffusers"]
        self.assertEqual(pinned["pipelineClass"], "AuraFlowPipeline")
        self.assertEqual(pinned["transformerClass"], "AuraFlowTransformer2DModel")
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        self.assertRegex(pinned["pipelineSha256"], SHA256)
        self.assertRegex(pinned["transformerSha256"], SHA256)
        self.assertTrue(
            all(SHA256.fullmatch(digest) for digest in self.review["metadataSha256"].values())
        )

    def test_resource_claim_remains_estimate_only(self):
        envelope = self.review["remoteResourceEnvelope"]
        self.assertEqual(envelope["status"], "estimate_only_qualification_pending")
        self.assertGreater(envelope["minimumSelectiveDiskBytes"], self.review["repository"]["weightBytes"])
        self.assertIn("physical_macos_execution", self.review["admission"]["unresolvedGates"])

    def test_portable_graph_uses_only_the_exact_auraflow_contract(self):
        graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        nodes = {node["data"].get("studioRole"): node for node in graph["nodes"]}
        loader_params = nodes["diffusersImagePipeline"]["data"]["params"]
        action_params = nodes["diffusersImageGenerate"]["data"]["params"]
        self.assertEqual(loader_params["pipeline_class"]["value"], "AuraFlowPipeline")
        self.assertEqual(loader_params["conditioning_kind"]["value"], "none")
        self.assertEqual(loader_params["conditioning_model_id"]["value"], "")
        self.assertEqual(loader_params["conditioning_revision"]["value"], "")
        self.assertEqual(action_params["width"]["max"], 1536)
        self.assertEqual(action_params["height"]["max"], 1536)
        self.assertEqual(action_params["image_contract"]["value"]["pipelineClass"], "AuraFlowPipeline")

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        entry = next(
            workflow
            for workflow in manifest["workflows"]
            if workflow["id"] == "AuraFlowPipeline:text_to_image"
        )
        self.assertEqual(entry["requiredArtifacts"], ["fal/AuraFlow-v0.3"])
        self.assertEqual(entry["runtimeQualificationStatus"], "unqualified")


if __name__ == "__main__":
    unittest.main()
