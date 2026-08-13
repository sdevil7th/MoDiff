import hashlib
import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "chroma1-hd-artifact-review.json"
GRAPH_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "graphs"
    / "studio"
    / "chroma-pipeline"
    / "text-to-image.json"
)
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "workflow-library-manifest.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Chroma1HDArtifactReviewTests(unittest.TestCase):
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

    def test_five_file_diffusers_safetensors_inventory_is_exact(self):
        files = sorted(self.review["weightFiles"], key=lambda item: item["path"])
        repository = self.review["repository"]
        self.assertEqual(repository["selectedWeightFileCount"], 5)
        self.assertEqual(repository["selectedWeightBytes"], 27492403238)
        self.assertEqual(sum(item["byteSize"] for item in files), 27492403238)
        canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(
            repository["selectedWeightInventorySha256"],
            hashlib.sha256(canonical).hexdigest(),
        )
        for item in files:
            self.assertTrue(item["path"].endswith(".safetensors"))
            self.assertRegex(item["sha256"], SHA256)
        excluded = self.review["excludedWeightFiles"]
        self.assertEqual(len(excluded), 1)
        self.assertEqual(
            repository["fullWeightBytes"],
            repository["selectedWeightBytes"] + excluded[0]["byteSize"],
        )

    def test_bounded_package_owned_text_to_image_recipe_is_explicit(self):
        contract = self.review["pipelineContract"]
        self.assertEqual(contract["admittedModes"], ["text_to_image"])
        self.assertEqual(contract["deferredModes"], ["image_to_image"])
        self.assertEqual(
            (
                contract["defaultWidth"],
                contract["defaultHeight"],
                contract["defaultNumInferenceSteps"],
                contract["defaultGuidanceScale"],
                contract["maxSequenceLength"],
            ),
            (1024, 1024, 40, 3.0, 512),
        )
        self.assertEqual(contract["maximumMoDiffOutputSide"], 1024)
        self.assertEqual(contract["maximumMoDiffInferenceSteps"], 40)
        pinned = self.review["pinnedDiffusers"]
        self.assertEqual(pinned["pipelineClass"], "ChromaPipeline")
        self.assertEqual(pinned["transformerClass"], "ChromaTransformer2DModel")
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        for field in (
            "pipelineSha256",
            "img2imgPipelineSha256",
            "pipelineOutputSha256",
            "transformerSha256",
        ):
            self.assertRegex(pinned[field], SHA256)
        self.assertTrue(
            all(SHA256.fullmatch(digest) for digest in self.review["metadataSha256"].values())
        )

    def test_safety_and_resource_claims_remain_explicitly_unqualified(self):
        self.assertFalse(self.review["safety"]["modelCardDeclaresSafetyAlignment"])
        self.assertFalse(self.review["safety"]["pipelineSafetyCheckerPresent"])
        envelope = self.review["remoteResourceEnvelope"]
        self.assertEqual(envelope["status"], "estimate_only_qualification_pending")
        self.assertGreater(
            envelope["minimumSelectiveDiskBytes"],
            self.review["repository"]["selectedWeightBytes"],
        )
        gates = self.review["admission"]["unresolvedGates"]
        self.assertIn("live_output_safety_and_quality_review", gates)
        self.assertIn("physical_macos_execution", gates)

    def test_portable_graph_uses_only_the_exact_chroma_contract(self):
        graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        nodes = {node["data"].get("studioRole"): node for node in graph["nodes"]}
        loader_params = nodes["diffusersImagePipeline"]["data"]["params"]
        action_params = nodes["diffusersImageGenerate"]["data"]["params"]
        self.assertEqual(loader_params["pipeline_class"]["value"], "ChromaPipeline")
        self.assertEqual(loader_params["conditioning_kind"]["value"], "none")
        self.assertEqual(loader_params["conditioning_model_id"]["value"], "")
        self.assertEqual(loader_params["conditioning_revision"]["value"], "")
        self.assertEqual(action_params["width"]["max"], 1024)
        self.assertEqual(action_params["height"]["max"], 1024)
        self.assertEqual(action_params["image_contract"]["value"]["pipelineClass"], "ChromaPipeline")

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        entry = next(
            workflow
            for workflow in manifest["workflows"]
            if workflow["id"] == "ChromaPipeline:text_to_image"
        )
        self.assertEqual(entry["requiredArtifacts"], ["lodestones/Chroma1-HD"])
        self.assertEqual(entry["runtimeQualificationStatus"], "unqualified")


if __name__ == "__main__":
    unittest.main()
