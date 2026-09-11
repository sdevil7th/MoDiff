import hashlib
import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin


REVIEW_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "pixart-sigma-artifact-review.json"
)
GRAPH_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "graphs"
    / "studio"
    / "pix-art-sigma-pipeline"
    / "text-to-image.json"
)
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "workflow-library-manifest.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PixArtSigmaArtifactReviewTests(unittest.TestCase):
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
        self.assertEqual(pin["license"], "openrail++")

    def test_four_file_safetensors_inventory_is_exact(self):
        files = sorted(self.review["weightFiles"], key=lambda item: item["path"])
        repository = self.review["repository"]
        self.assertEqual(repository["weightFileCount"], 4)
        self.assertEqual(repository["weightBytes"], 21827405446)
        self.assertEqual(sum(item["byteSize"] for item in files), 21827405446)
        canonical = json.dumps(
            files,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(
            repository["weightInventorySha256"],
            hashlib.sha256(canonical).hexdigest(),
        )
        for item in files:
            self.assertTrue(item["path"].endswith(".safetensors"))
            self.assertRegex(item["sha256"], SHA256)

    def test_bounded_package_owned_recipe_is_explicit(self):
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
            (1024, 1024, 20, 4.5, 300),
        )
        self.assertEqual(contract["maximumMoDiffInferenceSteps"], 50)
        self.assertTrue(contract["defaultUseResolutionBinning"])
        pinned = self.review["pinnedDiffusers"]
        self.assertEqual(pinned["pipelineClass"], "PixArtSigmaPipeline")
        self.assertEqual(pinned["transformerClass"], "PixArtTransformer2DModel")
        self.assertTrue(pinned["legacyPerStepCallbackAvailable"])
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

    def test_portable_graph_uses_only_the_exact_pixart_contract(self):
        graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        nodes = {node["data"].get("studioRole"): node for node in graph["nodes"]}
        loader_params = nodes["diffusersImagePipeline"]["data"]["params"]
        action_params = nodes["diffusersImageGenerate"]["data"]["params"]
        self.assertEqual(loader_params["pipeline_class"]["value"], "PixArtSigmaPipeline")
        self.assertEqual(loader_params["conditioning_kind"]["value"], "none")
        self.assertEqual(loader_params["conditioning_model_id"]["value"], "")
        self.assertEqual(loader_params["conditioning_revision"]["value"], "")
        self.assertEqual(
            action_params["image_contract"]["value"]["pipelineClass"],
            "PixArtSigmaPipeline",
        )

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        entry = next(
            workflow
            for workflow in manifest["workflows"]
            if workflow["id"] == "PixArtSigmaPipeline:text_to_image"
        )
        self.assertEqual(
            entry["requiredArtifacts"],
            ["PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"],
        )
        self.assertEqual(entry["runtimeQualificationStatus"], "unqualified")


if __name__ == "__main__":
    unittest.main()
