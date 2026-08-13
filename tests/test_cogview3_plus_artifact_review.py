import hashlib
import json
import re
import unittest
from pathlib import Path

from modiff.model_artifact_catalog import catalog_repository_pin


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "cogview3-plus-3b-artifact-review.json"
GRAPH_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "graphs"
    / "studio"
    / "cog-view3-plus-pipeline"
    / "text-to-image.json"
)
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "workflow-library-manifest.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CogView3PlusArtifactReviewTests(unittest.TestCase):
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

    def test_seven_file_bfloat16_safetensors_inventory_is_exact(self):
        files = sorted(self.review["weightFiles"], key=lambda item: item["path"])
        repository = self.review["repository"]
        self.assertEqual(repository["weightFileCount"], 7)
        self.assertEqual(repository["weightBytes"], 25559227422)
        self.assertEqual(sum(item["byteSize"] for item in files), 25559227422)
        canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(repository["weightInventorySha256"], hashlib.sha256(canonical).hexdigest())
        self.assertEqual(repository["safetensorsDtypeParameterCounts"], {"BF16": 2848836672})
        for item in files:
            self.assertTrue(item["path"].endswith(".safetensors"))
            self.assertRegex(item["sha256"], SHA256)

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
            (1024, 1024, 50, 7.0, 224),
        )
        self.assertEqual(
            (
                contract["minimumMoDiffOutputSide"],
                contract["maximumMoDiffOutputSide"],
                contract["outputSideIncrement"],
            ),
            (512, 2048, 32),
        )
        self.assertTrue(contract["vaeSlicing"])
        self.assertTrue(contract["vaeTiling"])
        pinned = self.review["pinnedDiffusers"]
        self.assertEqual(pinned["pipelineClass"], "CogView3PlusPipeline")
        self.assertEqual(pinned["transformerClass"], "CogView3PlusTransformer2DModel")
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        for field in ("pipelineSha256", "pipelineOutputSha256", "transformerSha256"):
            self.assertRegex(pinned[field], SHA256)
        self.assertTrue(
            all(SHA256.fullmatch(digest) for digest in self.review["metadataSha256"].values())
        )

    def test_license_and_resource_evidence_are_not_overstated(self):
        license_review = self.review["license"]
        self.assertEqual(license_review["id"], "apache-2.0")
        self.assertFalse(license_review["declaredLinkAvailableAtRevision"])
        self.assertFalse(license_review["licenseFilePresent"])
        envelope = self.review["remoteResourceEnvelope"]
        self.assertEqual(envelope["status"], "upstream_measured_moDiff_qualification_pending")
        self.assertGreater(envelope["minimumSelectiveDiskBytes"], self.review["repository"]["weightBytes"])
        self.assertIn("physical_macos_execution", self.review["admission"]["unresolvedGates"])

    def test_portable_graph_uses_only_the_exact_cogview3_contract(self):
        graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        nodes = {node["data"].get("studioRole"): node for node in graph["nodes"]}
        loader_params = nodes["diffusersImagePipeline"]["data"]["params"]
        action_params = nodes["diffusersImageGenerate"]["data"]["params"]
        self.assertEqual(loader_params["pipeline_class"]["value"], "CogView3PlusPipeline")
        self.assertEqual(loader_params["conditioning_kind"]["value"], "none")
        for field in ("width", "height"):
            self.assertEqual(
                {key: action_params[field][key] for key in ("min", "max", "step")},
                {"min": 512, "max": 2048, "step": 32},
            )
        self.assertEqual(
            action_params["image_contract"]["value"]["pipelineClass"],
            "CogView3PlusPipeline",
        )

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        entry = next(
            workflow
            for workflow in manifest["workflows"]
            if workflow["id"] == "CogView3PlusPipeline:text_to_image"
        )
        self.assertEqual(entry["requiredArtifacts"], ["zai-org/CogView3-Plus-3B"])
        self.assertEqual(entry["runtimeQualificationStatus"], "unqualified")


if __name__ == "__main__":
    unittest.main()
