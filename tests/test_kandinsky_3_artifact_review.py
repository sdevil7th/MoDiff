import hashlib
import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.optional_runtimes import OPTIONAL_RUNTIME_PROFILES


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data" / "kandinsky-3-artifact-review.json"
MANIFEST_PATH = ROOT / "data" / "workflow-library-manifest.json"
GRAPH_ROOT = ROOT / "data" / "graphs" / "studio" / "kandinsky3-pipeline"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Kandinsky3ArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_exact_single_stage_repository_is_admitted_remote_only(self):
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

    def test_seven_file_fp16_safetensors_inventory_is_exact(self):
        files = sorted(self.review["weightFiles"], key=lambda item: item["path"])
        repository = self.review["repository"]
        self.assertEqual(repository["weightFileCount"], 7)
        self.assertEqual(repository["weightBytes"], 28390829958)
        self.assertEqual(sum(item["byteSize"] for item in files), 28390829958)
        canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(repository["weightInventorySha256"], hashlib.sha256(canonical).hexdigest())
        for item in files:
            self.assertTrue(item["path"].endswith(".safetensors"))
            self.assertIn("fp16", item["path"])
            self.assertRegex(item["sha256"], SHA256)

    def test_package_owned_generation_and_edit_contracts_are_bounded(self):
        contracts = self.review["pipelineContracts"]
        text = contracts["Kandinsky3Pipeline"]
        edit = contracts["Kandinsky3Img2ImgPipeline"]
        self.assertEqual(text["requiredInputs"], ["prompt"])
        self.assertEqual(edit["requiredInputs"], ["prompt", "image"])
        self.assertEqual(
            (
                text["defaultWidth"],
                text["defaultHeight"],
                text["defaultNumInferenceSteps"],
                text["defaultGuidanceScale"],
                text["maximumPromptTokens"],
            ),
            (1024, 1024, 25, 3.0, 128),
        )
        self.assertEqual(edit["defaultStrength"], 0.3)
        self.assertEqual(edit["maximumMoDiffInputPixels"], 1024 * 1024)
        pinned = self.review["pinnedDiffusers"]
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        self.assertTrue(pinned["cooperativeCancellationAvailable"])
        self.assertRegex(pinned["pipelineSha256"], SHA256)
        self.assertRegex(pinned["imageToImagePipelineSha256"], SHA256)
        self.assertRegex(pinned["unetSha256"], SHA256)
        self.assertTrue(
            all(SHA256.fullmatch(digest) for digest in self.review["metadataSha256"].values())
        )

    def test_runtime_symbols_and_license_caveat_are_explicit(self):
        profile = OPTIONAL_RUNTIME_PROFILES[self.review["runtime"]["optionalProfile"]]
        transformers = next(package for package in profile.packages if package.distribution == "transformers")
        self.assertTrue(
            set(self.review["runtime"]["transformersRequiredSymbols"]).issubset(
                transformers.required_class_symbols
            )
        )
        self.assertTrue(
            set(self.review["runtime"]["diffusersRequiredSymbols"]).issubset(
                profile.required_diffusers_symbols
            )
        )
        license_review = self.review["license"]
        self.assertFalse(license_review["modelSnapshotLicenseFilePresent"])
        self.assertRegex(license_review["upstreamRevision"], SHA1)
        self.assertRegex(license_review["upstreamLicenseSha256"], SHA256)
        self.assertIn(
            "model_snapshot_license_file_clarification",
            self.review["admission"]["unresolvedGates"],
        )

    def test_resource_and_output_safety_claims_remain_pending(self):
        envelope = self.review["remoteResourceEnvelope"]
        self.assertEqual(envelope["status"], "estimate_only_qualification_pending")
        self.assertGreater(envelope["minimumSelectiveDiskBytes"], self.review["repository"]["weightBytes"])
        self.assertFalse(self.review["safety"]["packageSafetyCheckerPresent"])
        self.assertIn("physical_macos_execution", self.review["admission"]["unresolvedGates"])

    def test_portable_graphs_use_only_the_exact_kandinsky3_contract(self):
        expected = {
            "text-to-image.json": ("Kandinsky3Pipeline", "text_to_image"),
            "edit-image.json": ("Kandinsky3Img2ImgPipeline", "edit_image"),
        }
        for filename, (pipeline_class, mode) in expected.items():
            graph = json.loads((GRAPH_ROOT / filename).read_text(encoding="utf-8"))
            nodes = {node["data"].get("studioRole"): node for node in graph["nodes"]}
            loader_params = nodes["diffusersImagePipeline"]["data"]["params"]
            action_role = "diffusersImageGenerate" if mode == "text_to_image" else "diffusersImageEdit"
            action_params = nodes[action_role]["data"]["params"]
            self.assertEqual(loader_params["pipeline_class"]["value"], pipeline_class)
            self.assertEqual(
                loader_params["model_id"]["value"],
                {"source": "hub", "value": "kandinsky-community/kandinsky-3"},
            )
            self.assertEqual(loader_params["conditioning_kind"]["value"], "none")
            self.assertEqual(loader_params["conditioning_model_id"]["value"], "")
            self.assertEqual(loader_params["conditioning_revision"]["value"], "")
            self.assertEqual(action_params["image_contract"]["value"]["pipelineClass"], pipeline_class)
            self.assertEqual(action_params["max_sequence_length"]["default"], 128)
            self.assertEqual(action_params["max_sequence_length"]["max"], 128)

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        entries = {
            workflow["id"]: workflow
            for workflow in manifest["workflows"]
            if workflow["modelType"] == "Kandinsky3Pipeline"
        }
        self.assertEqual(set(entries), {"Kandinsky3Pipeline:text_to_image", "Kandinsky3Pipeline:edit_image"})
        for entry in entries.values():
            self.assertEqual(entry["requiredArtifacts"], ["kandinsky-community/kandinsky-3"])
            self.assertEqual(entry["runtimeQualificationStatus"], "unqualified")


if __name__ == "__main__":
    unittest.main()
