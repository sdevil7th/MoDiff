import hashlib
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from PIL import Image

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.optional_runtimes import (
    OPTIONAL_RUNTIME_PROFILES,
    TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID,
)
from modiff.studio_execution_specs import (
    STUDIO_EXECUTION_SPEC_DEFINITIONS,
    studio_execution_spec_for_pair,
)
from modules.DiffusersImage.main import (
    Edit,
    Generate,
    IMAGE_PIPELINE_ADAPTERS,
    _tag_image_pipeline,
)


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data" / "joyimage-artifact-review.json"
MANIFEST_PATH = ROOT / "data" / "workflow-library-manifest.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class JoyImageArtifactReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_two_exact_public_snapshots_are_admitted_remote_only(self):
        repositories = self.review["repositories"]
        self.assertEqual(len(repositories), 2)
        for repository in repositories:
            self.assertRegex(repository["revision"], SHA1)
            self.assertFalse(repository["private"])
            self.assertFalse(repository["gated"])
            self.assertFalse(repository["trustRemoteCodeRequired"])
            pin = catalog_repository_pin(repository["repository"])
            self.assertIsNotNone(pin)
            self.assertEqual(pin["revision"], repository["revision"])
            self.assertEqual(pin["license"], "apache-2.0")

        admission = self.review["admission"]
        self.assertEqual(admission["status"], "source_complete_execution_pending")
        self.assertTrue(admission["runtimeCatalogExposed"])
        self.assertTrue(admission["downloadCatalogExposed"])
        self.assertTrue(admission["capabilityExposed"])
        self.assertFalse(admission["liveQualified"])
        self.assertFalse(admission["autoEligible"])
        self.assertFalse(admission["galleryEligible"])
        self.assertFalse(admission["repositoryPythonExecuted"])
        self.assertEqual(admission["weightBytesDownloaded"], 0)

    def test_bfloat16_weight_inventories_are_exact(self):
        repositories = {item["role"]: item for item in self.review["repositories"]}
        expected = {
            "single_image_edit_and_text_to_image": ("edit", 50315602078),
            "multi_image_instruction_edit": ("editPlus", 50315602038),
        }
        for role, (files_key, byte_size) in expected.items():
            repository = repositories[role]
            files = sorted(self.review["weightFiles"][files_key], key=lambda item: item["path"])
            self.assertEqual(repository["weightFileCount"], 12)
            self.assertEqual(repository["weightBytes"], byte_size)
            self.assertEqual(sum(item["byteSize"] for item in files), byte_size)
            canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
            self.assertEqual(
                hashlib.sha256(canonical).hexdigest(),
                repository["weightInventorySha256"],
            )
            self.assertEqual(repository["safetensorsDtypeParameterCounts"], {"BF16": 16263675968})
            for item in files:
                self.assertTrue(item["path"].endswith(".safetensors"))
                self.assertRegex(item["sha256"], SHA256)

    def test_package_runtime_and_backend_bounds_are_explicit(self):
        pinned = self.review["pinnedRuntime"]
        self.assertEqual(
            pinned["diffusersRevision"],
            "bb56997d4b7e87f0743f26a612f49ec4e7ce7213",
        )
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        self.assertTrue(pinned["interruptAvailable"])
        self.assertFalse(pinned["repositoryPythonRequired"])
        self.assertFalse(pinned["safetyCheckerAvailable"])
        self.assertTrue(all(SHA256.fullmatch(value) for value in pinned["sourceSha256"].values()))
        self.assertTrue(all(SHA256.fullmatch(value) for value in self.review["metadataSha256"].values()))

        edit = self.review["pipelineContracts"]["edit"]
        plus = self.review["pipelineContracts"]["editPlus"]
        self.assertEqual(edit["moDiffMaximumInferenceSteps"], 40)
        self.assertEqual(edit["moDiffMaximumReferenceImages"], 1)
        self.assertEqual(plus["moDiffMaximumInferenceSteps"], 30)
        self.assertEqual(plus["moDiffMaximumReferenceImages"], 5)
        self.assertEqual(plus["moDiffImageParameter"], "images")
        for contract in (edit, plus):
            self.assertEqual(contract["moDiffMaximumSequenceLength"], 2048)
            self.assertEqual(contract["moDiffOutputBucket"]["maximumPixels"], 1024 * 1024)

        profile = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID]
        transformers = profile.packages[0]
        transformers_symbols = {"Qwen3VLForConditionalGeneration", "Qwen3VLProcessor"}
        self.assertTrue(transformers_symbols.issubset(transformers.required_symbols))
        self.assertTrue(transformers_symbols.issubset(transformers.required_class_symbols))
        diffusers_symbols = {
            "JoyImageEditPipeline",
            "JoyImageEditPlusPipeline",
            "JoyImageEditTransformer3DModel",
            "JoyImageEditPlusTransformer3DModel",
        }
        self.assertTrue(diffusers_symbols.issubset(profile.required_diffusers_symbols))

    def test_plus_adapter_maps_generic_input_to_images_parameter(self):
        adapter = IMAGE_PIPELINE_ADAPTERS["JoyImageEditPlusPipeline"]
        self.assertEqual(adapter.image_parameter, "images")
        self.assertEqual(adapter.max_reference_images, 5)

        class JoyImageEditPlusPipeline:
            pass

        pipeline = JoyImageEditPlusPipeline()
        _tag_image_pipeline(
            pipeline,
            adapter,
            "multi_image_reference_edit",
            adapter.default_repo,
            "hub",
            "c2686460c7b64d8aa11bc4d0da423fb316b33f9e",
        )
        references = [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))]
        node = Edit()
        with patch.object(Edit, "_execute_conditioned", return_value={"images": []}) as execute:
            node.execute(pipeline=pipeline, image=references, prompt="combine")
        extra = execute.call_args.args[1]
        self.assertNotIn("image", extra)
        self.assertEqual(extra["images"], references)

    def test_bucketed_text_output_reports_actual_dimensions(self):
        calls = {}

        class JoyImageEditPipeline:
            _execution_device = "cpu"

            def __call__(self, **kwargs):
                calls.update(kwargs)
                return type("Result", (), {"images": [Image.new("RGB", (1024, 768))]})()

        pipeline = JoyImageEditPipeline()
        adapter = IMAGE_PIPELINE_ADAPTERS["JoyImageEditPipeline"]
        _tag_image_pipeline(
            pipeline,
            adapter,
            "text_to_image",
            adapter.default_repo,
            "hub",
            "4b41fb25d961f37668750178ccbb380da326201c",
        )
        result = Generate().execute(
            pipeline=pipeline,
            prompt="reviewed bucket fixture",
            negative_prompt="",
            width=800,
            height=640,
            num_inference_steps=1,
            guidance_scale=4,
            max_sequence_length=2048,
            seed=7,
            output_type="pil",
        )
        self.assertEqual((calls["width"], calls["height"]), (800, 640))
        self.assertEqual((result["width_out"], result["height_out"]), (1024, 768))

    def test_expert_capabilities_and_specs_cover_four_modes(self):
        expected = {
            ("JoyImageEditPipeline", "text_to_image"),
            ("JoyImageEditPipeline", "edit_image"),
            ("JoyImageEditPlusPipeline", "edit_image"),
            ("JoyImageEditPlusPipeline", "multi_image_reference_edit"),
        }
        for model_type, mode in expected:
            specification = studio_execution_spec_for_pair(model_type, mode)
            self.assertIsNotNone(specification)
            self.assertEqual(specification["modelType"], model_type)
            self.assertEqual(specification["mode"], mode)
            definition = next(
                item
                for item in STUDIO_EXECUTION_SPEC_DEFINITIONS.values()
                if item["modelType"] == model_type and item["mode"] == mode
            )
            capability = definition["capability"]
            self.assertFalse(capability["autoEligible"])
            self.assertTrue(capability["templateEligible"])
            self.assertFalse(capability["galleryEligible"])
            self.assertEqual(capability["qualificationStatus"], "graph-qualified-execution-pending")

    def test_four_portable_graphs_pin_the_reviewed_repositories_and_bounds(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        repositories = {
            item["modelIndexClass"]: item for item in self.review["repositories"]
        }
        expected = {
            "JoyImageEditPipeline:text_to_image": ("diffusersImageGenerate", 40),
            "JoyImageEditPipeline:edit_image": ("diffusersImageEdit", 40),
            "JoyImageEditPlusPipeline:edit_image": ("diffusersImageEdit", 30),
            "JoyImageEditPlusPipeline:multi_image_reference_edit": ("diffusersImageEdit", 30),
        }
        for workflow_id, (generation_role, steps) in expected.items():
            with self.subTest(workflow=workflow_id):
                entry = next(
                    item for item in manifest["workflows"] if item["id"] == workflow_id
                )
                model_type = entry["modelType"]
                repository = repositories[model_type]
                self.assertEqual(entry["requiredArtifacts"], [repository["repository"]])
                self.assertEqual(entry["runtimeQualificationStatus"], "unqualified")
                graph = json.loads(
                    (ROOT / "data" / "graphs" / entry["graphPath"]).read_text(
                        encoding="utf-8"
                    )
                )
                nodes = {node["data"].get("studioRole"): node for node in graph["nodes"]}
                loader = nodes["diffusersImagePipeline"]["data"]["params"]
                generation = nodes[generation_role]["data"]["params"]
                self.assertEqual(loader["pipeline_class"]["value"], model_type)
                self.assertEqual(loader["revision"]["value"], repository["revision"])
                self.assertEqual(generation["width"]["value"], 1024)
                self.assertEqual(generation["height"]["value"], 1024)
                self.assertEqual(generation["num_inference_steps"]["value"], steps)
                self.assertEqual(generation["guidance_scale"]["value"], 4)
                self.assertEqual(generation["max_sequence_length"]["value"], 2048)

    def test_license_gap_and_resource_claims_remain_conservative(self):
        license_review = self.review["licenseReview"]
        self.assertEqual(license_review["declaredLicenseId"], "apache-2.0")
        self.assertTrue(license_review["commercialUsePermitted"])
        self.assertFalse(license_review["modelRepositoryLicensePayloadPresent"])
        self.assertTrue(license_review["modelRepositoryLicenseLinksBroken"])
        self.assertTrue(license_review["weightSnapshotLicenseClarificationPending"])
        self.assertRegex(license_review["upstreamProjectLicense"]["sha256"], SHA256)

        envelope = self.review["remoteResourceEnvelope"]
        self.assertEqual(envelope["status"], "exact_disk_surface_runtime_memory_unqualified")
        self.assertIsNone(envelope["minimumAcceleratorMemoryBytesEstimate"])
        self.assertIsNone(envelope["minimumSystemRamBytesEstimate"])


if __name__ == "__main__":
    unittest.main()
