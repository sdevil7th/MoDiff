import hashlib
import json
from pathlib import Path
import re
import unittest

from PIL import Image

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.optional_runtimes import OPTIONAL_RUNTIME_PROFILES, TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID
from modiff.studio_execution_specs import studio_capability_definitions
from modules.DiffusersImage.main import (
    IMAGE_PIPELINE_ADAPTERS,
    prepare_reference_images,
    prepare_reference_prompt,
)


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data" / "omnigen-artifact-review.json"
MANIFEST_PATH = ROOT / "data" / "workflow-library-manifest.json"
GRAPH_ROOT = ROOT / "data" / "graphs" / "studio"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class OmniGenArtifactReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_exact_public_snapshot_is_admitted_remote_only(self):
        repository = self.review["repository"]
        self.assertEqual(repository["repository"], "Shitao/OmniGen-v1-diffusers")
        self.assertRegex(repository["revision"], SHA1)
        self.assertFalse(repository["private"])
        self.assertFalse(repository["gated"])
        self.assertFalse(repository["pythonFilesPresent"])
        self.assertFalse(repository["trustRemoteCodeRequired"])
        self.assertEqual(repository["modelIndexClass"], "OmniGenPipeline")
        pin = catalog_repository_pin(repository["repository"], model_type="OmniGenPipeline")
        self.assertIsNotNone(pin)
        self.assertEqual(pin["revision"], repository["revision"])
        self.assertEqual(pin["license"], "mit")

        admission = self.review["admission"]
        self.assertEqual(admission["status"], "source_complete_execution_pending")
        self.assertTrue(admission["runtimeCatalogExposed"])
        self.assertTrue(admission["downloadCatalogExposed"])
        self.assertTrue(admission["capabilityExposed"])
        self.assertTrue(admission["remoteOnly"])
        self.assertFalse(admission["liveQualified"])
        self.assertFalse(admission["autoEligible"])
        self.assertFalse(admission["galleryEligible"])

    def test_exact_safetensors_inventory_is_sealed(self):
        repository = self.review["repository"]
        files = sorted(self.review["weightFiles"], key=lambda item: item["path"])
        self.assertEqual(len(files), repository["weightFileCount"])
        self.assertEqual(sum(item["byteSize"] for item in files), repository["weightBytes"])
        canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), repository["weightInventorySha256"])
        self.assertTrue(all(item["path"].endswith(".safetensors") for item in files))
        self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in files))

    def test_backend_owns_multimodal_placeholders_and_generation_bounds(self):
        adapter = IMAGE_PIPELINE_ADAPTERS["OmniGenPipeline"]
        self.assertEqual(
            adapter.modes,
            frozenset({"text_to_image", "edit_image", "multi_image_reference_edit"}),
        )
        self.assertTrue(adapter.safe_serialization_required)
        self.assertEqual(adapter.max_inference_steps, 50)
        self.assertEqual(adapter.max_output_pixels, 1024 * 1024)
        self.assertEqual(adapter.image_parameter, "input_images")
        self.assertEqual(adapter.multi_image_strategy, "always_list")
        self.assertTrue(adapter.reference_prompt_placeholders)
        self.assertEqual(adapter.max_input_image_size, 1024)
        self.assertEqual(adapter.max_reference_images, 3)
        self.assertEqual(adapter.default_image_guidance_scale, 1.6)
        self.assertEqual(adapter.image_guidance_parameter, "img_guidance_scale")

        images = [Image.new("RGB", (16, 16), "black"), Image.new("RGB", (16, 16), "white")]
        prepared = prepare_reference_images(tuple(images), adapter)
        self.assertEqual(prepared, images)
        self.assertEqual(
            prepare_reference_prompt("combine them", prepared, adapter),
            "<img><|image_1|></img> <img><|image_2|></img> combine them",
        )
        with self.assertRaisesRegex(ValueError, "reserved image-placeholder syntax"):
            prepare_reference_prompt("reuse <|image_1|>", prepared, adapter)
        with self.assertRaisesRegex(ValueError, "one string prompt"):
            prepare_reference_prompt(["one", "two"], prepared, adapter)

        class OmniGenCall:
            def __call__(
                self,
                prompt=None,
                input_images=None,
                guidance_scale=2.5,
                img_guidance_scale=1.6,
                max_input_image_size=1024,
            ):
                return None

        target = {}
        adapter.apply_generation_parameters(
            OmniGenCall(),
            {"guidance_scale": 2.5, "image_guidance_scale": 1.6},
            target,
        )
        self.assertEqual(
            target,
            {"guidance_scale": 2.5, "img_guidance_scale": 1.6, "max_input_image_size": 1024},
        )

    def test_runtime_callback_symbols_and_license_receipts_are_explicit(self):
        runtime = self.review["pinnedRuntime"]
        profile = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID]
        transformers = next(package for package in profile.packages if package.distribution == "transformers")
        self.assertTrue(set(runtime["requiredDiffusersSymbols"]).issubset(profile.required_diffusers_symbols))
        self.assertTrue(set(runtime["requiredTransformersSymbols"]).issubset(transformers.required_class_symbols))
        safety = self.review["cancellationAndSafety"]
        self.assertTrue(safety["callbackOnStepEndAvailable"])
        self.assertTrue(safety["serverStopPropagatesThroughCallback"])
        self.assertFalse(safety["interruptFlagReadByDenoisingLoop"])
        self.assertFalse(safety["packageSafetyCheckerPresent"])
        self.assertTrue(safety["reservedPromptSyntaxRejected"])
        license_review = self.review["license"]["upstream"]
        self.assertRegex(license_review["revision"], SHA1)
        self.assertRegex(license_review["licenseSha256"], SHA256)

    def test_download_is_app_reserved_without_deleting_models(self):
        queue = self.review["appDownloadQueue"]
        self.assertTrue(queue["submittedThroughAppOnly"])
        self.assertFalse(queue["directWeightDownloadPerformed"])
        self.assertFalse(queue["olderModelsDeleted"])
        expected_headroom = (
            queue["preflightFreeBytes"]
            - queue["preflightExistingQueuedReservationBytes"]
            - queue["snapshotReservationBytes"]
            - queue["reserveBytes"]
        )
        self.assertTrue(queue["aggregateFits"])
        self.assertEqual(queue["aggregateHeadroomAfterReserveBytes"], expected_headroom)
        self.assertGreater(expected_headroom, 0)

    def test_portable_graphs_pin_all_three_modes(self):
        expected = {
            "OmniGenPipeline:text_to_image": "omni-gen-pipeline/text-to-image.json",
            "OmniGenPipeline:edit_image": "omni-gen-pipeline/edit-image.json",
            "OmniGenPipeline:multi_image_reference_edit": (
                "omni-gen-pipeline/multi-image-reference-edit.json"
            ),
        }
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        by_id = {workflow["id"]: workflow for workflow in manifest["workflows"]}
        for workflow_id, path in expected.items():
            workflow = by_id[workflow_id]
            graph = json.loads((GRAPH_ROOT / path).read_text(encoding="utf-8"))
            nodes = {node["data"].get("studioRole"): node for node in graph["nodes"]}
            loader = nodes["diffusersImagePipeline"]["data"]["params"]
            self.assertEqual(loader["pipeline_class"]["value"], "OmniGenPipeline")
            self.assertEqual(
                loader["model_id"]["value"],
                {"source": "hub", "value": self.review["repository"]["repository"]},
            )
            self.assertEqual(loader["revision"]["value"], self.review["repository"]["revision"])
            self.assertEqual(loader["dtype"]["value"], "bfloat16")
            self.assertEqual(workflow["requiredArtifacts"], [self.review["repository"]["repository"]])
            self.assertEqual(workflow["runtimeQualificationStatus"], "unqualified")

        capabilities = {item["modelType"]: item for item in studio_capability_definitions().values()}
        capability = capabilities["OmniGenPipeline"]
        self.assertEqual(capability["conditioningScale"], 1.6)
        self.assertTrue(capability["supportsMultiImage"])
        self.assertFalse(capability["autoEligible"])
        self.assertFalse(capability["galleryEligible"])


if __name__ == "__main__":
    unittest.main()
