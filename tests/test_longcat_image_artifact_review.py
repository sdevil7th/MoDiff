import hashlib
import json
from pathlib import Path
import re
import unittest

from PIL import Image

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.optional_runtimes import (
    OPTIONAL_RUNTIME_PROFILES,
    TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID,
)
from modules.DiffusersImage.main import (
    IMAGE_PIPELINE_ADAPTERS,
    _validate_image_media,
)


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data" / "longcat-image-artifact-review.json"
MANIFEST_PATH = ROOT / "data" / "workflow-library-manifest.json"
GRAPH_ROOT = ROOT / "data" / "graphs" / "studio"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LongCatImageArtifactReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_exact_public_safetensors_snapshots_are_admitted_remote_only(self):
        expected = {
            "textToImage": ("LongCatImagePipeline", "meituan-longcat/LongCat-Image"),
            "imageEdit": ("LongCatImageEditPipeline", "meituan-longcat/LongCat-Image-Edit"),
        }
        for key, (model_type, repository_id) in expected.items():
            repository = self.review["repositories"][key]
            self.assertEqual(repository["repository"], repository_id)
            self.assertRegex(repository["revision"], SHA1)
            self.assertFalse(repository["private"])
            self.assertFalse(repository["gated"])
            self.assertFalse(repository["pythonFilesPresent"])
            self.assertFalse(repository["trustRemoteCodeRequired"])
            self.assertFalse(repository["licenseFilePresent"])
            pin = catalog_repository_pin(repository_id, model_type=model_type)
            self.assertIsNotNone(pin)
            self.assertEqual(pin["revision"], repository["revision"])
            self.assertEqual(pin["license"], "apache-2.0")

        admission = self.review["admission"]
        self.assertEqual(admission["status"], "source_complete_execution_pending")
        self.assertTrue(admission["runtimeCatalogExposed"])
        self.assertTrue(admission["downloadCatalogExposed"])
        self.assertTrue(admission["capabilityExposed"])
        self.assertTrue(admission["remoteOnly"])
        self.assertFalse(admission["liveQualified"])
        self.assertFalse(admission["autoEligible"])
        self.assertFalse(admission["galleryEligible"])

    def test_two_exact_seven_file_weight_inventories_are_sealed(self):
        for key in ("textToImage", "imageEdit"):
            repository = self.review["repositories"][key]
            files = sorted(self.review["weightFiles"][key], key=lambda item: item["path"])
            self.assertEqual(repository["weightFileCount"], 7)
            self.assertEqual(sum(item["byteSize"] for item in files), repository["weightBytes"])
            canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
            self.assertEqual(hashlib.sha256(canonical).hexdigest(), repository["weightInventorySha256"])
            self.assertTrue(all(item["path"].endswith(".safetensors") for item in files))
            self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in files))

        text_files = {item["path"]: item for item in self.review["weightFiles"]["textToImage"]}
        edit_files = {item["path"]: item for item in self.review["weightFiles"]["imageEdit"]}
        shared = set(text_files) - {"transformer/diffusion_pytorch_model.safetensors"}
        self.assertTrue(all(text_files[path] == edit_files[path] for path in shared))
        self.assertNotEqual(
            text_files["transformer/diffusion_pytorch_model.safetensors"]["sha256"],
            edit_files["transformer/diffusion_pytorch_model.safetensors"]["sha256"],
        )

    def test_backend_owned_generation_and_edit_bounds_are_exact(self):
        text_adapter = IMAGE_PIPELINE_ADAPTERS["LongCatImagePipeline"]
        edit_adapter = IMAGE_PIPELINE_ADAPTERS["LongCatImageEditPipeline"]
        self.assertTrue(text_adapter.safe_serialization_required)
        self.assertEqual(text_adapter.max_inference_steps, 50)
        self.assertEqual(text_adapter.max_output_pixels, 1024 * 1024)
        self.assertFalse(text_adapter.enable_prompt_rewrite)
        self.assertEqual(edit_adapter.max_reference_pixels, 1024 * 1024)
        self.assertEqual(edit_adapter.min_reference_aspect_ratio, 0.25)
        self.assertEqual(edit_adapter.max_reference_aspect_ratio, 4.0)

        class LongCatCall:
            def __call__(self, prompt=None, enable_prompt_rewrite=True):
                return None

        target = {}
        text_adapter.apply_generation_parameters(LongCatCall(), {}, target)
        self.assertEqual(target, {"enable_prompt_rewrite": False})

        _validate_image_media(
            Image.new("RGB", (2048, 512)),
            field="LongCat edit source",
            max_items=1,
            max_pixels=1024 * 1024,
            min_aspect_ratio=0.25,
            max_aspect_ratio=4.0,
        )
        with self.assertRaisesRegex(ValueError, "aspect ratio must be at most 4"):
            _validate_image_media(
                Image.new("RGB", (4096, 128)),
                field="LongCat edit source",
                max_items=1,
                max_pixels=1024 * 1024,
                min_aspect_ratio=0.25,
                max_aspect_ratio=4.0,
            )

    def test_runtime_cancellation_symbols_and_license_receipt_are_explicit(self):
        runtime = self.review["pinnedRuntime"]
        profile = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID]
        transformers = next(package for package in profile.packages if package.distribution == "transformers")
        self.assertTrue(set(runtime["requiredDiffusersSymbols"]).issubset(profile.required_diffusers_symbols))
        self.assertTrue(
            set(runtime["requiredTransformersSymbols"]).issubset(transformers.required_class_symbols)
        )
        self.assertTrue(self.review["cancellationAndSafety"]["interruptFlagAvailable"])
        self.assertTrue(
            self.review["cancellationAndSafety"]["serverStopPropagatesToActivePipelineInterrupt"]
        )
        self.assertFalse(self.review["cancellationAndSafety"]["packageSafetyCheckerPresent"])
        license_review = self.review["license"]
        self.assertRegex(license_review["upstreamRevision"], SHA1)
        self.assertRegex(license_review["upstreamLicenseSha256"], SHA256)

    def test_downloads_are_app_reserved_without_deleting_models(self):
        queue = self.review["appDownloadQueue"]
        self.assertTrue(queue["submittedThroughAppOnly"])
        self.assertFalse(queue["directWeightDownloadPerformed"])
        self.assertFalse(queue["olderModelsDeleted"])
        expected_headroom = (
            queue["preflightFreeBytes"]
            - queue["preflightExistingQueuedReservationBytes"]
            - queue["textToImageReservationBytes"]
            - queue["imageEditReservationBytes"]
            - queue["reserveBytes"]
        )
        self.assertTrue(queue["aggregateFits"])
        self.assertEqual(queue["aggregateHeadroomAfterReserveBytes"], expected_headroom)
        self.assertGreater(expected_headroom, 0)

    def test_portable_graphs_pin_the_two_distinct_snapshots(self):
        expected = {
            "LongCatImagePipeline:text_to_image": (
                "long-cat-image-pipeline/text-to-image.json",
                "LongCatImagePipeline",
                "meituan-longcat/LongCat-Image",
                "textToImage",
            ),
            "LongCatImageEditPipeline:edit_image": (
                "long-cat-image-edit-pipeline/edit-image.json",
                "LongCatImageEditPipeline",
                "meituan-longcat/LongCat-Image-Edit",
                "imageEdit",
            ),
        }
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        by_id = {workflow["id"]: workflow for workflow in manifest["workflows"]}
        for workflow_id, (path, pipeline_class, repository_id, review_key) in expected.items():
            workflow = by_id[workflow_id]
            graph = json.loads((GRAPH_ROOT / path).read_text(encoding="utf-8"))
            nodes = {node["data"].get("studioRole"): node for node in graph["nodes"]}
            loader = nodes["diffusersImagePipeline"]["data"]["params"]
            self.assertEqual(loader["pipeline_class"]["value"], pipeline_class)
            self.assertEqual(loader["model_id"]["value"], {"source": "hub", "value": repository_id})
            self.assertEqual(loader["revision"]["value"], self.review["repositories"][review_key]["revision"])
            self.assertEqual(loader["dtype"]["value"], "bfloat16")
            self.assertEqual(workflow["requiredArtifacts"], [repository_id])
            self.assertEqual(workflow["runtimeQualificationStatus"], "unqualified")


if __name__ == "__main__":
    unittest.main()
