import hashlib
import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.optional_runtimes import (
    OPTIONAL_RUNTIME_PROFILES,
    TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID,
)
from modiff.studio_execution_specs import LUMINA2_DIFFUSERS_FILES, studio_capability_definitions
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data" / "lumina-image-artifact-review.json"
MANIFEST_PATH = ROOT / "data" / "workflow-library-manifest.json"
GRAPH_ROOT = ROOT / "data" / "graphs" / "studio"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LuminaImageArtifactReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_exact_public_snapshots_are_admitted_remote_only(self):
        expected = {
            "luminaNext": ("LuminaPipeline", "Alpha-VLLM/Lumina-Next-SFT-diffusers"),
            "lumina2": ("Lumina2Pipeline", "Alpha-VLLM/Lumina-Image-2.0"),
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

    def test_exact_safetensors_inventories_and_pickle_exclusion_are_sealed(self):
        for key in ("luminaNext", "lumina2"):
            repository = self.review["repositories"][key]
            files = sorted(self.review["weightFiles"][key], key=lambda item: item["path"])
            self.assertEqual(len(files), repository["weightFileCount"])
            self.assertEqual(sum(item["byteSize"] for item in files), repository["weightBytes"])
            canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
            self.assertEqual(hashlib.sha256(canonical).hexdigest(), repository["weightInventorySha256"])
            self.assertTrue(all(item["path"].endswith(".safetensors") for item in files))
            self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in files))

        lumina2 = self.review["repositories"]["lumina2"]
        self.assertEqual(len(LUMINA2_DIFFUSERS_FILES), lumina2["selectedDownloadFileCount"])
        excluded_paths = {item["path"] for item in lumina2["excludedLegacyPickleFiles"]}
        self.assertEqual(excluded_paths, {"consolidated.00-of-01.pth", "model_args.pth"})
        self.assertTrue(excluded_paths.isdisjoint(LUMINA2_DIFFUSERS_FILES))
        self.assertTrue(all(not path.endswith(".pth") for path in LUMINA2_DIFFUSERS_FILES))
        capabilities = {item["modelType"]: item for item in studio_capability_definitions().values()}
        self.assertEqual(capabilities["Lumina2Pipeline"]["downloadFiles"], LUMINA2_DIFFUSERS_FILES)

    def test_backend_owned_generation_bounds_and_fixed_recipe_are_exact(self):
        lumina = IMAGE_PIPELINE_ADAPTERS["LuminaPipeline"]
        lumina2 = IMAGE_PIPELINE_ADAPTERS["Lumina2Pipeline"]
        for adapter in (lumina, lumina2):
            self.assertTrue(adapter.safe_serialization_required)
            self.assertEqual(adapter.max_inference_steps, 50)
            self.assertEqual(adapter.max_output_pixels, 1024 * 1024)
            self.assertEqual(adapter.max_sequence_length, 256)
        self.assertEqual(lumina.artifact_pipeline_classes, ("LuminaText2ImgPipeline",))
        self.assertFalse(lumina.clean_caption)
        self.assertEqual(lumina2.cfg_trunc_ratio, 0.25)
        self.assertTrue(lumina2.cfg_normalization)

        class LuminaCall:
            def __call__(self, prompt=None, clean_caption=True):
                return None

        class Lumina2Call:
            def __call__(self, prompt=None, cfg_trunc_ratio=1.0, cfg_normalization=False):
                return None

        first, second = {}, {}
        lumina.apply_generation_parameters(LuminaCall(), {}, first)
        lumina2.apply_generation_parameters(Lumina2Call(), {}, second)
        self.assertEqual(first, {"clean_caption": False})
        self.assertEqual(second, {"cfg_trunc_ratio": 0.25, "cfg_normalization": True})

    def test_runtime_callback_symbols_and_license_receipts_are_explicit(self):
        runtime = self.review["pinnedRuntime"]
        profile = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID]
        transformers = next(package for package in profile.packages if package.distribution == "transformers")
        self.assertTrue(set(runtime["requiredDiffusersSymbols"]).issubset(profile.required_diffusers_symbols))
        self.assertTrue(set(runtime["requiredTransformersSymbols"]).issubset(transformers.required_class_symbols))
        safety = self.review["cancellationAndSafety"]
        self.assertTrue(safety["callbackOnStepEndAvailable"])
        self.assertTrue(safety["serverStopPropagatesThroughCallback"])
        self.assertFalse(safety["packageSafetyCheckerPresent"])
        for receipt in ("luminaNextUpstream", "lumina2Upstream"):
            license_review = self.review["license"][receipt]
            self.assertRegex(license_review["revision"], SHA1)
            self.assertRegex(license_review["licenseSha256"], SHA256)

    def test_downloads_are_app_reserved_without_deleting_models(self):
        queue = self.review["appDownloadQueue"]
        self.assertTrue(queue["submittedThroughAppOnly"])
        self.assertFalse(queue["directWeightDownloadPerformed"])
        self.assertFalse(queue["olderModelsDeleted"])
        expected_headroom = (
            queue["preflightFreeBytes"]
            - queue["preflightExistingQueuedReservationBytes"]
            - queue["luminaNextReservationBytes"]
            - queue["lumina2ReservationBytes"]
            - queue["reserveBytes"]
        )
        self.assertTrue(queue["aggregateFits"])
        self.assertEqual(queue["aggregateHeadroomAfterReserveBytes"], expected_headroom)
        self.assertGreater(expected_headroom, 0)

    def test_portable_graphs_pin_the_two_distinct_snapshots(self):
        expected = {
            "LuminaPipeline:text_to_image": (
                "lumina-pipeline/text-to-image.json",
                "LuminaPipeline",
                "Alpha-VLLM/Lumina-Next-SFT-diffusers",
                "luminaNext",
            ),
            "Lumina2Pipeline:text_to_image": (
                "lumina2-pipeline/text-to-image.json",
                "Lumina2Pipeline",
                "Alpha-VLLM/Lumina-Image-2.0",
                "lumina2",
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
