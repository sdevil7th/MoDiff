import hashlib
import json
from pathlib import Path
import re
import unittest

from PIL import Image

from modiff.model_artifact_catalog import catalog_repository_pin, catalog_revision
from modiff.optional_runtimes import OPTIONAL_RUNTIME_PROFILES, TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID
from modiff.studio_execution_specs import PRX_DIFFUSERS_FILES, studio_capability_definitions
from modules.DiffusersImage import Generate
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS, _tag_image_pipeline


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data" / "prx-artifact-review.json"
MANIFEST_PATH = ROOT / "data" / "workflow-library-manifest.json"
GRAPH_ROOT = ROOT / "data" / "graphs" / "studio"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PRXArtifactReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_exact_public_python_free_snapshot_is_admitted_remote_only(self):
        repository = self.review["repository"]
        self.assertEqual(repository["repository"], "Photoroom/prx-512-t2i-sft")
        self.assertRegex(repository["revision"], SHA1)
        self.assertFalse(repository["private"])
        self.assertFalse(repository["gated"])
        self.assertFalse(repository["pythonFilesPresent"])
        self.assertFalse(repository["pythonFilesSelected"])
        self.assertFalse(repository["trustRemoteCodeRequired"])
        self.assertEqual(repository["modelIndexClass"], "PRXPipeline")
        self.assertEqual(self.review["selectedFiles"], PRX_DIFFUSERS_FILES)
        self.assertTrue(all(not path.endswith(".py") for path in self.review["selectedFiles"]))
        pin = catalog_repository_pin(repository["repository"], model_type="PRXPipeline")
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

    def test_exact_safetensors_inventory_is_sealed(self):
        repository = self.review["repository"]
        files = sorted(self.review["weightFiles"], key=lambda item: item["path"])
        self.assertEqual(len(files), repository["weightFileCount"])
        self.assertEqual(sum(item["byteSize"] for item in files), repository["weightBytes"])
        canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), repository["weightInventorySha256"])
        self.assertTrue(all(item["path"].endswith(".safetensors") for item in files))
        self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in files))

    def test_backend_bounds_sequence_alias_and_runtime_symbols_are_explicit(self):
        adapter = IMAGE_PIPELINE_ADAPTERS["PRXPipeline"]
        self.assertEqual(adapter.modes, frozenset({"text_to_image"}))
        self.assertTrue(adapter.safe_serialization_required)
        self.assertEqual(adapter.max_inference_steps, 28)
        self.assertEqual(adapter.min_output_side, 352)
        self.assertEqual(adapter.max_output_side, 704)
        self.assertEqual(adapter.output_side_step, 32)
        self.assertEqual(adapter.max_output_pixels, 512 * 512)
        self.assertEqual(adapter.max_sequence_length, 256)
        self.assertEqual(adapter.max_sequence_length_parameter, "tokenizer_max_length")

        received = {}

        class PRXPipeline:
            _execution_device = "cpu"

            def __call__(
                self,
                *,
                prompt,
                negative_prompt,
                width,
                height,
                num_inference_steps,
                guidance_scale,
                tokenizer_max_length,
                generator,
                output_type,
                return_dict,
            ):
                received.update(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    tokenizer_max_length=tokenizer_max_length,
                    generator=generator,
                    output_type=output_type,
                    return_dict=return_dict,
                )
                return type("Output", (), {"images": [Image.new("RGB", (8, 8), "white")]})()

        pipeline = PRXPipeline()
        _tag_image_pipeline(
            pipeline,
            adapter,
            "text_to_image",
            adapter.default_repo,
            "hub",
            catalog_revision(adapter.default_repo),
        )
        Generate("prx-sequence-alias").execute(
            pipeline=pipeline,
            prompt="reviewed fixture",
            negative_prompt="",
            width=512,
            height=512,
            num_inference_steps=28,
            guidance_scale=5,
            max_sequence_length=256,
        )
        self.assertEqual(received["tokenizer_max_length"], 256)
        self.assertNotIn("max_sequence_length", received)

        runtime = self.review["pinnedRuntime"]
        profile = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID]
        transformers = next(package for package in profile.packages if package.distribution == "transformers")
        self.assertTrue(set(runtime["requiredDiffusersSymbols"]).issubset(profile.required_diffusers_symbols))
        self.assertTrue(set(runtime["requiredTransformersSymbols"]).issubset(transformers.required_class_symbols))
        self.assertRegex(runtime["pipelineSha256"], SHA256)
        self.assertRegex(runtime["transformerSha256"], SHA256)

    def test_cancellation_terms_and_app_queue_evidence_are_bounded(self):
        safety = self.review["cancellationAndSafety"]
        self.assertTrue(safety["callbackOnStepEndAvailable"])
        self.assertFalse(safety["interruptFlagAvailable"])
        self.assertFalse(safety["interruptFlagReadByDenoisingLoop"])
        self.assertTrue(safety["serverStopPropagatesThroughCallback"])
        self.assertFalse(safety["packageSafetyCheckerPresent"])

        license_review = self.review["license"]
        self.assertEqual(license_review["modelCardLicenseTag"], "apache-2.0")
        self.assertEqual(
            license_review["additionalTerms"],
            ["T5-Gemma terms", "Gemma prohibited-use policy"],
        )
        self.assertRegex(license_review["licenseGitBlob"], SHA1)
        self.assertRegex(license_review["licenseSha256"], SHA256)
        self.assertRegex(license_review["noticeGitBlob"], SHA1)
        self.assertRegex(license_review["noticeSha256"], SHA256)

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

    def test_portable_graph_pins_exact_text_to_image_snapshot(self):
        workflow_id = "PRXPipeline:text_to_image"
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        workflow = {item["id"]: item for item in manifest["workflows"]}[workflow_id]
        graph = json.loads((GRAPH_ROOT / "prx-pipeline" / "text-to-image.json").read_text(encoding="utf-8"))
        nodes = {node["data"].get("studioRole"): node for node in graph["nodes"]}
        loader = nodes["diffusersImagePipeline"]["data"]["params"]
        self.assertEqual(loader["pipeline_class"]["value"], "PRXPipeline")
        self.assertEqual(
            loader["model_id"]["value"],
            {"source": "hub", "value": self.review["repository"]["repository"]},
        )
        self.assertEqual(loader["revision"]["value"], self.review["repository"]["revision"])
        self.assertEqual(loader["dtype"]["value"], "bfloat16")
        self.assertEqual(workflow["requiredArtifacts"], [self.review["repository"]["repository"]])
        self.assertEqual(workflow["runtimeQualificationStatus"], "unqualified")

        capabilities = {item["modelType"]: item for item in studio_capability_definitions().values()}
        capability = capabilities["PRXPipeline"]
        self.assertEqual(capability["recommendedGuidance"], 5.0)
        self.assertEqual(capability["recommendedMaxSequenceLength"], 256)
        self.assertEqual(capability["downloadFiles"], PRX_DIFFUSERS_FILES)
        self.assertFalse(capability["autoEligible"])
        self.assertFalse(capability["galleryEligible"])


if __name__ == "__main__":
    unittest.main()
