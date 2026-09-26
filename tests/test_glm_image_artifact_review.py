import hashlib
import json
import re
import unittest
from pathlib import Path

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.optional_runtimes import (
    OPTIONAL_RUNTIME_PROFILES,
    TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID,
)


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data" / "glm-image-artifact-review.json"
GRAPH_PATH = ROOT / "data" / "graphs" / "studio" / "glm-image-pipeline" / "text-to-image.json"
MANIFEST_PATH = ROOT / "data" / "workflow-library-manifest.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class GlmImageArtifactReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_exact_safe_repository_is_admitted_remote_only(self):
        repository = self.review["repository"]
        self.assertEqual(repository["repository"], "zai-org/GLM-Image")
        self.assertRegex(repository["revision"], SHA1)
        self.assertFalse(repository["private"])
        self.assertFalse(repository["gated"])
        self.assertFalse(repository["pythonFilesPresent"])
        self.assertFalse(repository["trustRemoteCodeRequired"])
        self.assertEqual(repository["format"], "safetensors")
        pin = catalog_repository_pin(repository["repository"])
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

    def test_nine_file_mixed_dtype_safetensors_inventory_is_exact(self):
        files = sorted(self.review["weightFiles"], key=lambda item: item["path"])
        repository = self.review["repository"]
        self.assertEqual(repository["weightFileCount"], 9)
        self.assertEqual(repository["weightBytes"], 35765307854)
        self.assertEqual(sum(item["byteSize"] for item in files), repository["weightBytes"])
        canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), repository["weightInventorySha256"])
        aggregate = {}
        for item in files:
            self.assertTrue(item["path"].endswith(".safetensors"))
            self.assertRegex(item["sha256"], SHA256)
            for dtype, count in item["dtypeParameterCounts"].items():
                aggregate[dtype] = aggregate.get(dtype, 0) + count
        self.assertEqual(aggregate, repository["safetensorsDtypeParameterCounts"])
        self.assertEqual(
            sum(item["remoteHeaderBytesFetched"] for item in files),
            self.review["admission"]["remoteSafetensorsHeaderBytesFetched"],
        )
        self.assertFalse(self.review["admission"]["fullWeightFilesDownloaded"])

    def test_package_owned_contract_is_bounded_cancellable_and_mixed_dtype(self):
        contract = self.review["pipelineContract"]
        self.assertEqual(contract["modes"], ["text_to_image"])
        self.assertEqual((contract["moDiffWidth"], contract["moDiffHeight"]), (1024, 1024))
        self.assertEqual(contract["moDiffMaximumInferenceSteps"], 50)
        self.assertEqual(contract["moDiffDefaultGuidanceScale"], 1.5)
        self.assertEqual(contract["moDiffMaximumSequenceLength"], 2048)
        self.assertFalse(contract["negativePromptParameterAvailable"])

        pinned = self.review["pinnedRuntime"]
        self.assertEqual(pinned["pipelineClass"], "GlmImagePipeline")
        self.assertEqual(pinned["textEncoderDtype"], "float32")
        self.assertEqual(
            pinned["modelCpuOffloadSequence"],
            "vision_language_encoder->text_encoder->transformer->vae",
        )
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        self.assertEqual(pinned["callbackTensorInputs"], ["latents", "prompt_embeds"])
        self.assertFalse(pinned["safetyCheckerAvailable"])
        for key, value in pinned.items():
            if key.endswith("Sha256"):
                self.assertRegex(value, SHA256)
        self.assertTrue(all(SHA256.fullmatch(value) for value in self.review["metadataSha256"].values()))

        profile = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID]
        transformers = profile.packages[0]
        symbols = {"GlmImageForConditionalGeneration", "GlmImageProcessor", "ByT5Tokenizer"}
        self.assertTrue(symbols.issubset(transformers.required_symbols))
        self.assertTrue(symbols.issubset(transformers.required_class_symbols))
        self.assertTrue(
            {"GlmImagePipeline", "GlmImageTransformer2DModel"}.issubset(profile.required_diffusers_symbols)
        )
        self.assertNotIn("GlmImagePipeline", profile.pipeline_adapter_symbols)

    def test_license_and_resource_claims_are_conservative(self):
        license_review = self.review["license"]
        self.assertEqual(license_review["id"], "mit")
        self.assertFalse(license_review["licenseFilePresent"])
        self.assertTrue(license_review["commercialUsePermitted"])
        self.assertEqual(license_review["incorporatedXOmniWeightsLicense"], "apache-2.0")
        self.assertTrue(license_review["clarificationPending"])
        envelope = self.review["remoteResourceEnvelope"]
        self.assertEqual(envelope["status"], "upstream_claimed_modiff_qualification_pending")
        self.assertGreater(envelope["minimumSelectiveDiskBytes"], self.review["repository"]["weightBytes"])
        self.assertGreater(envelope["minimumSystemRamBytes"], envelope["minimumSelectiveDiskBytes"])

    def test_portable_graph_uses_only_the_exact_text_to_image_contract(self):
        graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        nodes = {node["data"].get("studioRole"): node for node in graph["nodes"]}
        loader = nodes["diffusersImagePipeline"]["data"]["params"]
        generate = nodes["diffusersImageGenerate"]["data"]["params"]
        self.assertEqual(loader["pipeline_class"]["value"], "GlmImagePipeline")
        self.assertEqual(loader["model_id"]["value"], {"source": "hub", "value": "zai-org/GLM-Image"})
        self.assertEqual(loader["revision"]["value"], self.review["repository"]["revision"])
        self.assertEqual(loader["dtype"]["value"], "bfloat16")
        self.assertEqual(loader["offload_mode"]["value"], "model_cpu")
        self.assertEqual(generate["width"]["value"], 1024)
        self.assertEqual(generate["height"]["value"], 1024)
        self.assertEqual(generate["num_inference_steps"]["value"], 50)
        self.assertEqual(generate["guidance_scale"]["value"], 1.5)
        self.assertEqual(generate["max_sequence_length"]["value"], 2048)
        self.assertTrue(generate["negative_prompt"]["hidden"])

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        workflow = next(
            item for item in manifest["workflows"] if item["id"] == "GlmImagePipeline:text_to_image"
        )
        self.assertEqual(workflow["requiredArtifacts"], ["zai-org/GLM-Image"])
        self.assertEqual(workflow["runtimeQualificationStatus"], "unqualified")


if __name__ == "__main__":
    unittest.main()
