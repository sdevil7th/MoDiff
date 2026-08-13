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
REVIEW_PATH = ROOT / "data" / "ernie-image-turbo-artifact-review.json"
GRAPH_PATH = ROOT / "data" / "graphs" / "studio" / "ernie-image-pipeline" / "text-to-image.json"
MANIFEST_PATH = ROOT / "data" / "workflow-library-manifest.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ErnieImageTurboArtifactReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_exact_safe_repository_is_admitted_remote_only(self):
        repository = self.review["repository"]
        self.assertEqual(repository["repository"], "baidu/ERNIE-Image-Turbo")
        self.assertRegex(repository["revision"], SHA1)
        self.assertFalse(repository["private"])
        self.assertFalse(repository["gated"])
        self.assertFalse(repository["pythonFilesPresent"])
        self.assertFalse(repository["trustRemoteCodeRequired"])
        self.assertEqual(repository["format"], "safetensors")
        pin = catalog_repository_pin(repository["repository"])
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

    def test_five_file_bfloat16_safetensors_inventory_is_exact(self):
        files = sorted(self.review["weightFiles"], key=lambda item: item["path"])
        repository = self.review["repository"]
        self.assertEqual(repository["weightFileCount"], 5)
        self.assertEqual(repository["weightBytes"], 31596733630)
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

    def test_package_owned_turbo_contract_is_bounded_and_cancellable(self):
        contract = self.review["pipelineContract"]
        self.assertEqual(contract["modes"], ["text_to_image"])
        self.assertEqual((contract["moDiffWidth"], contract["moDiffHeight"]), (1024, 1024))
        self.assertEqual(contract["moDiffMaximumInferenceSteps"], 8)
        self.assertEqual(contract["turboRequiredGuidanceScale"], 1.0)
        self.assertEqual(contract["tokenizerModelMaxLength"], 2048)
        self.assertTrue(contract["promptEnhancerEnabled"])
        self.assertFalse(contract["negativePromptEffectiveAtGuidanceOne"])

        pinned = self.review["pinnedRuntime"]
        self.assertEqual(pinned["pipelineClass"], "ErnieImagePipeline")
        self.assertEqual(pinned["modelCpuOffloadSequence"], "pe->text_encoder->transformer->vae")
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        self.assertEqual(pinned["callbackTensorInputs"], ["latents"])
        self.assertFalse(pinned["safetyCheckerAvailable"])
        for key, value in pinned.items():
            if key.endswith("Sha256"):
                self.assertRegex(value, SHA256)
        self.assertTrue(all(SHA256.fullmatch(value) for value in self.review["metadataSha256"].values()))

        transformers = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID].packages[0]
        ernie_symbols = {"Mistral3Model", "Ministral3ForCausalLM"}
        self.assertTrue(ernie_symbols.issubset(transformers.required_symbols))
        self.assertTrue(ernie_symbols.issubset(transformers.required_class_symbols))

    def test_license_resource_and_sibling_claims_are_conservative(self):
        license_review = self.review["license"]
        self.assertEqual(license_review["id"], "apache-2.0")
        self.assertTrue(license_review["licenseFilePresent"])
        self.assertTrue(license_review["commercialUsePermitted"])
        self.assertRegex(license_review["licenseFileSha256"], SHA256)
        envelope = self.review["remoteResourceEnvelope"]
        self.assertEqual(envelope["status"], "upstream_claimed_modiff_qualification_pending")
        self.assertGreater(envelope["minimumSelectiveDiskBytes"], self.review["repository"]["weightBytes"])
        self.assertGreater(envelope["minimumSystemRamBytes"], envelope["minimumSelectiveDiskBytes"])
        sibling = self.review["reviewedSibling"]
        self.assertEqual(sibling["repository"], "baidu/ERNIE-Image")
        self.assertFalse(sibling["runtimeCatalogExposed"])
        self.assertIsNone(catalog_repository_pin(sibling["repository"]))

    def test_portable_graph_uses_only_the_exact_turbo_contract(self):
        graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        nodes = {node["data"].get("studioRole"): node for node in graph["nodes"]}
        loader = nodes["diffusersImagePipeline"]["data"]["params"]
        generate = nodes["diffusersImageGenerate"]["data"]["params"]
        self.assertEqual(loader["pipeline_class"]["value"], "ErnieImagePipeline")
        self.assertEqual(loader["model_id"]["value"], {"source": "hub", "value": "baidu/ERNIE-Image-Turbo"})
        self.assertEqual(loader["revision"]["value"], self.review["repository"]["revision"])
        self.assertEqual(loader["dtype"]["value"], "bfloat16")
        self.assertEqual(loader["offload_mode"]["value"], "model_cpu")
        self.assertEqual(generate["width"]["value"], 1024)
        self.assertEqual(generate["height"]["value"], 1024)
        self.assertEqual(generate["num_inference_steps"]["value"], 8)
        self.assertEqual(generate["guidance_scale"]["value"], 1.0)
        self.assertTrue(generate["negative_prompt"]["hidden"])
        self.assertTrue(generate["guidance_scale"]["hidden"])

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        workflow = next(
            item for item in manifest["workflows"] if item["id"] == "ErnieImagePipeline:text_to_image"
        )
        self.assertEqual(workflow["requiredArtifacts"], ["baidu/ERNIE-Image-Turbo"])
        self.assertEqual(workflow["runtimeQualificationStatus"], "unqualified")


if __name__ == "__main__":
    unittest.main()
