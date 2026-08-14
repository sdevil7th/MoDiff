from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import unittest

from modiff.model_artifact_catalog import catalog_revision
from modules.HuggingFaceTransformers import main as transformers_nodes
from modules.HuggingFaceTransformers.main import (
    ANY_TO_ANY_ADAPTER_CONTRACTS,
    SECURITY_CONTRACT,
)


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data" / "janus-transformers-artifact-review.json"
LICENSE_PATH = ROOT / "licenses" / "DeepSeek-Model-License-1.0.txt"
MODEL_TYPE = "HuggingFaceAnyToAnyModel"
REPOSITORY = "deepseek-community/Janus-Pro-1B"
REVISION = "1655280bb75959cc1cb85529a2a8b26e7016072e"
WEIGHT_SHA256 = "9d1a416f95fb58d6e02858623c9c676003d66006d51fb5d5cc93348ba78cb942"
LICENSE_SHA256 = "09b2b4b4614509ff8baccd3c220e9d9b99e55152925b501f0d5f8b5e36eea982"


class JanusTransformersArtifactReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_review_pins_one_public_native_python_free_safetensors_snapshot(self):
        repository = self.review["repository"]
        self.assertEqual(self.review["schemaVersion"], 1)
        self.assertEqual(self.review["format"], "safetensors")
        self.assertEqual(repository["modelType"], MODEL_TYPE)
        self.assertEqual(repository["repository"], REPOSITORY)
        self.assertEqual(repository["revision"], REVISION)
        self.assertFalse(repository["private"])
        self.assertFalse(repository["gated"])
        self.assertFalse(repository["disabled"])
        self.assertFalse(repository["pythonFilesPresent"])
        self.assertFalse(repository["trustRemoteCodeRequired"])
        self.assertEqual(repository["pipelineTag"], "any-to-any")
        self.assertEqual(repository["libraryName"], "transformers")
        self.assertEqual(
            catalog_revision(REPOSITORY, model_type=MODEL_TYPE),
            REVISION,
        )

        configuration = repository["configuration"]
        self.assertEqual(configuration["modelType"], "janus")
        self.assertEqual(configuration["architectures"], ["JanusForConditionalGeneration"])
        self.assertFalse(configuration["autoMapPresent"])
        self.assertEqual(configuration["textModelType"], "llama")
        self.assertEqual(configuration["maxPositionEmbeddings"], 16_384)
        self.assertEqual(configuration["visionModelType"], "janus_vision_model")
        self.assertEqual(configuration["imageSize"], 384)
        self.assertEqual(configuration["patchSize"], 16)
        self.assertEqual(configuration["numImageTokens"], 576)
        self.assertEqual(configuration["vqModelType"], "janus_vqgan")
        self.assertEqual(configuration["vqCodebookSize"], 16_384)
        self.assertEqual(
            repository["processor"],
            {
                "processorClass": "JanusProcessor",
                "imageProcessorClass": "JanusImageProcessor",
                "imageHeight": 384,
                "imageWidth": 384,
            },
        )

    def test_selected_inventory_is_exact_safe_and_self_consistent(self):
        repository = self.review["repository"]
        selected = repository["selectedFiles"]
        self.assertEqual(
            [item["path"] for item in selected],
            [
                ".gitattributes",
                "README.md",
                "chat_template.jinja",
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "preprocessor_config.json",
                "processor_config.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
        )
        self.assertEqual(repository["selectedBytes"], 4_161_125_359)
        self.assertEqual(
            repository["selectedBytes"],
            sum(item["byteSize"] for item in selected),
        )
        self.assertFalse(
            any(
                item["path"].endswith((".bin", ".ckpt", ".pt", ".pth", ".py"))
                for item in selected
            )
        )

        weights = sorted(repository["weightFiles"], key=lambda item: item["path"])
        self.assertEqual(
            weights,
            [
                {
                    "path": "model.safetensors",
                    "byteSize": 4_153_396_574,
                    "sha256": WEIGHT_SHA256,
                }
            ],
        )
        canonical = json.dumps(weights, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(
            repository["weightInventorySha256"],
            hashlib.sha256(canonical).hexdigest(),
        )
        self.assertEqual(repository["weightFileCount"], len(weights))
        self.assertEqual(repository["weightBytes"], sum(item["byteSize"] for item in weights))
        self.assertEqual(repository["safetensorsParameterCount"], 2_076_636_811)

    def test_review_records_exact_metadata_hashes(self):
        self.assertEqual(
            self.review["repository"]["metadataSha256"],
            {
                "modelCard": "aa4645c510580ced8ce1be183a8f88888519ff8c431c9a1f6ed03802969a30d0",
                "configuration": "931a5d566cbaed44fe0114412755c7b1efe76bf8e30c7d04ee3e337b121805aa",
                "generationConfiguration": "cb3fe41f2dbb0cf759909e9346b01cbd946d27a4a27c56d54d1fed02db0869be",
                "chatTemplate": "ca86eba863fa87feee121798ddc615b986b2642757e8e472d66d82fe256124a2",
                "preprocessorConfiguration": "00bc5d559ff85d8a11452ef01e42ec478feee6de880ba921f5e74381002a984d",
                "processorConfiguration": "b9df983688f30166d870e495de7ac9882d320e76205f96c34da97fc538bca870",
                "specialTokensMap": "0427a4e49abdd6028f4152fc841009b6c892009107b91fc067e518741f077e92",
                "tokenizerConfiguration": "867aca294b5e986d9551939db41dbf9d8f05b2c061f50824b20a704285b63b96",
            },
        )

    def test_weights_keep_the_deepseek_model_license_and_compliance_gate(self):
        rights = self.review["rightsReview"]
        self.assertEqual(rights["state"], "reviewed_use_restricted_model_license")
        self.assertEqual(rights["modelCardLicenseTag"], "mit")
        self.assertEqual(rights["modelCardLicenseName"], "deepseek")
        self.assertEqual(rights["codeLicense"], "MIT")
        self.assertEqual(rights["weightsLicense"], "DeepSeek Model License Agreement v1.0")
        self.assertFalse(rights["modelSnapshotLicenseFilePresent"])
        self.assertTrue(rights["modelCardDistinguishesCodeAndWeightTerms"])
        self.assertTrue(rights["modelUseGrantedSubjectToRestrictions"])
        self.assertTrue(rights["commercialUseNotCategoricallyProhibited"])
        self.assertTrue(rights["distributionAndHostedUseCarryLicenseDuties"])
        self.assertTrue(rights["useRestrictionsPresent"])
        self.assertTrue(rights["productAndUserComplianceReviewRequired"])
        self.assertTrue(rights["licenseMustNotBeFlattenedToMit"])

        external = rights["immutableExternalModelLicense"]
        self.assertEqual(external["repository"], "deepseek-ai/DeepSeek-LLM")
        self.assertEqual(external["revision"], "6712a86bfb7dd25c73383c5ad2eb7a8db540258b")
        self.assertEqual(external["path"], "LICENSE-MODEL")
        license_bytes = LICENSE_PATH.read_bytes()
        self.assertEqual(len(license_bytes), external["byteSize"])
        self.assertEqual(hashlib.sha256(license_bytes).hexdigest(), external["sha256"])
        self.assertEqual(external["sha256"], LICENSE_SHA256)
        self.assertIn(b"Attachment A", license_bytes)
        self.assertIn(b"Use Restrictions", license_bytes)

    def test_native_adapter_is_finite_local_only_and_download_free(self):
        contract = ANY_TO_ANY_ADAPTER_CONTRACTS["janus"]
        self.assertEqual(
            contract,
            {
                "schemaVersion": 1,
                "adapterId": "janus-v1",
                "status": "executable",
                "modelClass": "JanusForConditionalGeneration",
                "processorClass": "JanusProcessor",
                "inputModalities": ["image", "text"],
                "outputModalities": ["image", "text"],
                "evidence": [
                    "transformers@5.14.1:models/janus/configuration_janus.py:31-61",
                    "transformers@5.14.1:models/janus/modeling_janus.py:1180-1349",
                    "transformers@5.14.1:models/janus/processing_janus.py:122-151",
                ],
            },
        )
        self.assertEqual(
            SECURITY_CONTRACT,
            {
                "localFilesOnly": True,
                "trustRemoteCode": False,
                "safetensorsOnly": True,
                "hostedInference": False,
            },
        )
        source = inspect.getsource(transformers_nodes)
        self.assertIn("local_files_only", source)
        self.assertIn("trust_remote_code", source)
        self.assertNotIn("trust_remote_code=True", source)
        self.assertNotIn("use_safetensors=False", source)
        self.assertNotIn("snapshot_download", source)
        self.assertNotIn("InferenceClient", source)

        runtime = self.review["pinnedRuntime"]
        self.assertEqual(runtime["baseTransformersVersion"], "5.14.1")
        self.assertEqual(
            runtime["baseTransformersWheelSha256"],
            "9db974c4079ede2d1a3ea7ca5a240df33f2cc26fc2b36ba64c5f2a4f43b6e725",
        )
        self.assertEqual(
            set(runtime["nativeSymbols"]),
            {
                "AnyToAnyPipeline",
                "AutoModelForMultimodalLM",
                "AutoProcessor",
                "JanusForConditionalGeneration",
                "JanusProcessor",
            },
        )
        self.assertFalse(runtime["probeDownloadedWeights"])

    def test_admission_is_expert_only_pending_install_execution_and_rights_gate(self):
        policy = self.review["downloadPolicy"]
        self.assertTrue(policy["appOnly"])
        self.assertFalse(policy["directWeightDownloadPerformed"])
        self.assertEqual(policy["modelInstallStatus"], "not_started")
        self.assertFalse(policy["olderModelsDeleted"])
        self.assertIn("free disk space", policy["requiredPreflight"])

        admission = self.review["admission"]
        self.assertEqual(
            admission["status"],
            "source_complete_execution_and_product_review_pending",
        )
        self.assertEqual(
            set(admission["executableModes"]),
            {
                f"{MODEL_TYPE}:text_generation",
                f"{MODEL_TYPE}:image_to_text",
                f"{MODEL_TYPE}:text_to_image",
            },
        )
        self.assertTrue(admission["downloadCatalogExposed"])
        self.assertTrue(admission["runtimeCatalogExposed"])
        self.assertTrue(admission["templateEligible"])
        self.assertFalse(admission["autoEligible"])
        self.assertFalse(admission["galleryEligible"])
        self.assertFalse(admission["liveQualified"])
        self.assertEqual(
            set(admission["unresolvedGates"]),
            {
                "app_only_model_install",
                "real_weight_text_execution",
                "real_weight_image_execution",
                "human_output_review",
                "product_and_user_license_compliance_review",
                "physical_macos_execution",
            },
        )


if __name__ == "__main__":
    unittest.main()
