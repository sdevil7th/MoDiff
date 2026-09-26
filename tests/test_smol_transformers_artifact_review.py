from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from modiff.model_artifact_catalog import catalog_revision
from modiff.optional_runtimes import (
    OPTIONAL_RUNTIME_PROFILES,
    TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID,
)
from modiff.studio_execution_specs import (
    SMOLLM2_135M_INSTRUCT_TRANSFORMERS_FILES,
    SMOLVLM_256M_INSTRUCT_TRANSFORMERS_FILES,
    studio_capability_definitions,
    studio_execution_profile_definitions,
    studio_execution_spec_for_pair,
    validate_studio_execution_specs,
)
from modules import MODULE_MAP


REVIEW_PATH = Path("data/smol-transformers-artifact-review.json")


class SmolTransformersArtifactReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
        cls.repositories = {
            item["modelType"]: item for item in cls.review["repositories"]
        }

    def test_review_pins_two_public_python_free_safetensors_surfaces(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        self.assertEqual(self.review["format"], "safetensors")
        self.assertEqual(
            set(self.repositories),
            {
                "HuggingFaceTextGenerationModel",
                "HuggingFaceImageTextToTextModel",
            },
        )
        selected_total = 0
        weight_total = 0
        for repository in self.review["repositories"]:
            with self.subTest(repository=repository["repository"]):
                self.assertFalse(repository["private"])
                self.assertFalse(repository["gated"])
                self.assertFalse(repository["pythonFilesPresent"])
                self.assertFalse(repository["trustRemoteCodeRequired"])
                self.assertEqual(repository["license"], "apache-2.0")
                self.assertEqual(len(repository["revision"]), 40)
                self.assertEqual(
                    catalog_revision(
                        repository["repository"],
                        model_type=repository["modelType"],
                    ),
                    repository["revision"],
                )
                selected = repository["selectedFiles"]
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
                canonical = json.dumps(
                    weights,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                self.assertEqual(
                    repository["weightInventorySha256"],
                    hashlib.sha256(canonical).hexdigest(),
                )
                self.assertEqual(repository["weightFileCount"], len(weights))
                self.assertEqual(
                    repository["weightBytes"],
                    sum(item["byteSize"] for item in weights),
                )
                self.assertTrue(all(item["path"].endswith(".safetensors") for item in weights))
                selected_total += repository["selectedBytes"]
                weight_total += repository["weightBytes"]
        self.assertEqual(self.review["selectedBytesTotal"], selected_total)
        self.assertEqual(self.review["weightBytesTotal"], weight_total)

    def test_download_selections_match_capability_and_review_exactly(self):
        capabilities = studio_capability_definitions()
        expected = {
            "HuggingFaceTextGenerationModel": SMOLLM2_135M_INSTRUCT_TRANSFORMERS_FILES,
            "HuggingFaceImageTextToTextModel": SMOLVLM_256M_INSTRUCT_TRANSFORMERS_FILES,
        }
        for model_type, files in expected.items():
            with self.subTest(model_type=model_type):
                reviewed = self.repositories[model_type]
                self.assertEqual(
                    [item["path"] for item in reviewed["selectedFiles"]],
                    files,
                )
                capability = capabilities[model_type]
                self.assertEqual(capability["downloadFiles"], files)
                self.assertEqual(capability["qualificationStatus"], "graph-qualified-execution-pending")
                self.assertEqual(capability["qualifiedModes"], [])
                self.assertTrue(capability["templateEligible"])
                self.assertFalse(capability["autoEligible"])
                self.assertFalse(capability["galleryEligible"])

    def test_runtime_profile_contains_native_architecture_symbols(self):
        package = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID].packages[0]
        expected = {
            "AutoModelForCausalLM",
            "LlamaForCausalLM",
            "AutoModelForImageTextToText",
            "Idefics3ForConditionalGeneration",
            "Idefics3Processor",
        }
        self.assertEqual(package.version, "5.14.1")
        self.assertTrue(expected.issubset(package.required_symbols))
        self.assertTrue(expected.issubset(package.required_class_symbols))
        self.assertEqual(
            self.review["pinnedRuntime"]["transformersWheelSha256"],
            "9db974c4079ede2d1a3ea7ca5a240df33f2cc26fc2b36ba64c5f2a4f43b6e725",
        )

    def test_profiles_and_specs_are_static_hidden_candidate_contracts(self):
        validated = {
            item["id"]: item for item in validate_studio_execution_specs(MODULE_MAP)
        }
        profiles = studio_execution_profile_definitions()
        cases = (
            (
                "HuggingFaceTextGenerationModel",
                "text_generation",
                "smollm2-135m-instruct:direct",
                "direct-huggingface-transformers-text",
                "AutoModelForCausalLM",
                {"transformersTextModel", "transformersTextGenerate", "transformersTextPreview"},
            ),
            (
                "HuggingFaceImageTextToTextModel",
                "image_to_text",
                "smolvlm-256m-instruct:direct",
                "direct-huggingface-transformers-image-text",
                "AutoModelForImageTextToText",
                {
                    "transformersImageTextModel",
                    "loadImage",
                    "transformersImageTextGenerate",
                    "transformersTextPreview",
                },
            ),
        )
        for model_type, mode, profile_id, path, pipeline_class, roles in cases:
            with self.subTest(model_type=model_type, mode=mode):
                profile = profiles[profile_id]
                self.assertEqual(profile["execution_path"], path)
                self.assertEqual(profile["pipeline_class"], pipeline_class)
                self.assertFalse(profile["live_proof"])
                spec = studio_execution_spec_for_pair(model_type, mode)
                self.assertIsNotNone(spec)
                self.assertEqual(spec["executionProfileId"], profile_id)
                self.assertEqual({item[0] for item in spec["roles"]}, roles)
                self.assertEqual(validated[spec["id"]], spec)

        image_capability = studio_capability_definitions()["HuggingFaceImageTextToTextModel"]
        self.assertEqual(
            image_capability["modeRequirements"]["image_to_text"]["requiredImages"],
            ["referenceImages"],
        )

    def test_review_records_app_only_pending_install_without_deletion(self):
        policy = self.review["downloadPolicy"]
        self.assertTrue(policy["appOnly"])
        self.assertFalse(policy["directWeightDownloadPerformed"])
        self.assertFalse(policy["olderModelsDeleted"])
        self.assertEqual(policy["modelInstallStatus"], "not_started")
        self.assertEqual(
            self.review["admission"]["status"],
            "source_complete_execution_pending",
        )


if __name__ == "__main__":
    unittest.main()
