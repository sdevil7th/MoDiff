from __future__ import annotations

import json
import unittest

import modules as module_registry

from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
from modiff.model_artifact_catalog import catalog_revision
from modiff.optional_runtimes import (
    OPTIONAL_RUNTIME_PROFILES,
    TRANSFORMERS_MAIN_PEFT_RUNTIME_PROFILE_ID,
    TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID,
)
from modiff.server import WebServer, studio_download_files_for_repo
from modiff.studio_execution_specs import (
    JANUS_PRO_1B_REPO,
    JANUS_PRO_1B_TRANSFORMERS_FILES,
    studio_execution_spec_for_pair,
    validate_studio_execution_specs,
)


MODEL_TYPE = "HuggingFaceAnyToAnyModel"
REVISION = "1655280bb75959cc1cb85529a2a8b26e7016072e"
MODES = ("text_generation", "image_to_text", "text_to_image")


class FakeRequest:
    query = {}


class JanusStudioContractTests(unittest.IsolatedAsyncioTestCase):
    def test_exact_artifact_is_selectable_only_through_the_reviewed_app_inventory(self):
        self.assertEqual(
            JANUS_PRO_1B_TRANSFORMERS_FILES,
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
        self.assertEqual(
            studio_download_files_for_repo(JANUS_PRO_1B_REPO),
            JANUS_PRO_1B_TRANSFORMERS_FILES,
        )
        self.assertEqual(catalog_revision(JANUS_PRO_1B_REPO, model_type=MODEL_TYPE), REVISION)
        self.assertFalse(
            any(
                path.endswith((".bin", ".ckpt", ".pt", ".pth", ".py"))
                for path in JANUS_PRO_1B_TRANSFORMERS_FILES
            )
        )

    def test_pinned_transformers_runtimes_require_every_native_janus_symbol(self):
        janus_symbols = {
            "JanusForConditionalGeneration",
            "JanusProcessor",
            "JanusImageProcessor",
        }
        for profile_id in (
            TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID,
            TRANSFORMERS_MAIN_PEFT_RUNTIME_PROFILE_ID,
        ):
            with self.subTest(profile=profile_id):
                profile = OPTIONAL_RUNTIME_PROFILES[profile_id]
                transformers_package = next(
                    package
                    for package in profile.packages
                    if package.distribution == "transformers"
                )
                self.assertTrue(janus_symbols.issubset(transformers_package.required_symbols))
                self.assertTrue(janus_symbols.issubset(transformers_package.required_class_symbols))

    def test_execution_profile_is_exact_expert_only_and_mps_unqualified(self):
        profile = DIFFUSERS_EXECUTION_PROFILES["janus-pro-1b:direct"]
        self.assertEqual(profile.model_type, MODEL_TYPE)
        self.assertEqual(profile.modes, MODES)
        self.assertEqual(profile.backend_path, "modules.HuggingFaceTransformers.LoadAnyToAnyModel")
        self.assertEqual(profile.execution_path, "direct-huggingface-transformers-any-to-any")
        self.assertEqual(profile.pipeline_class, "JanusForConditionalGeneration")
        self.assertEqual(profile.default_repo, JANUS_PRO_1B_REPO)
        self.assertIsNone(profile.fallback_repo)
        self.assertFalse(profile.live_proof)
        self.assertEqual(profile.supported_offload_modes, ("none",))
        self.assertEqual(
            profile.to_public_dict()["expert_mps_policy"],
            {
                "schema_version": 1,
                "qualification": "unqualified",
                "fallback_action": "open_setup",
            },
        )

    def test_three_specs_have_exact_mode_topology_and_output_kind(self):
        validate_studio_execution_specs(module_registry.MODULE_MAP)
        expectations = {
            "text_generation": {
                "generationSource": "anyToAnyText",
                "requiredImage": False,
                "outputEdge": (
                    "transformersAnyToAnyGenerate",
                    "result",
                    "transformersTextPreview",
                    "value",
                ),
            },
            "image_to_text": {
                "generationSource": "anyToAnyText",
                "requiredImage": True,
                "outputEdge": (
                    "transformersAnyToAnyGenerate",
                    "result",
                    "transformersTextPreview",
                    "value",
                ),
            },
            "text_to_image": {
                "generationSource": "anyToAnyImage",
                "requiredImage": False,
                "outputEdge": (
                    "transformersAnyToAnyGenerate",
                    "image",
                    "preview",
                    "image",
                ),
            },
        }
        for mode, expected in expectations.items():
            with self.subTest(mode=mode):
                specification = studio_execution_spec_for_pair(MODEL_TYPE, mode)
                self.assertIsNotNone(specification)
                self.assertEqual(specification["executionProfileId"], "janus-pro-1b:direct")
                self.assertEqual(specification["defaultRepo"], JANUS_PRO_1B_REPO)
                self.assertEqual(
                    specification["contentHash"],
                    {
                        "text_generation": "studio-spec-v1-e769a28b",
                        "image_to_text": "studio-spec-v1-234664aa",
                        "text_to_image": "studio-spec-v1-b484288e",
                    }[mode],
                )
                self.assertIn(
                    (
                        "transformersAnyToAnyGenerate",
                        "generation_mode",
                        expected["generationSource"],
                    ),
                    specification["bindings"],
                )
                if mode == "text_to_image":
                    self.assertIn(
                        ("transformersAnyToAnyGenerate", "do_sample", "true"),
                        specification["bindings"],
                    )
                else:
                    self.assertNotIn(
                        ("transformersAnyToAnyGenerate", "do_sample", "true"),
                        specification["bindings"],
                    )
                self.assertEqual(
                    ("loadImage", "file", "referenceImages") in specification["bindings"],
                    expected["requiredImage"],
                )
                self.assertIn(expected["outputEdge"], specification["edges"])

    async def test_capability_and_task_contracts_keep_compliance_and_live_gates_closed(self):
        response = await WebServer(module_registry.MODULE_MAP).model_capabilities(FakeRequest())
        payload = json.loads(response.text)
        capability = next(
            item for item in payload["capabilities"] if item["modelType"] == MODEL_TYPE
        )
        self.assertEqual(capability["runnableModes"], sorted(MODES))
        self.assertEqual(
            capability["modeOutputKinds"],
            {
                "text_generation": "json",
                "image_to_text": "json",
                "text_to_image": "image",
            },
        )
        self.assertEqual(capability["executionStatus"], "expert_only")
        self.assertEqual(capability["qualificationStatus"], "graph-qualified-execution-pending")
        self.assertEqual(capability["qualifiedModes"], [])
        self.assertFalse(capability["autoEligible"])
        self.assertTrue(capability["templateEligible"])
        self.assertFalse(capability["galleryEligible"])
        self.assertEqual(capability["revisionCandidates"], [REVISION])
        self.assertEqual(capability["license"], "DeepSeek Model License Agreement v1.0")
        self.assertEqual(
            capability["licenseCompliance"],
            {
                "state": "product_and_user_review_required",
                "codeLicense": "MIT",
                "weightsLicense": "DeepSeek Model License Agreement v1.0",
                "noticePath": "licenses/DeepSeek-Model-License-1.0.txt",
                "useRestrictionsPresent": True,
                "distributionAndHostedUseCarryDuties": True,
                "sourceExecutable": True,
                "liveExecutionQualified": False,
            },
        )

        contracts = {
            contract["mode"]: contract
            for contract in payload["taskTemplateContracts"]
            if contract["modelType"] == MODEL_TYPE
        }
        self.assertEqual(set(contracts), set(MODES))
        self.assertEqual(
            {mode: contract["mediaKind"] for mode, contract in contracts.items()},
            {
                "text_generation": "json",
                "image_to_text": "json",
                "text_to_image": "image",
            },
        )
        self.assertEqual(contracts["text_generation"]["requiredMedia"], [])
        self.assertEqual(
            contracts["image_to_text"]["requiredMedia"],
            [{"kind": "image", "field": "referenceImages", "minimumCount": 1}],
        )
        self.assertEqual(contracts["text_to_image"]["requiredMedia"], [])
        self.assertTrue(all(not contract["galleryEligible"] for contract in contracts.values()))


if __name__ == "__main__":
    unittest.main()
