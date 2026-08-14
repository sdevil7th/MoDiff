import importlib.util
import subprocess
import sys
import unittest
from unittest.mock import patch

import diffusers

from modiff.auto_resource import AUTO_MODEL_REQUIREMENTS
from modiff.diffusers_profiles import public_experimental_pipelines
from modiff.modular_contract_only_registry import (
    CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME,
    CURRENT_PIN_CONTRACT_ONLY_MODULAR_IMAGE_PIPELINES,
    CURRENT_PIN_CONTRACT_ONLY_MODULAR_MULTIMODAL_PIPELINES,
    CURRENT_PIN_CONTRACT_ONLY_MODULAR_PIPELINES,
    CURRENT_PIN_CONTRACT_ONLY_MODULAR_VIDEO_PIPELINES,
)
from modiff.modular_workflow_contracts import PINNED_MODULAR_WORKFLOW_TRUTH
from modiff.modular_workflow_discovery import reviewed_modular_workflow_contract
from modules.ModularDiffusers.loaders import ModelsLoader
from modules.ModularDiffusers.modular_utils import (
    _get_registry_instance,
    get_all_model_types,
    get_model_type_metadata,
    pipeline_class_to_modiff_node_config,
)


requires_transformers = unittest.skipUnless(
    importlib.util.find_spec("transformers"),
    "requires the staged optional Transformers runtime",
)


class ContractOnlyModularRegistryTests(unittest.TestCase):
    def test_data_only_registry_does_not_import_diffusers(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import modiff.modular_contract_only_registry; "
                    "raise SystemExit(1 if 'diffusers' in sys.modules else 0)"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @requires_transformers
    def test_current_pin_batches_cover_exact_exported_classes_and_normalized_schemas(self):
        self.assertEqual(len(CURRENT_PIN_CONTRACT_ONLY_MODULAR_IMAGE_PIPELINES), 8)
        self.assertEqual(len(CURRENT_PIN_CONTRACT_ONLY_MODULAR_VIDEO_PIPELINES), 9)
        self.assertEqual(len(CURRENT_PIN_CONTRACT_ONLY_MODULAR_MULTIMODAL_PIPELINES), 5)
        self.assertTrue(
            all(item.batch == "image" for item in CURRENT_PIN_CONTRACT_ONLY_MODULAR_IMAGE_PIPELINES)
        )
        self.assertTrue(
            all(item.batch == "video" for item in CURRENT_PIN_CONTRACT_ONLY_MODULAR_VIDEO_PIPELINES)
        )
        self.assertTrue(
            all(
                item.batch == "multimodal"
                for item in CURRENT_PIN_CONTRACT_ONLY_MODULAR_MULTIMODAL_PIPELINES
            )
        )
        self.assertEqual(len(CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME), 22)

        exported_modular_classes = {
            name
            for name in dir(diffusers)
            if name.endswith("ModularPipeline")
            and isinstance(getattr(diffusers, name), type)
            and getattr(diffusers, name) is not diffusers.ModularPipeline
            and issubclass(getattr(diffusers, name), diffusers.ModularPipeline)
        }
        self.assertEqual(
            set(PINNED_MODULAR_WORKFLOW_TRUTH) | set(CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME),
            exported_modular_classes,
        )

        for specification in CURRENT_PIN_CONTRACT_ONLY_MODULAR_PIPELINES:
            with self.subTest(pipeline=specification.class_name):
                pipeline_class = getattr(diffusers, specification.class_name)
                self.assertTrue(issubclass(pipeline_class, diffusers.ModularPipeline))
                contract = reviewed_modular_workflow_contract(specification.class_name)
                self.assertEqual(contract["pipelineClass"], specification.class_name)
                self.assertTrue(contract["workflows"])
                self.assertTrue(contract["components"])

    def test_expert_registry_includes_contract_only_classes_without_widening_executable_registry(self):
        executable = get_all_model_types()
        expert = get_all_model_types(include_contract_only=True)
        contract_only_names = set(CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME)
        executable_registry_names = {
            pipeline_class.__name__ for pipeline_class in _get_registry_instance().get_all()
        }

        self.assertTrue(contract_only_names.isdisjoint(executable))
        self.assertTrue(contract_only_names.isdisjoint(executable_registry_names))
        self.assertTrue(contract_only_names.issubset(expert))
        for name in contract_only_names:
            metadata = get_model_type_metadata(name)
            self.assertEqual(metadata["execution_status"], "contract_only")
            self.assertEqual(metadata["contract_batch"], CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME[name].batch)
            self.assertFalse(metadata["auto_eligible"])
            self.assertEqual(pipeline_class_to_modiff_node_config(getattr(diffusers, name), "denoise"), (None, None))

    def test_krea2_base_and_turbo_publish_distinct_exact_upstream_contracts(self):
        base = reviewed_modular_workflow_contract("Krea2ModularPipeline")
        turbo = reviewed_modular_workflow_contract("Krea2TurboModularPipeline")

        self.assertEqual(base["blocksClass"], "Krea2AutoBlocks")
        self.assertEqual(turbo["blocksClass"], "Krea2TurboAutoBlocks")
        for contract in (base, turbo):
            self.assertEqual([workflow["taskId"] for workflow in contract["workflows"]], ["text_to_image"])
            self.assertEqual(contract["workflows"][0]["requiredInputs"], ["prompt"])

        base_inputs = {item["name"]: item for item in base["workflows"][0]["inputs"]}
        turbo_inputs = {item["name"]: item for item in turbo["workflows"][0]["inputs"]}
        self.assertEqual(base_inputs["num_inference_steps"]["default"], 28)
        self.assertIn("negative_prompt", base_inputs)
        self.assertIn("guider", {item["name"] for item in base["components"]})
        self.assertEqual(turbo_inputs["num_inference_steps"]["default"], 8)
        self.assertNotIn("negative_prompt", turbo_inputs)
        self.assertNotIn("guider", {item["name"] for item in turbo["components"]})

    def test_minimax_h3_publishes_three_generic_joint_video_audio_contracts(self):
        contract = reviewed_modular_workflow_contract("MiniMaxH3ModularPipeline")
        self.assertEqual(contract["blocksClass"], "MiniMaxH3Blocks")
        workflows = {workflow["id"]: workflow for workflow in contract["workflows"]}
        self.assertEqual(
            {name: workflow["taskId"] for name, workflow in workflows.items()},
            {
                "t2va": "text_to_video_with_audio",
                "fl2va": "first_last_frame_to_video_with_audio",
                "ref2va": "reference_to_video_with_audio",
            },
        )
        self.assertEqual(workflows["t2va"]["requiredInputs"], ["num_inference_steps", "prompt"])
        self.assertEqual(workflows["fl2va"]["requiredInputs"], ["num_inference_steps", "prompt"])
        self.assertEqual(
            workflows["fl2va"]["requiredInputAlternatives"],
            [
                ["image", "num_inference_steps", "prompt"],
                ["last_image", "num_inference_steps", "prompt"],
            ],
        )
        self.assertEqual(
            workflows["ref2va"]["requiredInputs"],
            ["num_frames", "num_inference_steps", "prompt", "references"],
        )
        component_names = {item["name"] for item in contract["components"]}
        self.assertTrue(
            {
                "text_encoder",
                "vae",
                "audio_vae",
                "scheduler",
                "audio_scheduler",
                "transformer",
                "transformer_ref",
            }.issubset(component_names)
        )

    def test_ltx2_and_ltx25_publish_exact_joint_workflows_and_distinct_decoders(self):
        expected_tasks = {
            "text2video": "text_to_video_with_audio",
            "image2video": "image_to_video_with_audio",
            "condition": "condition_to_video_with_audio",
            "in_context": "in_context_to_video_with_audio",
        }
        required_inputs = {
            "text2video": ["prompt"],
            "image2video": ["image", "prompt"],
            "condition": ["conditions", "prompt"],
            "in_context": ["num_frames", "prompt", "reference_conditions"],
        }

        ltx2 = reviewed_modular_workflow_contract("LTX2ModularPipeline")
        ltx25 = reviewed_modular_workflow_contract("LTX25ModularPipeline")
        self.assertEqual(ltx2["blocksClass"], "LTX2AutoBlocks")
        self.assertEqual(ltx25["blocksClass"], "LTX25AutoBlocks")

        for contract in (ltx2, ltx25):
            workflows = {workflow["id"]: workflow for workflow in contract["workflows"]}
            self.assertEqual(
                {name: workflow["taskId"] for name, workflow in workflows.items()},
                expected_tasks,
            )
            self.assertEqual(
                {name: workflow["requiredInputs"] for name, workflow in workflows.items()},
                required_inputs,
            )
            for workflow in workflows.values():
                output_names = {output["name"] for output in workflow["outputs"]}
                self.assertTrue({"videos", "audio"}.issubset(output_names))

        ltx2_components = {item["name"] for item in ltx2["components"]}
        ltx25_components = {item["name"] for item in ltx25["components"]}
        shared_components = {
            "prompt_enhancer",
            "text_encoder",
            "connectors",
            "duration_head",
            "transformer",
            "scheduler",
            "audio_vae",
            "vocoder",
        }
        self.assertTrue(shared_components.issubset(ltx2_components))
        self.assertTrue(shared_components.issubset(ltx25_components))
        self.assertNotIn("diffusion_decoder", ltx2_components)
        self.assertIn("diffusion_decoder", ltx25_components)

        for workflow in ltx2["workflows"]:
            input_names = {item["name"] for item in workflow["inputs"]}
            self.assertTrue({"decode_timestep", "decode_noise_scale"}.issubset(input_names))
            self.assertIn("LTX2VaeDecoderStep", {step["className"] for step in workflow["steps"]})
        for workflow in ltx25["workflows"]:
            input_names = {item["name"] for item in workflow["inputs"]}
            self.assertTrue({"decode_timestep", "decode_noise_scale"}.isdisjoint(input_names))
            self.assertIn(
                "LTX2DiffusionVaeDecoderStep",
                {step["className"] for step in workflow["steps"]},
            )

    def test_wan_animate_2_base_and_distilled_publish_distinct_exact_contracts(self):
        base = reviewed_modular_workflow_contract("WanAnimate2ModularPipeline")
        distilled = reviewed_modular_workflow_contract("WanAnimate2DistilledModularPipeline")

        self.assertEqual(base["blocksClass"], "WanAnimate2Blocks")
        self.assertEqual(distilled["blocksClass"], "WanAnimate2DistilledBlocks")
        for contract in (base, distilled):
            self.assertEqual(len(contract["workflows"]), 1)
            workflow = contract["workflows"][0]
            self.assertEqual((workflow["id"], workflow["taskId"]), ("default", "character_animate"))
            self.assertEqual(workflow["requiredInputs"], ["driving_video", "image", "prompt"])
            self.assertIn("videos", {output["name"] for output in workflow["outputs"]})

        base_workflow = base["workflows"][0]
        distilled_workflow = distilled["workflows"][0]
        base_inputs = {item["name"]: item for item in base_workflow["inputs"]}
        distilled_inputs = {item["name"]: item for item in distilled_workflow["inputs"]}
        self.assertEqual(base_inputs["num_inference_steps"]["default"], 40)
        # The pinned distilled class describes a ten-step recipe, but its
        # composed input schema still inherits the base 40-step default. Keep
        # the reviewed snapshot exact instead of silently rewriting upstream.
        self.assertEqual(distilled_inputs["num_inference_steps"]["default"], 40)
        self.assertIn("WanAnimate2DenoiseStep", {step["className"] for step in base_workflow["steps"]})
        self.assertIn(
            "WanAnimate2DistilledDenoiseStep",
            {step["className"] for step in distilled_workflow["steps"]},
        )

    def test_models_loader_rejects_contract_only_class_before_artifact_or_index_resolution(self):
        for specification in CURRENT_PIN_CONTRACT_ONLY_MODULAR_PIPELINES:
            with (
                self.subTest(pipeline=specification.class_name),
                patch("modules.ModularDiffusers.loaders.require_catalog_revision") as catalog_revision,
                patch("modules.ModularDiffusers.loaders._validate_reviewed_pipeline_index") as validate_index,
                self.assertRaisesRegex(ValueError, "Expert contract discovery only"),
            ):
                ModelsLoader._reviewed_builtin_selection(
                    model_type=specification.class_name,
                    repo_id={"source": "hub", "value": "attacker/repository"},
                    revision="a" * 40,
                )
            catalog_revision.assert_not_called()
            validate_index.assert_not_called()

    def test_public_contracts_are_expert_only_and_publish_no_runnable_modes(self):
        published = {
            capability["modelType"]: capability
            for capability in public_experimental_pipelines()
            if capability["modelType"] in CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME
        }
        self.assertEqual(set(published), set(CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME))
        self.assertTrue(set(published).isdisjoint(AUTO_MODEL_REQUIREMENTS))

        for name, capability in published.items():
            with self.subTest(pipeline=name):
                contract = reviewed_modular_workflow_contract(name)
                self.assertEqual(capability["qualificationStatus"], "contract_only")
                self.assertEqual(capability["runnableModes"], [])
                self.assertEqual(
                    capability["upstreamWorkflows"],
                    [workflow["taskId"] for workflow in contract["workflows"]],
                )
                self.assertTrue(capability["expertVisible"])
                self.assertFalse(capability["autoEligible"])
                self.assertFalse(capability["templateEligible"])
                self.assertFalse(capability["galleryEligible"])
                self.assertEqual(capability["optionalRuntimeProfileIds"], [])
                self.assertEqual(capability["optionalRuntimeProfiles"], [])
                self.assertNotIn("defaultRepo", capability)


if __name__ == "__main__":
    unittest.main()
