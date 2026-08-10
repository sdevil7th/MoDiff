import json
import unittest
from pathlib import Path
from unittest.mock import patch

import diffusers

import modules as module_registry
from modiff.diffusers_profiles import public_execution_profiles, public_experimental_pipelines
from modiff.modular_workflow_contracts import (
    PINNED_DIFFUSERS_REVISION,
    PINNED_MODULAR_WORKFLOW_TRUTH,
)
from modiff.server import WebServer
from modules.ModularDiffusers.loaders import ModelsLoader
from modules.ModularDiffusers.modular_utils import (
    get_all_model_types,
    get_model_type_metadata,
    require_modiff_node_contract,
)


MODULAR_BACKEND_PATH = "modules.ModularDiffusers.ModelsLoader"


def _pipeline_blocks(model_type):
    truth = PINNED_MODULAR_WORKFLOW_TRUTH[model_type]
    pipeline_class = getattr(diffusers, model_type)
    constructor_config = dict(truth.constructor_config)
    pipeline = pipeline_class(config_dict=constructor_config) if constructor_config else pipeline_class()
    return pipeline_class, pipeline.blocks


def _advertised_modular_modes():
    advertised = {}
    for capability in public_experimental_pipelines():
        if capability.get("executionKind") != "modular":
            continue
        advertised.setdefault(capability["modelType"], set()).update(capability["runnableModes"])
    for profile in public_execution_profiles():
        if profile["backend_path"] != MODULAR_BACKEND_PATH:
            continue
        advertised.setdefault(profile["pipeline_class"], set()).update(profile["modes"])
    return advertised


class ModularWorkflowTruthTests(unittest.TestCase):
    def test_matrix_covers_all_eleven_registered_pipelines(self):
        registered = set(get_all_model_types()) - {"", "DummyCustomPipeline"}
        self.assertEqual(len(PINNED_MODULAR_WORKFLOW_TRUTH), 11)
        self.assertEqual(set(PINNED_MODULAR_WORKFLOW_TRUTH), registered)
        self.assertEqual(PINNED_DIFFUSERS_REVISION, "13a7bee4878d62fccc8d25f97e480e68de96fa03")
        dependency_contract = Path("pyproject.toml").read_text(encoding="utf-8")
        self.assertIn(
            f"diffusers.git@{PINNED_DIFFUSERS_REVISION}",
            dependency_contract,
            "A Diffusers pin update requires an explicit review of the Modular workflow truth matrix.",
        )

    def test_pinned_workflow_maps_and_fixed_sequences_are_exact(self):
        for model_type, truth in PINNED_MODULAR_WORKFLOW_TRUTH.items():
            with self.subTest(model_type=model_type):
                _pipeline_class, blocks = _pipeline_blocks(model_type)
                self.assertEqual(type(blocks).__name__, truth.blocks_class)

                if truth.workflows:
                    self.assertFalse(truth.fixed_block_sequence)
                    expected = {workflow.name: workflow.required_inputs for workflow in truth.workflows}
                    actual_map = blocks._workflow_map
                    self.assertIsNotNone(actual_map)
                    actual = {
                        name: frozenset(input_name for input_name, required in inputs.items() if required)
                        for name, inputs in actual_map.items()
                    }
                    self.assertEqual(actual, expected)
                    self.assertEqual(set(blocks.available_workflows), set(expected))
                    for workflow_name, required_inputs in expected.items():
                        workflow = blocks.get_workflow(workflow_name)
                        self.assertTrue(tuple(workflow.block_names))
                        self.assertTrue(set(required_inputs).issubset(workflow.input_names))
                        self.assertTrue(workflow.output_names)
                else:
                    self.assertTrue(truth.fixed_block_sequence)
                    self.assertIsNone(blocks._workflow_map)
                    with self.assertRaises(NotImplementedError):
                        _ = blocks.available_workflows
                    self.assertEqual(tuple(blocks.block_names), truth.fixed_block_sequence)

    def test_flux_qwen_and_wan_advertised_modular_modes_are_exact(self):
        expected = {
            "StableDiffusionXLModularPipeline": {"text_to_image", "image_to_image", "control_image"},
            "QwenImageModularPipeline": {"control_image"},
            "QwenImageEditModularPipeline": {"edit_image"},
            "QwenImageEditPlusModularPipeline": {"edit_image", "multi_image_reference_edit"},
            "QwenImageLayeredModularPipeline": {"layer_decomposition"},
            "FluxModularPipeline": {"text_to_image", "image_to_image"},
            "ZImageModularPipeline": {"text_to_image"},
            "WanModularPipeline": {"text_to_video"},
            "WanImage2VideoModularPipeline": {"image_to_video"},
        }
        self.assertEqual(_advertised_modular_modes(), expected)
        self.assertEqual(
            {model_type: set(dict(truth.modes)) for model_type, truth in PINNED_MODULAR_WORKFLOW_TRUTH.items()},
            {
                model_type: expected.get(model_type, set())
                for model_type in PINNED_MODULAR_WORKFLOW_TRUTH
            },
        )
        self.assertNotIn("FluxKontextModularPipeline", expected)
        self.assertNotIn("Flux2KleinModularPipeline", expected)

    def test_every_advertised_mode_has_a_constructible_action_and_state_contract(self):
        for model_type, modes in _advertised_modular_modes().items():
            truth = PINNED_MODULAR_WORKFLOW_TRUTH[model_type]
            pipeline_class, _blocks = _pipeline_blocks(model_type)
            metadata = get_model_type_metadata(model_type)
            self.assertIsNotNone(metadata)

            for mode in sorted(modes):
                with self.subTest(model_type=model_type, mode=mode):
                    mode_truth = truth.mode(mode)
                    self.assertIsNotNone(mode_truth, f"{model_type}:{mode} has no reviewed MoDiff mode contract")

                    if mode_truth.upstream_workflow is not None:
                        workflows = {workflow.name: workflow for workflow in truth.workflows}
                        self.assertIn(mode_truth.upstream_workflow, workflows)
                        self.assertEqual(
                            mode_truth.required_upstream_inputs,
                            workflows[mode_truth.upstream_workflow].required_inputs,
                        )
                    else:
                        self.assertTrue(truth.fixed_block_sequence)

                    action_contracts = {}
                    for action in mode_truth.action_sequence:
                        action_contract = metadata["node_params"].get(action)
                        self.assertIsNotNone(action_contract, f"{model_type}:{mode} lacks action {action}")
                        action_contracts[action] = action_contract
                        blocks, resolved_contract = require_modiff_node_contract(
                            pipeline_class,
                            action,
                            require_blocks=not (
                                action == "controlnet"
                                and action_contract.get("block_name") is None
                            ),
                        )
                        self.assertIsNotNone(resolved_contract)
                        if action_contract.get("block_name") is not None:
                            self.assertIsNotNone(blocks)

                    action_inputs = {
                        input_name
                        for contract in action_contracts.values()
                        for input_name in contract["input_names"]
                    }
                    self.assertTrue(mode_truth.required_upstream_inputs.issubset(action_inputs))

                    for edge in mode_truth.state_edges:
                        self.assertIn(edge.producer_action, action_contracts)
                        self.assertIn(edge.consumer_action, action_contracts)
                        self.assertIn(
                            edge.producer_output,
                            action_contracts[edge.producer_action]["output_names"],
                        )
                        self.assertIn(
                            edge.consumer_input,
                            action_contracts[edge.consumer_action]["input_names"],
                        )

    def test_sdxl_public_mode_edges_are_route_aware_and_exact(self):
        truth = PINNED_MODULAR_WORKFLOW_TRUTH["StableDiffusionXLModularPipeline"]
        expected_edges = {
            "text_to_image": (
                ("text_encoder", "embeddings", "denoise", "embeddings"),
                ("denoise", "latents", "decoder", "latents"),
                ("denoise", "route_state_out", "decoder", "route_state_in"),
            ),
            "image_to_image": (
                ("text_encoder", "embeddings", "denoise", "embeddings"),
                ("vae_encoder", "image_latents", "denoise", "image_latents"),
                ("vae_encoder", "route_state_out", "denoise", "route_state_in"),
                ("denoise", "latents", "decoder", "latents"),
                ("denoise", "route_state_out", "decoder", "route_state_in"),
            ),
            "control_image": (
                ("text_encoder", "embeddings", "denoise", "embeddings"),
                ("controlnet", "controlnet_bundle", "denoise", "controlnet_bundle"),
                ("denoise", "latents", "decoder", "latents"),
                ("denoise", "route_state_out", "decoder", "route_state_in"),
            ),
        }

        self.assertEqual(set(dict(truth.modes)), set(expected_edges))
        for name, mode in truth.modes:
            with self.subTest(mode=name):
                actual_edges = tuple(
                    (
                        edge.producer_action,
                        edge.producer_output,
                        edge.consumer_action,
                        edge.consumer_input,
                    )
                    for edge in mode.state_edges
                )
                self.assertEqual(actual_edges, expected_edges[name])

    def test_sdxl_base_inpaint_state_flow_is_exact_constructible_and_nonadvertised(
        self,
    ):
        model_type = "StableDiffusionXLModularPipeline"
        truth = PINNED_MODULAR_WORKFLOW_TRUTH[model_type]
        pipeline_class, blocks = _pipeline_blocks(model_type)
        metadata = get_model_type_metadata(model_type)
        workflows = {workflow.name: workflow for workflow in truth.workflows}
        state_flow = truth.state_flow("inpainting")

        self.assertEqual(set(dict(truth.state_flows)), {"inpainting"})
        self.assertEqual(
            set(dict(truth.modes)),
            {"text_to_image", "image_to_image", "control_image"},
        )
        self.assertEqual(
            _advertised_modular_modes()[model_type],
            {"text_to_image", "image_to_image", "control_image"},
        )
        self.assertIsNotNone(state_flow)
        self.assertEqual(state_flow.upstream_workflow, "inpainting")
        self.assertEqual(
            state_flow.required_upstream_inputs,
            frozenset({"mask_image", "image", "prompt"}),
        )
        self.assertEqual(state_flow.required_upstream_inputs, workflows["inpainting"].required_inputs)
        self.assertEqual(
            state_flow.upstream_block_sequence,
            (
                "text_encoder",
                "vae_encoder",
                "denoise.input",
                "denoise.before_denoise.set_timesteps",
                "denoise.before_denoise.prepare_latents",
                "denoise.before_denoise.prepare_add_cond",
                "denoise.denoise",
                "decode",
            ),
        )
        self.assertEqual(
            state_flow.action_sequence,
            ("text_encoder", "vae_encoder", "denoise", "decoder"),
        )

        workflow = blocks.get_workflow("inpainting")
        self.assertEqual(tuple(workflow.block_names), state_flow.upstream_block_sequence)
        self.assertTrue(state_flow.required_upstream_inputs.issubset(workflow.input_names))
        for output_name in (
            "image_latents",
            "mask",
            "masked_image_latents",
            "crops_coords",
            "latents",
            "images",
        ):
            self.assertIn(output_name, workflow.output_names)

        action_contracts = {}
        action_blocks = {}
        for action in state_flow.action_sequence:
            action_contract = metadata["node_params"].get(action)
            self.assertIsNotNone(
                action_contract,
                f"{model_type}:inpainting lacks action {action}",
            )
            action_contracts[action] = action_contract
            resolved_blocks, resolved_contract = require_modiff_node_contract(
                pipeline_class,
                action,
            )
            self.assertIsNotNone(resolved_blocks)
            self.assertIsNotNone(resolved_contract)
            action_blocks[action] = resolved_blocks

        denoise_contract = action_contracts["denoise"]
        decoder_contract = action_contracts["decoder"]
        self.assertIn("vae", denoise_contract["model_input_names"])
        self.assertIn("vae", action_blocks["denoise"].component_names)
        self.assertEqual(
            denoise_contract["params"]["vae"]["type"],
            decoder_contract["params"]["vae"]["type"],
        )
        self.assertEqual(
            denoise_contract["params"]["vae"]["type"],
            ModelsLoader.params["vae_out"]["type"],
        )
        self.assertEqual(denoise_contract["params"]["vae"]["display"], "input")
        self.assertEqual(ModelsLoader.params["vae_out"]["display"], "output")

        action_inputs = {
            input_name
            for contract in action_contracts.values()
            for input_name in contract["input_names"]
        }
        self.assertTrue(state_flow.required_upstream_inputs.issubset(action_inputs))

        expected_edges = (
            ("text_encoder", "embeddings", "denoise", "embeddings"),
            ("vae_encoder", "image_latents", "denoise", "image_latents"),
            ("vae_encoder", "mask", "denoise", "mask"),
            ("vae_encoder", "masked_image_latents", "denoise", "masked_image_latents"),
            ("vae_encoder", "route_state_out", "denoise", "route_state_in"),
            ("denoise", "latents", "decoder", "latents"),
            ("denoise", "route_state_out", "decoder", "route_state_in"),
        )
        actual_edges = tuple(
            (
                edge.producer_action,
                edge.producer_output,
                edge.consumer_action,
                edge.consumer_input,
            )
            for edge in state_flow.state_edges
        )
        self.assertEqual(actual_edges, expected_edges)

        for edge in state_flow.state_edges:
            self.assertIn(
                edge.producer_output,
                action_contracts[edge.producer_action]["output_names"],
            )
            self.assertIn(
                edge.consumer_input,
                action_contracts[edge.consumer_action]["input_names"],
            )

        for field_name, field_type in (
            ("mask", "latent_mask"),
            ("masked_image_latents", "masked_latents"),
        ):
            producer_param = action_contracts["vae_encoder"]["params"][field_name]
            consumer_param = action_contracts["denoise"]["params"][field_name]
            self.assertEqual(producer_param["display"], "output")
            self.assertEqual(consumer_param["display"], "input")
            self.assertEqual(producer_param["type"], field_type)
            self.assertEqual(consumer_param["type"], field_type)

    def test_qwen_state_flows_are_exact_constructible_and_nonadvertised(self):
        model_type = "QwenImageModularPipeline"
        truth = PINNED_MODULAR_WORKFLOW_TRUTH[model_type]
        pipeline_class, blocks = _pipeline_blocks(model_type)
        metadata = get_model_type_metadata(model_type)
        workflows = {workflow.name: workflow for workflow in truth.workflows}
        expected_names = {
            "image2image",
            "inpainting",
            "controlnet_image2image",
            "controlnet_inpainting",
        }

        self.assertEqual(set(dict(truth.state_flows)), expected_names)
        self.assertEqual(set(dict(truth.modes)), {"control_image"})
        self.assertEqual(_advertised_modular_modes()[model_type], {"control_image"})

        for name, state_flow in truth.state_flows:
            with self.subTest(state_flow=name):
                self.assertEqual(name, state_flow.upstream_workflow)
                self.assertIn(name, workflows)
                self.assertEqual(state_flow.required_upstream_inputs, workflows[name].required_inputs)

                workflow = blocks.get_workflow(name)
                self.assertEqual(tuple(workflow.block_names), state_flow.upstream_block_sequence)
                self.assertTrue(state_flow.required_upstream_inputs.issubset(workflow.input_names))
                self.assertIn("latents", workflow.output_names)
                self.assertIn("images", workflow.output_names)

                action_contracts = {}
                for action in state_flow.action_sequence:
                    action_contract = metadata["node_params"].get(action)
                    self.assertIsNotNone(action_contract, f"{model_type}:{name} lacks action {action}")
                    action_contracts[action] = action_contract
                    resolved_blocks, resolved_contract = require_modiff_node_contract(
                        pipeline_class,
                        action,
                        require_blocks=not (action == "controlnet" and action_contract.get("block_name") is None),
                    )
                    self.assertIsNotNone(resolved_contract)
                    if action_contract.get("block_name") is not None:
                        self.assertIsNotNone(resolved_blocks)

                action_inputs = {
                    input_name
                    for contract in action_contracts.values()
                    for input_name in contract["input_names"]
                }
                self.assertTrue(state_flow.required_upstream_inputs.issubset(action_inputs))

                for edge in state_flow.state_edges:
                    self.assertIn(edge.producer_action, action_contracts)
                    self.assertIn(edge.consumer_action, action_contracts)
                    self.assertIn(edge.producer_output, action_contracts[edge.producer_action]["output_names"])
                    self.assertIn(edge.consumer_input, action_contracts[edge.consumer_action]["input_names"])

                is_inpaint = name.endswith("inpainting") or name == "inpainting"
                is_control = name.startswith("controlnet_")
                self.assertEqual("processed_mask_image" in workflow.output_names, is_inpaint)
                self.assertEqual("mask_overlay_kwargs" in workflow.output_names, is_inpaint)
                self.assertEqual("mask" in workflow.output_names, is_inpaint)
                self.assertEqual("control_image_latents" in workflow.output_names, is_control)

    def test_qwen_state_flow_edges_are_exact_and_ordered(self):
        truth = PINNED_MODULAR_WORKFLOW_TRUTH["QwenImageModularPipeline"]
        expected_edges = {
            "image2image": (
                ("text_encoder", "embeddings", "denoise", "embeddings"),
                ("vae_encoder", "image_latents", "denoise", "image_latents"),
                ("vae_encoder", "route_state_out", "denoise", "route_state_in"),
                ("denoise", "latents", "decoder", "latents"),
                ("denoise", "route_state_out", "decoder", "route_state_in"),
            ),
            "inpainting": (
                ("text_encoder", "embeddings", "denoise", "embeddings"),
                ("vae_encoder", "image_latents", "denoise", "image_latents"),
                ("vae_encoder", "route_state_out", "denoise", "route_state_in"),
                ("denoise", "latents", "decoder", "latents"),
                ("denoise", "route_state_out", "decoder", "route_state_in"),
            ),
            "controlnet_image2image": (
                ("text_encoder", "embeddings", "denoise", "embeddings"),
                ("vae_encoder", "image_latents", "denoise", "image_latents"),
                ("vae_encoder", "route_state_out", "controlnet", "route_state_in"),
                ("controlnet", "controlnet_bundle", "denoise", "controlnet_bundle"),
                ("controlnet", "route_state_out", "denoise", "route_state_in"),
                ("denoise", "latents", "decoder", "latents"),
                ("denoise", "route_state_out", "decoder", "route_state_in"),
            ),
            "controlnet_inpainting": (
                ("text_encoder", "embeddings", "denoise", "embeddings"),
                ("vae_encoder", "image_latents", "denoise", "image_latents"),
                ("vae_encoder", "route_state_out", "controlnet", "route_state_in"),
                ("controlnet", "controlnet_bundle", "denoise", "controlnet_bundle"),
                ("controlnet", "route_state_out", "denoise", "route_state_in"),
                ("denoise", "latents", "decoder", "latents"),
                ("denoise", "route_state_out", "decoder", "route_state_in"),
            ),
        }

        self.assertEqual(len(truth.state_flows), 4)
        self.assertEqual(
            {name for name, _state_flow in truth.state_flows},
            {"image2image", "inpainting", "controlnet_image2image", "controlnet_inpainting"},
        )
        for name, state_flow in truth.state_flows:
            with self.subTest(state_flow=name):
                actual_edges = tuple(
                    (
                        edge.producer_action,
                        edge.producer_output,
                        edge.consumer_action,
                        edge.consumer_input,
                    )
                    for edge in state_flow.state_edges
                )
                self.assertEqual(actual_edges, expected_edges[name])

    def test_false_modular_claims_are_explicitly_unsupported(self):
        experimental = {item["modelType"]: item for item in public_experimental_pipelines()}

        sdxl = experimental["StableDiffusionXLModularPipeline"]
        self.assertNotIn("inpaint", sdxl["runnableModes"])
        self.assertEqual(sdxl["unsupportedModes"]["inpaint"]["status"], "unsupported")
        self.assertEqual(sdxl["unsupportedModes"]["inpaint"]["upstreamWorkflow"], "inpainting")
        self.assertIn("state flow internally", sdxl["unsupportedModes"]["inpaint"]["reason"])
        self.assertIn("no reviewed execution profile", sdxl["unsupportedModes"]["inpaint"]["reason"])
        self.assertIn("live qualification", sdxl["unsupportedModes"]["inpaint"]["reason"])
        self.assertEqual(sdxl["unsupportedModes"]["inpaint"]["missingState"], [])

        flux = experimental["FluxModularPipeline"]
        self.assertNotIn("control_image", flux["runnableModes"])
        self.assertIn("control_image", flux["unsupportedModes"])

        self.assertEqual(PINNED_MODULAR_WORKFLOW_TRUTH["FluxKontextModularPipeline"].modes, ())
        self.assertEqual(PINNED_MODULAR_WORKFLOW_TRUTH["Flux2KleinModularPipeline"].modes, ())

    def test_flux2_klein_legacy_model_type_publishes_only_the_standard_execution_path(self):
        flux2 = {
            item["modelType"]: item for item in public_experimental_pipelines()
        }["Flux2KleinModularPipeline"]

        self.assertEqual(flux2["modelType"], "Flux2KleinModularPipeline")
        self.assertEqual(flux2["label"], "FLUX.2 Klein (Standard Diffusers)")
        self.assertEqual(flux2["executionKind"], "standard")
        self.assertEqual(flux2["executionModelType"], "Flux2KleinPipeline")
        self.assertEqual(flux2["pipelineClasses"], ["Flux2KleinPipeline"])
        self.assertEqual(flux2["backendPath"], "modules.DiffusersImage.LoadPipeline")
        self.assertEqual(len(flux2["executionProfiles"]), 1)
        profile = flux2["executionProfiles"][0]
        self.assertEqual(profile["id"], "flux2-klein:direct")
        self.assertEqual(profile["model_type"], "Flux2KleinPipeline")
        self.assertEqual(flux2["runnableModes"], profile["modes"])
        self.assertNotIn("Flux2KleinModularPipeline", flux2["pipelineClasses"])

    def test_dangling_experimental_profile_reference_fails_closed(self):
        invalid = {
            "modelType": "LegacyModularPipeline",
            "label": "Invalid fixture",
            "mediaKind": "image",
            "pipelineClasses": ["LegacyModularPipeline"],
            "backendPath": MODULAR_BACKEND_PATH,
            "executionKind": "standard",
            "executionProfileIds": ["missing:profile"],
            "runnableModes": ["text_to_image"],
        }
        with patch("modiff.diffusers_profiles.EXPERIMENTAL_DIFFUSERS_PIPELINES", [invalid]):
            capability = public_experimental_pipelines()[0]

        self.assertEqual(capability["executionProfiles"], [])
        self.assertEqual(capability["pipelineClasses"], [])
        self.assertEqual(capability["runnableModes"], [])
        self.assertIsNone(capability["backendPath"])
        self.assertEqual(capability["qualificationStatus"], "invalid_contract")


class FakeRequest:
    query = {}


class ModularWorkflowCapabilitySerializationTests(unittest.IsolatedAsyncioTestCase):
    async def test_public_endpoint_serializes_unsupported_and_execution_truth(self):
        response = await WebServer(module_registry.MODULE_MAP).model_capabilities(FakeRequest())
        payload = json.loads(response.text)
        experimental = {item["modelType"]: item for item in payload["experimentalCapabilities"]}

        self.assertEqual(payload["schemaVersion"], 2)
        self.assertIn("inpaint", experimental["StableDiffusionXLModularPipeline"]["unsupportedModes"])
        self.assertNotIn("inpaint", experimental["StableDiffusionXLModularPipeline"]["runnableModes"])
        self.assertEqual(
            experimental["Flux2KleinModularPipeline"]["pipelineClasses"],
            ["Flux2KleinPipeline"],
        )
        self.assertEqual(
            experimental["Flux2KleinModularPipeline"]["executionProfiles"][0]["id"],
            "flux2-klein:direct",
        )


if __name__ == "__main__":
    unittest.main()
