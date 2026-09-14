import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from diffusers.modular_pipelines.modular_pipeline_utils import InputParam, OutputParam

from modules.ModularDiffusers.modular_utils import require_modiff_node_contract
from modules.ModularDiffusers.pipeline_schema import MoDiffParam, MoDiffPipelineConfig, input_param_to_modiff_param


class PipelineSchemaTests(unittest.TestCase):
    def test_parameter_templates_serialize_without_internal_fields(self):
        prompt = MoDiffParam.prompt(default="a lighthouse")

        self.assertEqual(prompt.name, "prompt")
        self.assertEqual(
            prompt.to_dict(),
            {
                "label": "Prompt",
                "type": "string",
                "display": "textarea",
                "default": "a lighthouse",
            },
        )

    def test_output_names_are_disambiguated_from_inputs(self):
        config = MoDiffPipelineConfig(
            node_specs={
                "denoise": {
                    "inputs": [MoDiffParam.latents(display="input")],
                    "model_inputs": [],
                    "outputs": [MoDiffParam.latents(display="output")],
                    "required_inputs": ["latents"],
                    "block_name": "denoise",
                }
            }
        )

        denoise = config.node_params["denoise"]
        self.assertEqual(denoise["input_names"], ["latents"])
        self.assertEqual(denoise["output_names"], ["out_latents"])
        self.assertEqual(denoise["params"]["latents"]["label"], "Latents *")
        self.assertEqual(denoise["params"]["out_latents"]["display"], "output")

    def test_custom_block_metadata_uses_modiff_schema(self):
        input_param = SimpleNamespace(name="prompt", default="hello", metadata={"modiff": "textbox"})

        converted = input_param_to_modiff_param(input_param)

        self.assertEqual(converted.name, "prompt")
        self.assertEqual(converted.to_dict()["display"], "textarea")
        self.assertEqual(converted.to_dict()["default"], "hello")

    def test_custom_block_preserves_upstream_required_inputs(self):
        block = SimpleNamespace(
            inputs=[
                InputParam(name="prompt", type_hint=str, required=True, metadata={"modiff": "textbox"}),
                InputParam(name="strength", type_hint=float, default=0.5, required=False),
            ],
            outputs=[OutputParam(name="latents")],
            component_names=[],
        )

        config = MoDiffPipelineConfig.from_custom_block(block, node_label="Required fixture")
        custom = config.node_params["custom"]

        self.assertEqual(custom["input_names"], ["prompt", "strength"])
        self.assertEqual(custom["params"]["prompt"]["label"], "Prompt *")
        self.assertEqual(custom["params"]["strength"]["label"], "Strength")

    def test_config_round_trip_uses_modiff_owned_filename(self):
        config = MoDiffPipelineConfig(
            node_specs={"custom": None},
            label="Custom Pipeline",
            default_repo="example/pipeline",
            default_dtype="float16",
        )

        with tempfile.TemporaryDirectory() as directory:
            config.save(directory)
            config_path = Path(directory, "modiff_pipeline_config.json")

            self.assertTrue(config_path.is_file())
            self.assertEqual(MoDiffPipelineConfig.load(directory).to_dict(), config.to_dict())

    def test_default_vae_encoder_projects_only_an_upstream_generator_to_a_seed_field(self):
        for upstream_inputs, expected_inputs in (
            (["image", "generator"], ["image", "seed"]),
            (["image"], ["image"]),
        ):
            with self.subTest(upstream_inputs=upstream_inputs):
                block = SimpleNamespace(
                    input_names=upstream_inputs,
                    intermediate_output_names=["image_latents"],
                    component_names=["vae"],
                )
                blocks = SimpleNamespace(sub_blocks={"vae_encoder": block})

                node_config = MoDiffPipelineConfig.from_blocks(blocks).node_params["vae_encoder"]

                self.assertEqual(node_config["input_names"], expected_inputs)
                self.assertNotIn("generator", node_config["params"])
                if "generator" in upstream_inputs:
                    self.assertEqual(node_config["params"]["seed"]["min"], 0)
                    self.assertEqual(node_config["params"]["seed"]["max"], 4294967295)

    def test_resolved_node_contract_does_not_mutate_deserialized_custom_config(self):
        config = MoDiffPipelineConfig.from_dict(
            {
                "label": "Custom fixture",
                "node_params": {
                    "denoise": {
                        "block_name": "denoise",
                        "params": {
                            "unet": {"label": "Denoiser", "type": "diffusers_auto_model"},
                            "steps": {"label": "Steps", "type": "int", "default": 4},
                        },
                        "input_names": ["steps"],
                        "model_input_names": ["unet"],
                        "output_names": ["latents"],
                    }
                },
            }
        )
        block = object()

        class CustomPipeline:
            def __init__(self):
                self.blocks = SimpleNamespace(sub_blocks={"denoise": block})

        registry = SimpleNamespace(get=lambda _pipeline_class: config)
        with patch("modules.ModularDiffusers.modular_utils._get_registry_instance", return_value=registry):
            resolved_blocks, first = require_modiff_node_contract(CustomPipeline, "denoise")
            first["params"].pop("unet")
            first["params"]["steps"]["default"] = 99
            _, second = require_modiff_node_contract(CustomPipeline, "denoise")

        self.assertIs(resolved_blocks, block)
        self.assertIn("unet", config.node_params["denoise"]["params"])
        self.assertEqual(config.node_params["denoise"]["params"]["steps"]["default"], 4)
        self.assertIn("unet", second["params"])
        self.assertEqual(second["params"]["steps"]["default"], 4)


if __name__ == "__main__":
    unittest.main()
