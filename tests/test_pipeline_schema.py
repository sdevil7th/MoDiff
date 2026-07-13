import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()
