import inspect
import unittest
from unittest.mock import MagicMock, patch

import diffusers
import torch
from diffusers import ComponentSpec, ComponentsManager, ModularPipeline
from diffusers.modular_pipelines import InputParam, LoopSequentialPipelineBlocks, ModularPipelineBlocks, OutputParam

from modules.ModularDiffusers.modular_utils import get_all_model_types
from modules.ModularDiffusers import FLUX_BLOCKS, QWEN_IMAGE_BLOCKS, SDXL_BLOCKS
from modules.ModularDiffusers.denoise import Denoise
from modules.ModularDiffusers.dynamic_node import DynamicBlockNode
from modules.ModularDiffusers.guiders import Guider, Layers
from modules.ModularDiffusers.guiders import GUIDER_OPTIONS
from modules.ModularDiffusers.loaders import AutoModelLoader, ModelsLoader, QuantizationConfigNode
from modules.ModularDiffusers.pipeline_schema import (
    MoDiffPipelineConfig,
    input_param_to_modiff_param,
    output_param_to_modiff_param,
)


_NO_EXPLICIT_GUIDER = object()


class ModularDiffusersUpstreamContractTests(unittest.TestCase):
    """Hardware-free checks for the experimental upstream API MoDiff consumes."""

    def _run_denoise_guider_contract(self, *, guider=_NO_EXPLICIT_GUIDER, pipeline_components=("guider",)):
        pipeline = MagicMock()
        pipeline.component_names = list(pipeline_components)
        pipeline._execution_device = "cpu"
        pipeline.transformer = None
        pipeline.return_value = {}

        blocks = MagicMock()
        blocks.component_names = ["guider"]
        blocks.input_names = []
        blocks.init_pipeline.return_value = pipeline
        node_config = {
            "params": {"guidance_scale": {"type": "float"}},
            "input_names": ["guidance_scale"],
            "model_input_names": ["unet", "guider"],
            "output_names": [],
        }
        kwargs = {
            "unet": {"repo_id": None},
            "guidance_scale": 4.5,
        }
        if guider is not _NO_EXPLICIT_GUIDER:
            kwargs["guider"] = guider

        node = Denoise("guider-install-contract")
        node._pipeline_class = object()
        with (
            patch(
                "modules.ModularDiffusers.denoise.pipeline_class_to_modiff_node_config",
                return_value=(blocks, node_config),
            ),
            patch("modules.ModularDiffusers.denoise.deepcopy", return_value=blocks),
            patch("modules.ModularDiffusers.denoise.insert_preview_block"),
        ):
            result = node.execute(**kwargs)

        return pipeline, result

    def test_core_symbols_and_loader_signature_are_present(self):
        self.assertTrue(inspect.isclass(ModularPipeline))
        self.assertTrue(inspect.isclass(ComponentsManager))
        self.assertTrue(inspect.isclass(ComponentSpec))
        self.assertTrue(inspect.isclass(ModularPipelineBlocks))
        self.assertTrue(inspect.isclass(LoopSequentialPipelineBlocks))

        loader_params = inspect.signature(ModularPipeline.from_pretrained).parameters
        self.assertIn("pretrained_model_name_or_path", loader_params)
        self.assertIn("trust_remote_code", loader_params)
        self.assertIn("components_manager", loader_params)
        self.assertIn("collection", loader_params)

    def test_every_remote_code_loader_exposes_an_immutable_revision_field(self):
        for loader in (DynamicBlockNode, AutoModelLoader, ModelsLoader):
            with self.subTest(loader=loader.__name__):
                self.assertIn("trust_remote_code", loader.params)
                self.assertIn("revision", loader.params)

    def test_public_guider_registry_exposes_resolved_options_mapping(self):
        from modules import MODULE_MAP

        options = MODULE_MAP["modules.ModularDiffusers"]["Guider"]["params"]["guider"]["options"]
        self.assertIsInstance(options, dict)
        self.assertEqual(options, GUIDER_OPTIONS)

    def test_reviewed_dynamic_block_resolves_its_catalog_revision(self):
        node = DynamicBlockNode("dynamic-revision-probe")
        with patch(
            "modules.ModularDiffusers.dynamic_node.PipelineConfig.load",
            return_value=object(),
        ) as load_config:
            node._get_custom_config("diffusers/FLUX.2-klein-4B-modular")

        load_config.assert_called_once_with(
            "diffusers/FLUX.2-klein-4B-modular",
            revision="62ac375aa5308588f111fcd12115f5c54a8b1f4f",
        )

    def test_models_loader_resolves_known_base_revision(self):
        node = ModelsLoader("modular-revision-probe")
        with (
            patch("modules.ModularDiffusers.loaders.configure_components_manager_offload"),
            patch(
                "modules.ModularDiffusers.loaders.ModularPipeline.from_pretrained",
                side_effect=RuntimeError("stop after loader call"),
            ) as loader,
        ):
            with self.assertRaisesRegex(RuntimeError, "stop after loader call"):
                node.execute(
                    model_type="ZImageModularPipeline",
                    repo_id={"source": "hub", "value": "Tongyi-MAI/Z-Image-Turbo"},
                    device="cpu",
                    dtype="float16",
                    trust_remote_code=False,
                    auto_offload=False,
                    offload_mode="none",
                )

        self.assertEqual(
            loader.call_args.kwargs["revision"],
            "f332072aa78be7aecdf3ee76d5c247082da564a6",
        )

    def test_quantization_layer_probe_uses_the_public_auto_model_boundary(self):
        fake_model = unittest.mock.Mock()
        fake_model.named_modules.return_value = [
            ("transformer_blocks.0.attn.to_q", torch.nn.Linear(2, 2)),
            ("transformer_blocks.0.norm", torch.nn.LayerNorm(2)),
        ]
        node = QuantizationConfigNode("quant-layer-contract")

        with (
            patch("diffusers.AutoModel.load_config", return_value={"_class_name": "FixtureModel"}),
            patch("diffusers.AutoModel.from_config", return_value=fake_model) as from_config,
        ):
            layers = node._get_model_layers("unit/model", "transformer")

        from_config.assert_called_once_with({"_class_name": "FixtureModel"})
        self.assertEqual(layers, {"transformer_blocks.0": ["transformer_blocks.0.attn.to_q"]})
        self.assertNotIn("pipeline_loading_utils", inspect.getsource(QuantizationConfigNode._get_model_layers))

    def test_input_and_output_metadata_converts_to_graph_fields(self):
        prompt = input_param_to_modiff_param(
            InputParam(name="prompt", default="", required=True, metadata={"modiff": "textbox"})
        )
        image = output_param_to_modiff_param(OutputParam(name="images", metadata={"modiff": "image"}))

        self.assertEqual(prompt.name, "prompt")
        self.assertEqual(prompt.display, "textarea")
        self.assertEqual(image.name, "images")
        self.assertEqual(image.type, "image")
        self.assertEqual(image.display, "output")

    def test_modiff_schema_round_trip_does_not_require_model_weights(self):
        config = MoDiffPipelineConfig(
            node_specs={
                "encode": {
                    "inputs": [
                        input_param_to_modiff_param(
                            InputParam(name="prompt", default="", metadata={"modiff": "textbox"})
                        )
                    ],
                    "outputs": [
                        output_param_to_modiff_param(OutputParam(name="images", metadata={"modiff": "image"}))
                    ],
                    "required_inputs": ["prompt"],
                    "block_name": "text_encoder",
                }
            },
            label="Contract fixture",
            default_repo="local/fixture",
            default_dtype="bfloat16",
        )

        restored = MoDiffPipelineConfig.from_dict(config.to_dict())
        self.assertEqual(restored.to_dict(), config.to_dict())
        self.assertEqual(restored.node_params["encode"]["block_name"], "text_encoder")
        self.assertIn("prompt", restored.node_params["encode"]["params"])

    def test_required_pipeline_registry_matches_installed_diffusers(self):
        required = {
            "StableDiffusionXLModularPipeline",
            "QwenImageModularPipeline",
            "QwenImageEditModularPipeline",
            "QwenImageEditPlusModularPipeline",
            "QwenImageLayeredModularPipeline",
            "FluxModularPipeline",
            "FluxKontextModularPipeline",
            "Flux2KleinModularPipeline",
            "ZImageModularPipeline",
            "WanModularPipeline",
            "WanImage2VideoModularPipeline",
        }

        missing_exports = sorted(name for name in required if not hasattr(diffusers, name))
        self.assertEqual(missing_exports, [], f"Installed Diffusers removed exports: {missing_exports}")
        registered = set(get_all_model_types())
        self.assertTrue(required.issubset(registered), f"MoDiff registry is missing: {sorted(required - registered)}")

    def test_layer_options_identify_module_list_stacks(self):
        self.assertEqual(QWEN_IMAGE_BLOCKS, ["transformer_blocks"])
        self.assertEqual(FLUX_BLOCKS, ["transformer_blocks", "single_transformer_blocks"])
        self.assertTrue(SDXL_BLOCKS)
        self.assertTrue(all(value.endswith(".transformer_blocks") for value in SDXL_BLOCKS))
        self.assertTrue(all(value == value.strip() for value in [*SDXL_BLOCKS, *QWEN_IMAGE_BLOCKS, *FLUX_BLOCKS]))

    def test_layers_preserve_exact_stack_fqn_and_validate_indices(self):
        node = object.__new__(Layers)
        output = node.execute(
            blocks_select=["transformer_blocks"],
            transformer_blocks={"indices": "0, 18", "dropout": 0.5},
        )

        self.assertEqual(
            output["layers_config"],
            [
                {
                    "indices": [0, 18],
                    "fqn": "transformer_blocks",
                    "dropout": 0.5,
                    "skip_attention": False,
                    "skip_attention_scores": False,
                    "skip_ff": False,
                }
            ],
        )
        with self.assertRaisesRegex(ValueError, "comma-separated integers"):
            node.execute(transformer_blocks={"indices": "zero"})

    def test_layer_dependent_guiders_require_an_explicit_nonempty_selection(self):
        node = object.__new__(Guider)
        node.node_id = "guider-contract"

        for guider in ("SkipLayerGuidance", "AutoGuidance", "SmoothedEnergyGuidance"):
            with self.subTest(guider=guider), self.assertRaisesRegex(ValueError, "non-empty Layers connection"):
                node.execute(guider, layers_config=[])

    def test_guider_converts_validated_layer_mapping_to_upstream_config(self):
        node = object.__new__(Guider)
        node.node_id = "guider-contract"
        with patch.object(diffusers, "SkipLayerGuidance", return_value="configured") as constructor:
            result = node.execute(
                "SkipLayerGuidance",
                layers_config=[
                    {
                        "indices": [1, 3],
                        "fqn": "transformer_blocks",
                        "dropout": 1.0,
                        "skip_attention": False,
                        "skip_attention_scores": True,
                        "skip_ff": False,
                    }
                ],
            )

        self.assertEqual(result, {"guider_out": "configured"})
        config = constructor.call_args.kwargs["skip_layer_config"][0]
        self.assertEqual(config.indices, [1, 3])
        self.assertEqual(config.fqn, "transformer_blocks")

    def test_frequency_decoupled_guider_uses_upstream_plural_scale_argument(self):
        node = object.__new__(Guider)
        node.node_id = "frequency-guider-contract"
        with patch.object(diffusers, "FrequencyDecoupledGuidance", return_value="configured") as constructor:
            result = node.execute("FrequencyDecoupledGuidance", guidance_scale=4.5)

        self.assertEqual(result, {"guider_out": "configured"})
        self.assertEqual(constructor.call_args.kwargs["guidance_scales"], [4.5])
        self.assertNotIn("guidance_scale", constructor.call_args.kwargs)

    def test_denoise_installs_connected_diffusers_guider_without_legacy_overwrite(self):
        guider = diffusers.ClassifierFreeGuidance(guidance_scale=7.0)

        pipeline, result = self._run_denoise_guider_contract(guider=guider)

        self.assertEqual(result, {})
        pipeline.update_components.assert_called_once_with(guider=guider)
        pipeline.get_component_spec.assert_not_called()

    def test_denoise_keeps_guidance_scale_fallback_without_model_ids_or_explicit_guider(self):
        pipeline, result = self._run_denoise_guider_contract()
        created_guider = pipeline.get_component_spec.return_value.create.return_value

        # No model IDs exercise the formerly uninitialized component-update path.
        self.assertEqual(result, {})
        pipeline.get_component_spec.assert_called_once_with("guider")
        pipeline.get_component_spec.return_value.create.assert_called_once_with(guidance_scale=4.5)
        pipeline.update_components.assert_called_once_with(guider=created_guider)

    def test_denoise_rejects_non_diffusers_guider_with_actionable_error(self):
        with self.assertRaisesRegex(TypeError, "Diffusers BaseGuidance instance"):
            self._run_denoise_guider_contract(guider=object())

    def test_denoise_rejects_guider_for_pipeline_without_guider_component(self):
        guider = diffusers.ClassifierFreeGuidance(guidance_scale=7.0)

        with self.assertRaisesRegex(ValueError, "does not expose a 'guider' component"):
            self._run_denoise_guider_contract(guider=guider, pipeline_components=())


if __name__ == "__main__":
    unittest.main()
