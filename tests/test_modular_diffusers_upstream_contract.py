import inspect
import unittest
from unittest.mock import MagicMock, patch

import diffusers
import torch
from diffusers import ComponentSpec, ComponentsManager, ModularPipeline
from diffusers.modular_pipelines import InputParam, LoopSequentialPipelineBlocks, ModularPipelineBlocks, OutputParam

from modiff.diffusers_profiles import public_execution_profiles, public_experimental_pipelines
from modules import MODULE_MAP
from modules.ModularDiffusers.modular_utils import (
    get_all_model_types,
    get_modular_layer_block_options,
    get_model_type_metadata,
    require_modiff_node_contract,
)
from modules.ModularDiffusers import FLUX_BLOCKS, MODULAR_LAYER_BLOCK_OPTIONS, QWEN_IMAGE_BLOCKS, SDXL_BLOCKS
from modules.ModularDiffusers.controlnet import Controlnet
from modules.ModularDiffusers.denoise import Denoise
from modules.ModularDiffusers.dynamic_node import DynamicBlockNode
from modules.ModularDiffusers.embeddings import EncodePrompt, ImageEmbeddings
from modules.ModularDiffusers.guiders import GUIDER_CONFIGS, GUIDER_OPTIONS, LAYER_CONFIG_MAPPING, Guider, Layers
from modules.ModularDiffusers.latents import DecodeLatents, ImageEncode
from modules.ModularDiffusers.loaders import (
    AutoModelLoader,
    ModelsLoader,
    QuantizationConfigNode,
    _reviewed_loader_component_outputs,
)
from modules.ModularDiffusers.pipeline_schema import (
    MoDiffParam,
    MoDiffPipelineConfig,
    input_param_to_modiff_param,
    output_param_to_modiff_param,
)
from modules.ModularDiffusers.route_state import ROUTE_STATE_INPUT, ROUTE_STATE_OUTPUT


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
                "modules.ModularDiffusers.denoise.require_modiff_node_contract",
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

    def test_pinned_guider_exports_and_constructor_signatures_are_exact(self):
        expected_signatures = {
            "AdaptiveProjectedMixGuidance": [
                "guidance_scale",
                "guidance_rescale",
                "adaptive_projected_guidance_scale",
                "adaptive_projected_guidance_momentum",
                "adaptive_projected_guidance_rescale",
                "eta",
                "use_original_formulation",
                "start",
                "stop",
                "adaptive_projected_guidance_start_step",
                "enabled",
            ],
            "PerturbedAttentionGuidance": [
                "guidance_scale",
                "perturbed_guidance_scale",
                "perturbed_guidance_start",
                "perturbed_guidance_stop",
                "perturbed_guidance_layers",
                "perturbed_guidance_config",
                "guidance_rescale",
                "use_original_formulation",
                "start",
                "stop",
                "enabled",
            ],
        }
        expected_defaults = {
            "AdaptiveProjectedMixGuidance": {
                "guidance_scale": 3.5,
                "adaptive_projected_guidance_scale": 10.0,
                "adaptive_projected_guidance_momentum": -0.5,
                "adaptive_projected_guidance_rescale": 10.0,
                "eta": 0.0,
                "adaptive_projected_guidance_start_step": 5,
            },
            "PerturbedAttentionGuidance": {
                "guidance_scale": 7.5,
                "perturbed_guidance_scale": 2.8,
                "perturbed_guidance_start": 0.01,
                "perturbed_guidance_stop": 0.2,
                "perturbed_guidance_layers": None,
                "perturbed_guidance_config": None,
            },
        }

        for guider_name, expected_parameters in expected_signatures.items():
            with self.subTest(guider=guider_name):
                self.assertTrue(hasattr(diffusers, guider_name))
                signature = inspect.signature(getattr(diffusers, guider_name))
                self.assertEqual(list(signature.parameters), expected_parameters)
                for parameter_name, default in expected_defaults[guider_name].items():
                    self.assertEqual(signature.parameters[parameter_name].default, default)

        self.assertIn("AdaptiveProjectedMixGuidance", GUIDER_OPTIONS)
        self.assertIn("PerturbedAttentionGuidance", GUIDER_OPTIONS)
        self.assertNotIn("MagnitudeAwareGuidance", GUIDER_OPTIONS)
        self.assertFalse(hasattr(diffusers, "MagnitudeAwareGuidance"))
        self.assertEqual(
            set(GUIDER_CONFIGS["AdaptiveProjectedMixGuidance"]),
            {
                "adaptive_projected_guidance_scale",
                "adaptive_projected_guidance_momentum",
                "adaptive_projected_guidance_rescale",
                "eta",
                "adaptive_projected_guidance_start_step",
            },
        )
        self.assertEqual(
            set(GUIDER_CONFIGS["PerturbedAttentionGuidance"]),
            {"perturbed_guidance_scale", "perturbed_guidance_start", "perturbed_guidance_stop"},
        )
        self.assertEqual(
            LAYER_CONFIG_MAPPING["PerturbedAttentionGuidance"],
            "perturbed_guidance_config",
        )

    def test_reviewed_dynamic_block_resolves_its_catalog_revision(self):
        node = DynamicBlockNode("dynamic-revision-probe")
        verified = MagicMock()
        verified.config = object()
        with patch(
            "modules.ModularDiffusers.dynamic_node.PipelineConfig.load_verified",
            return_value=verified,
        ) as load_verified:
            node._get_custom_config("diffusers/FLUX.2-klein-4B-modular")

        load_verified.assert_called_once_with(
            "diffusers/FLUX.2-klein-4B-modular",
            source="hub",
            revision="62ac375aa5308588f111fcd12115f5c54a8b1f4f",
        )

    def test_models_loader_resolves_known_base_revision(self):
        node = ModelsLoader("modular-revision-probe")
        with (
            patch("modules.ModularDiffusers.loaders.configure_components_manager_offload"),
            patch(
                "modules.ModularDiffusers.loaders._validate_reviewed_pipeline_index",
                return_value=("model_index.json", {"_class_name": "ZImagePipeline"}),
            ) as validate_index,
            patch(
                "modules.ModularDiffusers.loaders._instantiate_reviewed_builtin_pipeline",
                side_effect=RuntimeError("stop after loader call"),
            ),
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

        validate_index.assert_called_once_with(
            "ZImageModularPipeline",
            "Tongyi-MAI/Z-Image-Turbo",
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
            loader_component_outputs=("image_encoder",),
            layer_block_options=("transformer_blocks",),
        )

        restored = MoDiffPipelineConfig.from_dict(config.to_dict())
        self.assertEqual(restored.to_dict(), config.to_dict())
        self.assertEqual(restored.loader_component_outputs, ("image_encoder",))
        self.assertEqual(restored.layer_block_options, ("transformer_blocks",))
        self.assertEqual(restored.node_params["encode"]["block_name"], "text_encoder")
        self.assertIn("prompt", restored.node_params["encode"]["params"])

    def test_models_loader_component_outputs_come_from_reviewed_pipeline_metadata(self):
        registered = set(get_all_model_types()) - {"", "DummyCustomPipeline"}
        expected = {"WanImage2VideoModularPipeline": ("image_encoder",)}

        for model_type in sorted(registered):
            with self.subTest(model_type=model_type):
                outputs = _reviewed_loader_component_outputs(model_type)
                self.assertEqual(outputs, expected.get(model_type, ()))
                self.assertEqual(get_model_type_metadata(model_type)["loader_component_outputs"], list(outputs))

        self.assertNotIn(
            'model_type == "WanImage2VideoModularPipeline"',
            inspect.getsource(ModelsLoader.execute),
        )

    def test_models_loader_rejects_malformed_component_output_metadata(self):
        with patch(
            "modules.ModularDiffusers.loaders.get_model_type_metadata",
            return_value={"loader_component_outputs": ["not_a_loader_output"]},
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid loader component output contract"):
                _reviewed_loader_component_outputs("FixturePipeline")

        with self.assertRaisesRegex(ValueError, "list or tuple"):
            MoDiffPipelineConfig(node_specs={}, loader_component_outputs="image_encoder")
        with self.assertRaisesRegex(ValueError, "list or tuple"):
            MoDiffPipelineConfig(node_specs={}, layer_block_options="transformer_blocks")

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

    def test_registered_pipeline_action_matrix_resolves_real_contracts(self):
        expected = {
            "StableDiffusionXLModularPipeline": {"controlnet", "decoder", "denoise", "text_encoder", "vae_encoder"},
            "QwenImageModularPipeline": {"controlnet", "decoder", "denoise", "text_encoder", "vae_encoder"},
            "QwenImageEditModularPipeline": {"decoder", "denoise", "text_encoder", "vae_encoder"},
            "QwenImageEditPlusModularPipeline": {"decoder", "denoise", "text_encoder", "vae_encoder"},
            "QwenImageLayeredModularPipeline": {"decoder", "denoise", "text_encoder", "vae_encoder"},
            "FluxModularPipeline": {"decoder", "denoise", "text_encoder", "vae_encoder"},
            "FluxKontextModularPipeline": {"decoder", "denoise", "text_encoder", "vae_encoder"},
            "Flux2KleinModularPipeline": {"decoder", "denoise", "text_encoder", "vae_encoder"},
            "ZImageModularPipeline": {"decoder", "denoise", "text_encoder", "vae_encoder"},
            "WanModularPipeline": {"decoder", "denoise", "text_encoder"},
            "WanImage2VideoModularPipeline": {
                "decoder",
                "denoise",
                "image_encoder",
                "text_encoder",
                "vae_encoder",
            },
        }
        registered = set(get_all_model_types()) - {"", "DummyCustomPipeline"}
        self.assertEqual(registered, set(expected))

        actions = {"controlnet", "decoder", "denoise", "image_encoder", "text_encoder", "vae_encoder"}
        for model_type, supported_actions in expected.items():
            pipeline_class = getattr(diffusers, model_type)
            metadata = get_model_type_metadata(model_type)
            actual_actions = {name for name, config in metadata["node_params"].items() if config is not None}
            self.assertEqual(actual_actions, supported_actions)

            for action in actions:
                with self.subTest(model_type=model_type, action=action):
                    if action in supported_actions:
                        blocks, node_config = require_modiff_node_contract(
                            pipeline_class,
                            action,
                            require_blocks=action != "controlnet",
                        )
                        self.assertIsNotNone(node_config)
                        if action != "controlnet":
                            self.assertIsNotNone(blocks)
                    else:
                        with self.assertRaisesRegex(ValueError, "does not support the generic"):
                            require_modiff_node_contract(
                                pipeline_class,
                                action,
                                require_blocks=action != "controlnet",
                            )

    def test_qwen_layered_targeted_controls_match_the_pinned_upstream_contract_without_weights(self):
        expected_defaults = {
            "text_encoder": {
                "resolution": 640,
                "use_en_prompt": False,
                "max_sequence_length": 1024,
            },
            "vae_encoder": {"resolution": 640},
        }
        expected_modiff_inputs = {
            "text_encoder": {
                "prompt",
                "negative_prompt",
                "image",
                "resolution",
                "use_en_prompt",
                "max_sequence_length",
            },
            "vae_encoder": {"image", "resolution", "seed"},
        }
        runtime_input_aliases = {
            "text_encoder": {},
            "vae_encoder": {"seed": "generator"},
        }

        for action, defaults in expected_defaults.items():
            with self.subTest(action=action):
                blocks, node_config = require_modiff_node_contract(
                    diffusers.QwenImageLayeredModularPipeline,
                    action,
                )
                upstream_inputs = {param.name: param for param in blocks.inputs}
                aliases = runtime_input_aliases[action]
                self.assertEqual(
                    set(blocks.input_names),
                    (expected_modiff_inputs[action] - set(aliases)) | set(aliases.values()),
                )
                self.assertEqual(set(node_config["input_names"]), expected_modiff_inputs[action])
                for field_name, default in defaults.items():
                    self.assertIn(field_name, upstream_inputs)
                    self.assertEqual(upstream_inputs[field_name].default, default)

        metadata = get_model_type_metadata("QwenImageLayeredModularPipeline")
        text_params = metadata["node_params"]["text_encoder"]["params"]
        vae_params = metadata["node_params"]["vae_encoder"]["params"]

        for params in (text_params, vae_params):
            self.assertEqual(
                params["resolution"],
                {
                    "label": "Source Resolution",
                    "type": "int",
                    "default": 640,
                    "options": [640, 1024],
                    "fieldOptions": {
                        "controlTier": "advanced",
                        "studioBinding": {
                            "schemaVersion": 1,
                            "group": "source-resolution",
                            "formFields": ["width", "height"],
                            "transform": "nearest-option-to-long-edge",
                        },
                    },
                },
            )
        self.assertEqual(text_params["use_en_prompt"]["default"], False)
        self.assertEqual(text_params["use_en_prompt"]["type"], "boolean")
        self.assertEqual(
            vae_params["seed"],
            {
                "label": "Seed",
                "type": "int",
                "default": 0,
                "min": 0,
                "max": 4294967295,
                "display": "random",
            },
        )
        self.assertEqual(
            text_params["max_sequence_length"],
            {
                "label": "Maximum Sequence Length",
                "type": "int",
                "default": 1024,
                "min": 1,
                "max": 1024,
                "step": 1,
                "fieldOptions": {
                    "controlTier": "advanced",
                    "studioBinding": {
                        "schemaVersion": 1,
                        "group": "maximum-sequence-length",
                        "formFields": ["maxSequenceLength"],
                        "transform": "identity",
                    },
                },
            },
        )

    def test_registered_encoder_seed_ports_exactly_match_upstream_generator_inputs(self):
        expected_generator_actions = {
            ("StableDiffusionXLModularPipeline", "vae_encoder"),
            ("QwenImageModularPipeline", "vae_encoder"),
            ("QwenImageEditModularPipeline", "vae_encoder"),
            ("QwenImageEditPlusModularPipeline", "vae_encoder"),
            ("QwenImageLayeredModularPipeline", "vae_encoder"),
            ("FluxModularPipeline", "vae_encoder"),
            ("FluxKontextModularPipeline", "vae_encoder"),
            ("Flux2KleinModularPipeline", "vae_encoder"),
            ("ZImageModularPipeline", "vae_encoder"),
            ("WanImage2VideoModularPipeline", "vae_encoder"),
        }
        actual_generator_actions = set()

        for model_type in sorted(set(get_all_model_types()) - {"", "DummyCustomPipeline"}):
            metadata = get_model_type_metadata(model_type)
            for action in ("vae_encoder", "image_encoder"):
                if metadata["node_params"].get(action) is None:
                    continue
                with self.subTest(model_type=model_type, action=action):
                    blocks, node_config = require_modiff_node_contract(getattr(diffusers, model_type), action)
                    upstream_has_generator = "generator" in blocks.input_names
                    graph_has_seed = "seed" in node_config["input_names"]
                    self.assertEqual(graph_has_seed, upstream_has_generator)
                    self.assertNotIn("generator", node_config["input_names"])
                    self.assertNotIn("generator", node_config["params"])
                    if upstream_has_generator:
                        actual_generator_actions.add((model_type, action))
                        self.assertEqual(
                            node_config["params"]["seed"],
                            {
                                "label": "Seed",
                                "type": "int",
                                "default": 0,
                                "min": 0,
                                "max": 4294967295,
                                "display": "random",
                            },
                        )

        self.assertEqual(actual_generator_actions, expected_generator_actions)

    def test_registered_denoise_component_ports_exactly_match_upstream_blocks(self):
        port_components = {
            "unet": {"transformer", "unet"},
            "vae": {"vae"},
            "scheduler": {"scheduler"},
            "guider": {"guider"},
            "controlnet_bundle": {"controlnet"},
        }
        registered = set(get_all_model_types()) - {"", "DummyCustomPipeline"}
        provenance_only_ports = {"WanImage2VideoModularPipeline": {"vae"}}
        self.assertEqual(MoDiffParam.guider().required_block_params, ["guider"])

        for model_type in sorted(registered):
            with self.subTest(model_type=model_type):
                blocks, node_config = require_modiff_node_contract(getattr(diffusers, model_type), "denoise")
                upstream_components = set(blocks.component_names)
                expected_ports = {
                    port for port, component_names in port_components.items() if component_names & upstream_components
                }
                expected_ports.update(provenance_only_ports.get(model_type, ()))
                self.assertEqual(set(node_config["model_input_names"]), expected_ports)

        wan_blocks, wan_config = require_modiff_node_contract(
            diffusers.WanImage2VideoModularPipeline,
            "denoise",
        )
        self.assertNotIn("vae", wan_blocks.component_names)
        self.assertIn("vae", wan_config["model_input_names"])
        self.assertTrue(wan_config["params"]["vae"]["label"].endswith(" *"))

        self.assertEqual(
            set(get_model_type_metadata("FluxModularPipeline")["node_params"]["denoise"]["model_input_names"]),
            {"unet", "scheduler"},
        )
        self.assertEqual(
            set(get_model_type_metadata("WanModularPipeline")["node_params"]["denoise"]["model_input_names"]),
            {"unet", "guider", "scheduler"},
        )

    def test_advertised_modular_control_modes_have_a_registered_node_contract(self):
        advertised_control_models = {
            capability["modelType"]
            for capability in public_experimental_pipelines()
            if "control_image" in capability["runnableModes"]
        }
        advertised_control_models.update(
            profile["model_type"]
            for profile in public_execution_profiles()
            if "control_image" in profile["modes"]
            and profile["backend_path"] == "modules.ModularDiffusers.ModelsLoader"
        )
        self.assertTrue(advertised_control_models)

        for model_type in sorted(advertised_control_models):
            with self.subTest(model_type=model_type):
                metadata = get_model_type_metadata(model_type)
                self.assertIsNotNone(metadata, f"{model_type} is advertised but not registered")
                self.assertIsNotNone(
                    metadata["node_params"].get("controlnet"),
                    f"{model_type}:control_image is advertised without a ControlNet node contract",
                )

    def test_controlnet_model_signal_is_a_generic_passthrough(self):
        expected_actions = [
            {"action": "value", "target": "model_type"},
            {"action": "exec", "data": "update_node"},
        ]
        self.assertEqual(Controlnet.params["controlnet_bundle"]["onSignal"], expected_actions)

        node = Controlnet("generic-controlnet-signal")
        node.send_node_definition = MagicMock()
        node.update_node({"model_type": "QwenImageModularPipeline"}, None)

        refreshed = node.send_node_definition.call_args.args[0]
        self.assertEqual(refreshed["controlnet_bundle"]["onSignal"], expected_actions)

    def test_controlnet_update_rejects_an_unsupported_modular_pipeline(self):
        node = Controlnet("unsupported-controlnet-update")

        for _ in range(2):
            with self.assertRaisesRegex(
                ValueError,
                "FluxModularPipeline.*does not support the generic ControlNet node",
            ):
                node.update_node({"model_type": "FluxModularPipeline"}, None)

    def test_controlnet_execution_rejects_a_stale_unsupported_graph(self):
        node = Controlnet("unsupported-controlnet-execute")

        with self.assertRaisesRegex(
            ValueError,
            "FluxModularPipeline.*does not support the generic ControlNet node",
        ):
            node.execute(unet={"model_type": "FluxModularPipeline"})

    def test_every_generic_modular_action_rejects_unsupported_updates_consistently(self):
        cases = (
            (EncodePrompt, "Encode Prompt", False),
            (ImageEmbeddings, "Image Embeddings", False),
            (ImageEncode, "Encode Image", False),
            (Denoise, "Denoise", False),
            (DecodeLatents, "Decode Latents", False),
            (Controlnet, "ControlNet", True),
        )

        with patch(
            "modules.ModularDiffusers.modular_utils.pipeline_class_to_modiff_node_config",
            return_value=(None, None),
        ):
            for node_class, action_label, uses_explicit_model_type in cases:
                with self.subTest(node=node_class.__name__):
                    node = node_class(f"unsupported-{node_class.__name__}")
                    node.send_node_definition = MagicMock()
                    if not uses_explicit_model_type:
                        node.get_signal_value = MagicMock(return_value="FluxModularPipeline")

                    for _ in range(2):
                        with self.assertRaisesRegex(
                            ValueError,
                            f"FluxModularPipeline.*does not support the generic {action_label} node",
                        ):
                            node.update_node(
                                {"model_type": "FluxModularPipeline"} if uses_explicit_model_type else {},
                                None,
                            )

                    self.assertEqual(node._model_type, "")
                    self.assertIsNone(node._pipeline_class)
                    self.assertEqual(node.send_node_definition.call_count, 2)

    def test_every_generic_modular_action_rejects_unknown_model_types_during_update(self):
        cases = (
            (EncodePrompt, False),
            (ImageEmbeddings, False),
            (ImageEncode, False),
            (Denoise, False),
            (DecodeLatents, False),
            (Controlnet, True),
        )

        for node_class, uses_explicit_model_type in cases:
            with self.subTest(node=node_class.__name__):
                node = node_class(f"unknown-{node_class.__name__}")
                node._model_type = "FluxModularPipeline"
                node._pipeline_class = diffusers.FluxModularPipeline
                node.send_node_definition = MagicMock()
                if not uses_explicit_model_type:
                    node.get_signal_value = MagicMock(return_value="FutureModularPipeline")

                for _ in range(2):
                    with self.assertRaisesRegex(
                        ValueError,
                        "Unknown Diffusers modular pipeline class 'FutureModularPipeline'.*"
                        "refresh the node definition",
                    ):
                        node.update_node(
                            {"model_type": "FutureModularPipeline"} if uses_explicit_model_type else {},
                            None,
                        )

                self.assertEqual(node._model_type, "")
                self.assertIsNone(node._pipeline_class)
                self.assertEqual(node.send_node_definition.call_count, 2)

    def test_every_generic_modular_action_rejects_unsupported_execution_consistently(self):
        cases = (
            (EncodePrompt, "Encode Prompt", "text_encoders"),
            (ImageEmbeddings, "Image Embeddings", "image_encoder"),
            (ImageEncode, "Encode Image", "vae"),
            (Denoise, "Denoise", "unet"),
            (DecodeLatents, "Decode Latents", "vae"),
            (Controlnet, "ControlNet", "unet"),
        )

        with patch(
            "modules.ModularDiffusers.modular_utils.pipeline_class_to_modiff_node_config",
            return_value=(None, None),
        ):
            for node_class, action_label, connector in cases:
                with self.subTest(node=node_class.__name__):
                    node = node_class(f"stale-{node_class.__name__}")
                    runtime_component = {
                        "model_type": "FluxModularPipeline",
                        "repo_id": "local/fixture",
                    }
                    with self.assertRaisesRegex(
                        ValueError,
                        f"FluxModularPipeline.*does not support the generic {action_label} node",
                    ):
                        node.execute(**{connector: runtime_component})

    def test_sdxl_controlnet_keeps_its_supported_bundle_only_contract(self):
        pipeline_class = diffusers.StableDiffusionXLModularPipeline
        blocks, node_config = require_modiff_node_contract(
            pipeline_class,
            "controlnet",
            require_blocks=False,
        )

        self.assertIsNone(blocks)
        self.assertIsNotNone(node_config)
        self.assertFalse({"seed", ROUTE_STATE_INPUT}.intersection(node_config["input_names"]))
        self.assertNotIn(ROUTE_STATE_OUTPUT, node_config["output_names"])
        controlnet_component = {
            "model_id": "fixture-controlnet-id",
            "repo_id": "local/controlnet-fixture",
        }
        result = Controlnet("sdxl-bundle-only").execute(
            model_type=pipeline_class.__name__,
            controlnet=controlnet_component,
            control_image="fixture-control-image",
            controlnet_conditioning_scale=0.75,
            control_guidance_start=0.1,
            control_guidance_end=0.9,
        )
        self.assertEqual(
            result,
            {
                "controlnet_bundle": {
                    "controlnet": controlnet_component,
                    "control_image": "fixture-control-image",
                    "controlnet_conditioning_scale": 0.75,
                    "control_guidance_start": 0.1,
                    "control_guidance_end": 0.9,
                }
            },
        )

    def test_qwen_controlnet_generator_is_closed_by_seed_and_optional_opaque_route(self):
        blocks, node_config = require_modiff_node_contract(
            diffusers.QwenImageModularPipeline,
            "controlnet",
        )

        self.assertIn("generator", blocks.input_names)
        self.assertNotIn("seed", blocks.input_names)
        self.assertIn("seed", node_config["input_names"])
        self.assertIn(ROUTE_STATE_INPUT, node_config["input_names"])
        self.assertIn(ROUTE_STATE_OUTPUT, node_config["output_names"])
        self.assertFalse(node_config["params"][ROUTE_STATE_INPUT]["label"].endswith("*"))
        self.assertFalse(
            {"image_latents", "image_latents_with_strength", "strength"}.intersection(node_config["input_names"])
        )
        self.assertIn("control_image_latents", blocks.output_names)

    def test_layer_options_identify_module_list_stacks(self):
        self.assertEqual(QWEN_IMAGE_BLOCKS, ["transformer_blocks"])
        self.assertEqual(FLUX_BLOCKS, ["transformer_blocks", "single_transformer_blocks"])
        self.assertTrue(SDXL_BLOCKS)
        self.assertTrue(all(value.endswith(".transformer_blocks") for value in SDXL_BLOCKS))
        self.assertTrue(all(value == value.strip() for value in [*SDXL_BLOCKS, *QWEN_IMAGE_BLOCKS, *FLUX_BLOCKS]))

        expected = {
            "StableDiffusionXLModularPipeline": SDXL_BLOCKS,
            "QwenImageModularPipeline": QWEN_IMAGE_BLOCKS,
            "QwenImageEditModularPipeline": QWEN_IMAGE_BLOCKS,
            "QwenImageEditPlusModularPipeline": QWEN_IMAGE_BLOCKS,
            "FluxModularPipeline": FLUX_BLOCKS,
            "FluxKontextModularPipeline": FLUX_BLOCKS,
        }
        self.assertEqual(get_modular_layer_block_options(), expected)
        self.assertEqual(MODULAR_LAYER_BLOCK_OPTIONS, expected)
        self.assertEqual(Layers.params["layers_config"]["onSignal"]["data"], expected)
        self.assertEqual(
            MODULE_MAP["modules.ModularDiffusers"]["Layers"]["params"]["layers_config"]["onSignal"]["data"],
            expected,
        )
        for model_type in set(get_all_model_types()) - {"", "DummyCustomPipeline"}:
            self.assertEqual(
                get_model_type_metadata(model_type)["layer_block_options"],
                expected.get(model_type, []),
            )
        layer_source = inspect.getsource(Layers)
        self.assertNotIn("QwenImageModularPipeline", layer_source)
        self.assertNotIn("FluxModularPipeline", layer_source)

    def test_layers_preserve_exact_stack_fqn_and_validate_indices(self):
        node = object.__new__(Layers)
        with patch.object(Layers, "get_signal_value", return_value="QwenImageModularPipeline"):
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
        with patch.object(Layers, "get_signal_value", return_value="QwenImageModularPipeline"):
            with self.assertRaisesRegex(ValueError, "comma-separated integers"):
                node.execute(blocks_select=["transformer_blocks"], transformer_blocks={"indices": "zero"})

    def test_layers_field_action_and_execution_require_the_connected_reviewed_allowlist(self):
        node = object.__new__(Layers)
        node.send_node_definition = MagicMock()

        with patch.object(Layers, "get_signal_value", return_value="FluxModularPipeline"):
            node.set_blocks({"blocks_select": ["single_transformer_blocks"]}, None)
            node.send_node_definition.assert_called_once()
            with self.assertRaisesRegex(ValueError, "exactly match"):
                node.execute(
                    blocks_select=["single_transformer_blocks"],
                    single_transformer_blocks={"indices": "1"},
                    transformer_blocks={"indices": "2"},
                )

        for model_type, block in (
            (None, "transformer_blocks"),
            ({}, "transformer_blocks"),
            ("QwenImageModularPipeline", "single_transformer_blocks"),
            ("QwenImageLayeredModularPipeline", "transformer_blocks"),
        ):
            with self.subTest(model_type=model_type, block=block):
                with patch.object(Layers, "get_signal_value", return_value=model_type):
                    with self.assertRaisesRegex(ValueError, "connected reviewed Modular pipeline"):
                        node.execute(blocks_select=[block], **{block: {"indices": "0"}})

    def test_layer_dependent_guiders_require_an_explicit_nonempty_selection(self):
        node = object.__new__(Guider)
        node.node_id = "guider-contract"

        for guider in (
            "SkipLayerGuidance",
            "AutoGuidance",
            "SmoothedEnergyGuidance",
            "PerturbedAttentionGuidance",
        ):
            with self.subTest(guider=guider), self.assertRaisesRegex(ValueError, "non-empty Layers connection"):
                node.execute(guider, layers_config=[])

    def test_adaptive_projected_mix_guider_forwards_exact_typed_arguments(self):
        node = object.__new__(Guider)
        node.node_id = "adaptive-projected-mix-contract"
        with patch.object(diffusers, "AdaptiveProjectedMixGuidance", return_value="configured") as constructor:
            result = node.execute(
                "AdaptiveProjectedMixGuidance",
                guidance_scale=3.5,
                guidance_rescale=0.25,
                adaptive_projected_guidance_scale=10.0,
                adaptive_projected_guidance_momentum=-0.5,
                adaptive_projected_guidance_rescale=12.0,
                eta=0.1,
                use_original_formulation=True,
                start=0.05,
                stop=0.95,
                adaptive_projected_guidance_start_step=7,
                enabled=True,
            )

        self.assertEqual(result, {"guider_out": "configured"})
        constructor.assert_called_once_with(
            guidance_scale=3.5,
            guidance_rescale=0.25,
            adaptive_projected_guidance_scale=10.0,
            adaptive_projected_guidance_momentum=-0.5,
            adaptive_projected_guidance_rescale=12.0,
            eta=0.1,
            use_original_formulation=True,
            start=0.05,
            stop=0.95,
            adaptive_projected_guidance_start_step=7,
            enabled=True,
        )

    def test_new_pinned_guiders_construct_without_model_weights(self):
        node = object.__new__(Guider)
        node.node_id = "new-guider-construction-contract"

        adaptive = node.execute("AdaptiveProjectedMixGuidance")["guider_out"]
        perturbed = node.execute(
            "PerturbedAttentionGuidance",
            layers_config=[{"indices": [1], "fqn": "transformer_blocks", "dropout": 1.0}],
        )["guider_out"]

        self.assertIsInstance(adaptive, diffusers.AdaptiveProjectedMixGuidance)
        self.assertEqual(adaptive.adaptive_projected_guidance_start_step, 5)
        self.assertIsInstance(perturbed, diffusers.PerturbedAttentionGuidance)
        self.assertEqual(perturbed.skip_layer_config[0].indices, [1])
        self.assertTrue(perturbed.skip_layer_config[0].skip_attention_scores)

    def test_adaptive_projected_mix_rejects_fractional_start_step_before_construction(self):
        node = object.__new__(Guider)
        node.node_id = "adaptive-projected-mix-invalid-start"

        with patch.object(diffusers, "AdaptiveProjectedMixGuidance") as constructor:
            with self.assertRaisesRegex(ValueError, "adaptive_projected_guidance_start_step must be an integer"):
                node.execute(
                    "AdaptiveProjectedMixGuidance",
                    adaptive_projected_guidance_start_step=2.5,
                )
            constructor.assert_not_called()

    def test_perturbed_attention_guider_normalizes_the_generic_layers_contract(self):
        node = object.__new__(Guider)
        node.node_id = "perturbed-attention-contract"
        with patch.object(diffusers, "PerturbedAttentionGuidance", return_value="configured") as constructor:
            result = node.execute(
                "PerturbedAttentionGuidance",
                layers_config=[
                    {
                        "indices": [2, 7],
                        "fqn": "transformer_blocks",
                        "dropout": 1.0,
                        "skip_attention": True,
                        "skip_attention_scores": False,
                        "skip_ff": True,
                    }
                ],
                guidance_scale=7.5,
                perturbed_guidance_scale=2.8,
                perturbed_guidance_start=0.01,
                perturbed_guidance_stop=0.2,
            )

        self.assertEqual(result, {"guider_out": "configured"})
        config = constructor.call_args.kwargs["perturbed_guidance_config"][0]
        self.assertEqual(config.indices, [2, 7])
        self.assertEqual(config.fqn, "transformer_blocks")
        self.assertFalse(config.skip_attention)
        self.assertTrue(config.skip_attention_scores)
        self.assertFalse(config.skip_ff)
        self.assertNotIn("perturbed_guidance_layers", constructor.call_args.kwargs)

    def test_perturbed_attention_rejects_invalid_layers_before_construction(self):
        invalid_layers = (
            None,
            [],
            [{"indices": "2", "fqn": "transformer_blocks"}],
            [{"indices": [-1], "fqn": "transformer_blocks"}],
            [{"indices": [2], "fqn": " transformer_blocks"}],
            [{"indices": [2], "fqn": "transformer_blocks", "dropout": 0.5}],
            ["transformer_blocks.2"],
            [diffusers.LayerSkipConfig(indices=[2], fqn="transformer_blocks", dropout=0.5)],
            [diffusers.LayerSkipConfig(indices=[-1], fqn="transformer_blocks")],
            diffusers.LayerSkipConfig(indices=[2], fqn="transformer_blocks", dropout=0.5),
        )
        node = object.__new__(Guider)
        node.node_id = "perturbed-attention-invalid-layers"

        for layers_config in invalid_layers:
            with self.subTest(layers_config=layers_config):
                with patch.object(diffusers, "PerturbedAttentionGuidance") as constructor:
                    with self.assertRaises((TypeError, ValueError)):
                        node.execute("PerturbedAttentionGuidance", layers_config=layers_config)
                    constructor.assert_not_called()

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
