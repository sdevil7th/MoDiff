import inspect
import gc
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from diffusers import ClassifierFreeGuidance, StableDiffusionXLModularPipeline
from diffusers.models import ImageProjection
from diffusers.models.attention_processor import IPAdapterAttnProcessor
from PIL import Image

from modiff.auxiliary_ip_adapter import ResolvedSDXLIPAdapter
from modules.ModularDiffusers.ip_adapter import IPAdapter
from modules.ModularDiffusers.denoise import Denoise
from modules.ModularDiffusers.loaders import ModelsLoader, annotate_modular_loader_outputs
from modules.ModularDiffusers.modular_utils import require_modiff_node_contract
from modules.ModularDiffusers.route_state import (
    ROUTE_STATE_OUTPUT,
    issue_pipeline_instance_token,
    require_sdxl_ip_adapter_bundle,
    reset_owned_sdxl_ip_adapter_for_loader,
)


SDXL = "StableDiffusionXLModularPipeline"


class FixtureCLIPImageProcessor:
    size = {"shortest_edge": 224}
    crop_size = {"height": 224, "width": 224}
    do_convert_rgb = True
    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    do_center_crop = True
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]


class FixtureEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type(
            "FixtureEncoderConfig",
            (),
            {
                "image_size": 224,
                "hidden_size": 1664,
                "patch_size": 14,
                "num_channels": 3,
                "num_hidden_layers": 48,
                "num_attention_heads": 16,
                "projection_dim": 1280,
            },
        )()


class FixtureUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.float32
        self.encoder_hid_proj = None
        self.attn_processors = {"base.processor": torch.nn.Identity()}
        self.config = type("FixtureUNetConfig", (), {"encoder_hid_dim_type": None})()


def _loader_output(unet_id="fixture-unet"):
    token = issue_pipeline_instance_token(
        model_type=SDXL,
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        repo_source="hub",
        revision="a" * 40,
    )
    outputs = {
        "unet_out": {"model_id": unet_id},
        "vae_out": {"model_id": "fixture-vae"},
        "text_encoders": {"text_encoder": {"model_id": "fixture-text"}},
        "scheduler": {"model_id": "fixture-scheduler"},
    }
    annotate_modular_loader_outputs(
        outputs,
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        repo_source="hub",
        model_type=SDXL,
        revision="a" * 40,
        trust_remote_code=False,
        pipeline_instance_token=token,
    )
    return token, outputs


class FixturePipeline:
    def __init__(self, *, swap_manager=None, fail_call=False):
        self._execution_device = torch.device("cpu")
        self.blocks = type("FixtureBlocksDocument", (), {"doc": "fixture"})()
        self.feature_extractor = FixtureCLIPImageProcessor()
        self.swap_manager = swap_manager
        self.fail_call = fail_call
        self.load_calls = []
        self.unload_calls = 0

    def update_components(self, **values):
        for name, value in values.items():
            setattr(self, name, value)

    def load_ip_adapter(self, path, **kwargs):
        self.load_calls.append((path, dict(kwargs)))
        self.unet.encoder_hid_proj = type("FixtureProjection", (), {})()
        self.unet.encoder_hid_proj.image_projection_layers = torch.nn.ModuleList(
            [ImageProjection(image_embed_dim=1280, cross_attention_dim=8, num_image_text_embeds=4)]
        )
        self.unet.attn_processors = {
            "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor": IPAdapterAttnProcessor(
                hidden_size=8,
                cross_attention_dim=8,
                num_tokens=(4,),
                scale=1.0,
            )
        }
        self.unet.config.encoder_hid_dim_type = "ip_image_proj"

    def unload_ip_adapter(self):
        self.unload_calls += 1
        self.unet.encoder_hid_proj = None
        self.unet.attn_processors = {"base.processor": torch.nn.Identity()}
        self.unet.config.encoder_hid_dim_type = None

    def set_ip_adapter_scale(self, scale):
        for processor in self.unet.attn_processors.values():
            if isinstance(processor, IPAdapterAttnProcessor):
                processor.scale = [float(scale)]

    def __call__(self, **kwargs):
        if self.swap_manager is not None:
            self.swap_manager()
        if self.fail_call:
            raise RuntimeError("fixture encoder failure")
        self.call_kwargs = dict(kwargs)
        return {
            "ip_adapter_embeds": [torch.zeros((1, 1, 1280))],
            "negative_ip_adapter_embeds": [torch.ones((1, 1, 1280))],
        }


class FixtureBlocks:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def init_pipeline(self, *, components_manager):
        self.components_manager = components_manager
        return self.pipeline


class ModularIPAdapterTests(unittest.TestCase):
    def test_connected_guider_schema_cannot_inject_a_local_guidance_scale(self):
        _blocks, config = require_modiff_node_contract(
            StableDiffusionXLModularPipeline,
            "ip_adapter",
            resolve_blocks=False,
        )

        self.assertNotIn("guidance_scale", config["params"])
        self.assertNotIn("onChange", config["params"]["guider"])

    def test_separate_node_instances_reuse_one_exact_managed_image_encoder(self):
        encoder_class = type("CLIPVisionModelWithProjection", (FixtureEncoder,), {})
        candidate = encoder_class()
        candidate.dtype = torch.float32
        candidate._diffusers_load_id = "h94/IP-Adapter|sdxl_models/image_encoder|null|" + "0" * 40
        artifact = ResolvedSDXLIPAdapter(
            repository="h94/IP-Adapter",
            revision="0" * 40,
            weight_name="ip-adapter_sdxl.safetensors",
            content_sha256="1" * 64,
            byte_size=1,
            image_encoder_subfolder="sdxl_models/image_encoder",
            image_encoder_class="CLIPVisionModelWithProjection",
            load_directory=Path("/unused"),
        )
        nodes = (IPAdapter("first-ip-adapter"), IPAdapter("second-ip-adapter"))
        transformers_module = type(sys)("transformers")
        transformers_module.CLIPVisionModelWithProjection = encoder_class

        with (
            patch.dict(sys.modules, {"transformers": transformers_module}),
            patch(
                "modules.ModularDiffusers.ip_adapter.components.get_ids",
                return_value=["image_encoder_fixture"],
            ) as get_ids,
            patch(
                "modules.ModularDiffusers.ip_adapter.components.get_components_by_ids",
                return_value={"image_encoder_fixture": candidate},
            ) as get_components,
            patch("modules.ModularDiffusers.ip_adapter.sdxl_ip_adapter_image_encoder_contract"),
            patch("modules.ModularDiffusers.ip_adapter.ComponentSpec.load") as load,
        ):
            resolved = [
                node._load_image_encoder(artifact, dtype=torch.float32, device=torch.device("cpu"))
                for node in nodes
            ]

        self.assertEqual(resolved, [candidate, candidate])
        self.assertEqual(get_ids.call_count, 2)
        self.assertEqual(
            get_ids.call_args.kwargs,
            {"names": "image_encoder"},
        )
        self.assertEqual(get_components.call_count, 2)
        self.assertEqual(
            get_components.call_args.kwargs,
            {
                "ids": ["image_encoder_fixture"],
                "return_dict_with_names": False,
            },
        )
        load.assert_not_called()

    def test_ambiguous_or_wrong_dtype_managed_image_encoder_fails_closed(self):
        encoder_class = type("CLIPVisionModelWithProjection", (FixtureEncoder,), {})
        artifact = ResolvedSDXLIPAdapter(
            repository="h94/IP-Adapter",
            revision="0" * 40,
            weight_name="ip-adapter_sdxl.safetensors",
            content_sha256="1" * 64,
            byte_size=1,
            image_encoder_subfolder="sdxl_models/image_encoder",
            image_encoder_class="CLIPVisionModelWithProjection",
            load_directory=Path("/unused"),
        )
        compatible = encoder_class()
        compatible.dtype = torch.float32
        compatible._diffusers_load_id = "h94/IP-Adapter|sdxl_models/image_encoder|null|" + "0" * 40
        incompatible = encoder_class()
        incompatible.dtype = torch.float16
        incompatible._diffusers_load_id = compatible._diffusers_load_id
        transformers_module = type(sys)("transformers")
        transformers_module.CLIPVisionModelWithProjection = encoder_class

        for resident, message in (
            (
                {"image_encoder_one": compatible, "image_encoder_two": compatible},
                "multiple compatible",
            ),
            ({"image_encoder_wrong_dtype": incompatible}, "incompatible class or dtype"),
        ):
            with (
                self.subTest(message=message),
                patch.dict(sys.modules, {"transformers": transformers_module}),
                patch(
                    "modules.ModularDiffusers.ip_adapter.components.get_ids",
                    return_value=list(resident),
                ),
                patch(
                    "modules.ModularDiffusers.ip_adapter.components.get_components_by_ids",
                    return_value=resident,
                ),
                self.assertRaisesRegex(ValueError, message),
            ):
                IPAdapter("fixture-ip-adapter")._load_image_encoder(
                    artifact,
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                )

    def test_first_image_encoder_load_handles_an_empty_components_manager(self):
        encoder_class = type("CLIPVisionModelWithProjection", (FixtureEncoder,), {})
        loaded = encoder_class()
        loaded.dtype = torch.float32
        loaded.to = Mock(return_value=loaded)
        artifact = ResolvedSDXLIPAdapter(
            repository="h94/IP-Adapter",
            revision="0" * 40,
            weight_name="ip-adapter_sdxl.safetensors",
            content_sha256="1" * 64,
            byte_size=1,
            image_encoder_subfolder="sdxl_models/image_encoder",
            image_encoder_class="CLIPVisionModelWithProjection",
            load_directory=Path("/unused"),
        )
        transformers_module = type(sys)("transformers")
        transformers_module.CLIPVisionModelWithProjection = encoder_class

        with (
            patch.dict(sys.modules, {"transformers": transformers_module}),
            patch("modules.ModularDiffusers.ip_adapter.components.get_ids", return_value=[]),
            patch(
                "modules.ModularDiffusers.ip_adapter.components.get_components_by_ids",
                return_value={},
            ),
            patch(
                "modules.ModularDiffusers.ip_adapter.ComponentSpec.load",
                return_value=loaded,
            ) as load,
            patch("modules.ModularDiffusers.ip_adapter.sdxl_ip_adapter_image_encoder_contract"),
        ):
            result = IPAdapter("fixture-ip-adapter")._load_image_encoder(
                artifact,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )

        self.assertIs(result, loaded)
        load.assert_called_once_with(local_files_only=True, torch_dtype=torch.float32)
        loaded.to.assert_called_once_with(device=torch.device("cpu"))

    def _execute(self, *, pipeline=None, manager_swap=False, fail_call=False):
        token, outputs = _loader_output()
        unet = FixtureUNet()
        other_unet = FixtureUNet()
        current = {"value": unet}
        encoder = FixtureEncoder()
        guider = ClassifierFreeGuidance(guidance_scale=7.5)
        image = Image.new("RGB", (32, 32), "green")
        with tempfile.TemporaryDirectory() as directory:
            artifact = ResolvedSDXLIPAdapter(
                repository="h94/IP-Adapter",
                revision="0" * 40,
                weight_name="ip-adapter_sdxl.safetensors",
                content_sha256="1" * 64,
                byte_size=1,
                image_encoder_subfolder="sdxl_models/image_encoder",
                image_encoder_class="CLIPVisionModelWithProjection",
                load_directory=Path(directory),
            )

            def manager(*, ids, return_dict_with_names=False):
                self.assertFalse(return_dict_with_names)
                self.assertEqual(ids, [outputs["unet_out"]["model_id"]])
                return {ids[0]: current["value"]}

            def swap():
                if manager_swap:
                    current["value"] = other_unet

            pipeline = pipeline or FixturePipeline()
            pipeline.swap_manager = swap
            pipeline.fail_call = fail_call
            _real_blocks, config = require_modiff_node_contract(
                StableDiffusionXLModularPipeline,
                "ip_adapter",
                resolve_blocks=False,
            )
            node = IPAdapter("fixture-ip-adapter")
            with (
                patch(
                    "modules.ModularDiffusers.ip_adapter.pipeline_class_from_runtime_inputs",
                    return_value=StableDiffusionXLModularPipeline,
                ),
                patch(
                    "modules.ModularDiffusers.ip_adapter.require_modiff_node_contract",
                    return_value=(FixtureBlocks(pipeline), config),
                ),
                patch("modules.ModularDiffusers.ip_adapter.components.get_components_by_ids", side_effect=manager),
                patch("modules.ModularDiffusers.ip_adapter.resolve_reviewed_sdxl_ip_adapter", return_value=artifact),
                patch.object(node, "_load_image_encoder", return_value=encoder),
            ):
                result = node.execute(
                    unet=outputs["unet_out"],
                    guider=guider,
                    ip_adapter_image=image,
                    adapter_model={"source": "hub", "value": "h94/IP-Adapter"},
                    adapter_revision="0" * 40,
                    adapter_weight_name="sdxl_models/ip-adapter_sdxl.safetensors",
                    adapter_scale=0.75,
                )
        return {
            "token": token,
            "outputs": outputs,
            "unet": unet,
            "other_unet": other_unet,
            "encoder": encoder,
            "guider": guider,
            "image": image,
            "pipeline": pipeline,
            "result": result,
        }

    def test_action_loads_one_exact_local_adapter_and_publishes_the_bound_bundle(self):
        fixture = self._execute()
        pipeline = fixture["pipeline"]
        self.assertEqual(len(pipeline.load_calls), 1)
        _path, kwargs = pipeline.load_calls[0]
        self.assertEqual(kwargs["subfolder"], "")
        self.assertEqual(kwargs["weight_name"], "ip-adapter_sdxl.safetensors")
        self.assertIs(kwargs["local_files_only"], True)
        self.assertIs(pipeline.call_kwargs["ip_adapter_image"], fixture["image"])
        self.assertIsNotNone(
            require_sdxl_ip_adapter_bundle(
                fixture["result"]["ip_adapter"],
                binding=fixture["token"],
                unet=fixture["unet"],
                guider=fixture["guider"],
            )
        )

    def test_manager_swap_or_encoding_failure_unloads_partial_mutation_and_never_publishes(self):
        for option in ("manager-swap", "call-failure"):
            pipeline = FixturePipeline()
            with self.subTest(option=option):
                with self.assertRaisesRegex(
                    (ValueError, RuntimeError),
                    "changed during IP-Adapter encoding|fixture encoder failure",
                ):
                    self._execute(
                        pipeline=pipeline,
                        manager_swap=option == "manager-swap",
                        fail_call=option == "call-failure",
                    )
                self.assertEqual(pipeline.unload_calls, 1)
                self.assertIsNone(pipeline.unet.encoder_hid_proj)

    def test_loader_reset_removes_only_current_owned_adapter_state(self):
        fixture = self._execute()
        self.assertTrue(reset_owned_sdxl_ip_adapter_for_loader(fixture["pipeline"]))
        self.assertEqual(fixture["pipeline"].unload_calls, 1)
        self.assertIsNone(
            require_sdxl_ip_adapter_bundle(
                None,
                binding=fixture["token"],
                unet=fixture["unet"],
                guider=fixture["guider"],
            )
        )
        self.assertIn("reset_owned_sdxl_ip_adapter_for_loader(self.loader)", inspect.getsource(ModelsLoader.execute))

    def test_denoise_flattens_only_the_exact_adapter_bundle_and_revalidates_resident_components(self):
        fixture = self._execute()

        class FixtureVAE:
            config = type(
                "FixtureVAEConfig",
                (),
                {"latent_channels": 4, "block_out_channels": [32, 64, 128, 256]},
            )()

        vae = FixtureVAE()
        scheduler = object()
        calls = []
        _blocks, config = require_modiff_node_contract(
            StableDiffusionXLModularPipeline,
            "denoise",
            resolve_blocks=False,
        )
        block_component_names = ["unet", "vae", "scheduler", "guider"]
        block_input_names = [
            "prompt_embeds",
            "negative_prompt_embeds",
            "pooled_prompt_embeds",
            "negative_pooled_prompt_embeds",
            "generator",
            "ip_adapter_embeds",
            "negative_ip_adapter_embeds",
        ]

        class FixtureDenoisePipeline:
            _execution_device = torch.device("cpu")
            component_names = list(block_component_names)
            blocks = type("FixtureDenoiseDocument", (), {"doc": "fixture"})()
            transformer = None

            def update_components(self, **values):
                for name, value in values.items():
                    setattr(self, name, value)

            def __call__(self, **kwargs):
                calls.append(dict(kwargs))
                return {"latents": torch.zeros((1, 4, 8, 8))}

        pipeline = FixtureDenoisePipeline()

        class FixtureDenoiseBlocks:
            component_names = list(block_component_names)
            input_names = list(block_input_names)

            def __deepcopy__(self, memo):
                return self

            @staticmethod
            def init_pipeline(*, components_manager):
                return pipeline

        manager_values = {
            fixture["outputs"]["unet_out"]["model_id"]: ("unet", fixture["unet"]),
            fixture["outputs"]["vae_out"]["model_id"]: ("vae", vae),
            fixture["outputs"]["scheduler"]["model_id"]: ("scheduler", scheduler),
        }

        def manager(*, ids, return_dict_with_names=False):
            selected = [manager_values[model_id] for model_id in ids]
            if return_dict_with_names:
                return {name: value for name, value in selected}
            return {model_id: manager_values[model_id][1] for model_id in ids}

        with (
            patch(
                "modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs",
                return_value=StableDiffusionXLModularPipeline,
            ),
            patch(
                "modules.ModularDiffusers.denoise.require_modiff_node_contract",
                return_value=(FixtureDenoiseBlocks(), config),
            ),
            patch("modules.ModularDiffusers.denoise.components.get_components_by_ids", side_effect=manager),
            patch("modules.ModularDiffusers.denoise.insert_preview_block"),
        ):
            result = Denoise("fixture-ip-denoise").execute(
                unet=fixture["outputs"]["unet_out"],
                vae=fixture["outputs"]["vae_out"],
                scheduler=fixture["outputs"]["scheduler"],
                guider=fixture["guider"],
                embeddings={"prompt_embeds": torch.zeros((1, 1, 2048))},
                ip_adapter=fixture["result"]["ip_adapter"],
                seed=7,
                num_inference_steps=1,
            )

        self.assertEqual(len(calls), 1)
        self.assertIsNot(calls[0]["ip_adapter_embeds"], fixture["result"]["ip_adapter"]["ip_adapter_embeds"])
        self.assertIs(
            calls[0]["ip_adapter_embeds"][0],
            fixture["result"]["ip_adapter"]["ip_adapter_embeds"][0],
        )
        self.assertIsNot(
            calls[0]["negative_ip_adapter_embeds"],
            fixture["result"]["ip_adapter"]["negative_ip_adapter_embeds"],
        )
        self.assertIs(
            calls[0]["negative_ip_adapter_embeds"][0],
            fixture["result"]["ip_adapter"]["negative_ip_adapter_embeds"][0],
        )
        self.assertNotIn("ip_adapter", calls[0])
        self.assertNotIn("_IPAdapterStateKey", repr(calls[0]))
        self.assertIsNotNone(result[ROUTE_STATE_OUTPUT])

    def test_loader_reset_rejects_unreceipted_adapter_structure(self):
        unet = FixtureUNet()
        pipeline = FixturePipeline()
        pipeline.update_components(unet=unet)
        pipeline.load_ip_adapter("fixture", local_files_only=True)
        with self.assertRaisesRegex(ValueError, "unreceipted IP-Adapter state"):
            reset_owned_sdxl_ip_adapter_for_loader(pipeline)
        self.assertEqual(pipeline.unload_calls, 0)

    def test_loader_can_reset_owned_state_after_the_adapter_action_is_deleted(self):
        fixture = self._execute()
        unet = fixture["unet"]
        reset_pipeline = FixturePipeline()
        reset_pipeline.update_components(unet=unet)
        del fixture
        gc.collect()

        self.assertTrue(reset_owned_sdxl_ip_adapter_for_loader(reset_pipeline))
        self.assertEqual(reset_pipeline.unload_calls, 1)
        self.assertIsNone(unet.encoder_hid_proj)


if __name__ == "__main__":
    unittest.main()
