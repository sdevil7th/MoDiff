import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import modules as module_registry
from modiff.server import WebServer
from modules.DiffusersImage import ControlGenerate, Edit, Inpaint, LoadAdapter, LoadPipeline, MODULE_MAP
from modules.DiffusersImage.main import (
    FLUX_DEV_REPO,
    FluxReduxPipelineBundle,
    add_progress_callback,
    output_image_dimensions,
    quant_config_for,
)
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


class DiffusersImageRegistryTests(unittest.TestCase):
    def test_output_dimensions_support_pil_numpy_and_torch_layouts(self):
        self.assertEqual(output_image_dimensions([Image.new("RGB", (31, 19))], "pil"), (31, 19))
        self.assertEqual(output_image_dimensions(np.zeros((2, 19, 31, 3)), "np"), (31, 19))
        fake_tensor = SimpleNamespace(shape=(2, 3, 19, 31), size=lambda: 2 * 3 * 19 * 31)
        self.assertEqual(output_image_dimensions(fake_tensor, "pt"), (31, 19))

    def test_torchao_int8_weight_only_passes_an_aobase_config_instance(self):
        int8_config = object()
        torchao_config = object()
        with (
            patch.dict(
                "sys.modules",
                {
                    "torchao": SimpleNamespace(),
                    "torchao.quantization": SimpleNamespace(
                        Int8WeightOnlyConfig=Mock(return_value=int8_config),
                    ),
                },
            ),
            patch("diffusers.TorchAoConfig", return_value=torchao_config) as config,
        ):
            result = quant_config_for("torchao_int8_weight_only", None, ["proj_out"])

        self.assertIs(result, torchao_config)
        self.assertIs(config.call_args.kwargs["quant_type"], int8_config)
        self.assertEqual(config.call_args.kwargs["modules_to_not_convert"], ["proj_out"])

    def test_quanto_float8_uses_the_current_weights_dtype_argument(self):
        received = {}

        class FakeQuantoConfig:
            def __init__(self, **kwargs):
                received.update(kwargs)

        with patch("diffusers.QuantoConfig", FakeQuantoConfig):
            quant_config_for("quanto_float8", None)

        self.assertEqual(received, {"weights_dtype": "float8", "modules_to_not_convert": None})

    def test_quanto_int8_uses_weight_only_int8(self):
        received = {}

        class FakeQuantoConfig:
            def __init__(self, **kwargs):
                received.update(kwargs)

        with patch("diffusers.QuantoConfig", FakeQuantoConfig):
            quant_config_for("quanto_int8", None, ["norm"])

        self.assertEqual(received, {"weights_dtype": "int8", "modules_to_not_convert": ["norm"]})

    def test_torchao_float8_passes_an_aobase_config_instance(self):
        import sys
        from types import ModuleType

        received = {}

        class FakeFloat8WeightOnlyConfig:
            pass

        class FakeTorchAoConfig:
            def __init__(self, **kwargs):
                received.update(kwargs)

        fake_quantization = ModuleType("torchao.quantization")
        fake_quantization.Float8WeightOnlyConfig = FakeFloat8WeightOnlyConfig
        fake_torchao = ModuleType("torchao")
        fake_torchao.quantization = fake_quantization

        with (
            patch("diffusers.TorchAoConfig", FakeTorchAoConfig),
            patch.dict(
                sys.modules,
                {"torchao": fake_torchao, "torchao.quantization": fake_quantization},
            ),
        ):
            quant_config_for("torchao_float8", None)

        self.assertIsInstance(received["quant_type"], FakeFloat8WeightOnlyConfig)

    def test_torchao_float8_explains_when_optional_dependency_is_missing(self):
        import builtins

        real_import = builtins.__import__

        def import_without_torchao(name, *args, **kwargs):
            if name == "torchao.quantization":
                raise ImportError("torchao unavailable")
            return real_import(name, *args, **kwargs)

        with (
            patch("diffusers.TorchAoConfig", object),
            patch("builtins.__import__", side_effect=import_without_torchao),
        ):
            with self.assertRaisesRegex(RuntimeError, "optional torchao quantization package"):
                quant_config_for("torchao_float8", None)

    def test_inherited_nodes_are_registered_with_their_live_contracts(self):
        expected_inputs = {
            "Edit": {"pipeline", "image"},
            "Inpaint": {"pipeline", "image", "mask_image"},
            "ControlGenerate": {"pipeline", "control_image"},
        }

        for name, required_inputs in expected_inputs.items():
            with self.subTest(node=name):
                entry = MODULE_MAP[name]
                self.assertEqual(entry["type"], "custom")
                self.assertEqual(entry["category"], "Diffusers Image")
                self.assertTrue(entry["resizable"])
                self.assertTrue(required_inputs.issubset(entry["params"]))
                self.assertEqual(entry["params"]["images"]["display"], "output")

    def test_graph_contract_marks_mode_independent_image_inputs_as_required(self):
        self.assertTrue(Edit.params["pipeline"]["required"])
        self.assertTrue(Edit.params["image"]["required"])
        self.assertTrue(Inpaint.params["mask_image"]["required"])
        self.assertTrue(ControlGenerate.params["control_image"]["required"])
        self.assertTrue(LoadAdapter.params["pipeline"]["required"])

    def test_registered_classes_can_be_constructed(self):
        for node_class in (Edit, Inpaint, ControlGenerate):
            with self.subTest(node=node_class.__name__):
                node = node_class("registry-probe")
                self.assertEqual(node.node_id, "registry-probe")
                self.assertTrue(node.resizable)

    def test_unsupported_mode_fails_before_pipeline_resolution(self):
        node = LoadPipeline("mode-probe")
        with self.assertRaisesRegex(ValueError, "does not support outpaint"):
            node.execute(model_id="example/model", pipeline_class="FluxPipeline", mode="outpaint")

    def test_flux2_klein_supports_generation_and_reference_edits(self):
        node = LoadPipeline("flux2-mode-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, _repo, **_kwargs):
                return cls()

        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=FakePipeline),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            for mode in ("text_to_image", "edit_image", "multi_image_reference_edit"):
                node.execute(
                    model_id="black-forest-labs/FLUX.2-klein-4B",
                    pipeline_class="Flux2KleinPipeline",
                    mode=mode,
                    auto_offload=False,
                    offload_mode="none",
                )

    def test_flux2_klein_reuses_one_resident_loader_when_only_mode_changes(self):
        loaded = []

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, repo, **_kwargs):
                loaded.append(repo)
                return cls()

        node = LoadPipeline("flux2-cache-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        common = {
            "model_id": "black-forest-labs/FLUX.2-klein-4B",
            "pipeline_class": "Flux2KleinPipeline",
            "auto_offload": False,
            "offload_mode": "none",
        }
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=FakePipeline),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            first = node(mode="text_to_image", **common)
            second = node(mode="edit_image", **common)

        self.assertIs(first["pipeline"], second["pipeline"])
        self.assertEqual(loaded, ["black-forest-labs/FLUX.2-klein-4B"])
        self.assertEqual(node.params["mode"], "edit_image")
        self.assertFalse(node._has_changed)

    def test_cross_workflow_loader_reuse_removes_residual_lora(self):
        unloads = []
        pipeline = type("Pipeline", (), {"unload_lora_weights": lambda _self: unloads.append(True)})()
        node = LoadPipeline("image-loader-reuse")
        node.output["pipeline"] = pipeline

        node.prepare_for_workflow_reuse()

        self.assertEqual(unloads, [True])
        self.assertIs(node.output["pipeline"], pipeline)

    def test_two_repositories_use_the_same_generic_loader_contract(self):
        loaded = []

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.append(repo)
                return cls()

        node = LoadPipeline("repo-switch-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=FakePipeline),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            for repo in ("org/flux-compatible-a", "org/flux-compatible-b"):
                result = node.execute(
                    model_id=repo,
                    pipeline_class="FluxPipeline",
                    mode="text_to_image",
                    auto_offload=False,
                    offload_mode="none",
                )
                self.assertEqual(result["resolved_artifact"], repo)

        self.assertEqual(loaded, ["org/flux-compatible-a", "org/flux-compatible-b"])

    def test_execution_recipe_controls_load_placement_attention_and_offload(self):
        loaded = {}
        offload = {}

        class FakeTransformer:
            def __init__(self):
                self.backends = []

            def set_attention_backend(self, backend):
                self.backends.append(backend)

        class FakePipeline:
            def __init__(self):
                self.transformer = FakeTransformer()

            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.update({"repo": repo, "kwargs": kwargs})
                return cls()

        def capture_offload(_pipeline, **kwargs):
            offload.update(kwargs)

        quant_config = object()
        recipe = {
            "quantization_config": quant_config,
            "device_map": {"transformer": 0, "text_encoder_2": "cpu"},
            "max_memory": {0: "16GiB", "cpu": "48GiB"},
            "offload_mode": "group_cpu",
            "device": "cuda:0",
            "attention_backend": "native",
            "attention_components": ["transformer"],
            "vae_slicing": False,
            "vae_tiling": False,
        }

        node = LoadPipeline("recipe-loader-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=FakePipeline),
            patch("modules.DiffusersImage.main.apply_pipeline_offload", side_effect=capture_offload),
        ):
            with self.assertRaisesRegex(ValueError, "PipelineQuantizationConfig"):
                node.execute(
                    model_id="org/runtime-recipe-model",
                    pipeline_class="FluxPipeline",
                    mode="text_to_image",
                    execution_recipe=recipe,
                )
            recipe["quantization_config"] = None
            result = node.execute(
                model_id="org/runtime-recipe-model",
                pipeline_class="FluxPipeline",
                mode="text_to_image",
                execution_recipe=recipe,
            )

        self.assertNotIn("quantization_config", loaded["kwargs"])
        self.assertEqual(loaded["kwargs"]["device_map"], recipe["device_map"])
        self.assertEqual(loaded["kwargs"]["max_memory"], recipe["max_memory"])
        self.assertEqual(offload["mode"], "group_cpu")
        self.assertEqual(offload["device"], "cuda:0")
        self.assertEqual(result["pipeline"].transformer.backends, ["native"])

    def test_execution_recipe_applies_cache_and_compile_to_direct_image_pipeline(self):
        applied = {}

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, _repo, **_kwargs):
                return cls()

        recipe = {
            "offload_mode": "none",
            "device": "cuda:0",
            "denoiser_cache": "first_block",
            "cache_threshold": 0.08,
            "regional_compile": True,
            "compile_components": ["transformer"],
        }
        node = LoadPipeline("direct-image-runtime-recipe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=FakePipeline),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
            patch(
                "modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline",
                side_effect=lambda pipeline, runtime_recipe: applied.update(
                    {"pipeline": pipeline, "recipe": runtime_recipe}
                )
                or {"cache": {"requested": runtime_recipe["denoiser_cache"]}},
            ),
        ):
            result = node.execute(
                model_id="org/direct-image-model",
                pipeline_class="FluxPipeline",
                mode="text_to_image",
                execution_recipe=recipe,
                enable_vae_slicing=False,
                enable_vae_tiling=False,
            )

        self.assertIs(applied["pipeline"], result["pipeline"])
        self.assertEqual(applied["recipe"]["denoiser_cache"], "first_block")
        self.assertEqual(applied["recipe"]["cache_threshold"], 0.08)
        self.assertTrue(applied["recipe"]["regional_compile"])
        self.assertFalse(applied["recipe"]["vae_slicing"])
        self.assertFalse(applied["recipe"]["vae_tiling"])
        self.assertEqual(
            result["pipeline"]._modiff_runtime_config,
            {"cache": {"requested": "first_block"}},
        )

    def test_direct_image_defaults_still_apply_vae_memory_without_a_recipe_node(self):
        captured = {}

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, _repo, **_kwargs):
                return cls()

        node = LoadPipeline("direct-image-default-runtime")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=FakePipeline),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
            patch(
                "modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline",
                side_effect=lambda _pipeline, runtime_recipe: captured.update(runtime_recipe) or {},
            ),
        ):
            node.execute(
                model_id="org/direct-image-model",
                pipeline_class="FluxPipeline",
                mode="text_to_image",
                auto_offload=False,
                offload_mode="none",
                enable_vae_slicing=False,
                enable_vae_tiling=True,
            )

        self.assertFalse(captured["vae_slicing"])
        self.assertTrue(captured["vae_tiling"])

    def test_direct_device_map_streams_a_no_offload_pipeline_to_cuda_during_load(self):
        loaded = {}
        offload = {}

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.update({"repo": repo, "kwargs": kwargs})
                return cls()

        node = LoadPipeline("direct-device-map-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=FakePipeline),
            patch(
                "modules.DiffusersImage.main.apply_pipeline_offload",
                side_effect=lambda _pipeline, **kwargs: offload.update(kwargs),
            ),
        ):
            node.execute(
                model_id="org/native-model",
                pipeline_class="FluxPipeline",
                mode="text_to_image",
                device="cuda:0",
                device_map="cuda",
                auto_offload=False,
                offload_mode="none",
            )

        self.assertEqual(loaded["kwargs"]["device_map"], "cuda")
        self.assertEqual(offload["mode"], "none")
        self.assertEqual(offload["device"], "cuda:0")

    def test_direct_device_map_overrides_a_neutral_recipe_default(self):
        loaded = {}

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, _repo, **kwargs):
                loaded.update(kwargs)
                return cls()

        node = LoadPipeline("direct-map-precedence-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=FakePipeline),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            node.execute(
                model_id="org/native-model",
                pipeline_class="FluxPipeline",
                mode="text_to_image",
                execution_recipe={"device_map": "none", "offload_mode": "none", "device": "cuda:0"},
                device_map="cuda",
            )

        self.assertEqual(loaded["device_map"], "cuda")

    def test_flux_redux_loader_composes_prior_with_app_cached_flux_base(self):
        loaded = []

        class FakePrior:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.append(("prior", repo, kwargs))
                return cls()

        class FakeBase:
            text_encoder = "clip"
            text_encoder_2 = "t5"
            tokenizer = "clip-tokenizer"
            tokenizer_2 = "t5-tokenizer"

            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.append(("base", repo, kwargs))
                return cls()

            def register_modules(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        node = LoadPipeline("redux-loader-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("diffusers.FluxPriorReduxPipeline", FakePrior),
            patch("diffusers.FluxPipeline", FakeBase),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            result = node.execute(
                model_id="black-forest-labs/FLUX.1-Redux-dev",
                pipeline_class="FluxReduxPipeline",
                mode="edit_image",
                revision="redux-commit",
                auto_offload=False,
                offload_mode="none",
            )

        self.assertIsInstance(result["pipeline"], FluxReduxPipelineBundle)
        self.assertEqual(loaded[0][0:2], ("base", FLUX_DEV_REPO))
        self.assertEqual(loaded[0][2]["revision"], "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21")
        self.assertEqual(loaded[1][0:2], ("prior", "black-forest-labs/FLUX.1-Redux-dev"))
        self.assertEqual(loaded[1][2]["revision"], "redux-commit")
        self.assertEqual(loaded[1][2]["text_encoder"], "clip")
        self.assertEqual(loaded[1][2]["text_encoder_2"], "t5")
        self.assertEqual(loaded[1][2]["tokenizer"], "clip-tokenizer")
        self.assertEqual(loaded[1][2]["tokenizer_2"], "t5-tokenizer")
        self.assertTrue(loaded[0][2]["local_files_only"])
        self.assertTrue(loaded[1][2]["local_files_only"])
        self.assertIsNone(result["pipeline"].base.text_encoder)
        self.assertIsNone(result["pipeline"].base.text_encoder_2)

    def test_curated_image_loader_uses_catalog_pin_but_preserves_explicit_revision(self):
        loaded = []

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.append((repo, kwargs))
                return cls()

        node = LoadPipeline("image-revision-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=FakePipeline),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            node.execute(
                model_id="black-forest-labs/FLUX.1-schnell",
                pipeline_class="FluxPipeline",
                mode="text_to_image",
                auto_offload=False,
                offload_mode="none",
            )
            node.execute(
                model_id="black-forest-labs/FLUX.1-schnell",
                pipeline_class="FluxPipeline",
                mode="text_to_image",
                revision="reviewed-user-revision",
                auto_offload=False,
                offload_mode="none",
            )

        self.assertEqual(loaded[0][1]["revision"], "741f7c3ce8b383c54771c7003378a50191e9efe9")
        self.assertEqual(loaded[1][1]["revision"], "reviewed-user-revision")

    def test_flux_redux_bundle_delegates_multiple_reference_fusion_to_diffusers(self):
        import torch

        calls = {}

        class PriorOutput:
            # FluxPriorReduxPipeline already reduces the reference batch to
            # one weighted conditioning sample.
            prompt_embeds = torch.tensor([[[3.0, 5.0]]])
            pooled_prompt_embeds = torch.tensor([[4.0, 6.0]])

        class FakePrior:
            def __call__(self, **kwargs):
                calls["prior"] = kwargs
                return PriorOutput()

        class FakeBase:
            device = "cpu"

            def __call__(self, **kwargs):
                calls["base"] = kwargs
                return type("Result", (), {"images": [Image.new("RGB", (32, 32), "white")]})()

        bundle = FluxReduxPipelineBundle(FakePrior(), FakeBase())
        references = [Image.new("RGB", (16, 16), "red"), Image.new("RGB", (16, 16), "blue")]
        result = Edit("redux-edit-probe").execute(
            pipeline=bundle,
            image=references,
            prompt="combine material and silhouette",
            width=32,
            height=32,
            num_inference_steps=4,
            guidance_scale=2.5,
        )

        self.assertIs(calls["prior"]["image"], references)
        self.assertEqual(calls["prior"]["prompt"], "combine material and silhouette")
        self.assertEqual(calls["prior"]["prompt_embeds_scale"], [1.0, 1.0])
        self.assertEqual(calls["prior"]["pooled_prompt_embeds_scale"], [1.0, 1.0])
        torch.testing.assert_close(calls["base"]["prompt_embeds"], torch.tensor([[[3.0, 5.0]]]))
        torch.testing.assert_close(calls["base"]["pooled_prompt_embeds"], torch.tensor([[4.0, 6.0]]))
        self.assertEqual(calls["base"]["width"], 32)
        self.assertEqual(calls["base"]["height"], 32)
        self.assertEqual(result["images"][0].size, (32, 32))

    def test_flux_redux_bundle_scales_secondary_references_through_generic_conditioning(self):
        import torch

        calls = {}

        class PriorOutput:
            prompt_embeds = torch.tensor([[[1.0, 3.0]]])
            pooled_prompt_embeds = torch.tensor([[2.0, 4.0]])

        class FakePrior:
            def __call__(self, **kwargs):
                calls["prior"] = kwargs
                return PriorOutput()

        class FakeBase:
            device = "cpu"

            def __call__(self, **kwargs):
                return type("Result", (), {"images": [Image.new("RGB", (32, 32), "white")]})()

        bundle = FluxReduxPipelineBundle(FakePrior(), FakeBase())
        references = [Image.new("RGB", (16, 16), "red"), Image.new("RGB", (16, 16), "blue")]
        Edit("redux-weight-probe").execute(
            pipeline=bundle,
            image=references,
            prompt="architecture with a restrained material cue",
            width=32,
            height=32,
            num_inference_steps=4,
            guidance_scale=2.5,
            reference_strength=0.2,
        )

        self.assertEqual(calls["prior"]["prompt_embeds_scale"], [1.0, 0.2])
        self.assertEqual(calls["prior"]["pooled_prompt_embeds_scale"], [1.0, 0.2])

    def test_flux_kontext_stitches_multiple_references_through_generic_edit(self):
        received = {}

        class FakeKontextPipeline:
            _execution_device = "cpu"
            _modiff_image_adapter = IMAGE_PIPELINE_ADAPTERS["FluxKontextPipeline"]

            def __call__(self, **kwargs):
                received.update(kwargs)
                return type("Result", (), {"images": [Image.new("RGB", (32, 32), "white")]})()

        references = [Image.new("RGB", (16, 16), "red"), Image.new("RGB", (8, 16), "blue")]
        result = Edit("kontext-multi-probe").execute(
            pipeline=FakeKontextPipeline(),
            image=references,
            prompt="use first for identity and second for style",
            width=32,
            height=32,
            num_inference_steps=2,
        )

        self.assertIsInstance(received["image"], Image.Image)
        self.assertEqual(received["image"].size, (24, 16))
        self.assertEqual(received["image"].getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(received["image"].getpixel((23, 0)), (0, 0, 255))
        self.assertEqual(result["images"][0].size, (32, 32))

    def test_qwen_inpaint_uses_generic_guidance_and_crop_aliases(self):
        received = {}

        class FakeResult:
            images = [Image.new("RGB", (16, 16), "white")]

        class FakeQwenPipeline:
            _execution_device = "cpu"

            def __call__(
                self,
                *,
                image,
                mask_image,
                prompt,
                num_inference_steps,
                generator,
                output_type,
                return_dict,
                true_cfg_scale,
                padding_mask_crop,
                strength,
            ):
                received.update(
                    true_cfg_scale=true_cfg_scale,
                    padding_mask_crop=padding_mask_crop,
                    strength=strength,
                )
                return FakeResult()

        node = Inpaint("qwen-generic-probe")
        result = node.execute(
            pipeline=FakeQwenPipeline(),
            image=Image.new("RGB", (16, 16), "black"),
            mask_image=Image.new("L", (16, 16), "white"),
            prompt="replace the object",
            num_inference_steps=2,
            guidance_scale=4.0,
            padding_mask_crop=128,
            strength=1.0,
            output_type="pil",
        )

        self.assertEqual(received, {"true_cfg_scale": 4.0, "padding_mask_crop": 128, "strength": 1.0})
        self.assertEqual(result["images"][0].size, (16, 16))

    def test_zero_padding_mask_crop_maps_to_diffusers_no_crop(self):
        received = {}

        class Pipeline:
            def __call__(self, padding_mask_crop=None, **kwargs):
                return None

        adapter = IMAGE_PIPELINE_ADAPTERS["QwenImageEditInpaintPipeline"]
        adapter.apply_generation_parameters(
            Pipeline(),
            {"padding_mask_crop": 0},
            received,
        )

        self.assertNotIn("padding_mask_crop", received)

    def test_qwen_generic_loader_uses_the_official_default_repository(self):
        loaded = []

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, repo, **_kwargs):
                loaded.append(repo)
                return cls()

        node = LoadPipeline("qwen-loader-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=FakePipeline),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            node.execute(
                pipeline_class="QwenImageEditInpaintPipeline",
                mode="inpaint",
                auto_offload=False,
                offload_mode="none",
            )

        self.assertEqual(loaded, ["Qwen/Qwen-Image-Edit"])

    def test_generic_inpaint_preserves_every_black_mask_pixel(self):
        self.assertEqual(Inpaint.params["output_type"]["options"], ["pil"])

        class FakeResult:
            images = [Image.new("RGB", (4, 2), (200, 210, 220))]

        class FakePipeline:
            _execution_device = "cpu"

            def __call__(self, **_kwargs):
                return FakeResult()

        source = Image.new("RGB", (4, 2), (10, 20, 30))
        mask = Image.new("L", (4, 2), 0)
        for x in (2, 3):
            for y in (0, 1):
                mask.putpixel((x, y), 255)

        result = Inpaint("mask-contract-probe").execute(
            pipeline=FakePipeline(),
            image=source,
            mask_image=mask,
            prompt="replace",
            num_inference_steps=1,
            output_type="pil",
        )

        self.assertEqual(result["images"][0].getpixel((0, 0)), (10, 20, 30))
        self.assertEqual(result["images"][0].getpixel((3, 1)), (200, 210, 220))

    def test_generic_step_callback_propagates_interrupt_to_pipeline(self):
        class FakePipeline:
            _interrupt = False
            _num_timesteps = 4

            def __call__(self, *, callback_on_step_end=None, callback_on_step_end_tensor_inputs=None):
                pass

        node = Inpaint("interrupt-probe")
        node._interrupt = True
        progress = []
        node.progress = lambda *args, **kwargs: progress.append((args, kwargs))
        pipeline = FakePipeline()
        call_kwargs = {}

        add_progress_callback(node, pipeline, call_kwargs, 4)
        self.assertEqual(progress[0][0], (0,))
        self.assertEqual(progress[0][1]["phase"], "denoising")
        self.assertEqual(progress[0][1]["current_step"], 0)
        self.assertEqual(progress[0][1]["total_steps"], 4)
        with self.assertRaisesRegex(InterruptedError, "interrupted by the user"):
            call_kwargs["callback_on_step_end"](pipeline, 0, 1, {})

        self.assertTrue(pipeline._interrupt)

    def test_generic_execution_exposes_active_pipeline_during_inference(self):
        node = Inpaint("active-pipeline-probe")
        observed = []

        class FakeResult:
            images = [Image.new("RGB", (16, 16), "white")]

        class FakePipeline:
            _execution_device = "cpu"

            def __call__(self, **_kwargs):
                observed.append(node._active_pipeline is self)
                return FakeResult()

        node.execute(
            pipeline=FakePipeline(),
            image=Image.new("RGB", (16, 16), "black"),
            mask_image=Image.new("L", (16, 16), "white"),
            prompt="replace",
            num_inference_steps=1,
        )

        self.assertEqual(observed, [True])
        self.assertIsNone(node._active_pipeline)

    def test_adapter_uses_only_the_app_managed_cached_weight(self):
        calls = []
        cached_weight = Path("/cache/revision/adapter.safetensors")

        class FakePipeline:
            def load_lora_weights(self, path, **kwargs):
                calls.append((path, kwargs))

        with patch("utils.huggingface.cached_file_path", return_value=str(cached_weight)):
            LoadAdapter("adapter-probe").execute(
                pipeline=FakePipeline(),
                adapter_path={"source": "hub", "value": "unit/adapter"},
                weight_name="adapter.safetensors",
                adapter_name="gallery",
                scale=0.8,
            )

        self.assertEqual(calls[0][0], str(cached_weight.parent))
        self.assertEqual(calls[0][1]["weight_name"], "adapter.safetensors")

    def test_adapter_missing_from_app_cache_fails_before_pipeline_load(self):
        class FakePipeline:
            def load_lora_weights(self, *_args, **_kwargs):
                raise AssertionError("must not download or load")

        with patch("utils.huggingface.cached_file_path", return_value=False):
            with self.assertRaisesRegex(FileNotFoundError, "Model Manager"):
                LoadAdapter("missing-adapter-probe").execute(
                    pipeline=FakePipeline(),
                    adapter_path={"source": "hub", "value": "unit/adapter"},
                    weight_name="adapter.safetensors",
                )

    def test_adapter_verifies_pinned_hash_and_replaces_previous_pipeline_adapters(self):
        events = []

        class FakePipeline:
            def unload_lora_weights(self):
                events.append("unload")

            def load_lora_weights(self, path, **kwargs):
                events.append(("load", path, kwargs))

            def set_adapters(self, names, weights):
                events.append(("activate", names, weights))

        with tempfile.TemporaryDirectory() as directory:
            adapter_file = Path(directory) / "adapter.safetensors"
            adapter_file.write_bytes(b"pinned adapter bytes")
            expected = hashlib.sha256(adapter_file.read_bytes()).hexdigest()
            with patch("utils.huggingface.cached_file_path", return_value=str(adapter_file)):
                LoadAdapter("verified-adapter-probe").execute(
                    pipeline=FakePipeline(),
                    adapter_path={"source": "hub", "value": "unit/adapter"},
                    weight_name=adapter_file.name,
                    expected_sha256=expected,
                    adapter_name="theme",
                    scale=0.75,
                )

        self.assertEqual(events[0], "unload")
        self.assertEqual(events[1][0], "load")
        self.assertEqual(events[2], ("activate", ["theme"], [0.75]))

    def test_chained_adapters_preserve_prior_adapter_names_and_scales(self):
        events = []

        class FakePipeline:
            def unload_lora_weights(self):
                events.append("unload")

            def load_lora_weights(self, path, **kwargs):
                events.append(("load", path, kwargs))

            def set_adapters(self, names, weights):
                events.append(("activate", names, weights))

        pipeline = FakePipeline()
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.safetensors"
            second = Path(directory) / "second.safetensors"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            with patch(
                "utils.huggingface.cached_file_path",
                side_effect=[str(first), str(second)],
            ):
                LoadAdapter("first-adapter").execute(
                    pipeline=pipeline,
                    adapter_path={"source": "hub", "value": "unit/first"},
                    weight_name=first.name,
                    adapter_name="cinematic",
                    scale=0.8,
                    replace_existing=True,
                )
                LoadAdapter("second-adapter").execute(
                    pipeline=pipeline,
                    adapter_path={"source": "hub", "value": "unit/second"},
                    weight_name=second.name,
                    adapter_name="render_3d",
                    scale=0.18,
                    replace_existing=False,
                )

        self.assertEqual(events.count("unload"), 1)
        self.assertEqual(events[-1], ("activate", ["cinematic", "render_3d"], [0.8, 0.18]))
        self.assertEqual(pipeline._modiff_adapter_scales, {"cinematic": 0.8, "render_3d": 0.18})

    def test_adapter_scale_zero_remains_zero(self):
        activations = []

        class FakePipeline:
            def load_lora_weights(self, *_args, **_kwargs):
                pass

            def set_adapters(self, names, weights):
                activations.append((names, weights))

        with patch("utils.huggingface.cached_file_path", return_value="/cache/adapter.safetensors"):
            LoadAdapter("zero-scale-adapter").execute(
                pipeline=FakePipeline(),
                adapter_path={"source": "hub", "value": "unit/adapter"},
                weight_name="adapter.safetensors",
                adapter_name="optional",
                scale=0,
                replace_existing=False,
            )

        self.assertEqual(activations, [(["optional"], [0.0])])

    def test_adapter_hash_mismatch_does_not_mutate_pipeline(self):
        class FakePipeline:
            def unload_lora_weights(self):
                raise AssertionError("hash validation must happen before pipeline mutation")

            def load_lora_weights(self, *_args, **_kwargs):
                raise AssertionError("hash validation must happen before adapter loading")

        with tempfile.TemporaryDirectory() as directory:
            adapter_file = Path(directory) / "adapter.safetensors"
            adapter_file.write_bytes(b"unexpected bytes")
            with patch("utils.huggingface.cached_file_path", return_value=str(adapter_file)):
                with self.assertRaisesRegex(ValueError, "pinned SHA-256"):
                    LoadAdapter("invalid-adapter-probe").execute(
                        pipeline=FakePipeline(),
                        adapter_path={"source": "hub", "value": "unit/adapter"},
                        weight_name=adapter_file.name,
                        expected_sha256="0" * 64,
                    )


class FakeRequest:
    match_info = {}


class DiffusersImageNodeRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_nodes_route_exposes_flattened_facade_contracts(self):
        server = WebServer(module_registry.MODULE_MAP)
        response = await server.nodes(FakeRequest())
        nodes = json.loads(response.text)["nodes"]

        for action in ("Edit", "Inpaint", "ControlGenerate"):
            with self.subTest(node=action):
                key = f"modules.DiffusersImage.{action}"
                self.assertIn(key, nodes)
                self.assertTrue(nodes[key]["resizable"])
                self.assertEqual(nodes[key]["params"]["images"]["display"], "output")


if __name__ == "__main__":
    unittest.main()
