from source_contract_helpers import source_sha256

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from modiff.model_artifact_catalog import catalog_repository_pin, catalog_revision
from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION
from modiff.server import STUDIO_MODEL_CAPABILITIES
from modiff.studio_execution_specs import (
    QWEN_CONTROLNET_FILES,
    QWEN_IMAGE_2512_DIFFUSERS_FILES,
    QWEN_IMAGE_LAYERED_DIFFUSERS_FILES,
    studio_model_dependencies_for_pair,
)
from modules.DiffusersImage.main import (
    IMAGE_PIPELINE_ADAPTERS,
    QWEN_IMAGE_2512_REPO,
    QWEN_IMAGE_CONTROLNET_REPO,
    QWEN_IMAGE_LAYERED_REPO,
    ControlGenerate,
    LayerDecompose,
    LoadPipeline,
    _tag_image_pipeline,
    image_pipeline_contract,
)


LAYERED_SOURCE_SHA256 = "d857259e9d1ed65de4a27dfb20f2689f307ae81e792d8976f7394bbb2a2a90d8"
CONTROLNET_SOURCE_SHA256 = "df355aa553805f5ce87b1717a23b31985dffaaa742cfbc4fa4993663b69c94ba"
QWEN_2512_REVISION = "25468b98e3276ca6700de15c6628e51b7de54a26"
QWEN_LAYERED_REVISION = "8f0ca708dfff6ba1dd5f2d85d78f8c108a040bcf"
QWEN_CONTROLNET_REVISION = "b13036f066d6dee7c20513e263d3d673055e9de8"


def tag_pipeline(pipeline, pipeline_class, mode, repository):
    adapter = IMAGE_PIPELINE_ADAPTERS[pipeline_class]
    _tag_image_pipeline(
        pipeline,
        adapter,
        mode,
        repository,
        "hub",
        catalog_revision(repository),
        conditioning_repo=adapter.default_conditioning_repo,
        conditioning_revision=(
            catalog_revision(adapter.default_conditioning_repo)
            if adapter.default_conditioning_repo
            else None
        ),
    )
    return pipeline


class QwenImageLayeredPipeline:
    _execution_device = "cpu"

    def __init__(self, *, batches=1, numpy_output=False):
        self.batches = batches
        self.numpy_output = numpy_output
        self.calls = []

    def __call__(
        self,
        *,
        image,
        prompt,
        negative_prompt,
        true_cfg_scale,
        layers,
        num_inference_steps,
        generator,
        output_type,
        return_dict,
        max_sequence_length,
        resolution,
        cfg_normalize,
        use_en_prompt,
    ):
        self.calls.append(dict(locals()))
        if self.numpy_output:
            batch = np.zeros((layers, resolution, resolution, 3), dtype=np.float32)
        else:
            batch = [Image.new("RGBA", (resolution, resolution), (index, 0, 0, 255)) for index in range(layers)]
        return SimpleNamespace(images=[batch for _index in range(self.batches)])


class QwenImageControlNetPipeline:
    _execution_device = "cpu"

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        *,
        prompt,
        negative_prompt,
        true_cfg_scale,
        height,
        width,
        num_inference_steps,
        control_guidance_start,
        control_guidance_end,
        control_image,
        controlnet_conditioning_scale,
        generator,
        output_type,
        return_dict,
        max_sequence_length,
    ):
        self.calls.append(dict(locals()))
        return SimpleNamespace(images=[Image.new("RGB", (width, height), "white")])


class QwenStandardSafeWaveTests(unittest.TestCase):
    def test_exact_pinned_upstream_sources_and_call_signatures_are_preserved(self):
        import diffusers

        self.assertEqual(PINNED_DIFFUSERS_REVISION, "2f7e0154a9db246e95c9ede43edba7db5b130805")
        cases = (
            (
                "pipeline_qwenimage_layered.py",
                "QwenImageLayeredPipeline",
                LAYERED_SOURCE_SHA256,
                {
                    "image",
                    "prompt",
                    "negative_prompt",
                    "true_cfg_scale",
                    "layers",
                    "num_inference_steps",
                    "generator",
                    "output_type",
                    "return_dict",
                    "max_sequence_length",
                    "resolution",
                    "cfg_normalize",
                    "use_en_prompt",
                },
            ),
            (
                "pipeline_qwenimage_controlnet.py",
                "QwenImageControlNetPipeline",
                CONTROLNET_SOURCE_SHA256,
                {
                    "prompt",
                    "negative_prompt",
                    "true_cfg_scale",
                    "height",
                    "width",
                    "num_inference_steps",
                    "control_guidance_start",
                    "control_guidance_end",
                    "control_image",
                    "controlnet_conditioning_scale",
                    "generator",
                    "output_type",
                    "return_dict",
                    "max_sequence_length",
                },
            ),
        )
        diffusers_root = Path(diffusers.__file__).resolve().parent
        for filename, class_name, expected_digest, required_parameters in cases:
            with self.subTest(pipeline=class_name):
                source_path = diffusers_root / "pipelines" / "qwenimage" / filename
                self.assertTrue(source_path.is_file())
                self.assertEqual(source_sha256(source_path), expected_digest)
                tree = ast.parse(source_path.read_text(encoding="utf-8"))
                pipeline_node = next(
                    node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
                )
                call_node = next(
                    node
                    for node in pipeline_node.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__call__"
                )
                parameters = {
                    argument.arg
                    for argument in (
                        *call_node.args.posonlyargs,
                        *call_node.args.args,
                        *call_node.args.kwonlyargs,
                    )
                }
                self.assertTrue(required_parameters.issubset(parameters))

    def test_exact_adapters_are_separate_and_reuse_only_admitted_artifacts(self):
        control = IMAGE_PIPELINE_ADAPTERS["QwenImageControlNetPipeline"]
        layered = IMAGE_PIPELINE_ADAPTERS["QwenImageLayeredPipeline"]

        self.assertEqual(control.modes, frozenset({"control_image"}))
        self.assertEqual(control.default_repo, QWEN_IMAGE_2512_REPO)
        self.assertEqual(control.default_conditioning_repo, QWEN_IMAGE_CONTROLNET_REPO)
        self.assertEqual(control.conditioning_component_class, "QwenImageControlNetModel")
        self.assertEqual(control.conditioning_component_parameter, "controlnet")
        self.assertEqual(control.conditioning_scale_parameter, "controlnet_conditioning_scale")
        self.assertEqual(
            control.conditioning_config_requirements,
            (
                ("in_channels", 64),
                ("out_channels", 16),
                ("extra_condition_channels", 0),
                ("num_layers", 5),
            ),
        )
        self.assertEqual(layered.modes, frozenset({"layer_decomposition"}))
        self.assertEqual(layered.default_repo, QWEN_IMAGE_LAYERED_REPO)
        self.assertEqual((layered.min_layers, layered.max_layers), (1, 10))
        self.assertEqual(layered.layer_resolutions, (640, 1024))
        self.assertTrue(control.safe_serialization_required)
        self.assertTrue(layered.safe_serialization_required)

        pins = {
            QWEN_IMAGE_2512_REPO: QWEN_2512_REVISION,
            QWEN_IMAGE_LAYERED_REPO: QWEN_LAYERED_REVISION,
            QWEN_IMAGE_CONTROLNET_REPO: QWEN_CONTROLNET_REVISION,
        }
        for repository, revision in pins.items():
            with self.subTest(repository=repository):
                pin = catalog_repository_pin(repository)
                self.assertEqual(pin["revision"], revision)
                self.assertEqual(pin["license"], "apache-2.0")

        unsafe_suffixes = (".bin", ".ckpt", ".pt", ".pth", ".py")
        for files in (
            QWEN_IMAGE_2512_DIFFUSERS_FILES,
            QWEN_IMAGE_LAYERED_DIFFUSERS_FILES,
            QWEN_CONTROLNET_FILES,
        ):
            self.assertFalse(any(path.endswith(unsafe_suffixes) for path in files))
        self.assertEqual(
            STUDIO_MODEL_CAPABILITIES["QwenImageControlNetPipeline"]["downloadFiles"],
            QWEN_IMAGE_2512_DIFFUSERS_FILES,
        )
        self.assertEqual(
            STUDIO_MODEL_CAPABILITIES["QwenImageLayeredPipeline"]["downloadFiles"],
            QWEN_IMAGE_LAYERED_DIFFUSERS_FILES,
        )
        self.assertEqual(
            studio_model_dependencies_for_pair("QwenImageControlNetPipeline", "control_image"),
            studio_model_dependencies_for_pair("QwenImageModularPipeline", "control_image"),
        )

    def test_layer_decomposition_action_passes_only_the_exact_generic_contract(self):
        source = Image.new("RGB", (32, 24), "black")
        pipeline = tag_pipeline(
            QwenImageLayeredPipeline(),
            "QwenImageLayeredPipeline",
            "layer_decomposition",
            QWEN_IMAGE_LAYERED_REPO,
        )
        result = LayerDecompose("qwen-layer-test").execute(
            pipeline=pipeline,
            image=source,
            prompt="separate foreground, subject, and background",
            negative_prompt="artifacts",
            seed=9,
            num_inference_steps=12,
            guidance_scale=3.75,
            max_sequence_length=128,
            layers=3,
            resolution=640,
            cfg_normalize=True,
            use_en_prompt=False,
            output_type="pil",
        )

        self.assertEqual(len(result["images"]), 3)
        self.assertEqual((result["width_out"], result["height_out"]), (640, 640))
        call = pipeline.calls[0]
        self.assertEqual(call["image"], source)
        self.assertEqual(call["true_cfg_scale"], 3.75)
        self.assertEqual(call["layers"], 3)
        self.assertEqual(call["resolution"], 640)
        self.assertTrue(call["cfg_normalize"])
        self.assertFalse(call["use_en_prompt"])
        self.assertNotIn("width", call)
        self.assertNotIn("height", call)

    def test_layer_decomposition_rejects_invalid_inputs_and_multi_batch_outputs(self):
        source = Image.new("RGB", (16, 16), "black")
        pipeline = tag_pipeline(
            QwenImageLayeredPipeline(),
            "QwenImageLayeredPipeline",
            "layer_decomposition",
            QWEN_IMAGE_LAYERED_REPO,
        )
        valid = {"pipeline": pipeline, "image": source, "prompt": "layers"}
        invalid = (
            ({"layers": 0}, "between 1 and 10"),
            ({"layers": 11}, "between 1 and 10"),
            ({"resolution": 800}, "exactly one of: 640, 1024"),
            ({"cfg_normalize": "true"}, "must be a boolean"),
            ({"use_en_prompt": 0}, "must be a boolean"),
            ({"prompt": ["one", "two"]}, "one prompt"),
            ({"image": [source]}, "one PIL image, not a list"),
            ({"output_type": "pt"}, "exactly one of: np, pil"),
        )
        for values, message in invalid:
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, message):
                LayerDecompose("invalid-layer-test").execute(**{**valid, **values})
        self.assertEqual(pipeline.calls, [])

        multi_batch = tag_pipeline(
            QwenImageLayeredPipeline(batches=2),
            "QwenImageLayeredPipeline",
            "layer_decomposition",
            QWEN_IMAGE_LAYERED_REPO,
        )
        with self.assertRaisesRegex(ValueError, "exactly one output batch"):
            LayerDecompose("multi-batch-layer-test").execute(
                pipeline=multi_batch,
                image=source,
                prompt="layers",
                layers=2,
            )

    def test_layer_decomposition_flattens_exactly_one_numpy_batch(self):
        source = Image.new("RGB", (16, 16), "black")
        pipeline = tag_pipeline(
            QwenImageLayeredPipeline(numpy_output=True),
            "QwenImageLayeredPipeline",
            "layer_decomposition",
            QWEN_IMAGE_LAYERED_REPO,
        )
        result = LayerDecompose("numpy-layer-test").execute(
            pipeline=pipeline,
            image=source,
            prompt="layers",
            layers=2,
            resolution=640,
            output_type="np",
        )
        self.assertEqual(result["images"].shape, (2, 640, 640, 3))
        self.assertEqual((result["width_out"], result["height_out"]), (640, 640))

    def test_control_action_passes_exact_control_window_and_union_scale(self):
        control_image = Image.new("RGB", (32, 24), "black")
        pipeline = tag_pipeline(
            QwenImageControlNetPipeline(),
            "QwenImageControlNetPipeline",
            "control_image",
            QWEN_IMAGE_2512_REPO,
        )
        result = ControlGenerate("qwen-control-test").execute(
            pipeline=pipeline,
            control_image=control_image,
            prompt="controlled subject",
            negative_prompt="artifacts",
            width=64,
            height=48,
            seed=3,
            num_inference_steps=15,
            guidance_scale=4.25,
            max_sequence_length=128,
            conditioning_scale=0.8,
            control_guidance_start=0.2,
            control_guidance_end=0.7,
            layers=4,
            resolution=640,
            cfg_normalize=False,
            use_en_prompt=False,
            output_type="pil",
        )
        self.assertEqual((result["width_out"], result["height_out"]), (64, 48))
        call = pipeline.calls[0]
        self.assertEqual(call["control_image"], control_image)
        self.assertEqual(call["true_cfg_scale"], 4.25)
        self.assertEqual(call["controlnet_conditioning_scale"], 0.8)
        self.assertEqual(call["control_guidance_start"], 0.2)
        self.assertEqual(call["control_guidance_end"], 0.7)

        with self.assertRaisesRegex(ValueError, "start cannot exceed its end"):
            ControlGenerate("invalid-control-window").execute(
                pipeline=pipeline,
                control_image=control_image,
                prompt="controlled subject",
                control_guidance_start=0.8,
                control_guidance_end=0.2,
            )
        self.assertEqual(len(pipeline.calls), 1)

    def test_loader_reuses_exact_pins_and_never_requests_unsafe_or_remote_loading(self):
        component_calls = []
        pipeline_calls = []

        class FakeQwenImageControlNetModel:
            @classmethod
            def from_pretrained(cls, repository, **kwargs):
                component = SimpleNamespace(
                    config=SimpleNamespace(
                        in_channels=64,
                        out_channels=16,
                        extra_condition_channels=0,
                        num_layers=5,
                    )
                )
                component_calls.append((repository, kwargs, component))
                return component

        class FakeQwenImageControlNetPipeline:
            @classmethod
            def from_pretrained(cls, repository, **kwargs):
                pipeline_calls.append((repository, kwargs))
                return cls()

        loader = LoadPipeline("qwen-control-loader-test")
        loader.progress = Mock()
        loader.mm_add = Mock()
        loader.diffusers_loading_progress = lambda: nullcontext()
        classes = {
            "QwenImageControlNetModel": FakeQwenImageControlNetModel,
            "QwenImageControlNetPipeline": FakeQwenImageControlNetPipeline,
        }
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", side_effect=classes.__getitem__),
            patch("modules.DiffusersImage.main.apply_pipeline_offload", return_value=SimpleNamespace(method="test")),
            patch("modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline", return_value={}),
            patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("download attempted")),
            patch("huggingface_hub.snapshot_download", side_effect=AssertionError("download attempted")),
        ):
            result = loader.execute(
                pipeline_class="QwenImageControlNetPipeline",
                mode="control_image",
                model_id={"source": "hub", "value": QWEN_IMAGE_2512_REPO},
                revision="",
                conditioning_kind="controlnet",
                conditioning_model_id={"source": "hub", "value": QWEN_IMAGE_CONTROLNET_REPO},
                conditioning_revision="",
                dtype="bfloat16",
                device="cpu",
                quantization_mode="none",
                quantized_components=[],
                auto_offload=True,
                offload_mode="model_cpu",
            )

        self.assertIsInstance(result["pipeline"], FakeQwenImageControlNetPipeline)
        self.assertEqual(component_calls[0][0], QWEN_IMAGE_CONTROLNET_REPO)
        self.assertEqual(component_calls[0][1]["revision"], QWEN_CONTROLNET_REVISION)
        self.assertTrue(component_calls[0][1]["use_safetensors"])
        self.assertNotIn("trust_remote_code", component_calls[0][1])
        self.assertEqual(pipeline_calls[0][0], QWEN_IMAGE_2512_REPO)
        self.assertEqual(pipeline_calls[0][1]["revision"], QWEN_2512_REVISION)
        self.assertTrue(pipeline_calls[0][1]["use_safetensors"])
        self.assertIs(pipeline_calls[0][1]["controlnet"], component_calls[0][2])
        self.assertNotIn("trust_remote_code", pipeline_calls[0][1])
        self.assertEqual(result["pipeline"]._modiff_image_pipeline_class, "QwenImageControlNetPipeline")
        self.assertEqual(result["pipeline"]._modiff_image_mode, "control_image")
        self.assertEqual(result["pipeline"]._modiff_conditioning_repo, QWEN_IMAGE_CONTROLNET_REPO)

    def test_loader_rejects_union_config_mismatch_before_loading_base(self):
        class BadQwenImageControlNetModel:
            @classmethod
            def from_pretrained(cls, _repository, **_kwargs):
                return SimpleNamespace(
                    config=SimpleNamespace(
                        in_channels=32,
                        out_channels=16,
                        extra_condition_channels=0,
                        num_layers=5,
                    )
                )

        class ForbiddenBasePipeline:
            @classmethod
            def from_pretrained(cls, _repository, **_kwargs):
                raise AssertionError("base pipeline loaded after config mismatch")

        loader = LoadPipeline("qwen-control-mismatch-test")
        loader.progress = Mock()
        loader.diffusers_loading_progress = lambda: nullcontext()
        classes = {
            "QwenImageControlNetModel": BadQwenImageControlNetModel,
            "QwenImageControlNetPipeline": ForbiddenBasePipeline,
        }
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", side_effect=classes.__getitem__),
            patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("download attempted")),
            patch("huggingface_hub.snapshot_download", side_effect=AssertionError("download attempted")),
            self.assertRaisesRegex(RuntimeError, "config field 'in_channels' must be 64"),
        ):
            loader.execute(
                pipeline_class="QwenImageControlNetPipeline",
                mode="control_image",
                model_id={"source": "hub", "value": QWEN_IMAGE_2512_REPO},
                conditioning_kind="controlnet",
                conditioning_model_id={"source": "hub", "value": QWEN_IMAGE_CONTROLNET_REPO},
                dtype="bfloat16",
                device="cpu",
                quantization_mode="none",
                quantized_components=[],
            )

    def test_layered_loader_uses_only_exact_safe_pin_without_remote_code(self):
        calls = []

        class FakeLayeredPipeline:
            @classmethod
            def from_pretrained(cls, repository, **kwargs):
                calls.append((repository, kwargs))
                return cls()

        loader = LoadPipeline("qwen-layered-loader-test")
        loader.progress = Mock()
        loader.mm_add = Mock()
        loader.diffusers_loading_progress = lambda: nullcontext()
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=FakeLayeredPipeline),
            patch("modules.DiffusersImage.main.apply_pipeline_offload", return_value=SimpleNamespace(method="test")),
            patch("modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline", return_value={}),
            patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("download attempted")),
            patch("huggingface_hub.snapshot_download", side_effect=AssertionError("download attempted")),
        ):
            result = loader.execute(
                pipeline_class="QwenImageLayeredPipeline",
                mode="layer_decomposition",
                model_id={"source": "hub", "value": QWEN_IMAGE_LAYERED_REPO},
                revision="",
                dtype="bfloat16",
                device="cpu",
                quantization_mode="none",
                quantized_components=[],
                auto_offload=True,
                offload_mode="model_cpu",
            )

        self.assertIsInstance(result["pipeline"], FakeLayeredPipeline)
        self.assertEqual(calls[0][0], QWEN_IMAGE_LAYERED_REPO)
        self.assertEqual(calls[0][1]["revision"], QWEN_LAYERED_REVISION)
        self.assertTrue(calls[0][1]["use_safetensors"])
        self.assertNotIn("trust_remote_code", calls[0][1])
        self.assertEqual(result["pipeline"]._modiff_image_pipeline_class, "QwenImageLayeredPipeline")
        self.assertEqual(result["pipeline"]._modiff_image_mode, "layer_decomposition")

    def test_layer_contract_exposes_only_generic_action_and_exact_bounds(self):
        contract = image_pipeline_contract(
            IMAGE_PIPELINE_ADAPTERS["QwenImageLayeredPipeline"],
            "layer_decomposition",
        )
        self.assertEqual(contract["actions"], {"LayerDecompose": ["layer_decomposition"]})
        self.assertEqual(contract["fieldParams"]["layers"]["min"], 1)
        self.assertEqual(contract["fieldParams"]["layers"]["max"], 10)
        self.assertEqual(contract["fieldParams"]["resolution"]["options"], [640, 1024])
        self.assertTrue(contract["fieldParams"]["width"]["hidden"])
        self.assertTrue(contract["fieldParams"]["height"]["hidden"])


if __name__ == "__main__":
    unittest.main()
