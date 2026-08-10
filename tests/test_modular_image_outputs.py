import sys
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.Image.main import Preview  # noqa: E402
from modules.ModularDiffusers.latents import (  # noqa: E402
    ImageEncode,
    flatten_pil_images,
    prepare_image_for_vae_pipeline,
)
from modules.ModularDiffusers.embeddings import EncodePrompt  # noqa: E402


class ModularImageOutputTests(unittest.TestCase):
    def test_qwen_layered_vae_input_adds_an_opaque_alpha_channel(self):
        class QwenImageLayeredModularPipeline:
            pass

        source = Image.new("RGB", (8, 8), "red")
        prepared = prepare_image_for_vae_pipeline(source, QwenImageLayeredModularPipeline)

        self.assertEqual(prepared.mode, "RGBA")
        self.assertEqual(prepared.getpixel((0, 0)), (255, 0, 0, 255))
        self.assertEqual(source.mode, "RGB")

    def test_qwen_layered_vae_input_normalizes_nested_pil_batches(self):
        class QwenImageLayeredModularPipeline:
            pass

        source = [Image.new("RGB", (8, 8), "red"), (Image.new("RGBA", (8, 8), "blue"),)]
        prepared = prepare_image_for_vae_pipeline(source, QwenImageLayeredModularPipeline)

        self.assertEqual(prepared[0].mode, "RGBA")
        self.assertEqual(prepared[1][0].mode, "RGBA")

    def test_other_vae_pipelines_keep_their_source_unchanged(self):
        class QwenImageEditModularPipeline:
            pass

        source = Image.new("RGB", (8, 8), "red")

        self.assertIs(prepare_image_for_vae_pipeline(source, QwenImageEditModularPipeline), source)

    def test_image_encode_applies_layered_rgba_contract_before_running_pipeline(self):
        class QwenImageLayeredModularPipeline:
            pass

        observed = {}

        class FakePipeline:
            def __call__(self, **kwargs):
                observed.update(kwargs)
                return {"latents": "encoded"}

        class FakeBlocks:
            component_names = []
            input_names = ["image"]

            @staticmethod
            def init_pipeline(*, components_manager):
                return FakePipeline()

        node_config = {
            "params": {},
            "model_input_names": [],
            "input_names": ["image"],
            "output_names": ["latents"],
        }
        source = Image.new("RGB", (8, 8), "red")
        node = ImageEncode()
        node._pipeline_class = QwenImageLayeredModularPipeline

        with patch(
            "modules.ModularDiffusers.latents.require_modiff_node_contract",
            return_value=(FakeBlocks(), node_config),
        ):
            result = node.execute(vae={"repo_id": "fixture/layered"}, image=source)

        self.assertEqual(result["latents"], "encoded")
        summary = json.loads(result["encode_summary_data"])
        self.assertEqual(summary["schemaVersion"], 1)
        self.assertEqual(summary["status"], "encoded")
        self.assertGreaterEqual(summary["elapsedSeconds"], 0)
        self.assertEqual(observed["image"].mode, "RGBA")
        self.assertEqual(observed["image"].getpixel((0, 0)), (255, 0, 0, 255))
        self.assertEqual(source.mode, "RGB")

    def test_layered_prompt_controls_forward_exactly_without_loading_weights(self):
        observed = {}

        class FakeState:
            @staticmethod
            def get_by_kwargs(_name):
                return {"prompt_embeds": "encoded"}

        class FakePipeline:
            blocks = type("PipelineBlocks", (), {"doc": "fixture-doc"})()

            def __call__(self, **kwargs):
                observed.update(kwargs)
                return FakeState()

            @staticmethod
            def update_components(**_kwargs):
                return None

        class FakeBlocks:
            component_names = []
            input_names = [
                "image",
                "resolution",
                "prompt",
                "use_en_prompt",
                "negative_prompt",
                "max_sequence_length",
            ]

            @staticmethod
            def init_pipeline(*, components_manager):
                return FakePipeline()

        node_config = {
            "params": {
                "resolution": {"type": "int", "options": [640, 1024]},
                "use_en_prompt": {"type": "boolean"},
                "max_sequence_length": {"type": "int", "min": 1, "max": 1024},
            },
            "model_input_names": [],
            "input_names": [
                "prompt",
                "negative_prompt",
                "image",
                "resolution",
                "use_en_prompt",
                "max_sequence_length",
            ],
            "output_names": ["embeddings", "doc"],
        }
        node = EncodePrompt()
        node._pipeline_class = type("OpaqueLayeredPipeline", (), {})

        with patch(
            "modules.ModularDiffusers.embeddings.require_modiff_node_contract",
            return_value=(FakeBlocks(), node_config),
        ):
            result = node.execute(
                text_encoders={"repo_id": "fixture/layered"},
                prompt="separate the subject",
                negative_prompt="",
                image="fixture-image",
                resolution="1024",
                use_en_prompt=True,
                max_sequence_length="768",
            )

        self.assertEqual(
            observed,
            {
                "prompt": "separate the subject",
                "negative_prompt": "",
                "image": "fixture-image",
                "resolution": 1024,
                "use_en_prompt": True,
                "max_sequence_length": 768,
            },
        )
        self.assertEqual(result, {"embeddings": {"prompt_embeds": "encoded"}, "doc": "fixture-doc"})

        invalid_controls = (
            (
                {"resolution": "768", "use_en_prompt": False, "max_sequence_length": "768"},
                "resolution.*one of",
            ),
            (
                {"resolution": "640", "use_en_prompt": False, "max_sequence_length": "0"},
                "max_sequence_length.*greater than",
            ),
            (
                {"resolution": "640", "use_en_prompt": False, "max_sequence_length": "1025"},
                "max_sequence_length.*less than",
            ),
            (
                {"resolution": "640", "use_en_prompt": "false", "max_sequence_length": "768"},
                "use_en_prompt.*boolean",
            ),
            (
                {"resolution": 1024.9, "use_en_prompt": False, "max_sequence_length": "768"},
                "resolution.*expected int",
            ),
            (
                {"resolution": "640", "use_en_prompt": False, "max_sequence_length": 1024.9},
                "max_sequence_length.*expected int",
            ),
            (
                {"resolution": "640", "use_en_prompt": False, "max_sequence_length": True},
                "max_sequence_length.*expected int",
            ),
            (
                {"resolution": "640", "use_en_prompt": False, "max_sequence_length": float("nan")},
                "max_sequence_length.*expected int",
            ),
            (
                {"resolution": "640", "use_en_prompt": False, "max_sequence_length": "01024"},
                "max_sequence_length.*expected int",
            ),
            (
                {"resolution": "640", "use_en_prompt": False, "max_sequence_length": "-0"},
                "max_sequence_length.*expected int",
            ),
        )
        for overrides, message in invalid_controls:
            with self.subTest(overrides=overrides), patch(
                "modules.ModularDiffusers.embeddings.require_modiff_node_contract",
                return_value=(FakeBlocks(), node_config),
            ):
                observed.clear()
                with self.assertRaisesRegex(ValueError, message):
                    node.execute(
                        text_encoders={"repo_id": "fixture/layered"},
                        prompt="separate the subject",
                        negative_prompt="",
                        image="fixture-image",
                        **overrides,
                    )
                self.assertEqual(observed, {})

    def test_layered_vae_resolution_forwards_exactly_without_loading_weights(self):
        observed = {}

        class FakePipeline:
            blocks = type("PipelineBlocks", (), {"doc": "fixture-doc"})()

            def __call__(self, **kwargs):
                observed.update(kwargs)
                return {"image_latents": "encoded"}

            @staticmethod
            def update_components(**_kwargs):
                return None

        class FakeBlocks:
            component_names = []
            input_names = ["image", "resolution"]

            @staticmethod
            def init_pipeline(*, components_manager):
                return FakePipeline()

        node_config = {
            "params": {"resolution": {"type": "int", "options": [640, 1024]}},
            "model_input_names": [],
            "input_names": ["image", "resolution"],
            "output_names": ["image_latents", "doc"],
        }
        source = Image.new("RGB", (8, 8), "red")
        node = ImageEncode()
        node._pipeline_class = type("OpaqueLayeredPipeline", (), {})

        with patch(
            "modules.ModularDiffusers.latents.require_modiff_node_contract",
            return_value=(FakeBlocks(), node_config),
        ):
            result = node.execute(
                vae={"repo_id": "fixture/layered"},
                image=source,
                resolution="1024",
            )

        self.assertIs(observed["image"], source)
        self.assertEqual(observed["resolution"], 1024)
        self.assertEqual(result["image_latents"], "encoded")
        self.assertEqual(result["doc"], "fixture-doc")

        observed.clear()
        for invalid_resolution, message in (
            ("768", "resolution.*one of"),
            (1024.9, "resolution.*expected int"),
            (float("inf"), "resolution.*expected int"),
            (True, "resolution.*expected int"),
            ("01024", "resolution.*expected int"),
            ("-0", "resolution.*expected int"),
        ):
            with self.subTest(resolution=invalid_resolution), patch(
                "modules.ModularDiffusers.latents.require_modiff_node_contract",
                return_value=(FakeBlocks(), node_config),
            ), self.assertRaisesRegex(ValueError, message):
                node.execute(
                    vae={"repo_id": "fixture/layered"},
                    image=source,
                    resolution=invalid_resolution,
                )
            self.assertEqual(observed, {})

    def test_vae_seed_becomes_an_execution_device_generator_without_forwarding_raw_seed(self):
        observed = []

        class FakePipeline:
            blocks = type("PipelineBlocks", (), {"doc": "fixture-doc"})()
            _execution_device = torch.device("cpu")

            def __call__(self, **kwargs):
                observed.append(kwargs)
                return {"image_latents": "encoded"}

            @staticmethod
            def update_components(**_kwargs):
                return None

        class FakeBlocks:
            component_names = []
            input_names = ["image", "resolution", "generator"]

            @staticmethod
            def init_pipeline(*, components_manager):
                return FakePipeline()

        node_config = {
            "params": {
                "resolution": {"type": "int", "options": [640, 1024]},
                "seed": {"type": "int", "min": 0, "max": 4294967295},
            },
            "model_input_names": [],
            "input_names": ["image", "resolution", "seed"],
            "output_names": ["image_latents", "doc"],
        }
        node = ImageEncode()
        node._pipeline_class = type("OpaqueLayeredPipeline", (), {})

        with patch(
            "modules.ModularDiffusers.latents.require_modiff_node_contract",
            return_value=(FakeBlocks(), node_config),
        ):
            for seed in (0, 4294967295):
                with self.subTest(seed=seed):
                    result = node.execute(
                        vae={"repo_id": "fixture/layered"},
                        image="fixture-image",
                        resolution=640,
                        seed=str(seed),
                    )
                    call = observed[-1]
                    self.assertNotIn("seed", call)
                    self.assertEqual(call["generator"].initial_seed(), seed)
                    self.assertEqual(call["generator"].device, torch.device("cpu"))
                    self.assertEqual(result["image_latents"], "encoded")

        self.assertEqual(len(observed), 2)

    def test_invalid_or_undeclared_vae_generator_state_fails_before_pipeline_or_torch_init(self):
        initialization_count = 0

        class FakeBlocks:
            component_names = []
            input_names = ["image", "generator"]

            @staticmethod
            def init_pipeline(*, components_manager):
                nonlocal initialization_count
                initialization_count += 1
                raise AssertionError("invalid seed state must fail before pipeline initialization")

        node_config = {
            "params": {"seed": {"type": "int", "min": 0, "max": 4294967295}},
            "model_input_names": [],
            "input_names": ["image", "seed"],
            "output_names": ["image_latents"],
        }
        node = ImageEncode()
        node._pipeline_class = type("OpaqueLayeredPipeline", (), {})
        invalid_values = (True, False, 1.5, float("nan"), float("inf"), "01", "-0", -1, 4294967296)

        with patch(
            "modules.ModularDiffusers.latents.require_modiff_node_contract",
            return_value=(FakeBlocks(), node_config),
        ) as contract_resolver, patch(
            "modules.ModularDiffusers.modular_utils.torch.Generator"
        ) as generator_constructor:
            for seed in invalid_values:
                with self.subTest(seed=seed), self.assertRaises(ValueError):
                    node.execute(vae={"repo_id": "fixture/layered"}, image="fixture-image", seed=seed)

            resolver_calls_before_direct_inputs = contract_resolver.call_count
            for direct_kwargs in ({"generator": object()}, {"seed": 7, "generator": object()}):
                with self.subTest(direct_kwargs=direct_kwargs), self.assertRaisesRegex(
                    ValueError,
                    "Direct Modular Diffusers 'generator' values are not accepted",
                ):
                    node.execute(
                        vae={"repo_id": "fixture/layered"},
                        image="fixture-image",
                        **direct_kwargs,
                    )
            self.assertEqual(contract_resolver.call_count, resolver_calls_before_direct_inputs)

            generator_constructor.assert_not_called()

        self.assertEqual(initialization_count, 0)

    def test_vae_seed_contract_mismatch_fails_before_pipeline_initialization(self):
        initialization_count = 0

        class FakeBlocks:
            component_names = []
            input_names = ["image"]

            @staticmethod
            def init_pipeline(*, components_manager):
                nonlocal initialization_count
                initialization_count += 1
                raise AssertionError("a mismatched action contract must fail before initialization")

        node_config = {
            "params": {"seed": {"type": "int", "min": 0, "max": 4294967295}},
            "model_input_names": [],
            "input_names": ["image", "seed"],
            "output_names": ["image_latents"],
        }
        node = ImageEncode()
        node._pipeline_class = type("OpaquePipeline", (), {})

        with patch(
            "modules.ModularDiffusers.latents.require_modiff_node_contract",
            return_value=(FakeBlocks(), node_config),
        ), patch("modules.ModularDiffusers.modular_utils.torch.Generator") as generator_constructor:
            with self.assertRaisesRegex(ValueError, "does not expose the required generator input"):
                node.execute(vae={"repo_id": "fixture/vae"}, image="fixture-image", seed=0)
            generator_constructor.assert_not_called()

        self.assertEqual(initialization_count, 0)

    def test_flattens_nested_layered_diffusers_pil_batches(self):
        first = Image.new("RGB", (8, 8), "red")
        second = Image.new("RGBA", (8, 8), "blue")

        self.assertEqual(flatten_pil_images([[first], (second,)]), [first, second])

    def test_does_not_relabel_non_image_values_as_decoded_images(self):
        self.assertIsNone(flatten_pil_images([[Image.new("RGB", (8, 8))], object()]))

    def test_preview_missing_vae_respects_declared_output_contract(self):
        result = Preview.execute(object(), image=object(), export="", vae=None, device="cpu")

        self.assertEqual(result, {"output": None, "filtered": None})


if __name__ == "__main__":
    unittest.main()
