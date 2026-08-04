import unittest
from unittest.mock import patch

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from modules.ModularDiffusers.adapters import Lora
from modules.ModularDiffusers.loaders import apply_lora_scheduler_override
from modules.Spandrel import MODULE_MAP as SPANDREL_MODULE_MAP
from modules.Spandrel.main import Upscaler
from utils.huggingface import local_files_only


class AppManagedAuxiliaryModelTests(unittest.TestCase):
    def test_upscaler_socket_contract_accepts_stills_and_video_frame_batches(self):
        params = SPANDREL_MODULE_MAP["Upscaler"]["params"]
        self.assertEqual(params["image"]["type"], ["image", "video"])
        self.assertTrue(params["image"]["required"])
        self.assertEqual(params["output"]["type"], ["image", "video"])

    def test_execution_loaders_are_always_cache_only(self):
        self.assertTrue(local_files_only("example/model"))

    def test_modular_lora_rejects_an_empty_selection(self):
        with self.assertRaisesRegex(ValueError, "LoRA model is required"):
            Lora("empty-lora").execute({"source": "hub", "value": ""}, 1.0)

    def test_modular_lora_resolves_a_hub_weight_only_from_app_cache(self):
        with patch(
            "utils.huggingface.cached_file_path",
            return_value="/cache/revision/style.safetensors",
        ):
            result = Lora("cached-lora").execute(
                {"source": "hub", "value": "example/style"},
                0.75,
                weight_name="style.safetensors",
            )["lora"]

        self.assertEqual(result["lora_path"], "/cache/revision")
        self.assertEqual(result["weight_name"], "style.safetensors")

    def test_modular_lora_carries_a_generic_scheduler_contract(self):
        with patch("utils.huggingface.cached_file_path", return_value="/cache/revision/lightning.safetensors"):
            result = Lora("lightning-lora").execute(
                {"source": "hub", "value": "example/lightning"},
                1.0,
                weight_name="lightning.safetensors",
                scheduler_class="FlowMatchEulerDiscreteScheduler",
                scheduler_config='{"base_shift": 1.0986122886681098, "shift_terminal": null}',
            )["lora"]

        self.assertEqual(result["scheduler_class"], "FlowMatchEulerDiscreteScheduler")
        self.assertIsNone(result["scheduler_config"]["shift_terminal"])

    def test_loader_applies_explicit_lora_scheduler_contract(self):
        class FakePipeline:
            def __init__(self):
                self.scheduler = FlowMatchEulerDiscreteScheduler()

            def update_components(self, **components):
                for name, component in components.items():
                    setattr(self, name, component)

        pipeline = FakePipeline()
        scheduler = apply_lora_scheduler_override(
            pipeline,
            {
                "scheduler_class": "FlowMatchEulerDiscreteScheduler",
                "scheduler_config": {"base_shift": 1.0986122886681098, "shift_terminal": None},
            },
        )

        self.assertIs(pipeline.scheduler, scheduler)
        self.assertAlmostEqual(scheduler.config.base_shift, 1.0986122886681098)
        self.assertIsNone(scheduler.config.shift_terminal)

    def test_modular_lora_fails_if_model_manager_has_not_installed_weight(self):
        with patch("utils.huggingface.cached_file_path", return_value=False):
            with self.assertRaisesRegex(FileNotFoundError, "Model Manager"):
                Lora("missing-lora").execute(
                    {"source": "hub", "value": "example/style"},
                    1.0,
                    weight_name="style.safetensors",
                )

    def test_hub_upscaler_requires_a_pinned_filename(self):
        with self.assertRaisesRegex(ValueError, "pinned filename"):
            Upscaler("unpinned-upscaler").execute(
                image=object(),
                model_id={"source": "hub", "value": "example/upscaler"},
                device="cpu",
            )

    def test_hub_upscaler_missing_from_app_cache_fails_before_model_load(self):
        with patch("utils.huggingface.cached_file_path", return_value=False):
            with patch("modules.Spandrel.main.ModelLoader") as loader:
                with self.assertRaisesRegex(FileNotFoundError, "Model Manager"):
                    Upscaler("missing-upscaler").execute(
                        image=object(),
                        model_id={"source": "hub", "value": "example/upscaler/model.pth"},
                        device="cpu",
                    )
                loader.assert_not_called()

    def test_upscaler_tiles_and_stitches_model_agnostic_integer_scale(self):
        class FakeUpscaler:
            device = "cpu"

            def __call__(self, image):
                return torch.nn.functional.interpolate(image, scale_factor=2, mode="nearest")

        source = torch.arange(3 * 11 * 13, dtype=torch.float32).reshape(1, 3, 11, 13)
        expected = FakeUpscaler()(source)

        actual = Upscaler._upscale_tensor_tiled(source, FakeUpscaler(), tile_size=5, tile_overlap=2)

        self.assertEqual(tuple(actual.shape), (1, 3, 22, 26))
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
