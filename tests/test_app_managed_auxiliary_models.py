import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.numpy import save_file

from modules.ModularDiffusers.adapters import Lora
from modules.ModularDiffusers.loaders import apply_lora_scheduler_override
from modules.Spandrel import MODULE_MAP as SPANDREL_MODULE_MAP
from modules.Spandrel.main import Upscaler
from utils.huggingface import CONFIG, local_files_only


LORA_REVISION = "a" * 40


def _write_tiny_safetensors(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"lora.weight": np.asarray([1.0], dtype=np.float32)}, str(path))


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
        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory)
            cached_weight = (
                cache_root
                / "models--example--style"
                / "snapshots"
                / LORA_REVISION
                / "style.safetensors"
            )
            _write_tiny_safetensors(cached_weight)
            digest = hashlib.sha256(cached_weight.read_bytes()).hexdigest()
            with patch.dict(CONFIG.hf, {"cache_dir": str(cache_root)}):
                with patch("utils.huggingface.cached_file_path", return_value=str(cached_weight)) as cached:
                    result = Lora("cached-lora").execute(
                        {"source": "hub", "value": "example/style"},
                        0.75,
                        weight_name="style.safetensors",
                        revision=LORA_REVISION,
                        expected_sha256=digest,
                    )["lora"]

        self.assertEqual(result["artifact"]["repository"], "example/style")
        self.assertEqual(result["artifact"]["revision"], LORA_REVISION)
        self.assertEqual(result["artifact"]["weight_name"], "style.safetensors")
        self.assertEqual(cached.call_args.kwargs["revision"], LORA_REVISION)

    def test_modular_lora_carries_a_generic_scheduler_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / "lightning.safetensors"
            _write_tiny_safetensors(weight)
            result = Lora("lightning-lora").execute(
                {"source": "local", "value": str(weight)},
                1.0,
                weight_name="lightning.safetensors",
                scheduler_class="FlowMatchEulerDiscreteScheduler",
                scheduler_config='{"base_shift": 1.0986122886681098, "shift_terminal": null}',
            )["lora"]

        self.assertEqual(result["scheduler"]["class_name"], "FlowMatchEulerDiscreteScheduler")
        self.assertIsNone(result["scheduler"]["config"]["shift_terminal"])

    def test_loader_applies_explicit_lora_scheduler_contract(self):
        class FakePipeline:
            def __init__(self):
                self.scheduler = FlowMatchEulerDiscreteScheduler()

            def update_components(self, **components):
                for name, component in components.items():
                    setattr(self, name, component)

        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / "lightning.safetensors"
            _write_tiny_safetensors(weight)
            descriptor = Lora("lightning-lora").execute(
                {"source": "local", "value": str(weight)},
                1.0,
                weight_name=weight.name,
                scheduler_class="FlowMatchEulerDiscreteScheduler",
                scheduler_config={"base_shift": 1.0986122886681098, "shift_terminal": None},
            )["lora"]
            pipeline = FakePipeline()
            scheduler = apply_lora_scheduler_override(pipeline, descriptor)

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
                    revision=LORA_REVISION,
                    expected_sha256="b" * 64,
                )

    def test_modular_lora_model_selection_does_not_publish_a_blank_class_filter(self):
        options = Lora.params["model"]["fieldOptions"]
        self.assertEqual(options["sources"], ["hub", "local"])
        self.assertNotIn("filter", options)

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

    def test_exact_upscaler_selection_is_revalidated_before_model_load(self):
        managed_path = Path("C:/managed/exact.pth")
        selection = {
            "source": "hub",
            "value": "nateraw/real-esrgan/RealESRGAN_x2plus.pth",
            "revision": "42efb9c3eeed1f5c0c8a626cf5f7f4481dfbb094",
            "sha256": "4" * 64,
            "byteSize": 123,
        }
        node = Upscaler("exact-upscaler")
        model = MagicMock()
        model.eval.return_value = model
        node.mm_add = MagicMock()
        node.mm_exec = MagicMock(return_value=[])
        with (
            patch(
                "modiff.controlled_artifacts.resolve_upscaler_artifact",
                return_value=SimpleNamespace(
                    path=managed_path,
                    receipt={"artifact": {"weightName": "exact.pth"}},
                ),
            ) as resolve,
            patch("modules.Spandrel.main.ModelLoader") as loader,
        ):
            loader.return_value.load_from_file.return_value = model
            self.assertEqual(node.execute(image=object(), model_id=selection, device="cpu"), {"output": []})

        resolve.assert_called_once_with(selection)
        loader.return_value.load_from_file.assert_called_once_with(str(managed_path))

    def test_exact_upscaler_preserves_the_reviewed_suffix_for_an_extensionless_cache_blob(self):
        selection = {
            "source": "hub",
            "value": "nateraw/real-esrgan/RealESRGAN_x2plus.pth",
            "revision": "42efb9c3eeed1f5c0c8a626cf5f7f4481dfbb094",
            "sha256": "4" * 64,
            "byteSize": 123,
        }
        with tempfile.TemporaryDirectory() as directory:
            blob = Path(directory) / "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"
            blob.write_bytes(b"reviewed-upscaler")
            receipt = {"artifact": {"weightName": "RealESRGAN_x2plus.pth"}}
            node = Upscaler("extensionless-exact-upscaler")
            model = MagicMock()
            model.eval.return_value = model
            node.mm_add = MagicMock()
            node.mm_exec = MagicMock(return_value=[])
            with (
                patch.dict("modules.Spandrel.main.CONFIG.paths", {"temp": directory}),
                patch(
                    "modiff.controlled_artifacts.resolve_upscaler_artifact",
                    return_value=SimpleNamespace(path=blob, receipt=receipt),
                ),
                patch("modules.Spandrel.main.ModelLoader") as loader,
            ):
                loader.return_value.load_from_file.return_value = model
                self.assertEqual(node.execute(image=object(), model_id=selection, device="cpu"), {"output": []})
                loaded_path = Path(loader.return_value.load_from_file.call_args.args[0])
                self.assertEqual(loaded_path.suffix, ".pth")
                self.assertFalse(loaded_path.exists())

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
