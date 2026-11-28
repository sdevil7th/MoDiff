import unittest

import numpy as np
import pytest
from PIL import Image, ImageDraw

from modules.ImageFilters import MODULE_MAP as FILTER_DEFINITIONS, main as filters
from modules.ImageFilters.main import Canny
from utils.torch_utils import CPU_DEVICE


class ImageFilterTests(unittest.TestCase):
    def test_every_published_filter_executes_with_its_creator_defaults(self):
        source = Image.new("RGB", (64, 48), "black")
        ImageDraw.Draw(source).rectangle((16, 12, 47, 35), fill="white")
        for action, definition in FILTER_DEFINITIONS.items():
            with self.subTest(action=action):
                values = {name: field["default"] for name, field in definition["params"].items() if "default" in field}
                values.update(image=source, device=CPU_DEVICE)
                result = getattr(filters, action)("defaults-" + action)(**values)["output"]
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0].size, source.size)

    def test_unsharp_integral_float_and_even_radius_keep_the_integer_recipe(self):
        source = Image.new("RGB", (64, 48), "black")
        ImageDraw.Draw(source).rectangle((16, 12, 47, 35), fill="white")
        expected = filters.UnsharpMask().execute(image=source, radius=5, device=CPU_DEVICE)["output"][0]
        for radius in (5.0, "5", 4.0):
            with self.subTest(radius=radius):
                result = filters.UnsharpMask("edited-unsharp")(
                    image=source, radius=radius, device=CPU_DEVICE
                )["output"][0]
                self.assertEqual(result.tobytes(), expected.tobytes())

    def test_unsharp_rejects_invalid_radius_before_building_a_kernel(self):
        source = Image.new("RGB", (64, 48), "black")
        for radius in (0, -1, 64, 3.5, float("nan"), float("inf"), "bad"):
            with self.subTest(radius=radius), self.assertRaisesRegex(RuntimeError, "radius must be a whole number between 1 and 63"):
                filters.UnsharpMask("invalid-unsharp")(image=source, radius=radius, device=CPU_DEVICE)

    def test_canny_reversed_thresholds_preserve_the_same_edge_map(self):
        pixels = np.random.default_rng(42).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        source = Image.fromarray(pixels)
        expected = Canny().execute(
            image=source, low_threshold=0.1, high_threshold=0.7, device=CPU_DEVICE
        )["output"][0]
        node = Canny("canny-threshold-order")
        inputs = dict(image=source, low_threshold=0.7, high_threshold=0.1, device=CPU_DEVICE)
        reversed_thresholds = node(**inputs)["output"][0]
        self.assertEqual(reversed_thresholds.tobytes(), expected.tobytes())
        self.assertTrue(node._has_changed)
        self.assertEqual(node(**inputs)["output"][0].tobytes(), expected.tobytes())
        self.assertFalse(node._has_changed)

    def test_canny_preserves_source_extent_and_emits_edges(self):
        source = Image.new("RGB", (32, 24), "black")
        ImageDraw.Draw(source).rectangle((8, 6, 23, 17), fill="white")

        output = Canny().execute(
            image=source,
            low_threshold=0.1,
            high_threshold=0.2,
            device="cpu",
        )["output"]

        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].size, source.size)
        self.assertEqual(output[0].mode, "L")
        self.assertGreater(np.asarray(output[0]).max(), 0)


def test_canny_evidence_records_the_normalized_thresholds_and_exact_edge_map(monkeypatch):
    from modiff.builtin_media_evidence import media_identity

    monkeypatch.setenv("MODIFF_CAPTURE_BUILTIN_MEDIA", "1")
    source = Image.new("RGB", (32, 24), "black")
    ImageDraw.Draw(source).rectangle((8, 6, 23, 17), fill="white")
    canny = Canny("captured-canny")
    result = canny(image=source, low_threshold=0.7, high_threshold=0.1, output_mode="RGB", device=CPU_DEVICE)
    evidence = canny._execution_input_record["mediaEvidence"]
    assert evidence["controls"] == {"low_threshold": 0.1, "high_threshold": 0.7,
                                    "output_mode": "RGB", "device": CPU_DEVICE}
    assert evidence["inputs"]["image"] == [media_identity(source)]
    assert evidence["outputs"]["output"] == [media_identity(result["output"][0])]


@pytest.mark.parametrize("output_mode", ["L", "RGB"])
def test_sdxl_adapter_accepts_the_canny_maps_from_saved_and_new_workflows(output_mode):
    from types import SimpleNamespace

    from diffusers.models.adapter import FullAdapterXL
    from diffusers.pipelines.t2i_adapter.pipeline_stable_diffusion_xl_adapter import _preprocess_adapter_image

    from modules.DiffusersImage import main as image

    source = Image.new("RGB", (128, 128), "white")
    ImageDraw.Draw(source).rectangle((24, 24, 103, 103), fill="gray")
    options = {"output_mode": "RGB"} if output_mode == "RGB" else {}
    edges = Canny("canny-adapter")(image=source, device=CPU_DEVICE, **options)["output"]
    assert edges[0].mode == output_mode
    original_bytes = edges[0].tobytes()
    # Use the installed Diffusers preprocessing and the first convolution of
    # its XL adapter. The reduced output width avoids allocating other weights.
    adapter_kernel = FullAdapterXL(in_channels=3, channels=[8], num_res_blocks=1, downscale_factor=16)
    calls = []

    class StableDiffusionXLAdapterPipeline:
        _execution_device = "cpu"

        def __call__(self, **kwargs):
            calls.append(kwargs["image"])
            pixels = _preprocess_adapter_image(kwargs["image"], 128, 128)
            assert tuple(pixels.shape) == (1, 3, 128, 128)
            features = adapter_kernel.conv_in(adapter_kernel.unshuffle(pixels))
            assert tuple(features.shape) == (1, 8, 8, 8)
            return SimpleNamespace(images=[source])

    result = image.ControlGenerate("canny-adapter-image")(
        pipeline=StableDiffusionXLAdapterPipeline(), control_image=edges, width=128, height=128,
        prompt="A glass cube", num_inference_steps=1,
    )
    assert result["images"] == [source]
    assert calls[0][0].tobytes() == edges[0].convert("RGB").tobytes()
    assert edges[0].mode == output_mode and edges[0].tobytes() == original_bytes
    if output_mode == "RGB":
        assert calls[0][0] is edges[0]


def test_hunyuan_controlnet_accepts_a_grayscale_guide():
    from types import SimpleNamespace

    import torch
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.pipelines.controlnet_hunyuandit.pipeline_hunyuandit_controlnet import (
        HunyuanDiTControlNetPipeline as UpstreamPipeline,
    )

    from modules.DiffusersImage.main import ControlGenerate

    guide = Image.new("L", (512, 512))
    ImageDraw.Draw(guide).rectangle((100, 100, 300, 400), outline=255, width=4)
    pixels_before = guide.tobytes()
    projection = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3)

    class HunyuanDiTControlNetPipeline:
        _execution_device = "cpu"
        image_processor = VaeImageProcessor(vae_scale_factor=8)

        def __call__(self, *, control_image, use_resolution_binning=True, **kwargs):
            assert not use_resolution_binning
            pixels = UpstreamPipeline.prepare_image(
                self, control_image, 512, 512, 1, 1, "cpu", torch.float32,
            )
            features = projection(pixels)
            assert tuple(features.shape) == (1, 4, 510, 510)
            assert torch.equal(pixels[:, 0], pixels[:, 1]) and torch.equal(pixels[:, 0], pixels[:, 2])
            return SimpleNamespace(images=[guide.convert("RGB")])

    result = ControlGenerate("hunyuan-canny")(
        pipeline=HunyuanDiTControlNetPipeline(), control_image=guide, width=512, height=512,
        prompt="A glass cube", num_inference_steps=1,
    )
    assert result["images"][0].size == (512, 512)
    assert guide.mode == "L" and guide.tobytes() == pixels_before


if __name__ == "__main__":
    unittest.main()
