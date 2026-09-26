"""No-download checks against the already approved official runtime."""
import importlib.util
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from modules.HuggingFaceTransformers.depth import depth_geometry, predict_depth

pytestmark = pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="Requires reviewed optional Transformers runtime")


@pytest.mark.parametrize("image_size,resolution", [((1024, 1024), 0), ((1328, 1024), 0), ((640, 384), 392)])
def test_real_dpt_preprocessor_matches_preflight_geometry(image_size, resolution):
    from transformers import DPTImageProcessor
    processor = DPTImageProcessor(size={"height": 518, "width": 518}, keep_aspect_ratio=True, ensure_multiple_of=14)
    image = Image.new("RGB", image_size, color=(30, 70, 100))
    size, shape = depth_geometry(processor, image, resolution)
    actual = processor(images=[image], size=size, return_tensors="pt")
    assert set(actual) == {"pixel_values"}
    assert tuple(actual["pixel_values"].shape) == shape


def test_real_native_auto_class_and_postprocessing_contract_without_weights():
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation, DPTImageProcessor
    assert callable(AutoImageProcessor.from_pretrained)
    assert callable(AutoModelForDepthEstimation.from_pretrained)
    processor = DPTImageProcessor(size={"height": 28, "width": 28}, keep_aspect_ratio=True, ensure_multiple_of=14)
    class Model:
        config = SimpleNamespace(model_type="depth_anything", depth_estimation_type="relative")
        def __call__(self, pixel_values):
            h, w = pixel_values.shape[-2:]
            return SimpleNamespace(predicted_depth=torch.arange(h * w, dtype=torch.float32).reshape(1, h, w))
    result = predict_depth(Model(), processor, Image.new("RGB", (84, 56)), receipt={"runtime": {"dtype": "float32", "device": "cpu"}}, resolution=0, match_input=True, convention="model_default")
    assert result["native_depth"].shape == (56, 84)
    assert result["preview_images"][0].size == (84, 56)
    assert result["result"]["nativeDepth"]["resizedToInput"] is True
