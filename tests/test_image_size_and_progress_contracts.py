"""Native-run regressions exercised without constructing models or a graph."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
from PIL import Image

from modules.DiffusersImage.main import (
    _tag_image_pipeline,
    add_progress_callback,
    catalog_revision,
    get_image_pipeline_adapter,
    image_pipeline_contract,
    preflight_image_action,
)


@pytest.mark.parametrize("name", ["HunyuanDiTPipeline", "HunyuanDiTPAGPipeline", "HunyuanDiTControlNetPipeline"])
def test_explicit_size_is_not_fixed_or_silently_binned(name):
    adapter = get_image_pipeline_adapter(name)
    assert adapter.min_output_side <= 896 < 1152 <= adapter.max_output_side
    assert adapter.max_output_pixels == 1024 * 1024
    contract = image_pipeline_contract(adapter, adapter.mode_options[0])
    for field in ("width", "height"):
        assert {key: contract["fieldParams"][field][key] for key in ("min", "max", "step")} == {
            "min": 512,
            "max": 2048,
            "step": 32,
        }
    assert contract["maxOutputPixels"] == 1024 * 1024

    class Pipeline:
        def __call__(self, *, width, height, use_resolution_binning=True):
            pass

    values = {"width": 1152, "height": 896, "num_inference_steps": 25}
    target = {}
    adapter.apply_generation_parameters(Pipeline(), values, target)
    assert target == {"width": 1152, "height": 896, "use_resolution_binning": False}


def test_pipeline_without_step_callback_has_indeterminate_progress():
    class Pipeline:
        def __call__(self, *, prompt):
            pass

    node = Mock()
    pipeline = Pipeline()
    call_kwargs = {"prompt": "still working"}
    add_progress_callback(node, pipeline, call_kwargs, 50)
    node.progress.assert_called_once_with(-1, phase="denoising", message="Generating image")
    assert call_kwargs == {"prompt": "still working"}
    assert not hasattr(pipeline, "_num_timesteps")


@pytest.mark.parametrize("name", ["HunyuanDiTPipeline", "HunyuanDiTPAGPipeline", "HunyuanDiTControlNetPipeline"])
def test_explicit_size_keeps_backend_pixel_ceiling_and_alignment(name):
    adapter = get_image_pipeline_adapter(name)
    pipeline = type(name, (), {})()
    mode = adapter.mode_options[0]
    action = "ControlGenerate" if mode == "control_image" else "Generate"
    _tag_image_pipeline(
        pipeline,
        adapter,
        mode,
        adapter.default_repo,
        "hub",
        catalog_revision(adapter.default_repo),
        conditioning_repo=adapter.default_conditioning_repo,
        conditioning_revision=catalog_revision(adapter.default_conditioning_repo)
        if adapter.default_conditioning_repo
        else None,
    )
    values = {"width": 1152, "height": 896, "num_inference_steps": 25}
    if mode == "control_image":
        values["control_image"] = Image.new("RGB", (32, 32), "black")
    _, accepted = preflight_image_action(pipeline, action, values)
    assert (accepted["width"], accepted["height"]) == (1152, 896)
    for size, message in [((1152, 1024), "cannot exceed"), ((1153, 896), "increments"), ((480, 896), "between")]:
        with pytest.raises(ValueError, match=message):
            preflight_image_action(pipeline, action, {**values, "width": size[0], "height": size[1]})


def test_resolution_binning_policy_rejects_non_boolean_and_missing_upstream_contract():
    adapter = get_image_pipeline_adapter("HunyuanDiTPipeline")
    with pytest.raises(ValueError, match="exact boolean"):
        replace(adapter, use_resolution_binning=0)
    with pytest.raises(ValueError, match="resolution-binning control"):
        adapter.apply_generation_parameters(lambda: None, {"num_inference_steps": 25}, {})
