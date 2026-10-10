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


@pytest.mark.parametrize("source_size,output_size", [((512, 512), (640, 640)), ((640, 640), (640, 640)), ((512, 512), (768, 640))])
def test_z_image_edit_matches_source_latents_to_explicit_output_without_changing_recipe(source_size, output_size):
    from diffusers.image_processor import VaeImageProcessor

    class Pipeline:
        image_processor = VaeImageProcessor(vae_scale_factor=8)

        def __call__(self, *, image, width, height):
            pass

    pipeline = Pipeline()
    pipeline.image_processor.resize = Mock(wraps=pipeline.image_processor.resize)
    image = Image.new("RGB", source_size, "blue")
    before = image.tobytes()
    values = {"num_inference_steps": 8, "width": output_size[0], "height": output_size[1]}
    target = {"image": image}
    get_image_pipeline_adapter("ZImageImg2ImgPipeline").apply_generation_parameters(pipeline, values, target)
    # The reviewed SDK preprocesses without width/height before separately
    # allocating output-sized noise. Reproduce that exact geometry boundary.
    tensor = pipeline.image_processor.preprocess(target["image"])
    assert tuple(tensor.shape[-2:]) == (target["height"], target["width"])
    assert values == {"num_inference_steps": 8, "width": output_size[0], "height": output_size[1]}
    assert image.size == source_size and image.tobytes() == before
    if source_size == output_size:
        pipeline.image_processor.resize.assert_called_once()  # SDK preprocess only
        assert target["image"] is image
    else:
        assert pipeline.image_processor.resize.call_args_list[0].kwargs == {
            "height": output_size[1], "width": output_size[0],
        }


@pytest.mark.parametrize("name", ["ZImageInpaintPipeline", "Flux2KleinPipeline", "Flux2KleinInpaintPipeline", "QwenImageEditPipeline"])
def test_other_image_adapters_keep_their_native_input_resolution_policy(name):
    class Pipeline:
        image_processor = Mock()

        def __call__(self, *, image, width, height):
            pass

    pipeline = Pipeline()
    image = Image.new("RGB", (512, 512), "blue")
    target = {"image": image}
    get_image_pipeline_adapter(name).apply_generation_parameters(
        pipeline, {"num_inference_steps": 8, "width": 640, "height": 640}, target,
    )
    assert target["image"] is image
    pipeline.image_processor.resize.assert_not_called()


def test_z_image_edit_requires_its_reviewed_resize_boundary_for_different_output_size():
    class Pipeline:
        def __call__(self, *, image, width, height):
            pass

    adapter = get_image_pipeline_adapter("ZImageImg2ImgPipeline")
    with pytest.raises(ValueError, match="exact boolean"):
        replace(adapter, resize_reference_to_output=1)
    with pytest.raises(RuntimeError, match="reviewed image resize boundary"):
        adapter.apply_generation_parameters(
            Pipeline(), {"num_inference_steps": 8, "width": 640, "height": 640},
            {"image": Image.new("RGB", (512, 512), "blue")},
        )
