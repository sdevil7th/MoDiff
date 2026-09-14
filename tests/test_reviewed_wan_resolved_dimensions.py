"""Pinned Wan image/video geometry survives repeated creator area controls."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from PIL import Image

pytest.importorskip("transformers", reason="requires the reviewed optional Transformers runtime")
from diffusers.modular_pipelines import PipelineState
from diffusers.modular_pipelines.wan_animate_2.encoders import (
    WanAnimate2ProcessImagesInputStep, WanAnimate2ProcessVideosInputStep,
)
from diffusers.modular_pipelines.wan_animate_2.before_denoise import WanAnimate2PrepareSegmentsStep
from diffusers.modular_pipelines.wan_animate_2.encoders import WanAnimate2VideoProcessor
from modules.ModularDiffusers import reviewed_blocks

PIPELINES = ("WanAnimate2ModularPipeline", "WanAnimate2DistilledModularPipeline")


def resolved_state():
    processor = WanAnimate2VideoProcessor(vae_scale_factor=8, spatial_patch_size=(2, 2))
    components = SimpleNamespace(image_processor=processor, video_processor=processor,
                                 vae_scale_factor_spatial=8, vae_scale_factor_temporal=4,
                                 _execution_device=torch.device("cpu"), component_names=())
    state = PipelineState()
    for name, value in {"image": Image.new("RGB", (74, 102)), "height": 80, "width": 64}.items():
        state.set(name, value)
    WanAnimate2ProcessImagesInputStep()(components, state)
    return components, state


@pytest.mark.parametrize("family", PIPELINES)
def test_real_video_preprocess_and_segment_geometry_use_resolved_image_frame(family):
    components, state = resolved_state()
    expected = tuple(state.get("image_pixels").shape[-2:])
    assert expected != (80, 64)
    block = WanAnimate2ProcessVideosInputStep()
    components.blocks = SimpleNamespace(sub_blocks={"video": block})
    original = {"height": 80, "width": 64, "driving_video": [Image.new("RGB", (64, 40)) for _ in range(5)],
                "fps": 24, "segment_frame_length": 5, "prev_segment_conditioning_frames": 1}
    with patch.object(reviewed_blocks, "_reviewed_placement", return_value=({}, {"outputs": []})), patch.object(
        reviewed_blocks, "_new_runtime", return_value=(object(), components, state)
    ):
        reviewed_blocks.ReviewedModularWorkflowStep().execute(
            pipeline_class=family, workflow_id="default", execution_scope="unpruned_pipeline",
            placement_path=["video"], block_definition_id="fixture", block_contract_hash="fixture",
            block_class=type(block).__name__, execution_kind="step", **original,
        )
    state.set("reference_image_latents", torch.zeros(20, 1, expected[0] // 8, expected[1] // 8))
    WanAnimate2PrepareSegmentsStep()(components, state)
    assert tuple(state.get("driving_video_pixels").shape[-2:]) == expected
    assert (state.get("height"), state.get("width")) == expected
    assert (original["height"], original["width"]) == (80, 64)


@pytest.mark.parametrize("family", PIPELINES)
@pytest.mark.parametrize("block", ["WanAnimate2ProcessVideosInputStep", "WanAnimate2DenoiseStep", "WanAnimate2DistilledDenoiseStep"])
def test_downstream_geometry_preserves_native_state_without_mutating_controls(family, block):
    _components, state = resolved_state()
    values = {"height": 80, "width": 64, "num_inference_steps": 10}
    result = reviewed_blocks._reviewed_resolved_dimensions(family, block, state, values)
    assert (result["height"], result["width"]) == tuple(state.get("image_pixels").shape[-2:])
    assert result["num_inference_steps"] == 10
    assert values == {"height": 80, "width": 64, "num_inference_steps": 10}


def test_initial_area_controls_other_families_and_unresolved_states_stay_unchanged():
    _components, state = resolved_state()
    values = {"height": 80, "width": 64}
    for family, block, current in [
        (PIPELINES[0], "WanAnimate2ProcessImagesInputStep", state),
        ("WanModularPipeline", "WanAnimate2ProcessVideosInputStep", state),
        (PIPELINES[0], "WanAnimate2ProcessVideosInputStep", PipelineState()),
    ]:
        assert reviewed_blocks._reviewed_resolved_dimensions(family, block, current, values) is values
