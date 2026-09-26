"""Exercise the real pinned stage constructor without weights or inference."""
from types import SimpleNamespace
import importlib.util

import pytest
import torch
from diffusers import AnimaModularPipeline

from modules.ModularDiffusers.workflow_blocks import (
    WorkflowImageDenoise,
    WorkflowImageEncode,
    _anima_image_dimensions,
    _official_stage_block,
    _official_stage_component_dependencies,
)

requires_transformers = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="requires the staged optional Transformers runtime",
)


@pytest.mark.parametrize("workflow,stage", [("text2image", "denoise"), ("img2img", "denoise"), ("img2img", "vae_encoder")])
@requires_transformers
def test_anima_stage_retains_cross_stage_geometry_specs(workflow, stage):
    definition = AnimaModularPipeline()
    block = _official_stage_block(definition.blocks.get_workflow(workflow), stage)
    original_names = {spec.name for spec in block.expected_components}
    dependency = "vae" if stage == "denoise" else "transformer"
    wrapped = _official_stage_component_dependencies(
        block, pipeline_class="AnimaModularPipeline", block_path=stage, definition=definition,
    )
    assert wrapped.sub_blocks[stage] is block
    assert {spec.name for spec in wrapped.expected_components} == original_names | {dependency}
    assert wrapped.inputs == block.inputs
    pipeline = wrapped.init_pipeline()
    assert {"vae", "transformer"} <= set(pipeline.pretrained_component_names)
    # Real upstream properties must follow bound component geometry, not copied
    # constants. These metadata-only sentinels deliberately differ from defaults.
    pipeline.vae = SimpleNamespace(temperal_downsample=(True, True))
    pipeline.transformer = SimpleNamespace(config=SimpleNamespace(in_channels=32))
    assert pipeline.vae_scale_factor == 4
    assert pipeline.num_channels_latents == 32
    # Wrapping must not mutate the global official definition or the stage.
    assert {spec.name for spec in block.expected_components} == original_names


@requires_transformers
def test_unrelated_stages_are_not_wrapped_and_missing_specs_fail_closed():
    definition = AnimaModularPipeline()
    block = _official_stage_block(definition.blocks.get_workflow("text2image"), "text_encoder")
    assert _official_stage_component_dependencies(
        block, pipeline_class="AnimaModularPipeline", block_path="text_encoder", definition=definition,
    ) is block
    with pytest.raises(ValueError, match="geometry component"):
        _official_stage_component_dependencies(
            block, pipeline_class="AnimaModularPipeline", block_path="denoise",
            definition=SimpleNamespace(blocks=SimpleNamespace(expected_components=[])),
        )


def test_anima_controls_and_validation_use_the_upstream_patch_grid():
    for cls in (WorkflowImageDenoise, WorkflowImageEncode):
        for name in ("width", "height"):
            assert cls.params[name]["step"] == 16
    for width, height in ((512, 512), (528, 512), (1536, 1024)):
        _anima_image_dimensions(width, height)
    for value in (520, 504, 1544, True, 512.0):
        with pytest.raises(ValueError, match="multiple of 16"):
            _anima_image_dimensions(value, 512)
        with pytest.raises(ValueError, match="multiple of 16"):
            _anima_image_dimensions(512, value)


@requires_transformers
def test_real_anima_latent_preparation_uses_bound_vae_geometry():
    from diffusers.modular_pipelines.anima.before_denoise import AnimaPrepareLatentsStep

    definition = AnimaModularPipeline()
    wrapped = _official_stage_component_dependencies(
        AnimaPrepareLatentsStep(), pipeline_class="AnimaModularPipeline",
        block_path="denoise", definition=definition,
    )
    pipeline = wrapped.init_pipeline()
    pipeline.vae = SimpleNamespace(temperal_downsample=(True, True, True))
    pipeline.transformer = SimpleNamespace(config=SimpleNamespace(in_channels=16))
    state = pipeline(
        width=528, height=512, batch_size=1, num_images_per_prompt=1,
        dtype=torch.float32, generator=torch.Generator(device="cpu").manual_seed(42),
    )
    assert state.get("latents").shape == (1, 16, 1, 64, 66)
    assert state.get("padding_mask").shape == (1, 1, 512, 528)
