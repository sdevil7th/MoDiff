"""Real tensor regression for native Z-Image's whole-pipeline time convention."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from diffusers import ComponentsManager, FlowMatchEulerDiscreteScheduler, QwenImageModularPipeline, ZImageModularPipeline
from diffusers.modular_pipelines import BlockState
from diffusers.modular_pipelines.z_image.denoise import ZImageLoopBeforeDenoiser, ZImageLoopDenoiser

from modules.ModularDiffusers.modular_utils import require_modiff_node_contract
from modules.ModularDiffusers.native_blocks import (
    ZImageFloat32CFGDenoiser,
    ZImageFloat32TimeInput,
    prepare_native_pipeline_blocks,
)


def timesteps():
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3)
    scheduler.set_timesteps(sigmas=[1 - index / 8 for index in range(8)])
    return scheduler.timesteps


def run_inputs(block, steps, dtype):
    latents = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4)
    result = []
    components = SimpleNamespace()
    for index, timestep in enumerate(steps):
        state = BlockState(latents=latents, dtype=dtype)
        returned, state = block(components, state, i=index, t=timestep)
        assert returned is components
        assert len(state.latent_model_input) == 2
        assert all(value.dtype == dtype for value in state.latent_model_input)
        torch.testing.assert_close(
            torch.stack(state.latent_model_input), latents.unsqueeze(2).to(dtype), rtol=0, atol=0,
        )
        result.append(state.timestep)
    return torch.stack(result)


@pytest.mark.parametrize("branch", ["text2image", "image2image"])
def test_native_z_loop_preserves_float32_time_and_bfloat16_latent_inputs(branch):
    steps = timesteps()
    expected = ((1000 - steps) / 1000).unsqueeze(1).expand(-1, 2)
    upstream = run_inputs(ZImageLoopBeforeDenoiser(), steps, torch.bfloat16)
    assert (upstream[:, 0].float() != expected[:, 0]).sum().item() == 6
    blocks, _ = require_modiff_node_contract(ZImageModularPipeline, "denoise")
    before = blocks.sub_blocks[branch].sub_blocks["denoise"].sub_blocks["before_denoiser"]
    actual = run_inputs(before, steps, torch.bfloat16)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("time_dtype", [torch.float32, torch.float64])
def test_time_arithmetic_is_float32_without_changing_model_input_dtype(dtype, time_dtype):
    steps = timesteps().to(time_dtype)
    actual = run_inputs(ZImageFloat32TimeInput(), steps, dtype)
    expected = ((1000 - steps.float()) / 1000).unsqueeze(1).expand(-1, 2)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_adapter_is_scoped_to_fresh_exact_z_blueprints_and_keeps_upstream_unchanged():
    first = ZImageModularPipeline().blocks
    upstream = ZImageModularPipeline().blocks
    other = QwenImageModularPipeline().blocks
    assert prepare_native_pipeline_blocks(QwenImageModularPipeline, other) is other
    assert prepare_native_pipeline_blocks(type("ZImageModularPipeline", (), {}), other) is other
    assert prepare_native_pipeline_blocks(ZImageModularPipeline, first) is first
    for branch in ("text2image", "image2image"):
        original_loop = upstream.sub_blocks["denoise"].sub_blocks[branch].sub_blocks["denoise"]
        actual_loop = first.sub_blocks["denoise"].sub_blocks[branch].sub_blocks["denoise"]
        assert type(original_loop.sub_blocks["before_denoiser"]) is ZImageLoopBeforeDenoiser
        assert type(actual_loop.sub_blocks["before_denoiser"]) is ZImageFloat32TimeInput
        assert list(actual_loop.sub_blocks) == list(original_loop.sub_blocks)
        assert type(original_loop.sub_blocks["denoiser"]) is ZImageLoopDenoiser
        assert type(actual_loop.sub_blocks["denoiser"]) is ZImageFloat32CFGDenoiser
        assert type(actual_loop.sub_blocks["after_denoiser"]) is type(original_loop.sub_blocks["after_denoiser"])
        assert type(deepcopy(actual_loop).sub_blocks["before_denoiser"]) is ZImageFloat32TimeInput
    assert upstream.input_names == first.input_names
    assert upstream.component_names == first.component_names


@pytest.mark.parametrize("changed", ["blueprint", "branch", "loop", "before"])
def test_changed_z_blueprint_is_rejected_before_either_branch_is_mutated(changed):
    blocks = ZImageModularPipeline().blocks
    denoise = blocks.sub_blocks["denoise"]
    original_text = denoise.sub_blocks["text2image"].sub_blocks["denoise"].sub_blocks["before_denoiser"]
    if changed == "blueprint":
        blocks = SimpleNamespace(sub_blocks=blocks.sub_blocks)
    elif changed == "branch":
        denoise.sub_blocks["image2image"] = SimpleNamespace()
    elif changed == "loop":
        denoise.sub_blocks["image2image"].sub_blocks["denoise"] = SimpleNamespace()
    else:
        denoise.sub_blocks["image2image"].sub_blocks["denoise"].sub_blocks["before_denoiser"] = SimpleNamespace()
    with pytest.raises(ValueError, match="reviewed"):
        prepare_native_pipeline_blocks(ZImageModularPipeline, blocks)
    assert denoise.sub_blocks["text2image"].sub_blocks["denoise"].sub_blocks["before_denoiser"] is original_text


@pytest.mark.parametrize("workflow", ["text2image", "image2image"])
def test_reviewed_loader_constructs_the_same_adapted_native_tree_without_weights(workflow):
    from modules.ModularDiffusers.loaders import _instantiate_reviewed_builtin_pipeline

    document = {
        "_class_name": "ZImagePipeline", "transformer": ["diffusers", "ZImageTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"], "text_encoder": ["transformers", "Qwen3Model"],
        "tokenizer": ["transformers", "Qwen2Tokenizer"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    }
    with patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("no download allowed")):
        pipeline = _instantiate_reviewed_builtin_pipeline(
            "ZImageModularPipeline", "Tongyi-MAI/Z-Image-Turbo",
            index_filename="model_index.json", index_document=document,
            components_manager=ComponentsManager(), collection="native-z-time-regression", workflow_id=workflow,
        )
    assert all(getattr(pipeline, name) is None for name in ("transformer", "vae", "text_encoder", "tokenizer"))
    # The public workflow selection prunes the Auto branch into the ordinary
    # native denoise sequence while preserving our exact loop input block.
    def time_inputs(block):
        return ([block] if isinstance(block, ZImageLoopBeforeDenoiser) else []) + [
            value for child in block.sub_blocks.values() for value in time_inputs(child)
        ]

    before = time_inputs(pipeline.blocks)
    assert len(before) == 1
    assert type(before[0]) is ZImageFloat32TimeInput
