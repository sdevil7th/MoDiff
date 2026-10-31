"""Native Flux preserves the whole-pipeline timestep rounding contract."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from diffusers import (
    ComponentsManager,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinModularPipeline,
    FluxModularPipeline,
    FluxTransformer2DModel,
    ZImageModularPipeline,
)
from diffusers.modular_pipelines import BlockState
from diffusers.modular_pipelines.flux.before_denoise import calculate_shift
from diffusers.modular_pipelines.flux.denoise import FluxLoopDenoiser

from modules.ModularDiffusers.modular_utils import require_modiff_node_contract
from modules.ModularDiffusers.native_blocks import prepare_native_pipeline_blocks


def schedule(steps, dynamic, sequence_length):
    # Reviewed cached configs: Schnell741f7c3 uses shift1/static; Dev3de623
    # and Krea8162a9c use shift3/dynamic, with the same published shift bounds.
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3 if dynamic else 1, use_dynamic_shifting=dynamic)
    scheduler.set_timesteps(sigmas=np.linspace(1, 1 / steps, steps), mu=calculate_shift(sequence_length))
    return scheduler.timesteps


@pytest.fixture(scope="module")
def tiny_model_inputs():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        model = FluxTransformer2DModel(
            in_channels=4, out_channels=4, num_layers=1, num_single_layers=1,
            attention_head_dim=16, num_attention_heads=2, joint_attention_dim=8,
            pooled_projection_dim=8, guidance_embeds=True, axes_dims_rope=(4, 6, 6),
        ).to(dtype=torch.bfloat16).eval()
        latents = torch.randn((1, 4, 4), dtype=torch.bfloat16)
        prompt = torch.randn((1, 3, 8), dtype=torch.bfloat16)
        pooled = torch.randn((1, 8), dtype=torch.bfloat16)
    return model, dict(
        latents=latents, prompt_embeds=prompt, pooled_prompt_embeds=pooled,
        guidance=torch.tensor([3.5]), txt_ids=torch.zeros((3, 3), dtype=torch.bfloat16),
        img_ids=torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]], dtype=torch.bfloat16),
        joint_attention_kwargs=None,
    )


@pytest.mark.parametrize("steps,dynamic,sequence_length,changed", [
    (4, False, 4096, 0), (24, True, 3072, 6), (28, True, 4096, 7), (30, True, 4096, 8),
])
def test_real_bfloat16_flux_matches_whole_timestep_math_and_retains_schnell_collapse(
    tiny_model_inputs, steps, dynamic, sequence_length, changed,
):
    model, values = tiny_model_inputs
    times = schedule(steps, dynamic, sequence_length)
    whole = times.to(torch.bfloat16) / 1000
    native = times / 1000
    old_effective, native_effective = whole * 1000, native.to(torch.bfloat16) * 1000
    indices = (old_effective != native_effective).nonzero().flatten().tolist()
    assert len(indices) == changed
    index = indices[0] if indices else 1
    blocks, _ = require_modiff_node_contract(FluxModularPipeline, "denoise")
    predictor = blocks.sub_blocks["denoise"].sub_blocks["denoiser"]
    components = SimpleNamespace(transformer=model)
    upstream_state, actual_state = BlockState(**values), BlockState(**values)
    FluxLoopDenoiser()(components, upstream_state, i=index, t=times[index])
    with torch.no_grad():
        expected = model(
            hidden_states=values["latents"], timestep=whole[index].reshape(1), guidance=values["guidance"],
            encoder_hidden_states=values["prompt_embeds"], pooled_projections=values["pooled_prompt_embeds"],
            txt_ids=values["txt_ids"], img_ids=values["img_ids"], return_dict=False,
        )[0]
    captured = []
    def observe_time(_module, args):
        captured.append(args[0].detach().clone())
    handle = model.time_text_embed.register_forward_pre_hook(observe_time)
    try:
        with patch.object(model, "forward", wraps=model.forward) as forward:
            predictor(components, actual_state, i=index, t=times[index])
        forward.assert_called_once()
    finally:
        handle.remove()
    torch.testing.assert_close(captured[0], old_effective[index].reshape(1), rtol=0, atol=0)
    assert torch.isfinite(actual_state.noise_pred).all() and torch.count_nonzero(actual_state.noise_pred)
    torch.testing.assert_close(actual_state.noise_pred, expected, rtol=0, atol=0)
    assert torch.equal(upstream_state.noise_pred, expected) is (not changed)
    assert actual_state.guidance is values["guidance"]


@pytest.mark.parametrize("branch", ["text2image", "img2img"])
def test_shared_flux_predictor_cast_preserves_original_scheduler_time_for_both_branches(branch):
    blocks, _ = require_modiff_node_contract(FluxModularPipeline, "denoise")
    assert branch in blocks.sub_blocks["input"].sub_blocks
    assert branch in blocks.sub_blocks["before_denoise"].sub_blocks
    loop = blocks.sub_blocks["denoise"]
    latents = torch.ones((2, 4, 4), dtype=torch.bfloat16)
    state = BlockState(latents=latents, guidance=None, prompt_embeds=None, pooled_prompt_embeds=None,
                       joint_attention_kwargs=None, txt_ids=None, img_ids=None)
    original = torch.tensor(892.8571166992188)
    model = Mock(return_value=(torch.ones_like(latents),))
    scheduler = SimpleNamespace(step=Mock(return_value=(latents,)))
    components = SimpleNamespace(transformer=model, scheduler=scheduler)
    loop.sub_blocks["denoiser"](components, state, i=0, t=original)
    actual = model.call_args.kwargs["timestep"]
    torch.testing.assert_close(actual, original.expand(2).to(torch.bfloat16) / 1000, rtol=0, atol=0)
    loop.sub_blocks["after_denoiser"](components, state, i=0, t=original)
    assert scheduler.step.call_args.args[1] is original
    assert original.dtype == torch.float32
    assert state.latents is latents


@pytest.mark.parametrize("changed", ["blueprint", "core", "loop", "predictor", "image_input", "image_prepare"])
def test_unknown_flux_blueprint_is_rejected_before_mutating_the_shared_predictor(changed):
    blocks = FluxModularPipeline().blocks
    core = blocks.sub_blocks["denoise"]
    loop = core.sub_blocks["denoise"]
    original = loop.sub_blocks["denoiser"]
    if changed == "blueprint":
        blocks = SimpleNamespace(sub_blocks=blocks.sub_blocks)
    elif changed == "core":
        blocks.sub_blocks["denoise"] = SimpleNamespace()
    elif changed == "loop":
        core.sub_blocks["denoise"] = SimpleNamespace()
    elif changed == "predictor":
        loop.sub_blocks["denoiser"] = SimpleNamespace()
    elif changed == "image_input":
        core.sub_blocks["input"].sub_blocks["img2img"] = SimpleNamespace()
    else:
        core.sub_blocks["before_denoise"].sub_blocks["img2img"] = SimpleNamespace()
    with pytest.raises(ValueError, match="reviewed"):
        prepare_native_pipeline_blocks(FluxModularPipeline, blocks)
    if changed != "predictor":
        assert loop.sub_blocks["denoiser"] is original


def test_flux_adapter_preserves_upstream_classes_other_families_and_deepcopy_contracts():
    from modules.ModularDiffusers.native_blocks import FluxWholePipelineTimeDenoiser

    upstream = FluxModularPipeline().blocks
    adapted = prepare_native_pipeline_blocks(FluxModularPipeline, FluxModularPipeline().blocks)
    old_loop = upstream.sub_blocks["denoise"].sub_blocks["denoise"]
    new_loop = adapted.sub_blocks["denoise"].sub_blocks["denoise"]
    assert type(old_loop.sub_blocks["denoiser"]) is FluxLoopDenoiser
    assert type(new_loop.sub_blocks["denoiser"]) is FluxWholePipelineTimeDenoiser
    assert type(deepcopy(new_loop).sub_blocks["denoiser"]) is FluxWholePipelineTimeDenoiser
    assert set(upstream.input_names) <= set(adapted.input_names)
    assert set(adapted.input_names) - set(upstream.input_names) == {"negative_prompt", "negative_prompt_2"}
    assert set(adapted.component_names) - set(upstream.component_names) == {"guider"}
    for pipeline_class in (Flux2KleinModularPipeline,):
        blocks = pipeline_class().blocks
        assert prepare_native_pipeline_blocks(pipeline_class, blocks) is blocks
    blocks = ZImageModularPipeline().blocks
    with pytest.raises(ValueError, match="reviewed"):
        prepare_native_pipeline_blocks(FluxModularPipeline, blocks)


@pytest.mark.parametrize("workflow", ["text2image", "image2image"])
def test_real_reviewed_loader_workflows_preserve_the_adapted_flux_loop_without_weights(workflow):
    from modules.ModularDiffusers.loaders import _instantiate_reviewed_builtin_pipeline
    from modules.ModularDiffusers.native_blocks import FluxWholePipelineTimeDenoiser

    document = {
        "_class_name": "FluxPipeline", "transformer": ["diffusers", "FluxTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"], "text_encoder": ["transformers", "CLIPTextModel"],
        "text_encoder_2": ["transformers", "T5EncoderModel"],
        "tokenizer": ["transformers", "CLIPTokenizer"], "tokenizer_2": ["transformers", "T5TokenizerFast"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    }
    with patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("no download allowed")):
        pipeline = _instantiate_reviewed_builtin_pipeline(
            "FluxModularPipeline", "black-forest-labs/FLUX.1-schnell",
            index_filename="model_index.json", index_document=document,
            components_manager=ComponentsManager(), collection="flux-time-regression", workflow_id=workflow,
        )
    assert all(getattr(pipeline, name) is None for name in ("transformer", "vae", "text_encoder", "text_encoder_2"))
    def predictors(block):
        return ([block] if isinstance(block, FluxLoopDenoiser) else []) + [
            value for child in block.sub_blocks.values() for value in predictors(child)
        ]
    actual = predictors(pipeline.blocks)
    assert len(actual) == 1 and type(actual[0]) is FluxWholePipelineTimeDenoiser
