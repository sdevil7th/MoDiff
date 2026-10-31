"""Pinned native Z guidance preserves whole-pipeline batch and FP32 math."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from diffusers import ClassifierFreeGuidance, ZImageModularPipeline, ZImageTransformer2DModel
from diffusers.modular_pipelines import BlockState
from diffusers.modular_pipelines.z_image.denoise import ZImageLoopDenoiser

from modules.ModularDiffusers.modular_utils import require_modiff_node_contract
from modules.ModularDiffusers.native_blocks import prepare_native_pipeline_blocks


class PredictionTransformer(torch.nn.Module):
    """Bounded BF16 predictions reveal actual guider math and batch wiring."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, t, cap_feats, return_dict=False):
        self.calls.append({"x": [value.clone() for value in x], "t": t.clone(),
                           "cap_feats": [value.clone() for value in cap_feats]})
        return ([torch.full_like(value, float(cap[0, 0])) for value, cap in zip(x, cap_feats)],)


def run_block(
    block, *, enabled=True, original=True, scale=1.0, transformer=None, step=0, start=0, stop=1, rescale=0,
):
    guide = ClassifierFreeGuidance(
        guidance_scale=scale, enabled=enabled, use_original_formulation=original, start=start, stop=stop,
        guidance_rescale=rescale,
    )
    if transformer is None:
        transformer = PredictionTransformer()
    components = SimpleNamespace(guider=guide, transformer=transformer)
    state = BlockState(
        latent_model_input=[torch.full((1, 1, 2, 2), value, dtype=torch.bfloat16) for value in (3, 4)],
        timestep=torch.tensor([0.125, 0.125], dtype=torch.float32), dtype=torch.bfloat16,
        num_inference_steps=8,
        prompt_embeds=[torch.full((length, 1), value, dtype=torch.bfloat16)
                       for length, value in ((3, 1.0), (2, 2.0))],
        negative_prompt_embeds=[torch.full((1, 1), value, dtype=torch.bfloat16) for value in (0.1, 0.2)],
    )
    returned, result = block(components, state, i=step, t=torch.tensor(875.0))
    assert returned is components and result is state
    return state, components


@pytest.mark.parametrize("branch", ["text2image", "image2image"])
def test_actual_z_native_cfg_uses_one_ordered_double_batch_and_float32_guidance(branch):
    blocks, _ = require_modiff_node_contract(ZImageModularPipeline, "denoise")
    block = blocks.sub_blocks[branch].sub_blocks["denoise"].sub_blocks["denoiser"]
    upstream, original_components = run_block(ZImageLoopDenoiser())
    state, components = run_block(block)
    assert len(original_components.transformer.calls) == 2
    assert len(components.transformer.calls) == 1
    call = components.transformer.calls[0]
    assert [value.shape[0] for value in call["cap_feats"]] == [3, 2, 1, 1]
    assert [float(value[0, 0]) for value in call["cap_feats"]] == [1, 2, 0.10009765625, 0.2001953125]
    assert [float(value[0, 0, 0, 0]) for value in call["x"]] == [3, 4, 3, 4]
    torch.testing.assert_close(call["t"], torch.full((4,), 0.125), rtol=0, atol=0)
    assert all(value.dtype == torch.bfloat16 for value in call["x"] + call["cap_feats"])
    assert state.noise_pred.dtype == torch.float32
    pos = torch.tensor([1, 2], dtype=torch.bfloat16).float()
    neg = torch.tensor([0.1, 0.2], dtype=torch.bfloat16).float()
    expected = -(pos + (pos - neg))
    torch.testing.assert_close(state.noise_pred[:, 0, 0, 0], expected, rtol=0, atol=0)
    assert not torch.equal(upstream.noise_pred.float(), state.noise_pred)
    assert components.guider._count_prepared == 2


@pytest.mark.parametrize("enabled,original,scale,conditions", [
    (False, True, 5.0, 1), (False, False, 5.0, 1), (True, True, 0.0, 1),
    (True, False, 1.0, 1), (True, False, 2.0, 2), (True, True, 1.0, 2),
])
def test_actual_cfg_configuration_retains_disabled_zero_and_standard_formulation_semantics(
    enabled, original, scale, conditions,
):
    blocks, _ = require_modiff_node_contract(ZImageModularPipeline, "denoise")
    block = blocks.sub_blocks["text2image"].sub_blocks["denoise"].sub_blocks["denoiser"]
    state, components = run_block(block, enabled=enabled, original=original, scale=scale)
    assert len(components.transformer.calls) == 1
    assert len(components.transformer.calls[0]["x"]) == 2 * conditions
    pos = torch.tensor([1, 2], dtype=torch.bfloat16).float()
    neg = torch.tensor([0.1, 0.2], dtype=torch.bfloat16).float()
    expected = pos if conditions == 1 else (pos if original else neg) + scale * (pos - neg)
    torch.testing.assert_close(state.noise_pred[:, 0, 0, 0], -expected, rtol=0, atol=0)
    assert state.noise_pred.dtype == torch.float32
    assert components.guider._count_prepared == conditions


@pytest.mark.parametrize("step,conditions", [(0, 1), (4, 2), (7, 1)])
def test_actual_cfg_step_window_is_retained(step, conditions):
    blocks, _ = require_modiff_node_contract(ZImageModularPipeline, "denoise")
    block = blocks.sub_blocks["text2image"].sub_blocks["denoise"].sub_blocks["denoiser"]
    _, components = run_block(block, step=step, start=0.5, stop=0.75)
    assert len(components.transformer.calls) == 1
    assert len(components.transformer.calls[0]["x"]) == 2 * conditions


@pytest.mark.parametrize("rescale", [0, 0.3])
def test_real_tiny_bfloat16_z_transformer_matches_whole_pipeline_cfg_arithmetic_on_cpu(rescale):
    blocks, _ = require_modiff_node_contract(ZImageModularPipeline, "denoise")
    block = blocks.sub_blocks["text2image"].sub_blocks["denoise"].sub_blocks["denoiser"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = ZImageTransformer2DModel(
            in_channels=1, dim=32, n_layers=1, n_refiner_layers=1, n_heads=2, n_kv_heads=2,
            cap_feat_dim=1, axes_dims=[4, 6, 6], axes_lens=[128, 128, 128],
        ).to(dtype=torch.bfloat16).eval()
    latents = [torch.full((1, 1, 2, 2), value, dtype=torch.bfloat16) for value in (3, 4)]
    embeddings = [torch.full((length, 1), value, dtype=torch.bfloat16)
                  for length, value in ((3, 1), (2, 2), (1, 0.1), (1, 0.2))]
    original_latents = [value.clone() for value in latents]
    original_embeddings = [value.clone() for value in embeddings]
    # Whole Z-Image repeats the batched tensor before unbinding. Compare that
    # storage-independent reference to the native adapter's repeated references.
    whole_batch = list(torch.stack(latents).repeat(2, 1, 1, 1, 1).unbind())
    with torch.no_grad():
        predictions = model(whole_batch, torch.full((4,), 0.125), embeddings, return_dict=False)[0]
    assert all(value.dtype == torch.bfloat16 for value in predictions)
    positive = torch.stack([value.float() for value in predictions[:2]], dim=0)
    negative = torch.stack([value.float() for value in predictions[2:]], dim=0)
    expected = -(positive + (positive - negative)).squeeze(2)
    if rescale:
        reference_guide = ClassifierFreeGuidance(
            guidance_scale=1, use_original_formulation=True, guidance_rescale=rescale,
        )
        rescaled = -reference_guide.forward(positive.squeeze(2), negative.squeeze(2)).pred
        assert not torch.equal(rescaled, expected)
        expected = rescaled
    with patch.object(model, "forward", wraps=model.forward) as forward:
        state, _ = run_block(block, transformer=model, rescale=rescale)
    forward.assert_called_once()
    assert torch.isfinite(state.noise_pred).all() and torch.count_nonzero(state.noise_pred)
    torch.testing.assert_close(state.noise_pred, expected, rtol=0, atol=0)
    for actual, before in zip(latents, original_latents):
        torch.testing.assert_close(actual, before, rtol=0, atol=0)
    for actual, before in zip(embeddings, original_embeddings):
        torch.testing.assert_close(actual, before, rtol=0, atol=0)
    for actual, before in zip(state.latent_model_input, original_latents):
        torch.testing.assert_close(actual, before, rtol=0, atol=0)


def test_invalid_prediction_batch_cleans_up_all_prepared_cfg_conditions():
    from modules.ModularDiffusers.native_blocks import ZImageFloat32CFGDenoiser

    guide = ClassifierFreeGuidance(guidance_scale=1, use_original_formulation=True)
    model = PredictionTransformer()
    state = BlockState(
        latent_model_input=[torch.ones((1, 1, 2, 2), dtype=torch.bfloat16)],
        timestep=torch.tensor([0.125]), dtype=torch.bfloat16, num_inference_steps=8,
        prompt_embeds=[torch.ones((2, 1))], negative_prompt_embeds=[torch.zeros((1, 1))],
    )
    components = SimpleNamespace(guider=guide, transformer=model)
    with patch.object(model, "forward", return_value=([],)), patch.object(guide, "cleanup_models") as cleanup:
        with pytest.raises(ValueError, match="prediction batch"):
            ZImageFloat32CFGDenoiser()(components, state, i=0, t=torch.tensor(875.0))
    assert cleanup.call_count == 2


def test_other_guider_types_keep_the_pinned_upstream_execution_path():
    blocks, _ = require_modiff_node_contract(ZImageModularPipeline, "denoise")
    block = blocks.sub_blocks["text2image"].sub_blocks["denoise"].sub_blocks["denoiser"]
    components = SimpleNamespace(guider=SimpleNamespace())
    state = BlockState()
    with patch.object(ZImageLoopDenoiser, "__call__", return_value=(components, state)) as upstream:
        assert block(components, state, i=0, t=torch.tensor(1000.0)) == (components, state)
    upstream.assert_called_once()


@pytest.mark.parametrize("changed", ["denoiser", "field_mapping", "block_names"])
def test_changed_denoiser_contract_rejects_both_branch_adaptations_atomically(changed):
    blocks = ZImageModularPipeline().blocks
    text_loop = blocks.sub_blocks["denoise"].sub_blocks["text2image"].sub_blocks["denoise"]
    image_loop = blocks.sub_blocks["denoise"].sub_blocks["image2image"].sub_blocks["denoise"]
    before, denoiser = text_loop.sub_blocks["before_denoiser"], text_loop.sub_blocks["denoiser"]
    if changed == "denoiser":
        image_loop.sub_blocks["denoiser"] = SimpleNamespace()
    elif changed == "field_mapping":
        image_loop.sub_blocks["denoiser"]._guider_input_fields = {"cap_feats": "prompt_embeds"}
    else:
        image_loop.block_names = ["before_denoiser", "changed_denoiser", "after_denoiser"]
    with pytest.raises(ValueError, match="reviewed"):
        prepare_native_pipeline_blocks(ZImageModularPipeline, blocks)
    assert text_loop.sub_blocks["before_denoiser"] is before
    assert text_loop.sub_blocks["denoiser"] is denoiser


def test_guider_adapter_preserves_the_upstream_blueprint_and_deepcopy():
    upstream = ZImageModularPipeline().blocks
    blocks = prepare_native_pipeline_blocks(ZImageModularPipeline, ZImageModularPipeline().blocks)
    for branch in ("text2image", "image2image"):
        old_loop = upstream.sub_blocks["denoise"].sub_blocks[branch].sub_blocks["denoise"]
        new_loop = blocks.sub_blocks["denoise"].sub_blocks[branch].sub_blocks["denoise"]
        assert type(old_loop.sub_blocks["denoiser"]) is ZImageLoopDenoiser
        assert type(deepcopy(new_loop).sub_blocks["denoiser"]) is type(new_loop.sub_blocks["denoiser"])
        assert list(old_loop.sub_blocks) == list(new_loop.sub_blocks)
        assert old_loop.input_names == new_loop.input_names
        assert old_loop.component_names == new_loop.component_names
