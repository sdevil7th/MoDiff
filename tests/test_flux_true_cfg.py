"""Native Flux/Kontext executes the official two-condition CFG contract."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from diffusers import ClassifierFreeGuidance, Flux2KleinModularPipeline, FluxKontextModularPipeline, FluxModularPipeline, FluxTransformer2DModel
from diffusers.modular_pipelines import BlockState, PipelineState
from diffusers.modular_pipelines.flux.encoders import FluxTextEncoderStep
from transformers import CLIPTextConfig, CLIPTextModel, T5Config, T5EncoderModel

from modules.ModularDiffusers.modular_utils import get_model_type_metadata, require_modiff_node_contract
from modules.ModularDiffusers.native_blocks import FluxCFGTextInputStep, prepare_native_pipeline_blocks


@pytest.fixture(scope="module")
def tiny_flux():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(131)
        model = FluxTransformer2DModel(
            in_channels=4, out_channels=4, num_layers=1, num_single_layers=1,
            attention_head_dim=16, num_attention_heads=2, joint_attention_dim=8,
            pooled_projection_dim=8, guidance_embeds=True, axes_dims_rope=(4, 6, 6),
        ).to(dtype=torch.bfloat16).eval()
        values = dict(
            latents=torch.randn((2, 4, 4), dtype=torch.bfloat16),
            image_latents=torch.randn((2, 2, 4), dtype=torch.bfloat16),
            prompt_embeds=torch.randn((2, 3, 8), dtype=torch.bfloat16),
            negative_prompt_embeds=torch.randn((2, 3, 8), dtype=torch.bfloat16),
            pooled_prompt_embeds=torch.randn((2, 8), dtype=torch.bfloat16),
            negative_pooled_prompt_embeds=torch.randn((2, 8), dtype=torch.bfloat16),
            guidance=torch.tensor([3.5, 3.5]),
            txt_ids=torch.zeros((3, 3), dtype=torch.bfloat16),
            joint_attention_kwargs=None, num_inference_steps=24,
        )
    return model, values


def state_for(values, kontext):
    values = dict(values)
    count = 6 if kontext else 4
    values["img_ids"] = torch.tensor([[0, i // 2, i % 2] for i in range(count)], dtype=torch.bfloat16)
    if not kontext:
        values["image_latents"] = None
    return BlockState(**values)


@pytest.mark.parametrize("pipeline", [FluxModularPipeline, FluxKontextModularPipeline])
@pytest.mark.parametrize("enabled,original,rescale,step,start,stop", [
    (True, False, 0.0, 5, 0.0, 1.0), (True, True, 0.0, 5, 0.0, 1.0),
    (True, False, 0.4, 5, 0.0, 1.0), (False, False, 0.0, 5, 0.0, 1.0),
    (True, False, 0.0, 5, 0.5, 0.9), (True, False, 0.0, 23, 0.0, 0.9),
])
def test_actual_tiny_bfloat16_flux_matches_condition_calls_and_whole_formula(
    tiny_flux, pipeline, enabled, original, rescale, step, start, stop,
):
    model, values = tiny_flux
    kontext = pipeline is FluxKontextModularPipeline
    state = state_for(values, kontext)
    snapshots = {key: value.clone() for key, value in vars(state).items() if isinstance(value, torch.Tensor)}
    guider = ClassifierFreeGuidance(guidance_scale=3.0, enabled=enabled, use_original_formulation=original,
                                    guidance_rescale=rescale, start=start, stop=stop)
    reference = guider.new()
    timestep = torch.tensor(908.125)
    reference.set_state(step, 24, timestep)
    expected_batches = reference.prepare_inputs_from_block_state(state, {
        "prompt_embeds": ("prompt_embeds", "negative_prompt_embeds"),
        "pooled_prompt_embeds": ("pooled_prompt_embeds", "negative_pooled_prompt_embeds"),
    })
    model_input = torch.cat([state.latents, state.image_latents], dim=1) if kontext else state.latents
    with torch.no_grad():
        for batch in expected_batches:
            batch.noise_pred = model(
                hidden_states=model_input, timestep=timestep.expand(2).to(torch.bfloat16) / 1000,
                guidance=state.guidance, encoder_hidden_states=batch.prompt_embeds,
                pooled_projections=batch.pooled_prompt_embeds, txt_ids=state.txt_ids,
                img_ids=state.img_ids, return_dict=False,
            )[0][:, :4]
        expected = reference(expected_batches)[0]
    blocks, _ = require_modiff_node_contract(pipeline, "denoise")
    predictor = blocks.sub_blocks["denoise"].sub_blocks["denoiser"]
    components = SimpleNamespace(transformer=model, guider=guider)
    actual_contexts = []
    original_context = model.cache_context
    def context(name):
        actual_contexts.append(name)
        return original_context(name)
    with patch.object(model, "forward", wraps=model.forward) as forward, patch.object(model, "cache_context", side_effect=context):
        predictor(components, state, i=step, t=timestep)
    assert forward.call_count == len(expected_batches)
    assert actual_contexts == ["cond", "uncond"][:len(expected_batches)]
    assert forward.call_args_list[0].kwargs["encoder_hidden_states"] is values["prompt_embeds"]
    assert all(call.kwargs["guidance"] is values["guidance"] for call in forward.call_args_list)
    if len(expected_batches) == 2:
        assert forward.call_args_list[1].kwargs["encoder_hidden_states"] is values["negative_prompt_embeds"]
        # Compare directly with the whole-pipeline native formulation, not
        # merely two calls to the same guider implementation.
        if not original and not rescale:
            manual = expected_batches[1].noise_pred + 3.0 * (expected_batches[0].noise_pred - expected_batches[1].noise_pred)
            torch.testing.assert_close(expected, manual, rtol=0, atol=0)
            assert not torch.equal(expected, expected_batches[0].noise_pred)
    assert state.noise_pred.dtype == torch.bfloat16
    torch.testing.assert_close(state.noise_pred, expected, rtol=0, atol=0)
    assert torch.isfinite(state.noise_pred).all()
    for key, before in snapshots.items():
        torch.testing.assert_close(getattr(state, key), before, rtol=0, atol=0)
    assert guider.get_state()["count_prepared"] == len(expected_batches)


class Tokens:
    model_max_length = 4
    def __call__(self, prompts, *, max_length=None, **kwargs):
        length = max_length or self.model_max_length
        rows = [[1 + sum(map(ord, text)) % 15] * length for text in prompts]
        return SimpleNamespace(input_ids=torch.tensor(rows))
    def batch_decode(self, tokens):
        return ["truncated"] * len(tokens)


@pytest.fixture(scope="module")
def tiny_text_components():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(51)
        clip = CLIPTextModel(CLIPTextConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
                                          num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8)).eval()
        t5 = T5EncoderModel(T5Config(vocab_size=32, d_model=8, d_ff=16, d_kv=4, num_layers=1, num_heads=2)).eval()
    return SimpleNamespace(text_encoder=clip, text_encoder_2=t5, tokenizer=Tokens(), tokenizer_2=Tokens(),
                           _execution_device=torch.device("cpu"))


@pytest.mark.parametrize("pipeline", [FluxModularPipeline, FluxKontextModularPipeline])
def test_real_clip_t5_encoding_delivers_positive_and_negative_pooled_fields_and_repeats(tiny_text_components, pipeline):
    components = SimpleNamespace(**vars(tiny_text_components), guider=ClassifierFreeGuidance(guidance_scale=3.0))
    # Reusing a guider after a final excluded step must still encode negatives
    # needed when Denoise resets it for the next run.
    components.guider = ClassifierFreeGuidance(guidance_scale=3.0, stop=0.8)
    components.guider.set_state(23, 24, torch.tensor(1.0))
    blocks, _ = require_modiff_node_contract(pipeline, "text_encoder")
    state = PipelineState(values={"prompt": ["sun", "moon"], "negative_prompt": ["rain", "snow"], "max_sequence_length": 4})
    blocks(components, state)
    fields = state.get_by_kwargs("denoiser_input_fields")
    assert set(fields) == {"prompt_embeds", "pooled_prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"}
    with torch.no_grad():
        positive = FluxTextEncoderStep.encode_prompt(components, ["sun", "moon"], None, max_sequence_length=4)
        negative = FluxTextEncoderStep.encode_prompt(components, ["rain", "snow"], None, max_sequence_length=4)
    for key, expected in zip(("prompt_embeds", "pooled_prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"), positive + negative):
        torch.testing.assert_close(fields[key], expected, rtol=0, atol=0)
    assert not torch.equal(fields["prompt_embeds"], fields["negative_prompt_embeds"])
    repeated = PipelineState(values={**fields, "num_images_per_prompt": 2})
    FluxCFGTextInputStep()(components, repeated)
    for key, value in fields.items():
        torch.testing.assert_close(repeated.get(key), value.repeat_interleave(2, dim=0), rtol=0, atol=0)


@pytest.mark.parametrize("pipeline", [FluxModularPipeline, FluxKontextModularPipeline])
def test_default_real_component_disables_cfg_and_encoder_does_not_evaluate_negatives(tiny_text_components, pipeline):
    encoder, config = require_modiff_node_contract(pipeline, "text_encoder")
    real_pipeline = encoder.init_pipeline()
    assert type(real_pipeline.guider) is ClassifierFreeGuidance
    assert real_pipeline.guider.get_state()["enabled"] is False
    assert get_model_type_metadata(pipeline.__name__)["guider_options"] == ["ClassifierFreeGuidance"]
    assert config["params"]["guider"]["hidden"] is False
    components = SimpleNamespace(**vars(tiny_text_components), guider=real_pipeline.guider)
    state = PipelineState(values={"prompt": "positive", "negative_prompt": "negative", "max_sequence_length": 4})
    with patch.object(components.text_encoder, "forward", wraps=components.text_encoder.forward) as clip:
        encoder(components, state)
    clip.assert_called_once()
    assert state.get("negative_prompt_embeds") is None
    assert state.get("negative_pooled_prompt_embeds") is None


@pytest.mark.parametrize("pipeline", [FluxModularPipeline, FluxKontextModularPipeline])
@pytest.mark.parametrize("changed", ["encoder", "text_input", "image_input", "image_prepare", "predictor", "after"])
def test_both_complete_blueprints_are_validated_before_any_cfg_mutation(pipeline, changed):
    blocks = pipeline().blocks
    core = blocks.sub_blocks["denoise"]
    loop = core.sub_blocks["denoise"]
    original_encoder = blocks.sub_blocks["text_encoder"]
    original_predictor = loop.sub_blocks["denoiser"]
    image_branch = "img2img" if pipeline is FluxModularPipeline else "image_conditioned"
    if changed == "encoder":
        blocks.sub_blocks["text_encoder"] = SimpleNamespace()
    elif changed == "text_input":
        core.sub_blocks["input"].sub_blocks["text2image"] = SimpleNamespace()
    elif changed == "image_input":
        core.sub_blocks["input"].sub_blocks[image_branch].sub_blocks["text_inputs"] = SimpleNamespace()
    elif changed == "image_prepare":
        core.sub_blocks["before_denoise"].sub_blocks[image_branch] = SimpleNamespace()
    else:
        loop.sub_blocks["denoiser" if changed == "predictor" else "after_denoiser"] = SimpleNamespace()
    with pytest.raises(ValueError, match="reviewed"):
        prepare_native_pipeline_blocks(pipeline, blocks)
    if changed != "encoder":
        assert blocks.sub_blocks["text_encoder"] is original_encoder
    if changed != "predictor":
        assert loop.sub_blocks["denoiser"] is original_predictor


def test_unknown_guider_fails_before_encoder_or_model_execution(tiny_text_components, tiny_flux):
    class UnknownGuider(ClassifierFreeGuidance):
        pass
    for pipeline in (FluxModularPipeline, FluxKontextModularPipeline):
        encoder, _ = require_modiff_node_contract(pipeline, "text_encoder")
        components = SimpleNamespace(**vars(tiny_text_components), guider=UnknownGuider())
        with patch.object(components.text_encoder, "forward", wraps=components.text_encoder.forward) as forward:
            with pytest.raises(ValueError, match="exact ClassifierFreeGuidance"):
                encoder(components, PipelineState(values={"prompt": "positive"}))
        forward.assert_not_called()
        blocks, _ = require_modiff_node_contract(pipeline, "denoise")
        model, values = tiny_flux
        with patch.object(model, "forward", wraps=model.forward) as forward:
            with pytest.raises(ValueError, match="exact ClassifierFreeGuidance"):
                blocks.sub_blocks["denoise"].sub_blocks["denoiser"](
                    SimpleNamespace(transformer=model, guider=UnknownGuider()),
                    state_for(values, pipeline is FluxKontextModularPipeline), i=0, t=torch.tensor(1000.0),
                )
        forward.assert_not_called()


@pytest.mark.parametrize("pipeline", [FluxModularPipeline, FluxKontextModularPipeline])
def test_missing_negative_condition_fails_before_any_model_forward(tiny_flux, pipeline):
    model, values = tiny_flux
    state = state_for(values, pipeline is FluxKontextModularPipeline)
    state.negative_pooled_prompt_embeds = None
    blocks, _ = require_modiff_node_contract(pipeline, "denoise")
    with patch.object(model, "forward", wraps=model.forward) as forward:
        with pytest.raises(ValueError, match="conditioning batches"):
            blocks.sub_blocks["denoise"].sub_blocks["denoiser"](
                SimpleNamespace(transformer=model, guider=ClassifierFreeGuidance(guidance_scale=3.0)),
                state, i=0, t=torch.tensor(1000.0),
            )
    forward.assert_not_called()


@pytest.mark.parametrize("pipeline", [FluxModularPipeline, FluxKontextModularPipeline])
def test_failed_negative_forward_cleans_up_each_prepared_condition_and_next_run_resets(tiny_flux, pipeline):
    model, values = tiny_flux
    state = state_for(values, pipeline is FluxKontextModularPipeline)
    blocks, _ = require_modiff_node_contract(pipeline, "denoise")
    predictor = blocks.sub_blocks["denoise"].sub_blocks["denoiser"]
    guider = ClassifierFreeGuidance(guidance_scale=3.0)
    components = SimpleNamespace(transformer=model, guider=guider)
    original = model.forward
    seen = []
    def fail_negative(**kwargs):
        seen.append(kwargs["encoder_hidden_states"])
        if len(seen) == 2:
            raise RuntimeError("bounded regression failure")
        return original(**kwargs)
    with patch.object(model, "forward", side_effect=fail_negative), patch.object(guider, "cleanup_models", wraps=guider.cleanup_models) as cleanup:
        with pytest.raises(RuntimeError, match="bounded regression failure"):
            predictor(components, state, i=0, t=torch.tensor(1000.0))
    assert cleanup.call_count == 2
    assert guider.get_state()["count_prepared"] == 2
    with patch.object(model, "forward", wraps=model.forward) as forward:
        predictor(components, state, i=0, t=torch.tensor(1000.0))
    assert forward.call_count == 2 and guider.get_state()["count_prepared"] == 2
    assert torch.isfinite(state.noise_pred).all()


def test_encoder_runtime_disable_skips_negative_even_when_original_config_was_enabled(tiny_text_components):
    components = SimpleNamespace(**vars(tiny_text_components), guider=ClassifierFreeGuidance(guidance_scale=3.0))
    components.guider.disable()
    encoder, _ = require_modiff_node_contract(FluxModularPipeline, "text_encoder")
    state = PipelineState(values={"prompt": "positive", "negative_prompt": "negative", "max_sequence_length": 4})
    encoder(components, state)
    assert state.get("negative_prompt_embeds") is None


def test_other_pipeline_tree_and_upstream_global_classes_remain_unchanged():
    blocks = Flux2KleinModularPipeline().blocks
    assert prepare_native_pipeline_blocks(Flux2KleinModularPipeline, blocks) is blocks
    for pipeline in (FluxModularPipeline, FluxKontextModularPipeline):
        original = pipeline().blocks
        assert type(original.sub_blocks["text_encoder"]) is FluxTextEncoderStep
        adapted = prepare_native_pipeline_blocks(pipeline, pipeline().blocks)
        assert type(deepcopy(adapted).sub_blocks["text_encoder"]) is type(adapted.sub_blocks["text_encoder"])
        assert "guider" not in original.component_names
        assert "guider" in adapted.component_names
