"""Actual official CPU-target hooks; no accelerator residency/fit proof."""

from unittest.mock import patch

import pytest
import torch

from modules.DiffusersImage import stage_execution as core
from image_stage_test_support import tiny_pipeline
from test_image_stage_numerical import selected_owner, values, old_call
from modules.DiffusersImage import main as nodes

from image_stage_test_support import stage_offline_runtime as stage_offline_runtime

pytestmark = pytest.mark.usefixtures("stage_offline_runtime")


@pytest.mark.parametrize("kind", ["fill", "control"])
def test_cpu_target_official_model_hooks_survive_split_and_repeat(kind):
    owner, strategy, _ = selected_owner(kind)
    baseline = tiny_pipeline(kind)
    given = values()
    owner.enable_model_cpu_offload(device="cpu")
    baseline.enable_model_cpu_offload(device="cpu")
    assert owner._all_hooks and owner.transformer._hf_hook is not None
    first_hooks = tuple(owner._all_hooks)
    with patch.object(owner, "enable_model_cpu_offload", wraps=owner.enable_model_cpu_offload) as reinstall:
        encoded = core.encode(owner, strategy, given)
        assert tuple(owner._all_hooks) == first_hooks
        actual, binding = core.denoise(owner, strategy, encoded, given, {})
        assert reinstall.call_count == 1
        assert tuple(owner._all_hooks) != first_hooks
        decoder, matching = core.decoder_context(owner, strategy, actual, binding, 64, 80)
        decoded = nodes.DecodeLatents("actual-cpu-hook-decode")(
            pipeline=decoder, latents=actual, latent_binding=binding, width=64, height=80, output_type="pt"
        )["images"]
        assert reinstall.call_count == 2
        assert matching is encoded
        expected = baseline(**old_call(kind, given, "pt")).images
        torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
        repeated, _ = core.denoise(owner, strategy, encoded, given, {})
        torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
        assert reinstall.call_count == 3
    owner.remove_all_hooks()
    with pytest.raises(ValueError, match="stale"):
        # Cleanup destroys the official owner grant explicitly, independent of
        # whether ordinary CPU hooks happened to leave the same component refs.
        core.begin_mutation(owner)
        core.require_encoding(owner, strategy, encoded, given)


@pytest.mark.parametrize("kind", ["fill", "control"])
def test_supplied_embedding_single_image_and_final_pil_equal_full_call(kind):
    owner, strategy, _ = selected_owner(kind)
    baseline = tiny_pipeline(kind)
    given = values()
    with torch.no_grad():
        prompt, pooled, _ = owner.encode_prompt(
            prompt=["sun"], device="cpu", num_images_per_prompt=1, max_sequence_length=4
        )
    given.update(prompt=None, prompt_embeds=prompt, pooled_prompt_embeds=pooled)
    encoded = core.encode(owner, strategy, given)
    actual, binding = core.denoise(owner, strategy, encoded, given, {})
    decoder, matching = core.decoder_context(owner, strategy, actual, binding, 64, 80)
    decoded = nodes.DecodeLatents("embedding-pil")(
        pipeline=decoder, latents=actual, latent_binding=binding, width=64, height=80, output_type="pil"
    )["images"]
    assert matching is encoded
    actual = decoded
    call = old_call(kind, given, "pil")
    call.update(prompt_embeds=prompt, pooled_prompt_embeds=pooled)
    if kind == "fill":
        expected = nodes.Inpaint("embedding-full-pil")(pipeline=baseline, **given, output_type="pil")["images"]
    else:
        expected = baseline(**call).images
    assert [item.tobytes() for item in actual] == [item.tobytes() for item in expected]


@pytest.mark.parametrize("strength,total", [(1.0, 4), (0.5, 2), (0.25, 1)])
def test_fill_official_strength_step_total_is_preserved(strength, total):
    owner, strategy, _ = selected_owner("fill")
    given = {**values(), "num_inference_steps": 4, "strength": strength}
    encoded = core.encode(owner, strategy, given)
    seen = []

    def callback(pipe, step, timestep, kwargs):
        seen.append((pipe is owner, step, pipe.num_timesteps))
        return kwargs

    actual, _ = core.denoise(
        owner, strategy, encoded, given, dict(callback_on_step_end=callback, callback_on_step_end_tensor_inputs=[])
    )
    expected = owner(**old_call("fill", given, "latent")).images
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert seen == [(True, step, total) for step in range(total)]
