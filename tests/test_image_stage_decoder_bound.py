"""Existing decoder bounds must be refused before the official sampler."""
from unittest.mock import patch

import pytest

from test_image_stage_graph import runtime as _runtime
runtime = _runtime
from test_image_stage_lora_epoch import loaded
from modules.DiffusersImage import stage_execution as core
from image_stage_test_support import stage_offline_runtime as stage_offline_runtime

pytestmark = pytest.mark.usefixtures('stage_offline_runtime')


@pytest.mark.parametrize('kind', ['fill', 'control'])
def test_real_encoded_nine_batch_refuses_before_official_sampler(runtime, kind):
    owner, strategy, given = loaded(runtime, kind)
    given.update(prompt=['sun'] * 9, num_inference_steps=1, guidance_scale=1., strength=1.)
    encoded = core.encode(owner, strategy, given)
    assert encoded.material['prompt_embeds'].shape[0] == 9
    with patch.object(owner.transformer, 'forward', wraps=owner.transformer.forward) as forward:
        with pytest.raises(ValueError, match='decoder'):
            core.denoise(owner, strategy, encoded, given, {})
        assert forward.call_count == 0


def test_actual_text_only_encode_can_reject_existing_decoder_pixel_limit_before_large_call(runtime):
    owner, strategy, given = loaded(runtime, 'fill')
    given.update(prompt=['sun'] * 5, width=2048, height=2048,
                 num_inference_steps=1, guidance_scale=1., strength=1.)
    # Fill's real Encode stage performs text only. Source/mask remain tiny PIL
    # inputs; this does not allocate a large latent/image or run a large model.
    encoded = core.encode(owner, strategy, given)
    assert encoded.material['prompt_embeds'].shape[0] == 5
    with patch.object(strategy, 'denoise', side_effect=AssertionError(
            'Existing decoder pixel bound reached the expensive sampler')) as sampler:
        with pytest.raises(ValueError, match='decoder'):
            core.denoise(owner, strategy, encoded, given, {})
        assert sampler.call_count == 0
