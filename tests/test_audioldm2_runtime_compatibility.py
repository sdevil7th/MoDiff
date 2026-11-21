"""Exercise the official AudioLDM2/GPT-2 bridge without downloading weights."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from modules.DiffusersAudio.main import (
    _ensure_language_model_generation_api,
    _reviewed_audioldm2_generation_helper,
)


def _current_transformers():
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "5.18.0":
        pytest.skip("exact Transformers 5.18.0 compatibility regression")
    return transformers


def test_reviewed_base_runtime_runs_real_audioldm2_language_generation():
    transformers = _current_transformers()
    diffusers = pytest.importorskip("diffusers")
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.manual_seed(19)
            model = transformers.GPT2Model(transformers.GPT2Config(
                vocab_size=16, n_positions=32, n_embd=8, n_layer=1, n_head=2,
                bos_token_id=0, eos_token_id=1,
                resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
            )).eval()
            pipeline = SimpleNamespace(language_model=model)
            inputs = torch.randn(2, 3, 8)
            assert not hasattr(model, "_update_model_kwargs_for_generation")
            with pytest.raises(AttributeError, match="_update_model_kwargs_for_generation"):
                diffusers.AudioLDM2Pipeline.generate_language_model(
                    pipeline, inputs, max_new_tokens=4,
                    attention_mask=torch.ones(2, 3, dtype=torch.long),
                )

            _ensure_language_model_generation_api(pipeline, "AudioLDM2Pipeline")
            bound = model._update_model_kwargs_for_generation
            _ensure_language_model_generation_api(pipeline, "AudioLDM2Pipeline")
            assert model._update_model_kwargs_for_generation is bound
            actual = diffusers.AudioLDM2Pipeline.generate_language_model(
                pipeline, inputs, max_new_tokens=4,
                attention_mask=torch.ones(2, 3, dtype=torch.long),
            )
            repeated = diffusers.AudioLDM2Pipeline.generate_language_model(
                pipeline, inputs, max_new_tokens=4,
                attention_mask=torch.ones(2, 3, dtype=torch.long),
            )
            expected = inputs
            for _ in range(4):
                output = model(
                    inputs_embeds=expected, use_cache=False, output_hidden_states=True,
                    attention_mask=torch.ones(expected.shape[:2], dtype=torch.long),
                    return_dict=True,
                )
                expected = torch.cat((expected, output.hidden_states[-1][:, -1:, :]), dim=1)
            assert actual.shape == (2, 4, 8)
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected[:, -4:, :], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(actual, repeated, atol=0, rtol=0)
    finally:
        torch.set_num_threads(previous_threads)


def test_reviewed_helper_refuses_unknown_version_and_changed_source():
    transformers = _current_transformers()
    with patch.object(transformers, "__version__", "5.18.1"):
        with pytest.raises(RuntimeError, match="has not been reviewed"):
            _reviewed_audioldm2_generation_helper()
    with patch("inspect.getsource", return_value="changed upstream helper"):
        with pytest.raises(RuntimeError, match="has not been reviewed"):
            _reviewed_audioldm2_generation_helper()
