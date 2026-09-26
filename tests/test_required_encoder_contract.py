"""The registry must expose the model dependency before graph execution."""
from modules.ModularDiffusers.embeddings import EncodePrompt


def test_prompt_encoder_requires_models_but_accepts_inline_prompt():
    assert EncodePrompt.params['text_encoders']['required'] is True
    assert EncodePrompt.params['text_encoders']['display'] == 'input'
    assert not EncodePrompt.params['prompt_input'].get('required', False)
