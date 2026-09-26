"""ROCm multimodal attention fallback preserves unrelated attention choices."""

from types import SimpleNamespace

import pytest

from modules.DiffusersRuntime.main import configure_rocm_vision_attention


def model(implementation="sdpa", *, accepts=True):
    vision = SimpleNamespace(_attn_implementation=implementation)
    text = SimpleNamespace(_attn_implementation="sdpa")
    calls = []

    def set_attention(value):
        calls.append(value)
        if accepts:
            vision._attn_implementation = value["vision_config"]

    return SimpleNamespace(
        config=SimpleNamespace(vision_config=vision, text_config=text),
        set_attn_implementation=set_attention,
        calls=calls,
    )


def test_rocm_only_changes_the_sdpa_vision_subconfig_once():
    encoder = model()
    pipeline = SimpleNamespace(
        components={"text_encoder": encoder, "alias": encoder, "transformer": object(), "processor": object()}
    )
    result = configure_rocm_vision_attention(
        pipeline, torch_module=SimpleNamespace(version=SimpleNamespace(hip="7.2"))
    )
    assert result == ["text_encoder"]
    assert encoder.calls == [{"vision_config": "eager"}]
    assert encoder.config.text_config._attn_implementation == "sdpa"
    assert (
        configure_rocm_vision_attention(pipeline, torch_module=SimpleNamespace(version=SimpleNamespace(hip="7.2")))
        == []
    )


@pytest.mark.parametrize("hip,implementation", [(None, "sdpa"), ("7.2", "eager"), ("7.2", "flash_attention_2")])
def test_nvidia_and_explicit_non_sdpa_implementations_remain_unchanged(hip, implementation):
    encoder = model(implementation)
    pipeline = SimpleNamespace(components={"text_encoder": encoder})
    assert (
        configure_rocm_vision_attention(pipeline, torch_module=SimpleNamespace(version=SimpleNamespace(hip=hip))) == []
    )
    assert not encoder.calls


def test_a_rejected_fallback_is_not_reported_as_applied():
    encoder = model(accepts=False)
    pipeline = SimpleNamespace(components={"text_encoder": encoder})
    with pytest.raises(RuntimeError, match="vision attention"):
        configure_rocm_vision_attention(pipeline, torch_module=SimpleNamespace(version=SimpleNamespace(hip="7.2")))
