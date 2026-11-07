"""ROCm multimodal attention fallback preserves unrelated attention choices."""

from types import SimpleNamespace

import pytest

import modules.DiffusersRuntime.main as runtime
from modules.DiffusersRuntime.main import configure_rocm_vision_attention, rocm_vision_attention_requires_configuration


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


@pytest.mark.parametrize("hip,implementation,expected", [
    (None, "sdpa", False), ("7.2", "sdpa", True), ("7.2", "eager", False), ("7.2", "flash_attention_2", False),
])
def test_shared_vision_predicate_inspects_without_mutating_owned_or_shared_models(hip, implementation, expected):
    encoder = model(implementation)
    assert rocm_vision_attention_requires_configuration(
        encoder, torch_module=SimpleNamespace(version=SimpleNamespace(hip=hip)),
    ) is expected
    assert encoder.config.vision_config._attn_implementation == implementation
    assert encoder.config.text_config._attn_implementation == "sdpa"
    assert encoder.calls == []


def test_vision_without_a_public_setter_is_not_implicitly_claimed_as_configurable():
    encoder = model()
    encoder.set_attn_implementation = None
    assert not rocm_vision_attention_requires_configuration(
        encoder, torch_module=SimpleNamespace(version=SimpleNamespace(hip="7.2")),
    )
    assert encoder.config.vision_config._attn_implementation == "sdpa" and encoder.calls == []


def rocm_torch():
    import torch

    return SimpleNamespace(version=SimpleNamespace(hip="7.2"), bfloat16=torch.bfloat16, backends=torch.backends)


@pytest.mark.parametrize("model_type", ["qwen3_vl", "qwen3_vl_vision"])
def test_rocm_qwen3_bfloat16_uses_the_scoped_math_adapter_without_changing_text(monkeypatch, model_type):
    import torch

    encoder = model()
    encoder.config.vision_config.model_type = model_type
    encoder.dtype = torch.bfloat16
    registrations = []
    monkeypatch.setattr(runtime, "_register_rocm_bf16_vision_math", lambda: registrations.append(True), raising=False)
    assert rocm_vision_attention_requires_configuration(encoder, torch_module=rocm_torch())
    assert not registrations and not encoder.calls
    assert configure_rocm_vision_attention(
        SimpleNamespace(components={"text_encoder": encoder}), torch_module=rocm_torch(),
        component_policies={"text_encoder": "modiff_rocm_bf16_vision_math_v1"},
    ) == ["text_encoder"]
    assert encoder.calls == [{"vision_config": "modiff_rocm_bf16_vision_math_v1"}]
    assert encoder.config.text_config._attn_implementation == "sdpa"
    assert registrations == [True]


@pytest.mark.parametrize("model_type,dtype", [
    ("qwen2_5_vl", "bfloat16"), ("qwen3_vl", "bfloat16"),
    ("qwen3_vl", "float16"), ("qwen3_vl", "float32"),
])
def test_existing_vision_fallbacks_are_not_promoted_to_the_bfloat16_policy(model_type, dtype):
    import torch

    encoder = model()
    encoder.config.vision_config.model_type = model_type
    encoder.dtype = getattr(torch, dtype)
    configure_rocm_vision_attention(SimpleNamespace(components={"text_encoder": encoder}), torch_module=rocm_torch())
    assert encoder.calls == [{"vision_config": "eager"}]


@pytest.mark.parametrize("choice,dtype", [("eager", "bfloat16"), ("sdpa", "float32")])
def test_declared_reference_policy_cannot_claim_an_incompatible_owner(choice, dtype):
    import torch

    encoder = model(choice)
    encoder.config.vision_config.model_type = "qwen3_vl_vision"
    encoder.dtype = getattr(torch, dtype)
    with pytest.raises(RuntimeError, match="reference vision"):
        configure_rocm_vision_attention(
            SimpleNamespace(components={"text_encoder": encoder}), torch_module=rocm_torch(),
            component_policies={"text_encoder": runtime.ROCM_BF16_VISION_MATH},
        )
    assert not encoder.calls


def test_reference_policy_requires_a_real_loaded_vision_component():
    with pytest.raises(ValueError, match="loaded component"):
        configure_rocm_vision_attention(
            SimpleNamespace(components={}), torch_module=rocm_torch(),
            component_policies={"text_encoder": runtime.ROCM_BF16_VISION_MATH},
        )
    with pytest.raises(RuntimeError, match="BF16 Qwen3-VL"):
        configure_rocm_vision_attention(
            SimpleNamespace(components={"text_encoder": object()}), torch_module=rocm_torch(),
            component_policies={"text_encoder": runtime.ROCM_BF16_VISION_MATH},
        )


@pytest.mark.parametrize("invalid", [False, [], "math", {"text_encoder": "unknown"}])
def test_reference_vision_policy_rejects_malformed_metadata_before_mutation(invalid):
    encoder = model()
    with pytest.raises(ValueError, match="declared component policy"):
        configure_rocm_vision_attention(SimpleNamespace(components={"text_encoder": encoder}), torch_module=rocm_torch(), component_policies=invalid)
    assert not encoder.calls


def test_scoped_math_adapter_registers_the_official_sdpa_mask_as_well_as_attention(monkeypatch):
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

    calls = []
    monkeypatch.setattr(AttentionInterface, "register", classmethod(lambda cls, key, value: calls.append(("attention", key, value))))
    monkeypatch.setattr(AttentionMaskInterface, "register", classmethod(lambda cls, key, value: calls.append(("mask", key, value))))
    runtime._register_rocm_bf16_vision_math()
    assert calls == [
        ("attention", "modiff_rocm_bf16_vision_math_v1", runtime._rocm_bf16_vision_math_attention),
        ("mask", "modiff_rocm_bf16_vision_math_v1", sdpa_mask),
    ]


@pytest.mark.parametrize("fails,prior", [(False, False), (False, True), (True, False), (True, True)])
def test_scoped_math_adapter_delegates_and_restores_precision_and_kernel_flags(monkeypatch, fails, prior):
    import torch
    from transformers.integrations import sdpa_attention

    previous = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
    flags = lambda: (torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.math_sdp_enabled())
    original_flags = flags()
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(prior)
    query = SimpleNamespace(dtype=torch.bfloat16)
    key, value, mask, owner = (object() for _ in range(4))
    output = (torch.zeros(1, dtype=torch.bfloat16), None)

    def forward(*args, **kwargs):
        assert args == (owner, query, key, value, mask)
        assert kwargs == {"dropout": 0.0, "scaling": 0.125}
        assert torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed() is True
        assert flags() == (False, False, True)
        if fails:
            raise RuntimeError("upstream failure")
        return output

    monkeypatch.setattr(sdpa_attention, "sdpa_attention_forward", forward)
    try:
        if fails:
            with pytest.raises(RuntimeError, match="upstream failure"):
                runtime._rocm_bf16_vision_math_attention(owner, query, key, value, mask, dropout=0.0, scaling=0.125)
        else:
            assert runtime._rocm_bf16_vision_math_attention(owner, query, key, value, mask, dropout=0.0, scaling=0.125) is output
        assert torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed() is prior
        assert flags() == original_flags
    finally:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous)


def test_scoped_math_adapter_refuses_nonfinite_vision_before_later_generation(monkeypatch):
    import torch
    from transformers.integrations import sdpa_attention

    prior = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
    monkeypatch.setattr(sdpa_attention, "sdpa_attention_forward", lambda *args, **kwargs: (torch.tensor([float("nan")], dtype=torch.bfloat16), None))
    with pytest.raises(RuntimeError, match="Non-finite.*vision"):
        runtime._rocm_bf16_vision_math_attention(None, SimpleNamespace(dtype=torch.bfloat16), None, None, None)
    assert torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed() is prior


def test_registered_math_policy_runs_a_tiny_official_vision_model_without_downloads():
    import torch
    from transformers import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        config = Qwen3VLVisionConfig(
            hidden_size=32, intermediate_size=64, num_heads=4, depth=1,
            out_hidden_size=64, patch_size=2, temporal_patch_size=2,
            spatial_merge_size=2, num_position_embeddings=16, deepstack_visual_indexes=[],
        )
        config._attn_implementation = "eager"
        vision = Qwen3VLVisionModel(config).eval().to(dtype=torch.bfloat16)
        encoder = model()
        encoder.config.vision_config = vision.config
        encoder.dtype = torch.bfloat16
        encoder.set_attn_implementation = lambda value: vision.set_attn_implementation(value["vision_config"])
        pixels = torch.randn(4, 24, dtype=torch.bfloat16)
        grid = torch.tensor([[1, 2, 2]])
        prior = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        with torch.inference_mode():
            expected = vision(pixels, grid_thw=grid)
            vision.set_attn_implementation("sdpa")
            configure_rocm_vision_attention(
                SimpleNamespace(components={"text_encoder": encoder}), torch_module=rocm_torch(),
                component_policies={"text_encoder": "modiff_rocm_bf16_vision_math_v1"},
            )
            actual = vision(pixels, grid_thw=grid)
        assert vision.config._attn_implementation == "modiff_rocm_bf16_vision_math_v1"
        for field in ("last_hidden_state", "pooler_output"):
            value = getattr(actual, field)
            assert torch.isfinite(value).all()
            torch.testing.assert_close(value, getattr(expected, field), rtol=0.03, atol=0.03)
        assert torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed() is prior
