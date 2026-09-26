"""Exercise the pinned native cache with a tiny real CPU denoiser, without weights."""
import importlib.util
from types import SimpleNamespace

import pytest


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None,
                    reason="Reviewed optional model runtime not active")
def test_native_context_cache_prefills_once_and_is_private_to_each_invocation(monkeypatch):
    import torch
    from diffusers import (
        AutoencoderKLQwenImage21, QwenImage21Pipeline, QwenImage21Transformer2DModel,
        FlowMatchEulerDiscreteScheduler,
    )

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(71)
        transformer = QwenImage21Transformer2DModel(
            patch_size=1, in_channels=4, out_channels=4, num_layers=2,
            attention_head_dim=16, num_attention_heads=2, context_in_dim=8,
            mlp_ratio=2, axes_dims_rope=(4, 6, 6), causal_condition=True,
        ).eval()
        # The real denoiser receives supplied embeddings. Only constructor-time
        # template metadata is stubbed; this test makes no text-encoding claim.
        class ProcessorStub:
            tokenizer = SimpleNamespace(encode=lambda value: [2])

            def apply_chat_template(self, *args, **kwargs):
                return [[0, 1]]

        processor = ProcessorStub()
        vae = AutoencoderKLQwenImage21(
            base_dim=16, decoder_base_dim=16, z_dim=4, dim_mult=[1, 1, 1, 1, 1],
            num_res_blocks=1, latents_mean=[0.1] * 4, latents_std=[0.8] * 4,
        ).eval()
        pipeline = QwenImage21Pipeline(
            transformer=transformer, vae=vae, text_encoder=None, processor=processor,
            scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=False),
        )
        pipeline.set_progress_bar_config(disable=True)
        embeddings = torch.randn(1, 4, 8)
        events, caches = [], []
        forward = transformer.forward

        def observe(*args, **kwargs):
            mode, cache = kwargs.get("kv_cache_mode"), kwargs.get("kv_cache")
            events.append(mode)
            if mode == "extract":
                assert all(layer.k is None and layer.v is None for layer in cache.layer_caches)
                caches.append(cache)
            result = forward(*args, **kwargs)
            if mode == "extract":
                for layer in cache.layer_caches:
                    for tensor in layer.get():
                        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
            return result

        monkeypatch.setattr(transformer, "forward", observe)

        def run(prompt_embeds, *, enabled=True, interrupted=False, guidance=False):
            events.clear()

            def callback(owner, step, timestep, values):
                if interrupted:
                    owner._interrupt = True
                return values

            output = pipeline(
                prompt_embeds=prompt_embeds, height=64, width=64, num_inference_steps=3,
                true_cfg_scale=2.0 if guidance else 1.0,
                negative_prompt_embeds=-prompt_embeds if guidance else None,
                generator=torch.Generator().manual_seed(123), use_kv_cache=enabled,
                output_type="latent", callback_on_step_end=callback,
            ).images
            expected = (["extract", "cached", "cached"] if enabled else [None] * 3)
            if interrupted:
                expected = expected[:1]
            if guidance:
                expected = [event for event in expected for _ in range(2)]
            assert events == expected
            assert torch.isfinite(output).all()
            return output

        baseline = run(embeddings)
        torch.testing.assert_close(run(embeddings), baseline, atol=0, rtol=0)
        assert not torch.equal(run(embeddings + 0.5), baseline)
        run(embeddings, enabled=False)
        run(embeddings, interrupted=True)
        torch.testing.assert_close(run(embeddings), baseline, atol=0, rtol=0)
        run(embeddings, guidance=True)
        assert len(caches) == len({id(cache) for cache in caches}) == 7

        from modules.DiffusersImage.main import DecodeLatents

        original = baseline.clone()
        expected = pipeline(
            prompt_embeds=embeddings, height=64, width=64, num_inference_steps=3,
            generator=torch.Generator().manual_seed(123), use_kv_cache=True, output_type="pt",
        ).images
        decoded = object.__new__(DecodeLatents).execute(pipeline, baseline, width=64, height=64, output_type="pt")
        assert decoded["images"].shape == (1, 4, 64, 64)
        torch.testing.assert_close(decoded["images"], expected, atol=0, rtol=0)
        torch.testing.assert_close(baseline, original, atol=0, rtol=0)
        with pytest.raises(ValueError, match="incompatible layout"):
            object.__new__(DecodeLatents).execute(pipeline, baseline, width=128, height=64)
    finally:
        torch.set_num_threads(previous_threads)
