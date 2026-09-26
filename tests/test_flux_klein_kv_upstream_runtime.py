"""Pinned KV runtime investigation: tiny real CPU models, not MoDiff admission."""
import importlib.util

import pytest


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None,
                    reason="Optional model runtime not active")
def test_pinned_klein_kv_reference_cache_is_per_call_and_cleared(monkeypatch):
    import torch
    from PIL import Image
    from diffusers import (AutoencoderKLFlux2, Flux2Transformer2DModel,
                           FlowMatchEulerDiscreteScheduler, Flux2KleinKVPipeline)

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(42)
        transformer = Flux2Transformer2DModel(in_channels=8, out_channels=8, num_layers=1,
            num_single_layers=1, attention_head_dim=16, num_attention_heads=1,
            joint_attention_dim=8, axes_dims_rope=(4, 4, 4, 4), guidance_embeds=False)
        vae = AutoencoderKLFlux2(block_out_channels=(32, 32, 32, 32), layers_per_block=1,
                                latent_channels=2).eval()
        original_processor_type = type(transformer.transformer_blocks[0].attn.processor)
        pipeline = Flux2KleinKVPipeline(transformer=transformer, vae=vae, text_encoder=None,
            tokenizer=None, scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=False))
        # This constructor mutates its owned transformer's processors. A future
        # adapter must not silently share it with a non-KV cached pipeline.
        assert type(transformer.transformer_blocks[0].attn.processor) is not original_processor_type
        pipeline.set_progress_bar_config(disable=True)
        prompt_embeds = torch.randn((1, 3, 8))
        calls, caches = [], []
        forward = transformer.forward

        def observe(*args, **kwargs):
            calls.append(kwargs.get("kv_cache_mode", "standard"))
            result = forward(*args, **kwargs)
            if kwargs.get("kv_cache_mode") == "extract":
                caches.append(result[1])
            return result

        monkeypatch.setattr(transformer, "forward", observe)

        def run(images, *, interrupt=False):
            calls.clear()

            def on_step(owner, index, timestep, values):
                if interrupt:
                    owner._interrupt = True
                return values

            result = pipeline(prompt_embeds=prompt_embeds, image=images, height=64, width=64,
                num_inference_steps=3, generator=torch.Generator().manual_seed(99),
                output_type="latent", callback_on_step_end=on_step).images
            expected = ["standard"] * 3 if images is None else ["extract", "cached", "cached"]
            assert calls == (expected[:1] if interrupt else expected)
            assert torch.isfinite(result).all()
            for cache in caches:
                assert cache.num_ref_tokens == 0
                assert all(layer.k_ref is None and layer.v_ref is None for layer in
                           [*cache.double_block_caches, *cache.single_block_caches])
            return result

        baseline = run(None)
        with pytest.raises(ValueError, match="Both dimensions must be at least 64px"):
            run(Image.new("RGB", (32, 32), "red"))
        red, blue = Image.new("RGB", (64, 64), "red"), Image.new("RGB", (64, 64), "blue")
        first = run(red)
        torch.testing.assert_close(run(red), first, rtol=0, atol=0)
        assert not torch.equal(run(blue), first)
        assert not torch.equal(run([red, blue]), first)
        run(red, interrupt=True)
        # A preceding conditional/partially interrupted call cannot contaminate
        # an unconditional repeat. This is upstream interrupt, not app cancel proof.
        torch.testing.assert_close(run(None), baseline, rtol=0, atol=0)
        assert len({id(cache) for cache in caches}) == len(caches) == 5
    finally:
        torch.set_num_threads(previous_threads)
