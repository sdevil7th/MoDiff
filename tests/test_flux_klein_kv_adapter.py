"""KV-specific contracts and actual tiny MoDiff action execution, no downloads."""
import importlib.util
from unittest.mock import Mock

import pytest


def test_kv_contract_has_only_real_upstream_controls_and_separate_identity():
    from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS, image_pipeline_contract, LoadPipeline
    from modiff.studio_execution_specs import STUDIO_EXECUTION_SPEC_DEFINITIONS
    adapter = IMAGE_PIPELINE_ADAPTERS['Flux2KleinKVPipeline']
    assert adapter.load_pipeline_class == 'Flux2KleinKVPipeline'
    assert adapter.guidance_parameter is None
    assert 'pipeline_class' not in LoadPipeline.cache_ignored_params
    assert 'model_id' not in LoadPipeline.cache_ignored_params
    for mode in adapter.mode_options:
        contract = image_pipeline_contract(adapter, mode)
        for key in ('negative_prompt', 'guidance_scale', 'image_guidance_scale', 'strength'):
            assert contract['fieldParams'][key]['hidden']
        for key in ('width', 'height', 'max_sequence_length'):
            assert not contract['fieldParams'][key]['hidden']
        spec = STUDIO_EXECUTION_SPEC_DEFINITIONS[f"flux2-klein-kv:{mode.replace('_', '-')}:v1"]
        assert spec['profile']['pipeline_class'] == adapter.pipeline_class
        assert spec['profile']['default_repo'] == adapter.default_repo
        assert spec['profile']['default_quantized_components'] == ()
        assert not spec['profile']['live_proof']
        assert not spec['capability']['autoEligible']
        assert not spec['capability']['qualifiedModes']
        assert not any(binding[1] == 'guidance_scale' for binding in spec['bindings'])


@pytest.mark.skipif(importlib.util.find_spec('transformers') is None,
                    reason='Optional model runtime not active')
def test_real_kv_actions_repeat_changed_reference_multireference_and_cache_cleanup(monkeypatch):
    import torch
    import numpy as np
    from PIL import Image
    from diffusers import (AutoencoderKLFlux2, Flux2Transformer2DModel,
                           FlowMatchEulerDiscreteScheduler, Flux2KleinKVPipeline)
    from modules.DiffusersImage import main as nodes
    from modiff.model_artifact_catalog import catalog_revision
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(42)
        transformer = Flux2Transformer2DModel(in_channels=8, out_channels=8, num_layers=1,
            num_single_layers=1, attention_head_dim=16, num_attention_heads=1,
            joint_attention_dim=8, axes_dims_rope=(4,4,4,4), guidance_embeds=False)
        vae = AutoencoderKLFlux2(block_out_channels=(32,32,32,32), layers_per_block=1,
                                latent_channels=2).eval()
        pipeline = Flux2KleinKVPipeline(transformer=transformer, vae=vae, text_encoder=None,
            tokenizer=None, scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=False))
        pipeline.set_progress_bar_config(disable=True)
        embeds = torch.randn((1,3,8))
        original_encode = pipeline.encode_prompt
        def encode(*args, **kwargs):
            kwargs['prompt'] = None
            kwargs['prompt_embeds'] = embeds
            return original_encode(**kwargs)
        monkeypatch.setattr(pipeline, 'encode_prompt', encode)
        calls, caches = [], []
        forward = transformer.forward
        def observe(*args, **kwargs):
            calls.append(kwargs.get('kv_cache_mode', 'standard'))
            result = forward(*args, **kwargs)
            if kwargs.get('kv_cache_mode') == 'extract':
                caches.append(result[1])
            return result
        monkeypatch.setattr(transformer, 'forward', observe)
        adapter = nodes.IMAGE_PIPELINE_ADAPTERS['Flux2KleinKVPipeline']
        nodes._tag_image_pipeline(pipeline, adapter, 'text_to_image', adapter.default_repo,
                                  'hub', catalog_revision(adapter.default_repo))
        generate, edit = nodes.Generate('kv-generate'), nodes.Edit('kv-edit')
        generate.progress = edit.progress = Mock()
        base = dict(pipeline=pipeline, prompt='a geometric reference study', width=64,
            height=64, seed=99, num_inference_steps=3, guidance_scale=0., output_type='pil')
        def run(images=None):
            calls.clear()
            node = generate if images is None else edit
            mode = ('text_to_image' if images is None else
                    'multi_image_reference_edit' if isinstance(images, list) else 'edit_image')
            nodes._tag_image_pipeline(pipeline, adapter, mode, adapter.default_repo,
                                     'hub', catalog_revision(adapter.default_repo))
            result = node.execute(**base, **({} if images is None else {'image':images}))
            assert calls == (['standard']*3 if images is None else ['extract','cached','cached'])
            assert node._active_pipeline is None
            for cache in caches:
                assert cache.num_ref_tokens == 0
                assert all(layer.k_ref is None and layer.v_ref is None for layer in
                           [*cache.double_block_caches, *cache.single_block_caches])
            return np.asarray(result['images'][0])
        unconditional = run()
        red, blue = Image.new('RGB',(64,64),'red'), Image.new('RGB',(64,64),'blue')
        reference = run(red)
        np.testing.assert_array_equal(reference, run(red))
        assert not np.array_equal(reference, run(blue))
        assert not np.array_equal(reference, run([red,blue]))
        np.testing.assert_array_equal(unconditional, run())
        assert len({id(cache) for cache in caches}) == 4
    finally:
        torch.set_num_threads(old_threads)
