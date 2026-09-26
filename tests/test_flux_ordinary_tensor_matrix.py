"""Actual ordinary FLUX transformer/VAE calls through normal node dispatch.

Small seeded CPU components test execution, not pretrained visual quality. Text
embeddings are supplied through the public tensor sockets; nothing is downloaded.
"""
import importlib.util
from functools import wraps
from unittest.mock import Mock

import pytest


CASES = (
    ('FluxPipeline', 'Generate', 'text_to_image'),
    ('FluxImg2ImgPipeline', 'Edit', 'edit_image'),
    ('FluxInpaintPipeline', 'Inpaint', 'inpaint'),
    ('FluxKontextPipeline', 'Edit', 'edit_image'),
    ('FluxKontextInpaintPipeline', 'Inpaint', 'inpaint'),
    ('FluxFillPipeline', 'Inpaint', 'inpaint'),
    ('FluxControlPipeline', 'ControlGenerate', 'control_image'),
    ('FluxControlImg2ImgPipeline', 'ControlEdit', 'control_edit_image'),
    ('FluxControlInpaintPipeline', 'ControlInpaint', 'control_inpaint'),
    ('Flux2Pipeline', 'Generate', 'text_to_image'),
    ('Flux2KleinPipeline', 'Edit', 'edit_image'),
    ('Flux2KleinInpaintPipeline', 'Inpaint', 'inpaint'),
)


@pytest.mark.skipif(importlib.util.find_spec('transformers') is None, reason='Optional runtime inactive')
@pytest.mark.parametrize('name,action,mode', CASES)
@pytest.mark.parametrize('masked_attention', [False, True])
def test_ordinary_flux_dispatch_matches_real_upstream_tensor_output(monkeypatch, name, action, mode, masked_attention):
    import diffusers
    import torch
    from PIL import Image, ImageDraw
    from modules.DiffusersImage import main as nodes
    from modiff.model_artifact_catalog import catalog_revision

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(715)
        flux2 = name.startswith('Flux2')
        vae_args = dict(block_out_channels=(32,)*4, layers_per_block=1, latent_channels=2)
        if flux2:
            vae = diffusers.AutoencoderKLFlux2(**vae_args).eval()
            transformer = diffusers.Flux2Transformer2DModel(in_channels=8, out_channels=8,
                num_layers=1, num_single_layers=1, attention_head_dim=16, num_attention_heads=1,
                joint_attention_dim=8, axes_dims_rope=(4,4,4,4), guidance_embeds=name == 'Flux2Pipeline')
        else:
            vae = diffusers.AutoencoderKL(**vae_args, down_block_types=('DownEncoderBlock2D',)*4,
                up_block_types=('UpDecoderBlock2D',)*4, shift_factor=0.).eval()
            channels = 272 if name == 'FluxFillPipeline' else 16 if name.startswith('FluxControl') else 8
            transformer = diffusers.FluxTransformer2DModel(in_channels=channels, out_channels=8,
                num_layers=1, num_single_layers=1, attention_head_dim=16, num_attention_heads=1,
                joint_attention_dim=8, pooled_projection_dim=8, axes_dims_rope=(4,6,6), guidance_embeds=True)
        components = dict(vae=vae, transformer=transformer, text_encoder=None, tokenizer=None,
            scheduler=diffusers.FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=False))
        if not flux2:
            components.update(text_encoder_2=None, tokenizer_2=None)
        pipeline = getattr(diffusers, name)(**components)
        pipeline.set_progress_bar_config(disable=True)
        adapter = nodes.IMAGE_PIPELINE_ADAPTERS[name]
        nodes._tag_image_pipeline(pipeline, adapter, mode, adapter.default_repo, 'hub',
                                 catalog_revision(adapter.default_repo))
        source = Image.new('RGB', (64,64), 'navy')
        mask = Image.new('L', (64,64))
        ImageDraw.Draw(mask).rectangle((16,16,48,48), fill=255)
        kwargs = dict(pipeline=pipeline, prompt='authored text remains stored', width=64, height=64,
            num_inference_steps=3, guidance_scale=1., output_type='pt',
            generator=torch.Generator().manual_seed(19), prompt_embeds=torch.randn(1,3,8))
        if not flux2:
            kwargs['pooled_prompt_embeds'] = torch.randn(1,8)
        if action != 'Generate':
            kwargs.update(image=source, strength=1.)
        if 'Inpaint' in action:
            kwargs['mask_image'] = mask
        if action.startswith('Control'):
            kwargs['control_image'] = source
        if 'Kontext' in name:
            kwargs.update(_auto_resize=False, max_area=4096)
        attention_key = 'attention_kwargs' if flux2 else 'joint_attention_kwargs'
        attention_mask = torch.zeros(1, 1, 1, 1, dtype=torch.bool)
        if masked_attention:
            kwargs[attention_key] = {'attention_mask': attention_mask}
        captured = []
        original = type(pipeline).__call__

        @wraps(original)
        def observe(self, *args, **values):
            captured.append((dict(values), values['generator'].get_state().clone()))
            return original(self, *args, **values)

        monkeypatch.setattr(type(pipeline), '__call__', observe)
        node = getattr(nodes, action)(f'ordinary-real-{name}')
        node.progress = Mock()
        result = node(**kwargs)
        submitted, generator_state = captured[-1]
        assert submitted['prompt'] is None
        assert submitted['prompt_embeds'] is kwargs['prompt_embeds']
        assert submitted['output_type'] == 'pt'
        assert submitted['num_inference_steps'] == 3
        reference = dict(submitted, generator=torch.Generator().set_state(generator_state))
        reference.pop('callback_on_step_end', None)
        expected = original(pipeline, **reference).images
        torch.testing.assert_close(result['images'], expected, rtol=0, atol=0)
        if masked_attention:
            assert submitted[attention_key]['attention_mask'] is attention_mask
            unmasked = dict(reference, generator=torch.Generator().set_state(generator_state))
            unmasked.pop(attention_key)
            without_mask = original(pipeline, **unmasked).images
            assert not torch.equal(expected, without_mask), name
            assert not attention_mask.any()
        assert (result['width_out'], result['height_out']) == (64,64)
        assert kwargs['prompt'] == 'authored text remains stored'
        assert node._active_pipeline is None
    finally:
        torch.set_num_threads(threads)
