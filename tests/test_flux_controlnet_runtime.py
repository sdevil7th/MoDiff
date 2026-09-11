"""Actual tiny CPU FLUX ControlNet through MoDiff actions; no model downloads."""
import pytest
import importlib.util
from unittest.mock import Mock


@pytest.mark.skipif(importlib.util.find_spec('transformers') is None,
                    reason='Optional model runtime not active')
@pytest.mark.parametrize('pipeline_name,action_name,mode', [
    ('FluxControlNetPipeline', 'ControlGenerate', 'control_image'),
    ('FluxControlNetImg2ImgPipeline', 'ControlEdit', 'control_edit_image'),
    ('FluxControlNetInpaintPipeline', 'ControlInpaint', 'control_inpaint'),
])
@pytest.mark.parametrize('composition', ['single', 'multiple', 'union', 'union_multiple'])
def test_real_controlnet_actions_repeat_and_preserve_distinct_image_mask_inputs(monkeypatch, pipeline_name, action_name, mode, composition):
    import torch
    import numpy as np
    from PIL import Image, ImageDraw
    import diffusers
    from modules.DiffusersImage import main as image_nodes
    from modiff.model_artifact_catalog import catalog_revision

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(42)
        common = dict(in_channels=8, num_layers=1, num_single_layers=1,
            attention_head_dim=16, num_attention_heads=1, joint_attention_dim=8,
            pooled_projection_dim=8, axes_dims_rope=(4,6,6), guidance_embeds=True)
        transformer = diffusers.FluxTransformer2DModel(**common)
        controlnet = diffusers.FluxControlNetModel(**common,
            **({'num_mode': 3} if composition.startswith('union') else {}))
        # Upstream initializes residual heads to zero; nonzero weights exercise
        # actual conditioning propagation rather than a disconnected no-op.
        for head in [*controlnet.controlnet_blocks, *controlnet.controlnet_single_blocks]:
            torch.nn.init.normal_(head.weight, std=.1)
        if composition == 'multiple':
            second = diffusers.FluxControlNetModel(**common)
            for head in [*second.controlnet_blocks, *second.controlnet_single_blocks]:
                torch.nn.init.normal_(head.weight, std=.1)
            controlnet = [controlnet, second]
        elif composition == 'union_multiple':
            controlnet = [controlnet]
        vae = diffusers.AutoencoderKL(block_out_channels=(32,32,32,32), layers_per_block=1,
            latent_channels=2, down_block_types=('DownEncoderBlock2D',)*4,
            up_block_types=('UpDecoderBlock2D',)*4, shift_factor=0.).eval()
        pipeline = getattr(diffusers, pipeline_name)(transformer=transformer, controlnet=controlnet, vae=vae,
            text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None,
            scheduler=diffusers.FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=False))
        pipeline.set_progress_bar_config(disable=True)
        prompt_embeds, pooled = torch.randn((1,3,8)), torch.randn((1,8))
        # Fix only prompt embeddings; VAE, scheduler, transformer and ControlNet
        # execute normally in each call. This is not a text-encoder quality test.
        monkeypatch.setattr(pipeline, 'encode_prompt', lambda *args, **kwargs:
                            (prompt_embeds, pooled, torch.zeros((3,3))))
        adapter = image_nodes.IMAGE_PIPELINE_ADAPTERS[pipeline_name]
        image_nodes._tag_image_pipeline(pipeline, adapter, mode, image_nodes.FLUX_DEV_REPO, 'hub',
            catalog_revision(image_nodes.FLUX_DEV_REPO), conditioning_repo=adapter.default_conditioning_repo,
            conditioning_revision=catalog_revision(adapter.default_conditioning_repo))
        source = Image.new('RGB', (64,64), 'navy')
        control = Image.new('RGB', (64,64), 'black')
        ImageDraw.Draw(control).rectangle((16,16,48,48), outline='white', width=3)
        mask = Image.new('RGB', (64,64), 'black')
        ImageDraw.Draw(mask).rectangle((20,20,44,44), fill='white')
        node = getattr(image_nodes, action_name)('tiny-controlnet')
        node.progress = Mock()
        values = dict(pipeline=pipeline, prompt='a controlled geometric object',
            control_image=control, width=64, height=64, seed=99,
            num_inference_steps=3, guidance_scale=3.5, conditioning_scale=.8,
            control_guidance_start=0., control_guidance_end=1., output_type='pil')
        if action_name != 'ControlGenerate':
            values.update(image=source, strength=1.)
        if action_name == 'ControlInpaint':
            values['mask_image'] = mask
        if composition in {'multiple', 'union_multiple'}:
            values.update(control_image=[control, source], conditioning_scale=[.8, .3],
                          control_guidance_start=[0., .25], control_guidance_end=[1., .75])
        if composition.startswith('union'):
            values['control_mode'] = [0, 2] if composition == 'union_multiple' else 1
        first = node.execute(**values)
        repeat = node.execute(**values)
        pixels = np.asarray(first['images'][0])
        dispatched = node(**values)
        np.testing.assert_array_equal(pixels, np.asarray(dispatched['images'][0]))
        assert (first['width_out'], first['height_out']) == (64,64)
        np.testing.assert_array_equal(pixels, np.asarray(repeat['images'][0]))
        without_control = node.execute(**{**values, 'conditioning_scale':(
            [0., 0.] if composition in {'multiple', 'union_multiple'} else 0.)})
        assert not np.array_equal(pixels, np.asarray(without_control['images'][0]))
        if action_name == 'ControlInpaint':
            outside = np.asarray(mask)[:,:,0] == 0
            np.testing.assert_array_equal(pixels[outside], np.asarray(source)[outside])
        if composition == 'single':
            from functools import wraps
            attention_mask = torch.zeros(1, 1, 1, 1, dtype=torch.bool)
            original = type(pipeline).__call__
            calls = []
            @wraps(original)
            def capture(self, *args, **options):
                calls.append(dict(options))
                return original(self, *args, **options)
            monkeypatch.setattr(type(pipeline), '__call__', capture)
            masked = node.execute(**{**values, 'output_type': 'pt',
                'joint_attention_kwargs': {'attention_mask': attention_mask}})['images']
            actual_call = calls[-1]
            assert actual_call['joint_attention_kwargs']['attention_mask'] is attention_mask
            reference = dict(actual_call, generator=torch.Generator().manual_seed(99))
            reference.pop('callback_on_step_end', None)
            expected = original(pipeline, **reference).images
            torch.testing.assert_close(masked, expected, rtol=0, atol=0)
            reference.pop('joint_attention_kwargs')
            reference['generator'] = torch.Generator().manual_seed(99)
            unmasked = original(pipeline, **reference).images
            assert not torch.equal(masked, unmasked)
        assert node._active_pipeline is None
    finally:
        torch.set_num_threads(old_threads)
