"""Explicit latent outputs retain upstream tensors, never masquerade as images."""
import ast
import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from modules.DiffusersImage import main as nodes
from modules.DiffusersImage.call_inputs import PIPELINE_CALL_INPUTS


FLUX_CLASSES = tuple(name for name in PIPELINE_CALL_INPUTS if name != 'FluxReduxPipeline')


def fixture_pipeline(name):
    calls = []
    latent = torch.randn(1, 4, 8)

    def call(self, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(images=latent if kwargs['output_type'] == 'latent'
                               else [Image.new('RGB', (64, 64))])

    pipeline = type(name, (), {'__call__': call, 'device': 'cpu', '_execution_device': 'cpu'})()
    return pipeline, latent, calls


@pytest.mark.skipif(importlib.util.find_spec('transformers') is None, reason='Optional runtime inactive')
@pytest.mark.parametrize('family', ['flux1', 'flux2-kv'])
def test_explicit_decode_matches_upstream_pixels_without_reinterpreting_latents(family):
    from diffusers import (AutoencoderKL, AutoencoderKLFlux2, FluxPipeline,
        Flux2KleinKVPipeline, FluxTransformer2DModel, Flux2Transformer2DModel,
        FlowMatchEulerDiscreteScheduler)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(91)
        common = dict(text_encoder=None, tokenizer=None,
            scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=False))
        if family == 'flux1':
            pipeline = FluxPipeline(**common, text_encoder_2=None, tokenizer_2=None,
                transformer=FluxTransformer2DModel(in_channels=8, num_layers=1, num_single_layers=1,
                    attention_head_dim=16, num_attention_heads=1, joint_attention_dim=8,
                    pooled_projection_dim=8, axes_dims_rope=(4, 6, 6), guidance_embeds=True),
                vae=AutoencoderKL(block_out_channels=(32,)*4, layers_per_block=1, latent_channels=2,
                    down_block_types=('DownEncoderBlock2D',)*4, up_block_types=('UpDecoderBlock2D',)*4,
                    shift_factor=0.).eval())
            kwargs = dict(prompt_embeds=torch.randn(1,3,8), pooled_prompt_embeds=torch.randn(1,8))
        else:
            pipeline = Flux2KleinKVPipeline(**common,
                transformer=Flux2Transformer2DModel(in_channels=8, out_channels=8,
                    num_layers=1, num_single_layers=1, attention_head_dim=16,
                    num_attention_heads=1, joint_attention_dim=8,
                    axes_dims_rope=(4,4,4,4), guidance_embeds=False),
                vae=AutoencoderKLFlux2(block_out_channels=(32,)*4, layers_per_block=1,
                    latent_channels=2).eval())
            kwargs = dict(prompt_embeds=torch.randn(1,3,8))
        pipeline.set_progress_bar_config(disable=True)
        args = dict(**kwargs, height=64, width=64, num_inference_steps=3)
        latent = pipeline(**args, generator=torch.Generator().manual_seed(5), output_type='latent').images
        original = latent.clone()
        expected = pipeline(**args, generator=torch.Generator().manual_seed(5), output_type='pt').images
        decoded = nodes.DecodeLatents('explicit-decode')(pipeline=pipeline, latents=latent,
            height=64, width=64, output_type='pt')
        torch.testing.assert_close(decoded['images'], expected, rtol=0, atol=0)
        torch.testing.assert_close(latent, original, rtol=0, atol=0)
        assert decoded['width_out'] == decoded['height_out'] == 64
        wrong = torch.zeros(1,7,8) if family == 'flux1' else torch.zeros(1,7,8,8)
        with pytest.raises(RuntimeError, match='Latents.*layout'):
            nodes.DecodeLatents('wrong-layout')(pipeline=pipeline, latents=wrong, height=64, width=64)
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize('name', FLUX_CLASSES)
def test_every_reviewed_flux_call_really_supports_latent_output(name):
    root = Path(importlib.util.find_spec('diffusers').origin).parent / 'pipelines'
    matches = []
    for path in root.glob('flux*/pipeline_flux*.py'):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef) and node.name == name:
                method = next(member for member in node.body if isinstance(member, ast.FunctionDef)
                              and member.name == '__call__')
                matches.append(method)
    method, = matches
    assert 'output_type' in {arg.arg for arg in method.args.args}
    assert any(isinstance(node, ast.Constant) and node.value == 'latent' for node in ast.walk(method))


@pytest.mark.parametrize('name', FLUX_CLASSES)
def test_latent_output_is_typed_and_scoped_in_selected_contract(name):
    adapter = nodes.IMAGE_PIPELINE_ADAPTERS[name]
    for mode in adapter.mode_options:
        contract = nodes.image_pipeline_contract(adapter, mode)
        assert 'latent' in contract['fieldParams']['output_type']['options']
        assert contract['fieldParams']['latents_out']['hidden'] is False
    assert nodes.Generate.params['latents_out']['type'] == 'tensor'
    old = deepcopy(contract)
    old['fieldParams'].pop('control_mode', None)
    from modules.DiffusersImage.image_prompt_adapter import IP_ADAPTER_INPUTS
    for key in IP_ADAPTER_INPUTS:
        old['fieldParams'].pop(key, None)
    del old['fieldParams']['output_type']
    del old['fieldParams']['latents_out']
    assert nodes._compatible_image_contract(old, contract)
    old['mode'] = 'unreviewed_mode'
    assert not nodes._compatible_image_contract(old, contract)


@pytest.mark.parametrize('name,action', [
    ('FluxPipeline', 'Generate'), ('FluxImg2ImgPipeline', 'Edit'),
    ('FluxInpaintPipeline', 'Inpaint'), ('FluxKontextPipeline', 'Edit'),
    ('FluxKontextInpaintPipeline', 'Inpaint'), ('FluxFillPipeline', 'Inpaint'),
    ('FluxControlPipeline', 'ControlGenerate'), ('FluxControlImg2ImgPipeline', 'ControlEdit'),
    ('FluxControlInpaintPipeline', 'ControlInpaint'), ('FluxControlNetPipeline', 'ControlGenerate'),
    ('FluxControlNetImg2ImgPipeline', 'ControlEdit'), ('FluxControlNetInpaintPipeline', 'ControlInpaint'),
    ('Flux2Pipeline', 'Generate'), ('Flux2KleinPipeline', 'Generate'),
    ('Flux2KleinInpaintPipeline', 'Inpaint'), ('Flux2KleinKVPipeline', 'Generate'),
])
def test_dispatch_routes_latent_tensor_without_decoding_or_pil_compositing(name, action):
    pipeline, latent, calls = fixture_pipeline(name)
    from modiff.model_artifact_catalog import catalog_revision
    adapter = nodes.IMAGE_PIPELINE_ADAPTERS[name]
    mode = next(mode for mode in adapter.mode_options if mode in nodes.IMAGE_ACTION_MODES[action])
    nodes._tag_image_pipeline(pipeline, adapter, mode, adapter.default_repo,
                             'hub', catalog_revision(adapter.default_repo),
                             conditioning_repo=adapter.default_conditioning_repo,
                             conditioning_revision=catalog_revision(adapter.default_conditioning_repo)
                             if adapter.default_conditioning_repo else None)
    node = getattr(nodes, action)(f'latent-{name}')
    node.progress = Mock()
    values = dict(pipeline=pipeline, prompt='preserved prompt', width=64, height=64,
                  num_inference_steps=2, output_type='latent')
    if action != 'Generate':
        values.update(image=Image.new('RGB', (64, 64)), mask_image=Image.new('L', (64, 64), 255),
                      control_image=Image.new('RGB', (64, 64)))
    result = node(**values)
    assert calls[-1]['output_type'] == 'latent'
    assert result['latents_out'] is latent
    assert result['images'] is None
    assert result['width_out'] is None and result['height_out'] is None
    assert node._active_pipeline is None


def test_latent_crop_and_unreviewed_pipeline_fail_before_execution():
    pipeline, _, calls = fixture_pipeline('FluxInpaintPipeline')
    image = Image.new('RGB', (64, 64))
    with pytest.raises(ValueError, match='padding_mask_crop'):
        nodes.Inpaint('latent-crop')(pipeline=pipeline, image=image,
            mask_image=Image.new('L', (64, 64)), output_type='latent', padding_mask_crop=8)
    assert calls == []
    pipeline, _, calls = fixture_pipeline('QwenImagePipeline')
    with pytest.raises(ValueError, match='output_type'):
        nodes.Generate('latent-unsupported')(pipeline=pipeline, output_type='latent')
    assert calls == []


def test_default_image_outputs_and_creator_options_are_not_rewritten():
    defaults = deepcopy(nodes.Generate.params)
    pipeline, _, _ = fixture_pipeline('FluxPipeline')
    result = nodes.Generate('legacy-image-output')(pipeline=pipeline, width=64, height=64)
    assert isinstance(result['images'][0], Image.Image)
    assert result['latents_out'] is None
    assert nodes.Generate.params['output_type']['default'] == 'pil'
    assert nodes.Generate.params == defaults


def test_compiled_action_has_visible_tensor_output_and_exact_output_options():
    from modiff.studio_execution_specs import _execution_spec_role_params
    params = _execution_spec_role_params(
        {'pipelineClass': 'FluxControlNetInpaintPipeline', 'mode': 'control_inpaint',
         'executionPath': 'diffusers-image'},
        'modules.DiffusersImage.ControlInpaint', {'params': nodes.ControlInpaint.params})
    assert params['latents_out']['type'] == 'tensor'
    assert params['latents_out']['hidden'] is False
    assert params['output_type']['options'] == ['pil', 'np', 'pt', 'latent']
    assert params['output_type']['default'] == 'pil'


@pytest.mark.parametrize('output_type', ['np', 'pt'])
def test_explicit_decoded_tensor_and_array_outputs_do_not_require_pil_compositing(output_type):
    pipeline, _, _ = fixture_pipeline('FluxInpaintPipeline')
    data = torch.zeros(1, 3, 64, 64) if output_type == 'pt' else torch.zeros(1, 64, 64, 3).numpy()
    type(pipeline).__call__ = lambda self, **kwargs: SimpleNamespace(images=data)
    image = Image.new('RGB', (64, 64))
    result = nodes.Inpaint('array-inpaint')(pipeline=pipeline, image=image,
        mask_image=Image.new('L', (64, 64), 255), output_type=output_type)
    assert result['images'] is data
    assert result['latents_out'] is None
    assert result['width_out'] == result['height_out'] == 64


@pytest.mark.skipif(importlib.util.find_spec('transformers') is None, reason='Optional runtime inactive')
def test_actual_small_flux_dispatch_latents_match_upstream_and_feed_another_call():
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(37)
        pipeline = FluxPipeline(
            transformer=FluxTransformer2DModel(in_channels=8, num_layers=1, num_single_layers=1,
                attention_head_dim=16, num_attention_heads=1, joint_attention_dim=8,
                pooled_projection_dim=8, axes_dims_rope=(4, 6, 6), guidance_embeds=True),
            vae=AutoencoderKL(block_out_channels=(32,)*4, layers_per_block=1, latent_channels=2,
                down_block_types=('DownEncoderBlock2D',)*4, up_block_types=('UpDecoderBlock2D',)*4,
                shift_factor=0.).eval(),
            text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None,
            scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=False),
        )
        pipeline.set_progress_bar_config(disable=True)
        embeds, pooled, initial = torch.randn(1, 3, 8), torch.randn(1, 8), torch.randn(1, 16, 8)
        values = dict(pipeline=pipeline, prompt='retained description', prompt_embeds=embeds,
            pooled_prompt_embeds=pooled, latents=initial, width=64, height=64,
            num_inference_steps=3, sigmas=[1., .6, .2], guidance_scale=1., output_type='latent')
        node = nodes.Generate('real-latent-dispatch')
        node.progress = Mock()
        actual = node(**values)
        direct = pipeline(prompt=None, prompt_embeds=embeds, pooled_prompt_embeds=pooled,
            latents=initial, width=64, height=64, num_inference_steps=3, sigmas=[1., .6, .2],
            true_cfg_scale=1., output_type='latent').images
        torch.testing.assert_close(actual['latents_out'], direct, rtol=0, atol=0)
        next_node = nodes.Generate('reuse-latent-dispatch')
        next_node.progress = Mock()
        reused = next_node(**{**values, 'latents': actual['latents_out']})
        expected = pipeline(prompt=None, prompt_embeds=embeds, pooled_prompt_embeds=pooled,
            latents=direct, width=64, height=64, num_inference_steps=3, sigmas=[1., .6, .2],
            true_cfg_scale=1., output_type='latent').images
        torch.testing.assert_close(reused['latents_out'], expected, rtol=0, atol=0)
    finally:
        torch.set_num_threads(threads)
