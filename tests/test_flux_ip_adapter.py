"""Pinned ordinary image-prompt adapters use upstream loaders, not hidden mutation."""
import importlib.util
from types import SimpleNamespace

import pytest


def test_configuration_is_immutable_bounded_and_never_downloads():
    from modules.DiffusersImage.image_prompt_adapter import image_prompt_adapter_config
    values = dict(adapter_model={'source': 'hub', 'value': 'XLabs-AI/flux-ip-adapter'},
        revision='a'*40, weight_name='adapter.safetensors', expected_sha256='b'*64,
        image_encoder_model={'source': 'hub', 'value': 'openai/clip-vit-large-patch14'},
        image_encoder_revision='c'*40, image_encoder_sha256='d'*64, scale=.6)
    config = image_prompt_adapter_config(values)
    assert config[0].scale == .6
    assert len(image_prompt_adapter_config({**values, 'previous': config})) == 2
    for field, invalid in [('revision', 'main'), ('expected_sha256', ''),
            ('weight_name', '../weights.safetensors'), ('weight_name', 'weights.bin'),
            ('image_encoder_revision', ''), ('scale', float('nan')),
            ('scale', True), ('previous', [{'repository': 'fake'}])]:
        with pytest.raises((ValueError, TypeError), match=field):
            image_prompt_adapter_config({**values, field: invalid})
    with pytest.raises(ValueError, match='4'):
        image_prompt_adapter_config({**values, 'previous': config * 4})
    with pytest.raises(ValueError, match='encoder'):
        image_prompt_adapter_config({**values, 'previous': config,
                                     'image_encoder_revision': 'e'*40})


def test_unsupported_pipeline_rejected_before_any_weight_or_encoder_loading():
    from modules.DiffusersImage.image_prompt_adapter import validate_image_prompt_adapters
    for pipeline in ('Flux2Pipeline', 'FluxControlNetImg2ImgPipeline', 'QwenImagePipeline'):
        with pytest.raises(ValueError, match=pipeline):
            validate_image_prompt_adapters(pipeline, (object(),))


def test_missing_adapter_has_precise_local_install_error(monkeypatch):
    from modules.DiffusersImage import image_prompt_adapter as adapters
    monkeypatch.setattr(adapters, 'cached_file_path', lambda *args, **kwargs: None)
    config = adapters.image_prompt_adapter_config(dict(
        adapter_model={'source': 'hub', 'value': 'example/adapter'}, revision='a'*40,
        weight_name='adapter.safetensors', expected_sha256='b'*64,
        image_encoder_model={'source': 'hub', 'value': 'example/encoder'},
        image_encoder_revision='c'*40, image_encoder_sha256='d'*64))
    with pytest.raises(FileNotFoundError, match='Model Manager'):
        adapters.resolve_image_prompt_adapter_files(config)


@pytest.mark.skipif(importlib.util.find_spec('transformers') is None,
                    reason='Optional model runtime inactive')
@pytest.mark.parametrize('adapter_count', [1, 2])
def test_real_flux_adapter_scales_influence_pixels_and_keep_pipeline_owner(monkeypatch, adapter_count):
    from copy import deepcopy
    import torch
    import diffusers
    from modules.DiffusersImage import image_prompt_adapter as adapters
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(217)
        pipeline = diffusers.FluxPipeline(
            transformer=diffusers.FluxTransformer2DModel(in_channels=8, num_layers=2,
                num_single_layers=1, attention_head_dim=16, num_attention_heads=1,
                joint_attention_dim=8, pooled_projection_dim=8, axes_dims_rope=(4,6,6),
                guidance_embeds=True),
            vae=diffusers.AutoencoderKL(block_out_channels=(32,)*4, layers_per_block=1,
                latent_channels=2, down_block_types=('DownEncoderBlock2D',)*4,
                up_block_types=('UpDecoderBlock2D',)*4, shift_factor=0.),
            text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None,
            scheduler=diffusers.FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=False))
        pipeline.set_progress_bar_config(disable=True)
        # Genuine upstream-format small adapter: no fixture model downloads.
        state = {'image_proj': {'proj.weight': torch.randn(32,8)*.1,
            'proj.bias': torch.randn(32)*.1, 'norm.weight': torch.ones(8), 'norm.bias': torch.zeros(8)},
            'ip_adapter': {'0.to_k_ip.weight': torch.randn(16,8)*.1,
                           '0.to_v_ip.weight': torch.randn(16,8)*.1,
                           '0.to_k_ip.bias': torch.zeros(16), '0.to_v_ip.bias': torch.zeros(16)}}
        for key, value in list(state['ip_adapter'].items()):
            state['ip_adapter'][key.replace('0.', '1.')] = value.clone()
        config = adapters.image_prompt_adapter_config(dict(
            adapter_model={'source':'hub','value':'example/adapter'}, revision='a'*40,
            weight_name='adapter.safetensors', expected_sha256='b'*64, scale=.7))
        config = config * adapter_count
        monkeypatch.setattr(adapters, 'resolve_image_prompt_adapter_files',
            lambda value: ([state] * len(value), None))
        original_transformer = pipeline.transformer
        untouched = deepcopy(pipeline)
        adapters.attach_image_prompt_adapters(pipeline, 'FluxPipeline', config,
                                              dtype=torch.float32)
        assert pipeline.transformer is original_transformer
        assert pipeline.transformer.encoder_hid_proj.num_ip_adapters == adapter_count
        args = dict(prompt_embeds=torch.randn(1,3,8), pooled_prompt_embeds=torch.randn(1,8),
            ip_adapter_image_embeds=[torch.randn(1,1,8) for _ in range(adapter_count)],
            negative_ip_adapter_image_embeds=[torch.randn(1,1,8) for _ in range(adapter_count)], width=64,height=64,
            num_inference_steps=3,output_type='latent')
        first = pipeline(**args,generator=torch.Generator().manual_seed(12)).images
        pipeline.set_ip_adapter_scale(0.)
        without = pipeline(**args,generator=torch.Generator().manual_seed(12)).images
        assert not torch.equal(first, without)
        pipeline.set_ip_adapter_scale(.7)
        repeat = pipeline(**args,generator=torch.Generator().manual_seed(12)).images
        torch.testing.assert_close(first, repeat, rtol=0,atol=0)
        from modules.DiffusersImage.main import Generate
        actual = Generate('ip-adapter-real-dispatch')(pipeline=pipeline, **args, seed=12)['latents_out']
        torch.testing.assert_close(first, actual, rtol=0, atol=0)
        mask = torch.zeros(1, 1, 1, 1, dtype=torch.bool)
        masked = Generate('ip-adapter-masked-dispatch')(
            pipeline=pipeline, **args, seed=12,
            joint_attention_kwargs={'attention_mask': mask})['latents_out']
        expected_masked = pipeline(**args, generator=torch.Generator().manual_seed(12),
            joint_attention_kwargs={'attention_mask': mask}).images
        torch.testing.assert_close(masked, expected_masked, rtol=0, atol=0)
        assert not torch.equal(masked, actual)
        # A second owner must never share this transformer's adapter processors.
        with pytest.raises(ValueError, match='already'):
            adapters.attach_image_prompt_adapters(pipeline, 'FluxPipeline', config,
                                                  dtype=torch.float32)
        processors_before = dict(untouched.transformer.attn_processors)
        config_before = untouched.config
        transformer_config_before = untouched.transformer.config
        original_load = untouched.load_ip_adapter
        def fail_after_attachment(*args, **kwargs):
            original_load(*args, **kwargs)
            raise RuntimeError('deliberate adapter failure')
        monkeypatch.setattr(untouched, 'load_ip_adapter', fail_after_attachment)
        with pytest.raises(RuntimeError, match='deliberate'):
            adapters.attach_image_prompt_adapters(untouched, 'FluxPipeline', config, dtype=torch.float32)
        assert untouched.transformer.encoder_hid_proj is None
        assert untouched.config == config_before
        assert untouched.transformer.config == transformer_config_before
        assert all(untouched.transformer.attn_processors[key] is value for key,value in processors_before.items())
        assert not getattr(untouched, '_modiff_image_prompt_adapters', None)
    finally:
        torch.set_num_threads(threads)


def test_call_inputs_validate_paired_media_and_active_adapter():
    from modules.DiffusersImage.image_prompt_adapter import validate_image_prompt_inputs
    pipeline = SimpleNamespace(transformer=SimpleNamespace(encoder_hid_proj=None))
    with pytest.raises(ValueError, match='Image Prompt Adapter'):
        validate_image_prompt_inputs(pipeline, {'ip_adapter_image': object()})
    assert validate_image_prompt_inputs(pipeline, {}) is None
    pipeline.transformer.encoder_hid_proj = SimpleNamespace(num_ip_adapters=1)
    with pytest.raises(ValueError, match='reference'):
        validate_image_prompt_inputs(pipeline, {})
    with pytest.raises(ValueError, match='either'):
        validate_image_prompt_inputs(pipeline, {'ip_adapter_image': object(),
            'ip_adapter_image_embeds': object()})
    with pytest.raises(ValueError, match='encoder'):
        validate_image_prompt_inputs(pipeline, {'ip_adapter_image': object()})


def test_configuration_node_dispatch_preserves_defaults_and_chained_values():
    from copy import deepcopy
    from modules.DiffusersImage.main import ImagePromptAdapter
    defaults = deepcopy(ImagePromptAdapter.params)
    first = ImagePromptAdapter('image-adapter-config')
    result = first(scale='0.35')
    assert result['adapters'][0].scale == .35
    second = ImagePromptAdapter('second-image-adapter')
    chained = second(previous=result['adapters'], layer_scales=[.2, .8])
    assert len(chained['adapters']) == 2
    assert chained['adapters'][1].scale == (.2, .8)
    assert ImagePromptAdapter.params == defaults
    assert first(scale='0.35')['adapters'] is result['adapters']
    with pytest.raises(ValueError, match='scale'):
        first(scale=True)


def test_adapter_provenance_does_not_replace_base_model_revision():
    from modiff.execution_input_provenance import capture_generation_inputs, build_resolved_execution_inputs
    graph = {'nodes': {
        'adapter': {'module': 'modules.DiffusersImage', 'action': 'ImagePromptAdapter', 'params': {}},
        'loader': {'module': 'modules.DiffusersImage', 'action': 'LoadPipeline',
                   'params': {'image_prompt_adapter': {'sourceId': 'adapter', 'sourceKey': 'adapters'}}}}}
    records = {
        'adapter': capture_generation_inputs('adapter', graph['nodes']['adapter'], {
            'revision': 'a'*40, 'adapter_model': {'source':'hub','value':'example/adapter'},
            'image_encoder_model': {'source':'hub','value':'example/encoder'},
            'image_encoder_revision': 'c'*40, 'expected_sha256': 'b'*64, 'scale': .6,
            'token': 'never recorded'}),
        'loader': capture_generation_inputs('loader', graph['nodes']['loader'], {
            'revision': 'd'*40, 'model_id': 'example/base', 'pipeline_class': 'FluxPipeline'})}
    receipt = build_resolved_execution_inputs(graph, records, task_id='run', attempt_index=0, node_id='loader')
    assert receipt['summary']['revision'] == 'd'*40
    assert receipt['summary']['repo'] == 'example/base'
    assert receipt['summary']['ipAdapterRevision'] == 'a'*40
    assert receipt['summary']['ipAdapterRepo'] == 'example/adapter'
    assert receipt['summary']['ipAdapterEncoderRepo'] == 'example/encoder'
    assert not receipt['ambiguousFields']
    assert 'token' not in records['adapter']['fields']
