"""Ordinary-pipeline optional inputs are explicit, typed and opt-in."""
import ast
import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize('action_name,pipeline_name', [
    ('Generate', 'FluxPipeline'), ('Edit', 'FluxImg2ImgPipeline'),
    ('Inpaint', 'FluxInpaintPipeline'), ('ControlGenerate', 'FluxControlNetPipeline'),
    ('ControlEdit', 'FluxControlNetImg2ImgPipeline'), ('ControlInpaint', 'FluxControlNetInpaintPipeline'),
])
@pytest.mark.parametrize('external', [False, True])
@pytest.mark.parametrize('output_type', ['pil', 'latent'])
def test_dispatch_receipt_matches_effective_image_call(monkeypatch, action_name, pipeline_name, external, output_type):
    import torch
    from PIL import Image
    from types import SimpleNamespace
    import modules.DiffusersImage.main as image_module
    from modiff.execution_input_provenance import bind_generation_input_origins

    calls = []

    class Pipeline:
        device = 'cpu'

        def __call__(self, *, prompt=None, negative_prompt=None, width=64, height=64,
                     true_cfg_scale=1., guidance_scale=3.5, **kwargs):
            calls.append(dict(prompt=prompt, true_cfg_scale=true_cfg_scale,
                              guidance_scale=guidance_scale, **kwargs))
            return SimpleNamespace(images=torch.zeros(1, 4, 8) if output_type == 'latent'
                                   else [Image.new('RGB', (width, height))])

    adapter = image_module.IMAGE_PIPELINE_ADAPTERS[pipeline_name]
    monkeypatch.setattr(image_module, 'validate_image_action', lambda *args: adapter)
    node = getattr(image_module, action_name)('receipt')
    values = dict(pipeline=Pipeline(), prompt='authored prompt', seed=7, width=64, height=64, output_type=output_type,
                  num_inference_steps=5, guidance_scale=4., use_guidance_scale_2=bool(adapter.secondary_guidance_parameter),
                  guidance_scale_2=2., image=Image.new('RGB', (64, 64)),
                  mask_image=Image.new('L', (64, 64), 255), control_image=Image.new('RGB', (64, 64)))
    if external:
        generator = torch.Generator().manual_seed(99)
        torch.rand(1, generator=generator)  # initial_seed is not sufficient replay evidence.
        values.update(generator=generator, prompt_embeds=torch.zeros(1, 2, 8),
                      pooled_prompt_embeds=torch.zeros(1, 8), sigmas=[1., .5, .1])
    result = node(**values)
    if output_type == 'latent':
        assert result['images'] is None and result['latents_out'].shape == (1, 4, 8)
    else:
        assert result['latents_out'] is None and result['images'][0].size == (64, 64)
    fields = node._execution_input_record['fields']
    assert fields['prompt']['value'] == (None if external else 'authored prompt')
    assert fields['seed']['value'] == (None if external else 7)
    assert fields['num_inference_steps']['value'] == (None if external else 5)
    assert fields[adapter.guidance_parameter]['value'] == calls[0][adapter.guidance_parameter]
    if adapter.secondary_guidance_parameter:
        assert fields[adapter.secondary_guidance_parameter]['value'] == calls[0][adapter.secondary_guidance_parameter]
    if external:
        assert calls[0]['generator'] is generator
        assert calls[0]['prompt_embeds'] is values['prompt_embeds']
    graph_node = {'module': 'DiffusersImage', 'action': action_name, 'params': {
        'guidance_scale': {'sourceId': 'primary', 'sourceKey': 'value'},
        'guidance_scale_2': {'sourceId': 'secondary', 'sourceKey': 'value'},
    }}
    bound = bind_generation_input_origins('receipt', graph_node, node._execution_input_record,
        source_fields=getattr(node, '_execution_input_source_fields', {}))
    assert bound['fields'][adapter.guidance_parameter]['sourceNodeId'] == 'primary'
    if adapter.secondary_guidance_parameter:
        assert bound['fields'][adapter.secondary_guidance_parameter]['sourceNodeId'] == 'secondary'
    assert values['prompt'] == 'authored prompt' and values['seed'] == 7
    node(**values)
    assert len(calls) == 1


def test_reviewed_optional_fields_exist_in_each_pinned_upstream_call():
    from modules.DiffusersImage.call_inputs import PIPELINE_CALL_INPUTS
    root = Path(importlib.util.find_spec('diffusers').origin).parent / 'pipelines'
    signatures = {}
    for path in root.glob('flux*/pipeline_flux*.py'):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef):
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == '__call__':
                        signatures[node.name] = {arg.arg for arg in method.args.args + method.args.kwonlyargs}
    assert len(PIPELINE_CALL_INPUTS) == 17
    for name, fields in PIPELINE_CALL_INPUTS.items():
        upstream = 'FluxPriorReduxPipeline' if name == 'FluxReduxPipeline' else name
        assert set(fields) <= signatures[upstream], name


def test_redux_independent_scales_preserve_legacy_reference_strength_and_scope():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from PIL import Image
    from modules.DiffusersImage.main import Edit, FluxReduxPipelineBundle
    from modules.DiffusersImage.call_inputs import normalize_call_inputs
    images = [Image.new('RGB', (32, 32)), Image.new('RGB', (32, 32))]
    prior = Mock(return_value=SimpleNamespace(prompt_embeds=object(), pooled_prompt_embeds=object()))
    base = Mock(return_value=SimpleNamespace(images=[images[0]]))
    base.device = 'cpu'
    base._execution_device = 'cpu'
    bundle = FluxReduxPipelineBundle(prior, base)
    values = dict(pipeline=bundle, image=images, width=32, height=32,
                  prompt='combine shape and material', reference_strength=.2)
    Edit('redux-independent-scales').execute(**values, prompt_embeds_scale=[.7, .3],
                                            pooled_prompt_embeds_scale=[.1, .9])
    assert prior.call_args.kwargs['prompt_embeds_scale'] == [.7, .3]
    assert prior.call_args.kwargs['pooled_prompt_embeds_scale'] == [.1, .9]
    assert 'prompt_embeds_scale' not in base.call_args.kwargs
    assert 'pooled_prompt_embeds_scale' not in base.call_args.kwargs
    Edit('redux-one-scale').execute(**values, prompt_embeds_scale=.4)
    assert prior.call_args.kwargs['prompt_embeds_scale'] == .4
    assert prior.call_args.kwargs['pooled_prompt_embeds_scale'] == [1., .2]
    Edit('redux-legacy-scales').execute(**values)
    assert prior.call_args.kwargs['prompt_embeds_scale'] == [1., .2]
    assert prior.call_args.kwargs['pooled_prompt_embeds_scale'] == [1., .2]
    with pytest.raises(ValueError, match='prompt_embeds_scale'):
        Edit('redux-invalid-count').execute(**values, prompt_embeds_scale=[.5])
    for value in ([float('nan'), .2], True, {'weight': .2}, [], [1.] * 9):
        with pytest.raises(ValueError, match='prompt_embeds_scale'):
            normalize_call_inputs('FluxReduxPipeline', {'prompt_embeds_scale': value})
    with pytest.raises(ValueError, match='prompt_embeds_scale'):
        normalize_call_inputs('FluxPipeline', {'prompt_embeds_scale': .5})


def test_optional_inputs_are_absent_unless_explicit_and_never_mutate_defaults():
    from modules.DiffusersImage.call_inputs import normalize_call_inputs
    assert normalize_call_inputs('FluxPipeline', {}) == {}
    assert normalize_call_inputs('QwenImagePipeline', {'prompt_2': None}) == {}
    values = {'prompt_2': 'alternate wording', 'num_images_per_prompt': '2', 'sigmas': [1., .5, .1]}
    result = normalize_call_inputs('FluxPipeline', values)
    assert result == {**values, 'num_images_per_prompt': 2}
    assert values['num_images_per_prompt'] == '2'
    assert result['sigmas'] is not values['sigmas']


def test_attention_options_are_explicit_copied_and_pipeline_scoped():
    from modules.DiffusersImage.call_inputs import normalize_call_inputs
    for pipeline, field in (('FluxPipeline', 'joint_attention_kwargs'),
                            ('Flux2Pipeline', 'attention_kwargs'),
                            ('Flux2KleinKVPipeline', 'attention_kwargs')):
        original = {'scale': .35}
        result = normalize_call_inputs(pipeline, {field: original})
        assert result[field] == original
        assert result[field] is not original
        for invalid in ({'scale': float('inf')}, {'scale': True}, {'unknown': 1},
                        {'callback': lambda: None}, ['scale', 1]):
            with pytest.raises(ValueError, match=field):
                normalize_call_inputs(pipeline, {field: invalid})
    with pytest.raises(ValueError, match='joint_attention_kwargs'):
        normalize_call_inputs('Flux2Pipeline', {'joint_attention_kwargs': {'scale': 1}})


@pytest.mark.parametrize('pipeline,key,value', [
    ('Flux2KleinKVPipeline', 'prompt_2', 'not supported'),
    ('QwenImagePipeline', 'num_images_per_prompt', 2),
    ('FluxPipeline', 'num_images_per_prompt', True),
    ('FluxPipeline', 'num_images_per_prompt', 0),
    ('FluxPipeline', 'num_images_per_prompt', 9),
    ('FluxPipeline', 'num_images_per_prompt', 1.5),
    ('FluxPipeline', 'sigmas', []),
    ('FluxPipeline', 'sigmas', [float('nan')]),
    ('FluxPipeline', 'sigmas', [1., 1.]),
    ('FluxPipeline', 'sigmas', [0.2, 0.8]),
    ('FluxPipeline', 'sigmas', '1,0.5'),
    ('FluxPipeline', 'prompt_2', {'hidden': 'value'}),
    ('FluxPipeline', 'latents', [1, 2]),
    ('FluxPipeline', 'generator', {'seed': 1}),
    ('Flux2Pipeline', 'text_encoder_out_layers', [False, 2]),
    ('Flux2Pipeline', 'caption_upsample_temperature', -1),
    ('FluxKontextPipeline', '_auto_resize', 'false'),
])
def test_invalid_optional_field_has_actionable_error(pipeline, key, value):
    from modules.DiffusersImage.call_inputs import normalize_call_inputs
    with pytest.raises(ValueError, match=key):
        normalize_call_inputs(pipeline, {key: value})


def test_explicit_embeddings_replace_only_call_time_prompt_and_keep_tensor_identity():
    import torch
    from modules.DiffusersImage.call_inputs import apply_call_inputs, normalize_call_inputs
    embeds = torch.zeros(1, 3, 8)
    pooled = torch.zeros(1, 8)
    generator = torch.Generator().manual_seed(44)
    values = {'prompt': 'keep my authored prompt', 'prompt_embeds': embeds,
              'pooled_prompt_embeds': pooled, 'generator': generator}
    options = normalize_call_inputs('FluxPipeline', values)
    target = {'prompt': values['prompt'], 'generator': object(), 'return_dict': True}
    apply_call_inputs(options, target)
    assert target['prompt'] is None
    assert target['prompt_embeds'] is embeds
    assert target['pooled_prompt_embeds'] is pooled
    assert target['generator'] is generator
    assert target['return_dict'] is True
    assert values['prompt'] == 'keep my authored prompt'
    with pytest.raises(ValueError, match='pooled_prompt_embeds'):
        normalize_call_inputs('FluxPipeline', {'prompt_embeds': embeds})
    with pytest.raises(ValueError, match='negative_prompt_embeds'):
        normalize_call_inputs('FluxPipeline', {'negative_pooled_prompt_embeds': pooled})


def test_optional_parameters_are_normal_action_ports_not_unchecked_kwargs():
    from modules.DiffusersImage.main import Generate, IMAGE_PIPELINE_ADAPTERS, image_pipeline_contract
    from modules.DiffusersImage.call_inputs import CALL_INPUT_PARAMS, PIPELINE_CALL_INPUTS
    for key, field in CALL_INPUT_PARAMS.items():
        assert Generate.params[key]['type'] == field['type']
        assert Generate.params[key].get('default') is None
    for name, fields in PIPELINE_CALL_INPUTS.items():
        adapter = IMAGE_PIPELINE_ADAPTERS[name]
        contract = image_pipeline_contract(adapter, adapter.mode_options[0])
        assert all(not contract['fieldParams'][key]['hidden'] for key in fields)
    assert 'callback_on_step_end' not in CALL_INPUT_PARAMS
    assert 'return_dict' not in CALL_INPUT_PARAMS


def test_historical_contracts_remain_exact_and_switch_hides_unavailable_sockets():
    from unittest.mock import Mock
    from modules.DiffusersImage.main import (Generate, IMAGE_PIPELINE_ADAPTERS,
        image_pipeline_contract, _compatible_image_contract)
    from modules.DiffusersImage.call_inputs import CALL_INPUT_PARAMS, CALL_INPUT_ADDITIONS
    current = image_pipeline_contract(IMAGE_PIPELINE_ADAPTERS['FluxPipeline'], 'text_to_image')
    historical_fields = dict(current['fieldParams'])
    for additions in reversed(CALL_INPUT_ADDITIONS):
        historical_fields = {key: value for key, value in historical_fields.items() if key not in additions}
        assert _compatible_image_contract({**current, 'fieldParams': historical_fields}, current)
    previous = {**current, 'fieldParams': {key: dict(value) for key, value in historical_fields.items()
                                         if key not in CALL_INPUT_PARAMS}}
    assert _compatible_image_contract(previous, current)
    old = {**previous, 'fieldParams': {key: dict(value) for key, value in previous['fieldParams'].items()
                                     if key not in ('guidance_scale_2', 'use_guidance_scale_2')}}
    old['fieldParams']['guidance_scale'].pop('label')
    assert _compatible_image_contract(old, current)
    old['fieldParams']['width']['hidden'] = True
    assert not _compatible_image_contract(old, current)
    assert not _compatible_image_contract({**previous, 'mode': 'edit_image'}, current)
    node = Generate('optional-switch')
    node.set_field_params = Mock()
    kv = image_pipeline_contract(IMAGE_PIPELINE_ADAPTERS['Flux2KleinKVPipeline'], 'text_to_image')
    node.update_image_contract({'image_contract': kv}, {})
    node.set_field_params.assert_any_call('prompt_2', {'hidden': True})
    node.set_field_params.assert_any_call('num_images_per_prompt', {'hidden': False})


def test_optional_batch_bounds_and_malformed_embedding_errors():
    import torch
    from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS, _tag_image_pipeline, preflight_image_action
    from modules.DiffusersImage.call_inputs import normalize_call_inputs
    from modiff.model_artifact_catalog import catalog_revision
    pipeline = type('FluxPipeline', (), {})()
    adapter = IMAGE_PIPELINE_ADAPTERS['FluxPipeline']
    _tag_image_pipeline(pipeline, adapter, 'text_to_image', adapter.default_repo, 'hub',
                        catalog_revision(adapter.default_repo))
    with pytest.raises(ValueError, match='num_images_per_prompt'):
        preflight_image_action(pipeline, 'Generate', {'prompt': ['one', 'two'], 'num_images_per_prompt': 8})
    with pytest.raises(ValueError, match='num_images_per_prompt'):
        preflight_image_action(pipeline, 'Generate', {'prompt_embeds': torch.zeros(9, 3, 8),
                                                     'pooled_prompt_embeds': torch.zeros(9, 8)})
    with pytest.raises(ValueError, match='prompt_embeds'):
        normalize_call_inputs('FluxPipeline', {'prompt_embeds': torch.tensor(1.),
                                             'pooled_prompt_embeds': torch.zeros(1, 8)})


@pytest.mark.parametrize('pipeline,mode', [
    ('Flux2Pipeline', 'multi_image_reference_edit'),
    ('FluxControlNetPipeline', 'control_image'),
    ('FluxReduxPipeline', 'edit_image'),
])
def test_compiled_loader_fields_match_native_selected_pipeline_without_value_rewrites(pipeline, mode):
    from copy import deepcopy
    from unittest.mock import Mock
    from modules.DiffusersImage.main import LoadPipeline, IMAGE_PIPELINE_ADAPTERS
    from modiff.studio_execution_specs import _execution_spec_role_params
    adapter = IMAGE_PIPELINE_ADAPTERS[pipeline]
    original = deepcopy(LoadPipeline.params)
    node = LoadPipeline('selected-loader-fields')
    node.set_field_params = Mock()
    node.set_field_value = Mock()
    node.update_pipeline_contract({'pipeline_class': pipeline, 'mode': mode,
        'model_id': {'source': 'hub', 'value': adapter.default_repo}}, {'key': 'mode'})
    published = {call.args[0]: call.args[1] for call in node.set_field_params.call_args_list}
    compiled = _execution_spec_role_params({'pipelineClass': pipeline, 'mode': mode,
        'executionPath': 'diffusers-image'}, 'modules.DiffusersImage.LoadPipeline', {'params': original})
    for field in ('mode', 'model_id', 'conditioning_kind', 'conditioning_model_id', 'conditioning_revision'):
        for key, value in published[field].items():
            assert compiled[field].get(key) == value, (pipeline, field, key)
        assert compiled[field].get('value') == original[field].get('value')
    assert LoadPipeline.params == original


def test_compiled_action_role_initializes_selected_field_visibility_not_placeholder():
    from copy import deepcopy
    from modules.DiffusersImage.main import ControlGenerate, Generate
    from modiff.studio_execution_specs import _execution_spec_role_params
    original = deepcopy(ControlGenerate.params)
    public = {'pipelineClass': 'FluxControlNetPipeline', 'mode': 'control_image',
              'executionPath': 'diffusers-image'}
    params = _execution_spec_role_params(public, 'modules.DiffusersImage.ControlGenerate',
                                        {'params': ControlGenerate.params})
    assert params['prompt_2']['hidden'] is False
    assert params['num_images_per_prompt']['hidden'] is False
    assert params['use_guidance_scale_2']['hidden'] is False
    assert params['guidance_scale']['label'] == 'Distilled Guidance'
    assert params['prompt_2'].get('value') is None
    assert ControlGenerate.params == original
    kv = _execution_spec_role_params({**public, 'pipelineClass': 'Flux2KleinKVPipeline', 'mode': 'text_to_image'},
                                     'modules.DiffusersImage.Generate', {'params': Generate.params})
    assert kv['prompt_2']['hidden'] is True
    assert kv['num_images_per_prompt']['hidden'] is False
    assert 'guidance_scale_2' not in kv


@pytest.mark.skipif(importlib.util.find_spec('transformers') is None,
                    reason='Optional model runtime not active')
@pytest.mark.parametrize('masked_attention', [False, True])
def test_real_pipeline_uses_connected_embeddings_latents_sigmas_and_batch_through_dispatch(masked_attention):
    import numpy as np
    import torch
    from unittest.mock import Mock
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel
    from peft import LoraConfig
    from modules.DiffusersImage import main as nodes
    from modules.Tensor.main import AttentionArguments
    from modiff.model_artifact_catalog import catalog_revision
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(11)
        transformer = FluxTransformer2DModel(in_channels=8, num_layers=1, num_single_layers=1,
            attention_head_dim=16, num_attention_heads=1, joint_attention_dim=8,
            pooled_projection_dim=8, axes_dims_rope=(4, 6, 6), guidance_embeds=True)
        transformer.add_adapter(LoraConfig(r=2, lora_alpha=2, init_lora_weights=False,
                                          target_modules=['to_q', 'to_k', 'to_v']))
        vae = AutoencoderKL(block_out_channels=(32,)*4, layers_per_block=1, latent_channels=2,
            down_block_types=('DownEncoderBlock2D',)*4, up_block_types=('UpDecoderBlock2D',)*4,
            shift_factor=0.).eval()
        pipeline = FluxPipeline(transformer=transformer, vae=vae, text_encoder=None, tokenizer=None,
            text_encoder_2=None, tokenizer_2=None,
            scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=False))
        pipeline.set_progress_bar_config(disable=True)
        adapter = nodes.IMAGE_PIPELINE_ADAPTERS['FluxPipeline']
        nodes._tag_image_pipeline(pipeline, adapter, 'text_to_image', adapter.default_repo,
                                 'hub', catalog_revision(adapter.default_repo))
        node = nodes.Generate('optional-input-dispatch')
        node.progress = Mock()
        prompt_embeds, pooled = torch.randn(1, 3, 8), torch.randn(1, 8)
        latents = torch.randn(2, 16, 8)
        mask = None
        if masked_attention:
            mask = torch.ones(1, 1, 19, 19, dtype=torch.bool)
            mask[..., :8] = False
        options = AttentionArguments('ordinary-attention-builder')(
            attention_mask=mask, enable_lora_scale=True, lora_scale=.35)['options']
        values = dict(pipeline=pipeline, prompt='authored text retained', prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled, num_images_per_prompt=2, sigmas=[1., .6, .2],
            latents=latents, width=64, height=64, num_inference_steps=3, guidance_scale=1.,
            joint_attention_kwargs=options,
            generator=[torch.Generator().manual_seed(3), torch.Generator().manual_seed(4)])
        actual = node(**values)
        direct = pipeline(prompt=None, prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled,
            num_images_per_prompt=2, sigmas=[1., .6, .2], latents=latents, width=64, height=64,
            num_inference_steps=3, true_cfg_scale=1.,
            joint_attention_kwargs=dict(options),
            generator=[torch.Generator().manual_seed(3), torch.Generator().manual_seed(4)])
        assert len(actual['images']) == 2
        for first, second in zip(actual['images'], direct.images):
            np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
        assert values['prompt'] == 'authored text retained'
        assert values['joint_attention_kwargs']['scale'] == .35
        if masked_attention:
            assert values['joint_attention_kwargs']['attention_mask'] is mask
            assert not mask[..., :8].any() and mask[..., 8:].all()
            unmasked = pipeline(prompt=None, prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled,
                num_images_per_prompt=2, sigmas=[1., .6, .2], latents=latents, width=64, height=64,
                num_inference_steps=3, true_cfg_scale=1., joint_attention_kwargs={'scale': .35},
                generator=[torch.Generator().manual_seed(3), torch.Generator().manual_seed(4)])
            assert any(not np.array_equal(np.asarray(first), np.asarray(second))
                       for first, second in zip(actual['images'], unmasked.images))
        else:
            assert values['joint_attention_kwargs'] == {'scale': .35}
        for module in transformer.modules():
            if hasattr(module, 'scaling'):
                assert all(scale == 1. for scale in module.scaling.values())
        assert node._active_pipeline is None
    finally:
        torch.set_num_threads(threads)
