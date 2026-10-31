"""Actual Redux loader/stage dispatcher/cache teardown with tiny CPU components."""
from copy import deepcopy
from functools import wraps
import gc
import weakref

import pytest

import redux_stage_test_support as contract
from redux_stage_test_support import bounded_redux_stage_runtime as bounded_redux_stage_runtime


def require_stage_api():
    import modules
    import modules.DiffusersImage.main as image
    expected = ('EncodeInputs', 'Denoise', 'DecodeLatents')
    assert all(name in modules.MODULE_MAP['modules.DiffusersImage'] for name in expected), (
        'Real generic stage registry/ports are required; do not manufacture them in this fixture.'
    )
    assert all(hasattr(image, name) for name in expected)
    assert 'encoded_inputs' in image.EncodeInputs.params
    assert 'encoded_inputs' in image.Denoise.params
    assert 'latent_binding' in image.DecodeLatents.params
    return image


@pytest.fixture
def actual_graph_runtime(tmp_path, monkeypatch):
    image = require_stage_api()  # Current absence is a meaningful pre-model RED.
    import torch
    import modules
    import modules.Text.main as text
    from PIL import Image
    from modiff.NodeBase import NodeBase
    from modiff.server import WebServer, memory_manager
    from diffusers import FluxPipeline, FluxPriorReduxPipeline

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    events, pending, owner_refs = [], {}, []

    class SyntheticStageInput(NodeBase):
        """Registered fixture supplier, not a fake pipeline/model or Auto grant."""
        params = {'kind': {'type': 'string'}, 'value': {'display': 'output', 'type': 'any'}}
        skipParamsCheck = True

        def execute(self, kind):
            if kind == 'images':
                return {'value': [Image.new('RGB', (16, 16), (23, 51, 89)),
                                  Image.new('RGB', (16, 16), (149, 101, 37))]}
            generator = torch.Generator().manual_seed(41)
            value = torch.randn(2, 3, 8, generator=generator) if kind == 'embeddings' else torch.randn(2, 8, generator=generator)
            return {'value': value}

    SyntheticStageInput.__module__ = 'modules.Text.main'
    monkeypatch.setattr(text, 'SyntheticStageInput', SyntheticStageInput, raising=False)
    monkeypatch.setitem(modules.MODULE_MAP['modules.Text'], 'SyntheticStageInput', {
        'params': SyntheticStageInput.params, 'skipParamsCheck': True, 'outputs': ['value'],
    })
    monkeypatch.setattr(memory_manager, 'cache', {})
    monkeypatch.setattr('modiff.NodeBase.modelstore.is_hf_cached', lambda *args: True)
    monkeypatch.setattr('modiff.NodeBase.modelstore.is_local_cached', lambda *args: True)
    monkeypatch.setattr('modiff.server.get_hardware_snapshot', lambda *args, **kwargs: {
        'system': {'ram_total': 64 * 1024**3, 'ram_available': 48 * 1024**3}, 'devices': [],
    })

    def cached_construction(cls, repository, **kwargs):
        assert kwargs['local_files_only'] is True and kwargs['use_safetensors'] is True
        if cls is FluxPipeline:
            assert repository == contract.BASE_REPO and kwargs['revision'] == contract.BASE_REVISION
            bundle = contract.make_actual_tiny_bundle()
            # Match the loader's genuine pre-transfer base component ownership.
            bundle.base.register_modules(text_encoder=bundle.prior.text_encoder,
                text_encoder_2=bundle.prior.text_encoder_2, tokenizer=None, tokenizer_2=None)
            pending['construction'] = bundle
            return bundle.base
        assert cls is FluxPriorReduxPipeline
        assert repository == contract.PRIOR_REPO and kwargs['revision'] == contract.PRIOR_REVISION
        bundle = pending.pop('construction')
        assert kwargs['text_encoder'] is bundle.prior.text_encoder
        assert kwargs['text_encoder_2'] is bundle.prior.text_encoder_2
        return bundle.prior

    monkeypatch.setattr(image, 'load_cached_image_component', cached_construction)
    actual_prior_call, actual_base_call = FluxPriorReduxPipeline.__call__, FluxPipeline.__call__

    @wraps(actual_prior_call)
    def prior_call(self, **kwargs):
        events.append(('prior', kwargs.get('prompt_embeds_scale'), kwargs.get('pooled_prompt_embeds_scale')))
        return actual_prior_call(self, **kwargs)

    @wraps(actual_base_call)
    def base_call(self, **kwargs):
        # Store only scalar/identity observations; callbacks must not retain nodes.
        events.append(('base', kwargs['output_type'], kwargs['generator'].initial_seed(), kwargs['num_inference_steps']))
        return actual_base_call(self, **kwargs)

    monkeypatch.setattr(FluxPriorReduxPipeline, '__call__', prior_call)
    monkeypatch.setattr(FluxPipeline, '__call__', base_call)
    app = WebServer(modules=modules.MODULE_MAP, work_dir=str(tmp_path), data_dir=str(tmp_path / 'data'))
    monkeypatch.setattr(app, 'queue_message', lambda _: None)
    monkeypatch.setattr('modiff.server.server', app)
    monkeypatch.setattr(app, '_runtime_fingerprint', lambda **kwargs: {'fingerprint': 'actual-tiny-CPU-component-test'})
    monkeypatch.setattr(app, '_record_auto_resource_success', lambda *args, **kwargs: None)
    monkeypatch.setattr(app, '_record_optimization_observations', lambda *args, **kwargs: [])
    # Keep cleanup in this isolated unit cache; no real offload root/device/allocator work.
    monkeypatch.setattr(app, '_release_diffusers_offload_cache', lambda: (0, []))
    monkeypatch.setattr(app, '_best_effort_device_cache_clear', lambda: [])
    monkeypatch.setattr(app, '_best_effort_allocator_trim', lambda: (False, []))
    try:
        yield app, events, owner_refs
    finally:
        app.node_cache.clear()
        memory_manager.cache.clear()
        pending.clear()
        torch.set_num_threads(old_threads)


def stage_graph():
    from utils.torch_utils import CPU_DEVICE
    # Expert graph isolates dispatch; it creates NO synthetic Auto budget/credit.
    return {'sid': 'redux-stage-unit', 'nodes': {
        'models': {'module': 'modules.DiffusersImage', 'action': 'LoadPipeline', 'params': {
            'model_id': {'value': {'source': 'hub', 'value': contract.PRIOR_REPO}},
            'revision': {'value': contract.PRIOR_REVISION}, 'pipeline_class': {'value': 'FluxReduxPipeline'},
            'mode': {'value': 'multi_image_reference_edit'}, 'execution_profile_id': {'value': 'flux-redux:direct'},
            'device': {'value': CPU_DEVICE}, 'dtype': {'value': 'float32'},
            'auto_offload': {'value': False}, 'offload_mode': {'value': 'none'},
        }},
        **{name: {'module': 'modules.Text', 'action': 'SyntheticStageInput',
                  'params': {'kind': {'value': kind}}}
           for name, kind in (('source', 'images'), ('text', 'embeddings'), ('pooled', 'pooled'))},
        'encode': {'module': 'modules.DiffusersImage', 'action': 'EncodeInputs', 'params': {
            'pipeline': {'sourceId': 'models', 'sourceKey': 'pipeline'},
            'image': {'sourceId': 'source', 'sourceKey': 'value'},
            'prompt_embeds': {'sourceId': 'text', 'sourceKey': 'value'},
            'pooled_prompt_embeds': {'sourceId': 'pooled', 'sourceKey': 'value'},
            'prompt': {'value': ''}, 'reference_strength': {'value': .2},
            'prompt_embeds_scale': {'value': [.7, .3]}, 'pooled_prompt_embeds_scale': {'value': [.1, .9]},
        }},
        'denoise': {'module': 'modules.DiffusersImage', 'action': 'Denoise', 'params': {
            'pipeline': {'sourceId': 'models', 'sourceKey': 'pipeline'},
            'encoded_inputs': {'sourceId': 'encode', 'sourceKey': 'encoded_inputs'},
            'seed': {'value': 5}, 'width': {'value': 64}, 'height': {'value': 64},
            'num_inference_steps': {'value': 3}, 'guidance_scale': {'value': 1.},
        }},
        'decode': {'module': 'modules.DiffusersImage', 'action': 'DecodeLatents', 'params': {
            'pipeline': {'sourceId': 'models', 'sourceKey': 'pipeline'},
            'latents': {'sourceId': 'denoise', 'sourceKey': 'latents_out'},
            'latent_binding': {'sourceId': 'denoise', 'sourceKey': 'latent_binding'},
            'width': {'value': 64}, 'height': {'value': 64}, 'output_type': {'value': 'pt'},
        }},
    }, 'paths': [['models', 'source', 'text', 'pooled', 'encode', 'denoise', 'decode']]}


def dispatch(app, graph):
    app.current_task = {'task_id': 'actual-redux-stage-task', 'sid': graph['sid'],
                        'status': 'running', 'progress': 0}
    app._execute_graph(deepcopy(graph))
    return app.node_cache['decode'].output['images']


def test_actual_graph_prior_once_base_latent_only_and_matching_real_decoder(actual_graph_runtime):
    import torch
    app, events, refs = actual_graph_runtime
    graph = stage_graph()
    actual = dispatch(app, graph)
    assert [item[0] for item in events] == ['prior', 'base']
    assert events[0] == ('prior', [.7, .3], [.1, .9])
    assert events[1] == ('base', 'latent', 5, 3)
    bundle = app.node_cache['models'].output['pipeline']
    assert bundle._modiff_stage_loading_contract == ('none', 'none', False)
    refs.extend([weakref.ref(bundle), weakref.ref(bundle.prior), weakref.ref(bundle.base)])
    assert bundle.base.text_encoder is None and bundle.prior.text_encoder is not None
    # Genuine independently repeated old composite call is the reference. The
    # immutable stage output must not call prior during Denoise or decode.
    expected = bundle(image=app.node_cache['source'].output['value'], prompt='',
        prompt_embeds=app.node_cache['text'].output['value'].clone(),
        pooled_prompt_embeds=app.node_cache['pooled'].output['value'].clone(),
        prompt_embeds_scale=[.7, .3], pooled_prompt_embeds_scale=[.1, .9],
        width=64, height=64, num_inference_steps=3, guidance_scale=1.,
        generator=torch.Generator().manual_seed(5), output_type='pt').images
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    events.clear()
    graph['nodes']['denoise']['params']['seed']['value'] = 6
    dispatch(app, graph)
    assert events == [('base', 'latent', 6, 3)]  # Valid encoder cache, same owner.
    assert app.node_cache['models'].output['pipeline'] is refs[0]()


def test_actual_graph_released_owner_dies_but_detached_material_output_survives(actual_graph_runtime):
    import torch
    app, _events, refs = actual_graph_runtime
    pixels = dispatch(app, stage_graph()).clone()
    refs.extend([weakref.ref(app.node_cache['models'].output['pipeline']),
                 weakref.ref(app.node_cache['models'].output['pipeline'].prior),
                 weakref.ref(app.node_cache['models'].output['pipeline'].base)])
    result = app._release_runtime_caches_for_retry()
    gc.collect()
    assert not result['errors']
    assert not app.node_cache
    assert all(reference() is None for reference in refs)
    assert pixels.shape == (1, 3, 64, 64) and torch.isfinite(pixels).all()
    # Actual owner/cache object lifetime is proved here. This does not prove
    # physical allocator/RSS reclamation or disk-hook qualification.


@pytest.mark.parametrize('recipe,expected_map', [(None, 'none'),
    ({'offload_mode': 'none', 'device_map': 'balanced'}, 'balanced')])
def test_actual_loader_captures_effective_recipe_placement_before_bridge_admission(
    actual_graph_runtime, recipe, expected_map,
):
    from modules.DiffusersImage import stage_execution as core
    app, _events, _refs = actual_graph_runtime
    current = stage_graph()
    if recipe is not None:
        current['nodes']['models']['params']['execution_recipe'] = {'value': recipe}
    current['paths'] = [['models']]
    app.current_task = {'task_id': 'redux-loading-contract', 'sid': current['sid'],
                        'status': 'running', 'progress': 0}
    app._execute_graph(deepcopy(current))
    owner = app.node_cache['models'].output['pipeline']
    assert owner._modiff_stage_loading_contract == ('none', expected_map, False)
    # Acquisition is the genuine tiny factory substitute, so the balanced
    # record proves effective control capture/refusal, not distributed loading.
    if expected_map == 'none':
        transformer, clip = core._require_split_lora_owner(owner)
        assert transformer is owner.base.transformer and clip is owner.prior.text_encoder
    else:
        with pytest.raises(ValueError, match='loading contract'):
            core._require_split_lora_owner(owner)
