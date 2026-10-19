"""Image/text ownership through ordinary dispatch; synthetic CPU weights only."""
from collections import Counter
from copy import deepcopy
import gc
import importlib
from types import SimpleNamespace
from unittest.mock import Mock
import weakref

import pytest
import torch
from PIL import Image

import test_prompt_enhancement as stage_one
from modiff.NodeBase import memory_manager
from modiff.native_component_lora import native_component_graph_scope
from modiff.workflow_auto_lifecycle import release_owner_caches
from modules.HuggingFaceTransformers import main as nodes
from utils.torch_utils import DEFAULT_DEVICE

actual_pup = stage_one.actual_pup


@pytest.fixture(autouse=True)
def isolated_manager(monkeypatch):
    monkeypatch.setattr(memory_manager, 'cache', {})
    monkeypatch.setattr(memory_manager, '_active_ids', Counter())
    # Synthetic ownership checks isolate ambient RAM pressure, not model fit.
    # Keep real registration, leases, release and the production LRU policy.
    monkeypatch.setattr('modiff.hardware.system_memory_snapshot', lambda: {
        'available_bytes': 16 * 1024**3, 'total_bytes': 32 * 1024**3,
    })


def cached_pair(monkeypatch):
    import modules
    from modiff.server import WebServer
    def node(action, **params):
        return {'module': 'modules.HuggingFaceTransformers', 'action': action,
                'params': {key: value if type(value) is dict and 'sourceId' in value else {'value': value}
                           for key, value in params.items()}}
    graph = {'sid': None, 'nodes': {
        'pup-owner': node('LoadImageTextToTextModel',
            model_id={'source': 'hub', 'value': stage_one.REPOSITORY}, revision=stage_one.REVISION,
            dtype='float32', device=DEFAULT_DEVICE),
        'pup-proposal': node('GenerateImageVideoText', model={'sourceId': 'pup-owner', 'sourceKey': 'model'},
            prompt='A red cube.', operation='enhance_prompt', enhancement_max_new_tokens=2, min_new_tokens=2),
    }, 'paths': [['pup-owner', 'pup-proposal']]}
    app = object.__new__(WebServer)
    app.modules, app.node_cache = modules.MODULE_MAP, {}
    app.instance = 'explicit-pup-owner-unit-worker'
    app.backend_source_identity = {'fingerprint': stage_one.unit_source_fingerprint()}
    app.current_task = {'task_id': 'pup-owner-unit', 'attempt_index': 0, 'args': [graph], 'sid': None,
        'current_node': 'pup-owner', 'interrupt_requested': False, 'runtimeHints': {}}
    app._active_graph_node_ids = set(graph['nodes'])
    app.queue_message = Mock()
    app._best_effort_device_cache_clear = lambda: []
    app._best_effort_allocator_trim = lambda: (True, [])
    app._cuda_memory_snapshot = lambda: None
    monkeypatch.setattr(importlib.import_module('modiff.NodeBase'), '_server', lambda: app)
    dispatch(app, graph, 'pup-owner')
    dispatch(app, graph, 'pup-proposal')
    return app, graph


def dispatch(app, graph, node_id):
    app.current_task['current_node'] = node_id
    with native_component_graph_scope(graph, app.backend_source_identity['fingerprint']):
        app.execute_node(node_id, graph['nodes'][node_id], None, quiet=True)


def release(app):
    return release_owner_caches(app, {'ownerIds': ['pup-owner'],
        'nodeIds': app._dependent_cache_nodes(['pup-owner']),
        'retainOutputs': {'pup-proposal': ['result']}}, memory_manager)


def test_actual_loader_registers_exact_model_without_changing_public_handle(actual_pup, monkeypatch):
    app, _ = cached_pair(monkeypatch)
    owner = app.node_cache['pup-owner']
    handle = owner.output['model']
    registered = [key for key, entry in memory_manager.cache.items() if entry['model'] is handle['model']]
    assert owner._mm_models and registered == owner._mm_models
    assert set(handle) == nodes.MODEL_HANDLE_KEYS and handle['receipt'] == owner.output['receipt']
    assert app.node_cache['pup-proposal']._cache_input_sources == {'pup-owner'}


def test_actual_external_handle_stops_release_before_any_next_owner(actual_pup, monkeypatch):
    app, _ = cached_pair(monkeypatch)
    external = app.node_cache['pup-owner'].output['model']
    reference = weakref.ref(external['model'])
    with pytest.raises(ValueError, match='model references remain alive'):
        release(app)
    assert reference() is external['model']
    assert 'pup-owner' not in app.node_cache
    assert app.node_cache['pup-proposal'].output['result']['authoredPrompt'] == 'A red cube.'


def test_actual_exclusive_release_collects_model_and_keeps_detached_proposal(actual_pup, monkeypatch):
    app, _ = cached_pair(monkeypatch)
    reference = weakref.ref(app.node_cache['pup-owner'].output['model']['model'])
    proposal = deepcopy(app.node_cache['pup-proposal'].output['result'])
    result = release(app)
    gc.collect()
    assert reference() is None
    assert result['retainedOutputNodes'] == ['pup-proposal']
    assert app.node_cache['pup-proposal'].output['result'] == proposal


def test_actual_shared_owner_and_unrelated_registered_model_survive(actual_pup, monkeypatch):
    app, _ = cached_pair(monkeypatch)
    handle = app.node_cache['pup-owner'].output['model']
    ids = list(app.node_cache['pup-owner']._mm_models)
    unrelated = torch.nn.Linear(2, 2)
    unrelated_id = memory_manager.add(unrelated)
    app.node_cache['shared'] = SimpleNamespace(output={'model': handle}, params={}, _mm_models=ids)
    app.node_cache['unrelated'] = SimpleNamespace(output={}, params={}, _mm_models=[unrelated_id])
    result = release(app)
    assert result['releasedComponentCount'] == 0
    assert all(memory_manager.get_model(key) is handle['model'] for key in ids)
    assert memory_manager.get_model(unrelated_id) is unrelated
    assert app.node_cache['shared'].output['model'] is handle


def test_warm_eviction_reloads_owner_and_stale_consumer_cannot_reuse_cache(actual_pup, monkeypatch):
    _, calls = actual_pup
    app, graph = cached_pair(monkeypatch)
    old = app.node_cache['pup-owner'].output['model']
    ids = list(app.node_cache['pup-owner']._mm_models)
    assert ids
    consumer = nodes.GenerateImageVideoText('stale-consumer')
    args = dict(model=old, prompt='A red cube.', operation='enhance_prompt', enhancement_max_new_tokens=2)
    consumer(**args)
    for key in ids:
        memory_manager.remove(key)
    with pytest.raises(ValueError, match='no longer registered'):
        consumer(**args)
    count = len(calls)
    dispatch(app, graph, 'pup-owner')
    assert app.node_cache['pup-owner']._cache_reason == 'models_evicted'
    fresh = app.node_cache['pup-owner'].output['model']
    assert fresh['model'] is not old['model'] and len(calls) == count + 2
    dispatch(app, graph, 'pup-proposal')
    assert app.node_cache['pup-proposal'].output['result']['generatedTokens'] == 2


@pytest.mark.parametrize('operation', ['standard', 'enhance_prompt'])
def test_actual_generation_holds_exact_standard_manager_lease(actual_pup, monkeypatch, operation):
    handle, _ = actual_pup
    model = handle['model']
    assert getattr(model, '_mm_id', None) in memory_manager.cache
    original = model.generate
    calls = []
    def checked(**kwargs):
        calls.append(True)
        assert memory_manager.get_model(model._mm_id) is model
        assert memory_manager._active_ids[model._mm_id] > 0
        return original(**kwargs)
    monkeypatch.setattr(model, 'generate', checked)
    node = nodes.GenerateImageVideoText('leased-' + operation)
    node(model=handle, prompt='A red cube.', operation=operation, max_new_tokens=2,
         images=[Image.new('RGB', (256, 256), 'red')] if operation == 'standard' else None,
         enhancement_max_new_tokens=2, min_new_tokens=2)
    assert len(calls) == 1 and not memory_manager._active_ids
    assert node.output['result']['generatedTokens'] == 2


@pytest.mark.parametrize('operation,error', [('standard', torch.OutOfMemoryError),
    ('enhance_prompt', torch.OutOfMemoryError), ('enhance_prompt', InterruptedError)])
def test_error_or_cancel_releases_lease_and_does_not_replay(actual_pup, monkeypatch, operation, error):
    handle, _ = actual_pup
    model = handle['model']
    calls = []
    def failed(**kwargs):
        calls.append(True)
        assert memory_manager._active_ids[model._mm_id] > 0
        raise error('explicit unit neural fault')
    monkeypatch.setattr(model, 'generate', failed)
    node = nodes.GenerateImageVideoText('fault-' + operation)
    with pytest.raises(RuntimeError, match='explicit unit neural fault') as caught:
        node(model=handle, prompt='A red cube.', operation=operation, max_new_tokens=2,
         images=[Image.new('RGB', (256, 256), 'red')] if operation == 'standard' else None,
             enhancement_max_new_tokens=2)
    assert isinstance(caught.value.__cause__, error)
    assert len(calls) == 1 and not memory_manager._active_ids
    assert not node._cache_valid and all(value is None for value in node.output.values())


def test_owned_stop_during_actual_neural_call_releases_lease(actual_pup, monkeypatch):
    handle, _ = actual_pup
    node = nodes.GenerateImageVideoText('leased-stop')
    graph, task, _ = stage_one.owned_generation(monkeypatch, node)
    model = handle['model']
    original = model.generate
    calls = []
    def stopped(**kwargs):
        calls.append(True)
        assert memory_manager._active_ids[model._mm_id] > 0
        task['interrupt_requested'] = True
        return original(**kwargs)
    monkeypatch.setattr(model, 'generate', stopped)
    with native_component_graph_scope(graph, stage_one.unit_source_fingerprint()), pytest.raises(RuntimeError) as caught:
        node(model=handle, prompt='A red cube.', operation='enhance_prompt', enhancement_max_new_tokens=2)
    assert isinstance(caught.value.__cause__, InterruptedError)
    assert len(calls) == 1 and not memory_manager._active_ids


def test_direct_lower_level_unregistered_handle_keeps_existing_behavior(actual_pup):
    handle, _ = nodes._load_model(selection_value={'source': 'hub', 'value': stage_one.REPOSITORY},
        revision_value=stage_one.REVISION, dtype_value='float32', device_value=DEFAULT_DEVICE,
        task='image-video-to-text')
    assert not hasattr(handle['model'], '_mm_id')
    direct = nodes.GenerateImageVideoText()
    output = direct(model=handle, prompt='A red cube.', images=[Image.new('RGB', (256, 256), 'red')],
                    max_new_tokens=2, min_new_tokens=2)
    assert output['result']['generatedTokens'] == 2 and not hasattr(handle['model'], '_mm_id')


def test_retained_opaque_preflight_remains_before_registered_owner_mutation():
    model = torch.nn.Linear(2, 2)
    key = memory_manager.add(model)
    owner = SimpleNamespace(output={'model': model}, params={}, _mm_models=[key])
    app = SimpleNamespace(node_cache={'load': owner})
    with pytest.raises(ValueError, match='opaque'):
        release_owner_caches(app, {'ownerIds': ['load'], 'nodeIds': ['load'],
            'retainOutputs': {'load': ['model']}}, memory_manager)
    assert app.node_cache['load'] is owner and memory_manager.get_model(key) is model


def test_registration_failure_cannot_publish_or_warm_reuse_raw_success(actual_pup, monkeypatch):
    _, calls = actual_pup
    owner = nodes.LoadImageTextToTextModel('failed-registration')
    original = owner.mm_add
    registered = []
    def failed(model):
        key = original(model)
        registered.append(key)
        raise RuntimeError('explicit post-registration fault')
    monkeypatch.setattr(owner, 'mm_add', failed)
    args = dict(model_id={'source': 'hub', 'value': stage_one.REPOSITORY}, revision=stage_one.REVISION,
                dtype='float32', device=DEFAULT_DEVICE)
    with pytest.raises(RuntimeError, match='explicit post-registration fault'):
        owner(**args)
    assert not owner._cache_valid and all(value is None for value in owner.output.values())
    assert owner._mm_models == [] and all(memory_manager.get_model(key) is None for key in registered)
    count = len(calls)
    monkeypatch.setattr(owner, 'mm_add', original)
    owner(**args)
    assert len(calls) == count + 2 and owner._cache_valid


def test_warm_registration_is_idempotent_and_rejects_borrowed_cached_handle(actual_pup, monkeypatch):
    other, calls = actual_pup
    owner = nodes.LoadImageTextToTextModel('warm-owner')
    add = Mock(wraps=owner.mm_add)
    monkeypatch.setattr(owner, 'mm_add', add)
    args = dict(model_id={'source': 'hub', 'value': stage_one.REPOSITORY}, revision=stage_one.REVISION,
                dtype='float32', device=DEFAULT_DEVICE)
    owner(**args)
    owned = owner.output['model']['model']
    count = len(calls)
    owner(**args)
    assert add.call_count == 1 and len(calls) == count
    owner.output['model']['model'] = other['model']
    with pytest.raises(ValueError, match='owner'):
        owner(**args)
    assert add.call_count == 1
    assert memory_manager.get_model(other['model']._mm_id) is other['model']
    assert memory_manager.get_model(owned._mm_id) is owned
