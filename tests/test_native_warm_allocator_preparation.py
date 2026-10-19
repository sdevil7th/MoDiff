"""Do not destroy a valid native owner before observing reusable CUDA blocks.

Actual tiny official CPU loader/manager ownership is used. CUDA memory values
and the allocator observation are explicit policy fixtures, not GPU/fit proof.
"""
import asyncio
from copy import deepcopy
from unittest.mock import Mock

import pytest

import test_flux3_native_lora as native_tests
from modiff.native_component_lora import native_component_graph_scope

runtime = native_tests.runtime
GIB = 1024**3


def hardware(free):
    return {'system': {'ram_available': 16 * GIB, 'ram_total': 32 * GIB},
            'devices': [{'type': 'cuda', 'torch_vram_free': free, 'vram_free': free,
                         'torch_vram_total': 16 * GIB, 'vram_total': 16 * GIB}]}


@pytest.mark.parametrize('control', [None, 'still_low', 'ram_became_low', 'missing_observation',
    'invalidated', 'changed_source', 'changed_inputs', 'foreign_owner', 'queued_work',
    'outside_owned_callback', 'changed_recipe', 'initial_ram_low'])
def test_complete_native_owner_refreshes_real_capacity_before_destructive_cleanup(runtime, monkeypatch, control):
    app = runtime.server
    app.backend_source_identity = {'fingerprint': 'unit-native-owner-source'}
    graph = native_tests.source_graph(native_tests.PAIRS[1])
    node = runtime.run(graph)
    hints = app._runtime_cleanup_hints_for_graph(graph['nodes'], {'resourceMode': 'expert'})
    app._last_auto_model_family = hints['modelType']
    app._last_auto_resource_signature = app._auto_candidate_cache_signature(hints)
    app._workflow_auto_allocator_budget = {'used': False}
    low, high = hardware(1 * GIB), hardware(8 * GIB)
    if control == 'initial_ram_low':
        low['system']['ram_available'] = 3 * GIB
    if control == 'still_low':
        high = deepcopy(low)
    if control == 'ram_became_low':
        high['system']['ram_available'] = 3 * GIB
    import modiff.server as server_module
    monkeypatch.setattr(server_module, 'get_hardware_snapshot', lambda *_args, **_kwargs: deepcopy(low))
    fingerprint = {'backendSource': app.backend_source_identity, 'hardware': low, 'fingerprint': 'unit-runtime'}
    monkeypatch.setattr(app, '_runtime_fingerprint', lambda **_kwargs: deepcopy(fingerprint))
    observed = Mock(return_value=None if control == 'missing_observation' else {**fingerprint, 'hardware': high})
    monkeypatch.setattr(app, '_refresh_workflow_auto_cuda_capacity', observed)
    cleanup = Mock(return_value={'released': {}, 'errors': []})
    if control is None:
        cleanup.side_effect = AssertionError('A valid warm owner was destroyed before allocator recheck.')
    monkeypatch.setattr(app, '_release_runtime_caches_for_retry', cleanup)
    before_pipeline = node.loader
    before_components = dict(runtime.manager.components)
    if control == 'invalidated':
        node.invalidate_cache()
    if control == 'changed_source':
        app.backend_source_identity = {'fingerprint': 'unit-other-source'}
    if control == 'changed_inputs':
        graph['nodes']['models']['params']['dtype'] = {'value': 'float16'}
    if control == 'foreign_owner':
        runtime.manager.components['foreign'] = native_tests.torch.nn.Linear(1, 1)
        runtime.manager.collections['foreign'] = {'foreign'}
    if control == 'queued_work':
        app.queued_tasks['foreign-task'] = {'status': 'queued'}
    if control == 'changed_recipe':
        hints['dtype'] = 'float16'

    async def actual_owned_callback():
        app.loop = asyncio.get_running_loop()
        async with app._node_cache_lock:
            def prepare():
                assert app._workflow_auto_owned_callback() is (control != 'outside_owned_callback')
                current_graph = deepcopy(graph)
                app.current_task['args'] = (current_graph,)
                with native_component_graph_scope(current_graph, app.backend_source_identity['fingerprint']):
                    return app._prepare_auto_runtime_for_graph(hints)
            if control == 'outside_owned_callback':
                return prepare()
            return await app._run_executor_callback(prepare, serialize_model_io=True)

    try:
        result = asyncio.run(actual_owned_callback())
        if control is None:
            assert not result['performed'] and result['residentRecipeReusable']
            assert result['availableVramBytes'] == 8 * GIB
            observed.assert_called_once()
            cleanup.assert_not_called()
        else:
            assert result['performed']
            cleanup.assert_called_once_with()
            if control in {'still_low', 'ram_became_low', 'missing_observation'}:
                observed.assert_called_once()
            else:
                observed.assert_not_called()
        assert app.node_cache['models'] is node and node.loader is before_pipeline
        assert all(runtime.manager.components[key] is value for key, value in before_components.items())
    finally:
        app.loop = None
        app.queued_tasks.clear()
        if app._model_executor is not None:
            app._model_executor.shutdown(wait=True)
            app._model_executor = None
