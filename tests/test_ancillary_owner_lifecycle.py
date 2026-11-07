"""Private owner boundary: actual leases and tiny official CPU components only."""
import asyncio
from collections import Counter
from copy import deepcopy
import importlib
import threading
from types import SimpleNamespace
import weakref

import pytest

import test_ancillary_execution as ancillary
import test_prompt_enhancement as pup
from modiff.NodeBase import memory_manager
from modiff.server import WebServer

actual_pup = pup.actual_pup


class ReachedLegacyPreparation(AssertionError):
    pass


@pytest.fixture(autouse=True)
def isolated_manager(monkeypatch):
    monkeypatch.setattr(memory_manager, 'cache', {})
    monkeypatch.setattr(memory_manager, '_active_ids', Counter())


def app_for(tmp_path, monkeypatch, graph):
    import modules
    app = WebServer(modules=modules.MODULE_MAP, work_dir=str(tmp_path), data_dir=str(tmp_path))
    app.backend_source_identity = {'fingerprint': pup.unit_source_fingerprint()}
    app.current_task = {'task_id': 'explicit-owner-boundary-unit', 'attempt_index': 0,
                        'args': [graph], 'sid': graph['sid'], 'progress': 0}
    app.queue_message = lambda *_args: None
    app._runtime_fingerprint = lambda: {'fingerprint': 'explicit-controlled-unit-snapshot'}
    app._runtime_measurement = lambda **_kwargs: {'elapsedSeconds': 0}
    app._record_auto_resource_success = lambda *_args, **_kwargs: None
    app._record_optimization_observations = lambda *_args, **_kwargs: []
    app._persist_supervisor_queue_state = lambda **_kwargs: None
    monkeypatch.setattr(importlib.import_module('modiff.NodeBase'), '_server', lambda: app)
    return app


def leased(app, graph):
    """Use the existing shielded cache lease and single model executor."""
    async def run():
        app.loop = asyncio.get_running_loop()
        return await app._with_node_cache_lease(
            lambda: app._run_executor_callback(lambda: app._execute_graph(graph), serialize_model_io=True))
    try:
        return asyncio.run(run())
    finally:
        if app._model_executor is not None:
            app._model_executor.shutdown(wait=True)
            app._model_executor = None


def legacy_tripwire(app, seen):
    def reached(_hints):
        seen.append({'cacheLocked': app._node_cache_lock.locked(),
                     'modelThreadBound': app._model_executor_thread_binding[1] == threading.get_ident()})
        raise ReachedLegacyPreparation('claimed Preview reached legacy process-wide preparation without a private origin')
    return reached


def test_claimed_preview_unknown_origin_refuses_before_legacy_preparation(tmp_path, monkeypatch):
    import modules
    graph, _parent = ancillary.preview_graph(modules.MODULE_MAP)
    graph['runtimeHints']['resourceMode'] = 'expert'
    app = app_for(tmp_path, monkeypatch, graph)
    foreign = SimpleNamespace(output={}, params={}, _mm_models=[])
    app.node_cache['unassociated-resident-owner'] = foreign
    seen = []
    app._prepare_auto_runtime_for_graph = legacy_tripwire(app, seen)
    with pytest.raises(ValueError, match='(?i)origin|owner|turnover'):
        leased(app, graph)
    assert seen == []
    assert app.node_cache['unassociated-resident-owner'] is foreign


def test_unclaimed_graph_keeps_existing_serialized_preparation(tmp_path, monkeypatch):
    graph = ancillary.graph()
    graph['runtimeHints'] = {'resourceMode': 'expert'}
    app = app_for(tmp_path, monkeypatch, graph)
    seen = []
    app._prepare_auto_runtime_for_graph = legacy_tripwire(app, seen)
    with pytest.raises(ReachedLegacyPreparation):
        leased(app, graph)
    assert seen == [{'cacheLocked': True, 'modelThreadBound': True}]


def test_invalid_claim_refuses_before_existing_preparation(tmp_path, monkeypatch):
    graph = ancillary.graph()
    graph['runtimeHints'] = {'executionPurpose': None}
    app = app_for(tmp_path, monkeypatch, graph)
    seen = []
    app._prepare_auto_runtime_for_graph = legacy_tripwire(app, seen)
    with pytest.raises(ValueError, match='ancillary'):
        leased(app, graph)
    assert seen == []


@pytest.mark.parametrize('field', ['max_input_tokens', 'max_image_pixels', 'enhancement_max_new_tokens',
    'min_new_tokens', 'num_beams', 'do_sample', 'use_chat_template'])
def test_private_preview_preallocation_controls_cannot_depend_on_the_future_provider(field):
    import modules
    from modiff.ancillary_execution import validate_execution_purpose
    graph, _parent = ancillary.preview_graph(modules.MODULE_MAP)
    graph['nodes']['proposal']['params'][field] = {'sourceId': 'provider', 'sourceKey': 'receipt'}
    with pytest.raises(ValueError, match='preallocation literals'):
        validate_execution_purpose(graph, modules.MODULE_MAP)


def test_completed_real_preview_releases_its_exact_registered_owner_after_dispatch(actual_pup, tmp_path, monkeypatch):
    """Actual tiny official model calls; not pretrained output or hardware fit."""
    import modules
    graph, _parent = ancillary.preview_graph(modules.MODULE_MAP)
    graph['runtimeHints']['resourceMode'] = 'expert'
    # The fixture's setup-only sample is not an application cache owner.
    memory_manager.remove(actual_pup[0]['model'])
    graph['nodes']['provider']['params'].update(device={'value': pup.DEFAULT_DEVICE}, dtype={'value': 'float32'})
    graph['nodes']['proposal']['params'].update(enhancement_max_new_tokens={'value': 2}, min_new_tokens={'value': 2})
    app = app_for(tmp_path, monkeypatch, graph)
    # Isolate the missing terminal event from the distinct old global-cleanup
    # defect above. This test grants no incoming memory budget or release power.
    app._prepare_auto_runtime_for_graph = lambda _hints: None
    references = {}
    real_dispatch = app.execute_node
    def observe(node_id, node, sid):
        result = real_dispatch(node_id, node, sid, quiet=True)
        if node_id == 'provider':
            references['model'] = weakref.ref(app.node_cache[node_id].output['model']['model'])
            references['managerIds'] = list(app.node_cache[node_id]._mm_models)
        return result
    app.execute_node = observe
    before = deepcopy(graph)
    leased(app, graph)
    assert graph['nodes'] == before['nodes'] and graph['paths'] == before['paths']
    assert graph['runtimeHints']['executionPurpose'] == before['runtimeHints']['executionPurpose']
    assert app.node_cache['proposal'].output['proposal_summary']
    assert 'provider' not in app.node_cache, 'completed Preview still retains its provider owner'
    assert all(key not in memory_manager.cache for key in references['managerIds'])
    assert references['model']() is None


def cold_preview(actual_pup, tmp_path, monkeypatch):
    import modules
    memory_manager.remove(actual_pup[0]['model'])
    graph, _parent = ancillary.preview_graph(modules.MODULE_MAP)
    graph['runtimeHints']['resourceMode'] = 'expert'
    graph['nodes']['provider']['params']['device'] = {'value': pup.DEFAULT_DEVICE}
    graph['nodes']['proposal']['params'].update(enhancement_max_new_tokens={'value': 2}, min_new_tokens={'value': 2})
    app = app_for(tmp_path, monkeypatch, graph)
    app._best_effort_device_cache_clear = lambda: []
    app._best_effort_allocator_trim = lambda: (False, [])
    app.messages = []
    app.queue_message = app.messages.append
    return app, graph


def quiet_dispatch(app, after=None):
    original = app.execute_node
    def execute(node_id, node, sid):
        result = original(node_id, node, sid, quiet=True)
        if after is not None:
            after(node_id)
        return result
    app.execute_node = execute


def test_repeated_real_preview_replaces_detached_output_without_retaining_provider(actual_pup, tmp_path, monkeypatch):
    """Two real tiny CPU dispatcher attempts; not trained/GPU memory proof."""
    app, graph = cold_preview(actual_pup, tmp_path, monkeypatch)
    references = []
    def observe(node_id):
        if node_id == 'provider':
            references.append(weakref.ref(app.node_cache[node_id].output['model']['model']))
    quiet_dispatch(app, observe)
    for attempt in range(2):
        graph['nodes']['proposal']['params']['prompt'] = {'value': f'A tiny cube, attempt {attempt}.'}
        app.current_task = {'task_id': f'cpu-repeat-preview-{attempt}', 'attempt_index': 0,
            'args': [graph], 'sid': graph['sid'], 'progress': 0}
        leased(app, graph)
        assert app.node_cache['proposal'].output['proposal_summary']
        assert 'provider' not in app.node_cache and not memory_manager.cache
        assert all(reference() is None for reference in references)
    assert len(references) == 2


@pytest.mark.parametrize('mode', ['auto', None])
def test_no_implicit_expert_or_auto_admission(tmp_path, monkeypatch, mode):
    import modules
    graph, _parent = ancillary.preview_graph(modules.MODULE_MAP)
    graph['runtimeHints']['resourceMode'] = mode
    app = app_for(tmp_path, monkeypatch, graph)
    app.execute_node = lambda *_args: pytest.fail('No incoming factory is admitted')
    with pytest.raises(ValueError, match='Custom memory'):
        leased(app, graph)


@pytest.mark.parametrize('fault', ['none', 'owner', 'graph', 'budget-kind', 'schedule', 'preparation', 'budget-missing', 'budget-bool'])
def test_scoped_auto_cache_seam_never_uses_global_cleanup(tmp_path, monkeypatch, fault):
    """Controlled internal plan plumbing only, not source recipe/fit authority."""
    import modules
    from modiff.ancillary_owner_lifecycle import owner_scope, bind_preview_auto_plan
    from modiff.workflow_auto_resource import workflow_graph_hash
    graph, _parent = ancillary.preview_graph(modules.MODULE_MAP)
    graph['nodes']['provider']['params'].update(device={'value': 'cuda:0'},
        dtype={'value': 'bfloat16'}, quantization_mode={'value': 'bnb_4bit'})
    graph_hash = workflow_graph_hash(graph)
    graph['runtimeHints'].update(resourceMode='auto', workflowAutoPlan={'schemaVersion': 1, 'graphHash': graph_hash})
    app = app_for(tmp_path, monkeypatch, graph)
    app._release_runtime_caches_for_retry = lambda: pytest.fail('A Preview cannot use global cache preparation')
    # This private test-double is passed only by the actual callback. Production
    # obtains the owner from its closed source-reviewed planner, never hints.
    plan = {'graphHash': graph_hash, 'plannedGraphHash': graph_hash, 'canAutoRun': True,
        'patches': [], 'schedule': None, 'requiresPreparation': False,
        'requiresCachePreparation': True, 'loaders': [{'nodeId': 'provider',
        'modelType': 'ImageTextToText', 'workingDemandEvidence': {'kind': 'reviewed_bounded_image_text_storage',
            'cpuPreparationRequirements': {'systemRamBytes': 268435456, 'vramBytes': 0, 'diskFreeBytes': 0}},
        'requirements': {'systemRamBytes': 1, 'vramBytes': 1, 'diskFreeBytes': 0},
        'parameterResidencyRequirements': {'systemRamBytes': 0, 'vramBytes': 1},
        'parameterLoadingRequirements': {'systemRamBytes': 1, 'vramBytes': 1},
        'capacityRequirements': {}, 'consumers': ['proposal'], 'workingMemoryPolicy': 'runtime_headroom_policy',
        'settings': {'device': 'cuda:0', 'dtype': 'bfloat16', 'quantizationMode': 'bnb_4bit', 'offloadMode': 'none'}}]}
    if fault == 'owner': plan['loaders'][0]['nodeId'] = 'foreign'
    elif fault == 'graph': plan['plannedGraphHash'] = 'foreign'
    elif fault == 'budget-kind': plan['loaders'][0]['workingDemandEvidence']['kind'] = 'client-budget'
    elif fault == 'schedule': plan['schedule'] = {'owners': []}
    elif fault == 'preparation': plan['requiresPreparation'] = True
    elif fault == 'budget-missing': plan['loaders'][0].pop('parameterLoadingRequirements')
    elif fault == 'budget-bool': plan['loaders'][0]['requirements']['vramBytes'] = True
    def callback():
        with owner_scope(app, graph, graph['runtimeHints']['executionPurpose']):
            bind_preview_auto_plan(app, plan)
            result, cleanup = app._prepare_workflow_auto_cache(graph, plan)
            assert result is plan
            assert cleanup == {'performed': False, 'scopedUntilProcessorPreflight': True}
    async def run():
        app.loop = asyncio.get_running_loop()
        return await app._with_node_cache_lease(lambda: app._run_executor_callback(callback, serialize_model_io=True))
    try:
        if fault == 'none': asyncio.run(run())
        else:
            with pytest.raises(ValueError, match='server-computed Auto owner budget'):
                asyncio.run(run())
    finally:
        if app._model_executor is not None:
            app._model_executor.shutdown(wait=True)
            app._model_executor = None


def test_client_values_cannot_mint_an_actual_dispatch_observation(tmp_path, monkeypatch):
    import modules
    from modiff.ancillary_owner_lifecycle import observe_claimed_attempt, owner_scope
    graph, _parent = ancillary.preview_graph(modules.MODULE_MAP)
    app = app_for(tmp_path, monkeypatch, graph)
    observe_claimed_attempt(app, graph)
    assert not hasattr(app, '_ancillary_owner_attempt')
    with pytest.raises(ValueError, match='serialized'):
        with owner_scope(app, graph, graph['runtimeHints']['executionPurpose']):
            pytest.fail('No nonleased scope')
    assert not hasattr(app, '_ancillary_dispatched_origins')


@pytest.mark.parametrize('mutation', ['output', 'inputs', 'implementation', 'live_implementation', 'manager', 'active', 'task', 'worker', 'purpose'])
def test_changed_owner_or_attempt_refuses_before_consumer_and_release(actual_pup, tmp_path, monkeypatch, mutation):
    import torch
    app, graph = cold_preview(actual_pup, tmp_path, monkeypatch)
    calls = []
    quiet_dispatch(app, calls.append)
    from modiff import ancillary_owner_lifecycle as lifecycle
    original_record = lifecycle.record_dispatched_owner
    def record(server, node_id):
        original_record(server, node_id)
        if node_id != 'provider':
            return
        owner = app.node_cache[node_id]
        if mutation == 'output': owner.output = dict(owner.output)
        elif mutation == 'inputs': owner._cache_input_snapshot = ('foreign',)
        elif mutation == 'implementation': owner._cache_implementation = None
        elif mutation == 'live_implementation': monkeypatch.setattr(type(owner), 'execute', lambda _self, **_kwargs: {})
        elif mutation == 'manager': memory_manager.cache[owner._mm_models[0]]['model'] = torch.nn.Linear(2, 2)
        elif mutation == 'active': memory_manager._active_ids.update(owner._mm_models)
        elif mutation == 'task': app.current_task['attempt_index'] += 1
        elif mutation == 'worker': app.instance += '-replacement'
        elif mutation == 'purpose': graph['runtimeHints']['executionPurpose']['bindings'][0]['parent']['authoringContentHash'] = 'studio-authoring-v1-00000000'
    monkeypatch.setattr(lifecycle, 'record_dispatched_owner', record)
    with pytest.raises(ValueError, match='owner boundary'):
        leased(app, graph)
    assert calls == ['provider'] and 'provider' in app.node_cache
    assert not any(message['type'] == 'graph_completed' for message in app.messages)


def test_external_handle_is_stopped_after_release_without_completion_or_reload(actual_pup, tmp_path, monkeypatch):
    app, graph = cold_preview(actual_pup, tmp_path, monkeypatch)
    external = []
    def retain(node_id):
        if node_id == 'provider':
            external.append(app.node_cache[node_id].output['model'])
    quiet_dispatch(app, retain)
    with pytest.raises(ValueError, match='model references remain alive'):
        leased(app, graph)
    assert len(external) == 1 and 'provider' not in app.node_cache and not memory_manager.cache
    assert app.node_cache['proposal'].output['proposal_summary']
    assert not any(message['type'] == 'graph_completed' for message in app.messages)


@pytest.mark.parametrize('foreign', ['dependent', 'shared', 'unregistered'])
def test_foreign_retaining_or_shared_owner_refuses_before_removal(actual_pup, tmp_path, monkeypatch, foreign):
    app, graph = cold_preview(actual_pup, tmp_path, monkeypatch)
    def retain(node_id):
        if node_id == 'proposal':
            owner = app.node_cache['provider']
            app.node_cache['foreign'] = SimpleNamespace(output={'model': owner.output['model']} if foreign == 'unregistered' else {}, params={},
                _cache_input_sources={'provider'} if foreign == 'dependent' else set(),
                _mm_models=list(owner._mm_models) if foreign == 'shared' else [])
    quiet_dispatch(app, retain)
    with pytest.raises(ValueError, match='foreign|shared'):
        leased(app, graph)
    assert {'provider', 'proposal', 'foreign'} <= app.node_cache.keys()
    assert memory_manager.cache


def test_retained_opaque_output_refuses_before_removal(actual_pup, tmp_path, monkeypatch):
    app, graph = cold_preview(actual_pup, tmp_path, monkeypatch)
    def inject(node_id):
        if node_id == 'proposal':
            app.node_cache[node_id].output['result'] = object()
    quiet_dispatch(app, inject)
    with pytest.raises(ValueError, match='opaque'):
        leased(app, graph)
    assert 'provider' in app.node_cache and memory_manager.cache


def test_missing_native_summary_refuses_before_removal(actual_pup, tmp_path, monkeypatch):
    app, graph = cold_preview(actual_pup, tmp_path, monkeypatch)
    def missing(node_id):
        if node_id == 'proposal':
            app.node_cache[node_id].output['proposal_summary'] = None
    quiet_dispatch(app, missing)
    with pytest.raises(ValueError, match='summary'):
        leased(app, graph)
    assert 'provider' in app.node_cache and memory_manager.cache


@pytest.mark.parametrize('trim', [(True, []), (False, []), (False, ['explicit allocator error'])])
def test_allocator_result_is_observed_after_removal(trim):
    from modiff.workflow_auto_lifecycle import release_owner_caches
    app = SimpleNamespace(node_cache={'owner': SimpleNamespace(output={}, params={}, _mm_models=[])},
        _release_node_modular_components=lambda _ids: 0, _best_effort_device_cache_clear=lambda: [],
        _best_effort_allocator_trim=lambda: trim)
    event = {'ownerIds': ['owner'], 'nodeIds': ['owner'], 'retainOutputs': {}}
    if trim[1]:
        with pytest.raises(ValueError, match='allocator'):
            release_owner_caches(app, event, SimpleNamespace(cache={}))
    else:
        assert release_owner_caches(app, event, SimpleNamespace(cache={}))['releasedNodeIds'] == ['owner']
    assert not app.node_cache


def test_malformed_allocator_contract_stays_an_error_after_removal():
    from modiff.workflow_auto_lifecycle import release_owner_caches
    app = SimpleNamespace(node_cache={'owner': SimpleNamespace(output={}, params={}, _mm_models=[])},
        _release_node_modular_components=lambda _ids: 0, _best_effort_device_cache_clear=lambda: [],
        _best_effort_allocator_trim=lambda: None)
    with pytest.raises(TypeError):
        release_owner_caches(app, {'ownerIds': ['owner'], 'nodeIds': ['owner'], 'retainOutputs': {}}, SimpleNamespace(cache={}))
    assert not app.node_cache


def run_actual_worker(app, graph, *, queued_mutation=None, legacy_cleanup=False):
    """Queue through the real server, run its worker/finally, retain terminal status."""
    async def no_history(_operation, *_args):
        return None
    app._run_studio_history_mutation = no_history
    app._record_auto_resource_failure = lambda *_args: None
    statuses = []
    app._record_terminal_task = lambda status, **_kwargs: statuses.append(status) or {'completed_at': 0}
    if not legacy_cleanup:
        app._release_runtime_caches_for_retry = lambda: pytest.fail('Private Preview must preserve foreign/uncertain owners')
    async def run():
        app.loop = asyncio.get_running_loop()
        future = app.loop.create_future()
        await app.queue_task(app.execute_graph, (graph,), future, graph['sid'], name='Graph execution')
        if queued_mutation is not None:
            queued_mutation()
        worker = asyncio.create_task(app._main_worker())
        try:
            await asyncio.wait_for(app.main_queue.join(), 15)
            if future.done():
                try:
                    future.result()
                except (Exception, asyncio.CancelledError):
                    pass  # Actual terminal status and preserved ownership are asserted below.
        finally:
            await app.main_queue.put(None)
            await worker
    try:
        asyncio.run(run())
    finally:
        if app._model_executor is not None:
            app._model_executor.shutdown(wait=True)
            app._model_executor = None
    assert not hasattr(app, '_ancillary_owner_attempt'), 'private attempt marker must clear at terminal'
    return statuses


@pytest.mark.parametrize('stale_after_queue', [False, True])
def test_actual_failed_worker_preserves_foreign_owners_even_for_stale_queued_claim(tmp_path, monkeypatch, stale_after_queue):
    import modules
    import torch
    graph, _parent = ancillary.preview_graph(modules.MODULE_MAP)
    graph['runtimeHints']['resourceMode'] = 'expert'
    app = app_for(tmp_path, monkeypatch, graph)
    model = torch.nn.Linear(2, 2)
    model_id = memory_manager.add(model)
    foreign = SimpleNamespace(output={'model': model}, params={}, _mm_models=[model_id])
    app.node_cache['foreign'] = foreign
    def stale():
        graph['runtimeHints']['executionPurpose']['bindings'][0]['contentHash'] = 'studio-ancillary-v1-00000000'
    assert run_actual_worker(app, graph, queued_mutation=stale if stale_after_queue else None) == ['failed']
    assert app.node_cache['foreign'] is foreign
    assert memory_manager.get_model(model_id) is model


def test_actual_inactive_quantized_preview_worker_preserves_foreign_owner_before_scope(tmp_path, monkeypatch):
    """Actual optional-runtime refusal precedes the old inner preservation seam."""
    import modules
    import torch
    graph, _parent = ancillary.preview_graph(modules.MODULE_MAP)
    graph['runtimeHints']['resourceMode'] = 'expert'
    graph['nodes']['provider']['params']['quantization_mode'] = {'value': 'bnb_4bit'}
    graph['nodes']['provider']['params']['dtype'] = {'value': 'bfloat16'}
    app = app_for(tmp_path, monkeypatch, graph)
    model = torch.nn.Linear(2, 2)
    model_id = memory_manager.add(model)
    foreign = SimpleNamespace(output={'model': model}, params={}, _mm_models=[model_id])
    app.node_cache['foreign'] = foreign
    app.execute_node = lambda *_args: pytest.fail('inactive optional runtime must refuse before factories')
    assert run_actual_worker(app, graph, legacy_cleanup=True) == ['failed']
    assert app.node_cache.get('foreign') is foreign, 'early quantization refusal erased an unrelated cached owner'
    assert memory_manager.get_model(model_id) is model


def test_actual_cancelled_worker_preserves_uncertain_own_model_after_callback_drains(actual_pup, tmp_path, monkeypatch):
    app, graph = cold_preview(actual_pup, tmp_path, monkeypatch)
    def stop(node_id):
        if node_id == 'provider':
            assert app._node_cache_lock.locked()
            app.current_task['interrupt_requested'] = True
            app.interrupt_flag = True
    quiet_dispatch(app, stop)
    assert run_actual_worker(app, graph) == ['cancelled']
    assert 'provider' in app.node_cache and memory_manager.cache
    assert not any(message['type'] == 'graph_completed' for message in app.messages)


def test_unknown_cache_properties_are_not_executed_during_refusal(tmp_path, monkeypatch):
    import modules
    graph, _parent = ancillary.preview_graph(modules.MODULE_MAP)
    graph['runtimeHints']['resourceMode'] = 'expert'
    app = app_for(tmp_path, monkeypatch, graph)
    class Unknown:
        @property
        def __dict__(self):
            raise AssertionError('unreviewed cache property was executed')
    app.node_cache['unknown'] = Unknown()
    with pytest.raises(ValueError, match='unknown|foreign'):
        leased(app, graph)


@pytest.mark.parametrize('mutation', ['workflow_kind', 'remove', 'worker_drift'])
def test_queue_issued_preservation_survives_live_claim_or_worker_drift(tmp_path, monkeypatch, mutation):
    import modules
    from modiff.ancillary_owner_lifecycle import preserve_failed_attempt
    graph, _parent = ancillary.preview_graph(modules.MODULE_MAP)
    graph['runtimeHints']['resourceMode'] = 'expert'
    app = app_for(tmp_path, monkeypatch, graph)
    async def no_history(_operation, *_args):
        return None
    app._run_studio_history_mutation = no_history
    async def queued():
        task_id = await app.queue_task(app.execute_graph, (graph,), None, None, name='Graph execution')
        app.current_task = {'task_id': task_id, 'args': [graph], 'attempt_index': 0}
    asyncio.run(queued())
    app.node_cache['foreign'] = SimpleNamespace(output={}, params={}, _mm_models=[])
    if mutation == 'workflow_kind': graph['runtimeHints']['executionPurpose']['kind'] = 'workflow'
    elif mutation == 'remove': graph['runtimeHints'].pop('executionPurpose')
    prepared = []
    def preparation(_hints):
        prepared.append(True)
        raise ValueError('explicit refusal')
    app._prepare_auto_runtime_for_graph = preparation
    if mutation == 'worker_drift':
        from modiff import ancillary_owner_lifecycle as lifecycle
        real = lifecycle.observe_claimed_attempt
        def observe(server, value):
            real(server, value)
            server.instance += '-changed'
        monkeypatch.setattr(lifecycle, 'observe_claimed_attempt', observe)
    try:
        leased(app, graph)
    except ValueError:
        pass
    assert preserve_failed_attempt(app), 'actual queued Preview must not fall into global failure cleanup'
    assert not prepared, 'removing a queued claim cannot re-enable global preparation'


def test_actual_successful_worker_releases_own_owner_before_terminal_and_clears_marker(actual_pup, tmp_path, monkeypatch):
    app, graph = cold_preview(actual_pup, tmp_path, monkeypatch)
    quiet_dispatch(app)
    assert run_actual_worker(app, graph) == ['completed']
    assert 'provider' not in app.node_cache and not memory_manager.cache
    assert app.node_cache['proposal'].output['proposal_summary']
    assert any(message['type'] == 'graph_completed' for message in app.messages)


def test_unclaimed_failed_worker_keeps_historical_cleanup(tmp_path, monkeypatch):
    graph = ancillary.graph()
    app = app_for(tmp_path, monkeypatch, graph)
    app._prepare_auto_runtime_for_graph = lambda _hints: None
    calls = []
    def failed(*_args):
        raise RuntimeError('explicit unclaimed graph fault')
    app.execute_node = failed
    app._release_runtime_caches_for_retry = lambda: calls.append(True) or {'released': {}, 'errors': []}
    assert run_actual_worker(app, graph, legacy_cleanup=True) == ['failed']
    assert calls == [True]


def test_graph_argument_drift_preserves_the_same_issued_task(tmp_path, monkeypatch):
    import modules
    from modiff.ancillary_owner_lifecycle import observe_claimed_attempt, preserve_failed_attempt
    graph, _parent = ancillary.preview_graph(modules.MODULE_MAP)
    app = app_for(tmp_path, monkeypatch, graph)
    async def run():
        app.loop = asyncio.get_running_loop()
        def actual_callback():
            observe_claimed_attempt(app, graph)
            app.current_task['args'] = [deepcopy(graph)]
        await app._with_node_cache_lease(lambda: app._run_executor_callback(actual_callback, serialize_model_io=True))
    try:
        asyncio.run(run())
        assert preserve_failed_attempt(app), 'graph drift cannot grant global destruction in the same issued task'
    finally:
        app._model_executor.shutdown(wait=True)


def test_real_production_manager_pressure_policy_remains_operative(monkeypatch):
    import torch
    from utils.memory_menager import MemoryManager
    # A controlled low observation tests policy, never model fit or physical RAM.
    monkeypatch.setattr('modiff.hardware.system_memory_snapshot', lambda: {
        'available_bytes': 1024**3, 'total_bytes': 32 * 1024**3,
    })
    manager = MemoryManager()
    assert manager.policy == 'lru'
    first = manager.add(torch.nn.Linear(2, 2))
    second = manager.add(torch.nn.Linear(2, 2))
    assert first not in manager.cache and second in manager.cache
