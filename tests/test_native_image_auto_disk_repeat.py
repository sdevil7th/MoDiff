"""Repeated native image Auto placement with actual tiny official disk hooks.

Only CPU hook installation/serialization runs. Declaring a CUDA onload target
does not execute a forward or access CUDA; finite hardware/header fixtures are
not live model-fit evidence.
"""
from copy import deepcopy
from types import SimpleNamespace
import weakref

import pytest
import torch

import test_native_image_auto_placement as placement
import test_native_image_disk_lora as disk_lora
from modiff import workflow_auto_resource

placement_runtime = placement.placement_runtime
native_runtime = disk_lora.native_runtime
graph_runtime = disk_lora.graph_runtime


def install_disk_hooks(model, directory, monkeypatch, *, onload='cuda:0'):
    from diffusers.hooks import apply_group_offloading

    monkeypatch.setattr(torch.accelerator, 'current_accelerator', lambda: torch.device('cpu'))
    apply_group_offloading(model, onload_device=onload, offload_device='cpu',
                           offload_type='leaf_level', use_stream=False,
                           low_cpu_mem_usage=True, offload_to_disk_path=str(directory))


def warm_disk_owner(monkeypatch, tmp_path):
    graph = placement._graph('group_disk')
    app, manager, cached = placement._live_native_cache(graph, monkeypatch)
    for name in ('transformer', 'text_encoder', 'vae'):
        install_disk_hooks(getattr(cached.loader, name), tmp_path / name, monkeypatch)
    return app, manager, cached


def test_actual_disk_placeholders_are_not_resident_or_reclaimable_cpu_weights(monkeypatch, tmp_path):
    app, manager, cached = warm_disk_owner(monkeypatch, tmp_path)
    # This independent component is genuine CPU storage and must stay counted.
    ordinary = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    manager.add('ordinary', ordinary, collection='unrelated')
    snapshot = app._workflow_auto_cache_snapshot()
    assert snapshot['owners']['models']['loadingPhaseReusable'] is True
    assert snapshot['owners']['models']['systemRamBytes'] == 0
    assert snapshot['reclaimable']['systemRamBytes'] == 32
    # Invalidating the owner must not restore placeholder reclamation credit.
    cached._cache_valid = False
    snapshot = app._workflow_auto_cache_snapshot()
    assert not snapshot['owners']
    assert snapshot['reclaimable']['systemRamBytes'] == 32


def test_a_valid_plain_owner_alias_cannot_revive_disk_placeholder_storage(monkeypatch, tmp_path):
    app, manager, cached = warm_disk_owner(monkeypatch, tmp_path)
    alias = type(cached)()
    alias.module_name, alias.class_name = cached.module_name, cached.class_name
    alias._cache_valid, alias._cache_invalidated = True, False
    alias._mm_models, alias._cache_input_sources, alias.params = [], (), {}
    alias.loader = type(cached)()
    alias.loader.transformer = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    alias.loader.transformer.weight = cached.loader.transformer.weight
    component_id = manager.add('alias', alias.loader.transformer, collection='alias')
    alias.output = {'pipeline_components': {'transformer': {'model_id': component_id}}}
    alias._loader_diagnostics = {'required_components': ['transformer']}
    app.node_cache['alias'] = alias
    graph = {'nodes': {'alias': placement._graph()['nodes']['models']}, 'paths': [['alias']]}
    app._record_workflow_auto_owner({
        'nodeId': 'alias', 'cacheKey': workflow_auto_resource.workflow_owner_cache_key(graph, 'alias'),
        'parameterResidencyRequirements': {'systemRamBytes': 32, 'vramBytes': 0},
    })
    snapshot = app._workflow_auto_cache_snapshot()
    assert set(snapshot['owners']) == {'models', 'alias'}
    assert all(owner['loadingPhaseReusable'] for owner in snapshot['owners'].values())
    assert all(owner['systemRamBytes'] == 0 for owner in snapshot['owners'].values())
    assert snapshot['reclaimable']['systemRamBytes'] == 0


@pytest.mark.parametrize('onload', ['cpu', 'cuda:0'])
def test_installed_hook_inspection_preserves_unrelated_real_cpu_storage(monkeypatch, tmp_path, onload):
    app, manager, cached = placement._live_native_cache(placement._graph('group_disk'), monkeypatch)
    for name in ('transformer', 'text_encoder', 'vae'):
        model = getattr(cached.loader, name)
        install_disk_hooks(model, tmp_path / name, monkeypatch, onload=onload)
        # Added after installation, this buffer is not a disk-group placeholder.
        model.register_buffer('ordinary', torch.ones(2, dtype=torch.bfloat16))
        monkeypatch.setattr(model, 'parameters', lambda *args, **kwargs: pytest.fail('model method ran'))
        monkeypatch.setattr(model, 'buffers', lambda *args, **kwargs: pytest.fail('model method ran'))
        monkeypatch.setattr(model._diffusers_hook, 'get_hook', lambda *args, **kwargs: pytest.fail('hook method ran'))
    snapshot = app._workflow_auto_cache_snapshot()
    expected = 12 + (96 if onload == 'cpu' else 0)
    assert snapshot['owners']['models']['systemRamBytes'] == expected
    assert snapshot['reclaimable']['systemRamBytes'] == expected


@pytest.mark.parametrize('gpu_free', [15 * placement.GIB, 2 * placement.GIB])
def test_original_authored_placement_prefers_complete_warm_disk_over_projected_cleanup(
    placement_runtime, monkeypatch, tmp_path, gpu_free,
):
    plan, hardware, bound, _, _ = placement_runtime
    placement._small_parameter_bound(bound)
    app, manager, cached = warm_disk_owner(monkeypatch, tmp_path)
    # Genuine unrelated CPU weights make a cold cleanup proposal appear viable.
    # This exposes both original-result and first-alternative short circuits.
    manager.add('ordinary', torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16), collection='unrelated')
    hardware['systemMemory']['availableBytes'] = 4 * placement.GIB
    hardware['accelerator']['freeBytes'] = gpu_free
    graph = placement._graph()  # The template remains authored as model_cpu.
    before = deepcopy(graph)
    snapshot = app._workflow_auto_cache_snapshot()
    result = plan(graph, cache_snapshot=snapshot)
    assert result['canAutoRun'], result['issues']
    assert result['loaders'][0]['settings']['offloadMode'] == 'group_disk'
    assert result['reusedOwnerIds'] == ['models']
    assert not result['requiresCachePreparation']
    assert result['schedule'] is None
    assert result['requirements']['systemRamBytes'] == 4 * placement.GIB
    assert app.node_cache['models'] is cached
    assert graph == before
    placement._assert_only_placement_changes(graph, result)


@pytest.mark.parametrize('drift', ['partial_eviction', 'component_replaced', 'owner_replaced',
                                 'vae_policy', 'revision', 'scheduled', 'unknown_headers'])
def test_disk_repeat_never_grants_warm_credit_to_an_incompatible_owner(
    placement_runtime, monkeypatch, tmp_path, drift,
):
    plan, hardware, bound, _, _ = placement_runtime
    placement._small_parameter_bound(bound)
    app, manager, cached = warm_disk_owner(monkeypatch, tmp_path)
    hardware['systemMemory']['availableBytes'] = 4 * placement.GIB
    graph = placement._graph()
    if drift == 'partial_eviction':
        manager.components.pop(cached.output['pipeline_components']['vae']['model_id'])
    elif drift == 'component_replaced':
        key = cached.output['pipeline_components']['vae']['model_id']
        manager.components[key] = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    elif drift == 'owner_replaced':
        app.node_cache['models'] = object()
    elif drift == 'vae_policy':
        graph['nodes']['models']['params']['vae_tiling']['value'] = True
    elif drift == 'revision':
        graph['nodes']['models']['params']['revision']['value'] = 'a' * 40
    elif drift == 'scheduled':
        graph = placement._two_graphs()
    elif drift == 'unknown_headers':
        monkeypatch.setattr(placement.auto_resource, 'artifact_parameter_lower_bound', lambda *args, **kwargs: None)
    snapshot = app._workflow_auto_cache_snapshot()
    # No projected cleanup credit participates in this warm-identity witness.
    snapshot.pop('reclaimable', None)
    result = plan(graph, cache_snapshot=snapshot)
    assert result['reusedOwnerIds'] == []
    if drift == 'unknown_headers':
        assert result['loaders'][0]['settings']['offloadMode'] == 'model_cpu'
    else:
        assert not result['canAutoRun']
        assert result['patches'] == []


def test_shared_memory_does_not_acquire_new_dedicated_cuda_warm_preference(
    placement_runtime, monkeypatch, tmp_path,
):
    plan, hardware, bound, _, _ = placement_runtime
    placement._small_parameter_bound(bound)
    app, manager, _ = warm_disk_owner(monkeypatch, tmp_path)
    manager.add('ordinary', torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16), collection='unrelated')
    hardware['systemMemory']['availableBytes'] = 6 * placement.GIB
    hardware['accelerator'].update(memoryKind='shared', sharedMemory=True, freeBytes=2 * placement.GIB)
    result = plan(placement._graph(), cache_snapshot=app._workflow_auto_cache_snapshot())
    assert result['sharedMemory'] is True
    assert result['requiresCachePreparation']
    assert result['reusedOwnerIds'] == []


def test_placeholder_only_cleanup_cannot_claim_cold_placement_fit(
    placement_runtime, monkeypatch, tmp_path,
):
    plan, hardware, bound, _, _ = placement_runtime
    placement._small_parameter_bound(bound)
    app, _, cached = warm_disk_owner(monkeypatch, tmp_path)
    cached._cache_valid = False
    hardware['systemMemory']['availableBytes'] = 4 * placement.GIB
    result = plan(placement._graph(), cache_snapshot=app._workflow_auto_cache_snapshot())
    assert not result['canAutoRun']
    assert not result['requiresCachePreparation']
    assert result['patches'] == []


def test_warm_preference_keeps_the_shared_evaluation_cap(
    placement_runtime, monkeypatch, tmp_path,
):
    plan, hardware, bound, _, _ = placement_runtime
    placement._small_parameter_bound(bound)
    app, manager, _ = warm_disk_owner(monkeypatch, tmp_path)
    manager.add('ordinary', torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16), collection='unrelated')
    hardware['systemMemory']['availableBytes'] = 4 * placement.GIB
    monkeypatch.setattr(workflow_auto_resource, '_PLACEMENT_EVALUATION_LIMIT', 1)
    ordinary = workflow_auto_resource._build_workflow_auto_plan
    evaluations = []

    def observe(*args, **kwargs):
        result = ordinary(*args, **kwargs)
        evaluations.append(deepcopy(result))
        return result

    monkeypatch.setattr(workflow_auto_resource, '_build_workflow_auto_plan', observe)
    result = plan(placement._graph(), cache_snapshot=app._workflow_auto_cache_snapshot())
    assert len(evaluations) == 1
    assert result == evaluations[0] | {'resolvedFields': result['resolvedFields']}


def test_preference_and_ordinary_search_share_one_cap_and_memoized_evaluations(
    placement_runtime, monkeypatch,
):
    plan, _, _, _, _ = placement_runtime
    monkeypatch.setattr(workflow_auto_resource, '_PLACEMENT_EVALUATION_LIMIT', 6)
    ordinary = workflow_auto_resource._build_workflow_auto_plan
    choices = []
    results = []

    def observe(*args, **kwargs):
        choices.append(tuple(sorted(kwargs['_placement_choices'].items())))
        result = ordinary(*args, **kwargs)
        results.append(deepcopy(result))
        return result

    monkeypatch.setattr(workflow_auto_resource, '_build_workflow_auto_plan', observe)
    graph = placement._two_graphs()
    before = deepcopy(graph)
    result = plan(graph)
    assert len(choices) == 6
    assert len(set(choices)) == len(choices)
    assert any(len(choice) == 2 for choice in choices), 'The ordinary search did not share the budget.'
    assert not result['canAutoRun'] and result['patches'] == []
    assert result['loaders'] == results[0]['loaders']
    assert graph == before


def test_actual_auto_dispatch_records_disk_placement_and_retains_the_owner_on_authored_repeat(
    graph_runtime, placement_runtime, monkeypatch, tmp_path,
):
    """Actual Graph/ModelsLoader/cache dispatch with tiny CPU-only components.

    Finite artifact/hardware admission is synthetic, and no image inference or
    accelerator forward runs. The server must publish the successful key itself.
    """
    server, loaders, manager, events, pipelines = graph_runtime
    plan, hardware, _, _, _ = placement_runtime
    monkeypatch.setattr(disk_lora.TinyPipeline, 'pretrained_component_names',
                        ['transformer', 'text_encoder', 'vae', 'scheduler'])
    load = disk_lora.TinySpec.load

    def load_complete_component(spec, **kwargs):
        if spec.name != 'scheduler':
            return load(spec, **kwargs)
        from diffusers import FlowMatchEulerDiscreteScheduler
        events.append(('load', 'scheduler'))
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler._diffusers_load_id = spec.load_id
        return scheduler

    monkeypatch.setattr(disk_lora.TinySpec, 'load', load_complete_component)

    def offload(pipeline, *, component_names, **kwargs):
        for name in component_names:
            model = getattr(pipeline, name, None)
            if isinstance(model, torch.nn.Module) and not hasattr(model, '_diffusers_hook'):
                install_disk_hooks(model, tmp_path / ('actual-dispatch-' + name), monkeypatch)
                events.append(('hook', name))
        return SimpleNamespace(applied=True, components=list(component_names), mode='group_disk',
                               method='group_offload', disk_path=None, detail=None)

    monkeypatch.setattr(loaders, 'apply_component_group_offload', offload)
    # Keep the registered native task actions and ports. Their tiny CPU execute
    # stand-ins exercise dispatch/cache publication, not diffusion inference.
    import modules.ModularDiffusers.main as native
    from PIL import Image
    monkeypatch.setattr(native.EncodePrompt, 'execute', lambda self, **kwargs:
                        {'embeddings': torch.zeros(1)})
    monkeypatch.setattr(native.Denoise, 'execute', lambda self, **kwargs:
                        {'latents': torch.zeros(1)})
    monkeypatch.setattr(native.DecodeLatents, 'execute', lambda self, **kwargs:
                        {'images': [Image.new('RGB', (4, 4))]})
    monkeypatch.setattr(server, '_build_workflow_auto_plan', lambda graph, **kwargs:
                        plan(graph, cache_snapshot=server._workflow_auto_cache_snapshot()))
    server._release_runtime_caches_for_retry = lambda: pytest.fail('An unchanged warm owner was destroyed.')
    authored = placement._graph()
    # This native loader resolves its unique profile from its actual contract;
    # execution_profile_id is planner fixture metadata, not its executable input.
    authored['nodes']['models']['params'].pop('execution_profile_id')
    authored['nodes']['preview']['params']['export'] = {
        'value': server.modules['modules.Image']['Preview']['params']['export']['default']}
    authored['sid'] = 'unit'

    def execute(current):
        server.current_task = {'task_id': 'tiny-task', 'sid': 'unit', 'status': 'running', 'progress': 0}
        server._execute_graph(deepcopy(current))
        assert server.node_cache['decode'].output['images'][0].size == (4, 4)

    def submitted_graph():
        current = deepcopy(authored)
        selected = plan(current, cache_snapshot=server._workflow_auto_cache_snapshot())
        assert selected['canAutoRun'], selected['issues']
        assert selected['loaders'][0]['settings']['offloadMode'] == 'group_disk'
        assert not selected['requiresCachePreparation']
        for patch in selected['patches']:
            current['nodes'][patch['nodeId']]['params'][patch['field']]['value'] = patch['value']
        current['runtimeHints'] = {'resourceMode': 'auto', 'workflowAutoPlan': {
            'schemaVersion': 1, 'graphHash': workflow_auto_resource.workflow_graph_hash(current)}}
        return current

    execute(submitted_graph())
    initial = weakref.ref(server.node_cache['models'].loader)
    snapshot = server._workflow_auto_cache_snapshot()
    assert 'models' in snapshot['owners'], {
        'records': server._workflow_auto_owner_records,
        'diagnostics': server.node_cache['models']._loader_diagnostics,
        'bundle': server.node_cache['models'].output['pipeline_components'],
        'collection': server.node_cache['models'].loader.manager.collections,
    }
    assert snapshot['owners']['models']['loadingPhaseReusable'] is True
    assert snapshot['owners']['models']['cacheKey'] == workflow_auto_resource.workflow_owner_cache_key(
        submitted_graph(), 'models')
    assert snapshot['owners']['models']['systemRamBytes'] == 0
    # With real unrelated weights, the original model_cpu proposal fits only
    # through projected destruction. A genuinely complete disk owner fits now.
    unrelated = manager.add('ordinary', torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16),
                            collection='unrelated')
    hardware['systemMemory']['availableBytes'] = placement.PARAMETERS['totalBytes'] + 4 * placement.GIB - 256
    execute(submitted_graph())
    authored['nodes']['prompt']['params']['prompt']['value'] = 'prompt only'
    execute(submitted_graph())
    assert server.node_cache['prompt'].params['prompt'] == 'prompt only'
    assert server.node_cache['models'].loader is initial() and len(pipelines) == 1
    assert unrelated in manager.components
