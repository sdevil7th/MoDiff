"""Cold GPU parameter minima and warm-cache controls, without GPU execution.

The actual native profile and both planners come from the placement fixture.
Only finite hardware/cache/header samples and the explicit placement override
are controlled; these are resource-contract regressions, not live fit proof.
"""

from copy import deepcopy

import pytest

from modiff import workflow_auto_resource
from test_native_image_auto_placement import GIB, PARAMETERS, _graph, _two_graphs, placement_runtime as _placement_runtime

placement_runtime = _placement_runtime


def _gpu_case(runtime, mode):
    plan, hardware, _, overrides, _ = runtime
    overrides['offloadMode'] = mode  # Keep this accepted placement under test.
    # The existing fully resident Z tier requires 80 GiB GPU / 64 GiB host;
    # satisfy those unchanged totals so that the none case isolates free bytes.
    hardware['systemMemory'].update(totalBytes=(80 if mode == 'none' else 32) * GIB,
                                    availableBytes=(60 if mode == 'none' else 30) * GIB)
    hardware['accelerator'].update(totalBytes=(96 if mode == 'none' else 16) * GIB,
                                   freeBytes=(12 if mode == 'none' else 3) * GIB)
    return plan, hardware, _graph(mode)


@pytest.mark.parametrize('mode', ['none', 'model_cpu'])
def test_cold_native_gpu_parameters_cannot_fit_from_total_capacity_and_headroom_alone(placement_runtime, mode):
    plan, hardware, graph = _gpu_case(placement_runtime, mode)
    required = PARAMETERS['totalBytes'] if mode == 'none' else PARAMETERS['largestComponentBytes']
    assert required < hardware['accelerator']['totalBytes']
    assert required > hardware['accelerator']['freeBytes']
    before = deepcopy(graph)
    result = plan(graph)
    assert not result['canAutoRun'], 'Total GPU capacity does not make a cold component allocation available.'
    assert result['patches'] == []
    assert graph == before


@pytest.mark.parametrize('mode', ['none', 'model_cpu'])
def test_intact_native_gpu_owner_keeps_pool_specific_storage_credit(placement_runtime, mode):
    plan, _, graph = _gpu_case(placement_runtime, mode)
    gpu_weights = PARAMETERS['totalBytes'] if mode == 'none' else PARAMETERS['largestComponentBytes']
    cache = {'owners': {'models': {
        'cacheKey': workflow_auto_resource.workflow_owner_cache_key(graph, 'models'),
        'loadingPhaseReusable': True,
        'systemRamBytes': PARAMETERS['totalBytes'] - gpu_weights,
        'vramBytes': gpu_weights,
    }}}
    result = plan(graph, cache_snapshot=cache)
    assert result['canAutoRun'], result['issues']
    assert result['reusedOwnerIds'] == ['models']


def test_exact_offloaded_owner_still_needs_observed_free_capacity_beyond_advisory_reservations(placement_runtime):
    plan, hardware, graph = _gpu_case(placement_runtime, 'model_cpu')
    gpu_weights = PARAMETERS['components']['text_encoder']
    # Inactive allocator reservations are advisory. Their release must occur
    # under the existing server lease and be followed by an actual fresh sample;
    # this pure planner cannot turn the five GiB counter into free capacity.
    hardware['runtime'] = {'hardware': {'devices': [{
        'type': 'cuda', 'index': 0, 'memory_kind': 'dedicated',
        'torch_vram_total': 16 * GIB, 'torch_vram_free': 3 * GIB,
        'torch_allocated': gpu_weights, 'torch_reserved': gpu_weights + 5 * GIB,
    }]}}
    cache = {'owners': {'models': {
        'cacheKey': workflow_auto_resource.workflow_owner_cache_key(graph, 'models'),
        'loadingPhaseReusable': True,
        'systemRamBytes': PARAMETERS['totalBytes'] - gpu_weights,
        'vramBytes': gpu_weights,
    }}}
    before = deepcopy(graph)
    result = plan(graph, cache_snapshot=cache)
    assert not result['canAutoRun']
    assert result['patches'] == []
    assert result['reusedOwnerIds'] == ['models']
    assert graph == before


def test_fresh_sample_can_admit_the_same_exact_owner_after_unused_reservations_are_released(placement_runtime):
    plan, hardware, graph = _gpu_case(placement_runtime, 'model_cpu')
    gpu_weights = PARAMETERS['components']['text_encoder']
    cache = {'owners': {'models': {
        'cacheKey': workflow_auto_resource.workflow_owner_cache_key(graph, 'models'),
        'loadingPhaseReusable': True,
        'systemRamBytes': PARAMETERS['totalBytes'] - gpu_weights,
        'vramBytes': gpu_weights,
    }}}
    hardware['accelerator']['freeBytes'] = 8 * GIB
    before = deepcopy(graph)
    result = plan(graph, cache_snapshot=cache)
    assert result['canAutoRun'], result['issues']
    assert result['reusedOwnerIds'] == ['models']
    assert result['patches'] == []
    assert graph == before


def test_unbound_allocator_numbers_do_not_grant_cold_native_gpu_parameter_fit(placement_runtime):
    plan, hardware, graph = _gpu_case(placement_runtime, 'model_cpu')
    hardware['runtime'] = {'hardware': {'devices': [{
        'type': 'cuda', 'index': 0, 'memory_kind': 'dedicated',
        'torch_vram_total': 16 * GIB, 'torch_vram_free': 3 * GIB,
        'torch_allocated': 0, 'torch_reserved': 13 * GIB,
    }]}}
    result = plan(graph)
    assert not result['canAutoRun'], 'Advisory allocator fields cannot replace actual allocation-boundary capacity.'


@pytest.mark.parametrize('mode', ['none', 'model_cpu'])
def test_gpu_storage_credit_from_a_different_owner_policy_cannot_hide_cold_demand(placement_runtime, mode):
    plan, _, graph = _gpu_case(placement_runtime, mode)
    cache = {'owners': {'models': {
        'cacheKey': 'another-owner-or-policy', 'loadingPhaseReusable': True,
        'systemRamBytes': PARAMETERS['totalBytes'], 'vramBytes': PARAMETERS['totalBytes'],
    }}}
    result = plan(graph, cache_snapshot=cache)
    assert not result['canAutoRun']
    assert result['reusedOwnerIds'] == []


@pytest.mark.parametrize('complete', [None, False])
def test_incomplete_native_owner_cannot_grant_gpu_parameter_storage_credit(placement_runtime, complete):
    plan, _, graph = _gpu_case(placement_runtime, 'model_cpu')
    cached = {
        'cacheKey': workflow_auto_resource.workflow_owner_cache_key(graph, 'models'),
        'systemRamBytes': PARAMETERS['totalBytes'], 'vramBytes': PARAMETERS['largestComponentBytes'],
    }
    if complete is not None:
        cached['loadingPhaseReusable'] = complete
    result = plan(graph, cache_snapshot={'owners': {'models': cached}})
    assert not result['canAutoRun']
    assert result['requirements']['vramBytes'] == PARAMETERS['largestComponentBytes'] + 2 * GIB
    assert result['requirements']['systemRamBytes'] == 4 * GIB  # Existing measured CPU credit stays pool-specific.


def test_complete_cpu_storage_does_not_credit_the_gpu_parameter_pool(placement_runtime):
    plan, _, graph = _gpu_case(placement_runtime, 'model_cpu')
    cache = {'owners': {'models': {
        'cacheKey': workflow_auto_resource.workflow_owner_cache_key(graph, 'models'),
        'loadingPhaseReusable': True, 'systemRamBytes': PARAMETERS['totalBytes'], 'vramBytes': 0,
    }}}
    result = plan(graph, cache_snapshot=cache)
    assert not result['canAutoRun']
    assert result['reusedOwnerIds'] == ['models']


def test_zero_observed_gpu_free_bytes_remains_a_refusal_for_every_declared_placement(placement_runtime):
    plan, hardware, _, _, _ = placement_runtime
    hardware['systemMemory']['availableBytes'] = 30 * GIB
    hardware['accelerator']['freeBytes'] = 0
    graph = _graph()
    before = deepcopy(graph)
    result = plan(graph)
    assert not result['canAutoRun']
    assert result['available']['vramBytes'] == 0
    assert result['patches'] == []
    assert graph == before


@pytest.mark.parametrize('mode', ['group_cpu', 'group_disk'])
def test_unknown_group_gpu_storage_keeps_its_existing_declared_headroom_policy(placement_runtime, mode):
    plan, hardware, _, overrides, _ = placement_runtime
    overrides['offloadMode'] = mode
    hardware['systemMemory']['availableBytes'] = 30 * GIB
    hardware['accelerator']['freeBytes'] = 3 * GIB
    result = plan(_graph(mode))
    assert result['canAutoRun'], result['issues']
    assert result['requirements']['vramBytes'] == 2 * GIB


def test_unknown_header_does_not_invent_a_gpu_floor_or_new_placement_proof(placement_runtime):
    _, _, bound, _, _ = placement_runtime
    bound.clear()
    plan, _, graph = _gpu_case(placement_runtime, 'model_cpu')
    result = plan(graph)
    assert result['canAutoRun'], result['issues']
    assert result['loaders'][0]['settings']['offloadMode'] == 'model_cpu'
    assert 'parameterResidencyRequirements' not in result['loaders'][0]
    assert result['patches'] == []


@pytest.mark.parametrize('accelerator_change', [
    {'backend': 'rocm'}, {'memoryKind': 'shared'}, {'sharedMemory': True},
])
def test_other_cuda_api_backends_and_shared_pools_keep_the_existing_gpu_policy(placement_runtime, accelerator_change):
    plan, hardware, graph = _gpu_case(placement_runtime, 'model_cpu')
    hardware['accelerator'].update(accelerator_change)
    result = plan(graph)
    assert result['canAutoRun'], result['issues']
    assert result['requirements']['vramBytes'] == 2 * GIB
    assert not result['loaders'][0].get('parameterResidencyRequirements', {}).get('vramBytes')


def test_nonprimary_cuda_device_keeps_existing_policy_without_primary_gpu_metric_authority(placement_runtime):
    plan, _, graph = _gpu_case(placement_runtime, 'model_cpu')
    graph['nodes']['models']['params']['device']['value'] = 'cuda:1'
    result = plan(graph)
    assert not result['canAutoRun']
    assert any('differs from the accelerator covered' in issue for issue in result['issues'])
    assert result['loaders'] == [] and result['patches'] == []


def test_two_native_image_owners_progress_across_an_unchanged_gpu_peak(placement_runtime):
    plan, hardware, _, _, reads = placement_runtime
    hardware['systemMemory'].update(totalBytes=80 * GIB, availableBytes=70 * GIB)
    hardware['accelerator']['freeBytes'] = 3 * GIB
    graph = _two_graphs()
    before = deepcopy(graph)
    result = plan(graph)
    assert result['canAutoRun'], result['issues']
    assert result['schedule'] is not None
    assert len(result['loaders']) == 2
    assert all(owner['settings']['offloadMode'] in {'group_cpu', 'group_disk'} for owner in result['loaders'])
    assert len(reads) == 1
    assert all(patch['field'] in {'offload_mode', 'auto_offload'} for patch in result['patches'])
    assert graph == before


def test_explicit_gpu_working_budget_above_storage_retains_its_uncredited_remainder(placement_runtime, monkeypatch):
    from modiff import auto_resource
    plan, hardware, graph = _gpu_case(placement_runtime, 'model_cpu')
    original_recipe = auto_resource.build_auto_resource_plan
    def declared_budget(*args, **kwargs):
        result = original_recipe(*args, **kwargs)
        for candidate in result['candidates']:
            candidate['workingMemoryRequirements'] = {
                'systemRamBytes': PARAMETERS['totalBytes'], 'vramBytes': 14 * GIB, 'diskFreeBytes': 0,
            }
        return result
    monkeypatch.setattr(auto_resource, 'build_auto_resource_plan', declared_budget)
    gpu_weights = PARAMETERS['largestComponentBytes']
    cache = {'owners': {'models': {
        'cacheKey': workflow_auto_resource.workflow_owner_cache_key(graph, 'models'), 'loadingPhaseReusable': True,
        'systemRamBytes': PARAMETERS['totalBytes'] - gpu_weights, 'vramBytes': gpu_weights,
    }}}
    remaining = 14 * GIB - gpu_weights
    hardware['accelerator']['freeBytes'] = remaining - 1
    refused = plan(graph, cache_snapshot=cache)
    assert not refused['canAutoRun']
    assert refused['requirements']['vramBytes'] == remaining
    hardware['accelerator']['freeBytes'] = remaining
    admitted = plan(graph, cache_snapshot=cache)
    assert admitted['canAutoRun'], admitted['issues']
    assert admitted['requirements']['vramBytes'] == remaining
    assert admitted['workingMemoryPolicy'] == 'explicit_working_demand'


def test_scheduled_next_owner_rechecks_known_gpu_storage_and_headroom_at_exact_threshold(placement_runtime):
    from modiff.workflow_auto_lifecycle import assert_next_owner_capacity
    plan, hardware, _, overrides, _ = placement_runtime
    overrides['offloadMode'] = 'model_cpu'
    hardware['systemMemory'].update(totalBytes=80 * GIB, availableBytes=70 * GIB)
    result = plan(_two_graphs())
    assert result['canAutoRun'] and result['schedule'] is not None
    owner = result['schedule']['owners'][1]
    assert owner['parameterResidencyRequirements']['vramBytes'] == PARAMETERS['largestComponentBytes']
    threshold = PARAMETERS['largestComponentBytes'] + 2 * GIB
    hardware['accelerator']['freeBytes'] = threshold - 1
    with pytest.raises(ValueError, match='actual free memory'):
        assert_next_owner_capacity(None, owner, hardware)
    hardware['accelerator']['freeBytes'] = threshold
    assert_next_owner_capacity(None, owner, hardware)
