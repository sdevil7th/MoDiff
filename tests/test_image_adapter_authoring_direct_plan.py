"""Actual Redux starter/adapter headers and recipe patches; no runtime fit proof.

Only the recipe callback is a declared logical unit observation. The planner,
producer, existing recipe inspector and adapter header accounting are actual.
"""
from copy import deepcopy
from unittest.mock import patch

import pytest

from test_image_stage_auto_contracts import catalog as _catalog, graph, adapter_chain
from test_image_stage_direct_lora_memory import tiny_weight
from modiff import workflow_auto_resource as planner
from modiff.workflow_auto_values import inspect_execution_recipe_controls

catalog = _catalog


def selected_graph(catalog, tmp_path, *, empty=False):
    current, _spec = graph(catalog, 'FluxReduxPipeline', 'edit_image')
    current = adapter_chain(current)
    owner = current['nodes']['diffusion.load_models']['params']
    owner.update({name: {'value': value} for name, value in {
        'offload_mode': 'group_cpu', 'auto_offload': True, 'device_map': 'none',
        'quantization_mode': 'none', 'prequantized_transformer': None,
    }.items()})
    if not empty:
        path = tmp_path / 'style.safetensors'
        digest = tiny_weight(path)
        for index, key in enumerate(('adapter1', 'adapter2')):
            current['nodes'][key]['params'].update({
                'adapter_path': {'value': {'source': 'local', 'value': str(path)}},
                'weight_name': {'value': path.name}, 'expected_sha256': {'value': digest},
                'adapter_name': {'value': 'style' + str(index)}, 'scale': {'value': .35},
                'replace_existing': {'value': False},
            })
    return current


def add_recipe(current, *, connected_offload=False):
    current['nodes']['recipe'] = {
        'module': 'modules.DiffusersRuntime', 'action': 'DiffusersExecutionRecipe',
        'params': {'offload_mode': {'value': 'group_cpu'}, 'device_map': {'value': 'none'},
                   'device': {'value': 'cuda:0'}},
    }
    current['nodes']['diffusion.load_models']['params']['execution_recipe'] = {
        'sourceId': 'recipe', 'sourceKey': 'execution_recipe',
    }
    current['paths'][0].insert(0, 'recipe')
    if connected_offload:
        current['nodes']['custom'] = {'module': 'custom.Unknown', 'action': 'Compute', 'params': {}}
        current['nodes']['recipe']['params']['offload_mode'] = {'sourceId': 'custom', 'sourceKey': 'offload'}
        current['paths'][0].insert(0, 'custom')


def plan(current, tmp_path, *, selected_mode='none', can_run=True):
    requests = []

    def recipe(payload, **kwargs):
        from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
        form = payload['form']
        profile = DIFFUSERS_EXECUTION_PROFILES[form['executionProfileId']]
        owner = current['nodes']['diffusion.load_models']['params']
        requests.append(deepcopy(form))
        candidate = {
            'id': 'explicit-unit-recipe', 'executionProfileId': profile.id,
            'modelRepo': form['modelRepo'], 'modelType': form['modelType'], 'mode': form['mode'],
            'loaderModule': profile.loader_module, 'loaderAction': profile.loader_action,
            'executionPath': profile.execution_path, 'artifactRevision': owner['revision']['value'],
            'dtype': form['dtype'], 'quantizationMode': form['quantizationMode'],
            'offloadMode': selected_mode, 'autoOffload': selected_mode != 'none',
            'canAutoRun': can_run, 'proof': {'status': 'declared_safe'},
            'generation': {'width': form['width'], 'height': form['height'], 'steps': form['steps'],
                           'maxSequenceLength': form['maxSequenceLength'], 'batchSize': 1},
            'requirements': {'systemRamBytes': 300, 'vramBytes': 200, 'diskFreeBytes': 10},
        }
        return {'candidates': [candidate]}

    with patch('modiff.NodeBase.NodeBase.__init__', side_effect=AssertionError('supplier constructed')):
        result = planner.build_workflow_auto_plan(current, runtime_fingerprint={}, local_models=[],
            data_dir=str(tmp_path), plan_recipe=recipe,
            hardware={'accelerator': {'freeBytes': 1000, 'memoryKind': 'dedicated'},
                      'systemMemory': {'availableBytes': 1000}, 'offloadDisk': {'freeBytes': 1000}})
    return result, requests


def test_actual_populated_redux_recipe_is_patched_before_required_controls_replan(catalog, tmp_path):
    current = selected_graph(catalog, tmp_path)
    add_recipe(current)
    before = deepcopy(current)
    result, requests = plan(current, tmp_path)
    assert result['canAutoRun'], result['issues']
    assert len(requests) == 1 and requests[0]['offloadMode'] == 'group_cpu'
    assert {'nodeId': 'recipe', 'field': 'offload_mode', 'value': 'none'} in result['patches']
    assert result['requirements']['systemRamBytes'] == 620
    assert result['requirements']['vramBytes'] == 520
    assert sum(row['systemRamBytes'] for row in result['adapters']) == 320
    assert result['reusedOwnerIds'] == [] and len(result['adapters']) == 2
    patched = deepcopy(current)
    for row in result['patches']:
        param = patched['nodes'][row['nodeId']]['params'][row['field']]
        assert param.get('sourceId') is None and row['field'] in before['nodes'][row['nodeId']]['params']
        param['value'] = row['value']
    controls, _bindings = inspect_execution_recipe_controls(patched['nodes'], 'diffusion.load_models', planner._literal)
    assert controls['offload_mode'] == 'none' and controls['auto_offload'] is False
    rerun, requests2 = plan(patched, tmp_path)
    assert rerun['canAutoRun'], rerun['issues']
    assert requests2[0]['offloadMode'] == 'none' and rerun['patches'] == []
    assert rerun['requirements'] == result['requirements']
    assert rerun['adapters'] == result['adapters'] and rerun['reusedOwnerIds'] == []
    assert current == before
    profile = requests2[0]['executionProfileId']
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
    assert 'none' not in DIFFUSERS_EXECUTION_PROFILES[profile].supported_offload_modes


def test_populated_direct_recipe_cannot_ignore_connected_offload(catalog, tmp_path):
    current = selected_graph(catalog, tmp_path)
    add_recipe(current, connected_offload=True)
    before = deepcopy(current)
    result, requests = plan(current, tmp_path)
    assert not result['canAutoRun'] and result['patches'] == [] and requests == []
    assert current == before


@pytest.mark.parametrize('empty_selection', [None, '', {'source': 'local', 'value': ''}])
def test_actual_empty_redux_chain_preserves_ordinary_recipe_and_zero_adapter_budget(catalog, tmp_path, empty_selection):
    current = selected_graph(catalog, tmp_path, empty=True)
    for key in ('adapter1', 'adapter2'):
        current['nodes'][key]['params']['adapter_path'] = {'value': empty_selection}
    add_recipe(current)
    before = deepcopy(current)
    result, requests = plan(current, tmp_path, selected_mode='group_cpu')
    assert result['canAutoRun'], result['issues']
    assert requests[0]['offloadMode'] == 'group_cpu' and result['patches'] == []
    assert result['adapters'] == [] and result['reusedOwnerIds'] == []
    assert result['requirements']['systemRamBytes'] == 300
    assert result['requirements']['vramBytes'] == 200
    assert current == before


def test_route_controls_do_not_override_recipe_readiness_refusal(catalog, tmp_path):
    current = selected_graph(catalog, tmp_path)
    add_recipe(current)
    result, _requests = plan(current, tmp_path, can_run=False)
    assert not result['canAutoRun'] and result['patches'] == []


def test_populated_adapter_missing_weights_does_not_receive_zero_budget(catalog, tmp_path):
    current = selected_graph(catalog, tmp_path)
    for key in ('adapter1', 'adapter2'):
        current['nodes'][key]['params']['adapter_path']['value']['value'] = str(tmp_path / 'missing.safetensors')
    result, _requests = plan(current, tmp_path)
    assert not result['canAutoRun'] and result['patches'] == []
