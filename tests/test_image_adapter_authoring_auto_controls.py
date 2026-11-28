"""Actual planner with source-declared private policy; resource recipes are unit fixtures."""
from copy import deepcopy
import hashlib
import json
import struct

import pytest
from modiff import workflow_auto_resource as planner
from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
from test_native_image_component_lora import graph as cosmos_graph, MODEL, REPO, REV


def tiny_adapter(tmp_path):
    header = {
        'transformer.proj_out.lora_A.weight': {'dtype': 'F32', 'shape': [1, 8], 'data_offsets': [0, 32]},
        'transformer.proj_out.lora_B.weight': {'dtype': 'F32', 'shape': [1, 1], 'data_offsets': [32, 36]},
    }
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    path = tmp_path / 'adapter.safetensors'
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + bytes(36))
    return {'module': 'modules.ModularDiffusers', 'action': 'Lora', 'params': {
        key: {'value': value} for key, value in {
            'model': {'source': 'local', 'value': str(path)}, 'weight_name': path.name,
            'expected_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'adapter_name': 'unit',
            'scale': 1., 'previous_loras': None, 'scheduler_class': '', 'scheduler_config': '{}',
        }.items()}}


@pytest.fixture
def setup(tmp_path):
    profile = DIFFUSERS_EXECUTION_PROFILES['cosmos3-super-text-to-image:official-modular-workflow']
    assert profile.adapter_authoring.required_controls
    source = cosmos_graph()
    source['nodes'] = {'adapter': tiny_adapter(tmp_path), **source['nodes']}
    source['nodes']['models']['params']['lora_list'] = {'sourceId': 'adapter', 'sourceKey': 'lora'}
    source['paths'][0].insert(0, 'adapter')
    candidate = {
        'id': 'unit-observed-recipe', 'executionProfileId': profile.id, 'modelRepo': REPO,
        'artifactRevision': REV, 'modelType': MODEL, 'mode': 'text_to_image',
        'loaderModule': profile.loader_module, 'loaderAction': profile.loader_action,
        'executionPath': profile.execution_path, 'dtype': 'float32', 'device': 'cpu',
        'quantizationMode': 'none', 'offloadMode': 'group_cpu', 'autoOffload': True,
        'weight_loading_mode': 'cpu', 'canAutoRun': True, 'proof': {'status': 'declared_safe'},
        'generation': {'width': 8, 'height': 8, 'steps': 50},
        'requirements': {'systemRamBytes': 300, 'vramBytes': 200, 'diskFreeBytes': 10},
    }
    hardware = {'accelerator': {'kind': 'cuda', 'memoryKind': 'dedicated', 'freeBytes': 2**34},
                'systemMemory': {'availableBytes': 2**35}, 'offloadDisk': {'freeBytes': 2**35}}
    candidates = [candidate]
    def plan(graph):
        return planner.build_workflow_auto_plan(graph, runtime_fingerprint={}, local_models=[],
            data_dir=str(tmp_path), hardware=hardware,
            plan_recipe=lambda *_args, **_kwargs: {'candidates': deepcopy(candidates)})
    return source, candidate, candidates, plan


def test_populated_native_route_refuses_accepted_placement_outside_declared_controls(setup):
    source, _candidate, _candidates, plan = setup
    before = deepcopy(source)
    result = plan(source)
    assert result['canAutoRun'] is False
    assert result['patches'] == []
    assert 'adapter route' in ' '.join(result['issues'])
    assert source == before


def test_populated_route_selects_existing_admitted_none_instead_of_unsafe_preferred(setup):
    source, candidate, candidates, plan = setup
    source['nodes']['models']['params']['offload_mode']['value'] = 'group_cpu'
    source['nodes']['models']['params']['auto_offload']['value'] = True
    candidates.append({**candidate, 'id': 'unit-none-control', 'offloadMode': 'none', 'autoOffload': False})
    result = plan(source)
    assert result['canAutoRun'], result['issues']
    assert result['loaders'][0]['candidateId'] == 'unit-none-control'
    assert {row['field'] for row in result['patches']} == {'offload_mode', 'auto_offload'}


def test_vanilla_preserves_ordinary_accepted_placement(setup):
    source, _candidate, _candidates, plan = setup
    source['nodes'].pop('adapter')
    source['paths'][0].remove('adapter')
    source['nodes']['models']['params']['lora_list'] = {'value': None}
    result = plan(source)
    assert result['canAutoRun'], result['issues']
    assert result['loaders'][0]['candidateId'] == 'unit-observed-recipe'
    assert {row['field'] for row in result['patches']} == {'offload_mode', 'auto_offload'}


@pytest.mark.parametrize('field', ['weight_loading_mode', 'quant_config', 'trust_remote_code'])
def test_missing_required_owner_descriptor_cannot_obtain_policy_by_insertion(setup, field):
    source, candidate, _candidates, plan = setup
    candidate.update(offloadMode='none', autoOffload=False)
    source['nodes']['models']['params'].pop(field)
    result = plan(source)
    assert result['canAutoRun'] is False
    assert result['patches'] == []
    assert field in ' '.join(result['issues'])
