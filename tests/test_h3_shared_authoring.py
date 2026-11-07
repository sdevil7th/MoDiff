"""New H3 graphs use shared stages; saved source-bound specifications stay exact."""
import json
from pathlib import Path

import pytest

from modiff.operation_catalog import build_operation_catalog
from modiff.operation_starters import resolve_operation_starter
from modiff.supplemental_modular_contracts import PIKA_WORKFLOW_TASKS


@pytest.fixture(scope='module')
def catalog():
    from modules import MODULE_MAP
    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    return MODULE_MAP, contracts


@pytest.mark.parametrize('workflow,task', PIKA_WORKFLOW_TASKS.items())
def test_new_h3_graph_uses_shared_stages_and_exact_provider(catalog, workflow, task):
    from modiff.diffusers_profiles import resolve_execution_profiles_for_loader
    from modiff.studio_execution_specs import studio_execution_spec_for_pair
    modules, contracts = catalog
    selected = studio_execution_spec_for_pair('MiniMaxH3ModularPipeline', task)
    assert selected['executionProfileId'] == 'minimax-h3:official-provider:v1'
    result = resolve_operation_starter(modules, contracts, {
        'pipelineClass': 'MiniMaxH3ModularPipeline', 'task': task,
        'executionProfileId': selected['executionProfileId'],
    })
    actions = {node['action'] for node in result['nodes']}
    assert {'WorkflowJointAVTextEncode', 'WorkflowJointAVDenoise', 'WorkflowJointAVDecode'} <= actions
    assert not any(action.startswith('WorkflowMiniMaxH3') for action in actions)
    loader = next(node for node in result['nodes'] if node['action'] == 'ModelsLoader')
    profiles, reason = resolve_execution_profiles_for_loader(loader['module'], loader['action'], loader['values'])
    assert reason is None and len(profiles) == 1
    assert profiles[0].required_diffusers_provider == 'huggingface-diffusers-video-d0b9b1b'
    assert loader['values']['workflow_id'] == workflow
    assert loader['values']['revision'] == '42ed227ee7df40d41602854ae760620d6eb651fe'
    assert ('denoise', 'num_inference_steps', 'steps') in selected['bindings']
    denoise = next(node for node in result['nodes'] if node['action'] == 'WorkflowJointAVDenoise')
    assert denoise['params']['num_inference_steps']['hidden'] is False
    assert denoise['values']['num_inference_steps'] == 50


def test_all_saved_h3_specifications_retain_their_exact_original_hash_and_fields():
    from modiff.studio_execution_specs import studio_execution_spec_for_pair
    fixture = json.loads((Path(__file__).parent / 'fixtures/h3-legacy-studio-specs.v1.json').read_bytes())
    for expected in fixture['specifications']:
        actual = studio_execution_spec_for_pair(expected['modelType'], expected['mode'],
            content_hash=expected['contentHash'], execution_profile_id=expected['executionProfileId'])
        assert json.loads(json.dumps(actual)) == expected


def test_shared_h3_defaults_leave_all_qualification_gates_closed():
    from modiff.studio_execution_specs import studio_capability_definition
    capability = studio_capability_definition('MiniMaxH3ModularPipeline')
    assert capability['defaultSize'] == {'width': 960, 'height': 544, 'aspectRatio': '30:17'}
    for field in ('autoEligible', 'templateEligible', 'galleryEligible', 'liveProof'):
        assert capability[field] is False
    assert capability['qualifiedModes'] == [] and capability['downloadFiles'] == []
