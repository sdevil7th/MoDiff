"""New native starters carry the same selected workflow as their declaration."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from modiff.operation_catalog import build_operation_catalog
from modiff.operation_starters import resolve_operation_starter
from modules import MODULE_MAP


@pytest.fixture(scope='module')
def contracts():
    return build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})[0]


@pytest.mark.parametrize('pipeline', ['Flux3ModularPipeline', 'Flux3DistilledModularPipeline',
                                    'Flux2KleinModularPipeline', 'StableDiffusionXLModularPipeline'])
@pytest.mark.parametrize('task', ['text_to_image', 'edit_image'])
def test_new_native_starter_publishes_exact_resolved_workflow(contracts, pipeline, task):
    if pipeline.startswith('Flux3'):
        from test_modular_weight_formats import require_flux3_api
        require_flux3_api()
    if pipeline == 'StableDiffusionXLModularPipeline' and task == 'edit_image':
        task = 'image_to_image'
    before = deepcopy(contracts)
    with patch('modiff.NodeBase.NodeBase.__init__', side_effect=AssertionError('Metadata constructed a node.')):
        result = resolve_operation_starter(MODULE_MAP, contracts, {'pipelineClass': pipeline, 'task': task})
    loader = next(node for node in result['nodes'] if node['action'] == 'ModelsLoader')
    assert result['workflowId'] == loader['operation']['workflowId']
    assert loader['values']['workflow_id'] == result['workflowId']
    assert loader['params']['workflow_id']['value'] == result['workflowId']
    assert result['workflowId']
    assert contracts == before


def test_integrated_audio_starter_has_no_synthetic_modular_workflow(contracts):
    result = resolve_operation_starter(MODULE_MAP, contracts,
        {'pipelineClass': 'StableAudioPipeline', 'task': 'text_to_audio'})
    assert result['workflowId'] is None
    assert all('workflow_id' not in node['values'] for node in result['nodes'])
