"""Pika starters use shared stage schemas and retain the package-owned sampler."""
from unittest.mock import patch

import pytest

from modiff.operation_catalog import build_operation_catalog
from modiff.operation_starters import resolve_operation_starter
from modiff.supplemental_modular_contracts import PIKA_PIPELINE, PIKA_WORKFLOW_TASKS


@pytest.fixture(scope='module')
def catalog():
    from modules import MODULE_MAP
    with patch('modiff.NodeBase.NodeBase.__init__', side_effect=AssertionError('Discovery must not construct executable nodes')):
        contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    return MODULE_MAP, contracts


@pytest.mark.parametrize('workflow,task', PIKA_WORKFLOW_TASKS.items())
def test_actual_pika_starter_has_complete_shared_stage_bindings(catalog, workflow, task):
    modules, contracts = catalog
    with patch('modiff.NodeBase.NodeBase.__init__', side_effect=AssertionError('Authoring must not construct executable nodes')):
        result = resolve_operation_starter(modules, contracts, {'pipelineClass': PIKA_PIPELINE, 'task': task})
    actions = {node['action'] for node in result['nodes']}
    assert {'ModelsLoader', 'WorkflowJointAVTextEncode', 'WorkflowJointAVDenoise', 'WorkflowJointAVDecode'} <= actions
    assert not any('MiniMaxH3' in action for action in actions)
    loader = next(node for node in result['nodes'] if node['action'] == 'ModelsLoader')
    for stage in result['nodes']:
        if stage['action'] in {'ModelsLoader', 'WorkflowJointAVReferenceAssembler'}:
            continue
        assert any(edge['source'] == loader['operation']['operationId']
            and edge['sourceHandle'] == edge['targetHandle'] == 'pipeline_components'
            and edge['target'] == stage['operation']['operationId'] for edge in result['edges']), stage['action']
    denoise = next(node for node in result['nodes'] if node['action'] == 'WorkflowJointAVDenoise')
    assert denoise['values'].get('num_inference_steps') is None
    assert denoise['values']['pipeline_class'] == PIKA_PIPELINE
    assert denoise['values']['workflow_id'] == workflow
    assert not any(row['field'] in ('num_inference_steps', 'guidance_scale', 'negative_prompt') for row in result['requiredInputs'])
    if workflow == 'ref2va':
        helper = next(node for node in result['nodes'] if node['action'] == 'WorkflowJointAVReferenceAssembler')
        assert helper['values']['pipeline_class'] == PIKA_PIPELINE
