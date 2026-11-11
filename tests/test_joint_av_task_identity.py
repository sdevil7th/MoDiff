"""Shared stage history recognizes exact routes without granting a runtime recipe."""
from copy import deepcopy

import pytest

from modiff.modular_action_bindings import MODULAR_ACTION_BINDINGS
from modiff.modular_whole_workflow_contracts import (
    PINNED_WHOLE_WORKFLOW_GRAPH_ADAPTERS,
    SUPPLEMENTAL_WHOLE_WORKFLOW_GRAPH_ADAPTERS,
)
from modiff.supplemental_modular_contracts import PIKA_WORKFLOW_TASKS
from modiff.workflow_task_identity import graph_task_receipts, modular_graph_tasks, resource_consumers


PIPELINES = ('MiniMaxH3ModularPipeline', 'PikaH3FlexModularPipeline')


def joint_graph(pipeline, workflow, *, legacy=False):
    routes = PINNED_WHOLE_WORKFLOW_GRAPH_ADAPTERS if legacy else SUPPLEMENTAL_WHOLE_WORKFLOW_GRAPH_ADAPTERS
    route = routes[pipeline, workflow]
    nodes = {'loader': {'module': 'modules.ModularDiffusers', 'action': 'ModelsLoader', 'params': {
        'model_type': {'value': pipeline}, 'workflow_id': {'value': workflow},
    }}}
    for action, block in zip(route['actionSequence'], route['upstreamBlockSequence']):
        module, name = MODULAR_ACTION_BINDINGS[action][1].rsplit('.', 1)
        nodes[action] = {'module': module, 'action': name, 'params': {
            'pipeline_components': {'sourceId': 'loader', 'sourceKey': 'pipeline_components'},
            'pipeline_class': {'value': pipeline}, 'workflow_id': {'value': workflow},
            'block_path': {'value': block},
        }}
    for edge in route['stateEdges']:
        nodes[edge['consumerAction']]['params'][edge['consumerInput']] = {
            'sourceId': edge['producerAction'], 'sourceKey': edge['producerOutput'],
        }
    for name in route['requiredInputs']:
        nodes[route['actionSequence'][0]]['params'][name] = {'value': 'test-input'}
    return nodes


def tasks(nodes, pipeline):
    return modular_graph_tasks(nodes, 'loader', resource_consumers(nodes, 'loader', {'loader'}), pipeline)


@pytest.mark.parametrize('pipeline', PIPELINES)
@pytest.mark.parametrize('workflow,task', PIKA_WORKFLOW_TASKS.items())
def test_shared_joint_routes_record_the_exact_task_without_changing_graph(pipeline, workflow, task):
    from modiff.execution_input_provenance import capture_generation_inputs

    nodes = joint_graph(pipeline, workflow)
    before = deepcopy(nodes)
    assert tasks(nodes, pipeline) == {task}
    records = {key: capture_generation_inputs(key, node, {
        name: value['value'] for name, value in node['params'].items() if 'value' in value
    }) for key, node in nodes.items()}
    assert graph_task_receipts(nodes, records) == [{'loaderId': 'loader', 'pipelineClass': pipeline, 'task': task}]
    assert graph_task_receipts(nodes, {}) == []  # A graph alone is not an observed owner receipt.
    assert nodes == before


@pytest.mark.parametrize('workflow,task', PIKA_WORKFLOW_TASKS.items())
def test_saved_legacy_h3_routes_keep_their_existing_task(workflow, task):
    assert tasks(joint_graph(PIPELINES[0], workflow, legacy=True), PIPELINES[0]) == {task}


@pytest.mark.parametrize('pipeline', PIPELINES)
@pytest.mark.parametrize('workflow', PIKA_WORKFLOW_TASKS)
@pytest.mark.parametrize('defect', ('block', 'pipeline', 'workflow', 'owner', 'state', 'duplicate', 'partial', 'loader'))
def test_shared_joint_task_refuses_conflicting_or_partial_routes(pipeline, workflow, defect):
    nodes = joint_graph(pipeline, workflow)
    first = next(key for key in nodes if key != 'loader')
    last = list(nodes)[-1]
    if defect == 'block':
        nodes[first]['params']['block_path'] = {'value': 'decode'}
    elif defect == 'pipeline':
        nodes[first]['params']['pipeline_class'] = {'value': PIPELINES[1] if pipeline == PIPELINES[0] else PIPELINES[0]}
    elif defect == 'workflow':
        nodes[first]['params']['workflow_id'] = {'value': 'unknown'}
    elif defect == 'owner':
        nodes[first]['params']['pipeline_components'] = {'sourceId': 'foreign', 'sourceKey': 'pipeline_components'}
    elif defect == 'state':
        nodes[last]['params']['state_in']['sourceKey'] = 'foreign_state'
    elif defect == 'duplicate':
        nodes['duplicate'] = deepcopy(nodes[first])
    elif defect == 'partial':
        del nodes[last]
    else:
        nodes['loader']['params']['workflow_id'] = {'value': 'unknown'}
    before = deepcopy(nodes)
    assert tasks(nodes, pipeline) == set()
    assert nodes == before
