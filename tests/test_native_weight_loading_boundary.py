"""Existing graph admission and consumed-input receipts carry loading mode."""

from copy import deepcopy

import pytest

from modiff.execution_input_provenance import build_resolved_execution_inputs, capture_generation_inputs


@pytest.mark.parametrize('field,value', [('weight_loading_mode', 'cpu'),
                                       ('weight_loading_mode', 'cuda'), ('inpaint_compatibility', 'native')])
def test_sparse_owner_identity_preserves_explicit_inspected_controls(field, value):
    from modiff.workflow_auto_values import inspect_native_runtime_controls

    nodes = {'models': {'module': 'modules.ModularDiffusers', 'action': 'ModelsLoader',
                        'params': {field: {'value': value}}}}
    controls = inspect_native_runtime_controls(
        nodes, 'models', lambda graph, node_id, key: graph[node_id]['params'][key]['value'],
    )
    assert controls == {field: value}


def test_shared_generic_loading_control_survives_real_runtime_hint_boundary(tmp_path):
    from modiff.server import WebServer

    server = WebServer(modules={}, work_dir=str(tmp_path), data_dir=str(tmp_path))
    receipt = {'schemaVersion': 1, 'graphHash': 'controlled-graph', 'resourceControlGroups': [[
        {'nodeId': 'inside', 'field': 'weight_loading_mode'},
        {'nodeId': 'outside', 'field': 'weight_loading_mode'},
    ]]}
    before = deepcopy(receipt)
    assert server._coerce_runtime_hints({'workflowAutoPlan': receipt})['workflowAutoPlan'] == before
    assert receipt == before


@pytest.mark.parametrize('field', ['weight_loading_mode_extra', 'prompt', {}, []])
def test_shared_loading_control_does_not_admit_other_or_malformed_fields(tmp_path, field):
    from modiff.server import WebServer

    server = WebServer(modules={}, work_dir=str(tmp_path), data_dir=str(tmp_path))
    receipt = {'schemaVersion': 1, 'graphHash': 'controlled-graph', 'resourceControlGroups': [[
        {'nodeId': 'inside', 'field': field}, {'nodeId': 'outside', 'field': field},
    ]]}
    with pytest.raises(RuntimeError, match='shared-control groups'):
        server._coerce_runtime_hints({'workflowAutoPlan': receipt})


@pytest.mark.parametrize('mode', ['cpu', 'cuda'])
def test_consumed_loading_mode_is_captured_in_output_ancestry(mode):
    loader = {'module': 'modules.ModularDiffusers', 'action': 'ModelsLoader',
              'params': {'weight_loading_mode': {'value': 'cpu'}}}
    preview = {'module': 'modules.Image', 'action': 'Preview',
               'params': {'images': {'sourceId': 'models', 'sourceKey': 'pipeline'}}}
    record = capture_generation_inputs('models', loader, {'weight_loading_mode': mode})
    assert record['fields']['weight_loading_mode'] == {'value': mode, 'source': 'literal'}
    receipt = build_resolved_execution_inputs(
        {'nodes': {'models': loader, 'preview': preview}}, {'models': record},
        task_id='loading-run', attempt_index=0, node_id='preview',
    )
    assert receipt['summary']['weightLoadingMode'] == mode
    assert loader['params']['weight_loading_mode']['value'] == 'cpu'
