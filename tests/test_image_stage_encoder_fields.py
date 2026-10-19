"""Current selected encoders publish scalar controls they actually consume."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from image_stage_test_support import stage_offline_runtime as _stage_offline_runtime

runtime = _stage_offline_runtime

PAIRS = (
    ('FluxReduxPipeline', 'edit_image'), ('FluxReduxPipeline', 'multi_image_reference_edit'),
    ('FluxFillPipeline', 'inpaint'), ('FluxFillPipeline', 'outpaint'),
    ('FluxCannyPipeline', 'control_image'), ('FluxDepthPipeline', 'control_image'),
)


@pytest.fixture(scope='module')
def catalog():
    from modules import MODULE_MAP
    from modiff.operation_catalog import build_operation_catalog
    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    return MODULE_MAP, contracts


@pytest.mark.parametrize('model_type,task', PAIRS)
def test_selected_starter_and_current_contract_omit_only_unconsumed_scalar_encoder_controls(catalog, model_type, task):
    from modiff.operation_starters import resolve_operation_starter
    from modiff.studio_execution_specs import studio_execution_spec_for_pair
    spec = studio_execution_spec_for_pair(model_type, task)
    before = deepcopy(catalog[0]['modules.DiffusersImage']['EncodeInputs']['params'])
    with patch('modiff.NodeBase.NodeBase.__init__', side_effect=AssertionError('node constructed')):
        starter = resolve_operation_starter(*catalog, {'pipelineClass': spec['pipelineClass'], 'task': task,
            'executionProfileId': spec['executionProfileId']})
    encoder, = [row for row in starter['nodes'] if row['action'] == 'EncodeInputs']
    sampler, = [row for row in starter['nodes'] if row['action'] == 'Denoise']
    scalars = {'seed', 'width', 'height', 'max_sequence_length', 'num_images_per_prompt'}
    expected = set() if model_type == 'FluxReduxPipeline' else scalars - ({'seed'} if model_type == 'FluxFillPipeline' else set())
    assert set(encoder['params']) & scalars == expected
    assert {port['name'] for port in encoder['operation']['ports']} & scalars == expected
    assert sampler['params']['seed']['default'] == 0
    assert 'seed' in sampler['params'] and 'width' in sampler['params'] and 'height' in sampler['params']
    if 'num_images_per_prompt' in expected:
        assert encoder['params']['num_images_per_prompt']['default'] == 1
    assert catalog[0]['modules.DiffusersImage']['EncodeInputs']['params'] == before


@pytest.mark.parametrize('model_type,task', PAIRS)
def test_retained_whole_action_still_publishes_original_geometry_and_rng(catalog, model_type, task):
    from modiff.operation_starters import resolve_operation_starter
    from modiff.studio_execution_specs import studio_execution_spec_for_pair
    spec = studio_execution_spec_for_pair(model_type, task)
    whole = resolve_operation_starter(*catalog, {'pipelineClass': spec['pipelineClass'], 'task': task})
    consumer, = [row for row in whole['nodes'] if row['action'] != 'LoadPipeline']
    assert {'seed', 'width', 'height', 'max_sequence_length'} <= set(consumer['params'])





@pytest.mark.parametrize('kind,task', [('fill', 'inpaint'), ('control', 'control_image')])
def test_successful_bound_decode_records_actual_executed_owner_task(runtime, kind, task):
    from test_image_stage_numerical import selected_owner, values
    from modules.DiffusersImage import main, stage_execution as core
    owner, strategy, _ = selected_owner(kind)
    given = values(count=2)
    encoded = core.encode(owner, strategy, given)
    latent, binding = core.denoise(owner, strategy, encoded, given, {})
    node = main.DecodeLatents('actual-bound-capture')
    result = node(pipeline=owner, latents=latent, latent_binding=binding, width=64, height=80, output_type='pt')
    assert result['images'].shape[0] == 2
    assert node._execution_input_record['fields']['executed_task']['value'] == task
    assert node._execution_input_record['fields']['width']['value'] == 64
    assert node._execution_input_record['fields']['height']['value'] == 80
    raw = main.DecodeLatents('retained-raw-capture')
    raw(pipeline=owner, latents=latent.clone(), width=64, height=80, output_type='pt')
    assert 'executed_task' not in raw._execution_input_record['fields']
