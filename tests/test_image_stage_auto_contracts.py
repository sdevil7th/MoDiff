"""Static stage inspection and captured tasks do not execute model suppliers."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from test_redux_image_stage_graph import actual_graph_runtime as _actual_graph_runtime

actual_graph_runtime = _actual_graph_runtime

PAIRS = (
    ('FluxReduxPipeline', 'edit_image'),
    ('FluxReduxPipeline', 'multi_image_reference_edit'),
    ('FluxFillPipeline', 'inpaint'),
    ('FluxFillPipeline', 'outpaint'),
    ('FluxCannyPipeline', 'control_image'),
    ('FluxDepthPipeline', 'control_image'),
)


@pytest.fixture(scope='module')
def catalog():
    from modules import MODULE_MAP
    from modiff.operation_catalog import build_operation_catalog
    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    return MODULE_MAP, contracts


def graph(catalog, model_type, task, *, whole=False):
    from modiff.operation_starters import resolve_operation_starter
    from modiff.studio_execution_specs import studio_execution_spec_for_pair
    spec = studio_execution_spec_for_pair(model_type, task)
    selection = {'pipelineClass': spec['pipelineClass'], 'task': task}
    if not whole:
        selection['executionProfileId'] = spec['executionProfileId']
    with patch('modiff.NodeBase.NodeBase.__init__', side_effect=AssertionError('executable supplier constructed')):
        starter = resolve_operation_starter(*catalog, selection)
    nodes = {}
    for row in starter['nodes']:
        params = {name: {'value': deepcopy(param.get('value', param.get('default')))}
                  for name, param in row['params'].items() if param.get('display') != 'output'}
        params.update({name: {'value': value} for name, value in {
            'width': 128, 'height': 128, 'seed': 5, 'max_sequence_length': 4,
            'num_inference_steps': 3, 'num_images_per_prompt': 1, 'prompt': 'one prompt', 'prompt_2': None,
        }.items() if name in params})
        nodes[row['operation']['operationId']] = {
            'module': row['module'], 'action': row['action'], 'params': params,
        }
    for edge in starter['edges']:
        nodes[edge['target']]['params'][edge['targetHandle']] = {
            'sourceId': edge['source'], 'sourceKey': edge['sourceHandle'],
        }
    return {'nodes': nodes, 'paths': [[row['operation']['operationId'] for row in starter['nodes']]]}, spec


def controlled_plan(current, tmp_path):
    """Inject only a logical accepted recipe, never real hardware/fit evidence."""
    from modiff.workflow_auto_resource import build_workflow_auto_plan
    requests = []

    def recipe(payload, **kwargs):
        from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
        form = payload['form']
        profile = DIFFUSERS_EXECUTION_PROFILES[form['executionProfileId']]
        owner, = [node for node in current['nodes'].values() if node['action'] == 'LoadPipeline']
        requests.append(deepcopy(form))
        candidate = {'id': 'controlled-metadata-recipe', 'executionProfileId': form['executionProfileId'],
            'modelRepo': form['modelRepo'], 'modelType': form['modelType'], 'mode': form['mode'],
            'loaderModule': 'modules.DiffusersImage', 'loaderAction': 'LoadPipeline',
            'executionPath': profile.execution_path, 'artifactRevision': owner['params']['revision']['value'],
            'dtype': form['dtype'], 'quantizationMode': form['quantizationMode'],
            'offloadMode': 'none', 'autoOffload': False, 'canAutoRun': True,
            'proof': {'status': 'declared_safe'},
            'generation': {'width': form['width'], 'height': form['height'], 'steps': form['steps'],
                           'maxSequenceLength': form['maxSequenceLength'], 'batchSize': 1},
            'requirements': {'systemRamBytes': 300, 'vramBytes': 200, 'diskFreeBytes': 10}}
        return {'candidates': [candidate]}

    with patch('modiff.NodeBase.NodeBase.__init__', side_effect=AssertionError('supplier constructed')):
        result = build_workflow_auto_plan(current, runtime_fingerprint={}, local_models=[],
            data_dir=str(tmp_path), plan_recipe=recipe,
            hardware={'accelerator': {'freeBytes': 1000, 'memoryKind': 'dedicated'},
                      'systemMemory': {'availableBytes': 1000}, 'offloadDisk': {'freeBytes': 1000}})
    return result, requests


@pytest.mark.parametrize('model_type,task', PAIRS)
def test_source_generated_six_stage_routes_reach_existing_recipe_inspection_without_suppliers(catalog, tmp_path, model_type, task):
    current, _spec = graph(catalog, model_type, task)
    before = deepcopy(current)
    result, requests = controlled_plan(current, tmp_path)
    assert result['canAutoRun'], result['issues']
    assert len(requests) == len(result['loaders']) == 1
    assert requests[0]['width'] == requests[0]['height'] == 128
    if model_type == 'FluxReduxPipeline':
        assert 'num_images_per_prompt' not in current['nodes']['diffusion.encode_image']['params']
        assert 'batchSize' not in requests[0]  # The actual Redux prior has fixed implicit count1.
    else:
        assert requests[0]['batchSize'] == 1
    assert len(result['loaders'][0]['consumers']) == 3
    assert current == before


@pytest.mark.parametrize('corruption', ['alias', 'latents', 'binding', 'owner', 'supplier', 'fraction'])
def test_geometry_forecast_requires_exact_builtin_latent_binding_owner_and_scalar_controls(catalog, corruption):
    from modiff.workflow_auto_values import inspect_resource_value
    current, _spec = graph(catalog, 'FluxFillPipeline', 'inpaint')
    nodes = current['nodes']
    decode = nodes['diffusion.decode_latents']['params']
    denoise = nodes['diffusion.denoise']['params']
    if corruption == 'alias':
        decode['width']['sourceKey'] = 'output_width'
    elif corruption == 'latents':
        decode['latents']['sourceKey'] = 'images'
    elif corruption == 'binding':
        decode.pop('latent_binding')
    elif corruption == 'owner':
        decode['pipeline'] = {'sourceId': 'foreign', 'sourceKey': 'pipeline'}
    elif corruption == 'fraction':
        denoise['width'] = {'value': 127.5}
    else:
        nodes['custom'] = {'module': 'custom.Unknown', 'action': 'ComputeDimension', 'params': {}}
        denoise['width'] = {'sourceId': 'custom', 'sourceKey': 'width'}
    with patch('modiff.NodeBase.NodeBase.__init__', side_effect=AssertionError('supplier constructed')):
        with pytest.raises(ValueError):
            inspect_resource_value(nodes, 'diffusion.decode_latents', 'width')


@pytest.mark.parametrize('model_type,field', [
    ('FluxFillPipeline', 'width'), ('FluxCannyPipeline', 'seed'), ('FluxDepthPipeline', 'max_sequence_length'),
])
def test_declared_shared_encoder_sampler_controls_refuse_mismatch_before_recipe(catalog, tmp_path, model_type, field):
    task = 'inpaint' if model_type == 'FluxFillPipeline' else 'control_image'
    current, _spec = graph(catalog, model_type, task)
    current['nodes']['diffusion.encode_image']['params'][field]['value'] += 1
    result, requests = controlled_plan(current, tmp_path)
    assert not result['canAutoRun'] and not requests and result['patches'] == []
    assert 'known matching integers' in ' '.join(result['issues'])


@pytest.mark.parametrize('corruption', ['prompt_batch', 'embedding', 'count', 'sampler_count', 'generator'])
def test_unknown_batch_and_external_stage_material_cannot_acquire_an_auto_envelope(catalog, tmp_path, corruption):
    current, _spec = graph(catalog, 'FluxFillPipeline', 'inpaint')
    encode = current['nodes']['diffusion.encode_image']['params']
    if corruption == 'prompt_batch':
        encode['prompt']['value'] = ['one', 'two']
    elif corruption == 'embedding':
        current['nodes']['unknown'] = {'module': 'custom.Unknown', 'action': 'Embeddings', 'params': {}}
        encode['prompt_embeds'] = {'sourceId': 'unknown', 'sourceKey': 'embedding'}
        current['paths'][0].insert(0, 'unknown')
    elif corruption == 'generator':
        encode['generator'] = {'value': 'opaque external generator'}
    elif corruption == 'sampler_count':
        current['nodes']['diffusion.denoise']['params']['num_images_per_prompt'] = {'value': 2}
    else:
        encode['num_images_per_prompt'] = {'value': None}
    result, requests = controlled_plan(current, tmp_path)
    assert not result['canAutoRun'] and not requests and result['patches'] == []


def test_redux_nonconsuming_encoder_seed_and_geometry_are_not_false_alignment_constraints(catalog, tmp_path):
    current, _spec = graph(catalog, 'FluxReduxPipeline', 'edit_image')
    encode = current['nodes']['diffusion.encode_image']['params']
    encode['seed'] = {'value': 101}
    encode['width'] = {'value': 256}
    result, requests = controlled_plan(current, tmp_path)
    assert result['canAutoRun'], result['issues']
    assert requests[0]['width'] == 256  # Existing conservative maxima remain unchanged.


def captured_records(current, spec, task):
    from modiff.execution_input_provenance import capture_generation_inputs
    nodes = current['nodes']
    records = {'diffusion.load_models': capture_generation_inputs('diffusion.load_models',
        nodes['diffusion.load_models'], {'pipeline_class': spec['pipelineClass'], 'mode': task})}
    for identity in ('diffusion.encode_image', 'diffusion.denoise', 'diffusion.decode_latents'):
        records[identity] = capture_generation_inputs(identity, nodes[identity], {'executed_task': task})
    return records


@pytest.mark.parametrize('model_type,task', PAIRS)
def test_stage_task_receipts_use_the_captured_owner_and_actual_executed_tasks(catalog, model_type, task):
    from modiff.workflow_task_identity import graph_task_receipts
    current, spec = graph(catalog, model_type, task)
    records = captured_records(current, spec, task)
    assert graph_task_receipts(current['nodes'], records) == [
        {'loaderId': 'diffusion.load_models', 'pipelineClass': spec['pipelineClass'], 'task': task}]


@pytest.mark.parametrize('corruption', ['missing_owner', 'invalid_owner', 'missing_stage', 'conflict', 'consistent_wrong_task'])
def test_stage_task_receipts_do_not_trust_mode_labels_or_conflicting_capture(catalog, corruption):
    from modiff.workflow_task_identity import graph_task_receipts
    current, spec = graph(catalog, 'FluxFillPipeline', 'inpaint')
    records = captured_records(current, spec, 'inpaint')
    if corruption == 'missing_owner':
        records['diffusion.load_models']['fields'].pop('mode')
    elif corruption == 'invalid_owner':
        records['diffusion.load_models']['fields']['mode']['value'] = []
    elif corruption == 'missing_stage':
        records['diffusion.denoise']['fields'].pop('executed_task')
    elif corruption == 'consistent_wrong_task':
        for identity in ('diffusion.encode_image', 'diffusion.denoise', 'diffusion.decode_latents'):
            records[identity]['fields']['executed_task']['value'] = 'outpaint'
    else:
        records['diffusion.denoise']['fields']['executed_task']['value'] = 'outpaint'
    assert graph_task_receipts(current['nodes'], records)[0]['task'] is None


def test_actual_tiny_redux_dispatcher_final_receipt_contains_the_consumed_task(actual_graph_runtime):
    from modiff.execution_input_provenance import build_resolved_execution_inputs
    from modiff.workflow_task_identity import graph_task_receipts
    from test_redux_image_stage_graph import stage_graph, dispatch
    app, _events, _refs = actual_graph_runtime
    current = stage_graph()
    dispatch(app, current)
    expected = [{'loaderId': 'models', 'pipelineClass': 'FluxReduxPipeline', 'task': 'multi_image_reference_edit'}]
    assert graph_task_receipts(current['nodes'], app._resolved_input_records) == expected
    receipt = build_resolved_execution_inputs(current, app._resolved_input_records,
        task_id='actual-redux-stage-task', attempt_index=0, node_id='decode')
    assert receipt['graphTasks'] == expected


def test_retained_whole_recipe_and_native_geometry_forecast_are_unchanged(catalog, tmp_path):
    from modiff.workflow_auto_values import inspect_resource_value
    current, _spec = graph(catalog, 'FluxCannyPipeline', 'control_image', whole=True)
    result, requests = controlled_plan(current, tmp_path)
    assert result['canAutoRun'], result['issues']
    assert len(requests) == 1 and len(result['loaders'][0]['consumers']) == 1
    native = {'denoise': {'module': 'modules.ModularDiffusers', 'action': 'Denoise', 'params': {'width': {'value': 512}}},
              'decode': {'module': 'modules.ModularDiffusers', 'action': 'DecodeLatents', 'params': {
                  'width': {'sourceId': 'denoise', 'sourceKey': 'out_width'},
                  'latents': {'sourceId': 'denoise', 'sourceKey': 'latents'}}}}
    assert inspect_resource_value(native, 'decode', 'width') == 512


@pytest.mark.parametrize('model_type,task', [('FluxFillPipeline', 'inpaint'), ('FluxCannyPipeline', 'control_image'), ('FluxDepthPipeline', 'control_image')])
def test_shared_primitive_controls_are_inspected_without_execution(catalog, tmp_path, model_type, task):
    current, _spec = graph(catalog, model_type, task)
    fields = ('width', 'height', 'max_sequence_length', *(('seed',) if task == 'control_image' else ()))
    for field in fields:
        primitive = 'control_' + field
        value = current['nodes']['diffusion.denoise']['params'][field]['value']
        current['nodes'][primitive] = {'module': 'modules.Primitive', 'action': 'TextValue', 'params': {'text': {'value': str(value)}}}
        converted = primitive + '_integer'
        current['nodes'][converted] = {'module': 'modules.Text', 'action': 'ProcessText', 'params': {
            'source': {'sourceId': primitive, 'sourceKey': 'output'},
            'operation': {'value': 'data_conversion'}, 'target_type': {'value': 'integer'}}}
        for stage in ('diffusion.encode_image', 'diffusion.denoise'):
            current['nodes'][stage]['params'][field] = {'sourceId': converted, 'sourceKey': 'output'}
    assert 'TextValue' in catalog[0]['modules.Primitive'] and 'ProcessText' in catalog[0]['modules.Text']
    current['paths'][0] = [key for field in fields for key in ('control_' + field, 'control_' + field + '_integer')] + current['paths'][0]
    result, requests = controlled_plan(current, tmp_path)
    assert result['canAutoRun'], result['issues']
    assert len(requests) == 1 and requests[0]['width'] == 128


def adapter_chain(current):
    current = deepcopy(current)
    for key, source, output in (('adapter1', 'diffusion.load_models', 'pipeline'), ('adapter2', 'adapter1', 'output')):
        current['nodes'][key] = {'module': 'modules.DiffusersImage', 'action': 'LoadAdapter', 'params': {
            'pipeline': {'sourceId': source, 'sourceKey': output}, 'adapter_path': {'value': None}}}
    for stage in ('diffusion.encode_image', 'diffusion.denoise', 'diffusion.decode_latents'):
        current['nodes'][stage]['params']['pipeline'] = {'sourceId': 'adapter2', 'sourceKey': 'output'}
    current['paths'][0][1:1] = ['adapter1', 'adapter2']
    return current


@pytest.mark.parametrize('model_type,task', [('FluxFillPipeline', 'inpaint'), ('FluxCannyPipeline', 'control_image'), ('FluxDepthPipeline', 'control_image')])
def test_exact_empty_two_adapter_stage_chain_reaches_existing_recipe(catalog, tmp_path, model_type, task):
    current, _spec = graph(catalog, model_type, task)
    current = adapter_chain(current)
    result, requests = controlled_plan(current, tmp_path)
    assert result['canAutoRun'], result['issues']
    assert len(requests) == 1 and len(result['loaders'][0]['consumers']) == 5


@pytest.mark.parametrize('corruption', ['alias', 'bypass', 'shared', 'cycle', 'foreign', 'populated'])
def test_adapter_route_cannot_hide_shared_foreign_or_unbudgeted_outputs(catalog, tmp_path, corruption):
    current, _spec = graph(catalog, 'FluxCannyPipeline', 'control_image')
    current = adapter_chain(current)
    nodes = current['nodes']
    if corruption == 'alias':
        nodes['diffusion.decode_latents']['params']['pipeline']['sourceKey'] = 'pipeline'
    elif corruption == 'bypass':
        nodes['diffusion.encode_image']['params']['pipeline'] = {'sourceId': 'adapter1', 'sourceKey': 'output'}
    elif corruption == 'shared':
        nodes['extra'] = {'module': 'custom.Unknown', 'action': 'Observe', 'params': {'pipeline': {'sourceId': 'adapter1', 'sourceKey': 'output'}}}
        current['paths'][0].append('extra')
    elif corruption == 'cycle':
        nodes['adapter1']['params']['pipeline'] = {'sourceId': 'adapter2', 'sourceKey': 'output'}
        with pytest.raises(ValueError, match='cycle or an input ordered'):
            controlled_plan(current, tmp_path)
        return
    elif corruption == 'foreign':
        nodes['adapter1']['module'] = 'custom.Unknown'
    else:
        nodes['adapter1']['params']['adapter_path'] = {'value': {'source': 'local', 'value': 'not-inspected'}}
    result, _requests = controlled_plan(current, tmp_path)
    assert not result['canAutoRun'] and result['patches'] == []



@pytest.mark.parametrize('model_type,task', [('FluxFillPipeline', 'inpaint'), ('FluxCannyPipeline', 'control_image'), ('FluxDepthPipeline', 'control_image')])
def test_equivalent_numeric_string_and_primitive_integer_controls(catalog, tmp_path, model_type, task):
    current, _spec = graph(catalog, model_type, task)
    current['nodes']['diffusion.encode_image']['params']['width'] = {'value': '128'}
    current['nodes']['text'] = {'module': 'modules.Primitive', 'action': 'TextValue', 'params': {'text': {'value': '128'}}}
    current['nodes']['integer'] = {'module': 'modules.Text', 'action': 'ProcessText', 'params': {
        'source': {'sourceId': 'text', 'sourceKey': 'output'}, 'operation': {'value': 'data_conversion'},
        'target_type': {'value': 'integer'}}}
    current['nodes']['diffusion.denoise']['params']['height'] = {'sourceId': 'integer', 'sourceKey': 'output'}
    current['paths'][0][0:0] = ['text', 'integer']
    result, requests = controlled_plan(current, tmp_path)
    assert result['canAutoRun'], result['issues']
    assert len(requests) == 1 and requests[0]['width'] == requests[0]['height'] == 128


@pytest.mark.parametrize('field', ['generator', 'prompt_embeds', 'latents'])
def test_nonempty_supplied_material_descriptor_defaults_cannot_hide_unknown_batch(catalog, tmp_path, field):
    current, _spec = graph(catalog, 'FluxFillPipeline', 'inpaint')
    current['nodes']['diffusion.encode_image']['params'][field] = {'default': 'opaque supplied material'}
    result, requests = controlled_plan(current, tmp_path)
    assert not result['canAutoRun'] and not requests and result['patches'] == []


@pytest.mark.parametrize('selection', ['', '   ', {'source': 'local', 'value': ''}])
def test_cleared_direct_adapter_selection_retains_runtime_noop(catalog, tmp_path, selection):
    current, _spec = graph(catalog, 'FluxCannyPipeline', 'control_image')
    current = adapter_chain(current)
    for key in ('adapter1', 'adapter2'):
        current['nodes'][key]['params']['adapter_path'] = {'value': selection}
    result, requests = controlled_plan(current, tmp_path)
    assert result['canAutoRun'], result['issues']
    assert len(requests) == 1 and result['adapters'] == []


def _set_stage_scalar(current, field, value):
    for identity in ('diffusion.encode_image', 'diffusion.denoise'):
        params = current['nodes'][identity]['params']
        if field in params:
            params[field] = {'value': value}


@pytest.mark.parametrize('model_type,task', PAIRS)
def test_declared_runtime_bounds_refuse_all_six_off_grid_routes_before_recipe(catalog, tmp_path, model_type, task):
    current, _spec = graph(catalog, model_type, task)
    _set_stage_scalar(current, 'width', 129)
    result, requests = controlled_plan(current, tmp_path)
    assert not result['canAutoRun'] and not requests and result['patches'] == []
    assert 'increments of' in ' '.join(result['issues'])


@pytest.mark.parametrize('field,value', [
    ('width', 8), ('height', 2064), ('seed', 4294967296),
    ('max_sequence_length', 513), ('seed', None),
])
def test_declared_runtime_bounds_refuse_known_out_of_range_or_unknown_scalars(catalog, tmp_path, field, value):
    current, _spec = graph(catalog, 'FluxFillPipeline', 'inpaint')
    _set_stage_scalar(current, field, value)
    result, requests = controlled_plan(current, tmp_path)
    assert not result['canAutoRun'] and not requests and result['patches'] == []


def test_declared_runtime_bounds_use_the_selected_adapter_pixel_limit(catalog, tmp_path):
    from dataclasses import replace
    from modules.DiffusersImage.main import get_image_pipeline_adapter
    current, _spec = graph(catalog, 'FluxFillPipeline', 'inpaint')
    _set_stage_scalar(current, 'width', 256)
    _set_stage_scalar(current, 'height', 256)
    declared = replace(get_image_pipeline_adapter('FluxFillPipeline'), max_output_pixels=256 * 128)
    with patch('modules.DiffusersImage.main.get_image_pipeline_adapter', return_value=declared):
        result, requests = controlled_plan(current, tmp_path)
    assert not result['canAutoRun'] and not requests and result['patches'] == []
    assert 'reviewed output pixel bound' in ' '.join(result['issues'])


@pytest.mark.parametrize('count,side', [(9, 128), (5, 2048)])
def test_decoder_total_output_bounds_refuse_count_or_cumulative_pixels_before_recipe(catalog, tmp_path, count, side):
    current, _spec = graph(catalog, 'FluxFillPipeline', 'inpaint')
    _set_stage_scalar(current, 'width', side)
    _set_stage_scalar(current, 'height', side)
    current['nodes']['diffusion.encode_image']['params']['num_images_per_prompt'] = {'value': count}
    result, requests = controlled_plan(current, tmp_path)
    assert not result['canAutoRun'] and not requests and result['patches'] == []
    assert 'at most 8 images and 16 megapixels' in ' '.join(result['issues'])


def test_declared_runtime_bounds_preserve_normalized_boundary_values(catalog, tmp_path):
    current, _spec = graph(catalog, 'FluxFillPipeline', 'inpaint')
    for field, value in [('width', '16'), ('height', '32'), ('seed', '4294967295'), ('max_sequence_length', '512')]:
        _set_stage_scalar(current, field, value)
    result, requests = controlled_plan(current, tmp_path)
    assert result['canAutoRun'], result['issues']
    assert requests[0]['width'] == 16 and requests[0]['height'] == 32
    assert requests[0]['maxSequenceLength'] == 512


def test_decoder_total_output_bound_does_not_invent_a_recipe_for_valid_count_eight(catalog, tmp_path):
    current, _spec = graph(catalog, 'FluxFillPipeline', 'inpaint')
    current['nodes']['diffusion.encode_image']['params']['num_images_per_prompt'] = {'value': '8'}
    result, requests = controlled_plan(current, tmp_path)
    assert len(requests) == 1 and requests[0]['batchSize'] == 8
    # The controlled historical recipe is one image: valid static geometry is
    # not authority to reuse a smaller workload envelope.
    assert not result['canAutoRun'] and result['patches'] == []
