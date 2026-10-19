"""Custom schedules belong to the real sampler, never an inert encoder input."""
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from image_stage_test_support import stage_offline_runtime as stage_offline_runtime
from test_image_stage_numerical import old_call, selected_owner, values

pytestmark = pytest.mark.usefixtures('stage_offline_runtime')

PAIRS = (
    ('FluxFillPipeline', 'inpaint'), ('FluxFillPipeline', 'outpaint'),
    ('FluxCannyPipeline', 'control_image'), ('FluxDepthPipeline', 'control_image'),
    ('FluxReduxPipeline', 'edit_image'), ('FluxReduxPipeline', 'multi_image_reference_edit'),
)
SCHEDULE = [1.0, 0.5]


@pytest.fixture(scope='module')
def catalog():
    from modules import MODULE_MAP
    from modiff.operation_catalog import build_operation_catalog
    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    return MODULE_MAP, contracts


@pytest.mark.parametrize('model_type,task', PAIRS)
def test_selected_encoder_omits_schedule_but_retains_sampler_and_whole_contracts(catalog, model_type, task):
    from modiff.operation_starters import resolve_operation_starter
    from modiff.studio_execution_specs import studio_execution_spec_for_pair
    spec = studio_execution_spec_for_pair(model_type, task)
    before = deepcopy(catalog[0]['modules.DiffusersImage'])
    with patch('modiff.NodeBase.NodeBase.__init__', side_effect=AssertionError('node constructed')):
        staged = resolve_operation_starter(*catalog, {'pipelineClass': spec['pipelineClass'], 'task': task,
            'executionProfileId': spec['executionProfileId']})
        whole = resolve_operation_starter(*catalog, {'pipelineClass': spec['pipelineClass'], 'task': task})
    encoder, = [row for row in staged['nodes'] if row['action'] == 'EncodeInputs']
    assert 'sigmas' not in encoder['params']
    assert 'sigmas' not in {port['name'] for port in encoder['operation']['ports']}
    if model_type != 'FluxReduxPipeline':
        sampler, = [row for row in staged['nodes'] if row['action'] == 'Denoise']
        consumer, = [row for row in whole['nodes'] if row['action'] != 'LoadPipeline']
        assert 'sigmas' in sampler['params']
        assert 'sigmas' in {port['name'] for port in sampler['operation']['ports']}
        assert 'sigmas' in consumer['params']
        assert 'sigmas' in {port['name'] for port in consumer['operation']['ports']}
    assert catalog[0]['modules.DiffusersImage'] == before


@pytest.mark.parametrize('kind', ['fill', 'control'])
@pytest.mark.parametrize('warm', [False, True])
def test_actual_encoder_rejects_non_none_schedule_before_cold_or_cached_encoding(kind, warm):
    from modules.DiffusersImage import main
    owner, _strategy, _selection = selected_owner(kind)
    given = {'pipeline': owner, **values()}
    encoder = main.EncodeInputs('encoder-schedule-refusal')
    previous = encoder(**given)['encoded_inputs'] if warm else None
    with patch.object(owner.text_encoder, 'forward', wraps=owner.text_encoder.forward) as forward:
        with pytest.raises(ValueError, match='Connect the noise schedule to Denoise'):
            encoder(**{**given, 'sigmas': SCHEDULE})
        assert forward.call_count == 0
    assert encoder.output.get('encoded_inputs') is previous
    if warm:
        assert encoder(**given)['encoded_inputs'] is previous


@pytest.mark.parametrize('kind', ['fill', 'control'])
@pytest.mark.parametrize('explicit_none', [False, True])
def test_actual_encoder_keeps_missing_and_none_schedule_eligible(kind, explicit_none):
    from modules.DiffusersImage import main
    owner, strategy, _selection = selected_owner(kind)
    given = {'pipeline': owner, **values()}
    if explicit_none:
        given['sigmas'] = None
    encoder = main.EncodeInputs('encoder-default-schedule')
    result = encoder(**given)['encoded_inputs']
    from modules.DiffusersImage import stage_execution as core
    assert core.require_encoding(owner, strategy, result, given) is result
    assert encoder(**given)['encoded_inputs'] is result


@pytest.mark.parametrize('kind', ['fill', 'control'])
def test_actual_denoise_and_whole_facades_use_same_official_custom_schedule(kind):
    from modules.DiffusersImage import main
    owner, _strategy, _selection = selected_owner(kind)
    official = deepcopy(owner)
    given = {**values(), 'num_inference_steps': 4}
    expected = official(**{**old_call(kind, given, 'latent'), 'sigmas': SCHEDULE}).images
    encoder = main.EncodeInputs('custom-schedule-encode')
    encoded = encoder(pipeline=owner, **given)['encoded_inputs']
    sampler = main.Denoise('custom-schedule-denoise')
    result = sampler(pipeline=owner, encoded_inputs=encoded, **given, sigmas=SCHEDULE)
    torch.testing.assert_close(result['latents_out'], expected, rtol=0, atol=0)
    assert owner.num_timesteps == len(SCHEDULE)
    whole = main.Inpaint('custom-schedule-whole') if kind == 'fill' else main.ControlGenerate('custom-schedule-whole')
    result = whole(pipeline=owner, **given, output_type='latent', sigmas=SCHEDULE)
    torch.testing.assert_close(result['latents_out'], expected, rtol=0, atol=0)
    assert owner.num_timesteps == len(SCHEDULE)


def test_actual_redux_encoder_retains_original_unsupported_schedule_error():
    from modules.DiffusersImage import main
    from redux_stage_test_support import make_actual_tiny_bundle, PRIOR_REPO, PRIOR_REVISION, BASE_REPO, BASE_REVISION
    owner = make_actual_tiny_bundle()
    adapter = main.IMAGE_PIPELINE_ADAPTERS['FluxReduxPipeline']
    main._tag_image_pipeline(owner, adapter, 'edit_image', PRIOR_REPO, 'hub', PRIOR_REVISION)
    owner._modiff_loaded_stage_artifacts = ((PRIOR_REPO, PRIOR_REVISION), (BASE_REPO, BASE_REVISION))
    main._bind_loaded_image_stage(owner, adapter, {
        'execution_profile_id': 'flux-redux:direct', 'mode': 'edit_image',
        'pipeline_class': 'FluxReduxPipeline', 'model_id': {'source': 'hub', 'value': PRIOR_REPO},
        'revision': PRIOR_REVISION,
    })
    with patch.object(owner.prior.image_encoder, 'forward', wraps=owner.prior.image_encoder.forward) as forward:
        with pytest.raises(ValueError, match='FluxReduxPipeline does not support optional input.*sigmas'):
            main.EncodeInputs('redux-schedule-refusal')(pipeline=owner, **values(), sigmas=SCHEDULE)
        assert forward.call_count == 0
