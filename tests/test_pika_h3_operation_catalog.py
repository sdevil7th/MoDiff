"""Pika starters use shared stage schemas and retain the package-owned sampler."""
from unittest.mock import patch

import pytest

from modiff.operation_catalog import build_operation_catalog
from modiff.operation_starters import resolve_operation_starter
from modiff.supplemental_modular_contracts import PIKA_PIPELINE, PIKA_WORKFLOW_TASKS
from modiff.joint_av_studio_specs import PIKA_EXECUTION_PROFILE


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


@pytest.mark.parametrize('workflow,task', PIKA_WORKFLOW_TASKS.items())
def test_pika_explicit_profile_seals_source_and_leaves_qualification_closed(catalog, workflow, task):
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES, resolve_execution_profiles_for_loader
    from modiff.joint_av_model_adapters import PIKA_REPOSITORY, PIKA_REVISION
    from modiff.studio_execution_specs import studio_execution_spec_for_pair, studio_capability_definition
    from modiff.supplemental_modular_contracts import PIKA_RUNTIME_PROFILE

    profile = DIFFUSERS_EXECUTION_PROFILES[PIKA_EXECUTION_PROFILE]
    assert profile.required_diffusers_provider == PIKA_RUNTIME_PROFILE
    assert profile.public and not profile.live_proof
    assert profile.retry_offload_modes == ()
    modules, contracts = catalog
    result = resolve_operation_starter(modules, contracts,
        {'pipelineClass': PIKA_PIPELINE, 'task': task, 'executionProfileId': profile.id})
    loader = next(node for node in result['nodes'] if node['action'] == 'ModelsLoader')
    selected, reason = resolve_execution_profiles_for_loader(loader['module'], loader['action'], loader['values'])
    assert not reason and [item.id for item in selected] == [profile.id]
    assert loader['values']['repo_id'] == {'source': 'hub', 'value': PIKA_REPOSITORY}
    assert loader['values']['revision'] == PIKA_REVISION
    assert loader['values']['workflow_id'] == workflow
    spec = studio_execution_spec_for_pair(PIKA_PIPELINE, task)
    assert spec['executionProfileId'] == profile.id
    assert not any(field in {'num_inference_steps', 'guidance_scale', 'negative_prompt'} for _, field, _ in spec['bindings'])
    capability = studio_capability_definition(PIKA_PIPELINE)
    for field in ('autoEligible', 'templateEligible', 'galleryEligible', 'liveProof'):
        assert capability[field] is False
    assert capability['qualifiedModes'] == []
    assert capability['downloadFiles'] == []  # Never expose a full-repository download.


@pytest.mark.parametrize('pipeline,helper_action', [
    ('MiniMaxH3ModularPipeline', 'WorkflowJointAVReferenceAssembler'),
    (PIKA_PIPELINE, 'WorkflowJointAVReferenceAssembler'),
])
def test_reference_helpers_bind_only_fields_their_existing_schema_declares(catalog, pipeline, helper_action):
    modules, contracts = catalog
    result = resolve_operation_starter(modules, contracts,
        {'pipelineClass': pipeline, 'task': 'reference_to_video_with_audio'})
    helper = next(node for node in result['nodes'] if node['action'] == helper_action)
    assert set(helper['values']) <= set(helper['params'])
    assert helper['values']['pipeline_class'] == pipeline


def test_saved_legacy_h3_reference_helper_keeps_its_original_schema(catalog):
    modules, _contracts = catalog
    assert 'pipeline_class' not in modules['modules.ModularDiffusers']['WorkflowMiniMaxH3ReferenceAssembler']['params']


@pytest.mark.parametrize('pipeline', ['MiniMaxH3ModularPipeline', PIKA_PIPELINE])
def test_keyframe_starters_do_not_require_an_unrelated_reference_list(catalog, pipeline):
    modules, contracts = catalog
    result = resolve_operation_starter(modules, contracts,
        {'pipelineClass': pipeline, 'task': 'first_last_frame_to_video_with_audio'})
    assert not any(node['action'] == 'WorkflowJointAVReferenceAssembler' for node in result['nodes'])
    before = next(node for node in result['nodes'] if node['action'] == 'WorkflowJointAVBeforeEncode')
    assert before['params']['references']['hidden'] is True
    assert before['params']['image'].get('hidden') is not True
    assert before['params']['last_image'].get('hidden') is not True
    reference = resolve_operation_starter(modules, contracts,
        {'pipelineClass': pipeline, 'task': 'reference_to_video_with_audio'})
    assert any(node['action'] == 'WorkflowJointAVReferenceAssembler' for node in reference['nodes'])
    before = next(node for node in reference['nodes'] if node['action'] == 'WorkflowJointAVBeforeEncode')
    assert before['params']['references'].get('hidden') is not True
    assert before['params']['image']['hidden'] is True and before['params']['last_image']['hidden'] is True
