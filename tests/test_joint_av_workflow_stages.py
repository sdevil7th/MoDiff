"""Joint AV stages keep pipeline/state identity and upstream sampler semantics."""
from unittest import mock
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import pytest

from modiff.modular_whole_workflow_contracts import reviewed_whole_workflow_graph_adapter
from modules.ModularDiffusers import workflow_blocks as blocks


PIKA = 'PikaH3FlexModularPipeline'
H3 = 'MiniMaxH3ModularPipeline'


@pytest.mark.parametrize('workflow', ['t2va', 'fl2va', 'ref2va'])
def test_pika_uses_generic_joint_stages_without_a_steps_binding(workflow):
    contract = reviewed_whole_workflow_graph_adapter(PIKA, workflow)
    assert contract is not None
    assert 'num_inference_steps' not in contract['requiredInputs']
    assert contract['actionSequence'][-2:] == ['workflow_joint_av_denoise', 'workflow_joint_av_decoder']
    assert len(contract['stateEdges']) == len(contract['actionSequence']) - 1


@pytest.mark.parametrize('pipeline_class', [H3, PIKA])
def test_joint_text_stage_seals_actual_pipeline_identity(pipeline_class):
    node = blocks.WorkflowJointAVTextEncode('joint-text')
    token, state = object(), object()
    pipeline = mock.Mock(return_value=state)
    with mock.patch.object(node, '_prepare_pipeline', return_value=(token, pipeline)):
        result = node.execute(pipeline_class=pipeline_class, workflow_id='t2va',
                              block_path='text_encoder', prompt='A bell rings')
    assert blocks._require_workflow_state(result['state_out'], token=token,
        pipeline_class=pipeline_class, workflow_id='t2va', completed_stage='text_encoder') is state
    other = H3 if pipeline_class == PIKA else PIKA
    with pytest.raises(ValueError):
        blocks._require_workflow_state(result['state_out'], token=token,
            pipeline_class=other, workflow_id='t2va', completed_stage='text_encoder')


def test_pika_denoise_preserves_fixed_schedule_and_seed_without_steps_or_cfg():
    node = blocks.WorkflowJointAVDenoise('joint-denoise')
    token, state, denoised = object(), object(), object()
    sealed = blocks._issue_workflow_state(token=token, pipeline_class=PIKA,
        workflow_id='t2va', completed_stage='text_encoder', state=state)
    pipeline = mock.Mock(return_value=denoised)
    args = dict(pipeline_class=PIKA, workflow_id='t2va', block_path='denoise',
                state_in=sealed, width=832, height=480, num_frames=124, seed=17)
    with (mock.patch.object(node, '_prepare_pipeline', return_value=(token, pipeline)),
          mock.patch.object(blocks, 'modular_generator_from_seed', return_value='generator')):
        result = node.execute(**args)
    assert pipeline.call_args.kwargs == dict(state=state, generator='generator',
                                             width=832, height=480, num_frames=124)
    assert blocks._require_workflow_state(result['state_out'], token=token, pipeline_class=PIKA,
        workflow_id='t2va', completed_stage='denoise') is denoised
    for field in ('num_inference_steps', 'guidance_scale', 'negative_prompt'):
        with pytest.raises(ValueError, match='fixed'):
            node.execute(**args, **{field: 7})


def test_pika_keyframe_preparation_keeps_supplied_geometry_and_partial_frame_or():
    node = blocks.WorkflowJointAVBeforeEncode('joint-prepare')
    frame = Image.new('RGB', (832, 480), 'red')
    pipeline = mock.Mock(return_value=object())
    with mock.patch.object(node, '_prepare_pipeline', return_value=(object(), pipeline)):
        node.execute(pipeline_class=PIKA, workflow_id='fl2va', block_path='before_encode',
                     last_image=frame, width=832, height=480, num_frames=120)
    assert pipeline.call_args.kwargs == dict(image=None, last_image=frame,
                                             width=832, height=480, num_frames=120)


def test_legacy_h3_nodes_still_reject_other_pipeline_families():
    with pytest.raises(ValueError, match='exact reviewed'):
        blocks.WorkflowMiniMaxH3TextEncode('legacy').execute(
            pipeline_class=PIKA, workflow_id='t2va', prompt='A bell rings')


def test_joint_reference_assembler_preserves_order_and_rejects_other_family_objects():
    @dataclass
    class Reference:
        image: object
        kind = 'image'
    frame = Image.new('RGB', (32, 32), 'red')
    node = blocks.WorkflowJointAVReferenceAssembler('joint-reference')
    original = Reference(frame)
    with mock.patch.object(blocks, '_joint_av_reference_types',
                           return_value=(Reference, Reference, None, None)) as resolver:
        result = node.execute(pipeline_class=PIKA, references_in=[original],
                              reference_kind='image', image=frame)['references']
        assert result[0] is original and result[1].image is frame
        resolver.assert_called_once_with(PIKA)
        with pytest.raises(ValueError, match='dataclass'):
            node.execute(pipeline_class=PIKA, references_in=[object()],
                         reference_kind='image', image=frame)
        for invalid in ({}, '', False, 0):
            with pytest.raises(ValueError, match='ordered list'):
                node.execute(pipeline_class=PIKA, references_in=invalid,
                             reference_kind='image', image=frame)


def test_normal_registry_resolves_joint_stage_schemas_and_validation_flags():
    import modules
    import modules.ModularDiffusers as package
    parsed = modules.parse_module_file(str(Path(blocks.__file__)), package)
    for name in ('BeforeEncode', 'TextEncode', 'VaeEncode', 'Denoise', 'Decode'):
        item = parsed['WorkflowJointAV' + name]
        assert item['skipParamsCheck'] is True
        assert item['resizable'] is True
        assert isinstance(item['params'], dict)
        assert item['params']['pipeline_class']['default'] == H3
    assert parsed['WorkflowJointAVDenoise']['params']['num_inference_steps'] == {
        'type': 'int', 'hidden': True}
    assert parsed['WorkflowJointAVReferenceAssembler']['params']['references']['type'] == 'joint_av_references'
