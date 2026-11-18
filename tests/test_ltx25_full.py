"""No-weight Full/SFT integration tests; no pretrained-output claim."""
from types import SimpleNamespace
from unittest.mock import Mock
import inspect

import numpy as np
from PIL import Image
import pytest

from modules.DiffusersVideo import ltx25_full as full
from modules.DiffusersVideo.main import GenerateVideoAudio


class Component:
    pass


class Pipeline:
    _modiff_video_pipeline_class = full.MODEL
    _modiff_video_repo = full.REPOSITORY
    _modiff_video_revision = full.REVISION
    _execution_device = 'cpu'
    vae_spatial_compression_ratio = 32
    vae_temporal_compression_ratio = 8

    def __init__(self):
        for name in full.COMPONENTS:
            setattr(self, name, Component())
        self.transformer.config = SimpleNamespace(num_layers=48)
        self.scheduler.config = {'use_dynamic_shifting': True, 'shift_terminal': .1}
        self.vocoder.config = {'output_sampling_rate': 48000}
        self.received = []

    def __call__(self, **arguments):
        self.received.append(arguments)
        return SimpleNamespace(frames=[[Image.new('RGB', (32, 32))]],
                               audio=np.zeros((1, 4800), dtype=np.float32))


def bound():
    pipeline = Pipeline()
    full.bind(pipeline)
    return pipeline


def invoke(pipeline=None, **values):
    pipeline = pipeline or bound()
    result = GenerateVideoAudio().execute(
        pipeline=pipeline, mode=values.pop('mode', full.MODES[0]),
        prompt='A bird flies above a forest lake.', **values,
    )
    return pipeline, result


def test_native_full_arguments_match_official_api_and_recommended_guidance():
    from diffusers import LTX2ConditionPipeline
    signature = inspect.signature(LTX2ConditionPipeline.__call__)
    pipeline, result = invoke(expand_prompt=True, seed=42)
    arguments = pipeline.received[0]
    assert set(arguments) <= set(signature.parameters)
    for name in ['num_inference_steps', 'guidance_scale', *full.GUIDANCE_DEFAULTS,
                 'spatio_temporal_guidance_blocks', 'use_cross_timestep']:
        assert arguments[name] == signature.parameters[name].default
    assert arguments['enable_prompt_enhancement'] is True
    assert arguments['generator'].initial_seed() == 42
    assert 'sigmas' not in arguments and 'timesteps' not in arguments
    assert result['sample_rate_out'] == 48000
    assert result['duration_seconds'] == pytest.approx(.1)


def test_full_receipt_records_actual_native_guidance_and_keyframe_positions():
    image = Image.new('RGB', (32, 32))
    node = GenerateVideoAudio()
    result = node.execute(pipeline=bound(), mode='reference_to_video_with_audio', prompt='A calm lake.',
        reference_images=[image, image], condition_indices_json='[0, -1]', audio_guidance_scale=4.5,
        temporal_guidance_blocks_json='[27,28]', expand_prompt=True)
    fields = node._execution_input_record['fields']
    assert fields['audio_guidance_scale']['value'] == 4.5
    assert fields['condition_indices_json']['value'] == [0, -1]
    assert fields['temporal_guidance_blocks_json']['value'] == [27, 28]
    assert fields['expand_prompt']['value'] is True
    assert '_native_generation_inputs' not in result


def test_full_owner_accepts_native_scheduler_timestep_updates_between_runs():
    from diffusers import FlowMatchEulerDiscreteScheduler
    pipeline = Pipeline()
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, shift_terminal=.1)
    full.bind(pipeline)
    for count in (30, 17):
        pipeline.scheduler.set_timesteps(count, device='cpu', mu=1.04)
        full.require_owner(pipeline)
        assert len(pipeline.scheduler.timesteps) == count


@pytest.mark.parametrize('mode', full.MODES)
def test_generic_facade_preserves_each_real_conditioning_route(mode):
    image = Image.new('RGB', (32, 32))
    values = {'mode': mode, 'guidance_scale': 0., 'audio_guidance_scale': 0.}
    if mode == 'image_to_video_with_audio':
        values['reference_images'] = [image]
    elif mode == 'reference_to_video_with_audio':
        values.update(reference_images=[image, image], condition_indices_json='[0,-1]')
    elif mode == 'video_to_video_with_audio':
        values['video'] = [image] * 9
    pipeline, result = invoke(**values)
    arguments = pipeline.received[0]
    assert arguments['guidance_scale'] == arguments['audio_guidance_scale'] == 0.
    conditions = arguments['conditions']
    if mode == 'text_to_video_with_audio':
        assert conditions is None
    else:
        assert conditions[0].frames[0] is image and conditions[0].index == 0
        if mode == 'reference_to_video_with_audio':
            assert len(conditions) == 2 and conditions[1].index == -1
        if mode == 'video_to_video_with_audio':
            assert len(conditions[0].frames) == 9
    assert result['frames_out'] == 1 and result['audio']['channels'] == 1


@pytest.mark.parametrize('name,value', [
    ('width', 33), ('height', -1), ('num_frames', 8), ('seed', True),
    ('num_inference_steps', 0), ('frame_rate', float('nan')), ('guidance_scale', -1),
    ('audio_guidance_scale', float('inf')), ('audio_guidance_rescale', 1.1),
    ('expand_prompt', 'yes'), ('use_cross_timestep', 1), ('num_videos_per_prompt', 2),
    ('temporal_guidance_blocks_json', '[48]'), ('temporal_guidance_blocks_json', '[true]'),
    ('temporal_guidance_blocks_json', '{}'), ('temporal_guidance_blocks_json', 'not json'),
    ('condition_indices_json', '[0]'), ('output_type', 'latent'), ('mask', [Image.new('L', (32, 32))]),
])
def test_invalid_or_unadmitted_controls_fail_before_inference(name, value):
    pipeline = bound()
    with pytest.raises((ValueError, TypeError)):
        invoke(pipeline, **{name: value})
    assert pipeline.received == []


@pytest.mark.parametrize('mode,values', [
    ('image_to_video_with_audio', {}), ('image_to_video_with_audio', {'reference_images': []}),
    ('video_to_video_with_audio', {}), ('reference_to_video_with_audio', {}),
    ('reference_to_video_with_audio', {'reference_images': [Image.new('RGB', (32, 32))]}),
    ('reference_to_video_with_audio', {'reference_images': [Image.new('RGB', (32, 32))], 'condition_indices_json': '[0,-1]'}),
])
def test_missing_or_misaligned_conditioning_never_runs(mode, values):
    pipeline = bound()
    with pytest.raises(ValueError):
        invoke(pipeline, mode=mode, **values)
    assert pipeline.received == []


@pytest.mark.parametrize('change', ['transformer', 'scheduler', 'vae', 'processor', 'prompt_enhancer', 'config'])
def test_full_recipe_cannot_silently_reuse_different_components_or_a_distilled_schedule(change):
    pipeline = bound()
    if change == 'config':
        pipeline.scheduler.config['use_dynamic_shifting'] = False
    else:
        setattr(pipeline, change, Component())
    with pytest.raises((ValueError, AttributeError)):
        invoke(pipeline)
    assert pipeline.received == []


def test_full_loader_selects_full_transformer_locally_before_constructing_pipeline(monkeypatch, tmp_path):
    import diffusers
    from modules.DiffusersVideo import wan_vace
    from modules.DiffusersRuntime import main as runtime
    from modules.DiffusersVideo.main import LoadPipeline

    pipeline = Pipeline()
    transformer = pipeline.transformer
    transformer_load = Mock(return_value=transformer)
    pipeline_load = Mock(return_value=pipeline)
    scheduler = pipeline.scheduler
    scheduler_factory = Mock(return_value=scheduler)
    monkeypatch.setattr(diffusers.LTX2VideoTransformer3DModel, 'from_pretrained', transformer_load)
    monkeypatch.setattr(diffusers.LTX2ConditionPipeline, 'from_pretrained', pipeline_load)
    monkeypatch.setattr(diffusers.FlowMatchEulerDiscreteScheduler, 'from_config', scheduler_factory)
    monkeypatch.setattr(wan_vace, 'exact_cached_snapshot_path', lambda repo, revision: (
        str(tmp_path) if (repo, revision) == (full.REPOSITORY, full.REVISION) else pytest.fail('Wrong snapshot')))
    monkeypatch.setattr(wan_vace, 'apply_pipeline_offload', Mock())
    monkeypatch.setattr(runtime, 'apply_execution_recipe_to_pipeline', Mock())
    node = LoadPipeline()
    node.progress, node.mm_add = Mock(), Mock()
    result = node.execute(pipeline_class=full.MODEL, model_id={'source': 'hub', 'value': full.REPOSITORY},
                          revision=full.REVISION, dtype='bfloat16', device='cpu', offload_mode='none', auto_offload=False)
    assert result['pipeline'] is pipeline
    assert transformer_load.call_args.kwargs['subfolder'] == 'transformer_full'
    assert transformer_load.call_args.kwargs['local_files_only'] is True
    assert transformer_load.call_args.kwargs['use_safetensors'] is True
    assert pipeline_load.call_args.kwargs['transformer'] is transformer
    assert pipeline_load.call_args.kwargs['local_files_only'] is True
    assert scheduler_factory.call_args.kwargs == {'use_dynamic_shifting': True, 'shift_terminal': .1}
    full.require_owner(pipeline)


@pytest.mark.parametrize('mode', full.MODES)
def test_full_uses_existing_generic_nodes_without_promoting_qualification(mode):
    from modules import MODULE_MAP
    from modiff.operation_catalog import build_operation_catalog
    from modiff.operation_starters import resolve_operation_starter
    from modiff.studio_execution_specs import studio_capability_definition

    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    draft = resolve_operation_starter(MODULE_MAP, contracts, {'pipelineClass': full.MODEL, 'task': mode})
    assert [row['action'] for row in draft['nodes']] == ['LoadPipeline', 'GenerateVideoAudio']
    capability = studio_capability_definition(full.MODEL)
    assert not any(capability[key] for key in ['autoEligible', 'templateEligible', 'galleryEligible', 'liveProof'])
    fields = draft['nodes'][1]['params']
    assert fields['audio_guidance_scale']['hidden'] is False
    assert fields['condition_indices_json']['hidden'] is (mode != 'reference_to_video_with_audio')
    assert fields['condition_indices_json'].get('required', False) is (mode == 'reference_to_video_with_audio')
    assert fields['stg_scale']['fieldOptions']['controlTier'] == 'advanced'
    assert fields['width']['default'] == 768 and fields['height']['default'] == 512
    assert fields['num_inference_steps']['default'] == 30 and fields['guidance_scale']['default'] == 3.


def test_existing_distilled_contract_remains_exact_and_separate():
    from modiff.ltx25_execution_contract import LTX25_DISTILLED_EXECUTION
    assert LTX25_DISTILLED_EXECUTION.transformer_subfolder == 'transformer'
    assert LTX25_DISTILLED_EXECUTION.repository_revision == 'a6de4b5354f078db24d9cf4778c14846788aea3d'
    assert LTX25_DISTILLED_EXECUTION.guidance_scale == LTX25_DISTILLED_EXECUTION.audio_guidance_scale == 1.
    assert len(LTX25_DISTILLED_EXECUTION.sigmas) == 8
