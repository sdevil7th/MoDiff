"""No-weight dispatch, ownership and official split-video contracts."""
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import pytest
import torch

from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
from modiff.model_artifact_catalog import require_catalog_revision
from modules.DiffusersVideo import stage_execution as core
from modules.DiffusersVideo.stage_nodes import EncodeInputs, Denoise, DecodeLatents


@pytest.mark.parametrize('action', ('EncodeInputs', 'Denoise', 'DecodeLatents'))
def test_registered_video_stages_resolve_through_the_actual_executor_entrypoint(action):
    from importlib import import_module
    # The server dispatches getattr(import_module(module + '.main'), action).
    # Schema registration alone does not prove this executable lookup succeeds.
    klass = getattr(import_module('modules.DiffusersVideo.main'), action)
    assert klass is globals()[action]
    node = klass('video-executor-entrypoint-' + action)
    assert node.module_name == 'modules.DiffusersVideo' and node.class_name == action


@pytest.mark.parametrize('action', ('EncodeInputs', 'Denoise'))
def test_video_stage_authoring_callbacks_do_not_construct_executable_nodes(action):
    from modiff.field_metadata import is_metadata_field_action, metadata_field_callback
    from modiff.field_metadata_context import FieldMessageContext
    from modules.DiffusersVideo.main import get_video_pipeline_adapter, _adapter_signal
    assert is_metadata_field_action({'module': 'modules.DiffusersVideo', 'action': action,
                                    'fn': 'update_stage_contract'})
    with patch('modiff.NodeBase.NodeBase.__init__', side_effect=AssertionError('constructed')), patch.object(
        FieldMessageContext, 'set_field_params', autospec=True,
    ) as update:
        callback = metadata_field_callback(action, 'update_stage_contract', node_id='draft', sid='s', module='modules.DiffusersVideo')
        callback({'video_contract': _adapter_signal(get_video_pipeline_adapter('LTXConditionPipeline')),
                  'mode': 'image_to_video'}, None)
    assert any(call.args[1] == 'mode' for call in update.call_args_list)


class Tokenizer:
    def __call__(self, *args, **kwargs):
        return {'input_ids': [1, 2]}


class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter('weight', torch.nn.Parameter(torch.ones(1)))
        self.config = SimpleNamespace(latents_mean=[2.], latents_std=[3.], z_dim=1,
                                      scaling_factor=1., timestep_conditioning=True)
        self.latents_mean, self.latents_std = torch.tensor([2.]), torch.tensor([3.])
        self.last = None

    @property
    def dtype(self):
        return self.weight.dtype

    def decode(self, value, *args, **kwargs):
        assert not torch.is_grad_enabled(), 'separate decode must retain official inference mode'
        self.last = value.clone()
        return (value,)


class Scheduler:
    config = {}


class Pipeline:
    def __init__(self, identity):
        self.identity = identity
        self.text_encoder = torch.nn.Linear(1, 1)
        self.tokenizer = Tokenizer()
        self.transformer = torch.nn.Linear(1, 1)
        self.scheduler = Scheduler()
        self.vae = VAE()
        self.vae_scale_factor_temporal = 4
        self._execution_device = 'cpu'
        profile = DIFFUSERS_EXECUTION_PROFILES[identity]
        if profile.pipeline_class == 'WanImageToVideoPipeline':
            self.transformer_2 = torch.nn.Linear(1, 1)
            self.image_encoder = None
            self.image_processor = None
        self._modiff_video_pipeline_class = profile.pipeline_class
        self._modiff_video_repo = profile.default_repo
        self._modiff_video_revision = require_catalog_revision(profile.default_repo)
        self.calls = 0
        self.video_processor = SimpleNamespace(postprocess_video=lambda *args, **kwargs: [[Image.new('RGB', (32, 32)) for _ in range(9)]])

    def encode_prompt(self, **kwargs):
        assert not torch.is_grad_enabled(), 'separate encoding must retain official inference mode'
        self.calls += 1
        out = (torch.ones(1, 4, 2), torch.ones(1, 4, 2) if kwargs['do_classifier_free_guidance'] else None)
        if self.identity.startswith('ltx'):
            return out[0], torch.ones(1, 4), out[1], torch.ones(1, 4) if out[1] is not None else None
        return out

    @staticmethod
    def _denormalize_latents(value, mean, std, factor):
        return value * std.view(1, -1, 1, 1, 1) / factor + mean.view(1, -1, 1, 1, 1)

    def maybe_free_model_hooks(self):
        pass


def owner(identity='ltx-video:direct'):
    pipeline = Pipeline(identity)
    values = {'execution_profile_id': identity, 'pipeline_class': pipeline._modiff_video_pipeline_class,
              'model_id': {'source': 'hub', 'value': pipeline._modiff_video_repo},
              'revision': pipeline._modiff_video_revision,
              'device': 'cpu', 'dtype': 'bfloat16', 'auto_offload': False, 'offload_mode': 'none'}
    core.bind_owner(pipeline, values)
    return pipeline, values


def encoded(pipeline, **changes):
    mode = DIFFUSERS_EXECUTION_PROFILES[pipeline.identity].modes[0]
    media = {'reference_images': [Image.new('RGB', (32, 32))]} if mode == 'image_to_video' else (
        {'video': [Image.new('RGB', (32, 32)) for _ in range(9)]} if mode == 'video_to_video' else {})
    return core.encode(pipeline, {'mode': mode, 'prompt': 'A moving train', **media,
        'negative_prompt': '', 'guidance_scale': 1., 'max_sequence_length': 128, **changes})


@pytest.mark.parametrize('identity', core.PROFILE_CLASSES)
def test_real_profile_binding_rejects_foreign_mutable_and_advisory_artifacts(identity):
    pipeline, values = owner(identity)
    core.require_owner(pipeline)
    for key, changed in [('revision', 'main'), ('execution_profile_id', ''), ('model_id', {'source': 'local', 'value': 'arbitrary'})]:
        core.bind_owner(pipeline, {**values, key: changed})
        with pytest.raises(ValueError, match='ownership'):
            core.require_owner(pipeline)
        core.bind_owner(pipeline, values)


def test_encoder_node_dispatch_reuses_unchanged_text_and_reencodes_after_adapter_change():
    pipeline, _ = owner()
    node = EncodeInputs('video-encoding-unit')
    args = {'pipeline': pipeline, 'mode': 'text_to_video', 'prompt': 'A moving train',
            'negative_prompt': '', 'guidance_scale': 1., 'max_sequence_length': 128}
    first = node(**args)['encoded_inputs']
    assert node(**args)['encoded_inputs'] is first
    assert pipeline.calls == 1
    pipeline._modiff_adapter_scales = {'style': .7}
    second = node(**args)['encoded_inputs']
    assert second is not first and pipeline.calls == 2
    with pytest.raises(ValueError, match='recipe or adapters'):
        core.require_encoding(pipeline, first)


def test_encoding_rejects_foreign_owner_replaced_components_and_mutated_embeddings():
    pipeline, _ = owner()
    value = encoded(pipeline)
    other, _ = owner()
    with pytest.raises(ValueError, match='ownership'):
        core.require_encoding(other, value)
    value.embeddings['prompt_embeds'].add_(1)
    with pytest.raises(ValueError, match='modified'):
        core.require_encoding(pipeline, value)
    value = encoded(pipeline)
    pipeline.scheduler = Scheduler()
    with pytest.raises(ValueError, match='ownership'):
        core.require_encoding(pipeline, value)


def test_video_input_keeps_full_121_frame_recipe_and_ordered_owned_media():
    pipeline, _ = owner()
    frames = [Image.new('RGB', (32, 32), (index, 0, 0)) for index in range(121)]
    value = encoded(pipeline, mode='video_to_video', video=frames)
    frozen = [frame for chunk in value.media['video_chunks'] for frame in chunk]
    assert len(frozen) == 121
    assert [frame.getpixel((0, 0))[0] for frame in frozen] == list(range(121))
    frames[0].putpixel((0, 0), (255, 0, 0))
    assert frozen[0].getpixel((0, 0)) == (0, 0, 0)


def test_video_input_receipt_identifies_frozen_order_and_repeated_frames_without_paths():
    import hashlib
    import json
    pipeline, _ = owner()
    first, second = Image.new('RGB', (32, 32), 'red'), Image.new('RGB', (32, 32), 'blue')
    frames = [first, second, first]
    value = encoded(pipeline, mode='video_to_video', video=frames)
    descriptors = [[image.mode, list(image.size), hashlib.sha256(image.tobytes()).hexdigest()] for image in frames]
    expected = hashlib.sha256(b'modiff-video-source-pixels-v1\0' +
        json.dumps(descriptors, separators=(',', ':'), ensure_ascii=True).encode('ascii')).hexdigest()
    assert value.consumed['video_source_sha256'] == expected
    assert value.consumed['video_source_frames'] == 3
    assert encoded(pipeline, mode='video_to_video', video=[second, first, first]).consumed['video_source_sha256'] != expected
    first.putpixel((0, 0), (0, 0, 0))
    core.require_encoding(pipeline, value)
    assert value.consumed['video_source_sha256'] == expected
    assert 'video_source_sha256' not in encoded(pipeline).consumed


def test_video_input_receipt_does_not_read_tensor_pixels_or_invent_a_pixel_digest():
    pipeline, _ = owner()
    value = encoded(pipeline, mode='video_to_video', video=[torch.zeros(3, 32, 32)])
    assert value.consumed['video_source_frames'] == 1
    assert value.consumed['video_source_sha256'] is None


def test_actual_ltx_adapter_captures_condition_and_denoise_strength_separately():
    from modiff.execution_input_provenance import bind_generation_input_origins, build_resolved_execution_inputs
    from modules.DiffusersVideo.main import get_video_pipeline_adapter
    pipeline, _ = owner()
    encoder, sampler = EncodeInputs('encode-source'), Denoise('denoise-source')
    value = encoder(pipeline=pipeline, mode='video_to_video', video=[Image.new('RGB', (32, 32))],
        prompt='A moving train', negative_prompt='', guidance_scale=1., max_sequence_length=128)['encoded_inputs']
    calls = []
    def sample(pipe, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(frames=torch.ones(1, 1, 2, 2, 2))
    with patch.object(Pipeline, '__call__', sample, create=True), patch(
        'modules.DiffusersVideo.main._pipeline_adapter', return_value=get_video_pipeline_adapter('LTXConditionPipeline'),
    ), patch('modules.DiffusersVideo.main._install_ltx_dynamic_shift', return_value=None), patch.object(
        sampler, '_execute_adapter', side_effect=lambda p, a, m, v: sampler._execute_ltx(p, a, m, v),
    ):
        sampler(pipeline=pipeline, encoded_inputs=value, mode='video_to_video', guidance_scale=1.,
            width=32, height=32, num_frames=9, frame_rate=24., seed=7, num_inference_steps=8,
            strength=.65, denoise_strength=.8)
    assert calls[0]['conditions'][0].strength == .65
    assert calls[0]['denoise_strength'] == .8
    encode_node = {'module': 'modules.DiffusersVideo', 'action': 'EncodeInputs', 'params': {
        'video': {'sourceId': 'source', 'sourceKey': 'video'}}}
    denoise_node = {'module': 'modules.DiffusersVideo', 'action': 'Denoise', 'params': {
        'encoded_inputs': {'sourceId': encoder.node_id, 'sourceKey': 'encoded_inputs'},
        'strength': {'sourceId': 'control', 'sourceKey': 'value'}}}
    records = {node.node_id: bind_generation_input_origins(node.node_id, descriptor, node._execution_input_record,
        source_fields=node._execution_input_source_fields) for node, descriptor in (
            (encoder, encode_node), (sampler, denoise_node))}
    receipt = build_resolved_execution_inputs({'nodes': {encoder.node_id: encode_node, sampler.node_id: denoise_node}},
        records, task_id='actual-call', attempt_index=0, node_id=sampler.node_id)
    assert receipt['summary']['conditioningStrength'] == .65
    assert receipt['summary']['denoiseStrength'] == .8
    assert receipt['summary']['videoSourceFrames'] == 1
    assert len(receipt['summary']['videoSourceSha256']) == 64
    assert records[sampler.node_id]['fields']['conditioning_strength']['sourceNodeId'] == 'control'
    assert records[encoder.node_id]['fields']['video_source_sha256']['sourceNodeId'] == 'source'
    assert 'strength' not in receipt['summary'], 'Do not label condition strength as an ordinary denoise strength'


def test_ltx_encoder_normalizes_the_existing_shared_sequence_default_before_real_encoding():
    pipeline, _ = owner()
    with patch.object(pipeline, 'encode_prompt', wraps=pipeline.encode_prompt) as encode:
        value = encoded(pipeline, max_sequence_length=512)
    assert encode.call_args.kwargs['max_sequence_length'] == 128
    assert value.max_sequence_length == value.consumed['max_sequence_length'] == 128


@pytest.mark.parametrize('identity', [key for key, cls in core.PROFILE_CLASSES.items() if cls != 'WanVACEPipeline'])
def test_split_decode_uses_official_normalization_and_current_latent(identity):
    pipeline, _ = owner(identity)
    encoding = encoded(pipeline)
    tensor = torch.ones(1, 1, 2, 2, 2)
    generator = torch.Generator().manual_seed(7)
    latent = core.issue_latent(pipeline, encoding, tensor, (32, 32, 9), generator)
    # This is the exact inverse normalization from the pinned LTX and Wan
    # decoders. LTX's actual default decode timestep is zero, not 0.05.
    result = core.decode(pipeline, latent)
    assert torch.equal(pipeline.vae.last, torch.full_like(tensor, 5.))
    assert len(result['video_out']) == 9
    with pytest.raises(ValueError, match='modified'):
        core.require_latent(pipeline, latent, tensor)
    latent.tensor.add_(1)
    with pytest.raises(ValueError, match='modified'):
        core.decode(pipeline, latent)


def test_denoise_invokes_existing_sampler_with_real_embeddings_and_fresh_seed():
    pipeline, _ = owner()
    value = encoded(pipeline)
    node = Denoise('video-denoising-unit')
    seen = []
    def actual_adapter(pipe, adapter, mode, args):
        kwargs = {'width': 32, 'height': 32, 'num_frames': 9,
                  'generator': torch.Generator().manual_seed(args['seed']), 'prompt': None}
        seen.append(kwargs)
        return node._invoke_pipeline(pipe, kwargs)
    def call(**kwargs):
        assert kwargs['output_type'] == 'latent'
        assert kwargs['prompt_embeds'] is value.embeddings['prompt_embeds']
        assert 'prompt_attention_mask' in kwargs
        return SimpleNamespace(frames=torch.randn((1, 1, 2, 2, 2), generator=kwargs['generator']))
    from modules.DiffusersVideo.main import get_video_pipeline_adapter
    adapter = get_video_pipeline_adapter('LTXConditionPipeline')
    # The existing adapter remains the sampler call owner. These tiny tensors
    # prove dispatch/cache/RNG behavior, not model output quality.
    with patch.object(Pipeline, '__call__', lambda self, **kw: call(**kw), create=True), patch(
        'modules.DiffusersVideo.main._pipeline_adapter', return_value=adapter,
    ), patch.object(node, '_execute_adapter', side_effect=actual_adapter):
        args = {'pipeline': pipeline, 'mode': 'text_to_video', 'guidance_scale': 1.,
                'seed': 7, 'encoded_inputs': value}
        first = node(**args)
        node.invalidate_cache()
        second = node(**args)
    assert len(seen) == 2
    assert first['latent_binding'] is not second['latent_binding']
    assert torch.equal(first['latents_out'], second['latents_out'])


def test_decoder_dispatch_rejects_a_fabricated_binding_before_tensor_work():
    pipeline, _ = owner()
    with pytest.raises(ValueError, match='genuine'):
        DecodeLatents('video-decode-unit')(pipeline=pipeline, mode='text_to_video',
            latents=torch.ones(1), latent_binding={'seed': 7})


@pytest.mark.parametrize('identity', [key for key, cls in core.PROFILE_CLASSES.items() if cls != 'WanVACEPipeline'])
def test_actual_split_calls_capture_video_task_prompt_and_final_format_without_form_fallback(identity):
    from modiff.execution_input_provenance import capture_generation_inputs, build_resolved_execution_inputs, apply_resolved_execution_inputs
    from modules.DiffusersVideo.main import get_video_pipeline_adapter
    pipeline, _ = owner(identity)
    mode = DIFFUSERS_EXECUTION_PROFILES[identity].modes[0]
    media = {'reference_images': [Image.new('RGB', (32, 32))]} if mode == 'image_to_video' else (
        {'video': [Image.new('RGB', (32, 32)) for _ in range(9)]} if mode == 'video_to_video' else {})
    encoder, sampler, decoder = EncodeInputs('encode'), Denoise('sample'), DecodeLatents('decode')
    encoding = encoder(pipeline=pipeline, mode=mode, prompt='A moving train', **media,
                       negative_prompt='', guidance_scale=1., max_sequence_length=128)['encoded_inputs']
    def sample(pipe, adapter, mode, args):
        controls = {'guidance_scale_2': 3.5} if adapter.pipeline_class == 'WanImageToVideoPipeline' else {}
        return sampler._invoke_pipeline(pipe, {'width': 32, 'height': 32, 'num_frames': 9,
            'generator': torch.Generator().manual_seed(7), **controls})
    with patch.object(Pipeline, '__call__', lambda self, **kw: SimpleNamespace(frames=torch.ones(1, 1, 2, 2, 2)), create=True), patch(
        'modules.DiffusersVideo.main._pipeline_adapter', return_value=get_video_pipeline_adapter(pipeline._modiff_video_pipeline_class),
    ), patch.object(sampler, '_execute_adapter', side_effect=sample):
        latent = sampler(pipeline=pipeline, encoded_inputs=encoding, mode=mode, guidance_scale=1., seed=7)
    decoder(pipeline=pipeline, latents=latent['latents_out'], latent_binding=latent['latent_binding'])
    nodes = {'owner': {'module': 'modules.DiffusersVideo', 'action': 'LoadPipeline', 'params': {}}}
    for key, action in [('encode', 'EncodeInputs'), ('sample', 'Denoise'), ('decode', 'DecodeLatents')]:
        nodes[key] = {'module': 'modules.DiffusersVideo', 'action': action, 'params': {
            'pipeline': {'sourceId': 'owner', 'sourceKey': 'pipeline'}}}
    nodes['sample']['params']['encoded_inputs'] = {'sourceId': 'encode', 'sourceKey': 'encoded_inputs'}
    nodes['decode']['params']['latent_binding'] = {'sourceId': 'sample', 'sourceKey': 'latent_binding'}
    records = {node.node_id: node._execution_input_record for node in (encoder, sampler, decoder)}
    records['owner'] = capture_generation_inputs('owner', nodes['owner'], {'pipeline_class': pipeline._modiff_video_pipeline_class})
    receipt = build_resolved_execution_inputs({'nodes': nodes}, records, task_id='run', attempt_index=0, node_id='decode')
    assert receipt['graphTasksVersion'] == 2
    assert receipt['graphTasks'] == [{'loaderId': 'owner', 'pipelineClass': pipeline._modiff_video_pipeline_class, 'task': mode}]
    assert receipt['summary']['prompt'] == 'A moving train'
    # Blank text is normalized by the encoder; retain that exact observation.
    assert receipt['summary']['negativePrompt'] is None
    assert receipt['summary']['outputType'] == 'pil'
    assert receipt['summary']['executedTask'] == mode
    assert receipt['ambiguousFields'] == []
    result = apply_resolved_execution_inputs({'taskId': 'run', 'nodeId': 'decode', 'mode': 'text_to_image',
        'formSnapshot': {'mode': 'text_to_image', 'prompt': 'unused'}}, receipt)
    assert result['mode'] == mode and result['prompt'] == 'A moving train'
    assert result['formSnapshot'] == {'mode': 'text_to_image', 'prompt': 'unused'}


def test_video_task_receipts_require_complete_consistent_actual_stage_observations():
    from modiff.execution_input_provenance import capture_generation_inputs
    from modiff.workflow_task_identity import graph_task_receipts
    nodes = {'owner': {'module': 'modules.DiffusersVideo', 'action': 'LoadPipeline', 'params': {}}}
    records = {'owner': capture_generation_inputs('owner', nodes['owner'], {'pipeline_class': 'LTXConditionPipeline'})}
    for action in ('EncodeInputs', 'Denoise', 'DecodeLatents'):
        nodes[action] = {'module': 'modules.DiffusersVideo', 'action': action, 'params': {
            'pipeline': {'sourceId': 'owner', 'sourceKey': 'pipeline'}, 'mode': {'value': 'text_to_video'}}}
        records[action] = capture_generation_inputs(action, nodes[action], {'executed_task': 'text_to_video'})
    for task in (None, 'image_to_video', 'foreign_task'):
        changed = {**records, 'Denoise': capture_generation_inputs('Denoise', nodes['Denoise'], {'executed_task': task})}
        assert graph_task_receipts(nodes, changed)[0]['task'] is None
    assert graph_task_receipts({k: v for k, v in nodes.items() if k != 'DecodeLatents'}, records)[0]['task'] is None
    # Original whole-video receipts keep the legacy owner vocabulary.
    legacy = {k: capture_generation_inputs(k, v, {}) for k, v in nodes.items() if k != 'owner'}
    assert graph_task_receipts(nodes, {**legacy, 'owner': records['owner']}) == []
    foreign = {**records, 'owner': capture_generation_inputs('owner', nodes['owner'], {'pipeline_class': 'ForeignVideoPipeline'})}
    assert graph_task_receipts(nodes, foreign)[0]['task'] is None


def test_wan_stage_seals_latents_after_the_existing_adapter_restores_its_scheduler():
    pipeline, _ = owner('wan-22-ti2v-5b:direct')
    value = encoded(pipeline)
    node = Denoise('wan-stage-scheduler-unit')
    original, temporary = pipeline.scheduler, Scheduler()
    from modules.DiffusersVideo.main import get_video_pipeline_adapter
    adapter = get_video_pipeline_adapter('WanTI2VPipeline')
    def call(pipe, **kwargs):
        assert pipe.scheduler is temporary
        return SimpleNamespace(frames=torch.ones(1, 1, 2, 2, 2))
    with patch.object(Pipeline, '__call__', call, create=True), patch(
        'modules.DiffusersVideo.main._pipeline_adapter', return_value=adapter,
    ), patch('diffusers.UniPCMultistepScheduler.from_config', return_value=temporary), patch.object(
        node, '_execute_adapter', side_effect=lambda p, a, m, v: node._execute_wan_text_to_video(p, a, m, v),
    ):
        result = node(pipeline=pipeline, encoded_inputs=value, mode='text_to_video', guidance_scale=1.,
                      width=32, height=32, num_frames=9, scheduler_flow_shift=5., seed=7)
    assert pipeline.scheduler is original
    core.require_latent(pipeline, result['latent_binding'], result['latents_out'])
    assert node._video_sample is None


def test_wan_13b_stages_use_the_existing_adapter_with_owned_prompt_embeddings():
    import inspect
    from diffusers import WanPipeline
    from modules.DiffusersVideo.main import get_video_pipeline_adapter

    pipeline, _ = owner('wan-text-to-video:direct')
    value = encoded(pipeline, guidance_scale=4.5)
    sampler = Denoise('wan-13b-staged-call')
    calls = []

    @torch.no_grad()
    def sample(pipe, **kwargs):
        assert not torch.is_grad_enabled()
        assert set(kwargs) <= set(inspect.signature(WanPipeline.__call__).parameters)
        calls.append(kwargs)
        return SimpleNamespace(frames=torch.ones(1, 1, 3, 4, 4))

    with patch.object(Pipeline, '__call__', sample, create=True), patch(
        'modules.DiffusersVideo.main._pipeline_adapter', return_value=get_video_pipeline_adapter('WanPipeline'),
    ), patch.object(sampler, '_execute_adapter', side_effect=lambda p, a, m, v: sampler._execute_wan_text_to_video(p, a, m, v)):
        result = sampler(pipeline=pipeline, encoded_inputs=value, mode='text_to_video', guidance_scale=4.5,
                         width=32, height=32, num_frames=9, num_inference_steps=50, seed=8201)
    assert len(calls) == 1
    actual = calls[0]
    assert actual['prompt'] is None and actual['negative_prompt'] is None
    assert actual['prompt_embeds'] is value.embeddings['prompt_embeds']
    assert actual['negative_prompt_embeds'] is value.embeddings['negative_prompt_embeds']
    assert actual['num_inference_steps'] == 50 and actual['guidance_scale'] == 4.5
    assert actual['generator'].initial_seed() == 8201
    assert actual['output_type'] == 'latent'
    decoded = DecodeLatents('wan-13b-staged-decode')(
        pipeline=pipeline, latents=result['latents_out'], latent_binding=result['latent_binding'])
    assert decoded['frames_out'] == 9 and len(decoded['video_out']) == 9


@pytest.mark.parametrize(('mode', 'strength'), [('video_to_video', .7), ('video_color_edit', .2)])
def test_wan_source_video_stages_keep_actual_source_geometry_and_strength(mode, strength):
    import inspect
    from diffusers import WanVideoToVideoPipeline, FlowMatchEulerDiscreteScheduler
    from modules.DiffusersVideo.main import get_video_pipeline_adapter

    pipeline, _ = owner('wan-video-to-video:direct')
    frames = [Image.new('RGB', (32, 32), (index, 2, 3)) for index in range(9)]
    value = encoded(pipeline, mode=mode, video=frames, guidance_scale=5.)
    frames[0].putpixel((0, 0), (255, 255, 255))
    sampler, calls = Denoise('wan-video-stage-call'), []

    @torch.no_grad()
    def sample(pipe, **kwargs):
        assert set(kwargs) <= set(inspect.signature(WanVideoToVideoPipeline.__call__).parameters)
        calls.append(kwargs)
        return SimpleNamespace(frames=torch.ones(1, 1, 3, 4, 4))

    with patch.object(Pipeline, '__call__', sample, create=True), patch(
        'modules.DiffusersVideo.main._pipeline_adapter',
        return_value=get_video_pipeline_adapter('WanVideoToVideoPipeline'),
    ), patch.object(sampler, '_execute_adapter', side_effect=lambda p, a, m, v: sampler._execute_wan_video_to_video(p, a, m, v)):
        result = sampler(pipeline=pipeline, encoded_inputs=value, mode=mode, guidance_scale=5., width=32,
                         height=32, num_frames=173, num_inference_steps=50, seed=8412, strength=strength)
    assert len(calls) == 1
    actual = calls[0]
    assert actual['prompt'] is None and actual['prompt_embeds'] is value.embeddings['prompt_embeds']
    assert actual['video'][0].getpixel((0, 0)) == (0, 2, 3)
    assert len(actual['video']) == 9 and 'num_frames' not in actual
    assert actual['num_inference_steps'] == 50 and actual['strength'] == strength
    assert result['frames_out'] == 9, 'unused requested frame literal must not overwrite actual source-derived geometry'
    decoded = DecodeLatents('wan-video-stage-decode')(
        pipeline=pipeline, latents=result['latents_out'], latent_binding=result['latent_binding'])
    assert len(decoded['video_out']) == decoded['frames_out'] == 9

    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(50)
    effective = WanVideoToVideoPipeline.get_timesteps(SimpleNamespace(scheduler=scheduler),
        50, scheduler.timesteps, strength, 'cpu')
    assert effective[1] == int(50 * strength)


def test_a14b_stage_binding_tracks_both_experts_and_optional_image_components():
    pipeline, _ = owner('wan-22-image-to-video:direct')
    value = encoded(pipeline)
    original = pipeline.transformer_2
    pipeline.transformer_2 = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match='ownership'):
        core.require_encoding(pipeline, value)
    pipeline.transformer_2 = original
    core.require_encoding(pipeline, value)
    pipeline.image_encoder = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match='ownership'):
        core.require_encoding(pipeline, value)


def test_stage_binding_does_not_accept_missing_required_base_components():
    pipeline, values = owner()
    pipeline._modiff_video_stage_owner = None
    pipeline.text_encoder = None
    with pytest.raises(ValueError, match='base components'):
        core.bind_owner(pipeline, values)


def test_a14b_staged_call_keeps_both_guidance_controls_and_frozen_keyframes():
    import inspect
    from diffusers import WanImageToVideoPipeline
    from modules.DiffusersVideo.main import get_video_pipeline_adapter

    pipeline, _ = owner('wan-22-image-to-video:direct')
    opening, ending = Image.new('RGB', (32, 32), 'red'), Image.new('RGB', (32, 32), 'blue')
    value = encoded(pipeline, guidance_scale=3.5, reference_images=[opening], last_image=ending)
    opening.putpixel((0, 0), (0, 0, 0))
    ending.putpixel((0, 0), (0, 0, 0))
    sampler, calls = Denoise('a14b-stage-call'), []
    pipeline.video_processor.postprocess_video = lambda *args, **kwargs: [[Image.new('RGB', (32, 32)) for _ in range(81)]]

    @torch.no_grad()
    def sample(pipe, **kwargs):
        assert set(kwargs) <= set(inspect.signature(WanImageToVideoPipeline.__call__).parameters)
        calls.append(kwargs)
        return SimpleNamespace(frames=torch.ones(1, 1, 21, 4, 4))

    with patch.object(Pipeline, '__call__', sample, create=True), patch(
        'modules.DiffusersVideo.main._pipeline_adapter',
        return_value=get_video_pipeline_adapter('WanImageToVideoPipeline'),
    ), patch.object(sampler, '_execute_adapter', side_effect=lambda p, a, m, v: sampler._execute_wan_image_to_video(p, a, m, v)):
        result = sampler(pipeline=pipeline, encoded_inputs=value, mode='image_to_video', guidance_scale=3.5,
                         secondary_guidance_scale=2., width=32, height=32, num_frames=81,
                         num_inference_steps=40, seed=92021)
    assert len(calls) == 1
    actual = calls[0]
    assert actual['image'].getpixel((0, 0)) == (255, 0, 0)
    assert actual['last_image'].getpixel((0, 0)) == (0, 0, 255)
    assert actual['guidance_scale'] == 3.5 and actual['guidance_scale_2'] == 2.
    assert actual['num_frames'] == 81 and actual['num_inference_steps'] == 40
    assert actual['generator'].initial_seed() == 92021
    from modiff.execution_input_provenance import bind_generation_input_origins
    record = bind_generation_input_origins(sampler.node_id, {
        'module': 'modules.DiffusersVideo', 'action': 'Denoise', 'params': {
            'secondary_guidance_scale': {'sourceId': 'low-noise-control', 'sourceKey': 'value'},
        },
    }, sampler._execution_input_record, source_fields=sampler._execution_input_source_fields)
    assert record['fields']['secondary_guidance_scale']['value'] == 2.
    assert record['fields']['secondary_guidance_scale']['sourceNodeId'] == 'low-noise-control'
    decoded = DecodeLatents('a14b-stage-decode')(
        pipeline=pipeline, latents=result['latents_out'], latent_binding=result['latent_binding'])
    assert len(decoded['video_out']) == decoded['frames_out'] == 81


@pytest.mark.parametrize('mode', ['video_inpaint', 'video_outpaint', 'control_to_video'])
def test_vace_encoded_stage_keeps_native_windows_and_actual_decoded_continuity(mode):
    import inspect
    from diffusers import WanVACEPipeline
    from modules.DiffusersVideo.main import Generate
    from modiff.execution_input_provenance import build_resolved_execution_inputs, capture_generation_inputs

    pipeline, _ = owner('wan-vace:direct')
    source = [Image.new('RGB', (32, 32), (index, 1, 2)) for index in range(161)]
    media = {'video': source}
    if mode != 'control_to_video':
        media['mask'] = [Image.new('RGB', (32, 32), 'white') for _ in source]
    encoder, render = EncodeInputs('encode'), Generate('render')
    value = encoder(pipeline=pipeline, mode=mode, prompt='Amber glass in a moving scene',
                    negative_prompt='', guidance_scale=5., **media)['encoded_inputs']
    source[0].putpixel((0, 0), (255, 255, 255))
    calls, delivered = [], []

    @torch.no_grad()
    def sample(pipe, **kwargs):
        assert set(kwargs) <= set(inspect.signature(WanVACEPipeline.__call__).parameters)
        calls.append(kwargs)
        frames = [Image.new('RGB', (32, 32), (10 + len(calls), index, 3))
                  for index in range(kwargs['num_frames'])]
        delivered.append(frames)
        return SimpleNamespace(frames=[frames])

    with patch.object(Pipeline, '__call__', sample, create=True):
        result = render(pipeline=pipeline, mode=mode, encoded_inputs=value, width=32, height=32,
                        num_frames=161, num_inference_steps=30, guidance_scale=5.,
                        conditioning_scale=1., seed=8319, output_type='pil')
    assert pipeline.calls == 1, 'text encoding is reused across both native windows'
    assert [call['num_frames'] for call in calls] == [81, 81]
    assert all(call['prompt'] is None and call['prompt_embeds'] is value.embeddings['prompt_embeds'] for call in calls)
    assert all(call['generator'].initial_seed() == 8319 and call['num_inference_steps'] == 30 for call in calls)
    assert len(result['video_out']) == result['frames_out'] == 161
    assert result['video_out'] == delivered[0] + delivered[1][1:]
    if mode != 'control_to_video':
        assert calls[0]['video'][0].getpixel((0, 0)) == (127, 127, 127)
        assert calls[1]['video'][0] is delivered[0][-1]
        assert calls[1]['mask'][0].getpixel((0, 0)) == (0, 0, 0)
    else:
        assert calls[0]['video'][0].getpixel((0, 0)) == (0, 1, 2)
        assert all(call['mask'] is None for call in calls)
    graph = {'nodes': {
        'owner': {'module': 'modules.DiffusersVideo', 'action': 'LoadPipeline', 'params': {}},
        'encode': {'module': 'modules.DiffusersVideo', 'action': 'EncodeInputs', 'params': {
            'pipeline': {'sourceId': 'owner', 'sourceKey': 'pipeline'}}},
        'render': {'module': 'modules.DiffusersVideo', 'action': 'Generate', 'params': {
            'pipeline': {'sourceId': 'owner', 'sourceKey': 'pipeline'},
            'encoded_inputs': {'sourceId': 'encode', 'sourceKey': 'encoded_inputs'}}},
    }}
    records = {'owner': capture_generation_inputs('owner', graph['nodes']['owner'], {'pipeline_class': 'WanVACEPipeline'}),
               'encode': encoder._execution_input_record, 'render': render._execution_input_record}
    receipt = build_resolved_execution_inputs(graph, records, task_id='vace-staged', attempt_index=0, node_id='render')
    assert receipt['summary']['prompt'] == 'Amber glass in a moving scene'
    assert receipt['summary']['numFrames'] == 161
    assert receipt['summary']['seed'] == 8319
    assert receipt['graphTasks'] == [{'loaderId': 'owner', 'pipelineClass': 'WanVACEPipeline', 'task': mode}]


def test_integrated_encoded_video_rejects_foreign_or_mutated_conditioning_before_pipeline_call():
    from modules.DiffusersVideo.main import Generate
    pipeline, _ = owner('wan-vace:direct')
    frames = [Image.new('RGB', (32, 32)) for _ in range(9)]
    value = encoded(pipeline, mode='control_to_video', video=frames)
    foreign, _ = owner('wan-vace:direct')
    with pytest.raises(ValueError, match='ownership'):
        Generate('foreign-vace')(pipeline=foreign, encoded_inputs=value, mode='control_to_video', guidance_scale=1.)
    value.media['video_chunks'][0][0].putpixel((0, 0), (1, 2, 3))
    with pytest.raises(ValueError, match='modified'):
        Generate('mutated-vace')(pipeline=pipeline, encoded_inputs=value, mode='control_to_video', guidance_scale=1.)


@pytest.mark.parametrize('identity', [key for key, cls in core.PROFILE_CLASSES.items() if cls != 'WanVACEPipeline'])
def test_whole_video_operation_does_not_admit_an_undeclared_encoded_stage(identity):
    from modules.DiffusersVideo.main import Generate
    pipeline, _ = owner(identity)
    value = encoded(pipeline)
    node = Generate('undeclared-integrated-video')
    with patch.object(node, 'execute', side_effect=AssertionError('execution must remain unreachable')):
        with pytest.raises(ValueError, match='not declared'):
            node(pipeline=pipeline, encoded_inputs=value, mode=value.mode, guidance_scale=1.)


@pytest.mark.parametrize('task', DIFFUSERS_EXECUTION_PROFILES['wan-vace:direct'].modes)
def test_unbound_vace_starter_retains_its_whole_operation_beside_explicit_encoded_stages(task):
    from modules import MODULE_MAP
    from modiff.operation_catalog import build_operation_catalog
    from modiff.operation_starters import resolve_operation_starter
    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    with patch('modiff.NodeBase.NodeBase.__init__', side_effect=AssertionError('constructed')):
        result = resolve_operation_starter(MODULE_MAP, contracts, {'pipelineClass': 'WanVACEPipeline', 'task': task})
    assert {node['action'] for node in result['nodes']} == {'LoadPipeline', 'Generate'}
    assert len(result['nodes']) == 2
    whole = next(node for node in result['nodes'] if node['action'] == 'Generate')
    assert whole['operation']['operationId'] == 'diffusion.generate_video'
    assert whole['operation']['decomposition'] == 'pipeline'
    assert not any(edge['targetHandle'] == 'encoded_inputs' for edge in result['edges'])


def test_video_material_accepts_declared_241_masked_frames_without_widening_image_default():
    from modules.DiffusersImage.stage_execution import _freeze_material
    pipeline, _ = owner('wan-vace:direct')
    frames = [Image.new('RGB', (16, 16)) for _ in range(241)]
    value = encoded(pipeline, mode='video_inpaint', video=frames, mask=frames)
    assert sum(len(chunk) for chunk in value.media['mask_chunks']) == 241
    with pytest.raises(ValueError, match='finite structure'):
        _freeze_material({'one': [list(range(60)) for _ in range(5)]})


@pytest.mark.parametrize('identity,task', [
    ('ltx-video:direct', 'text_to_video'), ('ltx-video:direct', 'image_to_video'),
    ('ltx-video:direct', 'reference_to_video'), ('ltx-video:direct', 'video_to_video'),
    ('wan-22-ti2v-5b:direct', 'text_to_video'),
    ('wan-text-to-video:direct', 'text_to_video'),
    ('wan-video-to-video:direct', 'video_to_video'),
    ('wan-video-to-video:direct', 'video_color_edit'),
    ('wan-22-image-to-video:direct', 'image_to_video'),
    ('wan-vace:direct', 'video_inpaint'),
    ('wan-vace:direct', 'video_outpaint'),
    ('wan-vace:direct', 'control_to_video'),
])
def test_selected_starter_publishes_real_stage_edges_and_no_extra_whole_consumer(identity, task):
    from modules import MODULE_MAP
    from modiff.operation_catalog import build_operation_catalog
    from modiff.operation_starters import resolve_operation_starter
    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    profile = DIFFUSERS_EXECUTION_PROFILES[identity]
    with patch('modiff.NodeBase.NodeBase.__init__', side_effect=AssertionError('constructed')):
        result = resolve_operation_starter(MODULE_MAP, contracts, {
            'pipelineClass': profile.pipeline_class, 'task': task, 'executionProfileId': identity,
            'authoringRecipeId': 'generic-video-stages:v1',
        })
    actions = {node['action'] for node in result['nodes']}
    rendered = identity == 'wan-vace:direct'
    assert actions == ({'LoadPipeline', 'EncodeInputs', 'Generate'} if rendered else
                       {'LoadPipeline', 'EncodeInputs', 'Denoise', 'DecodeLatents'})
    assert {'source': 'diffusion.encode_video', 'sourceHandle': 'encoded_inputs',
            'target': 'diffusion.render_video' if rendered else 'diffusion.denoise',
            'targetHandle': 'encoded_inputs'} in result['edges']
    if not rendered:
        assert {'source': 'diffusion.denoise', 'sourceHandle': 'latent_binding',
                'target': 'diffusion.decode_latents', 'targetHandle': 'latent_binding'} in result['edges']
    else:
        generate = next(node for node in result['nodes'] if node['action'] == 'Generate')
        assert all(generate['params'][key]['hidden'] for key in ('prompt', 'negative_prompt', 'video', 'mask', 'reference_images'))
    assert not any(row['field'] in ('encoded_inputs', 'latent_binding', 'latents') for row in result['requiredInputs'])
    loader = next(node for node in result['nodes'] if node['action'] == 'LoadPipeline')
    assert loader['values']['revision'] == require_catalog_revision(profile.default_repo)
    assert result['authoring']['id'] == 'generic-video-stages:v1'
    assert {'name': 'guidance_scale', 'members': [
        {'operationId': 'diffusion.encode_video', 'field': 'guidance_scale'},
        {'operationId': 'diffusion.render_video' if rendered else 'diffusion.denoise', 'field': 'guidance_scale'},
    ]} in result['sharedInputs']
    assert not any(field == 'mode' for _role, field, _source in result['authoring']['bindings'])
    assert ('diffusersRecipe', 'attention_backend', 'attentionBackend') in result['authoring']['bindings']
    # The whole-operation source is historical authority, not a fresh-stage
    # default. An imported graph's literal policy is never rewritten here.
    if identity == 'ltx-video:direct':
        from modiff.studio_execution_specs import studio_execution_spec_for_pair
        retained = studio_execution_spec_for_pair('LTXVideoPipeline', task)
        assert ('diffusersRecipe', 'attention_backend', 'nativeMath') in retained['bindings']
    if identity == 'ltx-video:direct':
        for node in result['nodes']:
            for field, expected in (('guidance_scale', 1.), ('num_inference_steps', 8)):
                if field in node['params']:
                    assert node['params'][field]['min'] == node['params'][field]['max'] == expected
                    assert node['params'][field]['value'] == node['params'][field]['default'] == expected
                    assert node['values'][field] == expected
    for node in result['nodes']:
        if node['action'] == 'LoadPipeline':
            continue
        port = next(port for port in node['operation']['ports'] if port['name'] == 'pipeline')
        assert port['semantics']['kind'] == 'pipeline'
        assert port['semantics']['owner'] == 'same_loader'
