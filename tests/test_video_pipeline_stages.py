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
        self._execution_device = 'cpu'
        profile = DIFFUSERS_EXECUTION_PROFILES[identity]
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
    return core.encode(pipeline, {'mode': 'text_to_video', 'prompt': 'A moving train',
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


@pytest.mark.parametrize('identity', core.PROFILE_CLASSES)
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


@pytest.mark.parametrize('identity', core.PROFILE_CLASSES)
def test_actual_split_calls_capture_video_task_prompt_and_final_format_without_form_fallback(identity):
    from modiff.execution_input_provenance import capture_generation_inputs, build_resolved_execution_inputs, apply_resolved_execution_inputs
    from modules.DiffusersVideo.main import get_video_pipeline_adapter
    pipeline, _ = owner(identity)
    encoder, sampler, decoder = EncodeInputs('encode'), Denoise('sample'), DecodeLatents('decode')
    encoding = encoder(pipeline=pipeline, mode='text_to_video', prompt='A moving train',
                       negative_prompt='', guidance_scale=1., max_sequence_length=128)['encoded_inputs']
    def sample(pipe, adapter, mode, args):
        return sampler._invoke_pipeline(pipe, {'width': 32, 'height': 32, 'num_frames': 9,
            'generator': torch.Generator().manual_seed(7)})
    with patch.object(Pipeline, '__call__', lambda self, **kw: SimpleNamespace(frames=torch.ones(1, 1, 2, 2, 2)), create=True), patch(
        'modules.DiffusersVideo.main._pipeline_adapter', return_value=get_video_pipeline_adapter(pipeline._modiff_video_pipeline_class),
    ), patch.object(sampler, '_execute_adapter', side_effect=sample):
        latent = sampler(pipeline=pipeline, encoded_inputs=encoding, mode='text_to_video', guidance_scale=1., seed=7)
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
    assert receipt['graphTasks'] == [{'loaderId': 'owner', 'pipelineClass': pipeline._modiff_video_pipeline_class, 'task': 'text_to_video'}]
    assert receipt['summary']['prompt'] == 'A moving train'
    # Blank text is normalized by the encoder; retain that exact observation.
    assert receipt['summary']['negativePrompt'] is None
    assert receipt['summary']['outputType'] == 'pil'
    assert receipt['summary']['executedTask'] == 'text_to_video'
    assert receipt['ambiguousFields'] == []
    result = apply_resolved_execution_inputs({'taskId': 'run', 'nodeId': 'decode', 'mode': 'text_to_image',
        'formSnapshot': {'mode': 'text_to_image', 'prompt': 'unused'}}, receipt)
    assert result['mode'] == 'text_to_video' and result['prompt'] == 'A moving train'
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


@pytest.mark.parametrize('identity,task', [
    ('ltx-video:direct', 'text_to_video'), ('ltx-video:direct', 'image_to_video'),
    ('ltx-video:direct', 'reference_to_video'), ('ltx-video:direct', 'video_to_video'),
    ('wan-22-ti2v-5b:direct', 'text_to_video'),
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
    assert actions == {'LoadPipeline', 'EncodeInputs', 'Denoise', 'DecodeLatents'}
    assert {'source': 'diffusion.encode_video', 'sourceHandle': 'encoded_inputs',
            'target': 'diffusion.denoise', 'targetHandle': 'encoded_inputs'} in result['edges']
    assert {'source': 'diffusion.denoise', 'sourceHandle': 'latent_binding',
            'target': 'diffusion.decode_latents', 'targetHandle': 'latent_binding'} in result['edges']
    assert not any(row['field'] in ('encoded_inputs', 'latent_binding', 'latents') for row in result['requiredInputs'])
    loader = next(node for node in result['nodes'] if node['action'] == 'LoadPipeline')
    assert loader['values']['revision'] == require_catalog_revision(profile.default_repo)
    assert result['authoring']['id'] == 'generic-video-stages:v1'
    assert not any(field == 'mode' for _role, field, _source in result['authoring']['bindings'])
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
