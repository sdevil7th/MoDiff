"""LTX-2.5 Full calls through the ordinary resource-managed video facade.

The official library owns sampling, conditioning, prompt expansion and decode.
The Full transformer and scheduler are explicit; the distilled recipe is intact.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import weakref


MODEL = 'LTX25FullPipeline'
REPOSITORY = 'Lightricks/LTX-2.5-Diffusers'
REVISION = '4791a1100e0cf97f7ed5c55c12dd71dce13f9dd3'
MODES = ('text_to_video_with_audio', 'image_to_video_with_audio',
         'reference_to_video_with_audio', 'video_to_video_with_audio')
GUIDANCE_DEFAULTS = {
    'stg_scale': 1., 'modality_scale': 3., 'guidance_rescale': .7,
    'audio_guidance_scale': 7., 'audio_stg_scale': 1.,
    'audio_modality_scale': 3., 'audio_guidance_rescale': .7,
}
COMPONENTS = ('transformer', 'scheduler', 'vae', 'audio_vae', 'text_encoder',
              'tokenizer', 'connectors', 'vocoder', 'processor', 'prompt_enhancer')


def _scheduler_fingerprint(scheduler):
    raw = json.dumps(dict(scheduler.config), sort_keys=True, allow_nan=False).encode()
    if len(raw) > 65536:
        raise ValueError('The connected scheduler configuration exceeds its inspection bound.')
    if scheduler.config.get('use_dynamic_shifting') is not True or scheduler.config.get('shift_terminal') != .1:
        raise ValueError('LTX-2.5 Full requires its dynamic-shifting scheduler and terminal shift 0.1.')
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class FullOwner:
    pipeline: object
    components: tuple
    scheduler_sha256: str


def bind(pipeline):
    references = []
    for name in COMPONENTS:
        component = getattr(pipeline, name, None)
        if component is None:
            raise ValueError(f'LTX-2.5 Full is missing its required {name} component.')
        references.append((name, weakref.ref(component)))
    pipeline._modiff_ltx25_full_owner = FullOwner(
        weakref.ref(pipeline), tuple(references), _scheduler_fingerprint(pipeline.scheduler),
    )


def require_owner(pipeline):
    record = getattr(pipeline, '_modiff_ltx25_full_owner', None)
    if (not isinstance(record, FullOwner) or record.pipeline() is not pipeline
            or any(reference() is not getattr(pipeline, name, None) for name, reference in record.components)
            or record.scheduler_sha256 != _scheduler_fingerprint(pipeline.scheduler)):
        raise ValueError('Reload the selected LTX-2.5 Full recipe; its components or scheduler changed.')


def load(node, adapter, values):
    from utils.torch_utils import str_to_dtype
    from diffusers import FlowMatchEulerDiscreteScheduler, LTX2ConditionPipeline, LTX2VideoTransformer3DModel
    from modules.DiffusersRuntime.main import apply_execution_recipe_to_pipeline, loader_runtime_options
    from .wan_vace import apply_pipeline_offload, exact_cached_snapshot_path

    if values.get('model_id') != {'source': 'hub', 'value': REPOSITORY} or values.get('revision') != REVISION:
        raise ValueError('LTX-2.5 Full requires its exact reviewed Hub repository and revision.')
    recipe, device, offload, load_options = loader_runtime_options(
        values, default_device='cuda:0', default_offload_mode='model_cpu', direct_device_load=True,
    )
    if offload not in {'none', 'model_cpu'}:
        raise ValueError('The reviewed LTX-2.5 Full route supports full residency or model CPU offload.')
    snapshot = exact_cached_snapshot_path(REPOSITORY, REVISION)
    options = {'torch_dtype': str_to_dtype(values.get('dtype', 'bfloat16')), 'use_safetensors': True, 'local_files_only': True,
               'low_cpu_mem_usage': bool(values.get('low_cpu_mem_usage', True)), **load_options}
    node.progress(-1, phase='loading', message='Loading full video and audio components')
    transformer = LTX2VideoTransformer3DModel.from_pretrained(snapshot, subfolder='transformer_full', **options)
    pipeline = LTX2ConditionPipeline.from_pretrained(snapshot, transformer=transformer, **options)
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipeline.scheduler.config, use_dynamic_shifting=True, shift_terminal=.1,
    )
    apply_execution_recipe_to_pipeline(pipeline, recipe)
    apply_pipeline_offload(pipeline, mode=offload, device=device, node_id=node.node_id, scope=adapter.id)
    bind(pipeline)
    node.mm_add(pipeline, priority=2)
    return pipeline


def _number(values, field, default, *, minimum=0, maximum=None):
    value = values.get(field)
    value = default if value is None else value
    if (type(value) not in (int, float) or not math.isfinite(value) or value < minimum
            or maximum is not None and value > maximum):
        raise ValueError(f'{field} must be a finite value in its declared range.')
    return value


def _integer(values, field, default, *, minimum=1):
    value = values.get(field)
    value = default if value is None else value
    if type(value) is not int or not minimum <= value <= 2**53 - 1:
        raise ValueError(f'{field} must be an exact whole number of at least {minimum}.')
    return value


def _json_integers(raw, *, label, minimum, maximum, default):
    if raw is None or raw == '':
        return default
    if type(raw) is not str or len(raw.encode()) > 8192:
        raise ValueError(f'{label} must be a bounded JSON integer list.')
    try:
        result = json.loads(raw)
    except (ValueError, RecursionError) as error:
        raise ValueError(f'{label} must be a bounded JSON integer list.') from error
    if (not isinstance(result, list) or len(result) > 256
            or any(type(value) is not int or not minimum <= value <= maximum for value in result)):
        raise ValueError(f'{label} must contain integers in its declared range.')
    return result


def execute(node, pipeline, adapter, mode, values):
    import torch
    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
    from .wan_vace import (callback_tensor_inputs, ensure_reference_images, ensure_single_prompt,
                          ensure_video_list, none_if_blank, parse_json_object)

    require_owner(pipeline)
    if mode not in MODES:
        raise ValueError('Select an admitted Full video/audio task.')
    if _integer(values, 'num_videos_per_prompt', 1) != 1:
        raise ValueError('This Full facade generates one video per prompt.')
    images = ensure_reference_images(values.get('reference_images'))
    video = ensure_video_list(values.get('video'), 'video')
    if mode == 'text_to_video_with_audio' and (images or video):
        raise ValueError('Text generation accepts no image or video conditions.')
    if mode == 'image_to_video_with_audio' and (not images or len(images) != 1 or video):
        raise ValueError('Image generation requires exactly one opening image and no source video.')
    if mode == 'reference_to_video_with_audio' and (not images or video):
        raise ValueError('Keyframe generation requires ordered reference images and no source video.')
    if mode == 'video_to_video_with_audio' and (not video or images):
        raise ValueError('Video conditioning requires one source video and no image references.')
    for field in ('encoded_inputs', 'mask', 'control_video', 'reference_video', 'pose_video',
                  'face_video', 'background_video', 'last_image', 'prompt_embeds', 'negative_prompt_embeds'):
        if none_if_blank(values.get(field)) is not None:
            raise ValueError(f'This Full video facade does not admit {field}.')
    width, height = _integer(values, 'width', 768), _integer(values, 'height', 512)
    frames = _integer(values, 'num_frames', 121)
    fps = _number(values, 'frame_rate', 24., minimum=.001)
    spatial = pipeline.vae_spatial_compression_ratio
    temporal = pipeline.vae_temporal_compression_ratio
    if width % spatial or height % spatial or (frames - 1) % temporal:
        raise ValueError('Video size and frame count must follow the selected native VAE grid.')
    blocks = _json_integers(values.get('temporal_guidance_blocks_json'), label='Temporal guidance layers',
                           minimum=0, maximum=pipeline.transformer.config.num_layers - 1, default=[28])
    for field in ('expand_prompt', 'use_cross_timestep'):
        if values.get(field) is not None and type(values[field]) is not bool:
            raise ValueError(f'{field} must be a boolean.')
    condition_strength = _number(values, 'strength', 1., maximum=1.)
    indices = _json_integers(values.get('condition_indices_json'), label='Keyframe positions',
                            minimum=-1, maximum=frames - 1, default=None)
    if indices is not None and mode != 'reference_to_video_with_audio':
        raise ValueError('Keyframe positions apply only to ordered-reference generation.')
    conditions = None
    if images:
        if mode == 'reference_to_video_with_audio':
            if indices is None or len(indices) != len(images):
                raise ValueError('Provide one explicit frame position for every ordered reference image.')
        else:
            indices = [0]
        conditions = [LTX2VideoCondition(frames=[image], index=index, strength=condition_strength)
                      for image, index in zip(images, indices, strict=True)]
    elif video:
        conditions = [LTX2VideoCondition(frames=video, index=0, strength=condition_strength)]
    arguments = {
        'prompt': ensure_single_prompt(none_if_blank(values.get('prompt')), 'prompt'),
        'negative_prompt': ensure_single_prompt(none_if_blank(values.get('negative_prompt')), 'negative prompt'),
        'height': height, 'width': width, 'num_frames': frames, 'frame_rate': fps,
        'num_inference_steps': _integer(values, 'num_inference_steps', 30),
        'guidance_scale': _number(values, 'guidance_scale', 3.),
        **{field: _number(values, field, default, maximum=1. if field.endswith('rescale') else None)
           for field, default in GUIDANCE_DEFAULTS.items()},
        'spatio_temporal_guidance_blocks': blocks, 'use_cross_timestep': values.get('use_cross_timestep', True),
        'enable_prompt_enhancement': values.get('expand_prompt', False),
        'generator': torch.Generator(device=pipeline._execution_device).manual_seed(_integer(values, 'seed', 0, minimum=0)),
        'conditions': conditions, 'latents': values.get('latents'),
        'output_type': values.get('output_type', 'pil'), 'return_dict': True,
        'attention_kwargs': parse_json_object(values.get('attention_kwargs_json'), 'attention kwargs'),
        'callback_on_step_end': node.pipe_callback,
        'callback_on_step_end_tensor_inputs': callback_tensor_inputs(values.get('callback_on_step_end_tensor_inputs')),
        'max_sequence_length': _integer(values, 'max_sequence_length', 1024),
    }
    if arguments['output_type'] not in {'pil', 'np', 'pt'}:
        raise ValueError('The Full facade exports decoded PIL, NumPy or Torch video frames.')
    from .main import _validate_prompt_token_limit
    for field in ('prompt', 'negative_prompt'):
        _validate_prompt_token_limit(pipeline, arguments[field], field, adapter.max_prompt_tokens, family='LTX-2.5 Full')
    node._active_pipeline = pipeline
    try:
        result = node._invoke_pipeline(pipeline, arguments)
    finally:
        node._active_pipeline = None
    require_owner(pipeline)
    videos = result.frames
    if arguments['output_type'] == 'np':
        videos = pipeline.video_processor.numpy_to_pil(videos[0])
    elif arguments['output_type'] == 'pt':
        videos = pipeline.video_processor.postprocess(videos[0], output_type='pil', do_denormalize=[False] * len(videos[0]))
    elif isinstance(videos, list) and len(videos) == 1 and isinstance(videos[0], list):
        videos = videos[0]
    controls = {field: arguments[field] for field in ('prompt', 'negative_prompt', 'height', 'width',
                'num_frames', 'frame_rate', 'num_inference_steps', 'guidance_scale', *GUIDANCE_DEFAULTS,
                'use_cross_timestep', 'output_type', 'max_sequence_length')}
    controls.update(seed=arguments['generator'].initial_seed(),
                    expand_prompt=arguments['enable_prompt_enhancement'],
                    temporal_guidance_blocks_json=blocks)
    if conditions:
        controls.update(strength=condition_strength, condition_indices_json=[condition.index for condition in conditions])
    return {'video_out': videos, 'width_out': width, 'height_out': height,
            'frames_out': len(videos), '_audio': result.audio, '_native_generation_inputs': controls}
