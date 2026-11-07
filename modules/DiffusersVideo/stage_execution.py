"""Bounded, loader-owned material for the declared ordinary video stages.

Sampling remains in Generate's existing official-library adapters. These
process-local bindings are data, never another graph or executable callback.
"""
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
import weakref
from modiff.video_stage_contract import PROFILE_CLASSES, video_stage_spec
_ENCODINGS = weakref.WeakSet()
_LATENTS = weakref.WeakSet()


def starter_spec(pipeline_class, task, values):
    """Derive the same selected owner's real stage wiring, without loading."""
    from modiff.studio_execution_specs import studio_execution_spec_for_pair
    if pipeline_class not in PROFILE_CLASSES.values():
        return None
    try:
        profile = selected_profile(values)
    except ValueError:
        return None
    if task not in profile.modes:
        raise ValueError('The video stage task does not belong to the selected profile.')
    original = studio_execution_spec_for_pair(profile.model_type, task)
    if original is None or original['executionProfileId'] != profile.id:
        raise ValueError('The selected video stages lack an exact source specification.')
    return video_stage_spec(original)


def selected_profile(values):
    from modiff.diffusers_profiles import resolve_execution_profiles_for_loader
    from modiff.model_artifact_catalog import require_catalog_revision
    identity = values.get('execution_profile_id')
    if PROFILE_CLASSES.get(identity) != values.get('pipeline_class'):
        raise ValueError('Separate video stages require a declared exact execution profile.')
    profiles, reason = resolve_execution_profiles_for_loader('modules.DiffusersVideo', 'LoadPipeline', values)
    if reason or len(profiles) != 1 or profiles[0].id != identity:
        raise ValueError('One authoritative video loader profile is required.')
    profile = profiles[0]
    selection = values.get('model_id')
    if (not profile.public or type(selection) is not dict or set(selection) != {'source', 'value'}
            or selection != {'source': 'hub', 'value': profile.default_repo}
            or values.get('revision') != require_catalog_revision(profile.default_repo)):
        raise ValueError('Video stages require their exact reviewed Hub artifact and revision.')
    return profile


def _configuration(pipeline):
    return json.dumps({key: vars(pipeline).get(key) for key in (
        '_modiff_runtime_config', '_modiff_adapter_scales', '_modiff_video_pipeline_class',
        '_modiff_video_repo', '_modiff_video_revision',
    )}, sort_keys=True, allow_nan=False)


@dataclass(frozen=True, eq=False)
class Owner:
    pipeline: object
    profile_id: str
    pipeline_class: str
    tasks: tuple
    components: tuple
    repository: str
    revision: str


def bind_owner(pipeline, values):
    try:
        profile = selected_profile(values)
    except ValueError:
        pipeline._modiff_video_stage_owner = None
        return
    if (vars(pipeline).get('_modiff_video_repo') != profile.default_repo
            or vars(pipeline).get('_modiff_video_revision') != values['revision']):
        raise ValueError('The actual video loader artifact disagrees with its stage selection.')
    previous = vars(pipeline).get('_modiff_video_stage_owner')
    if isinstance(previous, Owner) and previous.profile_id == profile.id:
        require_owner(pipeline)
        return
    components = tuple((name, weakref.ref(getattr(pipeline, name)))
                       for name in ('text_encoder', 'tokenizer', 'transformer', 'vae', 'scheduler'))
    pipeline._modiff_video_stage_owner = Owner(weakref.ref(pipeline), profile.id,
                                              profile.pipeline_class, tuple(profile.modes), components,
                                              profile.default_repo, values['revision'])


def require_owner(pipeline, expected=None):
    record = vars(pipeline).get('_modiff_video_stage_owner')
    if (type(record) is not Owner or record.pipeline() is not pipeline
            or expected is not None and expected is not record
            or vars(pipeline).get('_modiff_video_pipeline_class') != record.pipeline_class
            or vars(pipeline).get('_modiff_video_repo') != record.repository
            or vars(pipeline).get('_modiff_video_revision') != record.revision
            or any(reference() is None or getattr(pipeline, name, None) is not reference()
                   for name, reference in record.components)):
        raise ValueError('Video stage ownership is incomplete, replaced or stale. Rerun Model Setup.')
    return record


@dataclass(frozen=True, eq=False)
class Encoding:
    owner: Owner
    configuration: str
    mode: str
    guidance_enabled: bool
    max_sequence_length: int
    embeddings: object
    media: object
    stamps: tuple
    consumed: object


@dataclass(frozen=True, eq=False)
class Latent:
    encoding: Encoding
    tensor: object
    stamp: tuple
    geometry: tuple
    generator_state: object
    generator_device: str


def _stamp(tensor):
    return (tuple(tensor.shape), tensor.dtype, tensor.device, tensor._version)


def _material(value):
    # Reuse existing bounded material normalization and mutation checks.
    from modules.DiffusersImage.stage_execution import _freeze_material
    return _freeze_material(value)


def _video_source_evidence(material, stamps):
    """Identify the frozen pixel sequence from existing validated media stamps.

    This is not a source-file hash. No paths, tensors or arbitrary runtime
    objects are read for evidence. Non-PIL media retains its frame count and
    explicitly unavailable pixel digest, rather than an invented identity.
    """
    chunks = material['video_chunks']
    if chunks is None:
        return {}
    frames = [frame for chunk in chunks for frame in chunk]
    by_identity = {id(reference()): stamp for kind, reference, stamp in stamps if kind == 'pil'}
    descriptors = []
    for frame in frames:
        stamp = by_identity.get(id(frame))
        if stamp is None:
            return {'video_source_sha256': None, 'video_source_frames': len(frames)}
        mode, size, digest = stamp
        descriptors.append([mode, list(size), digest.hex()])
    payload = json.dumps(descriptors, separators=(',', ':'), ensure_ascii=True).encode('ascii')
    return {'video_source_sha256': hashlib.sha256(b'modiff-video-source-pixels-v1\0' + payload).hexdigest(),
            'video_source_frames': len(frames)}


def require_encoding(pipeline, value, mode=None, guidance=None):
    from modules.DiffusersImage.stage_execution import _check_stamps
    if type(value) is not Encoding or value not in _ENCODINGS:
        raise ValueError('A genuine current video encoding is required.')
    require_owner(pipeline, value.owner)
    if value.configuration != _configuration(pipeline):
        raise ValueError('Video model recipe or adapters changed. Rerun Encode Inputs.')
    if mode is not None and value.mode != mode:
        raise ValueError('Video encoding belongs to another task.')
    if guidance is not None and value.guidance_enabled != (float(guidance) > 1):
        raise ValueError('Video guidance enablement changed. Rerun Encode Inputs.')
    _check_stamps(value.stamps)
    return value


def encode(pipeline, values):
    import torch
    with torch.inference_mode():
        return _encode(pipeline, values)


def _encode(pipeline, values):
    from .wan_vace import ensure_single_prompt, none_if_blank, ensure_reference_images, ensure_video_list
    from .main import _validate_prompt_token_limit, _pipeline_adapter
    owner = require_owner(pipeline)
    mode = values['mode']
    if mode not in owner.tasks:
        raise ValueError('Video task is outside the selected stage owner.')
    prompt = ensure_single_prompt(none_if_blank(values.get('prompt')), 'prompt')
    negative = ensure_single_prompt(none_if_blank(values.get('negative_prompt')), 'negative prompt')
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Encode Video Inputs requires a nonempty prompt.')
    adapter = _pipeline_adapter(pipeline)
    _validate_prompt_token_limit(pipeline, prompt, 'prompt', adapter.max_prompt_tokens)
    _validate_prompt_token_limit(pipeline, negative, 'negative prompt', adapter.max_prompt_tokens)
    guidance = float(values.get('guidance_scale', 1))
    length = int(values.get('max_sequence_length', adapter.max_prompt_tokens or 256))
    if owner.pipeline_class == 'LTXConditionPipeline':
        # Retain the existing LTX adapter's normalization of the shared 512
        # default. Capture the actual encoder length rather than the form value.
        length = min(length, adapter.max_prompt_tokens or 256)
    if not 1 <= length <= (adapter.max_prompt_tokens or 512):
        raise ValueError('Video sequence length is outside the declared encoder bound.')
    video = ensure_video_list(values.get('video'), 'video')
    references = ensure_reference_images(values.get('reference_images'))
    if owner.pipeline_class == 'WanTI2VImageToVideoPipeline' and (not references or len(references) != 1):
        raise ValueError('The dense Wan 5B image route needs exactly one opening image.')
    if (mode == 'text_to_video' and (video is not None or references)
            or mode in ('image_to_video', 'reference_to_video') and not references
            or mode == 'video_to_video' and video is None):
        raise ValueError('Video source media does not match the selected task.')
    if video is not None and len(video) > 192:
        raise ValueError('Separate video input encoding currently supports at most 192 source frames.')
    # Keep the shared finite-list and total-byte bounds, while allowing the
    # existing 81/121/161-frame video recipes rather than image-batch lengths.
    chunks = [video[i:i + 64] for i in range(0, len(video), 64)] if video is not None else None
    material, stamps, _logical, _copied = _material({'video_chunks': chunks, 'reference_images': references})
    device = getattr(pipeline, '_execution_device', None) or 'cpu'
    output = pipeline.encode_prompt(prompt=prompt, negative_prompt=negative,
        do_classifier_free_guidance=guidance > 1, num_videos_per_prompt=1,
        device=device, max_sequence_length=length)
    names = ('prompt_embeds', 'prompt_attention_mask', 'negative_prompt_embeds', 'negative_prompt_attention_mask') if owner.pipeline_class == 'LTXConditionPipeline' else ('prompt_embeds', 'negative_prompt_embeds')
    if not isinstance(output, tuple) or len(output) != len(names):
        raise ValueError('The official video encoder returned an incompatible contract.')
    embeddings, tensor_stamps, _logical, _copied = _material(dict(zip(names, output, strict=True)))
    result = Encoding(owner, _configuration(pipeline), mode, guidance > 1, length,
                      embeddings, material, stamps + tensor_stamps,
                      MappingProxyType({'executed_task': mode, 'prompt': prompt, 'negative_prompt': negative,
                                        'guidance_scale': guidance, 'max_sequence_length': length,
                                        **_video_source_evidence(material, stamps)}))
    _ENCODINGS.add(result)
    return result


def issue_latent(pipeline, encoding, tensor, geometry, generator):
    require_encoding(pipeline, encoding)
    import torch
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 5:
        raise ValueError('The official video sampler must return a five-dimensional latent tensor.')
    if tensor.device.type == 'meta' or tensor.numel() * tensor.element_size() > 1024**3:
        raise ValueError('Video latent material exceeds its bounded allocation.')
    with torch.inference_mode(False), torch.no_grad():
        tensor = tensor.detach().clone()
    result = Latent(encoding, tensor, _stamp(tensor), geometry,
                    generator.get_state().clone(), str(generator.device))
    _LATENTS.add(result)
    return result


def require_latent(pipeline, binding, tensor):
    if type(binding) is not Latent or binding not in _LATENTS:
        raise ValueError('A genuine current video latent binding is required.')
    require_encoding(pipeline, binding.encoding)
    if tensor is not binding.tensor or _stamp(binding.tensor) != binding.stamp:
        raise ValueError('Video latents were modified after denoising.')
    return binding


def decode(pipeline, binding):
    import torch
    with torch.inference_mode():
        return _decode(pipeline, binding)


def _decode(pipeline, binding):
    import torch
    from diffusers.utils.torch_utils import get_module_execution_device
    require_latent(pipeline, binding, binding.tensor)
    device = get_module_execution_device(pipeline.vae)
    latents = binding.tensor.to(device=device, dtype=pipeline.vae.dtype)
    if binding.encoding.owner.pipeline_class == 'LTXConditionPipeline':
        latents = pipeline._denormalize_latents(latents, pipeline.vae.latents_mean,
            pipeline.vae.latents_std, pipeline.vae.config.scaling_factor)
        timestep = None
        if pipeline.vae.config.timestep_conditioning:
            # The reviewed existing adapter leaves the official defaults at
            # decode_timestep=0 / decode_noise_scale=None. Continue the exact
            # post-denoise RNG state rather than resetting the seed here.
            from diffusers.utils.torch_utils import randn_tensor
            generator = torch.Generator(device=binding.generator_device)
            generator.set_state(binding.generator_state)
            noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
            timestep = torch.zeros((latents.shape[0],), device=device, dtype=latents.dtype)
            latents = latents + 0 * noise
        output = pipeline.vae.decode(latents, timestep, return_dict=False)[0]
    else:
        mean = torch.tensor(pipeline.vae.config.latents_mean).view(1, pipeline.vae.config.z_dim, 1, 1, 1).to(latents)
        std = 1.0 / torch.tensor(pipeline.vae.config.latents_std).view(1, pipeline.vae.config.z_dim, 1, 1, 1).to(latents)
        output = pipeline.vae.decode(latents / std + mean, return_dict=False)[0]
    frames = pipeline.video_processor.postprocess_video(output, output_type='pil')
    pipeline.maybe_free_model_hooks()
    if isinstance(frames, list) and len(frames) == 1 and isinstance(frames[0], list):
        frames = frames[0]
    width, height, count = binding.geometry
    if len(frames) != count or any(frame.size != (width, height) for frame in frames):
        raise ValueError('Decoded video geometry disagrees with the sampler binding.')
    return {'video_out': frames, 'width_out': width, 'height_out': height, 'frames_out': count}
