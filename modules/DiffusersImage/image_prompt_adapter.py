"""Exact local artifacts attached to one newly constructed pipeline owner.

No model downloads, shared-transformer cloning, remote Python or parallel graph
executor. The pipeline loader attaches adapters before installing offload hooks.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from utils.huggingface import cached_file_path, resolve_managed_hf_cache_file
from huggingface_hub.utils import validate_repo_id


IP_ADAPTER_PIPELINES = frozenset({
    'FluxPipeline', 'FluxImg2ImgPipeline', 'FluxInpaintPipeline',
    'FluxKontextPipeline', 'FluxKontextInpaintPipeline', 'FluxControlNetPipeline',
})
IP_ADAPTER_INPUTS = ('ip_adapter_image', 'ip_adapter_image_embeds',
                     'negative_ip_adapter_image', 'negative_ip_adapter_image_embeds')


@dataclass(frozen=True)
class ImagePromptAdapterConfig:
    repository: str
    revision: str
    weight_name: str
    sha256: str
    encoder_repository: str | None
    encoder_revision: str | None
    encoder_sha256: str | None
    scale: float | tuple[float, ...]


def _exact(value, field, length):
    if not isinstance(value, str) or not re.fullmatch(f'[0-9a-f]{{{length}}}', value):
        raise ValueError(f'{field} requires exactly {length} lowercase hexadecimal characters.')
    return value


def _repository(value, field):
    if not isinstance(value, dict) or value.get('source') != 'hub':
        raise ValueError(f'{field} requires an immutable Hub model selection.')
    repository = value.get('value')
    if not isinstance(repository, str) or repository != repository.strip() or '/' not in repository:
        raise ValueError(f'{field} requires a canonical owner/repository ID.')
    try:
        validate_repo_id(repository)
    except ValueError as error:
        raise ValueError(f'{field}: invalid Hub repository ID.') from error
    return repository


def _scale(value):
    if type(value) not in (int, float, str):
        raise ValueError('scale must be a finite number between -100 and 100.')
    try:
        numeric = float(value)
        if not math.isfinite(numeric) or not -100 <= numeric <= 100:
            raise ValueError
    except (ValueError, OverflowError):
        raise ValueError('scale must be a finite number between -100 and 100.') from None
    return numeric


def image_prompt_adapter_config(values):
    previous = values.get('previous') or ()
    if not isinstance(previous, tuple) or not all(isinstance(item, ImagePromptAdapterConfig) for item in previous):
        raise TypeError('previous must come from an Image Prompt Adapter configuration node.')
    if len(previous) >= 4:
        raise ValueError('Image Prompt Adapter supports at most 4 explicitly chained adapters.')
    weight = values.get('weight_name')
    if (not isinstance(weight, str) or not weight.endswith('.safetensors') or
            len(weight) > 512 or '\\' in weight or ':' in weight or
            any(part in ('', '.', '..') for part in weight.split('/'))):
        raise ValueError('weight_name must select one contained .safetensors repository file.')
    encoder = values.get('image_encoder_model')
    if not encoder or (isinstance(encoder, dict) and not encoder.get('value')):
        if values.get('image_encoder_revision') or values.get('image_encoder_sha256'):
            raise ValueError('image_encoder_revision/hash requires an image encoder selection.')
        encoder_repo = encoder_revision = encoder_hash = None
    else:
        encoder_repo = _repository(encoder, 'image_encoder_model')
        encoder_revision = _exact(values.get('image_encoder_revision'), 'image_encoder_revision', 40)
        encoder_hash = _exact(values.get('image_encoder_sha256'), 'image_encoder_sha256', 64)
    scales = values.get('layer_scales')
    if scales is None:
        scale = _scale(values.get('scale', 1.))
    else:
        if not isinstance(scales, (list, tuple)) or not 1 <= len(scales) <= 128:
            raise ValueError('layer_scales must be a list of 1–128 finite scale values.')
        scale = tuple(_scale(value) for value in scales)
    config = ImagePromptAdapterConfig(_repository(values.get('adapter_model'), 'adapter_model'),
        _exact(values.get('revision'), 'revision', 40), weight,
        _exact(values.get('expected_sha256'), 'expected_sha256', 64),
        encoder_repo, encoder_revision, encoder_hash, scale)
    if any((item.encoder_repository, item.encoder_revision, item.encoder_sha256) !=
           (encoder_repo, encoder_revision, encoder_hash) for item in previous):
        raise ValueError('All chained image-prompt adapters must use the same explicit vision encoder, or embeddings only.')
    return (*previous, config)


def validate_image_prompt_adapters(pipeline_class, configs):
    if configs is None:
        return
    if pipeline_class not in IP_ADAPTER_PIPELINES:
        raise ValueError(f'{pipeline_class} does not support this Image Prompt Adapter. Disconnect it or select a compatible pipeline.')
    if not isinstance(configs, tuple) or not 1 <= len(configs) <= 4 or not all(
            isinstance(item, ImagePromptAdapterConfig) for item in configs):
        raise TypeError('Image Prompt Adapter must come from its configuration node, not serialized graph data.')


def _cached(repository, revision, filename, *, sha256=None, limit=4 * 1024**3):
    alias = cached_file_path(repository, filename, revision=revision)
    if not alias:
        raise FileNotFoundError(f'Image Prompt Adapter needs {repository}/{filename}@{revision}. Install this exact Hub file through Model Manager first.')
    resolved = resolve_managed_hf_cache_file(alias)
    if not 0 < resolved.stat().st_size <= limit:
        raise ValueError(f'Image Prompt Adapter file {filename} exceeds its resource limit or is empty.')
    if sha256:
        digest = hashlib.sha256()
        with resolved.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
                digest.update(chunk)
        if digest.hexdigest() != sha256:
            raise ValueError(f'Image Prompt Adapter {filename} failed SHA-256 verification. Repair it through Model Manager.')
    return Path(alias)


def resolve_image_prompt_adapter_files(configs):
    paths = [_cached(item.repository, item.revision, item.weight_name, sha256=item.sha256)
             for item in configs]
    first = configs[0]
    encoder = None
    if first.encoder_repository:
        encoder = _cached(first.encoder_repository, first.encoder_revision, 'model.safetensors',
                          sha256=first.encoder_sha256).parent
        for filename in ('config.json', 'preprocessor_config.json'):
            config_file = _cached(first.encoder_repository, first.encoder_revision, filename, limit=1024**2)
            if config_file.parent != encoder:
                raise ValueError('Vision encoder files must belong to one exact Hub snapshot.')
            payload = json.loads(config_file.read_text())
            if not isinstance(payload, dict):
                raise ValueError(f'Vision encoder {filename} must contain an object.')
            if filename == 'config.json' and payload.get('model_type') not in ('clip', 'clip_vision_model'):
                raise ValueError('Image Prompt Adapter requires the pinned CLIP vision encoder architecture.')
    return paths, encoder


def attach_image_prompt_adapters(pipeline, pipeline_class, configs, *, dtype):
    validate_image_prompt_adapters(pipeline_class, configs)
    if configs is None:
        return
    transformer = pipeline.transformer
    if getattr(transformer, 'encoder_hid_proj', None) is not None or getattr(pipeline, '_modiff_image_prompt_adapters', None):
        raise ValueError('This pipeline already owns an image adapter. Configure adapters on a fresh loader instead.')
    if any(getattr(module, '_hf_hook', None) or getattr(module, '_diffusers_hook', None)
           for module in transformer.modules()):
        raise ValueError('Image Prompt Adapter must be attached before pipeline offload hooks.')
    layers = transformer.config.num_layers
    for item in configs:
        if isinstance(item.scale, tuple) and len(item.scale) != layers:
            raise ValueError(f'layer_scales requires exactly {layers} values for this transformer.')
    paths, encoder_path = resolve_image_prompt_adapter_files(configs)
    original_processors = dict(transformer.attn_processors)
    original_projection = getattr(transformer, 'encoder_hid_proj', None)
    original_dim_type = transformer.config.get('encoder_hid_dim_type', None)
    original_encoder = getattr(pipeline, 'image_encoder', None)
    original_extractor = getattr(pipeline, 'feature_extractor', None)
    original_pipeline_config = pipeline._internal_dict
    original_transformer_config = transformer._internal_dict
    try:
        if encoder_path:
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
            encoder = CLIPVisionModelWithProjection.from_pretrained(str(encoder_path),
                local_files_only=True, use_safetensors=True, trust_remote_code=False,
                torch_dtype=dtype).to(pipeline.device).eval()
            extractor = CLIPImageProcessor.from_pretrained(str(encoder_path), local_files_only=True)
            pipeline.register_modules(image_encoder=encoder, feature_extractor=extractor)
        # Resolve the encoder independently above. The upstream convenience
        # loader does not forward the adapter revision to its encoder download.
        pipeline.load_ip_adapter([str(path.parent) if isinstance(path, Path) else path for path in paths],
            weight_name=[path.name if isinstance(path, Path) else 'adapter.safetensors' for path in paths],
            image_encoder_pretrained_model_name_or_path=None, local_files_only=True,
            low_cpu_mem_usage=True)
        pipeline.set_ip_adapter_scale([list(item.scale) if isinstance(item.scale, tuple)
                                       else [item.scale] * layers for item in configs])
    except BaseException:
        # Restore exact original objects, not freshly initialized processors.
        transformer.set_attn_processor(original_processors)
        transformer.encoder_hid_proj = original_projection
        transformer.config.encoder_hid_dim_type = original_dim_type
        pipeline.register_modules(image_encoder=original_encoder, feature_extractor=original_extractor)
        pipeline._internal_dict = original_pipeline_config
        transformer._internal_dict = original_transformer_config
        raise
    pipeline._modiff_image_prompt_adapters = configs


def validate_image_prompt_inputs(pipeline, options):
    selected = {key: options[key] for key in IP_ADAPTER_INPUTS if options.get(key) is not None}
    projection = getattr(getattr(pipeline, 'transformer', None), 'encoder_hid_proj', None)
    if not selected:
        if projection is not None and getattr(projection, 'num_ip_adapters', 0):
            raise ValueError('This pipeline has an Image Prompt Adapter. Connect its reference image/embeddings, or disconnect the adapter configuration from the loader.')
        return
    if projection is None or not getattr(projection, 'num_ip_adapters', 0):
        raise ValueError('Connect an Image Prompt Adapter configuration to the pipeline loader before using its image/embedding inputs.')
    for prefix in ('', 'negative_'):
        media, embeds = prefix + 'ip_adapter_image', prefix + 'ip_adapter_image_embeds'
        if media in selected and embeds in selected:
            raise ValueError(f'Connect either {media} or {embeds}, not both.')
        if media in selected and getattr(pipeline, 'image_encoder', None) is None:
            raise ValueError(f'{media} needs the explicitly pinned vision encoder; use embeddings or configure the encoder.')
        if media in selected:
            media_count = len(selected[media]) if isinstance(selected[media], list) else 1
            if media_count != projection.num_ip_adapters:
                raise ValueError(f'{media} requires one image or image batch per installed adapter ({projection.num_ip_adapters}).')
        if embeds in selected:
            import torch
            values = selected[embeds] if isinstance(selected[embeds], list) else [selected[embeds]]
            if (not isinstance(values, list) or len(values) != projection.num_ip_adapters or
                    any(not isinstance(item, torch.Tensor) or item.ndim not in (3, 4) or
                        item.device.type == 'meta' or not item.is_floating_point() or
                        any(size <= 0 for size in item.shape) for item in values)):
                raise ValueError(f'{embeds} needs one nonempty floating-point embedding tensor per installed adapter.')
    if getattr(pipeline, 'image_encoder', None) is None and (
            ('ip_adapter_image_embeds' in selected) != ('negative_ip_adapter_image_embeds' in selected)):
        raise ValueError('Without a vision encoder, connect both positive and negative IP-Adapter embeddings. Upstream otherwise encodes a blank image for the missing side.')
