"""Explicit opt-in inputs for reviewed ordinary image pipeline calls.

No reflective kwargs forwarding: this allow-list is checked against the pinned
upstream source in tests. Omitted inputs keep existing workflow behavior. Tensor
shape compatibility stays with the actual upstream consumer, not JSON coercion.
"""
from __future__ import annotations

import math
from typing import Any
from .image_prompt_adapter import IP_ADAPTER_INPUTS


_COMMON = ('num_images_per_prompt', 'sigmas', 'generator', 'latents', 'prompt_embeds')
_FLUX1 = (*_COMMON, 'prompt_2', 'pooled_prompt_embeds', 'joint_attention_kwargs')
_NEGATIVE = ('negative_prompt_2', 'negative_prompt_embeds', 'negative_pooled_prompt_embeds')
_FLUX2 = (*_COMMON, 'text_encoder_out_layers', 'attention_kwargs')
PIPELINE_CALL_INPUTS = {
    'FluxPipeline': (*_FLUX1, *_NEGATIVE, *IP_ADAPTER_INPUTS),
    'FluxImg2ImgPipeline': (*_FLUX1, *_NEGATIVE, *IP_ADAPTER_INPUTS),
    'FluxInpaintPipeline': (*_FLUX1, *_NEGATIVE, 'masked_image_latents', *IP_ADAPTER_INPUTS),
    'FluxKontextPipeline': (*_FLUX1, *_NEGATIVE, 'max_area', '_auto_resize', *IP_ADAPTER_INPUTS),
    'FluxKontextInpaintPipeline': (*_FLUX1, *_NEGATIVE, 'image_reference', 'max_area', '_auto_resize', *IP_ADAPTER_INPUTS),
    'FluxFillPipeline': (*_FLUX1, 'masked_image_latents'),
    'FluxControlPipeline': _FLUX1,
    'FluxControlImg2ImgPipeline': _FLUX1,
    'FluxControlInpaintPipeline': (*_FLUX1, 'masked_image_latents'),
    'FluxControlNetPipeline': (*_FLUX1, *_NEGATIVE, *IP_ADAPTER_INPUTS, 'control_mode'),
    'FluxControlNetImg2ImgPipeline': (*_FLUX1, 'control_mode'),
    'FluxControlNetInpaintPipeline': (*_FLUX1, 'masked_image_latents', 'control_mode'),
    'Flux2Pipeline': (*_FLUX2, 'caption_upsample_temperature'),
    'Flux2KleinPipeline': (*_FLUX2, 'negative_prompt_embeds'),
    'Flux2KleinInpaintPipeline': (*_FLUX2, 'negative_prompt_embeds', 'image_reference'),
    'Flux2KleinKVPipeline': _FLUX2,
    'FluxReduxPipeline': ('prompt_embeds_scale', 'pooled_prompt_embeds_scale',
                          'prompt_2', 'prompt_embeds', 'pooled_prompt_embeds'),
}

# Append-only presentation revisions allow old saved contracts without admitting
# arbitrary partial field maps or changing any historical execution values.
CALL_INPUT_ADDITIONS = (
    frozenset({'prompt_embeds_scale', 'pooled_prompt_embeds_scale'}),
    frozenset({'joint_attention_kwargs', 'attention_kwargs'}),
    frozenset({'output_type', 'latents_out'}),
    frozenset(IP_ADAPTER_INPUTS),
    frozenset({'control_mode'}),
)

_TENSORS = {
    'latents': 'tensor', 'prompt_embeds': 'tensor', 'pooled_prompt_embeds': 'tensor',
    'negative_prompt_embeds': 'tensor', 'negative_pooled_prompt_embeds': 'tensor',
    'masked_image_latents': 'tensor',
}
_TYPES = {
    **_TENSORS, 'generator': 'generator', 'prompt_2': 'text', 'negative_prompt_2': 'text',
    'num_images_per_prompt': 'int', 'sigmas': 'float', 'text_encoder_out_layers': 'int',
    'caption_upsample_temperature': 'float', 'max_area': 'int', '_auto_resize': 'bool',
    'image_reference': 'image',
    'prompt_embeds_scale': 'float', 'pooled_prompt_embeds_scale': 'float',
    'joint_attention_kwargs': 'object', 'attention_kwargs': 'object',
    'ip_adapter_image': 'image', 'negative_ip_adapter_image': 'image',
    'ip_adapter_image_embeds': 'tensor', 'negative_ip_adapter_image_embeds': 'tensor',
    'control_mode': 'int',
}
CALL_INPUT_PARAMS = {
    name: {
        'label': name.replace('_', ' ').strip().title(), 'type': kind,
        'display': 'input', 'hidden': True,
        'description': 'Optional upstream input. Leave disconnected to preserve the current call defaults.',
    }
    for name, kind in _TYPES.items()
}


def _number(value: Any, field: str, lower: float, upper: float, *, integer: bool = False):
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError
        numeric = float(value)
        if not math.isfinite(numeric) or not lower <= numeric <= upper or (integer and not numeric.is_integer()):
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        raise ValueError(f'{field} must be a finite {"integer" if integer else "number"} between {lower:g} and {upper:g}.') from None
    return int(numeric) if integer else numeric


def normalize_call_inputs(pipeline_class: str, values: dict[str, Any]) -> dict[str, Any]:
    allowed = PIPELINE_CALL_INPUTS.get(pipeline_class, ())
    selected = {key: values[key] for key in CALL_INPUT_PARAMS if values.get(key) is not None}
    unsupported = selected.keys() - set(allowed)
    if unsupported:
        raise ValueError(f'{pipeline_class} does not support optional input(s): {", ".join(sorted(unsupported))}. Disconnect these inputs or choose a compatible pipeline.')
    result = {}
    for key, value in selected.items():
        if key in ('prompt_2', 'negative_prompt_2'):
            if not isinstance(value, str) and not (isinstance(value, list) and 1 <= len(value) <= 8 and all(isinstance(item, str) for item in value)):
                raise ValueError(f'{key} must be text or a list of 1–8 prompt strings.')
            result[key] = list(value) if isinstance(value, list) else value
        elif key == 'num_images_per_prompt':
            result[key] = _number(value, key, 1, 8, integer=True)
        elif key == 'control_mode':
            if isinstance(value, (tuple, list)):
                if not 1 <= len(value) <= 4:
                    raise ValueError('control_mode requires 1–4 per-condition mode IDs.')
                result[key] = [None if item is None else _number(item, key, -1, 255, integer=True) for item in value]
            else:
                result[key] = _number(value, key, 0, 255, integer=True)
        elif key in ('joint_attention_kwargs', 'attention_kwargs'):
            # Data-only upstream arguments; never inject processors, callbacks,
            # or mutable KV execution state through a graph input.
            if not isinstance(value, dict) or value.keys() - {'scale', 'attention_mask'}:
                raise ValueError(f'{key} must contain only reviewed scale and attention_mask keys.')
            options = {'scale': _number(value['scale'], f'{key}.scale', -100, 100)} if 'scale' in value else {}
            if 'attention_mask' in value:
                if pipeline_class == 'Flux2KleinKVPipeline':
                    raise ValueError('Flux2KleinKVPipeline does not apply attention_mask during reference KV caching. Disconnect the mask or choose ordinary Klein.')
                from modiff.attention_arguments import validate_attention_mask
                options['attention_mask'] = validate_attention_mask(value['attention_mask'], f'{key}.attention_mask')
            result[key] = options
        elif key in ('prompt_embeds_scale', 'pooled_prompt_embeds_scale'):
            if isinstance(value, (list, tuple)):
                if not 1 <= len(value) <= 8:
                    raise ValueError(f'{key} must be a finite scale or 1–8 per-reference scales.')
                result[key] = [_number(item, key, -100, 100) for item in value]
            else:
                result[key] = _number(value, key, -100, 100)
        elif key == 'sigmas':
            if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 100:
                raise ValueError('sigmas must be a list of 1–100 finite descending noise levels.')
            sigmas = [_number(item, key, 0, 1) for item in value]
            if any(first <= second for first, second in zip(sigmas, sigmas[1:])):
                raise ValueError('sigmas must be strictly descending.')
            result[key] = sigmas
        elif key == 'text_encoder_out_layers':
            if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 128:
                raise ValueError('text_encoder_out_layers must be a nonempty bounded list of layer indices.')
            result[key] = [_number(item, key, -128, 128, integer=True) for item in value]
        elif key == 'max_area':
            result[key] = _number(value, key, 4096, 16 * 1024 * 1024, integer=True)
        elif key == 'caption_upsample_temperature':
            result[key] = _number(value, key, 0, 10)
        elif key == '_auto_resize':
            if type(value) is not bool:
                raise ValueError('_auto_resize must be a boolean.')
            result[key] = value
        elif key in _TENSORS or key == 'generator':
            import torch
            if key == 'generator':
                generators = value if isinstance(value, list) else [value]
                if not 1 <= len(generators) <= 8 or not all(isinstance(item, torch.Generator) for item in generators):
                    raise ValueError('generator must be a Torch Generator or list of 1–8 Generators, not a serialized object.')
            elif not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.device.type == 'meta' or not value.is_floating_point():
                raise ValueError(f'{key} must be a materialized floating-point Tensor.')
            elif key.endswith('prompt_embeds'):
                rank = 2 if 'pooled' in key else 3
                if value.ndim != rank or any(size <= 0 for size in value.shape):
                    raise ValueError(f'{key} must have {rank} nonempty dimensions.')
            result[key] = value
        else:
            # The action's existing media boundary validates image_reference.
            result[key] = value
    for prefix in ('', 'negative_'):
        embedding, pooled = f'{prefix}prompt_embeds', f'{prefix}pooled_prompt_embeds'
        if pooled in allowed and ((embedding in result) != (pooled in result)):
            raise ValueError(f'{embedding} and {pooled} must be connected together.')
        if pooled in result and result[embedding].shape[0] != result[pooled].shape[0]:
            raise ValueError(f'{embedding} and {pooled} must have matching batch sizes.')
    return result


def apply_call_inputs(options: dict[str, Any], target: dict[str, Any]) -> None:
    target.update(options)
    for prefix in ('', 'negative_'):
        if f'{prefix}prompt_embeds' in options:
            target[f'{prefix}prompt'] = None
            target.pop(f'{prefix}prompt_2', None)


def record_image_call_inputs(node, values, call_kwargs, adapter, encoded_inputs=None):
    """Capture effective bounded controls without claiming opaque RNG state.

    The adapter may consume text while preparing embeddings before dispatch.
    Supplied embeddings replace that text; supplied generators and custom
    schedules do not establish the saved seed or number of executed steps.
    """
    from modiff.execution_input_provenance import FIELD_NAMES

    effective = {**call_kwargs, **(encoded_inputs or {})}
    source_fields = {}
    for source, destination in (
        ('guidance_scale', adapter.guidance_parameter),
        ('guidance_scale_2', adapter.secondary_guidance_parameter),
    ):
        if destination and destination in call_kwargs:
            source_fields[destination] = source
    # These aliases retain the generic receipt field's existing meaning.
    for source, destination in (
        ('max_sequence_length', adapter.max_sequence_length_parameter),
        ('conditioning_scale', adapter.conditioning_scale_parameter),
    ):
        if destination and destination in call_kwargs:
            effective[source] = call_kwargs[destination]
    for prefix in ('', 'negative_'):
        if values.get(f'{prefix}prompt_embeds') is not None:
            effective[f'{prefix}prompt'] = None
            effective[f'{prefix}prompt_2'] = None
    effective['seed'] = values.get('seed') if values.get('generator') is None else None
    if call_kwargs.get('sigmas') is not None:
        effective['num_inference_steps'] = None
    # Explicit absence prevents stale saved controls or ancestor values from
    # being presented as the final consumer's effective controls.
    for key in values.keys() & FIELD_NAMES.keys():
        effective.setdefault(key, None)
    node.record_generation_inputs(effective)
    node._execution_input_source_fields = source_fields
