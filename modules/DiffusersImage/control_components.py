"""Owned ControlNet composition and bounded upstream condition routing."""
from dataclasses import dataclass
import math

CONTROL_PIPELINES = frozenset({'FluxControlNetPipeline', 'FluxControlNetImg2ImgPipeline',
                               'FluxControlNetInpaintPipeline'})


@dataclass(frozen=True)
class ControlComponentConfig:
    repository: str
    revision: str
    shared_conditions: bool = False


def validate_control_components(pipeline_class, configs):
    if configs is None:
        return
    if pipeline_class not in CONTROL_PIPELINES:
        raise ValueError(f'{pipeline_class} cannot accept Control Components. Select a compatible ControlNet pipeline or disconnect this input.')
    if not isinstance(configs, tuple) or not 1 <= len(configs) <= 4 or not all(
            isinstance(item, ControlComponentConfig) for item in configs):
        raise TypeError('Control Components must come from Control Component configuration nodes.')
    if len(configs) > 1 and any(item.shared_conditions for item in configs):
        raise ValueError('Shared conditions requires one Union component, not multiple loaded components.')


def control_component_config(values, resolver):
    previous = values.get('previous')
    if previous is None:
        previous = ()
    if not isinstance(previous, tuple) or not all(isinstance(item, ControlComponentConfig) for item in previous):
        raise TypeError('Previous components must come from Control Component configuration nodes.')
    if len(previous) >= 4:
        raise ValueError('At most four Control Components may be chained.')
    shared = values.get('shared_conditions', False)
    if type(shared) is not bool:
        raise ValueError('Shared conditions must be a boolean.')
    selection, revision = resolver(values.get('model_id'), values.get('revision'))
    result = (*previous, ControlComponentConfig(selection['value'], revision, shared))
    validate_control_components('FluxControlNetPipeline', result)
    return result


def load_control_components(configs, adapter, dtype, low_cpu_mem_usage):
    """Construct separate genuine components, never mutate resident owners."""
    from diffusers import FluxControlNetModel, FluxMultiControlNetModel
    from utils.huggingface import local_files_only

    validate_control_components(adapter.load_pipeline_class, configs)
    loaded = []
    for config in configs:
        component = FluxControlNetModel.from_pretrained(config.repository,
            revision=config.revision, torch_dtype=dtype, use_safetensors=True,
            low_cpu_mem_usage=low_cpu_mem_usage,
            local_files_only=local_files_only(config.repository))
        for field, expected in adapter.conditioning_config_requirements:
            actual = getattr(component.config, field, None)
            if type(actual) is not int or actual != expected:
                raise ValueError(f'Control Component {config.repository}: {field} must be {expected}; received {actual}.')
        if config.shared_conditions and not getattr(component, 'union', False):
            raise ValueError(f'Control Component {config.repository}: Shared conditions requires a Union model with num_mode.')
        loaded.append(component)
    # The pinned upstream preprocessor chooses a shared preparation strategy.
    # Reject incompatible raw-pixel and VAE-latent encoders before inference.
    if len({getattr(item, 'input_hint_block', None) is None for item in loaded}) != 1:
        raise ValueError('Control Components use incompatible raw-pixel and VAE-latent input strategies. Use components with the same input contract.')
    return FluxMultiControlNetModel(loaded) if len(loaded) > 1 or configs[0].shared_conditions else loaded[0]


def _number(value, field, lower, upper, integer=False):
    try:
        if type(value) not in (int, float, str):
            raise ValueError
        number = float(value)
        if not math.isfinite(number) or not lower <= number <= upper or (integer and not number.is_integer()):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{field} requires a finite {"integer" if integer else "number"} between {lower} and {upper}.') from None
    return int(number) if integer else number


def normalize_control_inputs(pipeline, values):
    """Keep scalar defaults; normalize explicit per-condition inputs only."""
    controlnet = getattr(pipeline, 'controlnet', None)
    nets = getattr(controlnet, 'nets', None)
    multiple = nets is not None
    nets = list(nets) if multiple else [controlnet]
    images = values.get('control_image')
    if multiple:
        if not isinstance(images, (list, tuple)) or not 1 <= len(images) <= 4:
            raise ValueError('Control Image requires a list of 1–4 conditions for Multi-ControlNet.')
        count = len(images)
        if len(nets) != count and not (len(nets) == 1 and getattr(nets[0], 'union', False)):
            raise ValueError(f'Control Image count {count} must match {len(nets)} loaded Control Components.')
        values['control_image'] = list(images)
    else:
        count = 1
    normalized = {}
    for key, default, high in [('conditioning_scale', 1., 2.),
                              ('control_guidance_start', 0., 1.), ('control_guidance_end', 1., 1.)]:
        raw = values.get(key)
        if raw is None:
            raw = default
        if isinstance(raw, (list, tuple)):
            if len(raw) != count:
                raise ValueError(f'{key} requires exactly {count} per-condition values.')
            entries = [_number(item, key, 0., high) for item in raw]
            values[key] = entries if multiple else entries[0]
        else:
            scalar = _number(raw, key, 0., high)
            entries = [scalar] * count
            values[key] = entries if multiple else scalar
        normalized[key] = entries
    for index, (start, end) in enumerate(zip(normalized['control_guidance_start'], normalized['control_guidance_end'])):
        if start > end:
            raise ValueError(f'Control guidance start cannot exceed end for condition {index + 1}.')
    modes = values.get('control_mode')
    modes = list(modes) if isinstance(modes, (tuple, list)) else [modes] * count
    if len(modes) != count:
        raise ValueError(f'control_mode requires exactly {count} per-condition values.')
    for index, mode in enumerate(modes):
        net = nets[index] if len(nets) > 1 else nets[0]
        if getattr(net, 'union', False):
            total = getattr(net.config, 'num_mode', None)
            if type(total) is not int or total <= 0:
                raise ValueError('Union Control Component has no valid num_mode contract.')
            modes[index] = _number(mode, f'control_mode[{index}]', 0, total - 1, integer=True)
        elif mode is not None:
            # -1 is upstream's explicit non-Union sentinel. Keep validation
            # idempotent across NodeBase preflight and execute boundaries.
            if not (multiple and mode == -1 and type(mode) is int):
                raise ValueError(f'control_mode[{index}] requires a Union Control Component; disconnect this mode.')
        elif multiple:
            # Pinned img2img/inpaint do not broadcast a scalar None into the
            # sentinel list used by T2I. Supply its identical explicit mode
            # to avoid an empty zip in FluxMultiControlNetModel.forward.
            modes[index] = -1
    if multiple or values.get('control_mode') is not None:
        values['control_mode'] = modes if multiple else modes[0]
    # Flatten only for bounded media validation; preserve upstream list layout.
    if multiple:
        return [image for group in images for image in (group if isinstance(group, (list, tuple)) else [group])], 8
    return images, 1
