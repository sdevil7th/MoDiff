"""Bounded inspection of built-in data controls; runtime preparation markers."""
from __future__ import annotations
from contextvars import ContextVar
import json
import math

RUNTIME_VALUES = ContextVar('workflow_auto_values', default={})
DATA_MODULES = {'modules.Primitive', 'modules.Text', 'modules.Image', 'modules.ImageOperations', 'modules.Audio', 'modules.Video'}
DATA_ACTIONS = {
    ('modules.Tensor', 'SeededGenerator'), ('modules.Tensor', 'AttentionArguments'),
    ('modules.DiffusersRuntime', 'PipelineQuantizationConfigV2'),
    ('modules.DiffusersRuntime', 'DiffusersExecutionRecipe'),
    ('modules.DiffusersImage', 'OutpaintCanvas'),
}


class DeferredResourceValue(ValueError):
    def __init__(self, node_id, field, source_id, source_key, nodes):
        required, pending = set(), [source_id]
        while pending:
            key = pending.pop()
            if key in required:
                continue
            source = nodes[key]
            if source['module'] not in DATA_MODULES and (source['module'], source['action']) not in DATA_ACTIONS:
                raise ValueError(f'{node_id}.{field} is computed by {source_id}; its supplier needs a reviewed data-only preparation contract.')
            required.add(key)
            pending.extend(p['sourceId'] for p in source['params'].values() if p.get('sourceId'))
        self.node_ids = required
        self.target = {'nodeId': node_id, 'field': field, 'sourceId': source_id, 'sourceKey': source_key}
        super().__init__(f'{node_id}.{field} will be resolved by data preparation when Run is pressed.')


def inspect_execution_recipe_controls(nodes, loader_id, literal):
    """Inspect only declared controls, never construct a recipe or quantizer."""
    param = nodes[loader_id]['params'].get('execution_recipe') or {}
    recipe_id = param.get('sourceId')
    if not recipe_id:
        if param.get('value') not in (None, {}):
            raise ValueError('Auto needs a reviewed Diffusers Execution Recipe connection; inline recipe overrides require Custom memory.')
        return {}, {}
    recipe = nodes[recipe_id]
    if (recipe['module'], recipe['action'], param.get('sourceKey')) != (
        'modules.DiffusersRuntime', 'DiffusersExecutionRecipe', 'execution_recipe'
    ):
        raise ValueError('Auto cannot inspect this execution recipe supplier; use the reviewed Diffusers Execution Recipe node or Custom memory.')

    def control(field, default):
        if field not in recipe['params']:
            return default
        return literal(nodes, recipe_id, field)

    # The eager native/math paths retain ordinary SDPA behavior. Other kernels,
    # placement, compilation and caches need their separate accepted recipes.
    if str(control('attention_backend', 'auto') or 'auto') not in {'auto', 'native', '_native_math'}:
        raise ValueError('The connected attention backend has no accepted workflow Auto recipe; select Custom memory to preserve it.')
    for field in ('regional_compile', 'layerwise_casting', 'channels_last'):
        if bool(control(field, False)):
            raise ValueError(f'The connected {field} setting needs a separately accepted Auto recipe; select Custom memory to preserve it.')
    for field, default in (('device_map', 'none'), ('denoiser_cache', 'none')):
        if str(control(field, default) or default) != default:
            raise ValueError(f'The connected {field} override has no accepted workflow Auto recipe; select Custom memory.')
    for field in ('max_memory', 'device_map_overrides'):
        value = control(field, {})
        if isinstance(value, str):
            try:
                value = json.loads(value or '{}')
            except json.JSONDecodeError as error:
                raise ValueError(f'{recipe_id}.{field} must contain a JSON object.') from error
        if value not in (None, {}):
            raise ValueError(f'The connected {field} limit needs Custom memory; Auto cannot replace it.')
    for field in ('vae_slicing', 'vae_tiling'):
        if not bool(control(field, True)):
            raise ValueError(f'Disabling connected {field} has no accepted workflow Auto memory recipe; select Custom memory.')

    quant_param = recipe['params'].get('quantization_config') or {}
    quant_id = quant_param.get('sourceId')
    if quant_id:
        quant = nodes[quant_id]
        if (quant['module'], quant['action'], quant_param.get('sourceKey')) != (
            'modules.DiffusersRuntime', 'PipelineQuantizationConfigV2', 'quantization_config'
        ):
            raise ValueError('Auto needs the reviewed Pipeline Quantization Config V2 supplier or Custom memory.')
        backend = literal(nodes, quant_id, 'backend') or 'none'
        overrides = literal(nodes, quant_id, 'component_overrides') if 'component_overrides' in quant['params'] else {}
        if isinstance(overrides, str):
            try:
                overrides = json.loads(overrides or '{}')
            except json.JSONDecodeError as error:
                raise ValueError(f'{quant_id}.component_overrides must contain a JSON object.') from error
        if backend != 'none' or overrides not in (None, {}):
            raise ValueError('The connected per-component quantization needs its own accepted Auto recipe; select Custom memory to preserve it.')
    elif quant_param.get('value') is not None:
        raise ValueError('Inline quantization configuration requires Custom memory.')
    offload = str(control('offload_mode', 'none') or 'none')
    device = str(control('device', 'cuda:0') or 'cuda:0')
    return {'offload_mode': offload, 'auto_offload': offload != 'none', 'device': device}, {
        'offload_mode': (recipe_id, 'offload_mode'), 'device': (recipe_id, 'device'),
    }


def inspect_native_runtime_controls(nodes, loader_id, literal):
    """Capture the exact bounded native policies without importing model code."""
    loader = nodes[loader_id]
    if (loader['module'], loader['action']) != ('modules.ModularDiffusers', 'ModelsLoader'):
        return {}
    controls = {}
    if 'attention_backend' in loader['params']:
        backend = literal(nodes, loader_id, 'attention_backend')
        if backend in (None, ''):
            backend = 'inherit'
        if not isinstance(backend, str) or backend not in {'inherit', 'auto', 'native', '_native_math'}:
            raise ValueError('Native Modular attention has no accepted Auto policy; use inherit, auto, native, or _native_math.')
        controls['attention_backend'] = backend
    for field in ('vae_slicing', 'vae_tiling'):
        if field in loader['params']:
            value = literal(nodes, loader_id, field)
            if value is not None and type(value) is not bool:
                raise ValueError(f'{loader_id}.{field} must be an explicit boolean or unset.')
            controls[field] = value
    return controls


def inspect_resource_value(nodes, node_id, field, visited=frozenset()):
    key = (node_id, field)
    if key in visited or len(visited) > 128:
        raise ValueError(f'{node_id}.{field} contains a resource input cycle or exceeds 128 control links.')
    param = nodes[node_id]['params'].get(field, {})
    if not param.get('sourceId'):
        return param.get('value')
    source_id, source_field = param['sourceId'], param.get('sourceKey')
    if (source_id, source_field) in RUNTIME_VALUES.get():
        return RUNTIME_VALUES.get()[(source_id, source_field)]
    source = nodes[source_id]
    def value(name):
        return inspect_resource_value(nodes, source_id, name, visited | {key})
    if (
        (nodes[node_id]['module'], nodes[node_id]['action']) == ('modules.ModularDiffusers', 'DecodeLatents')
        and (source['module'], source['action']) == ('modules.ModularDiffusers', 'Denoise')
        and source_field in {'out_width', 'out_height'}
        and field == source_field.removeprefix('out_')
        and nodes[node_id]['params'].get('latents', {}).get('sourceId') == source_id
        and nodes[node_id]['params'].get('latents', {}).get('sourceKey') == 'latents'
    ):
        # These split-decoder ports carry the originating denoiser's geometry.
        # Forecast only its explicitly inspected integer dimension. The normal
        # executor compares the actual connected output to this captured value
        # before decoder allocation, so changed/normalized geometry cannot
        # silently reuse this envelope. Never execute a model during planning.
        dimension = value(field)
        if isinstance(dimension, bool):
            raise ValueError(f'{source_id}.{field} must declare a positive integer dimension.')
        try:
            number = float(dimension)
        except (TypeError, ValueError) as error:
            raise ValueError(f'{source_id}.{field} must declare a positive integer dimension.') from error
        if not math.isfinite(number) or number <= 0 or not number.is_integer():
            raise ValueError(f'{source_id}.{field} must declare a positive integer dimension.')
        return int(number)
    if source['module'] == 'modules.Primitive':
        if source['action'] in {'String', 'Integer', 'Float', 'Boolean'}:
            return value('value')
        if source['action'] == 'TextValue' and source_field == 'output':
            return value('text') or ''
        if source['action'] == 'ToList' and source_field == 'list':
            return [value(name) for name in source['params'] if name == 'item' or name.startswith('item>>>')]
    if source['module'] == 'modules.Text' and source['action'] == 'ProcessText':
        from modules.Text.main import evaluate_data_operation
        args = {name: value(name) for name, param in source['params'].items() if param.get('display') != 'output' and name not in {'output', 'selected_index', 'item_count'}}
        try:
            outputs = evaluate_data_operation(args)
        except (TypeError, ValueError) as error:
            raise ValueError(f'{source_id}: {error}') from error
        if source_field not in outputs:
            raise ValueError(f'{source_id} has no data output {source_field}.')
        return outputs[source_field]
    raise DeferredResourceValue(node_id, field, source_id, source_field, nodes)
