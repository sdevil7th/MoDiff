"""Bounded source declarations for the existing native FLUX 3 Auto path.

These are parameter/loading minima and ordinary headroom policy, not measured
peak inference budgets or native output qualification. No model is imported.
"""
from __future__ import annotations

import hashlib
import json
import math

from modiff.artifact_weight_memory import (
    _loading_json_file, _snapshot_file, artifact_parameter_lower_bound,
    artifact_parameter_conversion_reserve,
)
from modiff.modular_weight_formats import _header, _selected_recipe


RECIPE_ID = 'flux3-fp8r-windows-model-cpu:v1'
PUBLIC_PROFILES = frozenset({'flux3-klein:official-provider:v1', 'flux3-klein-base:official-provider:v1'})


def selected_artifacts(model_type, repository, revision):
    """Observe the complete exact FP8r manifest using execution cache precedence."""
    from utils.huggingface import exact_cached_snapshot_path, observed_catalog_cache_selections

    recipe = _selected_recipe(model_type, repository, revision, 'fp8r')
    files = sorted(recipe['downloadFiles'])
    observations = observed_catalog_cache_selections(repository) or []
    complete = any(row == {'revision': revision, 'files': files, 'complete': True} for row in observations)
    if not complete:
        return {'installed': False, 'complete': False,
                'reason': 'Install or repair the complete exact FP8r model selection, including its variant file.'}
    snapshot = exact_cached_snapshot_path(repository, revision, 'modular_model_index.json')
    return {'installed': True, 'complete': True, 'snapshots': [{'path': str(snapshot), 'complete': True}],
            'plannedRevision': revision, 'plannedFiles': files}


def parameter_census(status, model_type, repository, revision, dtype):
    """Count pinned packed storage and a conservative whole-file loading overlap.

    The official BFL loader retains FP8 weights and FP32 row scales. Norm Q/K
    weights stay FP32. Single-file conversion retains a whole source checkpoint;
    reserve a complete destination as well, including its scale/shift remapping.
    This conservative reserve cannot claim mmap sharing or physical residency.
    """
    if type(status) is not dict or not status.get('complete') or dtype != 'bfloat16':
        return None, None, None
    snapshots = status.get('snapshots')
    if (type(snapshots) is not list or len(snapshots) != 1 or type(snapshots[0]) is not dict
            or snapshots[0].get('complete') is not True or type(snapshots[0].get('path')) is not str
            or not snapshots[0]['path']):
        raise ValueError('The FP8r census requires one exact complete snapshot observation.')
    recipe = _selected_recipe(model_type, repository, revision, 'fp8r')
    from pathlib import Path
    snapshot = Path(snapshots[0]['path'])
    if snapshot.name != revision:
        raise ValueError('The FP8r census snapshot does not match the immutable revision.')
    index = _loading_json_file(_snapshot_file(snapshot / 'modular_model_index.json', snapshot))
    if type(index) is not dict or not 0 < len(index) <= 128:
        raise ValueError('The installed Modular component index must be a bounded object.')
    expected = {'text_encoder': ['transformers', 'Qwen3VLForConditionalGeneration'],
                'vae': ['diffusers', 'AutoencoderKLFlux2'],
                'transformer': ['diffusers', recipe['loaderClass']]}
    if (index.get('_class_name') != model_type or any(type(index.get(name)) is not list
            or len(index[name]) != 3 or index[name][:2] != pair for name, pair in expected.items())):
        raise ValueError('The installed Modular component index differs from the reviewed FP8r route.')
    selected = frozenset({'text_encoder', 'vae'})
    bound = artifact_parameter_lower_bound(status, revision, dtype,
        _index_file='modular_model_index.json', _components=selected)
    reserve = artifact_parameter_conversion_reserve(status, revision, dtype,
        _index_file='modular_model_index.json', _components=selected)
    if not bound or not reserve or set(bound['components']) != selected or set(reserve['components']) != selected:
        raise ValueError('The exact FP8r recipe needs complete text encoder and VAE parameter/loading headers.')
    weight = _snapshot_file(snapshot / recipe['weight']['path'], snapshot)
    if weight.stat().st_size != recipe['weight']['byteSize']:
        raise ValueError('The FP8r variant file differs from its reviewed size.')
    tensors = _header(weight, 'fp8r')
    sizes = {'BF16': 2, 'F16': 2, 'F32': 4, 'F8_E4M3': 1}
    target = source = largest_matrix = 0
    for name, item in tensors.items():
        count = math.prod(item['shape'])
        if item['dtype'] not in sizes:
            raise ValueError('Unreviewed FP8r parameter dtype.')
        source += count * sizes[item['dtype']]
        if item['dtype'] == 'F8_E4M3':
            target += count
            largest_matrix = max(largest_matrix, count)
        elif name.endswith('.weight_scale') or '.norm.query_norm.' in name or '.norm.key_norm.' in name:
            target += count * 4
        else:
            target += count * 2
    if not 0 < source <= 2**63 or not 0 < target <= 2**63 or not largest_matrix:
        raise ValueError('Invalid bounded FP8r parameter census.')
    bound['components']['transformer'] = target
    bound.update(totalBytes=sum(bound['components'].values()), largestComponentBytes=max(bound['components'].values()))
    reserve['components']['transformer'] = {
        'loaderLibrary': 'diffusers', 'loaderClass': recipe['loaderClass'],
        'layoutSha256': hashlib.sha256(json.dumps(tensors, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'sourcePayloadBytes': source, 'requestedPayloadBytes': target,
        'convertedTargetBytes': target, 'largestSourceShardBytes': source,
        'largestConvertedTensorBytes': max(math.prod(item['shape']) * (4 if item['dtype'] == 'F32' else 2)
                                         for item in tensors.values()),
        'serialCpuOverlapBytes': source + target, 'conservativeCpuOverlapBytes': source + target,
        'serialCudaOverlapBytes': source + target, 'conservativeCudaOverlapBytes': source + target,
    }
    # The official fallback multiplies two FP32 full matrices before BF16 cast.
    # Runtime allocator/vendor headroom is still added independently by Auto.
    return bound, reserve, {'vramBytes': largest_matrix * 8, 'proofScope': 'largest_fp8r_linear_dequantization_only'}


def workload_missing(form, hardware, profile_id):
    """Validate the admitted boundary without rewriting any authored value."""
    accelerator = hardware.get('accelerator') or {}
    if (profile_id not in PUBLIC_PROFILES or hardware.get('platform') != 'windows'
            or hardware.get('architecture') != 'x86_64' or accelerator.get('kind') != 'cuda'
            or accelerator.get('backend', 'cuda') != 'cuda'
            or accelerator.get('memoryKind') != 'dedicated'
            or hardware.get('runtimeBackend') == 'rocm' or accelerator.get('sharedMemory') is True
            or form.get('device') not in {'cuda', 'cuda:0'}):
        return ['This FP8r Auto recipe requires Windows x86_64 and primary dedicated NVIDIA CUDA.']
    checks = {'width': 512, 'height': 512, 'batchSize': 1}
    if any(_number(form.get(key, expected)) != expected for key, expected in checks.items()):
        return ['This FP8r Auto recipe covers only 512×512 output and batch one; authored settings are preserved.']
    if (form.get('weightFormat') != 'fp8r' or form.get('dtype') != 'bfloat16'
            or form.get('quantizationMode', 'none') != 'none'
            or form.get('weight_loading_mode', 'cpu') != 'cpu'):
        return ['This Auto recipe requires explicit FP8r storage, BF16 compute, CPU weight loading and no additional quantization.']
    steps = 50 if profile_id.startswith('flux3-klein-base:') else 16
    if _number(form.get('steps', steps)) != steps or _number(form.get('maxSequenceLength', 8192)) not in range(1, 8193):
        return ['This FP8r Auto recipe requires the reviewed step count and at most 8192 prompt tokens.']
    if profile_id.startswith('flux3-klein-base:') and (
            _number(form.get('guidanceScale', 3.5)) != 3.5 or _number(form.get('guidanceRescale', 0)) != 0):
        return ['Base Auto preserves the reviewed 50 steps, CFG 3.5 and rescale zero recipe.']
    references = form.get('referenceWorkload', [])
    if form.get('mode') == 'text_to_image':
        return [] if references == [] else ['Text-to-image Auto cannot consume reference images.']
    if form.get('mode') != 'edit_image' or type(references) is not list or len(references) not in {1, 2}:
        return ['Edit Auto requires one or two references with actual prepared geometry.']
    pixels = 0
    for row in references:
        if type(row) is not dict or set(row) != {'sourceSize', 'processedSize'}:
            return ['Edit Auto needs actual source and processed reference geometry.']
        source, processed = row['sourceSize'], row['processedSize']
        if (type(source) is not list or len(source) != 2 or any(type(v) is not int or not 16 <= v <= 1024 for v in source)
                or processed != [v // 16 * 16 for v in source]):
            return ['Edit Auto references must fit the reviewed 16-pixel crop within 1024 pixels per side.']
        pixels += math.prod(processed)
    return [] if pixels <= 512**2 + 1024**2 else ['Edit Auto processed references exceed the reviewed combined pixel boundary.']


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None
