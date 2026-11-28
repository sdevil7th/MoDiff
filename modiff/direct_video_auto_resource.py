"""Reviewed direct-video placement and partial storage admission, without imports.

This is a loading/parameter contract, not a measured inference peak. Ordinary
workflow headroom, owner lifetimes and dispatch-time free-memory checks remain.
"""
from __future__ import annotations

import math

REPOSITORY = 'Lightricks/LTX-Video-0.9.8-13B-distilled'
REVISION = '7c64400e1861cc0d7b98d570a1926d5408ec60cd'
_COMPONENTS = {'text_encoder': ('transformers', 'T5EncoderModel'),
               'transformer': ('diffusers', 'LTXVideoTransformer3DModel'),
               'vae': ('diffusers', 'AutoencoderKLLTXVideo')}
_MODES = {'text_to_video', 'image_to_video', 'reference_to_video', 'video_to_video'}
CONTRACT = 'ltx-direct-shared-rocm-storage-v1'


def shared_video_policy(form: dict, hardware: dict) -> bool:
    """Only this declared direct pipeline on Linux's primary ROCm UMA device.

    Capability-list ordering, CPU, dedicated CUDA, MPS, XPU and other models
    keep their existing policy. A stale Expert choice is not Auto authority.
    """
    accelerator = hardware.get('accelerator') or {}
    return (form.get('modelType') == 'LTXVideoPipeline' and form.get('mode') in _MODES
            and (form.get('modelRepo') or REPOSITORY) == REPOSITORY
            and hardware.get('platform') == 'linux' and accelerator.get('kind') == 'cuda'
            and accelerator.get('backend') == 'rocm' and accelerator.get('vendor') == 'amd'
            and accelerator.get('memoryKind') in {'shared', 'unified'}
            and form.get('device', 'cuda:0') in {'cuda', 'cuda:0'})


def shared_video_storage(candidate: dict, form: dict, hardware: dict) -> dict:
    """Require exact selected dense headers and conservative source/destination.

    Group CPU hooks retain selected host weights; initial loading can overlap
    source and requested storage. Budget both entire sets, independent of
    component order, serial/parallel flags or streaming eligibility. GPU group
    and activation peaks remain unmeasured, covered only by existing headroom.
    """
    if not shared_video_policy(form, hardware):
        return {}
    limits = {'width': 768, 'height': 512, 'steps': 8, 'batchSize': 1, 'guidanceScale': 1,
              'numFrames': 161 if form.get('mode') == 'image_to_video' else
                           121 if form.get('mode') == 'text_to_video' else 81}
    for field, maximum in limits.items():
        value = form.get(field)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)
                                  or not 0 < value <= maximum):
            raise ValueError(f'Shared-memory video Auto has no reviewed storage route for {field}={value!r}; keep the workload in Custom memory.')
    bound = candidate.get('parameterStorageLowerBound') or {}
    conversion = candidate.get('parameterConversionReserve') or {}
    resolved = (candidate.get('artifactResolution') or {}).get('resolved') or {}
    if (candidate.get('executionPath') != 'direct-diffusers-video'
            or candidate.get('executionProfileId') != 'ltx-video:direct'
            or candidate.get('modelRepo') != REPOSITORY or resolved.get('revision') != REVISION
            or form.get('revision') not in (None, '', REVISION) or candidate.get('dtype') != 'bfloat16'
            or candidate.get('offloadMode') != 'group_cpu' or candidate.get('autoOffload') is not True
            or candidate.get('deviceMap') not in (None, 'none')
            or candidate.get('quantizationMode', 'none') != 'none'
            or candidate.get('loadedQuantization', 'none') != 'none'
            or bound.get('proofScope') != 'parameter_storage_lower_bound_only'
            or bound.get('revision') != REVISION or bound.get('dtype') != 'bfloat16'
            or conversion.get('proofScope') != 'source_requested_loading_overlap_only'
            or conversion.get('revision') != REVISION or conversion.get('dtype') != 'bfloat16'
            or conversion.get('complete') is not False):
        raise ValueError('Shared-memory video Auto requires its exact BF16 pin, CPU group loading without a device map, and complete component headers.')
    components, sources = bound.get('components'), conversion.get('components')
    if (type(components) is not dict or set(components) != set(_COMPONENTS)
            or type(sources) is not dict or set(sources) != set(_COMPONENTS)
            or any(type(value) is not int or not 0 < value <= 2**63 for value in components.values())
            or type(bound.get('totalBytes')) is not int or bound['totalBytes'] != sum(components.values())):
        raise ValueError('Shared-memory video Auto needs all three neural component storage headers.')
    source_total = 0
    for name, (library, klass) in _COMPONENTS.items():
        record = sources[name]
        if (type(record) is not dict or record.get('loaderLibrary') != library or record.get('loaderClass') != klass
                or type(record.get('sourcePayloadBytes')) is not int or not 0 < record['sourcePayloadBytes'] <= 2**63
                or record.get('requestedPayloadBytes') != components[name]):
            raise ValueError('The selected video component loading contract is incomplete or changed.')
        source_total += record['sourcePayloadBytes']
    cold = 2 * max(source_total, bound['totalBytes'])
    if cold > 2**63:
        raise ValueError('The video cold-loading storage exceeds the finite resource bound.')
    return {'contract': CONTRACT, 'systemRamBytes': bound['totalBytes'], 'coldSystemRamBytes': cold,
            'proofScope': 'parameter_and_cold_storage_only', 'inferencePeakMeasured': False}
