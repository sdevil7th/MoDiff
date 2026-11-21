"""Owned selection of Diffusers' documented FP8-storage linear fallback.

The checkpoint, packed parameters, quantizer and module types remain official.
Only two separately reproduced runtime rows select this adapter; ordinary
Diffusers dispatch remains unchanged elsewhere. Heavy imports stay lazy.
"""
from dataclasses import dataclass
from types import MethodType
import weakref


FALLBACK_MODE = 'fp8-storage/per-layer-dequantized-linear'


@dataclass(frozen=True)
class _OwnedFallback:
    owner: weakref.ReferenceType
    layers: tuple
    dtypes: tuple


def _official_fallback_forward(layer, inputs):
    from diffusers.quantizers.bfl.utils import dequantize_fp8r
    from torch.nn.functional import linear

    # The exact official fallback expression, without converting the stored
    # parameter or quantizing activations. The temporary weight dies per call.
    weight = dequantize_fp8r(layer.weight, layer.weight_scale).to(layer.compute_dtype)
    return linear(inputs, weight, layer.bias)


def install_owned_fallback(component, *, owner, mode):
    """Bind before hooks/PEFT; never reconfigure another source's component."""
    if mode != FALLBACK_MODE:
        raise ValueError('Unreviewed BFL FP8 execution mode.')
    import torch
    from diffusers.quantizers.bfl.utils import BFLFP8RLinear, BFLNVFP4Linear
    from peft.tuners.tuners_utils import BaseTunerLayer

    existing = vars(component).get('_modiff_bfl_fp8_execution')
    if existing is not None and (type(existing) is not _OwnedFallback or existing.owner() is not owner):
        raise ValueError('BFL FP8 compatibility belongs to a different component owner.')
    layers = []
    for module in component.modules():
        if (getattr(module, '_hf_hook', None) is not None
                or getattr(module, '_diffusers_hook', None) is not None
                or isinstance(module, BaseTunerLayer)):
            raise ValueError('BFL FP8 compatibility must precede hooks and adapters.')
        if isinstance(module, BFLNVFP4Linear):
            raise ValueError('BFL FP8 compatibility cannot change another stored scheme.')
        if not isinstance(module, BFLFP8RLinear):
            continue
        method = module.forward
        expected = BFLFP8RLinear.forward if existing is None else _official_fallback_forward
        if (type(module) is not BFLFP8RLinear
                or type(method) is not MethodType or method.__self__ is not module or method.__func__ is not expected
                or module.compute_dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or module.weight.dtype != torch.float8_e4m3fn
                or tuple(module.weight.shape) != (module.out_features, module.in_features)
                or module.weight_scale.dtype != torch.float32
                or tuple(module.weight_scale.shape) != (module.out_features,)):
            raise ValueError('BFL FP8 layer differs from the reviewed official storage/forward contract.')
        layers.append(module)
    if not layers:
        raise ValueError('BFL FP8 compatibility requires actual official packed layers.')
    dtypes = tuple(sorted({str(layer.compute_dtype).removeprefix('torch.') for layer in layers}))
    if existing is not None:
        if tuple(reference() for reference in existing.layers) != tuple(layers) or existing.dtypes != dtypes:
            raise ValueError('BFL FP8 compatibility layer inventory changed after installation.')
    else:
        marker = _OwnedFallback(weakref.ref(owner), tuple(weakref.ref(layer) for layer in layers), dtypes)
        for layer in layers:
            layer.forward = MethodType(_official_fallback_forward, layer)
        component._modiff_bfl_fp8_execution = marker
    return {'mode': FALLBACK_MODE, 'weightStorage': 'float8_e4m3fn',
            'computeDtypes': list(dtypes), 'activationQuantization': False, 'layers': len(layers)}
