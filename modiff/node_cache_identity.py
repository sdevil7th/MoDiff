"""Process-local input snapshots; never copy resident model weights."""
import hashlib

import numpy as np
import torch
from PIL.Image import Image


def input_snapshot(value, depth=0, _memo=None):
    """Freeze data mutations separately from NodeBase's authority/equality checks.

    Tensor version counters detect ordinary in-place operations without a device
    readback or weight copy. Opaque runtime objects remain identity-bound; their
    producers must invalidate descendants when changing adapters/components.
    """
    kind = type(value)
    if value is None or kind in (str, bytes, int, float, bool):
        return kind, value
    if depth > 32:
        return kind, id(value)
    if isinstance(value, torch.Tensor):
        try:
            version = value._version
        except RuntimeError:  # Inference tensors do not expose a version counter.
            version = None
        return kind, id(value), version, value.device, value.dtype, tuple(value.shape)
    if isinstance(value, torch.Generator):
        return kind, id(value), hashlib.sha256(value.get_state().numpy().tobytes()).digest()
    if isinstance(value, np.ndarray):
        return kind, value.dtype.str, value.shape, hashlib.sha256(value.tobytes()).digest()
    if isinstance(value, Image):
        return kind, value.mode, value.size, hashlib.sha256(value.tobytes()).digest()
    if kind in (dict, list, tuple, set, frozenset):
        memo = {} if _memo is None else _memo
        if id(value) in memo:
            return memo[id(value)]
        # Runtime inputs may contain cycles or repeated container aliases.
        # Visit each once; never expand a branching cycle to the depth limit.
        memo[id(value)] = (kind, id(value))
        if kind is dict:
            snapshot = kind, frozenset((input_snapshot(k, depth + 1, memo), input_snapshot(v, depth + 1, memo))
                                      for k, v in value.items())
        elif kind in (list, tuple):
            snapshot = kind, tuple(input_snapshot(item, depth + 1, memo) for item in value)
        else:
            snapshot = kind, frozenset(input_snapshot(item, depth + 1, memo) for item in value)
        memo[id(value)] = snapshot
        return snapshot
    return kind, id(value)


def implementation_identity(node):
    """Identify executable code actually loaded, including local reloads/patches."""
    callback = getattr(node, node.CALLBACK)
    callback = getattr(callback, '__func__', callback)
    return type(node), getattr(callback, '__code__', callback)
