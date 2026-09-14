"""Shared data-only validation for PyTorch attention masks; no processor injection."""
from __future__ import annotations


def validate_attention_mask(value, field: str = 'attention_mask'):
    import torch

    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.device.type == 'meta' or value.ndim not in (2, 3, 4)
            or any(size <= 0 for size in value.shape)
            or not (value.dtype == torch.bool or value.is_floating_point())):
        raise ValueError(f'{field} must be a materialized boolean or floating-point Tensor with 2–4 nonempty dimensions.')
    if value.is_floating_point() and (torch.isnan(value).any().item() or torch.isposinf(value).any().item()):
        raise ValueError(f'{field} must not contain NaN or positive infinity; additive negative infinity is allowed.')
    # Shape broadcasting and device/dtype compatibility belong to the actual
    # attention consumer. Do not reshape, move, or mutate a supplied mask.
    return value
