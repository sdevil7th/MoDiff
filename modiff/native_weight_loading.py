"""Declared cold loading policies for the existing native image loader.

This module inspects metadata only. The actual official component loader must
recheck its class/configuration and resources before allocating weights.
"""
from __future__ import annotations

from modiff.artifact_weight_memory import (
    native_disk_lora_loading_floor, native_parameter_conversion_reserve,
)

# Initial reviewed uniform-BF16 loading contracts, not namespace-wide trust.
# Qwen3's small nonpersistent FP32 rotary buffers remain runtime overhead.
REVIEWED_CUDA_WEIGHT_LOADER_COMPONENTS = {
    "transformer": ("diffusers", "ZImageTransformer2DModel"),
    "vae": ("diffusers", "AutoencoderKL"),
    "text_encoder": ("transformers", "Qwen3Model"),
}
REVIEWED_CUDA_WEIGHT_LOADER_CLASSES = frozenset(REVIEWED_CUDA_WEIGHT_LOADER_COMPONENTS.values())
CUDA_WEIGHT_LOADING_COMPONENTS = frozenset(REVIEWED_CUDA_WEIGHT_LOADER_COMPONENTS)


def normalize_weight_loading_mode(value="cpu") -> str:
    if type(value) is not str or value not in {"cpu", "cuda"}:
        raise ValueError("Weight loading mode must be cpu or cuda.")
    return value


def declared_cuda_weight_loaders(reserve: dict | None, lower_bound: dict | None) -> bool:
    """Require the exact reviewed heavy classes without importing a model."""
    if not isinstance(reserve, dict) or not isinstance(lower_bound, dict):
        return False
    records, weights = reserve.get("components"), lower_bound.get("components")
    if (not isinstance(records, dict) or not isinstance(weights, dict) or not records
            or set(records) != set(weights) or not set(records) <= CUDA_WEIGHT_LOADING_COMPONENTS
            or reserve.get("dtype") != "bfloat16" or lower_bound.get("dtype") != "bfloat16"):
        return False
    return all(isinstance(record, dict)
               and all(type(record.get(key)) is str for key in ("loaderLibrary", "loaderClass"))
               and (record["loaderLibrary"], record["loaderClass"]) == REVIEWED_CUDA_WEIGHT_LOADER_COMPONENTS[name]
               for name, record in records.items())


def native_disk_lora_conversion_reserve(
    reserve, *, lower_bound, targets: tuple[str, ...], dtype: str, device: str,
    parallel_loading: bool,
) -> dict:
    """Add prior deferred targets beside the current source/cast overlap.

    Existing native disk LoRA loads its reviewed mutation targets last and keeps
    all of them until official injection. Other components offload beforehand.
    Adapter storage remains the caller's independent additive budget.
    """
    if not native_disk_lora_loading_floor(lower_bound, targets=targets, dtype=dtype):
        return {}
    baseline = native_parameter_conversion_reserve(
        reserve, lower_bound=lower_bound, execution_path="modular-diffusers",
        offload_mode="group_disk", device=device, dtype=dtype,
        parallel_loading=parallel_loading,
    )
    if not baseline:
        return {}
    requested = {name: lower_bound["components"][name] for name in targets}
    selected_bound = {**lower_bound, "components": requested, "totalBytes": sum(requested.values()),
                      "largestComponentBytes": max(requested.values())}
    selected_reserve = {**reserve, "components": {name: reserve["components"][name] for name in targets}}
    staged = native_parameter_conversion_reserve(
        selected_reserve, lower_bound=selected_bound, execution_path="modular-diffusers",
        offload_mode="model_cpu", device=device, dtype=dtype,
        parallel_loading=parallel_loading,
    )
    return {"systemRamBytes": max(baseline["systemRamBytes"], staged["systemRamBytes"]), "vramBytes": 0} if staged else {}
