"""Composable runtime configuration nodes shared by Diffusers pipelines."""

import gc
import hashlib
import json
import importlib.util
import os
from pathlib import Path
from typing import Any

from modiff.NodeBase import NodeBase
from modiff.model_artifact_catalog import (
    catalog_artifact_file,
    catalog_repository_pin,
    resolve_model_revision,
)
from modules.DiffusersImage.main import QUANT_COMPONENTS, quant_config_for
from utils.torch_utils import str_to_dtype


ATTENTION_BACKENDS = [
    "auto",
    "native",
    "_native_flash",
    "_native_efficient",
    "_native_math",
    "_native_cudnn",
    "flex",
    "flash",
    "flash_hub",
    "flash_varlen",
    "flash_varlen_hub",
    "flash_4_hub",
    "_flash_3",
    "_flash_varlen_3",
    "_flash_3_hub",
    "_flash_3_varlen_hub",
    "sage",
    "sage_hub",
    "sage_varlen",
    "xformers",
]

ATTENTION_COMPONENTS = (
    "transformer",
    "transformer_2",
    "unet",
    "controlnet",
    "prior",
)

SAFETENSORS_DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}

NO_QUANTIZATION_CONFIG = {
    "schema_version": 1,
    "backend": "none",
    "disabled": True,
}


def verify_cataloged_artifact_file(path: Path, contract: dict[str, Any]) -> None:
    """Verify one downloaded model file against its immutable catalog identity."""

    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size != contract["byteSize"]:
        raise RuntimeError("The cataloged model file has an unexpected size.")
    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != contract["sha256"]:
        raise RuntimeError("The cataloged model file failed its immutable SHA-256 check.")


def _normalize_optional_quantization_config(value: Any):
    if isinstance(value, dict) and value.get("disabled") is True and value.get("backend") == "none":
        return None
    return value


def enable_parallel_weight_loading() -> bool:
    """Enable Hugging Face's supported sharded-loader fast path.

    Diffusers recommends pairing parallel shard loading with a direct CUDA
    device map so the allocator can reserve the destination up front. Respect
    an explicit environment choice; otherwise enable the upstream default
    worker count for complete pipelines that opt into direct loading.
    """

    configured = os.environ.get("HF_ENABLE_PARALLEL_LOADING")
    if configured is None:
        os.environ["HF_ENABLE_PARALLEL_LOADING"] = "YES"
        return True
    return configured.strip().upper() in {"1", "ON", "TRUE", "YES"}


def _string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        values = value.replace("\n", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]
    return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _component_names(value: Any) -> list[str]:
    return [name for name in _string_list(value) if name in QUANT_COMPONENTS]


def _model_value(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("value") or value.get("id") or "").strip()
    if isinstance(value, list):
        return _model_value(value[0]) if value else ""
    return str(value or "").strip()


def _component_for_weight(filename: str) -> str:
    normalized = str(filename).replace("\\", "/").lstrip("/")
    parts = [part for part in normalized.split("/") if part]
    return parts[0] if len(parts) > 1 else "root"


def summarize_safetensors_files(files: list[dict[str, Any]]) -> dict[str, Any]:
    components: dict[str, dict[str, Any]] = {}
    total_source_bytes = 0
    total_weight_bytes = 0
    total_parameters = 0
    largest_shard = None

    for item in files:
        name = str(item.get("name") or "")
        component = _component_for_weight(name)
        source_bytes = item.get("source_bytes")
        source_bytes = int(source_bytes) if isinstance(source_bytes, int) and source_bytes >= 0 else None
        dtype_counts = {
            str(dtype): int(count)
            for dtype, count in dict(item.get("parameter_count") or {}).items()
            if isinstance(count, int) and count >= 0
        }
        parameter_count = sum(dtype_counts.values())
        weight_bytes = sum(SAFETENSORS_DTYPE_BYTES.get(dtype, 0) * count for dtype, count in dtype_counts.items())
        entry = components.setdefault(
            component,
            {
                "component": component,
                "source_bytes": 0,
                "source_bytes_known": True,
                "weight_bytes": 0,
                "parameter_count": 0,
                "dtype_counts": {},
                "files": [],
                "largest_shard": None,
            },
        )
        entry["files"].append(name)
        entry["parameter_count"] += parameter_count
        entry["weight_bytes"] += weight_bytes
        for dtype, count in dtype_counts.items():
            entry["dtype_counts"][dtype] = entry["dtype_counts"].get(dtype, 0) + count
        if source_bytes is None:
            entry["source_bytes_known"] = False
        else:
            entry["source_bytes"] += source_bytes
            total_source_bytes += source_bytes
            shard = {"name": name, "bytes": source_bytes}
            if entry["largest_shard"] is None or source_bytes > entry["largest_shard"]["bytes"]:
                entry["largest_shard"] = shard
            if largest_shard is None or source_bytes > largest_shard["bytes"]:
                largest_shard = shard
        total_parameters += parameter_count
        total_weight_bytes += weight_bytes

    return {
        "components": [components[name] for name in sorted(components)],
        "total_source_bytes": total_source_bytes,
        "total_weight_bytes": total_weight_bytes,
        "total_parameter_count": total_parameters,
        "largest_shard": largest_shard,
        "file_count": len(files),
    }


def inspect_diffusers_components(model_id: str, revision: str | None = None) -> dict[str, Any]:
    """Read Safetensors headers without materializing model weights."""
    from huggingface_hub import HfApi, parse_local_safetensors_file_metadata, parse_safetensors_file_metadata
    from modiff.config import CONFIG

    model_path = Path(model_id).expanduser()
    files = []
    if model_path.exists():
        if model_path.is_file():
            candidates = [model_path] if model_path.suffix == ".safetensors" else []
            root = model_path.parent
        else:
            candidates = sorted(model_path.rglob("*.safetensors"))
            root = model_path
        for candidate in candidates:
            metadata = parse_local_safetensors_file_metadata(candidate)
            files.append(
                {
                    "name": candidate.relative_to(root).as_posix(),
                    "source_bytes": candidate.stat().st_size,
                    "parameter_count": dict(metadata.parameter_count),
                }
            )
        source = "local"
    else:
        revision = resolve_model_revision(model_id, revision)
        api = HfApi(token=CONFIG.hf["token"], library_name="MoDiff")
        try:
            info = api.model_info(model_id, revision=revision, files_metadata=True)
        except TypeError:
            info = api.model_info(model_id, revision=revision)
        siblings = getattr(info, "siblings", []) or []
        for sibling in siblings:
            filename = str(getattr(sibling, "rfilename", "") or "")
            if not filename.endswith(".safetensors"):
                continue
            size = getattr(sibling, "size", None)
            metadata = parse_safetensors_file_metadata(
                model_id,
                filename,
                revision=revision,
                token=CONFIG.hf["token"],
            )
            files.append(
                {
                    "name": filename,
                    "source_bytes": int(size) if isinstance(size, int) else None,
                    "parameter_count": dict(metadata.parameter_count),
                }
            )
        source = "hub"

    summary = summarize_safetensors_files(files)
    summary.update({"model_id": model_id, "revision": revision, "source": source})
    return summary


def build_quantization_config_v2(
    *,
    backend: str,
    components: Any,
    dtype: Any,
    excluded_modules: Any = None,
    component_overrides: Any = None,
):
    """Build a real per-component PipelineQuantizationConfig.

    ``component_overrides`` maps component names to either a backend string or
    ``{"backend": ..., "excluded_modules": [...]}``. This keeps the common UI
    compact while allowing graph authors to preserve sensitive projections,
    norms, or modulation layers per component.
    """
    selected = _component_names(components)
    default_backend = str(backend or "none")
    default_excluded = _string_list(excluded_modules)

    if component_overrides in (None, ""):
        overrides = {}
    elif isinstance(component_overrides, str):
        try:
            overrides = json.loads(component_overrides)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Component overrides must be valid JSON: {exc.msg}.") from exc
    elif isinstance(component_overrides, dict):
        overrides = component_overrides
    else:
        raise TypeError("Component overrides must be a JSON object or dictionary.")
    if not isinstance(overrides, dict):
        raise ValueError("Component overrides must decode to a JSON object.")

    unknown = sorted(set(overrides) - set(QUANT_COMPONENTS))
    if unknown:
        raise ValueError(f"Unknown quantized components: {', '.join(unknown)}.")

    component_order = list(dict.fromkeys([*selected, *overrides.keys()]))
    quant_mapping = {}
    summary = {}
    for component in component_order:
        override = overrides.get(component, {})
        if isinstance(override, str):
            component_backend = override
            component_excluded = default_excluded
        elif isinstance(override, dict):
            component_backend = str(override.get("backend") or default_backend)
            component_excluded = _string_list(override.get("excluded_modules", default_excluded))
        else:
            raise TypeError(f"Override for {component} must be a backend string or object.")
        if component_backend == "none":
            continue
        config = quant_config_for(component_backend, dtype, component_excluded)
        if config is None:
            raise ValueError(f"Unsupported quantization backend {component_backend!r} for {component}.")
        quant_mapping[component] = config
        summary[component] = {
            "backend": component_backend,
            "excluded_modules": component_excluded,
        }

    if not quant_mapping:
        return None, summary
    from diffusers.quantizers import PipelineQuantizationConfig

    return PipelineQuantizationConfig(quant_mapping=quant_mapping), summary


def apply_attention_backend(pipeline: Any, backend: str, components: Any = None) -> dict[str, Any]:
    requested = str(backend or "auto")
    if requested not in ATTENTION_BACKENDS:
        raise ValueError(f"Unsupported attention backend {requested!r}.")
    if requested == "auto":
        return {"requested": "auto", "applied": [], "default_selection": True}

    requested_components = _string_list(components)
    candidate_names = requested_components or list(ATTENTION_COMPONENTS)
    applied = []
    unsupported = []
    seen = set()
    for name in candidate_names:
        component = pipeline if name in ("pipeline", "self") else getattr(pipeline, name, None)
        if component is None or id(component) in seen:
            continue
        seen.add(id(component))
        setter = getattr(component, "set_attention_backend", None)
        if not callable(setter):
            unsupported.append(name)
            continue
        try:
            setter(requested)
        except Exception as exc:
            raise RuntimeError(f"Could not apply attention backend {requested!r} to {name}: {exc}") from exc
        applied.append(name)

    if not applied:
        raise RuntimeError(
            f"Attention backend {requested!r} could not be applied because the selected pipeline components "
            "do not expose set_attention_backend()."
        )
    return {
        "requested": requested,
        "applied": applied,
        "unsupported": unsupported,
        "default_selection": False,
    }


def configure_vae_memory(pipeline: Any, *, slicing: bool, tiling: bool) -> dict[str, Any]:
    vae = getattr(pipeline, "vae", None)
    owner = vae if vae is not None else pipeline
    applied = []
    unsupported = []

    def configure(enabled: bool, enable_names: tuple[str, ...], disable_names: tuple[str, ...], label: str):
        names = enable_names if enabled else disable_names
        method = next((getattr(owner, name, None) for name in names if callable(getattr(owner, name, None))), None)
        if method is None:
            unsupported.append(label)
            return
        method()
        applied.append({"feature": label, "enabled": enabled})

    configure(slicing, ("enable_slicing", "enable_vae_slicing"), ("disable_slicing", "disable_vae_slicing"), "slicing")
    configure(tiling, ("enable_tiling", "enable_vae_tiling"), ("disable_tiling", "disable_vae_tiling"), "tiling")
    return {"applied": applied, "unsupported": unsupported}


def configure_denoiser_cache(
    pipeline: Any,
    *,
    strategy: str,
    threshold: float = 0.05,
    options: Any = None,
) -> dict[str, Any]:
    requested = str(strategy or "none")
    cache_options = _json_object(options, "Denoiser cache options")
    candidates = [
        (name, getattr(pipeline, name, None))
        for name in ("transformer", "transformer_2", "unet", "prior")
        if getattr(pipeline, name, None) is not None
    ]
    if requested == "none":
        disabled = []
        for name, component in candidates:
            if bool(getattr(component, "is_cache_enabled", False)):
                disable = getattr(component, "disable_cache", None)
                if callable(disable):
                    disable()
                    disabled.append(name)
        return {"requested": requested, "applied": [], "disabled": disabled}
    from diffusers.hooks import (
        FasterCacheConfig,
        FirstBlockCacheConfig,
        MagCacheConfig,
        PyramidAttentionBroadcastConfig,
        TaylorSeerCacheConfig,
        TextKVCacheConfig,
    )

    if requested == "first_block":
        config = FirstBlockCacheConfig(threshold=float(cache_options.get("threshold", threshold)))
    elif requested == "magcache":
        calibrate = bool(cache_options.get("calibrate", False))
        mag_ratios = cache_options.get("mag_ratios")
        if not calibrate and mag_ratios is None:
            raise ValueError(
                "MagCache needs model-specific mag_ratios. Set calibrate=true and run one representative "
                "generation to measure them, or paste a previously calibrated mag_ratios array into cache options."
            )
        config = MagCacheConfig(
            threshold=float(cache_options.get("threshold", 0.06)),
            max_skip_steps=int(cache_options.get("max_skip_steps", 3)),
            retention_ratio=float(cache_options.get("retention_ratio", 0.2)),
            num_inference_steps=int(cache_options.get("num_inference_steps", 28)),
            mag_ratios=mag_ratios,
            calibrate=calibrate,
        )
    elif requested == "taylorseer":
        disable_after = cache_options.get("disable_cache_after_step")
        config = TaylorSeerCacheConfig(
            cache_interval=int(cache_options.get("cache_interval", 5)),
            disable_cache_before_step=int(cache_options.get("disable_cache_before_step", 3)),
            disable_cache_after_step=int(disable_after) if disable_after is not None else None,
            max_order=int(cache_options.get("max_order", 1)),
            use_lite_mode=bool(cache_options.get("use_lite_mode", False)),
        )
    elif requested == "pab":
        config = PyramidAttentionBroadcastConfig(
            spatial_attention_block_skip_range=cache_options.get("spatial_attention_block_skip_range", 2),
            temporal_attention_block_skip_range=cache_options.get("temporal_attention_block_skip_range"),
            cross_attention_block_skip_range=cache_options.get("cross_attention_block_skip_range"),
            spatial_attention_timestep_skip_range=tuple(
                cache_options.get("spatial_attention_timestep_skip_range", (100, 800))
            ),
            temporal_attention_timestep_skip_range=tuple(
                cache_options.get("temporal_attention_timestep_skip_range", (100, 800))
            ),
            cross_attention_timestep_skip_range=tuple(
                cache_options.get("cross_attention_timestep_skip_range", (100, 800))
            ),
            current_timestep_callback=lambda: getattr(pipeline, "current_timestep", None),
        )
    elif requested == "fastercache":
        config = FasterCacheConfig(
            spatial_attention_block_skip_range=int(cache_options.get("spatial_attention_block_skip_range", 2)),
            temporal_attention_block_skip_range=cache_options.get("temporal_attention_block_skip_range"),
            tensor_format=str(cache_options.get("tensor_format", "BCFHW")),
            is_guidance_distilled=bool(cache_options.get("is_guidance_distilled", False)),
            current_timestep_callback=lambda: getattr(pipeline, "current_timestep", None),
        )
    elif requested == "text_kv":
        config = TextKVCacheConfig()
    else:
        raise ValueError(f"Unsupported denoiser cache strategy {requested!r}.")

    applied = []
    unsupported = []
    for name, component in candidates:
        enable = getattr(component, "enable_cache", None)
        if not callable(enable):
            unsupported.append(name)
            continue
        try:
            if bool(getattr(component, "is_cache_enabled", False)):
                component.disable_cache()
            enable(config)
        except Exception as exc:
            raise RuntimeError(f"Could not enable {requested} cache on {name}: {exc}") from exc
        applied.append(name)
    if not applied:
        raise RuntimeError(f"{requested} cache is not supported by this pipeline's denoiser components.")
    return {
        "requested": requested,
        "config": type(config).__name__,
        "options": cache_options,
        "applied": applied,
        "unsupported": unsupported,
        "quality_warning": "Denoiser caching can improve speed but may reduce generation quality; validate each model, scheduler, and shape.",
    }


def configure_regional_compile(
    pipeline: Any,
    *,
    enabled: bool,
    components: Any = None,
    backend: str = "inductor",
    mode: str = "default",
    fullgraph: bool = False,
    dynamic: bool = False,
) -> dict[str, Any]:
    if not enabled:
        return {"requested": False, "applied": []}
    requested_components = _string_list(components) or ["transformer", "transformer_2", "unet", "prior"]
    applied = []
    unsupported = []
    for name in requested_components:
        component = getattr(pipeline, name, None)
        if component is None:
            continue
        compile_regions = getattr(component, "compile_repeated_blocks", None)
        if not callable(compile_regions):
            unsupported.append(name)
            continue
        try:
            compile_regions(
                backend=str(backend or "inductor"),
                mode=str(mode or "default"),
                fullgraph=bool(fullgraph),
                dynamic=bool(dynamic),
            )
        except Exception as exc:
            raise RuntimeError(f"Could not regionally compile {name}: {exc}") from exc
        applied.append(name)
    if not applied:
        raise RuntimeError("Regional compilation is not supported by the selected pipeline components.")
    return {
        "requested": True,
        "applied": applied,
        "unsupported": unsupported,
        "backend": str(backend or "inductor"),
        "mode": str(mode or "default"),
        "fullgraph": bool(fullgraph),
        "dynamic": bool(dynamic),
    }


def configure_layerwise_casting(
    pipeline: Any,
    *,
    enabled: bool,
    components: Any = None,
    storage_dtype: str = "float8_e4m3fn",
    compute_dtype: str = "bfloat16",
) -> dict[str, Any]:
    """Apply Diffusers' supported layerwise-casting hooks once per component.

    Layerwise casting mutates model weights and installs forward hooks. It
    cannot be safely changed in-place after a pipeline has been cached, so a
    changed signature is rejected with an actionable reload message instead of
    stacking a second set of hooks.
    """
    if not enabled:
        return {"requested": False, "applied": []}

    import torch
    from diffusers.hooks import apply_layerwise_casting

    storage = getattr(torch, str(storage_dtype or ""), None)
    compute = getattr(torch, str(compute_dtype or ""), None)
    if storage not in {
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    }:
        raise ValueError(
            "Layerwise casting storage dtype must be float8_e4m3fn or float8_e5m2 "
            "on a runtime that exposes that dtype."
        )
    if compute not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("Layerwise casting compute dtype must be float16, bfloat16, or float32.")

    requested_components = _string_list(components) or ["transformer", "transformer_2", "unet", "prior"]
    signature = {
        "storage_dtype": str(storage_dtype),
        "compute_dtype": str(compute_dtype),
    }
    applied = []
    already_applied = []
    unavailable = []
    for name in requested_components:
        component = getattr(pipeline, name, None)
        if component is None:
            continue
        current = getattr(component, "_modiff_layerwise_casting_signature", None)
        if current == signature:
            already_applied.append(name)
            continue
        if current is not None:
            raise RuntimeError(
                f"Layerwise casting for {name} is already configured differently. "
                "Reload the pipeline before changing its storage or compute dtype."
            )
        try:
            apply_layerwise_casting(
                component,
                storage_dtype=storage,
                compute_dtype=compute,
                skip_modules_pattern="auto",
                non_blocking=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Could not apply layerwise casting to {name}: {exc}") from exc
        component._modiff_layerwise_casting_signature = dict(signature)
        applied.append(name)
    if not applied and not already_applied:
        unavailable = requested_components
        raise RuntimeError(
            "Layerwise casting is not available because this pipeline has none of the selected denoiser components."
        )
    return {
        "requested": True,
        "applied": applied,
        "alreadyApplied": already_applied,
        "unavailable": unavailable,
        **signature,
    }


def configure_channels_last(
    pipeline: Any,
    *,
    enabled: bool,
    components: Any = None,
) -> dict[str, Any]:
    """Use channels-last only for explicitly selected convolutional modules."""
    if not enabled:
        return {"requested": False, "applied": []}

    import torch

    requested_components = _string_list(components) or ["unet", "vae"]
    applied = []
    already_applied = []
    unavailable = []
    for name in requested_components:
        component = getattr(pipeline, name, None)
        if component is None:
            unavailable.append(name)
            continue
        if bool(getattr(component, "_modiff_channels_last", False)):
            already_applied.append(name)
            continue
        move = getattr(component, "to", None)
        if not callable(move):
            unavailable.append(name)
            continue
        try:
            move(memory_format=torch.channels_last)
        except Exception as exc:
            raise RuntimeError(f"Could not apply channels-last layout to {name}: {exc}") from exc
        component._modiff_channels_last = True
        applied.append(name)
    if not applied and not already_applied:
        raise RuntimeError(
            "Channels-last layout is not available because this pipeline has none of the selected components."
        )
    return {
        "requested": True,
        "applied": applied,
        "alreadyApplied": already_applied,
        "unavailable": unavailable,
    }


def release_pipeline_memory(
    pipeline: Any,
    *,
    move_to_cpu: bool = True,
    disable_cache: bool = True,
    reset_compile_cache: bool = False,
) -> dict[str, Any]:
    released = []
    errors = []
    if disable_cache:
        for name in ("transformer", "transformer_2", "unet", "prior"):
            component = getattr(pipeline, name, None)
            disable = getattr(component, "disable_cache", None)
            if callable(disable) and bool(getattr(component, "is_cache_enabled", False)):
                try:
                    disable()
                    released.append(f"{name}_cache")
                except Exception as exc:
                    errors.append(f"{name} cache: {exc}")
    free_hooks = getattr(pipeline, "maybe_free_model_hooks", None)
    if callable(free_hooks):
        try:
            free_hooks()
            released.append("model_hooks")
        except Exception as exc:
            errors.append(f"model hooks: {exc}")
    reset_map = getattr(pipeline, "reset_device_map", None)
    if callable(reset_map) and getattr(pipeline, "hf_device_map", None):
        try:
            reset_map()
            released.append("device_map")
        except Exception as exc:
            errors.append(f"device map: {exc}")
    if move_to_cpu:
        move = getattr(pipeline, "to", None)
        if callable(move):
            try:
                move("cpu")
                released.append("pipeline_to_cpu")
            except Exception as exc:
                errors.append(f"pipeline to CPU: {exc}")

    try:
        import torch

        if reset_compile_cache:
            reset = getattr(getattr(torch, "_dynamo", None), "reset", None)
            if callable(reset):
                reset()
                released.append("torch_compile_cache")
        if bool(getattr(getattr(torch, "cuda", None), "is_available", lambda: False)()):
            torch.cuda.empty_cache()
            released.append("cuda_allocator_cache")
        elif bool(getattr(getattr(torch, "xpu", None), "is_available", lambda: False)()):
            empty = getattr(torch.xpu, "empty_cache", None)
            if callable(empty):
                empty()
                released.append("xpu_allocator_cache")
        elif bool(getattr(getattr(getattr(torch, "backends", None), "mps", None), "is_available", lambda: False)()):
            empty = getattr(getattr(torch, "mps", None), "empty_cache", None)
            if callable(empty):
                empty()
                released.append("mps_allocator_cache")
    except Exception as exc:
        errors.append(f"accelerator cache: {exc}")
    gc.collect()
    released.append("python_garbage_collection")
    return {"released": released, "errors": errors}


def _json_object(value: Any, label: str) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} must be valid JSON: {exc.msg}.") from exc
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"{label} must be a JSON object.")


def build_execution_recipe(
    *,
    quantization_config: Any = None,
    device_map: str = "none",
    device_map_overrides: Any = None,
    max_memory: Any = None,
    offload_mode: str = "none",
    device: str = "cuda:0",
    attention_backend: str = "auto",
    attention_components: Any = None,
    vae_slicing: bool = True,
    vae_tiling: bool = True,
    layerwise_casting: bool = False,
    layerwise_casting_components: Any = None,
    layerwise_storage_dtype: str = "float8_e4m3fn",
    layerwise_compute_dtype: str = "bfloat16",
    channels_last: bool = False,
    channels_last_components: Any = None,
    regional_compile: bool = False,
    compile_components: Any = None,
    compile_backend: str = "inductor",
    compile_mode: str = "default",
    compile_fullgraph: bool = False,
    compile_dynamic: bool = False,
    denoiser_cache: str = "none",
    cache_threshold: float = 0.05,
    cache_options: Any = None,
) -> dict[str, Any]:
    quantization_config = _normalize_optional_quantization_config(quantization_config)
    normalized_device_map = str(device_map or "none")
    if normalized_device_map not in {"none", "cuda", "auto", "balanced", "balanced_low_0", "cpu", "manual"}:
        raise ValueError(f"Unsupported device map strategy {normalized_device_map!r}.")
    memory = _json_object(max_memory, "Max memory")
    manual_map = _json_object(device_map_overrides, "Manual device map")
    if normalized_device_map == "manual" and not manual_map:
        raise ValueError("Manual device mapping needs at least one component placement entry.")
    normalized_offload = str(offload_mode or "none")
    normalized_attention_backend = str(attention_backend or "auto")
    if normalized_attention_backend not in ATTENTION_BACKENDS:
        raise ValueError(f"Unsupported attention backend {normalized_attention_backend!r}.")
    if quantization_config is not None and (
        normalized_offload != "none" or normalized_device_map not in {"none", "cuda"}
    ):
        raise ValueError(
            "Runtime quantization cannot use CPU/disk offload or a split device map. "
            "Use Create optimized copy, or select no offload on a GPU that can hold the unquantized source model."
        )
    normalized_memory = {}
    for key, value in memory.items():
        normalized_key = int(key) if isinstance(key, str) and key.isdigit() else key
        normalized_memory[normalized_key] = value
    return {
        "schema_version": 1,
        "quantization_config": quantization_config,
        "device_map": manual_map if normalized_device_map == "manual" else normalized_device_map,
        "device_map_strategy": normalized_device_map,
        "max_memory": normalized_memory,
        "offload_mode": normalized_offload,
        "device": str(device or "cuda:0"),
        "attention_backend": normalized_attention_backend,
        "attention_components": _string_list(attention_components),
        "vae_slicing": bool(vae_slicing),
        "vae_tiling": bool(vae_tiling),
        "layerwise_casting": bool(layerwise_casting),
        "layerwise_casting_components": _string_list(layerwise_casting_components),
        "layerwise_storage_dtype": str(layerwise_storage_dtype or "float8_e4m3fn"),
        "layerwise_compute_dtype": str(layerwise_compute_dtype or "bfloat16"),
        "channels_last": bool(channels_last),
        "channels_last_components": _string_list(channels_last_components),
        "regional_compile": bool(regional_compile),
        "compile_components": _string_list(compile_components),
        "compile_backend": str(compile_backend or "inductor"),
        "compile_mode": str(compile_mode or "default"),
        "compile_fullgraph": bool(compile_fullgraph),
        "compile_dynamic": bool(compile_dynamic),
        "denoiser_cache": str(denoiser_cache or "none"),
        "cache_threshold": float(cache_threshold),
        "cache_options": _json_object(cache_options, "Denoiser cache options"),
    }


def execution_recipe_summary(recipe: dict[str, Any]) -> dict[str, Any]:
    summary = dict(recipe)
    quant = summary.pop("quantization_config", None)
    mapping = getattr(quant, "quant_mapping", None)
    if isinstance(mapping, dict):
        summary["quantized_components"] = sorted(mapping)
    elif isinstance(quant, dict):
        summary["quantized_components"] = sorted(str(key) for key in quant)
    else:
        summary["quantized_components"] = []
    return summary


def loader_runtime_options(
    kwargs: dict[str, Any],
    *,
    default_device: str,
    default_offload_mode: str,
    direct_device_load: bool = False,
) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    """Resolve a connected recipe without breaking legacy loader controls.

    Complete Diffusers pipelines can be materialized directly on their final
    accelerator for a no-offload run.  This avoids constructing an entire
    CPU-resident copy and then migrating every component with ``pipeline.to``.
    Component-by-component loaders must leave ``direct_device_load`` disabled
    because they may need to assemble mixed-dtype CPU components first.
    """
    from modiff.diffusers_offload import normalize_offload_mode

    recipe = kwargs.get("execution_recipe") or {}
    if not isinstance(recipe, dict):
        raise TypeError("Execution Recipe must come from a Diffusers Execution Recipe node.")
    device = str(recipe.get("device") or kwargs.get("device") or default_device)
    recipe_offload = recipe.get("offload_mode")
    auto_offload = bool(kwargs.get("auto_offload", True))
    if recipe_offload is not None:
        auto_offload = str(recipe_offload) != "none"
    offload_mode = normalize_offload_mode(
        recipe_offload if recipe_offload is not None else kwargs.get("offload_mode") or default_offload_mode,
        auto_offload=auto_offload,
        device=device,
    )
    load_kwargs = {}
    quantization_config = recipe.get("quantization_config")
    if quantization_config is not None:
        assert_runtime_quantization_full_residency(
            model_id=_model_value(kwargs.get("model_id")),
            revision=str(kwargs.get("revision") or "").strip() or None,
            quantization_config=quantization_config,
            device=device,
            offload_mode=offload_mode,
            device_map=recipe.get("device_map") or "none",
        )
        load_kwargs["quantization_config"] = quantization_config
    device_map = recipe.get("device_map") or "none"
    if direct_device_load and offload_mode == "none" and str(device) in {"cuda", "cuda:0"}:
        if device_map == "none":
            device_map = "cuda"
        if device_map == "cuda":
            enable_parallel_weight_loading()
    if device_map != "none":
        load_kwargs["device_map"] = device_map
    max_memory = recipe.get("max_memory")
    if isinstance(max_memory, dict) and max_memory:
        load_kwargs["max_memory"] = max_memory
    return recipe, device, offload_mode, load_kwargs


def runtime_residency_reserve_bytes(total_vram: int | None = None) -> int:
    """Workspace and OS reserve used before any on-load quantization."""
    gib = 1024**3
    mib = 1024**2
    os_reserve = 600 * mib if os.name == "nt" else 400 * mib
    if os.name == "nt" and isinstance(total_vram, int) and total_vram > 15 * gib:
        os_reserve += 100 * mib
    return int(0.8 * gib) + os_reserve


def assert_runtime_quantization_full_residency(
    *,
    model_id: str,
    revision: str | None,
    quantization_config: Any,
    device: str,
    offload_mode: str,
    device_map: Any,
) -> dict[str, int]:
    """Admit expert on-load quantization only when the source fits entirely on one GPU."""
    if quantization_config is None:
        return {"source_weight_bytes": 0, "required_bytes": 0, "free_bytes": 0}
    if not model_id:
        raise ValueError("Runtime quantization needs a source model so full GPU residency can be checked.")
    if not str(device).startswith("cuda"):
        raise ValueError("Runtime quantization requires a CUDA/ROCm GPU. Use a pre-quantized artifact on this backend.")
    if str(offload_mode or "none") != "none" or device_map not in {None, "none", "cuda"}:
        raise ValueError(
            "Runtime quantization is blocked while CPU/disk offload or split placement is enabled. "
            "Use Create optimized copy instead."
        )

    inventory = inspect_diffusers_components(model_id, revision)
    source_bytes = int(inventory.get("total_weight_bytes") or 0)
    if source_bytes <= 0:
        raise ValueError(
            "MoDiff could not prove the source model's unquantized GPU residency from Safetensors metadata. "
            "Use the dedicated optimized-artifact workflow instead."
        )

    from modiff.config import CONFIG
    from modiff.hardware import get_hardware_snapshot

    hardware = get_hardware_snapshot(CONFIG.paths.get("data"), refresh=True)
    requested_index = 0
    try:
        requested_index = int(str(device).split(":", 1)[1])
    except (IndexError, ValueError):
        pass
    gpu = next(
        (
            item
            for item in hardware.get("devices", [])
            if item.get("type") == "cuda" and int(item.get("index") or 0) == requested_index
        ),
        None,
    )
    if not gpu:
        raise ValueError(f"Runtime quantization cannot find the requested accelerator {device}.")
    free_bytes = int(gpu.get("torch_vram_free") or gpu.get("vram_free") or 0)
    total_vram = int(gpu.get("torch_vram_total") or gpu.get("vram_total") or 0)
    reserve = runtime_residency_reserve_bytes(total_vram)
    required = source_bytes + reserve
    if free_bytes <= 0 or required > free_bytes:
        gib = 1024**3
        raise ValueError(
            f"Runtime quantization needs the full unquantized source plus workspace on {device}: "
            f"{required / gib:.1f} GiB required, {free_bytes / gib:.1f} GiB free. "
            "Use Create optimized copy on a larger GPU or install a qualified pre-quantized artifact."
        )
    return {"source_weight_bytes": source_bytes, "required_bytes": required, "free_bytes": free_bytes}


def apply_execution_recipe_to_pipeline(pipeline: Any, recipe: dict[str, Any]) -> dict[str, Any]:
    if not recipe:
        return {"attention": None, "vae": None}
    vae = configure_vae_memory(
        pipeline,
        slicing=bool(recipe.get("vae_slicing", True)),
        tiling=bool(recipe.get("vae_tiling", True)),
    )
    layerwise = configure_layerwise_casting(
        pipeline,
        enabled=bool(recipe.get("layerwise_casting", False)),
        components=recipe.get("layerwise_casting_components"),
        storage_dtype=recipe.get("layerwise_storage_dtype") or "float8_e4m3fn",
        compute_dtype=recipe.get("layerwise_compute_dtype") or "bfloat16",
    )
    channels_last = configure_channels_last(
        pipeline,
        enabled=bool(recipe.get("channels_last", False)),
        components=recipe.get("channels_last_components"),
    )
    attention = apply_attention_backend(
        pipeline,
        recipe.get("attention_backend") or "auto",
        recipe.get("attention_components"),
    )
    cache = configure_denoiser_cache(
        pipeline,
        strategy=recipe.get("denoiser_cache") or "none",
        threshold=float(recipe.get("cache_threshold", 0.05)),
        options=recipe.get("cache_options"),
    )
    compile_result = configure_regional_compile(
        pipeline,
        enabled=bool(recipe.get("regional_compile", False)),
        components=recipe.get("compile_components"),
        backend=recipe.get("compile_backend") or "inductor",
        mode=recipe.get("compile_mode") or "default",
        fullgraph=bool(recipe.get("compile_fullgraph", False)),
        dynamic=bool(recipe.get("compile_dynamic", False)),
    )
    return {
        "attention": attention,
        "vae": vae,
        "layerwiseCasting": layerwise,
        "channelsLast": channels_last,
        "cache": cache,
        "compile": compile_result,
    }


def build_runtime_capabilities(
    hardware: dict[str, Any],
    *,
    torch_module: Any,
    package_available=None,
) -> dict[str, Any]:
    """Build conservative runtime candidates from live backend/package facts."""
    if package_available is None:
        def package_available(name: str) -> bool:
            try:
                return importlib.util.find_spec(name) is not None
            except (ImportError, ModuleNotFoundError, ValueError):
                return False

    devices = list(hardware.get("devices") or [])
    accelerator = next((item for item in devices if item.get("type") != "cpu"), None)
    backend = str((accelerator or {}).get("type") or "cpu")
    torch_version = getattr(torch_module, "version", None)
    hip_version = getattr(torch_version, "hip", None)
    cuda_version = getattr(torch_version, "cuda", None)
    vendor = (
        "amd"
        if backend == "cuda" and hip_version
        else "nvidia"
        if backend == "cuda"
        else "apple"
        if backend == "mps"
        else "intel"
        if backend == "xpu"
        else "cpu"
    )

    capability = None
    if backend == "cuda":
        try:
            capability = tuple(int(item) for item in torch_module.cuda.get_device_capability(0))
        except Exception:
            capability = None

    bf16 = False
    if backend == "cuda":
        probe = getattr(torch_module.cuda, "is_bf16_supported", None)
        try:
            bf16 = bool(probe()) if callable(probe) else bool(capability and capability >= (8, 0))
        except Exception:
            bf16 = False
    elif backend == "mps":
        try:
            probe_tensor = torch_module.empty(
                (1,),
                dtype=torch_module.bfloat16,
                device="mps",
            )
            bf16 = probe_tensor is not None
            del probe_tensor
        except Exception:
            bf16 = False
    elif backend == "xpu":
        probe = getattr(getattr(torch_module, "xpu", None), "is_bf16_supported", None)
        try:
            bf16 = bool(probe()) if callable(probe) else False
        except Exception:
            bf16 = False
    elif backend == "cpu":
        cpu_probe = getattr(getattr(torch_module, "cpu", None), "_is_avx512_bf16_supported", None)
        try:
            bf16 = bool(cpu_probe()) if callable(cpu_probe) else False
        except Exception:
            bf16 = False

    flash_attn_available = backend == "cuda" and package_available("flash_attn")
    flash_attn_3_available = vendor == "nvidia" and package_available("flash_attn_interface")
    hub_kernels_available = vendor == "nvidia" and package_available("kernels")
    sage_available = backend == "cuda" and package_available("sageattention")
    xformers_available = vendor == "nvidia" and package_available("xformers")
    attention = {
        "native": {"available": True, "reason": "PyTorch SDPA fallback"},
        "_native_math": {"available": True, "reason": "Portable PyTorch math fallback"},
        "_native_flash": {
            "available": backend == "cuda",
            "reason": "PyTorch accelerator flash SDPA" if backend == "cuda" else "Requires a CUDA/ROCm torch backend",
        },
        "_native_efficient": {
            "available": backend == "cuda",
            "reason": "PyTorch memory-efficient SDPA" if backend == "cuda" else "Requires a CUDA/ROCm torch backend",
        },
        "_native_cudnn": {
            "available": vendor == "nvidia",
            "reason": "NVIDIA cuDNN attention" if vendor == "nvidia" else "Requires NVIDIA CUDA",
        },
        "flex": {
            "available": backend == "cuda" and hasattr(getattr(torch_module, "nn", None), "attention"),
            "reason": (
                "PyTorch FlexAttention is available"
                if backend == "cuda" and hasattr(getattr(torch_module, "nn", None), "attention")
                else "Requires a PyTorch accelerator build with FlexAttention"
            ),
        },
        "flash": {
            "available": flash_attn_available,
            "reason": (
                "flash-attn package detected"
                if flash_attn_available
                else "flash-attn package is not installed"
                if backend == "cuda"
                else "Requires a CUDA/ROCm torch backend"
            ),
        },
        "flash_varlen": {
            "available": flash_attn_available,
            "reason": "flash-attn variable-length kernels detected" if flash_attn_available else "Requires flash-attn",
        },
        "flash_hub": {
            "available": hub_kernels_available,
            "reason": "Hugging Face Hub kernels runtime detected" if hub_kernels_available else "Requires the kernels package on NVIDIA CUDA",
        },
        "flash_varlen_hub": {
            "available": hub_kernels_available,
            "reason": "Hugging Face variable-length kernels are available" if hub_kernels_available else "Requires the kernels package on NVIDIA CUDA",
        },
        "flash_4_hub": {
            "available": hub_kernels_available and bool(capability and capability >= (9, 0)),
            "reason": (
                "FlashAttention 4 Hub kernels are available on Hopper/Blackwell"
                if hub_kernels_available and capability and capability >= (9, 0)
                else "Requires Hub kernels and NVIDIA compute capability 9.0 or newer"
            ),
        },
        "_flash_3": {
            "available": flash_attn_3_available,
            "reason": "FlashAttention 3 interface detected" if flash_attn_3_available else "Requires the FlashAttention 3 interface on NVIDIA Hopper",
        },
        "_flash_varlen_3": {
            "available": flash_attn_3_available,
            "reason": "FlashAttention 3 variable-length interface detected" if flash_attn_3_available else "Requires the FlashAttention 3 interface on NVIDIA Hopper",
        },
        "_flash_3_hub": {
            "available": hub_kernels_available and bool(capability and capability >= (9, 0)),
            "reason": "FlashAttention 3 Hub kernels are available" if hub_kernels_available else "Requires Hub kernels on NVIDIA Hopper",
        },
        "_flash_3_varlen_hub": {
            "available": hub_kernels_available and bool(capability and capability >= (9, 0)),
            "reason": "FlashAttention 3 variable-length Hub kernels are available" if hub_kernels_available else "Requires Hub kernels on NVIDIA Hopper",
        },
        "sage": {
            "available": sage_available,
            "reason": (
                "SageAttention package detected"
                if sage_available
                else "SageAttention package is not installed"
                if backend == "cuda"
                else "Requires a CUDA/ROCm torch backend"
            ),
        },
        "sage_hub": {
            "available": hub_kernels_available,
            "reason": "SageAttention Hub kernels are available" if hub_kernels_available else "Requires the kernels package on NVIDIA CUDA",
        },
        "sage_varlen": {
            "available": sage_available,
            "reason": "SageAttention variable-length kernels detected" if sage_available else "Requires SageAttention",
        },
        "xformers": {
            "available": xformers_available,
            "reason": (
                "xFormers package detected"
                if xformers_available
                else "xFormers package is not installed"
                if vendor == "nvidia"
                else "MoDiff only enables xFormers on NVIDIA CUDA"
            ),
        },
    }

    torchao_installed = package_available("torchao")
    quanto_installed = package_available("optimum.quanto") or package_available("quanto")
    bitsandbytes_installed = package_available("bitsandbytes")
    torchao_fp8_device = vendor == "nvidia" and capability is not None and capability >= (8, 9)
    blackwell_device = vendor == "nvidia" and capability is not None and capability >= (10, 0)
    quantization = {
        "none": {
            "available": True,
            "reason": "No runtime quantization requested",
        },
        "bnb_4bit": {
            "available": bitsandbytes_installed and vendor == "nvidia",
            "reason": "bitsandbytes detected on NVIDIA CUDA"
            if vendor == "nvidia"
            else "MoDiff has not qualified this backend on the active platform",
        },
        "bnb_8bit": {
            "available": bitsandbytes_installed and vendor == "nvidia",
            "reason": "bitsandbytes detected on NVIDIA CUDA"
            if vendor == "nvidia"
            else "MoDiff has not qualified this backend on the active platform",
        },
        "quanto_float8": {
            "available": quanto_installed,
            "reason": "Quanto package detected" if quanto_installed else "Quanto package is not installed",
        },
        "quanto_int8": {
            "available": quanto_installed,
            "reason": "Quanto package detected" if quanto_installed else "Quanto package is not installed",
        },
        "torchao_int8_weight_only": {
            "available": torchao_installed,
            "reason": "TorchAO package detected" if torchao_installed else "TorchAO package is not installed",
        },
        "torchao_float8": {
            "available": torchao_installed and torchao_fp8_device,
            "reason": "TorchAO detected on an FP8-capable NVIDIA device"
            if torchao_fp8_device
            else "Requires TorchAO and a qualified NVIDIA compute capability of at least 8.9",
        },
        "torchao_mxfp8": {
            "available": torchao_installed and blackwell_device,
            "reason": "TorchAO MXFP8 detected on NVIDIA Blackwell"
            if blackwell_device
            else "Requires TorchAO and NVIDIA compute capability 10.0 or newer",
        },
        "torchao_nvfp4": {
            "available": torchao_installed and blackwell_device and package_available("mslk"),
            "reason": "TorchAO NVFP4 and MSLK detected on NVIDIA Blackwell"
            if blackwell_device
            else "Requires TorchAO, MSLK, and NVIDIA compute capability 10.0 or newer",
        },
    }

    mps_memory = None
    if backend == "mps":
        runtime = getattr(torch_module, "mps", None)
        mps_memory = {}
        for name in ("current_allocated_memory", "driver_allocated_memory", "recommended_max_memory"):
            probe = getattr(runtime, name, None)
            try:
                mps_memory[name] = int(probe()) if callable(probe) else None
            except Exception as exc:
                mps_memory[name] = None
                mps_memory[f"{name}_error"] = str(exc)

    system = hardware.get("system") if isinstance(hardware.get("system"), dict) else {}
    disk = hardware.get("disk") if isinstance(hardware.get("disk"), dict) else {}
    return {
        "backend": backend,
        "vendor": vendor,
        "device": (accelerator or {"device": "cpu:0"}).get("device"),
        "device_capability": list(capability) if capability else None,
        "torch_version": str(getattr(torch_module, "__version__", "unknown")),
        "cuda_version": str(cuda_version) if cuda_version else None,
        "hip_version": str(hip_version) if hip_version else None,
        "dtypes": {"float32": True, "float16": backend != "cpu", "bfloat16": bf16},
        "attention_backends": attention,
        "quantization_backends": quantization,
        "compile": {
            "available": callable(getattr(torch_module, "compile", None)),
            "regional_requires_model_probe": True,
        },
        "denoiser_cache": {
            name: {
                "available": package_available("diffusers"),
                "requires_model_probe": True,
                "quality_neutral": False,
            }
            for name in ("first_block", "magcache", "taylorseer", "pab", "fastercache", "text_kv")
        },
        "mps_memory": mps_memory,
        "memory": {
            "accelerator_total_bytes": (accelerator or {}).get("vram_total"),
            "accelerator_free_bytes": (accelerator or {}).get("vram_free"),
            "system_total_bytes": system.get("ram_total"),
            "system_available_bytes": system.get("ram_available"),
            "disk_free_bytes": disk.get("free_bytes"),
        },
    }


QUANTIZATION_NOMINAL_BITS = {
    "bnb_4bit": 4,
    "bnb_8bit": 8,
    "quanto_float8": 8,
    "torchao_float8": 8,
    "torchao_mxfp8": 8,
    "torchao_nvfp4": 4,
}


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def estimate_pipeline_memory(
    inventory: Any,
    *,
    width: int,
    height: int,
    frames: int = 1,
    batch_size: int = 1,
    dtype_bytes: int = 2,
    latent_channels: int = 16,
    spatial_compression: int = 8,
    temporal_compression: int = 4,
    quantization_summary: Any = None,
    offload_mode: str = "none",
) -> dict[str, Any]:
    """Return weight and latent floors without presenting a guessed peak VRAM value."""
    inventory = _json_object(inventory, "Component inventory")
    quantization = _json_object(quantization_summary, "Quantization summary")
    dimensions = {
        "width": max(1, int(width)),
        "height": max(1, int(height)),
        "frames": max(1, int(frames)),
        "batch_size": max(1, int(batch_size)),
        "dtype_bytes": max(1, int(dtype_bytes)),
        "latent_channels": max(1, int(latent_channels)),
        "spatial_compression": max(1, int(spatial_compression)),
        "temporal_compression": max(1, int(temporal_compression)),
    }
    latent_width = _ceil_div(dimensions["width"], dimensions["spatial_compression"])
    latent_height = _ceil_div(dimensions["height"], dimensions["spatial_compression"])
    latent_frames = _ceil_div(dimensions["frames"], dimensions["temporal_compression"])
    latent_elements = (
        dimensions["batch_size"] * dimensions["latent_channels"] * latent_frames * latent_height * latent_width
    )
    latent_bytes = latent_elements * dimensions["dtype_bytes"]

    component_rows = []
    unknown_dtypes = set()
    idealized_total = 0
    for item in inventory.get("components") or []:
        if not isinstance(item, dict):
            continue
        component = str(item.get("component") or "unknown")
        parameters = max(0, int(item.get("parameter_count") or 0))
        original_bytes = max(0, int(item.get("weight_bytes") or 0))
        for dtype in dict(item.get("dtype_counts") or {}):
            if dtype not in SAFETENSORS_DTYPE_BYTES:
                unknown_dtypes.add(str(dtype))
        policy = quantization.get(component) if isinstance(quantization.get(component), dict) else {}
        backend = str(policy.get("backend") or "none")
        nominal_bits = QUANTIZATION_NOMINAL_BITS.get(backend)
        idealized_bytes = _ceil_div(parameters * nominal_bits, 8) if nominal_bits else original_bytes
        exclusions = _string_list(policy.get("excluded_modules"))
        component_rows.append(
            {
                "component": component,
                "parameter_count": parameters,
                "unquantized_weight_floor_bytes": original_bytes,
                "quantization_backend": backend,
                "nominal_bits": nominal_bits,
                "idealized_weight_floor_bytes": idealized_bytes,
                "has_preserved_modules": bool(exclusions),
            }
        )
        idealized_total += idealized_bytes

    unquantized_total = max(0, int(inventory.get("total_weight_bytes") or 0))
    largest_component = max(component_rows, key=lambda item: item["idealized_weight_floor_bytes"], default=None)
    normalized_offload = str(offload_mode or "none")
    if normalized_offload == "none":
        residency_proxy = idealized_total
        residency_basis = "all component weight floors"
    else:
        residency_proxy = int((largest_component or {}).get("idealized_weight_floor_bytes") or 0)
        residency_basis = "largest component weight floor; runtime hook and layer granularity can differ"

    warnings = [
        "Peak accelerator memory is not estimated: attention, intermediate activations, allocator state, and backend workspaces require a measured run.",
        "Quantized weight floors exclude packing metadata, scales, preserved modules, and backend overhead.",
    ]
    if unknown_dtypes:
        warnings.append(
            f"Unknown Safetensors dtypes were excluded from byte totals: {', '.join(sorted(unknown_dtypes))}."
        )
    if not component_rows:
        warnings.append("No Safetensors component metadata was available for this model.")

    return {
        "schema_version": 1,
        "confidence": {
            "unquantized_weight_floor": "high" if component_rows and not unknown_dtypes else "limited",
            "idealized_quantized_weight_floor": "low" if quantization else "high",
            "latent_tensor_floor": "high for the explicit compression and channel assumptions",
            "peak_accelerator_memory": "unavailable until measured",
        },
        "shape": {
            **dimensions,
            "latent_width": latent_width,
            "latent_height": latent_height,
            "latent_frames": latent_frames,
            "latent_elements": latent_elements,
            "latent_tensor_bytes": latent_bytes,
        },
        "weights": {
            "unquantized_weight_floor_bytes": unquantized_total,
            "idealized_weight_floor_bytes": idealized_total,
            "accelerator_resident_weight_proxy_bytes": residency_proxy,
            "accelerator_residency_basis": residency_basis,
            "components": component_rows,
        },
        "offload_mode": normalized_offload,
        "peak_accelerator_memory_bytes": None,
        "warnings": warnings,
    }


def plan_execution_recipes(
    capabilities: Any,
    inventory: Any,
    *,
    preference: str = "balanced",
) -> dict[str, Any]:
    """Rank conservative recipes using facts available before a proof run.

    Weight residency is the only admission signal available here. Candidates
    therefore remain explicitly unproven until the executor records a real
    workload measurement for the same model, shape, and runtime fingerprint.
    """
    capabilities = _json_object(capabilities, "Hardware capabilities")
    inventory = _json_object(inventory, "Component inventory")
    requested = str(preference or "balanced")
    if requested not in {"quality", "balanced", "low_memory"}:
        raise ValueError(f"Unsupported recipe preference {requested!r}.")

    memory = capabilities.get("memory") if isinstance(capabilities.get("memory"), dict) else {}
    free_accelerator = memory.get("accelerator_free_bytes")
    free_accelerator = int(free_accelerator) if isinstance(free_accelerator, int) and free_accelerator > 0 else None
    backend = str(capabilities.get("backend") or "cpu")
    device = str(capabilities.get("device") or "cpu:0")
    dtypes = capabilities.get("dtypes") if isinstance(capabilities.get("dtypes"), dict) else {}
    dtype = "bfloat16" if dtypes.get("bfloat16") else "float16" if dtypes.get("float16") else "float32"
    inventory_components = {
        str(item.get("component"))
        for item in inventory.get("components") or []
        if isinstance(item, dict) and item.get("component")
    }
    quant_components = [
        name
        for name in ("transformer", "transformer_2", "text_encoder", "text_encoder_2")
        if name in inventory_components
    ]
    weight_floor = max(0, int(inventory.get("total_weight_bytes") or 0))

    # Runtime recipes never manufacture quantized weights. A selected artifact
    # may already be quantized, but the execution recipe only manages its
    # residency and attention policy.
    balanced_quant = "none"
    low_quant = "none"

    def idealized_weight(quant_backend: str) -> int:
        bits = QUANTIZATION_NOMINAL_BITS.get(quant_backend)
        return _ceil_div(weight_floor * bits, 16) if bits else weight_floor

    def fits(weight_bytes: int, fraction: float) -> bool | None:
        return (
            None if free_accelerator is None or weight_bytes <= 0 else weight_bytes <= int(free_accelerator * fraction)
        )

    quality_fits = fits(weight_floor, 0.75)
    balanced_fits = fits(idealized_weight(balanced_quant), 0.65)
    specs = {
        "quality": {
            "quantization_backend": "none",
            "offload_mode": "none" if quality_fits is True else "model_cpu",
            "tradeoff": "Preserves model precision; may require CPU movement when the weight floor lacks accelerator headroom.",
        },
        "balanced": {
            "quantization_backend": balanced_quant,
            "offload_mode": "none" if balanced_fits is True else "group_cpu",
            "tradeoff": "Balances runtime movement without changing model weights or enabling quality-changing denoiser caches.",
        },
        "low_memory": {
            "quantization_backend": low_quant,
            "offload_mode": "sequential_cpu" if backend != "cpu" else "none",
            "tradeoff": "Minimizes accelerator residency and accepts substantially longer generation time; use a pre-built artifact when further compression is required.",
        },
    }
    order = [requested, *[name for name in ("quality", "balanced", "low_memory") if name != requested]]
    candidates = []
    for rank, name in enumerate(order, start=1):
        spec = specs[name]
        quant_backend = spec["quantization_backend"]
        resident_floor = idealized_weight(quant_backend)
        if spec["offload_mode"] != "none":
            component_floors = [
                max(0, int(item.get("weight_bytes") or 0))
                for item in inventory.get("components") or []
                if isinstance(item, dict)
            ]
            resident_floor = max(component_floors, default=resident_floor)
        candidate_fits = fits(resident_floor, 0.65)
        candidates.append(
            {
                "rank": rank,
                "profile": name,
                "preference_match": name == requested,
                "device": device,
                "dtype": dtype,
                "quantization_backend": quant_backend,
                "quantized_components": quant_components if quant_backend != "none" else [],
                "offload_mode": spec["offload_mode"],
                "attention_backend": "auto",
                "vae_slicing": True,
                "vae_tiling": True,
                "regional_compile": False,
                "denoiser_cache": "none",
                "weight_residency_floor_bytes": resident_floor,
                "weight_floor_fits_known_free_memory": candidate_fits,
                "proof_status": "required",
                "tradeoff": spec["tradeoff"],
            }
        )
    warnings = [
        "These recipes are ranked from hardware and model metadata, not a successful generation.",
        "Activation, attention, allocator, compile, and backend workspace memory must be established by a measured proof run.",
    ]
    warnings.append(
        "Execution recipes keep original precision; use a qualified pre-built artifact when compression is required."
    )
    return {
        "schema_version": 1,
        "requested_preference": requested,
        "backend": backend,
        "device": device,
        "free_accelerator_bytes": free_accelerator,
        "unquantized_weight_floor_bytes": weight_floor,
        "candidates": candidates,
        "warnings": warnings,
    }


class PipelineQuantizationConfigV2(NodeBase):
    """Create a component-selective Diffusers quantization configuration."""

    label = "Pipeline Quantization Config V2"
    category = "Diffusers Runtime"
    resizable = True
    params = {
        "backend": {
            "label": "Backend",
            "type": "string",
            "options": [
                "none",
                "bnb_4bit",
                "bnb_8bit",
                "quanto_float8",
                "quanto_int8",
                "torchao_float8",
                "torchao_int8_weight_only",
            ],
            "default": "none",
        },
        "components": {
            "label": "Components",
            "type": "string",
            "display": "select",
            "options": QUANT_COMPONENTS,
            "fieldOptions": {"multiple": True},
            "default": ["transformer"],
        },
        "dtype": {
            "label": "Compute DType",
            "type": "string",
            "options": ["float32", "float16", "bfloat16"],
            "default": "bfloat16",
        },
        "excluded_modules": {
            "label": "Preserve Modules",
            "display": "textarea",
            "type": "text",
            "default": "",
            "description": "Comma-separated module paths or patterns to keep in their original precision.",
        },
        "component_overrides": {
            "label": "Per-component Overrides (JSON)",
            "display": "textarea",
            "type": "text",
            "default": "{}",
        },
        "quantization_config": {"label": "Quant Config", "display": "output", "type": "quantization_config"},
        "summary": {"label": "Summary", "display": "output", "type": "string"},
    }

    def execute(self, **kwargs):
        config, summary = build_quantization_config_v2(
            backend=kwargs.get("backend") or "none",
            components=kwargs.get("components"),
            dtype=str_to_dtype(kwargs.get("dtype") or "bfloat16"),
            excluded_modules=kwargs.get("excluded_modules"),
            component_overrides=kwargs.get("component_overrides"),
        )
        return {
            # Connected output sockets must carry a concrete value. A disabled
            # marker keeps the quantization flow visible and executable while
            # build_execution_recipe normalizes it back to no quantization.
            "quantization_config": config if config is not None else dict(NO_QUANTIZATION_CONFIG),
            "summary": json.dumps(summary, sort_keys=True),
        }


class DiffusersComponentInventory(NodeBase):
    """Inspect component weight floors from local or Hub Safetensors headers."""

    label = "Diffusers Component Inventory"
    category = "Diffusers Runtime"
    resizable = True
    params = {
        "model_id": {
            "label": "Model",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": ""},
            "fieldOptions": {"noValidation": True, "sources": ["hub", "local"]},
        },
        "revision": {"label": "Revision", "type": "string", "default": ""},
        "inventory": {"label": "Inventory", "display": "output", "type": "string"},
        "source_bytes": {"label": "Source Bytes", "display": "output", "type": "int"},
        "weight_bytes": {"label": "Weight Floor Bytes", "display": "output", "type": "int"},
        "parameter_count": {"label": "Parameters", "display": "output", "type": "int"},
        "largest_shard_bytes": {"label": "Largest Shard Bytes", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        model_id = _model_value(kwargs.get("model_id"))
        if not model_id:
            raise ValueError("Diffusers Component Inventory needs a model.")
        revision = str(kwargs.get("revision") or "").strip() or None
        self.progress(-1, phase="inventory", message="Reading model component metadata")
        inventory = inspect_diffusers_components(model_id, revision)
        largest = inventory.get("largest_shard") or {}
        return {
            "inventory": json.dumps(inventory, sort_keys=True),
            "source_bytes": int(inventory.get("total_source_bytes") or 0),
            "weight_bytes": int(inventory.get("total_weight_bytes") or 0),
            "parameter_count": int(inventory.get("total_parameter_count") or 0),
            "largest_shard_bytes": int(largest.get("bytes") or 0),
        }


class LoadPrequantizedDiffusersComponent(NodeBase):
    """Load a Diffusers-native or GGUF denoiser component without quantizing at runtime."""

    label = "Load Pre-quantized Diffusers Component"
    category = "Diffusers Runtime"
    resizable = True
    params = {
        "artifact": {
            "label": "Artifact",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": "city96/FLUX.1-schnell-gguf"},
            "fieldOptions": {"noValidation": True, "sources": ["hub", "local"]},
        },
        "filename": {"label": "GGUF Filename", "type": "string", "default": "flux1-schnell-Q4_0.gguf"},
        "revision": {
            "label": "Revision",
            "type": "string",
            "default": "f495746ed9c5efcf4661f53ef05401dceadc17d2",
        },
        "component_class": {
            "label": "Component Architecture",
            "type": "string",
            "options": [
                "FluxTransformer2DModel",
                "QwenImageTransformer2DModel",
                "WanTransformer3DModel",
                "LTXVideoTransformer3DModel",
            ],
            "default": "FluxTransformer2DModel",
        },
        "config_model": {
            "label": "Base Config",
            "type": "string",
            "default": "black-forest-labs/FLUX.1-schnell",
            "description": "Optional base repository used to validate the component architecture.",
        },
        "config_revision": {
            "label": "Base Config Revision",
            "type": "string",
            "default": "741f7c3ce8b383c54771c7003378a50191e9efe9",
            "description": "Optional immutable revision for the base configuration repository.",
        },
        "subfolder": {"label": "Config Subfolder", "type": "string", "default": "transformer"},
        "compute_dtype": {
            "label": "Compute DType",
            "type": "string",
            "options": ["float16", "bfloat16", "float32"],
            "default": "bfloat16",
        },
        "component": {"label": "Component", "display": "output", "type": "any"},
        "resolved_artifact": {"label": "Resolved Artifact", "display": "output", "type": "string"},
        "quantization": {"label": "Quantization", "display": "output", "type": "string"},
    }

    def execute(self, **kwargs):
        import diffusers

        artifact_selection = kwargs.get("artifact")
        artifact = _model_value(artifact_selection)
        filename = str(kwargs.get("filename") or "").strip()
        artifact_source = artifact_selection.get("source") if isinstance(artifact_selection, dict) else None
        artifact_pin = catalog_repository_pin(artifact) if artifact_source in {None, "hub"} else None
        file_contract = catalog_artifact_file(artifact, filename) if artifact_pin is not None else None
        if artifact_pin is not None and artifact_pin.get("format") == "gguf" and file_contract is None:
            raise ValueError("Choose an exact reviewed GGUF filename from the model artifact catalog.")
        revision = resolve_model_revision(
            artifact,
            kwargs.get("revision"),
            source=artifact_source,
        )
        class_name = str(kwargs.get("component_class") or "FluxTransformer2DModel")
        allowed = set(self.default_params["component_class"]["options"])
        if class_name not in allowed:
            raise ValueError(f"Unsupported component architecture {class_name!r}.")
        component_class = getattr(diffusers, class_name, None)
        if component_class is None or not callable(getattr(component_class, "from_single_file", None)):
            raise RuntimeError(f"Installed Diffusers does not expose GGUF loading for {class_name}.")
        if not artifact:
            raise ValueError("Choose a pre-quantized artifact.")
        if file_contract is not None and class_name != file_contract["componentClass"]:
            raise ValueError("The selected GGUF file does not match the reviewed component architecture.")

        source = Path(artifact).expanduser()
        if source.is_file():
            resolved = source
        else:
            if not filename.lower().endswith(".gguf"):
                raise ValueError("Hub GGUF artifacts require an explicit .gguf filename.")
            from huggingface_hub import hf_hub_download
            from modiff.config import CONFIG

            resolved = Path(
                hf_hub_download(
                    repo_id=artifact,
                    filename=filename,
                    revision=revision,
                    token=CONFIG.hf.get("token"),
                    cache_dir=CONFIG.hf.get("cache_dir"),
                )
            )
        if resolved.suffix.lower() != ".gguf":
            raise ValueError(f"Expected a GGUF artifact, got {resolved.name!r}.")
        if file_contract is not None:
            verify_cataloged_artifact_file(resolved, file_contract)

        from diffusers import GGUFQuantizationConfig

        dtype = str_to_dtype(kwargs.get("compute_dtype") or "bfloat16")
        load_kwargs: dict[str, Any] = {
            "quantization_config": GGUFQuantizationConfig(compute_dtype=dtype),
            "torch_dtype": dtype,
        }
        config_model = str(kwargs.get("config_model") or "").strip()
        subfolder = str(kwargs.get("subfolder") or "transformer").strip()
        if file_contract is not None:
            reviewed_config = str(file_contract["baseConfigRepo"])
            reviewed_subfolder = str(file_contract["subfolder"])
            reviewed_dtype = str(file_contract["computeDtype"])
            if config_model and config_model != reviewed_config:
                raise ValueError("The selected GGUF file requires its reviewed base configuration repository.")
            if subfolder != reviewed_subfolder or str(kwargs.get("compute_dtype") or "bfloat16") != reviewed_dtype:
                raise ValueError("The selected GGUF file requires its reviewed subfolder and compute dtype.")
            config_model = reviewed_config
        if config_model:
            load_kwargs["config"] = config_model
            config_revision = resolve_model_revision(config_model, kwargs.get("config_revision"))
            if file_contract is not None and config_revision != file_contract["baseConfigRevision"]:
                raise ValueError("The selected GGUF file requires its reviewed immutable base configuration revision.")
            if config_revision:
                load_kwargs["config_revision"] = config_revision
            if subfolder:
                load_kwargs["subfolder"] = subfolder
        self.progress(-1, phase="loading", message=f"Loading {class_name} GGUF artifact")
        try:
            component = component_class.from_single_file(str(resolved), **load_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"{resolved.name} does not match {class_name} or its selected base config: {exc}"
            ) from exc
        if file_contract is not None:
            component._modiff_prequantized_component_contract = {
                "schemaVersion": 1,
                "artifactRepo": artifact_pin["repo"],
                "artifactRevision": revision,
                "filename": filename,
                "sha256": file_contract["sha256"],
                "byteSize": file_contract["byteSize"],
                "componentClass": class_name,
                "baseConfigRepo": config_model,
                "baseConfigRevision": file_contract["baseConfigRevision"],
                "subfolder": subfolder,
                "computeDtype": str(kwargs.get("compute_dtype") or "bfloat16"),
            }
        return {
            "component": component,
            "resolved_artifact": (
                f"{artifact}@{revision or ('local' if source.is_file() else 'unversioned')}:"
                f"{filename or resolved.name}"
            ),
            "quantization": "GGUF",
        }


class DiffusersExecutionRecipe(NodeBase):
    """Combine load-time and runtime choices into one reusable pipeline recipe."""

    label = "Diffusers Execution Recipe"
    category = "Diffusers Runtime"
    resizable = True
    params = {
        "quantization_config": {"label": "Quant Config", "display": "input", "type": "quantization_config"},
        "device_map": {
            "label": "Device Placement",
            "type": "string",
            "options": ["none", "cuda", "auto", "balanced", "balanced_low_0", "cpu", "manual"],
            "default": "none",
        },
        "max_memory": {
            "label": "Max Memory (JSON)",
            "display": "textarea",
            "type": "text",
            "default": "{}",
            "description": 'Optional device limits, for example {"0": "16GiB", "cpu": "48GiB"}.',
        },
        "device_map_overrides": {
            "label": "Manual Device Map (JSON)",
            "display": "textarea",
            "type": "text",
            "default": "{}",
            "description": 'Used only with manual placement, for example {"transformer": 0, "text_encoder_2": "cpu"}.',
        },
        "offload_mode": {
            "label": "Runtime Offload",
            "type": "string",
            "options": ["none", "model_cpu", "sequential_cpu", "group_cpu", "group_disk"],
            "default": "none",
        },
        "device": {"label": "Execution Device", "type": "string", "default": "cuda:0"},
        "attention_backend": {
            "label": "Attention Backend",
            "type": "string",
            "options": ATTENTION_BACKENDS,
            "default": "auto",
        },
        "attention_components": {"label": "Attention Components", "type": "string", "default": ""},
        "vae_slicing": {"label": "VAE Slicing", "type": "bool", "default": True},
        "vae_tiling": {"label": "VAE Tiling", "type": "bool", "default": True},
        "layerwise_casting": {"label": "Layerwise Casting", "type": "bool", "default": False},
        "layerwise_casting_components": {
            "label": "Layerwise Casting Components",
            "type": "string",
            "default": "transformer",
        },
        "layerwise_storage_dtype": {
            "label": "Layerwise Storage DType",
            "type": "string",
            "options": ["float8_e4m3fn", "float8_e5m2"],
            "default": "float8_e4m3fn",
        },
        "layerwise_compute_dtype": {
            "label": "Layerwise Compute DType",
            "type": "string",
            "options": ["bfloat16", "float16", "float32"],
            "default": "bfloat16",
        },
        "channels_last": {"label": "Channels Last", "type": "bool", "default": False},
        "channels_last_components": {
            "label": "Channels Last Components",
            "type": "string",
            "default": "unet,vae",
        },
        "regional_compile": {"label": "Regional Compile", "type": "bool", "default": False},
        "compile_components": {"label": "Compile Components", "type": "string", "default": "transformer"},
        "compile_backend": {"label": "Compile Backend", "type": "string", "default": "inductor"},
        "compile_mode": {
            "label": "Compile Mode",
            "type": "string",
            "options": ["default", "reduce-overhead", "max-autotune"],
            "default": "default",
        },
        "compile_fullgraph": {"label": "Compile Full Graph", "type": "bool", "default": False},
        "compile_dynamic": {"label": "Compile Dynamic Shapes", "type": "bool", "default": False},
        "denoiser_cache": {
            "label": "Denoiser Cache",
            "type": "string",
            "options": ["none", "first_block", "magcache", "taylorseer", "pab", "fastercache", "text_kv"],
            "default": "none",
        },
        "cache_threshold": {"label": "Cache Threshold", "type": "float", "default": 0.05, "min": 0.0},
        "cache_options": {
            "label": "Cache Options (JSON)",
            "display": "textarea",
            "type": "text",
            "default": "{}",
        },
        "execution_recipe": {"label": "Execution Recipe", "display": "output", "type": "diffusers_execution_recipe"},
        "summary": {"label": "Summary", "display": "output", "type": "string"},
    }

    def execute(self, **kwargs):
        recipe = build_execution_recipe(
            quantization_config=kwargs.get("quantization_config"),
            device_map=kwargs.get("device_map") or "none",
            device_map_overrides=kwargs.get("device_map_overrides"),
            max_memory=kwargs.get("max_memory"),
            offload_mode=kwargs.get("offload_mode") or "none",
            device=kwargs.get("device") or "cuda:0",
            attention_backend=kwargs.get("attention_backend") or "auto",
            attention_components=kwargs.get("attention_components"),
            vae_slicing=bool(kwargs.get("vae_slicing", True)),
            vae_tiling=bool(kwargs.get("vae_tiling", True)),
            layerwise_casting=bool(kwargs.get("layerwise_casting", False)),
            layerwise_casting_components=kwargs.get("layerwise_casting_components"),
            layerwise_storage_dtype=kwargs.get("layerwise_storage_dtype") or "float8_e4m3fn",
            layerwise_compute_dtype=kwargs.get("layerwise_compute_dtype") or "bfloat16",
            channels_last=bool(kwargs.get("channels_last", False)),
            channels_last_components=kwargs.get("channels_last_components"),
            regional_compile=bool(kwargs.get("regional_compile", False)),
            compile_components=kwargs.get("compile_components"),
            compile_backend=kwargs.get("compile_backend") or "inductor",
            compile_mode=kwargs.get("compile_mode") or "default",
            compile_fullgraph=bool(kwargs.get("compile_fullgraph", False)),
            compile_dynamic=bool(kwargs.get("compile_dynamic", False)),
            denoiser_cache=kwargs.get("denoiser_cache") or "none",
            cache_threshold=float(kwargs.get("cache_threshold", 0.05)),
            cache_options=kwargs.get("cache_options"),
        )
        return {
            "execution_recipe": recipe,
            "summary": json.dumps(execution_recipe_summary(recipe), sort_keys=True),
        }


class ApplyPipelineRuntimeConfig(NodeBase):
    """Apply attention and VAE memory settings to any compatible pipeline."""

    label = "Apply Pipeline Runtime Config"
    category = "Diffusers Runtime"
    resizable = True
    params = {
        "pipeline": {
            "label": "Pipeline",
            "display": "input",
            "type": [
                "image_diffusion_pipeline",
                "video_diffusion_pipeline",
                "audio_diffusion_pipeline",
                "modular_pipeline",
                "any",
            ],
        },
        "attention_backend": {
            "label": "Attention Backend",
            "type": "string",
            "options": ATTENTION_BACKENDS,
            "default": "auto",
        },
        "attention_components": {
            "label": "Attention Components",
            "type": "string",
            "default": "",
            "description": "Optional comma-separated component names. Empty applies to compatible denoisers.",
        },
        "vae_slicing": {"label": "VAE Slicing", "type": "bool", "default": True},
        "vae_tiling": {"label": "VAE Tiling", "type": "bool", "default": True},
        "layerwise_casting": {"label": "Layerwise Casting", "type": "bool", "default": False},
        "layerwise_casting_components": {
            "label": "Layerwise Casting Components",
            "type": "string",
            "default": "transformer",
        },
        "layerwise_storage_dtype": {
            "label": "Layerwise Storage DType",
            "type": "string",
            "options": ["float8_e4m3fn", "float8_e5m2"],
            "default": "float8_e4m3fn",
        },
        "layerwise_compute_dtype": {
            "label": "Layerwise Compute DType",
            "type": "string",
            "options": ["bfloat16", "float16", "float32"],
            "default": "bfloat16",
        },
        "channels_last": {"label": "Channels Last", "type": "bool", "default": False},
        "channels_last_components": {
            "label": "Channels Last Components",
            "type": "string",
            "default": "unet,vae",
        },
        "regional_compile": {"label": "Regional Compile", "type": "bool", "default": False},
        "compile_components": {"label": "Compile Components", "type": "string", "default": "transformer"},
        "compile_backend": {"label": "Compile Backend", "type": "string", "default": "inductor"},
        "compile_mode": {
            "label": "Compile Mode",
            "type": "string",
            "options": ["default", "reduce-overhead", "max-autotune"],
            "default": "default",
        },
        "denoiser_cache": {
            "label": "Denoiser Cache",
            "type": "string",
            "options": ["none", "first_block", "magcache", "taylorseer", "pab", "fastercache", "text_kv"],
            "default": "none",
        },
        "cache_threshold": {"label": "Cache Threshold", "type": "float", "default": 0.05, "min": 0.0},
        "cache_options": {
            "label": "Cache Options (JSON)",
            "display": "textarea",
            "type": "text",
            "default": "{}",
        },
        "configured_pipeline": {"label": "Pipeline", "display": "output", "type": "any"},
        "summary": {"label": "Summary", "display": "output", "type": "string"},
    }

    def execute(self, **kwargs):
        pipeline = kwargs.get("pipeline")
        if pipeline is None:
            raise ValueError("Apply Pipeline Runtime Config needs a pipeline input.")
        attention = apply_attention_backend(
            pipeline,
            kwargs.get("attention_backend") or "auto",
            kwargs.get("attention_components"),
        )
        vae = configure_vae_memory(
            pipeline,
            slicing=bool(kwargs.get("vae_slicing", True)),
            tiling=bool(kwargs.get("vae_tiling", True)),
        )
        layerwise = configure_layerwise_casting(
            pipeline,
            enabled=bool(kwargs.get("layerwise_casting", False)),
            components=kwargs.get("layerwise_casting_components"),
            storage_dtype=kwargs.get("layerwise_storage_dtype") or "float8_e4m3fn",
            compute_dtype=kwargs.get("layerwise_compute_dtype") or "bfloat16",
        )
        channels_last = configure_channels_last(
            pipeline,
            enabled=bool(kwargs.get("channels_last", False)),
            components=kwargs.get("channels_last_components"),
        )
        cache = configure_denoiser_cache(
            pipeline,
            strategy=kwargs.get("denoiser_cache") or "none",
            threshold=float(kwargs.get("cache_threshold", 0.05)),
            options=kwargs.get("cache_options"),
        )
        compile_result = configure_regional_compile(
            pipeline,
            enabled=bool(kwargs.get("regional_compile", False)),
            components=kwargs.get("compile_components"),
            backend=kwargs.get("compile_backend") or "inductor",
            mode=kwargs.get("compile_mode") or "default",
        )
        return {
            "configured_pipeline": pipeline,
            "summary": json.dumps(
                {
                    "attention": attention,
                    "vae": vae,
                    "layerwiseCasting": layerwise,
                    "channelsLast": channels_last,
                    "cache": cache,
                    "compile": compile_result,
                },
                sort_keys=True,
            ),
        }


class PipelineMemoryEstimate(NodeBase):
    """Expose metadata-derived memory floors for a requested image/video shape."""

    label = "Pipeline Memory Estimate"
    category = "Diffusers Runtime"
    resizable = True
    params = {
        "inventory": {"label": "Component Inventory", "display": "input", "type": "string"},
        "quantization_summary": {
            "label": "Quantization Summary",
            "display": "input",
            "type": "string",
            "default": "{}",
        },
        "width": {"label": "Width", "type": "int", "default": 1024, "min": 1},
        "height": {"label": "Height", "type": "int", "default": 1024, "min": 1},
        "frames": {"label": "Frames", "type": "int", "default": 1, "min": 1},
        "batch_size": {"label": "Batch", "type": "int", "default": 1, "min": 1},
        "dtype_bytes": {"label": "Latent DType Bytes", "type": "int", "default": 2, "min": 1, "max": 8},
        "latent_channels": {"label": "Latent Channels", "type": "int", "default": 16, "min": 1},
        "spatial_compression": {"label": "Spatial Compression", "type": "int", "default": 8, "min": 1},
        "temporal_compression": {"label": "Temporal Compression", "type": "int", "default": 4, "min": 1},
        "offload_mode": {
            "label": "Runtime Offload",
            "type": "string",
            "options": ["none", "model_cpu", "sequential_cpu", "group_cpu", "group_disk"],
            "default": "none",
        },
        "estimate": {"label": "Estimate", "display": "output", "type": "string"},
        "weight_floor_bytes": {"label": "Weight Floor", "display": "output", "type": "int"},
        "resident_weight_proxy_bytes": {
            "label": "Resident Weight Proxy",
            "display": "output",
            "type": "int",
        },
        "latent_tensor_bytes": {"label": "Latent Tensor Floor", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        estimate = estimate_pipeline_memory(
            kwargs.get("inventory"),
            width=kwargs.get("width") or 1024,
            height=kwargs.get("height") or 1024,
            frames=kwargs.get("frames") or 1,
            batch_size=kwargs.get("batch_size") or 1,
            dtype_bytes=kwargs.get("dtype_bytes") or 2,
            latent_channels=kwargs.get("latent_channels") or 16,
            spatial_compression=kwargs.get("spatial_compression") or 8,
            temporal_compression=kwargs.get("temporal_compression") or 4,
            quantization_summary=kwargs.get("quantization_summary"),
            offload_mode=kwargs.get("offload_mode") or "none",
        )
        weights = estimate["weights"]
        return {
            "estimate": json.dumps(estimate, sort_keys=True),
            "weight_floor_bytes": weights["idealized_weight_floor_bytes"],
            "resident_weight_proxy_bytes": weights["accelerator_resident_weight_proxy_bytes"],
            "latent_tensor_bytes": estimate["shape"]["latent_tensor_bytes"],
        }


class ExecutionRecipePlanner(NodeBase):
    """Rank quality, balanced, and low-memory recipes for the active machine."""

    label = "Execution Recipe Planner"
    category = "Diffusers Runtime"
    resizable = True
    params = {
        "capabilities": {"label": "Hardware Capabilities", "display": "input", "type": "string"},
        "inventory": {"label": "Component Inventory", "display": "input", "type": "string"},
        "preference": {
            "label": "Preference",
            "type": "string",
            "options": ["quality", "balanced", "low_memory"],
            "default": "balanced",
        },
        "execution_recipe": {
            "label": "Recommended Recipe",
            "display": "output",
            "type": "diffusers_execution_recipe",
        },
        "recommendations": {"label": "Recommendations", "display": "output", "type": "string"},
        "selected_profile": {"label": "Selected Profile", "display": "output", "type": "string"},
    }

    def execute(self, **kwargs):
        plan = plan_execution_recipes(
            kwargs.get("capabilities"),
            kwargs.get("inventory"),
            preference=kwargs.get("preference") or "balanced",
        )
        selected = plan["candidates"][0]
        quantization_config = None
        quant_backend = selected["quantization_backend"]
        if quant_backend != "none":
            quantization_config, _summary = build_quantization_config_v2(
                backend=quant_backend,
                components=selected["quantized_components"],
                dtype=str_to_dtype(selected["dtype"]),
            )
        recipe = build_execution_recipe(
            quantization_config=quantization_config,
            offload_mode=selected["offload_mode"],
            device=selected["device"],
            attention_backend=selected["attention_backend"],
            vae_slicing=selected["vae_slicing"],
            vae_tiling=selected["vae_tiling"],
            regional_compile=False,
            denoiser_cache="none",
        )
        recipe["planner_profile"] = selected["profile"]
        recipe["proof_status"] = "required"
        return {
            "execution_recipe": recipe,
            "recommendations": json.dumps(plan, sort_keys=True),
            "selected_profile": selected["profile"],
        }


class ReleasePipelineMemory(NodeBase):
    """Move a pipeline out of accelerator memory and clear optional runtime state."""

    label = "Release Pipeline Memory"
    category = "Diffusers Runtime"
    params = {
        "pipeline": {
            "label": "Pipeline",
            "display": "input",
            "type": [
                "image_diffusion_pipeline",
                "video_diffusion_pipeline",
                "audio_diffusion_pipeline",
                "modular_pipeline",
                "any",
            ],
        },
        "move_to_cpu": {"label": "Move Pipeline to CPU", "type": "bool", "default": True},
        "disable_cache": {"label": "Disable Denoiser Cache", "type": "bool", "default": True},
        "reset_compile_cache": {"label": "Reset Compile Cache", "type": "bool", "default": False},
        "report": {"label": "Release Report", "display": "output", "type": "string"},
    }

    def execute(self, **kwargs):
        pipeline = kwargs.get("pipeline")
        if pipeline is None:
            raise ValueError("Release Pipeline Memory needs a pipeline input.")
        report = release_pipeline_memory(
            pipeline,
            move_to_cpu=bool(kwargs.get("move_to_cpu", True)),
            disable_cache=bool(kwargs.get("disable_cache", True)),
            reset_compile_cache=bool(kwargs.get("reset_compile_cache", False)),
        )
        return {"report": json.dumps(report, sort_keys=True)}


class HardwareCapabilityProbe(NodeBase):
    """Report runtime choices supported by the active hardware and packages."""

    label = "Hardware Capability Probe"
    category = "Diffusers Runtime"
    resizable = True
    params = {
        "capabilities": {"label": "Capabilities", "display": "output", "type": "string"},
        "backend": {"label": "Backend", "display": "output", "type": "string"},
        "device": {"label": "Device", "display": "output", "type": "string"},
    }

    def execute(self, **kwargs):
        import torch
        from modiff.config import CONFIG
        from modiff.hardware import get_hardware_snapshot

        hardware = get_hardware_snapshot(CONFIG.paths.get("data"), refresh=True)
        capabilities = build_runtime_capabilities(hardware, torch_module=torch)
        return {
            "capabilities": json.dumps(capabilities, sort_keys=True),
            "backend": capabilities["backend"],
            "device": capabilities["device"],
        }
