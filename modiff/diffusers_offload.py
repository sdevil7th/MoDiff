import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

logger = logging.getLogger("modiff")

OFFLOAD_MODE_NONE = "none"
OFFLOAD_MODE_MODEL_CPU = "model_cpu"
OFFLOAD_MODE_SEQUENTIAL_CPU = "sequential_cpu"
OFFLOAD_MODE_GROUP_CPU = "group_cpu"
OFFLOAD_MODE_GROUP_DISK = "group_disk"
OFFLOAD_MODE_AUTO_CPU = "auto_cpu"

OFFLOAD_MODE_OPTIONS = [
    OFFLOAD_MODE_NONE,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_SEQUENTIAL_CPU,
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
]
OFFLOAD_MODE_LEGACY_OPTIONS = [OFFLOAD_MODE_AUTO_CPU, *OFFLOAD_MODE_OPTIONS]

DEFAULT_GROUP_COMPONENTS = (
    "transformer",
    "unet",
    "text_encoder",
    "text_encoder_2",
    "text_encoder_3",
    "vae",
)

# Diffusers invokes VAE ``encode`` and ``decode`` directly. Block-level group
# offload installs the fallback hook on the component's top-level ``forward``,
# so unmatched VAE layers (for example Qwen Image's ``encoder.conv_in``) can
# remain on CPU while their inputs are prepared on CUDA. Leaf-level hooks are
# attached to the actual convolution/linear calls and cover both entry points.
LEAF_LEVEL_GROUP_COMPONENTS = frozenset({"vae"})


@dataclass
class OffloadResult:
    mode: str
    applied: bool
    method: str
    components: list[str]
    disk_path: str | None = None
    detail: str | None = None


def normalize_offload_mode(mode, auto_offload=True):
    if not auto_offload:
        return OFFLOAD_MODE_NONE
    if mode in (None, "", OFFLOAD_MODE_AUTO_CPU):
        return OFFLOAD_MODE_MODEL_CPU
    mode = str(mode)
    if mode in OFFLOAD_MODE_OPTIONS:
        return mode
    return OFFLOAD_MODE_MODEL_CPU


def offload_mode_param(default=OFFLOAD_MODE_MODEL_CPU, modes=None):
    mode_options = list(modes or OFFLOAD_MODE_OPTIONS)
    options = [OFFLOAD_MODE_AUTO_CPU, *mode_options] if OFFLOAD_MODE_MODEL_CPU in mode_options else mode_options
    return {
        "label": "Offload Mode",
        "type": "string",
        "options": options,
        "value": default,
        "default": default,
    }


def _disk_path(node_id, scope, component_name=None):
    path = Path("data") / "offload" / "diffusers" / str(node_id or "node") / str(scope or "pipeline")
    if component_name:
        path = path / str(component_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_group_offloaded(module):
    try:
        from diffusers.hooks.group_offloading import _is_group_offload_enabled

        return bool(_is_group_offload_enabled(module))
    except Exception:
        return False


def _apply_group_to_module(module, *, component_name, device, node_id, scope, mode, offload_type="block_level", num_blocks_per_group=2):
    if module is None or not isinstance(module, torch.nn.Module):
        return None
    if _is_group_offloaded(module):
        return component_name

    try:
        from diffusers.hooks import apply_group_offloading
    except Exception as exc:
        raise RuntimeError("Diffusers group offloading is not available in this backend.") from exc

    offload_to_disk_path = None
    if mode == OFFLOAD_MODE_GROUP_DISK:
        offload_to_disk_path = str(_disk_path(node_id, scope, component_name))

    try:
        apply_group_offloading(
            module,
            onload_device=torch.device(device),
            offload_device=torch.device("cpu"),
            offload_type=offload_type,
            num_blocks_per_group=num_blocks_per_group if offload_type == "block_level" else None,
            low_cpu_mem_usage=True,
            offload_to_disk_path=offload_to_disk_path,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not apply {mode} group offload to {component_name}: {exc}") from exc
    return component_name


def apply_component_group_offload(
    pipeline,
    *,
    component_names: Iterable[str] = DEFAULT_GROUP_COMPONENTS,
    device,
    mode,
    node_id,
    scope="pipeline",
):
    applied = []
    for component_name in component_names:
        offload_type = "leaf_level" if component_name in LEAF_LEVEL_GROUP_COMPONENTS else "block_level"
        applied_name = _apply_group_to_module(
            getattr(pipeline, component_name, None),
            component_name=component_name,
            device=device,
            node_id=node_id,
            scope=scope,
            mode=mode,
            offload_type=offload_type,
        )
        if applied_name:
            applied.append(applied_name)
    disk_path = str(_disk_path(node_id, scope)) if mode == OFFLOAD_MODE_GROUP_DISK and applied else None
    return OffloadResult(mode=mode, applied=bool(applied), method="component_group_offload", components=applied, disk_path=disk_path)


def apply_model_offload(model, *, component_name, mode, device, node_id, scope="model"):
    """Apply the same explicit execution-device policy to a standalone Diffusers model.

    Components loaded outside a pipeline (for example a ControlNet) do not
    inherit pipeline CPU/group hooks. Leaving them on CPU while a modular
    pipeline prepares CUDA inputs produces a late tensor-device mismatch.
    """
    mode = normalize_offload_mode(mode, auto_offload=mode != OFFLOAD_MODE_NONE)
    device_text = str(device)
    if not isinstance(model, torch.nn.Module):
        return OffloadResult(
            mode=mode,
            applied=False,
            method="not_torch_module",
            components=[],
            detail=f"{component_name} is not a torch.nn.Module.",
        )
    if mode == OFFLOAD_MODE_NONE or not device_text.startswith("cuda"):
        model.to(torch.device(device))
        return OffloadResult(mode=mode, applied=True, method="to_device", components=[component_name])

    effective_group_mode = mode
    if mode in (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU):
        effective_group_mode = OFFLOAD_MODE_GROUP_CPU
    applied_name = _apply_group_to_module(
        model,
        component_name=component_name,
        device=device,
        node_id=node_id,
        scope=scope,
        mode=effective_group_mode,
    )
    disk_path = (
        str(_disk_path(node_id, scope, component_name))
        if effective_group_mode == OFFLOAD_MODE_GROUP_DISK and applied_name
        else None
    )
    return OffloadResult(
        mode=mode,
        applied=bool(applied_name),
        method="standalone_group_offload",
        components=[applied_name] if applied_name else [],
        disk_path=disk_path,
        detail=f"Standalone {component_name} uses {effective_group_mode} hooks.",
    )


def apply_pipeline_offload(
    pipeline,
    *,
    mode,
    device,
    node_id,
    scope="pipeline",
    component_names: Iterable[str] = DEFAULT_GROUP_COMPONENTS,
    prefer_pipeline_group=True,
):
    mode = normalize_offload_mode(mode, auto_offload=mode != OFFLOAD_MODE_NONE)
    device_text = str(device)

    if mode == OFFLOAD_MODE_NONE:
        pipeline.to(torch.device(device))
        return OffloadResult(mode=mode, applied=True, method="to_device", components=[])

    if not device_text.startswith("cuda"):
        pipeline.to(torch.device(device))
        return OffloadResult(
            mode=mode,
            applied=False,
            method="to_device",
            components=[],
            detail=f"{mode} offload is CUDA-oriented; moved pipeline to {device_text}.",
        )

    if mode == OFFLOAD_MODE_MODEL_CPU:
        if not hasattr(pipeline, "enable_model_cpu_offload"):
            raise RuntimeError("This Diffusers pipeline does not expose enable_model_cpu_offload().")
        pipeline.enable_model_cpu_offload(device=device)
        return OffloadResult(mode=mode, applied=True, method="enable_model_cpu_offload", components=[])

    if mode == OFFLOAD_MODE_SEQUENTIAL_CPU:
        if not hasattr(pipeline, "enable_sequential_cpu_offload"):
            raise RuntimeError("This Diffusers pipeline does not expose enable_sequential_cpu_offload().")
        pipeline.enable_sequential_cpu_offload(device=device)
        return OffloadResult(mode=mode, applied=True, method="enable_sequential_cpu_offload", components=[])

    if mode in (OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK):
        disk_path = str(_disk_path(node_id, scope)) if mode == OFFLOAD_MODE_GROUP_DISK else None
        if prefer_pipeline_group and hasattr(pipeline, "enable_group_offload"):
            try:
                pipeline.enable_group_offload(
                    onload_device=torch.device(device),
                    offload_device=torch.device("cpu"),
                    offload_type="block_level",
                    num_blocks_per_group=2,
                    low_cpu_mem_usage=True,
                    offload_to_disk_path=disk_path,
                )
                return OffloadResult(mode=mode, applied=True, method="enable_group_offload", components=[], disk_path=disk_path)
            except Exception as exc:
                logger.warning("Pipeline group offload failed; trying component-level group offload: %s", exc)

        result = apply_component_group_offload(
            pipeline,
            component_names=component_names,
            device=device,
            mode=mode,
            node_id=node_id,
            scope=scope,
        )
        if not result.applied:
            raise RuntimeError("No compatible torch.nn.Module components were available for group offload.")
        return result

    raise RuntimeError(f"Unsupported Diffusers offload mode: {mode}")
