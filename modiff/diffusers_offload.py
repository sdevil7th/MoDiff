import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import Iterable

import torch

from modiff.diffusers_offload_modes import (
    OFFLOAD_MODE_AUTO_CPU,
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_NONE,
    OFFLOAD_MODE_OPTIONS,
    OFFLOAD_MODE_SEQUENTIAL_CPU,
)

logger = logging.getLogger("modiff")

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
MIN_STREAMING_RAM_BYTES = 8 * 1024**3
CPU_OFFLOAD_ACCELERATOR_TYPES = frozenset({"cuda"})


@dataclass
class OffloadResult:
    mode: str
    applied: bool
    method: str
    components: list[str]
    disk_path: str | None = None
    detail: str | None = None


def normalize_execution_device(device) -> torch.device:
    """Return one canonical torch device for residency and offload decisions.

    PyTorch exposes AMD ROCm accelerators through the ``cuda`` device type, so
    the CUDA branch intentionally covers both NVIDIA CUDA and AMD ROCm. Invalid
    device strings are rejected here before an upstream offload helper can
    install a partially configured hook.
    """
    try:
        normalized = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported execution device {device!r}.") from exc
    if normalized.type == "cuda" and normalized.index is None:
        return torch.device("cuda:0")
    return normalized


def supports_accelerator_cpu_offload(device) -> bool:
    """Whether Diffusers/Accelerate CPU-offload hooks may target ``device``."""
    return normalize_execution_device(device).type in CPU_OFFLOAD_ACCELERATOR_TYPES


def normalize_offload_mode(mode, auto_offload=True, device=None):
    if not auto_offload:
        return OFFLOAD_MODE_NONE
    if mode in (None, "", OFFLOAD_MODE_AUTO_CPU):
        normalized_mode = OFFLOAD_MODE_MODEL_CPU
    else:
        mode = str(mode)
        normalized_mode = mode if mode in OFFLOAD_MODE_OPTIONS else OFFLOAD_MODE_MODEL_CPU
    if (
        device is not None
        and normalized_mode != OFFLOAD_MODE_NONE
        and not supports_accelerator_cpu_offload(device)
    ):
        return OFFLOAD_MODE_NONE
    return normalized_mode


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


def _streaming_offload_allowed(device, mode):
    """Use pinned asynchronous transfers only on qualified accelerator paths."""
    if mode != OFFLOAD_MODE_GROUP_CPU or not supports_accelerator_cpu_offload(device):
        return False
    try:
        if torch.are_deterministic_algorithms_enabled():
            return False
        from modiff.hardware import system_memory_snapshot

        available = system_memory_snapshot().get("available_bytes")
        return isinstance(available, int) and available >= MIN_STREAMING_RAM_BYTES
    except Exception:
        return False


def _apply_group_to_module(module, *, component_name, device, node_id, scope, mode, offload_type="block_level", num_blocks_per_group=2):
    if module is None or not isinstance(module, torch.nn.Module):
        return None
    if _is_group_offloaded(module):
        return component_name

    # FLUX.2 pipelines read VAE BatchNorm statistics directly, without calling
    # BatchNorm.forward. Disk offload replaces its buffers with uninitialized
    # storage until that hook runs, corrupting latent normalization. Keep this
    # VAE on synchronous CPU leaf offload; the large denoiser/encoders retain
    # the requested disk strategy. No prompt/precision/generation values change.
    direct_vae_statistics = component_name == "vae" and isinstance(
        getattr(module, "bn", None), torch.nn.BatchNorm2d
    )
    if mode == OFFLOAD_MODE_GROUP_DISK and direct_vae_statistics:
        logger.info("Keeping VAE normalization buffers in CPU memory under group-disk offload.")
        mode = OFFLOAD_MODE_GROUP_CPU

    try:
        from diffusers.hooks import apply_group_offloading
    except Exception as exc:
        raise RuntimeError("Diffusers group offloading is not available in this backend.") from exc

    offload_to_disk_path = None
    if mode == OFFLOAD_MODE_GROUP_DISK:
        # Upstream names files by internal group name and reuses any existing
        # file. A node can load different weights later; component separation
        # alone cannot make those old files safe. Allocate storage per hook
        # owner/load, beneath the node's existing recursive cleanup boundary.
        offload_to_disk_path = mkdtemp(
            prefix="load-", dir=_disk_path(node_id, scope, component_name)
        )
    use_stream = False if direct_vae_statistics else _streaming_offload_allowed(device, mode)

    try:
        apply_group_offloading(
            module,
            onload_device=torch.device(device),
            offload_device=torch.device("cpu"),
            offload_type=offload_type,
            num_blocks_per_group=num_blocks_per_group if offload_type == "block_level" else None,
            low_cpu_mem_usage=True,
            offload_to_disk_path=offload_to_disk_path,
            non_blocking=use_stream,
            use_stream=use_stream,
            record_stream=False,
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
    requested_mode = normalize_offload_mode(mode, auto_offload=mode != OFFLOAD_MODE_NONE)
    normalized_device = normalize_execution_device(device)
    mode = normalize_offload_mode(requested_mode, auto_offload=True, device=normalized_device)
    if mode == OFFLOAD_MODE_NONE:
        moved = []
        for component_name in component_names:
            module = getattr(pipeline, component_name, None)
            if isinstance(module, torch.nn.Module):
                module.to(normalized_device)
                moved.append(component_name)
        return OffloadResult(
            mode=mode,
            applied=bool(moved),
            method="to_device",
            components=moved,
            detail=(
                f"{requested_mode} hooks require a CUDA/ROCm execution device; "
                f"moved components to {normalized_device} without offload hooks."
            ),
        )

    applied = []
    for component_name in component_names:
        offload_type = "leaf_level" if component_name in LEAF_LEVEL_GROUP_COMPONENTS else "block_level"
        applied_name = _apply_group_to_module(
            getattr(pipeline, component_name, None),
            component_name=component_name,
            device=normalized_device,
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
    requested_mode = normalize_offload_mode(mode, auto_offload=mode != OFFLOAD_MODE_NONE)
    normalized_device = normalize_execution_device(device)
    mode = normalize_offload_mode(requested_mode, auto_offload=True, device=normalized_device)
    if not isinstance(model, torch.nn.Module):
        return OffloadResult(
            mode=mode,
            applied=False,
            method="not_torch_module",
            components=[],
            detail=f"{component_name} is not a torch.nn.Module.",
        )
    if mode == OFFLOAD_MODE_NONE:
        model.to(normalized_device)
        return OffloadResult(
            mode=mode,
            applied=True,
            method="to_device",
            components=[component_name],
            detail=(
                f"{requested_mode} hooks require a CUDA/ROCm execution device; "
                f"moved {component_name} to {normalized_device} without offload hooks."
                if requested_mode != OFFLOAD_MODE_NONE
                else None
            ),
        )

    effective_group_mode = mode
    if mode in (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU):
        effective_group_mode = OFFLOAD_MODE_GROUP_CPU
    applied_name = _apply_group_to_module(
        model,
        component_name=component_name,
        device=normalized_device,
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


def configure_components_manager_offload(manager, *, mode, device):
    """Safely configure Modular Diffusers' shared ComponentsManager.

    ``ComponentsManager.enable_auto_cpu_offload`` queries accelerator free
    memory through ``mem_get_info``. Calling it with CPU or MPS therefore
    raises before a graph can load. Keep that upstream API behind the same
    execution-device invariant as pipeline/model offload.
    """
    requested_mode = normalize_offload_mode(mode, auto_offload=mode != OFFLOAD_MODE_NONE)
    normalized_device = normalize_execution_device(device)
    effective_mode = normalize_offload_mode(
        requested_mode,
        auto_offload=True,
        device=normalized_device,
    )
    enabled = bool(getattr(manager, "_auto_offload_enabled", False))

    if effective_mode == OFFLOAD_MODE_MODEL_CPU:
        configured_device = getattr(manager, "_auto_offload_device", None)
        configured_device = (
            normalize_execution_device(configured_device)
            if configured_device is not None
            else None
        )
        if not enabled or configured_device != normalized_device:
            manager.enable_auto_cpu_offload(device=normalized_device)
        return OffloadResult(
            mode=effective_mode,
            applied=True,
            method="components_manager_auto_cpu_offload",
            components=[],
        )

    if enabled:
        manager.disable_auto_cpu_offload()
    return OffloadResult(
        mode=effective_mode,
        applied=False,
        method="components_manager_no_offload",
        components=[],
        detail=(
            f"{requested_mode} hooks require a CUDA/ROCm execution device; "
            f"kept components on {normalized_device} without offload hooks."
            if requested_mode != OFFLOAD_MODE_NONE and effective_mode == OFFLOAD_MODE_NONE
            else None
        ),
    )


def reset_pipeline_device_map_for_runtime(pipeline):
    """Detach Accelerate placement before applying runtime movement/offload.

    Diffusers pipelines loaded with ``device_map`` cannot safely be passed to
    ``.to()``, model/sequential CPU offload, or group offload until their
    placement hooks are reset. Quantized loaders commonly use a device map, so
    centralize the transition here instead of relying on every loader to
    remember the ordering contract.
    """
    device_map = getattr(pipeline, "hf_device_map", None)
    if not device_map:
        return False
    reset = getattr(pipeline, "reset_device_map", None)
    if not callable(reset):
        raise RuntimeError(
            "This pipeline was loaded with a device map, but it cannot reset that placement before runtime offload. "
            "Load it without device_map or use a Diffusers pipeline that exposes reset_device_map()."
        )
    reset()
    return True


def _device_map_is_fully_on_target(pipeline, device):
    device_map = getattr(pipeline, "hf_device_map", None)
    target = torch.device(device)
    if target.type != "cuda":
        return False
    target_index = 0 if target.index is None else target.index

    # Diffusers intentionally preserves an explicit device-type strategy such
    # as ``device_map="cuda"`` as a string on the pipeline. It already means
    # every component was materialized on that accelerator, so resetting it
    # before a no-offload run would create a CPU copy and then migrate the
    # entire pipeline a second time.
    if isinstance(device_map, str):
        try:
            placement_device = torch.device(device_map)
        except (RuntimeError, TypeError):
            return False
        placement_index = 0 if placement_device.index is None else placement_device.index
        return placement_device.type == "cuda" and placement_index == target_index

    if not isinstance(device_map, dict) or not device_map:
        return False

    for placement in device_map.values():
        if isinstance(placement, int):
            placement_device = torch.device("cuda", placement)
        else:
            try:
                placement_device = torch.device(placement)
            except (RuntimeError, TypeError):
                return False
        placement_index = 0 if placement_device.index is None else placement_device.index
        if placement_device.type != "cuda" or placement_index != target_index:
            return False
    return True


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
    requested_mode = normalize_offload_mode(mode, auto_offload=mode != OFFLOAD_MODE_NONE)
    normalized_device = normalize_execution_device(device)
    mode = normalize_offload_mode(requested_mode, auto_offload=True, device=normalized_device)
    device_text = str(normalized_device)

    # A pipeline streamed directly to one accelerator already satisfies the
    # no-offload contract. Resetting that map first can materialize a second
    # CPU copy before ``pipeline.to()``, which is fatal on unified-memory hosts
    # even when the resident model itself fits.
    if mode == OFFLOAD_MODE_NONE and _device_map_is_fully_on_target(pipeline, normalized_device):
        return OffloadResult(
            mode=mode,
            applied=True,
            method="preserve_device_map",
            components=[],
            detail=f"Pipeline is already fully resident on {device_text}.",
        )

    reset_pipeline_device_map_for_runtime(pipeline)

    if mode == OFFLOAD_MODE_NONE:
        pipeline.to(normalized_device)
        return OffloadResult(
            mode=mode,
            applied=True,
            method="to_device",
            components=[],
            detail=(
                f"{requested_mode} hooks require a CUDA/ROCm execution device; "
                f"moved pipeline to {device_text} without offload hooks."
                if requested_mode != OFFLOAD_MODE_NONE
                else None
            ),
        )

    if mode == OFFLOAD_MODE_MODEL_CPU:
        if not hasattr(pipeline, "enable_model_cpu_offload"):
            raise RuntimeError("This Diffusers pipeline does not expose enable_model_cpu_offload().")
        pipeline.enable_model_cpu_offload(device=normalized_device)
        return OffloadResult(mode=mode, applied=True, method="enable_model_cpu_offload", components=[])

    if mode == OFFLOAD_MODE_SEQUENTIAL_CPU:
        if not hasattr(pipeline, "enable_sequential_cpu_offload"):
            raise RuntimeError("This Diffusers pipeline does not expose enable_sequential_cpu_offload().")
        pipeline.enable_sequential_cpu_offload(device=normalized_device)
        return OffloadResult(mode=mode, applied=True, method="enable_sequential_cpu_offload", components=[])

    if mode in (OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK):
        disk_path = str(_disk_path(node_id, scope)) if mode == OFFLOAD_MODE_GROUP_DISK else None
        use_stream = _streaming_offload_allowed(normalized_device, mode)
        # The pipeline-wide upstream helper shares one disk directory across
        # components. Transformer/ControlNet group names collide, silently
        # substituting equal-shaped weights (or failing for unequal shapes).
        # Enumerate genuine owned components, including auxiliary encoders and
        # ControlNets not present in our default component-name fallback.
        component_names = tuple(component_names)
        if mode == OFFLOAD_MODE_GROUP_DISK:
            owned_components = getattr(pipeline, "components", {})
            if isinstance(owned_components, dict):
                component_names = tuple(dict.fromkeys((*component_names, *(
                    name for name, module in owned_components.items()
                    if isinstance(module, torch.nn.Module)
                ))))
        if mode != OFFLOAD_MODE_GROUP_DISK and prefer_pipeline_group and hasattr(pipeline, "enable_group_offload"):
            try:
                leaf_components = sorted(name for name in LEAF_LEVEL_GROUP_COMPONENTS
                                         if isinstance(getattr(pipeline, name, None), torch.nn.Module))
                pipeline.enable_group_offload(
                    onload_device=normalized_device,
                    offload_device=torch.device("cpu"),
                    offload_type="block_level",
                    num_blocks_per_group=2,
                    low_cpu_mem_usage=True,
                    offload_to_disk_path=disk_path,
                    non_blocking=use_stream,
                    use_stream=use_stream,
                    record_stream=False,
                    exclude_modules=leaf_components,
                )
                # The pipeline-wide block-level helper hooks `forward`, but
                # pipelines call VAE encode/decode directly. Exclude those
                # components from that helper and install the same leaf-level
                # policy used by our component fallback. All other components
                # (including auxiliary ControlNet/image encoders) stay owned by
                # the upstream pipeline helper.
                if leaf_components:
                    apply_component_group_offload(
                        pipeline, component_names=leaf_components,
                        device=normalized_device, mode=mode, node_id=node_id, scope=scope,
                    )
                detail = "Pinned asynchronous prefetch enabled." if use_stream else "Synchronous group transfers."
                return OffloadResult(
                    mode=mode,
                    applied=True,
                    method="enable_group_offload",
                    components=[],
                    disk_path=disk_path,
                    detail=detail,
                )
            except Exception as exc:
                logger.warning("Pipeline group offload failed; trying component-level group offload: %s", exc)

        result = apply_component_group_offload(
            pipeline,
            component_names=component_names,
            device=normalized_device,
            mode=mode,
            node_id=node_id,
            scope=scope,
        )
        if not result.applied:
            raise RuntimeError("No compatible torch.nn.Module components were available for group offload.")
        return result

    raise RuntimeError(f"Unsupported Diffusers offload mode: {mode}")
