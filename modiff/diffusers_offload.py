import logging
import hashlib
import os
import shutil
import stat
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
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
_MAX_STREAMING_MODULES = 65536
_MAX_STREAMING_TENSOR_REFERENCES = 262144
CPU_OFFLOAD_ACCELERATOR_TYPES = frozenset({"cuda"})
_DISK_LEASE_LOCK = threading.RLock()
_DISK_LEASES = weakref.WeakSet()
_DIRECTORY_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


@dataclass(frozen=True)
class _OwnedDiskDirectory:
    root: Path
    path: Path
    ancestors: tuple[tuple[Path, tuple[int, int, int | None]], ...]


@dataclass(frozen=True, eq=False)
class _DiskDirectoryLease:
    """Shared by hook users, never a strong reference to models or hooks."""

    directory: _OwnedDiskDirectory

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self


def _directory_identity(path):
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & _DIRECTORY_REPARSE_POINT):
        raise ValueError("Diffusers offload directories must not contain symlinks or junctions.")
    if not info.st_dev or not info.st_ino:
        raise ValueError("Diffusers offload directory identity is unavailable.")
    return info.st_dev, info.st_ino, getattr(info, "st_birthtime_ns", None)


def _offload_root():
    # Resolve the explicit working directory once, then reject indirection in
    # the dedicated cache hierarchy rather than following graph-supplied paths.
    return Path.cwd().resolve(strict=True) / "data" / "offload" / "diffusers"


def _explicit_offload_root(value):
    if value is None:
        return _offload_root()
    root = Path(value)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("Selected offload storage requires an explicit absolute plain directory.")
    # Inspect the lexical chain before creation, never resolve away a junction.
    current = Path(root.anchor)
    _directory_identity(current)
    missing = False
    for part in root.parts[1:]:
        current /= part
        if not missing:
            try:
                _directory_identity(current)
            except FileNotFoundError:
                missing = True
    return root


def _directory_chain(root, path, *, create=False):
    if not root.is_absolute() or not path.is_absolute():
        raise ValueError("Diffusers offload directory paths must be absolute.")
    path.relative_to(root)
    anchor = root.parent.parent.parent
    current = anchor
    chain = [(current, _directory_identity(current))]
    for part in path.relative_to(anchor).parts:
        current = current / part
        if create:
            try:
                current.mkdir()
            except FileExistsError:
                pass
        chain.append((current, _directory_identity(current)))
    if path.resolve(strict=True) != path:
        raise ValueError("Diffusers offload directory escapes its owned cache root.")
    return tuple(chain)


def _validate_owned_disk_directory(directory):
    if directory.path == directory.root:
        raise ValueError("A per-load offload lease cannot remove the cache root.")
    directory.path.relative_to(directory.root)
    current = _directory_chain(directory.root, directory.path)
    if current != directory.ancestors:
        raise ValueError("Diffusers offload directory identity changed; storage was preserved.")


def _remove_owned_disk_directory(directory):
    with _DISK_LEASE_LOCK:
        try:
            try:
                directory.path.lstat()
            except FileNotFoundError:
                return
            _validate_owned_disk_directory(directory)
            shutil.rmtree(directory.path)
        except Exception:
            # Cleanup cannot replace the original loading/cancellation error.
            # A later free-disk check still observes any preserved bytes.
            logger.warning("Could not release owned Diffusers disk offload directory %s",
                           directory.path, exc_info=True)


def _retain_disk_lease(owner, lease):
    leases = getattr(owner, "_modiff_disk_offload_leases", ())
    if lease not in leases:
        owner._modiff_disk_offload_leases = (*leases, lease)


def _bind_disk_hook_lease(module, lease):
    from diffusers.hooks.group_offloading import GroupOffloadingHook

    for child in module.modules():
        _retain_disk_lease(child, lease)
        for hook in getattr(getattr(child, "_diffusers_hook", None), "hooks", {}).values():
            if (isinstance(hook, GroupOffloadingHook)
                    and hook.config.offload_to_disk_path == str(lease.directory.path)):
                # Official PEFT reapplication retains this same config and may
                # create new hook modules not present in the original tree.
                for owner in (hook, hook.group, hook.config):
                    _retain_disk_lease(owner, lease)


def _new_disk_offload_lease(module, *, node_id, scope, component_name, offload_root=None):
    with _DISK_LEASE_LOCK:
        root = _explicit_offload_root(offload_root)
        parent = _disk_path(node_id, scope, component_name, offload_root=offload_root)
        path = Path(mkdtemp(prefix="load-", dir=parent))
        directory = _OwnedDiskDirectory(root, path, _directory_chain(root, path))
        lease = _DiskDirectoryLease(directory)
        finalizer = weakref.finalize(lease, _remove_owned_disk_directory, directory)
        # A live hook remains authoritative even during interpreter shutdown.
        # Process termination can leave remnants for explicit global cleanup.
        finalizer.atexit = False
        _DISK_LEASES.add(lease)
        for child in module.modules():
            _retain_disk_lease(child, lease)
        return lease


def release_disk_offload_cache():
    """Explicit global cleanup after ownership release, preserving live leases."""
    with _DISK_LEASE_LOCK:
        root = _offload_root()
        try:
            root.lstat()
        except FileNotFoundError:
            return 0, []
        try:
            original = _directory_chain(root, root)
            if any(lease.directory.root == root for lease in _DISK_LEASES):
                return 0, ["Diffusers disk offload cache: live hook storage was preserved."]
            count = 0
            for parent, directories, files in os.walk(root, followlinks=False):
                # Windows junctions are directory reparse points; do not walk
                # them when counting cache entries before native rmtree.
                directories[:] = [name for name in directories
                    if not (getattr(Path(parent, name).lstat(), "st_file_attributes", 0)
                            & _DIRECTORY_REPARSE_POINT)]
                count += len(files)
            if _directory_chain(root, root) != original:
                raise ValueError("Diffusers offload cache root identity changed.")
            shutil.rmtree(root)
            return count, []
        except Exception as exc:
            logger.debug("Diffusers disk offload cache cleanup failed", exc_info=True)
            return 0, [f"Diffusers disk offload cache: {exc}"]


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


def _owner_disk_segment(node_id):
    # Graph IDs are opaque (including colon-rich Block V2 projections), never
    # filesystem paths. A bounded digest preserves their exact distinct identity.
    return "owner-" + hashlib.sha256(str(node_id or "node").encode("utf-8")).hexdigest()


def _disk_path(node_id, scope, component_name=None, *, offload_root=None):
    parts = [_owner_disk_segment(node_id), str(scope or "pipeline")]
    if component_name is not None:
        parts.append(str(component_name))
    for part in parts:
        if (not part or len(part) > 255 or part in {".", ".."}
                or any(char in '<>:"/\\|?*' or ord(char) < 32 for char in part)
                or part.rstrip(" .") != part or PureWindowsPath(part).is_reserved()):
            raise ValueError("Diffusers offload directory segments must be single local names.")
    root = _explicit_offload_root(offload_root)
    path = root.joinpath(*parts)
    _directory_chain(root, path, create=True)
    return path


def _is_group_offloaded(module):
    try:
        from diffusers.hooks.group_offloading import _is_group_offload_enabled

        return bool(_is_group_offload_enabled(module))
    except Exception:
        return False


def _streaming_component_bytes(module):
    """Bound registered native storage without invoking model traversal hooks.

    Count shared physical storage once, but also retain the logical bytes of
    every registered reference: upstream may pin separate views independently.
    Unknown/subclass/placeholder storage selects synchronous offload instead.
    This is an execution-time transfer policy, not an Auto capacity estimate.
    """
    pending, seen_modules, storages = [module], set(), set()
    physical_bytes = logical_bytes = references = 0
    while pending:
        current = pending.pop()
        if not isinstance(current, torch.nn.Module):
            return None
        if id(current) in seen_modules:
            continue
        seen_modules.add(id(current))
        if len(seen_modules) > _MAX_STREAMING_MODULES:
            return None
        state = vars(current)
        children, parameters, buffers = (state.get(name) for name in ("_modules", "_parameters", "_buffers"))
        if any(type(mapping) is not dict for mapping in (children, parameters, buffers)):
            return None
        if len(pending) + len(children) > _MAX_STREAMING_MODULES:
            return None
        pending.extend(child for child in children.values() if child is not None)
        for mapping in (parameters, buffers):
            for tensor in mapping.values():
                if tensor is None:
                    continue
                references += 1
                if references > _MAX_STREAMING_TENSOR_REFERENCES:
                    return None
                if (type(tensor) not in (torch.Tensor, torch.nn.Parameter)
                        or tensor.layout != torch.strided or tensor.device.type not in ("cpu", "cuda")
                        or tensor.is_quantized):
                    return None
                storage = tensor.untyped_storage()
                size = storage.nbytes()
                logical = tensor.numel() * tensor.element_size()
                if any(type(value) is not int or not 0 <= value < 2**63 for value in (size, logical)):
                    return None
                key = (tensor.device, storage.data_ptr(), size)
                if key not in storages:
                    storages.add(key)
                    physical_bytes += size
                logical_bytes += logical
                if max(physical_bytes, logical_bytes) >= 2**63:
                    return None
    return max(physical_bytes, logical_bytes) if references else None


def _streaming_offload_allowed(device, mode, module=None):
    """Retain the free-RAM floor after conservative streamed host staging."""
    if mode != OFFLOAD_MODE_GROUP_CPU or not supports_accelerator_cpu_offload(device):
        return False
    try:
        if torch.are_deterministic_algorithms_enabled():
            return False
        from modiff.hardware import system_memory_snapshot

        available = system_memory_snapshot().get("available_bytes")
        if type(available) is not int or available < MIN_STREAMING_RAM_BYTES:
            return False
        component_bytes = _streaming_component_bytes(module)
        # Official streamed group hooks retain CPU source tensors and create
        # pinned transfer material. Reserve two component copies even when
        # only part of the component is in flight; keep the existing RAM floor
        # free after that reservation. Synchronous hooks avoid this staging.
        return (component_bytes is not None
                and available >= MIN_STREAMING_RAM_BYTES + 2 * component_bytes)
    except Exception:
        return False


def _apply_group_to_module(module, *, component_name, device, node_id, scope, mode, offload_type="block_level", num_blocks_per_group=2, allow_streaming=True, offload_root=None):
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
    disk_lease = None
    if mode == OFFLOAD_MODE_GROUP_DISK:
        # Upstream names files by internal group name and reuses any existing
        # file. A node can load different weights later; component separation
        # alone cannot make those old files safe. Allocate storage per hook
        # owner/load, held until every associated module/hook user is released.
        disk_lease = _new_disk_offload_lease(module, node_id=node_id, scope=scope,
                                          component_name=component_name, offload_root=offload_root)
        offload_to_disk_path = str(disk_lease.directory.path)
    use_stream = False if direct_vae_statistics or not allow_streaming else _streaming_offload_allowed(device, mode, module)

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
    finally:
        if disk_lease is not None:
            _bind_disk_hook_lease(module, disk_lease)
    return component_name


def apply_component_group_offload(
    pipeline,
    *,
    component_names: Iterable[str] = DEFAULT_GROUP_COMPONENTS,
    device,
    mode,
    node_id,
    scope="pipeline",
    leaf_level_components: Iterable[str] = LEAF_LEVEL_GROUP_COMPONENTS,
    allow_streaming: bool = True,
    offload_root=None,
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
        offload_type = "leaf_level" if component_name in leaf_level_components else "block_level"
        applied_name = _apply_group_to_module(
            getattr(pipeline, component_name, None),
            component_name=component_name,
            device=normalized_device,
            node_id=node_id,
            scope=scope,
            mode=mode,
            offload_type=offload_type,
            allow_streaming=allow_streaming,
            offload_root=offload_root,
        )
        if applied_name:
            applied.append(applied_name)
    disk_path = str(_disk_path(node_id, scope, offload_root=offload_root)) if mode == OFFLOAD_MODE_GROUP_DISK and applied else None
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
    leaf_level_components: Iterable[str] = LEAF_LEVEL_GROUP_COMPONENTS,
    resident_components: Iterable[str] = (),
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

    # Some reviewed pipelines read a component's tensors directly, outside its
    # forward hook. Preserve those components explicitly rather than exposing
    # meta tensors or disk placeholders to the pipeline. This is per-instance;
    # never mutate Diffusers' class-level exclusion list.
    resident_components = tuple(resident_components)
    if resident_components:
        pipeline._exclude_from_cpu_offload = list(dict.fromkeys((
            *getattr(pipeline, "_exclude_from_cpu_offload", ()), *resident_components,
        )))

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
        component_names = tuple(name for name in component_names if name not in resident_components)
        for name in resident_components:
            module = getattr(pipeline, name, None)
            if isinstance(module, torch.nn.Module):
                module.to(normalized_device)
        if mode != OFFLOAD_MODE_GROUP_DISK and prefer_pipeline_group and hasattr(pipeline, "enable_group_offload"):
            try:
                leaf_components = sorted(name for name in leaf_level_components
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
                    exclude_modules=list(dict.fromkeys((*leaf_components, *resident_components))),
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
                        leaf_level_components=leaf_level_components,
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
            leaf_level_components=leaf_level_components,
        )
        if not result.applied:
            raise RuntimeError("No compatible torch.nn.Module components were available for group offload.")
        return result

    raise RuntimeError(f"Unsupported Diffusers offload mode: {mode}")
