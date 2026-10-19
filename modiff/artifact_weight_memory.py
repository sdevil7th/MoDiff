"""Read-only parameter-storage lower bounds; these never establish model fit."""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import re
import struct

from modiff.hf_cache_layout import resolve_snapshot_cache_file

_FLOAT_DTYPES = {"F16", "BF16", "F32", "F64"}
_DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}
_DTYPE_NAMES = {"float16": "F16", "bfloat16": "BF16", "float32": "F32", "float64": "F64"}
_STORED_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1,
                 "I16": 2, "U16": 2, "I32": 4, "U32": 4, "I64": 8, "U64": 8, "BOOL": 1}
_MAX_HEADER_BYTES = 16 * 1024 * 1024


def _snapshot_file(alias: Path, snapshot: Path) -> Path:
    repository_cache = snapshot.parent.parent
    repository = repository_cache.name.removeprefix("models--").replace("--", "/")
    return resolve_snapshot_cache_file(
        alias, snapshot=snapshot, cache_root=repository_cache.parent, repository=repository,
    )


def _json_file(path: Path):
    with path.open("rb") as stream:
        raw = stream.read(_MAX_HEADER_BYTES + 1)
    if len(raw) > _MAX_HEADER_BYTES:
        raise ValueError("Oversized model metadata")
    return json.loads(raw)


def _weight_bytes(path: Path, dtype_bytes: int, tensor_names: set[str] | None):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if size > _MAX_HEADER_BYTES:
            raise ValueError("Oversized safetensors metadata")
        header = json.loads(stream.read(size))
    if not isinstance(header, dict) or len(header) > 100_000:
        raise ValueError("Invalid tensor metadata")
    names = tensor_names if tensor_names is not None else set(header) - {"__metadata__"}
    total = 0
    for name in names:
        tensor = header[name]
        shape, offsets, dtype = tensor["shape"], tensor["data_offsets"], tensor["dtype"]
        if (not isinstance(shape, list) or len(shape) > 32
                or any(type(value) is not int or value < 0 for value in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(value) is not int or value < 0 for value in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= path.stat().st_size - size - 8
                or dtype not in _STORED_BYTES):
            raise ValueError("Invalid tensor storage")
        count = math.prod(shape)
        if count > 2**63 or offsets[1] - offsets[0] != count * _STORED_BYTES[dtype]:
            raise ValueError("Invalid tensor shape/storage size")
        total += count * (dtype_bytes if dtype in _FLOAT_DTYPES else _STORED_BYTES[dtype])
    return total


def artifact_parameter_lower_bound(status: dict, revision: str | None, dtype: str) -> dict | None:
    """Inspect only an already validated, exact local Diffusers snapshot.

    Indexed tensors are counted once. Unknown/unsupported layouts yield no
    estimate rather than granting readiness or fabricating a working budget.
    Snapshot files use the same repository/shared blob boundary as execution.
    No weights, Python model code, optional imports or downloads are executed.
    """
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision) or dtype not in _DTYPE_BYTES or not status.get("complete"):
        return None
    for snapshot in status.get("snapshots") or []:
        if not snapshot.get("complete") or not snapshot.get("path"):
            continue
        path = Path(snapshot["path"]).absolute()
        if path.name.lower() != revision.lower():
            continue
        try:
            index = _json_file(_snapshot_file(path / "model_index.json", path))
            if not isinstance(index, dict) or len(index) > 128:
                continue
            components = {}
            for component, definition in index.items():
                if component.startswith("_") or not isinstance(definition, list) or len(definition) != 2 or not all(definition):
                    continue
                if Path(component).name != component or component in {".", ".."} or any(char in component for char in "\\/:"):
                    raise ValueError("Invalid component path")
                folder = path / component
                if (folder.is_symlink() or getattr(folder, "is_junction", lambda: False)()
                        or not folder.resolve().is_relative_to(path.resolve())):
                    raise ValueError("Escaping component path")
                files = {}
                indexes = list(folder.glob("*.safetensors.index.json"))
                if len(indexes) > 1:
                    return None
                if indexes:
                    shard_index = _json_file(_snapshot_file(indexes[0], path))
                    if not isinstance(shard_index, dict):
                        return None
                    weight_map = shard_index.get("weight_map")
                    if not isinstance(weight_map, dict) or len(weight_map) > 100_000:
                        return None
                    for tensor, filename in weight_map.items():
                        if not isinstance(tensor, str) or not isinstance(filename, str) or Path(filename).name != filename or any(char in filename for char in "\\/:"):
                            raise ValueError("Invalid shard path")
                        files.setdefault(filename, set()).add(tensor)
                else:
                    unsharded = list(folder.glob("*.safetensors"))
                    if len(unsharded) > 1:
                        return None
                    files = {file.name: None for file in unsharded}
                if len(files) > 1024:
                    return None
                amount = 0
                for filename, tensors in files.items():
                    file = _snapshot_file(folder / filename, path)
                    amount += _weight_bytes(file, _DTYPE_BYTES[dtype], tensors)
                if amount:
                    components[component] = amount
            if components:
                return {"revision": revision, "dtype": dtype, "components": components,
                        "totalBytes": sum(components.values()), "largestComponentBytes": max(components.values()),
                        "proofScope": "parameter_storage_lower_bound_only"}
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, struct.error):
            continue
    return None


def _unique_json_pairs(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Duplicate loading metadata key")
        result[name] = value
    return result


def _loading_json_file(path: Path):
    with path.open("rb") as stream:
        raw = stream.read(_MAX_HEADER_BYTES + 1)
    if len(raw) > _MAX_HEADER_BYTES:
        raise ValueError("Oversized model metadata")
    return json.loads(raw, object_pairs_hook=_unique_json_pairs)


def _loading_shard(path: Path, dtype: str, selected: set[str] | None) -> dict:
    """Read every source tensor, while counting only selected destinations."""
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if header_size > _MAX_HEADER_BYTES:
            raise ValueError("Oversized safetensors metadata")
        raw = stream.read(header_size)
    header = json.loads(raw, object_pairs_hook=_unique_json_pairs)
    if not isinstance(header, dict) or len(header) > 100_000:
        raise ValueError("Invalid tensor metadata")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(type(key) is not str or type(value) is not str for key, value in metadata.items()):
        raise ValueError("Invalid safetensors metadata")
    names = set(header) - {"__metadata__"}
    selected = names if selected is None else selected
    if not selected <= names:
        raise ValueError("Missing indexed tensor")
    payload_size = path.stat().st_size - header_size - 8
    source = requested = converted = largest_cast = 0
    intervals = []
    for name in names:
        tensor = header[name]
        shape, offsets, stored = tensor["shape"], tensor["data_offsets"], tensor["dtype"]
        if (type(name) is not str or not name or not isinstance(shape, list) or len(shape) > 32
                or any(type(value) is not int or value < 0 for value in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(value) is not int or value < 0 for value in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= payload_size
                or stored not in _STORED_BYTES):
            raise ValueError("Invalid source tensor storage")
        count = math.prod(shape)
        size = count * _STORED_BYTES[stored]
        if count > 2**63 or offsets[1] - offsets[0] != size:
            raise ValueError("Invalid source tensor shape/storage size")
        source += size
        if size:
            intervals.append(tuple(offsets))
        if name in selected:
            target = count * (_DTYPE_BYTES[dtype] if stored in _FLOAT_DTYPES else _STORED_BYTES[stored])
            requested += target
            if stored in _FLOAT_DTYPES and stored != _DTYPE_NAMES[dtype]:
                converted += target
                largest_cast = max(largest_cast, target)
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError("Overlapping or incomplete source tensor payload")
        cursor = end
    if cursor != payload_size or any(value > 2**63 for value in (source, requested, converted)):
        raise ValueError("Invalid source tensor payload size")
    return {"source": source, "requested": requested, "converted": converted,
            "largestCast": largest_cast, "headerSha256": hashlib.sha256(raw).hexdigest(),
            "fileBytes": payload_size + header_size + 8}


def artifact_parameter_conversion_reserve(status: dict, revision: str | None, dtype: str) -> dict | None:
    """Companion cold-loading evidence; addressable overlap is not resident RAM.

    The parameter lower-bound record remains unchanged. Official Diffusers sorts
    unique index filenames and retains a whole source shard while casting its
    selected tensors. Same-dtype destinations alias that source, while differing
    dtypes need distinct destinations. All source tensors count, even unused
    index entries. Parallel or unreviewed ordering uses the conservative envelope.
    Only metadata from an already validated immutable local snapshot is read.
    No tensor imports, model code, weights, installation or downloads run here.
    """
    if (type(revision) is not str or not re.fullmatch(r"[0-9a-f]{40}", revision)
            or type(dtype) is not str or dtype not in _DTYPE_BYTES
            or not isinstance(status, dict) or status.get("complete") is not True):
        return None
    snapshots = status.get("snapshots")
    if not isinstance(snapshots, list) or len(snapshots) > 1024:
        return None
    for snapshot in snapshots:
        if (not isinstance(snapshot, dict) or snapshot.get("complete") is not True
                or type(snapshot.get("path")) is not str or not snapshot["path"]):
            continue
        try:
            path = Path(snapshot["path"]).absolute()
            if path.name != revision:
                continue
            index = _loading_json_file(_snapshot_file(path / "model_index.json", path))
            if not isinstance(index, dict) or len(index) > 128:
                continue
            components = {}
            for component, definition in index.items():
                if component.startswith("_") or not isinstance(definition, list) or len(definition) != 2 or not all(definition):
                    continue
                if (Path(component).name != component or component in {".", ".."}
                        or any(char in component for char in "\\/:")
                        or any(type(value) is not str or not value or len(value) > 256 for value in definition)):
                    raise ValueError("Invalid loading component metadata")
                folder = path / component
                if (folder.is_symlink() or getattr(folder, "is_junction", lambda: False)()
                        or not folder.resolve().is_relative_to(path.resolve())):
                    raise ValueError("Escaping component path")
                indexes = list(folder.glob("*.safetensors.index.json"))
                if len(indexes) > 1:
                    return None
                files = {}
                if indexes:
                    shard_index = _loading_json_file(_snapshot_file(indexes[0], path))
                    if not isinstance(shard_index, dict):
                        return None
                    weight_map = shard_index.get("weight_map")
                    if not isinstance(weight_map, dict) or not 0 < len(weight_map) <= 100_000:
                        return None
                    for tensor, filename in weight_map.items():
                        if (type(tensor) is not str or not tensor or type(filename) is not str
                                or Path(filename).name != filename or any(char in filename for char in "\\/:")
                                or not filename.endswith(".safetensors")):
                            raise ValueError("Invalid loading shard path")
                        files.setdefault(filename, set()).add(tensor)
                else:
                    unsharded = list(folder.glob("*.safetensors"))
                    if len(unsharded) > 1:
                        return None
                    files = {file.name: None for file in unsharded}
                if len(files) > 1024:
                    return None
                source = requested = converted = serial_cpu = serial_cuda = largest_source = largest_cast = 0
                layout = []
                for filename in sorted(files):
                    shard = _loading_shard(_snapshot_file(folder / filename, path), dtype, files[filename])
                    overlap = shard["source"] + shard["converted"]
                    serial_cpu = max(serial_cpu, requested + overlap)
                    serial_cuda = max(serial_cuda, overlap)
                    source += shard["source"]
                    requested += shard["requested"]
                    converted += shard["converted"]
                    largest_source = max(largest_source, shard["source"])
                    largest_cast = max(largest_cast, shard["largestCast"])
                    layout.append([filename, sorted(files[filename]) if files[filename] is not None else None,
                                   shard["fileBytes"], shard["headerSha256"]])
                if requested:
                    conservative = source + converted
                    if any(value > 2**63 for value in (source, requested, conservative)):
                        raise ValueError("Oversized loading overlap")
                    components[component] = {
                        "loaderLibrary": definition[0], "loaderClass": definition[1],
                        "layoutSha256": hashlib.sha256(json.dumps(layout, separators=(",", ":")).encode()).hexdigest(),
                        "sourcePayloadBytes": source, "requestedPayloadBytes": requested,
                        "convertedTargetBytes": converted, "largestSourceShardBytes": largest_source,
                        "largestConvertedTensorBytes": largest_cast,
                        "serialCpuOverlapBytes": max(requested, serial_cpu),
                        "conservativeCpuOverlapBytes": conservative,
                        "serialCudaOverlapBytes": serial_cuda,
                        "conservativeCudaOverlapBytes": conservative,
                    }
            if components:
                return {"revision": revision, "dtype": dtype, "components": components,
                        "proofScope": "source_requested_loading_overlap_only", "complete": False}
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, struct.error):
            continue
    return None


def native_parameter_conversion_reserve(
    reserve: dict | None, *, lower_bound: dict | None, execution_path: str,
    offload_mode: str, device: str, dtype: str, weight_loading_mode: str = "cpu",
    parallel_loading: bool = True,
) -> dict:
    """Select a conservative cold CPU overlap, separately from parameter storage.

    Serial narrowing requires the reviewed Diffusers loader contract and an
    effective pre-import flag supplied by the caller. Transformers/other loaders
    remain conservative. A CUDA loading route still casts on CPU and serializes
    a complete component through CPU disk copies; no zero-CPU claim is made.
    Cold phases, retained-owner waivers and headroom remain the caller's policy.
    """
    if (any(type(value) is not str for value in (execution_path, offload_mode, device, dtype, weight_loading_mode))
            or type(parallel_loading) is not bool):
        return {}
    baseline = native_cpu_parameter_residency(lower_bound, execution_path=execution_path,
        offload_mode=offload_mode, device=device, dtype=dtype)
    if (not baseline or not isinstance(reserve, dict)
            or reserve.get("proofScope") != "source_requested_loading_overlap_only"
            or reserve.get("complete") is not False or reserve.get("revision") != lower_bound["revision"]
            or reserve.get("dtype") != dtype
            or weight_loading_mode not in {"cpu", "cuda"}):
        return {}
    components = reserve.get("components")
    if not isinstance(components, dict) or set(components) != set(lower_bound["components"]):
        return {}
    numeric = ("sourcePayloadBytes", "requestedPayloadBytes", "convertedTargetBytes",
               "largestSourceShardBytes", "largestConvertedTensorBytes", "serialCpuOverlapBytes",
               "conservativeCpuOverlapBytes", "serialCudaOverlapBytes", "conservativeCudaOverlapBytes")
    cpu_peaks, cuda_peaks = [], []
    for name, item in components.items():
        if (not isinstance(item, dict) or any(type(item.get(key)) is not int or not 0 <= item[key] <= 2**63 for key in numeric)
                or any(type(item.get(key)) is not str or not item[key] or len(item[key]) > 256
                       for key in ("loaderLibrary", "loaderClass"))
                or type(item.get("layoutSha256")) is not str
                or not re.fullmatch(r"[0-9a-f]{64}", item["layoutSha256"])):
            return {}
        source, requested, converted = (item[key] for key in numeric[:3])
        conservative = source + converted
        if (requested != lower_bound["components"][name] or source <= 0 or conservative > 2**63
                or converted > requested or requested - converted > source
                or not 0 < item["largestSourceShardBytes"] <= source
                or item["largestConvertedTensorBytes"] > converted
                or bool(item["largestConvertedTensorBytes"]) != bool(converted)
                or item["serialCudaOverlapBytes"] < item["largestSourceShardBytes"]
                or not max(requested, item["serialCudaOverlapBytes"]) <= item["serialCpuOverlapBytes"] <= conservative
                or item["conservativeCpuOverlapBytes"] != conservative
                or item["conservativeCudaOverlapBytes"] != conservative):
            return {}
        serial = not parallel_loading and item["loaderLibrary"] == "diffusers"
        cpu_peaks.append(item["serialCpuOverlapBytes"] if serial else conservative)
        cuda_peaks.append(max(requested, item["serialCudaOverlapBytes"] if serial else conservative))
    cpu_execution = str(device).split(":", 1)[0] == "cpu"
    if weight_loading_mode == "cuda":
        if offload_mode != "group_disk" or device not in {"cuda", "cuda:0"}:
            return {}
        return {"systemRamBytes": max(cuda_peaks), "vramBytes": 0}
    if not cpu_execution and offload_mode == "group_disk":
        return {"systemRamBytes": max(cpu_peaks), "vramBytes": 0}
    # Component iteration order is pipeline-owned, not model_index JSON order.
    # Budget all other requested components beside the largest conversion excess.
    excess = max(peak - lower_bound["components"][name]
                 for name, peak in zip(components, cpu_peaks))
    amount = lower_bound["totalBytes"] + excess
    return {"systemRamBytes": amount, "vramBytes": 0} if amount <= 2**63 else {}


def impossible_parameter_residency(hardware: dict, lower_bound: dict | None, offload_mode: str) -> list[str]:
    if not lower_bound:
        return []
    accelerator = hardware.get("accelerator") or {}
    if accelerator.get("kind") not in {"cuda", "xpu", "mps"}:
        return []
    needed = lower_bound["totalBytes"] if offload_mode == "none" else lower_bound["largestComponentBytes"] if offload_mode == "model_cpu" else 0
    total = accelerator.get("totalBytes")
    if type(total) is int and needed > total:
        return [f"GPU memory requires at least {needed / 1024**3:.2f} GiB of parameter storage for {offload_mode} placement ({needed} bytes); {total} total accelerator bytes are available. Select a reviewed lower-residency recipe or Custom memory."]
    return []


def native_cpu_parameter_residency(
    lower_bound: dict | None, *, execution_path: str, offload_mode: str, device: str, dtype: str,
) -> dict:
    """Minimum CPU storage for the native loader's proven loading phases.

    CPU/model/group CPU placements retain all selected weights. Accelerator
    ``none`` loads every component on CPU before placing them on the device.
    Incremental disk offload still loads one entire component before its disk
    hook is installed, so only the largest component is a proven disk minimum.
    These bounds exclude loading overhead, activations and disk placeholders.
    """
    if (execution_path != "modular-diffusers" or not isinstance(lower_bound, dict)
            or lower_bound.get("proofScope") != "parameter_storage_lower_bound_only"
            or lower_bound.get("dtype") != dtype
            or not isinstance(lower_bound.get("revision"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", lower_bound["revision"])
            or dtype not in _DTYPE_BYTES):
        return {}
    components = lower_bound.get("components")
    if (not isinstance(components, dict) or not 0 < len(components) <= 128
            or any(not isinstance(name, str) or not name or len(name) > 256
                   or type(value) is not int or not 0 < value <= 2**63
                   for name, value in components.items())):
        return {}
    total = lower_bound.get("totalBytes")
    largest = lower_bound.get("largestComponentBytes")
    if (type(total) is not int or total != sum(components.values()) or total > 2**63
            or type(largest) is not int or largest != max(components.values())):
        return {}
    cpu_execution = str(device).split(":", 1)[0] == "cpu"
    if not cpu_execution and offload_mode not in {"none", "model_cpu", "auto_cpu", "group_cpu", "group_disk"}:
        return {}
    return {"systemRamBytes": largest if not cpu_execution and offload_mode == "group_disk" else total,
            "vramBytes": 0}


def native_cpu_parameter_phase(*, offload_mode: str, device: str) -> str:
    """Classify an already validated native CPU storage requirement."""
    return ("cpu_loading" if not str(device).startswith("cpu") and offload_mode in {"none", "group_disk"}
            else "cpu_resident")


def native_primary_cuda_parameter_owner(
    lower_bound: dict | None, *, execution_path: str, device: str, dtype: str, hardware: dict,
) -> bool:
    """Validate dense native parameter identity and primary dedicated CUDA scope.

    This predicate declares no GPU minimum for group placement. It also lets an
    exact complete warm group owner request one observed allocator refresh.
    """
    accelerator = hardware.get('accelerator') if isinstance(hardware.get('accelerator'), dict) else {}
    runtime = hardware.get('runtime') if isinstance(hardware.get('runtime'), dict) else {}
    runtime_hardware = runtime.get('hardware') if isinstance(runtime.get('hardware'), dict) else {}
    torch_state = runtime_hardware.get('torch') if isinstance(runtime_hardware.get('torch'), dict) else {}
    backend = str(hardware.get('backend') or accelerator.get('backend') or accelerator.get('kind') or '').lower()
    if (accelerator.get('kind') != 'cuda' or backend != 'cuda'
            or str(hardware.get('runtimeBackend') or '').lower() == 'rocm'
            or torch_state.get('hip_version')
            or str(accelerator.get('memoryKind') or '').lower() != 'dedicated'
            or accelerator.get('sharedMemory') is True
            or device not in {'cuda', 'cuda:0'}
            or not native_cpu_parameter_residency(
                lower_bound, execution_path=execution_path, offload_mode='model_cpu',
                device=device, dtype=dtype,
            )):
        return False
    return True


def native_gpu_parameter_residency(
    lower_bound: dict | None, *, execution_path: str, offload_mode: str,
    device: str, dtype: str, hardware: dict,
) -> dict:
    """Known parameter storage for dedicated NVIDIA CUDA native placements.

    Resident placement needs every component; official model CPU offload still
    places one complete component on the device. Group GPU storage is unknown.
    This minimum grants neither allocator credit nor an inference-fit estimate.
    """
    if (offload_mode not in {'none', 'model_cpu', 'auto_cpu'}
            or not native_primary_cuda_parameter_owner(lower_bound, execution_path=execution_path,
                device=device, dtype=dtype, hardware=hardware)):
        return {}
    return {'vramBytes': lower_bound['totalBytes'] if offload_mode == 'none'
            else lower_bound['largestComponentBytes']}


def native_disk_lora_loading_floor(lower_bound: dict | None, *, targets: tuple[str, ...], dtype: str) -> dict:
    """Disk LoRA stages every reviewed mutation target until upstream injection.

    Other components still offload incrementally. This is only a parameter
    minimum, excluding adapters, conversion temporaries and hook placeholders.
    Missing target headers never establish a smaller loading phase.
    """
    baseline = native_cpu_parameter_residency(lower_bound, execution_path='modular-diffusers',
        offload_mode='group_disk', device='cuda', dtype=dtype)
    if (not baseline or not isinstance(targets, tuple) or not 0 < len(targets) <= 32
            or any(not isinstance(name, str) for name in targets)
            or len(set(targets)) != len(targets)
            or any(name not in lower_bound['components'] for name in targets)):
        return {}
    components = lower_bound['components']
    staged = sum(components[name] for name in targets)
    other = max((size for name, size in components.items() if name not in targets), default=0)
    return {'systemRamBytes': max(staged, other), 'vramBytes': 0}
