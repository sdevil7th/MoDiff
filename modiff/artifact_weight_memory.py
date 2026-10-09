"""Read-only parameter-storage lower bounds; these never establish model fit."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import struct

_FLOAT_DTYPES = {"F16", "BF16", "F32", "F64"}
_DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}
_STORED_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1,
                 "I16": 2, "U16": 2, "I32": 4, "U32": 4, "I64": 8, "U64": 8, "BOOL": 1}
_MAX_HEADER_BYTES = 16 * 1024 * 1024


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
    No weights, Python model code, optional imports or downloads are executed.
    """
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision) or dtype not in _DTYPE_BYTES or not status.get("complete"):
        return None
    for snapshot in status.get("snapshots") or []:
        if not snapshot.get("complete") or not snapshot.get("path"):
            continue
        path = Path(snapshot["path"])
        if path.name.lower() != revision.lower():
            continue
        try:
            cache_root = path.parent.parent.resolve()
            if not path.resolve().is_relative_to(cache_root) or not (path / "model_index.json").resolve().is_relative_to(cache_root):
                continue
            index = _json_file(path / "model_index.json")
            if not isinstance(index, dict) or len(index) > 128:
                continue
            components = {}
            for component, definition in index.items():
                if component.startswith("_") or not isinstance(definition, list) or len(definition) != 2 or not all(definition):
                    continue
                if Path(component).name != component or component in {".", ".."} or any(char in component for char in "\\/:"):
                    raise ValueError("Invalid component path")
                folder = path / component
                if not folder.resolve().is_relative_to(cache_root):
                    raise ValueError("Escaping component path")
                files = {}
                indexes = list(folder.glob("*.safetensors.index.json"))
                if len(indexes) > 1:
                    return None
                if indexes:
                    if not indexes[0].resolve().is_relative_to(cache_root):
                        raise ValueError("Escaping weight index")
                    weight_map = _json_file(indexes[0]).get("weight_map")
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
                    file = folder / filename
                    if not file.resolve().is_relative_to(cache_root):
                        raise ValueError("Escaping weight path")
                    amount += _weight_bytes(file, _DTYPE_BYTES[dtype], tensors)
                if amount:
                    components[component] = amount
            if components:
                return {"revision": revision, "dtype": dtype, "components": components,
                        "totalBytes": sum(components.values()), "largestComponentBytes": max(components.values()),
                        "proofScope": "parameter_storage_lower_bound_only"}
        except (OSError, ValueError, KeyError, TypeError, struct.error):
            continue
    return None


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
