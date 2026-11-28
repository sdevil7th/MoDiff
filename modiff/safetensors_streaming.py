"""Reviewed, read-only placement of dense checkpoints directly from Hub shards."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from modiff.artifact_weight_memory import _loading_json_file, _snapshot_file


SSD_STREAM = "ssd_stream"
GIB = 1024**3

# Each class has a forward entry point that can use Accelerate's layer hooks.
# Other architectures need an explicit review before joining this path.
STREAMABLE_CLASSES = {
    ("diffusers", "QwenImageTransformer2DModel"),
    ("transformers", "Qwen2_5_VLForConditionalGeneration"),
    ("diffusers", "Flux2Transformer2DModel"),
    ("transformers", "Qwen3ForCausalLM"),
    ("transformers", "LlamaForCausalLM"),
}
RESIDENT_CLASSES = {
    ("diffusers", "AutoencoderKLQwenImage"),
    ("diffusers", "AutoencoderKLFlux2"),
}
STREAMABLE_PROFILES = {
    "QwenImageModularPipeline": ("Qwen/Qwen-Image", "Qwen/Qwen-Image-2512"),
    "Flux2KleinModularPipeline": ("black-forest-labs/FLUX.2-klein-4B",),
}


def supports_streaming_profile(model_type, repository, workflow_id, dtype):
    return (dtype == "bfloat16" and workflow_id == "text2image"
            and repository in STREAMABLE_PROFILES.get(model_type, ()))


def require_streaming_capacity(budget, device):
    """Recheck live resources before constructing or reusing a streaming owner."""
    import torch

    from modiff.hardware import _linux_amd_memory_regions, rocm_memory_topology, system_memory_snapshot

    if str(device) not in {"cuda", "cuda:0"} or not torch.cuda.is_available():
        raise ValueError("SSD streaming requires the primary CUDA or ROCm accelerator.")
    memory = system_memory_snapshot()
    free_gpu, total_gpu = torch.cuda.mem_get_info(device)
    available = {"systemRamBytes": memory["available_bytes"], "vramBytes": free_gpu}
    requirements = dict(budget)
    if torch.version.hip:
        _, shared = rocm_memory_topology(
            torch.cuda.get_device_name(device), total_gpu, memory["total_bytes"],
            torch.cuda.get_device_properties(device), _linux_amd_memory_regions(), torch.cuda.device_count(),
        )
        if shared:
            # An APU must fit both working sets in the same physical pool.
            requirements["systemRamBytes"] += budget["vramBytes"]
    if memory["source"] == "GlobalMemoryStatusEx":
        available["systemCommitBytes"] = memory["page_file_available_bytes"]
    else:
        import sys

        if sys.platform == "win32":
            raise ValueError("SSD streaming could not inspect Windows commit headroom.")
    for key, free in available.items():
        if type(free) is not int or free < requirements[key]:
            raise ValueError(f"SSD streaming needs {requirements[key]} bytes of {key}, {free} available.")


def streaming_weight_budget(census):
    """Reserve shard mappings, active weights and resident decoder storage.

    The serial reader can overlap mappings for successive shards on Windows.
    Account for two source mappings in commit and keep separate physical and
    device working reserves. A normal group-disk recipe has different costs.
    """
    if not isinstance(census, dict) or census.get("dtype") != "bfloat16":
        return None
    components = census.get("components")
    if not isinstance(components, dict) or not components:
        return None
    streamed, resident = [], 0
    for item in components.values():
        if not isinstance(item, dict):
            return None
        identity = (item.get("loaderLibrary"), item.get("loaderClass"))
        size, shard = item.get("requestedPayloadBytes"), item.get("largestSourceShardBytes")
        if (type(size) is not int or type(shard) is not int or not 0 < shard <= size
                or item.get("convertedTargetBytes") != 0):
            return None
        if identity in STREAMABLE_CLASSES:
            streamed.append(shard)
        elif identity in RESIDENT_CLASSES:
            resident += size
        else:
            return None
    if not streamed:
        return None
    largest = max(streamed)
    return {
        "systemRamBytes": largest + 2 * resident + 4 * GIB,
        "systemCommitBytes": 2 * largest + 2 * resident + 4 * GIB,
        "vramBytes": largest + resident + 4 * GIB,
    }


def snapshot_component_file(snapshot, subfolder, filename):
    if subfolder is not None and (not isinstance(subfolder, str) or Path(subfolder).name != subfolder
                                  or subfolder in {".", ".."} or any(char in subfolder for char in "\\/:")):
        raise ValueError("SSD streaming requires a reviewed checkpoint component folder.")
    if (not isinstance(filename, str) or Path(filename).name != filename
            or filename in {".", ".."} or any(char in filename for char in "\\/:")):
        raise ValueError("Invalid SSD-streaming checkpoint filename.")
    folder = snapshot if subfolder is None else snapshot / subfolder
    if folder.is_symlink() or folder.is_junction():
        raise ValueError("SSD streaming does not accept a redirected component folder.")
    return _snapshot_file(folder / filename, snapshot)


def _shard_index(snapshot, subfolder):
    from safetensors import safe_open

    folder = snapshot if subfolder is None else snapshot / subfolder
    # Validate the component directory before enumerating its members.
    snapshot_component_file(snapshot, subfolder, "config.json")
    indexes = sorted(folder.glob("*.safetensors.index.json"))
    if len(indexes) > 1:
        raise ValueError("SSD streaming found multiple checkpoint indexes.")
    if indexes:
        document = _loading_json_file(snapshot_component_file(snapshot, subfolder, indexes[0].name))
        weights = document.get("weight_map") if isinstance(document, dict) else None
        if not isinstance(weights, dict) or not 0 < len(weights) <= 100_000:
            raise ValueError("SSD streaming found an invalid checkpoint index.")
    else:
        shards = sorted(folder.glob("*.safetensors"))
        if len(shards) != 1:
            raise ValueError("SSD streaming requires one safetensors file or one shard index.")
        with safe_open(snapshot_component_file(snapshot, subfolder, shards[0].name), framework="pt") as f:
            weights = dict.fromkeys(f.keys(), shards[0].name)
    paths = {}
    for name, filename in weights.items():
        if not isinstance(name, str) or not isinstance(filename, str) or not filename.endswith(".safetensors"):
            raise ValueError("SSD streaming found an invalid shard mapping.")
        if filename not in paths:
            if len(paths) >= 1024:
                raise ValueError("SSD streaming found too many checkpoint shards.")
            paths[filename] = snapshot_component_file(snapshot, subfolder, filename)
    return weights, paths


def load_streamed_component(snapshot, subfolder, cls, library, dtype, device):
    """Create a single hook-owned component without materializing all its weights."""
    import torch
    from accelerate import dispatch_model, init_empty_weights
    from safetensors import safe_open

    if (library, cls.__name__) not in STREAMABLE_CLASSES or dtype is not torch.bfloat16:
        raise ValueError("This class or precision has no reviewed SSD-streaming loader.")
    if (getattr(cls, "_keep_in_fp32_modules", None) or getattr(cls, "_keep_in_fp32_modules_strict", None)
            or getattr(cls, "_dtype_plan", None)):
        raise ValueError("SSD streaming requires components without per-layer precision overrides.")
    snapshot = Path(snapshot).absolute()
    config = _loading_json_file(snapshot_component_file(snapshot, subfolder, "config.json"))
    if not isinstance(config, dict) or config.get("auto_map") or config.get("quantization_config"):
        raise ValueError("SSD streaming requires a built-in dense component config.")
    weights, paths = _shard_index(snapshot, subfolder)
    folder = snapshot if subfolder is None else snapshot / subfolder
    if library == "transformers":
        from transformers import GenerationConfig
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import WeightRenaming
        from transformers.integrations.accelerate import accelerate_disk_offload

        if config.get("model_type") != cls.config_class.model_type:
            raise ValueError("SSD-streamed encoder config does not match its reviewed class.")
        with init_empty_weights(include_buffers=False):
            model = cls._from_config(cls.config_class.from_dict(config), dtype=dtype)
        generation_config = folder / "generation_config.json"
        if model.can_generate() and generation_config.exists():
            model.generation_config = GenerationConfig.from_dict(_loading_json_file(
                snapshot_component_file(snapshot, subfolder, generation_config.name),
            ))
        mapping = get_model_conversion_mapping(model, None, None)
        if any(not isinstance(item, WeightRenaming) for item in mapping or ()):
            raise ValueError("SSD streaming cannot apply checkpoint tensor conversions.")
        index = accelerate_disk_offload(
            model, None, [str(folder / filename) for filename in paths],
            {"": "disk"}, {"weight_map": weights}, mapping,
        )
    else:
        if config.get("_class_name") != cls.__name__:
            raise ValueError("SSD-streamed denoiser config does not match its reviewed class.")
        # Match Diffusers' native initialization dtype without casting
        # nonpersistent buffers that the architecture explicitly builds in FP32.
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            with init_empty_weights(include_buffers=False):
                model = cls.from_config(config)
        finally:
            torch.set_default_dtype(original_dtype)
        index = {name: {"safetensors_file": str(folder / filename), "weight_name": name}
                 for name, filename in weights.items()}
    params = dict(model.named_parameters(remove_duplicate=False))
    if (params.keys() != index.keys()
            or any(param.device.type != "meta" or param.dtype != dtype for param in params.values())):
        raise ValueError("SSD-streamed checkpoint keys or parameter placement do not match the native class.")
    sources = defaultdict(list)
    resolved_index = {}
    for name, record in index.items():
        filename = Path(record["safetensors_file"]).name
        source = record.get("weight_name", name)
        if filename not in paths or weights.get(source) != filename:
            raise ValueError("SSD-streamed checkpoint key mapping does not match its verified shards.")
        resolved_index[name] = {**record, "safetensors_file": str(paths[filename])}
        sources[filename].append((source, params[name].shape))
    for filename, path in paths.items():
        expected = {name for name, _ in sources[filename]}
        with safe_open(path, framework="pt") as shard:
            if set(shard.keys()) != expected:
                raise ValueError("SSD-streamed checkpoint contains extra or missing tensors.")
            for name, shape in sources[filename]:
                tensor = shard.get_slice(name)
                if tuple(tensor.get_shape()) != tuple(shape) or tensor.get_dtype() != "BF16":
                    raise ValueError("SSD-streamed tensor shape or precision differs from its declaration.")
    dispatch_model(model.eval(), device_map={"": "disk"}, main_device=str(device),
                   offload_index=resolved_index, skip_keys=getattr(model, "_skip_keys_device_placement", None))
    model._modiff_ssd_streamed = True
    return model


def standalone_streaming_budget(snapshot, subfolder, library, class_name):
    """Inspect the exact component layout without fetching tensor data."""
    from modiff.artifact_weight_memory import _loading_shard

    weights, paths = _shard_index(Path(snapshot).absolute(), subfolder)
    names_by_file = defaultdict(set)
    for tensor, filename in weights.items():
        names_by_file[filename].add(tensor)
    census = [_loading_shard(path, "bfloat16", names_by_file[filename])
              for filename, path in paths.items()]
    budget = streaming_weight_budget({"dtype": "bfloat16", "components": {"model": {
        "loaderLibrary": library, "loaderClass": class_name,
        "requestedPayloadBytes": sum(item["requested"] for item in census),
        "convertedTargetBytes": sum(item["converted"] for item in census),
        "largestSourceShardBytes": max(item["source"] for item in census),
    }}})
    if budget is None:
        raise ValueError("The standalone component does not have a reviewed dense BF16 streaming layout.")
    return budget


def prepare_standalone_streaming(repository, revision, subfolder, library, class_name, device):
    """Resolve and check the same exact source on cold and cached node loads."""
    from utils.huggingface import exact_cached_snapshot_path

    marker = "model_index.json" if subfolder else "config.json"
    snapshot = exact_cached_snapshot_path(repository, revision, marker)
    require_streaming_capacity(standalone_streaming_budget(snapshot, subfolder, library, class_name), device)
    return snapshot


def dispose_streamed_component(model):
    """Dispose a final owner without the manager converting meta weights to CPU."""
    from accelerate.hooks import remove_hook_from_module

    remove_hook_from_module(model, recurse=True)
    for module in model.modules():
        module._parameters.clear()
        module._buffers.clear()
