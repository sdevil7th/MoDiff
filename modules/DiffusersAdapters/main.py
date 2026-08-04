"""Model-neutral Diffusers LoRA inspection and lifecycle nodes."""

import json
from pathlib import Path
from typing import Any

from modiff.NodeBase import NodeBase


def _string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        values = value.replace("\n", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _resolve_local_adapter(selection: Any, weight_name: str | None = None) -> tuple[Path, str | None]:
    if isinstance(selection, dict):
        value = str(selection.get("value") or "").strip()
        source = selection.get("source") or "hub"
    else:
        value = str(selection or "").strip()
        source = "local" if Path(value).expanduser().exists() else "hub"
    if not value:
        raise ValueError("A LoRA adapter is required.")
    weight_name = str(weight_name or "").strip() or None
    if source == "hub":
        from utils.huggingface import cached_file_path

        repo_id = value
        if not weight_name:
            parts = value.split("/")
            if len(parts) >= 3:
                repo_id, weight_name = "/".join(parts[:2]), "/".join(parts[2:])
        if not weight_name:
            raise ValueError("A Hub LoRA needs a pinned weight name installed through Model Manager.")
        cached = cached_file_path(repo_id, weight_name)
        if not cached:
            raise FileNotFoundError(f"LoRA {repo_id}/{weight_name} is not installed.")
        path = Path(cached)
        return path.parent, path.name
    path = Path(value).expanduser()
    if path.is_file():
        return path.parent, path.name
    if not path.is_dir():
        raise FileNotFoundError(f"LoRA path does not exist: {path}")
    return path, weight_name


def inspect_lora_file(path: Path, *, base_model: str = "") -> dict[str, Any]:
    from safetensors import safe_open

    if path.suffix.lower() != ".safetensors":
        raise ValueError("LoRA inspection currently requires a Safetensors weight file.")
    ranks = set()
    targets = set()
    key_count = 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            key_count += 1
            normalized = str(key)
            if ".lora_A." in normalized or ".lora_down." in normalized:
                shape = tuple(handle.get_slice(key).get_shape())
                if shape:
                    ranks.add(int(shape[0]))
            prefix = normalized.split(".", 1)[0]
            if prefix in {"transformer", "unet", "text_encoder", "text_encoder_2"}:
                targets.add(prefix)
            elif normalized.startswith("lora_unet_"):
                targets.add("unet")
            elif normalized.startswith(("lora_te_", "lora_te1_")):
                targets.add("text_encoder")
            elif normalized.startswith("lora_te2_"):
                targets.add("text_encoder_2")
    declared_base = next(
        (metadata.get(key) for key in ("ss_base_model_version", "modelspec.architecture", "base_model") if metadata.get(key)),
        None,
    )
    requested_base = str(base_model or "").strip() or None
    compatibility = "unknown"
    compatibility_reason = "Safetensors keys and metadata cannot prove base-model compatibility."
    if requested_base and declared_base:
        compatibility = "declared_match" if requested_base.lower() in str(declared_base).lower() else "declared_mismatch"
        compatibility_reason = f"Adapter declares {declared_base!r}; requested base is {requested_base!r}."
    triggers = _string_list(metadata.get("ss_tag_frequency") or metadata.get("trigger_words"))
    return {
        "schema_version": 1,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "tensor_count": key_count,
        "ranks": sorted(ranks),
        "target_components": sorted(targets),
        "declared_base_model": declared_base,
        "requested_base_model": requested_base,
        "compatibility": compatibility,
        "compatibility_reason": compatibility_reason,
        "trigger_words": triggers,
        "metadata": metadata,
    }


def _adapter_descriptor(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("LoRA operations need adapter objects from a LoRA node.")
    required = ("lora_path", "adapter_name")
    missing = [name for name in required if not value.get(name)]
    if missing:
        raise ValueError(f"LoRA adapter is missing: {', '.join(missing)}.")
    return dict(value)


def active_adapter_names(pipeline: Any) -> set[str]:
    getter = getattr(pipeline, "get_list_adapters", None)
    if not callable(getter):
        return set()
    listed = getter() or {}
    if isinstance(listed, dict):
        return {str(name) for names in listed.values() for name in (names or [])}
    return set(_string_list(listed))


def apply_lora_mix(pipeline: Any, adapters: Any) -> dict[str, Any]:
    if pipeline is None:
        raise ValueError("LoRA Stack / Mix needs a pipeline.")
    values = adapters if isinstance(adapters, list) else [adapters]
    values = [_adapter_descriptor(item) for item in values if item is not None]
    if not values:
        raise ValueError("LoRA Stack / Mix needs at least one adapter.")
    load = getattr(pipeline, "load_lora_weights", None)
    activate = getattr(pipeline, "set_adapters", None)
    if not callable(load) or not callable(activate):
        raise ValueError("This Diffusers pipeline does not expose the multi-adapter LoRA API.")
    loaded = active_adapter_names(pipeline)
    names = []
    weights = []
    for adapter in values:
        name = str(adapter["adapter_name"])
        if name not in loaded:
            load_kwargs = {"adapter_name": name}
            if adapter.get("weight_name"):
                load_kwargs["weight_name"] = adapter["weight_name"]
            load(adapter["lora_path"], **load_kwargs)
            loaded.add(name)
        names.append(name)
        weights.append(float(adapter.get("scale", 1.0)))
    activate(names, weights)
    return {"adapter_names": names, "adapter_weights": weights}


class LoRAInspectValidate(NodeBase):
    label = "Inspect / Validate Diffusers LoRA"
    category = "Diffusers Adapters"
    resizable = True
    params = {
        "adapter": {"label": "Adapter", "display": "modelselect", "type": "string", "fieldOptions": {"noValidation": True, "sources": ["hub", "local"]}},
        "weight_name": {"label": "Weight Name", "type": "string", "default": ""},
        "base_model": {"label": "Expected Base Model", "type": "string", "default": ""},
        "report": {"label": "Inspection", "display": "output", "type": "string"},
        "compatibility": {"label": "Compatibility", "display": "output", "type": "string"},
        "rank": {"label": "Maximum Rank", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        directory, weight_name = _resolve_local_adapter(kwargs.get("adapter"), kwargs.get("weight_name"))
        if not weight_name:
            candidates = sorted(directory.glob("*.safetensors"))
            if len(candidates) != 1:
                raise ValueError("Choose a weight name when the adapter directory does not contain exactly one Safetensors file.")
            path = candidates[0]
        else:
            path = directory / weight_name
        if not path.is_file():
            raise FileNotFoundError(f"LoRA weight file does not exist: {path}")
        report = inspect_lora_file(path, base_model=kwargs.get("base_model") or "")
        return {
            "report": json.dumps(report, sort_keys=True),
            "compatibility": report["compatibility"],
            "rank": max(report["ranks"], default=0),
        }


class LoRAStackMix(NodeBase):
    label = "Diffusers LoRA Stack / Mix"
    category = "Diffusers Adapters"
    params = {
        "pipeline": {"label": "Pipeline", "display": "input", "type": "any"},
        "adapters": {"label": "LoRAs", "display": "input", "type": ["custom_lora", "collection"]},
        "output": {"label": "Pipeline", "display": "output", "type": "any"},
        "active_mix": {"label": "Active Mix", "display": "output", "type": "string"},
    }

    def execute(self, **kwargs):
        report = apply_lora_mix(kwargs.get("pipeline"), kwargs.get("adapters"))
        return {"output": kwargs.get("pipeline"), "active_mix": json.dumps(report, sort_keys=True)}


class LoRAHotswap(NodeBase):
    label = "Hotswap Diffusers LoRA"
    category = "Diffusers Adapters"
    params = {
        "pipeline": {"label": "Pipeline", "display": "input", "type": "any"},
        "replacement": {"label": "Replacement", "display": "input", "type": "custom_lora"},
        "slot_name": {"label": "Existing Slot", "type": "string", "default": "default_0"},
        "output": {"label": "Pipeline", "display": "output", "type": "any"},
    }

    def execute(self, **kwargs):
        pipeline = kwargs.get("pipeline")
        adapter = _adapter_descriptor(kwargs.get("replacement"))
        slot = str(kwargs.get("slot_name") or "default_0")
        if slot not in active_adapter_names(pipeline):
            raise ValueError(f"LoRA hotswap slot {slot!r} is not loaded. Load the initial adapter before hotswapping it.")
        load_kwargs = {"adapter_name": slot, "hotswap": True}
        if adapter.get("weight_name"):
            load_kwargs["weight_name"] = adapter["weight_name"]
        pipeline.load_lora_weights(adapter["lora_path"], **load_kwargs)
        pipeline.set_adapters([slot], [float(adapter.get("scale", 1.0))])
        return {"output": pipeline}


class LoRAFuseUnfuse(NodeBase):
    label = "Fuse / Unfuse Diffusers LoRA"
    category = "Diffusers Adapters"
    params = {
        "pipeline": {"label": "Pipeline", "display": "input", "type": "any"},
        "operation": {"label": "Operation", "type": "string", "options": ["fuse", "unfuse"], "default": "fuse"},
        "adapter_names": {"label": "Adapters", "type": "string", "default": ""},
        "components": {"label": "Components", "type": "string", "default": ""},
        "scale": {"label": "Scale", "type": "float", "default": 1.0},
        "safe_fusing": {"label": "Safe Fusing", "type": "bool", "default": True},
        "output": {"label": "Pipeline", "display": "output", "type": "any"},
    }

    def execute(self, **kwargs):
        pipeline = kwargs.get("pipeline")
        operation = str(kwargs.get("operation") or "fuse")
        components = _string_list(kwargs.get("components")) or None
        if operation == "fuse":
            method = getattr(pipeline, "fuse_lora", None)
            if not callable(method):
                raise ValueError("This pipeline does not support LoRA fusion.")
            method(
                components=components,
                adapter_names=_string_list(kwargs.get("adapter_names")) or None,
                lora_scale=float(kwargs.get("scale") or 1.0),
                safe_fusing=bool(kwargs.get("safe_fusing", True)),
            )
        elif operation == "unfuse":
            method = getattr(pipeline, "unfuse_lora", None)
            if not callable(method):
                raise ValueError("This pipeline does not support LoRA unfusing.")
            method(components=components)
        else:
            raise ValueError(f"Unsupported LoRA fusion operation {operation!r}.")
        return {"output": pipeline}


class LoRAUnloadReset(NodeBase):
    label = "Unload / Reset Diffusers LoRA"
    category = "Diffusers Adapters"
    params = {
        "pipeline": {"label": "Pipeline", "display": "input", "type": "any"},
        "adapter_names": {"label": "Adapters (empty = all)", "type": "string", "default": ""},
        "output": {"label": "Pipeline", "display": "output", "type": "any"},
    }

    def execute(self, **kwargs):
        pipeline = kwargs.get("pipeline")
        names = _string_list(kwargs.get("adapter_names"))
        if names:
            delete = getattr(pipeline, "delete_adapters", None)
            if not callable(delete):
                raise ValueError("This pipeline cannot delete individual LoRA adapters.")
            for name in names:
                delete(name)
        else:
            unload = getattr(pipeline, "unload_lora_weights", None)
            if not callable(unload):
                raise ValueError("This pipeline cannot unload LoRA weights.")
            unload()
        return {"output": pipeline}


class LoRAMergeArtifact(NodeBase):
    """Merge loaded adapters with PEFT and save a reloadable adapter artifact."""

    label = "Merge Diffusers LoRA Artifact"
    category = "Diffusers Adapters"
    resizable = True
    params = {
        "pipeline": {"label": "Pipeline", "display": "input", "type": "any"},
        "component": {"label": "Component", "type": "string", "options": ["transformer", "unet"], "default": "transformer"},
        "adapter_names": {"label": "Adapters", "type": "string", "default": ""},
        "weights": {"label": "Weights", "type": "string", "default": ""},
        "merge_method": {
            "label": "Method",
            "type": "string",
            "options": ["cat", "linear", "svd", "ties", "ties_svd", "dare_ties", "dare_linear", "magnitude_prune"],
            "default": "ties",
        },
        "density": {"label": "Density", "type": "float", "default": 0.5, "min": 0.01, "max": 1.0},
        "merged_name": {"label": "Merged Adapter Name", "type": "string", "default": "merged"},
        "output_directory": {"label": "Output Directory", "type": "string", "default": "{PATH:models}/merged_lora_{HASH:6}"},
        "artifact_path": {"label": "Artifact Path", "display": "output", "type": "string"},
        "manifest": {"label": "Manifest", "display": "output", "type": "string"},
    }

    def execute(self, **kwargs):
        from utils.paths import parse_filename

        pipeline = kwargs.get("pipeline")
        component_name = str(kwargs.get("component") or "transformer")
        component = getattr(pipeline, component_name, None)
        add_weighted = getattr(component, "add_weighted_adapter", None)
        save = getattr(component, "save_pretrained", None)
        if not callable(add_weighted) or not callable(save):
            raise ValueError(
                f"Pipeline component {component_name!r} does not expose PEFT weighted-adapter merge and save APIs."
            )
        names = _string_list(kwargs.get("adapter_names"))
        weights = [float(value) for value in _string_list(kwargs.get("weights"))]
        if not names or len(names) != len(weights):
            raise ValueError("LoRA merge needs the same non-zero number of adapter names and weights.")
        loaded = active_adapter_names(pipeline)
        missing = [name for name in names if name not in loaded]
        if missing:
            raise ValueError(f"LoRA merge adapters are not loaded: {', '.join(missing)}.")
        merged_name = str(kwargs.get("merged_name") or "merged").strip()
        method = str(kwargs.get("merge_method") or "ties")
        density = float(kwargs.get("density") or 0.5)
        merge_kwargs = {"combination_type": method}
        if method.startswith(("ties", "dare", "magnitude_prune")):
            merge_kwargs["density"] = density
        add_weighted(names, weights, merged_name, **merge_kwargs)

        destination = Path(parse_filename(kwargs.get("output_directory") or "{PATH:models}/merged_lora_{HASH:6}"))
        destination.mkdir(parents=True, exist_ok=False)
        save(str(destination), safe_serialization=True, selected_adapters=[merged_name])
        manifest = {
            "schema_version": 1,
            "component": component_name,
            "source_adapters": names,
            "weights": weights,
            "merged_adapter": merged_name,
            "combination_type": method,
            "density": density if "density" in merge_kwargs else None,
            "artifact_path": str(destination),
        }
        (destination / "modiff_merge_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {"artifact_path": str(destination), "manifest": json.dumps(manifest, sort_keys=True)}


class LoRAComparisonJobs(NodeBase):
    label = "Build LoRA Comparison Jobs"
    category = "Diffusers Adapters"
    params = {
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
        "mixes": {"label": "Mixes (JSON)", "display": "textarea", "type": "text", "default": "[]"},
        "seed": {"label": "Seed", "type": "int", "default": 0},
        "jobs": {"label": "Comparison Jobs", "display": "output", "type": "collection"},
    }

    def execute(self, **kwargs):
        raw = kwargs.get("mixes") or "[]"
        mixes = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(mixes, list) or any(not isinstance(item, dict) for item in mixes):
            raise ValueError("LoRA comparison mixes must be a JSON array of objects.")
        if not mixes:
            raise ValueError("Add at least one LoRA comparison mix.")
        return {
            "jobs": [
                {"index": index, "prompt": str(kwargs.get("prompt") or ""), "seed": int(kwargs.get("seed") or 0), "mix": mix}
                for index, mix in enumerate(mixes)
            ]
        }
