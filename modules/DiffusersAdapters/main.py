"""Model-neutral Diffusers LoRA inspection and lifecycle nodes."""

import json
from pathlib import Path
from typing import Any

from modiff.NodeBase import NodeBase
from modiff.auxiliary_lora import resolve_lora_descriptor, resolve_lora_descriptors


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
    resolved = resolve_lora_descriptors(adapters)
    if any(item.scheduler_class_name is not None for item in resolved):
        raise ValueError(
            "Diffusers LoRA Stack / Mix cannot apply scheduler-bearing descriptors; "
            "connect them through the Modular models loader instead."
        )
    load = getattr(pipeline, "load_lora_weights", None)
    activate = getattr(pipeline, "set_adapters", None)
    if not callable(load) or not callable(activate):
        raise ValueError("This Diffusers pipeline does not expose the multi-adapter LoRA API.")
    loaded = active_adapter_names(pipeline)
    identities = dict(getattr(pipeline, "_modiff_lora_identities", {}) or {})
    replace = {
        item.adapter_name
        for item in resolved
        if item.adapter_name in loaded and identities.get(item.adapter_name) != item.descriptor_sha256
    }
    delete = getattr(pipeline, "delete_adapters", None)
    if replace and not callable(delete):
        raise ValueError("This Diffusers pipeline cannot replace a loaded adapter with a new immutable identity.")

    names = []
    weights = []
    for item in resolved:
        name = item.adapter_name
        if name in replace:
            delete(name)
            loaded.remove(name)
            identities.pop(name, None)
        if name not in loaded:
            load(
                str(item.load_directory),
                weight_name=item.weight_name,
                adapter_name=name,
                use_safetensors=True,
            )
            loaded.add(name)
            identities[name] = item.descriptor_sha256
        names.append(name)
        weights.append(item.scale)
    activate(names, weights)
    pipeline._modiff_lora_identities = identities
    return {"adapter_names": names, "adapter_weights": weights}


class LoRAInspectValidate(NodeBase):
    label = "Inspect / Validate Diffusers LoRA"
    category = "Diffusers Adapters"
    resizable = True
    params = {
        "adapter": {
            "label": "Adapter",
            "display": "input",
            "type": "custom_lora",
            "required": True,
        },
        "base_model": {"label": "Expected Base Model", "type": "string", "default": ""},
        "report": {"label": "Inspection", "display": "output", "type": "string"},
        "compatibility": {"label": "Compatibility", "display": "output", "type": "string"},
        "rank": {"label": "Maximum Rank", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        adapter = resolve_lora_descriptor(kwargs.get("adapter"))
        path = adapter.load_directory / adapter.weight_name
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
        adapter = resolve_lora_descriptor(kwargs.get("replacement"))
        if adapter.scheduler_class_name is not None:
            raise ValueError(
                "Diffusers LoRA hotswap cannot apply a scheduler-bearing descriptor; "
                "connect it through the Modular models loader instead."
            )
        slot = str(kwargs.get("slot_name") or "default_0")
        if slot not in active_adapter_names(pipeline):
            raise ValueError(f"LoRA hotswap slot {slot!r} is not loaded. Load the initial adapter before hotswapping it.")
        load = getattr(pipeline, "load_lora_weights", None)
        activate = getattr(pipeline, "set_adapters", None)
        if not callable(load) or not callable(activate):
            raise ValueError("This Diffusers pipeline does not expose the reviewed LoRA hotswap API.")
        load(
            str(adapter.load_directory),
            weight_name=adapter.weight_name,
            adapter_name=slot,
            hotswap=True,
            use_safetensors=True,
        )
        activate([slot], [adapter.scale])
        identities = dict(getattr(pipeline, "_modiff_lora_identities", {}) or {})
        identities[slot] = adapter.descriptor_sha256
        pipeline._modiff_lora_identities = identities
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
