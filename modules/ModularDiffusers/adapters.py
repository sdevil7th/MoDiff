# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
from pathlib import PurePosixPath
from copy import deepcopy

from modiff.NodeBase import NodeBase
from modiff.auxiliary_lora import build_lora_descriptor, generated_lora_adapter_name, resolve_lora_descriptors


class Lora(NodeBase):
    label = "Lora"
    category = "adapters"
    resizable = True
    params = {
        "model": {
            "label": "Model",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": ""},
            "fieldOptions": {
                "noValidation": True,
                "sources": ["hub", "local"],
            },
        },
        "weight_name": {
            "label": "Weight Name",
            "type": "string",
        },
        "revision": {
            "label": "Revision",
            "type": "string",
            "default": "",
            "description": "Required exact commit for a Hub LoRA; unused for a local Safetensors file.",
        },
        "expected_sha256": {
            "label": "Expected SHA-256",
            "type": "string",
            "default": "",
            "description": "Required for Hub weights; local weights are hashed when this descriptor is created.",
        },
        "scale": {
            "label": "Scale",
            "type": "float",
            "display": "slider",
            "default": 1.0,
            "min": -20,
            "max": 20,
            "step": 0.1,
        },
        "adapter_name": {
            "label": "Adapter Name",
            "type": "string",
            "default": "",
            "description": "Optional exact name; an empty value retains the stable generated adapter identity.",
        },
        "previous_loras": {
            "label": "Previous LoRAs",
            "display": "input",
            "type": "custom_lora",
            "description": "Append this adapter to the connected ordered descriptor or descriptor list.",
        },
        "scheduler_class": {
            "label": "Scheduler Class",
            "type": "string",
            "value": "",
        },
        "scheduler_config": {
            "label": "Scheduler Config (JSON)",
            "type": "text",
            "display": "textarea",
            "value": "{}",
        },
        "lora": {
            "label": "Lora",
            "type": "custom_lora",
            "display": "output",
        },
    }

    def execute(
        self,
        model,
        scale,
        weight_name=None,
        revision="",
        expected_sha256="",
        scheduler_class="",
        scheduler_config="{}",
        adapter_name="",
        previous_loras=None,
    ):
        if not isinstance(model, dict) or not model.get("value"):
            raise ValueError("A LoRA model is required and must explicitly select a hub or local source.")
        requested_weight = str(weight_name or "")
        name_seed = PurePosixPath(requested_weight.replace("\\", "/")).stem
        if not name_seed:
            name_seed = PurePosixPath(str(model.get("value") or "").replace("\\", "/")).stem or "lora"
        descriptor = build_lora_descriptor(
            selection=model,
            weight_name=weight_name,
            revision=revision,
            expected_sha256=expected_sha256,
            adapter_name=(generated_lora_adapter_name(name_seed, self.node_id)
                          if adapter_name is None or adapter_name == "" else adapter_name),
            scale=scale,
            scheduler_class=scheduler_class,
            scheduler_config=scheduler_config,
        )
        if previous_loras is not None:
            previous = previous_loras if isinstance(previous_loras, list) else [previous_loras]
            combined = [*previous, descriptor]
            # Validate hashes, exact bytes, adapter-name collisions and bounds
            # before publishing a new chain. The loader revalidates it again.
            resolve_lora_descriptors(combined)
            return {"lora": deepcopy(combined)}
        return {"lora": descriptor}
