# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
from pathlib import PurePosixPath

from modiff.NodeBase import NodeBase
from modiff.auxiliary_lora import build_lora_descriptor, generated_lora_adapter_name


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
            adapter_name=generated_lora_adapter_name(name_seed, self.node_id),
            scale=scale,
            scheduler_class=scheduler_class,
            scheduler_config=scheduler_config,
        )
        return {"lora": descriptor}
