# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
import json
import hashlib
import os
from pathlib import Path

from modiff.NodeBase import NodeBase


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
                "filter": {
                    "hub": {"className": [""]},
                    "local": {"className": [""]},
                },
            },
        },
        "weight_name": {
            "label": "Weight Name",
            "type": "string",
        },
        "expected_sha256": {
            "label": "Expected SHA-256",
            "type": "string",
            "default": "",
            "description": "Optional immutable hash for the selected adapter weight file.",
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
        expected_sha256="",
        scheduler_class="",
        scheduler_config="{}",
    ):
        if isinstance(model, dict):
            lora_path = model.get("value")
            if not lora_path:
                raise ValueError("A LoRA model is required.")
            filename = os.path.splitext(os.path.basename(lora_path))[0]
            if model.get("source") == "hub" and lora_path:
                from utils.huggingface import cached_file_path

                repo_id = lora_path
                if not weight_name:
                    parts = lora_path.split("/")
                    if len(parts) >= 3:
                        repo_id, weight_name = "/".join(parts[:2]), "/".join(parts[2:])
                if not weight_name:
                    raise ValueError("A Hub LoRA requires a pinned weight_name for app-managed installation.")
                cached = cached_file_path(repo_id, weight_name)
                if not cached:
                    raise FileNotFoundError(
                        f"LoRA {repo_id}/{weight_name} is not installed. Install the pinned file through Model Manager first."
                    )
                cached_path = Path(cached)
                lora_path = str(cached_path.parent)
                weight_name = cached_path.name
                expected_sha256 = str(expected_sha256 or "").strip().lower().removeprefix("sha256:")
                if expected_sha256:
                    digest = hashlib.sha256()
                    with cached_path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                            digest.update(chunk)
                    if digest.hexdigest() != expected_sha256:
                        raise ValueError(
                            f"LoRA {repo_id}/{weight_name} failed its pinned SHA-256 verification. "
                            "Repair the adapter through Model Manager before running this graph."
                        )
        else:
            lora_path = None
            filename = ""

        adapter_name = f"{filename}_{self.node_id}"

        if isinstance(scheduler_config, str):
            try:
                scheduler_config = json.loads(scheduler_config or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(f"LoRA scheduler config must be valid JSON: {exc}") from exc
        if not isinstance(scheduler_config, dict):
            raise TypeError("LoRA scheduler config must decode to a JSON object.")

        # Return the LoRA configuration directly, including optional generic
        # inference metadata for distilled adapters. Models without that
        # metadata continue to use the repository scheduler unchanged.
        return {
            "lora": {
                "lora_path": lora_path,
                "weight_name": weight_name,
                "adapter_name": adapter_name,
                "scale": scale,
                "scheduler_class": scheduler_class or None,
                "scheduler_config": scheduler_config,
            }
        }
