import json
import logging
from pathlib import Path
from typing import Any

from modiff.NodeBase import NodeBase
from modiff.config import CONFIG
from modules.DiffusersImage.main import build_pipeline_quantization_config, normalize_component_list, pipeline_class_from_name, repo_value
from utils.huggingface import local_files_only
from utils.torch_utils import str_to_dtype

logger = logging.getLogger("modiff")


class QuantizeDiffusersComponents(NodeBase):
    """Create a local Diffusers artifact with selected quantized components."""

    label = "Quantize Diffusers Components"
    category = "Model Artifact"
    resizable = True
    params = {
        "model_id": {
            "label": "Source Model",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": ""},
            "fieldOptions": {"noValidation": True, "sources": ["hub", "local"]},
        },
        "pipeline_class": {
            "label": "Pipeline Class",
            "type": "string",
            "default": "FluxPipeline",
            "fieldOptions": {"noValidation": True},
        },
        "dtype": {
            "label": "DType",
            "type": "string",
            "options": ["float32", "float16", "bfloat16"],
            "default": "bfloat16",
        },
        "quantization_mode": {
            "label": "Quantization",
            "type": "string",
            "options": ["bnb_4bit", "bnb_8bit", "quanto_float8", "torchao_float8"],
            "default": "quanto_float8",
        },
        "quantized_components": {
            "label": "Components",
            "type": "string",
            "display": "select",
            "options": ["transformer", "text_encoder", "text_encoder_2", "vae"],
            "fieldOptions": {"multiple": True},
            "default": ["transformer"],
        },
        "output_dir": {
            "label": "Output folder",
            "type": "str",
            "default": "{PATH:models}/quantized/{MODEL}_{QUANT}",
        },
        "safe_serialization": {"label": "Safe serialization", "type": "bool", "default": True},
        "artifact_path": {"label": "Artifact Path", "display": "output", "type": "str"},
        "manifest": {"label": "Manifest", "display": "output", "type": "str"},
    }

    def execute(self, **kwargs):
        model_id = repo_value(kwargs.get("model_id"))
        if not model_id:
            raise ValueError("Quantize Diffusers Components needs a source model.")
        pipeline_class_name = str(kwargs.get("pipeline_class") or "FluxPipeline")
        pipeline_class = pipeline_class_from_name(pipeline_class_name)
        dtype = str_to_dtype(kwargs.get("dtype") or "bfloat16")
        quantization_mode = str(kwargs.get("quantization_mode") or "quanto_float8")
        quantized_components = normalize_component_list(kwargs.get("quantized_components"))
        if not quantized_components:
            raise ValueError("Select at least one quantized component.")
        quant_config = build_pipeline_quantization_config(quantization_mode, quantized_components, dtype)
        if quant_config is None:
            raise ValueError(f"Quantization mode {quantization_mode} is not available.")

        output_template = str(kwargs.get("output_dir") or "{PATH:models}/quantized/{MODEL}_{QUANT}")
        safe_name = model_id.replace("/", "--")
        output_text = output_template.replace("{MODEL}", safe_name).replace("{QUANT}", quantization_mode)
        output_text = output_text.replace("{PATH:models}", str(Path(CONFIG.paths["data"]) / "models"))
        output_path = Path(output_text)
        if not output_path.is_absolute():
            output_path = Path(CONFIG.paths["data"]) / output_path
        output_path.mkdir(parents=True, exist_ok=True)

        self.progress(-1, phase="loading", message="Loading source model with quantization config")
        pipeline = pipeline_class.from_pretrained(
            model_id,
            torch_dtype=dtype,
            quantization_config=quant_config,
            local_files_only=local_files_only(model_id),
        )
        self.progress(-1, phase="saving", message="Saving quantized Diffusers artifact")
        pipeline.save_pretrained(output_path, safe_serialization=bool(kwargs.get("safe_serialization", True)))
        manifest = {
            "source_model": model_id,
            "pipeline_class": pipeline_class_name,
            "dtype": str(dtype),
            "quantization_mode": quantization_mode,
            "quantized_components": quantized_components,
            "artifact_path": str(output_path),
        }
        manifest_path = output_path / "modiff_quantization_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {"artifact_path": str(output_path), "manifest": str(manifest_path)}
