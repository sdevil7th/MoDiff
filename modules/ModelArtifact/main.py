import json
import logging
import gc
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from modiff.NodeBase import NodeBase
from modiff.config import CONFIG
from modules.DiffusersImage.main import normalize_component_list, pipeline_class_from_name, repo_value
from modules.DiffusersRuntime.main import (
    assert_runtime_quantization_full_residency,
    build_quantization_config_v2,
)
from utils.huggingface import local_files_only
from utils.torch_utils import str_to_dtype

logger = logging.getLogger("modiff")


BLACKWELL_MODES = {"torchao_mxfp8", "torchao_nvfp4"}


def _json_object(value: Any, label: str) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON: {exc.msg}.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return parsed


def _string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    values = value.replace("\n", ",").split(",") if isinstance(value, str) else value
    return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _selective_exclusions(model_id: str) -> list[str]:
    lowered = model_id.lower()
    common = ["embed", "norm_out", "proj_out"]
    if "qwen" in lowered:
        return [
            *common,
            "img_in",
            "txt_in",
            "img_mod",
            "txt_mod",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
            "txt_mlp",
        ]
    if "ltx" in lowered:
        return [
            *common,
            "patch_embed",
            "proj_in",
            "caption_projection",
            "adaln_single",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        ]
    return common


def _resolve_source(model_id: str, revision: str | None) -> tuple[str | None, dict[str, Any]]:
    source = Path(model_id).expanduser()
    if source.exists():
        return None, {"source": "local", "license": None, "requested_revision": revision}
    from huggingface_hub import HfApi

    info = HfApi(token=CONFIG.hf.get("token"), library_name="MoDiff").model_info(
        model_id,
        revision=revision,
        files_metadata=True,
    )
    sha = str(getattr(info, "sha", "") or "").strip()
    if not sha:
        raise RuntimeError("Hugging Face did not return an immutable source revision.")
    card = getattr(info, "card_data", None)
    license_name = getattr(card, "license", None) if card is not None else None
    return sha, {"source": "hub", "license": license_name}


def _file_checksums(root: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "modiff_quantization_manifest.json":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        checksums[path.relative_to(root).as_posix()] = digest.hexdigest()
    return checksums


def _blackwell_admission(mode: str) -> None:
    if mode not in BLACKWELL_MODES:
        return
    import torch

    if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability(0)) < (10, 0):
        raise ValueError(f"{mode.removeprefix('torchao_').upper()} artifact creation requires NVIDIA Blackwell (SM 10.0+).")
    if torch.are_deterministic_algorithms_enabled():
        raise ValueError("Blackwell artifact creation is disabled during deterministic execution.")


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
        "source_revision": {"label": "Source Revision", "type": "string", "default": "main"},
        "dtype": {
            "label": "DType",
            "type": "string",
            "options": ["float32", "float16", "bfloat16"],
            "default": "bfloat16",
        },
        "quantization_mode": {
            "label": "Quantization",
            "type": "string",
            "options": [
                "bnb_4bit",
                "bnb_8bit",
                "quanto_float8",
                "quanto_int8",
                "torchao_float8",
                "torchao_int8_weight_only",
                "torchao_mxfp8",
                "torchao_nvfp4",
            ],
            "default": "",
        },
        "excluded_modules": {
            "label": "Preserved Modules",
            "type": "string",
            "default": "",
            "description": "Comma-separated module names kept at source precision.",
        },
        "component_overrides": {
            "label": "Component Policies (JSON)",
            "display": "textarea",
            "type": "text",
            "default": "{}",
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
        "smoke_generation": {
            "label": "Smoke Generation (JSON)",
            "display": "textarea",
            "type": "text",
            "default": "{}",
            "description": "Optional small deterministic pipeline call used before qualification.",
        },
        "smoke_seed": {"label": "Smoke Seed", "type": "int", "default": 0},
        "quality_comparison": {
            "label": "Quality Comparison (JSON)",
            "display": "textarea",
            "type": "text",
            "default": "{}",
        },
        "artifact_path": {"label": "Artifact Path", "display": "output", "type": "str"},
        "manifest": {"label": "Manifest", "display": "output", "type": "str"},
    }

    def execute(self, **kwargs):
        model_id = repo_value(kwargs.get("model_id"))
        if not model_id:
            raise ValueError("Quantize Diffusers Components needs a source model.")
        pipeline_class_name = str(kwargs.get("pipeline_class") or "FluxPipeline")
        quantization_mode = str(kwargs.get("quantization_mode") or "").strip()
        if not quantization_mode:
            raise ValueError("Select an installed quantization backend before creating an optimized artifact.")
        pipeline_class = pipeline_class_from_name(pipeline_class_name)
        dtype = str_to_dtype(kwargs.get("dtype") or "bfloat16")
        _blackwell_admission(quantization_mode)
        quantized_components = normalize_component_list(kwargs.get("quantized_components"))
        if not quantized_components:
            raise ValueError("Select at least one quantized component.")
        requested_revision = str(kwargs.get("source_revision") or "main").strip() or None
        pinned_revision, provenance = _resolve_source(model_id, requested_revision)
        excluded_modules = _string_list(kwargs.get("excluded_modules"))
        if quantization_mode in BLACKWELL_MODES:
            excluded_modules = list(dict.fromkeys([*excluded_modules, *_selective_exclusions(model_id)]))
        component_overrides = _json_object(kwargs.get("component_overrides"), "Component policies")
        quant_config, quantization_summary = build_quantization_config_v2(
            backend=quantization_mode,
            components=quantized_components,
            dtype=dtype,
            excluded_modules=excluded_modules,
            component_overrides=component_overrides,
        )
        if quant_config is None:
            raise ValueError(f"Quantization mode {quantization_mode} is not available.")

        residency = assert_runtime_quantization_full_residency(
            model_id=model_id,
            revision=pinned_revision,
            quantization_config=quant_config,
            device="cuda:0",
            offload_mode="none",
            device_map="cuda",
        )

        output_template = str(kwargs.get("output_dir") or "{PATH:models}/quantized/{MODEL}_{QUANT}")
        safe_name = model_id.replace("/", "--")
        output_text = output_template.replace("{MODEL}", safe_name).replace("{QUANT}", quantization_mode)
        output_text = output_text.replace("{PATH:models}", str(Path(CONFIG.paths["data"]) / "models"))
        output_path = Path(output_text)
        if not output_path.is_absolute():
            output_path = Path(CONFIG.paths["data"]) / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            raise FileExistsError(f"Artifact output already exists: {output_path}. Choose a new versioned folder.")
        staging_path = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.partial-", dir=output_path.parent))
        pipeline = None
        try:
            self.progress(-1, phase="loading", message="Loading source model with quantization config")
            pipeline = pipeline_class.from_pretrained(
                model_id,
                revision=pinned_revision,
                torch_dtype=dtype,
                quantization_config=quant_config,
                device_map="cuda",
                local_files_only=local_files_only(model_id),
            )
            smoke_kwargs = _json_object(kwargs.get("smoke_generation"), "Smoke generation")
            smoke = {"status": "not_requested", "seed": int(kwargs.get("smoke_seed") or 0)}
            if smoke_kwargs:
                import torch

                smoke_kwargs["generator"] = torch.Generator(device="cuda").manual_seed(smoke["seed"])
                self.progress(-1, phase="validating", message="Running deterministic smoke generation")
                result = pipeline(**smoke_kwargs)
                if result is None:
                    raise RuntimeError("Smoke generation returned no result.")
                smoke["status"] = "passed"

            self.progress(-1, phase="saving", message="Saving quantized Diffusers artifact")
            pipeline.save_pretrained(staging_path, safe_serialization=bool(kwargs.get("safe_serialization", True)))
            checksums = _file_checksums(staging_path)
            quality = _json_object(kwargs.get("quality_comparison"), "Quality comparison")
            manifest = {
                "schema_version": 1,
                "source_model": model_id,
                "source_requested_revision": requested_revision,
                "source_revision": pinned_revision,
                "source": provenance,
                "pipeline_class": pipeline_class_name,
                "dtype": str(dtype),
                "quantization_mode": quantization_mode,
                "quantized_components": quantized_components,
                "quantization_summary": quantization_summary,
                "preserved_modules": excluded_modules,
                "full_residency_admission": residency,
                "smoke_generation": smoke,
                "quality_comparison": quality or {"status": "not_provided"},
                "qualified": smoke["status"] == "passed" and bool(quality),
                "checksums": checksums,
                "artifact_path": str(output_path),
            }
            (staging_path / "modiff_quantization_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(staging_path, output_path)
        finally:
            if pipeline is not None:
                del pipeline
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            if staging_path.exists():
                shutil.rmtree(staging_path)
        manifest_path = output_path / "modiff_quantization_manifest.json"
        return {"artifact_path": str(output_path), "manifest": str(manifest_path)}
