"""Bounded, local-only text and multimodal generation with Transformers."""

from __future__ import annotations

import hashlib
import json
from math import isfinite
from pathlib import Path
import re
from typing import Any

import numpy as np

from modiff.NodeBase import NodeBase
from utils.huggingface import validate_hf_repo_id
from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST, str_to_dtype


SCHEMA_VERSION = 1
IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
MODEL_SELECTION_KEYS = frozenset({"source", "value"})
MODEL_HANDLE_KEYS = frozenset({"schemaVersion", "kind", "model", "preprocessor", "receipt"})
ANY_TO_ANY_HANDLE_KEYS = frozenset(
    {"schemaVersion", "kind", "adapter", "model", "preprocessor", "pipeline", "receipt"}
)
SECURITY_CONTRACT = {
    "localFilesOnly": True,
    "trustRemoteCode": False,
    "safetensorsOnly": True,
    "hostedInference": False,
}

MAX_PROMPT_CHARACTERS = 65_536
MAX_PROMPT_UTF8_BYTES = 262_144
MAX_RENDERED_PROMPT_CHARACTERS = 131_072
MAX_INPUT_TOKENS = 32_768
MAX_NEW_TOKENS = 2_048
MAX_OUTPUT_CHARACTERS = 262_144
MAX_OUTPUT_UTF8_BYTES = 1_048_576
MAX_IMAGES = 16
MAX_VIDEO_FRAMES = 128
MAX_MEDIA_SIDE = 4_096
MAX_MEDIA_PIXELS = 16_777_216
MAX_TOTAL_MEDIA_PIXELS = 67_108_864
MAX_PROCESSED_TENSOR_ELEMENTS = 268_435_456
MAX_LOCAL_CONFIG_BYTES = 2 * 1024 * 1024
MAX_LOCAL_CONFIG_NODES = 100_000
MAX_RECEIPT_BYTES = 16_384
MAX_ANY_TO_ANY_IMAGE_TOKENS = 4_096

# This registry is intentionally finite. An adapter becomes executable only after
# the pinned production source exposes enough family-specific information to
# preflight its work and normalize every supported output truthfully.
ANY_TO_ANY_ADAPTER_CONTRACTS = {
    "janus": {
        "schemaVersion": SCHEMA_VERSION,
        "adapterId": "janus-v1",
        "status": "executable",
        "modelClass": "JanusForConditionalGeneration",
        "processorClass": "JanusProcessor",
        "inputModalities": ["image", "text"],
        "outputModalities": ["image", "text"],
        "evidence": [
            "transformers@5.14.1:models/janus/configuration_janus.py:31-61",
            "transformers@5.14.1:models/janus/modeling_janus.py:1180-1349",
            "transformers@5.14.1:models/janus/processing_janus.py:122-151",
        ],
    },
    "qwen2_5_omni": {
        "schemaVersion": SCHEMA_VERSION,
        "adapterId": "qwen2-5-omni-v1",
        "status": "contract-only",
        "modelClass": "Qwen2_5OmniForConditionalGeneration",
        "processorClass": "Qwen2_5OmniProcessor",
        "inputModalities": ["audio", "image", "text", "video"],
        "outputModalities": ["audio", "text"],
        "audioOutputSampleRate": 24_000,
        "blocker": (
            "The official 5.14.1 loader unconditionally reads spk_dict.pt with torch.load(weights_only=True); "
            "that family cannot satisfy MoDiff's safetensors-only graph-loading contract."
        ),
        "evidence": [
            "transformers@5.14.1:models/qwen2_5_omni/modeling_qwen2_5_omni.py:3760-3801",
            "transformers@5.14.1:models/qwen2_5_omni/modeling_qwen2_5_omni.py:3806-3898",
            "transformers@5.14.1:models/qwen2_5_omni/processing_qwen2_5_omni.py:323-352",
            "transformers@5.14.1:docs/source/en/model_doc/qwen2_5_omni.md:90-98",
        ],
    },
}


def _model_selection(value: Any) -> dict[str, str]:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError("A Transformers model selection is required; no repository is selected by default.")
    if isinstance(value, str):
        source, selected = "hub", value.strip()
    elif isinstance(value, dict) and set(value).issubset(MODEL_SELECTION_KEYS):
        source = str(value.get("source") or "").strip().casefold()
        selected = str(value.get("value") or "").strip()
    else:
        raise ValueError("Transformers model selection must be a repository ID or hub/local selection object.")
    if not selected or "\x00" in selected or len(selected) > 4_096:
        raise ValueError("Transformers model selection is empty or invalid.")
    if source == "hub":
        if selected.count("/") != 1:
            raise ValueError("Transformers model repository must use the namespace/repository form.")
        validate_hf_repo_id(selected)
        return {"source": "hub", "value": selected}
    if source != "local":
        raise ValueError("Transformers model source must be exactly hub or local.")
    try:
        resolved = Path(selected).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("Local Transformers model directory does not exist.") from error
    if not resolved.is_dir():
        raise ValueError("Local Transformers model selection must be a directory.")
    return {"source": "local", "value": str(resolved)}


def _model_revision(selection: dict[str, str], value: Any) -> str | None:
    if selection["source"] == "local":
        if value not in (None, ""):
            raise ValueError("A local Transformers model must not carry a Hub revision.")
        return None
    revision = str(value or "").strip()
    if not IMMUTABLE_REVISION.fullmatch(revision):
        raise ValueError("Hub model revision must be an exact lowercase 40-character commit SHA.")
    return revision


def _local_config_receipt(directory: str) -> dict[str, Any]:
    config_path = Path(directory, "config.json")
    try:
        if config_path.is_symlink() or not config_path.is_file():
            raise ValueError("Local Transformers model requires a regular, non-symlink config.json.")
        size = config_path.stat().st_size
    except OSError as error:
        raise ValueError("Local Transformers model config.json could not be inspected.") from error
    if size <= 0 or size > MAX_LOCAL_CONFIG_BYTES:
        raise ValueError("Local Transformers model config.json has an invalid or excessive size.")
    try:
        payload = config_path.read_bytes()
    except OSError as error:
        raise ValueError("Local Transformers model config.json could not be read.") from error
    if len(payload) != size or len(payload) > MAX_LOCAL_CONFIG_BYTES:
        raise ValueError("Local Transformers model config.json changed while being inspected.")
    try:
        config = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Local Transformers model config.json must be valid bounded JSON.") from error
    if not isinstance(config, dict):
        raise ValueError("Local Transformers model config.json must contain an object.")
    pending = [config]
    visited = 0
    while pending:
        item = pending.pop()
        visited += 1
        if visited > MAX_LOCAL_CONFIG_NODES:
            raise ValueError("Local Transformers model config.json is too complex.")
        if isinstance(item, dict):
            if any(str(key).casefold() == "auto_map" for key in item):
                raise ValueError("Local Transformers model config.json must not declare auto_map remote code.")
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type.strip() or len(model_type) > 128:
        model_type = None
    raw_architectures = config.get("architectures")
    architectures: list[str] = []
    if isinstance(raw_architectures, list):
        for architecture in raw_architectures[:32]:
            if isinstance(architecture, str) and architecture.strip() and len(architecture) <= 256:
                architectures.append(architecture.strip())
    return {
        "configSha256": f"sha256:{hashlib.sha256(payload).hexdigest()}",
        "configByteSize": len(payload),
        "modelType": model_type,
        "architectures": architectures,
    }


def _bounded_string(
    value: Any,
    *,
    field: str,
    maximum_characters: int = MAX_PROMPT_CHARACTERS,
    maximum_bytes: int = MAX_PROMPT_UTF8_BYTES,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text.")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain NUL characters.")
    normalized = value.strip()
    if not allow_empty and not normalized:
        raise ValueError(f"{field} must not be empty.")
    if len(normalized) > maximum_characters or len(normalized.encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{field} exceeds its bounded text size.")
    return normalized


def _bounded_int(value: Any, *, field: str, default: int, minimum: int, maximum: int) -> int:
    raw = default if value is None else value
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{field} must be an integer.")
    if raw < minimum or raw > maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}.")
    return raw


def _bounded_float(value: Any, *, field: str, default: float, minimum: float, maximum: float) -> float:
    raw = default if value is None else value
    if isinstance(raw, bool):
        raise ValueError(f"{field} must be a number.")
    try:
        parsed = float(raw)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field} must be a number.") from error
    if not isfinite(parsed) or parsed < minimum or parsed > maximum:
        raise ValueError(f"{field} must be finite and between {minimum} and {maximum}.")
    return parsed


def _bounded_bool(value: Any, *, field: str, default: bool) -> bool:
    raw = default if value is None else value
    if not isinstance(raw, bool):
        raise ValueError(f"{field} must be a boolean.")
    return raw


def _normalized_device(value: Any) -> str:
    selected = str(value or DEFAULT_DEVICE).strip()
    if selected not in DEVICE_LIST:
        raise ValueError("Transformers device must be one of the devices discovered by MoDiff.")
    return selected


def _normalized_dtype(value: Any) -> tuple[str, Any]:
    selected = str(value or "float32").strip().casefold()
    if selected not in {"float32", "float16", "bfloat16"}:
        raise ValueError("Transformers dtype must be exactly float32, float16, or bfloat16.")
    return selected, str_to_dtype(selected)


def _canonical_json(value: Any) -> bytes:
    try:
        encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Transformers model receipt is not canonical JSON.") from error
    if len(encoded) > MAX_RECEIPT_BYTES:
        raise ValueError("Transformers model receipt exceeded its size limit.")
    return encoded


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _sealed_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    sealed = _json_copy(receipt)
    sealed["receiptDigest"] = f"sha256:{hashlib.sha256(_canonical_json(sealed)).hexdigest()}"
    return sealed


def _receipt_digest_is_valid(receipt: Any) -> bool:
    if not isinstance(receipt, dict):
        return False
    digest = receipt.get("receiptDigest")
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        return False
    unsigned = dict(receipt)
    unsigned.pop("receiptDigest", None)
    expected = f"sha256:{hashlib.sha256(_canonical_json(unsigned)).hexdigest()}"
    return digest == expected


def _model_source_receipt(selection: dict[str, str], revision: str | None) -> dict[str, Any]:
    if selection["source"] == "hub":
        return {"kind": "hub", "repository": selection["value"], "revision": revision}
    return {"kind": "local", "path": selection["value"], **_local_config_receipt(selection["value"])}


def _class_name(value: Any) -> str:
    name = type(value).__name__
    return name[:256] if isinstance(name, str) else "unknown"


def _config_metadata(model: Any) -> tuple[str | None, list[str]]:
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    if not isinstance(model_type, str) or not model_type.strip() or len(model_type) > 128:
        model_type = None
    raw_architectures = getattr(config, "architectures", None)
    architectures = []
    if isinstance(raw_architectures, (list, tuple)):
        for architecture in raw_architectures[:32]:
            if isinstance(architecture, str) and architecture.strip() and len(architecture) <= 256:
                architectures.append(architecture.strip())
    return model_type, architectures


def _load_model(*, selection_value: Any, revision_value: Any, dtype_value: Any, device_value: Any, task: str):
    import transformers

    selection = _model_selection(selection_value)
    revision = _model_revision(selection, revision_value)
    dtype_name, dtype = _normalized_dtype(dtype_value)
    device = _normalized_device(device_value)
    source_receipt = _model_source_receipt(selection, revision)
    source = selection["value"]
    common = {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if revision is not None:
        common["revision"] = revision
    if task == "text-generation":
        preprocessor_auto_class = "AutoTokenizer"
        model_auto_class = "AutoModelForCausalLM"
        preprocessor = transformers.AutoTokenizer.from_pretrained(source, use_fast=True, **common)
        auto_model = transformers.AutoModelForCausalLM
        handle_kind = "transformers-causal-lm"
    elif task == "image-video-to-text":
        preprocessor_auto_class = "AutoProcessor"
        model_auto_class = "AutoModelForImageTextToText"
        preprocessor = transformers.AutoProcessor.from_pretrained(source, **common)
        auto_model = transformers.AutoModelForImageTextToText
        handle_kind = "transformers-image-text-to-text"
    else:  # pragma: no cover - internal programming error
        raise RuntimeError("Unsupported internal Transformers task.")
    model = auto_model.from_pretrained(
        source,
        dtype=dtype,
        use_safetensors=True,
        weights_only=True,
        low_cpu_mem_usage=True,
        **common,
    )
    model.to(device)
    model.eval()
    model_type, architectures = _config_metadata(model)
    receipt = _sealed_receipt(
        {
            "schemaVersion": SCHEMA_VERSION,
            "library": "transformers",
            "task": task,
            "source": source_receipt,
            "loader": {
                "preprocessorAutoClass": preprocessor_auto_class,
                "modelAutoClass": model_auto_class,
            },
            "security": SECURITY_CONTRACT,
            "runtime": {
                "transformersVersion": str(getattr(transformers, "__version__", "unknown"))[:128],
                "dtype": dtype_name,
                "device": device,
                "preprocessorClass": _class_name(preprocessor),
                "modelClass": _class_name(model),
                "modelType": model_type,
                "architectures": architectures,
            },
        }
    )
    handle = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": handle_kind,
        "model": model,
        "preprocessor": preprocessor,
        "receipt": receipt,
    }
    return handle, _json_copy(receipt)


def _normalized_modalities(value: Any, *, field: str, expected: set[str]) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise ValueError(f"Any-to-any {field} must be a finite modality sequence.")
    if not values or len(values) > 4 or any(not isinstance(item, str) for item in values):
        raise ValueError(f"Any-to-any {field} must contain one to four modality names.")
    normalized = {item.strip().casefold() for item in values}
    if normalized != expected:
        raise ValueError(f"Any-to-any {field} does not match the reviewed adapter contract.")
    return sorted(normalized)


def _janus_runtime_contract(model: Any, processor: Any) -> dict[str, Any]:
    contract = ANY_TO_ANY_ADAPTER_CONTRACTS["janus"]
    if _class_name(model) != contract["modelClass"] or _class_name(processor) != contract["processorClass"]:
        raise ValueError("The Janus adapter requires the reviewed native model and processor classes.")
    config = getattr(model, "config", None)
    if str(getattr(config, "model_type", "")).casefold() != "janus":
        raise ValueError("The Janus adapter requires model_type=janus.")
    input_modalities = _normalized_modalities(
        getattr(model, "input_modalities", None), field="input modalities", expected={"image", "text"}
    )
    output_modalities = _normalized_modalities(
        getattr(model, "output_modalities", None), field="output modalities", expected={"image", "text"}
    )
    vision_config = getattr(config, "vision_config", None)
    runtime_vision_config = getattr(getattr(getattr(model, "model", None), "vision_model", None), "config", None)
    image_size = getattr(vision_config, "image_size", None)
    patch_size = getattr(vision_config, "patch_size", None)
    image_tokens = getattr(vision_config, "num_image_tokens", None)
    runtime_image_tokens = getattr(runtime_vision_config, "num_image_tokens", None)
    processor_image_tokens = getattr(processor, "num_image_tokens", None)
    for value, field in (
        (image_size, "image_size"),
        (patch_size, "patch_size"),
        (image_tokens, "num_image_tokens"),
        (runtime_image_tokens, "runtime num_image_tokens"),
        (processor_image_tokens, "processor num_image_tokens"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"The Janus adapter requires a positive integer {field}.")
    if (
        image_size > MAX_MEDIA_SIDE
        or image_size * image_size > MAX_MEDIA_PIXELS
        or patch_size > image_size
        or image_size % patch_size
    ):
        raise ValueError("The Janus adapter image geometry exceeds the bounded image contract.")
    derived_tokens = (image_size // patch_size) ** 2
    if (
        image_tokens != derived_tokens
        or runtime_image_tokens != image_tokens
        or processor_image_tokens != image_tokens
        or image_tokens > MAX_ANY_TO_ANY_IMAGE_TOKENS
    ):
        raise ValueError("The Janus adapter image-token geometry is inconsistent or exceeds its bound.")
    return {
        "adapterId": contract["adapterId"],
        "modelType": "janus",
        "inputModalities": input_modalities,
        "outputModalities": output_modalities,
        "imageGeneration": {
            "width": image_size,
            "height": image_size,
            "patchSize": patch_size,
            "tokenCount": image_tokens,
        },
    }


def _load_any_to_any_model(
    *, selection_value: Any, revision_value: Any, dtype_value: Any, device_value: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    import transformers

    selection = _model_selection(selection_value)
    revision = _model_revision(selection, revision_value)
    dtype_name, dtype = _normalized_dtype(dtype_value)
    device = _normalized_device(device_value)
    source = selection["value"]
    source_receipt = _model_source_receipt(selection, revision)
    common = {"local_files_only": True, "trust_remote_code": False}
    if revision is not None:
        common["revision"] = revision
    config = transformers.AutoConfig.from_pretrained(source, **common)
    model_type = str(getattr(config, "model_type", "")).strip().casefold()
    adapter_contract = ANY_TO_ANY_ADAPTER_CONTRACTS.get(model_type)
    if adapter_contract is None:
        raise ValueError("No reviewed Any-to-Any adapter exists for this model_type.")
    if adapter_contract["status"] != "executable":
        raise ValueError(
            f"Any-to-Any adapter {adapter_contract['adapterId']} is contract-only: {adapter_contract['blocker']}"
        )
    processor = transformers.AutoProcessor.from_pretrained(source, **common)
    model = transformers.AutoModelForMultimodalLM.from_pretrained(
        source,
        dtype=dtype,
        use_safetensors=True,
        weights_only=True,
        low_cpu_mem_usage=True,
        **common,
    )
    model.to(device)
    model.eval()
    if model_type != "janus":  # pragma: no cover - registry status is the internal guard
        raise RuntimeError("The executable Any-to-Any adapter registry is inconsistent.")
    adapter = _janus_runtime_contract(model, processor)
    pipeline = transformers.AnyToAnyPipeline(model=model, processor=processor, device=device, dtype=dtype)
    if _class_name(pipeline) != "AnyToAnyPipeline" or not callable(pipeline):
        raise ValueError("The Any-to-Any loader requires the reviewed native pipeline class.")
    model_type_value, architectures = _config_metadata(model)
    receipt = _sealed_receipt(
        {
            "schemaVersion": SCHEMA_VERSION,
            "library": "transformers",
            "task": "any-to-any-generation",
            "source": source_receipt,
            "loader": {
                "configAutoClass": "AutoConfig",
                "preprocessorAutoClass": "AutoProcessor",
                "modelAutoClass": "AutoModelForMultimodalLM",
                "pipelineClass": "AnyToAnyPipeline",
            },
            "adapter": adapter,
            "security": SECURITY_CONTRACT,
            "runtime": {
                "transformersVersion": str(getattr(transformers, "__version__", "unknown"))[:128],
                "dtype": dtype_name,
                "device": device,
                "preprocessorClass": _class_name(processor),
                "modelClass": _class_name(model),
                "pipelineClass": _class_name(pipeline),
                "modelType": model_type_value,
                "architectures": architectures,
            },
        }
    )
    handle = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "transformers-any-to-any",
        "adapter": adapter["adapterId"],
        "model": model,
        "preprocessor": processor,
        "pipeline": pipeline,
        "receipt": receipt,
    }
    return handle, _json_copy(receipt)


def _validated_any_to_any_handle(value: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != ANY_TO_ANY_HANDLE_KEYS:
        raise ValueError("Any-to-Any generation requires an intact reviewed model handle.")
    if (
        value.get("schemaVersion") != SCHEMA_VERSION
        or value.get("kind") != "transformers-any-to-any"
        or value.get("adapter") != "janus-v1"
    ):
        raise ValueError("Any-to-Any generation received an unsupported model adapter handle.")
    model = value.get("model")
    processor = value.get("preprocessor")
    pipeline = value.get("pipeline")
    if model is None or processor is None or not callable(pipeline):
        raise ValueError("Any-to-Any generation requires a loaded model, processor, and pipeline.")
    receipt = value.get("receipt")
    try:
        valid_digest = _receipt_digest_is_valid(receipt)
    except ValueError:
        valid_digest = False
    if not valid_digest:
        raise ValueError("Any-to-Any model receipt is missing, oversized, or has been modified.")
    adapter = receipt.get("adapter")
    loader = receipt.get("loader")
    runtime = receipt.get("runtime")
    if (
        receipt.get("schemaVersion") != SCHEMA_VERSION
        or receipt.get("library") != "transformers"
        or receipt.get("task") != "any-to-any-generation"
        or receipt.get("security") != SECURITY_CONTRACT
        or not isinstance(adapter, dict)
        or adapter.get("adapterId") != value["adapter"]
        or adapter.get("modelType") != "janus"
        or not isinstance(loader, dict)
        or loader
        != {
            "configAutoClass": "AutoConfig",
            "preprocessorAutoClass": "AutoProcessor",
            "modelAutoClass": "AutoModelForMultimodalLM",
            "pipelineClass": "AnyToAnyPipeline",
        }
        or not isinstance(runtime, dict)
        or runtime.get("modelClass") != _class_name(model)
        or runtime.get("preprocessorClass") != _class_name(processor)
        or runtime.get("pipelineClass") != _class_name(pipeline)
    ):
        raise ValueError("Any-to-Any model receipt does not match the reviewed adapter task.")
    source = receipt.get("source")
    if not isinstance(source, dict) or source.get("kind") not in {"hub", "local"}:
        raise ValueError("Any-to-Any model receipt has an invalid source.")
    if source["kind"] == "hub":
        if not isinstance(source.get("repository"), str) or not IMMUTABLE_REVISION.fullmatch(
            str(source.get("revision"))
        ):
            raise ValueError("Any-to-Any model receipt has a mutable Hub source.")
    elif not isinstance(source.get("path"), str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", str(source.get("configSha256"))
    ):
        raise ValueError("Any-to-Any model receipt has an invalid local source.")
    return model, processor, pipeline, _json_copy(receipt)


def _validated_handle(value: Any, *, task: str) -> tuple[Any, Any, dict[str, Any]]:
    expected_kind = {
        "text-generation": "transformers-causal-lm",
        "image-video-to-text": "transformers-image-text-to-text",
    }[task]
    expected_loader = {
        "text-generation": ("AutoTokenizer", "AutoModelForCausalLM"),
        "image-video-to-text": ("AutoProcessor", "AutoModelForImageTextToText"),
    }[task]
    if not isinstance(value, dict) or set(value) != MODEL_HANDLE_KEYS:
        raise ValueError("Generation requires an intact Transformers model handle.")
    if value.get("schemaVersion") != SCHEMA_VERSION or value.get("kind") != expected_kind:
        raise ValueError("Generation received the wrong Transformers model handle type.")
    model = value.get("model")
    preprocessor = value.get("preprocessor")
    if model is None or preprocessor is None or not callable(getattr(model, "generate", None)):
        raise ValueError("Generation requires a loaded Transformers model and preprocessor.")
    receipt = value.get("receipt")
    try:
        valid_digest = _receipt_digest_is_valid(receipt)
    except ValueError:
        valid_digest = False
    if not valid_digest:
        raise ValueError("Transformers model receipt is missing, oversized, or has been modified.")
    loader = receipt.get("loader")
    if (
        receipt.get("schemaVersion") != SCHEMA_VERSION
        or receipt.get("library") != "transformers"
        or receipt.get("task") != task
        or receipt.get("security") != SECURITY_CONTRACT
        or not isinstance(loader, dict)
        or (loader.get("preprocessorAutoClass"), loader.get("modelAutoClass")) != expected_loader
    ):
        raise ValueError("Transformers model receipt does not match the requested generation task.")
    source = receipt.get("source")
    if not isinstance(source, dict) or source.get("kind") not in {"hub", "local"}:
        raise ValueError("Transformers model receipt has an invalid source.")
    if source["kind"] == "hub":
        repository = source.get("repository")
        revision = source.get("revision")
        if (
            not isinstance(repository, str)
            or repository.count("/") != 1
            or not IMMUTABLE_REVISION.fullmatch(str(revision))
        ):
            raise ValueError("Transformers model receipt has a mutable Hub source.")
    else:
        if not isinstance(source.get("path"), str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(source.get("configSha256"))
        ):
            raise ValueError("Transformers model receipt has an invalid local source.")
    return model, preprocessor, _json_copy(receipt)


def _generation_controls(kwargs: dict[str, Any]) -> dict[str, Any]:
    maximum = _bounded_int(
        kwargs.get("max_new_tokens"), field="max_new_tokens", default=256, minimum=1, maximum=MAX_NEW_TOKENS
    )
    minimum = _bounded_int(kwargs.get("min_new_tokens"), field="min_new_tokens", default=0, minimum=0, maximum=maximum)
    sample = _bounded_bool(kwargs.get("do_sample"), field="do_sample", default=False)
    controls: dict[str, Any] = {
        "max_new_tokens": maximum,
        "min_new_tokens": minimum,
        "do_sample": sample,
        "num_beams": _bounded_int(kwargs.get("num_beams"), field="num_beams", default=1, minimum=1, maximum=8),
        "repetition_penalty": _bounded_float(
            kwargs.get("repetition_penalty"), field="repetition_penalty", default=1.0, minimum=0.1, maximum=10.0
        ),
        "return_dict_in_generate": False,
    }
    if sample:
        controls.update(
            {
                "temperature": _bounded_float(
                    kwargs.get("temperature"), field="temperature", default=1.0, minimum=0.01, maximum=5.0
                ),
                "top_p": _bounded_float(kwargs.get("top_p"), field="top_p", default=1.0, minimum=0.01, maximum=1.0),
                "top_k": _bounded_int(kwargs.get("top_k"), field="top_k", default=50, minimum=0, maximum=1_000),
            }
        )
    return controls


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        normalized = tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError, OverflowError):
        return None
    if any(dimension < 0 for dimension in normalized):
        return None
    return normalized


def _batch_to_device(batch: Any, *, device: str) -> dict[str, Any]:
    if not hasattr(batch, "items"):
        raise RuntimeError("Transformers preprocessing returned a non-mapping batch.")
    values = dict(batch.items())
    if not values or len(values) > 64:
        raise RuntimeError("Transformers preprocessing returned an empty or excessive batch.")
    total_elements = 0
    for value in values.values():
        shape = _shape(value)
        if shape is None:
            continue
        elements = 1
        for dimension in shape:
            elements *= dimension
        total_elements += elements
        if total_elements > MAX_PROCESSED_TENSOR_ELEMENTS:
            raise RuntimeError("Transformers preprocessing exceeded the tensor size limit.")
    input_ids = values.get("input_ids")
    input_shape = _shape(input_ids)
    if input_shape is None or len(input_shape) != 2 or input_shape[0] != 1 or input_shape[1] <= 0:
        raise RuntimeError("Transformers preprocessing must produce exactly one nonempty token sequence.")
    if input_shape[1] > MAX_INPUT_TOKENS:
        raise RuntimeError("Transformers preprocessing exceeded the input token limit.")
    moved: dict[str, Any] = {}
    for key, value in values.items():
        mover = getattr(value, "to", None)
        moved[key] = mover(device) if callable(mover) else value
    return moved


def _output_sequences(value: Any) -> Any:
    sequences = getattr(value, "sequences", value)
    shape = _shape(sequences)
    if shape is None or len(shape) != 2 or shape[0] != 1 or shape[1] <= 0:
        raise RuntimeError("Transformers generation returned an invalid token sequence.")
    return sequences


def _token_prefix_matches(sequences: Any, input_ids: Any, input_tokens: int) -> bool:
    try:
        comparison = sequences[0, :input_tokens] == input_ids[0]
        reducer = getattr(comparison, "all", None)
        reduced = reducer() if callable(reducer) else all(comparison)
        item = getattr(reduced, "item", None)
        return bool(item() if callable(item) else reduced)
    except (IndexError, TypeError, ValueError, RuntimeError):
        return False


def _decode(preprocessor: Any, token_ids: Any) -> str:
    decoder = getattr(preprocessor, "decode", None)
    if callable(decoder):
        text = decoder(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    else:
        batch_decoder = getattr(preprocessor, "batch_decode", None)
        if not callable(batch_decoder):
            raise RuntimeError("Transformers preprocessor does not provide a safe token decoder.")
        decoded = batch_decoder(token_ids[None, ...], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if not isinstance(decoded, (list, tuple)) or len(decoded) != 1:
            raise RuntimeError("Transformers preprocessor returned an invalid decoded batch.")
        text = decoded[0]
    if not isinstance(text, str) or "\x00" in text:
        raise RuntimeError("Transformers generation returned invalid text.")
    text = text.strip()
    if len(text) > MAX_OUTPUT_CHARACTERS or len(text.encode("utf-8")) > MAX_OUTPUT_UTF8_BYTES:
        raise RuntimeError("Transformers generation exceeded the output text limit.")
    return text


def _normalized_media_item(value: Any, *, field: str):
    from PIL import Image

    try:
        import torch
    except ImportError:  # pragma: no cover - Torch is a backend runtime dependency
        torch = None
    if isinstance(value, Image.Image):
        width, height = value.size
        if width <= 0 or height <= 0:
            raise ValueError(f"{field} contains an empty image.")
        array = None
        image = value
    elif torch is not None and isinstance(value, torch.Tensor):
        shape = _shape(value)
        if shape is None or len(shape) not in {2, 3}:
            raise ValueError(f"{field} tensors must be two- or three-dimensional images.")
        if len(shape) == 2:
            tensor_height, tensor_width = shape
        elif shape[-1] in {1, 3, 4}:
            tensor_height, tensor_width = shape[:2]
        elif shape[0] in {1, 3, 4}:
            tensor_height, tensor_width = shape[1:]
        else:
            raise ValueError(f"{field} image channels must be grayscale, RGB, or RGBA.")
        tensor_pixels = tensor_width * tensor_height
        if (
            tensor_width <= 0
            or tensor_height <= 0
            or tensor_width > MAX_MEDIA_SIDE
            or tensor_height > MAX_MEDIA_SIDE
            or tensor_pixels > MAX_MEDIA_PIXELS
        ):
            raise ValueError(f"{field} image dimensions exceed the media bounds.")
        array = value.detach().cpu().numpy()
        image = None
    elif isinstance(value, np.ndarray):
        if value.ndim not in {2, 3}:
            raise ValueError(f"{field} arrays must be two- or three-dimensional images.")
        array = value
        image = None
    else:
        raise ValueError(f"{field} accepts only in-memory PIL images, NumPy arrays, or Torch tensors.")
    if array is not None:
        if array.ndim == 2:
            height, width = array.shape
        elif array.shape[-1] in {1, 3, 4}:
            height, width = array.shape[:2]
        elif array.shape[0] in {1, 3, 4}:
            array = np.moveaxis(array, 0, -1)
            height, width = array.shape[:2]
        else:
            raise ValueError(f"{field} image channels must be grayscale, RGB, or RGBA.")
    pixels = int(width) * int(height)
    if width <= 0 or height <= 0 or width > MAX_MEDIA_SIDE or height > MAX_MEDIA_SIDE or pixels > MAX_MEDIA_PIXELS:
        raise ValueError(f"{field} image dimensions exceed the media bounds.")
    if array is not None:
        if np.issubdtype(array.dtype, np.bool_) or not (
            np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)
        ):
            raise ValueError(f"{field} image values must use a numeric non-boolean dtype.")
        if np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all() or (array.size and (float(array.min()) < 0.0 or float(array.max()) > 1.0)):
                raise ValueError(f"{field} floating-point images must be finite values in [0, 1].")
            array = np.rint(array * 255.0).astype(np.uint8)
        else:
            if array.size and (int(array.min()) < 0 or int(array.max()) > 255):
                raise ValueError(f"{field} integer images must contain values in [0, 255].")
            array = array.astype(np.uint8, copy=False)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        image = Image.fromarray(np.ascontiguousarray(array))
    return image.convert("RGB"), pixels


def _media_items(value: Any, *, field: str, maximum: int) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray, Path, dict)):
        raise ValueError(f"{field} must be in-memory media, not a path, URL, or object reference.")
    shape = _shape(value)
    if shape is not None and len(shape) == 4:
        if shape[0] > maximum:
            raise ValueError(f"{field} exceeds the media item count limit of {maximum}.")
        items = [value[index] for index in range(shape[0])]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    if len(items) > maximum:
        raise ValueError(f"{field} exceeds the media item count limit of {maximum}.")
    return items


def _normalized_media(images_value: Any, video_value: Any) -> tuple[list[Any], list[Any], int]:
    image_values = _media_items(images_value, field="images", maximum=MAX_IMAGES)
    video_values = _media_items(video_value, field="video", maximum=MAX_VIDEO_FRAMES)
    if not image_values and not video_values:
        raise ValueError("Image/video-to-text generation requires at least one in-memory image or video frame.")
    images: list[Any] = []
    video: list[Any] = []
    total_pixels = 0
    for field, values, target in (("images", image_values, images), ("video", video_values, video)):
        for value in values:
            image, pixels = _normalized_media_item(value, field=field)
            total_pixels += pixels
            if total_pixels > MAX_TOTAL_MEDIA_PIXELS:
                raise ValueError("Image/video-to-text media exceeded the total pixel limit.")
            target.append(image)
    return images, video, total_pixels


def _multimodal_prompt(
    processor: Any, *, prompt: str, image_count: int, video_count: int, use_chat_template: bool
) -> str:
    if not use_chat_template:
        return prompt
    template = getattr(processor, "apply_chat_template", None)
    if not callable(template):
        raise ValueError(
            "This processor has no chat template; disable use_chat_template for its native prompt format."
        )
    content = [{"type": "image"} for _ in range(image_count)]
    if video_count:
        content.append({"type": "video"})
    content.append({"type": "text", "text": prompt})
    rendered = template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return _bounded_string(
        rendered,
        field="rendered prompt",
        maximum_characters=MAX_RENDERED_PROMPT_CHARACTERS,
        maximum_bytes=MAX_PROMPT_UTF8_BYTES * 2,
    )


def _normalized_optional_images(value: Any) -> tuple[list[Any], int]:
    values = _media_items(value, field="images", maximum=MAX_IMAGES) if value is not None else []
    images: list[Any] = []
    total_pixels = 0
    for item in values:
        image, pixels = _normalized_media_item(item, field="images")
        total_pixels += pixels
        if total_pixels > MAX_TOTAL_MEDIA_PIXELS:
            raise ValueError("Any-to-Any images exceeded the total pixel limit.")
        images.append(image)
    return images, total_pixels


def _one_any_to_any_record(value: Any, *, output_key: str) -> dict[str, Any]:
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise RuntimeError("Any-to-Any pipeline must return exactly one result record.")
    record = value[0]
    if set(record) != {"input_text", output_key}:
        raise RuntimeError("Any-to-Any pipeline returned an unexpected result schema.")
    return record


def _generate_janus_any_to_any(
    *, model: Any, processor: Any, pipeline: Any, receipt: dict[str, Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    mode = str(kwargs.get("generation_mode") or "text").strip().casefold()
    if mode not in {"text", "image"}:
        raise ValueError("The reviewed Janus adapter can generate exactly text or image outputs.")
    if kwargs.get("video") is not None or kwargs.get("audio_input") is not None:
        raise ValueError("The reviewed Janus adapter accepts only text and optional in-memory images.")
    prompt = _bounded_string(kwargs.get("prompt"), field="prompt")
    images, media_pixels = _normalized_optional_images(kwargs.get("images"))
    if mode == "image" and images:
        raise ValueError("The reviewed Janus image-generation adapter accepts a text prompt without source images.")
    image_token = getattr(processor, "image_token", None)
    if images:
        if not isinstance(image_token, str) or not image_token or len(image_token) > 256 or "\x00" in image_token:
            raise ValueError("The reviewed Janus processor has an invalid image placeholder token.")
        adapted_prompt = _bounded_string(
            (image_token * len(images)) + prompt,
            field="adapted Janus prompt",
            maximum_characters=MAX_RENDERED_PROMPT_CHARACTERS,
            maximum_bytes=MAX_PROMPT_UTF8_BYTES * 2,
        )
    else:
        adapted_prompt = prompt
    processor_kwargs: dict[str, Any] = {
        "text": adapted_prompt,
        "return_tensors": "pt",
        "generation_mode": mode,
    }
    if images:
        processor_kwargs["images"] = images
    preview = _batch_to_device(processor(**processor_kwargs), device=receipt["runtime"]["device"])
    input_ids = preview["input_ids"]
    input_tokens = _shape(input_ids)[1]
    controls = _generation_controls(kwargs)
    controls["num_beams"] = 1
    invocation: dict[str, Any] = {"text": adapted_prompt}
    if images:
        invocation["images"] = images
    call_kwargs: dict[str, Any] = {
        "generation_mode": mode,
        "generate_kwargs": controls,
        "processor_kwargs": {"generation_mode": mode},
    }
    if mode == "text":
        call_kwargs["return_tensors"] = True
        record = _one_any_to_any_record(pipeline(invocation, **call_kwargs), output_key="generated_token_ids")
        sequence = record["generated_token_ids"]
        sequence_shape = _shape(sequence)
        if sequence_shape is None or len(sequence_shape) != 1 or sequence_shape[0] <= 0:
            raise RuntimeError("Any-to-Any text generation returned an invalid token sequence.")
        sequence_batch = sequence[None, ...]
        output_tokens = sequence_shape[0]
        if output_tokens < input_tokens or not _token_prefix_matches(sequence_batch, input_ids, input_tokens):
            raise RuntimeError("Any-to-Any text generation did not preserve the input token prefix.")
        generated_tokens = output_tokens - input_tokens
        if generated_tokens > controls["max_new_tokens"]:
            raise RuntimeError("Any-to-Any text generation exceeded max_new_tokens.")
        text = _decode(processor, sequence[input_tokens:])
        result = {
            "schemaVersion": SCHEMA_VERSION,
            "task": "any-to-any-generation",
            "adapter": receipt["adapter"]["adapterId"],
            "outputModality": "text",
            "text": text,
            "image": None,
            "audio": None,
            "finishReason": "length" if generated_tokens == controls["max_new_tokens"] else "stop",
            "inputTokens": input_tokens,
            "generatedTokens": generated_tokens,
            "inputImageCount": len(images),
            "inputMediaPixels": media_pixels,
            "modelReceipt": receipt,
        }
        return {"text": text, "image": None, "result": result}
    record = _one_any_to_any_record(pipeline(invocation, **call_kwargs), output_key="generated_image")
    image, output_pixels = _normalized_media_item(record["generated_image"], field="generated image")
    geometry = receipt["adapter"]["imageGeneration"]
    if image.size != (geometry["width"], geometry["height"]):
        raise RuntimeError("The Janus adapter returned an image outside its reviewed output geometry.")
    result = {
        "schemaVersion": SCHEMA_VERSION,
        "task": "any-to-any-generation",
        "adapter": receipt["adapter"]["adapterId"],
        "outputModality": "image",
        "text": None,
        "image": {"width": image.width, "height": image.height, "pixels": output_pixels},
        "audio": None,
        "finishReason": "fixed-image-token-contract",
        "inputTokens": input_tokens,
        "generatedTokens": geometry["tokenCount"],
        "inputImageCount": 0,
        "inputMediaPixels": 0,
        "modelReceipt": receipt,
    }
    return {"text": None, "image": image, "result": result}


class LoadTextGenerationModel(NodeBase):
    """Load an immutable local-only causal language model and tokenizer."""

    label = "Load Text Generation Model"
    category = "Hugging Face Transformers"
    resizable = True
    params = {
        "model": {"label": "Model", "display": "output", "type": "transformers_causal_lm"},
        "pipeline_class": {
            "label": "Execution Class",
            "type": "string",
            "default": "AutoModelForCausalLM",
            "hidden": True,
        },
        "model_id": {
            "label": "Model",
            "display": "modelselect",
            "type": "string",
            "fieldOptions": {"noValidation": True, "sources": ["hub", "local"]},
        },
        "revision": {"label": "Exact Hub Commit", "type": "string", "default": ""},
        "dtype": {
            "label": "DType",
            "type": "string",
            "options": ["float32", "float16", "bfloat16"],
            "default": "float32",
        },
        "device": {"label": "Device", "type": "string", "options": DEVICE_LIST, "default": DEFAULT_DEVICE},
        "receipt": {"label": "Model Receipt", "display": "output", "type": "object"},
    }

    def execute(self, **kwargs):
        if kwargs.get("pipeline_class", "AutoModelForCausalLM") != "AutoModelForCausalLM":
            raise ValueError("Text generation requires the AutoModelForCausalLM execution contract.")
        model, receipt = _load_model(
            selection_value=kwargs.get("model_id"),
            revision_value=kwargs.get("revision"),
            dtype_value=kwargs.get("dtype"),
            device_value=kwargs.get("device"),
            task="text-generation",
        )
        return {"model": model, "receipt": receipt}


class GenerateText(NodeBase):
    """Generate a bounded text continuation from one loaded causal language model."""

    label = "Generate Text"
    category = "Hugging Face Transformers"
    resizable = True
    params = {
        "model": {"label": "Model", "display": "input", "type": "transformers_causal_lm"},
        "prompt": {"label": "Prompt", "type": "text", "default": ""},
        "max_new_tokens": {"label": "Max New Tokens", "type": "int", "default": 256, "min": 1, "max": 2048},
        "min_new_tokens": {"label": "Min New Tokens", "type": "int", "default": 0, "min": 0, "max": 2048},
        "do_sample": {"label": "Sample", "type": "boolean", "default": False},
        "temperature": {"label": "Temperature", "type": "float", "default": 1.0, "min": 0.01, "max": 5.0},
        "top_p": {"label": "Top P", "type": "float", "default": 1.0, "min": 0.01, "max": 1.0},
        "top_k": {"label": "Top K", "type": "int", "default": 50, "min": 0, "max": 1000},
        "num_beams": {"label": "Beams", "type": "int", "default": 1, "min": 1, "max": 8},
        "repetition_penalty": {
            "label": "Repetition Penalty",
            "type": "float",
            "default": 1.0,
            "min": 0.1,
            "max": 10.0,
        },
        "text": {"label": "Generated Text", "display": "output", "type": "string"},
        "result": {"label": "Generation Result", "display": "output", "type": "object"},
    }

    def execute(self, **kwargs):
        import torch

        model, tokenizer, receipt = _validated_handle(kwargs.get("model"), task="text-generation")
        prompt = _bounded_string(kwargs.get("prompt"), field="prompt")
        controls = _generation_controls(kwargs)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
        device = receipt["runtime"]["device"]
        batch = _batch_to_device(encoded, device=device)
        input_ids = batch["input_ids"]
        input_tokens = _shape(input_ids)[1]
        with torch.inference_mode():
            sequences = _output_sequences(model.generate(**batch, **controls))
        output_tokens = _shape(sequences)[1]
        if output_tokens < input_tokens or not _token_prefix_matches(sequences, input_ids, input_tokens):
            raise RuntimeError("Causal text generation did not preserve the input token prefix.")
        generated_tokens = output_tokens - input_tokens
        if generated_tokens > controls["max_new_tokens"]:
            raise RuntimeError("Transformers generation exceeded max_new_tokens.")
        text = _decode(tokenizer, sequences[0, input_tokens:])
        result = {
            "schemaVersion": SCHEMA_VERSION,
            "task": "text-generation",
            "text": text,
            "finishReason": "length" if generated_tokens == controls["max_new_tokens"] else "stop",
            "inputTokens": input_tokens,
            "generatedTokens": generated_tokens,
            "modelReceipt": receipt,
        }
        return {"text": text, "result": result}


class LoadImageTextToTextModel(NodeBase):
    """Load an immutable local-only image/video-to-text model and processor."""

    label = "Load Image/Video-to-Text Model"
    category = "Hugging Face Transformers"
    resizable = True
    params = {
        "model": {"label": "Model", "display": "output", "type": "transformers_image_text_to_text"},
        "pipeline_class": {
            "label": "Execution Class",
            "type": "string",
            "default": "AutoModelForImageTextToText",
            "hidden": True,
        },
        "model_id": {
            "label": "Model",
            "display": "modelselect",
            "type": "string",
            "fieldOptions": {"noValidation": True, "sources": ["hub", "local"]},
        },
        "revision": {"label": "Exact Hub Commit", "type": "string", "default": ""},
        "dtype": {
            "label": "DType",
            "type": "string",
            "options": ["float32", "float16", "bfloat16"],
            "default": "float32",
        },
        "device": {"label": "Device", "type": "string", "options": DEVICE_LIST, "default": DEFAULT_DEVICE},
        "receipt": {"label": "Model Receipt", "display": "output", "type": "object"},
    }

    def execute(self, **kwargs):
        if kwargs.get("pipeline_class", "AutoModelForImageTextToText") != "AutoModelForImageTextToText":
            raise ValueError("Image/video-to-text generation requires the AutoModelForImageTextToText contract.")
        model, receipt = _load_model(
            selection_value=kwargs.get("model_id"),
            revision_value=kwargs.get("revision"),
            dtype_value=kwargs.get("dtype"),
            device_value=kwargs.get("device"),
            task="image-video-to-text",
        )
        return {"model": model, "receipt": receipt}


class GenerateImageVideoText(NodeBase):
    """Generate bounded text from bounded in-memory images or video frames."""

    label = "Generate Image/Video Text"
    category = "Hugging Face Transformers"
    resizable = True
    params = {
        "model": {"label": "Model", "display": "input", "type": "transformers_image_text_to_text"},
        "prompt": {"label": "Prompt", "type": "text", "default": ""},
        "images": {"label": "Images", "display": "input", "type": "image", "required": False},
        "video": {"label": "Video", "display": "input", "type": "video", "required": False},
        "use_chat_template": {"label": "Use Chat Template", "type": "boolean", "default": True},
        "max_new_tokens": {"label": "Max New Tokens", "type": "int", "default": 256, "min": 1, "max": 2048},
        "min_new_tokens": {"label": "Min New Tokens", "type": "int", "default": 0, "min": 0, "max": 2048},
        "do_sample": {"label": "Sample", "type": "boolean", "default": False},
        "temperature": {"label": "Temperature", "type": "float", "default": 1.0, "min": 0.01, "max": 5.0},
        "top_p": {"label": "Top P", "type": "float", "default": 1.0, "min": 0.01, "max": 1.0},
        "top_k": {"label": "Top K", "type": "int", "default": 50, "min": 0, "max": 1000},
        "num_beams": {"label": "Beams", "type": "int", "default": 1, "min": 1, "max": 8},
        "repetition_penalty": {
            "label": "Repetition Penalty",
            "type": "float",
            "default": 1.0,
            "min": 0.1,
            "max": 10.0,
        },
        "text": {"label": "Generated Text", "display": "output", "type": "string"},
        "result": {"label": "Generation Result", "display": "output", "type": "object"},
    }

    def execute(self, **kwargs):
        import torch

        model, processor, receipt = _validated_handle(kwargs.get("model"), task="image-video-to-text")
        prompt = _bounded_string(kwargs.get("prompt"), field="prompt")
        use_chat_template = _bounded_bool(kwargs.get("use_chat_template"), field="use_chat_template", default=True)
        controls = _generation_controls(kwargs)
        images, video, total_pixels = _normalized_media(kwargs.get("images"), kwargs.get("video"))
        rendered = _multimodal_prompt(
            processor,
            prompt=prompt,
            image_count=len(images),
            video_count=len(video),
            use_chat_template=use_chat_template,
        )
        processor_kwargs: dict[str, Any] = {"text": rendered, "return_tensors": "pt"}
        if images:
            processor_kwargs["images"] = images
        if video:
            processor_kwargs["videos"] = [video]
        batch = _batch_to_device(processor(**processor_kwargs), device=receipt["runtime"]["device"])
        input_ids = batch["input_ids"]
        input_tokens = _shape(input_ids)[1]
        with torch.inference_mode():
            sequences = _output_sequences(model.generate(**batch, **controls))
        output_tokens = _shape(sequences)[1]
        prefix = output_tokens >= input_tokens and _token_prefix_matches(sequences, input_ids, input_tokens)
        generated = sequences[0, input_tokens:] if prefix else sequences[0]
        generated_tokens = _shape(generated)[0]
        if generated_tokens > controls["max_new_tokens"]:
            raise RuntimeError("Transformers generation exceeded max_new_tokens.")
        text = _decode(processor, generated)
        result = {
            "schemaVersion": SCHEMA_VERSION,
            "task": "image-video-to-text",
            "text": text,
            "finishReason": "length" if generated_tokens == controls["max_new_tokens"] else "stop",
            "inputTokens": input_tokens,
            "generatedTokens": generated_tokens,
            "imageCount": len(images),
            "videoFrameCount": len(video),
            "mediaPixels": total_pixels,
            "modelReceipt": receipt,
        }
        return {"text": text, "result": result}


class LoadAnyToAnyModel(NodeBase):
    """Load one immutable model through a finite reviewed Any-to-Any adapter."""

    label = "Load Any-to-Any Model"
    category = "Hugging Face Transformers"
    resizable = True
    params = {
        "model": {"label": "Model", "display": "output", "type": "transformers_any_to_any"},
        "pipeline_class": {
            "label": "Execution Class",
            "type": "string",
            "default": "JanusForConditionalGeneration",
            "hidden": True,
        },
        "model_id": {
            "label": "Model",
            "display": "modelselect",
            "type": "string",
            "fieldOptions": {"noValidation": True, "sources": ["hub", "local"]},
        },
        "revision": {"label": "Exact Hub Commit", "type": "string", "default": ""},
        "dtype": {
            "label": "DType",
            "type": "string",
            "options": ["float32", "float16", "bfloat16"],
            "default": "float32",
        },
        "device": {"label": "Device", "type": "string", "options": DEVICE_LIST, "default": DEFAULT_DEVICE},
        "receipt": {"label": "Model Receipt", "display": "output", "type": "object"},
    }

    def execute(self, **kwargs):
        if kwargs.get("pipeline_class", "JanusForConditionalGeneration") != "JanusForConditionalGeneration":
            raise ValueError("Any-to-any generation requires the reviewed Janus execution contract.")
        model, receipt = _load_any_to_any_model(
            selection_value=kwargs.get("model_id"),
            revision_value=kwargs.get("revision"),
            dtype_value=kwargs.get("dtype"),
            device_value=kwargs.get("device"),
        )
        return {"model": model, "receipt": receipt}


class GenerateAnyToAny(NodeBase):
    """Generate one bounded output through the handle's reviewed family adapter."""

    label = "Generate Any-to-Any"
    category = "Hugging Face Transformers"
    resizable = True
    params = {
        "model": {"label": "Model", "display": "input", "type": "transformers_any_to_any"},
        "prompt": {"label": "Prompt", "type": "text", "default": ""},
        "images": {"label": "Images", "display": "input", "type": "image", "required": False},
        "generation_mode": {
            "label": "Output Modality",
            "type": "string",
            "options": ["text", "image"],
            "default": "text",
        },
        "max_new_tokens": {"label": "Max New Tokens", "type": "int", "default": 256, "min": 1, "max": 2048},
        "min_new_tokens": {"label": "Min New Tokens", "type": "int", "default": 0, "min": 0, "max": 2048},
        "do_sample": {"label": "Sample", "type": "boolean", "default": False},
        "temperature": {"label": "Temperature", "type": "float", "default": 1.0, "min": 0.01, "max": 5.0},
        "top_p": {"label": "Top P", "type": "float", "default": 1.0, "min": 0.01, "max": 1.0},
        "top_k": {"label": "Top K", "type": "int", "default": 50, "min": 0, "max": 1000},
        "num_beams": {"label": "Beams", "type": "int", "default": 1, "min": 1, "max": 8},
        "repetition_penalty": {
            "label": "Repetition Penalty",
            "type": "float",
            "default": 1.0,
            "min": 0.1,
            "max": 10.0,
        },
        "text": {"label": "Generated Text", "display": "output", "type": "string"},
        "image": {"label": "Generated Image", "display": "output", "type": "image"},
        "result": {"label": "Generation Result", "display": "output", "type": "object"},
    }

    def execute(self, **kwargs):
        model, processor, pipeline, receipt = _validated_any_to_any_handle(kwargs.get("model"))
        if receipt["adapter"]["adapterId"] != "janus-v1":  # pragma: no cover - validated finite registry
            raise RuntimeError("The Any-to-Any adapter registry is inconsistent.")
        return _generate_janus_any_to_any(
            model=model,
            processor=processor,
            pipeline=pipeline,
            receipt=receipt,
            kwargs=kwargs,
        )
