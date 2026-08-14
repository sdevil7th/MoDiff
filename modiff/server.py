# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
import logging
import asyncio
import math
from aiohttp import web, WSMsgType
from aiohttp.web_fileresponse import CONTENT_TYPES as AIOHTTP_CONTENT_TYPES
from aiohttp_cors import setup as cors_setup, ResourceOptions
import mimetypes

mimetypes.add_type("image/webp", ".webp")
AIOHTTP_CONTENT_TYPES.add_type("image/webp", ".webp")

logging.getLogger("asyncio").setLevel(logging.WARNING)
from functools import partial
from importlib import import_module, metadata, invalidate_caches
import os
import platform
import base64
import csv
import hashlib
import html
import ipaddress
import io
import json
import nanoid
import random
import re
import shutil
import stat
import subprocess
import configparser
import threading
from utils.paths import list_files
from pathlib import Path
import sys
import traceback
import time
import gc
from urllib.parse import quote, unquote, unquote_to_bytes, urlparse, parse_qs
from copy import deepcopy
from modiff.path_identifiers import (
    data_path_identifier,
    is_data_path_identifier,
    resolve_data_path_identifier,
    resolve_managed_path_identifier,
)
from modiff.disk_activity import DiskActivitySampler
from modiff.supervisor_control import compact_task_history
from modiff.template_gallery import (
    TEMPLATE_GALLERY_SOURCE_PATH,
    TemplateGalleryError,
    fetch_template_gallery_contract,
    install_template_gallery,
    load_template_gallery_source,
    plan_template_gallery_install,
    template_gallery_destination_present,
    verify_template_gallery_tree,
)

logger = logging.getLogger("modiff")

SUPERVISED_RESTART_EXIT_CODE = 75
SUPPORTED_AUDIO_DOWNLOAD_SAMPLE_RATES = {44100, 48000, 88200, 96000}
DEFAULT_CLIENT_MAX_SIZE = 1024**3
MAX_WORKFLOW_SHARE_MEDIA_BYTES = 256 * 1024 * 1024
TEMPLATE_GALLERY_ROOT = Path("web/template-gallery")
MAX_PREVIEW_IMAGE_PIXELS = 40_000_000
_PREVIEW_IMAGE_FORMATS = {
    "jpeg": ("JPEG", "image/jpeg"),
    "jpg": ("JPEG", "image/jpeg"),
    "png": ("PNG", "image/png"),
    "webp": ("WEBP", "image/webp"),
    "bmp": ("BMP", "image/bmp"),
    "ico": ("ICO", "image/x-icon"),
    "gif": ("GIF", "image/gif"),
    "tiff": ("TIFF", "image/tiff"),
}


def render_image_preview(file_path, width=0, height=0, format_id="jpeg", quality=95):
    """Decode and resize a bounded image for the async preview route."""

    from PIL import Image, ImageOps
    from utils.image import cover

    descriptor = _PREVIEW_IMAGE_FORMATS.get(str(format_id).lower())
    if descriptor is None:
        raise ValueError(f"Unsupported preview image format: {format_id}.")
    pillow_format, content_type = descriptor
    try:
        with Image.open(file_path) as opened:
            pixel_count = int(opened.width) * int(opened.height)
            if pixel_count > MAX_PREVIEW_IMAGE_PIXELS:
                raise ValueError(
                    f"The image is too large to preview safely ({pixel_count:,} pixels; "
                    f"limit {MAX_PREVIEW_IMAGE_PIXELS:,})."
                )
            opened.load()
            image = ImageOps.exif_transpose(opened).copy()
    except Image.DecompressionBombError as error:
        raise ValueError("The image exceeds Pillow's safe decompression limit.") from error

    requested_width = int(width)
    requested_height = int(height)
    if requested_width > 0 or requested_height > 0:
        requested_width = min(2048, requested_width) if requested_width > 0 else min(2048, requested_height)
        requested_height = min(2048, requested_height) if requested_height > 0 else min(2048, requested_width)
    else:
        requested_width = min(2048, image.width)
        requested_height = min(2048, image.height)

    if requested_width != image.width or requested_height != image.height:
        image = cover(image, requested_width, requested_height, resample="BICUBIC")
    if image.mode != "RGB" and pillow_format in {"JPEG", "BMP", "ICO"}:
        image = image.convert("RGB")

    output = io.BytesIO()
    save_options = {"format": pillow_format}
    if pillow_format in {"JPEG", "WEBP"}:
        save_options["quality"] = max(1, min(100, int(quality)))
    image.save(output, **save_options)
    return output.getvalue(), content_type


def is_hidden_path(path):
    """Return dotfile or native Windows hidden-attribute state."""

    if path.name.startswith("."):
        return True
    file_attributes = getattr(path.stat(), "st_file_attributes", 0)
    hidden_attribute = getattr(stat, "FILE_ATTRIBUTE_HIDDEN", 0)
    return bool(hidden_attribute and file_attributes & hidden_attribute)


def parse_audio_download_sample_rate(value):
    if value in (None, ""):
        return None
    try:
        sample_rate = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Audio download sample rate must be an integer.") from exc
    if sample_rate not in SUPPORTED_AUDIO_DOWNLOAD_SAMPLE_RATES:
        supported = ", ".join(str(rate) for rate in sorted(SUPPORTED_AUDIO_DOWNLOAD_SAMPLE_RATES))
        raise ValueError(f"Audio download sample rate must be one of: {supported}.")
    return sample_rate


def resample_wav_bytes(body, target_sample_rate):
    """Return WAV bytes whose encoded sample rate matches the requested export rate."""
    from math import gcd

    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    source = io.BytesIO(bytes(body))
    source_sample_rate, samples = wavfile.read(source)
    source_sample_rate = int(source_sample_rate)
    target_sample_rate = int(target_sample_rate)
    if source_sample_rate == target_sample_rate:
        return bytes(body)

    divisor = gcd(source_sample_rate, target_sample_rate)
    resampled = resample_poly(
        samples.astype(np.float64),
        target_sample_rate // divisor,
        source_sample_rate // divisor,
        axis=0,
    )
    if np.issubdtype(samples.dtype, np.integer):
        limits = np.iinfo(samples.dtype)
        resampled = np.clip(np.rint(resampled), limits.min, limits.max).astype(samples.dtype)
    else:
        resampled = resampled.astype(samples.dtype)

    output = io.BytesIO()
    wavfile.write(output, target_sample_rate, resampled)
    return output.getvalue()


def audio_download_filename(filename, sample_rate):
    path = Path(str(filename or "MoDiff-audio.wav"))
    label = {
        44100: "44.1kHz",
        48000: "48kHz",
        88200: "88.2kHz",
        96000: "96kHz",
    }[int(sample_rate)]
    suffix = path.suffix if path.suffix.lower() == ".wav" else ".wav"
    return f"{path.stem}-{label}{suffix}"


def classify_hf_download_error(error: Exception) -> tuple[int, str, str, bool]:
    """Map Hub failures to stable, actionable API errors without discarding cache data."""
    name = error.__class__.__name__.lower()
    message = str(error)
    status_code = getattr(getattr(error, "response", None), "status_code", None)
    # Hugging Face's RepositoryNotFoundError text includes a generic suggestion
    # about private or gated repositories. Classify the concrete 404/name first
    # so a misspelled or retired repo is not incorrectly shown as "Add HF token".
    if "repositorynotfound" in name or status_code == 404:
        return (404, "huggingface_repo_not_found", "The Hugging Face repository was not found or is private.", False)
    if status_code in {401, 403} or "gatedrepo" in name:
        return (
            403,
            "huggingface_access_required",
            "Hugging Face access is required. Accept the repository license, then add a read token in Model Manager and retry.",
            False,
        )
    if isinstance(error, (TimeoutError, ConnectionError)) or status_code in {408, 429, 500, 502, 503, 504}:
        return (503, "huggingface_network_error", f"Download interrupted by a retryable Hub error: {message}", True)
    return (500, "huggingface_download_failed", message, True)


def node_execution_phase(module: str, action: str) -> str:
    name = f"{module}.{action}".lower()
    if "load" in name:
        return "loading"
    if "encode" in name or "embedding" in name:
        return "encoding"
    if "denoise" in name or "generate" in name or "inpaint" in name or "outpaint" in name:
        return "denoising"
    if "decode" in name:
        return "decoding"
    if "save" in name or "export" in name:
        return "export"
    if "preview" in name:
        return "previewing"
    return "unknown"


def node_execution_message(module: str, action: str, phase: str) -> str:
    name = f"{module}.{action}"
    if phase == "loading":
        return f"Loading {name}"
    if phase == "encoding":
        return f"Encoding {name}"
    if phase == "denoising":
        return f"Running {name}"
    if phase == "decoding":
        return f"Decoding {name}"
    if phase == "export":
        return f"Exporting {name}"
    if phase == "previewing":
        return f"Previewing {name}"
    return f"Running {name}"


def node_execution_weight(module: str, action: str) -> float:
    phase = node_execution_phase(module, action)
    if phase == "denoising":
        return 8.0
    if phase in {"encoding", "decoding"}:
        return 2.0
    if phase == "loading":
        return 1.5
    if phase in {"saving", "previewing"}:
        return 0.5
    return 1.0


def is_image_data_type(data_type):
    if data_type == "image":
        return True
    if isinstance(data_type, (list, tuple, set)):
        return any(item == "image" for item in data_type)
    return False


def is_cache_servable_data_type(data_type):
    if is_image_data_type(data_type):
        return True
    if isinstance(data_type, str):
        return data_type in {"audio", "video", "text"} or data_type.startswith("str")
    if isinstance(data_type, (list, tuple, set)):
        return any(
            item in {"audio", "video", "text"}
            or (isinstance(item, str) and item.startswith("str"))
            for item in data_type
        )
    return False


def image_dimensions(value):
    width = getattr(value, "width", None)
    height = getattr(value, "height", None)
    if isinstance(width, int) and isinstance(height, int):
        return width, height
    return None, None


def cache_image_artifact(node_id, field_key, index, url, value, image_format):
    mime_type = f"image/{str(image_format or 'WEBP').lower()}"
    width, height = image_dimensions(value)
    filename = f"modiff-{node_id}-{field_key}-{index}.{str(image_format or 'WEBP').lower()}"
    return {
        "url": url,
        "nodeId": node_id,
        "fieldKey": field_key,
        "index": index,
        "mimeType": mime_type,
        "filename": filename,
        "width": width,
        "height": height,
        "source": "cache",
    }


def attach_run_identity_to_artifact(artifact, *, task_id=None, attempt_index=None, runtime_hints=None):
    if not isinstance(artifact, dict):
        return artifact
    if task_id is not None:
        artifact["task_id"] = task_id
    if attempt_index is not None:
        artifact["attempt_index"] = attempt_index
    if isinstance(runtime_hints, dict):
        client_run_id = runtime_hints.get("clientRunId")
        run_input_hash = runtime_hints.get("runInputHash")
        if client_run_id:
            artifact["client_run_id"] = client_run_id
        if run_input_hash:
            artifact["run_input_hash"] = run_input_hash
    return artifact


def file_backed_media_preview(value):
    """Return a browser URL for a file-backed media output.

    UI audio/video fields whose source is a string path must point at the file
    route. Sending them through /cache serves the path itself as text because
    the source field's declared type is ``str``.
    """
    if not isinstance(value, (str, os.PathLike)):
        return None
    value = os.fspath(value)
    if value.startswith(("http://", "https://", "data:", "blob:", "/file?")):
        return value
    return f"/file?file={quote(value, safe='')}&t={time.time()}"


def byte_range_response(request, body, *, content_type, charset=None, filename=None):
    """Serve generated in-memory media with single-range HTTP semantics.

    Browser media controls seek by requesting a byte range. Returning the
    entire cached WAV with ``200`` makes Chromium briefly move the playhead and
    then snap back because the resource has no seekable range.
    """
    body = bytes(body)
    total_length = len(body)
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    if filename:
        headers["Content-Disposition"] = f'inline; filename="{filename}"'

    range_header = request.headers.get("Range")
    if not range_header:
        headers["Content-Length"] = str(total_length)
        return web.Response(
            body=body,
            content_type=content_type,
            charset=charset,
            headers=headers,
        )

    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
    if not match or total_length == 0:
        headers["Content-Range"] = f"bytes */{total_length}"
        return web.Response(status=416, headers=headers)

    start_text, end_text = match.groups()
    if not start_text and not end_text:
        headers["Content-Range"] = f"bytes */{total_length}"
        return web.Response(status=416, headers=headers)

    if start_text:
        start = int(start_text)
        if start >= total_length:
            headers["Content-Range"] = f"bytes */{total_length}"
            return web.Response(status=416, headers=headers)
        end = total_length - 1 if not end_text else min(int(end_text), total_length - 1)
        if end < start:
            headers["Content-Range"] = f"bytes */{total_length}"
            return web.Response(status=416, headers=headers)
    else:
        suffix_length = int(end_text)
        if suffix_length <= 0:
            headers["Content-Range"] = f"bytes */{total_length}"
            return web.Response(status=416, headers=headers)
        start = max(total_length - suffix_length, 0)
        end = total_length - 1

    partial = body[start : end + 1]
    headers["Content-Range"] = f"bytes {start}-{end}/{total_length}"
    headers["Content-Length"] = str(len(partial))
    return web.Response(
        status=206,
        body=partial,
        content_type=content_type,
        charset=charset,
        headers=headers,
    )


from modiff.config import CONFIG
from modiff.controlled_artifacts import controlled_artifact_receipts_from_graph
from modiff.diffusers_offload import (
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_NONE,
    OFFLOAD_MODE_SEQUENTIAL_CPU,
)
from modiff.diffusers_profiles import (
    QWEN_IMAGE_2512_PREQUANTIZED_REPO,
    VERIFIED_REPAIR_SOURCES,
    execution_profiles_for_execution,
    optional_runtime_profile_ids_for_execution,
    public_execution_profiles,
    public_experimental_pipelines,
)
from modiff.hardware import format_hardware_summary, get_hardware_snapshot, legacy_torch_status
from modiff.runtime_profile import runtime_profile
from modiff.auto_resource import (
    AUTO_RESOURCE_SCHEMA_VERSION,
    PROVEN_PROOF_STATUSES,
    artifact_cache_status,
    auto_resource_pair_is_declared,
    build_auto_resource_plan,
    build_auto_resource_plans,
    clear_auto_resource_history,
    matching_auto_resource_success_history,
    read_auto_resource_history,
    record_auto_resource_failure,
    record_auto_resource_success,
)
from modiff.model_artifact_catalog import (
    IMMUTABLE_HUB_REVISION,
    catalog_revision,
    public_model_artifact_catalog,
    refreshed_hub_metadata,
)
from modiff.optimization_packages import (
    activate_optional_runtime_environment,
    activate_environment as activate_optimization_environment,
    install_optional_runtime,
    install_capability as install_optimization_capability,
    optimization_selections_from_graph,
    probe_capability as probe_optimization_capability,
    public_catalog as public_optimization_catalog,
    public_optional_runtime_catalog,
    qualify_receipt as qualify_optimization_receipt,
    read_receipts as read_optimization_receipts,
    record_workload_observation as record_optimization_workload_observation,
    record_workload_baseline as record_optimization_workload_baseline,
    rollback_environment as rollback_optimization_environment,
    rollback_optional_runtime_environment,
    set_capability_enabled as set_optimization_capability_enabled,
    workload_key_for_form as optimization_workload_key_for_form,
    validate_optional_runtime_activation_request,
    validate_optional_runtime_install_request,
)
from modiff.runtime_overlays import (
    OverlayCancelled,
    OverlayInstallBusy,
    cancel_install as cancel_runtime_install,
    release_install as release_runtime_install,
    reserve_install as reserve_runtime_install,
)
from modiff.optional_runtimes import public_optional_runtime_profiles
from modiff.optional_runtime_execution import (
    assert_optional_runtime_ready,
    field_action_optional_runtime_requirement,
    graph_optional_runtime_requirement,
    loader_optional_runtime_requirement,
    optional_runtime_blocker_payload,
    optional_runtime_requirement_blocks_execution,
    optional_runtime_requirement_for_execution,
)
from modiff.studio_execution_specs import (
    ACE_STEP_DIFFUSERS_FILES,
    FLUX_FILL_DIFFUSERS_FILES,
    FLUX_KONTEXT_DIFFUSERS_FILES,
    LTX_VIDEO_DIFFUSERS_FILES,
    QWEN_IMAGE_2512_DIFFUSERS_FILES,
    QWEN_IMAGE_EDIT_2511_DIFFUSERS_FILES,
    QWEN_IMAGE_EDIT_DIFFUSERS_FILES,
    QWEN_IMAGE_LAYERED_DIFFUSERS_FILES,
    WAN_T2V_1_3B_DIFFUSERS_FILES,
    WAN_VACE_1_3B_DIFFUSERS_FILES,
    Z_IMAGE_DIFFUSERS_FILES,
    assert_studio_execution_graph,
    studio_capability_definitions,
    studio_execution_spec_for_pair,
    studio_model_dependencies_for_pair,
    studio_model_requirements_for_pair,
    validate_studio_execution_specs,
)
from modiff.model_artifact_catalog import require_catalog_revision
from modiff.task_template_contracts import (
    TASK_TEMPLATE_CONTRACT_SCHEMA_VERSION,
    build_task_template_contracts,
)
from modiff.modelstore import modelstore
from modules import MODULE_MAP, parse_module_map
from utils.huggingface import (
    delete_model,
    download_hub_model,
    get_cache_diagnostics,
    get_local_models,
    plan_hub_model_download,
    search_hub,
    validate_hf_repo_id,
)
from utils.memory_menager import memory_manager
from utils.torch_utils import reset_memory_stats, get_memory_stats

MODULAR_OFFLOAD_SUPPORT = {
    "default": OFFLOAD_MODE_MODEL_CPU,
    "lowVram": OFFLOAD_MODE_MODEL_CPU,
    "emergency": OFFLOAD_MODE_GROUP_DISK,
    "modes": [OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK],
}

QWEN_MODULAR_OFFLOAD_SUPPORT = {
    "default": OFFLOAD_MODE_MODEL_CPU,
    "lowVram": OFFLOAD_MODE_MODEL_CPU,
    "emergency": OFFLOAD_MODE_GROUP_DISK,
    "modes": [
        OFFLOAD_MODE_NONE,
        OFFLOAD_MODE_MODEL_CPU,
        OFFLOAD_MODE_SEQUENTIAL_CPU,
        OFFLOAD_MODE_GROUP_CPU,
        OFFLOAD_MODE_GROUP_DISK,
    ],
}

# Runtime-hint copies are persisted and included in task events, so these
# execution selectors must be primitive, bounded, and drawn from the same
# closed vocabularies as the generic Diffusers runtime nodes. Keep these
# values local instead of importing modules.DiffusersRuntime during server
# startup; that module owns model-runtime imports which remain lazy.
RUNTIME_ATTENTION_BACKENDS = {
    "auto",
    "native",
    "_native_flash",
    "_native_efficient",
    "_native_math",
    "_native_cudnn",
    "flex",
    "flash",
    "flash_hub",
    "flash_varlen",
    "flash_varlen_hub",
    "flash_4_hub",
    "_flash_3",
    "_flash_varlen_3",
    "_flash_3_hub",
    "_flash_3_varlen_hub",
    "aiter",
    "sage",
    "sage_hub",
    "sage_varlen",
    "xformers",
}
RUNTIME_DENOISER_CACHE_MODES = {
    "none",
    "first_block",
    "magcache",
    "taylorseer",
    "pab",
    "fastercache",
    "text_kv",
}
RUNTIME_DEVICE_MAPS = {
    "none",
    "cuda",
    "auto",
    "balanced",
    "balanced_low_0",
    "cpu",
    "manual",
}

DIRECT_OFFLOAD_SUPPORT = {
    "default": OFFLOAD_MODE_MODEL_CPU,
    "lowVram": OFFLOAD_MODE_MODEL_CPU,
    "emergency": OFFLOAD_MODE_GROUP_DISK,
    "modes": [
        OFFLOAD_MODE_NONE,
        OFFLOAD_MODE_MODEL_CPU,
        OFFLOAD_MODE_SEQUENTIAL_CPU,
        OFFLOAD_MODE_GROUP_CPU,
        OFFLOAD_MODE_GROUP_DISK,
    ],
}

QWEN_IMAGE_EDIT_INPAINT_CONTRACT = {
    "available": True,
    "status": "supported",
    "reason": "Qwen Image Edit inpaint is available through the generic modules.DiffusersImage LoadPipeline -> Inpaint contract. Outpaint additionally uses the model-neutral Outpaint Canvas node to build the expanded image and boundary mask.",
    "source": "modules.DiffusersImage.Inpaint",
    "checkedInputs": {
        "loader": ["model_id", "dtype", "device", "auto_offload", "offload_mode", "quant_config"],
        "outpaint": [
            "image",
            "width",
            "height",
            "left",
            "right",
            "top",
            "bottom",
            "overlap",
            "feather",
            "fill_color",
            "canvas",
            "mask_image",
        ],
        "inpaint": [
            "pipeline",
            "image",
            "mask_image",
            "prompt",
            "negative_prompt",
            "true_cfg_scale",
            "strength",
            "num_inference_steps",
        ],
    },
    "missingInputs": [],
}

QWEN_IMAGE_EDIT_PLUS_INPAINT_CONTRACT = {
    "available": False,
    "status": "blocked",
    "reason": "Qwen Image Edit Plus does not yet have a confirmed native mask, mask_image, or masked_image_latents execution contract in MoDiff.",
    "source": "modules.ModularDiffusers.modular_utils.QWEN_IMAGE_EDIT_PLUS_NODE_SPECS",
    "checkedInputs": {
        "denoise": ["embeddings", "seed", "num_inference_steps", "guidance_scale", "image_latents"],
        "vae_encoder": ["image"],
        "text_encoder": ["prompt", "negative_prompt", "image"],
    },
    "missingInputs": ["mask", "mask_image", "masked_image_latents"],
}


class MissingConnectedOutputError(RuntimeError):
    pass


STUDIO_MODEL_CAPABILITIES = {
    "ZImageModularPipeline": {
        "modelType": "ZImageModularPipeline",
        "label": "Z-Image Turbo",
        "displayName": "Z-Image-Turbo",
        "family": "Z-Image",
        "defaultRepo": "Tongyi-MAI/Z-Image-Turbo",
        "downloadFiles": Z_IMAGE_DIFFUSERS_FILES,
        "artifactLabel": "Diffusers repo",
        "defaultDtype": "bfloat16",
        "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
        "recommendedSteps": 8,
        "recommendedGuidance": 1,
        "guidanceLabel": "Guidance",
        "supportsNegativePrompt": False,
        "supportsImageInput": True,
        "supportsMask": False,
        "supportsMultiImage": False,
        "supportsControlImage": False,
        "supportsLayers": False,
        "supportsLora": True,
        "offloadSupport": QWEN_MODULAR_OFFLOAD_SUPPORT,
        "lowVram": {"dtype": "bfloat16", "autoOffload": True, "offloadMode": OFFLOAD_MODE_MODEL_CPU, "steps": 8},
        "modes": ["text_to_image", "edit_image"],
        "modeRequirements": {
            "edit_image": {
                "requiredImages": ["referenceImages"],
                "note": "Requires one source image for image-to-image transformation.",
            }
        },
        "revisionCandidates": [
            require_catalog_revision(
                "Tongyi-MAI/Z-Image-Turbo",
                model_type="ZImageModularPipeline",
            )
        ],
        "executionStatus": "supported",
    },
    "QwenImageModularPipeline": {
        "modelType": "QwenImageModularPipeline",
        "label": "Qwen-Image-2512",
        "displayName": "Qwen-Image-2512",
        "family": "Qwen Image",
        "defaultRepo": "Qwen/Qwen-Image-2512",
        "downloadFiles": QWEN_IMAGE_2512_DIFFUSERS_FILES,
        "artifactLabel": "bfloat16 Diffusers repo",
        "defaultDtype": "bfloat16",
        "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
        "recommendedSteps": 50,
        "recommendedGuidance": 4.5,
        "guidanceLabel": "Guidance",
        "supportsImageInput": True,
        "supportsMask": True,
        "supportsMultiImage": False,
        "supportsControlImage": True,
        "supportsLayers": False,
        "supportsLora": True,
        "offloadSupport": QWEN_MODULAR_OFFLOAD_SUPPORT,
        "lowVram": {
            "dtype": "bfloat16",
            "quantizationMode": "bnb_4bit",
            "autoOffload": True,
            "offloadMode": OFFLOAD_MODE_MODEL_CPU,
            "steps": 28,
        },
        "modes": ["text_to_image", "edit_image", "inpaint", "control_image"],
        "executionStatus": "supported_with_model",
        "additionalRequirements": studio_model_requirements_for_pair(
            "QwenImageModularPipeline", "control_image"
        ),
        "modeRequirements": {
            "edit_image": {
                "requiredImages": ["referenceImages"],
                "note": "Requires one source image for image-to-image transformation.",
            },
            "inpaint": {
                "requiredImages": ["referenceImages", "maskImage"],
                "note": "Requires one source image and one mask image for inpainting.",
            },
            "control_image": {
                "modelRequirements": studio_model_requirements_for_pair(
                    "QwenImageModularPipeline", "control_image"
                ),
                "requiredImages": ["controlImage"],
                "note": "Requires the Qwen ControlNet Union model plus one control image.",
            }
        },
        "revisionCandidates": [
            require_catalog_revision(
                "Qwen/Qwen-Image-2512",
                model_type="QwenImageModularPipeline",
            )
        ],
    },
    "QwenImageEditModularPipeline": {
        "modelType": "QwenImageEditModularPipeline",
        "label": "Qwen-Image-Edit",
        "displayName": "Qwen-Image-Edit",
        "family": "Qwen Image",
        "defaultRepo": "Qwen/Qwen-Image-Edit",
        "downloadFiles": QWEN_IMAGE_EDIT_DIFFUSERS_FILES,
        "artifactLabel": "bfloat16 Diffusers repo",
        "defaultDtype": "bfloat16",
        "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
        "recommendedSteps": 40,
        "recommendedGuidance": 4,
        "guidanceLabel": "Guidance",
        "supportsImageInput": True,
        "supportsMask": True,
        "supportsMultiImage": False,
        "supportsControlImage": False,
        "supportsLayers": False,
        "supportsLora": True,
        "offloadSupport": DIRECT_OFFLOAD_SUPPORT,
        "lowVram": {
            "dtype": "bfloat16",
            "quantizationMode": "bnb_4bit",
            "autoOffload": True,
            "offloadMode": OFFLOAD_MODE_MODEL_CPU,
            "steps": 24,
        },
        "modes": ["edit_image", "inpaint", "outpaint"],
        "executionStatus": "supported_with_model",
        "notes": [
            "Inpaint and outpaint use generic Diffusers image nodes with the QwenImageEditInpaintPipeline adapter."
        ],
        "inpaintContract": QWEN_IMAGE_EDIT_INPAINT_CONTRACT,
        "modeRequirements": {
            "inpaint": {
                "requiredImages": ["referenceImages", "maskImage"],
                "note": "Requires one source image and one mask image.",
            },
            "outpaint": {
                "requiredImages": ["referenceImages"],
                "note": "Requires one source image; MoDiff builds the expanded canvas and boundary mask.",
            },
        },
    },
    "QwenImageEditPlusModularPipeline": {
        "modelType": "QwenImageEditPlusModularPipeline",
        "label": "Qwen-Image-Edit-2511",
        "displayName": "Qwen-Image-Edit-2511",
        "family": "Qwen Image",
        "defaultRepo": "Qwen/Qwen-Image-Edit-2511",
        "downloadFiles": QWEN_IMAGE_EDIT_2511_DIFFUSERS_FILES,
        "artifactLabel": "bfloat16 Diffusers repo",
        "defaultDtype": "bfloat16",
        "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
        "recommendedSteps": 40,
        "recommendedGuidance": 4,
        "guidanceLabel": "Guidance",
        "supportsImageInput": True,
        "supportsMask": False,
        "supportsMultiImage": True,
        "supportsControlImage": False,
        "supportsLayers": False,
        "supportsLora": True,
        "offloadSupport": QWEN_MODULAR_OFFLOAD_SUPPORT,
        "lowVram": {
            "dtype": "bfloat16",
            "quantizationMode": "bnb_4bit",
            "autoOffload": True,
            "offloadMode": OFFLOAD_MODE_MODEL_CPU,
            "steps": 24,
        },
        # Legacy clients fall back to this list when schema-v2 runnableModes is
        # unavailable. Keep it aligned with the executable profile so an
        # imported Edit Plus form cannot revive the unimplemented mask path.
        "modes": ["edit_image", "multi_image_reference_edit"],
        "executionStatus": "supported_with_model",
        "notes": ["Inpaint mask execution still requires a confirmed backend mask graph contract."],
        "inpaintContract": QWEN_IMAGE_EDIT_PLUS_INPAINT_CONTRACT,
        "modeRequirements": {
            "inpaint": {
                "requiredImages": ["referenceImages", "maskImage"],
                "note": QWEN_IMAGE_EDIT_PLUS_INPAINT_CONTRACT["reason"],
            }
        },
    },
    "QwenImageLayeredModularPipeline": {
        "modelType": "QwenImageLayeredModularPipeline",
        "label": "Qwen-Image-Layered",
        "displayName": "Qwen-Image-Layered",
        "family": "Qwen Image",
        "defaultRepo": "Qwen/Qwen-Image-Layered",
        "downloadFiles": QWEN_IMAGE_LAYERED_DIFFUSERS_FILES,
        "artifactLabel": "bfloat16 Diffusers repo",
        "defaultDtype": "bfloat16",
        "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
        "recommendedSteps": 50,
        "recommendedGuidance": 4,
        "guidanceLabel": "Guidance",
        "supportsImageInput": True,
        "supportsMask": False,
        "supportsMultiImage": False,
        "supportsControlImage": False,
        "supportsLayers": True,
        "supportsLora": True,
        "offloadSupport": MODULAR_OFFLOAD_SUPPORT,
        "lowVram": {
            "dtype": "bfloat16",
            "quantizationMode": "bnb_4bit",
            "autoOffload": True,
            "offloadMode": OFFLOAD_MODE_MODEL_CPU,
            "steps": 30,
        },
        "modes": ["layer_decomposition"],
        "executionStatus": "supported_with_model",
    },
    "WanVACEPipeline": {
        "modelType": "WanVACEPipeline",
        "label": "Wan VACE 1.3B",
        "displayName": "Wan2.1-VACE-1.3B-diffusers",
        "family": "Wan Video",
        "qualificationStatus": "qualified",
        "qualifiedModes": ["text_to_video", "video_inpaint", "video_outpaint", "control_to_video"],
        "defaultRepo": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        "downloadFiles": WAN_VACE_1_3B_DIFFUSERS_FILES,
        "artifactLabel": "Diffusers repo",
        "defaultDtype": "bfloat16",
        "defaultSize": {"width": 832, "height": 480, "aspectRatio": "16:9"},
        "recommendedSteps": 30,
        "recommendedGuidance": 5.0,
        "guidanceLabel": "Guidance",
        "supportsImageInput": True,
        "supportsMask": True,
        "supportsMultiImage": True,
        "supportsControlImage": True,
        "supportsLayers": False,
        "supportsLora": True,
        "supportsVideoInput": True,
        "supportsVideoMask": True,
        "outputKind": "video",
        "recommendedFrames": 81,
        "recommendedFps": 16,
        "conditioningScale": 1.0,
        "offloadSupport": DIRECT_OFFLOAD_SUPPORT,
        "lowVram": {
            "dtype": "bfloat16",
            "autoOffload": True,
            "offloadMode": OFFLOAD_MODE_MODEL_CPU,
            "steps": 24,
            "width": 832,
            "height": 480,
            "numFrames": 49,
        },
        "modes": [
            "text_to_video",
            "video_inpaint",
            "video_outpaint",
            "control_to_video",
        ],
        "executionStatus": "supported_with_model",
        "notes": [
            "Wan VACE is exposed as a direct Diffusers pipeline because this runtime does not expose a WanVACEModularPipeline.",
            "Exact color correction is provided by deterministic Video Color nodes; Wan VACE color edits are generative.",
            "Image-only and free-reference conditioning remain planning contracts after failing source-fidelity qualification and are not advertised as runnable modes.",
        ],
        "modeRequirements": {
            "video_inpaint": {
                "requiredVideos": ["sourceVideo", "maskVideo"],
                "note": "Requires source video and matching mask video.",
            },
            "video_outpaint": {
                "requiredVideos": ["sourceVideo", "maskVideo"],
                "note": "Requires source video and boundary/generation mask video.",
            },
            "control_to_video": {"requiredVideos": ["controlVideo"], "note": "Requires a prepared control video."},
        },
    },
    "WanVideoPipeline": {
        "modelType": "WanVideoPipeline",
        "label": "Wan 2.1 T2V 1.3B",
        "displayName": "Wan2.1-T2V-1.3B-Diffusers",
        "family": "Wan Video",
        "defaultRepo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "downloadFiles": WAN_T2V_1_3B_DIFFUSERS_FILES,
        "artifactLabel": "Diffusers repo",
        "defaultDtype": "bfloat16",
        "defaultSize": {"width": 832, "height": 480, "aspectRatio": "16:9"},
        "supportTier": "supported",
        "qualificationStatus": "qualified",
        "qualifiedModes": ["text_to_video"],
        "recommendedSteps": 50,
        "recommendedGuidance": 5.0,
        "guidanceLabel": "Guidance",
        "supportsImageInput": False,
        "supportsMask": False,
        "supportsMultiImage": False,
        "supportsControlImage": False,
        "supportsLayers": False,
        "supportsLora": True,
        "supportsVideoInput": True,
        "supportsVideoMask": False,
        "outputKind": "video",
        "recommendedFrames": 81,
        "recommendedFps": 15,
        "offloadSupport": DIRECT_OFFLOAD_SUPPORT,
        "lowVram": {
            "dtype": "bfloat16",
            "autoOffload": True,
            "offloadMode": OFFLOAD_MODE_MODEL_CPU,
            "steps": 24,
            "width": 832,
            "height": 480,
            "numFrames": 49,
        },
        "modes": ["text_to_video", "video_to_video", "video_color_edit"],
        "executionStatus": "supported_with_model",
        "notes": [
            "Uses the generic Diffusers video node with WanPipeline for text generation and WanVideoToVideoPipeline when strength is a real denoise control.",
            "VACE-only mask, control, and reference inputs are rejected before model load.",
            "The 832x480, 81-frame, 15 fps, 50-step text-to-video contract is mechanically qualified on Radeon 8060S; human creative review remains pending.",
        ],
        "modeRequirements": {
            "video_to_video": {"requiredVideos": ["sourceVideo"], "note": "Requires one source video."},
            "video_color_edit": {"requiredVideos": ["sourceVideo"], "note": "Requires one source video."},
        },
    },
    "LTXVideoPipeline": {
        "modelType": "LTXVideoPipeline",
        "label": "LTX-Video",
        "displayName": "LTX-Video Diffusers",
        "family": "LTX Video",
        "supportTier": "supported",
        "qualificationStatus": "qualified",
        "qualifiedModes": ["text_to_video", "image_to_video", "video_to_video", "reference_to_video"],
        "defaultRepo": "Lightricks/LTX-Video-0.9.8-13B-distilled",
        "artifactLabel": "Diffusers repo",
        "downloadFiles": LTX_VIDEO_DIFFUSERS_FILES,
        "defaultDtype": "bfloat16",
        "defaultSize": {"width": 704, "height": 480, "aspectRatio": "22:15"},
        "recommendedSteps": 8,
        "recommendedGuidance": 1.0,
        "guidanceLabel": "Guidance",
        "maxPromptTokens": 128,
        "supportsImageInput": True,
        "supportsMask": False,
        "supportsMultiImage": True,
        "supportsControlImage": False,
        "supportsLayers": False,
        "supportsLora": True,
        "supportsVideoInput": True,
        "supportsVideoMask": False,
        "outputKind": "video",
        "recommendedFrames": 81,
        "recommendedFps": 16,
        "conditioningScale": 1.0,
        "offloadSupport": DIRECT_OFFLOAD_SUPPORT,
        "lowVram": {
            "dtype": "bfloat16",
            "autoOffload": True,
            "offloadMode": OFFLOAD_MODE_MODEL_CPU,
            "steps": 8,
            "width": 704,
            "height": 480,
            "numFrames": 65,
        },
        "modes": ["text_to_video", "image_to_video", "video_to_video", "reference_to_video"],
        "executionStatus": "supported_with_model",
        "notes": [
            "LTX uses the generic Diffusers video nodes and the official LTXConditionPipeline contract.",
            "Text, image, source-video, and multi-reference conditioning use the same model-neutral graph contract.",
            "Prompts are validated against the artifact tokenizer and rejected above 128 tokens before diffusion begins.",
            "Mask and control-video modes are not advertised until a matching official Diffusers adapter is qualified.",
        ],
        "modeRequirements": {
            "image_to_video": {"requiredImages": ["referenceImages"], "note": "Requires one starting image."},
            "video_to_video": {"requiredVideos": ["sourceVideo"], "note": "Requires one source video."},
            "reference_to_video": {
                "requiredImages": ["referenceImages"],
                "note": "Requires one or more frame references.",
            },
        },
    },
    "AceStepAudioPipeline": {
        "modelType": "AceStepAudioPipeline",
        "label": "ACE-Step Audio",
        "displayName": "acestep-v15-xl-turbo-diffusers",
        "family": "ACE Audio",
        "defaultRepo": "ACE-Step/acestep-v15-xl-turbo-diffusers",
        "downloadFiles": ACE_STEP_DIFFUSERS_FILES,
        "artifactLabel": "Diffusers repo",
        "defaultDtype": "bfloat16",
        "defaultSize": {"width": 0, "height": 0, "aspectRatio": "custom"},
        "recommendedSteps": 8,
        "recommendedGuidance": 1.0,
        "guidanceLabel": "Guidance",
        "supportsImageInput": False,
        "supportsMask": False,
        "supportsMultiImage": False,
        "supportsControlImage": False,
        "supportsLayers": False,
        "supportsLora": False,
        "supportsAudioInput": True,
        "outputKind": "audio",
        "recommendedSampleRate": 48000,
        "recommendedDuration": 30,
        "offloadSupport": DIRECT_OFFLOAD_SUPPORT,
        "lowVram": {"dtype": "bfloat16", "autoOffload": True, "offloadMode": OFFLOAD_MODE_MODEL_CPU, "steps": 8},
        "modes": ["text_to_audio", "audio_variation", "audio_continuation", "audio_repaint"],
        "executionStatus": "supported_with_model",
        "modeRequirements": {
            "audio_variation": {"requiredAudio": ["sourceAudio"], "note": "Requires a source audio clip."},
            "audio_continuation": {
                "requiredAudio": ["sourceAudio"],
                "note": "Requires a source audio clip to continue.",
            },
            "audio_repaint": {"requiredAudio": ["sourceAudio"], "note": "Requires source audio plus repaint timing."},
        },
    },
    "FluxKontextPipeline": {
        "modelType": "FluxKontextPipeline",
        "label": "FLUX.1 Kontext dev",
        "displayName": "FLUX.1-Kontext-dev",
        "family": "FLUX Image",
        "defaultRepo": "black-forest-labs/FLUX.1-Kontext-dev",
        "downloadFiles": FLUX_KONTEXT_DIFFUSERS_FILES,
        "alternateArtifact": "black-forest-labs/FLUX.1-Kontext-dev-NVFP4",
        "artifactLabel": "Diffusers repo",
        "defaultDtype": "bfloat16",
        "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
        "recommendedSteps": 28,
        "recommendedGuidance": 3.5,
        "guidanceLabel": "Guidance",
        "supportsImageInput": True,
        "supportsMask": False,
        "supportsMultiImage": True,
        "supportsControlImage": False,
        "supportsLayers": False,
        "supportsLora": True,
        "offloadSupport": DIRECT_OFFLOAD_SUPPORT,
        "lowVram": {
            "dtype": "bfloat16",
            "autoOffload": True,
            "offloadMode": OFFLOAD_MODE_GROUP_DISK,
            "steps": 20,
            "width": 768,
            "height": 768,
        },
        "modes": ["edit_image", "multi_image_reference_edit"],
        "executionStatus": "expert_only",
        "modeRequirements": {
            "edit_image": {"requiredImages": ["referenceImages"], "note": "Requires a source image."}
        },
    },
    "FluxFillPipeline": {
        "modelType": "FluxFillPipeline",
        "label": "FLUX.1 Fill dev",
        "displayName": "FLUX.1-Fill-dev",
        "family": "FLUX Image",
        "defaultRepo": "black-forest-labs/FLUX.1-Fill-dev",
        "downloadFiles": FLUX_FILL_DIFFUSERS_FILES,
        "artifactLabel": "Diffusers repo",
        "defaultDtype": "bfloat16",
        "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
        "recommendedSteps": 28,
        "recommendedGuidance": 3.5,
        "guidanceLabel": "Guidance",
        "supportsImageInput": True,
        "supportsMask": True,
        "supportsMultiImage": False,
        "supportsControlImage": False,
        "supportsLayers": False,
        "supportsLora": True,
        "offloadSupport": DIRECT_OFFLOAD_SUPPORT,
        "lowVram": {
            "dtype": "bfloat16",
            "autoOffload": True,
            "offloadMode": OFFLOAD_MODE_GROUP_DISK,
            "steps": 20,
            "width": 768,
            "height": 768,
        },
        "modes": ["inpaint", "outpaint"],
        "executionStatus": "expert_only",
        "modeRequirements": {
            "inpaint": {"requiredImages": ["referenceImages", "maskImage"], "note": "Requires source and mask images."}
        },
    },
}

# The migrated exact pairs are generated from the execution-spec registry.
# The remaining records stay on the schema-v2 migration path until P0.3e.
STUDIO_MODEL_CAPABILITIES.update(studio_capability_definitions())


def studio_download_files_for_repo(repo_id):
    """Return the one reviewed app-download selection for a Studio repository."""

    matches = []
    for capability in STUDIO_MODEL_CAPABILITIES.values():
        if capability.get("defaultRepo") == repo_id and capability.get("downloadFiles"):
            matches.append(capability["downloadFiles"])
        requirements = list(capability.get("additionalRequirements") or [])
        for mode_requirement in (capability.get("modeRequirements") or {}).values():
            requirements.extend(mode_requirement.get("modelRequirements") or [])
        matches.extend(
            requirement["downloadFiles"]
            for requirement in requirements
            if requirement.get("repo") == repo_id and requirement.get("downloadFiles")
        )

    normalized = {tuple(sorted(set(selection))) for selection in matches}
    if len(normalized) > 1:
        raise RuntimeError(f"Conflicting reviewed Studio download selections for {repo_id}.")
    return list(next(iter(normalized), ()))


class WebServer:
    def __init__(
        self,
        modules: dict = {},
        host: str = "127.0.0.1",
        port: int = 8088,
        secure: bool = False,
        certfile: str = None,
        keyfile: str = None,
        cors: bool = False,
        cors_routes: list = [],
        client_max_size: int = DEFAULT_CLIENT_MAX_SIZE,
        work_dir: str = "data",
        data_dir: str = "data",
    ):
        self.instance = nanoid.generate(size=10)

        self.modules = modules
        self.ws_sessions = {}
        self.pending_ws_requests = {}

        self.interrupt_flag = False
        self._forced_restart_timer = None
        supervisor_queue_state = os.environ.get("MODIFF_SUPERVISOR_QUEUE_STATE")
        # The queue snapshot is a single-writer contract owned by a worker
        # explicitly launched by `main.py`'s process supervisor. Standalone
        # WebServer instances (tests, embeddings, tools) must never guess the
        # production snapshot path and overwrite an active run.
        self._supervisor_queue_state_path = Path(supervisor_queue_state) if supervisor_queue_state else None
        self._supervisor_queue_state_lock = threading.RLock() if supervisor_queue_state else None
        self._supervisor_queue_last_write = 0.0
        self.node_cache = {}
        self._active_graph_node_ids = set()
        self._last_auto_model_family = None
        self._last_auto_resource_signature = None
        self._last_runtime_fingerprint = None
        self._runtime_resource_lock = threading.RLock()
        self._runtime_resource_process = None
        self._runtime_resource_cached_at = 0.0
        self._runtime_resource_cached_snapshot = None
        self._runtime_disk_activity_sampler = DiskActivitySampler()
        try:
            import psutil

            # cpu_percent is interval based. Keep one Process instance and
            # prime both counters once so later samples describe the interval
            # between requests instead of repeatedly returning a first-call 0.
            psutil.cpu_percent(interval=None)
            self._runtime_resource_process = psutil.Process()
            self._runtime_resource_process.cpu_percent(interval=None)
        except Exception:
            self._runtime_resource_process = None

        self.queued_tasks = {}
        self.current_task = {}
        self.recent_tasks = []
        if supervisor_queue_state:
            try:
                persisted_queue = json.loads(self._supervisor_queue_state_path.read_text(encoding="utf-8"))
                persisted_recent = persisted_queue.get("recent") if isinstance(persisted_queue, dict) else None
                if isinstance(persisted_recent, list):
                    self.recent_tasks = [item for item in persisted_recent if isinstance(item, dict)][:30]
            except (OSError, TypeError, ValueError):
                pass
        self.task_graphs = {}
        self.optimization_jobs = {}
        self._runtime_install_leases = {}
        self._runtime_install_gate_tokens = {}
        self._runtime_mutation_gate = None
        self._active_nonruntime_mutations = 0

        self.main_queue = asyncio.Queue()
        self.background_queue = asyncio.Queue()
        self._shutdown_event = asyncio.Event()
        self.studio_history_lock = asyncio.Lock()
        # Graph execution runs in an executor thread while the Studio output
        # routes run on the aiohttp loop.  The asyncio lock cannot serialize
        # those two callers, so protect the shared history file with a small
        # process-local lock as well.
        self.studio_history_file_lock = threading.RLock()
        self.hf_download_semaphore = asyncio.Semaphore(2)
        self.download_reservation_lock = asyncio.Lock()
        self.hf_download_tasks = {}
        self.template_gallery_install_task = None
        self.template_gallery_reserved_bytes = 0
        # ROCm and MPS commonly use system RAM as accelerator memory. Loading a
        # large pipeline while hf-xet is assembling another model can exhaust
        # the same physical pool and let the kernel kill the app. Keep graph
        # execution and app-managed model I/O mutually exclusive on those
        # shared-memory runtimes; discrete CUDA retains concurrent downloads.
        try:
            import torch

            self.serialize_model_io = bool(getattr(torch.version, "hip", None)) or bool(
                getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)()
            )
        except Exception:
            self.serialize_model_io = False
        self.model_io_lock = asyncio.Lock()

        self.main_worker_task = None
        self.background_worker_task = None
        # ``run`` binds the server to the active application loop.  Keeping an
        # explicit pre-run state lets node callbacks safely emit best-effort
        # progress while a WebServer is used by tests or embedding tools.
        self.loop = None
        self.runner = None
        self.site = None

        self.host = host
        self.port = port
        self.ssl_context = None
        if secure and certfile and keyfile:
            import ssl

            self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            self.ssl_context.load_cert_chain(certfile, keyfile)

        self.client_max_size = client_max_size
        self.work_dir = work_dir
        self.data_dir = data_dir
        self._load_runtime_jobs()
        # Prime interval counters at startup so the first browser request can
        # usually report active time instead of waiting for a second poll.
        self._runtime_disk_activity_sampler.sample(self.data_dir)
        self.app = web.Application(
            client_max_size=self.client_max_size,
            middlewares=[self._mutation_origin_middleware],
        )

        # A remote Gallery build resolves media directly from its immutable
        # public Hugging Face Dataset and intentionally has no local Gallery
        # directory. Offline/local builds materialize that directory and keep
        # the same-origin route.
        routes = [
            web.static("/assets", "web/assets", append_version=True),
        ]
        if TEMPLATE_GALLERY_ROOT.is_dir():
            routes.append(web.static("/template-gallery", str(TEMPLATE_GALLERY_ROOT), append_version=True))
        routes.extend(
            [
                web.get("/", self.index),
                web.get("/favicon.ico", self.favicon),
                web.get("/ws", self.websocket),
                web.get(r"/nodes{id:/?([\w\d_-]+/[\w\d_-]+)?}", self.nodes),
                web.post("/fields/action", self.field_action),
                web.get("/cache/{node}/{field}", self.cache),
                web.get("/cache/{node}/{field}/{index}", self.cache),
                web.delete("/cache", self.delete_cache),
                web.get("/listdir", self.listdir),
                web.get("/listgraphs", self.listgraphs),
                web.get("/workflows", self.workflows_list),
                web.get("/workflows/{workflow_id}", self.workflow_get),
                web.put("/workflows/{workflow_id}", self.workflow_put),
                web.delete("/workflows/{workflow_id}", self.workflow_delete),
                web.get("/file", self.fileGet),
                web.post("/file", self.filePost),
                web.get("/media/capabilities", self.media_capabilities),
                web.get("/media/probe", self.media_probe),
                web.get("/media/export", self.media_export),
                web.get("/media/preview", self.media_preview),
                web.get("/preview", self.preview),
                web.post("/graph", self.graph),
                web.get("/queue", self.get_queue),
                web.get("/runs/{task_id}", self.get_run),
                web.delete("/queue/{task_id}", self.delete_task),
                web.post("/stop", self.stop_execution),
                web.get("/health", self.runtime_status),
                web.get("/runtime/status", self.runtime_status),
                web.get("/runtime/resources", self.runtime_resources),
                web.get("/runtime/options", self.runtime_options),
                web.get("/runtime/optimizations", self.runtime_optimizations),
                web.post("/runtime/optimizations/install", self.runtime_optimization_install),
                web.get("/runtime/optimizations/jobs/{job_id}", self.runtime_optimization_job),
                web.post(
                    "/runtime/optimizations/jobs/{job_id}/cancel",
                    self.runtime_optimization_job_cancel,
                ),
                web.post("/runtime/optimizations/activate", self.runtime_optimization_activate),
                web.post("/runtime/optimizations/rollback", self.runtime_optimization_rollback),
                web.post("/runtime/optimizations/enable", self.runtime_optimization_enable),
                web.post("/runtime/optimizations/probe", self.runtime_optimization_probe),
                web.get("/runtime/optimizations/receipts", self.runtime_optimization_receipts),
                web.post("/runtime/optimizations/qualify", self.runtime_optimization_qualify),
                web.get("/runtime/optional-runtimes", self.runtime_optional_runtimes),
                web.post("/runtime/optional-runtimes/install", self.runtime_optional_runtime_install),
                web.get(
                    "/runtime/optional-runtimes/jobs/{job_id}",
                    self.runtime_optimization_job,
                ),
                web.post(
                    "/runtime/optional-runtimes/jobs/{job_id}/cancel",
                    self.runtime_optimization_job_cancel,
                ),
                web.post(
                    "/runtime/optional-runtimes/activate",
                    self.runtime_optional_runtime_activate,
                ),
                web.post(
                    "/runtime/optional-runtimes/rollback",
                    self.runtime_optional_runtime_rollback,
                ),
                web.get("/system_stats", self.system_stats),
                web.get("/runtime/gpu_processes", self.runtime_gpu_processes),
                web.post("/runtime/gpu_cleanup", self.runtime_gpu_cleanup),
                web.get("/media_assets", self.media_assets_list),
                web.delete("/media_assets", self.media_assets_cleanup),
                web.get("/model_capabilities", self.model_capabilities),
                web.get("/model_artifact_catalog", self.model_artifact_catalog),
                web.post("/auto_resource/plan", self.auto_resource_plan),
                web.post("/auto_resource/plans", self.auto_resource_plans),
                web.get("/auto_resource/history", self.auto_resource_history),
                web.delete("/auto_resource/history", self.auto_resource_history_clear),
                web.get("/model_fingerprints", self.model_fingerprints),
                web.get("/local_models", self.local_models),
                web.get("/hf_cache", self.hf_cache),
                web.post("/hf_token", self.hf_token),
                web.get("/model_cache/diagnostics", self.model_cache_diagnostics),
                web.get("/custom_modules", self.custom_modules_list),
                web.post("/custom_modules/refresh", self.custom_modules_refresh),
                web.post("/custom_modules/install", self.custom_modules_install),
                web.post("/custom_modules/{name}/update", self.custom_modules_update),
                web.post("/custom_modules/{name}/disable", self.custom_modules_disable),
                web.post("/custom_modules/{name}/enable", self.custom_modules_enable),
                web.get("/studio_outputs", self.studio_outputs_get),
                web.post("/studio_outputs", self.studio_outputs_post),
                web.patch("/studio_outputs/{output_id}", self.studio_outputs_patch),
                web.delete("/studio_outputs/{output_id}", self.studio_outputs_delete),
                web.get("/studio/blocks", self.studio_blocks_get),
                web.post("/studio/blocks", self.studio_blocks_post),
                web.get("/studio/blocks/{block_id}", self.studio_block_get),
                web.delete("/studio/blocks/{block_id}", self.studio_block_delete),
                web.get("/workflow_shares", self.workflow_shares_list),
                web.post("/workflows/share", self.workflow_share_post),
                web.get("/workflows/share/{share_id}/media/{filename}", self.workflow_share_media_get),
                web.get("/workflows/share/{share_id}", self.workflow_share_get),
                web.delete("/hf_cache/{hash}", self.hf_cache_delete),
                web.get("/hf_hub", self.hf_hub),
                web.get("/hf_download/plan", self.hf_download_plan),
                web.get("/hf_download/status", self.hf_download_status),
                web.post("/hf_download", self.hf_download),
                web.get("/template_gallery/status", self.template_gallery_status),
                web.get("/template_gallery/plan", self.template_gallery_plan),
                web.post("/template_gallery/install", self.template_gallery_install),
                web.get("/static/{module}/{file}", self.user_assets),
                web.get("/stream", self.stream),
            ]
        )
        self.app.add_routes(routes)

        # serve the user assets
        try:
            self.app.add_routes(web.static("/user", "web/user", append_version=True))
        except Exception:
            pass

        # set up the cors routes
        if cors:
            cors = cors_setup(
                self.app,
                defaults={
                    cors_route: ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*")
                    for cors_route in cors_routes
                },
            )
            for route in list(self.app.router.routes()):
                cors.add(route)

    async def run(self):
        try:
            hardware = get_hardware_snapshot(self.data_dir, refresh=True)
            logger.info(format_hardware_summary(hardware))
        except Exception:
            logger.warning("Unable to log the startup hardware summary", exc_info=True)

        # Get the current event loop
        self.loop = asyncio.get_event_loop()

        # Start both workers
        self.main_worker_task = self.loop.create_task(self._main_worker())
        self.background_worker_task = self.loop.create_task(self._background_worker())

        self.runner = web.AppRunner(self.app, access_log=None)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, host=self.host, port=self.port, ssl_context=self.ssl_context)
        await self.site.start()
        self._persist_supervisor_queue_state(force=True)

    async def cleanup(self):
        # Signal shutdown to workers
        self._shutdown_event.set()

        # Stop the web server from accepting new connections.
        if self.site:
            await self.site.stop()

        # Close all active websocket connections with timeout
        if self.ws_sessions:
            close_coroutines = []
            for sid, ws in list(self.ws_sessions.items()):
                try:
                    if not ws.closed:
                        close_coroutines.append(ws.close())
                except Exception as e:
                    logger.debug(f"Error preparing to close websocket {sid}: {e}")

            if close_coroutines:
                try:
                    await asyncio.wait_for(asyncio.gather(*close_coroutines, return_exceptions=True), timeout=0.5)
                except asyncio.TimeoutError:
                    logger.warning("Websocket connections did not close within timeout, forcing shutdown")

            # Clear the sessions dict
            self.ws_sessions.clear()

        # Cancel and clean up the background workers.
        if self.main_worker_task:
            self.main_worker_task.cancel()
        if self.background_worker_task:
            self.background_worker_task.cancel()

        # Add sentinel values to queues to wake up workers
        try:
            self.main_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        try:
            self.background_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

        # Wait for the workers to finish with a timeout
        tasks_to_wait = [task for task in [self.main_worker_task, self.background_worker_task] if task]
        if tasks_to_wait:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks_to_wait, return_exceptions=True), timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("Worker tasks did not finish within timeout, forcing shutdown")

        # Cleanup the runner.
        if self.runner:
            await self.runner.cleanup()

    """
    ╭───────────────╮
          Queue
    ╰───────────────╯
    """

    def _current_task_snapshot(self):
        if not self.current_task:
            return None
        runtime_hints = self.current_task.get("runtimeHints")
        return {
            "task_id": self.current_task.get("task_id"),
            "name": self.current_task.get("name") or "Graph execution",
            "sid": self.current_task.get("sid"),
            "started_at": self.current_task.get("started_at"),
            "updated_at": self.current_task.get("updated_at"),
            "progress": self.current_task.get("progress", 0),
            "status": "running",
            "attempt_index": self.current_task.get("attempt_index", 0),
            "current_node": self.current_task.get("current_node"),
            "current_node_name": self.current_task.get("current_node_name"),
            "node_progress": self.current_task.get("node_progress"),
            "phase": self.current_task.get("phase"),
            "message": self.current_task.get("message"),
            "current_step": self.current_task.get("current_step"),
            "total_steps": self.current_task.get("total_steps"),
            "component": self.current_task.get("component"),
            "shard_current": self.current_task.get("shard_current"),
            "shard_total": self.current_task.get("shard_total"),
            "elapsed_seconds": self.current_task.get("elapsed_seconds"),
            "average_step_seconds": self.current_task.get("average_step_seconds"),
            "eta_seconds": self.current_task.get("eta_seconds"),
            "last_heartbeat_at": self.current_task.get("last_heartbeat_at"),
            "resource_snapshot": self.current_task.get("resource_snapshot"),
            "phase_timings": self.current_task.get("phase_timings"),
            "runtimeFingerprint": self.current_task.get("runtimeFingerprint"),
            "resourceCandidateId": self.current_task.get("resourceCandidateId"),
            "runtimeMeasurement": self.current_task.get("runtimeMeasurement"),
            "deterministicMode": self.current_task.get("deterministicMode"),
            **self._current_run_identity_payload(),
            **self._run_navigation_payload(runtime_hints),
        }

    def _initial_graph_execution_state(self, args):
        """Describe the first executable node before the worker enters model code."""
        graph = args[0] if isinstance(args, tuple) and args else None
        if not isinstance(graph, dict):
            return {}
        nodes = graph.get("nodes")
        paths = graph.get("paths")
        if not isinstance(nodes, dict) or not isinstance(paths, list):
            return {}
        first_node_id = next(
            (
                node_id
                for path in paths
                if isinstance(path, list)
                for node_id in path
                if node_id in nodes and isinstance(nodes[node_id], dict)
            ),
            None,
        )
        if first_node_id is None:
            return {}
        node = nodes[first_node_id]
        module = str(node.get("module") or "")
        action = str(node.get("action") or "")
        phase = node_execution_phase(module, action)
        return {
            "current_node": first_node_id,
            "current_node_name": f"{module}.{action}".strip("."),
            "node_progress": -1,
            "phase": phase,
            "message": node_execution_message(module, action, phase),
            "updated_at": time.time(),
        }

    def _record_terminal_task(self, status, *, error_payload=None):
        if not self.current_task:
            return None
        completed_at = time.time()
        active_phase = self.current_task.get("phase")
        active_phase_started_at = self.current_task.get("_phase_started_at")
        phase_timings = self.current_task.setdefault("phase_timings", {})
        if active_phase and isinstance(phase_timings, dict) and isinstance(active_phase_started_at, (int, float)):
            phase_timings[active_phase] = float(phase_timings.get(active_phase, 0.0)) + max(
                0.0, completed_at - float(active_phase_started_at)
            )
            self.current_task["_phase_started_at"] = completed_at
        entry = {
            "task_id": self.current_task.get("task_id"),
            "name": self.current_task.get("name"),
            "sid": self.current_task.get("sid"),
            "started_at": self.current_task.get("started_at"),
            "completed_at": completed_at,
            "updated_at": completed_at,
            "progress": 100 if status == "completed" else self.current_task.get("progress", 0),
            "status": status,
            "attempt_index": self.current_task.get("attempt_index", 0),
            "current_node": self.current_task.get("current_node"),
            "current_node_name": self.current_task.get("current_node_name"),
            "node_progress": self.current_task.get("node_progress"),
            "phase": self.current_task.get("phase"),
            "message": self.current_task.get("message"),
            "current_step": self.current_task.get("current_step"),
            "total_steps": self.current_task.get("total_steps"),
            "component": self.current_task.get("component"),
            "shard_current": self.current_task.get("shard_current"),
            "shard_total": self.current_task.get("shard_total"),
            "elapsed_seconds": self.current_task.get("elapsed_seconds"),
            "average_step_seconds": self.current_task.get("average_step_seconds"),
            "eta_seconds": self.current_task.get("eta_seconds"),
            "last_heartbeat_at": self.current_task.get("last_heartbeat_at"),
            "resource_snapshot": self.current_task.get("resource_snapshot"),
            "phase_timings": self.current_task.get("phase_timings"),
            "runtimeFingerprint": self.current_task.get("runtimeFingerprint"),
            "resourceCandidateId": self.current_task.get("resourceCandidateId"),
            "runtimeMeasurement": self.current_task.get("runtimeMeasurement"),
            **self._current_run_identity_payload(),
            # Retain navigation metadata with recent runs as well as the live
            # queue snapshot. A refreshed client can then open the originating
            # workflow (or its failure details) even after execution completed
            # while it was disconnected.
            **self._run_navigation_payload(self.current_task.get("runtimeHints")),
        }
        if isinstance(error_payload, dict):
            for key in (
                "message",
                "error",
                "exception_type",
                "category",
                "error_code",
                "recovery_hint",
                "node",
                "node_name",
                "oom",
            ):
                if error_payload.get(key) is not None:
                    entry[key] = error_payload.get(key)
        self.recent_tasks = [entry, *[item for item in self.recent_tasks if item.get("task_id") != entry["task_id"]]][
            :30
        ]
        retained_ids = {str(item.get("task_id")) for item in self.recent_tasks if item.get("task_id")}
        retained_ids.update(str(value) for value in self.queued_tasks)
        if self.current_task.get("task_id"):
            retained_ids.add(str(self.current_task["task_id"]))
        self.task_graphs = {key: value for key, value in self.task_graphs.items() if key in retained_ids}
        return entry

    def record_node_progress(self, payload):
        if not self.current_task or not isinstance(payload, dict):
            return payload
        task_id = payload.get("task_id")
        if task_id and task_id != self.current_task.get("task_id"):
            return payload
        now = time.time()
        node_progress = payload.get("progress")
        prior_node_progress = self.current_task.get("node_progress")
        prior_current_step = self.current_task.get("current_step")
        prior_phase = self.current_task.get("phase")
        next_phase = payload.get("phase") or prior_phase
        phase_started_at = self.current_task.get("_phase_started_at")
        phase_timings = self.current_task.setdefault("phase_timings", {})
        if not isinstance(phase_timings, dict):
            phase_timings = {}
            self.current_task["phase_timings"] = phase_timings
        if next_phase != prior_phase:
            if prior_phase and isinstance(phase_started_at, (int, float)):
                phase_timings[prior_phase] = float(phase_timings.get(prior_phase, 0.0)) + max(
                    0.0, now - float(phase_started_at)
                )
            self.current_task["_phase_started_at"] = now
        elif not isinstance(phase_started_at, (int, float)):
            self.current_task["_phase_started_at"] = now
        self.current_task.update(
            {
                "updated_at": now,
                "last_heartbeat_at": payload.get("last_heartbeat_at") or now,
                "current_node": payload.get("node") or self.current_task.get("current_node"),
                "node_progress": node_progress,
                "phase": next_phase,
                "message": payload.get("message") or self.current_task.get("message"),
            }
        )
        # Node lifecycle events without step metrics must not erase the latest
        # denoising sample. Preserving the final sample gives reconnects and
        # terminal receipts an honest duration/step record through decode/save.
        for field in (
            "current_step",
            "total_steps",
            "elapsed_seconds",
            "average_step_seconds",
            "eta_seconds",
            "component",
            "shard_current",
            "shard_total",
            "resource_snapshot",
        ):
            if payload.get(field) is not None:
                self.current_task[field] = payload.get(field)
        if isinstance(node_progress, (int, float)) and node_progress >= 0:
            completed = float(self.current_task.get("completed_progress", 0.0))
            node_weight = float(self.current_task.get("current_node_weight", 0.0))
            overall = min(100.0, max(completed, completed + node_weight * min(100.0, float(node_progress)) / 100.0))
            self.current_task["progress"] = int(overall)
            payload["overall_progress"] = int(overall)
        payload["updated_at"] = now
        payload["last_heartbeat_at"] = self.current_task.get("last_heartbeat_at")
        payload["phase_timings"] = deepcopy(phase_timings)
        # The node-start snapshot is force-written with indeterminate progress.
        # A generator often publishes 0/N immediately afterward, inside the
        # normal 200 ms write throttle, and may then spend minutes in its first
        # offloaded model step. Force only that indeterminate-to-measurable
        # transition so a refreshed client and the process-external
        # notification shelf retain honest denoising state throughout it.
        first_measurable_sample = (
            isinstance(node_progress, (int, float))
            and node_progress >= 0
            and (not isinstance(prior_node_progress, (int, float)) or prior_node_progress < 0)
        ) or (
            payload.get("current_step") == 0 and prior_current_step is None and payload.get("total_steps") is not None
        )
        self._persist_supervisor_queue_state(force=first_measurable_sample)
        return payload

    def _get_queue(self):
        # task_list_sorted = {k: v for k, v in sorted(self.queued_tasks.items(), key=lambda x: x[1]['queued_at'], reverse=True)}
        # filter out keys that are not needed for the client
        queued_tasks = {
            k: {
                "name": v["name"],
                "sid": v["sid"],
                "queued_at": v["queued_at"],
                "task_id": k,
                "queue_position": index + 1,
                **self._run_identity_payload(v.get("runtimeHints")),
                **self._run_navigation_payload(v.get("runtimeHints")),
            }
            for index, (k, v) in enumerate(self.queued_tasks.items())
        }

        current_task = self._current_task_snapshot()

        return queued_tasks, current_task

    def _persist_supervisor_queue_state(self, *, force=False):
        """Expose queue truth to the process-external emergency control plane."""
        path = getattr(self, "_supervisor_queue_state_path", None)
        lock = getattr(self, "_supervisor_queue_state_lock", None)
        if path is None or lock is None:
            # Lightweight WebServer fixtures and unsupervised embeddings do
            # not configure the process-external control plane.
            return
        now = time.monotonic()
        last_write = getattr(self, "_supervisor_queue_last_write", 0.0)
        if not force and now - last_write < 0.2:
            return
        with lock:
            now = time.monotonic()
            last_write = getattr(self, "_supervisor_queue_last_write", 0.0)
            if not force and now - last_write < 0.2:
                return
            queued, current = self._get_queue()
            payload = {
                "workerPid": os.getpid(),
                "updatedAt": time.time(),
                "queued": queued,
                "current": current,
                "recent": self.recent_tasks,
            }
            temporary = path.with_suffix(path.suffix + ".tmp")
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                os.replace(temporary, path)
                self._supervisor_queue_last_write = now
            except Exception:
                logger.warning("Could not persist supervisor queue state", exc_info=True)
                try:
                    temporary.unlink(missing_ok=True)
                except (OSError, TypeError, ValueError):
                    pass

    async def queue_task(
        self,
        task,
        args,
        future,
        sid,
        name=None,
        runtime_hints=None,
        optional_runtime_requirement=None,
    ):
        if self._runtime_mutation_gate is not None:
            raise OverlayInstallBusy(
                "Runtime work is unavailable during runtime mutation or recovery."
            )
        task_name = name or f"Unnamed task ({task.__name__})"
        graph = (
            args[0]
            if task_name == "Graph execution" and isinstance(args, tuple) and args and isinstance(args[0], dict)
            else None
        )
        overlay_status = os.environ.get("MODIFF_RUNTIME_OVERLAY_STATUS", "base")
        if overlay_status in {
            "busy_recovery_only",
            "repair_required",
            "restart_required",
        }:
            requirement = (
                graph_optional_runtime_requirement(graph)
                if graph is not None
                else optional_runtime_requirement
            )
            base_requirement = bool(
                isinstance(requirement, dict)
                and requirement.get("delivery") == "base"
                and requirement.get("requiredNow") is False
                and requirement.get("state") == "base_satisfied"
            )
            if requirement is not None and not base_requirement:
                raise OverlayInstallBusy(
                    "Runtime work is unavailable during runtime mutation or recovery."
                )
        task_id = nanoid.generate(size=12)
        runtime_hints = (
            self._coerce_runtime_hints(graph.get("runtimeHints"))
            if graph
            else self._coerce_runtime_hints(runtime_hints)
        )
        if graph is not None:
            if runtime_hints is None:
                graph.pop("runtimeHints", None)
            else:
                # Replace the untrusted object before the graph is copied,
                # queued, persisted, or later inspected by plan application.
                graph["runtimeHints"] = runtime_hints

        self.queued_tasks[task_id] = {
            "task": task,
            "args": args,
            "future": future,
            "sid": sid,
            "queued_at": time.time(),
            "name": task_name,
            "runtimeHints": runtime_hints,
        }
        preview_state = None
        if graph is not None:
            self.task_graphs[task_id] = deepcopy(graph)
            preview_state = self._mark_studio_preview_slots_pending(graph, task_id)
        await self.main_queue.put((task, args, future, task_id))
        self._persist_supervisor_queue_state(force=True)

        task_list, current_task = self._get_queue()

        queued_message = {
            "type": "task_queued",
            "task_id": task_id,
            "sid": sid,
            **self._run_identity_payload(runtime_hints),
            "queued": task_list,
            "current": current_task,
        }
        if preview_state and preview_state["previewSlots"]:
            queued_message["preview_slots"] = preview_state["previewSlots"]
            queued_message["preview_state_revision"] = preview_state["revision"]
        self.queue_message(queued_message)

        return task_id

    async def get_queue(self, _):
        """
        HTTP endpoint to return the tasks queue and the current task.
        """
        task_list, current_task = self._get_queue()
        return web.json_response(
            {
                "queued": task_list,
                "current": current_task,
                # Completed workflow snapshots are available lazily from
                # /runs/{task_id}; do not resend dozens of full graphs on every
                # queue poll.
                "recent": compact_task_history(self.recent_tasks),
            }
        )

    def _studio_outputs_for_run(self, task_id, client_run_id=None):
        """Return persisted outputs whose recorded run identity matches exactly."""
        normalized_task_id = str(task_id or "").strip()
        normalized_client_run_id = str(client_run_id or "").strip()
        if not normalized_task_id:
            return []

        def identity_values(output, key, provenance_key):
            values = set()

            def add(value):
                if value is None:
                    return
                normalized = str(value).strip()
                if normalized:
                    values.add(normalized)

            add(output.get(key))
            for container_key in ("provenance", "backendProvenance"):
                container = output.get(container_key)
                if isinstance(container, dict):
                    add(container.get(provenance_key))
            media_items = output.get("mediaItems")
            if isinstance(media_items, list):
                for item in media_items:
                    if isinstance(item, dict):
                        add(item.get(key))
            return values

        matches = []
        for output in self._read_studio_outputs():
            if not isinstance(output, dict):
                continue
            task_ids = identity_values(output, "taskId", "backendExecutionId")
            if task_ids != {normalized_task_id}:
                continue
            if normalized_client_run_id:
                client_run_ids = identity_values(output, "clientRunId", "clientRunId")
                # Exact task identity is sufficient for legacy records that
                # predate client-run IDs. When present, however, every recorded
                # client identity must agree with the originating run.
                if client_run_ids and client_run_ids != {normalized_client_run_id}:
                    continue
            matches.append(output)
        return matches

    async def get_run(self, request):
        task_id = request.match_info.get("task_id")
        task = None
        if self.current_task and self.current_task.get("task_id") == task_id:
            task = self._current_task_snapshot()
        elif task_id in self.queued_tasks:
            task = self._get_queue()[0].get(task_id)
        else:
            task = next((item for item in self.recent_tasks if item.get("task_id") == task_id), None)
        if task is None:
            return web.json_response({"error": True, "message": "Run not found."}, status=404)
        graph = self.task_graphs.get(task_id)
        runtime_hints = graph.get("runtimeHints") if isinstance(graph, dict) else None
        client_run_id = runtime_hints.get("clientRunId") if isinstance(runtime_hints, dict) else None
        if not client_run_id and isinstance(task, dict):
            client_run_id = task.get("client_run_id")
        outputs = self._studio_outputs_for_run(task_id, client_run_id)
        if not isinstance(runtime_hints, dict):
            for output in reversed(outputs):
                api_graph = output.get("apiGraphSnapshot") if isinstance(output, dict) else None
                persisted_hints = api_graph.get("runtimeHints") if isinstance(api_graph, dict) else None
                if isinstance(persisted_hints, dict):
                    runtime_hints = self._coerce_runtime_hints(persisted_hints)
                    break
        workflow_id = runtime_hints.get("workflowTabId") if isinstance(runtime_hints, dict) else None
        workflow_title = runtime_hints.get("workflowTitle") if isinstance(runtime_hints, dict) else None
        workflow_snapshot = runtime_hints.get("workflowSnapshot") if isinstance(runtime_hints, dict) else None
        if isinstance(task, dict):
            workflow_id = workflow_id or task.get("workflow_tab_id")
            workflow_title = workflow_title or task.get("workflow_title")
            workflow_snapshot = workflow_snapshot or task.get("workflow_snapshot")
        return web.json_response(
            {
                "task": task,
                "workflow_id": workflow_id,
                "workflow_title": workflow_title,
                "workflow_snapshot": workflow_snapshot,
                "outputs": outputs,
            }
        )

    async def delete_task(self, request):
        """
        HTTP endpoint to delete a task from the queue.
        """
        task_id = request.match_info.get("task_id")
        if task_id in self.queued_tasks:
            task = self.queued_tasks.pop(task_id)
            self.task_graphs.pop(task_id, None)
            preview_state = self._mark_studio_preview_run_terminal(task_id, "cancelled")
            self._persist_supervisor_queue_state(force=True)
            logger.info(f"Task {task_id} {task['name']} deleted from queue.")
            task_list, current_task = self._get_queue()
            self.queue_message(
                {
                    "type": "task_cancelled",
                    "task_id": task_id,
                    **self._run_identity_payload(task.get("runtimeHints")),
                    "queued": task_list,
                    "current": current_task,
                }
            )
            response = {"error": False, "task_id": task_id, "queued": task_list, "current": current_task}
            if preview_state:
                response.update(
                    preview_slots=preview_state["previewSlots"],
                    preview_state_revision=preview_state["revision"],
                )
            return web.json_response(response)
        elif self.current_task and self.current_task["task_id"] == task_id:
            return web.json_response(
                {"error": True, "message": "Task is already running and cannot be cancelled.", "task_id": task_id},
                status=400,
            )

        return web.json_response(
            {"error": True, "message": "Task not found in queue.", "task_id": task_id}, status=404
        )

    async def _main_worker(self):
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Use wait_for with a timeout to check shutdown periodically
                    queue_item = await asyncio.wait_for(self.main_queue.get(), timeout=1.0)

                    # Check for sentinel value (None) indicating shutdown
                    if queue_item is None:
                        self.main_queue.task_done()
                        break

                    task, args, future, task_id = queue_item

                    if task_id not in self.queued_tasks:
                        logger.debug(f"Task {task_id} was cancelled, skipping.")
                        if future:
                            future.set_exception(asyncio.CancelledError(f"Task {task_id} was cancelled"))
                        self.main_queue.task_done()
                        continue

                    current_task = self.queued_tasks.pop(task_id)
                    runtime_hints = current_task.get("runtimeHints")

                    self.current_task = {
                        "task_id": task_id,
                        "task": task,
                        "name": current_task["name"],
                        "sid": current_task["sid"],
                        "started_at": time.time(),
                        "progress": 0,
                        "attempt_index": 0,
                        "args": args,
                        "runtimeHints": runtime_hints,
                        # The selected Auto recipe is known at admission time.
                        # Expose it throughout execution instead of leaving the
                        # supervisor/notification snapshot blank until the
                        # terminal receipt is assembled.
                        "resourceCandidateId": (
                            runtime_hints.get("autoResourceCandidateId") if isinstance(runtime_hints, dict) else None
                        ),
                    }
                    if current_task.get("name") == "Graph execution":
                        self.current_task.update(self._initial_graph_execution_state(args))
                        self.current_task["_phase_started_at"] = time.time()
                        self.current_task["phase_timings"] = {}
                    self._persist_supervisor_queue_state(force=True)
                    task_list, current_task = self._get_queue()
                    self.queue_message(
                        {
                            "type": "task_started",
                            "task_id": task_id,
                            "attempt_index": 0,
                            **self._current_run_identity_payload(),
                            "queued": task_list,
                            "current": current_task,
                        }
                    )
                    # Give aiohttp one scheduling turn to flush the graph-queued
                    # response before model loading begins in the executor. Some
                    # pipeline loaders hold the GIL for long stretches; without
                    # this grace period the client can time out even though the
                    # graph was accepted and is already running.
                    if current_task.get("name") == "Graph execution":
                        await asyncio.sleep(0.05)
                    terminal_status = "completed"
                    failure_payload = None
                    try:
                        if isinstance(args, tuple):
                            callback = partial(task, *args)
                        elif isinstance(args, dict):
                            callback = partial(task, **args)
                        else:
                            callback = partial(task, args)
                        serialize_model_io = current_task.get("name") == "Graph execution"
                        if serialize_model_io and self.serialize_model_io and self.model_io_lock.locked():
                            self.current_task.update(
                                {
                                    "phase": "waiting_for_model_io",
                                    "message": "Waiting for the active app-managed model download to finish safely.",
                                    "updated_at": time.time(),
                                }
                            )
                            self.queue_message(
                                {
                                    "type": "task_progress",
                                    "task_id": task_id,
                                    **self._current_run_identity_payload(),
                                    "progress": 0,
                                    "phase": self.current_task["phase"],
                                    "message": self.current_task["message"],
                                }
                            )
                        result = await self._run_executor_callback(
                            callback,
                            serialize_model_io=serialize_model_io,
                        )

                        if self.current_task and self.current_task.get("interrupt_requested"):
                            terminal_status = "cancelled"
                        if future and terminal_status == "completed":
                            future.set_result(result)
                        elif future and terminal_status == "cancelled":
                            future.set_exception(asyncio.CancelledError("Execution interrupted by the user."))
                    except Exception as e:
                        interrupted_by_user = bool(self.current_task and self.current_task.get("interrupt_requested"))
                        terminal_status = "cancelled" if interrupted_by_user else "failed"
                        if interrupted_by_user:
                            if future:
                                future.set_exception(asyncio.CancelledError("Execution interrupted by the user."))
                            continue
                        traceback_text = getattr(e, "modiff_traceback", None) or traceback.format_exc()
                        logger.error(f"Error occurred in {traceback_text}")
                        task_list, _ = self._get_queue()
                        failure_payload = self._exception_payload(
                            e,
                            task_id=task_id,
                            sid=self.current_task["sid"] if self.current_task else None,
                            node_id=getattr(e, "modiff_node_id", None),
                            node_name=getattr(e, "modiff_node_name", None),
                            traceback_text=traceback_text,
                        )
                        self._record_auto_resource_failure(
                            e,
                            {
                                "category": failure_payload.get("category"),
                                "error_code": failure_payload.get("error_code"),
                                "message": failure_payload.get("message"),
                                "recovery_hint": failure_payload.get("recovery_hint"),
                            },
                        )
                        if future:
                            future.set_exception(e)
                    finally:
                        runtime_cleanup = None
                        if terminal_status in {"cancelled", "failed"}:
                            # A failed or cancelled graph must not leave model,
                            # node, component, or allocator ownership behind for
                            # the next queued run. Cancellation is cooperative
                            # inside third-party model loading, so teardown runs
                            # immediately after that call returns and before the
                            # worker advances the queue.
                            runtime_cleanup = await self.loop.run_in_executor(
                                None,
                                self._release_runtime_caches_for_retry,
                            )
                            self._last_auto_model_family = None
                            self._last_auto_resource_signature = None
                        if self.current_task:
                            task_sid = self.current_task.get("sid")
                            task_name = self.current_task.get("name")
                            attempt_index = self.current_task.get("attempt_index")
                            terminal_entry = self._record_terminal_task(terminal_status, error_payload=failure_payload)
                            preview_state = self._mark_studio_preview_run_terminal(task_id, terminal_status)
                            task_list, _ = self._get_queue()
                            terminal_message = {
                                "type": "task_completed"
                                if terminal_status == "completed"
                                else "task_cancelled"
                                if terminal_status == "cancelled"
                                else "task_failed",
                                "task_id": task_id,
                                "name": task_name,
                                "attempt_index": attempt_index,
                                **self._current_run_identity_payload(),
                                "completed_at": terminal_entry.get("completed_at") if terminal_entry else time.time(),
                                "queued": task_list,
                                "current": None,
                                "sid": task_sid,
                                "status": terminal_status,
                                "recent": self.recent_tasks,
                            }
                            if terminal_status == "completed":
                                terminal_message["args"] = args
                            elif terminal_status == "cancelled":
                                terminal_message["message"] = "Execution interrupted by the user."
                            elif isinstance(failure_payload, dict):
                                terminal_message.update(failure_payload)
                            if (
                                runtime_cleanup is not None
                                and not (
                                    isinstance(failure_payload, dict)
                                    and failure_payload.get("category")
                                    == "optional_runtime"
                                )
                            ):
                                terminal_message["runtimeCleanup"] = runtime_cleanup
                            if preview_state:
                                terminal_message["preview_slots"] = preview_state["previewSlots"]
                                terminal_message["preview_state_revision"] = preview_state["revision"]
                            self.queue_message(terminal_message)
                            self.current_task = None
                            self._persist_supervisor_queue_state(force=True)
                        self.main_queue.task_done()
                        self.interrupt_flag = False

                except asyncio.TimeoutError:
                    # Timeout is expected, just continue to check shutdown
                    continue

        except Exception as e:
            logger.error(f"Main worker error: {e}")
        finally:
            logger.debug("Main worker shutting down")

    async def _run_executor_callback(self, callback, *, serialize_model_io=False):
        if serialize_model_io and self.serialize_model_io:
            async with self.model_io_lock:
                return await self.loop.run_in_executor(None, callback)
        return await self.loop.run_in_executor(None, callback)

    async def _background_worker(self):
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Use wait_for with a timeout to check shutdown periodically
                    queue_item = await asyncio.wait_for(self.background_queue.get(), timeout=1.0)

                    # Check for sentinel value (None) indicating shutdown
                    if queue_item is None:
                        self.background_queue.task_done()
                        break

                    task, args = queue_item

                    try:
                        if isinstance(args, tuple):
                            await task(*args)
                        elif isinstance(args, dict):
                            await task(**args)
                        else:
                            await task(args)
                    except Exception as e:
                        logger.error(f"Error processing background task: {e}")
                        # logger.error(f"Error occurred in {traceback.format_exc()}")
                    finally:
                        self.background_queue.task_done()

                except asyncio.TimeoutError:
                    # Timeout is expected, just continue to check shutdown
                    continue

        except Exception as e:
            logger.error(f"Background worker error: {e}")
        finally:
            logger.debug("Background worker shutting down")

    """
    ╭─────────────────────╮
       Basic HTTP Routes
    ╰─────────────────────╯
    """

    async def index(self, _):
        response = web.FileResponse("web/index.html")
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    async def favicon(self, _):
        return web.FileResponse("web/favicon.ico")

    async def user_assets(self, request):
        module = request.match_info.get("module")
        file = request.match_info.get("file")
        fileName = f"custom/{module}/web/{file}"

        if not Path(fileName).exists():
            return web.HTTPNotFound(text="File not found")

        response = web.FileResponse(fileName)
        # response.headers["Content-Type"] = "application/javascript"
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

        return response

    """
    ╭─────────────────────────╮
       Nodes & Fields Routes
    ╰─────────────────────────╯
    """

    def _available_runtime_devices(self):
        # Keep both spellings because older node contracts expose ``cpu:0``
        # while newer Diffusers loaders use the canonical ``cpu`` spelling.
        # PyTorch accepts both and neither should be presented as unavailable.
        devices = ["cpu", "cpu:0"]
        try:
            import torch as torch_runtime

            if torch_runtime.cuda.is_available():
                devices.extend(f"cuda:{index}" for index in range(max(1, torch_runtime.cuda.device_count())))
        except Exception:
            torch_runtime = None
        try:
            xpu = getattr(torch_runtime, "xpu", None)
            if xpu is not None and callable(getattr(xpu, "is_available", None)) and xpu.is_available():
                count = xpu.device_count() if callable(getattr(xpu, "device_count", None)) else 1
                devices.extend(f"xpu:{index}" for index in range(max(1, int(count))))
        except Exception:
            pass
        try:
            mps = getattr(getattr(torch_runtime, "backends", None), "mps", None)
            if mps is not None and callable(getattr(mps, "is_available", None)) and mps.is_available():
                devices.append("mps")
        except Exception:
            pass
        return list(dict.fromkeys(devices))

    @staticmethod
    def _option_label(value, fallback):
        if isinstance(value, dict):
            label = value.get("label") or value.get("name") or value.get("title") or fallback
            if isinstance(label, (list, tuple)):
                label = next((item for item in label if str(item).strip()), fallback)
            return str(label)
        text = str(value)
        return text if text else str(fallback)

    def _runtime_choice_capabilities(self):
        cached = getattr(self, "_runtime_choice_capabilities_cache", None)
        if isinstance(cached, dict):
            return cached
        try:
            import torch as torch_runtime
            from modules.DiffusersRuntime.main import build_runtime_capabilities

            devices = self._available_runtime_devices()
            if any(value.startswith("cuda:") for value in devices):
                device = {"type": "cuda", "device": next(value for value in devices if value.startswith("cuda:"))}
            elif any(value.startswith("xpu:") for value in devices):
                device = {"type": "xpu", "device": next(value for value in devices if value.startswith("xpu:"))}
            elif "mps" in devices:
                device = {"type": "mps", "device": "mps"}
            else:
                device = {"type": "cpu", "device": "cpu"}
            cached = build_runtime_capabilities(
                {"devices": [device]},
                torch_module=torch_runtime,
            )
        except Exception as exc:
            logger.debug("Could not build runtime option capabilities: %s", exc)
            cached = {}
        self._runtime_choice_capabilities_cache = cached
        return cached

    def _runtime_choice_compatibility(self, field_name, value):
        capabilities = self._runtime_choice_capabilities()
        normalized_field = str(field_name or "").strip().lower()
        normalized_value = str(value or "").strip()
        if normalized_field == "attention_backend":
            if normalized_value == "auto":
                return True, None
            option = (capabilities.get("attention_backends") or {}).get(normalized_value)
            if isinstance(option, dict) and not option.get("available", False):
                return False, str(option.get("reason") or "This attention backend is unavailable.")
        quantization_fields = {"backend", "quant_type", "quantization_mode"}
        if normalized_field in quantization_fields:
            option = (capabilities.get("quantization_backends") or {}).get(normalized_value)
            if isinstance(option, dict) and not option.get("available", False):
                return False, str(option.get("reason") or "This quantization backend is unavailable.")
        if normalized_field in {"dtype", "compute_dtype", "bnb_4bit_compute_dtype"} and normalized_value:
            option = (capabilities.get("dtypes") or {}).get(normalized_value)
            if option is False:
                return False, f"{normalized_value} is not supported by the active runtime."
        return True, None

    def _option_descriptors(self, field_name, field_definition):
        options = field_definition.get("options")
        if not isinstance(options, (list, tuple, dict)):
            return None
        # UI groups use ``options`` as an ordered list of child field keys,
        # not as user-selectable choices. Keep that structural contract intact.
        if str(field_definition.get("display") or "").strip().lower() == "ui_group":
            return None
        dependencies = field_definition.get("optionDependencies")
        if dependencies is None:
            source = field_definition.get("optionsSource")
            dependencies = source if isinstance(source, dict) else None

        entries = []
        normalized_field_name = str(field_name or "").strip().lower()
        is_device_field = normalized_field_name in {
            "device",
            "execution_device",
            "generator_device",
            "offload_device",
        } or normalized_field_name.endswith("_device")
        available_devices = set(self._available_runtime_devices()) if is_device_field else None
        if available_devices is not None:
            declared = options.keys() if isinstance(options, dict) else options
            declared_values = {str(value) for value in declared if not str(value).startswith("__")}
            runtime_values = sorted(
                available_devices,
                key=lambda value: (
                    0 if value.startswith(("cuda:", "xpu:")) or value == "mps" else 1,
                    value,
                ),
            )
            if isinstance(options, dict):
                options = {
                    **options,
                    **{value: value for value in runtime_values if value not in declared_values},
                }
            else:
                options = [
                    *options,
                    *(value for value in runtime_values if value not in declared_values),
                ]
        iterable = options.items() if isinstance(options, dict) else ((str(value), value) for value in options)
        for value, option in iterable:
            if str(value).startswith("__"):
                entries.append((str(value), option))
                continue
            option_value = str(value) if isinstance(options, dict) else str(option)
            compatible = available_devices is None or option_value in available_devices
            disabled_reason = None
            if compatible:
                compatible, disabled_reason = self._runtime_choice_compatibility(field_name, option_value)
            descriptor = {
                "schemaVersion": 1,
                "value": option_value,
                "label": self._option_label(option, option_value),
                "compatibility": "compatible" if compatible else "incompatible",
                "availability": "installed" if compatible else "unavailable",
                "installationState": "installed" if compatible else "unavailable",
            }
            if dependencies:
                descriptor["dependencies"] = deepcopy(dependencies)
            if not compatible:
                descriptor["disabledReason"] = disabled_reason or (
                    "This device is not available in the current runtime."
                    if available_devices is not None
                    else "This option is not available in the current runtime."
                )
            entries.append((str(value), descriptor))
        if isinstance(options, dict):
            return {key: value for key, value in entries}
        return [value for _, value in entries]

    def _runtime_option_catalog(self):
        catalog = {}
        for module, actions in self.modules.items():
            for action, values in actions.items():
                if values.get("hidden", False):
                    continue
                node_options = {}
                for field_name, field_definition in (values.get("params") or {}).items():
                    descriptors = self._option_descriptors(field_name, field_definition)
                    if descriptors is not None:
                        node_options[field_name] = descriptors
                if node_options:
                    catalog[f"{module}.{action}"] = node_options
        return catalog

    def describe_node_params(self, params):
        """Return browser-safe fields using the same versioned option contract."""
        described = deepcopy(params or {})
        for field_name, field_definition in described.items():
            if not isinstance(field_definition, dict):
                continue
            field_definition.pop("postProcess", None)
            option_descriptors = self._option_descriptors(field_name, field_definition)
            if option_descriptors is not None:
                field_definition["options"] = option_descriptors
        return described

    async def runtime_options(self, _request):
        return web.json_response(
            {
                "schemaVersion": 1,
                "generatedAt": time.time(),
                "nodes": self._runtime_option_catalog(),
            }
        )

    async def nodes(self, request):
        id = request.match_info.get("id", "").strip("/")
        modules = self.modules

        if id:
            m, n = id.split("/")
            if m not in modules:
                return web.json_response(
                    {"error": f"The module {m} was not found. Try refreshing the page and restarting the server."},
                    status=404,
                )
            if n not in modules[m]:
                return web.json_response(
                    {
                        "error": f"The node {n} was not found in the module {m}. Try refreshing the page and restarting the server."
                    },
                    status=404,
                )
            modules = {m: {n: modules[m][n]}}

        output = {}
        for module, actions in modules.items():
            for action, values in actions.items():
                if values.get("hidden", False) and not id:
                    continue
                params = self.describe_node_params(values.get("params", {}))

                output[f"{module}.{action}"] = {
                    "module": module,
                    "action": action,
                    "type": values.get("type", "custom"),
                    "label": values.get("label", f"{module}: {action}"),
                    "category": values.get("category", "default"),
                    "description": values.get("description", ""),
                    "resizable": values.get("resizable", False),
                    "skipParamsCheck": values.get("skipParamsCheck", False),
                    "style": values.get("style", ""),
                    "params": params,
                    "time": [0, 0, 0],
                    "memory": [0, 0, 0],
                    "cache": False,
                }

        return web.json_response({"instance": self.instance, "nodes": output})

    def _field_action_runtime_hints(self, data):
        if not isinstance(data, dict):
            return None
        return self._coerce_runtime_hints(
            {
                "workflowTabId": data.get("workflowTabId"),
                "workflowCanvasEpoch": data.get("workflowCanvasEpoch"),
                "workflowFormEpoch": data.get("workflowFormEpoch"),
            }
        )

    def _execute_field_action(
        self,
        fn,
        identity,
        include_current_task,
        module,
        action,
        values,
        ref,
    ):
        assert_optional_runtime_ready(
            field_action_optional_runtime_requirement(
                module,
                action,
                fn.__name__,
                values,
            )
        )
        from modiff.NodeBase import node_message_context

        message_identity = dict(identity) if isinstance(identity, dict) else {}
        if include_current_task and self.current_task:
            task_id = self.current_task.get("task_id")
            attempt_index = self.current_task.get("attempt_index")
            if task_id:
                message_identity["task_id"] = task_id
            if attempt_index is not None:
                message_identity["attempt_index"] = attempt_index
        with node_message_context(message_identity):
            return fn(values, ref)

    @staticmethod
    def _declared_field_exec_actions(field_definition):
        """Collect backend method names from one authoritative field contract."""

        if not isinstance(field_definition, dict):
            return set()
        allowed = set()
        for event_name in ("onChange", "onSignal"):
            pending = [field_definition.get(event_name)]
            while pending:
                descriptor = pending.pop()
                if isinstance(descriptor, str):
                    if descriptor:
                        allowed.add(descriptor)
                elif isinstance(descriptor, (list, tuple)):
                    pending.extend(descriptor)
                elif isinstance(descriptor, dict) and descriptor.get("action") == "exec":
                    method_name = descriptor.get("data")
                    if isinstance(method_name, str) and method_name:
                        allowed.add(method_name)
        return allowed

    def _authorize_field_action(self, *, module, action, field_key, method_name):
        """Validate a field RPC against the live backend node definition."""

        if not all(isinstance(value, str) and value for value in (module, action, field_key, method_name)):
            raise ValueError("Field actions require non-empty module, action, fieldKey, and fn strings.")
        module_definition = self.modules.get(module)
        if not isinstance(module_definition, dict):
            raise ValueError(f"Unknown field-action module {module!r}.")
        action_definition = module_definition.get(action)
        if not isinstance(action_definition, dict):
            raise ValueError(f"Unknown field-action node {module}.{action}.")
        params = action_definition.get("params")
        field_definition = params.get(field_key) if isinstance(params, dict) else None
        if not isinstance(field_definition, dict):
            raise ValueError(f"Unknown field {field_key!r} for field-action node {module}.{action}.")
        allowed = self._declared_field_exec_actions(field_definition)
        if method_name not in allowed:
            raise ValueError(
                f"Field {module}.{action}.{field_key} does not authorize backend action {method_name!r}."
            )

    async def field_action(self, request):
        data = await request.json()
        if not isinstance(data, dict):
            return web.json_response(
                {"error": True, "message": "Field action payload must be a JSON object."},
                status=400,
            )
        if self._runtime_mutation_gate is not None:
            return web.json_response(
                {
                    "error": True,
                    "error_code": "runtime_mutation_busy",
                    "message": "Field actions are unavailable during runtime mutation or recovery.",
                },
                status=409,
            )
        node = data.get("node")
        sid = data.get("sid")
        method_name = data.get("fn")
        values = data.get("values")
        key = data.get("fieldKey", None)
        queue = data.get("queue", False)
        module = data.get("module")
        action = data.get("action")
        if not isinstance(node, str) or not node:
            return web.json_response(
                {"error": True, "message": "Field actions require a non-empty node id."},
                status=400,
            )
        if not isinstance(values, dict):
            return web.json_response(
                {"error": True, "message": "Field action values must be a JSON object."},
                status=400,
            )
        if type(queue) is not bool:
            return web.json_response(
                {"error": True, "message": "Field action queue must be a JSON boolean."},
                status=400,
            )
        try:
            self._authorize_field_action(
                module=module,
                action=action,
                field_key=key,
                method_name=method_name,
            )
        except ValueError as error:
            return web.json_response(
                {"error": True, "message": str(error)},
                status=400,
            )
        optional_runtime_requirement = field_action_optional_runtime_requirement(
            module,
            action,
            method_name,
            values,
        )
        if optional_runtime_requirement_blocks_execution(
            optional_runtime_requirement
        ):
            return web.json_response(
                optional_runtime_blocker_payload(optional_runtime_requirement),
                status=409,
            )
        runtime_hints = self._field_action_runtime_hints(data)
        message_identity = self._run_identity_payload(runtime_hints)
        if sid:
            message_identity["sid"] = sid

        if node not in self.node_cache:
            work_module = import_module(f"{module}.main")
            work_action = getattr(work_module, action)
            work_action = work_action(node_id=node)
            self.node_cache[node] = work_action

        cached_node = self.node_cache[node]
        if (
            getattr(cached_node, "module_name", None) != module
            or getattr(cached_node, "class_name", None) != action
        ):
            return web.json_response(
                {
                    "error": True,
                    "message": "The cached node does not match the requested field-action module and action.",
                },
                status=409,
            )

        cached_node._sid = sid  # always update the sid as it may change over time

        callback = getattr(cached_node, method_name, None)
        if not callable(callback):
            return web.json_response(
                {"error": True, "message": "The authorized backend field action is not callable."},
                status=409,
            )
        ref = {
            "node": node,
            "key": key,
            "queue": queue,
        }

        if queue:
            task = partial(
                self._execute_field_action,
                callback,
                message_identity,
                True,
                module,
                action,
            )
            task_id = await self.queue_task(
                task,
                (values, ref),
                None,
                sid,
                name="Field action",
                runtime_hints=runtime_hints,
                optional_runtime_requirement=optional_runtime_requirement,
            )
        else:
            # Run field action in executor to avoid blocking the event loop
            if not getattr(self, "loop", None):
                self.loop = asyncio.get_event_loop()
            try:
                await self.loop.run_in_executor(
                    None,
                    partial(
                        self._execute_field_action,
                        callback,
                        message_identity,
                        False,
                        module,
                        action,
                        values,
                        ref,
                    ),
                )
            except Exception as e:
                logger.error(f"Error executing field action synchronously: {e}")
                blocked_requirement = getattr(
                    e,
                    "modiff_optional_runtime_requirement",
                    None,
                )
                if isinstance(blocked_requirement, dict):
                    return web.json_response(
                        optional_runtime_blocker_payload(blocked_requirement),
                        status=409,
                    )
                classification = self._classify_exception(e)
                status = 409 if getattr(e, "modiff_error_code", None) else 500
                recovery_hint = classification.get("recovery_hint")
                response_message = f"Field action error: {e}"
                if recovery_hint:
                    response_message = f"{response_message} {recovery_hint}"
                return web.json_response(
                    {
                        "error": True,
                        "message": response_message,
                        "category": classification.get("category"),
                        "error_code": classification.get("error_code"),
                        "recovery_hint": recovery_hint,
                        "sid": sid,
                        "ref": ref,
                    },
                    status=status,
                )
            task_id = None

        return web.json_response(
            {
                "error": False,
                "message": f"Field action `{method_name}` for node `{node}` queued for processing",
                "sid": sid,
                "task_id": task_id,
                "ref": ref,
            }
        )

    """
    ╭─────────────────────╮
       Node Cache Routes
    ╰─────────────────────╯
    """

    async def cache(self, request):
        node = request.match_info.get("node")
        field = request.match_info.get("field")
        index = request.match_info.get("index", None)

        if node not in self.node_cache:
            return web.HTTPNotFound(text=f"Node {node} not found in cache.")

        cached_node = self.node_cache[node]
        cached_output = getattr(cached_node, "output", None)
        cached_params = getattr(cached_node, "params", None)
        if isinstance(cached_output, dict) and dict.__contains__(cached_output, field):
            cached_values = cached_output
        elif isinstance(cached_params, dict) and dict.__contains__(cached_params, field):
            cached_values = cached_params
        else:
            return web.HTTPNotFound(text=f"Field {field} not found in node {node} cache.")

        # Dynamic outputs may carry process-local capabilities or other opaque
        # runtime state. Only fields in the authoritative static node contract
        # are cache-servable, and validate that contract before touching the
        # cached value or invoking any media conversion path.
        module = getattr(cached_node, "module_name", None)
        action = getattr(cached_node, "class_name", None)
        modules = self.modules
        module_definition = (
            dict.get(modules, module)
            if isinstance(modules, dict) and isinstance(module, str)
            else None
        )
        action_definition = (
            dict.get(module_definition, action)
            if isinstance(module_definition, dict) and isinstance(action, str)
            else None
        )
        params = dict.get(action_definition, "params") if isinstance(action_definition, dict) else None
        field_definition = dict.get(params, field) if isinstance(params, dict) else None
        if not isinstance(field_definition, dict) or not dict.__contains__(field_definition, "type"):
            return web.HTTPBadRequest(
                text=f"Field {field} is not declared as a cache-served field for node {node}."
            )

        type = dict.get(field_definition, "type")
        if not is_cache_servable_data_type(type):
            return web.HTTPBadRequest(
                text=f"Field {field} has a type that cannot be served from node cache."
            )

        data = dict.__getitem__(cached_values, field)
        if data is None:
            return web.HTTPNotFound(text=f"Field {field} is empty in node {node} cache.")

        if isinstance(data, list):
            index = max(0, min(len(data) - 1, int(index))) if index else 0
            data = data[index]

        fieldOptions = dict.get(field_definition, "fieldOptions", {})
        if not isinstance(fieldOptions, dict):
            fieldOptions = {}

        filename = request.query.get("filename", f"{field}")
        download_format = str(request.query.get("download_format") or "").strip().lower()
        export_options = None
        if download_format:
            try:
                export_options = self._media_export_options(request)
            except ValueError as exc:
                return web.json_response({"error": str(exc)}, status=400)

        charset = None
        if is_image_data_type(type):
            format = request.query.get("format", "WEBP").upper()
            quality = request.query.get("quality", 100)
            out = to_bytes(type, data, {"format": format, "quality": quality})
            if download_format:
                from modiff.media_io import export_media_bytes

                try:
                    export_path, content_type, export_filename = await asyncio.to_thread(
                        export_media_bytes,
                        out,
                        source_suffix=f".{format.lower()}",
                        kind="image",
                        format_id=download_format,
                        options=export_options,
                        cache_root=Path(self.data_dir) / ".media-exports",
                    )
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    return web.json_response({"error": str(exc)}, status=422)
                return web.FileResponse(
                    export_path,
                    headers={
                        "Content-Disposition": f'attachment; filename="{export_filename}"',
                        "Content-Type": content_type,
                        "Cache-Control": "private, max-age=31536000, immutable",
                    },
                )
            content_type = f"image/{format.lower()}"
            if not str(filename).lower().endswith(f".{format.lower()}"):
                filename = f"{filename}.{format.lower()}"
        elif type == "audio" or (isinstance(type, list) and "audio" in type):
            out = to_bytes("audio", data, fieldOptions)
            if download_format:
                from modiff.media_io import export_media_bytes

                try:
                    export_path, content_type, export_filename = await asyncio.to_thread(
                        export_media_bytes,
                        out,
                        source_suffix=".wav",
                        kind="audio",
                        format_id=download_format,
                        options=export_options,
                        cache_root=Path(self.data_dir) / ".media-exports",
                    )
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    return web.json_response({"error": str(exc)}, status=422)
                return web.FileResponse(
                    export_path,
                    headers={
                        "Content-Disposition": f'attachment; filename="{export_filename}"',
                        "Content-Type": content_type,
                        "Cache-Control": "private, max-age=31536000, immutable",
                    },
                )
            content_type = "audio/wav"
            if not str(filename).lower().endswith(".wav"):
                filename = f"{filename}.wav"
            try:
                download_sample_rate = parse_audio_download_sample_rate(request.query.get("download_sample_rate"))
            except ValueError as exc:
                return web.json_response({"error": str(exc)}, status=400)
            if download_sample_rate is not None:
                try:
                    out = resample_wav_bytes(out, download_sample_rate)
                except (OSError, ValueError) as exc:
                    return web.json_response(
                        {"error": f"Could not prepare the requested WAV download: {exc}"},
                        status=422,
                    )
                filename = audio_download_filename(filename, download_sample_rate)
        elif type == "video" or (isinstance(type, list) and "video" in type):
            data_path = self._resolve_managed_path_identifier(data) if isinstance(data, (str, os.PathLike)) else None
            if download_format and data_path is not None and data_path.is_file():
                from modiff.media_io import export_media_file

                try:
                    export_path, content_type, export_filename = await asyncio.to_thread(
                        export_media_file,
                        data_path,
                        kind="video",
                        format_id=download_format,
                        options=export_options,
                        cache_root=Path(self.data_dir) / ".media-exports",
                    )
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    return web.json_response({"error": str(exc)}, status=422)
                return web.FileResponse(
                    export_path,
                    headers={
                        "Content-Disposition": f'attachment; filename="{export_filename}"',
                        "Content-Type": content_type,
                        "Cache-Control": "private, max-age=31536000, immutable",
                    },
                )
            if data_path is not None and data_path.is_file():
                resp = web.FileResponse(data_path)
                resp.headers["Content-Disposition"] = f'inline; filename="{filename}"'
                resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                return resp
            return web.json_response(
                {"error": "Run an Export Video node before downloading this in-memory video."},
                status=422,
            )
        elif (
            type == "text"
            or (isinstance(type, str) and type.startswith("str"))
            or (isinstance(type, list) and any(isinstance(t, str) and t.startswith("str") for t in type))
        ):
            out = str(data).encode("utf-8")
            content_type = "text/plain"
            charset = "utf-8"
            filename = f"{filename}.txt"
        else:
            resp = web.FileResponse(data)
            resp.headers["Content-Disposition"] = f'inline; filename="{filename}"'
            resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            return resp

        return byte_range_response(
            request,
            out,
            content_type=content_type,
            charset=charset,
            filename=filename,
        )

    async def delete_cache(self, request):
        data = await request.json()
        nodes = data.get("nodes", [])

        if isinstance(nodes, str):
            nodes = list(self.node_cache.keys()) if nodes == "*" else [nodes]

        # this might take a while because it could be freeing up VRAM
        for node in nodes:
            if node in self.node_cache:
                del self.node_cache[node]

        logger.debug(f"Removed {len(nodes)} nodes from cache.")

        return web.json_response({"error": False, "nodes": nodes})

    """
    ╭───────────────────╮
       File Management
    ╰───────────────────╯
    """

    @staticmethod
    def _resolve_path_under_root(value, root):
        """Resolve ``value`` and reject traversal or symlink escapes from ``root``."""

        root_path = Path(root).expanduser().resolve(strict=False)
        candidate = Path(str(value))
        if not candidate.is_absolute():
            candidate = root_path / candidate
        candidate = candidate.expanduser().resolve(strict=False)
        try:
            candidate.relative_to(root_path)
        except ValueError:
            return None
        return candidate

    def _resolve_managed_path_identifier(self, value):
        """Resolve a public file identifier within the configured local roots."""

        return resolve_managed_path_identifier(
            value,
            work_root=self.work_dir,
            data_root=self.data_dir,
        )

    def _public_path_identifier(self, path):
        """Return a portable identifier without exposing an absolute host path."""

        candidate = Path(path).expanduser().resolve(strict=False)
        work_root = Path(self.work_dir).expanduser().resolve(strict=False)
        try:
            return candidate.relative_to(work_root).as_posix()
        except ValueError:
            return data_path_identifier(candidate, self.data_dir)

    @staticmethod
    def _loopback_host(host):
        normalized = str(host or "").strip().strip("[]").lower()
        if normalized == "localhost":
            return True
        try:
            return ipaddress.ip_address(normalized).is_loopback
        except ValueError:
            return False

    @staticmethod
    def _request_host(request):
        headers = getattr(request, "headers", {}) or {}
        try:
            return urlparse(f"//{getattr(request, 'host', None) or headers.get('Host', '')}").hostname
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _request_peer_host(request):
        try:
            peer_host = getattr(request, "remote", None)
        except (AttributeError, RuntimeError):
            peer_host = None
        if peer_host:
            return peer_host
        transport = getattr(request, "transport", None)
        peer = transport.get_extra_info("peername") if transport is not None else None
        return peer[0] if isinstance(peer, tuple) and peer else peer

    def _loopback_request_boundary(self, request):
        return self._loopback_host(self._request_host(request)) and self._loopback_host(
            self._request_peer_host(request)
        )

    def _trusted_browser_origin(self, request):
        """Authorize one local browser or native client mutation.

        Comparing two attacker-controlled DNS hostnames is not a trust check:
        a DNS-rebinding page can make its Origin and Host names equal. Require
        literal loopback request/peer addresses, plus a literal loopback HTTP
        Origin whenever a browser supplied one.
        """

        headers = getattr(request, "headers", {}) or {}
        if not self._loopback_request_boundary(request):
            return False
        origin = headers.get("Origin")
        if not origin:
            return True
        try:
            parsed_origin = urlparse(origin)
        except (TypeError, ValueError):
            return False
        return parsed_origin.scheme in {"http", "https"} and self._loopback_host(parsed_origin.hostname)

    def _trusted_websocket_origin(self, request):
        """Restrict the unauthenticated WebSocket to the local trust boundary.

        Browsers always send an ``Origin`` header for a WebSocket handshake, so
        a present origin must itself be loopback. Native clients, including
        :class:`modiff.client.WebSocketClient`, do not necessarily send one;
        those clients remain supported only when both the HTTP destination and
        the connected peer are loopback.
        """

        headers = getattr(request, "headers", {}) or {}
        if not self._loopback_request_boundary(request):
            return False

        origin = headers.get("Origin")
        if not origin:
            return True
        try:
            parsed_origin = urlparse(origin)
        except (TypeError, ValueError):
            return False
        return parsed_origin.scheme in {"http", "https"} and self._loopback_host(parsed_origin.hostname)

    @web.middleware
    async def _mutation_origin_middleware(self, request, handler):
        # Every endpoint, including read-only workflow/queue/media metadata,
        # belongs to the same unauthenticated loopback trust boundary. Checking
        # only mutations would still let a DNS-rebinding hostname read local
        # state through an attacker-controlled Host header.
        origin_error = self._untrusted_origin_response(request)
        if origin_error is not None:
            return origin_error
        method = str(getattr(request, "method", "GET")).upper()
        path = str(getattr(request, "path", ""))
        runtime_transaction = bool(
            method == "POST"
            and (
                path
                in {
                    "/runtime/optimizations/install",
                    "/runtime/optimizations/activate",
                    "/runtime/optimizations/rollback",
                    "/runtime/optional-runtimes/install",
                    "/runtime/optional-runtimes/activate",
                    "/runtime/optional-runtimes/rollback",
                }
                or re.fullmatch(
                    r"/runtime/(?:optimizations|optional-runtimes)/jobs/optjob-[A-Za-z0-9_-]{12}/cancel",
                    path,
                )
            )
        )
        query = getattr(request, "query", {}) or {}
        converting_get = bool(
            method == "GET"
            and (
                path in {"/media/export", "/media/preview", "/preview"}
                or (path == "/file" and query.get("download_format"))
                or (path.startswith("/cache/") and query.get("download_format"))
            )
        )
        tracked_mutation = (
            method in {"POST", "PUT", "PATCH", "DELETE"} and not runtime_transaction
        ) or converting_get
        if tracked_mutation:
            if self._runtime_mutation_gate is not None:
                return web.json_response(
                    {
                        "error": True,
                        "error_code": "runtime_mutation_busy",
                        "message": "This mutation is unavailable during runtime mutation or recovery.",
                    },
                    status=409,
                )
            self._active_nonruntime_mutations += 1
        try:
            return await handler(request)
        finally:
            if tracked_mutation:
                self._active_nonruntime_mutations = max(
                    0, self._active_nonruntime_mutations - 1
                )

    def _untrusted_origin_response(self, request):
        if self._trusted_browser_origin(request):
            return None
        return web.json_response(
            {
                "error": "Requests require a loopback client, loopback Host, and loopback browser Origin.",
                "code": "untrusted_request_boundary",
            },
            status=403,
        )

    def _untrusted_websocket_origin_response(self, request):
        if self._trusted_websocket_origin(request):
            return None
        return web.json_response(
            {"error": "WebSocket connections require a trusted loopback client and browser origin."},
            status=403,
        )

    async def listdir(self, request):
        from modiff.media_io import media_capabilities

        req_basepath = self.data_dir if request.query.get("basepath") == "data" else self.work_dir
        req_path = request.query.get("path", req_basepath)
        data_namespace = request.query.get("basepath") == "data" or is_data_path_identifier(req_path)
        req_type = request.query.get("type", None)
        if req_type:
            req_type = [t.strip() for t in req_type.lower().split(",")]

        base_root = Path(req_basepath).expanduser().resolve(strict=False)
        if is_data_path_identifier(req_path):
            base_root = Path(self.data_dir).expanduser().resolve(strict=False)
            try:
                full_path = resolve_data_path_identifier(req_path, base_root)
            except ValueError:
                full_path = None
        else:
            full_path = self._resolve_path_under_root(req_path, base_root)

        runtime_media = media_capabilities()["media"]
        file_types = {
            "image": [extension.lstrip(".") for extension in runtime_media["image"]["importExtensions"]],
            "audio": [extension.lstrip(".") for extension in runtime_media["audio"]["importExtensions"]],
            "video": [extension.lstrip(".") for extension in runtime_media["video"]["importExtensions"]],
            "text": [
                "txt",
                "md",
                "csv",
                "json",
                "xml",
                "yaml",
                "yml",
                "ini",
                "toml",
                "cfg",
                "conf",
                "log",
                "html",
                "css",
                "js",
                "ts",
                "py",
                "rb",
                "php",
                "sql",
                "sh",
                "bash",
            ],
            "archive": ["zip", "rar", "tar", "gz", "bz2", "7z"],
            "3d": ["glb", "gltf", "stl", "obj", "fbx", "dae", "ply", "3ds", "max", "blend"],
        }

        if full_path is None:
            return web.json_response({"error": f"Cannot access paths outside of {base_root}."}, status=403)

        contents = {
            "files": [],
            "path": "",
            "abs_path": "",
        }

        try:
            if full_path.exists():
                contents["path"] = (
                    data_path_identifier(full_path, self.data_dir)
                    if data_namespace
                    else self._public_path_identifier(full_path)
                )
                # Kept for the existing response schema, but deliberately no
                # longer contains an absolute host path.
                contents["abs_path"] = contents["path"]

                for item in full_path.iterdir():
                    suffix = item.suffix.lstrip(".").lower()
                    # if any of the requested types don't match the file type, skip it
                    if (
                        not item.is_dir()
                        and req_type
                        and not any(suffix in exts for ftype, exts in file_types.items() if ftype in req_type)
                    ):
                        continue

                    file = {
                        "is_dir": item.is_dir(),
                        "is_hidden": is_hidden_path(item),
                        "name": item.name,
                        "path": (
                            data_path_identifier(item, self.data_dir)
                            if data_namespace
                            else self._public_path_identifier(item)
                        ),
                        #'abs_path': str(item),
                        "modified": item.stat().st_mtime,
                        "size": None,
                        "ext": None,
                        "type": None,
                    }
                    if not item.is_dir():
                        file["size"] = item.stat().st_size
                        file["ext"] = suffix
                        file["type"] = next((ftype for ftype, exts in file_types.items() if suffix in exts), "other")

                    contents["files"].append(file)

                return web.json_response(contents)
            else:
                return web.json_response({"error": f"The path {req_path} does not exist."}, status=404)

        except Exception as e:
            logger.error(f"Error listing directory: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def listgraphs(self, request):
        path = Path(self.data_dir) / "graphs"
        if not path.exists():
            return web.json_response({"error": True, "message": "No graph directory found."}, status=404)

        graphs = list_files(str(path), recursive=True, extensions=["json"])
        workflow_metadata: dict[str, dict] = {}
        optional_runtime_catalog_snapshot = None

        def request_optional_runtime_catalog():
            nonlocal optional_runtime_catalog_snapshot
            if optional_runtime_catalog_snapshot is None:
                optional_runtime_catalog_snapshot = public_optional_runtime_catalog()
            return optional_runtime_catalog_snapshot

        manifest_path = Path(self.data_dir) / "workflow-library-manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_workflows = [
                    *manifest.get("workflows", []),
                    *manifest.get("experimentalWorkflows", []),
                ]
                workflow_metadata = {
                    str(item.get("graphPath", "")).replace("\\", "/"): item
                    for item in manifest_workflows
                    if isinstance(item, dict) and item.get("graphPath")
                }
            except (OSError, ValueError, TypeError) as exc:
                logger.warning(f"Could not read workflow library manifest: {exc}")

        # build a nested tree of directories and files
        tree: list[dict] = []

        for g in graphs:
            rel_dir = g.get("rel_directory") or "."
            # normalize and split directory parts
            parts = [] if rel_dir in (None, ".", "") else [p for p in rel_dir.strip("/").split("/") if p]

            parent_children = tree
            current_path_parts: list[str] = []
            # ensure directory nodes exist for each part
            for part in parts:
                current_path_parts.append(part)
                node = next((n for n in parent_children if n.get("isDir") and n.get("name") == part), None)
                if not node:
                    node = {
                        "isDir": True,
                        "name": part,
                        "path": f"{str(path)}/{'/'.join(current_path_parts)}",
                        "children": [],
                    }
                    parent_children.append(node)
                parent_children = node["children"]

            raw_name = g.get("name") or Path(g.get("path", "")).name
            try:
                relative_graph_path = Path(g.get("path", "")).resolve().relative_to(path.resolve()).as_posix()
            except (ValueError, TypeError):
                relative_graph_path = ""
            metadata = workflow_metadata.get(relative_graph_path, {})
            optional_runtime_profile_ids = optional_runtime_profile_ids_for_execution(
                metadata.get("modelType"),
                metadata.get("mode"),
            )
            optional_runtime_requirement = optional_runtime_requirement_for_execution(
                metadata.get("modelType"),
                metadata.get("mode"),
                catalog_resolver=request_optional_runtime_catalog,
            )
            file_item = {
                "isDir": False,
                "name": Path(raw_name).stem,
                "path": g.get("path"),
                "modelType": metadata.get("modelType"),
                "mode": metadata.get("mode"),
                "mediaKind": metadata.get("mediaKind"),
                "supportTier": metadata.get("supportTier"),
                "qualificationStatus": metadata.get("qualificationStatus"),
                "requiredArtifacts": metadata.get("requiredArtifacts", []),
                "optionalRuntimeProfileIds": list(optional_runtime_profile_ids),
                "optionalRuntimeProfiles": public_optional_runtime_profiles(
                    optional_runtime_profile_ids
                ),
                "optionalRuntimeRequirement": optional_runtime_requirement,
            }
            parent_children.append(file_item)

        # recursively sort directories (dirs first, then files) by name
        def sort_children(children: list[dict]):
            dirs = [c for c in children if c["isDir"]]
            files = [c for c in children if not c["isDir"]]
            dirs.sort(key=lambda x: x["name"].lower())
            files.sort(key=lambda x: x["name"].lower())
            for d in dirs:
                sort_children(d["children"])
            # mutate list in-place to preserve references
            children[:] = dirs + files

        sort_children(tree)

        return web.json_response(tree)

    async def workflows_list(self, _request):
        from modiff.workflow_store import list_workflows

        return web.json_response({"workflows": list_workflows(self.data_dir)})

    async def workflow_get(self, request):
        from modiff.workflow_store import get_workflow

        record = get_workflow(self.data_dir, request.match_info.get("workflow_id"))
        if record is None:
            return web.json_response({"error": True, "message": "Workflow not found."}, status=404)
        return web.json_response(record)

    async def workflow_put(self, request):
        from modiff.workflow_store import save_workflow

        try:
            record = save_workflow(
                self.data_dir,
                request.match_info.get("workflow_id"),
                await request.json(),
            )
        except (ValueError, json.JSONDecodeError) as exc:
            return web.json_response({"error": True, "message": str(exc)}, status=400)
        self.queue_message({"type": "workflow_updated", "workflow": record})
        return web.json_response(record)

    async def workflow_delete(self, request):
        from modiff.workflow_store import delete_workflow

        try:
            workflow_id = request.match_info.get("workflow_id")
            deleted = delete_workflow(self.data_dir, workflow_id)
        except ValueError as exc:
            return web.json_response({"error": True, "message": str(exc)}, status=400)
        if not deleted:
            return web.json_response({"error": True, "message": "Workflow not found."}, status=404)
        self.queue_message({"type": "workflow_deleted", "workflow_id": workflow_id})
        return web.json_response({"error": False, "workflow_id": workflow_id})

    async def fileGet(self, request):
        file = request.query.get("file")

        if not file:
            return web.json_response({"error": "Incorrect request, `file` is required."}, status=400)

        file_path = self._resolve_managed_path_identifier(file)
        if file_path is None:
            return web.json_response(
                {"error": "Files outside the configured MoDiff roots cannot be opened."}, status=403
            )

        if not file_path.exists():
            return web.json_response({"error": f"The file {file} does not exist."}, status=404)

        download_format = str(request.query.get("download_format") or "").strip().lower()
        if download_format:
            from modiff.media_io import export_media_file, probe_media_file

            try:
                media_kind = str(request.query.get("media_kind") or "").rstrip("s").lower()
                if not media_kind:
                    media_kind = str(probe_media_file(file_path).get("kind") or "")
                export_path, content_type, filename = await asyncio.to_thread(
                    export_media_file,
                    file_path,
                    kind=media_kind,
                    format_id=download_format,
                    options=self._media_export_options(request),
                    cache_root=Path(self.data_dir) / ".media-exports",
                )
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                return web.json_response({"error": str(exc)}, status=422)
            return web.FileResponse(
                export_path,
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"',
                    "Content-Type": content_type,
                    "Cache-Control": "private, max-age=31536000, immutable",
                },
            )

        try:
            download_sample_rate = parse_audio_download_sample_rate(request.query.get("download_sample_rate"))
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)

        if download_sample_rate is not None:
            if file_path.suffix.lower() != ".wav":
                return web.json_response(
                    {"error": "Sample-rate conversion is supported only for WAV downloads."},
                    status=422,
                )
            try:
                body = resample_wav_bytes(file_path.read_bytes(), download_sample_rate)
            except (OSError, ValueError) as exc:
                return web.json_response(
                    {"error": f"Could not prepare the requested WAV download: {exc}"},
                    status=422,
                )
            filename = audio_download_filename(file_path.name, download_sample_rate)
            return web.Response(
                body=body,
                content_type="audio/wav",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"',
                    "Cache-Control": "no-store",
                    "X-MoDiff-Audio-Sample-Rate": str(download_sample_rate),
                },
            )

        return web.FileResponse(file_path)

    async def filePost(self, request):
        data = await request.post()
        file = data.get("file")
        if file is None or not getattr(file, "filename", None):
            return web.json_response({"error": "A file upload is required."}, status=400)
        type = data.get("type", "images")
        type = type if type in ["images", "audio", "videos", "text", "3d"] else "images"
        safe_name = Path(str(file.filename).replace("\\", "/")).name
        if not safe_name or safe_name in {".", ".."}:
            return web.json_response({"error": "The uploaded filename is invalid."}, status=400)
        file_path = (Path(self.data_dir) / type / safe_name).resolve()
        destination_root = (Path(self.data_dir) / type).resolve()
        try:
            file_path.relative_to(destination_root)
        except ValueError:
            return web.json_response({"error": "The uploaded filename is invalid."}, status=400)

        if file_path.exists():
            file_path = file_path.with_name(f"{file_path.stem}_{nanoid.generate(size=6)}{file_path.suffix}")

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            max_upload_bytes = int(self.client_max_size)
            with open(file_path, "wb") as f:
                total = 0
                while True:
                    chunk = file.file.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_upload_bytes:
                        limit_mib = max_upload_bytes // (1024 * 1024)
                        raise ValueError(f"The uploaded file exceeds the {limit_mib} MB local import limit.")
                    f.write(chunk)

            from modiff.media_io import probe_media_file

            expected_kind = {"images": "image", "videos": "video", "audio": "audio", "text": "text"}.get(type)
            metadata = (
                await asyncio.to_thread(probe_media_file, file_path, expected_kind)
                if expected_kind
                else {
                    "filename": file_path.name,
                    "extension": file_path.suffix.lower(),
                    "sizeBytes": file_path.stat().st_size,
                    "kind": "3d",
                }
            )
            return web.json_response(
                {
                    "error": False,
                    "path": data_path_identifier(file_path, self.data_dir),
                    "media": metadata,
                }
            )
        except Exception as e:
            file_path.unlink(missing_ok=True)
            logger.error(f"Error saving file: {e}")
            return web.json_response({"error": str(e)}, status=400 if isinstance(e, ValueError) else 500)

    @staticmethod
    def _media_export_options(request):
        options = {}
        fields = {
            "sample_rate": ("sampleRate", int),
            "channels": ("channels", int),
            "bit_depth": ("bitDepth", int),
            "bitrate": ("bitrate", int),
            "quality": ("quality", int),
            "compression": ("compression", int),
            "fps": ("fps", int),
            "width": ("width", int),
            "background": ("background", str),
            "speed": ("speed", str),
        }
        for query_key, (option_key, converter) in fields.items():
            value = request.query.get(query_key)
            if value in (None, ""):
                continue
            try:
                options[option_key] = converter(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid media export option: {query_key}.") from exc
        return options

    async def media_capabilities(self, _request):
        from modiff.media_io import media_capabilities

        return web.json_response(await asyncio.to_thread(media_capabilities))

    async def media_probe(self, request):
        from modiff.media_io import probe_media_file

        file = request.query.get("file")
        if not file:
            return web.json_response({"error": "`file` is required."}, status=400)
        file_path = self._resolve_managed_path_identifier(file)
        if file_path is None:
            return web.json_response(
                {"error": "Files outside the configured MoDiff roots cannot be inspected."}, status=403
            )
        try:
            metadata = await asyncio.to_thread(
                probe_media_file,
                file_path,
                request.query.get("media_kind"),
            )
        except FileNotFoundError as exc:
            return web.json_response({"error": str(exc)}, status=404)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return web.json_response({"error": str(exc)}, status=422)
        return web.json_response(metadata)

    async def media_export(self, request):
        from modiff.media_io import export_media_file

        file = request.query.get("file")
        format_id = request.query.get("format")
        media_kind = request.query.get("media_kind")
        if not file or not format_id or not media_kind:
            return web.json_response({"error": "`file`, `format`, and `media_kind` are required."}, status=400)
        file_path = self._resolve_managed_path_identifier(file)
        if file_path is None:
            return web.json_response(
                {"error": "Files outside the configured MoDiff roots cannot be exported."}, status=403
            )
        try:
            export_path, content_type, filename = await asyncio.to_thread(
                export_media_file,
                file_path,
                kind=media_kind,
                format_id=format_id,
                options=self._media_export_options(request),
                cache_root=Path(self.data_dir) / ".media-exports",
            )
        except FileNotFoundError as exc:
            return web.json_response({"error": str(exc)}, status=404)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return web.json_response({"error": str(exc)}, status=422)
        return web.FileResponse(
            export_path,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Type": content_type,
                "Cache-Control": "private, max-age=31536000, immutable",
            },
        )

    async def media_preview(self, request):
        """Serve a cached browser-safe representation of imported media."""

        from modiff.media_io import export_media_file, media_capabilities

        file = request.query.get("file")
        media_kind = str(request.query.get("media_kind") or "").rstrip("s").lower()
        if not file or media_kind not in {"audio", "video"}:
            return web.json_response({"error": "`file` and an audio/video `media_kind` are required."}, status=400)
        file_path = self._resolve_managed_path_identifier(file)
        if file_path is None:
            return web.json_response(
                {"error": "Files outside the configured MoDiff roots cannot be previewed."}, status=403
            )

        available = {descriptor["value"] for descriptor in media_capabilities()["media"][media_kind]["exportFormats"]}
        format_id = (
            ("mp3" if "mp3" in available else "wav")
            if media_kind == "audio"
            else ("mp4" if "mp4" in available else "webm")
        )
        if format_id not in available:
            return web.json_response({"error": f"No browser-safe {media_kind} preview is available."}, status=422)
        options = (
            {"sampleRate": 48000, "bitrate": 192} if media_kind == "audio" else {"quality": 23, "speed": "veryfast"}
        )
        try:
            preview_path, content_type, filename = await asyncio.to_thread(
                export_media_file,
                file_path,
                kind=media_kind,
                format_id=format_id,
                options=options,
                cache_root=Path(self.data_dir) / ".media-previews",
            )
        except FileNotFoundError as exc:
            return web.json_response({"error": str(exc)}, status=404)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return web.json_response({"error": str(exc)}, status=422)
        return web.FileResponse(
            preview_path,
            headers={
                "Content-Disposition": f'inline; filename="{filename}"',
                "Content-Type": content_type,
                "Cache-Control": "private, max-age=31536000, immutable",
            },
        )

    async def preview(self, request):
        file = request.query.get("file")
        if not file:
            return web.json_response({"error": "Incorrect request, `file` is required."}, status=400)

        file_path = self._resolve_managed_path_identifier(file)
        if file_path is None:
            return web.json_response({"error": "Cannot access paths outside the configured MoDiff roots."}, status=403)

        if not file_path.exists():
            return web.json_response({"error": f"The file {file} does not exist."}, status=404)

        try:
            width = int(request.query.get("width", 0))
            height = int(request.query.get("height", 0))
            format_id = str(request.query.get("format", "jpeg")).lower()
            quality = int(request.query.get("quality", 95))
            body, content_type = await asyncio.to_thread(
                render_image_preview,
                file_path,
                width,
                height,
                format_id,
                quality,
            )
        except (OSError, ValueError):
            return web.json_response({"error": f"The file {file} is not a supported image."}, status=400)

        return web.Response(
            body=body,
            content_type=content_type,
            headers={
                "Content-Disposition": f'inline; filename="{file_path.name}"',
                "Content-Length": str(len(body)),
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    async def stream(self, request):
        file = request.query.get("file")
        if not file:
            return web.json_response({"error": "Incorrect request, `file` is required."}, status=400)

        file_path = self._resolve_managed_path_identifier(file)
        if file_path is None:
            return web.json_response({"error": "Cannot access paths outside the configured MoDiff roots."}, status=403)

        if not file_path.exists():
            return web.json_response({"error": f"The file {file} does not exist."}, status=404)

        resp = web.FileResponse(file_path)
        resp.headers["Content-Disposition"] = f'inline; filename="{file_path.name}"'
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"

        return resp

    async def local_models(self, request):
        refresh = request.query.get("refresh", False)
        path_match = request.query.get("match", "")

        if refresh:
            modelstore.update_local()

        files = modelstore.get_local_ids(name=path_match)

        return web.json_response(files)

    def _studio_history_file(self):
        return Path(self.data_dir) / "studio" / "outputs.json"

    def _studio_outputs_dir(self):
        return Path(self.data_dir) / "studio" / "outputs"

    def _studio_blocks_dir(self):
        return Path(self.data_dir) / "studio" / "blocks"

    def _safe_block_id(self, block_id=None):
        raw_id = str(block_id or nanoid.generate(size=12))
        safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw_id).strip("_")[:80]
        return safe_id or nanoid.generate(size=12)

    def _studio_block_file(self, block_id):
        return self._studio_blocks_dir() / f"{self._safe_block_id(block_id)}.json"

    def _validate_studio_block(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("User block must be a JSON object.")

        block_id = self._safe_block_id(payload.get("id"))
        name = str(payload.get("name") or "User Block").strip()[:120] or "User Block"
        version = payload.get("version", 1)
        if version != 1:
            raise ValueError("Unsupported user block version.")

        required_arrays = ["nodes", "edges", "inputs", "outputs", "exposedParams"]
        for key in required_arrays:
            if not isinstance(payload.get(key), list):
                raise ValueError(f"User block field {key} must be a list.")

        for node in payload["nodes"]:
            if not isinstance(node, dict):
                raise ValueError("User block nodes must be JSON objects.")
            data = node.get("data")
            if not isinstance(data, dict):
                raise ValueError("User block node data must be a JSON object.")
            if (
                node.get("type") == "block"
                or data.get("type") == "block"
                or data.get("userBlockId")
                or data.get("userBlockSnapshot")
            ):
                raise ValueError("Nested user blocks are not supported. Flatten the selected block before saving.")

        block = dict(payload)
        block["id"] = block_id
        block["name"] = name
        block["version"] = 1
        block["nodes"] = payload["nodes"]
        block["edges"] = payload["edges"]
        block["inputs"] = payload["inputs"]
        block["outputs"] = payload["outputs"]
        block["exposedParams"] = payload["exposedParams"]
        now = int(time.time() * 1000)
        block["createdAt"] = int(payload.get("createdAt") or now)
        block["updatedAt"] = int(payload.get("updatedAt") or now)
        return block

    def _read_studio_block(self, block_id):
        block_file = self._studio_block_file(block_id)
        if not block_file.exists():
            return None
        with open(block_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return self._validate_studio_block(payload)

    def _write_studio_block(self, block):
        blocks_dir = self._studio_blocks_dir()
        blocks_dir.mkdir(parents=True, exist_ok=True)
        validated = self._validate_studio_block(block)
        validated["updatedAt"] = int(time.time() * 1000)
        target = self._studio_block_file(validated["id"])
        temp_file = target.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(validated, f, ensure_ascii=False)
        temp_file.replace(target)
        return validated, target

    def _list_studio_blocks(self):
        blocks_dir = self._studio_blocks_dir()
        if not blocks_dir.exists():
            return []

        blocks = []
        for block_file in blocks_dir.glob("*.json"):
            try:
                with open(block_file, "r", encoding="utf-8") as f:
                    blocks.append(self._validate_studio_block(json.load(f)))
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
                logger.error(f"Error reading Studio user block {block_file}: {e}")
        return sorted(blocks, key=lambda block: block.get("updatedAt") or 0, reverse=True)

    def _studio_preview_slot_key(self, workflow_tab_id, node_id, field_key):
        if not workflow_tab_id or not node_id or not field_key:
            return None
        return json.dumps(
            [str(workflow_tab_id), str(node_id), str(field_key)],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _studio_state_int(self, value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return default

    def _normalize_studio_preview_slot(self, slot):
        if not isinstance(slot, dict):
            return None
        key = self._studio_preview_slot_key(
            slot.get("workflowTabId"), slot.get("nodeId"), slot.get("fieldKey")
        )
        if key is None:
            return None
        status = str(slot.get("status") or "empty")
        if status not in {
            "empty",
            "pending",
            "ready",
            "failed",
            "cancelled",
            "completed_without_output",
        }:
            status = "empty"
        normalized = {
            "schemaVersion": 1,
            "workflowTabId": str(slot["workflowTabId"]),
            "nodeId": str(slot["nodeId"]),
            "fieldKey": str(slot["fieldKey"]),
            "currentOutputId": str(slot["currentOutputId"]) if slot.get("currentOutputId") else None,
            "pendingClientRunId": str(slot["pendingClientRunId"]) if slot.get("pendingClientRunId") else None,
            "pendingTaskId": str(slot["pendingTaskId"]) if slot.get("pendingTaskId") else None,
            "generation": max(self._studio_state_int(slot.get("generation"), 0), 0),
            "attemptIndex": (
                self._studio_state_int(slot["attemptIndex"])
                if slot.get("attemptIndex") is not None
                else None
            ),
            "status": status,
            "updatedAt": self._studio_state_int(slot.get("updatedAt"), 0),
        }
        return key, normalized

    def _legacy_studio_preview_slots(self, outputs):
        slots = {}
        for output in self._sort_studio_outputs(outputs):
            key = self._studio_preview_slot_key(
                output.get("workflowTabId"), output.get("nodeId"), output.get("fieldKey")
            )
            if key is None or key in slots or not output.get("id"):
                continue
            slots[key] = {
                "schemaVersion": 1,
                "workflowTabId": str(output["workflowTabId"]),
                "nodeId": str(output["nodeId"]),
                "fieldKey": str(output["fieldKey"]),
                "currentOutputId": str(output["id"]),
                "pendingClientRunId": None,
                "pendingTaskId": None,
                "generation": 1,
                "attemptIndex": output.get("attemptIndex"),
                "status": "ready",
                "updatedAt": int(output.get("createdAt") or 0),
            }
        return slots

    def _read_studio_output_state(self):
        history_file = self._studio_history_file()
        if not history_file.exists():
            return {"revision": 0, "previewSlots": {}, "outputs": []}

        try:
            with open(history_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            logger.error(f"Error reading Studio output history: {e}")
            return {"revision": 0, "previewSlots": {}, "outputs": []}

        if isinstance(payload, list):
            outputs = payload
            version = 1
            revision = 0
            raw_slots = None
        elif isinstance(payload, dict) and isinstance(payload.get("outputs"), list):
            outputs = payload["outputs"]
            version = self._studio_state_int(payload.get("version"), 1)
            revision = max(self._studio_state_int(payload.get("revision"), 0), 0)
            raw_slots = payload.get("previewSlots")
        else:
            outputs = []
            version = 1
            revision = 0
            raw_slots = None

        outputs = [output for output in outputs if isinstance(output, dict)]
        slots = {}
        slot_values = list(raw_slots.values()) if isinstance(raw_slots, dict) else raw_slots
        if isinstance(slot_values, (list, tuple)):
            for slot in slot_values:
                normalized = self._normalize_studio_preview_slot(slot)
                if normalized is not None:
                    slots[normalized[0]] = normalized[1]
        if version < 2:
            slots = self._legacy_studio_preview_slots(outputs)
        return {"revision": revision, "previewSlots": slots, "outputs": outputs}

    def _read_studio_outputs(self):
        return self._read_studio_output_state()["outputs"]

    def _write_studio_output_state(self, outputs, preview_slots, *, revision=None):
        history_file = self._studio_history_file()
        history_file.parent.mkdir(parents=True, exist_ok=True)
        current_ids = {
            str(slot.get("currentOutputId"))
            for slot in preview_slots.values()
            if isinstance(slot, dict) and slot.get("currentOutputId")
        }
        bounded_outputs = list(outputs[:200])
        bounded_ids = {str(output.get("id")) for output in bounded_outputs if output.get("id")}
        for output in outputs[200:]:
            output_id = str(output.get("id")) if output.get("id") else None
            if output_id in current_ids and output_id not in bounded_ids:
                bounded_outputs.append(output)
                bounded_ids.add(output_id)
        next_revision = max(self._studio_state_int(revision, 0), 0)
        payload = {
            "version": 2,
            "revision": next_revision,
            "updatedAt": int(time.time() * 1000),
            "previewSlots": preview_slots,
            "outputs": bounded_outputs,
        }
        temp_file = history_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        temp_file.replace(history_file)
        return bounded_outputs

    def _write_studio_outputs(self, outputs):
        state = self._read_studio_output_state()
        return self._write_studio_output_state(
            outputs,
            state["previewSlots"],
            revision=state["revision"] + 1,
        )

    def _studio_output_key(self, output):
        return str(
            output.get("id") or f"{output.get('nodeId', '')}:{output.get('fieldKey', '')}:{output.get('url', '')}"
        )

    def _hash_file(self, path):
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"sha256:bytes:{digest.hexdigest()}"

    def _hash_collection(self, hashes):
        digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8"))
        return f"sha256:collection:{digest.hexdigest()}"

    def _sort_studio_outputs(self, outputs):
        return sorted(outputs, key=lambda output: output.get("createdAt") or 0, reverse=True)

    def _merge_studio_outputs(self, existing, incoming):
        merged = {}
        order = []

        # Existing records establish durable paths/favorites; incoming records
        # then enrich or replace the same logical output.  Processing in the
        # opposite order silently kept stale retry media and discarded richer
        # frontend metadata for backend-captured outputs.
        for output in existing + incoming:
            key = self._studio_output_key(output)
            if not key:
                continue
            if key not in merged:
                order.append(key)
                merged[key] = output
                continue

            previous = merged[key]
            merged[key] = {
                **previous,
                **output,
                "favorite": bool(previous.get("favorite", False) or output.get("favorite", False)),
                "backendImagePath": output.get("backendImagePath") or previous.get("backendImagePath"),
                "backendMediaPath": output.get("backendMediaPath") or previous.get("backendMediaPath"),
                "backendSyncedAt": output.get("backendSyncedAt") or previous.get("backendSyncedAt"),
            }

        return self._sort_studio_outputs([merged[key] for key in order])

    def _generated_preview_fields_for_graph(self, graph):
        if not isinstance(graph, dict):
            return []
        runtime_hints = graph.get("runtimeHints")
        workflow_tab_id = runtime_hints.get("workflowTabId") if isinstance(runtime_hints, dict) else None
        nodes = graph.get("nodes")
        if not workflow_tab_id or not isinstance(nodes, dict):
            return []
        fields = []
        for node_id, node in nodes.items():
            if not isinstance(node, dict):
                continue
            module = node.get("module")
            action = node.get("action")
            definitions = self.modules.get(module, {}).get(action, {}).get("params", {})
            if not isinstance(definitions, dict):
                continue
            submitted_params = node.get("params") if isinstance(node.get("params"), dict) else {}
            for field_key, definition in definitions.items():
                if not isinstance(definition, dict):
                    continue
                if definition.get("display") not in {"ui_image", "ui_video", "ui_audio", "ui_text"}:
                    continue
                if definition.get("hidden") or (module == "modules.Audio" and action == "Load"):
                    continue
                # Only fields present in the submitted graph can receive an
                # update for this run. This avoids clearing unrelated optional
                # previews declared by a module but omitted from the graph.
                if field_key not in submitted_params:
                    continue
                fields.append((str(workflow_tab_id), str(node_id), str(field_key)))
        return fields

    def _mark_studio_preview_slots_pending(self, graph, task_id):
        fields = self._generated_preview_fields_for_graph(graph)
        if not fields:
            return {"revision": 0, "previewSlots": []}
        runtime_hints = graph.get("runtimeHints") if isinstance(graph, dict) else {}
        client_run_id = runtime_hints.get("clientRunId") if isinstance(runtime_hints, dict) else None
        attempt_index = runtime_hints.get("attemptIndex") if isinstance(runtime_hints, dict) else None
        now = int(time.time() * 1000)
        with self.studio_history_file_lock:
            state = self._read_studio_output_state()
            slots = state["previewSlots"]
            changed = []
            for workflow_tab_id, node_id, field_key in fields:
                key = self._studio_preview_slot_key(workflow_tab_id, node_id, field_key)
                previous = slots.get(key, {})
                slot = {
                    "schemaVersion": 1,
                    "workflowTabId": workflow_tab_id,
                    "nodeId": node_id,
                    "fieldKey": field_key,
                    "currentOutputId": None,
                    "pendingClientRunId": str(client_run_id) if client_run_id else None,
                    "pendingTaskId": str(task_id),
                    "generation": max(self._studio_state_int(previous.get("generation"), 0), 0) + 1,
                    "attemptIndex": self._studio_state_int(attempt_index) if attempt_index is not None else None,
                    "status": "pending",
                    "updatedAt": now,
                }
                slots[key] = slot
                changed.append(slot)
            revision = state["revision"] + 1
            self._write_studio_output_state(state["outputs"], slots, revision=revision)
        return {"revision": revision, "previewSlots": changed}

    def _studio_preview_slots_for_task(self, task_id):
        normalized_task_id = str(task_id or "")
        with self.studio_history_file_lock:
            state = self._read_studio_output_state()
        slots = [
            slot
            for slot in state["previewSlots"].values()
            if str(slot.get("pendingTaskId") or "") == normalized_task_id
            or (
                slot.get("currentOutputId")
                and any(
                    str(output.get("id") or "") == str(slot["currentOutputId"])
                    and str(output.get("taskId") or "") == normalized_task_id
                    for output in state["outputs"]
                )
            )
        ]
        return {"revision": state["revision"], "previewSlots": slots}

    def _mark_studio_preview_run_terminal(self, task_id, status):
        normalized_task_id = str(task_id or "")
        if not normalized_task_id:
            return None
        terminal_status = {
            "failed": "failed",
            "cancelled": "cancelled",
            "completed": "completed_without_output",
        }.get(status)
        if terminal_status is None:
            return None
        now = int(time.time() * 1000)
        with self.studio_history_file_lock:
            state = self._read_studio_output_state()
            changed = []
            for key, slot in state["previewSlots"].items():
                if str(slot.get("pendingTaskId") or "") != normalized_task_id:
                    continue
                updated = {
                    **slot,
                    "currentOutputId": None,
                    "pendingClientRunId": None,
                    "pendingTaskId": None,
                    "status": terminal_status,
                    "updatedAt": now,
                }
                state["previewSlots"][key] = updated
                changed.append(updated)
            if not changed:
                return None
            revision = state["revision"] + 1
            self._write_studio_output_state(
                state["outputs"], state["previewSlots"], revision=revision
            )
        return {"revision": revision, "previewSlots": changed}

    def _save_studio_output_image(self, output):
        image_data = output.pop("image_data", None)
        if not image_data or output.get("backendImagePath"):
            return output

        try:
            header = ""
            payload = image_data
            if isinstance(image_data, str) and image_data.startswith("data:") and "," in image_data:
                header, payload = image_data.split(",", 1)

            if not isinstance(payload, str):
                return output

            mime_type = header.split(";")[0].removeprefix("data:") if header else ""
            extension = {
                "image/webp": ".webp",
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "image/jpg": ".jpg",
                "image/gif": ".gif",
            }.get(mime_type, ".webp")

            output_id = "".join(
                ch for ch in str(output.get("id") or nanoid.generate(size=12)) if ch.isalnum() or ch in ("-", "_")
            )[:80]
            if not output_id:
                output_id = nanoid.generate(size=12)

            image_bytes = base64.b64decode(payload)
            output_dir = self._studio_outputs_dir()
            output_dir.mkdir(parents=True, exist_ok=True)
            image_path = output_dir / f"{output_id}{extension}"
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            image_file = data_path_identifier(image_path, self.data_dir)
            output["backendImagePath"] = image_file
            output["url"] = f"/file?file={quote(image_file)}"
            output["backendSyncedAt"] = int(time.time() * 1000)
            output["mediaHash"] = self._hash_file(image_path)
        except Exception as e:
            logger.error(f"Error saving Studio output image: {e}")

        return output

    def _save_studio_output_media(self, output):
        output = self._save_studio_output_image(output)
        if output.get("backendMediaPath") or output.get("backendImagePath"):
            return output
        if isinstance(output.get("mediaItems"), list) and output.get("mediaItems"):
            return output

        display_type = output.get("displayType")
        preview_url = output.get("url")
        if display_type != "video" and not str(preview_url or "").lower().split("?")[0].endswith(
            (".mp4", ".webm", ".mov", ".mkv")
        ):
            return output

        try:
            media = self._share_media_bytes_from_url(preview_url)
            if not media or not media.get("bytes"):
                return output

            content_type = media.get("contentType") or "application/octet-stream"
            extension = self._content_type_extension(content_type, media.get("filename") or preview_url)
            if extension.lower() not in (".mp4", ".webm", ".mov", ".mkv"):
                extension = ".mp4"

            output_id = "".join(
                ch for ch in str(output.get("id") or nanoid.generate(size=12)) if ch.isalnum() or ch in ("-", "_")
            )[:80]
            if not output_id:
                output_id = nanoid.generate(size=12)

            output_dir = self._studio_outputs_dir()
            output_dir.mkdir(parents=True, exist_ok=True)
            media_path = output_dir / f"{output_id}{extension}"
            with open(media_path, "wb") as f:
                f.write(media["bytes"])

            media_file = data_path_identifier(media_path, self.data_dir)
            output["backendMediaPath"] = media_file
            output["url"] = f"/file?file={quote(media_file)}"
            output["backendSyncedAt"] = int(time.time() * 1000)
            output["mediaHash"] = self._hash_file(media_path)
        except Exception as e:
            logger.error(f"Error saving Studio output media: {e}")

        return output

    def _safe_studio_output_id(self, output):
        output_id = "".join(
            ch for ch in str(output.get("id") or nanoid.generate(size=12)) if ch.isalnum() or ch in ("-", "_")
        )[:80]
        return output_id or nanoid.generate(size=12)

    def _save_studio_output_media_items(self, output):
        media_items = output.get("mediaItems")
        if not isinstance(media_items, list) or len(media_items) == 0:
            return output

        output_dir = self._studio_outputs_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_id = self._safe_studio_output_id(output)
        saved_items = []

        for index, item in enumerate(media_items):
            if not isinstance(item, dict):
                continue

            normalized = deepcopy(item)
            normalized["index"] = int(normalized.get("index", index) or index)
            if normalized.get("backendPath") and normalized.get("mediaHash"):
                saved_items.append(normalized)
                continue

            preview_url = normalized.get("url") or normalized.get("value")
            media = self._share_media_bytes_from_url(preview_url)
            if not media or not isinstance(media.get("bytes"), (bytes, bytearray)):
                saved_items.append(normalized)
                continue

            media_bytes = bytes(media["bytes"])
            content_type = media.get("contentType") or "application/octet-stream"
            extension = self._content_type_extension(content_type, media.get("filename") or preview_url)
            filename = f"{output_id}_item_{normalized['index']:02d}{extension}"
            media_path = output_dir / filename
            with open(media_path, "wb") as f:
                f.write(media_bytes)

            media_file = data_path_identifier(media_path, self.data_dir)

            normalized["backendPath"] = media_file
            normalized["url"] = f"/file?file={quote(media_file)}"
            normalized["mediaHash"] = self._hash_file(media_path)
            normalized["contentType"] = content_type
            normalized["byteSize"] = len(media_bytes)
            saved_items.append(normalized)

        if saved_items:
            output["mediaItems"] = saved_items
            first_item = saved_items[0]
            if first_item.get("url"):
                output["url"] = first_item["url"]
            if first_item.get("backendPath"):
                output["backendMediaPath"] = first_item["backendPath"]
                output["backendSyncedAt"] = int(time.time() * 1000)
            item_hashes = [item.get("mediaHash") for item in saved_items if item.get("mediaHash")]
            if item_hashes:
                output["mediaCollectionHash"] = self._hash_collection(item_hashes)
                if len(item_hashes) > 1:
                    output["mediaHash"] = output["mediaCollectionHash"]
                elif first_item.get("mediaHash"):
                    output["mediaHash"] = first_item["mediaHash"]

        return output

    def _normalize_studio_output(self, output):
        normalized = deepcopy(output)
        if not isinstance(normalized.get("id"), str) or not normalized.get("id"):
            normalized["id"] = nanoid.generate(size=12)
        if not normalized.get("createdAt"):
            normalized["createdAt"] = int(time.time() * 1000)
        if "favorite" not in normalized:
            normalized["favorite"] = False
        normalized = self._save_studio_output_media(normalized)
        normalized = self._save_studio_output_media_items(normalized)
        provenance = normalized.get("provenance") if isinstance(normalized.get("provenance"), dict) else {}
        runtime_fingerprint = provenance.get("runtimeFingerprint")
        if not runtime_fingerprint and self.current_task:
            runtime_fingerprint = self.current_task.get("runtimeFingerprint")
        normalized["backendProvenance"] = {
            "schemaVersion": 1,
            "source": "backend-record",
            "capturedAt": int(time.time() * 1000),
            "backendExecutionId": normalized.get("taskId") or normalized.get("runId"),
            "clientRunId": normalized.get("clientRunId"),
            "runInputHash": normalized.get("runInputHash"),
            "workflowTabId": normalized.get("workflowTabId"),
            "attemptIndex": normalized.get("attemptIndex"),
            "nodeId": normalized.get("nodeId"),
            "fieldKey": normalized.get("fieldKey"),
            "historyPath": str(self._studio_history_file()),
            "mediaPath": normalized.get("backendMediaPath") or normalized.get("backendImagePath"),
            "mediaHash": normalized.get("mediaHash"),
            "mediaCollectionHash": normalized.get("mediaCollectionHash"),
            "mediaItems": normalized.get("mediaItems"),
            "runtimeFingerprint": runtime_fingerprint,
            "templateId": normalized.get("templateId"),
            "templateLockHash": normalized.get("templateLockHash"),
            "promptSettingsHash": normalized.get("promptSettingsHash"),
        }
        return normalized

    def _generated_output_id(self, task_id, attempt_index, node_id, field_key):
        identity = json.dumps(
            [str(task_id), int(attempt_index or 0), str(node_id), str(field_key)],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        return f"run-output-{digest}"

    def _persist_generated_output_update(self, message, *, display):
        """Capture a generated preview before its cache entry can be replaced.

        Frontend history enrichment is useful but must not be the only durable
        record: the browser can reload or disconnect between update_value and
        its POST to /studio_outputs.
        """
        if not isinstance(message, dict) or display not in {"ui_image", "ui_video", "ui_audio", "ui_text"}:
            return None, False

        task_id = message.get("task_id")
        node_id = message.get("node")
        field_key = message.get("key")
        if not task_id or not node_id or not field_key:
            return None, False

        display_type = {
            "ui_image": "image",
            "ui_video": "video",
            "ui_audio": "audio",
            "ui_text": "text",
        }[display]
        value = message.get("value")
        values = value if isinstance(value, list) else [value]
        artifacts = message.get("artifacts") if isinstance(message.get("artifacts"), list) else []
        media_items = []

        if display_type == "text":
            text_value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
            media_items.append(
                {
                    "index": 0,
                    "value": value,
                    "url": f"data:text/plain;charset=utf-8,{quote(text_value, safe='')}",
                    "displayType": "text",
                    "taskId": task_id,
                    "clientRunId": message.get("client_run_id"),
                    "runInputHash": message.get("run_input_hash"),
                    "attemptIndex": message.get("attempt_index"),
                }
            )
        else:
            for index, item in enumerate(values):
                artifact = artifacts[index] if index < len(artifacts) and isinstance(artifacts[index], dict) else {}
                url = artifact.get("url") if isinstance(artifact.get("url"), str) else item
                if not isinstance(url, str) or not url:
                    continue
                media_items.append(
                    {
                        "index": index,
                        "value": item,
                        "url": url,
                        "displayType": display_type,
                        "contentType": artifact.get("mimeType"),
                        "width": artifact.get("width"),
                        "height": artifact.get("height"),
                        "durationSeconds": artifact.get("durationSeconds"),
                        "taskId": task_id,
                        "clientRunId": message.get("client_run_id"),
                        "runInputHash": message.get("run_input_hash"),
                        "attemptIndex": message.get("attempt_index"),
                    }
                )

        if not media_items:
            return None, False

        graph = self.task_graphs.get(str(task_id))
        graph_runtime_hints = graph.get("runtimeHints") if isinstance(graph, dict) else None
        runtime_hints = (
            graph_runtime_hints
            if isinstance(graph_runtime_hints, dict)
            else (self.current_task.get("runtimeHints") if self.current_task else {})
        )
        workflow_snapshot = runtime_hints.get("workflowSnapshot") if isinstance(runtime_hints, dict) else None
        workflow_snapshot = workflow_snapshot if isinstance(workflow_snapshot, dict) else {}
        form_snapshot = workflow_snapshot.get("studioForm")
        form_snapshot = form_snapshot if isinstance(form_snapshot, dict) else {}
        graph_snapshot = {
            key: deepcopy(workflow_snapshot[key]) for key in ("nodes", "edges", "viewport") if key in workflow_snapshot
        }
        output_id = self._generated_output_id(task_id, message.get("attempt_index"), node_id, field_key)
        output = {
            "id": output_id,
            "taskId": task_id,
            "clientRunId": message.get("client_run_id"),
            "runInputHash": message.get("run_input_hash"),
            "workflowTabId": message.get("workflow_tab_id"),
            "attemptIndex": message.get("attempt_index"),
            "nodeId": node_id,
            "fieldKey": field_key,
            "value": value,
            "url": media_items[0]["url"],
            "createdAt": int(time.time() * 1000),
            "favorite": False,
            "displayType": "image_collection" if display_type == "image" and len(media_items) > 1 else display_type,
            "mediaItems": media_items,
            "sid": self.current_task.get("sid") if self.current_task else None,
            "mode": form_snapshot.get("mode"),
            "modelType": form_snapshot.get("modelType")
            or (runtime_hints.get("modelType") if isinstance(runtime_hints, dict) else None),
            "modelLabel": runtime_hints.get("modelName") if isinstance(runtime_hints, dict) else None,
            "repo": runtime_hints.get("resolvedArtifact") or runtime_hints.get("modelRepo")
            if isinstance(runtime_hints, dict)
            else None,
            "prompt": form_snapshot.get("prompt"),
            "negativePrompt": form_snapshot.get("negativePrompt"),
            "seed": form_snapshot.get("seed"),
            "width": form_snapshot.get("width"),
            "height": form_snapshot.get("height"),
            "steps": form_snapshot.get("steps"),
            "guidanceScale": form_snapshot.get("guidanceScale"),
            "referenceImages": form_snapshot.get("referenceImages"),
            "formSnapshot": deepcopy(form_snapshot) if form_snapshot else None,
            "graphSnapshot": graph_snapshot or None,
            "graphBindingSnapshot": deepcopy(workflow_snapshot.get("studioGraphBinding")),
            "templateId": workflow_snapshot.get("activeTemplateId"),
            "sourceOutputId": workflow_snapshot.get("sourceOutputId"),
            "provenance": {
                "schemaVersion": 1,
                "source": "backend-record",
                "capturedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "backendExecutionId": task_id,
                "clientRunId": message.get("client_run_id"),
                "runInputHash": message.get("run_input_hash"),
                "workflowTabId": message.get("workflow_tab_id"),
                "attemptIndex": message.get("attempt_index"),
                "nodeId": node_id,
                "runtimeFingerprint": message.get("runtimeFingerprint"),
                "mediaItems": media_items,
            },
        }
        output = {key: item for key, item in output.items() if item is not None}

        try:
            with self.studio_history_file_lock:
                normalized = self._normalize_studio_output(output)
                state = self._read_studio_output_state()
                outputs = self._merge_studio_outputs(state["outputs"], [normalized])
                slot_key = self._studio_preview_slot_key(
                    output.get("workflowTabId"), node_id, field_key
                )
                promoted_slot = None
                if slot_key is not None:
                    previous = state["previewSlots"].get(slot_key)
                    pending_matches = bool(
                        previous
                        and str(previous.get("pendingTaskId") or "") == str(task_id)
                        and (
                            not previous.get("pendingClientRunId")
                            or str(previous.get("pendingClientRunId"))
                            == str(message.get("client_run_id") or "")
                        )
                    )
                    # A newer accepted run owns the pending slot and must not
                    # be displaced by a late output from the run ahead of it.
                    # Otherwise every update from the currently executing run
                    # may refresh its own durable current output (including an
                    # automatic retry with a new attempt index).
                    may_promote = previous is None or pending_matches or not previous.get("pendingTaskId")
                    if may_promote:
                        promoted_slot = {
                            "schemaVersion": 1,
                            "workflowTabId": str(output["workflowTabId"]),
                            "nodeId": str(node_id),
                            "fieldKey": str(field_key),
                            "currentOutputId": output_id,
                            "pendingClientRunId": None,
                            "pendingTaskId": None,
                            "generation": max(
                                self._studio_state_int((previous or {}).get("generation"), 0), 0
                            )
                            or 1,
                            "attemptIndex": message.get("attempt_index"),
                            "status": "ready",
                            "updatedAt": int(time.time() * 1000),
                        }
                        state["previewSlots"][slot_key] = promoted_slot
                revision = state["revision"] + 1
                self._write_studio_output_state(
                    outputs, state["previewSlots"], revision=revision
                )
                if promoted_slot is not None:
                    message["preview_slot"] = promoted_slot
                    message["preview_state_revision"] = revision
            return output_id, True
        except Exception as error:
            logger.error(f"Error preserving generated Studio output {output_id}: {error}")
            return output_id, False

    async def studio_outputs_get(self, request):
        limit = min(max(int(request.query.get("limit", 80)), 1), 200)
        with self.studio_history_file_lock:
            state = self._read_studio_output_state()
        outputs = state["outputs"]
        response_outputs = list(outputs[:limit])
        response_ids = {str(output.get("id")) for output in response_outputs if output.get("id")}
        current_ids = {
            str(slot.get("currentOutputId"))
            for slot in state["previewSlots"].values()
            if slot.get("currentOutputId")
        }
        for output in outputs[limit:]:
            output_id = str(output.get("id")) if output.get("id") else None
            if output_id in current_ids and output_id not in response_ids:
                response_outputs.append(output)
                response_ids.add(output_id)
        return web.json_response(
            {
                "error": False,
                "count": len(outputs),
                "outputs": response_outputs,
                "previewSlots": list(state["previewSlots"].values()),
                "revision": state["revision"],
                "path": str(self._studio_history_file()),
            }
        )

    async def studio_outputs_post(self, request):
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": True, "message": "Invalid JSON body."}, status=400)

        raw_outputs = (
            payload.get("outputs") if isinstance(payload, dict) and isinstance(payload.get("outputs"), list) else None
        )
        if raw_outputs is None:
            raw_outputs = [payload] if isinstance(payload, dict) else []

        async with self.studio_history_lock:
            with self.studio_history_file_lock:
                incoming = [
                    self._normalize_studio_output(output) for output in raw_outputs if isinstance(output, dict)
                ]
                if not incoming:
                    return web.json_response({"error": True, "message": "No Studio outputs supplied."}, status=400)
                existing = self._read_studio_outputs()
                outputs = self._write_studio_outputs(self._merge_studio_outputs(existing, incoming))
                state = self._read_studio_output_state()

        return web.json_response(
            {
                "error": False,
                "count": len(outputs),
                "outputs": outputs,
                "previewSlots": list(state["previewSlots"].values()),
                "revision": state["revision"],
            }
        )

    async def studio_outputs_patch(self, request):
        output_id = request.match_info.get("output_id")
        if not output_id:
            return web.json_response({"error": True, "message": "Missing Studio output id."}, status=400)

        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": True, "message": "Invalid JSON body."}, status=400)

        async with self.studio_history_lock:
            with self.studio_history_file_lock:
                outputs = self._read_studio_outputs()
                updated = False
                for index, output in enumerate(outputs):
                    if str(output.get("id")) != output_id:
                        continue
                    outputs[index] = {
                        **output,
                        **payload,
                        "id": output_id,
                        "updatedAt": int(time.time() * 1000),
                    }
                    updated = True
                    break

                if not updated:
                    return web.json_response(
                        {"error": True, "message": f"Studio output {output_id} was not found."}, status=404
                    )

                outputs = self._write_studio_outputs(self._sort_studio_outputs(outputs))
                state = self._read_studio_output_state()

        return web.json_response(
            {
                "error": False,
                "count": len(outputs),
                "outputs": outputs,
                "previewSlots": list(state["previewSlots"].values()),
                "revision": state["revision"],
            }
        )

    async def studio_outputs_delete(self, request):
        output_id = request.match_info.get("output_id")
        if not output_id:
            return web.json_response({"error": True, "message": "Missing Studio output id."}, status=400)

        async with self.studio_history_lock:
            with self.studio_history_file_lock:
                state = self._read_studio_output_state()
                outputs = state["outputs"]
                next_outputs = [output for output in outputs if str(output.get("id")) != output_id]
                if len(next_outputs) == len(outputs):
                    return web.json_response(
                        {"error": True, "message": f"Studio output {output_id} was not found."}, status=404
                    )
                now = int(time.time() * 1000)
                for key, slot in state["previewSlots"].items():
                    if str(slot.get("currentOutputId") or "") != output_id:
                        continue
                    state["previewSlots"][key] = {
                        **slot,
                        "currentOutputId": None,
                        "status": "empty",
                        "updatedAt": now,
                    }
                revision = state["revision"] + 1
                outputs = self._write_studio_output_state(
                    next_outputs, state["previewSlots"], revision=revision
                )

        return web.json_response(
            {
                "error": False,
                "count": len(outputs),
                "outputs": outputs,
                "previewSlots": list(state["previewSlots"].values()),
                "revision": revision,
            }
        )

    async def studio_blocks_get(self, request):
        limit = min(max(int(request.query.get("limit", 200)), 1), 500)
        blocks = self._list_studio_blocks()
        return web.json_response(
            {
                "error": False,
                "count": len(blocks),
                "blocks": blocks[:limit],
                "path": str(self._studio_blocks_dir()),
            }
        )

    async def studio_blocks_post(self, request):
        try:
            payload = await request.json()
            block, block_file = self._write_studio_block(payload)
        except json.JSONDecodeError:
            return web.json_response({"error": True, "message": "Invalid JSON body."}, status=400)
        except ValueError as e:
            return web.json_response({"error": True, "message": str(e)}, status=400)
        except OSError as e:
            logger.error(f"Error saving Studio user block: {e}")
            return web.json_response({"error": True, "message": "Could not save user block."}, status=500)

        return web.json_response(
            {
                "error": False,
                "block": block,
                "path": str(block_file),
            }
        )

    async def studio_block_get(self, request):
        block_id = request.match_info.get("block_id")
        if not block_id:
            return web.json_response({"error": True, "message": "Missing user block id."}, status=400)
        try:
            block = self._read_studio_block(block_id)
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            logger.error(f"Error reading Studio user block {block_id}: {e}")
            return web.json_response({"error": True, "message": "Could not read user block."}, status=500)
        if block is None:
            return web.json_response({"error": True, "message": f"User block {block_id} was not found."}, status=404)
        return web.json_response(
            {
                "error": False,
                "block": block,
            }
        )

    async def studio_block_delete(self, request):
        block_id = request.match_info.get("block_id")
        if not block_id:
            return web.json_response({"error": True, "message": "Missing user block id."}, status=400)
        block_file = self._studio_block_file(block_id)
        if not block_file.exists():
            return web.json_response({"error": True, "message": f"User block {block_id} was not found."}, status=404)
        try:
            block_file.unlink()
        except OSError as e:
            logger.error(f"Error deleting Studio user block {block_id}: {e}")
            return web.json_response({"error": True, "message": "Could not delete user block."}, status=500)
        return web.json_response(
            {
                "error": False,
                "id": self._safe_block_id(block_id),
            }
        )

    def _workflow_shares_dir(self):
        return Path(self.data_dir) / "studio" / "shares"

    def _safe_share_id(self, share_id=None):
        raw_id = str(share_id or nanoid.generate(size=12))
        safe_id = "".join(ch for ch in raw_id if ch.isalnum() or ch in ("-", "_"))[:80]
        return safe_id or nanoid.generate(size=12)

    def _workflow_share_file(self, share_id):
        return self._workflow_shares_dir() / f"{self._safe_share_id(share_id)}.json"

    def _workflow_share_media_dir(self, share_id):
        return self._workflow_shares_dir() / self._safe_share_id(share_id) / "media"

    def _path_within(self, path, root):
        try:
            Path(path).resolve(strict=False).relative_to(Path(root).resolve(strict=False))
            return True
        except ValueError:
            return False

    def _safe_media_filename(self, filename, fallback="preview.bin"):
        raw_name = Path(str(filename or fallback)).name
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "-" for ch in raw_name)[:120].strip(".-")
        return safe_name or fallback

    def _share_media_hash(self, data):
        return f"sha256:bytes:{hashlib.sha256(data).hexdigest()}"

    def _content_type_extension(self, content_type, fallback_url=""):
        normalized = str(content_type or "").split(";")[0].strip().lower()
        explicit = {
            "image/webp": ".webp",
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/gif": ".gif",
            "video/mp4": ".mp4",
            "video/webm": ".webm",
            "audio/wav": ".wav",
            "audio/mpeg": ".mp3",
            "audio/flac": ".flac",
            "application/json": ".json",
            "text/plain": ".txt",
        }.get(normalized)
        if explicit:
            return explicit

        parsed_suffix = Path(urlparse(str(fallback_url or "")).path).suffix.lower()
        if parsed_suffix:
            return parsed_suffix
        return (mimetypes.guess_extension(normalized) or ".bin") if normalized else ".bin"

    def _resolve_file_route_path(self, file):
        if not file:
            return None

        file_path = self._resolve_managed_path_identifier(unquote(str(file)))
        if file_path is None or not file_path.exists():
            return None
        return file_path

    def _cache_media_bytes_from_url(self, preview_url):
        parsed = urlparse(str(preview_url))
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        if len(parts) < 3 or parts[0] != "cache":
            return None

        node, field = parts[1], parts[2]
        index = parts[3] if len(parts) > 3 else None
        if node not in self.node_cache:
            return None

        cached_node = self.node_cache[node]
        if field in cached_node.output:
            data = cached_node.output[field]
        elif field in cached_node.params:
            data = cached_node.params[field]
        else:
            return None

        if data is None:
            return None

        if isinstance(data, list):
            try:
                selected_index = max(0, min(len(data) - 1, int(index))) if index is not None else 0
            except (TypeError, ValueError):
                selected_index = 0
            data = data[selected_index] if data else None

        if data is None:
            return None

        params = self.modules.get(cached_node.module_name, {}).get(cached_node.class_name, {}).get("params", {})
        data_type = params.get(field, {}).get("type")
        type_values = data_type if isinstance(data_type, list) else [data_type]
        query = parse_qs(parsed.query)
        filename = query.get("filename", [field])[0]

        if "image" in type_values:
            image_format = query.get("format", ["WEBP"])[0].upper()
            quality = query.get("quality", [100])[0]
            return {
                "bytes": to_bytes(data_type, data, {"format": image_format, "quality": quality}),
                "contentType": f"image/{image_format.lower()}",
                "filename": f"{filename}.{image_format.lower()}",
            }

        if data_type == "text" or any(isinstance(item, str) and item.startswith("str") for item in type_values):
            return {
                "bytes": str(data).encode("utf-8"),
                "contentType": "text/plain",
                "filename": f"{filename}.txt",
            }

        data_path = Path(str(data))
        if data_path.exists() and (
            self._path_within(data_path, self.work_dir) or self._path_within(data_path, self.data_dir)
        ):
            content_type = mimetypes.guess_type(str(data_path))[0] or "application/octet-stream"
            return {
                "bytes": data_path.read_bytes(),
                "contentType": content_type,
                "filename": data_path.name,
            }
        return None

    def _share_media_bytes_from_url(self, preview_url):
        if not isinstance(preview_url, str) or not preview_url:
            return None

        if preview_url.startswith("data:") and "," in preview_url:
            header, payload = preview_url.split(",", 1)
            content_type = header.split(";")[0].removeprefix("data:") or "application/octet-stream"
            if len(payload) > (MAX_WORKFLOW_SHARE_MEDIA_BYTES * 4 // 3) + 4:
                raise ValueError("Embedded workflow share media exceeds the 256 MB limit.")
            data = base64.b64decode(payload) if ";base64" in header else unquote_to_bytes(payload)
            if len(data) > MAX_WORKFLOW_SHARE_MEDIA_BYTES:
                raise ValueError("Embedded workflow share media exceeds the 256 MB limit.")
            return {
                "bytes": data,
                "contentType": content_type,
                "filename": f"preview{self._content_type_extension(content_type, preview_url)}",
            }

        parsed = urlparse(preview_url)
        if parsed.path.startswith("/workflows/share/"):
            return None
        if parsed.scheme in ("http", "https") and not parsed.path.startswith("/cache/") and parsed.path != "/file":
            return None

        if parsed.path.startswith("/cache/"):
            return self._cache_media_bytes_from_url(preview_url)

        if parsed.path == "/file":
            file_path = self._resolve_file_route_path(parse_qs(parsed.query).get("file", [""])[0])
            if not file_path:
                return None
            if file_path.stat().st_size > MAX_WORKFLOW_SHARE_MEDIA_BYTES:
                raise ValueError("Workflow share preview media exceeds the 256 MB limit.")
            return {
                "bytes": file_path.read_bytes(),
                "contentType": mimetypes.guess_type(str(file_path))[0] or "application/octet-stream",
                "filename": file_path.name,
            }

        return None

    def _persist_share_preview_media(self, share_id, package):
        preview = self._share_preview_url(package)
        media = self._share_media_bytes_from_url(preview)
        if not media:
            return package, None
        if len(media.get("bytes") or b"") > MAX_WORKFLOW_SHARE_MEDIA_BYTES:
            raise ValueError("Workflow share preview media exceeds the 256 MB limit.")

        try:
            safe_share_id = self._safe_share_id(share_id)
            content_type = media.get("contentType") or "application/octet-stream"
            filename = self._safe_media_filename(
                media.get("filename"), f"preview{self._content_type_extension(content_type, preview)}"
            )
            if not Path(filename).suffix:
                filename = f"{filename}{self._content_type_extension(content_type, preview)}"

            target_dir = self._workflow_share_media_dir(safe_share_id)
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = (target_dir / filename).resolve()
            if not self._path_within(target_path, target_dir):
                raise ValueError("Resolved share media path escaped the share media directory.")

            media_bytes = media.get("bytes")
            if not isinstance(media_bytes, (bytes, bytearray)):
                return package, None

            with open(target_path, "wb") as f:
                f.write(media_bytes)

            byte_hash = self._share_media_hash(bytes(media_bytes))
            media_url = f"/workflows/share/{quote(safe_share_id)}/media/{quote(filename)}"
            next_package = deepcopy(package)

            manifest = next_package.setdefault("manifest", {}) if isinstance(next_package, dict) else {}
            manifest_media = manifest.get("media") if isinstance(manifest.get("media"), dict) else {}
            manifest_media.update(
                {
                    "url": media_url,
                    "persistedUrl": media_url,
                    "byteHash": byte_hash,
                    "contentType": content_type,
                }
            )
            manifest_media.pop("backendShareMediaPath", None)
            manifest["media"] = manifest_media

            metadata = next_package.setdefault("metadata", {}) if isinstance(next_package, dict) else {}
            metadata["preview"] = media_url

            latest_output = (
                next_package.get("latestOutput") if isinstance(next_package.get("latestOutput"), dict) else None
            )
            if latest_output is not None:
                latest_output["url"] = media_url
                latest_output["backendShareMediaHash"] = byte_hash
                latest_output.pop("backendShareMediaPath", None)

            persisted = {
                "url": media_url,
                "filename": filename,
                "byteHash": byte_hash,
                "contentType": content_type,
            }
            return next_package, persisted
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error persisting workflow share media {share_id}: {e}")
            return package, None

    def _public_workflow_share(self, share):
        """Strip backend filesystem details from old and new public shares."""

        public_share = deepcopy(share) if isinstance(share, dict) else {}
        persisted_media = public_share.get("persistedMedia")
        if isinstance(persisted_media, dict):
            persisted_media.pop("path", None)
        package = public_share.get("package")
        if isinstance(package, dict):
            manifest = package.get("manifest")
            media = manifest.get("media") if isinstance(manifest, dict) else None
            if isinstance(media, dict):
                media.pop("backendShareMediaPath", None)
            latest_output = package.get("latestOutput")
            if isinstance(latest_output, dict):
                latest_output.pop("backendShareMediaPath", None)
        return public_share

    def _share_summary(self, share):
        share = self._public_workflow_share(share)
        package = share.get("package", {}) if isinstance(share, dict) else {}
        metadata = package.get("metadata", {}) if isinstance(package, dict) else {}
        studio = metadata.get("studio", {}) if isinstance(metadata, dict) else {}
        return {
            "share_id": share.get("share_id"),
            "createdAt": share.get("createdAt"),
            "updatedAt": share.get("updatedAt"),
            "url": share.get("url"),
            "modelType": studio.get("modelType"),
            "mode": studio.get("mode"),
            "prompt": studio.get("prompt"),
            "preview": self._share_preview_url(package),
            "persistedMedia": share.get("persistedMedia"),
        }

    def _share_preview_url(self, package):
        if not isinstance(package, dict):
            return None
        metadata = package.get("metadata", {})
        manifest = package.get("manifest", {})
        media = manifest.get("media", {}) if isinstance(manifest, dict) else {}
        preview = metadata.get("preview") if isinstance(metadata, dict) else None
        if not preview and isinstance(media, dict):
            preview = media.get("url")
        if not isinstance(preview, str):
            return None
        if preview.startswith(("http://", "https://", "data:image/", "/")):
            return preview
        return None

    def _workflow_share_preview_html(self, request, share):
        share_id = self._safe_share_id(share.get("share_id"))
        package = share.get("package", {}) if isinstance(share, dict) else {}
        metadata = package.get("metadata", {}) if isinstance(package, dict) else {}
        manifest = package.get("manifest", {}) if isinstance(package, dict) else {}
        studio = metadata.get("studio", {}) if isinstance(metadata, dict) else {}
        template = manifest.get("template", {}) if isinstance(manifest, dict) else {}
        provenance = manifest.get("provenance", {}) if isinstance(manifest, dict) else {}
        frontend_url = f"/?share={quote(share_id)}"
        json_url = f"/workflows/share/{quote(share_id)}?format=json"
        preview = self._share_preview_url(package)
        title = studio.get("prompt") or template.get("templateLabel") or f"MoDiff workflow {share_id}"
        prompt = studio.get("prompt") or ""
        mode = studio.get("mode") or "workflow"
        model_type = studio.get("modelType") or "unknown model"
        exported_at = metadata.get("exportedAt") or manifest.get("exportedAt") or share.get("createdAt") or ""
        media_hash = None
        if isinstance(provenance, dict):
            frontend = provenance.get("frontend", {})
            backend = provenance.get("backend", {})
            if isinstance(frontend, dict):
                media_hash = frontend.get("mediaHash")
            if not media_hash and isinstance(backend, dict):
                media_hash = backend.get("mediaHash")

        def esc(value):
            return html.escape(str(value or ""), quote=True)

        preview_html = ""
        if preview:
            preview_html = f'<img class="preview" src="{esc(preview)}" alt="Shared workflow preview" />'

        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MoDiff Share - {esc(share_id)}</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin: 0; font-family: system-ui, sans-serif; background: Canvas; color: CanvasText; }}
    main {{ max-width: 920px; margin: 0 auto; padding: 24px; }}
    .preview {{ display: block; max-width: 100%; max-height: 520px; object-fit: contain; border: 1px solid ButtonBorder; }}
    .panel {{ display: grid; gap: 12px; border: 1px solid ButtonBorder; padding: 16px; background: Field; }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    a.button {{ display: inline-block; padding: 8px 12px; border: 1px solid ButtonBorder; color: LinkText; text-decoration: none; font-weight: 700; }}
    dl {{ display: grid; grid-template-columns: minmax(120px, max-content) 1fr; gap: 8px 12px; }}
    dt {{ font-weight: 700; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <h1>{esc(title)}</h1>
      {preview_html}
      <div class="actions">
        <a class="button" href="{esc(frontend_url)}">Open and restore in MoDiff</a>
        <a class="button" href="{esc(json_url)}">View JSON package</a>
      </div>
      <dl>
        <dt>Share</dt><dd>{esc(share_id)}</dd>
        <dt>Mode</dt><dd>{esc(mode)}</dd>
        <dt>Model</dt><dd>{esc(model_type)}</dd>
        <dt>Exported</dt><dd>{esc(exported_at)}</dd>
        <dt>Template</dt><dd>{esc(template.get("templateLabel") if isinstance(template, dict) else "")}</dd>
        <dt>Media hash</dt><dd>{esc(media_hash)}</dd>
        <dt>Prompt</dt><dd>{esc(prompt)}</dd>
      </dl>
    </section>
  </main>
</body>
</html>"""

    async def workflow_shares_list(self, request):
        limit = min(max(int(request.query.get("limit", 50)), 1), 200)
        shares_dir = self._workflow_shares_dir()
        summaries = []

        if shares_dir.exists():
            for share_file in shares_dir.glob("*.json"):
                try:
                    with open(share_file, "r", encoding="utf-8") as f:
                        share = json.load(f)
                    summaries.append(self._share_summary(share))
                except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
                    logger.error(f"Error reading workflow share {share_file}: {e}")

        summaries.sort(key=lambda item: item.get("createdAt") or "", reverse=True)
        return web.json_response(
            {
                "error": False,
                "count": len(summaries),
                "shares": summaries[:limit],
            }
        )

    async def workflow_share_post(self, request):
        try:
            package = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": True, "message": "Invalid JSON body."}, status=400)

        if not isinstance(package, dict):
            return web.json_response(
                {"error": True, "message": "Workflow share package must be a JSON object."}, status=400
            )

        share_id = self._safe_share_id(package.get("share_id") or package.get("shareId"))
        try:
            package, persisted_media = self._persist_share_preview_media(share_id, package)
        except ValueError as exc:
            return web.json_response({"error": True, "message": str(exc)}, status=413)
        created_at = package.get("createdAt") or int(time.time() * 1000)
        url = f"/workflows/share/{share_id}"
        share = {
            "share_id": share_id,
            "createdAt": created_at,
            "updatedAt": int(time.time() * 1000),
            "url": url,
            "package": package,
        }
        if persisted_media:
            share["persistedMedia"] = persisted_media

        shares_dir = self._workflow_shares_dir()
        shares_dir.mkdir(parents=True, exist_ok=True)
        share_file = self._workflow_share_file(share_id)
        temp_file = share_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(share, f, ensure_ascii=False)
            temp_file.replace(share_file)
        except OSError as e:
            logger.error(f"Error saving workflow share {share_id}: {e}")
            return web.json_response({"error": True, "message": str(e)}, status=500)

        return web.json_response(
            {
                "error": False,
                "share_id": share_id,
                "url": url,
                "persistedMedia": persisted_media,
                "share": self._public_workflow_share(share),
            }
        )

    async def workflow_share_media_get(self, request):
        share_id = self._safe_share_id(request.match_info.get("share_id"))
        filename = self._safe_media_filename(request.match_info.get("filename"))
        media_dir = self._workflow_share_media_dir(share_id)
        media_path = (media_dir / filename).resolve()

        if not self._path_within(media_path, media_dir):
            return web.json_response({"error": True, "message": "Invalid workflow share media path."}, status=403)
        if not media_path.exists() or not media_path.is_file():
            return web.json_response(
                {"error": True, "message": f"Workflow share media {filename} was not found."}, status=404
            )

        resp = web.FileResponse(media_path)
        resp.headers["Content-Disposition"] = f'inline; filename="{filename}"'
        resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return resp

    async def workflow_share_get(self, request):
        share_id = request.match_info.get("share_id")
        if not share_id:
            return web.json_response({"error": True, "message": "Missing workflow share id."}, status=400)

        share_file = self._workflow_share_file(share_id)
        if not share_file.exists():
            return web.json_response(
                {"error": True, "message": f"Workflow share {share_id} was not found."}, status=404
            )

        try:
            with open(share_file, "r", encoding="utf-8") as f:
                share = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            logger.error(f"Error reading workflow share {share_id}: {e}")
            return web.json_response({"error": True, "message": str(e)}, status=500)

        wants_html = request.query.get("format") != "json" and "text/html" in request.headers.get("Accept", "")
        if wants_html:
            return web.Response(
                text=self._workflow_share_preview_html(request, share),
                content_type="text/html",
            )

        return web.json_response(
            {
                "error": False,
                **self._public_workflow_share(share),
            }
        )

    """
    ╭────────────────────────╮
       Main graph execution
    ╰────────────────────────╯
    """

    async def graph(self, request):
        graph = await request.json()
        sid = graph.get("sid")
        # if not sid:
        #    return web.json_response({"error": True, "message": "Missing session id"}, status=400)

        if self._runtime_mutation_gate is not None:
            return web.json_response(
                {
                    "error": True,
                    "error_code": "runtime_mutation_busy",
                    "message": "A runtime install, activation, rollback, or restart is in progress.",
                },
                status=409,
            )

        optional_runtime_requirement = graph_optional_runtime_requirement(graph)

        if optional_runtime_requirement_blocks_execution(optional_runtime_requirement):
            return web.json_response(
                optional_runtime_blocker_payload(optional_runtime_requirement),
                status=409,
            )

        runtime_block = self._auto_resource_runtime_block()
        if runtime_block:
            issue = runtime_block["issue"]
            repair_action = runtime_block["repairAction"]
            return web.json_response(
                {
                    "error": True,
                    "message": issue["message"],
                    "category": issue["category"],
                    "error_code": issue["code"],
                    "recovery_hint": repair_action["label"],
                    "repair_action": repair_action,
                    "runtime_profile": runtime_block["runtimeProfile"],
                },
                status=409,
            )

        task_id = await self.queue_task(self.execute_graph, (graph,), None, sid, name="Graph execution")
        preview_state = self._studio_preview_slots_for_task(task_id)
        return web.json_response(
            {
                "error": False,
                "message": "Graph queued for processing",
                "sid": sid,
                "task_id": task_id,
                "preview_slots": preview_state["previewSlots"],
                "preview_state_revision": preview_state["revision"],
            }
        )

    def _exception_chain(self, e):
        chain = []
        seen = set()
        current = e
        while current is not None and id(current) not in seen:
            chain.append(current)
            seen.add(id(current))
            current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        return chain

    def _loader_diagnostics_snapshot(self):
        diagnostics = {}
        runtime_hints = self.current_task.get("runtimeHints") if self.current_task else None
        runtime_hint_offload = runtime_hints.get("offloadMode") if isinstance(runtime_hints, dict) else None
        runtime_hint_resource_mode = runtime_hints.get("resourceMode") if isinstance(runtime_hints, dict) else None
        runtime_hint_resolved_mode = (
            runtime_hints.get("resolvedResourceMode") if isinstance(runtime_hints, dict) else None
        )

        node_cache = getattr(self, "node_cache", {}) or {}
        for node_id, node in node_cache.items():
            value = getattr(node, "_loader_diagnostics", None)
            if not isinstance(value, dict):
                continue
            entry = deepcopy(value)
            entry["runtime_hint_offload_mode"] = runtime_hint_offload
            entry["runtime_hint_resource_mode"] = runtime_hint_resource_mode
            entry["runtime_hint_resolved_resource_mode"] = runtime_hint_resolved_mode
            diagnostics[str(node_id)] = entry
        return diagnostics

    def _run_identity_payload(self, runtime_hints):
        if not isinstance(runtime_hints, dict):
            return {}
        payload = {}
        client_run_id = runtime_hints.get("clientRunId")
        run_input_hash = runtime_hints.get("runInputHash")
        workflow_tab_id = runtime_hints.get("workflowTabId")
        workflow_canvas_epoch = runtime_hints.get("workflowCanvasEpoch")
        workflow_form_epoch = runtime_hints.get("workflowFormEpoch")
        node_id = runtime_hints.get("nodeId")
        if client_run_id:
            payload["client_run_id"] = client_run_id
        if run_input_hash:
            payload["run_input_hash"] = run_input_hash
        if workflow_tab_id:
            payload["workflow_tab_id"] = workflow_tab_id
        if workflow_canvas_epoch is not None:
            payload["workflow_canvas_epoch"] = workflow_canvas_epoch
        if workflow_form_epoch is not None:
            payload["workflow_form_epoch"] = workflow_form_epoch
        if node_id:
            payload["node_id"] = node_id
        return payload

    def _run_navigation_payload(self, runtime_hints):
        """Expose the captured workflow needed to navigate while execution is busy.

        This payload is intentionally attached only to queue/current snapshots,
        not every progress event, so a reconnect or activity click can restore
        the owning graph without repeatedly broadcasting a full document.
        """
        if not isinstance(runtime_hints, dict):
            return {}
        payload = {}
        workflow_title = runtime_hints.get("workflowTitle")
        workflow_snapshot = runtime_hints.get("workflowSnapshot")
        if workflow_title:
            payload["workflow_title"] = workflow_title
        if isinstance(workflow_snapshot, dict):
            payload["workflow_snapshot"] = workflow_snapshot
        return payload

    def _current_run_identity_payload(self):
        runtime_hints = self.current_task.get("runtimeHints") if self.current_task else None
        return self._run_identity_payload(runtime_hints)

    def _current_dynamic_message_identity_payload(self):
        """Correlate node-owned UI mutations with the executing workflow."""

        payload = self._current_run_identity_payload()
        if not self.current_task:
            return payload
        task_id = self.current_task.get("task_id")
        attempt_index = self.current_task.get("attempt_index")
        sid = self.current_task.get("sid")
        if task_id:
            payload["task_id"] = task_id
        if attempt_index is not None:
            payload["attempt_index"] = attempt_index
        if sid:
            payload["sid"] = sid
        return payload

    def _exception_payload(self, e, task_id=None, sid=None, node_id=None, node_name=None, traceback_text=None):
        optional_runtime_requirement = getattr(
            e,
            "modiff_optional_runtime_requirement",
            None,
        )
        if isinstance(optional_runtime_requirement, dict):
            payload = optional_runtime_blocker_payload(
                optional_runtime_requirement
            )
            if isinstance(task_id, str) and re.fullmatch(
                r"[A-Za-z0-9_-]{1,64}", task_id
            ):
                payload["task_id"] = task_id
            if isinstance(node_id, str) and re.fullmatch(
                r"[A-Za-z0-9_-]{1,128}", node_id
            ):
                payload["node"] = node_id
            if isinstance(node_name, str) and re.fullmatch(
                r"[A-Za-z0-9_.:-]{1,256}", node_name
            ):
                payload["node_name"] = node_name
            return payload

        exception_type = type(e).__name__
        message = str(e) or exception_type
        classification = self._classify_exception(e, message=message, exception_type=exception_type)
        if classification["error_code"] == "missing_prompt_embeddings":
            message = "Prompt embeddings are missing from Encode Prompt. Update or recreate the Studio graph after node definitions finish refreshing."
        elif classification.get("message"):
            message = classification["message"]
        if classification["category"] == "custom_pipeline" and classification.get("recovery_hint"):
            message = f"{message} {classification['recovery_hint']}"
        oom = classification["category"] == "oom"
        memory_summary = None
        if oom:
            memory_summary = message.split("\n")[0]

        payload = {
            "task_id": task_id,
            "sid": sid,
            "node": node_id,
            "node_name": node_name,
            "message": message,
            "exception_type": exception_type,
            "traceback": traceback_text,
            "category": classification["category"],
            "error_code": classification["error_code"],
            "recovery_hint": classification["recovery_hint"],
            "oom": oom,
            "memory_summary": memory_summary,
            "cuda_memory_snapshot": self._cuda_memory_snapshot(),
            "gpu_processes": self._gpu_process_snapshot(),
            "runtime_hints": self.current_task.get("runtimeHints") if self.current_task else None,
            "runtime_budget": self.current_task.get("runtimeBudget") if self.current_task else None,
            "loader_diagnostics": self._loader_diagnostics_snapshot(),
        }
        return payload

    def _classify_exception(self, e, message=None, exception_type=None):
        exception_type = exception_type or type(e).__name__
        message = message or str(e) or exception_type
        explicit_error_code = getattr(e, "modiff_error_code", None)
        if explicit_error_code:
            return {
                "category": getattr(e, "modiff_category", "runtime"),
                "error_code": explicit_error_code,
                "message": message,
                "recovery_hint": getattr(e, "modiff_recovery_hint", None),
            }
        chain = self._exception_chain(e)
        chain_text = " | ".join(f"{type(item).__name__} {str(item) or type(item).__name__}" for item in chain)
        normalized = f"{exception_type} {message} {chain_text}".lower()

        if (
            "outofmemory" in normalized
            or "out of memory" in normalized
            or "cuda out of memory" in normalized
            or "cublas_status_alloc_failed" in normalized
            or "cusolver_status_alloc_failed" in normalized
        ):
            return {
                "category": "oom",
                "error_code": "cuda_oom",
                "message": next(
                    (
                        str(item)
                        for item in reversed(chain)
                        if "out of memory" in (str(item) or "").lower() or "outofmemory" in type(item).__name__.lower()
                    ),
                    message,
                ),
                "recovery_hint": "Release accelerator cache, close other accelerator-heavy apps, apply the Low VRAM preset, or switch to a smaller compatible model.",
            }

        if any(isinstance(item, MissingConnectedOutputError) for item in chain):
            return {
                "category": "graph_incomplete",
                "error_code": "missing_connected_output",
                "recovery_hint": "An upstream node did not produce a connected output. Update or recreate the graph; if this followed a loader failure, inspect loader diagnostics.",
            }

        if "illegal memory access" in normalized or "cudaerrorillegaladdress" in normalized:
            return {
                "category": "cuda_context",
                "error_code": "cuda_context_poisoned",
                "message": next(
                    (str(item) for item in reversed(chain) if "illegal memory access" in (str(item) or "").lower()),
                    message,
                ),
                "recovery_hint": "CUDA reported an illegal memory access. Stop this run, restart the backend process, and retry with a safer execution plan so the current Python CUDA context is not reused.",
            }

        if "cublas_status_not_supported" in normalized or "cublasltmatmulalgogetheuristic" in normalized:
            return {
                "category": "cuda_kernel",
                "error_code": "cuda_kernel_unsupported",
                "message": next(
                    (str(item) for item in reversed(chain) if "cublas" in (str(item) or "").lower()), message
                ),
                "recovery_hint": "The quantized CUDA kernel used by this model path is not supported by the current PyTorch/bitsandbytes/CUDA combination. Try a non-quantized smaller model path, update the CUDA/PyTorch/bitsandbytes stack, or use a backend path that provides compatible Qwen weights.",
            }

        if (
            (isinstance(e, KeyError) and str(e).strip("'\"") == "embeddings")
            or "keyerror 'embeddings'" in normalized
            or 'keyerror "embeddings"' in normalized
        ):
            return {
                "category": "graph_incomplete",
                "error_code": "missing_prompt_embeddings",
                "recovery_hint": "Prompt embeddings were not ready or connected. Update or recreate the Studio graph after node definitions finish refreshing, then retry.",
            }

        if (
            isinstance(e, (ModuleNotFoundError, ImportError))
            or "no module named" in normalized
            or "cannot import name" in normalized
        ):
            return {
                "category": "missing_dependency",
                "error_code": "missing_dependency",
                "recovery_hint": "Install or repair the missing Python package, then restart the backend.",
            }

        if (
            isinstance(e, FileNotFoundError)
            or "model not found" in normalized
            or "missing model" in normalized
            or "no such file or directory" in normalized
            or "localentrynotfound" in normalized
            or "entrynotfound" in normalized
            or "repo not found" in normalized
            or "repository not found" in normalized
        ):
            return {
                "category": "missing_model",
                "error_code": "missing_model",
                "recovery_hint": "Open Setup, refresh model indexes, then install or relink the missing model package.",
            }

        if (
            isinstance(e, ConnectionError)
            or "connection refused" in normalized
            or "connection reset" in normalized
            or "backend unavailable" in normalized
            or "server disconnected" in normalized
        ):
            return {
                "category": "backend_unavailable",
                "error_code": "backend_unavailable",
                "recovery_hint": "Check that the backend is still running, then retry the workflow.",
            }

        if isinstance(e, PermissionError) or "permission denied" in normalized or "access is denied" in normalized:
            return {
                "category": "permission",
                "error_code": "permission_denied",
                "recovery_hint": "Check file permissions and whether another process is locking the target path.",
            }

        if isinstance(e, asyncio.CancelledError) or "cancelled" in normalized or "interrupted" in normalized:
            return {
                "category": "interrupted",
                "error_code": "run_interrupted",
                "recovery_hint": "The run was interrupted. Retry when the backend queue is idle.",
            }

        if any(isinstance(item, (ValueError, TypeError)) for item in chain):
            return {
                "category": "input_validation",
                "error_code": "invalid_node_input",
                "message": next(
                    (str(item) for item in reversed(chain) if isinstance(item, (ValueError, TypeError)) and str(item)),
                    message,
                ),
                "recovery_hint": "Correct the referenced prompt, dimensions, mode, or node input and retry. The installed model remains runnable.",
            }

        return {
            "category": "runtime_error",
            "error_code": "runtime_error",
            "recovery_hint": "Review the run details, fix the referenced node or input, and retry.",
        }

    def _runtime_fingerprint(self):
        packages = {
            "python": sys.version.split(" ")[0],
            "platform": platform.platform(),
        }
        for package_name in ("diffusers", "transformers", "accelerate", "bitsandbytes"):
            try:
                packages[package_name] = metadata.version(package_name)
            except Exception:
                packages[package_name] = None
        try:
            # Runtime proof must observe settings applied immediately before
            # execution. The normal hardware snapshot cache can otherwise retain
            # pre-run deterministic flags and make identical duplicate runs look
            # like different runtimes.
            hardware = get_hardware_snapshot(self.data_dir, refresh=True)
            torch_metadata = hardware.get("torch") if isinstance(hardware.get("torch"), dict) else {}
            legacy_status = legacy_torch_status(hardware)
            if torch_metadata.get("available"):
                packages["torch"] = torch_metadata.get("version")
                torch_state = {
                    "cuda_available": legacy_status.get("cuda_available", False),
                    "cuda_device_count": legacy_status.get("cuda_device_count", 0),
                    "cuda_device_name": legacy_status.get("cuda_device_name"),
                    "xpu_available": legacy_status.get("xpu_available", False),
                    "xpu_device_count": legacy_status.get("xpu_device_count", 0),
                    "xpu_devices": legacy_status.get("xpu_devices"),
                    "cudnn_version": torch_metadata.get("cudnn_version"),
                    "cudnn_deterministic": torch_metadata.get("cudnn_deterministic"),
                    "cudnn_benchmark": torch_metadata.get("cudnn_benchmark"),
                    "deterministic_algorithms": torch_metadata.get("deterministic_algorithms"),
                }
            else:
                errors = torch_metadata.get("errors") if isinstance(torch_metadata.get("errors"), dict) else {}
                torch_state = {"error": errors.get("import") or "torch is unavailable"}

            if torch_state.get("cuda_available") and torch_state.get("cuda_device_count", 0) > 0:
                torch_state.update(
                    {
                        "cuda_device_total_memory_bytes": legacy_status.get("cuda_device_total_memory_bytes"),
                        "cuda_memory_free_bytes": legacy_status.get("cuda_memory_free_bytes"),
                        "cuda_memory_total_bytes": legacy_status.get("cuda_memory_total_bytes"),
                    }
                )
                try:
                    torch = import_module("torch")
                    capability = torch.cuda.get_device_capability(0)
                    torch_state["cuda_device_capability"] = ".".join(str(item) for item in capability)
                except Exception as capability_error:
                    torch_state["cuda_device_capability_error"] = str(capability_error)
        except Exception as hardware_error:
            hardware = {"error": str(hardware_error)}
            torch_state = {"error": str(hardware_error)}

        returned_payload = {
            "packages": packages,
            "torch": torch_state,
            "work_dir": str(self.work_dir),
            "data_dir": str(self.data_dir),
        }
        execution_torch_identity = {
            key: value for key, value in torch_state.items() if key not in {"cuda_memory_free_bytes"}
        }
        resource_torch_identity = {
            key: value
            for key, value in execution_torch_identity.items()
            if key
            not in {
                "cudnn_deterministic",
                "cudnn_benchmark",
                "deterministic_algorithms",
            }
        }
        execution_identity = {
            **returned_payload,
            "torch": execution_torch_identity,
        }
        resource_identity = {
            **returned_payload,
            "torch": resource_torch_identity,
        }
        fingerprint = hashlib.sha256(
            json.dumps(execution_identity, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        resource_fingerprint = hashlib.sha256(
            json.dumps(resource_identity, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        result = {
            "fingerprint": f"sha256:{fingerprint}",
            "resourceFingerprint": f"sha256:{resource_fingerprint}",
            **returned_payload,
            "hardware": hardware,
        }
        self._last_runtime_fingerprint = deepcopy(result)
        return result

    def _extract_graph_seed(self, graph, deterministic_options):
        seed = deterministic_options.get("seed") if isinstance(deterministic_options, dict) else None
        if seed is not None:
            try:
                return int(seed)
            except (TypeError, ValueError):
                return None

        for node in graph.get("nodes", {}).values():
            if not isinstance(node, dict):
                continue
            params = node.get("params", {})
            if not isinstance(params, dict):
                continue
            for key, param in params.items():
                if key != "seed" or not isinstance(param, dict):
                    continue
                value = param.get("value")
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue

        return None

    def _deterministic_warnings(self, graph):
        warnings = []
        for node_id, node in graph.get("nodes", {}).items():
            if not isinstance(node, dict):
                continue
            params = node.get("params", {})
            if not isinstance(params, dict):
                continue
            for key, param in params.items():
                if not isinstance(param, dict):
                    continue
                if param.get("display") == "random":
                    warnings.append(f"{node_id}.{key} is still marked random in the API graph.")
                if key == "seed" and param.get("value") in (None, "", -1):
                    warnings.append(f"{node_id}.seed is not locked.")
        return warnings

    def _apply_deterministic_mode(self, graph):
        options = graph.get("deterministicMode")
        if options is True:
            options = {"enabled": True}
        if not isinstance(options, dict) or not options.get("enabled"):
            return None

        seed = self._extract_graph_seed(graph, options)
        applied = {
            "enabled": True,
            "seed": seed,
            "strict": bool(options.get("strict", True)),
            "warnings": self._deterministic_warnings(graph),
            "settings": {
                "python_random": False,
                "numpy_random": False,
                "torch_manual_seed": False,
                "torch_cuda_manual_seed_all": False,
                "torch_deterministic_algorithms": False,
                "cudnn_benchmark": None,
                "cudnn_deterministic": None,
                "allow_tf32": None,
            },
        }

        if seed is None:
            if applied["strict"]:
                raise ValueError("Strict deterministic execution requires a fixed seed.")
            applied["warnings"].append("No fixed seed found for deterministic execution.")
        else:
            os.environ["PYTHONHASHSEED"] = str(seed)
            random.seed(seed)
            applied["settings"]["python_random"] = True

            try:
                import numpy as np

                np.random.seed(seed % (2**32))
                applied["settings"]["numpy_random"] = True
            except Exception as e:
                if applied["strict"]:
                    raise RuntimeError(f"Strict deterministic mode could not seed NumPy: {e}") from e
                applied["warnings"].append(f"NumPy seed was not applied: {e}")

            try:
                import torch

                torch.manual_seed(seed)
                applied["settings"]["torch_manual_seed"] = True
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                    applied["settings"]["torch_cuda_manual_seed_all"] = True
                if applied["strict"]:
                    if not hasattr(torch, "use_deterministic_algorithms"):
                        raise RuntimeError("This Torch build does not expose deterministic algorithm enforcement.")
                    torch.use_deterministic_algorithms(True, warn_only=False)
                    applied["settings"]["torch_deterministic_algorithms"] = True
                if applied["strict"] and hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                    applied["settings"]["cudnn_benchmark"] = False
                    applied["settings"]["cudnn_deterministic"] = True
                if applied["strict"] and hasattr(torch.backends, "cuda"):
                    torch.backends.cuda.matmul.allow_tf32 = False
                    applied["settings"]["allow_tf32"] = False
                if applied["strict"] and hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.allow_tf32 = False
            except Exception as e:
                if applied["strict"]:
                    raise RuntimeError(f"Strict deterministic Torch settings could not be applied: {e}") from e
                applied["warnings"].append(f"Torch deterministic settings were not fully applied: {e}")

        return applied

    @staticmethod
    def _bounded_auto_value(value, *, field_name, depth, budget):
        if depth > 8:
            raise ValueError("nested value is too deep")
        budget["entries"] += 1
        if budget["entries"] > 4096:
            raise ValueError("too many nested values")
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("numeric value is not finite")
            return value
        if isinstance(value, str):
            limit = 512 if field_name in {
                "id",
                "candidateId",
                "modelType",
                "mode",
                "loaderModule",
                "loaderAction",
                "executionPath",
                "pipelineClass",
                "artifact",
                "baseArtifact",
                "modelRepo",
                "resolvedArtifact",
                "repo",
                "revision",
            } else 4096
            if len(value) > limit:
                raise ValueError("string value is too long")
            budget["chars"] += len(value)
            if budget["chars"] > 65536:
                raise ValueError("too much string data")
            return value
        if isinstance(value, list):
            if len(value) > 32:
                raise ValueError("array is too large")
            return [
                WebServer._bounded_auto_value(
                    item,
                    field_name=field_name,
                    depth=depth + 1,
                    budget=budget,
                )
                for item in value
            ]
        if isinstance(value, dict):
            if len(value) > 128:
                raise ValueError("object is too large")
            output = {}
            for key, item in value.items():
                if not isinstance(key, str) or len(key) > 128:
                    raise ValueError("object key is invalid")
                output[key] = WebServer._bounded_auto_value(
                    item,
                    field_name=key,
                    depth=depth + 1,
                    budget=budget,
                )
            return output
        raise ValueError("value is not JSON-compatible")

    @staticmethod
    def _project_bounded_auto_mapping(value, *, retry=False):
        if not isinstance(value, dict):
            raise WebServer._auto_resource_contract_error(
                "Auto candidate data is malformed. Refresh Auto before running this workflow.",
                code="auto_resource_candidate_mismatch",
            )
        allowed = {"id", *WebServer._auto_candidate_execution_fields()}
        if retry:
            allowed.update({"candidateId", "index", "reason", "onCategories", "onErrorCodes"})
        budget = {"entries": 0, "chars": 0}
        try:
            projected = {
                key: WebServer._bounded_auto_value(
                    value[key],
                    field_name=key,
                    depth=0,
                    budget=budget,
                )
                for key in allowed
                if key in value
            }
            if len(json.dumps(projected, ensure_ascii=False, separators=(",", ":"))) > 65536:
                raise ValueError("mapping is too large")
        except (OverflowError, RecursionError, TypeError, ValueError):
            raise WebServer._auto_resource_contract_error(
                "Auto candidate data exceeds the supported execution contract. Refresh Auto before running this workflow.",
                code="auto_resource_candidate_mismatch",
            ) from None
        return projected

    @staticmethod
    def _bounded_runtime_container_value(
        value,
        *,
        depth,
        budget,
        max_depth,
        max_entries,
        max_items,
        max_keys,
        max_string_chars,
        max_total_chars,
    ):
        if depth > max_depth:
            raise ValueError("nested value is too deep")
        budget["entries"] += 1
        if budget["entries"] > max_entries:
            raise ValueError("too many nested values")
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("numeric value is not finite")
            return value
        if isinstance(value, str):
            if len(value) > max_string_chars:
                raise ValueError("string value is too long")
            budget["chars"] += len(value)
            if budget["chars"] > max_total_chars:
                raise ValueError("too much string data")
            return value
        if isinstance(value, list):
            if len(value) > max_items:
                raise ValueError("array is too large")
            return [
                WebServer._bounded_runtime_container_value(
                    item,
                    depth=depth + 1,
                    budget=budget,
                    max_depth=max_depth,
                    max_entries=max_entries,
                    max_items=max_items,
                    max_keys=max_keys,
                    max_string_chars=max_string_chars,
                    max_total_chars=max_total_chars,
                )
                for item in value
            ]
        if isinstance(value, dict):
            if len(value) > max_keys:
                raise ValueError("object is too large")
            output = {}
            for key, item in value.items():
                if not isinstance(key, str) or len(key) > 256:
                    raise ValueError("object key is invalid")
                budget["chars"] += len(key)
                if budget["chars"] > max_total_chars:
                    raise ValueError("too much string data")
                output[key] = WebServer._bounded_runtime_container_value(
                    item,
                    depth=depth + 1,
                    budget=budget,
                    max_depth=max_depth,
                    max_entries=max_entries,
                    max_items=max_items,
                    max_keys=max_keys,
                    max_string_chars=max_string_chars,
                    max_total_chars=max_total_chars,
                )
            return output
        raise ValueError("value is not JSON-compatible")

    @staticmethod
    def _project_bounded_runtime_container(
        value,
        *,
        expected_type,
        max_depth=8,
        max_entries=4096,
        max_items=256,
        max_keys=256,
        max_string_chars=65536,
        max_total_chars=262144,
        max_serialized_chars=262144,
    ):
        if not isinstance(value, expected_type):
            raise WebServer._auto_resource_contract_error(
                "Runtime hints contain malformed structured data. Refresh the workflow before running it.",
                code="auto_resource_candidate_mismatch",
            )
        budget = {"entries": 0, "chars": 0}
        try:
            projected = WebServer._bounded_runtime_container_value(
                value,
                depth=0,
                budget=budget,
                max_depth=max_depth,
                max_entries=max_entries,
                max_items=max_items,
                max_keys=max_keys,
                max_string_chars=max_string_chars,
                max_total_chars=max_total_chars,
            )
            if len(json.dumps(projected, ensure_ascii=False, separators=(",", ":"))) > max_serialized_chars:
                raise ValueError("structured value is too large")
        except (OverflowError, RecursionError, TypeError, ValueError):
            raise WebServer._auto_resource_contract_error(
                "Runtime hints exceed the supported structured-data contract. Refresh the workflow before running it.",
                code="auto_resource_candidate_mismatch",
            ) from None
        return projected

    def _coerce_runtime_hints(self, value):
        if not isinstance(value, dict):
            return None

        allowed = {
            "source",
            "device",
            "cudaIndex",
            "cudaMemoryFreeBytes",
            "cudaMemoryTotalBytes",
            "modelType",
            "mode",
            "modelRepo",
            "modelName",
            "resolvedModelRepo",
            "resolvedArtifact",
            "modelDependencies",
            "loaderModule",
            "loaderAction",
            "executionPath",
            "pipelineClass",
            "dtype",
            "resourceMode",
            "resolvedResourceMode",
            "quantizationMode",
            "quantizedComponents",
            "autoOffload",
            "offloadMode",
            "deviceMap",
            "attentionBackend",
            "regionalCompile",
            "denoiserCache",
            "channelsLast",
            "layerwiseCasting",
            "offloadDiskPath",
            "supportedOffloadModes",
            "resourcePlan",
            "autoResourcePlan",
            "autoResourceCandidates",
            "autoResourceProofStatus",
            "autoResourceCandidateId",
            "resourceRetryModes",
            "resourceRetryPlans",
            "resourceRetryAttempt",
            "resourceRetryHistory",
            "resourceRetryLastError",
            "resourceRetryLastCode",
            "cudaBudgetPolicy",
            "enforceCudaBudget",
            "compatibilityProbe",
            "compatibilityStatus",
            "requestedCudaReserveBytes",
            "requestedCudaBudgetBytes",
            "clientRunId",
            "runInputHash",
            "workflowTabId",
            "workflowCanvasEpoch",
            "workflowFormEpoch",
            "workflowTitle",
            "workflowSnapshot",
            "nodeId",
            "maxRuntimeSeconds",
            "autoFieldOverrides",
            "optimizationQualificationForm",
            "studioExecutionSpec",
        }
        hints = {key: value.get(key) for key in allowed if key in value}

        for key in (
            "source",
            "device",
            "modelType",
            "mode",
            "modelRepo",
            "modelName",
            "resolvedModelRepo",
            "resolvedArtifact",
            "loaderModule",
            "loaderAction",
            "executionPath",
            "pipelineClass",
            "dtype",
            "resourceMode",
            "resolvedResourceMode",
            "quantizationMode",
            "offloadMode",
            "offloadDiskPath",
            "cudaBudgetPolicy",
            "compatibilityStatus",
            "autoResourceProofStatus",
            "autoResourceCandidateId",
            "clientRunId",
            "runInputHash",
            "workflowTabId",
            "nodeId",
        ):
            if key in hints and hints[key] is not None and not isinstance(hints[key], str):
                raw_value = hints[key]
                if (
                    not isinstance(raw_value, (bool, int, float))
                    or isinstance(raw_value, float) and not math.isfinite(raw_value)
                ):
                    raise self._auto_resource_contract_error(
                        "Runtime hint identity has an invalid primitive shape. Refresh the workflow before running it.",
                        code="auto_resource_candidate_mismatch",
                    )
                hints[key] = str(raw_value)
            if key in hints and isinstance(hints[key], str) and len(hints[key]) > 512:
                raise self._auto_resource_contract_error(
                    "Auto runtime identity exceeds the supported execution contract. "
                    "Refresh Auto before running this workflow.",
                    code="auto_resource_candidate_mismatch",
                )

        enum_fields = {
            "deviceMap": RUNTIME_DEVICE_MAPS,
            "attentionBackend": RUNTIME_ATTENTION_BACKENDS,
            "denoiserCache": RUNTIME_DENOISER_CACHE_MODES,
        }
        for key, admitted_values in enum_fields.items():
            if key not in hints or hints[key] is None:
                hints.pop(key, None)
                continue
            if not isinstance(hints[key], str) or hints[key] not in admitted_values:
                raise self._auto_resource_contract_error(
                    "Runtime hint execution selector is unsupported. Refresh the workflow before running it.",
                    code="auto_resource_candidate_mismatch",
                )

        for key in (
            "autoOffload",
            "enforceCudaBudget",
            "regionalCompile",
            "channelsLast",
            "layerwiseCasting",
        ):
            if key not in hints or hints[key] is None:
                hints.pop(key, None)
                continue
            if not isinstance(hints[key], bool):
                raise self._auto_resource_contract_error(
                    "Runtime hint flag has an invalid primitive shape. Refresh the workflow before running it.",
                    code="auto_resource_candidate_mismatch",
                )

        workflow_canvas_epoch = hints.get("workflowCanvasEpoch")
        if workflow_canvas_epoch is not None and (
            isinstance(workflow_canvas_epoch, bool)
            or not isinstance(workflow_canvas_epoch, int)
            or workflow_canvas_epoch < 0
            or workflow_canvas_epoch > 9_007_199_254_740_991
        ):
            hints.pop("workflowCanvasEpoch", None)

        workflow_form_epoch = hints.get("workflowFormEpoch")
        if workflow_form_epoch is not None and (
            isinstance(workflow_form_epoch, bool)
            or not isinstance(workflow_form_epoch, int)
            or workflow_form_epoch < 0
            or workflow_form_epoch > 9_007_199_254_740_991
        ):
            hints.pop("workflowFormEpoch", None)

        known_offload_modes = {
            OFFLOAD_MODE_NONE,
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_SEQUENTIAL_CPU,
            OFFLOAD_MODE_GROUP_CPU,
            OFFLOAD_MODE_GROUP_DISK,
        }
        for key in ("quantizedComponents", "supportedOffloadModes", "resourceRetryModes"):
            if key in hints and hints[key] is not None:
                if isinstance(hints[key], list):
                    if len(hints[key]) > 32:
                        raise self._auto_resource_contract_error(
                            "Runtime hint list exceeds the supported execution contract. Refresh the workflow before running it.",
                            code="auto_resource_candidate_mismatch",
                        )
                    normalized = [
                        item
                        for item in hints[key]
                        if isinstance(item, str) and len(item) <= 128
                    ]
                    if key != "quantizedComponents":
                        normalized = [item for item in normalized if item in known_offload_modes]
                    hints[key] = normalized
                else:
                    hints.pop(key, None)

        if "modelDependencies" in hints and hints["modelDependencies"] is not None:
            dependencies = self._model_dependencies_signature(hints["modelDependencies"])
            if dependencies is None:
                raise self._auto_resource_contract_error(
                    "Model dependency receipt is malformed. Refresh Auto before running this workflow.",
                    code="auto_resource_candidate_mismatch",
                )
            hints["modelDependencies"] = dependencies

        if (
            "resourcePlan" in hints
            and hints["resourcePlan"] is not None
            and not isinstance(hints["resourcePlan"], dict)
        ):
            hints.pop("resourcePlan", None)
        elif isinstance(hints.get("resourcePlan"), dict):
            hints["resourcePlan"] = self._project_bounded_runtime_container(
                hints["resourcePlan"],
                expected_type=dict,
                max_depth=8,
                max_entries=4096,
                max_items=64,
                max_keys=128,
                max_string_chars=4096,
                max_total_chars=65536,
                max_serialized_chars=65536,
            )
            # Active retry state is produced only by this worker after a
            # failed attempt; submitted copies are not execution authority.
            hints["resourcePlan"].pop("activeRetryPlan", None)

        if "studioExecutionSpec" in hints:
            receipt = hints["studioExecutionSpec"]
            if not isinstance(receipt, dict):
                raise self._auto_resource_contract_error(
                    "Studio execution specification receipt is malformed. Rebuild the managed graph.",
                    code="studio_execution_spec_mismatch",
                )
            allowed_receipt_keys = {"schemaVersion", "id", "contentHash", "nodes"}
            if set(receipt) != allowed_receipt_keys or not isinstance(receipt.get("nodes"), dict):
                raise self._auto_resource_contract_error(
                    "Studio execution specification receipt is malformed. Rebuild the managed graph.",
                    code="studio_execution_spec_mismatch",
                )
            nodes = receipt["nodes"]
            if (
                receipt.get("schemaVersion") != 1
                or not isinstance(receipt.get("id"), str)
                or len(receipt["id"]) > 128
                or not isinstance(receipt.get("contentHash"), str)
                or re.fullmatch(r"studio-spec-v1-[0-9a-f]{8}", receipt["contentHash"]) is None
                or len(nodes) > 32
                or any(
                    not isinstance(role, str)
                    or not role
                    or len(role) > 64
                    or not isinstance(node_id, str)
                    or not node_id
                    or len(node_id) > 128
                    for role, node_id in nodes.items()
                )
            ):
                raise self._auto_resource_contract_error(
                    "Studio execution specification receipt is invalid. Rebuild the managed graph.",
                    code="studio_execution_spec_mismatch",
                )
            hints["studioExecutionSpec"] = {
                key: deepcopy(receipt[key]) for key in ("schemaVersion", "id", "contentHash", "nodes")
            }
            specification = studio_execution_spec_for_pair(
                str(hints.get("modelType") or ""),
                str(hints.get("mode") or ""),
            )
            if (
                specification is None
                or receipt["id"] != specification["id"]
                or receipt["contentHash"] != specification["contentHash"]
                or set(nodes) != {role[0] for role in specification["roles"]}
            ):
                raise self._auto_resource_contract_error(
                    "Studio execution specification receipt does not match this workflow. Rebuild the managed graph.",
                    code="studio_execution_spec_mismatch",
                )

        if (
            "autoResourcePlan" in hints
            and hints["autoResourcePlan"] is not None
            and not isinstance(hints["autoResourcePlan"], dict)
        ):
            hints.pop("autoResourcePlan", None)
        elif isinstance(hints.get("autoResourcePlan"), dict):
            hints["autoResourcePlan"] = self._project_bounded_auto_mapping(hints["autoResourcePlan"])

        if (
            "autoResourceCandidates" in hints
            and hints["autoResourceCandidates"] is not None
            and not isinstance(hints["autoResourceCandidates"], list)
        ):
            hints.pop("autoResourceCandidates", None)
        elif isinstance(hints.get("autoResourceCandidates"), list):
            if len(hints["autoResourceCandidates"]) > 64:
                raise self._auto_resource_contract_error(
                    "Auto candidate list exceeds the supported execution contract. Refresh Auto before running this workflow.",
                    code="auto_resource_candidate_mismatch",
                )
            if any(not isinstance(candidate, dict) for candidate in hints["autoResourceCandidates"]):
                raise self._auto_resource_contract_error(
                    "Auto candidate list is malformed. Refresh Auto before running this workflow.",
                    code="auto_resource_candidate_mismatch",
                )
            hints["autoResourceCandidates"] = [
                self._project_bounded_auto_mapping(candidate)
                for candidate in hints["autoResourceCandidates"]
                if isinstance(candidate, dict)
            ]
            if len(json.dumps(hints["autoResourceCandidates"], ensure_ascii=False)) > 1_048_576:
                raise self._auto_resource_contract_error(
                    "Auto candidate list exceeds the supported execution contract. Refresh Auto before running this workflow.",
                    code="auto_resource_candidate_mismatch",
                )

        # Retry state is derived by the worker. Submitted copies must never
        # influence a new run or survive into queue/completion payloads.
        for key in (
            "resourceRetryAttempt",
            "resourceRetryHistory",
            "resourceRetryLastError",
            "resourceRetryLastCode",
        ):
            hints.pop(key, None)

        if "resourceRetryPlans" in hints and hints["resourceRetryPlans"] is not None:
            if isinstance(hints["resourceRetryPlans"], list):
                if len(hints["resourceRetryPlans"]) > 32:
                    raise self._auto_resource_contract_error(
                        "Auto retry list exceeds the supported execution contract. Refresh Auto before running this workflow.",
                        code="auto_resource_candidate_mismatch",
                    )
                if any(not isinstance(item, dict) for item in hints["resourceRetryPlans"]):
                    raise self._auto_resource_contract_error(
                        "Auto retry list is malformed. Refresh Auto before running this workflow.",
                        code="auto_resource_candidate_mismatch",
                    )
                plans = []
                for item in hints["resourceRetryPlans"]:
                    if isinstance(item, dict):
                        plans.append(self._project_bounded_auto_mapping(item, retry=True))
                hints["resourceRetryPlans"] = plans
            else:
                hints.pop("resourceRetryPlans", None)

        if (
            "compatibilityProbe" in hints
            and hints["compatibilityProbe"] is not None
            and not isinstance(hints["compatibilityProbe"], dict)
        ):
            hints.pop("compatibilityProbe", None)
        elif isinstance(hints.get("compatibilityProbe"), dict):
            hints["compatibilityProbe"] = self._project_bounded_runtime_container(
                hints["compatibilityProbe"],
                expected_type=dict,
                max_depth=8,
                max_entries=4096,
                max_items=256,
                max_keys=256,
                max_string_chars=4096,
                max_total_chars=262144,
                max_serialized_chars=262144,
            )

        for key in (
            "cudaIndex",
            "cudaMemoryFreeBytes",
            "cudaMemoryTotalBytes",
            "requestedCudaReserveBytes",
            "requestedCudaBudgetBytes",
            "resourceRetryAttempt",
            "maxRuntimeSeconds",
        ):
            if key in hints and hints[key] is not None:
                try:
                    hints[key] = int(hints[key])
                except (TypeError, ValueError):
                    hints.pop(key, None)
        if "workflowTitle" in hints:
            workflow_title = hints["workflowTitle"]
            if workflow_title is None:
                hints.pop("workflowTitle", None)
            elif not isinstance(workflow_title, (str, bool, int, float)) or (
                isinstance(workflow_title, float) and not math.isfinite(workflow_title)
            ):
                raise self._auto_resource_contract_error(
                    "Workflow title has an invalid primitive shape. Refresh the workflow before running it.",
                    code="auto_resource_candidate_mismatch",
                )
            else:
                hints["workflowTitle"] = str(workflow_title).strip()[:256]
                if not hints["workflowTitle"]:
                    hints.pop("workflowTitle", None)
        if "workflowSnapshot" in hints:
            workflow_snapshot = hints["workflowSnapshot"]
            if isinstance(workflow_snapshot, dict):
                hints["workflowSnapshot"] = self._project_bounded_runtime_container(
                    workflow_snapshot,
                    expected_type=dict,
                    max_depth=32,
                    max_entries=200000,
                    max_items=20000,
                    max_keys=20000,
                    max_string_chars=1_048_576,
                    max_total_chars=8_388_608,
                    max_serialized_chars=8_388_608,
                )
            else:
                hints.pop("workflowSnapshot", None)
        if "autoFieldOverrides" in hints:
            overrides = hints["autoFieldOverrides"]
            if isinstance(overrides, list):
                if len(overrides) > 256:
                    raise self._auto_resource_contract_error(
                        "Auto field override list exceeds the supported execution contract. Refresh the workflow before running it.",
                        code="auto_resource_candidate_mismatch",
                    )
                projected_overrides = []
                for item in overrides:
                    if (
                        not isinstance(item, dict)
                        or not isinstance(item.get("nodeId"), str)
                        or not isinstance(item.get("fieldKey"), str)
                        or len(item["nodeId"]) > 512
                        or len(item["fieldKey"]) > 512
                    ):
                        raise self._auto_resource_contract_error(
                            "Auto field override identity is malformed. Refresh the workflow before running it.",
                            code="auto_resource_candidate_mismatch",
                        )
                    projected_overrides.append(
                        self._project_bounded_runtime_container(
                            {
                                key: item[key]
                                for key in ("schemaVersion", "nodeId", "fieldKey", "formKey", "value", "updatedAt")
                                if key in item
                            },
                            expected_type=dict,
                            max_depth=8,
                            max_entries=4096,
                            max_items=64,
                            max_keys=16,
                            max_string_chars=4096,
                            max_total_chars=65536,
                            max_serialized_chars=65536,
                        )
                    )
                if len(json.dumps(projected_overrides, ensure_ascii=False)) > 1_048_576:
                    raise self._auto_resource_contract_error(
                        "Auto field override data exceeds the supported execution contract. Refresh the workflow before running it.",
                        code="auto_resource_candidate_mismatch",
                    )
                hints["autoFieldOverrides"] = projected_overrides
            else:
                hints.pop("autoFieldOverrides", None)
        if "optimizationQualificationForm" in hints:
            form = hints["optimizationQualificationForm"]
            if isinstance(form, dict):
                hints["optimizationQualificationForm"] = self._project_bounded_runtime_container(
                    form,
                    expected_type=dict,
                    max_depth=12,
                    max_entries=16384,
                    max_items=512,
                    max_keys=512,
                    max_string_chars=65536,
                    max_total_chars=524288,
                    max_serialized_chars=524288,
                )
            else:
                hints.pop("optimizationQualificationForm", None)
        if "maxRuntimeSeconds" in hints:
            # Quality-first local video models can legitimately need more than
            # six hours at their upstream-recommended step count. Keep a hard
            # safety ceiling, but do not force users to reduce sampling quality
            # merely to fit the old gallery-oriented limit.
            hints["maxRuntimeSeconds"] = max(60, min(43200, hints["maxRuntimeSeconds"]))

        if hints.get("cudaBudgetPolicy") not in (None, "advisory", "enforced"):
            hints.pop("cudaBudgetPolicy", None)

        if isinstance(hints.get("resourceRetryPlans"), list):
            canonical_plans = self._coerce_retry_plan_list(hints)
            hints["resourceRetryPlans"] = [
                self._sanitize_retry_plan_for_hints(plan)
                for plan in canonical_plans
            ]

        return hints

    def _cuda_index_from_runtime_hints(self, hints):
        if not hints:
            return None
        if isinstance(hints.get("cudaIndex"), int):
            return hints.get("cudaIndex")

        device = str(hints.get("device") or "").strip().lower()
        match = re.match(r"^cuda(?::(\d+))?$", device)
        if not match:
            return None
        return int(match.group(1) or 0)

    def _apply_cuda_runtime_budget(self, runtime_hints):
        result = {
            "applied": False,
            "reason": "No CUDA runtime hints were provided.",
        }
        cuda_index = self._cuda_index_from_runtime_hints(runtime_hints)
        if cuda_index is None:
            if runtime_hints:
                result["reason"] = "Runtime hints did not target a CUDA device."
            return result

        try:
            torch = import_module("torch")
        except Exception as e:
            return {
                **result,
                "reason": f"torch import failed: {e}",
                "cuda_index": cuda_index,
            }

        if not torch.cuda.is_available():
            return {
                **result,
                "reason": "CUDA is not available in this backend process.",
                "cuda_index": cuda_index,
            }

        device_count = int(torch.cuda.device_count())
        if cuda_index < 0 or cuda_index >= device_count:
            return {
                **result,
                "reason": f"CUDA device {cuda_index} is not available.",
                "cuda_index": cuda_index,
                "device_count": device_count,
            }

        enforce_budget = (runtime_hints or {}).get("enforceCudaBudget") is True or (runtime_hints or {}).get(
            "cudaBudgetPolicy"
        ) == "enforced"
        if not enforce_budget:
            try:
                torch.cuda.set_per_process_memory_fraction(1.0, cuda_index)
                reset_reason = "CUDA budget is advisory; reset PyTorch process memory fraction to full device."
            except Exception as e:
                reset_reason = f"CUDA budget is advisory; could not reset PyTorch process memory fraction: {e}"
            return {
                **result,
                "reason": reset_reason,
                "cuda_index": cuda_index,
                "cuda_budget_policy": "advisory",
                "fraction": 1.0,
                "model_repo": runtime_hints.get("modelRepo") if runtime_hints else None,
                "model_name": runtime_hints.get("modelName") if runtime_hints else None,
                "execution_path": runtime_hints.get("executionPath") if runtime_hints else None,
                "offload_mode": runtime_hints.get("offloadMode") if runtime_hints else None,
            }

        try:
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(cuda_index)
            except TypeError:
                with torch.cuda.device(cuda_index):
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_bytes = int(free_bytes)
            total_bytes = int(total_bytes)
        except Exception as e:
            return {
                **result,
                "reason": f"CUDA memory info unavailable: {e}",
                "cuda_index": cuda_index,
            }

        gib = 1024**3
        requested_reserve = runtime_hints.get("requestedCudaReserveBytes") if runtime_hints else None
        reserve_bytes = (
            requested_reserve
            if isinstance(requested_reserve, int) and requested_reserve > 0
            else max(gib, int(total_bytes * 0.1))
        )
        reserve_bytes = min(reserve_bytes, max(total_bytes - 1, 0))

        free_budget = max(0, free_bytes - reserve_bytes)
        total_budget = max(0, total_bytes - reserve_bytes)
        requested_budget = runtime_hints.get("requestedCudaBudgetBytes") if runtime_hints else None
        budget_candidates = [free_budget, total_budget]
        if isinstance(requested_budget, int) and requested_budget > 0:
            budget_candidates.append(requested_budget)
        applied_budget = (
            min(candidate for candidate in budget_candidates if candidate > 0)
            if any(candidate > 0 for candidate in budget_candidates)
            else 0
        )

        if applied_budget <= 0:
            return {
                **result,
                "reason": "No CUDA budget remained after reserve calculation.",
                "cuda_index": cuda_index,
                "free_bytes": free_bytes,
                "total_bytes": total_bytes,
                "reserve_bytes": reserve_bytes,
            }

        fraction = max(0.05, min(1.0, applied_budget / total_bytes))
        try:
            torch.cuda.set_per_process_memory_fraction(fraction, cuda_index)
        except Exception as e:
            return {
                **result,
                "reason": f"Could not apply CUDA memory fraction: {e}",
                "cuda_index": cuda_index,
                "free_bytes": free_bytes,
                "total_bytes": total_bytes,
                "reserve_bytes": reserve_bytes,
                "applied_budget_bytes": applied_budget,
                "fraction": fraction,
            }

        return {
            "applied": True,
            "reason": "Applied CUDA memory fraction from runtime hints.",
            "cuda_index": cuda_index,
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "reserve_bytes": reserve_bytes,
            "applied_budget_bytes": applied_budget,
            "fraction": fraction,
            "model_repo": runtime_hints.get("modelRepo") if runtime_hints else None,
            "model_name": runtime_hints.get("modelName") if runtime_hints else None,
            "dtype": runtime_hints.get("dtype") if runtime_hints else None,
            "quantization_mode": runtime_hints.get("quantizationMode") if runtime_hints else None,
            "auto_offload": runtime_hints.get("autoOffload") if runtime_hints else None,
            "offload_mode": runtime_hints.get("offloadMode") if runtime_hints else None,
        }

    def _resource_retry_modes(self, runtime_hints):
        if not runtime_hints:
            return []
        allowed = [
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_SEQUENTIAL_CPU,
            OFFLOAD_MODE_GROUP_CPU,
            OFFLOAD_MODE_GROUP_DISK,
        ]
        if runtime_hints.get("resourceMode") == "auto":
            selected = runtime_hints.get("autoResourcePlan")
            target = selected if isinstance(selected, dict) else runtime_hints
            profile = self._resource_plan_execution_profile(target, runtime_hints=runtime_hints)
            modes = list(profile.retry_offload_modes)
        else:
            requested = runtime_hints.get("resourceRetryModes")
            modes = requested if isinstance(requested, list) else allowed
        modes = [mode for mode in modes if mode in allowed]
        current_mode = runtime_hints.get("offloadMode")
        if current_mode in allowed:
            current_index = allowed.index(current_mode)
            return [mode for mode in modes if allowed.index(mode) > current_index]
        return modes

    def _coerce_retry_plan_list(self, runtime_hints):
        if not runtime_hints:
            return []

        raw_plans = runtime_hints.get("resourceRetryPlans")
        if isinstance(raw_plans, list):
            plans = []
            for index, raw_plan in enumerate(raw_plans):
                if not isinstance(raw_plan, dict):
                    continue
                if runtime_hints.get("resourceMode") == "auto":
                    plan = self._canonical_auto_retry_plan(runtime_hints, raw_plan, index=index)
                else:
                    plan = deepcopy(raw_plan)
                    plan["index"] = index
                    plan["reason"] = f"retry_plan_{index + 1}"
                    plan.pop("candidateId", None)
                    plan.pop("id", None)
                    self._normalize_retry_plan_triggers(plan)
                    profile = self._resource_plan_execution_profile(plan, runtime_hints=runtime_hints)
                    self._assert_resource_plan_values_supported(plan, profile)
                plans.append(plan)
            return plans

        selected = runtime_hints.get("autoResourcePlan")
        target_source = selected if isinstance(selected, dict) else runtime_hints
        target = {
            key: target_source.get(key)
            for key in (
                "modelType",
                "mode",
                "loaderModule",
                "loaderAction",
                "executionPath",
                "pipelineClass",
            )
        }
        target.update(
            (key, target_source[key])
            for key in ("autoResourceSchemaVersion", "executionProfileId")
            if key in target_source
        )
        if not all(isinstance(target[key], str) and target[key] for key in (
            "modelType",
            "mode",
            "loaderModule",
            "loaderAction",
            "executionPath",
            "pipelineClass",
        )):
            return []
        return [
            {
                "index": index,
                "reason": f"{mode}_after_oom",
                "candidateId": target_source.get("id"),
                **target,
                "offloadMode": mode,
                "onCategories": ["oom"],
            }
            for index, mode in enumerate(self._resource_retry_modes(runtime_hints))
        ]

    @staticmethod
    def _auto_resource_contract_error(message, *, code="auto_resource_target_mismatch"):
        error = RuntimeError(message)
        setattr(error, "modiff_error_code", code)
        setattr(error, "modiff_category", "auto_resource")
        setattr(
            error,
            "modiff_recovery_hint",
            "Refresh Auto so every plan is resolved from the current exact candidate and loader contract.",
        )
        setattr(error, "modiff_auto_resource_status", "expert_only")
        return error

    @staticmethod
    def _controlled_artifact_contract_error():
        error = RuntimeError(
            "A controlled workflow artifact does not match its exact executable receipt. "
            "Repair or rebuild the workflow before running it."
        )
        setattr(error, "modiff_error_code", "controlled_artifact_mismatch")
        setattr(error, "modiff_category", "model")
        setattr(
            error,
            "modiff_recovery_hint",
            "Repair the pinned artifact in Model Manager or rebuild the controlled workflow block.",
        )
        return error

    def _bind_controlled_artifact_receipts(self, graph, runtime_hints):
        if not isinstance(runtime_hints, dict):
            return []
        try:
            selected = runtime_hints.get("autoResourcePlan")
            receipts = controlled_artifact_receipts_from_graph(
                graph,
                primary_candidate=selected if isinstance(selected, dict) else None,
            )
        except Exception as exc:
            raise self._controlled_artifact_contract_error() from exc
        runtime_hints["controlledArtifacts"] = deepcopy(receipts)
        if runtime_hints.get("resourceMode") == "auto":
            if isinstance(selected, dict):
                selected["controlledArtifacts"] = deepcopy(receipts)
            candidates = runtime_hints.get("autoResourceCandidates")
            if isinstance(candidates, list):
                for candidate in candidates:
                    if isinstance(candidate, dict):
                        candidate["controlledArtifacts"] = deepcopy(receipts)
            if receipts:
                runtime_fingerprint = self._runtime_fingerprint()
                auto_history = read_auto_resource_history(self.data_dir)
                candidates_to_check = [selected] if isinstance(selected, dict) else []
                candidates_to_check.extend(
                    candidate
                    for candidate in (candidates if isinstance(candidates, list) else [])
                    if isinstance(candidate, dict)
                )
                for candidate in candidates_to_check:
                    proof = candidate.get("proof")
                    if not isinstance(proof, dict) or proof.get("status") != "live_proven":
                        continue
                    exact_history = matching_auto_resource_success_history(
                        self.data_dir,
                        candidate=candidate,
                        runtime_fingerprint=runtime_fingerprint,
                        history=auto_history,
                    )
                    if exact_history is None:
                        candidate["proof"] = {
                            **proof,
                            "status": "skipped",
                            "source": "controlled_artifact_history_required",
                            "message": (
                                "Earlier base-only Auto evidence does not qualify this exact controlled artifact set."
                            ),
                        }
                        candidate["successHistory"] = None
                    else:
                        candidate["successHistory"] = exact_history
        return receipts

    @staticmethod
    def _normalize_retry_plan_triggers(plan):
        # These are the stable resource-pressure classifications for which a
        # loader recipe change can be corrective. Keep this narrower than the
        # full runtime classifier while preserving the existing Qwen kernel
        # fallback contract.
        allowed_categories = {"oom", "cuda_kernel"}
        allowed_error_codes = {"cuda_oom", "cuda_kernel_unsupported"}
        if "onCategories" in plan:
            plan["onCategories"] = [
                value
                for value in plan.get("onCategories") or []
                if isinstance(value, str) and value in allowed_categories
            ]
        if "onErrorCodes" in plan:
            plan["onErrorCodes"] = [
                value
                for value in plan.get("onErrorCodes") or []
                if isinstance(value, str) and value in allowed_error_codes
            ]

    @staticmethod
    def _auto_candidate_execution_fields():
        return (
            "autoResourceSchemaVersion",
            "executionProfileId",
            "modelType",
            "mode",
            "loaderModule",
            "loaderAction",
            "executionPath",
            "pipelineClass",
            "artifact",
            "baseArtifact",
            "modelRepo",
            "resolvedArtifact",
            "artifactRevision",
            "artifactResolution",
            "dtype",
            "quantizationMode",
            "loadedQuantization",
            "quantizedComponents",
            "bnb4ComputeDtype",
            "offloadMode",
            "autoOffload",
            "deviceMap",
            "generation",
            "attentionBackend",
            "regionalCompile",
            "denoiserCache",
            "channelsLast",
            "layerwiseCasting",
            "modelDependencies",
            "optionalRuntimeProfileIds",
            "optionalRuntimeRequirement",
            "studioExecutionSpecContract",
            "proof",
            "readiness",
            "canAutoRun",
            "exactPairDeclared",
            "profileArtifactCompatible",
            "requiresConfirmation",
            "artifactTrust",
            "knownBadReasons",
            "requirementsMissing",
            "requirements",
        )

    def _runtime_auto_candidate(self, runtime_hints, candidate_id):
        candidates = runtime_hints.get("autoResourceCandidates") if isinstance(runtime_hints, dict) else None
        if (
            not isinstance(candidate_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", candidate_id) is None
            or not isinstance(candidates, list)
        ):
            raise self._auto_resource_contract_error(
                "Auto candidate binding is missing. Refresh Auto before running this workflow.",
                code="auto_resource_candidate_mismatch",
            )
        matches = [
            candidate
            for candidate in candidates
            if isinstance(candidate, dict) and candidate.get("id") == candidate_id
        ]
        if len(matches) != 1:
            raise self._auto_resource_contract_error(
                "Auto candidate binding is stale or ambiguous. Refresh Auto before running this workflow.",
                code="auto_resource_candidate_mismatch",
            )
        return matches[0]

    def _assert_same_auto_candidate_recipe(self, plan, candidate):
        if not isinstance(plan, dict) or not isinstance(candidate, dict):
            raise self._auto_resource_contract_error(
                "Auto candidate recipe is unavailable. Refresh Auto before running this workflow.",
                code="auto_resource_candidate_mismatch",
            )
        mismatches = [
            key
            for key in self._auto_candidate_execution_fields()
            if plan.get(key) != candidate.get(key)
        ]
        if plan.get("controlledArtifacts") != candidate.get("controlledArtifacts"):
            mismatches.append("controlledArtifacts")
        if mismatches:
            raise self._auto_resource_contract_error(
                "Auto candidate recipe does not match the current candidate list. Refresh Auto before running this workflow.",
                code="auto_resource_candidate_mismatch",
            )

    @staticmethod
    def _resource_plan_artifact(plan):
        if not isinstance(plan, dict):
            return None
        resolution = plan.get("artifactResolution")
        resolved = resolution.get("resolved") if isinstance(resolution, dict) else None
        values = [
            plan.get("modelRepo"),
            plan.get("resolvedArtifact"),
            plan.get("artifact"),
            resolved.get("repo") if isinstance(resolved, dict) else None,
        ]
        repos = [value for value in values if isinstance(value, str) and value]
        if not repos:
            return None
        if any(repo != repos[0] for repo in repos[1:]):
            raise WebServer._auto_resource_contract_error(
                "Auto resource plan contains inconsistent artifact identities. Refresh Auto before running this workflow."
            )
        return repos[0]

    @staticmethod
    def _assert_resource_plan_artifact_compatible(plan, profile):
        repo = WebServer._resource_plan_artifact(plan)
        if repo is None:
            return
        compatible = {
            item
            for item in (profile.default_repo, profile.fallback_repo, *profile.compatible_repos)
            if isinstance(item, str) and item
        }
        if repo not in compatible:
            raise WebServer._auto_resource_contract_error(
                "Auto resource plan artifact is not admitted by its exact execution profile. "
                "Refresh Auto before running this workflow."
            )

    @staticmethod
    def _assert_resource_plan_values_supported(plan, profile):
        offload_mode = plan.get("offloadMode")
        if offload_mode is not None and offload_mode not in profile.supported_offload_modes:
            raise WebServer._auto_resource_contract_error(
                "Auto resource plan contains an unsupported offload mode. Refresh Auto before running this workflow."
            )
        dtype = plan.get("dtype")
        if dtype is not None and dtype not in {"float32", "float16", "bfloat16"}:
            raise WebServer._auto_resource_contract_error(
                "Auto resource plan contains an unsupported dtype. Refresh Auto before running this workflow."
            )
        compute_dtype = plan.get("bnb4ComputeDtype")
        if compute_dtype is not None and compute_dtype not in {"float32", "float16", "bfloat16"}:
            raise WebServer._auto_resource_contract_error(
                "Auto resource plan contains an unsupported compute dtype. Refresh Auto before running this workflow."
            )
        quantization_mode = plan.get("quantizationMode")
        if quantization_mode is not None and quantization_mode not in {
            "none",
            "bnb_4bit",
            "bnb_8bit",
            "quanto_float8",
            "quanto_int8",
            "torchao_float8",
            "torchao_int8_weight_only",
        }:
            raise WebServer._auto_resource_contract_error(
                "Auto resource plan contains an unsupported quantization mode. Refresh Auto before running this workflow."
            )
        device_map = plan.get("deviceMap")
        if device_map is not None and device_map not in RUNTIME_DEVICE_MAPS:
            raise WebServer._auto_resource_contract_error(
                "Auto resource plan contains an unsupported device map. Refresh Auto before running this workflow."
            )
        attention_backend = plan.get("attentionBackend")
        if attention_backend is not None and attention_backend not in RUNTIME_ATTENTION_BACKENDS:
            raise WebServer._auto_resource_contract_error(
                "Auto resource plan contains an unsupported attention backend. Refresh Auto before running this workflow."
            )
        denoiser_cache = plan.get("denoiserCache")
        if denoiser_cache is not None and denoiser_cache not in RUNTIME_DENOISER_CACHE_MODES:
            raise WebServer._auto_resource_contract_error(
                "Auto resource plan contains an unsupported denoiser cache. Refresh Auto before running this workflow."
            )
        for key in ("autoOffload", "regionalCompile", "channelsLast", "layerwiseCasting"):
            if key in plan and plan.get(key) is not None and not isinstance(plan.get(key), bool):
                raise WebServer._auto_resource_contract_error(
                    "Auto resource plan contains an invalid runtime flag. Refresh Auto before running this workflow."
                )
        components = plan.get("quantizedComponents")
        if components is not None:
            admitted = set(profile.quantizable_components)
            if (
                not isinstance(components, list)
                or any(not isinstance(item, str) or item not in admitted for item in components)
            ):
                raise WebServer._auto_resource_contract_error(
                    "Auto resource plan contains unsupported quantized components. "
                    "Refresh Auto before running this workflow."
                )

    def _canonical_auto_retry_plan(self, runtime_hints, raw_plan, *, index):
        candidate_id = raw_plan.get("candidateId") if isinstance(raw_plan, dict) else None
        candidate = self._runtime_auto_candidate(runtime_hints, candidate_id)
        retry_identity_fields = (
            "modelType",
            "mode",
            "loaderModule",
            "loaderAction",
            "executionPath",
            "pipelineClass",
        )
        if any(raw_plan.get(key) != candidate.get(key) for key in retry_identity_fields):
            raise self._auto_resource_contract_error(
                "Auto retry identity does not match its current candidate. Refresh Auto before running this workflow.",
                code="auto_resource_candidate_mismatch",
            )
        if any(
            key in raw_plan and raw_plan.get(key) != candidate.get(key)
            for key in self._auto_candidate_execution_fields()
            if key not in retry_identity_fields
        ):
            raise self._auto_resource_contract_error(
                "Auto retry recipe does not match its current candidate. Refresh Auto before running this workflow.",
                code="auto_resource_candidate_mismatch",
            )
        selected = runtime_hints.get("autoResourcePlan")
        selected_id = selected.get("id") if isinstance(selected, dict) else None
        selected_candidate = self._runtime_auto_candidate(runtime_hints, selected_id)
        self._assert_same_auto_candidate_recipe(selected, selected_candidate)

        selected_profile = self._resource_plan_execution_profile(selected, runtime_hints=runtime_hints)
        retry_profile = self._resource_plan_execution_profile(candidate, runtime_hints=runtime_hints)
        if retry_profile.id != selected_profile.id:
            raise self._auto_resource_contract_error(
                "Auto retry candidate does not belong to the selected execution profile. "
                "Refresh Auto before running this workflow.",
                code="auto_resource_candidate_mismatch",
            )
        if not self._auto_resource_candidate_is_proven(candidate):
            raise self._auto_resource_contract_error(
                "Auto retry candidate is not runnable. Refresh Auto before running this workflow.",
                code="auto_resource_candidate_mismatch",
            )
        self._assert_resource_plan_artifact_compatible(candidate, retry_profile)
        self._assert_resource_plan_values_supported(candidate, retry_profile)

        plan = deepcopy(candidate)
        plan["candidateId"] = candidate_id
        plan["index"] = index
        plan["reason"] = f"candidate_retry_{index + 1}"
        for key in ("onCategories", "onErrorCodes"):
            values = raw_plan.get(key)
            if isinstance(values, list):
                plan[key] = list(values)
        self._normalize_retry_plan_triggers(plan)
        return plan

    def _set_param_value_if_present(self, node, key, value):
        params = node.get("params") if isinstance(node, dict) else None
        if not isinstance(params, dict) or key not in params or not isinstance(params[key], dict):
            return False
        current = params[key].get("value", params[key].get("default"))
        try:
            if bool(current == value):
                return False
        except (TypeError, ValueError):
            # Retry-controlled parameters are expected to be JSON-like. An
            # exotic value must remain replaceable without pulling tensor
            # comparison into the server planner.
            pass
        params[key]["value"] = deepcopy(value)
        return True

    def _set_model_repo_if_present(self, node, key, repo):
        params = node.get("params") if isinstance(node, dict) else None
        if not repo or not isinstance(params, dict) or key not in params or not isinstance(params[key], dict):
            return False
        current = params[key].get("value")
        current_repo = current.get("value") if isinstance(current, dict) else current
        current_source = current.get("source") if isinstance(current, dict) else "hub"
        if current_repo == repo and current_source == "hub":
            return False
        if isinstance(current, dict):
            params[key]["value"] = {**current, "source": "hub", "value": repo}
        else:
            params[key]["value"] = {"source": "hub", "value": repo}
        return True

    @staticmethod
    def _auto_resource_artifact_revision(plan, repo):
        """Resolve the exact Hub commit owned by one Auto artifact."""

        if not isinstance(plan, dict) or not isinstance(repo, str) or not repo:
            raise WebServer._auto_resource_contract_error(
                "Auto repository mutation requires a reviewed artifact identity."
            )
        resolution = plan.get("artifactResolution")
        resolved = resolution.get("resolved") if isinstance(resolution, dict) else None
        declared = plan.get("artifactRevision")
        if declared in (None, "") and isinstance(resolved, dict):
            declared = resolved.get("revision")
        if declared not in (None, ""):
            if (
                not isinstance(declared, str)
                or declared != declared.strip()
                or declared != declared.lower()
                or not IMMUTABLE_HUB_REVISION.fullmatch(declared)
            ):
                raise WebServer._auto_resource_contract_error(
                    "Auto artifact requires an exact lowercase 40-character commit revision."
                )
        else:
            declared = None

        reviewed = catalog_revision(repo)
        if reviewed is not None and declared is not None and declared != reviewed:
            raise WebServer._auto_resource_contract_error(
                "Auto artifact revision does not match the reviewed catalog commit."
            )
        revision = reviewed or declared
        if revision is None:
            raise WebServer._auto_resource_contract_error(
                "Auto artifact has no immutable reviewed revision."
            )
        return revision

    def _set_model_repo_and_revision_if_present(
        self,
        node,
        key,
        repo,
        revision,
        *,
        node_id,
        pinned_fields,
        dry_run=False,
    ):
        """Mutate a generic loader's Hub repository and commit as one identity."""

        params = node.get("params") if isinstance(node, dict) else None
        field = params.get(key) if isinstance(params, dict) else None
        if not repo or not isinstance(field, dict):
            return False
        node_key = str(node_id)
        if (node_key, key) in pinned_fields:
            # A pinned repository selection owns its existing revision too.
            return False

        current = field.get("value", field.get("default"))
        current_repo = current.get("value") if isinstance(current, dict) else current
        current_source = current.get("source") if isinstance(current, dict) else "hub"
        repository_changes = current_repo != repo or current_source != "hub"
        revision_field = params.get("revision") if isinstance(params, dict) else None
        if not isinstance(revision_field, dict):
            if repository_changes:
                raise self._auto_resource_contract_error(
                    "Auto cannot change a loader repository without a revision field on the same loader."
                )
            return False

        revision_is_pinned = (node_key, "revision") in pinned_fields
        current_revision = revision_field.get("value", revision_field.get("default"))
        if revision_is_pinned and repository_changes and current_revision != revision:
            raise self._auto_resource_contract_error(
                "Auto cannot change a loader repository while its revision override is pinned to a different commit. "
                "Unpin both fields or switch to Expert."
            )

        revision_changes = not revision_is_pinned and current_revision != revision
        if dry_run:
            return repository_changes or revision_changes
        changed = self._set_model_repo_if_present(node, key, repo)
        if revision_changes:
            revision_field["value"] = revision
            changed = True
        return changed

    @staticmethod
    def _resource_plan_execution_profile(plan, *, runtime_hints=None):
        """Resolve and validate the one backend-owned target for a plan."""

        if not isinstance(plan, dict):
            raise WebServer._auto_resource_contract_error(
                "Auto resource plan must be a JSON object with an exact loader target."
            )
        hints = runtime_hints if isinstance(runtime_hints, dict) else {}
        for key in ("modelType", "mode"):
            plan_value = plan.get(key)
            hint_value = hints.get(key)
            if (
                isinstance(plan_value, str)
                and isinstance(hint_value, str)
                and hint_value
                and plan_value != hint_value
            ):
                raise WebServer._auto_resource_contract_error(
                    f"Auto resource plan {key} does not match the active workflow. "
                    "Refresh Auto before running this workflow."
                )
        model_type = plan.get("modelType")
        mode = plan.get("mode")
        if (
            not isinstance(model_type, str)
            or not model_type
            or model_type != model_type.strip()
            or not isinstance(mode, str)
            or not mode
            or mode != mode.strip()
        ):
            raise WebServer._auto_resource_contract_error(
                "Auto resource plan requires an exact modelType and mode before it can target a loader."
            )
        profiles = execution_profiles_for_execution(model_type, mode)
        if len(profiles) != 1:
            raise WebServer._auto_resource_contract_error(
                "Auto resource plan does not resolve to one exact execution profile."
            )
        profile = profiles[0]
        expected = {
            "loaderModule": profile.loader_module,
            "loaderAction": profile.loader_action,
            "executionPath": profile.execution_path,
            "pipelineClass": profile.pipeline_class,
        }
        if hints.get("resourceMode") == "auto" or any(
            key in plan for key in ("autoResourceSchemaVersion", "executionProfileId")
        ):
            expected.update(
                autoResourceSchemaVersion=AUTO_RESOURCE_SCHEMA_VERSION,
                executionProfileId=profile.id,
            )
        mismatches = [
            key
            for key, value in expected.items()
            if plan.get(key) != value
        ]
        if mismatches:
            raise WebServer._auto_resource_contract_error(
                f"Auto resource plan target does not match execution profile {profile.id!r}: "
                + ", ".join(mismatches)
                + ". Refresh Auto before running this workflow."
            )
        hint_mismatches = [
            key
            for key, value in expected.items()
            if hints.get(key) not in (None, "") and hints.get(key) != value
        ]
        if hint_mismatches:
            raise WebServer._auto_resource_contract_error(
                f"Active workflow target does not match execution profile {profile.id!r}: "
                + ", ".join(hint_mismatches)
                + ". Refresh Auto before running this workflow."
            )
        if hints.get("resourceMode") == "auto" or any(
            key in plan for key in ("optionalRuntimeProfileIds", "optionalRuntimeRequirement")
        ):
            expected_profile_ids = list(
                optional_runtime_profile_ids_for_execution(model_type, mode)
            )
            expected_requirement = WebServer._optional_runtime_contract_signature(
                optional_runtime_requirement_for_execution(model_type, mode)
            )
            if (
                plan.get("optionalRuntimeProfileIds") != expected_profile_ids
                or WebServer._optional_runtime_contract_signature(
                    plan.get("optionalRuntimeRequirement")
                )
                != expected_requirement
            ):
                raise WebServer._auto_resource_contract_error(
                    "Auto resource plan optional-runtime receipt does not match its execution profile. "
                    "Refresh Auto before running this workflow."
                )
        if hints.get("resourceMode") == "auto" or "studioExecutionSpecContract" in plan:
            specification = studio_execution_spec_for_pair(model_type, mode)
            expected_contract = WebServer._studio_execution_spec_contract_signature(
                specification
            )
            declared_contract = plan.get("studioExecutionSpecContract")
            declared_keys_are_exact = (
                isinstance(declared_contract, dict)
                and set(declared_contract)
                == {"schemaVersion", "id", "contentHash", "executionProfileId"}
            )
            if (
                (expected_contract is None and declared_contract is not None)
                or (
                    expected_contract is not None
                    and (
                        not declared_keys_are_exact
                        or WebServer._studio_execution_spec_contract_signature(declared_contract)
                        != expected_contract
                    )
                )
            ):
                raise WebServer._auto_resource_contract_error(
                    "Auto resource plan graph receipt does not match the current Studio execution specification. "
                    "Refresh Auto before running this workflow."
                )
        if hints.get("resourceMode") == "auto" or "modelDependencies" in plan:
            expected_dependencies = WebServer._model_dependencies_signature(
                studio_model_dependencies_for_pair(model_type, mode)
            )
            if (
                WebServer._model_dependencies_signature(plan.get("modelDependencies"))
                != expected_dependencies
                or "modelDependencies" in hints
                and WebServer._model_dependencies_signature(hints.get("modelDependencies"))
                != expected_dependencies
            ):
                raise WebServer._auto_resource_contract_error(
                    "Auto resource plan model dependencies do not match the current reviewed artifact contract. "
                    "Refresh Auto before running this workflow."
                )
        return profile

    @staticmethod
    def _optional_runtime_contract_signature(requirement):
        if not isinstance(requirement, dict):
            return None
        profile_ids = requirement.get("profileIds")
        execution_profile_ids = requirement.get("executionProfileIds")
        if (
            isinstance(requirement.get("schemaVersion"), bool)
            or not isinstance(requirement.get("schemaVersion"), int)
            or not isinstance(requirement.get("delivery"), str)
            or type(requirement.get("requiredNow")) is not bool
            or not isinstance(profile_ids, list)
            or any(not isinstance(item, str) for item in profile_ids)
            or not isinstance(execution_profile_ids, list)
            or any(not isinstance(item, str) for item in execution_profile_ids)
        ):
            return None
        return {
            "schemaVersion": requirement["schemaVersion"],
            "delivery": requirement["delivery"],
            "requiredNow": requirement["requiredNow"],
            "profileIds": list(profile_ids),
            "executionProfileIds": list(execution_profile_ids),
        }

    @staticmethod
    def _studio_execution_spec_contract_signature(contract):
        if not isinstance(contract, dict):
            return None
        if (
            isinstance(contract.get("schemaVersion"), bool)
            or not isinstance(contract.get("schemaVersion"), int)
            or not isinstance(contract.get("id"), str)
            or not isinstance(contract.get("contentHash"), str)
            or not isinstance(contract.get("executionProfileId"), str)
        ):
            return None
        return {
            "schemaVersion": contract["schemaVersion"],
            "id": contract["id"],
            "contentHash": contract["contentHash"],
            "executionProfileId": contract["executionProfileId"],
        }

    @staticmethod
    def _model_dependencies_signature(dependencies):
        if not isinstance(dependencies, list) or len(dependencies) > 32:
            return None
        output = []
        for dependency in dependencies:
            if not isinstance(dependency, dict) or set(dependency) != {
                "id",
                "kind",
                "repo",
                "revision",
            }:
                return None
            if not all(
                isinstance(dependency.get(key), str)
                and dependency[key]
                and len(dependency[key]) <= 512
                for key in dependency
            ):
                return None
            output.append({key: dependency[key] for key in ("id", "kind", "repo", "revision")})
        if len({dependency["id"] for dependency in output}) != len(output):
            return None
        return sorted(output, key=lambda dependency: (dependency["kind"], dependency["id"], dependency["repo"]))

    @staticmethod
    def _resource_plan_node_matches_profile(node, profile):
        if (
            not isinstance(node, dict)
            or node.get("module") != profile.loader_module
            or node.get("action") != profile.loader_action
        ):
            return False
        params = node.get("params")
        if not isinstance(params, dict):
            return False
        identity_key = "model_type" if profile.loader_action == "ModelsLoader" else "pipeline_class"
        identity_param = params.get(identity_key)
        if not isinstance(identity_param, dict):
            return False
        identity = identity_param.get("value", identity_param.get("default"))
        expected_identity = profile.model_type if identity_key == "model_type" else profile.pipeline_class
        return identity == expected_identity

    def _resource_plan_targets_node_family(self, node, plan, *, runtime_hints=None):
        profile = self._resource_plan_execution_profile(plan, runtime_hints=runtime_hints)
        return self._resource_plan_node_matches_profile(node, profile)

    def _retry_plan_matches(self, plan, classification):
        if not isinstance(plan, dict) or not isinstance(classification, dict):
            return False

        error_code = classification.get("error_code")
        category = classification.get("category")
        on_error_codes = plan.get("onErrorCodes")
        on_categories = plan.get("onCategories")

        if isinstance(on_error_codes, list) and error_code in on_error_codes:
            return True
        if isinstance(on_categories, list) and category in on_categories:
            return True
        if on_error_codes is None and on_categories is None:
            return category == "oom"
        return False

    def _next_retry_plan_index(self, plans, current_index, classification):
        for index in range(current_index + 1, len(plans)):
            if self._retry_plan_matches(plans[index], classification):
                return index
        return None

    def _next_applicable_retry_plan_index(self, graph, plans, current_index, classification):
        """Return a retry that can change at least one unpinned runtime field."""
        skipped = []
        index = self._next_retry_plan_index(plans, current_index, classification)
        while index is not None:
            graph_copy = deepcopy(graph)
            if self._apply_resource_retry_plan_to_graph(graph_copy, plans[index]):
                return index, skipped
            skipped.append(index)
            index = self._next_retry_plan_index(plans, index, classification)
        return None, skipped

    def _sanitize_retry_plan_for_hints(self, plan):
        if not isinstance(plan, dict):
            return None
        allowed = {
            "index",
            "reason",
            "autoResourceSchemaVersion",
            "executionProfileId",
            "modelType",
            "mode",
            "loaderModule",
            "loaderAction",
            "executionPath",
            "modelRepo",
            "resolvedArtifact",
            "artifactRevision",
            "artifactResolution",
            "quantizationMode",
            "quantizedComponents",
            "bnb4ComputeDtype",
            "dtype",
            "pipelineClass",
            "candidateId",
            "id",
            "offloadMode",
            "deviceMap",
            "generation",
            "onCategories",
            "onErrorCodes",
        }
        sanitized = {key: deepcopy(plan.get(key)) for key in allowed if key in plan}
        for key in ("candidateId", "id"):
            value = sanitized.get(key)
            if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value) is None:
                sanitized.pop(key, None)
        index = sanitized.get("index")
        sanitized["reason"] = (
            f"retry_plan_{index + 1}"
            if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < 32
            else "retry_plan"
        )
        self._normalize_retry_plan_triggers(sanitized)
        return sanitized

    def _apply_resource_retry_to_graph(self, graph, offload_mode):
        nodes = graph.get("nodes", {})
        if not isinstance(nodes, dict):
            return []

        updated = []
        for node_id, node in nodes.items():
            if not isinstance(node, dict):
                continue
            action = node.get("action")
            module = node.get("module")
            compatible_loader = (
                module == "modules.ModularDiffusers" and action in ("ModelsLoader", "DynamicPipelineLoader")
            ) or (
                module in ("modules.DiffusersImage", "modules.DiffusersAudio", "modules.DiffusersVideo")
                and action == "LoadPipeline"
            )
            if not compatible_loader:
                continue
            changed = self._set_param_value_if_present(node, "offload_mode", offload_mode)
            changed = (
                self._set_param_value_if_present(node, "auto_offload", offload_mode != OFFLOAD_MODE_NONE) or changed
            )
            if changed:
                updated.append(str(node_id))

        return updated

    def _apply_resource_retry_plan_to_graph(self, graph, plan):
        nodes = graph.get("nodes", {})
        if not isinstance(nodes, dict) or not isinstance(plan, dict):
            return []

        runtime_hints = self._coerce_runtime_hints(graph.get("runtimeHints"))
        profile = self._resource_plan_execution_profile(plan, runtime_hints=runtime_hints)
        if isinstance(runtime_hints, dict) and runtime_hints.get("resourceMode") == "auto":
            self._assert_resource_plan_artifact_compatible(plan, profile)
            self._assert_resource_plan_values_supported(plan, profile)
        paths = graph.get("paths")
        executable_node_ids = None
        if isinstance(paths, list):
            executable_node_ids = {
                str(node_id)
                for path in paths
                if isinstance(path, list)
                for node_id in path
            }
        target_nodes = [
            (node_id, node)
            for node_id, node in nodes.items()
            if (executable_node_ids is None or str(node_id) in executable_node_ids)
            and self._resource_plan_node_matches_profile(node, profile)
        ]
        if not target_nodes:
            raise self._auto_resource_contract_error(
                f"Auto resource plan target {profile.loader_module}.{profile.loader_action} matched zero exact "
                "loader identities. Refresh Auto before running this workflow."
            )
        target_node_ids = {str(node_id) for node_id, _node in target_nodes}
        raw_overrides = runtime_hints.get("autoFieldOverrides") if isinstance(runtime_hints, dict) else None
        pinned_fields = {
            (str(item.get("nodeId")), str(item.get("fieldKey")))
            for item in (raw_overrides if isinstance(raw_overrides, list) else [])
            if isinstance(item, dict) and isinstance(item.get("nodeId"), str) and isinstance(item.get("fieldKey"), str)
        }

        def set_param(node_id, node, param_key, value):
            if (str(node_id), param_key) in pinned_fields:
                return False
            return self._set_param_value_if_present(node, param_key, value)

        def set_model_repo(node_id, node, param_key, value):
            return self._set_model_repo_and_revision_if_present(
                node,
                param_key,
                value,
                artifact_revision,
                node_id=node_id,
                pinned_fields=pinned_fields,
            )

        offload_mode = plan.get("offloadMode")
        device_map = plan.get("deviceMap")
        model_repo = plan.get("modelRepo") or plan.get("resolvedArtifact")
        artifact_revision = (
            self._auto_resource_artifact_revision(plan, model_repo)
            if isinstance(model_repo, str) and model_repo
            else None
        )
        quantization_mode = plan.get("quantizationMode")
        quantized_components = plan.get("quantizedComponents")
        compute_dtype = plan.get("bnb4ComputeDtype")
        dtype = plan.get("dtype")
        applicable_param_keys = set()
        if isinstance(offload_mode, str):
            applicable_param_keys.update(("offload_mode", "auto_offload"))
        if isinstance(device_map, str):
            applicable_param_keys.add("device_map")
        if isinstance(model_repo, str) and model_repo:
            applicable_param_keys.update(("model_id", "repo_id"))
        if isinstance(dtype, str):
            applicable_param_keys.add("dtype")
        if profile.loader_module == "modules.DiffusersImage":
            if isinstance(quantization_mode, str):
                applicable_param_keys.add("quantization_mode")
            if isinstance(quantized_components, list):
                applicable_param_keys.add("quantized_components")
            if isinstance(compute_dtype, str):
                applicable_param_keys.add("bnb_4bit_compute_dtype")
        if not any(
            isinstance(node.get("params"), dict)
            and any(key in node["params"] for key in applicable_param_keys)
            for _node_id, node in target_nodes
        ):
            raise self._auto_resource_contract_error(
                f"Auto resource plan target {profile.loader_module}.{profile.loader_action} matched loader nodes "
                "but none exposes an applicable plan field. Refresh Auto before running this workflow."
            )
        target_recipe_ids = set()
        for _node_id, node in target_nodes:
            recipe_param = (node.get("params") or {}).get("execution_recipe")
            recipe_source_id = recipe_param.get("sourceId") if isinstance(recipe_param, dict) else None
            if isinstance(recipe_source_id, str) and recipe_source_id:
                target_recipe_ids.add(recipe_source_id)

        # Validate every target before mutating any node. A pinned stale commit
        # must fail the whole Auto rewrite without leaving a partially changed
        # graph behind.
        if isinstance(model_repo, str) and model_repo:
            for node_id, node in target_nodes:
                for param_key in ("model_id", "repo_id"):
                    self._set_model_repo_and_revision_if_present(
                        node,
                        param_key,
                        model_repo,
                        artifact_revision,
                        node_id=node_id,
                        pinned_fields=pinned_fields,
                        dry_run=True,
                    )

        updated = []
        for node_id, node in nodes.items():
            if not isinstance(node, dict):
                continue
            action = node.get("action")
            module = node.get("module")
            if (
                str(node_id) in target_recipe_ids
                and (executable_node_ids is None or str(node_id) in executable_node_ids)
                and module == "modules.DiffusersRuntime"
                and action == "DiffusersExecutionRecipe"
            ):
                recipe_changed = False
                if isinstance(offload_mode, str):
                    recipe_changed = set_param(node_id, node, "offload_mode", offload_mode) or recipe_changed
                if isinstance(device_map, str):
                    recipe_changed = set_param(node_id, node, "device_map", device_map) or recipe_changed
                if recipe_changed:
                    updated.append(str(node_id))

            if str(node_id) in target_node_ids:
                changed = False
                if isinstance(offload_mode, str):
                    changed = set_param(node_id, node, "offload_mode", offload_mode) or changed
                    changed = set_param(node_id, node, "auto_offload", offload_mode != OFFLOAD_MODE_NONE) or changed
                if isinstance(device_map, str):
                    changed = set_param(node_id, node, "device_map", device_map) or changed
                if isinstance(model_repo, str) and model_repo:
                    changed = set_model_repo(node_id, node, "model_id", model_repo) or changed
                    changed = set_model_repo(node_id, node, "repo_id", model_repo) or changed
                if profile.loader_action == "ModelsLoader":
                    changed = set_param(node_id, node, "model_type", profile.model_type) or changed
                else:
                    changed = set_param(node_id, node, "pipeline_class", profile.pipeline_class) or changed
                if isinstance(dtype, str):
                    changed = set_param(node_id, node, "dtype", dtype) or changed
                if module == "modules.DiffusersImage" and action == "LoadPipeline":
                    if isinstance(quantization_mode, str):
                        changed = set_param(node_id, node, "quantization_mode", quantization_mode) or changed
                    if isinstance(quantized_components, list):
                        changed = (
                            set_param(
                                node_id,
                                node,
                                "quantized_components",
                                [str(item) for item in quantized_components],
                            )
                            or changed
                        )
                    if isinstance(compute_dtype, str):
                        changed = set_param(node_id, node, "bnb_4bit_compute_dtype", compute_dtype) or changed
                    if model_repo == QWEN_IMAGE_2512_PREQUANTIZED_REPO:
                        changed = set_param(node_id, node, "quantization_mode", "none") or changed
                        changed = set_param(node_id, node, "quantized_components", []) or changed
                if changed:
                    updated.append(str(node_id))

        return updated

    def _auto_resource_candidate_is_proven(self, candidate):
        if not isinstance(candidate, dict):
            return False
        proof = candidate.get("proof")
        status = proof.get("status") if isinstance(proof, dict) else None
        return status in PROVEN_PROOF_STATUSES

    def _auto_resource_requires_proven_candidate(self, runtime_hints):
        # Proof is qualification evidence, not a deterministic execution
        # prerequisite. Missing artifacts, inputs, devices, and incompatible
        # types are rejected by graph/runtime validation; an otherwise
        # complete but not-yet-qualified Auto recipe remains runnable with a
        # warning.
        return False

    def _assert_auto_resource_candidate_ready(self, runtime_hints):
        if not isinstance(runtime_hints, dict) or runtime_hints.get("resourceMode") != "auto":
            return

        auto_plan = runtime_hints.get("autoResourcePlan")
        plan_model_type = str(auto_plan.get("modelType") or "").strip() if isinstance(auto_plan, dict) else ""
        plan_mode = str(auto_plan.get("mode") or "").strip() if isinstance(auto_plan, dict) else ""
        hint_model_type = str(runtime_hints.get("modelType") or "").strip()
        hint_mode = str(runtime_hints.get("mode") or "").strip()
        pair_mismatch = bool(
            (plan_model_type and hint_model_type and plan_model_type != hint_model_type)
            or (plan_mode and hint_mode and plan_mode != hint_mode)
        )
        if pair_mismatch:
            error = RuntimeError(
                "Auto resource plan pair does not match the requested workflow pair. "
                "Refresh the Auto plan or switch to Expert before executing this workflow."
            )
            setattr(error, "modiff_error_code", "auto_resource_pair_mismatch")
            setattr(error, "modiff_category", "auto_resource")
            setattr(
                error,
                "modiff_recovery_hint",
                "Refresh Auto so the selected candidate is resolved for the current model and task. "
                "Structurally valid workflows can still be configured explicitly in Expert mode.",
            )
            setattr(error, "modiff_auto_resource_status", "expert_only")
            raise error

        model_type = plan_model_type
        mode = plan_mode
        if (
            not isinstance(auto_plan, dict)
            or not model_type
            or not mode
            or not auto_resource_pair_is_declared(model_type, mode)
        ):
            error = RuntimeError(
                "Auto has no declared execution recipe for the exact model/task pair. "
                "Refresh the Auto plan or switch to Expert before executing this workflow."
            )
            setattr(error, "modiff_error_code", "auto_resource_pair_undeclared")
            setattr(error, "modiff_category", "auto_resource")
            setattr(
                error,
                "modiff_recovery_hint",
                "Use Auto only for a model and task pair declared by both the backend resource requirements "
                "and execution profiles. Structurally valid workflows can still be configured explicitly in Expert mode.",
            )
            setattr(error, "modiff_auto_resource_status", "expert_only")
            raise error

        plan_candidate_id = auto_plan.get("id")
        selected_candidate_id = runtime_hints.get("autoResourceCandidateId")
        if (
            not isinstance(plan_candidate_id, str)
            or not plan_candidate_id
            or not isinstance(selected_candidate_id, str)
            or not selected_candidate_id
            or selected_candidate_id != plan_candidate_id
        ):
            error = RuntimeError(
                "Auto resource candidate ID does not match the selected plan ID. "
                "Refresh Auto before executing this workflow."
            )
            setattr(error, "modiff_error_code", "auto_resource_candidate_mismatch")
            setattr(error, "modiff_category", "auto_resource")
            setattr(error, "modiff_recovery_hint", "Refresh Auto and reselect a candidate for the current graph.")
            setattr(error, "modiff_auto_resource_status", "expert_only")
            raise error

        candidate = self._runtime_auto_candidate(runtime_hints, plan_candidate_id)
        self._assert_same_auto_candidate_recipe(auto_plan, candidate)

        try:
            profile = self._resource_plan_execution_profile(auto_plan, runtime_hints=runtime_hints)
            self._assert_resource_plan_artifact_compatible(auto_plan, profile)
            self._assert_resource_plan_values_supported(auto_plan, profile)
        except RuntimeError as exc:
            error = RuntimeError(str(exc))
            setattr(error, "modiff_error_code", "auto_resource_target_mismatch")
            setattr(error, "modiff_category", "auto_resource")
            setattr(
                error,
                "modiff_recovery_hint",
                "Refresh Auto so the selected candidate targets the exact loader contract in this workflow.",
            )
            setattr(error, "modiff_auto_resource_status", "expert_only")
            raise error from exc

        if not self._auto_resource_requires_proven_candidate(runtime_hints):
            return
        if self._auto_resource_candidate_is_proven(auto_plan):
            return

        candidates = runtime_hints.get("autoResourceCandidates") if isinstance(runtime_hints, dict) else None
        proven_candidates = (
            [candidate for candidate in candidates if self._auto_resource_candidate_is_proven(candidate)]
            if isinstance(candidates, list)
            else []
        )
        if proven_candidates:
            runtime_hints["autoResourcePlan"] = proven_candidates[0]
            runtime_hints["autoResourceCandidateId"] = proven_candidates[0].get("id")
            proof = proven_candidates[0].get("proof") if isinstance(proven_candidates[0].get("proof"), dict) else {}
            runtime_hints["autoResourceProofStatus"] = proof.get("status")
            return

        status = runtime_hints.get("compatibilityStatus") or runtime_hints.get("autoResourceProofStatus") or "unproven"
        error = RuntimeError(
            "Auto resource plan is not ready for this workflow. Refresh the Auto plan or choose Expert "
            "settings before executing this workflow."
        )
        setattr(error, "modiff_error_code", "auto_resource_unproven")
        setattr(error, "modiff_category", "auto_resource")
        setattr(
            error,
            "modiff_recovery_hint",
            (
                "Auto has not found a runnable artifact in the local model/cache and hardware metadata. "
                "Install the suggested compatible artifact or switch to Expert if you want to choose the configuration yourself."
            ),
        )
        setattr(error, "modiff_auto_resource_status", status)
        raise error

    def _record_auto_resource_success(self, runtime_hints, runtime_fingerprint, measurement=None):
        try:
            record_auto_resource_success(
                self.data_dir,
                runtime_fingerprint=runtime_fingerprint
                if isinstance(runtime_fingerprint, dict)
                else self._runtime_fingerprint(),
                runtime_hints=runtime_hints,
                measurement=measurement,
            )
        except Exception as exc:
            logger.debug(f"Could not record Auto resource success: {exc}")

    def _record_optimization_observations(
        self,
        runtime_hints,
        runtime_fingerprint,
        measurement=None,
        graph=None,
    ):
        if not isinstance(runtime_hints, dict) or not isinstance(measurement, dict):
            return []
        if not isinstance(graph, dict):
            task_id = str((getattr(self, "current_task", None) or {}).get("task_id") or "")
            task_graphs = getattr(self, "task_graphs", {})
            graph = task_graphs.get(task_id) if isinstance(task_graphs, dict) else None
        if not isinstance(graph, dict):
            return []
        selections = optimization_selections_from_graph(graph)
        form = runtime_hints.get("optimizationQualificationForm")
        workload_key = optimization_workload_key_for_form(form if isinstance(form, dict) else {})
        runtime_identity = (
            runtime_fingerprint.get("resourceFingerprint")
            if isinstance(runtime_fingerprint, dict)
            else runtime_fingerprint
        )
        model_type = str(runtime_hints.get("modelType") or "")
        mode = str((form or {}).get("mode") or "") if isinstance(form, dict) else ""
        artifact = str(
            runtime_hints.get("resolvedArtifact")
            or runtime_hints.get("resolvedModelRepo")
            or runtime_hints.get("modelRepo")
            or ""
        )
        receipts = []
        if not selections:
            try:
                return [
                    record_optimization_workload_baseline(
                        runtime_fingerprint=runtime_identity,
                        model_type=model_type,
                        mode=mode,
                        artifact=artifact,
                        workload_key=workload_key,
                        measurement=measurement,
                    )
                ]
            except Exception as exc:
                logger.debug("Could not record optimization workload baseline: %s", exc)
                return []
        for selection in selections:
            capability_id = str(selection.get("capabilityId") or "")
            try:
                receipts.append(
                    record_optimization_workload_observation(
                        capability_id=capability_id,
                        runtime_fingerprint=runtime_identity,
                        model_type=model_type,
                        mode=mode,
                        artifact=artifact,
                        workload_key=workload_key,
                        selection={key: value for key, value in selection.items() if key != "capabilityId"},
                        measurement=measurement,
                    )
                )
            except Exception as exc:
                logger.debug("Could not record optimization workload observation: %s", exc)
        return receipts

    def _record_auto_resource_failure(self, error, classification=None):
        # Auto history is a machine/artifact admission signal. User-authored
        # prompt, dimension, graph, or media-input failures must never poison a
        # model candidate and prevent the corrected graph from running.
        resource_categories = {"oom", "cuda_context", "cuda_kernel", "missing_dependency", "missing_model"}
        if not isinstance(classification, dict) or classification.get("category") not in resource_categories:
            return
        try:
            runtime_hints = self.current_task.get("runtimeHints") if self.current_task else None
            record_auto_resource_failure(
                self.data_dir,
                runtime_fingerprint=self._runtime_fingerprint(),
                runtime_hints=runtime_hints,
                classification=classification,
                error=error,
            )
        except Exception as exc:
            logger.debug(f"Could not record Auto resource failure: {exc}")

    def _reset_runtime_measurement(self):
        """Reset accelerator peak counters immediately before one graph attempt."""
        try:
            torch = import_module("torch")
            if bool(torch.cuda.is_available()):
                for index in range(int(torch.cuda.device_count())):
                    try:
                        torch.cuda.reset_peak_memory_stats(index)
                    except TypeError:
                        with torch.cuda.device(index):
                            torch.cuda.reset_peak_memory_stats()
            elif bool(getattr(getattr(torch, "xpu", None), "is_available", lambda: False)()):
                xpu = torch.xpu
                for index in range(int(xpu.device_count())):
                    reset = getattr(xpu, "reset_peak_memory_stats", None)
                    if callable(reset):
                        reset(index)
        except Exception as exc:
            logger.debug(f"Could not reset runtime memory counters: {exc}")

    def _runtime_measurement(self, *, elapsed_seconds):
        measurement = {"elapsedSeconds": max(0.0, float(elapsed_seconds))}
        try:
            torch = import_module("torch")
            if bool(torch.cuda.is_available()):
                index = 0
                hip_version = getattr(getattr(torch, "version", None), "hip", None)
                measurement.update(
                    {
                        "backend": "rocm" if hip_version else "cuda",
                        "device": f"cuda:{index}",
                        "allocatedBytes": int(torch.cuda.memory_allocated(index)),
                        "reservedBytes": int(torch.cuda.memory_reserved(index)),
                        "peakAllocatedBytes": int(torch.cuda.max_memory_allocated(index)),
                        "peakReservedBytes": int(torch.cuda.max_memory_reserved(index)),
                    }
                )
            elif bool(getattr(getattr(torch, "xpu", None), "is_available", lambda: False)()):
                index = 0
                xpu = torch.xpu
                measurement.update({"backend": "xpu", "device": f"xpu:{index}"})
                for key, method_name in (
                    ("allocatedBytes", "memory_allocated"),
                    ("reservedBytes", "memory_reserved"),
                    ("peakAllocatedBytes", "max_memory_allocated"),
                    ("peakReservedBytes", "max_memory_reserved"),
                ):
                    method = getattr(xpu, method_name, None)
                    if callable(method):
                        measurement[key] = int(method(index))
            elif bool(getattr(getattr(torch, "backends", None), "mps", None)) and torch.backends.mps.is_available():
                mps = getattr(torch, "mps", None)
                measurement.update({"backend": "mps", "device": "mps:0"})
                current = getattr(mps, "current_allocated_memory", None)
                driver = getattr(mps, "driver_allocated_memory", None)
                if callable(current):
                    measurement["allocatedBytes"] = int(current())
                if callable(driver):
                    measurement["driverAllocatedBytes"] = int(driver())
            else:
                measurement.update({"backend": "cpu", "device": "cpu:0"})
        except Exception as exc:
            measurement["acceleratorMeasurementError"] = str(exc)
        try:
            import psutil

            measurement["processRssBytes"] = int(psutil.Process().memory_info().rss)
        except Exception:
            pass
        return measurement

    def _release_runtime_caches_for_retry(self):
        errors = []
        released = {
            "nodes": len(self.node_cache),
            "models": 0,
            "diffusers_components": 0,
            "offload_files": 0,
        }

        try:
            released["models"] = memory_manager.clear()
        except Exception as e:
            errors.append(f"memory manager: {e}")
            try:
                memory_manager.cache.clear()
            except Exception as clear_error:
                errors.append(f"memory manager fallback: {clear_error}")

        # Detach MemoryManager ownership first. Node destructors otherwise call
        # remove(), which may try to materialize an offloaded pipeline on CPU
        # while its accelerator allocation is still live.
        try:
            self.node_cache.clear()
        except Exception as e:
            errors.append(f"node cache: {e}")

        released["diffusers_components"], diffusers_errors = self._release_modular_diffusers_components()
        errors.extend(diffusers_errors)
        released["offload_files"], offload_errors = self._release_diffusers_offload_cache()
        errors.extend(offload_errors)

        try:
            gc.collect()
        except Exception as e:
            errors.append(f"gc.collect: {e}")
        errors.extend(self._best_effort_device_cache_clear())
        allocator_trimmed, allocator_errors = self._best_effort_allocator_trim()
        errors.extend(allocator_errors)

        return {
            "released": released,
            "allocatorTrimmed": allocator_trimmed,
            "errors": errors,
        }

    @staticmethod
    def _auto_candidate_minimums(runtime_hints):
        if not isinstance(runtime_hints, dict):
            return {}
        candidate = runtime_hints.get("autoResourcePlan")
        if not isinstance(candidate, dict):
            return {}
        requirements = candidate.get("requirements")
        if not isinstance(requirements, dict):
            return {}
        minimum = requirements.get("minimum")
        return minimum if isinstance(minimum, dict) else {}

    @staticmethod
    def _auto_candidate_cache_signature(runtime_hints):
        if not isinstance(runtime_hints, dict):
            return None
        candidate = runtime_hints.get("autoResourcePlan")
        if not isinstance(candidate, dict):
            return None
        payload = {
            "modelType": candidate.get("modelType") or runtime_hints.get("modelType"),
            "artifact": (
                candidate.get("resolvedArtifact")
                or candidate.get("artifact")
                or candidate.get("modelRepo")
                or runtime_hints.get("resolvedArtifact")
                or runtime_hints.get("modelRepo")
            ),
            "loaderModule": candidate.get("loaderModule") or runtime_hints.get("loaderModule"),
            "loaderAction": candidate.get("loaderAction") or runtime_hints.get("loaderAction"),
            "executionPath": candidate.get("executionPath") or runtime_hints.get("executionPath"),
            "pipelineClass": candidate.get("pipelineClass") or runtime_hints.get("pipelineClass"),
            # The same model/artifact can be represented by an assembled
            # Diffusers pipeline node or by Diffusers component-loader nodes.
            # Those resident objects are not interchangeable.
            "loaderContract": runtime_hints.get("loaderContract"),
            "dtype": candidate.get("dtype") or runtime_hints.get("dtype"),
            "quantizationMode": (candidate.get("quantizationMode") or runtime_hints.get("quantizationMode")),
            "quantizedComponents": sorted(
                str(item)
                for item in (candidate.get("quantizedComponents") or runtime_hints.get("quantizedComponents") or [])
            ),
            "offloadMode": candidate.get("offloadMode") or runtime_hints.get("offloadMode"),
            "deviceMap": candidate.get("deviceMap") or runtime_hints.get("deviceMap"),
            "attentionBackend": candidate.get("attentionBackend") or runtime_hints.get("attentionBackend"),
            "regionalCompile": candidate.get("regionalCompile") or runtime_hints.get("regionalCompile"),
            "denoiserCache": candidate.get("denoiserCache") or runtime_hints.get("denoiserCache"),
            "channelsLast": candidate.get("channelsLast") or runtime_hints.get("channelsLast"),
            "layerwiseCasting": candidate.get("layerwiseCasting") or runtime_hints.get("layerwiseCasting"),
            "controlledArtifacts": runtime_hints.get("controlledArtifacts") or [],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    @staticmethod
    def _graph_loader_contract(nodes):
        if not isinstance(nodes, dict):
            return []
        loader_actions = {"LoadPipeline", "ModelsLoader", "AutoModelLoader"}
        contract = {
            f"{node.get('module', '')}.{node.get('action', '')}"
            for node in nodes.values()
            if isinstance(node, dict) and node.get("action") in loader_actions
        }
        return sorted(item for item in contract if item != ".")

    def _prepare_auto_runtime_for_graph(self, runtime_hints):
        """Release stale app-owned caches before an Auto run when warranted.

        Same-family cache is intentionally retained while memory has headroom;
        it is useful, not stale. A model-family switch or live RAM/VRAM
        pressure releases all app-owned graph/model caches before loading the
        selected candidate.
        """
        if not isinstance(runtime_hints, dict) or runtime_hints.get("resourceMode") != "auto":
            return None

        candidate = runtime_hints.get("autoResourcePlan")
        previous_family = self._last_auto_model_family
        previous_signature = self._last_auto_resource_signature
        incoming_family = str(
            (candidate.get("modelType") if isinstance(candidate, dict) else None)
            or runtime_hints.get("modelType")
            or ""
        ).strip()
        incoming_signature = self._auto_candidate_cache_signature(runtime_hints)
        has_runtime_cache = bool(self.node_cache or memory_manager.cache)
        resident_recipe_reusable = bool(
            has_runtime_cache
            and previous_family
            and incoming_family
            and previous_family == incoming_family
            and previous_signature
            and incoming_signature
            and previous_signature == incoming_signature
        )
        reasons = []
        if has_runtime_cache and incoming_family and not previous_family:
            reasons.append("cached model family is unknown")
        if has_runtime_cache and previous_family and incoming_family and previous_family != incoming_family:
            reasons.append(f"model family changed from {previous_family} to {incoming_family}")
        if (
            has_runtime_cache
            and previous_family
            and previous_family == incoming_family
            and previous_signature
            and incoming_signature
            and previous_signature != incoming_signature
        ):
            reasons.append(f"Auto resource recipe changed within {incoming_family}")

        minimums = self._auto_candidate_minimums(runtime_hints)
        try:
            hardware = get_hardware_snapshot(self.data_dir, refresh=True)
        except Exception:
            logger.debug("Could not sample resources before Auto execution", exc_info=True)
            hardware = {}

        system = hardware.get("system") if isinstance(hardware, dict) else {}
        available_ram = system.get("ram_available") if isinstance(system, dict) else None
        total_ram = system.get("ram_total") if isinstance(system, dict) else None
        required_ram = minimums.get("systemRamBytes")
        ram_floor = max(
            4 * 1024**3,
            int(total_ram * 0.1) if isinstance(total_ram, int) else 0,
        )
        if has_runtime_cache and isinstance(available_ram, int):
            if available_ram < ram_floor:
                reasons.append("available system memory is below the safety floor")
            elif not resident_recipe_reusable and isinstance(required_ram, int) and available_ram < required_ram:
                reasons.append("available system memory is below the selected candidate minimum")

        cuda_devices = [
            device
            for device in (hardware.get("devices") if isinstance(hardware, dict) else []) or []
            if isinstance(device, dict) and device.get("type") == "cuda"
        ]
        accelerator = cuda_devices[0] if cuda_devices else None
        available_vram = (
            accelerator.get("torch_vram_free") or accelerator.get("vram_free")
            if isinstance(accelerator, dict)
            else None
        )
        total_vram = (
            accelerator.get("torch_vram_total") or accelerator.get("vram_total")
            if isinstance(accelerator, dict)
            else None
        )
        required_vram = minimums.get("vramBytes")
        vram_floor = max(
            2 * 1024**3,
            int(total_vram * 0.1) if isinstance(total_vram, int) else 0,
        )
        if has_runtime_cache and isinstance(available_vram, int):
            if available_vram < vram_floor:
                reasons.append("available accelerator memory is below the safety floor")
            elif not resident_recipe_reusable and isinstance(required_vram, int) and available_vram < required_vram:
                reasons.append("available accelerator memory is below the selected candidate minimum")

        cleanup = self._release_runtime_caches_for_retry() if reasons else None
        if incoming_family:
            self._last_auto_model_family = incoming_family
        if incoming_signature:
            self._last_auto_resource_signature = incoming_signature
        result = {
            "performed": cleanup is not None,
            "reasons": reasons,
            "incomingModelFamily": incoming_family or None,
            "previousModelFamily": previous_family,
            "residentRecipeReusable": resident_recipe_reusable,
            "resourceRecipeChanged": bool(
                previous_signature and incoming_signature and previous_signature != incoming_signature
            ),
            "availableRamBytes": available_ram,
            "availableVramBytes": available_vram,
            "cleanup": cleanup,
        }
        if cleanup is not None:
            self.queue_message(
                {
                    "type": "auto_resource_cleanup",
                    "task_id": self.current_task.get("task_id") if self.current_task else None,
                    **self._current_run_identity_payload(),
                    **result,
                }
            )
        return result

    def _prepare_graph_loops(self, graph):
        raw_loops = graph.get("loops")
        if raw_loops in (None, []):
            return {"loops": [], "by_node": {}}
        if not isinstance(raw_loops, list):
            raise ValueError("Graph loops must be a list.")

        nodes = graph.get("nodes") if isinstance(graph.get("nodes"), dict) else {}
        paths = graph.get("paths") if isinstance(graph.get("paths"), list) else []
        global_order = []
        for path in paths:
            if not isinstance(path, list):
                continue
            for node_id in path:
                if node_id in nodes and node_id not in global_order:
                    global_order.append(node_id)

        prepared = []
        for position, item in enumerate(raw_loops):
            if not isinstance(item, dict):
                raise ValueError(f"Loop {position + 1} must be an object.")
            loop_id = str(item.get("id") or "").strip()
            if not loop_id:
                raise ValueError(f"Loop {position + 1} needs an id.")
            body_ids = [str(value) for value in item.get("bodyNodeIds") or []]
            body_ids = list(dict.fromkeys(body_ids))
            if not body_ids:
                raise ValueError(f"Loop {loop_id} needs at least one body node.")
            missing = [node_id for node_id in body_ids if node_id not in nodes]
            if missing:
                raise ValueError(f"Loop {loop_id} references missing body nodes: {', '.join(missing)}.")
            max_iterations = max(1, min(10000, int(item.get("maxIterations") or 100)))
            iterations = int(item.get("iterations") or 1)
            if iterations < 1 or iterations > max_iterations:
                raise ValueError(f"Loop {loop_id} iterations must be between 1 and its maximum of {max_iterations}.")
            ordered_body = [node_id for node_id in global_order if node_id in body_ids]
            if set(ordered_body) != set(body_ids):
                unresolved = sorted(set(body_ids) - set(ordered_body))
                raise ValueError(f"Loop {loop_id} body is not present in executable paths: {', '.join(unresolved)}.")

            input_id = str(item.get("inputNodeId") or "").strip() or None
            index_id = str(item.get("indexNodeId") or "").strip() or None
            item_id = str(item.get("itemNodeId") or "").strip() or None
            result_id = str(item.get("resultNodeId") or "").strip() or None
            for label, boundary_id in (
                ("input", input_id),
                ("index", index_id),
                ("items", item_id),
                ("result", result_id),
            ):
                if boundary_id and boundary_id not in body_ids:
                    raise ValueError(f"Loop {loop_id} {label} node must be inside its visual container.")
            if not result_id:
                raise ValueError(f"Loop {loop_id} needs one Loop Result node.")
            iteration_mode = str(item.get("iterationMode") or "count")
            if iteration_mode not in {"count", "collection"}:
                raise ValueError(f"Loop {loop_id} has unsupported iteration mode {iteration_mode!r}.")
            if iteration_mode == "collection" and not item_id:
                raise ValueError(f"Loop {loop_id} needs a Loop Items node in collection mode.")
            ordered_body = [node_id for node_id in ordered_body if node_id != result_id] + [result_id]

            body_set = set(body_ids)
            for target_id, target in nodes.items():
                if target_id in body_set and target_id != result_id:
                    for param in (target.get("params") or {}).values():
                        if isinstance(param, dict) and param.get("sourceId") == result_id:
                            raise ValueError(
                                f"Loop {loop_id} result cannot feed another node inside the same iteration; "
                                "use Loop Input for carried state."
                            )
                if target_id in body_set:
                    continue
                for param in (target.get("params") or {}).values():
                    if not isinstance(param, dict):
                        continue
                    source_id = param.get("sourceId")
                    if source_id in body_set and source_id != result_id:
                        raise ValueError(
                            f"Loop {loop_id} can only expose values through its Loop Result node; "
                            f"{target_id} reads directly from {source_id}."
                        )

            prepared_loop = {
                "id": loop_id,
                "body": ordered_body,
                "iterations": iterations,
                "max_iterations": max_iterations,
                "input_id": input_id,
                "index_id": index_id,
                "item_id": item_id,
                "result_id": result_id,
                "iteration_mode": iteration_mode,
                "carry": bool(item.get("carry", True)),
                "collect": bool(item.get("collect", True)),
                "durable": bool(item.get("durable", False)),
                "max_retries": max(0, min(10, int(item.get("maxRetries") or 0))),
            }
            prepared.append(prepared_loop)
        body_sets = {loop["id"]: set(loop["body"]) for loop in prepared}
        for index, loop in enumerate(prepared):
            for other in prepared[index + 1 :]:
                left = body_sets[loop["id"]]
                right = body_sets[other["id"]]
                overlap = left & right
                if overlap and not (left < right or right < left):
                    raise ValueError(
                        f"Loop {other['id']} overlaps another loop at: {', '.join(sorted(overlap))}. "
                        "Nested loop bodies must be strictly contained rather than partially overlapping."
                    )

        loops_by_id = {loop["id"]: loop for loop in prepared}
        for loop in prepared:
            supersets = [other for other in prepared if body_sets[loop["id"]] < body_sets[other["id"]]]
            parent = min(supersets, key=lambda item: len(body_sets[item["id"]])) if supersets else None
            loop["parent_id"] = parent["id"] if parent else None
            loop["child_by_node"] = {}
        for child in prepared:
            if not child["parent_id"]:
                continue
            parent = loops_by_id[child["parent_id"]]
            for node_id in child["body"]:
                parent["child_by_node"][node_id] = child

        root_by_node = {}
        for loop in prepared:
            root = loop
            while root["parent_id"]:
                root = loops_by_id[root["parent_id"]]
            for node_id in loop["body"]:
                root_by_node[node_id] = root
        return {"loops": prepared, "by_node": root_by_node, "loops_by_id": loops_by_id}

    def _durable_loop_checkpoint_path(self, loop, checkpoint_key):
        """Resolve one input-scoped checkpoint without exposing identity in its filename."""

        if not loop.get("durable"):
            return None
        runtime_hints = (self.current_task or {}).get("runtimeHints") or {}
        workflow_id = str(runtime_hints.get("workflowTabId") or "").strip()
        run_input_hash = str(runtime_hints.get("runInputHash") or "").strip()
        if not workflow_id or not run_input_hash:
            raise ValueError(
                f"Durable loop {loop['id']} needs workflowTabId and runInputHash runtime identity."
            )
        identity = json.dumps(
            [workflow_id, run_input_hash, checkpoint_key],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        data_dir = Path(getattr(self, "data_dir", "data")).resolve()
        return data_dir / "runtime" / "loop-checkpoints" / f"{digest}.json"

    @staticmethod
    def _durable_loop_asset(value):
        """Validate and normalize the retained-video value a durable loop may save."""

        if not isinstance(value, dict) or value.get("storage") != "file" or value.get("media_type") != "video":
            raise ValueError("Durable loops may checkpoint only retained file-backed video assets.")
        from modiff.media_assets import asset_root, coerce_video_asset

        asset = coerce_video_asset(value)
        path = Path(asset["path"]).resolve()
        try:
            path.relative_to(asset_root())
        except ValueError as exc:
            raise ValueError("Durable loop assets must be inside MoDiff's retained-media directory.") from exc
        projected = {
            key: asset[key]
            for key in (
                "schema_version",
                "asset_id",
                "storage",
                "path",
                "media_type",
                "width",
                "height",
                "fps",
                "frame_count",
                "duration_seconds",
                "task_id",
                "temporary",
                "pinned",
                "created_at",
                "updated_at",
                "source_asset_ids",
                "operation",
            )
            if key in asset
        }
        encoded = json.dumps(projected, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > 65536:
            raise ValueError("Durable loop asset metadata exceeds the 64 KiB checkpoint limit.")
        return projected

    def _load_durable_loop_checkpoint(self, path, signature):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(f"Durable loop checkpoint {path.name} is unreadable.") from exc
        if not isinstance(value, dict) or value.get("schema_version") != 1 or value.get("signature") != signature:
            raise ValueError(f"Durable loop checkpoint {path.name} does not match this loop contract.")
        try:
            next_index = int(value["next_index"])
            collection = [self._durable_loop_asset(item) for item in value.get("collection") or []]
            carry_value = value.get("carry_value")
            carry_value = self._durable_loop_asset(carry_value) if carry_value is not None else None
            stopped = bool(value.get("stopped"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Durable loop checkpoint {path.name} contains unusable retained media.") from exc
        if next_index < 0 or next_index != len(collection):
            raise ValueError(f"Durable loop checkpoint {path.name} has inconsistent iteration metadata.")
        return {
            "signature": signature,
            "next_index": next_index,
            "collection": collection,
            "carry_value": carry_value,
            "stopped": stopped,
            "durable_path": path,
        }

    def _persist_durable_loop_checkpoint(self, checkpoint):
        path = checkpoint.get("durable_path")
        if path is None:
            return
        collection = [self._durable_loop_asset(item) for item in checkpoint["collection"]]
        carry_value = checkpoint.get("carry_value")
        carry_value = self._durable_loop_asset(carry_value) if carry_value is not None else None
        payload = {
            "schema_version": 1,
            "signature": checkpoint["signature"],
            "next_index": int(checkpoint["next_index"]),
            "collection": collection,
            "carry_value": carry_value,
            "stopped": bool(checkpoint["stopped"]),
            "updated_at": time.time(),
        }
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > 8 * 1024 * 1024:
            raise ValueError("Durable loop checkpoint exceeds the 8 MiB metadata limit.")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def _clear_loop_checkpoints(self, task_id):
        checkpoints = self.__dict__.setdefault("_loop_checkpoints", {}).pop(str(task_id), {})
        for checkpoint in checkpoints.values():
            path = checkpoint.get("durable_path") if isinstance(checkpoint, dict) else None
            if isinstance(path, Path):
                path.unlink(missing_ok=True)

    def _loop_checkpoint(self, loop, *, iterations, scope=""):
        """Return a compatible in-process or retained-media checkpoint."""
        task_id = str((self.current_task or {}).get("task_id") or "session")
        checkpoints = self.__dict__.setdefault("_loop_checkpoints", {})
        if len(checkpoints) > 20:
            oldest_task = next(iter(checkpoints))
            if oldest_task != task_id:
                checkpoints.pop(oldest_task, None)
        task_checkpoints = checkpoints.setdefault(task_id, {})
        checkpoint_key = f"{scope}/{loop['id']}" if scope else loop["id"]
        checkpoint = task_checkpoints.get(checkpoint_key)
        signature = {
            "iteration_mode": loop["iteration_mode"],
            "iterations": int(iterations),
            "carry": bool(loop["carry"]),
            "collect": bool(loop["collect"]),
            "durable": bool(loop.get("durable")),
        }
        if not isinstance(checkpoint, dict) or checkpoint.get("signature") != signature:
            path = self._durable_loop_checkpoint_path(loop, checkpoint_key)
            checkpoint = self._load_durable_loop_checkpoint(path, signature) if path is not None else None
            if checkpoint is None:
                checkpoint = {
                    "signature": signature,
                    "next_index": 0,
                    "collection": [],
                    "carry_value": None,
                    "stopped": False,
                    "durable_path": path,
                }
            task_checkpoints[checkpoint_key] = checkpoint
        return checkpoint

    def _restore_loop_result(self, loop, nodes, sid, checkpoint):
        """Recreate the lightweight result boundary after a cache-clearing graph retry."""
        result_id = loop["result_id"]
        if result_id not in self.node_cache:
            result_node = nodes[result_id]
            work_module = import_module(f"{result_node['module']}.main")
            self.node_cache[result_id] = getattr(work_module, result_node["action"])(result_id)
        result_node = self.node_cache[result_id]
        result_node._sid = sid
        result_node.output = {
            "collection": list(checkpoint["collection"]) if loop["collect"] else [],
            "value": checkpoint["carry_value"],
            "stopped": bool(checkpoint["stopped"]),
        }
        return result_node

    def _execute_graph_loop(self, loop, nodes, sid, checkpoint_scope=""):
        iterations = loop["iterations"]
        if loop["iteration_mode"] == "collection":
            self.execute_node(
                loop["item_id"],
                nodes[loop["item_id"]],
                sid,
                param_overrides={"item_index": 0},
            )
            iterations = int(self.node_cache[loop["item_id"]].output.get("count") or 0)
            if iterations < 1:
                raise ValueError(f"Loop {loop['id']} cannot iterate an empty collection.")
            if iterations > loop["max_iterations"]:
                raise ValueError(
                    f"Loop {loop['id']} collection has {iterations} items, above its maximum of {loop['max_iterations']}."
                )
        checkpoint = self._loop_checkpoint(loop, iterations=iterations, scope=checkpoint_scope)
        collected = list(checkpoint["collection"])
        carry_value = checkpoint["carry_value"]
        stopped = bool(checkpoint["stopped"])
        completed_iterations = min(int(checkpoint["next_index"]), iterations)
        if completed_iterations:
            self.queue_message(
                {
                    "type": "progress",
                    "node": loop["id"],
                    "task_id": self.current_task.get("task_id") if self.current_task else None,
                    "attempt_index": self.current_task.get("attempt_index") if self.current_task else None,
                    **self._current_run_identity_payload(),
                    "status": "running",
                    "phase": "loop",
                    "message": f"Resuming after {completed_iterations} completed iteration(s)",
                    "progress": int(completed_iterations / iterations * 100),
                    "current_step": completed_iterations,
                    "total_steps": iterations,
                }
            )
        for index in range(completed_iterations, iterations):
            if self.interrupt_flag:
                raise InterruptedError(f"Loop {loop['id']} was interrupted before iteration {index + 1}.")
            runtime_limit = ((self.current_task or {}).get("runtimeHints") or {}).get("maxRuntimeSeconds")
            started_at = (self.current_task or {}).get("started_at")
            if runtime_limit and started_at and time.time() - float(started_at) >= float(runtime_limit):
                raise TimeoutError(
                    f"Loop {loop['id']} reached the configured {int(runtime_limit)} second runtime limit."
                )
            self.queue_message(
                {
                    "type": "progress",
                    "node": loop["id"],
                    "task_id": self.current_task.get("task_id") if self.current_task else None,
                    "attempt_index": self.current_task.get("attempt_index") if self.current_task else None,
                    **self._current_run_identity_payload(),
                    "status": "running",
                    "phase": "loop",
                    "message": f"Iteration {index + 1}/{iterations}",
                    "progress": int(index / iterations * 100),
                    "current_step": index + 1,
                    "total_steps": iterations,
                }
            )
            retry = 0
            while True:
                try:
                    executed_children = set()
                    for node_id in loop["body"]:
                        child_loop = loop.get("child_by_node", {}).get(node_id)
                        if child_loop is not None:
                            if child_loop["id"] not in executed_children:
                                child_scope = f"{checkpoint_scope}/{loop['id']}:{index}".strip("/")
                                self._execute_graph_loop(child_loop, nodes, sid, checkpoint_scope=child_scope)
                                executed_children.add(child_loop["id"])
                            continue
                        overrides = None
                        if node_id == loop["index_id"]:
                            overrides = {"index_value": index, "iteration_count": iterations}
                        elif node_id == loop["item_id"]:
                            overrides = {"item_index": index}
                        elif node_id == loop["input_id"] and index > 0 and loop["carry"]:
                            overrides = {"initial": carry_value}
                        self.execute_node(node_id, nodes[node_id], sid, param_overrides=overrides)
                    break
                except InterruptedError:
                    raise
                except Exception:
                    if retry >= loop["max_retries"]:
                        raise
                    retry += 1
                    for node_id in loop["body"]:
                        self.node_cache.pop(node_id, None)
                    self.queue_message(
                        {
                            "type": "progress",
                            "node": loop["id"],
                            "task_id": self.current_task.get("task_id") if self.current_task else None,
                            "attempt_index": self.current_task.get("attempt_index") if self.current_task else None,
                            **self._current_run_identity_payload(),
                            "status": "running",
                            "phase": "loop",
                            "message": f"Retrying iteration {index + 1}/{iterations} ({retry}/{loop['max_retries']})",
                            "progress": int(index / iterations * 100),
                            "current_step": index + 1,
                            "total_steps": iterations,
                        }
                    )

            result = self.node_cache[loop["result_id"]].output
            carry_value = result.get("value")
            collected.append(carry_value)
            stopped = bool(result.get("stopped"))
            completed_iterations = index + 1
            checkpoint.update(
                {
                    "next_index": completed_iterations,
                    "collection": list(collected),
                    "carry_value": carry_value,
                    "stopped": stopped,
                }
            )
            self._persist_durable_loop_checkpoint(checkpoint)
            if stopped:
                break

        result_node = self._restore_loop_result(loop, nodes, sid, checkpoint)
        result_node.output["collection"] = collected if loop["collect"] else []
        result_node.output["value"] = carry_value
        result_node.output["stopped"] = stopped
        self.queue_message(
            {
                "type": "progress",
                "node": loop["id"],
                "task_id": self.current_task.get("task_id") if self.current_task else None,
                "attempt_index": self.current_task.get("attempt_index") if self.current_task else None,
                **self._current_run_identity_payload(),
                "status": "succeeded",
                "phase": "loop",
                "message": f"Completed {completed_iterations} iteration(s)",
                "progress": 100,
                "current_step": completed_iterations,
                "total_steps": iterations,
            }
        )
        return {"iterations": completed_iterations, "stopped": stopped, "collection": collected}

    def _capture_execution_process_state(self):
        """Capture process-wide RNG and Torch backend settings changed by a run."""

        state = {
            "python_random": random.getstate(),
            "python_hash_seed_present": "PYTHONHASHSEED" in os.environ,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        }
        try:
            np = import_module("numpy")
            state["numpy_module"] = np
            state["numpy_random"] = np.random.get_state()
        except Exception:
            pass
        try:
            torch = import_module("torch")
            state["torch_module"] = torch
            get_rng_state = getattr(torch, "get_rng_state", None)
            if callable(get_rng_state):
                state["torch_rng"] = get_rng_state()
            deterministic_probe = getattr(torch, "are_deterministic_algorithms_enabled", None)
            if callable(deterministic_probe):
                state["torch_deterministic_algorithms"] = bool(deterministic_probe())
            warn_only_probe = getattr(torch, "is_deterministic_algorithms_warn_only_enabled", None)
            if callable(warn_only_probe):
                state["torch_deterministic_warn_only"] = bool(warn_only_probe())

            cuda = getattr(torch, "cuda", None)
            is_initialized = getattr(cuda, "is_initialized", None)
            get_cuda_rng = getattr(cuda, "get_rng_state_all", None)
            if callable(is_initialized) and is_initialized() and callable(get_cuda_rng):
                state["torch_cuda_rng"] = get_cuda_rng()

            cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
            if cudnn is not None:
                for name in ("benchmark", "deterministic", "allow_tf32"):
                    if hasattr(cudnn, name):
                        state[f"cudnn_{name}"] = getattr(cudnn, name)
            matmul = getattr(getattr(getattr(torch, "backends", None), "cuda", None), "matmul", None)
            if matmul is not None and hasattr(matmul, "allow_tf32"):
                state["cuda_matmul_allow_tf32"] = matmul.allow_tf32
        except Exception:
            pass
        return state

    def _restore_execution_process_state(self, state, graph):
        try:
            random.setstate(state["python_random"])
            if state.get("python_hash_seed_present"):
                os.environ["PYTHONHASHSEED"] = state.get("python_hash_seed") or ""
            else:
                os.environ.pop("PYTHONHASHSEED", None)

            np = state.get("numpy_module")
            if np is not None and "numpy_random" in state:
                np.random.set_state(state["numpy_random"])

            torch = state.get("torch_module")
            if torch is None:
                return
            set_rng_state = getattr(torch, "set_rng_state", None)
            if callable(set_rng_state) and "torch_rng" in state:
                set_rng_state(state["torch_rng"])
            set_cuda_rng = getattr(getattr(torch, "cuda", None), "set_rng_state_all", None)
            if callable(set_cuda_rng) and "torch_cuda_rng" in state:
                set_cuda_rng(state["torch_cuda_rng"])

            deterministic = state.get("torch_deterministic_algorithms")
            deterministic_setter = getattr(torch, "use_deterministic_algorithms", None)
            if deterministic is not None and callable(deterministic_setter):
                deterministic_setter(
                    deterministic,
                    warn_only=bool(state.get("torch_deterministic_warn_only", False)),
                )
            cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
            if cudnn is not None:
                for name in ("benchmark", "deterministic", "allow_tf32"):
                    key = f"cudnn_{name}"
                    if key in state:
                        setattr(cudnn, name, state[key])
            matmul = getattr(getattr(getattr(torch, "backends", None), "cuda", None), "matmul", None)
            if matmul is not None and "cuda_matmul_allow_tf32" in state:
                matmul.allow_tf32 = state["cuda_matmul_allow_tf32"]

            runtime_hints = graph.get("runtimeHints") if isinstance(graph, dict) else None
            cuda_index = self._cuda_index_from_runtime_hints(runtime_hints)
            cuda = getattr(torch, "cuda", None)
            if cuda_index is not None and callable(getattr(cuda, "set_per_process_memory_fraction", None)):
                cuda.set_per_process_memory_fraction(1.0, cuda_index)
        except Exception as exc:
            logger.warning("Could not fully restore process-wide execution settings: %s", exc)

    def execute_graph(self, graph):
        assert_optional_runtime_ready(graph_optional_runtime_requirement(graph))
        process_state = self._capture_execution_process_state()
        try:
            return self._execute_graph(graph)
        finally:
            self._restore_execution_process_state(process_state, graph)

    def _execute_graph(self, graph):
        sid = graph["sid"]
        nodes = graph["nodes"]
        paths = graph["paths"]
        self._active_graph_node_ids = set(nodes)
        graph_loops = self._prepare_graph_loops(graph)

        graph_execution_time = time.time()
        base_runtime_hints = self._coerce_runtime_hints(graph.get("runtimeHints"))
        assert_studio_execution_graph(graph, base_runtime_hints)
        if isinstance(base_runtime_hints, dict):
            self._bind_controlled_artifact_receipts(graph, base_runtime_hints)
            base_runtime_hints["loaderContract"] = self._graph_loader_contract(nodes)
            graph["runtimeHints"] = deepcopy(base_runtime_hints)
        auto_runtime_preparation = self._prepare_auto_runtime_for_graph(base_runtime_hints)
        if self.current_task and auto_runtime_preparation is not None:
            self.current_task["autoRuntimePreparation"] = auto_runtime_preparation
        retry_plans = self._coerce_retry_plan_list(base_runtime_hints)
        retry_history = []
        deterministic_receipt = None
        attempt_index = 0
        retry_plan_index = -1

        while True:
            if self.current_task:
                self.current_task["attempt_index"] = attempt_index
                self.current_task["progress"] = 0
            runtime_hints = deepcopy(base_runtime_hints) if base_runtime_hints else None
            active_retry_plan = (
                retry_plans[retry_plan_index]
                if retry_plan_index >= 0 and retry_plan_index < len(retry_plans)
                else None
            )
            if self.current_task and runtime_hints is not None:
                self.current_task["runtimeHints"] = runtime_hints
            if runtime_hints and active_retry_plan is None and attempt_index == 0:
                self._assert_auto_resource_candidate_ready(runtime_hints)
                auto_plan = runtime_hints.get("autoResourcePlan")
                if isinstance(auto_plan, dict):
                    proven = self._auto_resource_candidate_is_proven(auto_plan)
                    # Every Auto plan must address one exact visible loader,
                    # even when qualification is still advisory.  Simulate
                    # unproven plans on a copy so validation cannot mutate the
                    # user's graph.
                    target_graph = graph if proven else deepcopy(graph)
                    updated_nodes = self._apply_resource_retry_plan_to_graph(target_graph, auto_plan)
                    if proven and updated_nodes:
                        self.queue_message(
                            {
                                "type": "auto_resource_plan_applied",
                                "sid": sid,
                                "task_id": self.current_task.get("task_id") if self.current_task else None,
                                "attempt": attempt_index,
                                "attempt_index": attempt_index,
                                "candidateId": auto_plan.get("id"),
                                "updatedNodes": updated_nodes,
                                "message": "Applied the proven Auto resource plan before execution.",
                            }
                        )
            if runtime_hints and active_retry_plan:
                updated_nodes = self._apply_resource_retry_plan_to_graph(graph, active_retry_plan)
                retry_mode = active_retry_plan.get("offloadMode")
                if isinstance(retry_mode, str):
                    runtime_hints["offloadMode"] = retry_mode
                    runtime_hints["autoOffload"] = retry_mode != OFFLOAD_MODE_NONE
                    runtime_hints["offloadDiskPath"] = (
                        "data/offload/diffusers" if retry_mode == OFFLOAD_MODE_GROUP_DISK else None
                    )
                if isinstance(active_retry_plan.get("modelRepo"), str):
                    runtime_hints["modelRepo"] = active_retry_plan["modelRepo"]
                    runtime_hints["resolvedModelRepo"] = active_retry_plan["modelRepo"]
                if isinstance(active_retry_plan.get("resolvedArtifact"), str):
                    runtime_hints["resolvedArtifact"] = active_retry_plan["resolvedArtifact"]
                elif isinstance(active_retry_plan.get("modelRepo"), str):
                    runtime_hints["resolvedArtifact"] = active_retry_plan["modelRepo"]
                if isinstance(active_retry_plan.get("executionPath"), str):
                    runtime_hints["executionPath"] = active_retry_plan["executionPath"]
                if isinstance(active_retry_plan.get("quantizationMode"), str):
                    runtime_hints["quantizationMode"] = active_retry_plan["quantizationMode"]
                if isinstance(active_retry_plan.get("quantizedComponents"), list):
                    runtime_hints["quantizedComponents"] = [
                        str(item) for item in active_retry_plan["quantizedComponents"]
                    ]
                runtime_hints["resourceRetryAttempt"] = attempt_index
                runtime_hints["resourceRetryHistory"] = retry_history
                plan = runtime_hints.get("resourcePlan")
                if isinstance(plan, dict):
                    plan["activeRetryPlan"] = self._sanitize_retry_plan_for_hints(active_retry_plan)
                    if isinstance(retry_mode, str):
                        plan["offloadMode"] = retry_mode
                        plan["autoOffload"] = retry_mode != OFFLOAD_MODE_NONE
                retry_message = (
                    f"Retrying with validated resource plan {retry_plan_index + 1} "
                    f"after {retry_history[-1]['errorCode'] if retry_history else 'resource pressure'}."
                )
                retry_progress = self.record_node_progress(
                    {
                        "type": "progress",
                        "node": self.current_task.get("current_node") if self.current_task else None,
                        "task_id": self.current_task.get("task_id") if self.current_task else None,
                        "attempt_index": attempt_index,
                        **self._current_run_identity_payload(),
                        "status": "running",
                        "phase": "retry",
                        "message": retry_message,
                        "progress": -1,
                    }
                )
                self.queue_message(retry_progress)
                self.queue_message(
                    {
                        "type": "resource_retry",
                        "sid": sid,
                        "task_id": self.current_task.get("task_id") if self.current_task else None,
                        "attempt": attempt_index,
                        "attempt_index": attempt_index,
                        "offloadMode": retry_mode,
                        "retryPlan": self._sanitize_retry_plan_for_hints(active_retry_plan),
                        "updatedNodes": updated_nodes,
                        "message": retry_message,
                        "history": retry_history,
                    }
                )

            runtime_budget = self._apply_cuda_runtime_budget(runtime_hints)
            deterministic = self._apply_deterministic_mode(graph)
            if deterministic is not None:
                deterministic_receipt = deterministic
            runtime_fingerprint = self._runtime_fingerprint()
            if self.current_task:
                self.current_task["runtimeFingerprint"] = runtime_fingerprint.get("fingerprint")
                if deterministic is not None:
                    self.current_task["deterministicMode"] = deterministic
                self.current_task["runtimeHints"] = runtime_hints
                self.current_task["runtimeBudget"] = runtime_budget

            if deterministic:
                self.queue_message(
                    {
                        "type": "deterministic_execution",
                        "sid": sid,
                        "task_id": self.current_task.get("task_id") if self.current_task else None,
                        "attempt_index": attempt_index,
                        "deterministicMode": deterministic,
                        "runtimeFingerprint": runtime_fingerprint,
                    }
                )

            node_weights = {
                id: node_execution_weight(nodes[id].get("module", ""), nodes[id].get("action", ""))
                for path in paths
                for id in path
                if id in nodes
            }
            total_task_weight = sum(node_weights.values()) or 1
            task_progress = 0.0
            attempt_started_at = time.monotonic()
            self._reset_runtime_measurement()
            executed_loops = set()

            try:
                for path in paths:
                    for id in path:
                        if self.interrupt_flag:
                            if self.current_task:
                                self.current_task["interrupt_requested"] = True
                            return

                        if self.current_task:
                            node = nodes.get(id, {})
                            module = node.get("module", "")
                            action = node.get("action", "")
                            node_started_at = time.time()
                            next_phase = node_execution_phase(module, action)
                            prior_phase = self.current_task.get("phase")
                            prior_phase_started_at = self.current_task.get("_phase_started_at")
                            phase_timings = self.current_task.setdefault("phase_timings", {})
                            if (
                                prior_phase
                                and prior_phase != next_phase
                                and isinstance(phase_timings, dict)
                                and isinstance(prior_phase_started_at, (int, float))
                            ):
                                phase_timings[prior_phase] = float(phase_timings.get(prior_phase, 0.0)) + max(
                                    0.0, node_started_at - float(prior_phase_started_at)
                                )
                            self.current_task.update(
                                {
                                    "updated_at": node_started_at,
                                    "last_heartbeat_at": node_started_at,
                                    "current_node": id,
                                    "current_node_name": f"{module}.{action}",
                                    "node_progress": -1,
                                    "phase": next_phase,
                                    "_phase_started_at": node_started_at,
                                    "message": node_execution_message(module, action, next_phase),
                                    "current_step": None,
                                    "total_steps": None,
                                    "component": None,
                                    "shard_current": None,
                                    "shard_total": None,
                                    "elapsed_seconds": None,
                                    "average_step_seconds": None,
                                    "eta_seconds": None,
                                    "completed_progress": task_progress,
                                    "current_node_weight": node_weights.get(id, 1.0) / total_task_weight * 100,
                                }
                            )
                            self._persist_supervisor_queue_state(force=True)
                        graph_loop = graph_loops["by_node"].get(id)
                        if graph_loop is not None:
                            loop_id = graph_loop["id"]
                            if loop_id in executed_loops:
                                continue
                            self._execute_graph_loop(graph_loop, nodes, sid)
                            executed_loops.add(loop_id)
                        else:
                            self.execute_node(id, nodes[id], sid)

                        # broadcast the task progress
                        if self.current_task:
                            task_progress += node_weights.get(id, 1.0) / total_task_weight * 100
                            self.current_task["progress"] = int(task_progress)
                            self.current_task["completed_progress"] = task_progress
                            self.current_task["node_progress"] = 100
                            self.current_task["updated_at"] = time.time()
                        self.queue_message(
                            {
                                "type": "task_progress",
                                "task_id": self.current_task["task_id"],
                                "attempt_index": self.current_task.get("attempt_index"),
                                **self._current_run_identity_payload(),
                                "progress": self.current_task["progress"],
                            }
                        )
            except Exception as e:
                classification = self._classify_exception(e)
                next_retry_plan_index = None
                pinned_retry_plan_indexes = []
                if classification["error_code"] != "cuda_context_poisoned":
                    next_retry_plan_index, pinned_retry_plan_indexes = self._next_applicable_retry_plan_index(
                        graph,
                        retry_plans,
                        retry_plan_index,
                        classification,
                    )
                can_retry = next_retry_plan_index is not None
                if not can_retry:
                    if pinned_retry_plan_indexes:
                        setattr(e, "modiff_error_code", "auto_retry_requires_override_approval")
                        setattr(e, "modiff_category", "auto_resource")
                        setattr(
                            e,
                            "modiff_recovery_hint",
                            "A safer Auto retry would change one or more pinned workflow fields. "
                            "Reset the conflicting field to Auto or change it explicitly, then retry.",
                        )
                        self.queue_message(
                            {
                                "type": "auto_retry_requires_approval",
                                "sid": sid,
                                "task_id": self.current_task.get("task_id") if self.current_task else None,
                                "attempt_index": attempt_index,
                                **self._current_run_identity_payload(),
                                "node": getattr(e, "modiff_node_id", None),
                                "message": getattr(e, "modiff_recovery_hint"),
                                "retryPlans": [
                                    self._sanitize_retry_plan_for_hints(retry_plans[index])
                                    for index in pinned_retry_plan_indexes
                                ],
                            }
                        )
                    raise

                retry_history.append(
                    {
                        "attempt": attempt_index,
                        "offloadMode": runtime_hints.get("offloadMode") if runtime_hints else None,
                        "retryPlan": self._sanitize_retry_plan_for_hints(active_retry_plan),
                        "category": classification.get("category"),
                        "errorCode": classification.get("error_code"),
                        "error": str(e) or type(e).__name__,
                        "node": getattr(e, "modiff_node_id", None),
                        "nodeName": getattr(e, "modiff_node_name", None),
                        "loaderDiagnostics": self._loader_diagnostics_snapshot(),
                        "nextRetryPlan": self._sanitize_retry_plan_for_hints(retry_plans[next_retry_plan_index]),
                    }
                )
                if runtime_hints is not None:
                    runtime_hints["resourceRetryLastError"] = str(e) or type(e).__name__
                    runtime_hints["resourceRetryLastCode"] = classification.get("error_code")
                    runtime_hints["resourceRetryHistory"] = retry_history
                    if self.current_task:
                        self.current_task["runtimeHints"] = runtime_hints

                cleanup_progress = self.record_node_progress(
                    {
                        "type": "progress",
                        "node": getattr(e, "modiff_node_id", None)
                        or (self.current_task.get("current_node") if self.current_task else None),
                        "task_id": self.current_task.get("task_id") if self.current_task else None,
                        "attempt_index": attempt_index,
                        **self._current_run_identity_payload(),
                        "status": "running",
                        "phase": "cleanup",
                        "message": "Releasing failed-attempt model and accelerator caches before retry",
                        "progress": -1,
                    }
                )
                self.queue_message(cleanup_progress)
                cleanup = self._release_runtime_caches_for_retry()
                self.queue_message(
                    {
                        "type": "resource_retry_cleanup",
                        "sid": sid,
                        "task_id": self.current_task.get("task_id") if self.current_task else None,
                        "attempt": attempt_index,
                        "attempt_index": attempt_index,
                        **cleanup,
                    }
                )
                attempt_index += 1
                retry_plan_index = next_retry_plan_index
                continue

            # the graph has completed
            runtime_measurement = self._runtime_measurement(
                elapsed_seconds=time.monotonic() - attempt_started_at,
            )
            self._record_auto_resource_success(runtime_hints, runtime_fingerprint, runtime_measurement)
            optimization_receipts = self._record_optimization_observations(
                runtime_hints,
                runtime_fingerprint,
                runtime_measurement,
                graph=graph,
            )
            if self.current_task:
                self.current_task.update(
                    {
                        "runtimeFingerprint": runtime_fingerprint,
                        "resourceCandidateId": (
                            runtime_hints.get("autoResourceCandidateId") if isinstance(runtime_hints, dict) else None
                        ),
                        "runtimeMeasurement": runtime_measurement,
                        "optimizationReceiptIds": [item.get("id") for item in optimization_receipts],
                        "updated_at": time.time(),
                    }
                )
            task_id = str((self.current_task or {}).get("task_id") or "session")
            self._clear_loop_checkpoints(task_id)
            self.queue_message(
                {
                    "type": "graph_completed",
                    "sid": sid,
                    "task_id": self.current_task.get("task_id") if self.current_task else None,
                    **self._current_run_identity_payload(),
                    "executionTime": time.time() - graph_execution_time,
                    "runtimeFingerprint": runtime_fingerprint,
                    "deterministicMode": deterministic_receipt,
                    "runtimeHints": runtime_hints,
                    "runtimeBudget": runtime_budget,
                    "runtimeMeasurement": runtime_measurement,
                    "resourceRetryHistory": retry_history,
                }
            )
            return

    async def stop_execution(self, request):
        # check if there is a current task or any queued task
        if not self.current_task and not self.queued_tasks:
            return web.json_response(
                {
                    "error": True,
                    "message": "Nothing to do. No task is currently running or queued.",
                }
            )

        if self.interrupt_flag:
            return web.json_response(
                {
                    "error": True,
                    "message": "Execution is already set for interruption.",
                }
            )

        cancelled_queued = []
        for task_id, task in list(self.queued_tasks.items()):
            self.queued_tasks.pop(task_id, None)
            self.task_graphs.pop(task_id, None)
            cancelled_queued.append(task_id)
            self.queue_message(
                {
                    "type": "task_cancelled",
                    "task_id": task_id,
                    **self._run_identity_payload(task.get("runtimeHints")),
                    "status": "cancelled",
                    "message": "Cancelled before execution.",
                }
            )

        if not self.current_task:
            self.interrupt_flag = False
            self._persist_supervisor_queue_state(force=True)
            return web.json_response(
                {
                    "error": False,
                    "message": f"Cancelled {len(cancelled_queued)} queued task(s).",
                    "cancelled_queued_task_ids": cancelled_queued,
                    "cleanup_pending": False,
                }
            )

        self.interrupt_flag = True
        if self.current_task:
            self.current_task["interrupt_requested"] = True
            self.current_task["updated_at"] = time.time()
            self.current_task["phase"] = "stopping"
            self.current_task["message"] = "Stopping and releasing runtime resources"
            self._persist_supervisor_queue_state(force=True)
            self.queue_message(
                {
                    "type": "task_progress",
                    "task_id": self.current_task.get("task_id"),
                    **self._current_run_identity_payload(),
                    "status": "running",
                    "phase": "stopping",
                    "progress": self.current_task.get("progress", 0),
                    "message": self.current_task["message"],
                }
            )

        # set the interrupt flag for all the nodes in the cache
        for node in self.node_cache:
            self.node_cache[node]._interrupt = True
            active_pipeline = getattr(self.node_cache[node], "_active_pipeline", None)
            if active_pipeline is not None and hasattr(active_pipeline, "_interrupt"):
                active_pipeline._interrupt = True

        hard_restart_after_ms = self._schedule_forced_restart_if_still_running(
            self.current_task.get("task_id") if self.current_task else None
        )
        return web.json_response(
            {
                "error": False,
                "message": (
                    "Execution is stopping. Runtime resources will be released before the queue can advance."
                    if hard_restart_after_ms is None
                    else "Execution is stopping. If the active model call does not return promptly, "
                    "MoDiff will restart its backend worker to release RAM and VRAM."
                ),
                "task_id": self.current_task.get("task_id") if self.current_task else None,
                "cancelled_queued_task_ids": cancelled_queued,
                "cleanup_pending": True,
                "hard_restart_scheduled": hard_restart_after_ms is not None,
                "hard_restart_after_ms": hard_restart_after_ms,
            }
        )

    def _schedule_forced_restart_if_still_running(self, task_id):
        """Escalate cooperative cancellation by replacing the supervised worker."""
        if not task_id or os.environ.get("MODIFF_WORKER_SUPERVISED") != "1":
            return None
        try:
            grace_seconds = max(0.25, float(os.environ.get("MODIFF_HARD_CANCEL_GRACE_SECONDS", "2")))
        except (TypeError, ValueError):
            grace_seconds = 2.0
        prior = self._forced_restart_timer
        if prior is not None:
            prior.cancel()
        timer = threading.Timer(grace_seconds, self._force_restart_if_task_is_active, args=(task_id,))
        timer.daemon = True
        self._forced_restart_timer = timer
        timer.start()
        return int(grace_seconds * 1000)

    def _force_restart_if_task_is_active(self, task_id):
        current = self.current_task if isinstance(self.current_task, dict) else {}
        if current.get("task_id") != task_id or not current.get("interrupt_requested"):
            return
        logger.error(
            "Task %s did not honor cancellation; replacing the supervised backend worker to release runtime memory.",
            task_id,
        )
        self.queue_message(
            {
                "type": "task_cancelled",
                "task_id": task_id,
                **self._current_run_identity_payload(),
                "status": "cancelled",
                "message": "The backend worker is restarting to finish cancellation and release RAM/VRAM.",
                "backend_restart": True,
            }
        )
        # Give the event-loop queue one brief opportunity to flush the terminal
        # notification. The supervisor immediately replaces this process; the
        # operating system, rather than Python object finalizers, releases all
        # remaining accelerator allocations.
        time.sleep(0.05)
        os._exit(SUPERVISED_RESTART_EXIT_CODE)

    def _connected_output_value(self, *, target_node_id, target_node_name, target_param, source_node_id, source_key):
        source_node = self.node_cache.get(source_node_id)
        if source_node is None:
            error = MissingConnectedOutputError(
                f"Connected input '{target_param}' on {target_node_name} expected output '{source_key}' "
                f"from upstream node {source_node_id}, but that node has not executed."
            )
            setattr(error, "modiff_node_id", source_node_id)
            setattr(error, "modiff_node_name", "Unknown upstream node")
            setattr(error, "modiff_target_node_id", target_node_id)
            setattr(error, "modiff_target_node_name", target_node_name)
            raise error

        output = getattr(source_node, "output", None)
        source_name = (
            f"{getattr(source_node, 'module_name', 'unknown')}.{getattr(source_node, 'class_name', 'unknown')}"
        )
        if not isinstance(output, dict) or source_key not in output or output.get(source_key) is None:
            available = sorted(output.keys()) if isinstance(output, dict) else []
            error = MissingConnectedOutputError(
                f"Connected input '{target_param}' on {target_node_name} expected output '{source_key}' "
                f"from upstream node {source_name}, but that output was not produced. "
                f"Available outputs: {available or 'none'}."
            )
            setattr(error, "modiff_node_id", source_node_id)
            setattr(error, "modiff_node_name", source_name)
            setattr(error, "modiff_target_node_id", target_node_id)
            setattr(error, "modiff_target_node_name", target_node_name)
            raise error

        return output[source_key]

    def _adopt_reusable_loader_node(self, node_id, module, action):
        """Move a compatible loader cache entry to a new workflow node id.

        Workflow node ids are document-local, while an unchanged loader
        contract can safely keep its resident pipeline between sequential
        workflows. NodeBase revalidates every argument on the subsequent call;
        if anything differs, it performs the normal unload/reload path.
        """
        if action not in {"LoadPipeline", "ModelsLoader"}:
            return None
        for cached_id, cached_node in list(self.node_cache.items()):
            if cached_id == node_id or cached_id in self._active_graph_node_ids:
                continue
            if getattr(cached_node, "module_name", None) != module:
                continue
            if getattr(cached_node, "class_name", None) != action:
                continue
            prepare_for_reuse = getattr(cached_node, "prepare_for_workflow_reuse", None)
            if callable(prepare_for_reuse):
                prepare_for_reuse()
            self.node_cache.pop(cached_id, None)
            cached_node.node_id = node_id
            self.node_cache[node_id] = cached_node
            return cached_id
        return None

    def execute_node(self, id, node, sid, quiet=False, param_overrides=None):
        module = node["module"]
        action = node["action"]
        params = node["params"]

        if module not in self.modules:
            raise ValueError(f"Invalid module: {module}")

        if action not in self.modules[module]:
            raise ValueError(f"Invalid action: {action}")

        # get the arguments values
        args = {}
        ui_fields = {}
        upstream_changed = False

        for p in params:
            data_source_id = params[p].get("sourceId")
            data_param_key = params[p].get("sourceKey")

            # the field is a UI element, used mostly to display the data in the UI
            if "display" in params[p] and params[p]["display"] in [
                "ui_group",
                "ui_text",
                "ui_image",
                "ui_imagecompare",
                "ui_areaselect",
                "ui_audio",
                "ui_video",
                "ui_3d",
                "ui_label",
                "ui_button",
            ]:
                ui_fields[p] = data_param_key if data_param_key else None

            # the field is an input that gets its value from an output of another node
            elif data_source_id and data_param_key:
                source_node = self.node_cache.get(data_source_id)
                upstream_changed = upstream_changed or bool(getattr(source_node, "_has_changed", False))
                # spawn field handling
                # if '>>>' in p or self.modules[module][action]['params'][p].get('spawn'):
                if params[p].get("spawn"):
                    spawn_key = p.split(">>>")[0]
                    if not spawn_key in args:
                        args[spawn_key] = []

                    args[spawn_key].append(
                        self._connected_output_value(
                            target_node_id=id,
                            target_node_name=f"{module}.{action}",
                            target_param=p,
                            source_node_id=data_source_id,
                            source_key=data_param_key,
                        )
                    )
                else:
                    args[p] = self._connected_output_value(
                        target_node_id=id,
                        target_node_name=f"{module}.{action}",
                        target_param=p,
                        source_node_id=data_source_id,
                        source_key=data_param_key,
                    )
            # the field is a static value
            else:
                args[p] = params[p].get("value")

        if param_overrides:
            if not isinstance(param_overrides, dict):
                raise TypeError("Node parameter overrides must be a dictionary.")
            args.update(param_overrides)

        # Re-resolve the exact loader from authoritative node arguments at the
        # last boundary before importing its module.  This closes admission to
        # worker and connected-parameter TOCTOU gaps without trusting runtime
        # hints or triggering an install/activation path.
        assert_optional_runtime_ready(
            loader_optional_runtime_requirement(module, action, args)
        )

        if not quiet:
            reset_memory_stats()
            start_time = time.time()
            phase = node_execution_phase(module, action)

            # tell the client that the node is running
            starting_progress = self.record_node_progress(
                {
                    "type": "progress",
                    "node": id,
                    "name": f"{module}.{action}",
                    "task_id": self.current_task.get("task_id") if self.current_task else None,
                    "attempt_index": self.current_task.get("attempt_index") if self.current_task else None,
                    **self._current_run_identity_payload(),
                    "status": "running",
                    "phase": phase,
                    "message": node_execution_message(module, action, phase),
                    "current_node": id,
                    "progress": -1,  # -1 sets the progress to indeterminate
                }
            )
            self.queue_message(starting_progress)

        # if the node is not in the cache, initialize it
        if id not in self.node_cache:
            reused_node_id = self._adopt_reusable_loader_node(id, module, action)
            if reused_node_id is None:
                work_module = import_module(f"{module}.main")
                work_action = getattr(work_module, action)
                self.node_cache[id] = work_action(id)
            else:
                self.queue_message(
                    {
                        "type": "runtime_loader_reused",
                        "task_id": self.current_task.get("task_id") if self.current_task else None,
                        **self._current_run_identity_payload(),
                        "node": id,
                        "previous_node": reused_node_id,
                        "module": module,
                        "action": action,
                        "message": "Reusing the resident loader across workflows; inputs will be revalidated.",
                    }
                )

        if not callable(self.node_cache[id]):
            raise TypeError(
                f"The class `{module}.{action}` is not callable. Make sure the class has a `__call__` method or extends `NodeBase`."
            )

        # set the session id, it can be used to send messages from the node back to the client
        self.node_cache[id]._sid = sid
        if upstream_changed:
            invalidate_cache = getattr(self.node_cache[id], "invalidate_cache", None)
            if callable(invalidate_cache):
                invalidate_cache()

        heartbeat_stop = threading.Event()
        heartbeat_thread = None
        if not quiet:
            heartbeat_task_id = self.current_task.get("task_id") if self.current_task else None

            def publish_node_heartbeat():
                while not heartbeat_stop.wait(5.0):
                    current_task = self.current_task
                    if (
                        not current_task
                        or current_task.get("task_id") != heartbeat_task_id
                        or current_task.get("current_node") != id
                    ):
                        return
                    heartbeat_at = time.time()
                    progress_value = current_task.get("node_progress")
                    if not isinstance(progress_value, (int, float)):
                        progress_value = -1
                    try:
                        resource_snapshot = self._runtime_resource_snapshot(max_age_seconds=1.5)
                    except Exception:
                        resource_snapshot = None
                    heartbeat_payload = self.record_node_progress(
                        {
                            "type": "progress",
                            "node": id,
                            "name": f"{module}.{action}",
                            "task_id": heartbeat_task_id,
                            "attempt_index": current_task.get("attempt_index"),
                            **self._current_run_identity_payload(),
                            "status": "running",
                            "phase": current_task.get("phase") or node_execution_phase(module, action),
                            "message": current_task.get("message")
                            or node_execution_message(module, action, node_execution_phase(module, action)),
                            "component": current_task.get("component"),
                            "shard_current": current_task.get("shard_current"),
                            "shard_total": current_task.get("shard_total"),
                            "current_step": current_task.get("current_step"),
                            "total_steps": current_task.get("total_steps"),
                            "elapsed_seconds": max(0.0, heartbeat_at - start_time),
                            "average_step_seconds": current_task.get("average_step_seconds"),
                            "eta_seconds": current_task.get("eta_seconds"),
                            "last_heartbeat_at": heartbeat_at,
                            "resource_snapshot": resource_snapshot,
                            "progress": progress_value,
                        }
                    )
                    self.queue_message(heartbeat_payload)

            heartbeat_thread = threading.Thread(
                target=publish_node_heartbeat,
                name=f"modiff-progress-{id}",
                daemon=True,
            )
            heartbeat_thread.start()

        # *** execute the node ***
        try:
            self.node_cache[id](**args)
        except Exception as e:
            heartbeat_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=0.2)
            traceback_text = traceback.format_exc()
            logger.error(f"Error executing node {id} ({module}.{action})")
            logger.error(traceback_text)
            setattr(e, "modiff_node_id", id)
            setattr(e, "modiff_node_name", f"{module}.{action}")
            setattr(e, "modiff_traceback", traceback_text)
            self.queue_message(
                {
                    "type": "node_error",
                    **self._exception_payload(
                        e,
                        task_id=self.current_task.get("task_id") if self.current_task else None,
                        sid=sid,
                        node_id=id,
                        node_name=f"{module}.{action}",
                        traceback_text=traceback_text,
                    ),
                    **self._current_run_identity_payload(),
                    "status": "failed",
                    "current_node": id,
                }
            )
            if not quiet:
                self.queue_message(
                    {
                        "type": "progress",
                        "node": id,
                        "task_id": self.current_task.get("task_id") if self.current_task else None,
                        "attempt_index": self.current_task.get("attempt_index") if self.current_task else None,
                        **self._current_run_identity_payload(),
                        "status": "failed",
                        "phase": node_execution_phase(module, action),
                        "message": f"{module}.{action} failed",
                        "progress": 0,
                    }
                )
            raise e

        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=0.2)

        if not quiet:
            execution_time = time.time() - start_time
            self.node_cache[id]._execution_time["last"] = execution_time
            self.node_cache[id]._execution_time["min"] = (
                min(self.node_cache[id]._execution_time["min"], execution_time)
                if self.node_cache[id]._execution_time["min"] is not None
                else execution_time
            )
            self.node_cache[id]._execution_time["max"] = (
                max(self.node_cache[id]._execution_time["max"], execution_time)
                if self.node_cache[id]._execution_time["max"] is not None
                else execution_time
            )

            memory_stats = get_memory_stats()
            if memory_stats:
                self.node_cache[id]._memory_usage["last"] = memory_stats["peak"]
                self.node_cache[id]._memory_usage["min"] = (
                    min(self.node_cache[id]._memory_usage["min"], memory_stats["peak"])
                    if self.node_cache[id]._memory_usage["min"] is not None
                    else memory_stats["peak"]
                )
                self.node_cache[id]._memory_usage["max"] = (
                    max(self.node_cache[id]._memory_usage["max"], memory_stats["peak"])
                    if self.node_cache[id]._memory_usage["max"] is not None
                    else memory_stats["peak"]
                )

            # the node has completed
            self.queue_message(
                {
                    "type": "executed",
                    "node": id,
                    "name": f"{module}.{action}",
                    "task_id": self.current_task.get("task_id") if self.current_task else None,
                    "attempt_index": self.current_task.get("attempt_index") if self.current_task else None,
                    **self._current_run_identity_payload(),
                    "status": "cached" if not self.node_cache[id]._has_changed else "succeeded",
                    "phase": node_execution_phase(module, action),
                    "progress": 100,
                    "current_node": id,
                    "hasChanged": self.node_cache[id]._has_changed,
                    "executionTime": self.node_cache[id]._execution_time,
                    "memoryUsage": self.node_cache[id]._memory_usage,
                }
            )

        for ui_key, data_key in ui_fields.items():
            message = None
            display = self.modules[module][action]["params"][ui_key].get("display")

            # skip for button and group fields
            if display in ["ui_button", "ui_group"]:
                continue

            else:
                # if the data key is an output, get the value from the output otherwise from the params
                if data_key in self.node_cache[id].output:
                    source_value = self.node_cache[id].output[data_key]
                elif data_key in self.node_cache[id].params:
                    source_value = self.node_cache[id].params[data_key]
                else:
                    continue

            data_type = self.modules[module][action]["params"][data_key].get("type")  # data type of the source field
            data_format = self.modules[module][action]["params"][ui_key].get(
                "type", "text"
            )  # format of the returned value: text, raw, url
            fieldOptions = self.modules[module][action]["params"][ui_key].get("fieldOptions", {})

            source_value = source_value if isinstance(source_value, list) else [source_value]
            artifacts = None
            if data_format == "url":
                if is_image_data_type(data_type):
                    image_format = fieldOptions.get("format", "WEBP")
                    image_quality = fieldOptions.get("quality", 100)
                    data_value = []
                    artifacts = []
                    task_id = self.current_task.get("task_id") if self.current_task else None
                    attempt_index = self.current_task.get("attempt_index") if self.current_task else None
                    runtime_hints = self.current_task.get("runtimeHints") if self.current_task else None
                    for i in range(len(source_value)):
                        if source_value[i] is None:
                            continue
                        filename = f"modiff-{id}-{data_key}-{i}.{str(image_format).lower()}"
                        url = (
                            f"/cache/{id}/{data_key}/{i}"
                            f"?format={quote(str(image_format))}"
                            f"&quality={quote(str(image_quality))}"
                            f"&filename={quote(filename)}"
                            f"&t={time.time()}"
                        )
                        data_value.append(url)
                        artifacts.append(
                            attach_run_identity_to_artifact(
                                cache_image_artifact(id, data_key, i, url, source_value[i], image_format),
                                task_id=task_id,
                                attempt_index=attempt_index,
                                runtime_hints=runtime_hints,
                            )
                        )
                elif display in {"ui_audio", "ui_video"}:
                    data_value = []
                    artifacts = []
                    for i, item in enumerate(source_value):
                        if item is None:
                            continue
                        file_url = file_backed_media_preview(item)
                        url = file_url or f"/cache/{id}/{data_key}/{i}?t={time.time()}"
                        data_value.append(url)
                        if file_url:
                            filename = os.fspath(item)
                            mime_type, _ = mimetypes.guess_type(filename)
                            artifacts.append(
                                {
                                    "url": url,
                                    "nodeId": id,
                                    "fieldKey": data_key,
                                    "index": i,
                                    "mimeType": mime_type,
                                    "filename": filename,
                                    "source": "file",
                                }
                            )
                        else:
                            artifacts.append(
                                {
                                    "url": url,
                                    "nodeId": id,
                                    "fieldKey": data_key,
                                    "index": i,
                                    "source": "cache",
                                }
                            )
                else:
                    data_value = [
                        f"/cache/{id}/{data_key}/{i}?t={time.time()}"
                        for i in range(len(source_value))
                        if source_value[i] is not None
                    ]
            elif data_format == "raw":
                data_value = [to_bytes(data_type, item, fieldOptions) for item in source_value if item is not None]
            else:
                data_value = [to_base64(data_type, item, fieldOptions) for item in source_value if item is not None]

            message = {
                "client_id": sid,
                "type": "update_value",
                "node": id,
                "key": ui_key,
                "data_type": data_type,
                "value": data_value,
                "task_id": self.current_task.get("task_id") if self.current_task else None,
                "attempt_index": self.current_task.get("attempt_index") if self.current_task else None,
                **self._current_run_identity_payload(),
                "runtimeFingerprint": self.current_task.get("runtimeFingerprint") if self.current_task else None,
            }
            if artifacts is not None:
                message["artifacts"] = artifacts

            if message:
                ui_field_hidden = bool(self.modules[module][action]["params"][ui_key].get("hidden"))
                if not ui_field_hidden and not (module == "modules.Audio" and action == "Load"):
                    output_id, backend_persisted = self._persist_generated_output_update(message, display=display)
                    if output_id:
                        message["output_id"] = output_id
                        message["backend_persisted"] = backend_persisted
                self.queue_message(message)

    def trigger_node(self, source_id, output, sid):
        if not self.current_task:
            return

        graph = self.current_task["args"][0]
        nodes = graph["nodes"]

        for id in nodes:
            params = nodes[id]["params"]

            for p in params:
                data_source_id = params[p].get("sourceId")
                data_param_key = params[p].get("sourceKey")
                if data_source_id == source_id and data_param_key == output:
                    self.execute_node(id, nodes[id], sid, quiet=True)

    """
    ╭────────────────╮
       Hugging Face
    ╰────────────────╯
    """

    async def hf_cache(self, request):
        refresh = request.query.get("refresh", False)
        id = request.match_info.get("id", None)
        class_name = request.query.get("className", None)
        compact = request.query.get("compact", False)
        if refresh:
            modelstore.update_hf()

        models = modelstore.get_hf_models(id, class_name, "full")
        annotated = []
        for model in models:
            status = artifact_cache_status(model.get("id"), models)
            entry = dict(model)
            entry.update(
                {
                    "cached": bool(status.get("installed")),
                    "installed": bool(status.get("complete")),
                    "complete": bool(status.get("complete")),
                    "repair_required": bool(status.get("repairRequired")),
                    "install_reason": status.get("reason"),
                    "active_files": status.get("activeFiles") or [],
                    "missing_files": status.get("missingFiles") or [],
                    "corrupt_files": status.get("corruptFiles") or [],
                }
            )
            if compact:
                entry = {
                    key: entry[key]
                    for key in (
                        "id",
                        "class_names",
                        "cached",
                        "installed",
                        "complete",
                        "repair_required",
                        "install_reason",
                        "active_files",
                        "missing_files",
                        "corrupt_files",
                    )
                }
            annotated.append(entry)

        return web.json_response(annotated)

    def _package_status(self, module_name, distribution_name=None):
        package = {
            "available": False,
            "module": module_name,
            "distribution": distribution_name or module_name,
        }

        try:
            package["version"] = metadata.version(distribution_name or module_name)
        except Exception:
            pass

        try:
            module = import_module(module_name)
            package["available"] = True
            package["version"] = getattr(module, "__version__", package.get("version"))
        except Exception as e:
            package["error"] = str(e)

        return package

    def _runtime_fingerprint_for_control_request(self):
        """Avoid entering accelerator APIs while a model call owns the runtime."""
        if self.current_task and isinstance(self._last_runtime_fingerprint, dict):
            return deepcopy(self._last_runtime_fingerprint)
        return self._runtime_fingerprint()

    async def system_stats(self, _request):
        if self.current_task and isinstance(self._last_runtime_fingerprint, dict):
            cached_hardware = self._last_runtime_fingerprint.get("hardware")
            if isinstance(cached_hardware, dict):
                return web.json_response(deepcopy(cached_hardware))
        return web.json_response(get_hardware_snapshot(self.data_dir))

    async def runtime_status(self, request):
        runtime_fingerprint = self._runtime_fingerprint_for_control_request()
        hardware = runtime_fingerprint.get("hardware")
        if not isinstance(hardware, dict):
            hardware = get_hardware_snapshot(self.data_dir, refresh=True)
        profile = runtime_profile(hardware, venv=Path(sys.prefix))
        packages = {
            "aiohttp": self._package_status("aiohttp"),
            "aiohttp_cors": self._package_status("aiohttp_cors", "aiohttp-cors"),
            "torch": self._package_status("torch"),
            "diffusers": self._package_status("diffusers"),
            "transformers": self._package_status("transformers"),
            "huggingface_hub": self._package_status("huggingface_hub", "huggingface-hub"),
            "accelerate": self._package_status("accelerate"),
            "safetensors": self._package_status("safetensors"),
        }
        packages["torch"].update(legacy_torch_status(hardware))
        required = ["aiohttp", "aiohttp_cors", "torch", "diffusers", "huggingface_hub"]
        missing_required = [name for name in required if not packages.get(name, {}).get("available")]

        current_task = None
        if self.current_task:
            current_task = {
                "task_id": self.current_task.get("task_id"),
                "name": self.current_task.get("name"),
                "sid": self.current_task.get("sid"),
                "started_at": self.current_task.get("started_at"),
                "progress": self.current_task.get("progress"),
            }

        ready = len(missing_required) == 0 and bool(profile.get("execution_ready"))
        return web.json_response(
            {
                "error": False,
                "ready": ready,
                "runtime_fingerprint": (
                    runtime_fingerprint.get("resourceFingerprint") or runtime_fingerprint.get("fingerprint")
                ),
                "runtime_profile": profile,
                "instance": self.instance,
                "server": {
                    "host": self.host,
                    "port": self.port,
                    "scheme": "https" if self.ssl_context else "http",
                    "work_dir": self.work_dir,
                    "data_dir": self.data_dir,
                    "client_max_size": self.client_max_size,
                },
                "python": {
                    "version": sys.version,
                    "executable": sys.executable,
                    "platform": platform.platform(),
                    "cwd": os.getcwd(),
                },
                "config": {
                    "hf_cache_dir": CONFIG.hf.get("cache_dir"),
                    "hf_online_status": CONFIG.hf.get("online_status"),
                    "hf_token_configured": bool(CONFIG.hf.get("token")),
                    "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                    "paths": CONFIG.paths,
                },
                "packages": packages,
                "hardware": hardware,
                "missing_required_packages": missing_required,
                "modules": {
                    "registered_count": len(self.modules),
                    "module_map_count": len(MODULE_MAP),
                },
                "queue": {
                    "current": current_task,
                    "queued_count": len(self.queued_tasks),
                    "main_queue_size": self.main_queue.qsize(),
                    "background_queue_size": self.background_queue.qsize(),
                    "interrupt_requested": self.interrupt_flag,
                },
            }
        )

    def _optimization_runtime_context(self):
        runtime_fingerprint = self._runtime_fingerprint_for_control_request()
        hardware = runtime_fingerprint.get("hardware")
        if not isinstance(hardware, dict):
            hardware = get_hardware_snapshot(self.data_dir, refresh=not bool(self.current_task))
        return runtime_fingerprint, hardware, runtime_profile(hardware, venv=Path(sys.prefix))

    def _runtime_job_root(self, *, create=False):
        data_root_path = Path(self.data_dir)
        if create:
            data_root_path.mkdir(parents=True, exist_ok=True)
        data_root_info = data_root_path.lstat()
        if (
            not stat.S_ISDIR(data_root_info.st_mode)
            or stat.S_ISLNK(data_root_info.st_mode)
            or bool(
                getattr(data_root_info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            )
        ):
            raise OSError("The configured runtime data root is unsafe.")
        data_root = data_root_path.resolve(strict=True)
        current = data_root
        for name in ("runtime", "optimization-jobs"):
            candidate = current / name
            if create:
                try:
                    candidate.mkdir()
                except FileExistsError:
                    pass
            details = candidate.lstat()
            if (
                not stat.S_ISDIR(details.st_mode)
                or stat.S_ISLNK(details.st_mode)
                or bool(
                    getattr(details, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                )
            ):
                raise OSError("The runtime job receipt directory is unsafe.")
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(data_root)
            current = resolved
        return current

    @staticmethod
    def _parse_runtime_job_document(raw):
        def reject_duplicates(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate JSON key")
                value[key] = item
            return value

        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("non-finite")),
        )

    def _read_runtime_job_file(self, job_id):
        if not self._valid_runtime_job_id(job_id):
            return None
        try:
            job_root = self._runtime_job_root(create=False)
            path = job_root / f"{job_id}.json"
            details = path.lstat()
            if (
                not stat.S_ISREG(details.st_mode)
                or stat.S_ISLNK(details.st_mode)
                or details.st_nlink != 1
                or details.st_size > 64 * 1024
                or bool(
                    getattr(details, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                )
            ):
                return None
            resolved = path.resolve(strict=True)
            resolved.relative_to(job_root)
            raw = path.read_bytes()
            if len(raw) != details.st_size:
                return None
            value = self._parse_runtime_job_document(raw)
            return value if isinstance(value, dict) and value.get("id") == job_id else None
        except (OSError, TypeError, ValueError, UnicodeDecodeError, RecursionError):
            return None

    def _load_runtime_jobs(self):
        try:
            job_root = self._runtime_job_root(create=False)
        except OSError:
            return
        jobs = []
        scanned = 0
        try:
            with os.scandir(job_root) as entries:
                for entry in entries:
                    scanned += 1
                    if scanned > 10_000:
                        os.environ["MODIFF_RUNTIME_OVERLAY_STATUS"] = "repair_required"
                        break
                    match = re.fullmatch(r"(optjob-[A-Za-z0-9_-]{12})\.json", entry.name)
                    if not match:
                        continue
                    job = self._read_runtime_job_file(match.group(1))
                    if not isinstance(job, dict):
                        continue
                    if job.get("kind") not in {"optimization", "optional_runtime"}:
                        continue
                    if job.get("status") in {"queued", "running", "cancelling"}:
                        job["status"] = "failed"
                        job["error"] = "Optional-runtime installation failed."
                        job["progress"] = {
                            "phase": "failed",
                            "message": "The prior worker exited before this installation completed.",
                            "updatedAt": time.time(),
                        }
                        job["updatedAt"] = time.time()
                        try:
                            self._persist_optimization_job(job)
                        except (OSError, TypeError, ValueError):
                            logger.warning(
                                "Could not reconcile an interrupted runtime job", exc_info=True
                            )
                            os.environ["MODIFF_RUNTIME_OVERLAY_STATUS"] = "repair_required"
                    jobs.append(job)
        except OSError:
            return
        for job in sorted(
            jobs,
            key=lambda item: item.get("updatedAt")
            if isinstance(item.get("updatedAt"), (int, float))
            else 0,
            reverse=True,
        )[:100]:
            self.optimization_jobs[job["id"]] = job

    def _persist_optimization_job(self, job):
        job_id = str(job.get("id") or "") if isinstance(job, dict) else ""
        if not self._valid_runtime_job_id(job_id):
            raise OSError("The runtime job ID is invalid.")
        job_dir = self._runtime_job_root(create=True)
        path = job_dir / f"{job_id}.json"
        try:
            existing = path.lstat()
            if (
                not stat.S_ISREG(existing.st_mode)
                or stat.S_ISLNK(existing.st_mode)
                or existing.st_nlink != 1
                or bool(
                    getattr(existing, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                )
            ):
                raise OSError("The runtime job receipt target is unsafe.")
        except FileNotFoundError:
            pass
        temporary = job_dir / f".{job_id}.{nanoid.generate(size=12)}.tmp"
        body = (json.dumps(job, indent=2, allow_nan=False) + "\n").encode("utf-8")
        if len(body) > 64 * 1024:
            raise OSError("The runtime job receipt exceeds its safe size.")
        try:
            with temporary.open("xb") as output:
                output.write(body)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _update_optimization_job(self, job_id, **updates):
        job = self.optimization_jobs.get(job_id)
        if not isinstance(job, dict):
            return
        current_status = str(job.get("status") or "")
        requested_status = updates.get("status", current_status)
        terminal = {"ready", "failed", "cancelled"}
        transitions = {
            "queued": {"queued", "running", "cancelling", "cancelled", "failed", "ready"},
            "running": {"running", "cancelling", "cancelled", "failed", "ready"},
            "cancelling": {"cancelling", "cancelled", "failed"},
        }
        if current_status in terminal or requested_status not in transitions.get(
            current_status, set()
        ):
            return
        next_job = deepcopy(job)
        next_job.update(updates)
        next_job["updatedAt"] = time.time()
        try:
            self._persist_optimization_job(next_job)
        except (OSError, TypeError, ValueError):
            logger.warning("Could not persist optional-runtime installation job", exc_info=True)
            os.environ["MODIFF_RUNTIME_OVERLAY_STATUS"] = "repair_required"
            return
        self.optimization_jobs[job_id] = next_job

    def _reserve_worker_runtime_gate(self, kind, identifier):
        if (
            self._runtime_mutation_gate is not None
            or self.current_task
            or self.queued_tasks
            or self._active_nonruntime_mutations
            or self.hf_download_tasks
            or (self.template_gallery_install_task is not None and not self.template_gallery_install_task.done())
        ):
            raise OverlayInstallBusy(
                "Finish or stop active and queued runs before changing the runtime environment."
            )
        token = f"runtime-gate-{nanoid.generate(size=16)}"
        self._runtime_mutation_gate = {
            "token": token,
            "kind": str(kind)[:64],
            "identifier": str(identifier)[:256],
        }
        return token

    def _release_worker_runtime_gate(self, token):
        gate = self._runtime_mutation_gate
        if isinstance(gate, dict) and gate.get("token") == token:
            self._runtime_mutation_gate = None

    async def _strict_runtime_control_json(self, request, *, allowed, required=(), allow_empty=False):
        content_length = getattr(request, "content_length", None)
        if content_length is not None and (
            not isinstance(content_length, int)
            or isinstance(content_length, bool)
            or content_length < 0
            or content_length > 4096
        ):
            raise ValueError("Runtime control request exceeds 4096 bytes.")

        def reject_duplicates(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("Runtime control request contains a duplicate JSON key.")
                value[key] = item
            return value

        content = getattr(request, "content", None)
        if content is not None and hasattr(content, "read"):
            chunks = []
            total = 0
            while not content.at_eof():
                chunk = await content.read(min(4097 - total, 4097))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > 4096:
                    raise ValueError("Runtime control request exceeds 4096 bytes.")
            raw = b"".join(chunks)
        elif hasattr(request, "read"):
            raw = await request.read()
            if len(raw) > 4096:
                raise ValueError("Runtime control request exceeds 4096 bytes.")
        else:
            raw = None
        if raw is not None:
            if not raw and allow_empty:
                body = {}
            else:
                try:
                    body = json.loads(
                        raw.decode("utf-8"),
                        object_pairs_hook=reject_duplicates,
                        parse_constant=lambda _value: (_ for _ in ()).throw(
                            ValueError("Runtime control request contains a non-finite number.")
                        ),
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
                    raise ValueError("Runtime control request must be a JSON object.") from exc
        else:
            try:
                body = await request.json()
            except (TypeError, ValueError, RecursionError) as exc:
                raise ValueError("Runtime control request must be a JSON object.") from exc
        if not isinstance(body, dict):
            raise ValueError("Runtime control request must be a JSON object.")
        keys = set(body)
        if not keys.issubset(set(allowed)) or not set(required).issubset(keys):
            raise ValueError("Runtime control request has missing or unknown fields.")
        return body

    @staticmethod
    def _valid_runtime_job_id(job_id):
        return bool(re.fullmatch(r"optjob-[A-Za-z0-9_-]{12}", str(job_id or "")))

    @staticmethod
    def _public_runtime_job(job):
        if not isinstance(job, dict):
            return None
        job_id = str(job.get("id") or "")
        if not WebServer._valid_runtime_job_id(job_id):
            return None
        def environment_id(value):
            return (
                value
                if isinstance(value, str)
                and re.fullmatch(r"runtime-[0-9]{1,16}-[0-9a-f]{8}", value)
                else None
            )

        def contract_id(value):
            return (
                value
                if isinstance(value, str)
                and re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,127}", value)
                else None
            )

        def spec_digest(value):
            return (
                value
                if isinstance(value, str)
                and re.fullmatch(r"sha256:[0-9a-f]{64}", value)
                else None
            )

        def utc_timestamp(value):
            if not isinstance(value, str) or not re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
                value,
            ):
                return None
            try:
                parsed = time.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                return None
            return value if time.strftime("%Y-%m-%dT%H:%M:%SZ", parsed) == value else None

        def finite_time(value):
            return (
                float(value)
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                else None
            )

        result = job.get("result") if isinstance(job.get("result"), dict) else {}
        public_result = {
            "environmentId": environment_id(result.get("environmentId")),
            "specs": [
                {
                    "kind": item.get("kind"),
                    "id": contract_id(item.get("id")),
                    "specDigest": spec_digest(item.get("specDigest")),
                }
                for item in (result.get("specs") or [])
                if isinstance(item, dict)
                and item.get("kind") in {"optimization", "optional_runtime"}
                and contract_id(item.get("id")) is not None
                and spec_digest(item.get("specDigest")) is not None
            ][:32],
            "requiresActivation": result.get("requiresActivation") is True,
            "activeRuntimeChanged": result.get("activeRuntimeChanged") is True,
        }
        capabilities = result.get("capabilities")
        if isinstance(capabilities, list):
            public_result["capabilities"] = [
                item for item in capabilities[:32] if contract_id(item) is not None
            ]
        validation = result.get("validation")
        if isinstance(validation, dict):
            public_result["validation"] = {
                "status": (
                    validation.get("status")
                    if validation.get("status") in {"passed", "failed"}
                    else None
                ),
                "validatedAt": utc_timestamp(validation.get("validatedAt")),
                "bindingDigest": spec_digest(validation.get("bindingDigest")),
            }
        progress = job.get("progress") if isinstance(job.get("progress"), dict) else {}
        status = (
            str(job.get("status"))
            if job.get("status")
            in {"queued", "running", "cancelling", "cancelled", "failed", "ready"}
            else "failed"
        )
        phase = str(progress.get("phase") or "")
        phase_messages = {
            "queued": "Installation is queued.",
            "copying": "Preparing the isolated staged environment.",
            "downloading": "Acquiring reviewed runtime artifacts.",
            "installing": "Installing reviewed artifacts into the isolated stage.",
            "validating": "Validating the staged runtime in an isolated process.",
            "promoting": "Promoting the validated staged runtime.",
            "ready": "Validation passed. Explicit activation and restart are required.",
            "cancelling": "Cancelling the staged installation.",
            "cancelled": "Installation was cancelled; the active environment was unchanged.",
            "failed": "Installation failed; the active environment was unchanged.",
        }
        if phase not in phase_messages:
            phase = status if status in phase_messages else "failed"
        public = {
            "id": job_id,
            "status": status,
            "progress": {
                "phase": phase,
                "message": phase_messages[phase],
                "updatedAt": finite_time(progress.get("updatedAt")),
            },
            "createdAt": finite_time(job.get("createdAt")),
            "updatedAt": finite_time(job.get("updatedAt")),
        }
        for key in ("capabilityId", "profileId"):
            normalized = contract_id(job.get(key))
            if normalized is not None:
                public[key] = normalized
        normalized_digest = spec_digest(job.get("specDigest"))
        if normalized_digest is not None:
            public["specDigest"] = normalized_digest
        if result:
            public["result"] = public_result
        if job.get("error"):
            public["error"] = "Optional-runtime installation failed."
        return public

    @staticmethod
    def _public_optimization_receipt(receipt):
        if not isinstance(receipt, dict):
            return None
        receipt_id = receipt.get("id")
        if not isinstance(receipt_id, str) or not re.fullmatch(
            r"(?:probe|workload|baseline)-[0-9a-f]{32}", receipt_id
        ):
            return None
        kind = receipt.get("kind")
        status = receipt.get("status")
        if kind not in {"compatibility_probe", "workload", "workload_baseline"} or status not in {
            "probe_passed",
            "probe_failed",
            "observed",
            "qualified",
        }:
            return None

        def contract_id(value):
            return (
                value
                if isinstance(value, str)
                and re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,127}", value)
                else None
            )

        def environment_id(value):
            return (
                value
                if isinstance(value, str)
                and re.fullmatch(r"runtime-[0-9]{1,16}-[0-9a-f]{8}", value)
                else None
            )

        def timestamp(value):
            if not isinstance(value, str) or not re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
                value,
            ):
                return None
            try:
                parsed = time.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                return None
            return value if time.strftime("%Y-%m-%dT%H:%M:%SZ", parsed) == value else None

        public = {
            "id": receipt_id,
            "schemaVersion": 1,
            "kind": kind,
            "status": status,
            "capabilityId": contract_id(receipt.get("capabilityId")),
            "environmentId": environment_id(receipt.get("environmentId")),
            "createdAt": timestamp(receipt.get("createdAt")),
            "qualifiedAt": timestamp(receipt.get("qualifiedAt")),
            "autoEligible": receipt.get("autoEligible") is True,
        }
        fingerprint = receipt.get("runtimeFingerprintHash")
        if isinstance(fingerprint, str) and re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            public["runtimeFingerprintHash"] = fingerprint
        if kind == "compatibility_probe":
            result = receipt.get("result") if isinstance(receipt.get("result"), dict) else {}
            public["validationStatus"] = (
                result.get("status") if result.get("status") in {"passed", "failed"} else "failed"
            )
        evidence = receipt.get("benchmarkEvidence")
        if isinstance(evidence, dict):
            public_evidence = {"improved": evidence.get("improved") is True}
            for key in ("elapsedRatio", "peakMemoryRatio"):
                value = evidence.get(key)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                ):
                    public_evidence[key] = float(value)
            public["benchmarkEvidence"] = public_evidence
        baseline_id = receipt.get("baselineReceiptId")
        if isinstance(baseline_id, str) and re.fullmatch(r"baseline-[0-9a-f]{32}", baseline_id):
            public["baselineReceiptId"] = baseline_id
        return public

    @staticmethod
    def _public_runtime_mutation_result(result):
        value = result if isinstance(result, dict) else {}
        state = value.get("state") if isinstance(value.get("state"), dict) else {}
        enabled_capabilities = (
            state.get("enabledCapabilities")
            if isinstance(state.get("enabledCapabilities"), list)
            else []
        )

        def environment_id(raw):
            if not isinstance(raw, str) or not re.fullmatch(
                r"runtime-[0-9]{1,16}-[0-9a-f]{8}", raw
            ):
                return None
            return raw

        def timestamp(raw):
            if not isinstance(raw, str) or not re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
                raw,
            ):
                return None
            try:
                parsed = time.strptime(raw, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                return None
            return raw if time.strftime("%Y-%m-%dT%H:%M:%SZ", parsed) == raw else None

        return {
            "state": {
                "schemaVersion": state.get("schemaVersion")
                if isinstance(state.get("schemaVersion"), int)
                and not isinstance(state.get("schemaVersion"), bool)
                else 2,
                "activeEnvironmentId": environment_id(state.get("activeEnvironmentId")),
                "previousEnvironmentId": environment_id(state.get("previousEnvironmentId")),
                "enabledCapabilities": [
                    str(item)[:128]
                    for item in enabled_capabilities
                    if isinstance(item, str) and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", item)
                ][:64],
                "updatedAt": timestamp(state.get("updatedAt")),
            },
            "restartRequired": value.get("restartRequired") is True,
        }

    async def runtime_optimizations(self, _request):
        _fingerprint, hardware, profile = self._optimization_runtime_context()
        return web.json_response(public_optimization_catalog(runtime_profile=profile, hardware=hardware))

    async def runtime_optional_runtimes(self, _request):
        return web.json_response(public_optional_runtime_catalog())

    async def _run_optimization_install_job(
        self, job_id, capability_id, profile, hardware, lease, gate_token
    ):
        loop = asyncio.get_running_loop()

        def progress(update):
            loop.call_soon_threadsafe(
                partial(
                    self._update_optimization_job,
                    job_id,
                    status="running",
                    progress=update,
                )
            )

        try:
            result = await asyncio.to_thread(
                install_optimization_capability,
                capability_id,
                runtime_profile=profile,
                hardware=hardware,
                progress=progress,
                lease=lease,
            )
            self._update_optimization_job(
                job_id,
                status="ready",
                progress={
                    "phase": "ready",
                    "message": "Validation passed. Activate to restart MoDiff with this optional environment.",
                    "updatedAt": time.time(),
                },
                result=result,
            )
        except OverlayCancelled:
            self._update_optimization_job(
                job_id,
                status="cancelled",
                progress={
                    "phase": "cancelled",
                    "message": "Optional-runtime installation was cancelled and staging was removed.",
                    "updatedAt": time.time(),
                },
            )
        except Exception as exc:
            logger.warning("Optional runtime package installation failed: %s", exc)
            self._update_optimization_job(
                job_id,
                status="failed",
                progress={
                    "phase": "failed",
                    "message": "Optional-runtime installation failed. The active environment was unchanged.",
                    "updatedAt": time.time(),
                },
                error="Optional-runtime installation failed.",
            )
        finally:
            self._runtime_install_leases.pop(job_id, None)
            self._runtime_install_gate_tokens.pop(job_id, None)
            release_runtime_install(lease)
            self._release_worker_runtime_gate(gate_token)

    async def _run_optional_runtime_install_job(
        self, job_id, profile_id, spec_digest, lease, gate_token
    ):
        loop = asyncio.get_running_loop()

        def progress(update):
            loop.call_soon_threadsafe(
                partial(
                    self._update_optimization_job,
                    job_id,
                    status="running",
                    progress=update,
                )
            )

        try:
            result = await asyncio.to_thread(
                install_optional_runtime,
                profile_id,
                spec_digest,
                consent=True,
                lease=lease,
                progress=progress,
            )
            self._update_optimization_job(
                job_id,
                status="ready",
                result=result,
                progress={
                    "phase": "ready",
                    "message": "Validation passed. Explicit activation and restart are required.",
                    "updatedAt": time.time(),
                },
            )
        except OverlayCancelled:
            self._update_optimization_job(
                job_id,
                status="cancelled",
                progress={
                    "phase": "cancelled",
                    "message": "Optional-runtime installation was cancelled and staging was removed.",
                    "updatedAt": time.time(),
                },
            )
        except Exception:
            logger.warning("Optional model-runtime installation failed", exc_info=True)
            self._update_optimization_job(
                job_id,
                status="failed",
                error="Optional-runtime installation failed.",
                progress={
                    "phase": "failed",
                    "message": "Optional-runtime installation failed. The active environment was unchanged.",
                    "updatedAt": time.time(),
                },
            )
        finally:
            self._runtime_install_leases.pop(job_id, None)
            self._runtime_install_gate_tokens.pop(job_id, None)
            release_runtime_install(lease)
            self._release_worker_runtime_gate(gate_token)

    async def runtime_optimization_install(self, request):
        if self.current_task or self.queued_tasks:
            return web.json_response(
                {
                    "error": True,
                    "error_code": "optimization_install_busy",
                    "message": "Finish or stop active and queued runs before changing optional runtime packages.",
                },
                status=409,
            )
        try:
            body = await self._strict_runtime_control_json(
                request,
                allowed={"capabilityId"},
                required={"capabilityId"},
            )
        except ValueError as exc:
            return web.json_response({"error": True, "message": str(exc)}, status=400)
        capability_id = str(body.get("capabilityId") or "").strip()
        if not capability_id:
            return web.json_response({"error": True, "message": "capabilityId is required."}, status=400)
        _fingerprint, hardware, profile = self._optimization_runtime_context()
        catalog = public_optimization_catalog(runtime_profile=profile, hardware=hardware)
        capability = next(
            (item for item in catalog.get("capabilities", []) if item.get("id") == capability_id),
            None,
        )
        if (
            not capability
            or capability.get("kind") != "package"
            or not capability.get("compatible")
            or capability.get("canInstall") is not True
        ):
            return web.json_response(
                {"error": True, "message": "This optimization package is unavailable."},
                status=400,
            )
        try:
            gate_token = self._reserve_worker_runtime_gate("optimization_install", capability_id)
        except OverlayInstallBusy as exc:
            return web.json_response(
                {"error": True, "error_code": "optimization_install_busy", "message": str(exc)},
                status=409,
            )
        try:
            lease = reserve_runtime_install("optimization", capability_id)
        except OverlayInstallBusy as exc:
            self._release_worker_runtime_gate(gate_token)
            return web.json_response(
                {"error": True, "error_code": "optimization_install_busy", "message": str(exc)},
                status=409,
            )
        try:
            job_id = f"optjob-{nanoid.generate(size=12)}"
            job = {
                "id": job_id,
                "kind": "optimization",
                "capabilityId": capability_id,
                "status": "queued",
                "progress": {
                    "phase": "queued",
                    "message": "Waiting to stage the optional package.",
                    "updatedAt": time.time(),
                },
                "createdAt": time.time(),
                "updatedAt": time.time(),
            }
            self.optimization_jobs[job_id] = job
            self._runtime_install_leases[job_id] = lease
            self._runtime_install_gate_tokens[job_id] = gate_token
            self._persist_optimization_job(job)
            asyncio.create_task(
                self._run_optimization_install_job(
                    job_id, capability_id, profile, hardware, lease, gate_token
                )
            )
        except Exception:
            self.optimization_jobs.pop(locals().get("job_id", ""), None)
            self._runtime_install_leases.pop(locals().get("job_id", ""), None)
            self._runtime_install_gate_tokens.pop(locals().get("job_id", ""), None)
            release_runtime_install(lease)
            self._release_worker_runtime_gate(gate_token)
            logger.exception("Could not start the optional-runtime installation job")
            return web.json_response(
                {"error": True, "message": "Could not start the optional-runtime installation job."},
                status=500,
            )
        return web.json_response(
            {"error": False, "job": self._public_runtime_job(job)}, status=202
        )

    async def runtime_optional_runtime_install(self, request):
        if self.current_task or self.queued_tasks:
            return web.json_response(
                {
                    "error": True,
                    "error_code": "optional_runtime_install_busy",
                    "message": "Finish or stop active and queued runs before changing optional runtimes.",
                },
                status=409,
            )
        try:
            body = await self._strict_runtime_control_json(
                request,
                allowed={"profileId", "specDigest", "consent"},
                required={"profileId", "specDigest", "consent"},
            )
            profile_id = body.get("profileId")
            spec_digest = body.get("specDigest")
            if (
                not isinstance(profile_id, str)
                or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", profile_id)
                or not isinstance(spec_digest, str)
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", spec_digest)
                or body.get("consent") is not True
            ):
                raise ValueError(
                    "profileId, exact specDigest, and literal consent=true are required."
                )
            # Qualification, digest, artifact-lock, base binding, and managed
            # installer checks all run before a lease, job, staging path, or
            # subprocess can be created.
            validate_optional_runtime_install_request(profile_id, spec_digest, consent=True)
        except ValueError as exc:
            return web.json_response({"error": True, "message": str(exc)}, status=400)
        except RuntimeError as exc:
            return web.json_response({"error": True, "message": str(exc)}, status=409)
        try:
            gate_token = self._reserve_worker_runtime_gate("optional_runtime_install", profile_id)
        except OverlayInstallBusy as exc:
            return web.json_response(
                {"error": True, "error_code": "optional_runtime_install_busy", "message": str(exc)},
                status=409,
            )
        try:
            lease = reserve_runtime_install("optional_runtime", profile_id)
        except OverlayInstallBusy as exc:
            self._release_worker_runtime_gate(gate_token)
            return web.json_response(
                {"error": True, "error_code": "optional_runtime_install_busy", "message": str(exc)},
                status=409,
            )
        try:
            job_id = f"optjob-{nanoid.generate(size=12)}"
            job = {
                "id": job_id,
                "kind": "optional_runtime",
                "profileId": profile_id,
                "specDigest": spec_digest,
                "status": "queued",
                "progress": {
                    "phase": "queued",
                    "message": "Waiting to stage the reviewed optional runtime.",
                    "updatedAt": time.time(),
                },
                "createdAt": time.time(),
                "updatedAt": time.time(),
            }
            self.optimization_jobs[job_id] = job
            self._runtime_install_leases[job_id] = lease
            self._runtime_install_gate_tokens[job_id] = gate_token
            self._persist_optimization_job(job)
            asyncio.create_task(
                self._run_optional_runtime_install_job(
                    job_id, profile_id, spec_digest, lease, gate_token
                )
            )
        except Exception:
            self.optimization_jobs.pop(locals().get("job_id", ""), None)
            self._runtime_install_leases.pop(locals().get("job_id", ""), None)
            self._runtime_install_gate_tokens.pop(locals().get("job_id", ""), None)
            release_runtime_install(lease)
            self._release_worker_runtime_gate(gate_token)
            logger.exception("Could not start the optional-runtime installation job")
            return web.json_response(
                {"error": True, "message": "Could not start the optional-runtime installation job."},
                status=500,
            )
        return web.json_response(
            {"error": False, "job": self._public_runtime_job(job)}, status=202
        )

    async def runtime_optimization_job(self, request):
        job_id = str(request.match_info.get("job_id") or "")
        expected_kind = (
            "optional_runtime"
            if str(getattr(request, "path", "")).startswith("/runtime/optional-runtimes/")
            else "optimization"
        )
        if not self._valid_runtime_job_id(job_id):
            return web.json_response(
                {"error": True, "message": "Optimization installation job not found."}, status=404
            )
        job = self.optimization_jobs.get(job_id)
        if not isinstance(job, dict):
            job = self._read_runtime_job_file(job_id)
        if isinstance(job, dict) and job.get("kind") != expected_kind:
            job = None
        public_job = self._public_runtime_job(job)
        if not public_job:
            return web.json_response(
                {"error": True, "message": "Optimization installation job not found."}, status=404
            )
        return web.json_response({"error": False, "job": public_job})

    async def runtime_optimization_job_cancel(self, request):
        job_id = str(request.match_info.get("job_id") or "")
        expected_kind = (
            "optional_runtime"
            if str(getattr(request, "path", "")).startswith("/runtime/optional-runtimes/")
            else "optimization"
        )
        if not self._valid_runtime_job_id(job_id):
            return web.json_response(
                {"error": True, "message": "Optimization installation job not found."}, status=404
            )
        try:
            await self._strict_runtime_control_json(
                request, allowed=set(), required=set(), allow_empty=True
            )
        except ValueError as exc:
            return web.json_response({"error": True, "message": str(exc)}, status=400)
        job = self.optimization_jobs.get(job_id)
        if not isinstance(job, dict) or job.get("kind") != expected_kind:
            return web.json_response(
                {"error": True, "message": "The installation is not active in this worker."},
                status=409,
            )
        if job.get("status") in {"ready", "failed", "cancelled"}:
            return web.json_response(
                {"error": True, "message": "The installation job is already complete."}, status=409
            )
        lease = self._runtime_install_leases.get(job_id)
        cancelled = (
            await asyncio.to_thread(cancel_runtime_install, lease.token)
            if lease is not None
            else False
        )
        if not cancelled:
            return web.json_response(
                {"error": True, "message": "The installation can no longer be cancelled."}, status=409
            )
        self._update_optimization_job(
            job_id,
            status="cancelling",
            progress={
                "phase": "cancelling",
                "message": "Cancelling the exact installer process tree and removing staging.",
                "updatedAt": time.time(),
            },
        )
        return web.json_response(
            {"error": False, "job": self._public_runtime_job(self.optimization_jobs[job_id])},
            status=202,
        )

    async def runtime_optional_runtime_activate(self, request):
        if self.current_task or self.queued_tasks:
            return web.json_response(
                {
                    "error": True,
                    "error_code": "optional_runtime_activation_busy",
                    "message": "Finish or stop active and queued runs before activating an optional runtime.",
                },
                status=409,
            )
        gate_token = None
        keep_gate = False
        try:
            body = await self._strict_runtime_control_json(
                request,
                allowed={"environmentId", "profileId", "specDigest", "consent"},
                required={"environmentId", "profileId", "specDigest", "consent"},
            )
            environment_id = body.get("environmentId")
            profile_id = body.get("profileId")
            spec_digest = body.get("specDigest")
            if (
                not isinstance(environment_id, str)
                or not re.fullmatch(r"runtime-[0-9]{1,16}-[0-9a-f]{8}", environment_id)
                or not isinstance(profile_id, str)
                or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", profile_id)
                or not isinstance(spec_digest, str)
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", spec_digest)
                or body.get("consent") is not True
            ):
                raise ValueError(
                    "environmentId, profileId, exact specDigest, and literal consent=true are required."
                )
            validate_optional_runtime_activation_request(
                profile_id, spec_digest, consent=True
            )
            gate_token = self._reserve_worker_runtime_gate(
                "optional_runtime_activation", environment_id
            )
            result = await asyncio.to_thread(
                activate_optional_runtime_environment,
                environment_id,
                profile_id,
                spec_digest,
                consent=True,
            )
            result = self._public_runtime_mutation_result(result)
            restarting = bool(result.get("restartRequired")) and self._schedule_optional_runtime_restart()
            if result.get("restartRequired"):
                os.environ["MODIFF_RUNTIME_OVERLAY_STATUS"] = "restart_required"
            keep_gate = restarting
            return web.json_response(
                {
                    "error": False,
                    **result,
                    "restarting": restarting,
                    "message": (
                        "The validated optional runtime is active. MoDiff is restarting."
                        if restarting
                        else "The validated optional runtime is active. Restart MoDiff to load it."
                        if result.get("restartRequired")
                        else "This optional runtime is already active."
                    ),
                }
            )
        except ValueError as exc:
            return web.json_response({"error": True, "message": str(exc)}, status=400)
        except (RuntimeError, OverlayInstallBusy) as exc:
            return web.json_response({"error": True, "message": str(exc)}, status=409)
        finally:
            if gate_token is not None and not keep_gate:
                self._release_worker_runtime_gate(gate_token)

    async def runtime_optional_runtime_rollback(self, request):
        if self.current_task or self.queued_tasks:
            return web.json_response(
                {
                    "error": True,
                    "error_code": "optional_runtime_rollback_busy",
                    "message": "Finish or stop active and queued runs before rolling back an optional runtime.",
                },
                status=409,
            )
        gate_token = None
        keep_gate = False
        try:
            body = await self._strict_runtime_control_json(
                request, allowed={"consent"}, required={"consent"}
            )
            if body.get("consent") is not True:
                raise ValueError("Literal consent=true is required to roll back an optional runtime.")
            gate_token = self._reserve_worker_runtime_gate(
                "optional_runtime_rollback", "previous_environment"
            )
            result = await asyncio.to_thread(rollback_optional_runtime_environment, consent=True)
            result = self._public_runtime_mutation_result(result)
            restarting = bool(result.get("restartRequired")) and self._schedule_optional_runtime_restart()
            if result.get("restartRequired"):
                os.environ["MODIFF_RUNTIME_OVERLAY_STATUS"] = "restart_required"
            keep_gate = restarting
            return web.json_response(
                {
                    "error": False,
                    **result,
                    "restarting": restarting,
                    "message": (
                        "The previous optional runtime is restored. MoDiff is restarting."
                        if restarting
                        else "The previous optional runtime is selected. Restart MoDiff to finish rollback."
                    ),
                }
            )
        except ValueError as exc:
            return web.json_response({"error": True, "message": str(exc)}, status=400)
        except (RuntimeError, OverlayInstallBusy) as exc:
            return web.json_response({"error": True, "message": str(exc)}, status=409)
        finally:
            if gate_token is not None and not keep_gate:
                self._release_worker_runtime_gate(gate_token)

    def _schedule_optional_runtime_restart(self):
        if os.environ.get("MODIFF_WORKER_SUPERVISED") != "1":
            return False

        def restart_worker():
            os._exit(SUPERVISED_RESTART_EXIT_CODE)

        timer = threading.Timer(0.75, restart_worker)
        timer.daemon = True
        timer.start()
        return True

    async def runtime_optimization_activate(self, request):
        if self.current_task or self.queued_tasks:
            return web.json_response(
                {
                    "error": True,
                    "error_code": "optimization_activation_busy",
                    "message": "Finish or stop active and queued runs before activating an optional runtime.",
                },
                status=409,
            )
        gate_token = None
        keep_gate = False
        try:
            body = await self._strict_runtime_control_json(
                request, allowed={"environmentId"}, required={"environmentId"}
            )
            environment_id = body.get("environmentId")
            if not isinstance(environment_id, str) or not re.fullmatch(
                r"runtime-[0-9]{1,16}-[0-9a-f]{8}", environment_id
            ):
                raise ValueError("A valid environmentId is required.")
            gate_token = self._reserve_worker_runtime_gate(
                "optimization_activation", environment_id
            )
            result = await asyncio.to_thread(
                activate_optimization_environment, environment_id
            )
            result = self._public_runtime_mutation_result(result)
            restarting = bool(result.get("restartRequired")) and self._schedule_optional_runtime_restart()
            if result.get("restartRequired"):
                os.environ["MODIFF_RUNTIME_OVERLAY_STATUS"] = "restart_required"
            keep_gate = restarting
            return web.json_response(
                {
                    "error": False,
                    **result,
                    "restarting": restarting,
                    "message": (
                        "The validated optional runtime is active. MoDiff is restarting."
                        if restarting
                        else "The validated optional runtime is active. Restart MoDiff to load it."
                        if result.get("restartRequired")
                        else "This optional runtime is already active."
                    ),
                }
            )
        except ValueError as exc:
            return web.json_response({"error": True, "message": str(exc)}, status=400)
        except (RuntimeError, OverlayInstallBusy) as exc:
            return web.json_response({"error": True, "message": str(exc)}, status=409)
        finally:
            if gate_token is not None and not keep_gate:
                self._release_worker_runtime_gate(gate_token)

    async def runtime_optimization_rollback(self, request):
        if self.current_task or self.queued_tasks:
            return web.json_response(
                {
                    "error": True,
                    "error_code": "optimization_rollback_busy",
                    "message": "Finish or stop active and queued runs before rolling back the optional runtime.",
                },
                status=409,
            )
        gate_token = None
        keep_gate = False
        try:
            await self._strict_runtime_control_json(
                request, allowed=set(), required=set(), allow_empty=True
            )
            gate_token = self._reserve_worker_runtime_gate(
                "optimization_rollback", "previous_environment"
            )
            result = await asyncio.to_thread(rollback_optimization_environment)
            result = self._public_runtime_mutation_result(result)
            restarting = bool(result.get("restartRequired")) and self._schedule_optional_runtime_restart()
            if result.get("restartRequired"):
                os.environ["MODIFF_RUNTIME_OVERLAY_STATUS"] = "restart_required"
            keep_gate = restarting
            return web.json_response(
                {
                    "error": False,
                    **result,
                    "restarting": restarting,
                    "message": (
                        "The previous optional runtime is restored. MoDiff is restarting."
                        if restarting
                        else "The previous optional runtime is selected. Restart MoDiff to finish rollback."
                    ),
                }
            )
        except ValueError as exc:
            return web.json_response({"error": True, "message": str(exc)}, status=400)
        except (RuntimeError, OverlayInstallBusy) as exc:
            return web.json_response({"error": True, "message": str(exc)}, status=409)
        finally:
            if gate_token is not None and not keep_gate:
                self._release_worker_runtime_gate(gate_token)

    async def runtime_optimization_enable(self, request):
        try:
            body = await request.json()
            capability_id = str(body.get("capabilityId") or "")
            enabled = body.get("enabled") is True
            _fingerprint, hardware, profile = self._optimization_runtime_context()
            catalog = public_optimization_catalog(runtime_profile=profile, hardware=hardware)
            capability = next(
                (item for item in catalog.get("capabilities", []) if item.get("id") == capability_id),
                None,
            )
            if not capability:
                raise ValueError("Unknown optimization capability.")
            if enabled and not capability.get("compatible"):
                raise ValueError(capability.get("disabledReason") or "This optimization is incompatible.")
            if enabled and not capability.get("canEnable"):
                raise ValueError(capability.get("disabledReason") or "This optimization is not available to enable.")
            if (
                enabled
                and capability.get("kind") in {"package", "profile", "external"}
                and not capability.get("installed")
            ):
                raise ValueError("Install and validate this package before enabling it.")
            state = set_optimization_capability_enabled(capability_id, enabled)
        except (ValueError, RuntimeError) as exc:
            return web.json_response({"error": True, "message": str(exc)}, status=400)
        return web.json_response(
            {
                "error": False,
                "state": state,
                "message": (
                    "Opt-in enabled. Auto will still require an exact qualified workload receipt."
                    if enabled
                    else "Opt-in disabled. Auto will not select this optimization."
                ),
            }
        )

    async def runtime_optimization_probe(self, request):
        if self.current_task:
            return web.json_response(
                {
                    "error": True,
                    "message": "Compatibility probes cannot run while a graph owns the accelerator.",
                },
                status=409,
            )
        try:
            body = await request.json()
            capability_id = str(body.get("capabilityId") or "")
            runtime_fingerprint, hardware, profile = self._optimization_runtime_context()
            catalog = public_optimization_catalog(runtime_profile=profile, hardware=hardware)
            capability = next(
                (item for item in catalog.get("capabilities", []) if item.get("id") == capability_id),
                None,
            )
            if not capability:
                raise ValueError("Unknown optimization capability.")
            if not capability.get("canEnable"):
                raise ValueError(capability.get("disabledReason") or "This optimization is unavailable.")
            receipt = await asyncio.to_thread(
                probe_optimization_capability,
                capability_id,
                runtime_fingerprint=runtime_fingerprint,
            )
        except (ValueError, RuntimeError) as exc:
            return web.json_response({"error": True, "message": str(exc)}, status=400)
        return web.json_response(
            {
                "error": False,
                "receipt": self._public_optimization_receipt(receipt),
                "message": (
                    "Compatibility probe passed. This does not authorize Auto until a real workload is qualified."
                    if receipt.get("status") == "probe_passed"
                    else "Compatibility probe failed. The optimization remains unavailable to Auto."
                ),
            }
        )

    async def runtime_optimization_receipts(self, _request):
        document = read_optimization_receipts()
        receipts = document.get("receipts") if isinstance(document, dict) else []
        if not isinstance(receipts, list):
            receipts = []
        return web.json_response(
            {
                "schemaVersion": 1,
                "receipts": [
                    projected
                    for item in receipts
                    if (projected := self._public_optimization_receipt(item)) is not None
                ][:500],
            }
        )

    async def runtime_optimization_qualify(self, request):
        try:
            body = await request.json()
            receipt = qualify_optimization_receipt(
                str(body.get("receiptId") or ""),
                output_reviewed=body.get("outputReviewed") is True,
            )
        except (ValueError, RuntimeError) as exc:
            return web.json_response({"error": True, "message": str(exc)}, status=400)
        return web.json_response(
            {
                "error": False,
                "receipt": self._public_optimization_receipt(receipt),
                "message": "This exact runtime, model, workload, and optimization selection is now eligible for Auto.",
            }
        )

    def _cuda_memory_snapshot(self):
        snapshot = {
            "available": False,
            "device_count": 0,
            "devices": [],
        }
        try:
            torch = import_module("torch")
            cuda_available = bool(torch.cuda.is_available())
            snapshot["available"] = cuda_available
            if cuda_available:
                device_count = int(torch.cuda.device_count())
                snapshot["device_count"] = device_count
                devices = []
                for index in range(device_count):
                    device = {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                    }
                    try:
                        properties = torch.cuda.get_device_properties(index)
                        device["total_memory"] = int(getattr(properties, "total_memory", 0))
                    except Exception as properties_error:
                        device["properties_error"] = str(properties_error)
                    try:
                        try:
                            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                        except TypeError:
                            with torch.cuda.device(index):
                                free_bytes, total_bytes = torch.cuda.mem_get_info()
                        device["free_bytes"] = int(free_bytes)
                        device["total_bytes"] = int(total_bytes)
                    except Exception as memory_error:
                        device["mem_get_info_error"] = str(memory_error)
                    try:
                        device["allocated_bytes"] = int(torch.cuda.memory_allocated(index))
                        device["reserved_bytes"] = int(torch.cuda.memory_reserved(index))
                        device["max_allocated_bytes"] = int(torch.cuda.max_memory_allocated(index))
                        device["max_reserved_bytes"] = int(torch.cuda.max_memory_reserved(index))
                    except Exception as stats_error:
                        device["memory_stats_error"] = str(stats_error)
                    devices.append(device)

                snapshot["devices"] = devices
                if devices:
                    first_device = devices[0]
                    snapshot["free_bytes"] = first_device.get("free_bytes")
                    snapshot["total_bytes"] = first_device.get("total_bytes")
                    snapshot["allocated_bytes"] = first_device.get("allocated_bytes")
                    snapshot["reserved_bytes"] = first_device.get("reserved_bytes")
                    snapshot["device_name"] = first_device.get("name")
        except Exception as e:
            snapshot["error"] = str(e)
        return snapshot

    def _gpu_process_snapshot(self):
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            return {
                "available": False,
                "reason": "nvidia-smi not found",
                "gpus": [],
                "processes": [],
            }

        snapshot = {
            "available": True,
            "gpus": [],
            "processes": [],
        }

        try:
            gpu_result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=index,name,memory.used,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=4,
            )
            if gpu_result.returncode == 0:
                for row in csv.reader(gpu_result.stdout.splitlines()):
                    if len(row) < 5:
                        continue
                    index, name, used_mb, free_mb, total_mb = [item.strip() for item in row[:5]]
                    snapshot["gpus"].append(
                        {
                            "index": int(index) if index.isdigit() else index,
                            "name": name,
                            "memory_used_mb": self._safe_int(used_mb),
                            "memory_free_mb": self._safe_int(free_mb),
                            "memory_total_mb": self._safe_int(total_mb),
                        }
                    )
            else:
                snapshot["gpu_query_error"] = gpu_result.stderr.strip() or gpu_result.stdout.strip()
        except Exception as e:
            snapshot["gpu_query_error"] = str(e)

        try:
            process_result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=4,
            )
            if process_result.returncode == 0:
                for row in csv.reader(process_result.stdout.splitlines()):
                    if len(row) < 4:
                        continue
                    gpu_uuid, pid, process_name, used_memory_mb = [item.strip() for item in row[:4]]
                    snapshot["processes"].append(
                        {
                            "gpu_uuid": gpu_uuid,
                            "pid": self._safe_int(pid),
                            "process_name": process_name,
                            "used_memory_mb": self._safe_int(used_memory_mb),
                        }
                    )
            else:
                snapshot["process_query_error"] = process_result.stderr.strip() or process_result.stdout.strip()
        except Exception as e:
            snapshot["process_query_error"] = str(e)

        return snapshot

    def _safe_int(self, value):
        try:
            return int(str(value).strip())
        except Exception:
            return None

    def _storage_kind_for_path(self, path, *, sys_dev_root=Path("/sys/dev/block")):
        """Return a storage kind only when Linux exposes an authoritative rotational flag."""
        if platform.system() != "Linux":
            return "unknown", None
        candidate = Path(path).expanduser()
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        try:
            device_number = candidate.stat().st_dev
            device_link = Path(sys_dev_root) / f"{os.major(device_number)}:{os.minor(device_number)}"
            device_path = device_link.resolve(strict=True)
        except (OSError, RuntimeError):
            return "unknown", None

        for device_or_parent in (device_path, *device_path.parents):
            rotational_path = device_or_parent / "queue" / "rotational"
            try:
                rotational = rotational_path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if rotational == "0":
                return "ssd", "linux-sysfs"
            if rotational == "1":
                return "hdd", "linux-sysfs"
            return "unknown", None
        return "unknown", None

    def _runtime_storage_snapshot(self):
        path = Path(self.data_dir).expanduser()
        existing_path = path
        while not existing_path.exists() and existing_path != existing_path.parent:
            existing_path = existing_path.parent
        kind, detection_source = self._storage_kind_for_path(existing_path)
        active_percent, activity_source = self._runtime_disk_activity_sampler.sample(existing_path)
        usage = shutil.disk_usage(existing_path)
        total = int(usage.total)
        free = int(usage.free)
        used = int(usage.used)
        return {
            "path": str(existing_path.resolve()),
            "totalBytes": total,
            "freeBytes": free,
            "usedBytes": used,
            "percent": (used / total * 100.0) if total > 0 else None,
            "activePercent": active_percent,
            "activitySource": activity_source,
            "kind": kind,
            "detectionSource": detection_source,
        }

    async def runtime_gpu_processes(self, request):
        return web.json_response(
            {
                "error": False,
                "cuda_memory_snapshot": self._cuda_memory_snapshot(),
                "gpu_processes": self._gpu_process_snapshot(),
            }
        )

    def _runtime_resource_snapshot(self, *, max_age_seconds=1.0):
        with self._runtime_resource_lock:
            now = time.monotonic()
            if isinstance(
                self._runtime_resource_cached_snapshot, dict
            ) and now - self._runtime_resource_cached_at <= max(0.0, float(max_age_seconds)):
                return deepcopy(self._runtime_resource_cached_snapshot)
            snapshot = self._collect_runtime_resource_snapshot()
            self._runtime_resource_cached_at = now
            self._runtime_resource_cached_snapshot = snapshot
            return deepcopy(snapshot)

    def _collect_runtime_resource_snapshot(self):
        sampled_at = time.time()
        system = {
            "cpuPercent": None,
            "ramTotalBytes": None,
            "ramAvailableBytes": None,
            "ramUsedBytes": None,
            "ramPercent": None,
        }
        process = {
            "cpuPercent": None,
            "rssBytes": None,
        }
        errors = []
        storage = {
            "path": None,
            "totalBytes": None,
            "freeBytes": None,
            "usedBytes": None,
            "percent": None,
            "activePercent": None,
            "activitySource": None,
            "kind": "unknown",
            "detectionSource": None,
        }

        try:
            storage = self._runtime_storage_snapshot()
        except Exception as exc:
            errors.append(f"storage telemetry: {exc}")

        try:
            import psutil

            memory = psutil.virtual_memory()
            system.update(
                {
                    "cpuPercent": float(psutil.cpu_percent(interval=None)),
                    "ramTotalBytes": int(memory.total),
                    "ramAvailableBytes": int(memory.available),
                    "ramUsedBytes": int(memory.total - memory.available),
                    "ramPercent": float(memory.percent),
                }
            )
            current_process = self._runtime_resource_process or psutil.Process()
            self._runtime_resource_process = current_process
            process.update(
                {
                    "cpuPercent": float(current_process.cpu_percent(interval=None)),
                    "rssBytes": int(current_process.memory_info().rss),
                }
            )
        except Exception as exc:
            errors.append(f"process telemetry: {exc}")

        cuda_snapshot = self._cuda_memory_snapshot()
        accelerators = []
        try:
            torch = import_module("torch")
            hardware_snapshot = get_hardware_snapshot(self.data_dir)
            topology_by_device = {
                str(item.get("device")): item
                for item in (hardware_snapshot.get("devices") or [])
                if isinstance(item, dict) and item.get("device")
            }
            hip_version = getattr(getattr(torch, "version", None), "hip", None)
            runtime_backend = "rocm" if hip_version else "cuda"
            ram_total = system.get("ramTotalBytes")
            for raw_device in cuda_snapshot.get("devices") or []:
                device_name = f"cuda:{raw_device.get('index', len(accelerators))}"
                topology = topology_by_device.get(device_name, {})
                total = self._safe_int(raw_device.get("total_bytes") or raw_device.get("total_memory"))
                free = self._safe_int(raw_device.get("free_bytes"))
                allocated = self._safe_int(raw_device.get("allocated_bytes"))
                reserved = self._safe_int(raw_device.get("reserved_bytes"))
                used = max(0, total - free) if total is not None and free is not None else reserved
                shared = topology.get("memory_kind") == "shared" or bool(
                    hip_version and total is not None and ram_total is not None and total >= int(ram_total * 0.75)
                )
                planning_total = self._safe_int(topology.get("planning_memory_total")) or total
                planning_free = self._safe_int(topology.get("planning_memory_free"))
                if planning_free is None:
                    planning_free = free
                accelerators.append(
                    {
                        "device": device_name,
                        "index": raw_device.get("index", len(accelerators)),
                        "name": raw_device.get("name") or raw_device.get("device_name") or runtime_backend.upper(),
                        "backend": runtime_backend,
                        "vendor": topology.get("vendor") or ("amd" if hip_version else "nvidia"),
                        "architecture": topology.get("architecture"),
                        "memoryKind": "shared" if shared else "dedicated",
                        "utilizationPercent": None,
                        "utilizationSource": None,
                        "memoryTotalBytes": planning_total,
                        "memoryFreeBytes": planning_free,
                        "memoryUsedBytes": (
                            max(0, planning_total - planning_free)
                            if planning_total is not None and planning_free is not None
                            else used
                        ),
                        "accessibleMemoryTotalBytes": total,
                        "accessibleMemoryFreeBytes": free,
                        "dedicatedMemoryTotalBytes": self._safe_int(topology.get("dedicated_memory_total")),
                        "dedicatedMemoryFreeBytes": self._safe_int(topology.get("dedicated_memory_free")),
                        "sharedMemoryTotalBytes": self._safe_int(topology.get("shared_memory_total")),
                        "sharedMemoryFreeBytes": self._safe_int(topology.get("shared_memory_free")),
                        "allocatedBytes": allocated,
                        "reservedBytes": reserved,
                        "peakAllocatedBytes": self._safe_int(raw_device.get("max_allocated_bytes")),
                        "peakReservedBytes": self._safe_int(raw_device.get("max_reserved_bytes")),
                    }
                )

            xpu = getattr(torch, "xpu", None)
            if not accelerators and xpu is not None and bool(getattr(xpu, "is_available", lambda: False)()):
                for index in range(int(xpu.device_count())):
                    device_name = f"xpu:{index}"
                    topology = topology_by_device.get(device_name, {})
                    total = free = allocated = reserved = None
                    try:
                        properties = xpu.get_device_properties(index)
                        total = self._safe_int(getattr(properties, "total_memory", None))
                    except Exception:
                        pass
                    try:
                        free, runtime_total = xpu.mem_get_info(index)
                        free = self._safe_int(free)
                        total = total or self._safe_int(runtime_total)
                    except Exception:
                        pass
                    try:
                        allocated = self._safe_int(xpu.memory_allocated(index))
                        reserved = self._safe_int(xpu.memory_reserved(index))
                    except Exception:
                        pass
                    accelerators.append(
                        {
                            "device": device_name,
                            "index": index,
                            "name": str(getattr(xpu, "get_device_name", lambda _index: f"Intel XPU {index}")(index)),
                            "backend": "xpu",
                            "vendor": "intel",
                            "architecture": topology.get("architecture"),
                            "memoryKind": topology.get("memory_kind") or "dedicated",
                            "utilizationPercent": None,
                            "utilizationSource": None,
                            "memoryTotalBytes": self._safe_int(topology.get("planning_memory_total")) or total,
                            "memoryFreeBytes": self._safe_int(topology.get("planning_memory_free")) or free,
                            "accessibleMemoryTotalBytes": total,
                            "accessibleMemoryFreeBytes": free,
                            "dedicatedMemoryTotalBytes": self._safe_int(topology.get("dedicated_memory_total")),
                            "dedicatedMemoryFreeBytes": self._safe_int(topology.get("dedicated_memory_free")),
                            "sharedMemoryTotalBytes": self._safe_int(topology.get("shared_memory_total")),
                            "sharedMemoryFreeBytes": self._safe_int(topology.get("shared_memory_free")),
                            "memoryUsedBytes": max(0, total - free)
                            if total is not None and free is not None
                            else reserved,
                            "allocatedBytes": allocated,
                            "reservedBytes": reserved,
                        }
                    )

            mps = getattr(torch, "mps", None)
            if not accelerators and bool(
                getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available()
            ):
                allocated = None
                driver_allocated = None
                recommended_max = None
                try:
                    allocated = self._safe_int(mps.current_allocated_memory())
                    driver_allocated = self._safe_int(mps.driver_allocated_memory())
                    recommended = getattr(mps, "recommended_max_memory", None)
                    recommended_max = self._safe_int(recommended()) if callable(recommended) else None
                except Exception:
                    pass
                accelerators.append(
                    {
                        "device": "mps:0",
                        "index": 0,
                        "name": "Apple Metal Performance Shaders",
                        "backend": "mps",
                        "vendor": "apple",
                        "architecture": platform.machine(),
                        "memoryKind": "shared",
                        "utilizationPercent": None,
                        "utilizationSource": None,
                        "memoryTotalBytes": recommended_max,
                        "memoryFreeBytes": (
                            max(0, recommended_max - driver_allocated)
                            if recommended_max is not None and driver_allocated is not None
                            else None
                        ),
                        "memoryUsedBytes": driver_allocated,
                        "allocatedBytes": allocated,
                        "reservedBytes": driver_allocated,
                    }
                )
        except Exception as exc:
            errors.append(f"accelerator telemetry: {exc}")

        # Whole-device utilization is deliberately separate from Torch's
        # allocator counters. Prefer low-overhead vendor sources and leave the
        # value unavailable rather than inventing activity from allocation.
        try:
            if accelerators and accelerators[0].get("backend") == "cuda":
                nvidia_smi = shutil.which("nvidia-smi")
                if not nvidia_smi:
                    raise FileNotFoundError("nvidia-smi is unavailable")
                result = subprocess.run(
                    [
                        nvidia_smi,
                        "--query-gpu=index,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=1.5,
                )
                if result.returncode == 0:
                    for row in csv.reader(result.stdout.splitlines()):
                        if len(row) < 2:
                            continue
                        index = self._safe_int(row[0])
                        utilization = self._safe_int(row[1])
                        for accelerator in accelerators:
                            if accelerator.get("index") == index:
                                accelerator["utilizationPercent"] = utilization
                                accelerator["utilizationSource"] = "nvidia-smi"
            elif accelerators and accelerators[0].get("backend") == "rocm":
                amd_cards = []
                drm_root = Path("/sys/class/drm")
                if drm_root.is_dir():
                    for busy_path in sorted(drm_root.glob("card*/device/gpu_busy_percent")):
                        vendor_path = busy_path.parent / "vendor"
                        try:
                            if vendor_path.read_text(encoding="utf-8").strip().lower() != "0x1002":
                                continue
                            amd_cards.append(self._safe_int(busy_path.read_text(encoding="utf-8").strip()))
                        except Exception:
                            continue
                for index, utilization in enumerate(amd_cards):
                    if index < len(accelerators):
                        accelerators[index]["utilizationPercent"] = utilization
                        accelerators[index]["utilizationSource"] = "sysfs"
        except Exception as exc:
            errors.append(f"device utilization: {exc}")

        active_device = None
        if self.current_task:
            runtime_hints = self.current_task.get("runtimeHints")
            if isinstance(runtime_hints, dict):
                active_device = runtime_hints.get("device")
        if not active_device and accelerators:
            active_device = accelerators[0].get("device")

        current_run = None
        if self.current_task:
            current_run = {
                "taskId": self.current_task.get("task_id"),
                "name": self.current_task.get("name"),
                "startedAt": self.current_task.get("started_at"),
                "progress": self.current_task.get("progress"),
            }

        return {
            "schemaVersion": 1,
            "sampledAt": sampled_at,
            "system": system,
            "process": process,
            "storage": storage,
            "activeDevice": active_device,
            "accelerators": accelerators,
            "currentRun": current_run,
            "errors": errors,
        }

    async def runtime_resources(self, _request):
        # psutil and vendor probes are blocking system calls. Keep them off the
        # aiohttp loop so resource telemetry cannot delay graph or queue APIs.
        return web.json_response(await asyncio.to_thread(self._runtime_resource_snapshot))

    def _best_effort_device_cache_clear(self):
        errors = []
        try:
            torch = import_module("torch")
        except Exception as e:
            return [f"torch import failed: {e}"]

        if torch.cuda.is_available():
            for label, callback in (
                ("torch.cuda.empty_cache", torch.cuda.empty_cache),
                ("torch.cuda.ipc_collect", torch.cuda.ipc_collect),
            ):
                try:
                    callback()
                except Exception as e:
                    logger.debug(f"{label} failed during accelerator cleanup", exc_info=True)
                    errors.append(f"{label}: {e}")

        mps = getattr(torch, "mps", None)
        if mps is not None:
            try:
                if mps.is_available():
                    mps.empty_cache()
            except Exception as e:
                logger.debug("torch.mps.empty_cache failed during accelerator cleanup", exc_info=True)
                errors.append(f"torch.mps.empty_cache: {e}")

        return errors

    def _best_effort_allocator_trim(self):
        """Return freed glibc arenas to the OS after large unified-memory pipelines."""
        if platform.system() != "Linux":
            return False, []
        try:
            import ctypes

            libc = ctypes.CDLL(None)
            malloc_trim = getattr(libc, "malloc_trim", None)
            if malloc_trim is None:
                return False, []
            malloc_trim.argtypes = [ctypes.c_size_t]
            malloc_trim.restype = ctypes.c_int
            return bool(malloc_trim(0)), []
        except Exception as e:
            logger.debug("malloc_trim failed during accelerator cleanup", exc_info=True)
            return False, [f"malloc_trim: {e}"]

    def _release_modular_diffusers_components(self):
        try:
            modular_diffusers = import_module("modules.ModularDiffusers")
            manager = getattr(modular_diffusers, "components", None)
        except Exception as e:
            return 0, [f"Modular Diffusers components unavailable: {e}"]

        if manager is None:
            return 0, []

        components_dict = getattr(manager, "components", None)
        released_count = len(components_dict) if components_dict is not None else 0
        errors = []

        try:
            torch = import_module("torch")
        except Exception:
            torch = None

        hooks = list(getattr(manager, "model_hooks", None) or [])
        for hook in hooks:
            for label, callback in (
                ("offload", getattr(hook, "offload", None)),
                ("remove", getattr(hook, "remove", None)),
            ):
                if callback is None:
                    continue
                try:
                    callback()
                except Exception as e:
                    logger.debug(f"Modular Diffusers hook {label} failed during cleanup", exc_info=True)
                    errors.append(f"Modular Diffusers hook {label}: {e}")

        try:
            manager.model_hooks = None
            manager._auto_offload_enabled = False
            if hasattr(manager, "_auto_offload_device"):
                manager._auto_offload_device = None
        except Exception as e:
            errors.append(f"Modular Diffusers offload reset: {e}")

        if components_dict is not None:
            for component_id, component in list(components_dict.items()):
                try:
                    if torch is not None and isinstance(component, torch.nn.Module):
                        component.to("cpu")
                except Exception as e:
                    logger.debug(f"Could not move Modular Diffusers component {component_id} to CPU", exc_info=True)
                    errors.append(f"Modular Diffusers component {component_id}: {e}")

            try:
                components_dict.clear()
            except Exception as e:
                errors.append(f"Modular Diffusers component clear: {e}")

        for attr in ("added_time", "collections"):
            try:
                value = getattr(manager, attr, None)
                if value is not None:
                    value.clear()
            except Exception as e:
                errors.append(f"Modular Diffusers {attr} clear: {e}")

        return released_count, errors

    def _release_diffusers_offload_cache(self):
        offload_path = Path("data") / "offload" / "diffusers"
        if not offload_path.exists():
            return 0, []

        errors = []
        released_count = 0
        try:
            released_count = sum(1 for item in offload_path.rglob("*") if item.is_file())
            shutil.rmtree(offload_path)
        except Exception as e:
            logger.debug("Diffusers disk offload cache cleanup failed", exc_info=True)
            errors.append(f"Diffusers disk offload cache: {e}")
        return released_count, errors

    async def runtime_gpu_cleanup(self, request):
        # Clearing node_cache while a graph node is executing invalidates the
        # executor's own node object.  The node can finish its model call and
        # then fail with a KeyError for its node id when execution metrics or
        # outputs are recorded.  Refuse cleanup while the worker owns a task;
        # callers can retry once /queue reports no current task.
        if self.current_task:
            task_id = self.current_task.get("task_id")
            return web.json_response(
                {
                    "error": True,
                    "error_code": "runtime_cleanup_busy",
                    "message": "Accelerator cleanup cannot run while a task is active. Wait for the current task to finish, then retry.",
                    "task_id": task_id,
                },
                status=409,
            )

        before = self._cuda_memory_snapshot()
        cleanup_errors = []
        released_nodes = len(self.node_cache)
        released_models = 0
        released_diffusers_components = 0
        released_offload_files = 0

        # Drop memory-manager ownership before destroying cached nodes. Node
        # destructors call MemoryManager.remove() for their tracked ids; if the
        # manager still owns a large Accelerate-offloaded pipeline, remove()
        # tries to materialize it on CPU and flushes once per id. On unified
        # memory ROCm systems that turns cleanup into minutes of RAM/swap
        # thrashing. With the manager detached first, those destructor calls
        # are no-ops and the pipeline references are released exactly once.
        try:
            released_models = memory_manager.clear()
        except Exception as e:
            logger.debug("Failed to clear managed models during accelerator cleanup", exc_info=True)
            cleanup_errors.append(f"memory manager: {e}")
            try:
                released_models = len(memory_manager.cache)
                memory_manager.cache.clear()
            except Exception as clear_error:
                cleanup_errors.append(f"memory manager fallback: {clear_error}")

        try:
            self.node_cache.clear()
        except Exception as e:
            logger.debug("Failed to clear node cache during accelerator cleanup", exc_info=True)
            cleanup_errors.append(f"node cache: {e}")

        released_diffusers_components, diffusers_errors = self._release_modular_diffusers_components()
        cleanup_errors.extend(diffusers_errors)

        released_offload_files, offload_cache_errors = self._release_diffusers_offload_cache()
        cleanup_errors.extend(offload_cache_errors)

        try:
            gc.collect()
        except Exception as e:
            cleanup_errors.append(f"gc.collect: {e}")

        cleanup_errors.extend(self._best_effort_device_cache_clear())
        allocator_trimmed, allocator_errors = self._best_effort_allocator_trim()
        cleanup_errors.extend(allocator_errors)
        after = self._cuda_memory_snapshot()

        message = (
            f"Accelerator cleanup complete. Released {released_nodes} cached node object(s), "
            f"{released_models} managed model(s), {released_diffusers_components} Modular Diffusers component(s), "
            f"and {released_offload_files} Diffusers disk offload file(s)."
        )
        if cleanup_errors:
            message += " Some cleanup calls reported errors but references were dropped where possible."

        return web.json_response(
            {
                "error": False,
                "message": message,
                "released_node_count": released_nodes,
                "released_model_count": released_models,
                "released_diffusers_component_count": released_diffusers_components,
                "released_diffusers_offload_file_count": released_offload_files,
                "allocator_trimmed": allocator_trimmed,
                "cleanup_errors": cleanup_errors,
                "before": before,
                "after": after,
            }
        )

    async def model_capabilities(self, request):
        query = str(request.query.get("q", "")).lower().strip()
        optional_runtime_catalog_snapshot = None
        execution_specs = validate_studio_execution_specs(self.modules)
        specs_by_model = {}
        for specification in execution_specs:
            specs_by_model.setdefault(specification["modelType"], []).append(specification)

        def request_optional_runtime_catalog():
            nonlocal optional_runtime_catalog_snapshot
            if optional_runtime_catalog_snapshot is None:
                optional_runtime_catalog_snapshot = public_optional_runtime_catalog()
            return optional_runtime_catalog_snapshot

        profiles_by_model = {}
        for profile in public_execution_profiles(
            observe_optional_runtime=True,
            optional_runtime_catalog_resolver=request_optional_runtime_catalog,
        ):
            profiles_by_model.setdefault(profile.get("model_type"), []).append(profile)

        capabilities = []
        for raw_capability in STUDIO_MODEL_CAPABILITIES.values():
            capability = dict(raw_capability)
            profiles = profiles_by_model.get(capability.get("modelType"), [])
            pipeline_classes = sorted(
                {profile.get("pipeline_class") for profile in profiles if profile.get("pipeline_class")}
            )
            runnable_modes = sorted({mode for profile in profiles for mode in profile.get("modes", [])})
            output_kind = capability.get("outputKind") or "image"
            additional_requirements = capability.get("additionalRequirements") or []
            artifact_candidates = list(capability.get("artifactCandidates") or [capability.get("defaultRepo")])
            artifact_candidates.extend(
                artifact
                for profile in profiles
                for artifact in (profile.get("default_repo"), profile.get("fallback_repo"))
                if artifact
            )
            artifact_candidates.extend(
                requirement.get("repo") for requirement in additional_requirements if requirement.get("repo")
            )
            quantized_components = sorted(
                {component for profile in profiles for component in profile.get("quantizable_components", [])}
            )
            optional_runtime_profile_ids = list(
                dict.fromkeys(
                    profile_id
                    for profile in profiles
                    for profile_id in profile.get("optional_runtime_profiles", [])
                )
            )
            capability.update(
                {
                    "schemaVersion": 2,
                    "mediaKind": output_kind,
                    "supportTier": capability.get("supportTier") or "supported",
                    "pipelineClasses": pipeline_classes,
                    "executionProfiles": profiles,
                    "studioExecutionSpecs": specs_by_model.get(capability.get("modelType"), []),
                    "runnableModes": runnable_modes,
                    "inputContracts": capability.get("modeRequirements") or {},
                    "parameterAliases": {
                        "modelRepository": ["model_id", "model", "repo"],
                        "guidanceScale": ["guidance_scale", "true_cfg_scale", "guidance"],
                        "steps": ["num_inference_steps", "steps"],
                        "sourceImage": ["image", "reference_images"],
                        "maskImage": ["mask_image", "mask"],
                        "controlImage": ["control_image", "conditioning_image"],
                    },
                    "defaults": {
                        "dtype": capability.get("defaultDtype"),
                        "size": capability.get("defaultSize"),
                        "steps": capability.get("recommendedSteps"),
                        "guidanceScale": capability.get("recommendedGuidance"),
                        "offloadMode": (capability.get("offloadSupport") or {}).get("default"),
                    },
                    "artifactCandidates": list(dict.fromkeys(filter(None, artifact_candidates))),
                    "revisionCandidates": capability.get("revisionCandidates") or [],
                    "quantizationSupport": {
                        "defaultMode": (capability.get("lowVram") or {}).get("quantizationMode", "none"),
                        "components": quantized_components,
                        "offloadModes": (capability.get("offloadSupport") or {}).get("modes", []),
                    },
                    "qualificationStatus": capability.get("qualificationStatus") or "graph-qualified",
                    "optionalRuntimeProfileIds": optional_runtime_profile_ids,
                    "optionalRuntimeProfiles": public_optional_runtime_profiles(
                        optional_runtime_profile_ids
                    ),
                    "optionalRuntimeRequirement": optional_runtime_requirement_for_execution(
                        capability.get("modelType"),
                        catalog_resolver=request_optional_runtime_catalog,
                    ),
                }
            )
            if capability["studioExecutionSpecs"]:
                capability["studioExecutionSpecSchemaVersion"] = 1
                capability["studioExecutionSpecModes"] = sorted(
                    specification["mode"] for specification in capability["studioExecutionSpecs"]
                )
            capabilities.append(capability)
        task_template_contracts = build_task_template_contracts(capabilities, execution_specs)
        task_contracts_by_model = {}
        for contract in task_template_contracts:
            task_contracts_by_model.setdefault(contract["modelType"], []).append(contract)
        for capability in capabilities:
            contracts = task_contracts_by_model.get(capability["modelType"], [])
            capability["taskTemplateContracts"] = contracts
            if contracts:
                capability["taskTemplateContractSchemaVersion"] = TASK_TEMPLATE_CONTRACT_SCHEMA_VERSION
                capability["taskTemplateContractModes"] = sorted(contract["mode"] for contract in contracts)
        if query:
            capabilities = [
                capability
                for capability in capabilities
                if query in capability.get("modelType", "").lower()
                or query in capability.get("label", "").lower()
                or query in capability.get("family", "").lower()
                or query in capability.get("defaultRepo", "").lower()
            ]
            returned_models = {capability["modelType"] for capability in capabilities}
            task_template_contracts = [
                contract for contract in task_template_contracts if contract["modelType"] in returned_models
            ]

        return web.json_response(
            {
                "error": False,
                "schemaVersion": 2,
                "count": len(capabilities),
                "capabilities": capabilities,
                "taskTemplateContractSchemaVersion": TASK_TEMPLATE_CONTRACT_SCHEMA_VERSION,
                "taskTemplateContracts": task_template_contracts,
                "diffusersExecutionProfiles": public_execution_profiles(
                    observe_optional_runtime=True,
                    optional_runtime_catalog_resolver=request_optional_runtime_catalog,
                ),
                "studioExecutionSpecs": execution_specs,
                "optionalRuntimeProfiles": public_optional_runtime_profiles(),
                "experimentalCapabilities": public_experimental_pipelines(
                    observe_optional_runtime=True,
                    optional_runtime_catalog_resolver=request_optional_runtime_catalog,
                ),
                "source": "modiff-backend",
            }
        )

    def _auto_resource_runtime_block(self):
        cached = (
            self._last_runtime_fingerprint.get("hardware")
            if self.current_task and isinstance(self._last_runtime_fingerprint, dict)
            else None
        )
        hardware = deepcopy(cached) if isinstance(cached, dict) else get_hardware_snapshot(self.data_dir, refresh=True)
        profile = runtime_profile(hardware, venv=Path(sys.prefix))
        if profile.get("execution_ready"):
            return None
        profile_issues = [issue for issue in profile.get("issues", []) if issue.get("severity") == "error"]
        message = (
            profile_issues[0].get("message")
            if profile_issues
            else "The managed runtime environment is not ready for execution."
        )
        return {
            "error": False,
            "schemaVersion": 2,
            "resourceMode": "auto",
            "status": "needs_setup",
            "readiness": "needs_setup",
            "statusLabel": "Runtime needs repair",
            "blockingReason": message,
            "healthBadge": "Needs setup",
            "compatibility": {
                "state": "needs_setup",
                "severity": "error",
                "code": "runtime_profile_mismatch",
                "summary": "Runtime needs repair",
                "detail": message,
                "action": {
                    "type": "repair_environment",
                    "label": "Open Setup",
                    "command": profile.get("repair_command"),
                },
                "source": "backend_auto_planner",
            },
            "canAutoRun": False,
            "issue": {
                "code": "runtime_profile_mismatch",
                "category": "environment",
                "message": message,
                "issues": profile_issues,
            },
            "repairAction": {
                "type": "open_setup",
                "label": "Open Setup",
                "command": profile.get("repair_command"),
            },
            "runtimeProfile": profile,
            "selectedCandidate": None,
            "candidates": [],
            "checkedAt": int(time.time() * 1000),
        }

    def _auto_planning_runtime_fingerprint(self):
        """Report capacity available after releasing MoDiff-owned CUDA cache.

        Auto plans are requested between graph runs while the preceding
        pipeline is intentionally kept resident for reuse.  Sampling raw free
        VRAM at that point makes the planner count MoDiff's own reusable cache
        as external pressure.  A high-memory native recipe can consequently
        downshift to CPU offload, which changes the runtime signature and
        evicts the exact cache the next graph could have reused.

        Add only this worker's PyTorch reservation back to the free-memory
        sample, capped by physical capacity.  Memory owned by other processes
        remains unavailable, so real external pressure still selects a safer
        plan.
        """
        fingerprint = self._runtime_fingerprint_for_control_request()
        # During an active model call the cached fingerprint was captured
        # immediately before execution and already describes the capacity that
        # will be available after that run. Do not call torch.cuda memory APIs
        # here: some ROCm/CUDA loaders hold runtime locks while materializing
        # weights, and a refresh-time planning request must remain responsive.
        if self.current_task:
            return fingerprint
        if not (self.node_cache or memory_manager.cache):
            return fingerprint

        hardware = fingerprint.get("hardware") if isinstance(fingerprint, dict) else None
        devices = hardware.get("devices") if isinstance(hardware, dict) else None
        if not isinstance(devices, list):
            return fingerprint

        try:
            torch = import_module("torch")
            if not bool(torch.cuda.is_available()):
                return fingerprint
        except Exception:
            return fingerprint

        adjusted = deepcopy(fingerprint)
        adjusted_hardware = adjusted.get("hardware")
        adjusted_devices = adjusted_hardware.get("devices") if isinstance(adjusted_hardware, dict) else None
        if not isinstance(adjusted_devices, list):
            return fingerprint

        reclaimable_by_index = {}
        for index in range(int(torch.cuda.device_count())):
            try:
                reclaimable_by_index[index] = max(0, int(torch.cuda.memory_reserved(index)))
            except Exception:
                reclaimable_by_index[index] = 0

        for device in adjusted_devices:
            if not isinstance(device, dict) or device.get("type") != "cuda":
                continue
            try:
                index = int(device.get("index") or 0)
            except (TypeError, ValueError):
                index = 0
            reclaimable = reclaimable_by_index.get(index, 0)
            if reclaimable <= 0:
                continue
            totals = [
                value
                for value in (
                    device.get("torch_vram_total"),
                    device.get("vram_total"),
                )
                if isinstance(value, int) and value > 0
            ]
            total = min(totals) if totals else None
            for key in ("torch_vram_free", "vram_free"):
                free = device.get(key)
                if not isinstance(free, int):
                    continue
                device[key] = min(total, free + reclaimable) if total is not None else free + reclaimable
            device["modiff_reclaimable_vram"] = reclaimable

        torch_state = adjusted_hardware.get("torch") if isinstance(adjusted_hardware, dict) else None
        if isinstance(torch_state, dict):
            reclaimable = reclaimable_by_index.get(0, 0)
            free = torch_state.get("cuda_memory_free_bytes")
            total = torch_state.get("cuda_memory_total_bytes") or torch_state.get("cuda_device_total_memory_bytes")
            if reclaimable > 0 and isinstance(free, int):
                torch_state["cuda_memory_free_bytes"] = (
                    min(total, free + reclaimable) if isinstance(total, int) and total > 0 else free + reclaimable
                )
        return adjusted

    async def auto_resource_plan(self, request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        try:
            runtime_block = self._auto_resource_runtime_block()
            if runtime_block:
                return web.json_response(runtime_block)
            plan = build_auto_resource_plan(
                payload if isinstance(payload, dict) else {},
                runtime_fingerprint=self._auto_planning_runtime_fingerprint(),
                local_models=get_local_models(),
                data_dir=self.data_dir,
                history=read_auto_resource_history(self.data_dir),
            )
            return web.json_response(plan)
        except Exception as exc:
            return web.json_response(
                {
                    "error": True,
                    "schemaVersion": 2,
                    "status": "needs_setup",
                    "statusLabel": "Needs setup",
                    "message": str(exc) or type(exc).__name__,
                    "compatibility": {
                        "state": "needs_setup",
                        "severity": "error",
                        "code": "auto_planner_error",
                        "summary": "Auto planning failed",
                        "detail": str(exc) or type(exc).__name__,
                        "action": {"type": "open_setup", "label": "Open Setup"},
                        "source": "backend_auto_planner",
                    },
                },
                status=500,
            )

    async def model_artifact_catalog(self, request):
        try:
            live_metadata = None
            if request.query.get("refresh") in {"1", "true", "yes"}:
                live_metadata = await asyncio.to_thread(
                    refreshed_hub_metadata,
                    model_type=request.query.get("modelType"),
                    repo=request.query.get("repo"),
                )
            return web.json_response(
                {
                    "error": False,
                    **public_model_artifact_catalog(),
                    "liveMetadata": live_metadata,
                }
            )
        except Exception as exc:
            return web.json_response(
                {
                    "error": True,
                    "message": str(exc) or type(exc).__name__,
                    "models": [],
                },
                status=500,
            )

    async def auto_resource_plans(self, request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        try:
            runtime_block = self._auto_resource_runtime_block()
            if runtime_block:
                forms = payload.get("forms", []) if isinstance(payload, dict) else []
                keys = payload.get("keys", []) if isinstance(payload, dict) else []
                plans = []
                for index, _form in enumerate(forms):
                    plan = {**runtime_block, "requestIndex": index}
                    if index < len(keys) and keys[index]:
                        plan["planKey"] = str(keys[index])
                    plans.append(plan)
                return web.json_response(
                    {
                        "error": False,
                        "schemaVersion": 2,
                        "resourceMode": "auto",
                        "count": len(plans),
                        "plans": plans,
                        "checkedAt": int(time.time() * 1000),
                    }
                )
            result = build_auto_resource_plans(
                payload if isinstance(payload, dict) else {},
                runtime_fingerprint=self._auto_planning_runtime_fingerprint(),
                local_models=get_local_models(),
                data_dir=self.data_dir,
            )
            return web.json_response(result)
        except Exception as exc:
            return web.json_response(
                {
                    "error": True,
                    "message": str(exc) or type(exc).__name__,
                    "plans": [],
                    "count": 0,
                },
                status=500,
            )

    async def auto_resource_history(self, _request):
        return web.json_response(
            {
                "error": False,
                "history": read_auto_resource_history(self.data_dir),
            }
        )

    async def media_assets_list(self, _request):
        from modiff.media_assets import list_media_assets

        assets = list_media_assets()
        return web.json_response({"error": False, "assets": assets, "count": len(assets)})

    async def media_assets_cleanup(self, request):
        from modiff.media_assets import cleanup_media_assets

        if self.current_task is not None:
            return web.json_response(
                {
                    "error": True,
                    "message": "Temporary media cannot be cleaned while a generation is active.",
                },
                status=409,
            )
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        payload = payload if isinstance(payload, dict) else {}
        scope = str(payload.get("scope") or "all_unpinned")
        task_id = str(payload.get("taskId") or "").strip() or None
        older_seconds = None
        if scope == "older_than":
            older_seconds = max(0.0, float(payload.get("olderThanHours") or 24)) * 3600
        elif scope == "task":
            if not task_id:
                return web.json_response({"error": True, "message": "Task cleanup needs a taskId."}, status=400)
        elif scope != "all_unpinned":
            return web.json_response(
                {"error": True, "message": f"Unsupported media cleanup scope {scope!r}."}, status=400
            )
        report = cleanup_media_assets(
            task_id=task_id if scope == "task" else None,
            older_than_seconds=older_seconds,
        )
        return web.json_response({"error": bool(report["errors"]), **report})

    async def auto_resource_history_clear(self, request):
        model_type = request.query.get("modelType") or None
        mode = request.query.get("mode") or None
        artifact = request.query.get("artifact") or None
        removed = clear_auto_resource_history(
            self.data_dir,
            model_type=model_type,
            mode=mode,
            artifact=artifact,
        )
        return web.json_response(
            {
                "error": False,
                "removed": removed,
                "history": read_auto_resource_history(self.data_dir),
            }
        )

    def _timestamp_seconds(self, value):
        if hasattr(value, "timestamp"):
            return int(value.timestamp())
        if isinstance(value, (int, float)):
            return int(value)
        return None

    def _model_revision_records(self, model):
        revisions = []
        for index, revision in enumerate(model.get("revisions") or []):
            commit_hash = revision.get("hash") if isinstance(revision, dict) else None
            if not commit_hash:
                continue
            revisions.append(
                {
                    "hash": commit_hash,
                    "size": revision.get("size", 0),
                    "lastModified": self._timestamp_seconds(revision.get("last_modified")),
                    "order": index,
                }
            )
        revisions.sort(key=lambda item: ((item.get("lastModified") or 0), item.get("order") or 0))
        return revisions

    def _model_fingerprint(self, repo_id, revisions):
        payload = {
            "repoId": repo_id,
            "revisions": [revision.get("hash") for revision in revisions],
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    async def model_fingerprints(self, request):
        query = str(request.query.get("q", "")).lower().strip()
        model_type = str(request.query.get("modelType", "")).strip()
        repo_query = str(request.query.get("repo") or request.query.get("repoId") or "").strip()
        capability_by_model = {
            capability.get("modelType"): capability for capability in STUDIO_MODEL_CAPABILITIES.values()
        }
        repo_to_model_types = {}
        for capability in STUDIO_MODEL_CAPABILITIES.values():
            repo_id = capability.get("defaultRepo")
            if repo_id:
                repo_to_model_types.setdefault(repo_id, []).append(capability.get("modelType"))

        requested_repos = []
        if model_type and model_type in capability_by_model:
            default_repo = capability_by_model[model_type].get("defaultRepo")
            if default_repo:
                requested_repos.append(default_repo)
        if repo_query:
            requested_repos.append(repo_query)
        requested_repo_set = {repo.lower() for repo in requested_repos}

        models = []
        found_repo_ids = set()
        local_models = get_local_models()
        for model in local_models:
            repo_id = model.get("id")
            if not repo_id:
                continue
            repo_id_lower = repo_id.lower()
            if requested_repo_set and repo_id_lower not in requested_repo_set:
                continue
            if (
                query
                and query not in repo_id_lower
                and not any(query in str(name).lower() for name in model.get("class_names", []))
            ):
                continue

            revisions = self._model_revision_records(model)
            selected_revision = revisions[-1]["hash"] if revisions else None
            install_status = artifact_cache_status(repo_id, local_models)
            complete = bool(install_status.get("complete"))
            found_repo_ids.add(repo_id_lower)
            models.append(
                {
                    "repoId": repo_id,
                    "modelTypes": repo_to_model_types.get(repo_id, []),
                    "cached": bool(install_status.get("installed")),
                    "installed": complete,
                    "complete": complete,
                    "repairRequired": bool(install_status.get("repairRequired")),
                    "installReason": install_status.get("reason"),
                    "activeFiles": install_status.get("activeFiles") or [],
                    "missingFiles": install_status.get("missingFiles") or [],
                    "corruptFiles": install_status.get("corruptFiles") or [],
                    "selectedRevision": selected_revision,
                    "modelRevision": f"{repo_id}@{selected_revision}" if selected_revision else None,
                    "fingerprint": self._model_fingerprint(repo_id, revisions),
                    "size": model.get("size", 0),
                    "classNames": model.get("class_names", []),
                    "cacheDirs": model.get("cache_dirs")
                    or ([model.get("cache_dir")] if model.get("cache_dir") else []),
                    "revisions": revisions,
                }
            )

        for repo_id in requested_repos:
            if repo_id.lower() in found_repo_ids:
                continue
            models.append(
                {
                    "repoId": repo_id,
                    "modelTypes": repo_to_model_types.get(repo_id, [model_type] if model_type else []),
                    "installed": False,
                    "selectedRevision": None,
                    "modelRevision": None,
                    "fingerprint": None,
                    "size": 0,
                    "classNames": [],
                    "cacheDirs": [],
                    "revisions": [],
                }
            )

        runtime_fingerprint = self._runtime_fingerprint()
        return web.json_response(
            {
                "error": False,
                "count": len(models),
                "models": models,
                "runtimeFingerprint": runtime_fingerprint.get("fingerprint"),
                "source": "modiff-backend",
            }
        )

    async def model_cache_diagnostics(self, request):
        refresh = str(request.query.get("refresh", "")).lower() in ("1", "true", "yes")
        if refresh:
            modelstore.actualize()

        return web.json_response(get_cache_diagnostics())

    def _custom_modules_root(self):
        root = Path("custom").resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _disabled_custom_modules_root(self):
        root = (self._custom_modules_root() / ".disabled").resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _safe_custom_module_name(self, value):
        name = str(value or "").strip()
        if not name:
            raise ValueError("Module name is required.")
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$", name):
            raise ValueError(
                "Module name may only contain letters, numbers, dot, underscore, and dash, and must not start with a dot."
            )
        return name

    def _derive_custom_module_name(self, source):
        source_text = str(source or "").strip().rstrip("/\\")
        if not source_text:
            raise ValueError("Module source is required.")
        source_text = source_text[:-4] if source_text.endswith(".git") else source_text
        name = re.split(r"[/\\:]", source_text)[-1]
        name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-")
        return self._safe_custom_module_name(name)

    def _custom_module_path(self, name, disabled=False):
        safe_name = self._safe_custom_module_name(name)
        root = self._disabled_custom_modules_root() if disabled else self._custom_modules_root()
        target = (root / safe_name).resolve()
        if target.parent != root:
            raise ValueError("Resolved custom module path escaped the custom module directory.")
        return target

    def _is_git_source(self, source):
        source = str(source or "").strip().lower()
        return source.startswith(("https://", "http://", "ssh://", "git@")) or source.endswith(".git")

    def _run_git(self, args, cwd=None, timeout=300):
        git_bin = shutil.which("git")
        if not git_bin:
            raise RuntimeError("git is not available in the MoDiff backend environment.")

        completed = subprocess.run(
            [git_bin, *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        result = {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
        if completed.returncode != 0:
            message = result["stderr"] or result["stdout"] or f"git exited with {completed.returncode}"
            raise RuntimeError(message)
        return result

    def _git_value(self, module_path, args):
        try:
            return self._run_git(args, cwd=module_path, timeout=10).get("stdout", "")
        except Exception:
            return ""

    def _custom_module_git_info(self, module_path):
        is_git = bool(self._git_value(module_path, ["rev-parse", "--is-inside-work-tree"]))
        if not is_git:
            return {
                "hasGit": False,
                "canUpdate": False,
            }
        return {
            "hasGit": True,
            "canUpdate": True,
            "remote": self._git_value(module_path, ["config", "--get", "remote.origin.url"]),
            "branch": self._git_value(module_path, ["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": self._git_value(module_path, ["rev-parse", "--short", "HEAD"]),
        }

    def _custom_module_info(self, name, module_path, enabled=True):
        module_key = f"custom.{name}"
        node_actions = sorted((self.modules.get(module_key) or {}).keys()) if enabled else []
        git_info = self._custom_module_git_info(module_path)
        return {
            "name": name,
            "moduleKey": module_key,
            "source": "custom",
            "enabled": enabled,
            "status": "enabled" if enabled else "disabled",
            "path": str(module_path),
            "hasInit": (module_path / "__init__.py").exists(),
            "hasMain": (module_path / "main.py").exists(),
            "nodeCount": len(node_actions),
            "nodes": node_actions,
            "canDisable": enabled,
            "canEnable": not enabled,
            **git_info,
        }

    def _list_custom_modules(self):
        root = self._custom_modules_root()
        disabled_root = self._disabled_custom_modules_root()
        modules = []

        for entry in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if not entry.is_dir() or entry.name.startswith(".") or entry.name == "__pycache__":
                continue
            modules.append(self._custom_module_info(entry.name, entry, enabled=True))

        for entry in sorted(disabled_root.iterdir(), key=lambda item: item.name.lower()):
            if not entry.is_dir() or entry.name.startswith(".") or entry.name == "__pycache__":
                continue
            modules.append(self._custom_module_info(entry.name, entry, enabled=False))

        return modules

    def _refresh_custom_module_registry(self):
        for key in list(MODULE_MAP.keys()):
            if key.startswith("custom."):
                MODULE_MAP.pop(key, None)

        for key in list(sys.modules.keys()):
            if key == "custom" or key.startswith("custom."):
                sys.modules.pop(key, None)

        invalidate_caches()
        custom_root = self._custom_modules_root()
        if custom_root.exists():
            parse_module_map("custom")

        self.modules = MODULE_MAP
        self.instance = nanoid.generate(size=10)
        return self._list_custom_modules()

    def _prune_custom_node_cache(self, module_key=None):
        removed = []
        for node_id, cached_node in list(self.node_cache.items()):
            cached_module = getattr(cached_node, "module_name", "")
            if module_key is None:
                should_remove = str(cached_module).startswith("custom.")
            else:
                should_remove = cached_module == module_key
            if should_remove:
                self.node_cache.pop(node_id, None)
                removed.append(node_id)
        return removed

    def _custom_modules_payload(self):
        modules = self._list_custom_modules()
        return {
            "error": False,
            "root": str(self._custom_modules_root()),
            "disabledRoot": str(self._disabled_custom_modules_root()),
            "count": len(modules),
            "modules": modules,
        }

    async def custom_modules_list(self, request):
        return web.json_response(self._custom_modules_payload())

    async def custom_modules_refresh(self, request):
        self._prune_custom_node_cache()
        modules = self._refresh_custom_module_registry()
        return web.json_response(
            {
                "error": False,
                "message": "Custom module registry refreshed.",
                "instance": self.instance,
                "count": len(modules),
                "modules": modules,
            }
        )

    async def custom_modules_install(self, request):
        try:
            data = await request.json()
            source = str(data.get("source") or data.get("url") or "").strip()
            name = self._safe_custom_module_name(data.get("name") or self._derive_custom_module_name(source))
            target = self._custom_module_path(name)
            disabled_target = self._custom_module_path(name, disabled=True)

            if target.exists() or disabled_target.exists():
                return web.json_response(
                    {"error": True, "message": f"Custom module `{name}` already exists."}, status=409
                )

            if self._is_git_source(source):
                self._run_git(["clone", source, str(target)], timeout=900)
            else:
                source_path = Path(source).expanduser().resolve()
                if not source_path.is_dir():
                    return web.json_response(
                        {"error": True, "message": "Source must be a Git URL or an existing local directory."},
                        status=400,
                    )
                shutil.copytree(
                    source_path, target, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".mypy_cache")
                )

            modules = self._refresh_custom_module_registry()
            return web.json_response(
                {
                    "error": False,
                    "message": f"Custom module `{name}` installed.",
                    "module": next((item for item in modules if item["name"] == name), None),
                    "modules": modules,
                    "instance": self.instance,
                }
            )
        except Exception as e:
            logger.error(f"Error installing custom module: {e}", exc_info=True)
            return web.json_response({"error": True, "message": str(e)}, status=500)

    async def custom_modules_update(self, request):
        try:
            name = self._safe_custom_module_name(request.match_info.get("name"))
            enabled_path = self._custom_module_path(name)
            disabled_path = self._custom_module_path(name, disabled=True)
            module_path = enabled_path if enabled_path.exists() else disabled_path
            if not module_path.exists():
                return web.json_response(
                    {"error": True, "message": f"Custom module `{name}` was not found."}, status=404
                )

            if not (module_path / ".git").exists():
                return web.json_response(
                    {"error": True, "message": f"Custom module `{name}` is not a Git checkout."}, status=400
                )

            git_result = self._run_git(["pull", "--ff-only"], cwd=module_path, timeout=900)
            removed_cache_nodes = self._prune_custom_node_cache(f"custom.{name}")
            modules = self._refresh_custom_module_registry() if enabled_path.exists() else self._list_custom_modules()
            return web.json_response(
                {
                    "error": False,
                    "message": f"Custom module `{name}` updated.",
                    "git": git_result,
                    "removedCacheNodes": removed_cache_nodes,
                    "module": next((item for item in modules if item["name"] == name), None),
                    "modules": modules,
                    "instance": self.instance,
                }
            )
        except Exception as e:
            logger.error(f"Error updating custom module: {e}", exc_info=True)
            return web.json_response({"error": True, "message": str(e)}, status=500)

    async def custom_modules_disable(self, request):
        try:
            name = self._safe_custom_module_name(request.match_info.get("name"))
            source = self._custom_module_path(name)
            target = self._custom_module_path(name, disabled=True)
            if not source.exists():
                return web.json_response(
                    {"error": True, "message": f"Custom module `{name}` is not enabled."}, status=404
                )
            if target.exists():
                return web.json_response(
                    {"error": True, "message": f"Disabled custom module `{name}` already exists."}, status=409
                )

            shutil.move(str(source), str(target))
            removed_cache_nodes = self._prune_custom_node_cache(f"custom.{name}")
            modules = self._refresh_custom_module_registry()
            return web.json_response(
                {
                    "error": False,
                    "message": f"Custom module `{name}` disabled.",
                    "removedCacheNodes": removed_cache_nodes,
                    "module": next((item for item in modules if item["name"] == name), None),
                    "modules": modules,
                    "instance": self.instance,
                }
            )
        except Exception as e:
            logger.error(f"Error disabling custom module: {e}", exc_info=True)
            return web.json_response({"error": True, "message": str(e)}, status=500)

    async def custom_modules_enable(self, request):
        try:
            name = self._safe_custom_module_name(request.match_info.get("name"))
            source = self._custom_module_path(name, disabled=True)
            target = self._custom_module_path(name)
            if not source.exists():
                return web.json_response(
                    {"error": True, "message": f"Custom module `{name}` is not disabled."}, status=404
                )
            if target.exists():
                return web.json_response(
                    {"error": True, "message": f"Enabled custom module `{name}` already exists."}, status=409
                )

            shutil.move(str(source), str(target))
            modules = self._refresh_custom_module_registry()
            return web.json_response(
                {
                    "error": False,
                    "message": f"Custom module `{name}` enabled.",
                    "module": next((item for item in modules if item["name"] == name), None),
                    "modules": modules,
                    "instance": self.instance,
                }
            )
        except Exception as e:
            logger.error(f"Error enabling custom module: {e}", exc_info=True)
            return web.json_response({"error": True, "message": str(e)}, status=500)

    @staticmethod
    def _template_gallery_error_response(error, *, plan=None):
        status = 409
        if isinstance(error, TemplateGalleryError):
            code = error.code
            if code in {"template_gallery_contract_missing"}:
                status = 404
            elif code in {"template_gallery_download_unavailable"}:
                status = 503
        else:
            code = "template_gallery_install_failed"
            status = 500
        return web.json_response(
            {
                "error": str(error) if isinstance(error, TemplateGalleryError) else "Template Gallery operation failed.",
                "code": code,
                "retryable": status >= 500,
                **({"plan": plan} if isinstance(plan, dict) else {}),
            },
            status=status,
        )

    def _template_gallery_queue_reservation(self):
        return sum(int(task.get("reserved_bytes") or 0) for task in self.hf_download_tasks.values())

    async def template_gallery_status(self, request):
        source = None
        try:
            source = load_template_gallery_source(TEMPLATE_GALLERY_SOURCE_PATH)
            installing = self.template_gallery_install_task is not None and not self.template_gallery_install_task.done()
            if not template_gallery_destination_present(TEMPLATE_GALLERY_ROOT):
                return web.json_response(
                    {
                        "error": False,
                        "schemaVersion": 1,
                        "status": "installing" if installing else "missing",
                        "installed": False,
                        "complete": False,
                        "repairRequired": False,
                        "installing": installing,
                        "repoId": source["repoId"],
                        "revision": source["revision"],
                        "assetSetId": source["assetSetId"],
                    }
                )
            source, manifest = await asyncio.to_thread(fetch_template_gallery_contract, TEMPLATE_GALLERY_SOURCE_PATH)
            verification = await asyncio.to_thread(verify_template_gallery_tree, TEMPLATE_GALLERY_ROOT, manifest)
            return web.json_response(
                {
                    "error": False,
                    "schemaVersion": 1,
                    "status": "installing" if installing else "ready",
                    "installing": installing,
                    "repoId": source["repoId"],
                    "revision": source["revision"],
                    **verification,
                }
            )
        except TemplateGalleryError as error:
            return web.json_response(
                {
                    "error": False,
                    "schemaVersion": 1,
                    "status": (
                        "repair_required"
                        if template_gallery_destination_present(TEMPLATE_GALLERY_ROOT)
                        else "unavailable"
                    ),
                    "installed": False,
                    "complete": False,
                    "repairRequired": template_gallery_destination_present(TEMPLATE_GALLERY_ROOT),
                    "installing": False,
                    "code": error.code,
                    "message": str(error),
                    **(
                        {
                            "repoId": source["repoId"],
                            "revision": source["revision"],
                            "assetSetId": source["assetSetId"],
                        }
                        if isinstance(source, dict)
                        else {}
                    ),
                }
            )
        except Exception as error:
            logger.error("Could not inspect the Template Gallery install", exc_info=True)
            return self._template_gallery_error_response(error)

    async def _template_gallery_plan(self):
        async with self.download_reservation_lock:
            queued_reservation = self._template_gallery_queue_reservation()
            _source, _manifest, plan = await asyncio.to_thread(
                partial(
                    plan_template_gallery_install,
                    TEMPLATE_GALLERY_SOURCE_PATH,
                    TEMPLATE_GALLERY_ROOT,
                    queued_reservation_bytes=queued_reservation,
                )
            )
        plan["installing"] = self.template_gallery_install_task is not None and not self.template_gallery_install_task.done()
        return plan

    async def template_gallery_plan(self, request):
        try:
            return web.json_response({"error": False, **(await self._template_gallery_plan())})
        except TemplateGalleryError as error:
            return self._template_gallery_error_response(error)
        except Exception as error:
            logger.error("Could not plan the Template Gallery install", exc_info=True)
            return self._template_gallery_error_response(error)

    async def _run_template_gallery_install(self):
        source = manifest = plan = None
        try:
            async with self.download_reservation_lock:
                queued_reservation = self._template_gallery_queue_reservation()
                source, manifest, plan = await asyncio.to_thread(
                    partial(
                        plan_template_gallery_install,
                        TEMPLATE_GALLERY_SOURCE_PATH,
                        TEMPLATE_GALLERY_ROOT,
                        queued_reservation_bytes=queued_reservation,
                    )
                )
                plan["installing"] = True
                if not plan["sizeKnown"] or not plan["fitsWithQueue"]:
                    return {
                        "complete": False,
                        "httpStatus": 507,
                        "code": "insufficient_template_gallery_space",
                        "error": (
                            "The app refused the Template Gallery install because its exact download and staging "
                            "reservation, active model reservations, and the 64 GiB safety reserve do not fit."
                        ),
                        "plan": plan,
                    }
                if plan["installed"]:
                    return {"complete": True, "alreadyInstalled": True, "plan": plan}
                self.template_gallery_reserved_bytes = int(plan["reservationBytes"])
            async with self.hf_download_semaphore:
                result = await asyncio.to_thread(
                    partial(
                        install_template_gallery,
                        source,
                        manifest,
                        TEMPLATE_GALLERY_ROOT,
                        receipt_path=Path(self.data_dir) / "template-gallery-install.v1.json",
                    )
                )
            return {"complete": True, "alreadyInstalled": False, "plan": plan, "result": result}
        except TemplateGalleryError as error:
            return {
                "complete": False,
                "httpStatus": 409,
                "code": error.code,
                "error": str(error),
                **({"plan": plan} if isinstance(plan, dict) else {}),
            }
        except Exception:
            logger.error("Template Gallery installation failed", exc_info=True)
            return {
                "complete": False,
                "httpStatus": 500,
                "code": "template_gallery_install_failed",
                "error": "Template Gallery installation failed.",
                **({"plan": plan} if isinstance(plan, dict) else {}),
            }
        finally:
            self.template_gallery_reserved_bytes = 0

    async def template_gallery_install(self, request):
        if getattr(request, "can_read_body", False):
            try:
                payload = await request.json()
            except (TypeError, ValueError):
                return web.json_response({"error": "Invalid JSON body."}, status=400)
            if not isinstance(payload, dict) or payload:
                return web.json_response(
                    {"error": "The Template Gallery install request must be an empty JSON object."}, status=400
                )

        if self.template_gallery_install_task is None or self.template_gallery_install_task.done():
            self.template_gallery_install_task = asyncio.create_task(self._run_template_gallery_install())
        task = self.template_gallery_install_task
        try:
            result = await asyncio.shield(task)
            if not isinstance(result, dict) or result.get("complete") is not True:
                status = int(result.get("httpStatus") or 409) if isinstance(result, dict) else 500
                return web.json_response(
                    {
                        "error": result.get("error") if isinstance(result, dict) else "Template Gallery install failed.",
                        "code": result.get("code") if isinstance(result, dict) else "template_gallery_install_failed",
                        "retryable": status >= 500,
                        **({"plan": result.get("plan")} if isinstance(result, dict) and result.get("plan") else {}),
                    },
                    status=status,
                )
            return web.json_response({"error": False, **result})
        finally:
            if self.template_gallery_install_task is task and task.done():
                self.template_gallery_install_task = None

    async def hf_cache_delete(self, request):
        hashes = request.match_info.get("hash").split(",")
        if not hashes:
            return web.json_response({"error": "Incorrect request, `hash` is required."}, status=400)

        result = delete_model(*hashes)
        return web.json_response({"error": not result})

    async def hf_hub(self, request):
        query = request.query.get("q", "")
        sid = request.query.get("sid")

        future = asyncio.Future()
        try:
            await self.queue_task(search_hub, query, future, sid, name="Hugging Face search")
        except OverlayInstallBusy as exc:
            return web.json_response({"error": True, "message": str(exc)}, status=409)

        try:
            result = await future
            return web.json_response(result)
        except Exception as e:
            logger.error(f"Error in hf_hub endpoint: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def hf_download_plan(self, request):
        query = getattr(request, "query", {}) or {}
        repo_id = query.get("repo_id")
        if not repo_id:
            return web.json_response({"error": "Incorrect request, `repo_id` is required."}, status=400)
        try:
            repo_id = validate_hf_repo_id(repo_id)
        except (TypeError, ValueError) as error:
            return web.json_response(
                {"error": str(error), "code": "invalid_huggingface_repo_id", "retryable": False},
                status=400,
            )

        raw_revision = query.get("revision")
        if raw_revision is not None and (
            not isinstance(raw_revision, str)
            or raw_revision != raw_revision.strip()
            or raw_revision != raw_revision.lower()
            or not IMMUTABLE_HUB_REVISION.fullmatch(raw_revision)
        ):
            return web.json_response(
                {
                    "error": "Model download plans require an exact lowercase 40-character commit revision.",
                    "code": "invalid_huggingface_revision",
                    "retryable": False,
                },
                status=400,
            )
        revision = raw_revision or catalog_revision(repo_id)
        requested_files = []
        if hasattr(query, "getall"):
            requested_files = query.getall("file", [])
        elif query.get("file"):
            requested_files = [query.get("file")]
        requested_files = sorted(
            {item.strip() for raw in requested_files for item in str(raw or "").split(",") if item.strip()}
        )
        if not requested_files:
            requested_files = studio_download_files_for_repo(repo_id)

        try:
            plan = await asyncio.to_thread(
                plan_hub_model_download,
                repo_id,
                requested_files,
                revision,
            )
        except Exception as error:
            return web.json_response(
                {
                    "error": str(error) or type(error).__name__,
                    "code": "huggingface_download_plan_failed",
                    "retryable": True,
                },
                status=503,
            )
        queued_reservation = self.template_gallery_reserved_bytes + sum(
            int(task.get("reserved_bytes") or 0) for task in self.hf_download_tasks.values()
        )
        remaining_bytes = plan.get("remainingBytes")
        fits_with_queue = bool(
            plan.get("sizeKnown")
            and isinstance(remaining_bytes, int)
            and remaining_bytes + queued_reservation + int(plan.get("reserveBytes") or 0)
            <= int(plan.get("freeBytes") or 0)
        )
        return web.json_response(
            {
                "error": False,
                **plan,
                "queuedReservationBytes": queued_reservation,
                "fitsWithQueue": fits_with_queue,
            }
        )

    @staticmethod
    def _update_hf_download_progress(download_repo_id, entry, **updates):
        previous = entry.get("progress")
        if not isinstance(previous, dict):
            previous = {}
        snapshot = {
            **previous,
            "type": "hf_download_progress",
            "repo_id": download_repo_id,
            "task_id": entry["task_id"],
            "download_id": entry["task_id"],
            "started_at": entry.get("started_at"),
            "updated_at": time.time(),
            **updates,
        }
        entry["progress"] = snapshot
        return snapshot

    def _hf_download_status_snapshots(self):
        snapshots = []
        for repo_id, entry in self.hf_download_tasks.items():
            progress = entry.get("progress")
            if not isinstance(progress, dict):
                progress = {
                    "type": "hf_download_progress",
                    "repo_id": repo_id,
                    "task_id": entry.get("task_id"),
                    "download_id": entry.get("task_id"),
                    "status": "queued",
                    "phase": "queued",
                    "progress": 0,
                    "started_at": entry.get("started_at"),
                    "updated_at": entry.get("started_at"),
                }
            snapshot = dict(progress)
            snapshot["revision"] = entry.get("revision")
            snapshot["repair"] = bool(entry.get("repair"))
            snapshot["requested_file_count"] = len(entry.get("requested_files") or [])
            reserved_bytes = entry.get("reserved_bytes")
            if isinstance(reserved_bytes, int) and reserved_bytes >= 0:
                snapshot["reserved_bytes"] = reserved_bytes
                if snapshot.get("remaining_bytes") is None:
                    snapshot["remaining_bytes"] = reserved_bytes
            plan = entry.get("download_plan")
            if isinstance(plan, dict):
                plan_fields = {
                    "total_bytes": "totalBytes",
                    "completed_bytes": "completedBytes",
                    "total_file_count": "totalFileCount",
                    "size_known": "sizeKnown",
                    "cache_dir": "cacheRoot",
                }
                for target, source in plan_fields.items():
                    if snapshot.get(target) is None and plan.get(source) is not None:
                        snapshot[target] = plan[source]
            snapshots.append(snapshot)
        return sorted(
            snapshots,
            key=lambda snapshot: (
                float(snapshot.get("started_at") or 0),
                str(snapshot.get("repo_id") or ""),
            ),
        )

    async def hf_download_status(self, request):
        downloads = self._hf_download_status_snapshots()
        return web.json_response(
            {
                "error": False,
                "schemaVersion": 1,
                "downloads": downloads,
                "activeCount": len(downloads),
                "queuedReservationBytes": sum(
                    int(task.get("reserved_bytes") or 0) for task in self.hf_download_tasks.values()
                ),
                "templateGalleryReservationBytes": self.template_gallery_reserved_bytes,
            }
        )

    async def _reserve_hf_download_space(self, repo_id, entry):
        async with self.download_reservation_lock:
            try:
                plan = await asyncio.to_thread(
                    plan_hub_model_download,
                    repo_id,
                    entry.get("requested_files"),
                    entry.get("revision"),
                )
            except Exception as error:
                return {
                    "complete": False,
                    "errorCode": "huggingface_download_plan_failed",
                    "httpStatus": 503,
                    "validation": {"reason": str(error) or type(error).__name__},
                }

            remaining_bytes = plan.get("remainingBytes")
            if not plan.get("sizeKnown") or not isinstance(remaining_bytes, int):
                return {
                    "complete": False,
                    "errorCode": "huggingface_download_size_unknown",
                    "httpStatus": 503,
                    "downloadPlan": plan,
                    "validation": {
                        "reason": "The app could not prove the immutable model download size before writing files."
                    },
                }
            queued_reservation = self.template_gallery_reserved_bytes + sum(
                int(task.get("reserved_bytes") or 0)
                for task in self.hf_download_tasks.values()
                if task is not entry
            )
            required_with_reserve = remaining_bytes + queued_reservation + int(plan.get("reserveBytes") or 0)
            if required_with_reserve > int(plan.get("freeBytes") or 0):
                return {
                    "complete": False,
                    "errorCode": "insufficient_model_download_space",
                    "httpStatus": 507,
                    "downloadPlan": {**plan, "queuedReservationBytes": queued_reservation},
                    "validation": {
                        "reason": (
                            "The app refused the model download because its conservative remaining-size reservation, "
                            "the active download queue, and the 64 GiB safety reserve do not fit on the cache volume."
                        )
                    },
                }
            entry["reserved_bytes"] = remaining_bytes
            entry["download_plan"] = plan
            return None

    async def _run_hf_download_task(self, repo_id, entry):
        task_id = entry["task_id"]

        def progress_cb(progress):
            message = {
                "type": "hf_download_progress",
                "repo_id": repo_id,
                "task_id": task_id,
                "download_id": task_id,
            }
            if isinstance(progress, dict):
                message.update(progress)
                progress_value = message.get("progress")
                remaining_bytes = message.get("remaining_bytes")
                if isinstance(remaining_bytes, int) and remaining_bytes >= 0:
                    entry["reserved_bytes"] = remaining_bytes
            else:
                progress_value = float(progress or 0)
                message["progress"] = progress_value

            if "status" not in message:
                message["status"] = "downloading" if progress_value is None or progress_value < 1 else "complete"

            message = self._update_hf_download_progress(repo_id, entry, **message)

            for download_sid in list(entry.get("sids", [])):
                self.queue_message(message, download_sid)

        queued_message = self._update_hf_download_progress(
            repo_id,
            entry,
            progress=0,
            status="queued",
            phase="queued",
        )
        for download_sid in list(entry.get("sids", [])):
            self.queue_message(queued_message, download_sid)

        reservation_failure = await self._reserve_hf_download_space(repo_id, entry)
        if reservation_failure is not None:
            return reservation_failure

        try:
            return await self._run_reserved_hf_download(repo_id, entry, progress_cb)
        finally:
            entry["reserved_bytes"] = 0

    async def _run_reserved_hf_download(self, repo_id, entry, progress_cb):
        async with self.hf_download_semaphore:
            planning_message = self._update_hf_download_progress(
                repo_id,
                entry,
                progress=0,
                status="planning",
                phase="planning",
            )
            for download_sid in list(entry.get("sids", [])):
                self.queue_message(planning_message, download_sid)

            if self.serialize_model_io and self.model_io_lock.locked():
                waiting_message = self._update_hf_download_progress(
                    repo_id,
                    entry,
                    progress=0,
                    status="queued",
                    phase="waiting_for_model_io",
                    message="Waiting for active generation or model I/O to finish safely.",
                )
                for download_sid in list(entry.get("sids", [])):
                    self.queue_message(waiting_message, download_sid)
            result = await self._run_executor_callback(
                partial(
                    download_hub_model,
                    repo_id,
                    progress_cb,
                    bool(entry.get("repair")),
                    entry.get("repair_source_repo_id"),
                    entry.get("requested_files"),
                    entry.get("revision"),
                ),
                serialize_model_io=True,
            )
            if result:
                modelstore.actualize()
            return result

    async def hf_download(self, request):
        payload = {}
        if getattr(request, "can_read_body", False):
            try:
                payload = await request.json()
            except (ValueError, TypeError):
                return web.json_response({"error": "Invalid JSON body."}, status=400)
            if not isinstance(payload, dict):
                return web.json_response({"error": "The download request must be a JSON object."}, status=400)

        query = getattr(request, "query", {}) or {}
        repo_id = payload.get("repo_id") or query.get("repo_id")
        sid = payload.get("sid") or query.get("sid")
        raw_revision = payload.get("revision") if "revision" in payload else query.get("revision")
        revision = None
        if raw_revision is not None:
            if (
                not isinstance(raw_revision, str)
                or raw_revision != raw_revision.strip()
                or raw_revision != raw_revision.lower()
                or not IMMUTABLE_HUB_REVISION.fullmatch(raw_revision)
            ):
                return web.json_response(
                    {
                        "error": "Model downloads require an exact lowercase 40-character commit revision.",
                        "code": "invalid_huggingface_revision",
                        "retryable": False,
                    },
                    status=400,
                )
            revision = raw_revision
        repair_value = payload.get("repair") if "repair" in payload else query.get("repair")
        repair = repair_value is True or str(repair_value or "").lower() in {"1", "true", "yes"}
        repair_source_repo_id = (
            str(payload.get("repair_source_repo_id") or query.get("repair_source_repo_id") or "").strip() or None
        )
        raw_files = payload.get("files", payload.get("file", []))
        if isinstance(raw_files, str):
            raw_files = [raw_files]
        elif not isinstance(raw_files, list):
            raw_files = []
        if not raw_files and hasattr(query, "getall"):
            raw_files = query.getall("file", [])
        if not raw_files and query.get("file"):
            raw_files = [query.get("file")]
        requested_files = sorted(
            {item.strip() for raw in raw_files for item in str(raw or "").split(",") if item.strip()}
        )
        if not requested_files and repo_id:
            requested_files = studio_download_files_for_repo(repo_id)
        if repair and not repair_source_repo_id:
            repair_source_repo_id = VERIFIED_REPAIR_SOURCES.get(repo_id)

        if not repo_id:
            return web.json_response({"error": "Incorrect request, `repo_id` is required."}, status=400)
        try:
            repo_id = validate_hf_repo_id(repo_id)
            if repair_source_repo_id is not None:
                repair_source_repo_id = validate_hf_repo_id(repair_source_repo_id)
        except (TypeError, ValueError) as error:
            return web.json_response(
                {"error": str(error), "code": "invalid_huggingface_repo_id", "retryable": False},
                status=400,
            )

        if revision is None:
            revision = catalog_revision(repo_id)

        if repo_id in self.hf_download_tasks:
            entry = self.hf_download_tasks[repo_id]
            if (
                sorted(entry.get("requested_files") or []) != requested_files
                or entry.get("revision") != revision
            ):
                return web.json_response(
                    {
                        "error": "A different immutable snapshot or file selection is already downloading for this repository.",
                        "repo_id": repo_id,
                        "retryable": True,
                    },
                    status=409,
                )
            if sid:
                entry["sids"].add(sid)
                self.queue_message(
                    {
                        "type": "hf_download_progress",
                        "repo_id": repo_id,
                        "task_id": entry["task_id"],
                        "download_id": entry["task_id"],
                        "status": "joined",
                        "phase": "queued",
                        "progress": None,
                        "started_at": entry.get("started_at"),
                        "updated_at": time.time(),
                    },
                    sid,
                )
        else:
            task_id = nanoid.generate(size=12)
            entry = {
                "task_id": task_id,
                "sids": set([sid] if sid else []),
                "started_at": time.time(),
                "repair": repair,
                "repair_source_repo_id": repair_source_repo_id,
                "requested_files": requested_files,
                "revision": revision,
            }
            entry["future"] = self.loop.create_task(self._run_hf_download_task(repo_id, entry))
            self.hf_download_tasks[repo_id] = entry

        try:
            result = await asyncio.shield(entry["future"])
            complete = isinstance(result, dict) and result.get("complete") is True
            if not complete:
                reason = (
                    result.get("validation", {}).get("reason")
                    if isinstance(result, dict) and isinstance(result.get("validation"), dict)
                    else None
                ) or "The downloaded snapshot is incomplete and requires repair."
                response_status = (
                    int(result.get("httpStatus") or 409) if isinstance(result, dict) else 409
                )
                return web.json_response(
                    {
                        "error": reason,
                        "code": result.get("errorCode") if isinstance(result, dict) else None,
                        "result": result,
                        "task_id": entry["task_id"],
                        "repo_id": repo_id,
                        "repair_required": response_status == 409,
                    },
                    status=response_status,
                )
            return web.json_response(
                {"error": False, "result": result, "task_id": entry["task_id"], "repo_id": repo_id}
            )
        except Exception as e:
            logger.error(f"Error in hf_download endpoint: {e}")
            status, code, message, retryable = classify_hf_download_error(e)
            return web.json_response(
                {
                    "error": message,
                    "code": code,
                    "repo_id": repo_id,
                    "task_id": entry["task_id"],
                    "retryable": retryable,
                    "preserved_partial_download": True,
                },
                status=status,
            )
        finally:
            if (
                repo_id in self.hf_download_tasks
                and self.hf_download_tasks[repo_id].get("future") is entry.get("future")
                and entry["future"].done()
            ):
                self.hf_download_tasks.pop(repo_id, None)

    async def hf_token(self, request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body."}, status=400)
        token = str(payload.get("token") or "").strip() if isinstance(payload, dict) else ""
        if not token:
            return web.json_response({"error": "A Hugging Face access token is required."}, status=400)

        try:
            from huggingface_hub import HfApi

            identity = await self.loop.run_in_executor(None, lambda: HfApi(token=token).whoami())
        except Exception as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            message = "Hugging Face rejected this token. Create a read token and try again."
            if status:
                message = f"{message} HTTP {status}."
            return web.json_response({"error": message}, status=401)

        config_path = Path(__file__).resolve().parent.parent / "config.ini"
        parser = configparser.ConfigParser()
        parser.optionxform = str
        parser.read(config_path, encoding="utf-8")
        if not parser.has_section("huggingface"):
            parser.add_section("huggingface")
        parser.set("huggingface", "token", token)
        temp_path = config_path.with_suffix(".ini.tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                parser.write(handle)
            os.replace(temp_path, config_path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
        CONFIG.hf["token"] = token
        return web.json_response(
            {
                "error": False,
                "token_configured": True,
                "account_type": identity.get("type") if isinstance(identity, dict) else None,
            }
        )

    """
    ╭───────────────╮
        Websocket
    ╰───────────────╯
    """

    async def websocket(self, request):
        origin_error = self._untrusted_websocket_origin_response(request)
        if origin_error is not None:
            return origin_error

        ws = web.WebSocketResponse()
        await ws.prepare(request)
        sid = request.query.get("sid")
        if not sid:
            sid = nanoid.generate(size=10)
        if sid in self.ws_sessions:
            # close the connection and remove the old session
            logger.debug(f"Websocket session {sid} already exists, closing the old session.")
            # await self.ws_sessions[sid].close()
            sid = nanoid.generate(size=10)
            # del self.ws_sessions[sid]

        self.ws_sessions[sid] = ws
        logger.debug(f"Websocket connection opened: {sid}")

        # Restore global execution truth as part of the connection handshake so
        # panels never become responsible for discovering active work.
        queued_tasks, current_task = self._get_queue()
        await self.broadcast(
            {
                "type": "welcome",
                "instance": self.instance,
                "sid": sid,
                "cachedNodes": list(self.node_cache.keys()),
                "queued": queued_tasks,
                "current": current_task,
                "downloads": self._hf_download_status_snapshots(),
                # Keep the welcome contract aligned with /queue: workflow graphs
                # remain available lazily through /runs/{task_id}, but are not
                # disclosed or retransmitted in the initial handshake.
                "recent": compact_task_history(self.recent_tasks),
            },
            sid,
        )

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    data = json.loads(msg.data)

                    if data["type"] == "close":
                        await ws.close()
                        break
                    elif data["type"] == "ping":
                        await self.broadcast({"type": "pong"}, sid)
                    elif data["type"] == "signal_value":
                        request_id = data.get("request_id")
                        if not request_id:
                            logger.warning("[Websocket] signal_value received without request_id")
                            continue
                        # Resolve the pending request future if present
                        try:
                            promised_sid = data.get("sid", None)
                            future = self.pending_ws_requests.pop(request_id, None)
                            if future is None:
                                logger.debug(f"[Websocket] signal_value received for unknown request_id: {request_id}")
                                continue
                            if not future.done():
                                result = data.get("value")
                                if promised_sid != sid:
                                    result = {"__MODIFF_ERROR": "sid_mismatch"}
                                future.set_result(result)
                        except Exception as e:
                            logger.error(f"[Websocket] Error resolving signal_value for request_id {request_id}: {e}")

                    else:
                        logger.error(f"[Websocket] Invalid message type: {data['type']}")
                elif msg.type == WSMsgType.CLOSE:
                    await self.broadcast({"type": "close"}, sid)
                    break
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"[Websocket] Error: {ws.exception()}")
        except Exception as e:
            logger.error(f"[Websocket] Error: {e}")
        finally:
            if sid in self.ws_sessions:
                del self.ws_sessions[sid]
            logger.debug(f"Websocket connection closed: {sid}")

        return ws

    async def broadcast(self, message: dict | bytes, sid: list[str] | str = None, exclude: list[str] | str = None):
        sessions = []

        if sid:
            sessions = [sid] if not isinstance(sid, list) else sid
        else:
            sessions = list(self.ws_sessions.keys())

        if exclude:
            exclude = [exclude] if not isinstance(exclude, list) else exclude
            sessions = [s for s in sessions if s not in exclude]

        for session in sessions:
            websocket = self.ws_sessions.get(session)
            if websocket is None:
                continue
            if websocket.closed:
                if self.ws_sessions.get(session) is websocket:
                    self.ws_sessions.pop(session, None)
                continue
            try:
                if isinstance(message, dict):
                    await websocket.send_json(message)
                else:
                    await websocket.send_bytes(message)
            except Exception as e:
                # A browser refresh can close the transport after the `closed`
                # check but before aiohttp begins the write. Prune that stale
                # session immediately; otherwise every subsequent progress
                # event repeats the same noisy failure until the receive loop's
                # finally block gets scheduled.
                if self.ws_sessions.get(session) is websocket:
                    self.ws_sessions.pop(session, None)
                if websocket.closed or "closing transport" in str(e).lower():
                    logger.debug(f"[Websocket] Dropped closing session {session}: {e}")
                else:
                    logger.warning(f"[Websocket] Dropped failed session {session}: {e}")

    def queue_message(self, message: dict | bytes, sid: list[str] | str = None, exclude: list[str] | str = None):
        loop = self.loop
        if loop is not None and loop.is_running() and not self._shutdown_event.is_set():
            asyncio.run_coroutine_threadsafe(
                self.background_queue.put((self.broadcast, (message, sid, exclude))), loop
            )

    def get_signal_value(self, node: str, field: str, sid: str, timeout: int = 2):
        try:
            if not sid or sid not in self.ws_sessions or self.ws_sessions[sid].closed:
                return {"__MODIFF_ERROR": "invalid_sid"}

            if not getattr(self, "loop", None) or not self.loop.is_running():
                return {"__MODIFF_ERROR": "server_not_running"}

            try:
                running_loop = asyncio.get_running_loop()
                if running_loop is self.loop:
                    logger.warning(
                        "[Server] get_signal_value called from event loop thread; returning None to avoid deadlock."
                    )
                    return {"__MODIFF_ERROR": "called_from_event_loop"}
            except RuntimeError:
                # No running loop in this thread; safe to proceed
                pass

            # Create a future bound to the server loop and register it
            request_id = nanoid.generate(size=12)
            future = self.loop.create_future()
            self.pending_ws_requests[request_id] = future

            # Send the request to the target client
            self.queue_message(
                {"type": "get_signal_value", "request_id": request_id, "node": node, "field": field, "sid": sid}, sid
            )

            # Await the future result from outside the event loop thread
            # Use run_coroutine_threadsafe to wait with a timeout safely
            wrapped = asyncio.wait_for(future, timeout=timeout)
            cfut = asyncio.run_coroutine_threadsafe(wrapped, self.loop)
            try:
                result = cfut.result(timeout=timeout + 0.5)
            except Exception:
                result = {"__MODIFF_ERROR": "timeout"}
            finally:
                # Cleanup any leftover pending entry
                self.pending_ws_requests.pop(request_id, None)

            return result
        except Exception as e:
            logger.error(f"[Server] get_signal_value error: {e}")
            return {"__MODIFF_ERROR": "exception"}


def to_base64(type, value, options=None):
    import io
    import base64

    options = options or {}
    out = value

    if type == "image":
        format = options.get("format", "WEBP").upper()
        quality = options.get("quality")
        if format == "WEBP" and not quality:
            quality = 100
        elif format == "JPEG" and not quality:
            quality = 75
        elif format == "PNG" and not quality:
            quality = None
        mime_type = f"image/{format.lower()}"

        byte_arr = io.BytesIO()
        save_kwargs = {"format": format}
        if quality is not None:
            save_kwargs["quality"] = int(quality)
        value.save(byte_arr, **save_kwargs)
        header = f"data:{mime_type};base64,"
        out = header + base64.b64encode(byte_arr.getvalue()).decode("utf-8")

    return out


def to_bytes(data_type, value, options=None):
    import io
    import wave
    from PIL import Image

    options = options or {}
    out = value

    if isinstance(value, Image.Image):
        format = options.get("format", "WEBP").upper()
        quality = options.get("quality")
        if format == "WEBP" and not quality:
            quality = 100
        elif format == "JPEG" and not quality:
            quality = 75
        elif format == "PNG" and not quality:
            quality = None

        byte_arr = io.BytesIO()
        save_kwargs = {"format": format}
        if quality is not None:
            save_kwargs["quality"] = int(quality)
        value.save(byte_arr, **save_kwargs)
        out = byte_arr.getvalue()
    elif data_type == "audio" and isinstance(value, (str, os.PathLike)):
        from modiff.path_identifiers import resolve_runtime_input_path

        path = resolve_runtime_input_path(value)
        if path.is_file():
            out = path.read_bytes()
    elif data_type == "audio" and isinstance(value, dict):
        import numpy as np

        samples = value.get("samples", value.get("audio", value.get("array")))
        if samples is None:
            raise ValueError("Audio output must contain samples, audio, or array data.")
        if hasattr(samples, "detach"):
            samples = samples.detach().float().cpu().numpy()
        array = np.asarray(samples, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        elif array.ndim == 2 and array.shape[0] > array.shape[1]:
            array = array.T
        if array.ndim != 2:
            raise ValueError(f"Audio output must be one or two dimensional; received shape {array.shape}.")
        pcm = (np.clip(array, -1.0, 1.0).T * 32767.0).round().astype("<i2", copy=False)
        byte_arr = io.BytesIO()
        with wave.open(byte_arr, "wb") as wav:
            wav.setnchannels(int(array.shape[0]))
            wav.setsampwidth(2)
            wav.setframerate(int(value.get("sample_rate") or 48000))
            wav.writeframes(pcm.tobytes())
        out = byte_arr.getvalue()
    elif isinstance(value, str):
        out = value.encode("utf-8")

    return out


server = WebServer(
    MODULE_MAP,
    host=CONFIG.server["host"],
    port=CONFIG.server["port"],
    secure=CONFIG.server["secure"],
    certfile=CONFIG.server["certfile"],
    keyfile=CONFIG.server["keyfile"],
    cors=CONFIG.server["cors"],
    cors_routes=CONFIG.server["cors_routes"],
    client_max_size=CONFIG.server["client_max_size"],
    work_dir=CONFIG.paths["work_dir"],
    data_dir=CONFIG.paths["data"],
)
