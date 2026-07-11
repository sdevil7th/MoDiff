from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from modiff.diffusers_offload import (
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_NONE,
    OFFLOAD_MODE_SEQUENTIAL_CPU,
)
from modiff.diffusers_profiles import (
    ACE_STEP_REPO,
    FLUX_CANNY_REPO,
    FLUX_DEPTH_REPO,
    FLUX_DEV_REPO,
    FLUX_FILL_REPO,
    FLUX_KONTEXT_REPO,
    FLUX_KREA_REPO,
    FLUX_REDUX_REPO,
    FLUX_SCHNELL_REPO,
    QWEN_IMAGE_2512_PREQUANTIZED_REPO,
    QWEN_IMAGE_2512_REPO,
)
from modiff.hardware import disk_snapshot, get_hardware_snapshot, system_memory_snapshot


GIB = 1024**3
CAPACITY_CLASS_TOLERANCE = 0.01

QWEN_PRACTICAL_WIDTH = 1024
QWEN_PRACTICAL_HEIGHT = 1024
QWEN_NATIVE_WIDTH = 1328
QWEN_NATIVE_HEIGHT = 1328
QWEN_NATIVE_STEPS = 50
QWEN_NATIVE_TRUE_CFG = 4.0
QWEN_TRANSFORMER_ONLY_COMPONENTS = ["transformer"]

Z_IMAGE_REPO = "Tongyi-MAI/Z-Image-Turbo"
QWEN_IMAGE_EDIT_REPO = "Qwen/Qwen-Image-Edit"
QWEN_IMAGE_EDIT_PREQUANTIZED_REPO = "ovedrive/qwen-image-edit-4bit"
QWEN_IMAGE_EDIT_PLUS_REPO = "Qwen/Qwen-Image-Edit-2511"
QWEN_IMAGE_LAYERED_REPO = "Qwen/Qwen-Image-Layered"
WAN_VACE_REPO = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
FLUX_DEV_FP8_REPO = "black-forest-labs/FLUX.1-dev-FP8"
FLUX_KONTEXT_NVFP4_REPO = "black-forest-labs/FLUX.1-Kontext-dev-NVFP4"

READY_PROOF_STATUSES = {"passed", "declared_safe", "live_proven"}
PROVEN_PROOF_STATUSES = READY_PROOF_STATUSES
FAILED_HERE_PROOF_STATUS = "failed_here_before"
AUTO_HISTORY_VERSION = 1
AUTO_HISTORY_RELATIVE_PATH = Path("auto_resource") / "history.json"

MODELISH_EXTENSIONS = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx"}
CONFIG_JSON_NAMES = {"model_index.json", "config.json", "model_config.json", "scheduler_config.json"}

AUTO_MODEL_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "ZImageModularPipeline": {
        "supportedTasks": ["text_to_image"],
        "defaultRepo": Z_IMAGE_REPO,
        "executionPath": "modular-diffusers",
        "qualityDefaults": {"width": 1024, "height": 1024, "steps": 8, "guidanceScale": 1},
        "minimum": {"accelerator": "cuda_or_mps_or_cpu", "vramBytes": 0, "systemRamBytes": 8 * GIB},
        "recommended": {"accelerator": "cuda_or_mps", "vramBytes": 8 * GIB, "systemRamBytes": 16 * GIB},
        "supportedOffloadModes": [OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_DISK],
    },
    "QwenImageModularPipeline:text_to_image": {
        "supportedTasks": ["text_to_image"],
        "defaultRepo": QWEN_IMAGE_2512_REPO,
        "preferredLowerMemoryRepo": QWEN_IMAGE_2512_PREQUANTIZED_REPO,
        "executionPath": "direct-qwen-image",
        "qualityDefaults": {
            "width": QWEN_PRACTICAL_WIDTH,
            "height": QWEN_PRACTICAL_HEIGHT,
            "steps": QWEN_NATIVE_STEPS,
            "guidanceScale": QWEN_NATIVE_TRUE_CFG,
        },
        "nativeQuality": {
            "width": QWEN_NATIVE_WIDTH,
            "height": QWEN_NATIVE_HEIGHT,
            "steps": QWEN_NATIVE_STEPS,
            "guidanceScale": QWEN_NATIVE_TRUE_CFG,
        },
        "minimum": {"accelerator": "cuda", "vramBytes": 10 * GIB, "systemRamBytes": 24 * GIB},
        "recommended": {"accelerator": "cuda", "vramBytes": 16 * GIB, "systemRamBytes": 32 * GIB},
        "officialBf16": {"accelerator": "cuda", "vramBytes": 32 * GIB, "systemRamBytes": 30 * GIB},
        "supportedOffloadModes": [OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK],
        "knownBad": [
            "Do not Auto-quantize the Qwen text encoder with BitsAndBytes on unknown kernels; prior runs hit CUBLAS_STATUS_NOT_SUPPORTED.",
            "Do not start Auto from official BF16 native settings on constrained CUDA memory.",
        ],
    },
    "QwenImageModularPipeline:control_image": {
        "supportedTasks": ["control_image"],
        "defaultRepo": QWEN_IMAGE_2512_REPO,
        "executionPath": "modular-diffusers",
        "pipelineClass": "QwenImageModularPipeline",
        "qualityDefaults": {"width": 768, "height": 768, "steps": 28, "guidanceScale": 4, "maxSequenceLength": 512},
        "minimum": {"accelerator": "cuda", "vramBytes": 16 * GIB, "systemRamBytes": 32 * GIB, "diskFreeBytes": 20 * GIB},
        "recommended": {"accelerator": "cuda", "vramBytes": 24 * GIB, "systemRamBytes": 48 * GIB, "diskFreeBytes": 35 * GIB},
        "onLoadQuantization": {
            "accelerator": "cuda",
            "vramBytes": 16 * GIB,
            "systemRamBytes": 32 * GIB,
            "diskFreeBytes": 20 * GIB,
            "quantizationMode": "bnb_4bit",
            "quantizedComponents": ["transformer", "text_encoder"],
        },
        "supportedOffloadModes": [OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK],
        "requiredPackages": ["diffusers", "transformers", "accelerate", "torch", "bitsandbytes"],
        "guardedReason": "Qwen Control image is broad Auto coverage: MoDiff will try the generic modular Diffusers graph with quantization/offload, then remember any failure on this machine.",
    },
    "QwenImageEditModularPipeline": {
        "supportedTasks": ["edit_image", "inpaint", "outpaint"],
        "defaultRepo": QWEN_IMAGE_EDIT_REPO,
        "preferredLowerMemoryRepo": QWEN_IMAGE_EDIT_PREQUANTIZED_REPO,
        "executionPath": "direct-qwen-image-edit",
        "qualityDefaults": {"width": 1024, "height": 1024, "steps": 40, "guidanceScale": 4},
        "minimum": {"accelerator": "cuda", "vramBytes": 24 * GIB, "systemRamBytes": 40 * GIB},
        "recommended": {"accelerator": "cuda", "vramBytes": 32 * GIB, "systemRamBytes": 48 * GIB},
        "lowerMemory": {
            "accelerator": "cuda",
            "vramBytes": 16 * GIB,
            "systemRamBytes": 24 * GIB,
            "diskFreeBytes": 25 * GIB,
            "quantizationMode": "none",
            "quantizedComponents": [],
        },
        "onLoadQuantization": {
            "accelerator": "cuda",
            "vramBytes": 16 * GIB,
            "systemRamBytes": 32 * GIB,
            "diskFreeBytes": 30 * GIB,
            "quantizationMode": "bnb_4bit",
            "quantizedComponents": ["transformer", "text_encoder"],
        },
        "supportedOffloadModes": [OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK],
        "requiredPackages": ["diffusers", "transformers", "accelerate", "torch", "bitsandbytes"],
        "guardedReason": "Prefer the Apache-2.0 Diffusers-compatible prequantized Qwen Image Edit artifact on nominal 16 GiB CUDA systems before attempting official BF16 disk offload.",
    },
    "QwenImageEditPlusModularPipeline": {
        "supportedTasks": ["edit_image", "multi_image_reference_edit", "inpaint"],
        "defaultRepo": QWEN_IMAGE_EDIT_PLUS_REPO,
        "executionPath": "modular-diffusers",
        "pipelineClass": "QwenImageEditPlusModularPipeline",
        "qualityDefaults": {"width": 768, "height": 768, "steps": 24, "guidanceScale": 4, "maxSequenceLength": 512},
        "minimum": {"accelerator": "cuda", "vramBytes": 24 * GIB, "systemRamBytes": 48 * GIB, "diskFreeBytes": 35 * GIB},
        "recommended": {"accelerator": "cuda", "vramBytes": 32 * GIB, "systemRamBytes": 64 * GIB, "diskFreeBytes": 45 * GIB},
        "onLoadQuantization": {
            "accelerator": "cuda",
            "vramBytes": 16 * GIB,
            "systemRamBytes": 32 * GIB,
            "diskFreeBytes": 35 * GIB,
            "quantizationMode": "bnb_4bit",
            "quantizedComponents": ["transformer", "text_encoder"],
        },
        "supportedOffloadModes": [OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK],
        "requiredPackages": ["diffusers", "transformers", "accelerate", "torch", "bitsandbytes"],
        "guardedReason": "Qwen Edit Plus has guarded Auto coverage through the modular Diffusers graph. Failures are recorded per machine and candidate.",
    },
    "QwenImageLayeredModularPipeline": {
        "supportedTasks": ["layer_decomposition"],
        "defaultRepo": QWEN_IMAGE_LAYERED_REPO,
        "executionPath": "modular-diffusers",
        "pipelineClass": "QwenImageLayeredModularPipeline",
        "qualityDefaults": {"width": 768, "height": 768, "steps": 30, "guidanceScale": 4, "maxSequenceLength": 512},
        "minimum": {"accelerator": "cuda", "vramBytes": 24 * GIB, "systemRamBytes": 48 * GIB, "diskFreeBytes": 35 * GIB},
        "recommended": {"accelerator": "cuda", "vramBytes": 32 * GIB, "systemRamBytes": 64 * GIB, "diskFreeBytes": 45 * GIB},
        "onLoadQuantization": {
            "accelerator": "cuda",
            "vramBytes": 16 * GIB,
            "systemRamBytes": 32 * GIB,
            "diskFreeBytes": 35 * GIB,
            "quantizationMode": "bnb_4bit",
            "quantizedComponents": ["transformer", "text_encoder"],
        },
        "supportedOffloadModes": [OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK],
        "requiredPackages": ["diffusers", "transformers", "accelerate", "torch", "bitsandbytes"],
        "guardedReason": "Qwen Layered is guarded Auto coverage. MoDiff can try a low-memory modular recipe and remember failures.",
    },
    "WanVACEPipeline": {
        "supportedTasks": [
            "text_to_video",
            "image_to_video",
            "video_to_video",
            "video_inpaint",
            "video_outpaint",
            "reference_to_video",
            "control_to_video",
            "video_color_edit",
        ],
        "defaultRepo": WAN_VACE_REPO,
        "executionPath": "direct-wan-vace",
        "qualityDefaults": {"width": 832, "height": 480, "steps": 24, "guidanceScale": 5, "numFrames": 49},
        "minimum": {"accelerator": "cuda", "vramBytes": 10 * GIB, "systemRamBytes": 24 * GIB},
        "recommended": {"accelerator": "cuda", "vramBytes": 12 * GIB, "systemRamBytes": 32 * GIB},
        "highQuality": {"accelerator": "cuda", "vramBytes": 24 * GIB, "systemRamBytes": 48 * GIB},
        "supportedOffloadModes": [OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK],
    },
    "AceStepAudioPipeline": {
        "supportedTasks": ["text_to_audio", "audio_variation", "audio_continuation", "audio_repaint"],
        "defaultRepo": ACE_STEP_REPO,
        "executionPath": "direct-diffusers-audio",
        "pipelineClass": "AceStepPipeline",
        "qualityDefaults": {"audioDuration": 30, "steps": 8, "guidanceScale": 1, "shift": 3},
        "minimum": {"accelerator": "cuda", "vramBytes": 10 * GIB, "systemRamBytes": 24 * GIB, "diskFreeBytes": 20 * GIB},
        "recommended": {"accelerator": "cuda", "vramBytes": 16 * GIB, "systemRamBytes": 32 * GIB, "diskFreeBytes": 30 * GIB},
        "supportedOffloadModes": [OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK],
        "requiredPackages": ["diffusers", "transformers", "accelerate", "torch", "scipy"],
    },
    "FluxSchnellPipeline": {
        "supportedTasks": ["text_to_image"],
        "defaultRepo": FLUX_SCHNELL_REPO,
        "executionPath": "direct-diffusers-image",
        "pipelineClass": "FluxPipeline",
        "qualityDefaults": {"width": 1024, "height": 1024, "steps": 4, "guidanceScale": 0, "maxSequenceLength": 256},
        "minimum": {"accelerator": "cuda", "vramBytes": 12 * GIB, "systemRamBytes": 24 * GIB, "diskFreeBytes": 25 * GIB},
        "recommended": {"accelerator": "cuda", "vramBytes": 16 * GIB, "systemRamBytes": 32 * GIB, "diskFreeBytes": 35 * GIB},
        "supportedOffloadModes": [OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK],
        "requiredPackages": ["diffusers", "transformers", "accelerate", "torch"],
    },
    "FluxDevPipeline": {
        "supportedTasks": ["text_to_image"],
        "defaultRepo": FLUX_DEV_REPO,
        "preferredLowerMemoryRepo": FLUX_DEV_FP8_REPO,
        "executionPath": "direct-diffusers-image",
        "pipelineClass": "FluxPipeline",
        "qualityDefaults": {"width": 768, "height": 768, "steps": 20, "guidanceScale": 3.5, "maxSequenceLength": 256},
        "minimum": {"accelerator": "cuda", "vramBytes": 24 * GIB, "systemRamBytes": 48 * GIB, "diskFreeBytes": 45 * GIB},
        "recommended": {"accelerator": "cuda", "vramBytes": 32 * GIB, "systemRamBytes": 64 * GIB, "diskFreeBytes": 60 * GIB},
        "lowerMemory": {
            "accelerator": "cuda",
            "vramBytes": 16 * GIB,
            "systemRamBytes": 32 * GIB,
            "diskFreeBytes": 45 * GIB,
            "quantizationMode": "quanto_float8",
            "quantizedComponents": ["transformer", "text_encoder_2"],
        },
        "supportedOffloadModes": [OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK],
        "requiredPackages": ["diffusers", "transformers", "accelerate", "torch", "optimum-quanto"],
    },
    "FluxKreaPipeline": {
        "supportedTasks": ["text_to_image"],
        "defaultRepo": FLUX_KREA_REPO,
        "executionPath": "direct-diffusers-image",
        "pipelineClass": "FluxPipeline",
        "qualityDefaults": {"width": 768, "height": 768, "steps": 24, "guidanceScale": 3.5, "maxSequenceLength": 256},
        "minimum": {"accelerator": "cuda", "vramBytes": 24 * GIB, "systemRamBytes": 48 * GIB, "diskFreeBytes": 45 * GIB},
        "recommended": {"accelerator": "cuda", "vramBytes": 32 * GIB, "systemRamBytes": 64 * GIB, "diskFreeBytes": 60 * GIB},
        "onLoadQuantization": {
            "accelerator": "cuda",
            "vramBytes": 16 * GIB,
            "systemRamBytes": 32 * GIB,
            "diskFreeBytes": 45 * GIB,
            "quantizationMode": "quanto_float8",
            "quantizedComponents": ["transformer", "text_encoder_2"],
        },
        "supportedOffloadModes": [OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK],
        "requiredPackages": ["diffusers", "transformers", "accelerate", "torch", "optimum-quanto"],
        "guardedReason": "FLUX Krea has broad guarded Auto coverage through on-load float8 quantization and Diffusers offload.",
    },
    "FluxKontextPipeline": {
        "supportedTasks": ["edit_image"],
        "defaultRepo": FLUX_KONTEXT_REPO,
        "preferredLowerMemoryRepo": FLUX_KONTEXT_NVFP4_REPO,
        "executionPath": "direct-diffusers-image",
        "pipelineClass": "FluxKontextPipeline",
        "qualityDefaults": {"width": 768, "height": 768, "steps": 24, "guidanceScale": 3.5, "maxSequenceLength": 256},
        "minimum": {"accelerator": "cuda", "vramBytes": 24 * GIB, "systemRamBytes": 48 * GIB, "diskFreeBytes": 45 * GIB},
        "recommended": {"accelerator": "cuda", "vramBytes": 32 * GIB, "systemRamBytes": 64 * GIB, "diskFreeBytes": 60 * GIB},
        "lowerMemory": {
            "accelerator": "cuda",
            "vramBytes": 16 * GIB,
            "systemRamBytes": 32 * GIB,
            "diskFreeBytes": 45 * GIB,
            "quantizationMode": "torchao_float8",
            "quantizedComponents": ["transformer", "text_encoder_2"],
        },
        "supportedOffloadModes": [OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK],
        "requiredPackages": ["diffusers", "transformers", "accelerate", "torch", "torchao"],
        "guardedReason": "FLUX Kontext uses the NVFP4 lower-memory artifact when available; failures are remembered for this machine.",
    },
    "FluxFillPipeline": {
        "supportedTasks": ["inpaint", "outpaint"],
        "defaultRepo": FLUX_FILL_REPO,
        "executionPath": "direct-diffusers-image",
        "pipelineClass": "FluxFillPipeline",
        "qualityDefaults": {"width": 768, "height": 768, "steps": 24, "guidanceScale": 30, "maxSequenceLength": 256},
        "minimum": {"accelerator": "cuda", "vramBytes": 24 * GIB, "systemRamBytes": 48 * GIB, "diskFreeBytes": 45 * GIB},
        "recommended": {"accelerator": "cuda", "vramBytes": 32 * GIB, "systemRamBytes": 64 * GIB, "diskFreeBytes": 60 * GIB},
        "onLoadQuantization": {
            "accelerator": "cuda",
            "vramBytes": 16 * GIB,
            "systemRamBytes": 32 * GIB,
            "diskFreeBytes": 45 * GIB,
            "quantizationMode": "quanto_float8",
            "quantizedComponents": ["transformer", "text_encoder_2"],
        },
        "supportedOffloadModes": [OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK],
        "requiredPackages": ["diffusers", "transformers", "accelerate", "torch", "optimum-quanto"],
        "guardedReason": "FLUX Fill has guarded Auto coverage through generic Diffusers inpaint/outpaint nodes and on-load quantization.",
    },
    "FluxDepthPipeline": {
        "supportedTasks": ["control_image"],
        "defaultRepo": FLUX_DEPTH_REPO,
        "executionPath": "direct-diffusers-image",
        "pipelineClass": "FluxControlPipeline",
        "qualityDefaults": {"width": 768, "height": 768, "steps": 24, "guidanceScale": 10, "maxSequenceLength": 256},
        "minimum": {"accelerator": "cuda", "vramBytes": 24 * GIB, "systemRamBytes": 48 * GIB, "diskFreeBytes": 45 * GIB},
        "recommended": {"accelerator": "cuda", "vramBytes": 32 * GIB, "systemRamBytes": 64 * GIB, "diskFreeBytes": 60 * GIB},
        "onLoadQuantization": {
            "accelerator": "cuda",
            "vramBytes": 16 * GIB,
            "systemRamBytes": 32 * GIB,
            "diskFreeBytes": 45 * GIB,
            "quantizationMode": "quanto_float8",
            "quantizedComponents": ["transformer", "text_encoder_2"],
        },
        "supportedOffloadModes": [OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK],
        "requiredPackages": ["diffusers", "transformers", "accelerate", "torch", "optimum-quanto"],
        "guardedReason": "FLUX Depth has guarded Auto coverage through generic control-image Diffusers nodes.",
    },
    "FluxCannyPipeline": {
        "supportedTasks": ["control_image"],
        "defaultRepo": FLUX_CANNY_REPO,
        "executionPath": "direct-diffusers-image",
        "pipelineClass": "FluxControlPipeline",
        "qualityDefaults": {"width": 768, "height": 768, "steps": 24, "guidanceScale": 10, "maxSequenceLength": 256},
        "minimum": {"accelerator": "cuda", "vramBytes": 24 * GIB, "systemRamBytes": 48 * GIB, "diskFreeBytes": 45 * GIB},
        "recommended": {"accelerator": "cuda", "vramBytes": 32 * GIB, "systemRamBytes": 64 * GIB, "diskFreeBytes": 60 * GIB},
        "onLoadQuantization": {
            "accelerator": "cuda",
            "vramBytes": 16 * GIB,
            "systemRamBytes": 32 * GIB,
            "diskFreeBytes": 45 * GIB,
            "quantizationMode": "quanto_float8",
            "quantizedComponents": ["transformer", "text_encoder_2"],
        },
        "supportedOffloadModes": [OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK],
        "requiredPackages": ["diffusers", "transformers", "accelerate", "torch", "optimum-quanto"],
        "guardedReason": "FLUX Canny has guarded Auto coverage through generic control-image Diffusers nodes.",
    },
    "FluxReduxPipeline": {
        "supportedTasks": ["edit_image"],
        "defaultRepo": FLUX_REDUX_REPO,
        "executionPath": "direct-diffusers-image",
        "pipelineClass": "FluxPipeline",
        "qualityDefaults": {"width": 768, "height": 768, "steps": 24, "guidanceScale": 3.5, "maxSequenceLength": 256},
        "minimum": {"accelerator": "cuda", "vramBytes": 24 * GIB, "systemRamBytes": 48 * GIB, "diskFreeBytes": 45 * GIB},
        "recommended": {"accelerator": "cuda", "vramBytes": 32 * GIB, "systemRamBytes": 64 * GIB, "diskFreeBytes": 60 * GIB},
        "onLoadQuantization": {
            "accelerator": "cuda",
            "vramBytes": 16 * GIB,
            "systemRamBytes": 32 * GIB,
            "diskFreeBytes": 45 * GIB,
            "quantizationMode": "quanto_float8",
            "quantizedComponents": ["transformer", "text_encoder_2"],
        },
        "supportedOffloadModes": [OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK],
        "requiredPackages": ["diffusers", "transformers", "accelerate", "torch", "optimum-quanto"],
        "guardedReason": "FLUX Redux has guarded Auto coverage through generic Diffusers image/reference nodes.",
    },
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _repo_id_set(local_models: list[dict[str, Any]] | None) -> set[str]:
    output = set()
    for model in local_models or []:
        if isinstance(model, dict) and model.get("id"):
            output.add(str(model["id"]).lower())
            continue
        if isinstance(model, str):
            output.add(model.lower())
    return output


def _local_model_record(local_models: list[dict[str, Any]] | None, repo_id: str) -> dict[str, Any] | str | None:
    repo_id_lower = str(repo_id).lower()
    for model in local_models or []:
        if isinstance(model, dict) and str(model.get("id") or "").lower() == repo_id_lower:
            return model
        if isinstance(model, str) and model.lower() == repo_id_lower:
            return model
    return None


def _hf_repo_cache_dir(repo_id: str, cache_dir: str | os.PathLike[str]) -> Path:
    repo_folder = "models--" + str(repo_id).replace("/", "--")
    return Path(cache_dir).expanduser() / repo_folder


def _cache_dirs_for_record(record: dict[str, Any]) -> list[str]:
    cache_dirs = []
    for value in record.get("cache_dirs") or []:
        if value:
            cache_dirs.append(str(value))
    if record.get("cache_dir"):
        cache_dirs.append(str(record["cache_dir"]))

    seen_cache_dirs = []
    for cache_dir in cache_dirs:
        normalized = str(Path(cache_dir).expanduser()).lower()
        if normalized not in seen_cache_dirs:
            seen_cache_dirs.append(normalized)
    return seen_cache_dirs


def _snapshot_dirs_for_model(repo_id: str, record: dict[str, Any]) -> list[Path]:
    seen_cache_dirs = _cache_dirs_for_record(record)

    revision_hashes = [
        str(revision.get("hash"))
        for revision in record.get("revisions") or []
        if isinstance(revision, dict) and revision.get("hash")
    ]
    snapshot_dirs: list[Path] = []
    for cache_dir in seen_cache_dirs:
        snapshots_root = _hf_repo_cache_dir(repo_id, cache_dir) / "snapshots"
        if revision_hashes:
            snapshot_dirs.extend(snapshots_root / revision_hash for revision_hash in revision_hashes)
        elif snapshots_root.exists():
            try:
                snapshot_dirs.extend(path for path in snapshots_root.iterdir() if path.is_dir())
            except OSError:
                continue
    return snapshot_dirs


def _repo_plan_path(repo_id: str, cache_dir: str | os.PathLike[str]) -> Path:
    return _hf_repo_cache_dir(repo_id, cache_dir) / ".modiff_download_plan.json"


def _expected_files_for_repo(repo_id: str, cache_dirs: list[str]) -> list[dict[str, Any]]:
    for cache_dir in cache_dirs:
        plan_path = _repo_plan_path(repo_id, cache_dir)
        if not plan_path.exists():
            continue
        try:
            with plan_path.open("r", encoding="utf-8") as handle:
                plan = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        files = plan.get("files") if isinstance(plan, dict) else None
        if isinstance(files, list):
            return [item for item in files if isinstance(item, dict)]
    return []


def _active_repo_download_files(repo_path: Path, limit: int = 25) -> list[str]:
    active: list[str] = []
    if not repo_path.exists():
        return active
    try:
        for entry in repo_path.rglob("*"):
            if not entry.is_file():
                continue
            lower = entry.name.lower()
            if not (lower.endswith(".incomplete") or lower.endswith(".lock")):
                continue
            try:
                active.append(str(entry.relative_to(repo_path)).replace("\\", "/"))
            except ValueError:
                active.append(str(entry))
            if len(active) >= limit:
                break
    except OSError:
        return active
    return active


def _validate_json_files(snapshot_dir: Path) -> list[str]:
    corrupt: list[str] = []
    for json_path in snapshot_dir.rglob("*.json"):
        if json_path.name not in CONFIG_JSON_NAMES and not json_path.name.endswith(".index.json"):
            continue
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                json.load(handle)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
            try:
                rel_path = str(json_path.relative_to(snapshot_dir)).replace("\\", "/")
            except ValueError:
                rel_path = str(json_path)
            corrupt.append(f"{rel_path} ({exc})")
    return corrupt


def _zero_byte_model_files(snapshot_dir: Path) -> list[str]:
    zero: list[str] = []
    try:
        files = list(snapshot_dir.rglob("*"))
    except OSError:
        return zero
    for path in files:
        if not path.is_file():
            continue
        if path.suffix.lower() not in MODELISH_EXTENSIONS and path.name not in CONFIG_JSON_NAMES:
            continue
        try:
            if path.stat().st_size != 0:
                continue
            zero.append(str(path.relative_to(snapshot_dir)).replace("\\", "/"))
        except (OSError, ValueError):
            continue
    return zero


def _expected_file_mismatches(snapshot_dir: Path, expected_files: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    wrong_size: list[str] = []
    for expected_file in expected_files:
        name = expected_file.get("name")
        if not name:
            continue
        rel_name = str(name).replace("\\", "/")
        expected_path = snapshot_dir / rel_name
        if not expected_path.exists():
            missing.append(rel_name)
            continue
        expected_size = expected_file.get("size")
        if not isinstance(expected_size, int) or expected_size < 0:
            continue
        try:
            actual_size = expected_path.stat().st_size
        except OSError:
            continue
        if actual_size != expected_size:
            wrong_size.append(f"{rel_name} ({actual_size} of {expected_size} bytes)")
    return missing, wrong_size


def _validate_snapshot_shards(snapshot_dir: Path, expected_files: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    missing: list[str] = []
    corrupt: list[str] = []
    checked_indexes: list[str] = []
    checked_shards = 0
    expected_files = expected_files or []

    if not snapshot_dir.exists():
        return {
            "complete": False,
            "reason": f"Snapshot folder is missing: {snapshot_dir}",
            "missingFiles": [str(snapshot_dir)],
            "corruptFiles": [],
            "checkedIndexes": [],
            "checkedShardCount": 0,
        }

    try:
        index_files = list(snapshot_dir.rglob("*.index.json"))
    except OSError as exc:
        return {
            "complete": False,
            "reason": f"Could not inspect snapshot folder: {exc}",
            "missingFiles": [],
            "corruptFiles": [],
            "checkedIndexes": [],
            "checkedShardCount": 0,
        }

    corrupt.extend(_validate_json_files(snapshot_dir))
    zero_files = _zero_byte_model_files(snapshot_dir)
    if zero_files:
        corrupt.extend(f"{path} (zero bytes)" for path in zero_files)

    for index_file in index_files:
        try:
            rel_index = str(index_file.relative_to(snapshot_dir)).replace("\\", "/")
        except ValueError:
            rel_index = str(index_file)
        checked_indexes.append(rel_index)
        try:
            with index_file.open("r", encoding="utf-8") as handle:
                index_data = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            corrupt.append(f"{rel_index} (unreadable index: {exc})")
            continue

        weight_map = index_data.get("weight_map") if isinstance(index_data, dict) else None
        if not isinstance(weight_map, dict):
            continue
        for shard_name in sorted({str(value) for value in weight_map.values() if value}):
            checked_shards += 1
            expected = index_file.parent / shard_name
            fallback_expected = snapshot_dir / shard_name
            if expected.exists() or fallback_expected.exists():
                continue
            try:
                rel_missing = str(expected.relative_to(snapshot_dir)).replace("\\", "/")
            except ValueError:
                rel_missing = str(expected)
            missing.append(rel_missing)

    expected_missing, wrong_size = _expected_file_mismatches(snapshot_dir, expected_files)
    for item in expected_missing:
        if item not in missing:
            missing.append(item)
    corrupt.extend(wrong_size)

    if missing or corrupt:
        problem_preview = missing + corrupt
        preview = ", ".join(problem_preview[:3])
        suffix = f" and {len(problem_preview) - 3} more" if len(problem_preview) > 3 else ""
        reason = "Cached artifact snapshot is incomplete"
        if corrupt and not missing:
            reason = "Cached artifact snapshot needs repair"
        return {
            "complete": False,
            "reason": f"{reason}; {preview}{suffix}.",
            "missingFiles": missing[:25],
            "corruptFiles": corrupt[:25],
            "checkedIndexes": checked_indexes,
            "checkedShardCount": checked_shards,
        }

    if checked_indexes:
        return {
            "complete": True,
            "reason": "All sharded weight index references are present.",
            "missingFiles": [],
            "corruptFiles": [],
            "checkedIndexes": checked_indexes,
            "checkedShardCount": checked_shards,
        }

    has_weight_file = any(snapshot_dir.rglob("*.safetensors")) or any(snapshot_dir.rglob("*.bin"))
    return {
        "complete": bool(has_weight_file),
        "reason": "No shard index files found; direct weight files are present." if has_weight_file else "No local weight files were found in the snapshot.",
        "missingFiles": [],
        "corruptFiles": [],
        "checkedIndexes": [],
        "checkedShardCount": 0,
    }


def _artifact_cache_status(repo_id: str, local_models: list[dict[str, Any]] | None) -> dict[str, Any]:
    record = _local_model_record(local_models, repo_id)
    if record is None:
        return {
            "installed": False,
            "complete": False,
            "repairRequired": False,
            "reason": "Artifact is not installed in a configured Hugging Face cache.",
            "missingFiles": [],
            "corruptFiles": [],
            "activeFiles": [],
        }
    if not isinstance(record, dict):
        return {
            "installed": True,
            "complete": True,
            "repairRequired": False,
            "reason": "Artifact was reported installed, but no cache metadata was available for shard validation.",
            "missingFiles": [],
            "corruptFiles": [],
        }

    cache_dirs = _cache_dirs_for_record(record)
    expected_files = _expected_files_for_repo(repo_id, cache_dirs)
    active_files: list[str] = []
    for cache_dir in cache_dirs:
        active_files.extend(_active_repo_download_files(_hf_repo_cache_dir(repo_id, cache_dir)))
    active_files = list(dict.fromkeys(active_files))
    snapshot_dirs = _snapshot_dirs_for_model(repo_id, record)
    if not snapshot_dirs:
        return {
            "installed": True,
            "complete": False,
            "repairRequired": True,
            "reason": "Artifact is indexed locally, but no Hugging Face snapshot folder could be found.",
            "missingFiles": [],
            "corruptFiles": [],
            "activeFiles": active_files,
        }

    checked = []
    for snapshot_dir in snapshot_dirs:
        result = _validate_snapshot_shards(snapshot_dir, expected_files)
        result["path"] = str(snapshot_dir)
        checked.append(result)
        # A healthy snapshot is runnable even when another discovered cache root
        # contains an orphaned/parallel `.incomplete` file for the same repo.
        # Treating repo-wide partial markers as authoritative made a complete
        # configured snapshot flip between Ready and Repair after refresh.
        if result.get("complete"):
            reason = result.get("reason")
            if active_files:
                reason = f"{reason} A separate cache location also contains an unfinished download."
            return {
                "installed": True,
                "complete": True,
                "repairRequired": False,
                "reason": reason,
                "missingFiles": [],
                "corruptFiles": [],
                "activeFiles": active_files[:25],
                "snapshots": checked,
                "expectedFileCount": len(expected_files),
            }

    first_reason = checked[0].get("reason") if checked else "Artifact snapshot could not be validated."
    if active_files:
        first_reason = f"Artifact download is still incomplete or locked: {', '.join(active_files[:3])}."
    missing_files: list[str] = []
    corrupt_files: list[str] = []
    for result in checked:
        for item in result.get("missingFiles") or []:
            if item not in missing_files:
                missing_files.append(str(item))
        for item in result.get("corruptFiles") or []:
            if item not in corrupt_files:
                corrupt_files.append(str(item))
    return {
        "installed": True,
        "complete": False,
        "repairRequired": True,
        "reason": first_reason,
        "missingFiles": missing_files[:25],
        "corruptFiles": corrupt_files[:25],
        "activeFiles": active_files[:25],
        "snapshots": checked,
        "expectedFileCount": len(expected_files),
    }


def _runtime_key(runtime_fingerprint: dict[str, Any] | None) -> str:
    if not isinstance(runtime_fingerprint, dict):
        return "unknown-runtime"
    return str(runtime_fingerprint.get("fingerprint") or "unknown-runtime")


def _history_path(data_dir: str | os.PathLike[str]) -> Path:
    return Path(data_dir).expanduser() / AUTO_HISTORY_RELATIVE_PATH


def read_auto_resource_history(data_dir: str | os.PathLike[str]) -> dict[str, Any]:
    path = _history_path(data_dir)
    if not path.exists():
        return {"version": AUTO_HISTORY_VERSION, "entries": {}}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        return {"version": AUTO_HISTORY_VERSION, "entries": {}}
    entries = data.get("entries") if isinstance(data, dict) else None
    return {
        "version": AUTO_HISTORY_VERSION,
        "entries": entries if isinstance(entries, dict) else {},
    }


def _write_auto_resource_history(data_dir: str | os.PathLike[str], history: dict[str, Any]) -> None:
    path = _history_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    payload = {
        "version": AUTO_HISTORY_VERSION,
        "entries": history.get("entries") if isinstance(history.get("entries"), dict) else {},
    }
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def _hardware_history_key(runtime_fingerprint: dict[str, Any] | None = None, hardware: dict[str, Any] | None = None) -> str:
    if isinstance(hardware, dict) and hardware.get("runtimeFingerprint"):
        return str(hardware["runtimeFingerprint"])
    return _runtime_key(runtime_fingerprint)


def _candidate_history_signature(
    candidate: dict[str, Any],
    *,
    runtime_fingerprint: dict[str, Any] | None = None,
    hardware: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "hardwareFingerprint": _hardware_history_key(runtime_fingerprint, hardware),
        "modelType": str(candidate.get("modelType") or ""),
        "mode": str(candidate.get("mode") or ""),
        "artifact": str(candidate.get("resolvedArtifact") or candidate.get("artifact") or candidate.get("modelRepo") or ""),
        "dtype": str(candidate.get("dtype") or ""),
        "quantizationMode": str(candidate.get("quantizationMode") or "none"),
        "quantizedComponents": [str(item) for item in candidate.get("quantizedComponents") or []],
        "offloadMode": str(candidate.get("offloadMode") or ""),
        "pipelineClass": str(candidate.get("pipelineClass") or ""),
        "executionPath": str(candidate.get("executionPath") or ""),
    }


def auto_resource_history_key(
    candidate: dict[str, Any],
    *,
    runtime_fingerprint: dict[str, Any] | None = None,
    hardware: dict[str, Any] | None = None,
) -> str:
    payload = json.dumps(
        _candidate_history_signature(candidate, runtime_fingerprint=runtime_fingerprint, hardware=hardware),
        sort_keys=True,
        separators=(",", ":"),
    )
    import hashlib

    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _runtime_candidate_from_hints(runtime_hints: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(runtime_hints, dict) or runtime_hints.get("resourceMode") != "auto":
        return None
    auto_plan = runtime_hints.get("autoResourcePlan")
    if isinstance(auto_plan, dict):
        return dict(auto_plan)
    return {
        "id": runtime_hints.get("autoResourceCandidateId"),
        "modelType": runtime_hints.get("modelType"),
        "mode": runtime_hints.get("mode"),
        "modelRepo": runtime_hints.get("modelRepo"),
        "resolvedArtifact": runtime_hints.get("resolvedArtifact") or runtime_hints.get("resolvedModelRepo"),
        "dtype": runtime_hints.get("dtype"),
        "quantizationMode": runtime_hints.get("quantizationMode"),
        "quantizedComponents": runtime_hints.get("quantizedComponents") if isinstance(runtime_hints.get("quantizedComponents"), list) else [],
        "offloadMode": runtime_hints.get("offloadMode"),
        "pipelineClass": runtime_hints.get("pipelineClass"),
        "executionPath": runtime_hints.get("executionPath"),
    }


def _history_candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": candidate.get("id"),
        "modelType": candidate.get("modelType"),
        "mode": candidate.get("mode"),
        "artifact": candidate.get("resolvedArtifact") or candidate.get("artifact") or candidate.get("modelRepo"),
        "dtype": candidate.get("dtype"),
        "quantizationMode": candidate.get("quantizationMode"),
        "quantizedComponents": candidate.get("quantizedComponents") if isinstance(candidate.get("quantizedComponents"), list) else [],
        "offloadMode": candidate.get("offloadMode"),
        "pipelineClass": candidate.get("pipelineClass"),
        "executionPath": candidate.get("executionPath"),
    }


def record_auto_resource_success(
    data_dir: str | os.PathLike[str],
    *,
    runtime_fingerprint: dict[str, Any] | None,
    runtime_hints: dict[str, Any] | None,
) -> dict[str, Any] | None:
    candidate = _runtime_candidate_from_hints(runtime_hints)
    if not candidate:
        return None
    history = read_auto_resource_history(data_dir)
    key = auto_resource_history_key(candidate, runtime_fingerprint=runtime_fingerprint)
    entries = history.setdefault("entries", {})
    entry = entries.get(key) if isinstance(entries.get(key), dict) else {}
    now = _now_ms()
    entry.update({
        "key": key,
        "signature": _candidate_history_signature(candidate, runtime_fingerprint=runtime_fingerprint),
        "candidate": _history_candidate_summary(candidate),
        "successCount": int(entry.get("successCount") or 0) + 1,
        "lastSuccessAt": now,
        "lastStatus": "live_proven",
    })
    entries[key] = entry
    _write_auto_resource_history(data_dir, history)
    return entry


def record_auto_resource_failure(
    data_dir: str | os.PathLike[str],
    *,
    runtime_fingerprint: dict[str, Any] | None,
    runtime_hints: dict[str, Any] | None,
    classification: dict[str, Any] | None = None,
    error: BaseException | str | None = None,
) -> dict[str, Any] | None:
    candidate = _runtime_candidate_from_hints(runtime_hints)
    if not candidate:
        return None
    history = read_auto_resource_history(data_dir)
    key = auto_resource_history_key(candidate, runtime_fingerprint=runtime_fingerprint)
    entries = history.setdefault("entries", {})
    entry = entries.get(key) if isinstance(entries.get(key), dict) else {}
    now = _now_ms()
    error_message = str(error) if error is not None else ""
    entry.update({
        "key": key,
        "signature": _candidate_history_signature(candidate, runtime_fingerprint=runtime_fingerprint),
        "candidate": _history_candidate_summary(candidate),
        "failureCount": int(entry.get("failureCount") or 0) + 1,
        "lastFailureAt": now,
        "lastStatus": FAILED_HERE_PROOF_STATUS,
        "lastFailure": {
            "category": (classification or {}).get("category"),
            "errorCode": (classification or {}).get("error_code"),
            "message": (classification or {}).get("message") or error_message,
            "recoveryHint": (classification or {}).get("recovery_hint"),
            "checkedAt": now,
        },
    })
    entries[key] = entry
    _write_auto_resource_history(data_dir, history)
    return entry


def clear_auto_resource_history(
    data_dir: str | os.PathLike[str],
    *,
    model_type: str | None = None,
    mode: str | None = None,
    artifact: str | None = None,
) -> int:
    history = read_auto_resource_history(data_dir)
    entries = history.get("entries") if isinstance(history.get("entries"), dict) else {}
    removed = 0
    for key, entry in list(entries.items()):
        candidate = entry.get("candidate") if isinstance(entry, dict) else {}
        signature = entry.get("signature") if isinstance(entry, dict) else {}
        entry_model = str((candidate or {}).get("modelType") or (signature or {}).get("modelType") or "")
        entry_mode = str((candidate or {}).get("mode") or (signature or {}).get("mode") or "")
        entry_artifact = str((candidate or {}).get("artifact") or (signature or {}).get("artifact") or "")
        if model_type and entry_model != model_type:
            continue
        if mode and entry_mode != mode:
            continue
        if artifact and entry_artifact != artifact:
            continue
        entries.pop(key, None)
        removed += 1
    if removed:
        _write_auto_resource_history(data_dir, history)
    return removed


def _system_memory_snapshot(normalized_hardware: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        memory = system_memory_snapshot()
    except Exception:
        memory = {}

    system = normalized_hardware.get("system") if isinstance(normalized_hardware, dict) else None
    system = system if isinstance(system, dict) else {}
    total = _safe_int(system.get("ram_total"))
    free = _safe_int(system.get("ram_free"))
    available = _safe_int(system.get("ram_available"))
    return {
        "source": memory.get("source") or ("modiff.hardware" if system else "unavailable"),
        "totalBytes": total if total is not None else _safe_int(memory.get("total_bytes")),
        "freeBytes": free if free is not None else _safe_int(memory.get("free_bytes")),
        "availableBytes": available if available is not None else _safe_int(memory.get("available_bytes")),
        "pageFileTotalBytes": _safe_int(memory.get("page_file_total_bytes")),
        "pageFileAvailableBytes": _safe_int(memory.get("page_file_available_bytes")),
    }


def _disk_snapshot(
    path: str | os.PathLike[str],
    normalized_hardware: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_disk = normalized_hardware.get("disk") if isinstance(normalized_hardware, dict) else None
    normalized_disk = normalized_disk if isinstance(normalized_disk, dict) else None
    if normalized_disk is None or normalized_disk.get("path") is None:
        try:
            normalized_disk = disk_snapshot(path)
        except Exception as exc:
            normalized_disk = {
                "path": str(path),
                "error": str(exc),
            }

    snapshot = {
        "path": str(normalized_disk.get("path") or path),
        "totalBytes": _safe_int(normalized_disk.get("total_bytes")),
        "freeBytes": _safe_int(normalized_disk.get("free_bytes")),
        "usedBytes": _safe_int(normalized_disk.get("used_bytes")),
    }
    if normalized_disk.get("error"):
        snapshot["error"] = str(normalized_disk["error"])
    return snapshot


def _mps_accelerator(name: Any = None) -> dict[str, Any]:
    return {
        "kind": "mps",
        "name": str(name or "Apple Metal Performance Shaders"),
        "totalBytes": None,
        "freeBytes": None,
        "capability": None,
        "band": "shared_memory",
    }


def _cpu_accelerator(name: Any = None) -> dict[str, Any]:
    return {
        "kind": "cpu",
        "name": str(name or "CPU"),
        "totalBytes": None,
        "freeBytes": None,
        "capability": None,
        "band": "cpu_only",
    }


def _normalized_accelerator_snapshot(normalized_hardware: dict[str, Any] | None) -> dict[str, Any] | None:
    devices = normalized_hardware.get("devices") if isinstance(normalized_hardware, dict) else None
    devices = devices if isinstance(devices, list) else []
    device_by_kind = {
        kind: next(
            (item for item in devices if isinstance(item, dict) and str(item.get("type") or "").lower() == kind),
            None,
        )
        for kind in ("cuda", "mps", "cpu")
    }
    device = device_by_kind["cuda"]
    if device is not None:
        total = _safe_int(device.get("vram_total"))
        if total is None:
            total = _safe_int(device.get("torch_vram_total"))
        free = _safe_int(device.get("vram_free"))
        if free is None:
            free = _safe_int(device.get("torch_vram_free"))
        return {
            "kind": "cuda",
            "name": device.get("name"),
            "totalBytes": total,
            "freeBytes": free,
            "capability": device.get("capability"),
            "band": _vram_band(total),
        }

    torch_state = normalized_hardware.get("torch") if isinstance(normalized_hardware, dict) else None
    torch_state = torch_state if isinstance(torch_state, dict) else {}
    if bool(torch_state.get("cuda_available")):
        return {
            "kind": "cuda",
            "name": "CUDA",
            "totalBytes": None,
            "freeBytes": None,
            "capability": None,
            "band": "unknown",
        }

    device = device_by_kind["mps"]
    if device is not None:
        return _mps_accelerator(device.get("name"))
    if bool(torch_state.get("mps_available")):
        return _mps_accelerator()

    device = device_by_kind["cpu"]
    if device is not None:
        return _cpu_accelerator(device.get("name"))
    return None


def _legacy_accelerator_snapshot(runtime_fingerprint: dict[str, Any] | None) -> dict[str, Any] | None:
    torch_state = runtime_fingerprint.get("torch") if isinstance(runtime_fingerprint, dict) else None
    torch_state = torch_state if isinstance(torch_state, dict) else {}
    cuda_available = bool(torch_state.get("cuda_available"))
    total = _safe_int(torch_state.get("cuda_memory_total_bytes") or torch_state.get("cuda_device_total_memory_bytes"))
    free = _safe_int(torch_state.get("cuda_memory_free_bytes"))
    if cuda_available:
        return {
            "kind": "cuda",
            "name": torch_state.get("cuda_device_name"),
            "totalBytes": total,
            "freeBytes": free,
            "capability": torch_state.get("cuda_device_capability"),
            "band": _vram_band(total),
        }
    if bool(torch_state.get("mps_available")):
        mps_devices = torch_state.get("mps_devices")
        first_mps = mps_devices[0] if isinstance(mps_devices, list) and mps_devices and isinstance(mps_devices[0], dict) else {}
        return _mps_accelerator(first_mps.get("name"))
    if "cuda_available" in torch_state or "mps_available" in torch_state:
        return _cpu_accelerator()
    return None


def _accelerator_snapshot(
    runtime_fingerprint: dict[str, Any] | None,
    normalized_hardware: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_hardware = runtime_fingerprint.get("hardware") if isinstance(runtime_fingerprint, dict) else None
    runtime_hardware = runtime_hardware if isinstance(runtime_hardware, dict) else None
    if runtime_hardware is not None:
        accelerator = _normalized_accelerator_snapshot(runtime_hardware)
        if accelerator is not None:
            return accelerator

    accelerator = _legacy_accelerator_snapshot(runtime_fingerprint)
    if accelerator is not None:
        return accelerator
    return _normalized_accelerator_snapshot(normalized_hardware) or _cpu_accelerator()


def _hardware_snapshot(runtime_fingerprint: dict[str, Any] | None, data_dir: str | os.PathLike[str]) -> dict[str, Any]:
    runtime_hardware = runtime_fingerprint.get("hardware") if isinstance(runtime_fingerprint, dict) else None
    normalized_hardware = runtime_hardware if isinstance(runtime_hardware, dict) else None
    if normalized_hardware is None:
        try:
            normalized_hardware = get_hardware_snapshot(data_dir)
        except Exception:
            normalized_hardware = {}
    return {
        "runtimeFingerprint": _runtime_key(runtime_fingerprint),
        "runtime": runtime_fingerprint,
        "accelerator": _accelerator_snapshot(runtime_fingerprint, normalized_hardware),
        "systemMemory": _system_memory_snapshot(normalized_hardware),
        "offloadDisk": _disk_snapshot(data_dir, normalized_hardware),
    }


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _vram_band(total_bytes: int | None) -> str:
    if not total_bytes:
        return "unknown"
    if total_bytes < 10 * GIB:
        return "very_low"
    if total_bytes < 16 * GIB:
        return "low"
    if total_bytes < 24 * GIB:
        return "standard"
    if total_bytes < 32 * GIB:
        return "large"
    if total_bytes < 48 * GIB:
        return "high"
    return "workstation"


def _resource_value(hardware: dict[str, Any], section: str, key: str) -> int | None:
    value = hardware.get(section)
    if not isinstance(value, dict):
        return None
    return _safe_int(value.get(key))


def _has_installed(repo_id: str, installed: set[str]) -> bool:
    return bool(repo_id) and repo_id.lower() in installed


def _meets_total_capacity(actual_bytes: int | None, required_bytes: int | None) -> bool:
    """Treat near-tier hardware totals as their advertised capacity class.

    CUDA and operating-system reservations make nominal 16 GiB VRAM / 32 GiB RAM
    machines report slightly less than the marketed total. A one-percent tolerance
    admits that measurement variance without turning a materially smaller tier into
    a match. Free-disk requirements intentionally remain strict.
    """
    if not required_bytes:
        return True
    if not actual_bytes:
        return False
    return actual_bytes >= required_bytes * (1 - CAPACITY_CLASS_TOLERANCE)


def _requirements_missing(
    hardware: dict[str, Any],
    *,
    accelerator: str = "cuda",
    min_vram_bytes: int | None = None,
    min_system_ram_bytes: int | None = None,
    min_disk_free_bytes: int | None = None,
) -> list[str]:
    missing = []
    accelerator_state = hardware.get("accelerator") if isinstance(hardware.get("accelerator"), dict) else {}
    actual_accelerator = accelerator_state.get("kind")
    if accelerator == "cuda" and actual_accelerator != "cuda":
        missing.append("CUDA accelerator required")
    elif accelerator == "cuda_or_mps" and actual_accelerator not in {"cuda", "mps"}:
        missing.append("CUDA or MPS accelerator required")

    total_vram = _safe_int(accelerator_state.get("totalBytes"))
    if not _meets_total_capacity(total_vram, min_vram_bytes):
        missing.append(f"GPU memory requires at least {min_vram_bytes // GIB} GiB total")

    system_total = _resource_value(hardware, "systemMemory", "totalBytes")
    if not _meets_total_capacity(system_total, min_system_ram_bytes):
        missing.append(f"System memory requires at least {min_system_ram_bytes // GIB} GiB total")

    disk_free = _resource_value(hardware, "offloadDisk", "freeBytes")
    if min_disk_free_bytes and (not disk_free or disk_free < min_disk_free_bytes):
        missing.append(f"Offload disk requires at least {min_disk_free_bytes // GIB} GiB free")
    return missing


def _qwen_generation_defaults(form: dict[str, Any], hardware: dict[str, Any], *, native: bool = False) -> dict[str, Any]:
    negative = str(form.get("negativePrompt") or "").strip() or " "
    width = QWEN_NATIVE_WIDTH if native else QWEN_PRACTICAL_WIDTH
    height = QWEN_NATIVE_HEIGHT if native else QWEN_PRACTICAL_HEIGHT
    return {
        "width": width,
        "height": height,
        "steps": max(int(form.get("steps") or QWEN_NATIVE_STEPS), 40),
        "guidanceScale": QWEN_NATIVE_TRUE_CFG,
        "negativePrompt": negative,
        "maxSequenceLength": int(form.get("maxSequenceLength") or 512),
        "qualityPreset": "qwen-native-quality" if native else "qwen-practical-quality",
        "hardwareBand": (hardware.get("accelerator") or {}).get("band"),
    }


def _artifact_needs_repair(artifact_status: dict[str, Any] | None) -> bool:
    return bool(artifact_status and artifact_status.get("installed") and artifact_status.get("complete") is False)


def _candidate_health_badge(
    *,
    proof_status: str,
    artifact_status: dict[str, Any] | None,
    quality_tier: str,
    manual_only_reason: str | None,
    missing: list[str],
) -> str:
    if proof_status == "live_proven":
        return "Works here"
    if proof_status in READY_PROOF_STATUSES:
        if "quant" in quality_tier.lower() or "lower-memory" in quality_tier.lower():
            return "Works with quantized artifact"
        return "Works here"
    if _artifact_needs_repair(artifact_status):
        return "Repair required"
    if proof_status == FAILED_HERE_PROOF_STATUS:
        return "Failed here before"
    if proof_status == "known_bad":
        return "Will not work on this machine"
    if proof_status == "manual_only" or manual_only_reason:
        return "Expert only"
    if missing:
        blocked = [item for item in missing if "requires at least" in item.lower() or "requires cuda" in item.lower() or "offload disk" in item.lower()]
        if blocked and not (artifact_status and artifact_status.get("installed") is False):
            return "Will not work on this machine"
    return "Needs setup"


def _candidate_action_label(
    *,
    install_action_label: str | None,
    artifact_status: dict[str, Any] | None,
    quality_tier: str,
) -> str:
    if _artifact_needs_repair(artifact_status):
        return "Repair Auto artifact"
    if install_action_label:
        return install_action_label
    if "quant" in quality_tier.lower() or "lower-memory" in quality_tier.lower():
        return "Install quantized artifact"
    return "Install model"


def _candidate(
    *,
    candidate_id: str,
    rank: int,
    model_type: str,
    mode: str,
    execution_path: str,
    artifact: str,
    dtype: str,
    quantization_mode: str,
    quantized_components: list[str],
    offload_mode: str,
    quality_tier: str,
    reason: str,
    generation: dict[str, Any],
    installed: bool,
    requirements_missing: list[str] | None = None,
    manual_only_reason: str | None = None,
    known_bad_reasons: list[str] | None = None,
    artifact_status: dict[str, Any] | None = None,
    pipeline_class: str | None = None,
    requirements: dict[str, Any] | None = None,
    required_packages: list[str] | None = None,
    install_action_label: str | None = None,
) -> dict[str, Any]:
    missing = list(requirements_missing or [])
    known_bad = list(known_bad_reasons or [])
    if manual_only_reason:
        status = "manual_only"
        message = manual_only_reason
    elif known_bad:
        status = "known_bad"
        message = "; ".join(known_bad)
    elif not installed:
        status = "skipped"
        message = f"{artifact} is not installed in a runnable Hugging Face cache."
    elif missing:
        status = "skipped"
        message = "; ".join(missing)
    else:
        status = "declared_safe"
        message = reason
    repair_required = _artifact_needs_repair(artifact_status)
    health_badge = _candidate_health_badge(
        proof_status=status,
        artifact_status=artifact_status,
        quality_tier=quality_tier,
        manual_only_reason=manual_only_reason,
        missing=missing,
    )
    action_label = _candidate_action_label(
        install_action_label=install_action_label,
        artifact_status=artifact_status,
        quality_tier=quality_tier,
    )

    return {
        "id": candidate_id,
        "rank": rank,
        "modelType": model_type,
        "mode": mode,
        "executionPath": execution_path,
        "pipelineClass": pipeline_class,
        "artifact": artifact,
        "artifactSource": "huggingface-cache",
        "modelRepo": artifact,
        "resolvedArtifact": artifact,
        "dtype": dtype,
        "quantizationMode": quantization_mode,
        "quantizedComponents": quantized_components,
        "bnb4ComputeDtype": "bfloat16",
        "offloadMode": offload_mode,
        "autoOffload": offload_mode != OFFLOAD_MODE_NONE,
        "qualityTier": quality_tier,
        "qualityScore": max(0, 1000 - rank),
        "reason": reason,
        "candidateReasons": [reason],
        "knownBadReasons": known_bad,
        "generation": generation,
        "installed": installed,
        "artifactStatus": artifact_status,
        "artifactValidation": artifact_status,
        "repairRequired": repair_required,
        "healthBadge": health_badge,
        "canAutoRun": status in READY_PROOF_STATUSES,
        "installTarget": {
            "repo": artifact,
            "label": action_label,
            "reason": reason,
            "actionLabel": action_label,
            "repair": repair_required,
        },
        "requirements": requirements or {},
        "requiredPackages": list(required_packages or []),
        "readiness": "ready" if status in READY_PROOF_STATUSES else status,
        "requiresLocalProbe": False,
        "skipReason": None if status in READY_PROOF_STATUSES else message,
        "requirementsMatched": [] if missing else ["artifact", "hardware", "quality-defaults"],
        "requirementsMissing": missing,
        "proof": {
            "status": status,
            "source": "static_auto_requirements",
            "message": message,
            "checkedAt": _now_ms(),
        },
    }


def _qwen_auto_offload_for(hardware: dict[str, Any]) -> str:
    free = _resource_value(hardware, "accelerator", "freeBytes")
    if free is not None and free < 6 * GIB:
        return OFFLOAD_MODE_SEQUENTIAL_CPU
    return OFFLOAD_MODE_MODEL_CPU


def _qwen_text_to_image_candidates(
    form: dict[str, Any],
    local_models: list[dict[str, Any]] | None,
    hardware: dict[str, Any],
) -> list[dict[str, Any]]:
    installed = _repo_id_set(local_models)
    official_cache_status = _artifact_cache_status(QWEN_IMAGE_2512_REPO, local_models)
    prequantized_cache_status = _artifact_cache_status(QWEN_IMAGE_2512_PREQUANTIZED_REPO, local_models)
    official_installed = bool(official_cache_status.get("installed")) or _has_installed(QWEN_IMAGE_2512_REPO, installed)
    prequantized_installed = bool(prequantized_cache_status.get("installed")) or _has_installed(QWEN_IMAGE_2512_PREQUANTIZED_REPO, installed)
    model_type = str(form.get("modelType") or "QwenImageModularPipeline")
    mode = str(form.get("mode") or "text_to_image")

    prequantized_missing = _requirements_missing(
        hardware,
        accelerator="cuda",
        min_vram_bytes=10 * GIB,
        min_system_ram_bytes=24 * GIB,
    )
    official_missing = _requirements_missing(
        hardware,
        accelerator="cuda",
        min_vram_bytes=32 * GIB,
        min_system_ram_bytes=30 * GIB,
    )

    offload_mode = _qwen_auto_offload_for(hardware)
    prequantized_generation = _qwen_generation_defaults(form, hardware, native=False)
    official_generation = _qwen_generation_defaults(form, hardware, native=True)
    prequantized_cache_missing = []
    if prequantized_cache_status.get("installed") and not prequantized_cache_status.get("complete"):
        prequantized_cache_missing.append(str(prequantized_cache_status.get("reason") or "Cached prequantized artifact is incomplete."))
    official_cache_missing = []
    if official_cache_status.get("installed") and not official_cache_status.get("complete"):
        official_cache_missing.append(str(official_cache_status.get("reason") or "Cached official artifact is incomplete."))

    candidates = [
        _candidate(
            candidate_id="qwen-t2i-prequantized-model-cpu",
            rank=1,
            model_type=model_type,
            mode=mode,
            execution_path="direct-qwen-image",
            artifact=QWEN_IMAGE_2512_PREQUANTIZED_REPO,
            dtype="bfloat16",
            quantization_mode="none",
            quantized_components=[],
            offload_mode=offload_mode,
            quality_tier="diffusers-compatible-prequantized-quality",
            reason="Use the curated Diffusers-compatible prequantized Qwen artifact for quality settings on constrained CUDA resources.",
            generation=prequantized_generation,
            installed=prequantized_installed,
            requirements_missing=prequantized_missing + prequantized_cache_missing,
            artifact_status=prequantized_cache_status,
        ),
        _candidate(
            candidate_id="qwen-t2i-prequantized-sequential-cpu",
            rank=2,
            model_type=model_type,
            mode=mode,
            execution_path="direct-qwen-image",
            artifact=QWEN_IMAGE_2512_PREQUANTIZED_REPO,
            dtype="bfloat16",
            quantization_mode="none",
            quantized_components=[],
            offload_mode=OFFLOAD_MODE_SEQUENTIAL_CPU,
            quality_tier="diffusers-compatible-prequantized-quality",
            reason="Same Qwen quality settings with sequential CPU/system-memory offload when current free VRAM is tight.",
            generation=prequantized_generation,
            installed=prequantized_installed,
            requirements_missing=prequantized_missing + prequantized_cache_missing,
            artifact_status=prequantized_cache_status,
        ),
        _candidate(
            candidate_id="qwen-t2i-prequantized-group-disk",
            rank=3,
            model_type=model_type,
            mode=mode,
            execution_path="direct-qwen-image",
            artifact=QWEN_IMAGE_2512_PREQUANTIZED_REPO,
            dtype="bfloat16",
            quantization_mode="none",
            quantized_components=[],
            offload_mode=OFFLOAD_MODE_GROUP_DISK,
            quality_tier="diffusers-compatible-prequantized-emergency",
            reason="Same Qwen quality settings with SSD-backed Diffusers offload as a visible emergency fallback.",
            generation=prequantized_generation,
            installed=prequantized_installed,
            requirements_missing=_requirements_missing(
                hardware,
                accelerator="cuda",
                min_vram_bytes=10 * GIB,
                min_system_ram_bytes=20 * GIB,
                min_disk_free_bytes=20 * GIB,
            ) + prequantized_cache_missing,
            artifact_status=prequantized_cache_status,
        ),
        _candidate(
            candidate_id="qwen-t2i-official-bf16-native",
            rank=4,
            model_type=model_type,
            mode=mode,
            execution_path="direct-qwen-image",
            artifact=QWEN_IMAGE_2512_REPO,
            dtype="bfloat16",
            quantization_mode="none",
            quantized_components=[],
            offload_mode=OFFLOAD_MODE_MODEL_CPU,
            quality_tier="official-bf16-native-quality",
            reason="Use the official BF16 Diffusers repo only when hardware has enough GPU and system memory headroom.",
            generation=official_generation,
            installed=official_installed,
            requirements_missing=official_missing + official_cache_missing,
            artifact_status=official_cache_status,
        ),
        _candidate(
            candidate_id="qwen-t2i-official-transformer-bnb4-manual",
            rank=5,
            model_type=model_type,
            mode=mode,
            execution_path="direct-qwen-image",
            artifact=QWEN_IMAGE_2512_REPO,
            dtype="bfloat16",
            quantization_mode="bnb_4bit",
            quantized_components=QWEN_TRANSFORMER_ONLY_COMPONENTS,
            offload_mode=OFFLOAD_MODE_MODEL_CPU,
            quality_tier="manual-transformer-bnb4",
            reason="Official transformer-only BnB is available for Manual experimentation, not default Auto.",
            generation=prequantized_generation,
            installed=official_installed,
            manual_only_reason="On-the-fly BnB quantization is not an Auto default because package/kernel compatibility varies; use Manual if you want this configuration.",
            artifact_status=official_cache_status,
        ),
    ]
    return candidates


def _number_from_form_or_defaults(form: dict[str, Any], defaults: dict[str, Any], field: str, fallback: float):
    value = form.get(field)
    if value is None or value == "":
        value = defaults.get(field)
    if value is None or value == "":
        value = fallback
    return value


def _generation_for_requirements(model_type: str, form: dict[str, Any], generation_defaults: dict[str, Any]) -> dict[str, Any]:
    return {
        "width": int(_number_from_form_or_defaults(form, generation_defaults, "width", 1024)),
        "height": int(_number_from_form_or_defaults(form, generation_defaults, "height", 1024)),
        "steps": int(_number_from_form_or_defaults(form, generation_defaults, "steps", 30)),
        "guidanceScale": float(_number_from_form_or_defaults(form, generation_defaults, "guidanceScale", 4)),
        "negativePrompt": str(form.get("negativePrompt") or ""),
        "maxSequenceLength": int(_number_from_form_or_defaults(form, generation_defaults, "maxSequenceLength", 512)),
        "numFrames": int(_number_from_form_or_defaults(form, generation_defaults, "numFrames", 0)) or None,
        "audioDuration": float(_number_from_form_or_defaults(form, generation_defaults, "audioDuration", 0)) or None,
        "shift": float(_number_from_form_or_defaults(form, generation_defaults, "shift", 0)) or None,
        "qualityPreset": f"{model_type or 'studio'}-auto",
    }


def _requirements_for_candidate(requirements: dict[str, Any], key: str, fallback: dict[str, Any]) -> dict[str, Any]:
    value = requirements.get(key)
    return value if isinstance(value, dict) else fallback


def _requirements_missing_for_dict(hardware: dict[str, Any], requirement: dict[str, Any]) -> list[str]:
    return _requirements_missing(
        hardware,
        accelerator=str(requirement.get("accelerator") or "cuda_or_mps_or_cpu"),
        min_vram_bytes=_safe_int(requirement.get("vramBytes")),
        min_system_ram_bytes=_safe_int(requirement.get("systemRamBytes")),
        min_disk_free_bytes=_safe_int(requirement.get("diskFreeBytes")),
    )


def _cache_missing_for_status(status: dict[str, Any], label: str) -> list[str]:
    if status.get("installed") and not status.get("complete"):
        return [str(status.get("reason") or f"Cached {label} artifact is incomplete.")]
    return []


def _declared_profile_candidates(
    form: dict[str, Any],
    local_models: list[dict[str, Any]] | None,
    hardware: dict[str, Any],
) -> list[dict[str, Any]]:
    model_type = str(form.get("modelType") or "")
    mode = str(form.get("mode") or "")
    installed = _repo_id_set(local_models)
    key = f"{model_type}:{mode}" if f"{model_type}:{mode}" in AUTO_MODEL_REQUIREMENTS else model_type
    requirements = AUTO_MODEL_REQUIREMENTS.get(key) or {}
    default_repo = str(requirements.get("defaultRepo") or form.get("defaultRepo") or form.get("modelRepo") or "")
    lower_memory_repo = str(requirements.get("preferredLowerMemoryRepo") or "")
    manual_only_reason = requirements.get("manualOnlyReason")
    execution_path = str(requirements.get("executionPath") or ("direct-wan-vace" if model_type == "WanVACEPipeline" else "modular-diffusers"))
    pipeline_class = str(requirements.get("pipelineClass") or "")
    generation_defaults = requirements.get("qualityDefaults") if isinstance(requirements.get("qualityDefaults"), dict) else {}

    minimum = requirements.get("minimum") if isinstance(requirements.get("minimum"), dict) else requirements.get("recommended")
    minimum = minimum if isinstance(minimum, dict) else {}
    generation = _generation_for_requirements(model_type, form, generation_defaults)
    required_packages = requirements.get("requiredPackages") if isinstance(requirements.get("requiredPackages"), list) else []
    supported_offload = requirements.get("supportedOffloadModes") if isinstance(requirements.get("supportedOffloadModes"), list) else []
    preferred_offload = str(form.get("offloadMode") or (supported_offload[0] if supported_offload else OFFLOAD_MODE_MODEL_CPU))
    candidates: list[dict[str, Any]] = []

    lower_memory = requirements.get("lowerMemory") if isinstance(requirements.get("lowerMemory"), dict) else None
    if lower_memory_repo:
        lower_req = _requirements_for_candidate(requirements, "lowerMemory", minimum)
        lower_missing = _requirements_missing_for_dict(hardware, lower_req)
        lower_cache_status = _artifact_cache_status(lower_memory_repo, local_models)
        candidates.append(_candidate(
            candidate_id=f"{model_type or 'studio'}-{mode or 'mode'}-lower-memory-artifact",
            rank=10,
            model_type=model_type,
            mode=mode,
            execution_path=execution_path,
            artifact=lower_memory_repo,
            dtype=str(form.get("dtype") or "bfloat16"),
            quantization_mode=str((lower_memory or {}).get("quantizationMode") or requirements.get("quantizationMode") or "none"),
            quantized_components=list((lower_memory or {}).get("quantizedComponents") or requirements.get("quantizedComponents") or []),
            offload_mode=preferred_offload,
            quality_tier="lower-memory-quantized-artifact",
            reason=f"Use the known lower-memory Diffusers artifact {lower_memory_repo} for this model on constrained hardware.",
            generation=generation,
            installed=bool(lower_cache_status.get("installed")) or _has_installed(lower_memory_repo, installed),
            requirements_missing=lower_missing + _cache_missing_for_status(lower_cache_status, "lower-memory"),
            artifact_status=lower_cache_status,
            pipeline_class=pipeline_class or None,
            requirements={
                "minimum": requirements.get("minimum"),
                "recommended": requirements.get("recommended"),
                "lowerMemory": requirements.get("lowerMemory"),
                "supportedOffloadModes": requirements.get("supportedOffloadModes"),
            },
            required_packages=required_packages,
            install_action_label="Install quantized artifact",
        ))

    on_load = requirements.get("onLoadQuantization") if isinstance(requirements.get("onLoadQuantization"), dict) else None
    if on_load and default_repo:
        on_load_missing = _requirements_missing_for_dict(hardware, on_load)
        default_cache_status = _artifact_cache_status(default_repo, local_models)
        candidates.append(_candidate(
            candidate_id=f"{model_type or 'studio'}-{mode or 'mode'}-on-load-quantized",
            rank=20,
            model_type=model_type,
            mode=mode,
            execution_path=execution_path,
            artifact=default_repo,
            dtype=str(form.get("dtype") or "bfloat16"),
            quantization_mode=str(on_load.get("quantizationMode") or "none"),
            quantized_components=list(on_load.get("quantizedComponents") or []),
            offload_mode=OFFLOAD_MODE_GROUP_DISK if OFFLOAD_MODE_GROUP_DISK in supported_offload else preferred_offload,
            quality_tier="on-load-quantized-guarded",
            reason=str(requirements.get("guardedReason") or "Use a guarded on-load quantized Diffusers recipe with CPU/SSD offload where needed."),
            generation=generation,
            installed=bool(default_cache_status.get("installed")) or _has_installed(default_repo, installed),
            requirements_missing=on_load_missing + _cache_missing_for_status(default_cache_status, "default"),
            artifact_status=default_cache_status,
            pipeline_class=pipeline_class or None,
            requirements={
                "minimum": requirements.get("minimum"),
                "recommended": requirements.get("recommended"),
                "onLoadQuantization": requirements.get("onLoadQuantization"),
                "supportedOffloadModes": requirements.get("supportedOffloadModes"),
            },
            required_packages=required_packages,
            install_action_label="Install model for Auto quantization",
        ))

    if default_repo:
        native_req = requirements.get("minimum") if isinstance(requirements.get("minimum"), dict) else minimum
        native_missing = _requirements_missing_for_dict(hardware, native_req)
        default_cache_status = _artifact_cache_status(default_repo, local_models)
        candidates.append(_candidate(
            candidate_id=f"{model_type or 'studio'}-{mode or 'mode'}-native-bf16",
            rank=30,
            model_type=model_type,
            mode=mode,
            execution_path=execution_path,
            artifact=default_repo,
            dtype=str(form.get("dtype") or "bfloat16"),
            quantization_mode="none",
            quantized_components=[],
            offload_mode=preferred_offload,
            quality_tier="native-bf16-high-memory",
            reason="Use the native Diffusers artifact only when hardware and installed cache validation match.",
            generation=generation,
            installed=bool(default_cache_status.get("installed")) or _has_installed(default_repo, installed),
            requirements_missing=native_missing + _cache_missing_for_status(default_cache_status, "default"),
            manual_only_reason=str(manual_only_reason) if manual_only_reason else None,
            artifact_status=default_cache_status,
            pipeline_class=pipeline_class or None,
            requirements={
                "minimum": requirements.get("minimum"),
                "recommended": requirements.get("recommended"),
                "supportedOffloadModes": requirements.get("supportedOffloadModes"),
            },
            required_packages=required_packages,
        ))

    if not candidates:
        candidates.append(_candidate(
            candidate_id=f"{model_type or 'studio'}-{mode or 'mode'}-expert-only",
            rank=99,
            model_type=model_type,
            mode=mode,
            execution_path=execution_path,
            artifact=default_repo,
            dtype=str(form.get("dtype") or "bfloat16"),
            quantization_mode="none",
            quantized_components=[],
            offload_mode=preferred_offload,
            quality_tier="expert-only",
            reason="No Auto recipe is declared for this Studio profile.",
            generation=generation,
            installed=True,
            manual_only_reason=str(manual_only_reason or "Switch to Expert to configure this workflow."),
            pipeline_class=pipeline_class or None,
        ))

    return candidates


def _clone_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(candidate))


def _apply_history_to_candidates(
    candidates: list[dict[str, Any]],
    *,
    history: dict[str, Any] | None,
    hardware: dict[str, Any],
) -> list[dict[str, Any]]:
    entries = history.get("entries") if isinstance(history, dict) and isinstance(history.get("entries"), dict) else {}
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        item = _clone_candidate(candidate)
        key = auto_resource_history_key(item, hardware=hardware)
        entry = entries.get(key) if isinstance(entries.get(key), dict) else None
        item["historyKey"] = key
        item["failureHistory"] = entry if entry and entry.get("failureCount") else None
        item["successHistory"] = entry if entry and entry.get("successCount") else None
        if entry and entry.get("successCount") and not item.get("requirementsMissing") and item.get("installed"):
            proof = item.get("proof") if isinstance(item.get("proof"), dict) else {}
            proof = dict(proof)
            proof["status"] = "live_proven"
            proof["source"] = "auto_resource_history"
            proof["message"] = "This Auto candidate completed successfully on this machine before."
            proof["checkedAt"] = _now_ms()
            item["proof"] = proof
            item["readiness"] = "ready"
            item["canAutoRun"] = True
            item["healthBadge"] = "Works here"
            item["qualityScore"] = int(item.get("qualityScore") or 0) + 250
        if entry and entry.get("failureCount"):
            last_failure = entry.get("lastFailure") if isinstance(entry.get("lastFailure"), dict) else {}
            last_failure_at = int(entry.get("lastFailureAt") or 0)
            last_success_at = int(entry.get("lastSuccessAt") or 0)
            if last_failure_at >= last_success_at:
                proof = item.get("proof") if isinstance(item.get("proof"), dict) else {}
                message = str(last_failure.get("message") or "This Auto candidate failed on this machine before.")
                proof = dict(proof)
                proof["status"] = FAILED_HERE_PROOF_STATUS
                proof["source"] = "auto_resource_history"
                proof["message"] = message
                proof["category"] = last_failure.get("category")
                proof["errorCode"] = last_failure.get("errorCode")
                proof["checkedAt"] = _now_ms()
                item["proof"] = proof
                item["readiness"] = FAILED_HERE_PROOF_STATUS
                item["canAutoRun"] = False
                item["skipReason"] = message
                item["healthBadge"] = "Failed here before"
                known_bad = item.get("knownBadReasons") if isinstance(item.get("knownBadReasons"), list) else []
                item["knownBadReasons"] = list(dict.fromkeys([*known_bad, message]))
        output.append(item)
    output.sort(key=lambda item: (
        0 if (isinstance(item.get("proof"), dict) and item["proof"].get("status") == "live_proven") else 1,
        item.get("rank") if isinstance(item.get("rank"), int) else 999,
        -int(item.get("qualityScore") or 0),
    ))
    return output


def _candidate_repo(candidate: dict[str, Any]) -> str:
    install_target = candidate.get("installTarget") if isinstance(candidate.get("installTarget"), dict) else {}
    return str(
        install_target.get("repo")
        or candidate.get("resolvedArtifact")
        or candidate.get("artifact")
        or candidate.get("modelRepo")
        or ""
    )


def _candidate_needs_install_or_repair(candidate: dict[str, Any]) -> bool:
    status = candidate.get("artifactStatus") if isinstance(candidate.get("artifactStatus"), dict) else None
    if candidate.get("installed") is False:
        return True
    if status and status.get("installed") is False:
        return True
    if status and status.get("installed") and status.get("complete") is False:
        return True
    text = " ".join(
        str(item)
        for item in [
            candidate.get("skipReason"),
            (candidate.get("proof") or {}).get("message") if isinstance(candidate.get("proof"), dict) else None,
            status.get("reason") if status else None,
            *(candidate.get("requirementsMissing") or []),
        ]
        if item
    ).lower()
    return any(needle in text for needle in [
        "not installed",
        "artifact is not installed",
        "snapshot is incomplete",
        "cached artifact snapshot",
        "download is still incomplete",
        "local weight files",
        "zero bytes",
        ".safetensors",
    ])


def _candidate_has_resource_block(candidate: dict[str, Any]) -> bool:
    text = " ".join(str(item) for item in candidate.get("requirementsMissing") or []).lower()
    return any(needle in text for needle in [
        "gpu memory requires",
        "system memory requires",
        "offload disk requires",
        "requires cuda",
        "requires apple mps",
        "accelerator requires",
    ])


def _selected_install_target(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    for candidate in candidates:
        repo = _candidate_repo(candidate)
        if not repo or not _candidate_needs_install_or_repair(candidate):
            continue
        proof = candidate.get("proof") if isinstance(candidate.get("proof"), dict) else {}
        if proof.get("status") in {FAILED_HERE_PROOF_STATUS, "known_bad", "manual_only"}:
            continue
        if _candidate_has_resource_block(candidate) and not candidate.get("repairRequired"):
            continue
        install_target = candidate.get("installTarget") if isinstance(candidate.get("installTarget"), dict) else {}
        artifact_status = candidate.get("artifactStatus") if isinstance(candidate.get("artifactStatus"), dict) else {}
        repair = bool(candidate.get("repairRequired") or (artifact_status.get("installed") and artifact_status.get("complete") is False))
        return {
            "repo": repo,
            "label": install_target.get("label") or ("Repair Auto artifact" if repair else "Install Auto artifact"),
            "reason": install_target.get("reason") or artifact_status.get("reason") or candidate.get("skipReason"),
            "actionLabel": install_target.get("actionLabel") or ("Repair Auto artifact" if repair else "Install Auto artifact"),
            "repair": repair,
            "candidateId": candidate.get("id"),
        }
    return None


def _plan_health_badge(
    *,
    selected: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    selected_install_target: dict[str, Any] | None,
    readiness: str,
) -> str:
    if selected:
        return str(selected.get("healthBadge") or "Works here")
    if selected_install_target and selected_install_target.get("repair"):
        return "Repair required"
    if selected_install_target:
        return "Needs setup"
    if readiness == "manual_only":
        return "Expert only"
    if any(candidate.get("healthBadge") == "Failed here before" for candidate in candidates):
        return "Failed here before"
    if any(candidate.get("healthBadge") == "Will not work on this machine" for candidate in candidates):
        return "Will not work on this machine"
    return "Needs setup"


def build_auto_resource_plan(
    request_payload: dict[str, Any] | None,
    *,
    runtime_fingerprint: dict[str, Any] | None,
    local_models: list[dict[str, Any]] | None,
    data_dir: str | os.PathLike[str],
    history: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = request_payload if isinstance(request_payload, dict) else {}
    form = payload.get("form") if isinstance(payload.get("form"), dict) else payload
    model_type = str(form.get("modelType") or "")
    mode = str(form.get("mode") or "")
    hardware_override = payload.get("hardwareOverride") if isinstance(payload.get("hardwareOverride"), dict) else None
    hardware = hardware_override or _hardware_snapshot(runtime_fingerprint, data_dir)

    if model_type == "QwenImageModularPipeline" and mode == "text_to_image":
        candidates = _qwen_text_to_image_candidates(form, local_models, hardware)
    else:
        candidates = _declared_profile_candidates(form, local_models, hardware)

    candidates = _apply_history_to_candidates(candidates, history=history or read_auto_resource_history(data_dir), hardware=hardware)

    ready = [
        candidate for candidate in candidates
        if candidate.get("proof", {}).get("status") in READY_PROOF_STATUSES
    ]
    manual_only = [
        candidate for candidate in candidates
        if candidate.get("proof", {}).get("status") == "manual_only"
    ]

    selected = ready[0] if ready else None
    selected_install_target = _selected_install_target(candidates)
    if selected:
        status = "ready"
        status_label = "Ready with local Auto recipe"
        readiness = "ready"
        blocking_reason = None
    elif manual_only and len(manual_only) == len(candidates):
        status = "needs_setup"
        status_label = "Manual only"
        readiness = "manual_only"
        blocking_reason = manual_only[0].get("skipReason") or "This workflow does not have an Auto recipe yet."
    else:
        status = "needs_setup"
        status_label = "Needs setup"
        readiness = "needs_setup"
        skipped_reasons = [
            str(candidate.get("skipReason"))
            for candidate in candidates
            if candidate.get("skipReason")
        ]
        blocking_reason = skipped_reasons[0] if skipped_reasons else "No compatible local Auto recipe matched this hardware and model cache."
    will_not_work_reason = None
    failed_reasons = [
        str((candidate.get("proof") or {}).get("message") or candidate.get("skipReason"))
        for candidate in candidates
        if candidate.get("healthBadge") in {"Failed here before", "Will not work on this machine"}
    ]
    if failed_reasons and not selected:
        will_not_work_reason = failed_reasons[0]

    candidate_reasons = []
    requirements_missing = []
    known_bad_reasons = []
    for candidate in candidates:
        candidate_reasons.extend(candidate.get("candidateReasons") or [])
        requirements_missing.extend(candidate.get("requirementsMissing") or [])
        known_bad_reasons.extend(candidate.get("knownBadReasons") or [])

    return {
        "error": False,
        "resourceMode": "auto",
        "status": status,
        "readiness": readiness,
        "statusLabel": status_label,
        "blockingReason": blocking_reason,
        "willNotWorkReason": will_not_work_reason,
        "healthBadge": _plan_health_badge(
            selected=selected,
            candidates=candidates,
            selected_install_target=selected_install_target,
            readiness=readiness,
        ),
        "canAutoRun": bool(selected),
        "repairRequired": bool(selected_install_target and selected_install_target.get("repair")),
        "selectedInstallTarget": selected_install_target,
        "failureHistory": [candidate.get("failureHistory") for candidate in candidates if candidate.get("failureHistory")],
        "selectedCandidate": selected,
        "candidates": candidates,
        "hardware": hardware,
        "hardwareSnapshot": hardware,
        "modelRequirements": AUTO_MODEL_REQUIREMENTS,
        "requirementsMatched": selected.get("requirementsMatched") if isinstance(selected, dict) else [],
        "requirementsMissing": requirements_missing,
        "candidateReasons": list(dict.fromkeys(candidate_reasons)),
        "knownBadReasons": list(dict.fromkeys(known_bad_reasons)),
        "checkedAt": _now_ms(),
    }


def build_auto_resource_plans(
    request_payload: dict[str, Any] | None,
    *,
    runtime_fingerprint: dict[str, Any] | None,
    local_models: list[dict[str, Any]] | None,
    data_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    payload = request_payload if isinstance(request_payload, dict) else {}
    forms = payload.get("forms") if isinstance(payload.get("forms"), list) else []
    history = read_auto_resource_history(data_dir)
    plans = []
    for index, form in enumerate(forms):
        if not isinstance(form, dict):
            continue
        plan = build_auto_resource_plan(
            {"form": form, "hardwareOverride": payload.get("hardwareOverride")},
            runtime_fingerprint=runtime_fingerprint,
            local_models=local_models,
            data_dir=data_dir,
            history=history,
        )
        plan["requestIndex"] = index
        plan_key = payload.get("keys", [])[index] if isinstance(payload.get("keys"), list) and index < len(payload.get("keys", [])) else None
        if plan_key:
            plan["planKey"] = str(plan_key)
        plans.append(plan)
    return {
        "error": False,
        "resourceMode": "auto",
        "count": len(plans),
        "plans": plans,
        "checkedAt": _now_ms(),
    }
