from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from modiff.optimization_packages import qualified_auto_overrides, workload_key_for_form

from modiff.diffusers_offload_modes import (
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_NONE,
    OFFLOAD_MODE_SEQUENTIAL_CPU,
)
from modiff.diffusers_profiles import (
    ACE_STEP_REPO,
    FLUX_CANNY_REPO as FLUX_CANNY_REPO,
    FLUX_DEPTH_REPO as FLUX_DEPTH_REPO,
    FLUX_DEV_FP8_REPO as FLUX_DEV_FP8_REPO,
    FLUX_FILL_REPO,
    FLUX_KONTEXT_REPO,
    FLUX_KONTEXT_NVFP4_REPO,
    FLUX_KREA_REPO as FLUX_KREA_REPO,
    FLUX_REDUX_REPO as FLUX_REDUX_REPO,
    FLUX_SCHNELL_REPO as FLUX_SCHNELL_REPO,
    FLUX2_KLEIN_REPO,
    LTX_VIDEO_REPO,
    QWEN_IMAGE_2512_PREQUANTIZED_REPO,
    QWEN_IMAGE_2512_REPO,
    execution_profiles_for_execution,
    optional_runtime_profile_ids_for_execution,
)
from modiff.hardware import disk_snapshot, get_hardware_snapshot, system_memory_snapshot
from modiff.model_artifact_catalog import (
    AUTO_TRUST_LEVELS,
    catalog_artifact,
    catalog_model,
    community_artifact_is_discoverable,
)
from modiff.optional_runtimes import public_optional_runtime_profiles
from modiff.optional_runtime_execution import optional_runtime_requirement_for_execution
from modiff.studio_execution_specs import studio_auto_model_requirements


GIB = 1024**3
CAPACITY_CLASS_TOLERANCE = 0.01

QWEN_PRACTICAL_WIDTH = 1024
QWEN_PRACTICAL_HEIGHT = 1024
QWEN_NATIVE_WIDTH = 1328
QWEN_NATIVE_HEIGHT = 1328
QWEN_NATIVE_STEPS = 50
QWEN_NATIVE_TRUE_CFG = 4.0
QWEN_AUTO_MAX_DIMENSION = 1344
QWEN_PRACTICAL_PIXEL_BUDGET = QWEN_PRACTICAL_WIDTH * QWEN_PRACTICAL_HEIGHT
QWEN_TRANSFORMER_ONLY_COMPONENTS = ["transformer"]

Z_IMAGE_REPO = "Tongyi-MAI/Z-Image-Turbo"
QWEN_IMAGE_EDIT_REPO = "Qwen/Qwen-Image-Edit"
QWEN_IMAGE_EDIT_PREQUANTIZED_REPO = "ovedrive/qwen-image-edit-4bit"
QWEN_IMAGE_EDIT_PLUS_REPO = "Qwen/Qwen-Image-Edit-2511"
QWEN_IMAGE_LAYERED_REPO = "Qwen/Qwen-Image-Layered"
WAN_VACE_REPO = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"

READY_PROOF_STATUSES = {"passed", "declared_safe", "live_proven"}
PROVEN_PROOF_STATUSES = READY_PROOF_STATUSES
FAILED_HERE_PROOF_STATUS = "failed_here_before"
AUTO_HISTORY_VERSION = 2
AUTO_RESOURCE_SCHEMA_VERSION = 2
AUTO_HISTORY_RELATIVE_PATH = Path("auto_resource") / "history.json"

AUDIO_MODES = {"text_to_audio", "audio_variation", "audio_continuation", "audio_repaint"}
VIDEO_MODES = {
    "text_to_video",
    "image_to_video",
    "video_to_video",
    "reference_to_video",
    "control_to_video",
    "video_color_edit",
}
CPU_OR_DISK_OFFLOAD_MODES = {
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_SEQUENTIAL_CPU,
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
}

MODELISH_EXTENSIONS = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx"}
CONFIG_JSON_NAMES = {"model_index.json", "config.json", "model_config.json", "scheduler_config.json"}
HIGH_MEMORY_FULL_RESIDENCY = {
    "accelerator": "cuda",
    "vramBytes": 80 * GIB,
    "systemRamBytes": 64 * GIB,
}

AUTO_MODEL_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "ZImageModularPipeline": {
        "supportedTasks": ["text_to_image"],
        "defaultRepo": Z_IMAGE_REPO,
        "executionPath": "direct-diffusers-image",
        "qualityDefaults": {"width": 1024, "height": 1024, "steps": 8, "guidanceScale": 1},
        "minimum": {"accelerator": "gpu_or_cpu", "vramBytes": 0, "systemRamBytes": 8 * GIB},
        "recommended": {"accelerator": "gpu", "vramBytes": 8 * GIB, "systemRamBytes": 16 * GIB},
        "fullResidency": HIGH_MEMORY_FULL_RESIDENCY,
        "supportedOffloadModes": [OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_DISK],
    },
    "QwenImageModularPipeline:text_to_image": {
        "supportedTasks": ["text_to_image"],
        "defaultRepo": QWEN_IMAGE_2512_REPO,
        "preferredLowerMemoryRepo": QWEN_IMAGE_2512_PREQUANTIZED_REPO,
        "executionPath": "direct-diffusers-image",
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
        "supportedOffloadModes": [
            OFFLOAD_MODE_NONE,
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_SEQUENTIAL_CPU,
            OFFLOAD_MODE_GROUP_DISK,
        ],
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
        # Qwen Image BF16 occupies about 54 GiB on disk and the current Union
        # ControlNet adds about 3.4 GiB. The earlier 109 GiB pressure result was
        # produced by a graph-bridge defect that loaded a second full Qwen
        # pipeline in the ControlNet slot; it is not a valid residency
        # measurement. Leave conservative activation headroom while allowing
        # the qualified 97.7 GiB high-memory tier to avoid CPU offload.
        "fullResidency": {"accelerator": "cuda", "vramBytes": 80 * GIB, "systemRamBytes": 64 * GIB},
        "onLoadQuantization": {
            "accelerator": "cuda",
            "vramBytes": 16 * GIB,
            "systemRamBytes": 32 * GIB,
            "diskFreeBytes": 20 * GIB,
            "quantizationMode": "bnb_4bit",
            "quantizedComponents": ["transformer", "text_encoder"],
        },
        "supportedOffloadModes": [OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK],
        "requiredPackages": ["diffusers", "transformers", "accelerate", "torch", "bitsandbytes"],
        "guardedReason": "Qwen Control image is broad Auto coverage: MoDiff will try the generic modular Diffusers graph with quantization/offload, then remember any failure on this machine.",
    },
    "QwenImageEditModularPipeline": {
        "supportedTasks": ["edit_image", "inpaint", "outpaint"],
        "defaultRepo": QWEN_IMAGE_EDIT_REPO,
        "preferredLowerMemoryRepo": QWEN_IMAGE_EDIT_PREQUANTIZED_REPO,
        "executionPath": "direct-diffusers-image",
        "qualityDefaults": {"width": 1024, "height": 1024, "steps": 40, "guidanceScale": 4},
        "minimum": {"accelerator": "cuda", "vramBytes": 24 * GIB, "systemRamBytes": 40 * GIB},
        "recommended": {"accelerator": "cuda", "vramBytes": 32 * GIB, "systemRamBytes": 48 * GIB},
        "fullResidency": {"accelerator": "cuda", "vramBytes": 64 * GIB, "systemRamBytes": 64 * GIB},
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
        "supportedOffloadModes": [
            OFFLOAD_MODE_NONE,
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_SEQUENTIAL_CPU,
            OFFLOAD_MODE_GROUP_DISK,
        ],
        "requiredPackages": ["diffusers", "transformers", "accelerate", "torch", "bitsandbytes"],
        "guardedReason": "Prefer the Apache-2.0 Diffusers-compatible prequantized Qwen Image Edit artifact on nominal 16 GiB CUDA systems before attempting official BF16 disk offload.",
    },
    "QwenImageEditPlusModularPipeline": {
        "supportedTasks": ["edit_image", "multi_image_reference_edit"],
        "defaultRepo": QWEN_IMAGE_EDIT_PLUS_REPO,
        "executionPath": "modular-diffusers",
        "pipelineClass": "QwenImageEditPlusModularPipeline",
        "qualityDefaults": {"width": 768, "height": 768, "steps": 24, "guidanceScale": 4, "maxSequenceLength": 512},
        "minimum": {"accelerator": "cuda", "vramBytes": 24 * GIB, "systemRamBytes": 48 * GIB, "diskFreeBytes": 35 * GIB},
        "recommended": {"accelerator": "cuda", "vramBytes": 32 * GIB, "systemRamBytes": 64 * GIB, "diskFreeBytes": 45 * GIB},
        "fullResidency": {"accelerator": "cuda", "vramBytes": 64 * GIB, "systemRamBytes": 64 * GIB},
        "onLoadQuantization": {
            "accelerator": "cuda",
            "vramBytes": 16 * GIB,
            "systemRamBytes": 32 * GIB,
            "diskFreeBytes": 35 * GIB,
            "quantizationMode": "bnb_4bit",
            "quantizedComponents": ["transformer", "text_encoder"],
        },
        "supportedOffloadModes": [OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK],
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
        "fullResidency": {"accelerator": "cuda", "vramBytes": 64 * GIB, "systemRamBytes": 64 * GIB},
        "onLoadQuantization": {
            "accelerator": "cuda",
            "vramBytes": 16 * GIB,
            "systemRamBytes": 32 * GIB,
            "diskFreeBytes": 35 * GIB,
            "quantizationMode": "bnb_4bit",
            "quantizedComponents": ["transformer", "text_encoder"],
        },
        "supportedOffloadModes": [
            OFFLOAD_MODE_NONE,
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_GROUP_CPU,
            OFFLOAD_MODE_GROUP_DISK,
        ],
        "requiredPackages": ["diffusers", "transformers", "accelerate", "torch", "bitsandbytes"],
        "guardedReason": "Qwen Layered is guarded Auto coverage. MoDiff can try a low-memory modular recipe and remember failures.",
    },
    "WanVACEPipeline": {
        "supportedTasks": [
            "text_to_video",
            "video_inpaint",
            "video_outpaint",
            "control_to_video",
        ],
        "defaultRepo": WAN_VACE_REPO,
        "executionPath": "direct-wan-vace",
        "qualityDefaults": {"width": 832, "height": 480, "steps": 24, "guidanceScale": 5, "numFrames": 49},
        "minimum": {"accelerator": "cuda", "vramBytes": 10 * GIB, "systemRamBytes": 24 * GIB},
        "recommended": {"accelerator": "cuda", "vramBytes": 12 * GIB, "systemRamBytes": 32 * GIB},
        "highQuality": {"accelerator": "cuda", "vramBytes": 24 * GIB, "systemRamBytes": 48 * GIB},
        "supportedOffloadModes": [
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_SEQUENTIAL_CPU,
            OFFLOAD_MODE_GROUP_DISK,
            OFFLOAD_MODE_NONE,
        ],
    },
    "WanVideoPipeline": {
        "supportedTasks": ["text_to_video", "video_to_video", "video_color_edit"],
        "defaultRepo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "executionPath": "direct-diffusers-video",
        "pipelineClass": "WanVideoToVideoPipeline",
        "qualityDefaults": {"width": 832, "height": 480, "steps": 30, "guidanceScale": 5, "numFrames": 49},
        "minimum": {"accelerator": "cuda", "vramBytes": 10 * GIB, "systemRamBytes": 24 * GIB},
        "recommended": {"accelerator": "cuda", "vramBytes": 12 * GIB, "systemRamBytes": 32 * GIB},
        "highQuality": {"accelerator": "cuda", "vramBytes": 24 * GIB, "systemRamBytes": 48 * GIB},
        "supportedOffloadModes": [
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_SEQUENTIAL_CPU,
            OFFLOAD_MODE_GROUP_DISK,
            OFFLOAD_MODE_NONE,
        ],
    },
    "WanVideoPipeline:text_to_video": {
        "supportedTasks": ["text_to_video"],
        "defaultRepo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "executionPath": "direct-diffusers-video",
        "pipelineClass": "WanPipeline",
        "qualityDefaults": {"width": 832, "height": 480, "steps": 30, "guidanceScale": 5, "numFrames": 81},
        "minimum": {"accelerator": "cuda", "vramBytes": 10 * GIB, "systemRamBytes": 24 * GIB},
        "recommended": {"accelerator": "cuda", "vramBytes": 12 * GIB, "systemRamBytes": 32 * GIB},
        "highQuality": {"accelerator": "cuda", "vramBytes": 24 * GIB, "systemRamBytes": 48 * GIB},
        "supportedOffloadModes": [
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_SEQUENTIAL_CPU,
            OFFLOAD_MODE_GROUP_DISK,
            OFFLOAD_MODE_NONE,
        ],
    },
    "LTXVideoPipeline": {
        "supportedTasks": ["text_to_video", "image_to_video", "video_to_video", "reference_to_video"],
        "defaultRepo": LTX_VIDEO_REPO,
        "executionPath": "direct-diffusers-video",
        "pipelineClass": "LTXConditionPipeline",
        "qualityDefaults": {"width": 704, "height": 480, "steps": 8, "guidanceScale": 1, "numFrames": 81},
        "minimum": {"accelerator": "cuda", "vramBytes": 24 * GIB, "systemRamBytes": 48 * GIB, "diskFreeBytes": 50 * GIB},
        "recommended": {"accelerator": "cuda", "vramBytes": 32 * GIB, "systemRamBytes": 64 * GIB, "diskFreeBytes": 70 * GIB},
        "fullResidency": HIGH_MEMORY_FULL_RESIDENCY,
        "supportedOffloadModes": [
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_SEQUENTIAL_CPU,
            OFFLOAD_MODE_GROUP_CPU,
            OFFLOAD_MODE_GROUP_DISK,
            OFFLOAD_MODE_NONE,
        ],
        "requiredPackages": ["diffusers", "transformers", "accelerate", "torch"],
    },
    "AceStepAudioPipeline": {
        "supportedTasks": ["text_to_audio", "audio_variation", "audio_continuation", "audio_repaint"],
        "defaultRepo": ACE_STEP_REPO,
        "executionPath": "direct-diffusers-audio",
        "pipelineClass": "AceStepPipeline",
        "qualityDefaults": {"audioDuration": 30, "steps": 8, "guidanceScale": 1, "shift": 3},
        "minimum": {"accelerator": "cuda", "vramBytes": 10 * GIB, "systemRamBytes": 24 * GIB, "diskFreeBytes": 20 * GIB},
        "recommended": {"accelerator": "cuda", "vramBytes": 16 * GIB, "systemRamBytes": 32 * GIB, "diskFreeBytes": 30 * GIB},
        # The qualified ACE-Step run peaks below 14 GiB including allocator
        # reserve. A 16 GiB CUDA card can therefore load the complete pipeline
        # directly; forcing CPU offload here made cold loading dramatically
        # slower and did not reduce generation time.
        "fullResidency": {
            "accelerator": "cuda",
            "vramBytes": 16 * GIB,
            "systemRamBytes": 32 * GIB,
        },
        "coldLoadTarget": {
            "deviceName": "NVIDIA GeForce RTX 4080",
            "maxSeconds": 120,
            "recipe": {
                "dtype": "bfloat16",
                "offloadMode": OFFLOAD_MODE_NONE,
                "deviceMap": "cuda",
            },
        },
        "supportedOffloadModes": [
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_SEQUENTIAL_CPU,
            OFFLOAD_MODE_GROUP_DISK,
            OFFLOAD_MODE_NONE,
        ],
        "requiredPackages": ["diffusers", "transformers", "accelerate", "torch", "scipy"],
    },
    "Flux2KleinPipeline": {
        "supportedTasks": ["text_to_image", "edit_image", "multi_image_reference_edit"],
        "defaultRepo": FLUX2_KLEIN_REPO,
        "executionPath": "direct-diffusers-image",
        "pipelineClass": "Flux2KleinPipeline",
        "qualityDefaults": {"width": 1024, "height": 1024, "steps": 4, "guidanceScale": 1, "maxSequenceLength": 512},
        "minimum": {"accelerator": "cuda", "vramBytes": 13 * GIB, "systemRamBytes": 24 * GIB, "diskFreeBytes": 25 * GIB},
        "recommended": {"accelerator": "cuda", "vramBytes": 20 * GIB, "systemRamBytes": 32 * GIB, "diskFreeBytes": 35 * GIB},
        "fullResidency": HIGH_MEMORY_FULL_RESIDENCY,
        "supportedOffloadModes": [OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK],
        "requiredPackages": ["diffusers", "transformers", "accelerate", "torch"],
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
        "fullResidency": HIGH_MEMORY_FULL_RESIDENCY,
        "lowerMemory": {
            "accelerator": "cuda",
            "vramBytes": 16 * GIB,
            "systemRamBytes": 32 * GIB,
            "diskFreeBytes": 45 * GIB,
            "quantizationMode": "torchao_float8",
            "quantizedComponents": ["transformer", "text_encoder_2"],
        },
        "supportedOffloadModes": [
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_SEQUENTIAL_CPU,
            OFFLOAD_MODE_GROUP_DISK,
            OFFLOAD_MODE_NONE,
        ],
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
        "fullResidency": HIGH_MEMORY_FULL_RESIDENCY,
        "onLoadQuantization": {
            "accelerator": "cuda",
            "vramBytes": 16 * GIB,
            "systemRamBytes": 32 * GIB,
            "diskFreeBytes": 45 * GIB,
            "quantizationMode": "quanto_float8",
            "quantizedComponents": ["transformer", "text_encoder_2"],
        },
        "supportedOffloadModes": [
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_SEQUENTIAL_CPU,
            OFFLOAD_MODE_GROUP_DISK,
            OFFLOAD_MODE_NONE,
        ],
        "requiredPackages": ["diffusers", "transformers", "accelerate", "torch", "optimum-quanto"],
        "guardedReason": "FLUX Fill has guarded Auto coverage through generic Diffusers inpaint/outpaint nodes and on-load quantization.",
    },
}

AUTO_MODEL_REQUIREMENTS.update(studio_auto_model_requirements())


def _auto_requirements_for_pair(model_type: str, mode: str) -> dict[str, Any] | None:
    """Return the exact effective Auto specification for one declared pair.

    Resource requirements may be shared by several modes, but their loader
    target is never inferred from a pipeline-class name or a stale generic
    execution-path hint.  One unique execution profile owns the effective
    module, action, execution path, and pipeline class.
    """

    normalized_model = str(model_type or "").strip()
    normalized_mode = str(mode or "").strip()
    if not normalized_model or not normalized_mode:
        return None

    exact_key = f"{normalized_model}:{normalized_mode}"
    if exact_key in AUTO_MODEL_REQUIREMENTS:
        requirements = AUTO_MODEL_REQUIREMENTS[exact_key]
    else:
        requirements = AUTO_MODEL_REQUIREMENTS.get(normalized_model)
    if not isinstance(requirements, dict):
        return None

    supported_tasks = {
        str(task).strip()
        for task in requirements.get("supportedTasks") or []
        if str(task).strip()
    }
    if normalized_mode not in supported_tasks:
        return None

    profiles = execution_profiles_for_execution(normalized_model, normalized_mode)
    if len(profiles) != 1:
        return None
    profile = profiles[0]
    effective = {
        **requirements,
        "supportedTasks": [normalized_mode],
        "executionProfileId": profile.id,
        "loaderModule": profile.loader_module,
        "loaderAction": profile.loader_action,
        "executionPath": profile.execution_path,
        "pipelineClass": profile.pipeline_class,
        "defaultRepo": profile.default_repo,
        "fallbackRepo": profile.fallback_repo,
        "compatibleRepos": list(profile.compatible_repos),
    }
    allowed_lower_memory_repos = {
        repo
        for repo in (profile.fallback_repo, *profile.compatible_repos)
        if isinstance(repo, str) and repo
    }
    if effective.get("preferredLowerMemoryRepo") not in allowed_lower_memory_repos:
        effective.pop("preferredLowerMemoryRepo", None)
    return effective


def auto_resource_pair_is_declared(model_type: str, mode: str) -> bool:
    """Return whether both Auto requirements and an execution profile declare a pair."""

    return _auto_requirements_for_pair(model_type, mode) is not None


def _declared_auto_modes(model_type: str) -> list[str]:
    normalized_model = str(model_type or "").strip()
    modes = set()
    for key, requirements in AUTO_MODEL_REQUIREMENTS.items():
        if key != normalized_model and not key.startswith(f"{normalized_model}:"):
            continue
        modes.update(
            str(task).strip()
            for task in requirements.get("supportedTasks") or []
            if str(task).strip()
        )
    return sorted(mode for mode in modes if _auto_requirements_for_pair(normalized_model, mode) is not None)


def _public_auto_model_requirements() -> dict[str, dict[str, Any]]:
    """Publish only exact pair specifications with one canonical loader target."""

    specifications: dict[str, dict[str, Any]] = {}
    model_types = {
        str(key).split(":", 1)[0]
        for key in AUTO_MODEL_REQUIREMENTS
        if str(key).split(":", 1)[0]
    }
    for model_type in sorted(model_types):
        for mode in _declared_auto_modes(model_type):
            specification = _auto_requirements_for_pair(model_type, mode)
            if specification is not None:
                specifications[f"{model_type}:{mode}"] = specification
    return specifications


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

    # Auxiliary app-managed artifacts (for example Spandrel upscalers, LoRAs,
    # GGUF weights, and ONNX models) are valid runnable snapshots even though
    # they do not use Diffusers' usual .safetensors/.bin naming. Keep this in
    # sync with the model-ish extensions accepted by the download pipeline.
    has_weight_file = any(
        candidate.is_file() and candidate.suffix.lower() in MODELISH_EXTENSIONS
        for candidate in snapshot_dir.rglob("*")
    )
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


def artifact_cache_status(repo_id: str, local_models: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Return the runnable installation state for one cached Hub repository.

    Hugging Face exposes a revision as soon as its snapshot directory exists,
    which can be well before large blobs finish downloading. Public inventory
    endpoints must use the same shard/plan validation as Auto readiness instead
    of treating every scanned revision as installed.
    """
    return _artifact_cache_status(repo_id, local_models)


def _runtime_key(runtime_fingerprint: dict[str, Any] | None) -> str:
    if not isinstance(runtime_fingerprint, dict):
        return "unknown-runtime"
    return str(
        runtime_fingerprint.get("resourceFingerprint")
        or runtime_fingerprint.get("fingerprint")
        or "unknown-runtime"
    )


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
    resolution = candidate.get("artifactResolution") if isinstance(candidate.get("artifactResolution"), dict) else {}
    resolved = resolution.get("resolved") if isinstance(resolution.get("resolved"), dict) else {}
    workload = _candidate_workload_signature(candidate)
    return {
        "hardwareFingerprint": _hardware_history_key(runtime_fingerprint, hardware),
        "modelType": str(candidate.get("modelType") or ""),
        "mode": str(candidate.get("mode") or ""),
        "artifact": str(candidate.get("resolvedArtifact") or candidate.get("artifact") or candidate.get("modelRepo") or ""),
        "artifactRevision": str(resolved.get("revision") or candidate.get("artifactRevision") or "unversioned"),
        "dtype": str(candidate.get("dtype") or ""),
        "quantizationMode": str(candidate.get("quantizationMode") or "none"),
        "quantizedComponents": [str(item) for item in candidate.get("quantizedComponents") or []],
        "offloadMode": str(candidate.get("offloadMode") or ""),
        "device": str(candidate.get("device") or ""),
        "deviceMap": str(candidate.get("deviceMap") or ""),
        "attentionBackend": str(candidate.get("attentionBackend") or "auto"),
        "regionalCompile": bool(candidate.get("regionalCompile")),
        "denoiserCache": str(candidate.get("denoiserCache") or "none"),
        "channelsLast": bool(candidate.get("channelsLast")),
        "layerwiseCasting": bool(candidate.get("layerwiseCasting")),
        "pipelineClass": str(candidate.get("pipelineClass") or ""),
        "loaderModule": str(candidate.get("loaderModule") or ""),
        "loaderAction": str(candidate.get("loaderAction") or ""),
        "executionPath": str(candidate.get("executionPath") or ""),
        "workload": workload,
    }


def _candidate_workload_signature(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return only workload fields that affect this media kind's resource proof."""

    generation = candidate.get("generation") if isinstance(candidate.get("generation"), dict) else {}
    mode = str(candidate.get("mode") or "")
    model_type = str(candidate.get("modelType") or "")
    if mode in AUDIO_MODES or "audio" in model_type.lower():
        keys = ("audioDuration", "extensionDuration", "batchSize", "steps")
    elif mode in VIDEO_MODES or "video" in model_type.lower() or model_type.startswith("Wan"):
        keys = ("width", "height", "numFrames", "batchSize", "steps")
    else:
        keys = ("width", "height", "batchSize", "steps")
    return {key: generation.get(key) for key in keys if generation.get(key) is not None}


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

    # SHA-1 is retained only as the stable legacy cache-key shape. It is not a
    # signature or integrity boundary.
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()


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
        "device": runtime_hints.get("device"),
        "deviceMap": runtime_hints.get("deviceMap"),
        "attentionBackend": runtime_hints.get("attentionBackend"),
        "regionalCompile": runtime_hints.get("regionalCompile"),
        "denoiserCache": runtime_hints.get("denoiserCache"),
        "channelsLast": runtime_hints.get("channelsLast"),
        "layerwiseCasting": runtime_hints.get("layerwiseCasting"),
        "pipelineClass": runtime_hints.get("pipelineClass"),
        "loaderModule": runtime_hints.get("loaderModule"),
        "loaderAction": runtime_hints.get("loaderAction"),
        "executionPath": runtime_hints.get("executionPath"),
        "generation": runtime_hints.get("generation") if isinstance(runtime_hints.get("generation"), dict) else {},
        "artifactResolution": runtime_hints.get("artifactResolution") if isinstance(runtime_hints.get("artifactResolution"), dict) else {},
    }


def _history_candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    resolution = candidate.get("artifactResolution") if isinstance(candidate.get("artifactResolution"), dict) else {}
    resolved = resolution.get("resolved") if isinstance(resolution.get("resolved"), dict) else {}
    return {
        "id": candidate.get("id"),
        "modelType": candidate.get("modelType"),
        "mode": candidate.get("mode"),
        "artifact": candidate.get("resolvedArtifact") or candidate.get("artifact") or candidate.get("modelRepo"),
        "artifactRevision": resolved.get("revision") or candidate.get("artifactRevision"),
        "dtype": candidate.get("dtype"),
        "quantizationMode": candidate.get("quantizationMode"),
        "quantizedComponents": candidate.get("quantizedComponents") if isinstance(candidate.get("quantizedComponents"), list) else [],
        "offloadMode": candidate.get("offloadMode"),
        "device": candidate.get("device"),
        "deviceMap": candidate.get("deviceMap"),
        "attentionBackend": candidate.get("attentionBackend") or "auto",
        "regionalCompile": bool(candidate.get("regionalCompile")),
        "denoiserCache": candidate.get("denoiserCache") or "none",
        "channelsLast": bool(candidate.get("channelsLast")),
        "layerwiseCasting": bool(candidate.get("layerwiseCasting")),
        "pipelineClass": candidate.get("pipelineClass"),
        "loaderModule": candidate.get("loaderModule"),
        "loaderAction": candidate.get("loaderAction"),
        "executionPath": candidate.get("executionPath"),
        "generation": candidate.get("generation") if isinstance(candidate.get("generation"), dict) else {},
    }


def _normalized_measurement(measurement: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(measurement, dict):
        return None
    normalized = {}
    for key in (
        "elapsedSeconds",
        "peakAllocatedBytes",
        "peakReservedBytes",
        "allocatedBytes",
        "reservedBytes",
        "driverAllocatedBytes",
        "processRssBytes",
    ):
        value = measurement.get(key)
        if value is None:
            continue
        try:
            normalized[key] = float(value) if key == "elapsedSeconds" else int(value)
        except (TypeError, ValueError):
            continue
    for key in ("backend", "device"):
        value = measurement.get(key)
        if value not in (None, ""):
            normalized[key] = str(value)
    return normalized or None


def record_auto_resource_success(
    data_dir: str | os.PathLike[str],
    *,
    runtime_fingerprint: dict[str, Any] | None,
    runtime_hints: dict[str, Any] | None,
    measurement: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    candidate = _runtime_candidate_from_hints(runtime_hints)
    if not candidate or _auto_requirements_for_pair(
        str(candidate.get("modelType") or ""),
        str(candidate.get("mode") or ""),
    ) is None:
        return None
    history = read_auto_resource_history(data_dir)
    key = auto_resource_history_key(candidate, runtime_fingerprint=runtime_fingerprint)
    entries = history.setdefault("entries", {})
    entry = entries.get(key) if isinstance(entries.get(key), dict) else {}
    now = _now_ms()
    normalized_measurement = _normalized_measurement(measurement)
    entry.update({
        "key": key,
        "signature": _candidate_history_signature(candidate, runtime_fingerprint=runtime_fingerprint),
        "candidate": _history_candidate_summary(candidate),
        "successCount": int(entry.get("successCount") or 0) + 1,
        "lastSuccessAt": now,
        "lastStatus": "live_proven",
    })
    if normalized_measurement:
        entry["lastMeasurement"] = normalized_measurement
        elapsed = normalized_measurement.get("elapsedSeconds")
        previous_best = entry.get("bestElapsedSeconds")
        if elapsed is not None and (previous_best is None or elapsed < float(previous_best)):
            entry["bestElapsedSeconds"] = elapsed
        peak = normalized_measurement.get("peakAllocatedBytes")
        previous_peak = entry.get("maxObservedPeakAllocatedBytes")
        if peak is not None and (previous_peak is None or peak > int(previous_peak)):
            entry["maxObservedPeakAllocatedBytes"] = peak
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


def _mps_accelerator(name: Any = None, device: dict[str, Any] | None = None) -> dict[str, Any]:
    device = device if isinstance(device, dict) else {}
    total = _safe_int(device.get("planning_memory_total") or device.get("vram_total"))
    free = _safe_int(device.get("planning_memory_free") or device.get("vram_free"))
    return {
        "kind": "mps",
        "backend": "mps",
        "vendor": "apple",
        "name": str(name or "Apple Metal Performance Shaders"),
        "architecture": device.get("architecture"),
        "memoryKind": device.get("memory_kind") or "unified",
        "totalBytes": total,
        "freeBytes": free,
        "accessibleTotalBytes": _safe_int(device.get("torch_vram_total")) or total,
        "dedicatedTotalBytes": _safe_int(device.get("dedicated_memory_total")),
        "sharedTotalBytes": _safe_int(device.get("shared_memory_total")),
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
        for kind in ("cuda", "xpu", "mps", "cpu")
    }
    device = device_by_kind["cuda"]
    if device is not None:
        total = _safe_int(device.get("planning_memory_total") or device.get("vram_total"))
        if total is None:
            total = _safe_int(device.get("torch_vram_total"))
        free = _safe_int(device.get("planning_memory_free") or device.get("vram_free"))
        if free is None:
            free = _safe_int(device.get("torch_vram_free"))
        return {
            "kind": "cuda",
            "backend": device.get("backend") or "cuda",
            "vendor": device.get("vendor") or ("amd" if device.get("backend") == "rocm" else "nvidia"),
            "name": device.get("name"),
            "architecture": device.get("architecture"),
            "memoryKind": device.get("memory_kind") or "dedicated",
            "totalBytes": total,
            "freeBytes": free,
            "accessibleTotalBytes": _safe_int(device.get("torch_vram_total")) or total,
            "dedicatedTotalBytes": _safe_int(device.get("dedicated_memory_total")) or total,
            "sharedTotalBytes": _safe_int(device.get("shared_memory_total")),
            "capability": device.get("compute_capability") or device.get("capability"),
            "band": "shared_memory" if device.get("memory_kind") in {"shared", "unified"} else _vram_band(total),
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

    device = device_by_kind["xpu"]
    if device is not None:
        total = _safe_int(device.get("planning_memory_total") or device.get("vram_total") or device.get("torch_vram_total"))
        free = _safe_int(device.get("planning_memory_free") or device.get("vram_free") or device.get("torch_vram_free"))
        return {
            "kind": "xpu",
            "backend": "xpu",
            "vendor": "intel",
            "name": device.get("name") or "Intel XPU",
            "architecture": device.get("architecture"),
            "memoryKind": device.get("memory_kind") or "dedicated",
            "totalBytes": total,
            "freeBytes": free,
            "accessibleTotalBytes": _safe_int(device.get("torch_vram_total")) or total,
            "dedicatedTotalBytes": _safe_int(device.get("dedicated_memory_total")),
            "sharedTotalBytes": _safe_int(device.get("shared_memory_total")),
            "capability": device.get("capability"),
            "band": "shared_memory" if device.get("memory_kind") in {"shared", "unified"} else _vram_band(total),
        }
    if bool(torch_state.get("xpu_available")):
        return {
            "kind": "xpu",
            "name": "Intel XPU",
            "totalBytes": None,
            "freeBytes": None,
            "capability": None,
            "band": "unknown",
        }

    device = device_by_kind["mps"]
    if device is not None:
        return _mps_accelerator(device.get("name"), device)
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
    if bool(torch_state.get("xpu_available")):
        xpu_devices = torch_state.get("xpu_devices")
        first_xpu = xpu_devices[0] if isinstance(xpu_devices, list) and xpu_devices and isinstance(xpu_devices[0], dict) else {}
        return {
            "kind": "xpu",
            "name": str(first_xpu.get("name") or "Intel XPU"),
            "totalBytes": _safe_int(first_xpu.get("total_memory")),
            "freeBytes": _safe_int(first_xpu.get("memory_free_bytes")),
            "capability": None,
            "band": _vram_band(_safe_int(first_xpu.get("total_memory"))),
        }
    if "cuda_available" in torch_state or "xpu_available" in torch_state or "mps_available" in torch_state:
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
    system = normalized_hardware.get("system") if isinstance(normalized_hardware, dict) else None
    system = system if isinstance(system, dict) else {}
    return {
        "runtimeFingerprint": _runtime_key(runtime_fingerprint),
        "runtime": runtime_fingerprint,
        "platform": _normalized_platform_name(system.get("platform") or system.get("os") or system.get("os_name")),
        "architecture": str(system.get("architecture") or "unknown").lower(),
        "accelerator": _accelerator_snapshot(runtime_fingerprint, normalized_hardware),
        "systemMemory": _system_memory_snapshot(normalized_hardware),
        "offloadDisk": _disk_snapshot(data_dir, normalized_hardware),
    }


def _normalized_platform_name(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized.startswith("win") or normalized == "nt":
        return "windows"
    if normalized.startswith("darwin") or normalized.startswith("mac"):
        return "macos"
    if normalized.startswith("linux") or normalized == "posix":
        return "linux"
    return normalized or "unknown"


def _hardware_identity(hardware: dict[str, Any]) -> tuple[str, str, str, float | None]:
    runtime = hardware.get("runtime") if isinstance(hardware.get("runtime"), dict) else {}
    runtime_hardware = runtime.get("hardware") if isinstance(runtime.get("hardware"), dict) else {}
    system = runtime_hardware.get("system") if isinstance(runtime_hardware.get("system"), dict) else {}
    platform_name = _normalized_platform_name(
        hardware.get("platform") or hardware.get("os") or system.get("platform") or system.get("os") or system.get("os_name")
    )
    architecture = str(hardware.get("architecture") or system.get("architecture") or "unknown").strip().lower()
    accelerator = hardware.get("accelerator") if isinstance(hardware.get("accelerator"), dict) else {}
    backend = str(hardware.get("backend") or accelerator.get("backend") or accelerator.get("kind") or "cpu").lower()
    if backend == "cuda" and str(hardware.get("runtimeBackend") or "").lower() == "rocm":
        backend = "rocm"
    capability_value = accelerator.get("capability")
    try:
        if isinstance(capability_value, (list, tuple)) and len(capability_value) >= 2:
            capability = float(f"{int(capability_value[0])}.{int(capability_value[1])}")
        else:
            capability = float(str(capability_value)) if capability_value is not None else None
    except (TypeError, ValueError):
        capability = None
    return platform_name, architecture, backend, capability


def _apply_catalog_hardware_support(
    candidates: list[dict[str, Any]], hardware: dict[str, Any]
) -> list[dict[str, Any]]:
    platform_name, architecture, backend, capability = _hardware_identity(hardware)
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        item = _clone_candidate(candidate)
        if item.get("exactPairDeclared") is False:
            output.append(item)
            continue
        artifact = catalog_artifact(str(item.get("modelType") or ""), str(item.get("artifact") or ""))
        missing: list[str] = []
        if artifact:
            supported_platforms = {str(value).lower() for value in artifact.get("supportedPlatforms") or []}
            supported_architectures = {str(value).lower() for value in artifact.get("supportedArchitectures") or []}
            supported_backends = {str(value).lower() for value in artifact.get("supportedBackends") or []}
            if str(artifact.get("trust") or "community") in AUTO_TRUST_LEVELS and not artifact.get("revision"):
                missing.append("Qualified Auto artifact is missing an immutable catalog revision")
            if platform_name != "unknown" and supported_platforms and platform_name not in supported_platforms:
                missing.append(f"Artifact does not support {platform_name}")
            if architecture != "unknown" and supported_architectures and architecture not in supported_architectures:
                missing.append(f"Artifact does not support {architecture}")
            if backend and supported_backends and backend not in supported_backends:
                missing.append(f"Artifact does not support the {backend} backend")
            if str(artifact.get("format") or "").lower() in {"nvfp4", "mxfp8"} and (
                backend != "cuda" or capability is None or capability < 10.0
            ):
                missing.append("This artifact requires an NVIDIA Blackwell GPU (compute capability 10.0 or newer)")
        if missing:
            all_missing = list(dict.fromkeys([*(item.get("requirementsMissing") or []), *missing]))
            message = "; ".join(missing)
            item["requirementsMissing"] = all_missing
            item["requirementsMatched"] = []
            item["canAutoRun"] = False
            item["readiness"] = "known_bad"
            item["skipReason"] = message
            item["healthBadge"] = "Not suitable locally"
            proof = item.get("proof") if isinstance(item.get("proof"), dict) else {}
            item["proof"] = {**proof, "status": "known_bad", "source": "model_artifact_catalog", "message": message}
            evidence = item.get("compatibilityEvidence") if isinstance(item.get("compatibilityEvidence"), dict) else {}
            item["compatibilityEvidence"] = {
                **evidence,
                "level": "blocked",
                "label": "Not suitable locally",
                "source": "model_artifact_catalog",
                "message": message,
            }
        output.append(item)
    return output


def _apply_community_confirmation(
    candidates: list[dict[str, Any]], form: dict[str, Any]
) -> list[dict[str, Any]]:
    confirmed = str(form.get("confirmedCommunityArtifact") or "").strip().lower()
    if not confirmed:
        return candidates
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        item = _clone_candidate(candidate)
        if item.get("exactPairDeclared") is False:
            output.append(item)
            continue
        repo = str(item.get("resolvedArtifact") or item.get("artifact") or "").strip().lower()
        if (
            repo == confirmed
            and item.get("requiresConfirmation")
            and item.get("profileArtifactCompatible") is not False
            and item.get("installed")
            and not item.get("requirementsMissing")
        ):
            message = "Community artifact explicitly confirmed for this workflow and machine."
            proof = item.get("proof") if isinstance(item.get("proof"), dict) else {}
            item["proof"] = {
                **proof,
                "status": "declared_safe",
                "source": "user_community_confirmation",
                "message": message,
                "checkedAt": _now_ms(),
            }
            item["canAutoRun"] = True
            item["readiness"] = "ready"
            item["skipReason"] = None
            item["healthBadge"] = "Community option"
            evidence = item.get("compatibilityEvidence") if isinstance(item.get("compatibilityEvidence"), dict) else {}
            item["compatibilityEvidence"] = {
                **evidence,
                "level": "community",
                "label": "Community option",
                "source": "user_community_confirmation",
                "message": message,
                "checkedAt": _now_ms(),
            }
        output.append(item)
    return output


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
    offload_mode: str = OFFLOAD_MODE_NONE,
) -> list[str]:
    missing = []
    accelerator_state = hardware.get("accelerator") if isinstance(hardware.get("accelerator"), dict) else {}
    actual_accelerator = accelerator_state.get("kind")
    if accelerator == "cuda" and actual_accelerator != "cuda":
        missing.append("CUDA accelerator required")
    elif accelerator == "cuda_or_mps" and actual_accelerator not in {"cuda", "mps"}:
        missing.append("CUDA or MPS accelerator required")
    elif accelerator in {"cuda_or_mps_or_xpu", "gpu"} and actual_accelerator not in {"cuda", "mps", "xpu"}:
        missing.append("CUDA, Apple MPS, or Intel XPU accelerator required")
    elif accelerator in {"cuda_or_mps_or_xpu_or_cpu", "gpu_or_cpu"} and actual_accelerator not in {
        "cuda", "mps", "xpu", "cpu"
    }:
        missing.append("Supported accelerator or CPU required")

    total_vram = _safe_int(accelerator_state.get("totalBytes"))
    memory_kind = str(accelerator_state.get("memoryKind") or "dedicated").lower()
    if memory_kind in {"shared", "unified"} and (
        offload_mode in CPU_OR_DISK_OFFLOAD_MODES or actual_accelerator in {"mps", "xpu"}
    ):
        # Integrated and unified-memory accelerators do not have a discrete
        # VRAM pool. For offloaded recipes, admission is based on the memory
        # the runtime can actually address, while full-residency ranking keeps
        # using local/dedicated capacity so an APU is never mistaken for a
        # workstation GPU merely because it can borrow system RAM.
        shared_capacity = max(
            _safe_int(accelerator_state.get("accessibleTotalBytes")) or 0,
            _safe_int(accelerator_state.get("sharedTotalBytes")) or 0,
        )
        if shared_capacity:
            total_vram = max(total_vram or 0, shared_capacity)
    if not _meets_total_capacity(total_vram, min_vram_bytes):
        missing.append(f"GPU memory requires at least {min_vram_bytes // GIB} GiB total")

    system_total = _resource_value(hardware, "systemMemory", "totalBytes")
    if not _meets_total_capacity(system_total, min_system_ram_bytes):
        missing.append(f"System memory requires at least {min_system_ram_bytes // GIB} GiB total")

    disk_free = _resource_value(hardware, "offloadDisk", "freeBytes")
    if min_disk_free_bytes and (not disk_free or disk_free < min_disk_free_bytes):
        missing.append(f"Offload disk requires at least {min_disk_free_bytes // GIB} GiB free")
    return missing


def _qwen_dimension(value: Any, fallback: int) -> int:
    try:
        finite = float(value)
    except (TypeError, ValueError):
        finite = float(fallback)
    if not finite > 0:
        finite = float(fallback)
    return max(256, int((finite + 1e-6) // 16) * 16)


def _qwen_generation_dimensions(form: dict[str, Any], *, native: bool) -> tuple[int, int]:
    fallback_width = QWEN_NATIVE_WIDTH if native else QWEN_PRACTICAL_WIDTH
    fallback_height = QWEN_NATIVE_HEIGHT if native else QWEN_PRACTICAL_HEIGHT
    requested_width = _qwen_dimension(form.get("width"), fallback_width)
    requested_height = _qwen_dimension(form.get("height"), fallback_height)
    if native:
        # A high-memory candidate must honor the user's/template's basic
        # generation controls. The native quality preset supplies defaults
        # only when dimensions are absent; it must never force every portrait
        # and landscape workflow into a square 1328px render.
        return requested_width, requested_height

    requested_pixels = requested_width * requested_height
    scale = min(
        1.0,
        (QWEN_PRACTICAL_PIXEL_BUDGET / requested_pixels) ** 0.5,
        QWEN_AUTO_MAX_DIMENSION / max(requested_width, requested_height),
    )
    return (
        _qwen_dimension(requested_width * scale, QWEN_PRACTICAL_WIDTH),
        _qwen_dimension(requested_height * scale, QWEN_PRACTICAL_HEIGHT),
    )


def _qwen_generation_defaults(form: dict[str, Any], hardware: dict[str, Any], *, native: bool = False) -> dict[str, Any]:
    negative = str(form.get("negativePrompt") or "").strip() or " "
    width, height = _qwen_generation_dimensions(form, native=native)
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
        return "Ran here"
    if proof_status in READY_PROOF_STATUSES:
        return "This should work"
    if _artifact_needs_repair(artifact_status):
        return "Repair required"
    if proof_status == FAILED_HERE_PROOF_STATUS:
        return "Failed here before"
    if proof_status == "known_bad":
        return "Not suitable locally"
    if proof_status == "manual_only" or manual_only_reason:
        return "Expert only"
    if missing:
        blocked = [item for item in missing if "requires at least" in item.lower() or "requires cuda" in item.lower() or "offload disk" in item.lower()]
        if blocked and not (artifact_status and artifact_status.get("installed") is False):
            return "Not suitable locally"
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
    loader_module: str | None,
    loader_action: str | None,
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
    device_map: str | None = None,
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
    model_catalog = catalog_model(model_type) or {}
    artifact_catalog = catalog_artifact(model_type, artifact) or {}
    base_artifact = str(model_catalog.get("baseRepo") or artifact)
    trust = str(artifact_catalog.get("trust") or ("official" if artifact == base_artifact else "community"))
    artifact_format = str(artifact_catalog.get("format") or ("native" if artifact == base_artifact else "prequantized"))
    is_prequantized_artifact = artifact.lower() != base_artifact.lower() and artifact_format != "native"
    loaded_quantization = artifact_format if is_prequantized_artifact else quantization_mode
    resolved_revision = artifact_catalog.get("revision") or (
        model_catalog.get("baseRevision") if artifact.lower() == base_artifact.lower() else None
    )
    if trust not in AUTO_TRUST_LEVELS:
        health_badge = "Community option"
        if status in READY_PROOF_STATUSES:
            status = "manual_only"
            message = "Community artifact requires explicit confirmation for this workflow and machine."
    evidence_level = (
        "ran_here"
        if status == "live_proven"
        else "documented"
        if status in READY_PROOF_STATUSES
        else "community"
        if trust not in AUTO_TRUST_LEVELS
        else "blocked"
    )

    return {
        "id": candidate_id,
        "rank": rank,
        "modelType": model_type,
        "mode": mode,
        "loaderModule": loader_module,
        "loaderAction": loader_action,
        "executionPath": execution_path,
        "pipelineClass": pipeline_class,
        "artifact": artifact,
        "baseArtifact": base_artifact,
        "artifactSource": "huggingface-cache",
        "modelRepo": artifact,
        "resolvedArtifact": artifact,
        "artifactResolution": {
            "base": {"repo": base_artifact, "revision": model_catalog.get("baseRevision")},
            "resolved": {
                "repo": artifact,
                "revision": resolved_revision,
                "format": artifact_format,
                "bits": artifact_catalog.get("bits"),
                "quantization": loaded_quantization,
                "components": list(artifact_catalog.get("components") or quantized_components),
            },
            "substituted": artifact.lower() != base_artifact.lower(),
        },
        "compatibilityEvidence": {
            "level": evidence_level,
            "label": health_badge,
            "source": "model_artifact_catalog" if artifact_catalog else "static_auto_requirements",
            "message": message,
            "checkedAt": _now_ms(),
            "popularity": {
                "downloads": artifact_catalog.get("downloads"),
                "likes": artifact_catalog.get("likes"),
            } if artifact_catalog else None,
        },
        "artifactTrust": trust,
        "artifactFormat": artifact_format,
        "requiresConfirmation": trust not in AUTO_TRUST_LEVELS,
        "dtype": dtype,
        # This field controls runtime conversion in legacy/custom graphs. A
        # prequantized artifact must never turn that conversion back on.
        "quantizationMode": "none" if is_prequantized_artifact else quantization_mode,
        "loadedQuantization": loaded_quantization,
        "quantizedComponents": quantized_components,
        "bnb4ComputeDtype": "bfloat16",
        "offloadMode": offload_mode,
        "autoOffload": offload_mode != OFFLOAD_MODE_NONE,
        "deviceMap": device_map,
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


def _catalog_community_candidates(
    *,
    model_type: str,
    mode: str,
    loader_module: str,
    loader_action: str,
    execution_path: str,
    pipeline_class: str,
    generation: dict[str, Any],
    local_models: list[dict[str, Any]] | None,
    hardware: dict[str, Any],
    requirements: dict[str, Any],
    profile_artifacts: set[str],
    existing_artifacts: set[str],
) -> list[dict[str, Any]]:
    model = catalog_model(model_type) or {}
    installed = _repo_id_set(local_models)
    minimum = requirements.get("minimum") if isinstance(requirements.get("minimum"), dict) else {}
    missing = _requirements_missing_for_dict(hardware, minimum, offload_mode=OFFLOAD_MODE_MODEL_CPU)
    output = []
    for index, artifact in enumerate(model.get("artifacts") or []):
        if not isinstance(artifact, dict):
            continue
        repo = str(artifact.get("repo") or "")
        trust = str(artifact.get("trust") or "community")
        if not repo or repo.lower() in existing_artifacts or trust in AUTO_TRUST_LEVELS:
            continue
        if not community_artifact_is_discoverable(artifact):
            continue
        cache_status = _artifact_cache_status(repo, local_models)
        candidate = _candidate(
            candidate_id=f"{model_type or 'studio'}-{mode or 'mode'}-community-{index}",
            rank=70 + index,
            model_type=model_type,
            mode=mode,
            loader_module=loader_module,
            loader_action=loader_action,
            execution_path=execution_path,
            artifact=repo,
            dtype="bfloat16",
            quantization_mode="none",
            quantized_components=list(artifact.get("components") or []),
            offload_mode=OFFLOAD_MODE_MODEL_CPU,
            quality_tier="community-quantized-option",
            reason=f"Popular Hugging Face community option; review its evidence before installing {repo}.",
            generation=generation,
            installed=bool(cache_status.get("installed")) or _has_installed(repo, installed),
            requirements_missing=missing + _cache_missing_for_status(cache_status, "community"),
            manual_only_reason="Community artifact requires explicit confirmation until MoDiff qualifies it on this runtime.",
            artifact_status=cache_status,
            pipeline_class=pipeline_class or None,
            requirements={"minimum": minimum},
            install_action_label="Review community option",
        )
        candidate["healthBadge"] = "Community option"
        candidate["compatibilityEvidence"]["label"] = "Community option"
        candidate["requiresConfirmation"] = True
        candidate["profileArtifactCompatible"] = repo in profile_artifacts
        output.append(candidate)
    return output


def _qwen_auto_offload_for(hardware: dict[str, Any]) -> str:
    free = _resource_value(hardware, "accelerator", "freeBytes")
    if free is not None and free < 6 * GIB:
        return OFFLOAD_MODE_SEQUENTIAL_CPU
    return OFFLOAD_MODE_MODEL_CPU


def _qwen_native_offload_for(hardware: dict[str, Any]) -> str:
    free = _resource_value(hardware, "accelerator", "freeBytes")
    # The official BF16 artifact plus native-resolution activations fit with
    # useful headroom above this boundary. Keep lower-memory systems on the
    # established model-offload path.
    if free is not None and free >= 64 * GIB:
        return OFFLOAD_MODE_NONE
    return _qwen_auto_offload_for(hardware)


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
    specification = _auto_requirements_for_pair(model_type, mode)
    if specification is None:
        return _undeclared_pair_candidates(form)
    loader_module = str(specification["loaderModule"])
    loader_action = str(specification["loaderAction"])
    execution_path = str(specification["executionPath"])
    pipeline_class = str(specification["pipelineClass"])

    offload_mode = _qwen_auto_offload_for(hardware)
    native_offload_mode = _qwen_native_offload_for(hardware)
    prequantized_missing = _requirements_missing(
        hardware,
        accelerator="cuda",
        min_vram_bytes=10 * GIB,
        min_system_ram_bytes=24 * GIB,
        offload_mode=offload_mode,
    )
    official_missing = _requirements_missing(
        hardware,
        accelerator="cuda",
        min_vram_bytes=32 * GIB,
        min_system_ram_bytes=30 * GIB,
        offload_mode=native_offload_mode,
    )
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
            loader_module=loader_module,
            loader_action=loader_action,
            execution_path=execution_path,
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
            pipeline_class=pipeline_class,
        ),
        _candidate(
            candidate_id="qwen-t2i-prequantized-sequential-cpu",
            rank=2,
            model_type=model_type,
            mode=mode,
            loader_module=loader_module,
            loader_action=loader_action,
            execution_path=execution_path,
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
            pipeline_class=pipeline_class,
        ),
        _candidate(
            candidate_id="qwen-t2i-prequantized-group-disk",
            rank=3,
            model_type=model_type,
            mode=mode,
            loader_module=loader_module,
            loader_action=loader_action,
            execution_path=execution_path,
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
                offload_mode=OFFLOAD_MODE_GROUP_DISK,
            ) + prequantized_cache_missing,
            artifact_status=prequantized_cache_status,
            pipeline_class=pipeline_class,
        ),
        _candidate(
            candidate_id="qwen-t2i-official-bf16-native",
            rank=4,
            model_type=model_type,
            mode=mode,
            loader_module=loader_module,
            loader_action=loader_action,
            execution_path=execution_path,
            artifact=QWEN_IMAGE_2512_REPO,
            dtype="bfloat16",
            quantization_mode="none",
            quantized_components=[],
            offload_mode=native_offload_mode,
            device_map="cuda" if native_offload_mode == OFFLOAD_MODE_NONE else None,
            quality_tier="official-bf16-native-quality",
            reason="Use the official BF16 Diffusers repo only when hardware has enough GPU and system memory headroom.",
            generation=official_generation,
            installed=official_installed,
            requirements_missing=official_missing + official_cache_missing,
            artifact_status=official_cache_status,
            pipeline_class=pipeline_class,
        ),
        _candidate(
            candidate_id="qwen-t2i-official-transformer-bnb4-manual",
            rank=5,
            model_type=model_type,
            mode=mode,
            loader_module=loader_module,
            loader_action=loader_action,
            execution_path=execution_path,
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
            pipeline_class=pipeline_class,
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
        "extensionDuration": float(
            _number_from_form_or_defaults(form, generation_defaults, "extensionDuration", 0)
        )
        or None,
        "shift": float(_number_from_form_or_defaults(form, generation_defaults, "shift", 0)) or None,
        "qualityPreset": f"{model_type or 'studio'}-auto",
    }


def _requirements_for_candidate(requirements: dict[str, Any], key: str, fallback: dict[str, Any]) -> dict[str, Any]:
    value = requirements.get(key)
    return value if isinstance(value, dict) else fallback


def _requirements_missing_for_dict(
    hardware: dict[str, Any],
    requirement: dict[str, Any],
    *,
    offload_mode: str = OFFLOAD_MODE_NONE,
) -> list[str]:
    return _requirements_missing(
        hardware,
        accelerator=str(requirement.get("accelerator") or "cuda_or_mps_or_cpu"),
        min_vram_bytes=_safe_int(requirement.get("vramBytes")),
        min_system_ram_bytes=_safe_int(requirement.get("systemRamBytes")),
        min_disk_free_bytes=_safe_int(requirement.get("diskFreeBytes")),
        offload_mode=offload_mode,
    )


def _cache_missing_for_status(status: dict[str, Any], label: str) -> list[str]:
    if status.get("installed") and not status.get("complete"):
        return [str(status.get("reason") or f"Cached {label} artifact is incomplete.")]
    return []


def _undeclared_pair_candidates(form: dict[str, Any]) -> list[dict[str, Any]]:
    model_type = str(form.get("modelType") or "").strip()
    mode = str(form.get("mode") or "").strip()
    pair_label = f"{model_type or '<missing model type>'}:{mode or '<missing mode>'}"
    declared_modes = _declared_auto_modes(model_type)
    if declared_modes:
        reason = (
            f"No Auto recipe is declared for the exact model/task pair '{pair_label}'. "
            f"Declared Auto modes for {model_type} are: {', '.join(declared_modes)}. "
            "This workflow remains available for explicit Expert configuration when its graph structure "
            "and runtime inputs are valid."
        )
    else:
        reason = (
            f"No Auto recipe is declared for the exact model/task pair '{pair_label}'. "
            "This workflow remains available for explicit Expert configuration when its graph structure "
            "and runtime inputs are valid."
        )

    candidate = _candidate(
        candidate_id=f"{model_type or 'studio'}-{mode or 'mode'}-expert-only",
        rank=99,
        model_type=model_type,
        mode=mode,
        loader_module=None,
        loader_action=None,
        execution_path=str(form.get("executionPath") or ""),
        artifact=str(form.get("defaultRepo") or form.get("modelRepo") or ""),
        dtype=str(form.get("dtype") or "bfloat16"),
        quantization_mode="none",
        quantized_components=[],
        offload_mode=str(form.get("offloadMode") or OFFLOAD_MODE_NONE),
        quality_tier="expert-only",
        reason=reason,
        generation=_generation_for_requirements(model_type, form, {}),
        installed=True,
        manual_only_reason=reason,
        pipeline_class=str(form.get("pipelineClass") or "") or None,
    )
    candidate["exactPairDeclared"] = False
    return [candidate]


def _declared_profile_candidates(
    form: dict[str, Any],
    local_models: list[dict[str, Any]] | None,
    hardware: dict[str, Any],
) -> list[dict[str, Any]]:
    model_type = str(form.get("modelType") or "")
    mode = str(form.get("mode") or "")
    installed = _repo_id_set(local_models)
    requirements = _auto_requirements_for_pair(model_type, mode)
    if requirements is None:
        return _undeclared_pair_candidates(form)
    default_repo = str(requirements.get("defaultRepo") or form.get("defaultRepo") or form.get("modelRepo") or "")
    lower_memory_repo = str(requirements.get("preferredLowerMemoryRepo") or "")
    manual_only_reason = requirements.get("manualOnlyReason")
    loader_module = str(requirements["loaderModule"])
    loader_action = str(requirements["loaderAction"])
    execution_path = str(requirements["executionPath"])
    pipeline_class = str(requirements["pipelineClass"])
    generation_defaults = requirements.get("qualityDefaults") if isinstance(requirements.get("qualityDefaults"), dict) else {}

    minimum = requirements.get("minimum") if isinstance(requirements.get("minimum"), dict) else requirements.get("recommended")
    minimum = minimum if isinstance(minimum, dict) else {}
    generation = _generation_for_requirements(model_type, form, generation_defaults)
    required_packages = requirements.get("requiredPackages") if isinstance(requirements.get("requiredPackages"), list) else []
    supported_offload = requirements.get("supportedOffloadModes") if isinstance(requirements.get("supportedOffloadModes"), list) else []
    # Auto owns the resource choice. A saved/template form can contain a stale
    # Expert offload value from another machine, so use the declared constrained
    # default until this runtime proves that full residency is appropriate.
    # The supported list is a capability set, not a constrained-hardware
    # preference order. Several modular profiles list `none` first so the UI
    # can offer it in Expert mode; treating that as Auto's fallback silently
    # selected full residency even when the full-residency requirements failed.
    constrained_offload = next(
        (mode for mode in supported_offload if str(mode) != OFFLOAD_MODE_NONE),
        OFFLOAD_MODE_NONE if OFFLOAD_MODE_NONE in supported_offload else OFFLOAD_MODE_MODEL_CPU,
    )
    accelerator = hardware.get("accelerator") if isinstance(hardware.get("accelerator"), dict) else {}
    accelerator_kind = str(accelerator.get("kind") or "cpu")
    # Diffusers/Accelerate CPU-offload hooks are currently qualified only for
    # CUDA-device APIs (NVIDIA CUDA and AMD ROCm). MPS and XPU remain useful
    # direct-residency execution devices and must not inherit an invalid CUDA
    # offload recipe merely because a model profile lists one.
    preferred_offload = OFFLOAD_MODE_NONE if accelerator_kind in {"mps", "xpu", "cpu"} else str(constrained_offload)
    full_residency = requirements.get("fullResidency") if isinstance(requirements.get("fullResidency"), dict) else None
    high_quality = requirements.get("highQuality") if isinstance(requirements.get("highQuality"), dict) else None
    on_device_requirements = full_residency or high_quality
    full_residency_ready = bool(
        on_device_requirements
        and OFFLOAD_MODE_NONE in supported_offload
        and not _requirements_missing_for_dict(hardware, on_device_requirements)
    )
    if full_residency_ready:
        preferred_offload = OFFLOAD_MODE_NONE
    candidates: list[dict[str, Any]] = []

    lower_memory = requirements.get("lowerMemory") if isinstance(requirements.get("lowerMemory"), dict) else None
    if lower_memory_repo:
        lower_req = _requirements_for_candidate(requirements, "lowerMemory", minimum)
        lower_missing = _requirements_missing_for_dict(hardware, lower_req, offload_mode=preferred_offload)
        lower_cache_status = _artifact_cache_status(lower_memory_repo, local_models)
        candidates.append(_candidate(
            candidate_id=f"{model_type or 'studio'}-{mode or 'mode'}-lower-memory-artifact",
            rank=10,
            model_type=model_type,
            mode=mode,
            loader_module=loader_module,
            loader_action=loader_action,
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
            device_map="cuda" if preferred_offload == OFFLOAD_MODE_NONE and accelerator_kind == "cuda" else None,
            requirements={
                "minimum": requirements.get("minimum"),
                "recommended": requirements.get("recommended"),
                "fullResidency": requirements.get("fullResidency"),
                "lowerMemory": requirements.get("lowerMemory"),
                "supportedOffloadModes": requirements.get("supportedOffloadModes"),
                "coldLoadTarget": requirements.get("coldLoadTarget"),
            },
            required_packages=required_packages,
            install_action_label="Install quantized artifact",
        ))

    if default_repo:
        native_req = requirements.get("minimum") if isinstance(requirements.get("minimum"), dict) else minimum
        native_missing = _requirements_missing_for_dict(hardware, native_req, offload_mode=preferred_offload)
        default_cache_status = _artifact_cache_status(default_repo, local_models)
        candidates.append(_candidate(
            candidate_id=f"{model_type or 'studio'}-{mode or 'mode'}-native-bf16",
            rank=5 if full_residency_ready else 30,
            model_type=model_type,
            mode=mode,
            loader_module=loader_module,
            loader_action=loader_action,
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
            device_map="cuda" if preferred_offload == OFFLOAD_MODE_NONE and accelerator_kind == "cuda" else None,
            requirements={
                "minimum": requirements.get("minimum"),
                "recommended": requirements.get("recommended"),
                "fullResidency": requirements.get("fullResidency"),
                "supportedOffloadModes": requirements.get("supportedOffloadModes"),
                "coldLoadTarget": requirements.get("coldLoadTarget"),
            },
            required_packages=required_packages,
        ))

    existing_artifacts = {
        str(candidate.get("resolvedArtifact") or candidate.get("artifact") or "").lower()
        for candidate in candidates
    }
    profile_artifacts = {
        str(repo)
        for repo in (
            requirements.get("defaultRepo"),
            requirements.get("fallbackRepo"),
            *(requirements.get("compatibleRepos") or []),
        )
        if isinstance(repo, str) and repo
    }
    candidates.extend(_catalog_community_candidates(
        model_type=model_type,
        mode=mode,
        loader_module=loader_module,
        loader_action=loader_action,
        execution_path=execution_path,
        pipeline_class=pipeline_class,
        generation=generation,
        local_models=local_models,
        hardware=hardware,
        requirements=requirements,
        profile_artifacts=profile_artifacts,
        existing_artifacts=existing_artifacts,
    ))

    if not candidates:
        candidates.append(_candidate(
            candidate_id=f"{model_type or 'studio'}-{mode or 'mode'}-expert-only",
            rank=99,
            model_type=model_type,
            mode=mode,
            loader_module=loader_module,
            loader_action=loader_action,
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


def _normalized_history_entry_signature(entry: dict[str, Any]) -> dict[str, Any] | None:
    stored = entry.get("signature") if isinstance(entry.get("signature"), dict) else {}
    candidate = entry.get("candidate") if isinstance(entry.get("candidate"), dict) else None
    if not candidate:
        return None
    candidate = _clone_candidate(candidate)
    for key in (
        "modelType",
        "mode",
        "artifact",
        "artifactRevision",
        "dtype",
        "quantizationMode",
        "quantizedComponents",
        "offloadMode",
        "device",
        "deviceMap",
        "attentionBackend",
        "regionalCompile",
        "denoiserCache",
        "channelsLast",
        "layerwiseCasting",
        "pipelineClass",
        "loaderModule",
        "loaderAction",
        "executionPath",
    ):
        if candidate.get(key) is None and stored.get(key) is not None:
            candidate[key] = stored[key]
    hardware_fingerprint = stored.get("hardwareFingerprint")
    hardware = {"runtimeFingerprint": hardware_fingerprint} if hardware_fingerprint else None
    return _candidate_history_signature(candidate, hardware=hardware)


def _history_signatures_are_compatible(current: dict[str, Any], stored: dict[str, Any]) -> bool:
    current_base = {key: value for key, value in current.items() if key != "workload"}
    stored_base = {key: value for key, value in stored.items() if key != "workload"}
    if current_base != stored_base:
        return False
    current_workload = current.get("workload") if isinstance(current.get("workload"), dict) else {}
    stored_workload = stored.get("workload") if isinstance(stored.get("workload"), dict) else {}
    if not stored_workload:
        return False
    # Newer signatures may add a media-specific dimension that old receipts
    # could not record. Every dimension the older receipt did record must still
    # match; missing new dimensions are accepted only for this migration path.
    return all(key in current_workload and current_workload[key] == value for key, value in stored_workload.items())


def _compatible_success_history_entry(
    candidate: dict[str, Any],
    *,
    entries: dict[str, Any],
    hardware: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    current = _candidate_history_signature(candidate, hardware=hardware)
    matches = []
    for key, value in entries.items():
        if not isinstance(value, dict) or not value.get("successCount"):
            continue
        if int(value.get("lastFailureAt") or 0) > int(value.get("lastSuccessAt") or 0):
            continue
        stored = _normalized_history_entry_signature(value)
        if stored and _history_signatures_are_compatible(current, stored):
            matches.append((int(value.get("lastSuccessAt") or 0), str(key), value))
    if not matches:
        return None
    _, key, entry = max(matches, key=lambda item: item[0])
    return key, entry


def _requirements_after_success_evidence(
    candidate: dict[str, Any],
    entry: dict[str, Any],
    hardware: dict[str, Any],
) -> list[str]:
    missing = [str(item) for item in candidate.get("requirementsMissing") or []]
    measurement = entry.get("lastMeasurement") if isinstance(entry.get("lastMeasurement"), dict) else {}
    accelerator = hardware.get("accelerator") if isinstance(hardware.get("accelerator"), dict) else {}
    local_total = _safe_int(accelerator.get("totalBytes"))
    peak = max(
        _safe_int(measurement.get("peakReservedBytes")) or 0,
        _safe_int(measurement.get("peakAllocatedBytes")) or 0,
        _safe_int(measurement.get("reservedBytes")) or 0,
        _safe_int(measurement.get("allocatedBytes")) or 0,
    )
    system_total = _resource_value(hardware, "systemMemory", "totalBytes")
    process_rss = _safe_int(measurement.get("processRssBytes"))
    remaining = []
    for requirement in missing:
        lowered = requirement.lower()
        if lowered.startswith("gpu memory requires") and local_total and peak and peak <= local_total:
            continue
        if lowered.startswith("system memory requires") and system_total and process_rss and process_rss <= system_total:
            continue
        remaining.append(requirement)
    return remaining


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
        if item.get("exactPairDeclared") is False:
            item["historyKey"] = key
            output.append(item)
            continue
        entry = entries.get(key) if isinstance(entries.get(key), dict) else None
        compatible_key = None
        if entry is None:
            compatible = _compatible_success_history_entry(item, entries=entries, hardware=hardware)
            if compatible:
                compatible_key, entry = compatible
        item["historyKey"] = key
        if compatible_key:
            item["compatibleHistoryKey"] = compatible_key
        item["failureHistory"] = entry if entry and entry.get("failureCount") else None
        item["successHistory"] = entry if entry and entry.get("successCount") else None
        if entry and entry.get("successCount") and item.get("installed"):
            remaining_requirements = _requirements_after_success_evidence(item, entry, hardware)
            item["requirementsMissing"] = remaining_requirements
            if not remaining_requirements:
                item["requirementsMatched"] = ["artifact", "hardware", "quality-defaults", "local-success"]
        if entry and entry.get("successCount") and not item.get("requirementsMissing") and item.get("installed"):
            proof = item.get("proof") if isinstance(item.get("proof"), dict) else {}
            proof = dict(proof)
            proof["status"] = "live_proven"
            proof["source"] = "auto_resource_history_compatible" if compatible_key else "auto_resource_history"
            proof["message"] = (
                "A compatible earlier Auto receipt completed successfully on this machine."
                if compatible_key
                else "This Auto candidate completed successfully on this machine before."
            )
            proof["checkedAt"] = _now_ms()
            item["proof"] = proof
            item["readiness"] = "ready"
            item["canAutoRun"] = True
            item["healthBadge"] = "Ran here"
            evidence = item.get("compatibilityEvidence") if isinstance(item.get("compatibilityEvidence"), dict) else {}
            item["compatibilityEvidence"] = {
                **evidence,
                "level": "ran_here",
                "label": "Ran here",
                "source": "auto_resource_history",
                "message": proof["message"],
                "checkedAt": _now_ms(),
            }
            item["qualityScore"] = int(item.get("qualityScore") or 0) + 250
        if entry and entry.get("failureCount"):
            last_failure = entry.get("lastFailure") if isinstance(entry.get("lastFailure"), dict) else {}
            last_failure_at = int(entry.get("lastFailureAt") or 0)
            last_success_at = int(entry.get("lastSuccessAt") or 0)
            # A success recorded in the same millisecond is the newest event in
            # the synchronous completion path and must clear the prior failure.
            if last_failure_at > last_success_at:
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
    return output


def _preference_sort_key(candidate: dict[str, Any], preference: str) -> tuple[Any, ...]:
    proof = candidate.get("proof") if isinstance(candidate.get("proof"), dict) else {}
    status = str(proof.get("status") or "")
    quality_tier = str(candidate.get("qualityTier") or "").lower()
    trust = str(candidate.get("artifactTrust") or "community")
    ran_here = 0 if status == "live_proven" else 1
    blocked = 0 if status in READY_PROOF_STATUSES else 1
    trusted = 0 if trust in AUTO_TRUST_LEVELS else 1
    native = 0 if "native" in quality_tier or not (candidate.get("artifactResolution") or {}).get("substituted") else 1
    quantized = 0 if "quant" in quality_tier or "lower-memory" in quality_tier else 1
    rank = int(candidate.get("rank") or 999)
    install_only_needles = (
        "not installed",
        "snapshot is incomplete",
        "cached artifact",
        "download is still incomplete",
        "local weight files",
        "zero bytes",
        ".safetensors",
    )
    hardware_blocked = 0
    for requirement in candidate.get("requirementsMissing") or []:
        text = str(requirement).lower()
        if not any(needle in text for needle in install_only_needles):
            hardware_blocked = 1
            break
    if preference == "best_quality":
        return blocked, hardware_blocked, native, ran_here, trusted, rank
    if preference == "faster":
        history = candidate.get("successHistory") if isinstance(candidate.get("successHistory"), dict) else {}
        elapsed = float(history.get("bestElapsedSeconds") or 1e30)
        return blocked, hardware_blocked, ran_here, elapsed, trusted, rank
    if preference == "lowest_memory":
        bits = ((candidate.get("artifactResolution") or {}).get("resolved") or {}).get("bits")
        return blocked, hardware_blocked, 0 if bits else 1, int(bits or 128), quantized, ran_here, rank
    # Recommended is quality-leaning: use a compatible native artifact first,
    # but prefer an exact successful lower-memory recipe over an estimate.
    return blocked, hardware_blocked, ran_here, trusted, native, rank


def _rank_candidates_for_preference(candidates: list[dict[str, Any]], preference: str) -> list[dict[str, Any]]:
    normalized = preference if preference in {"recommended", "best_quality", "faster", "lowest_memory"} else "recommended"
    output = [_clone_candidate(candidate) for candidate in candidates]
    output.sort(key=lambda item: _preference_sort_key(item, normalized))
    for index, candidate in enumerate(output):
        candidate["preference"] = normalized
        candidate["preferenceRank"] = index + 1
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
        return str(selected.get("healthBadge") or "This should work")
    if selected_install_target and selected_install_target.get("repair"):
        return "Repair required"
    if selected_install_target:
        return "Needs setup"
    if readiness == "manual_only":
        return "Expert only"
    if any(candidate.get("healthBadge") == "Failed here before" for candidate in candidates):
        return "Failed here before"
    if any(candidate.get("healthBadge") == "Not suitable locally" for candidate in candidates):
        return "Not suitable locally"
    return "Needs setup"


def _compatibility_assessment(
    *,
    selected: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    selected_install_target: dict[str, Any] | None,
    readiness: str,
    status_label: str,
    blocking_reason: str | None,
    will_not_work_reason: str | None,
) -> dict[str, Any]:
    """Return the UI-facing result of the authoritative Auto assessment.

    Clients should render this result instead of independently interpreting
    GPU memory, operating-system, accelerator, or model-fit thresholds.
    """
    if selected:
        proof = selected.get("proof") if isinstance(selected.get("proof"), dict) else {}
        detail = (
            proof.get("message")
            or selected.get("reason")
            or "Auto selected a qualified recipe for the current runtime and installed models."
        )
        return {
            "state": "ready",
            "severity": "success",
            "code": "auto_recipe_ready",
            "summary": status_label,
            "detail": str(detail),
            "action": None,
            "source": "backend_auto_planner",
        }

    if selected_install_target:
        repair = bool(selected_install_target.get("repair"))
        return {
            "state": "needs_model",
            "severity": "warning",
            "code": "model_repair_required" if repair else "model_install_required",
            "summary": "Repair required" if repair else "Model setup required",
            "detail": str(
                selected_install_target.get("reason")
                or blocking_reason
                or "Install the selected Auto artifact before running this workflow."
            ),
            "action": {
                "type": "repair_model" if repair else "install_model",
                "label": selected_install_target.get("actionLabel")
                or ("Repair Auto artifact" if repair else "Install Auto artifact"),
                "repo": selected_install_target.get("repo"),
                "candidateId": selected_install_target.get("candidateId"),
            },
            "source": "backend_auto_planner",
        }

    if readiness == "manual_only":
        return {
            "state": "expert_only",
            "severity": "warning",
            "code": "expert_configuration_required",
            "summary": "Expert configuration required",
            "detail": str(blocking_reason or "This workflow does not have a qualified Auto recipe yet."),
            "action": {"type": "switch_to_expert", "label": "Review in Expert mode"},
            "source": "backend_auto_planner",
        }

    unsuitable = bool(will_not_work_reason) or any(
        candidate.get("healthBadge") in {"Failed here before", "Not suitable locally"}
        for candidate in candidates
    )
    return {
        "state": "unsuitable" if unsuitable else "needs_setup",
        "severity": "error" if unsuitable else "warning",
        "code": "no_qualified_local_recipe" if unsuitable else "auto_setup_required",
        "summary": "Not suitable on this runtime" if unsuitable else status_label,
        "detail": str(
            will_not_work_reason
            or blocking_reason
            or "No qualified local Auto recipe matched this runtime and model cache."
        ),
        "action": {"type": "open_setup", "label": "Open Setup"},
        "source": "backend_auto_planner",
    }


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
    preference = str(form.get("resourcePreference") or payload.get("resourcePreference") or "recommended")
    hardware_override = payload.get("hardwareOverride") if isinstance(payload.get("hardwareOverride"), dict) else None
    hardware = hardware_override or _hardware_snapshot(runtime_fingerprint, data_dir)

    exact_pair_requirements = _auto_requirements_for_pair(model_type, mode)
    if exact_pair_requirements is None:
        candidates = _undeclared_pair_candidates(form)
    elif model_type == "QwenImageModularPipeline" and mode == "text_to_image":
        candidates = _qwen_text_to_image_candidates(form, local_models, hardware)
    else:
        candidates = _declared_profile_candidates(form, local_models, hardware)

    exact_pair_declared = exact_pair_requirements is not None
    optional_runtime_profile_ids = optional_runtime_profile_ids_for_execution(
        model_type,
        mode,
    )
    optional_runtime_profiles = public_optional_runtime_profiles(
        optional_runtime_profile_ids
    )
    optional_runtime_requirement = optional_runtime_requirement_for_execution(
        model_type,
        mode,
    )
    for candidate in candidates:
        candidate["exactPairDeclared"] = exact_pair_declared
        candidate["optionalRuntimeProfileIds"] = list(optional_runtime_profile_ids)
        candidate["optionalRuntimeRequirement"] = dict(optional_runtime_requirement)

    candidates = _apply_catalog_hardware_support(candidates, hardware)
    candidates = _apply_community_confirmation(candidates, form)
    candidates = _apply_history_to_candidates(candidates, history=history or read_auto_resource_history(data_dir), hardware=hardware)
    candidates = _rank_candidates_for_preference(candidates, preference)
    runtime_identity = (
        runtime_fingerprint.get("resourceFingerprint")
        if isinstance(runtime_fingerprint, dict)
        else runtime_fingerprint
    )
    workload_key = workload_key_for_form(form)
    for candidate in candidates:
        if candidate.get("exactPairDeclared") is False:
            continue
        artifact = str(
            candidate.get("resolvedArtifact")
            or candidate.get("artifact")
            or candidate.get("modelRepo")
            or ""
        )
        overrides = qualified_auto_overrides(
            runtime_fingerprint=runtime_identity,
            model_type=str(candidate.get("modelType") or model_type),
            mode=str(candidate.get("mode") or mode),
            artifact=artifact,
            workload_key=workload_key,
        )
        if overrides:
            candidate.update(overrides)
            candidate["optimizationQualification"] = {
                "status": "qualified",
                "workloadKey": workload_key,
                "selection": overrides,
            }

    ready = [
        candidate for candidate in candidates
        if candidate.get("exactPairDeclared") is not False
        and candidate.get("proof", {}).get("status") in READY_PROOF_STATUSES
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
        if candidate.get("healthBadge") in {"Failed here before", "Not suitable locally"}
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

    compatibility = _compatibility_assessment(
        selected=selected,
        candidates=candidates,
        selected_install_target=selected_install_target,
        readiness=readiness,
        status_label=status_label,
        blocking_reason=blocking_reason,
        will_not_work_reason=will_not_work_reason,
    )

    return {
        "error": False,
        "schemaVersion": AUTO_RESOURCE_SCHEMA_VERSION,
        "resourceMode": "auto",
        "resourcePreference": preference,
        "exactPairDeclared": exact_pair_declared,
        "optionalRuntimeProfileIds": list(optional_runtime_profile_ids),
        "optionalRuntimeProfiles": optional_runtime_profiles,
        "optionalRuntimeRequirement": optional_runtime_requirement,
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
        "compatibility": compatibility,
        "canAutoRun": bool(selected),
        "repairRequired": bool(selected_install_target and selected_install_target.get("repair")),
        "selectedInstallTarget": selected_install_target,
        "failureHistory": [candidate.get("failureHistory") for candidate in candidates if candidate.get("failureHistory")],
        "selectedCandidate": selected,
        "candidates": candidates,
        "nextCandidate": next((candidate for candidate in candidates if candidate is not selected and candidate.get("proof", {}).get("status") in READY_PROOF_STATUSES), None),
        "hardware": hardware,
        "hardwareSnapshot": hardware,
        # Aggregate model-only requirements cannot faithfully represent models
        # whose modes use different loaders.  Schema v2 publishes one exact
        # specification per model/task pair instead.
        "modelRequirements": _public_auto_model_requirements(),
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
        "schemaVersion": AUTO_RESOURCE_SCHEMA_VERSION,
        "resourceMode": "auto",
        "count": len(plans),
        "plans": plans,
        "checkedAt": _now_ms(),
    }
