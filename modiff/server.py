import logging
import asyncio
from aiohttp import web, WSMsgType
from aiohttp.web_fileresponse import CONTENT_TYPES as AIOHTTP_CONTENT_TYPES
from aiohttp_cors import setup as cors_setup, ResourceOptions
import aiofiles
import mimetypes

mimetypes.add_type("image/webp", ".webp")
AIOHTTP_CONTENT_TYPES.add_type("image/webp", ".webp")

logging.getLogger('asyncio').setLevel(logging.WARNING)
from functools import partial
from importlib import import_module, metadata, invalidate_caches
import os
import platform
import base64
import csv
import hashlib
import html
import json
import nanoid
import random
import re
import shutil
import subprocess
import configparser
from utils.paths import list_files
from pathlib import Path
import sys
import traceback
import time
import gc
from urllib.parse import quote, unquote, unquote_to_bytes, urlparse, parse_qs
from copy import deepcopy
logger = logging.getLogger('modiff')

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
    if "save" in name:
        return "saving"
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
    if phase == "saving":
        return f"Saving {name}"
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
    if data_type == 'image':
        return True
    if isinstance(data_type, (list, tuple, set)):
        return any(item == 'image' for item in data_type)
    return False

def image_dimensions(value):
    width = getattr(value, 'width', None)
    height = getattr(value, 'height', None)
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

from modiff.config import CONFIG
from modiff.diffusers_offload import (
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_NONE,
    OFFLOAD_MODE_SEQUENTIAL_CPU,
)
from modiff.diffusers_profiles import QWEN_IMAGE_2512_PREQUANTIZED_REPO, public_execution_profiles
from modiff.hardware import format_hardware_summary, get_hardware_snapshot, legacy_torch_status
from modiff.auto_resource import (
    PROVEN_PROOF_STATUSES,
    build_auto_resource_plan,
    build_auto_resource_plans,
    clear_auto_resource_history,
    read_auto_resource_history,
    record_auto_resource_failure,
    record_auto_resource_success,
)
from modiff.modelstore import modelstore
from modules import MODULE_MAP, parse_module_map
from utils.huggingface import get_local_models, delete_model, search_hub, download_hub_model, get_local_model_ids, get_cache_diagnostics
from utils.memory_menager import memory_manager
from utils.torch_utils import reset_memory_stats, get_memory_stats

MODULAR_OFFLOAD_SUPPORT = {
    'default': OFFLOAD_MODE_MODEL_CPU,
    'lowVram': OFFLOAD_MODE_MODEL_CPU,
    'emergency': OFFLOAD_MODE_GROUP_DISK,
    'modes': [OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK],
}

QWEN_MODULAR_OFFLOAD_SUPPORT = {
    'default': OFFLOAD_MODE_MODEL_CPU,
    'lowVram': OFFLOAD_MODE_MODEL_CPU,
    'emergency': OFFLOAD_MODE_GROUP_DISK,
    'modes': [OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK],
}

DIRECT_OFFLOAD_SUPPORT = {
    'default': OFFLOAD_MODE_MODEL_CPU,
    'lowVram': OFFLOAD_MODE_MODEL_CPU,
    'emergency': OFFLOAD_MODE_GROUP_DISK,
    'modes': [OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK],
}

QWEN_IMAGE_EDIT_INPAINT_CONTRACT = {
    'available': True,
    'status': 'supported',
    'reason': 'Direct Diffusers Qwen Image Edit inpaint is available through modules.QwenImage.LoadInpaintPipeline -> modules.QwenImage.Inpaint with source image and mask_image inputs. Outpaint uses modules.QwenImage.OutpaintCanvas to build the expanded canvas and boundary mask before the same inpaint node.',
    'source': 'modules.QwenImage.Inpaint',
    'checkedInputs': {
        'loader': ['model_id', 'dtype', 'device', 'auto_offload', 'offload_mode', 'quant_config'],
        'outpaint': ['image', 'width', 'height', 'left', 'right', 'top', 'bottom', 'overlap', 'feather', 'fill_color', 'canvas', 'mask_image'],
        'inpaint': ['pipeline', 'image', 'mask_image', 'prompt', 'negative_prompt', 'true_cfg_scale', 'strength', 'num_inference_steps'],
    },
    'missingInputs': [],
}

QWEN_IMAGE_EDIT_PLUS_INPAINT_CONTRACT = {
    'available': False,
    'status': 'blocked',
    'reason': 'Qwen Image Edit Plus does not yet have a confirmed native mask, mask_image, or masked_image_latents execution contract in MoDiff.',
    'source': 'modules.ModularDiffusers.modular_utils.QWEN_IMAGE_EDIT_PLUS_NODE_SPECS',
    'checkedInputs': {
        'denoise': ['embeddings', 'seed', 'num_inference_steps', 'guidance_scale', 'image_latents'],
        'vae_encoder': ['image'],
        'text_encoder': ['prompt', 'negative_prompt', 'image'],
    },
    'missingInputs': ['mask', 'mask_image', 'masked_image_latents'],
}


class MissingConnectedOutputError(RuntimeError):
    pass

STUDIO_MODEL_CAPABILITIES = {
    'ZImageModularPipeline': {
        'modelType': 'ZImageModularPipeline',
        'label': 'Z-Image Turbo',
        'displayName': 'Z-Image-Turbo',
        'family': 'Z-Image',
        'defaultRepo': 'Tongyi-MAI/Z-Image-Turbo',
        'artifactLabel': 'Diffusers repo',
        'defaultDtype': 'bfloat16',
        'defaultSize': {'width': 1024, 'height': 1024, 'aspectRatio': '1:1'},
        'recommendedSteps': 8,
        'recommendedGuidance': 1,
        'guidanceLabel': 'Guidance',
        'supportsImageInput': False,
        'supportsMask': False,
        'supportsMultiImage': False,
        'supportsControlImage': False,
        'supportsLayers': False,
        'supportsLora': True,
        'offloadSupport': QWEN_MODULAR_OFFLOAD_SUPPORT,
        'lowVram': {'dtype': 'bfloat16', 'autoOffload': True, 'offloadMode': OFFLOAD_MODE_MODEL_CPU, 'steps': 8},
        'modes': ['text_to_image'],
        'executionStatus': 'supported',
    },
    'QwenImageModularPipeline': {
        'modelType': 'QwenImageModularPipeline',
        'label': 'Qwen-Image-2512',
        'displayName': 'Qwen-Image-2512',
        'family': 'Qwen Image',
        'defaultRepo': 'Qwen/Qwen-Image-2512',
        'artifactLabel': 'bfloat16 Diffusers repo',
        'comfyArtifact': 'qwen_image_2512_fp8_e4m3fn.safetensors',
        'defaultDtype': 'bfloat16',
        'defaultSize': {'width': 1024, 'height': 1024, 'aspectRatio': '1:1'},
        'recommendedSteps': 50,
        'recommendedGuidance': 4.5,
        'guidanceLabel': 'Guidance',
        'supportsImageInput': False,
        'supportsMask': False,
        'supportsMultiImage': False,
        'supportsControlImage': True,
        'supportsLayers': False,
        'supportsLora': True,
        'offloadSupport': QWEN_MODULAR_OFFLOAD_SUPPORT,
        'lowVram': {'dtype': 'bfloat16', 'quantizationMode': 'bnb_4bit', 'autoOffload': True, 'offloadMode': OFFLOAD_MODE_MODEL_CPU, 'steps': 28},
        'modes': ['text_to_image', 'control_image'],
        'executionStatus': 'supported_with_model',
        'additionalRequirements': [
            {
                'id': 'qwen-controlnet-union',
                'label': 'Qwen ControlNet Union',
                'repo': 'InstantX/Qwen-Image-ControlNet-Union',
                'kind': 'controlnet',
                'requiredForModes': ['control_image'],
                'description': 'Required for Qwen Image Control image workflows.',
            }
        ],
        'modeRequirements': {
            'control_image': {
                'modelRequirements': [
                    {
                        'id': 'qwen-controlnet-union',
                        'label': 'Qwen ControlNet Union',
                        'repo': 'InstantX/Qwen-Image-ControlNet-Union',
                        'kind': 'controlnet',
                        'requiredForModes': ['control_image'],
                        'description': 'Required for Qwen Image Control image workflows.',
                    }
                ],
                'requiredImages': ['controlImage'],
                'note': 'Requires the Qwen ControlNet Union model plus one control image.',
            }
        },
    },
    'QwenImageEditModularPipeline': {
        'modelType': 'QwenImageEditModularPipeline',
        'label': 'Qwen-Image-Edit',
        'displayName': 'Qwen-Image-Edit',
        'family': 'Qwen Image',
        'defaultRepo': 'Qwen/Qwen-Image-Edit',
        'artifactLabel': 'bfloat16 Diffusers repo',
        'defaultDtype': 'bfloat16',
        'defaultSize': {'width': 1024, 'height': 1024, 'aspectRatio': '1:1'},
        'recommendedSteps': 40,
        'recommendedGuidance': 4,
        'guidanceLabel': 'Guidance',
        'supportsImageInput': True,
        'supportsMask': True,
        'supportsMultiImage': False,
        'supportsControlImage': False,
        'supportsLayers': False,
        'supportsLora': True,
        'offloadSupport': DIRECT_OFFLOAD_SUPPORT,
        'lowVram': {'dtype': 'bfloat16', 'quantizationMode': 'bnb_4bit', 'autoOffload': True, 'offloadMode': OFFLOAD_MODE_MODEL_CPU, 'steps': 24},
        'modes': ['edit_image', 'inpaint', 'outpaint'],
        'executionStatus': 'supported_with_model',
        'notes': ['Inpaint and outpaint use the direct Diffusers QwenImageEditInpaintPipeline backend nodes.'],
        'inpaintContract': QWEN_IMAGE_EDIT_INPAINT_CONTRACT,
        'modeRequirements': {
            'inpaint': {
                'requiredImages': ['referenceImages', 'maskImage'],
                'note': 'Requires one source image and one mask image.',
            },
            'outpaint': {
                'requiredImages': ['referenceImages'],
                'note': 'Requires one source image; MoDiff builds the expanded canvas and boundary mask.',
            }
        },
    },
    'QwenImageEditPlusModularPipeline': {
        'modelType': 'QwenImageEditPlusModularPipeline',
        'label': 'Qwen-Image-Edit-2511',
        'displayName': 'Qwen-Image-Edit-2511',
        'family': 'Qwen Image',
        'defaultRepo': 'Qwen/Qwen-Image-Edit-2511',
        'artifactLabel': 'bfloat16 Diffusers repo',
        'comfyArtifact': 'qwen_image_edit_2511_bf16.safetensors',
        'defaultDtype': 'bfloat16',
        'defaultSize': {'width': 1024, 'height': 1024, 'aspectRatio': '1:1'},
        'recommendedSteps': 40,
        'recommendedGuidance': 4,
        'guidanceLabel': 'Guidance',
        'supportsImageInput': True,
        'supportsMask': False,
        'supportsMultiImage': True,
        'supportsControlImage': False,
        'supportsLayers': False,
        'supportsLora': True,
        'offloadSupport': QWEN_MODULAR_OFFLOAD_SUPPORT,
        'lowVram': {'dtype': 'bfloat16', 'quantizationMode': 'bnb_4bit', 'autoOffload': True, 'offloadMode': OFFLOAD_MODE_MODEL_CPU, 'steps': 24},
        'modes': ['edit_image', 'multi_image_reference_edit', 'inpaint'],
        'executionStatus': 'supported_with_model',
        'notes': ['Inpaint mask execution still requires a confirmed backend mask graph contract.'],
        'inpaintContract': QWEN_IMAGE_EDIT_PLUS_INPAINT_CONTRACT,
        'modeRequirements': {
            'inpaint': {
                'requiredImages': ['referenceImages', 'maskImage'],
                'note': QWEN_IMAGE_EDIT_PLUS_INPAINT_CONTRACT['reason'],
            }
        },
    },
    'QwenImageLayeredModularPipeline': {
        'modelType': 'QwenImageLayeredModularPipeline',
        'label': 'Qwen-Image-Layered',
        'displayName': 'Qwen-Image-Layered',
        'family': 'Qwen Image',
        'defaultRepo': 'Qwen/Qwen-Image-Layered',
        'artifactLabel': 'bfloat16 Diffusers repo',
        'comfyArtifact': 'qwen_image_layered_bf16.safetensors',
        'defaultDtype': 'bfloat16',
        'defaultSize': {'width': 1024, 'height': 1024, 'aspectRatio': '1:1'},
        'recommendedSteps': 50,
        'recommendedGuidance': 4,
        'guidanceLabel': 'Guidance',
        'supportsImageInput': True,
        'supportsMask': False,
        'supportsMultiImage': False,
        'supportsControlImage': False,
        'supportsLayers': True,
        'supportsLora': True,
        'offloadSupport': MODULAR_OFFLOAD_SUPPORT,
        'lowVram': {'dtype': 'bfloat16', 'quantizationMode': 'bnb_4bit', 'autoOffload': True, 'offloadMode': OFFLOAD_MODE_MODEL_CPU, 'steps': 30},
        'modes': ['layer_decomposition'],
        'executionStatus': 'supported_with_model',
    },
    'WanVACEPipeline': {
        'modelType': 'WanVACEPipeline',
        'label': 'Wan VACE 1.3B',
        'displayName': 'Wan2.1-VACE-1.3B-diffusers',
        'family': 'Wan Video',
        'defaultRepo': 'Wan-AI/Wan2.1-VACE-1.3B-diffusers',
        'artifactLabel': 'Diffusers repo',
        'defaultDtype': 'bfloat16',
        'defaultSize': {'width': 832, 'height': 480, 'aspectRatio': '16:9'},
        'recommendedSteps': 30,
        'recommendedGuidance': 5.0,
        'guidanceLabel': 'Guidance',
        'supportsImageInput': True,
        'supportsMask': True,
        'supportsMultiImage': True,
        'supportsControlImage': True,
        'supportsLayers': False,
        'supportsLora': True,
        'supportsVideoInput': True,
        'supportsVideoMask': True,
        'outputKind': 'video',
        'recommendedFrames': 81,
        'recommendedFps': 16,
        'conditioningScale': 1.0,
        'offloadSupport': DIRECT_OFFLOAD_SUPPORT,
        'lowVram': {'dtype': 'bfloat16', 'autoOffload': True, 'offloadMode': OFFLOAD_MODE_MODEL_CPU, 'steps': 24, 'width': 832, 'height': 480, 'numFrames': 49},
        'modes': [
            'text_to_video',
            'image_to_video',
            'video_to_video',
            'video_inpaint',
            'video_outpaint',
            'reference_to_video',
            'control_to_video',
            'video_color_edit',
        ],
        'executionStatus': 'supported_with_model',
        'notes': [
            'Wan VACE is exposed as a direct Diffusers pipeline because this runtime does not expose a WanVACEModularPipeline.',
            'Exact color correction is provided by deterministic Video Color nodes; Wan VACE color edits are generative.',
        ],
        'modeRequirements': {
            'image_to_video': {'requiredImages': ['referenceImages'], 'note': 'Requires at least one starting/reference image.'},
            'video_to_video': {'requiredVideos': ['sourceVideo'], 'note': 'Requires one source video.'},
            'video_inpaint': {'requiredVideos': ['sourceVideo', 'maskVideo'], 'note': 'Requires source video and matching mask video.'},
            'video_outpaint': {'requiredVideos': ['sourceVideo', 'maskVideo'], 'note': 'Requires source video and boundary/generation mask video.'},
            'reference_to_video': {'requiredImages': ['referenceImages'], 'note': 'Requires one or more reference images.'},
            'control_to_video': {'requiredVideos': ['controlVideo'], 'note': 'Requires a prepared control video.'},
            'video_color_edit': {'requiredVideos': ['sourceVideo'], 'note': 'Requires one source video.'},
        },
    },
    'AceStepAudioPipeline': {
        'modelType': 'AceStepAudioPipeline',
        'label': 'ACE-Step Audio',
        'displayName': 'acestep-v15-xl-turbo-diffusers',
        'family': 'ACE Audio',
        'defaultRepo': 'ACE-Step/acestep-v15-xl-turbo-diffusers',
        'artifactLabel': 'Diffusers repo',
        'defaultDtype': 'bfloat16',
        'defaultSize': {'width': 0, 'height': 0, 'aspectRatio': 'custom'},
        'recommendedSteps': 8,
        'recommendedGuidance': 1.0,
        'guidanceLabel': 'Guidance',
        'supportsImageInput': False,
        'supportsMask': False,
        'supportsMultiImage': False,
        'supportsControlImage': False,
        'supportsLayers': False,
        'supportsLora': False,
        'supportsAudioInput': True,
        'outputKind': 'audio',
        'recommendedSampleRate': 48000,
        'recommendedDuration': 30,
        'offloadSupport': DIRECT_OFFLOAD_SUPPORT,
        'lowVram': {'dtype': 'bfloat16', 'autoOffload': True, 'offloadMode': OFFLOAD_MODE_MODEL_CPU, 'steps': 8},
        'modes': ['text_to_audio', 'audio_variation', 'audio_continuation', 'audio_repaint'],
        'executionStatus': 'supported_with_model',
        'modeRequirements': {
            'audio_variation': {'requiredAudio': ['sourceAudio'], 'note': 'Requires a source audio clip.'},
            'audio_continuation': {'requiredAudio': ['sourceAudio'], 'note': 'Requires a source audio clip to continue.'},
            'audio_repaint': {'requiredAudio': ['sourceAudio'], 'note': 'Requires source audio plus repaint timing.'},
        },
    },
    'FluxSchnellPipeline': {
        'modelType': 'FluxSchnellPipeline',
        'label': 'FLUX.1 schnell',
        'displayName': 'FLUX.1-schnell',
        'family': 'FLUX Image',
        'defaultRepo': 'black-forest-labs/FLUX.1-schnell',
        'artifactLabel': 'Diffusers repo',
        'defaultDtype': 'bfloat16',
        'defaultSize': {'width': 1024, 'height': 1024, 'aspectRatio': '1:1'},
        'recommendedSteps': 4,
        'recommendedGuidance': 0.0,
        'guidanceLabel': 'Guidance',
        'supportsImageInput': False,
        'supportsMask': False,
        'supportsMultiImage': False,
        'supportsControlImage': False,
        'supportsLayers': False,
        'supportsLora': True,
        'offloadSupport': DIRECT_OFFLOAD_SUPPORT,
        'lowVram': {'dtype': 'bfloat16', 'autoOffload': True, 'offloadMode': OFFLOAD_MODE_MODEL_CPU, 'steps': 4, 'width': 1024, 'height': 1024},
        'modes': ['text_to_image'],
        'executionStatus': 'supported_with_model',
    },
    'FluxDevPipeline': {
        'modelType': 'FluxDevPipeline',
        'label': 'FLUX.1 dev',
        'displayName': 'FLUX.1-dev',
        'family': 'FLUX Image',
        'defaultRepo': 'black-forest-labs/FLUX.1-dev',
        'alternateArtifact': 'black-forest-labs/FLUX.1-dev-FP8',
        'artifactLabel': 'Diffusers repo',
        'defaultDtype': 'bfloat16',
        'defaultSize': {'width': 768, 'height': 768, 'aspectRatio': '1:1'},
        'recommendedSteps': 20,
        'recommendedGuidance': 3.5,
        'guidanceLabel': 'Guidance',
        'supportsImageInput': False,
        'supportsMask': False,
        'supportsMultiImage': False,
        'supportsControlImage': False,
        'supportsLayers': False,
        'supportsLora': True,
        'offloadSupport': DIRECT_OFFLOAD_SUPPORT,
        'lowVram': {'dtype': 'bfloat16', 'autoOffload': True, 'offloadMode': OFFLOAD_MODE_GROUP_DISK, 'steps': 20, 'width': 768, 'height': 768},
        'modes': ['text_to_image'],
        'executionStatus': 'supported_with_model',
        'notes': ['Auto prefers the FP8 artifact on 16 GB CUDA when available.'],
    },
    'FluxKreaPipeline': {
        'modelType': 'FluxKreaPipeline',
        'label': 'FLUX.1 Krea dev',
        'displayName': 'FLUX.1-Krea-dev',
        'family': 'FLUX Image',
        'defaultRepo': 'black-forest-labs/FLUX.1-Krea-dev',
        'artifactLabel': 'Diffusers repo',
        'defaultDtype': 'bfloat16',
        'defaultSize': {'width': 1024, 'height': 1024, 'aspectRatio': '1:1'},
        'recommendedSteps': 28,
        'recommendedGuidance': 3.5,
        'guidanceLabel': 'Guidance',
        'supportsImageInput': False,
        'supportsMask': False,
        'supportsMultiImage': False,
        'supportsControlImage': False,
        'supportsLayers': False,
        'supportsLora': True,
        'offloadSupport': DIRECT_OFFLOAD_SUPPORT,
        'lowVram': {'dtype': 'bfloat16', 'autoOffload': True, 'offloadMode': OFFLOAD_MODE_GROUP_DISK, 'steps': 20, 'width': 768, 'height': 768},
        'modes': ['text_to_image'],
        'executionStatus': 'expert_only',
    },
    'FluxKontextPipeline': {
        'modelType': 'FluxKontextPipeline',
        'label': 'FLUX.1 Kontext dev',
        'displayName': 'FLUX.1-Kontext-dev',
        'family': 'FLUX Image',
        'defaultRepo': 'black-forest-labs/FLUX.1-Kontext-dev',
        'alternateArtifact': 'black-forest-labs/FLUX.1-Kontext-dev-NVFP4',
        'artifactLabel': 'Diffusers repo',
        'defaultDtype': 'bfloat16',
        'defaultSize': {'width': 1024, 'height': 1024, 'aspectRatio': '1:1'},
        'recommendedSteps': 28,
        'recommendedGuidance': 3.5,
        'guidanceLabel': 'Guidance',
        'supportsImageInput': True,
        'supportsMask': False,
        'supportsMultiImage': False,
        'supportsControlImage': False,
        'supportsLayers': False,
        'supportsLora': True,
        'offloadSupport': DIRECT_OFFLOAD_SUPPORT,
        'lowVram': {'dtype': 'bfloat16', 'autoOffload': True, 'offloadMode': OFFLOAD_MODE_GROUP_DISK, 'steps': 20, 'width': 768, 'height': 768},
        'modes': ['edit_image'],
        'executionStatus': 'expert_only',
        'modeRequirements': {'edit_image': {'requiredImages': ['referenceImages'], 'note': 'Requires a source image.'}},
    },
    'FluxFillPipeline': {
        'modelType': 'FluxFillPipeline',
        'label': 'FLUX.1 Fill dev',
        'displayName': 'FLUX.1-Fill-dev',
        'family': 'FLUX Image',
        'defaultRepo': 'black-forest-labs/FLUX.1-Fill-dev',
        'artifactLabel': 'Diffusers repo',
        'defaultDtype': 'bfloat16',
        'defaultSize': {'width': 1024, 'height': 1024, 'aspectRatio': '1:1'},
        'recommendedSteps': 28,
        'recommendedGuidance': 3.5,
        'guidanceLabel': 'Guidance',
        'supportsImageInput': True,
        'supportsMask': True,
        'supportsMultiImage': False,
        'supportsControlImage': False,
        'supportsLayers': False,
        'supportsLora': True,
        'offloadSupport': DIRECT_OFFLOAD_SUPPORT,
        'lowVram': {'dtype': 'bfloat16', 'autoOffload': True, 'offloadMode': OFFLOAD_MODE_GROUP_DISK, 'steps': 20, 'width': 768, 'height': 768},
        'modes': ['inpaint', 'outpaint'],
        'executionStatus': 'expert_only',
        'modeRequirements': {'inpaint': {'requiredImages': ['referenceImages', 'maskImage'], 'note': 'Requires source and mask images.'}},
    },
    'FluxDepthPipeline': {
        'modelType': 'FluxDepthPipeline',
        'label': 'FLUX.1 Depth dev',
        'displayName': 'FLUX.1-Depth-dev',
        'family': 'FLUX Image',
        'defaultRepo': 'black-forest-labs/FLUX.1-Depth-dev',
        'artifactLabel': 'Diffusers repo',
        'defaultDtype': 'bfloat16',
        'defaultSize': {'width': 1024, 'height': 1024, 'aspectRatio': '1:1'},
        'recommendedSteps': 28,
        'recommendedGuidance': 3.5,
        'guidanceLabel': 'Guidance',
        'supportsImageInput': True,
        'supportsMask': False,
        'supportsMultiImage': False,
        'supportsControlImage': True,
        'supportsLayers': False,
        'supportsLora': True,
        'offloadSupport': DIRECT_OFFLOAD_SUPPORT,
        'lowVram': {'dtype': 'bfloat16', 'autoOffload': True, 'offloadMode': OFFLOAD_MODE_GROUP_DISK, 'steps': 20, 'width': 768, 'height': 768},
        'modes': ['control_image'],
        'executionStatus': 'expert_only',
    },
    'FluxCannyPipeline': {
        'modelType': 'FluxCannyPipeline',
        'label': 'FLUX.1 Canny dev',
        'displayName': 'FLUX.1-Canny-dev',
        'family': 'FLUX Image',
        'defaultRepo': 'black-forest-labs/FLUX.1-Canny-dev',
        'artifactLabel': 'Diffusers repo',
        'defaultDtype': 'bfloat16',
        'defaultSize': {'width': 1024, 'height': 1024, 'aspectRatio': '1:1'},
        'recommendedSteps': 28,
        'recommendedGuidance': 3.5,
        'guidanceLabel': 'Guidance',
        'supportsImageInput': True,
        'supportsMask': False,
        'supportsMultiImage': False,
        'supportsControlImage': True,
        'supportsLayers': False,
        'supportsLora': True,
        'offloadSupport': DIRECT_OFFLOAD_SUPPORT,
        'lowVram': {'dtype': 'bfloat16', 'autoOffload': True, 'offloadMode': OFFLOAD_MODE_GROUP_DISK, 'steps': 20, 'width': 768, 'height': 768},
        'modes': ['control_image'],
        'executionStatus': 'expert_only',
    },
    'FluxReduxPipeline': {
        'modelType': 'FluxReduxPipeline',
        'label': 'FLUX.1 Redux dev',
        'displayName': 'FLUX.1-Redux-dev',
        'family': 'FLUX Image',
        'defaultRepo': 'black-forest-labs/FLUX.1-Redux-dev',
        'artifactLabel': 'Diffusers repo',
        'defaultDtype': 'bfloat16',
        'defaultSize': {'width': 1024, 'height': 1024, 'aspectRatio': '1:1'},
        'recommendedSteps': 28,
        'recommendedGuidance': 3.5,
        'guidanceLabel': 'Guidance',
        'supportsImageInput': True,
        'supportsMask': False,
        'supportsMultiImage': False,
        'supportsControlImage': False,
        'supportsLayers': False,
        'supportsLora': True,
        'offloadSupport': DIRECT_OFFLOAD_SUPPORT,
        'lowVram': {'dtype': 'bfloat16', 'autoOffload': True, 'offloadMode': OFFLOAD_MODE_GROUP_DISK, 'steps': 20, 'width': 768, 'height': 768},
        'modes': ['edit_image'],
        'executionStatus': 'expert_only',
    },
}

class WebServer:
    def __init__(
            self,
            modules: dict = {},
            host: str = '127.0.0.1',
            port: int = 8088,
            secure: bool = False,
            certfile: str = None,
            keyfile: str = None,
            cors: bool = False,
            cors_routes: list = [],
            client_max_size: int = 1024**4,
            work_dir: str = 'data',
            data_dir: str = 'data'
        ):
        self.instance = nanoid.generate(size=10)

        self.modules = modules
        self.ws_sessions = {}
        self.pending_ws_requests = {}

        self.interrupt_flag = False
        self.node_cache = {}

        self.queued_tasks = {}
        self.current_task = {}
        self.recent_tasks = []

        self.main_queue = asyncio.Queue()
        self.background_queue = asyncio.Queue()
        self._shutdown_event = asyncio.Event()
        self.studio_history_lock = asyncio.Lock()
        self.hf_download_semaphore = asyncio.Semaphore(2)
        self.hf_download_tasks = {}

        self.main_worker_task = None
        self.background_worker_task = None
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
        self.app = web.Application(client_max_size=self.client_max_size)

        # set up the routes
        self.app.add_routes([
            web.static('/assets', 'web/assets', append_version=True),
            web.static('/template-gallery', 'web/template-gallery', append_version=True),
            web.get('/', self.index),
            web.get('/favicon.ico', self.favicon),
            web.get('/ws', self.websocket),
            web.get(r'/nodes{id:/?([\w\d_-]+/[\w\d_-]+)?}', self.nodes),
            web.post('/fields/action', self.field_action),
            web.get('/cache/{node}/{field}', self.cache),
            web.get('/cache/{node}/{field}/{index}', self.cache),
            web.delete('/cache', self.delete_cache),
            web.get('/listdir', self.listdir),
            web.get('/listgraphs', self.listgraphs),
            web.get('/file', self.fileGet),
            web.post('/file', self.filePost),
            web.get('/preview', self.preview),
            web.post('/graph', self.graph),
            web.get('/queue', self.get_queue),
            web.delete('/queue/{task_id}', self.delete_task),
            web.get('/stop', self.stop_execution),
            web.get('/health', self.runtime_status),
            web.get('/runtime/status', self.runtime_status),
            web.get('/system_stats', self.system_stats),
            web.get('/runtime/gpu_processes', self.runtime_gpu_processes),
            web.post('/runtime/gpu_cleanup', self.runtime_gpu_cleanup),
            web.get('/model_capabilities', self.model_capabilities),
            web.post('/auto_resource/plan', self.auto_resource_plan),
            web.post('/auto_resource/plans', self.auto_resource_plans),
            web.get('/auto_resource/history', self.auto_resource_history),
            web.delete('/auto_resource/history', self.auto_resource_history_clear),
            web.get('/model_fingerprints', self.model_fingerprints),
            web.get('/local_models', self.local_models),
            web.get('/hf_cache', self.hf_cache),
            web.post('/hf_token', self.hf_token),
            web.get('/model_cache/diagnostics', self.model_cache_diagnostics),
            web.get('/custom_modules', self.custom_modules_list),
            web.post('/custom_modules/refresh', self.custom_modules_refresh),
            web.post('/custom_modules/install', self.custom_modules_install),
            web.post('/custom_modules/{name}/update', self.custom_modules_update),
            web.post('/custom_modules/{name}/disable', self.custom_modules_disable),
            web.post('/custom_modules/{name}/enable', self.custom_modules_enable),
            web.get('/studio_outputs', self.studio_outputs_get),
            web.post('/studio_outputs', self.studio_outputs_post),
            web.patch('/studio_outputs/{output_id}', self.studio_outputs_patch),
            web.delete('/studio_outputs/{output_id}', self.studio_outputs_delete),
            web.get('/studio/blocks', self.studio_blocks_get),
            web.post('/studio/blocks', self.studio_blocks_post),
            web.get('/studio/blocks/{block_id}', self.studio_block_get),
            web.delete('/studio/blocks/{block_id}', self.studio_block_delete),
            web.get('/workflow_shares', self.workflow_shares_list),
            web.post('/workflows/share', self.workflow_share_post),
            web.get('/workflows/share/{share_id}/media/{filename}', self.workflow_share_media_get),
            web.get('/workflows/share/{share_id}', self.workflow_share_get),
            web.delete('/hf_cache/{hash}', self.hf_cache_delete),
            web.get('/hf_hub', self.hf_hub),
            web.get('/hf_download', self.hf_download),
            web.get('/static/{module}/{file}', self.user_assets),
            web.get('/stream', self.stream)
        ])

        # serve the user assets
        try:
            self.app.add_routes(web.static('/user', 'web/user', append_version=True))
        except Exception as e:
            pass

        # set up the cors routes
        if cors:
            cors = cors_setup(self.app, defaults={
                cors_route: ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*")
                for cors_route in cors_routes
            })
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
                    results = await asyncio.wait_for(
                        asyncio.gather(*close_coroutines, return_exceptions=True),
                        timeout=0.5
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Websocket connections did not close within timeout, forcing shutdown")

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
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_wait, return_exceptions=True),
                    timeout=2.0
                )
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
        return {
            "task_id": self.current_task["task_id"],
            "name": self.current_task["name"],
            "sid": self.current_task["sid"],
            "started_at": self.current_task["started_at"],
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
            "elapsed_seconds": self.current_task.get("elapsed_seconds"),
            "average_step_seconds": self.current_task.get("average_step_seconds"),
            "eta_seconds": self.current_task.get("eta_seconds"),
            "runtimeFingerprint": self.current_task.get("runtimeFingerprint"),
            "deterministicMode": self.current_task.get("deterministicMode"),
        }

    def _record_terminal_task(self, status, *, error_payload=None):
        if not self.current_task:
            return None
        completed_at = time.time()
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
            "elapsed_seconds": self.current_task.get("elapsed_seconds"),
            "average_step_seconds": self.current_task.get("average_step_seconds"),
            "eta_seconds": self.current_task.get("eta_seconds"),
            "runtimeFingerprint": self.current_task.get("runtimeFingerprint"),
        }
        if isinstance(error_payload, dict):
            for key in ("message", "error", "exception_type", "category", "error_code", "recovery_hint", "node", "node_name"):
                if error_payload.get(key) is not None:
                    entry[key] = error_payload.get(key)
        self.recent_tasks = [entry, *[item for item in self.recent_tasks if item.get("task_id") != entry["task_id"]]][:30]
        return entry

    def record_node_progress(self, payload):
        if not self.current_task or not isinstance(payload, dict):
            return payload
        task_id = payload.get("task_id")
        if task_id and task_id != self.current_task.get("task_id"):
            return payload
        now = time.time()
        node_progress = payload.get("progress")
        self.current_task.update({
            "updated_at": now,
            "current_node": payload.get("node") or self.current_task.get("current_node"),
            "node_progress": node_progress,
            "phase": payload.get("phase") or self.current_task.get("phase"),
            "message": payload.get("message") or self.current_task.get("message"),
        })
        # Node lifecycle events without step metrics must not erase the latest
        # denoising sample. Preserving the final sample gives reconnects and
        # terminal receipts an honest duration/step record through decode/save.
        for field in (
            "current_step",
            "total_steps",
            "elapsed_seconds",
            "average_step_seconds",
            "eta_seconds",
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
        return payload

    def _get_queue(self):
        #task_list_sorted = {k: v for k, v in sorted(self.queued_tasks.items(), key=lambda x: x[1]['queued_at'], reverse=True)}
        # filter out keys that are not needed for the client
        queued_tasks = {
            k: {
                'name': v['name'],
                'sid': v['sid'],
                'queued_at': v['queued_at'],
                'task_id': k,
                'queue_position': index + 1,
            }
            for index, (k, v) in enumerate(self.queued_tasks.items())
        }

        current_task = self._current_task_snapshot()

        return queued_tasks, current_task

    async def queue_task(self, task, args, future, sid, name=None):
        task_id = nanoid.generate(size=12)
        task_name = name or f'Unnamed task ({task.__name__})'

        self.queued_tasks[task_id] = {
            'task': task,
            'args': args,
            'future': future,
            "sid": sid,
            "queued_at": time.time(),
            "name": task_name,
        }
        await self.main_queue.put((task, args, future, task_id))

        task_list, current_task = self._get_queue()

        self.queue_message({
            "type": "task_queued",
            "task_id": task_id,
            "sid": sid,
            "queued": task_list,
            "current": current_task,
        })

        return task_id

    async def get_queue(self, _):
        """
        HTTP endpoint to return the tasks queue and the current task.
        """
        task_list, current_task = self._get_queue()
        return web.json_response({
            "queued": task_list,
            "current": current_task,
            "recent": self.recent_tasks,
        })

    async def delete_task(self, request):
        """
        HTTP endpoint to delete a task from the queue.
        """
        task_id = request.match_info.get('task_id')
        if task_id in self.queued_tasks:
            task = self.queued_tasks.pop(task_id)
            logger.info(f"Task {task_id} {task['name']} deleted from queue.")
            task_list, current_task = self._get_queue()
            self.queue_message({
                "type": "task_cancelled",
                "task_id": task_id,
                "queued": task_list,
                "current": current_task,
            })
            return web.json_response({"error": False, "task_id": task_id, "queued": task_list, "current": current_task})
        elif self.current_task and self.current_task["task_id"] == task_id:
            return web.json_response({"error": True, "message": f"Task is already running and cannot be cancelled.", "task_id": task_id}, status=400)

        return web.json_response({"error": True, "message": f"Task not found in queue.", "task_id": task_id}, status=404)

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

                    self.current_task = {
                        "task_id": task_id,
                        "task": task,
                        "name": current_task["name"],
                        "sid": current_task["sid"],
                        "started_at": time.time(),
                        "progress": 0,
                        "attempt_index": 0,
                        "args": args,
                    }
                    task_list, current_task = self._get_queue()
                    self.queue_message({
                        "type": "task_started",
                        "task_id": task_id,
                        "attempt_index": 0,
                        "queued": task_list,
                        "current": current_task,
                    })
                    terminal_status = "completed"
                    failure_payload = None
                    try:
                        if isinstance(args, tuple):
                            result = await self.loop.run_in_executor(None, partial(task, *args))
                        elif isinstance(args, dict):
                            result = await self.loop.run_in_executor(None, partial(task, **args))
                        else:
                            result = await self.loop.run_in_executor(None, partial(task, args))

                        if self.current_task and self.current_task.get("interrupt_requested"):
                            terminal_status = "cancelled"
                        if future and terminal_status == "completed":
                            future.set_result(result)
                        elif future and terminal_status == "cancelled":
                            future.set_exception(asyncio.CancelledError("Execution interrupted by the user."))
                    except Exception as e:
                        terminal_status = "failed"
                        traceback_text = (
                            getattr(e, 'modiff_traceback', None)
                            or getattr(e, 'mellon_traceback', None)
                            or traceback.format_exc()
                        )
                        logger.error(f"Error occurred in {traceback_text}")
                        task_list, _ = self._get_queue()
                        failure_payload = self._exception_payload(
                            e,
                            task_id=task_id,
                            sid=self.current_task["sid"] if self.current_task else None,
                            node_id=getattr(e, 'modiff_node_id', None) or getattr(e, 'mellon_node_id', None),
                            node_name=getattr(e, 'modiff_node_name', None) or getattr(e, 'mellon_node_name', None),
                            traceback_text=traceback_text,
                        )
                        self._record_auto_resource_failure(e, {
                            'category': failure_payload.get('category'),
                            'error_code': failure_payload.get('error_code'),
                            'message': failure_payload.get('message'),
                            'recovery_hint': failure_payload.get('recovery_hint'),
                        })
                        if future:
                            future.set_exception(e)
                    finally:
                        if self.current_task:
                            task_sid = self.current_task.get("sid")
                            task_name = self.current_task.get("name")
                            attempt_index = self.current_task.get("attempt_index")
                            terminal_entry = self._record_terminal_task(terminal_status, error_payload=failure_payload)
                            task_list, _ = self._get_queue()
                            terminal_message = {
                                "type": "task_completed" if terminal_status == "completed" else "task_cancelled" if terminal_status == "cancelled" else "task_failed",
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
                            self.queue_message(terminal_message, task_sid)
                            self.current_task = None
                        self.main_queue.task_done()
                        self.interrupt_flag = False

                except asyncio.TimeoutError:
                    # Timeout is expected, just continue to check shutdown
                    continue

        except Exception as e:
            logger.error(f"Main worker error: {e}")
        finally:
            logger.debug("Main worker shutting down")


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
                        #logger.error(f"Error occurred in {traceback.format_exc()}")
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
        response = web.FileResponse('web/index.html')
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    async def favicon(self, _):
        return web.FileResponse('web/favicon.ico')

    async def user_assets(self, request):
        module = request.match_info.get('module')
        file = request.match_info.get('file')
        fileName = f"custom/{module}/web/{file}"

        if not Path(fileName).exists():
            return web.HTTPNotFound(text='File not found')

        response = web.FileResponse(fileName)
        #response.headers["Content-Type"] = "application/javascript"
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

        return response


    """
    ╭─────────────────────────╮
       Nodes & Fields Routes
    ╰─────────────────────────╯
    """
    async def nodes(self, request):
        id = request.match_info.get('id', '').strip('/')
        modules = self.modules

        if id:
            m, n = id.split('/')
            if m not in modules:
                return web.json_response({"error": f"The module {m} was not found. Try refreshing the page and restarting the server."}, status=404)
            if n not in modules[m]:
                return web.json_response({"error": f"The node {n} was not found in the module {m}. Try refreshing the page and restarting the server."}, status=404)
            modules = {m: {n: modules[m][n]}}

        output = {}
        for module, actions in modules.items():
            for action, values in actions.items():
                params = deepcopy(values.get('params', {}))
                #spawn_fields = []
                for p in params:
                    if 'postProcess' in params[p]:
                        del params[p]["postProcess"]
                    #if 'spawn' in params[p] and params[p]['spawn']:
                    #    spawn_fields.append(p)
                # spawn fields are identified by the '>>>' suffix in the field key
                #for p in spawn_fields:
                #    params[f"{p}>>>0"] = params[p]
                #    del params[p]

                output[f"{module}.{action}"] = {
                    'module': module,
                    'action': action,
                    'type': values.get('type', 'custom'),
                    'label': values.get('label', f"{module}: {action}"),
                    'category': values.get('category', 'default'),
                    'description': values.get('description', ''),
                    'resizable': values.get('resizable', False),
                    'skipParamsCheck': values.get('skipParamsCheck', False),
                    'style': values.get('style', ''),
                    'params': params,
                    'time': [0,0,0],
                    'memory': [0,0,0],
                    'cache': False,
                }

        return web.json_response({
            'instance': self.instance,
            'nodes': output
        })

    async def field_action(self, request):
        data = await request.json()
        node = data.get('node')
        sid = data.get('sid')
        fn = data.get('fn')
        values = data.get('values')
        key = data.get('fieldKey', None)
        queue = data.get('queue', False)

        if node not in self.node_cache:
            module = data.get('module')
            action = data.get('action')
            work_module = import_module(f"{module}.main")
            work_action = getattr(work_module, action)
            work_action = work_action(node_id=node)
            self.node_cache[node] = work_action

        self.node_cache[node]._sid = sid # always update the sid as it may change over time

        fn = getattr(self.node_cache[node], fn)
        ref = {
            "node": node,
            "key": key,
            "queue": queue,
        }

        if queue:
            task_id = await self.queue_task(fn, (values, ref), None, sid, name=f"Field action")
        else:
            # Run field action in executor to avoid blocking the event loop
            if not getattr(self, 'loop', None):
                self.loop = asyncio.get_event_loop()
            try:
                await self.loop.run_in_executor(None, partial(fn, values, ref))
            except Exception as e:
                logger.error(f"Error executing field action synchronously: {e}")
                return web.json_response({
                    "error": True,
                    "message": f"Field action error: {e}",
                    "sid": sid,
                    "ref": ref,
                }, status=500)
            task_id = None

        return web.json_response({
            "error": False,
            "message": f"Field action `{fn}` for node `{node}` queued for processing",
            "sid": sid,
            "task_id": task_id,
            "ref": ref,
        })

    """
    ╭─────────────────────╮
       Node Cache Routes
    ╰─────────────────────╯
    """

    async def cache(self, request):
        node = request.match_info.get('node')
        field = request.match_info.get('field')
        index = request.match_info.get('index', None)

        if node not in self.node_cache:
            return web.HTTPNotFound(text=f"Node {node} not found in cache.")

        # get the actual value from the node cache
        if field in self.node_cache[node].output:
            data = self.node_cache[node].output[field]
        elif field in self.node_cache[node].params:
            data = self.node_cache[node].params[field]
        else:
            return web.HTTPNotFound(text=f"Field {field} not found in node {node} cache.")

        if data is None:
            return web.HTTPNotFound(text=f"Field {field} is empty in node {node} cache.")

        if isinstance(data, list):
            index = max(0, min(len(data) - 1, int(index))) if index else 0
            data = data[index]

        # check the registry for the type of the field
        module = self.node_cache[node].module_name
        action = self.node_cache[node].class_name
        type = self.modules[module][action]['params'][field].get('type')

        filename = request.query.get('filename', f"{field}")

        charset = None
        if is_image_data_type(type):
            format = request.query.get('format', 'WEBP').upper()
            quality = request.query.get('quality', 100)
            out = to_bytes(type, data, {'format': format, 'quality': quality})
            content_type = f'image/{format.lower()}'
            if not str(filename).lower().endswith(f".{format.lower()}"):
                filename = f"{filename}.{format.lower()}"
        elif type == 'text' or any(t.startswith('str') for t in type):
            out = str(data).encode('utf-8')
            content_type = f'text/plain'
            charset = 'utf-8'
            filename = f"{filename}.txt"
        else:
            resp = web.FileResponse(data)
            resp.headers['Content-Disposition'] = f'inline; filename="{filename}"'
            resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            resp.headers['Pragma'] = 'no-cache'
            resp.headers['Expires'] = '0'
            return resp

        return web.Response(
            body=out,
            content_type=content_type,
            charset=charset,
            headers={
                'Content-Disposition': f'inline; filename="{filename}"',
                'Content-Length': str(len(out)),
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        )

    async def delete_cache(self, request):
        data = await request.json()
        nodes = data.get('nodes', [])

        if isinstance(nodes, str):
            nodes = list(self.node_cache.keys()) if nodes == '*' else [nodes]

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

    async def listdir(self, request):
        req_basepath = self.data_dir if request.query.get('basepath') == 'data' else self.work_dir
        req_path = request.query.get('path', req_basepath)
        req_type = request.query.get('type', None)
        if req_type:
            req_type = [t.strip() for t in req_type.lower().split(',')]

        full_path = Path(req_path)
        if not full_path.is_absolute():
            full_path = Path(req_basepath) / full_path

        file_types = {
            'image': ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'ico', 'webp'],
            'audio': ['mp3', 'wav', 'ogg', 'm4a', 'flac', 'aac', 'wma', 'm4b', 'm4p', 'm4r'],
            'video': ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'mpeg', 'mpg', 'm4v', 'webm'],
            'text': ['txt', 'md', 'csv', 'json', 'xml', 'yaml', 'yml', 'ini', 'toml', 'cfg', 'conf', 'log', 'html', 'css', 'js', 'ts', 'py', 'rb', 'php', 'sql', 'sh', 'bash'],
            'archive': ['zip', 'rar', 'tar', 'gz', 'bz2', '7z'],
            '3d': ['glb', 'gltf', 'stl', 'obj', 'fbx', 'dae', 'ply', '3ds', 'max', 'blend'],
        }

        if not str(full_path).startswith(self.work_dir):
            return web.json_response({"error": f"Cannot access paths outside of {self.work_dir}."}, status=403)

        contents = {
            'files': [],
            'path': '',
            'abs_path': '',
        }

        try:
            if full_path.exists():
                contents['path'] = str(full_path.relative_to(self.work_dir))
                contents['abs_path'] = str(full_path)

                for item in full_path.iterdir():
                    suffix = item.suffix.lstrip('.').lower()
                    # if any of the requested types don't match the file type, skip it
                    if not item.is_dir() and req_type and not any(suffix in exts for ftype, exts in file_types.items() if ftype in req_type):
                        continue

                    file = {
                        'is_dir': item.is_dir(),
                        'is_hidden': item.name.startswith('.'), # TODO: Windows: bool(os.stat(item).st_mode & stat.FILE_ATTRIBUTE_HIDDEN),
                        'name': item.name,
                        'path': str(item.relative_to(self.work_dir)),
                        #'abs_path': str(item),
                        'modified': item.stat().st_mtime,
                        'size': None,
                        'ext': None,
                        'type': None,
                    }
                    if not item.is_dir():
                        file['size'] = item.stat().st_size
                        file['ext'] = suffix
                        file['type'] = next((ftype for ftype, exts in file_types.items() if suffix in exts), 'other')

                    contents['files'].append(file)

                return web.json_response(contents)
            else:
                return web.json_response({"error": f"The path {req_path} does not exist."}, status=404)

        except Exception as e:
            logger.error(f"Error listing directory: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def listgraphs(self, request):
        path = Path(self.data_dir) / 'graphs'
        if not path.exists():
            return web.json_response({"error": True, "message": "No graph directory found."}, status=404)

        graphs = list_files(str(path), recursive=True, extensions=['json'])

        # build a nested tree of directories and files
        tree: list[dict] = []

        for g in graphs:
            rel_dir = g.get('rel_directory') or '.'
            # normalize and split directory parts
            parts = [] if rel_dir in (None, '.', '') else [p for p in rel_dir.strip('/').split('/') if p]

            parent_children = tree
            current_path_parts: list[str] = []
            # ensure directory nodes exist for each part
            for part in parts:
                current_path_parts.append(part)
                node = next((n for n in parent_children if n.get('isDir') and n.get('name') == part), None)
                if not node:
                    node = {
                        "isDir": True,
                        "name": part,
                        "path": f"{str(path)}/{'/'.join(current_path_parts)}",
                        "children": []
                    }
                    parent_children.append(node)
                parent_children = node['children']

            raw_name = g.get('name') or Path(g.get('path', '')).name
            file_item = {
                "isDir": False,
                "name": Path(raw_name).stem,
                "path": g.get('path')
            }
            parent_children.append(file_item)

        # recursively sort directories (dirs first, then files) by name
        def sort_children(children: list[dict]):
            dirs = [c for c in children if c['isDir']]
            files = [c for c in children if not c['isDir']]
            dirs.sort(key=lambda x: x['name'].lower())
            files.sort(key=lambda x: x['name'].lower())
            for d in dirs:
                sort_children(d['children'])
            # mutate list in-place to preserve references
            children[:] = dirs + files

        sort_children(tree)

        return web.json_response(tree)

    async def fileGet(self, request):
        file = request.query.get('file')

        if not file:
            return web.json_response({"error": "Incorrect request, `file` is required."}, status=400)

        file_path = Path(file)
        if not file_path.is_absolute():
            file_path = Path(self.work_dir) / file_path

        if not file_path.exists():
            legacy_graph_root = Path(self.data_dir) / 'graphs' / 'mellon'
            modiff_graph_root = Path(self.data_dir) / 'graphs' / 'modiff'
            try:
                legacy_graph_relative_path = file_path.resolve(strict=False).relative_to(
                    legacy_graph_root.resolve(strict=False)
                )
            except ValueError:
                legacy_graph_relative_path = None

            if legacy_graph_relative_path is not None:
                migrated_file_path = modiff_graph_root / legacy_graph_relative_path
                if migrated_file_path.exists():
                    file_path = migrated_file_path

        if not file_path.exists():
            return web.json_response({"error": f"The file {file} does not exist."}, status=404)

        return web.FileResponse(file_path)

    async def filePost(self, request):
        data = await request.post()
        file = data.get('file')
        type = data.get('type', 'images')
        type = type if type in ['images', 'audio', 'videos', 'text', '3d'] else 'images'
        file_path = Path(self.data_dir) / type / file.filename

        if file_path.exists():
            file_path = file_path.with_name(f"{file_path.stem}_{nanoid.generate(size=6)}{file_path.suffix}")

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(file.file.read())

            return web.json_response({"error": False, "path": str(file_path.relative_to(self.work_dir))})
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def preview(self, request):
        from utils.image import cover
        from PIL import Image
        from io import BytesIO

        Image.MAX_IMAGE_PIXELS = None

        file = request.query.get('file')
        if not file:
            return web.json_response({"error": "Incorrect request, `file` is required."}, status=400)

        file_path = Path(file)
        if not file_path.is_absolute():
            file_path = Path(self.work_dir) / file_path

        if not str(file_path).startswith(self.work_dir):
            return web.json_response({"error": f"Cannot access paths outside of {self.work_dir}."}, status=403)

        if not file_path.exists():
            return web.json_response({"error": f"The file {file} does not exist."}, status=404)

        if not file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.ico', '.webp']:
            return web.json_response({"error": f"The file {file} is not an image."}, status=400)

        width = int(request.query.get('width', 0))
        height = int(request.query.get('height', 0))
        format = request.query.get('format', 'jpeg')
        format = format.lower()
        quality = int(request.query.get('quality', 95))

        image = Image.open(file_path)

        if width > 0 or height > 0:
            width = min(2048, width) if width > 0 else min(2048, height)
            height = min(2048, height) if height > 0 else min(2048, width)
        else:
            width = min(2048, image.width)
            height = min(2048, image.height)

        if width != image.width or height != image.height:
            image = cover(image, width, height, resample='BICUBIC')

        if image.mode != 'RGB' and format in ['jpeg', 'jpg', 'bmp', 'ico']:
            image = image.convert('RGB')

        bytes = BytesIO()
        image.save(bytes, format=format.upper(), quality=quality)
        bytes = bytes.getvalue()
        return web.Response(
            body=bytes,
            content_type=f'image/{format.lower()}',
            headers={
                'Content-Disposition': f'inline; filename="{file_path.name}"',
                'Content-Length': str(len(bytes)),
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        )

    async def stream(self, request):
        file = request.query.get('file')
        if not file:
            return web.json_response({"error": "Incorrect request, `file` is required."}, status=400)

        file_path = Path(file)
        if not file_path.is_absolute():
            file_path = Path(self.work_dir) / file_path

        if not str(file_path).startswith(self.work_dir):
            return web.json_response({"error": f"Cannot access paths outside of {self.work_dir}."}, status=403)

        if not file_path.exists():
            return web.json_response({"error": f"The file {file} does not exist."}, status=404)

        resp = web.FileResponse(file_path)
        resp.headers['Content-Disposition'] = f'inline; filename="{file_path.name}"'
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'

        return resp


    async def local_models(self, request):
        refresh = request.query.get('refresh', False)
        path_match = request.query.get('match', "")

        if refresh:
            modelstore.update_local()

        files = modelstore.get_local_ids(name=path_match)

        return web.json_response(files)

    def _studio_history_file(self):
        return Path(self.data_dir) / 'studio' / 'outputs.json'

    def _studio_outputs_dir(self):
        return Path(self.data_dir) / 'studio' / 'outputs'

    def _studio_blocks_dir(self):
        return Path(self.data_dir) / 'studio' / 'blocks'

    def _safe_block_id(self, block_id=None):
        raw_id = str(block_id or nanoid.generate(size=12))
        safe_id = re.sub(r'[^a-zA-Z0-9_-]+', '_', raw_id).strip('_')[:80]
        return safe_id or nanoid.generate(size=12)

    def _studio_block_file(self, block_id):
        return self._studio_blocks_dir() / f"{self._safe_block_id(block_id)}.json"

    def _validate_studio_block(self, payload):
        if not isinstance(payload, dict):
            raise ValueError('User block must be a JSON object.')

        block_id = self._safe_block_id(payload.get('id'))
        name = str(payload.get('name') or 'User Block').strip()[:120] or 'User Block'
        version = payload.get('version', 1)
        if version != 1:
            raise ValueError('Unsupported user block version.')

        required_arrays = ['nodes', 'edges', 'inputs', 'outputs', 'exposedParams']
        for key in required_arrays:
            if not isinstance(payload.get(key), list):
                raise ValueError(f'User block field {key} must be a list.')

        block = dict(payload)
        block['id'] = block_id
        block['name'] = name
        block['version'] = 1
        block['nodes'] = payload['nodes']
        block['edges'] = payload['edges']
        block['inputs'] = payload['inputs']
        block['outputs'] = payload['outputs']
        block['exposedParams'] = payload['exposedParams']
        now = int(time.time() * 1000)
        block['createdAt'] = int(payload.get('createdAt') or now)
        block['updatedAt'] = int(payload.get('updatedAt') or now)
        return block

    def _read_studio_block(self, block_id):
        block_file = self._studio_block_file(block_id)
        if not block_file.exists():
            return None
        with open(block_file, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        return self._validate_studio_block(payload)

    def _write_studio_block(self, block):
        blocks_dir = self._studio_blocks_dir()
        blocks_dir.mkdir(parents=True, exist_ok=True)
        validated = self._validate_studio_block(block)
        target = self._studio_block_file(validated['id'])
        temp_file = target.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(validated, f, ensure_ascii=False)
        temp_file.replace(target)
        return validated, target

    def _list_studio_blocks(self):
        blocks_dir = self._studio_blocks_dir()
        if not blocks_dir.exists():
            return []

        blocks = []
        for block_file in blocks_dir.glob('*.json'):
            try:
                with open(block_file, 'r', encoding='utf-8') as f:
                    blocks.append(self._validate_studio_block(json.load(f)))
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
                logger.error(f"Error reading Studio user block {block_file}: {e}")
        return sorted(blocks, key=lambda block: block.get('updatedAt') or 0, reverse=True)

    def _read_studio_outputs(self):
        history_file = self._studio_history_file()
        if not history_file.exists():
            return []

        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            logger.error(f"Error reading Studio output history: {e}")
            return []

        if isinstance(payload, list):
            outputs = payload
        elif isinstance(payload, dict) and isinstance(payload.get('outputs'), list):
            outputs = payload['outputs']
        else:
            outputs = []

        return [output for output in outputs if isinstance(output, dict)]

    def _write_studio_outputs(self, outputs):
        history_file = self._studio_history_file()
        history_file.parent.mkdir(parents=True, exist_ok=True)
        bounded_outputs = outputs[:200]
        payload = {
            'version': 1,
            'updatedAt': int(time.time() * 1000),
            'outputs': bounded_outputs,
        }
        temp_file = history_file.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False)
        temp_file.replace(history_file)
        return bounded_outputs

    def _studio_output_key(self, output):
        return str(output.get('id') or f"{output.get('nodeId', '')}:{output.get('fieldKey', '')}:{output.get('url', '')}")

    def _hash_file(self, path):
        digest = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(chunk)
        return f"sha256:bytes:{digest.hexdigest()}"

    def _hash_collection(self, hashes):
        digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode('utf-8'))
        return f"sha256:collection:{digest.hexdigest()}"

    def _sort_studio_outputs(self, outputs):
        return sorted(outputs, key=lambda output: output.get('createdAt') or 0, reverse=True)

    def _merge_studio_outputs(self, existing, incoming):
        merged = {}
        order = []

        for output in incoming + existing:
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
                'favorite': output.get('favorite', previous.get('favorite', False)),
                'backendImagePath': output.get('backendImagePath') or previous.get('backendImagePath'),
                'backendMediaPath': output.get('backendMediaPath') or previous.get('backendMediaPath'),
                'backendSyncedAt': output.get('backendSyncedAt') or previous.get('backendSyncedAt'),
            }

        return self._sort_studio_outputs([merged[key] for key in order])

    def _save_studio_output_image(self, output):
        image_data = output.pop('image_data', None)
        if not image_data or output.get('backendImagePath'):
            return output

        try:
            header = ''
            payload = image_data
            if isinstance(image_data, str) and image_data.startswith('data:') and ',' in image_data:
                header, payload = image_data.split(',', 1)

            if not isinstance(payload, str):
                return output

            mime_type = header.split(';')[0].removeprefix('data:') if header else ''
            extension = {
                'image/webp': '.webp',
                'image/png': '.png',
                'image/jpeg': '.jpg',
                'image/jpg': '.jpg',
                'image/gif': '.gif',
            }.get(mime_type, '.webp')

            output_id = ''.join(ch for ch in str(output.get('id') or nanoid.generate(size=12)) if ch.isalnum() or ch in ('-', '_'))[:80]
            if not output_id:
                output_id = nanoid.generate(size=12)

            image_bytes = base64.b64decode(payload)
            output_dir = self._studio_outputs_dir()
            output_dir.mkdir(parents=True, exist_ok=True)
            image_path = output_dir / f"{output_id}{extension}"
            with open(image_path, 'wb') as f:
                f.write(image_bytes)

            try:
                image_file = str(image_path.relative_to(self.work_dir)).replace('\\', '/')
            except ValueError:
                image_file = str(image_path)
            output['backendImagePath'] = image_file
            output['url'] = f"/file?file={quote(image_file)}"
            output['backendSyncedAt'] = int(time.time() * 1000)
            output['mediaHash'] = self._hash_file(image_path)
        except Exception as e:
            logger.error(f"Error saving Studio output image: {e}")

        return output

    def _save_studio_output_media(self, output):
        output = self._save_studio_output_image(output)
        if output.get('backendMediaPath') or output.get('backendImagePath'):
            return output

        display_type = output.get('displayType')
        preview_url = output.get('url')
        if display_type != 'video' and not str(preview_url or '').lower().split('?')[0].endswith(('.mp4', '.webm', '.mov', '.mkv')):
            return output

        try:
            media = self._share_media_bytes_from_url(preview_url)
            if not media or not media.get('bytes'):
                return output

            content_type = media.get('contentType') or 'application/octet-stream'
            extension = self._content_type_extension(content_type, media.get('filename') or preview_url)
            if extension.lower() not in ('.mp4', '.webm', '.mov', '.mkv'):
                extension = '.mp4'

            output_id = ''.join(ch for ch in str(output.get('id') or nanoid.generate(size=12)) if ch.isalnum() or ch in ('-', '_'))[:80]
            if not output_id:
                output_id = nanoid.generate(size=12)

            output_dir = self._studio_outputs_dir()
            output_dir.mkdir(parents=True, exist_ok=True)
            media_path = output_dir / f"{output_id}{extension}"
            with open(media_path, 'wb') as f:
                f.write(media['bytes'])

            try:
                media_file = str(media_path.relative_to(self.work_dir)).replace('\\', '/')
            except ValueError:
                media_file = str(media_path)
            output['backendMediaPath'] = media_file
            output['url'] = f"/file?file={quote(media_file)}"
            output['backendSyncedAt'] = int(time.time() * 1000)
            output['mediaHash'] = self._hash_file(media_path)
        except Exception as e:
            logger.error(f"Error saving Studio output media: {e}")

        return output

    def _safe_studio_output_id(self, output):
        output_id = ''.join(ch for ch in str(output.get('id') or nanoid.generate(size=12)) if ch.isalnum() or ch in ('-', '_'))[:80]
        return output_id or nanoid.generate(size=12)

    def _save_studio_output_media_items(self, output):
        media_items = output.get('mediaItems')
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
            normalized['index'] = int(normalized.get('index', index) or index)
            if normalized.get('backendPath') and normalized.get('mediaHash'):
                saved_items.append(normalized)
                continue

            preview_url = normalized.get('url') or normalized.get('value')
            media = self._share_media_bytes_from_url(preview_url)
            if not media or not isinstance(media.get('bytes'), (bytes, bytearray)):
                saved_items.append(normalized)
                continue

            media_bytes = bytes(media['bytes'])
            content_type = media.get('contentType') or 'application/octet-stream'
            extension = self._content_type_extension(content_type, media.get('filename') or preview_url)
            filename = f"{output_id}_item_{normalized['index']:02d}{extension}"
            media_path = output_dir / filename
            with open(media_path, 'wb') as f:
                f.write(media_bytes)

            try:
                media_file = str(media_path.relative_to(self.work_dir)).replace('\\', '/')
            except ValueError:
                media_file = str(media_path)

            normalized['backendPath'] = media_file
            normalized['url'] = f"/file?file={quote(media_file)}"
            normalized['mediaHash'] = self._hash_file(media_path)
            normalized['contentType'] = content_type
            normalized['byteSize'] = len(media_bytes)
            saved_items.append(normalized)

        if saved_items:
            output['mediaItems'] = saved_items
            item_hashes = [item.get('mediaHash') for item in saved_items if item.get('mediaHash')]
            if item_hashes:
                output['mediaCollectionHash'] = self._hash_collection(item_hashes)
                if len(item_hashes) > 1:
                    output['mediaHash'] = output['mediaCollectionHash']

        return output

    def _normalize_studio_output(self, output):
        normalized = deepcopy(output)
        if not isinstance(normalized.get('id'), str) or not normalized.get('id'):
            normalized['id'] = nanoid.generate(size=12)
        if not normalized.get('createdAt'):
            normalized['createdAt'] = int(time.time() * 1000)
        if 'favorite' not in normalized:
            normalized['favorite'] = False
        normalized = self._save_studio_output_media(normalized)
        normalized = self._save_studio_output_media_items(normalized)
        provenance = normalized.get('provenance') if isinstance(normalized.get('provenance'), dict) else {}
        runtime_fingerprint = provenance.get('runtimeFingerprint')
        if not runtime_fingerprint and self.current_task:
            runtime_fingerprint = self.current_task.get('runtimeFingerprint')
        normalized['backendProvenance'] = {
            'schemaVersion': 1,
            'source': 'backend-record',
            'capturedAt': int(time.time() * 1000),
            'backendExecutionId': normalized.get('taskId') or normalized.get('runId'),
            'clientRunId': normalized.get('clientRunId'),
            'runInputHash': normalized.get('runInputHash'),
            'workflowTabId': normalized.get('workflowTabId'),
            'attemptIndex': normalized.get('attemptIndex'),
            'nodeId': normalized.get('nodeId'),
            'fieldKey': normalized.get('fieldKey'),
            'historyPath': str(self._studio_history_file()),
            'mediaPath': normalized.get('backendMediaPath') or normalized.get('backendImagePath'),
            'mediaHash': normalized.get('mediaHash'),
            'mediaCollectionHash': normalized.get('mediaCollectionHash'),
            'mediaItems': normalized.get('mediaItems'),
            'runtimeFingerprint': runtime_fingerprint,
            'templateId': normalized.get('templateId'),
            'templateLockHash': normalized.get('templateLockHash'),
            'promptSettingsHash': normalized.get('promptSettingsHash'),
        }
        return normalized

    async def studio_outputs_get(self, request):
        limit = min(max(int(request.query.get('limit', 80)), 1), 200)
        outputs = self._read_studio_outputs()
        return web.json_response({
            'error': False,
            'count': len(outputs),
            'outputs': outputs[:limit],
            'path': str(self._studio_history_file()),
        })

    async def studio_outputs_post(self, request):
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return web.json_response({'error': True, 'message': 'Invalid JSON body.'}, status=400)

        raw_outputs = payload.get('outputs') if isinstance(payload, dict) and isinstance(payload.get('outputs'), list) else None
        if raw_outputs is None:
            raw_outputs = [payload] if isinstance(payload, dict) else []

        incoming = [self._normalize_studio_output(output) for output in raw_outputs if isinstance(output, dict)]
        if not incoming:
            return web.json_response({'error': True, 'message': 'No Studio outputs supplied.'}, status=400)

        async with self.studio_history_lock:
            existing = self._read_studio_outputs()
            outputs = self._write_studio_outputs(self._merge_studio_outputs(existing, incoming))

        return web.json_response({
            'error': False,
            'count': len(outputs),
            'outputs': outputs,
        })

    async def studio_outputs_patch(self, request):
        output_id = request.match_info.get('output_id')
        if not output_id:
            return web.json_response({'error': True, 'message': 'Missing Studio output id.'}, status=400)

        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return web.json_response({'error': True, 'message': 'Invalid JSON body.'}, status=400)

        async with self.studio_history_lock:
            outputs = self._read_studio_outputs()
            updated = False
            for index, output in enumerate(outputs):
                if str(output.get('id')) != output_id:
                    continue
                outputs[index] = {
                    **output,
                    **payload,
                    'id': output_id,
                    'updatedAt': int(time.time() * 1000),
                }
                updated = True
                break

            if not updated:
                return web.json_response({'error': True, 'message': f'Studio output {output_id} was not found.'}, status=404)

            outputs = self._write_studio_outputs(self._sort_studio_outputs(outputs))

        return web.json_response({
            'error': False,
            'count': len(outputs),
            'outputs': outputs,
        })

    async def studio_outputs_delete(self, request):
        output_id = request.match_info.get('output_id')
        if not output_id:
            return web.json_response({'error': True, 'message': 'Missing Studio output id.'}, status=400)

        async with self.studio_history_lock:
            outputs = self._read_studio_outputs()
            next_outputs = [output for output in outputs if str(output.get('id')) != output_id]
            if len(next_outputs) == len(outputs):
                return web.json_response({'error': True, 'message': f'Studio output {output_id} was not found.'}, status=404)
            outputs = self._write_studio_outputs(next_outputs)

        return web.json_response({
            'error': False,
            'count': len(outputs),
            'outputs': outputs,
        })

    async def studio_blocks_get(self, request):
        limit = min(max(int(request.query.get('limit', 200)), 1), 500)
        blocks = self._list_studio_blocks()
        return web.json_response({
            'error': False,
            'count': len(blocks),
            'blocks': blocks[:limit],
            'path': str(self._studio_blocks_dir()),
        })

    async def studio_blocks_post(self, request):
        try:
            payload = await request.json()
            block, block_file = self._write_studio_block(payload)
        except json.JSONDecodeError:
            return web.json_response({'error': True, 'message': 'Invalid JSON body.'}, status=400)
        except ValueError as e:
            return web.json_response({'error': True, 'message': str(e)}, status=400)
        except OSError as e:
            logger.error(f"Error saving Studio user block: {e}")
            return web.json_response({'error': True, 'message': 'Could not save user block.'}, status=500)

        return web.json_response({
            'error': False,
            'block': block,
            'path': str(block_file),
        })

    async def studio_block_get(self, request):
        block_id = request.match_info.get('block_id')
        if not block_id:
            return web.json_response({'error': True, 'message': 'Missing user block id.'}, status=400)
        try:
            block = self._read_studio_block(block_id)
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            logger.error(f"Error reading Studio user block {block_id}: {e}")
            return web.json_response({'error': True, 'message': 'Could not read user block.'}, status=500)
        if block is None:
            return web.json_response({'error': True, 'message': f'User block {block_id} was not found.'}, status=404)
        return web.json_response({
            'error': False,
            'block': block,
        })

    async def studio_block_delete(self, request):
        block_id = request.match_info.get('block_id')
        if not block_id:
            return web.json_response({'error': True, 'message': 'Missing user block id.'}, status=400)
        block_file = self._studio_block_file(block_id)
        if not block_file.exists():
            return web.json_response({'error': True, 'message': f'User block {block_id} was not found.'}, status=404)
        try:
            block_file.unlink()
        except OSError as e:
            logger.error(f"Error deleting Studio user block {block_id}: {e}")
            return web.json_response({'error': True, 'message': 'Could not delete user block.'}, status=500)
        return web.json_response({
            'error': False,
            'id': self._safe_block_id(block_id),
        })

    def _workflow_shares_dir(self):
        return Path(self.data_dir) / 'studio' / 'shares'

    def _safe_share_id(self, share_id=None):
        raw_id = str(share_id or nanoid.generate(size=12))
        safe_id = ''.join(ch for ch in raw_id if ch.isalnum() or ch in ('-', '_'))[:80]
        return safe_id or nanoid.generate(size=12)

    def _workflow_share_file(self, share_id):
        return self._workflow_shares_dir() / f"{self._safe_share_id(share_id)}.json"

    def _workflow_share_media_dir(self, share_id):
        return self._workflow_shares_dir() / self._safe_share_id(share_id) / 'media'

    def _path_within(self, path, root):
        try:
            Path(path).resolve(strict=False).relative_to(Path(root).resolve(strict=False))
            return True
        except ValueError:
            return False

    def _safe_media_filename(self, filename, fallback='preview.bin'):
        raw_name = Path(str(filename or fallback)).name
        safe_name = ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '-' for ch in raw_name)[:120].strip('.-')
        return safe_name or fallback

    def _share_media_hash(self, data):
        return f"sha256:bytes:{hashlib.sha256(data).hexdigest()}"

    def _content_type_extension(self, content_type, fallback_url=''):
        normalized = str(content_type or '').split(';')[0].strip().lower()
        explicit = {
            'image/webp': '.webp',
            'image/png': '.png',
            'image/jpeg': '.jpg',
            'image/jpg': '.jpg',
            'image/gif': '.gif',
            'video/mp4': '.mp4',
            'video/webm': '.webm',
            'audio/wav': '.wav',
            'audio/mpeg': '.mp3',
            'audio/flac': '.flac',
            'application/json': '.json',
            'text/plain': '.txt',
        }.get(normalized)
        if explicit:
            return explicit

        parsed_suffix = Path(urlparse(str(fallback_url or '')).path).suffix.lower()
        if parsed_suffix:
            return parsed_suffix
        return (mimetypes.guess_extension(normalized) or '.bin') if normalized else '.bin'

    def _resolve_file_route_path(self, file):
        if not file:
            return None

        file_path = Path(unquote(str(file)))
        if not file_path.is_absolute():
            file_path = Path(self.work_dir) / file_path

        if not file_path.exists():
            legacy_graph_root = Path(self.data_dir) / 'graphs' / 'mellon'
            modiff_graph_root = Path(self.data_dir) / 'graphs' / 'modiff'
            try:
                legacy_graph_relative_path = file_path.resolve(strict=False).relative_to(
                    legacy_graph_root.resolve(strict=False)
                )
            except ValueError:
                legacy_graph_relative_path = None

            if legacy_graph_relative_path is not None:
                migrated_file_path = modiff_graph_root / legacy_graph_relative_path
                if migrated_file_path.exists():
                    file_path = migrated_file_path

        if not file_path.exists():
            return None

        if self._path_within(file_path, self.work_dir) or self._path_within(file_path, self.data_dir):
            return file_path
        return None

    def _cache_media_bytes_from_url(self, preview_url):
        parsed = urlparse(str(preview_url))
        parts = [unquote(part) for part in parsed.path.split('/') if part]
        if len(parts) < 3 or parts[0] != 'cache':
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

        params = self.modules.get(cached_node.module_name, {}).get(cached_node.class_name, {}).get('params', {})
        data_type = params.get(field, {}).get('type')
        type_values = data_type if isinstance(data_type, list) else [data_type]
        query = parse_qs(parsed.query)
        filename = query.get('filename', [field])[0]

        if 'image' in type_values:
            image_format = query.get('format', ['WEBP'])[0].upper()
            quality = query.get('quality', [100])[0]
            return {
                'bytes': to_bytes(data_type, data, {'format': image_format, 'quality': quality}),
                'contentType': f"image/{image_format.lower()}",
                'filename': f"{filename}.{image_format.lower()}",
            }

        if data_type == 'text' or any(isinstance(item, str) and item.startswith('str') for item in type_values):
            return {
                'bytes': str(data).encode('utf-8'),
                'contentType': 'text/plain',
                'filename': f"{filename}.txt",
            }

        data_path = Path(str(data))
        if data_path.exists() and (self._path_within(data_path, self.work_dir) or self._path_within(data_path, self.data_dir)):
            content_type = mimetypes.guess_type(str(data_path))[0] or 'application/octet-stream'
            return {
                'bytes': data_path.read_bytes(),
                'contentType': content_type,
                'filename': data_path.name,
            }
        return None

    def _share_media_bytes_from_url(self, preview_url):
        if not isinstance(preview_url, str) or not preview_url:
            return None

        if preview_url.startswith('data:') and ',' in preview_url:
            header, payload = preview_url.split(',', 1)
            content_type = header.split(';')[0].removeprefix('data:') or 'application/octet-stream'
            data = base64.b64decode(payload) if ';base64' in header else unquote_to_bytes(payload)
            return {
                'bytes': data,
                'contentType': content_type,
                'filename': f"preview{self._content_type_extension(content_type, preview_url)}",
            }

        parsed = urlparse(preview_url)
        if parsed.path.startswith('/workflows/share/'):
            return None
        if parsed.scheme in ('http', 'https') and not parsed.path.startswith('/cache/') and parsed.path != '/file':
            return None

        if parsed.path.startswith('/cache/'):
            return self._cache_media_bytes_from_url(preview_url)

        if parsed.path == '/file':
            file_path = self._resolve_file_route_path(parse_qs(parsed.query).get('file', [''])[0])
            if not file_path:
                return None
            return {
                'bytes': file_path.read_bytes(),
                'contentType': mimetypes.guess_type(str(file_path))[0] or 'application/octet-stream',
                'filename': file_path.name,
            }

        return None

    def _persist_share_preview_media(self, share_id, package):
        preview = self._share_preview_url(package)
        media = self._share_media_bytes_from_url(preview)
        if not media:
            return package, None

        try:
            safe_share_id = self._safe_share_id(share_id)
            content_type = media.get('contentType') or 'application/octet-stream'
            filename = self._safe_media_filename(media.get('filename'), f"preview{self._content_type_extension(content_type, preview)}")
            if not Path(filename).suffix:
                filename = f"{filename}{self._content_type_extension(content_type, preview)}"

            target_dir = self._workflow_share_media_dir(safe_share_id)
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = (target_dir / filename).resolve()
            if not self._path_within(target_path, target_dir):
                raise ValueError('Resolved share media path escaped the share media directory.')

            media_bytes = media.get('bytes')
            if not isinstance(media_bytes, (bytes, bytearray)):
                return package, None

            with open(target_path, 'wb') as f:
                f.write(media_bytes)

            byte_hash = self._share_media_hash(bytes(media_bytes))
            media_url = f"/workflows/share/{quote(safe_share_id)}/media/{quote(filename)}"
            next_package = deepcopy(package)

            manifest = next_package.setdefault('manifest', {}) if isinstance(next_package, dict) else {}
            manifest_media = manifest.get('media') if isinstance(manifest.get('media'), dict) else {}
            manifest_media.update({
                'url': media_url,
                'persistedUrl': media_url,
                'byteHash': byte_hash,
                'contentType': content_type,
                'backendShareMediaPath': str(target_path),
            })
            manifest['media'] = manifest_media

            metadata = next_package.setdefault('metadata', {}) if isinstance(next_package, dict) else {}
            metadata['preview'] = media_url

            latest_output = next_package.get('latestOutput') if isinstance(next_package.get('latestOutput'), dict) else None
            if latest_output is not None:
                latest_output['url'] = media_url
                latest_output['backendShareMediaPath'] = str(target_path)
                latest_output['backendShareMediaHash'] = byte_hash

            persisted = {
                'url': media_url,
                'path': str(target_path),
                'byteHash': byte_hash,
                'contentType': content_type,
            }
            return next_package, persisted
        except Exception as e:
            logger.error(f"Error persisting workflow share media {share_id}: {e}")
            return package, None

    def _share_summary(self, share):
        package = share.get('package', {}) if isinstance(share, dict) else {}
        metadata = package.get('metadata', {}) if isinstance(package, dict) else {}
        studio = metadata.get('studio', {}) if isinstance(metadata, dict) else {}
        return {
            'share_id': share.get('share_id'),
            'createdAt': share.get('createdAt'),
            'updatedAt': share.get('updatedAt'),
            'url': share.get('url'),
            'modelType': studio.get('modelType'),
            'mode': studio.get('mode'),
            'prompt': studio.get('prompt'),
            'preview': self._share_preview_url(package),
            'persistedMedia': share.get('persistedMedia'),
        }

    def _share_preview_url(self, package):
        if not isinstance(package, dict):
            return None
        metadata = package.get('metadata', {})
        manifest = package.get('manifest', {})
        media = manifest.get('media', {}) if isinstance(manifest, dict) else {}
        preview = metadata.get('preview') if isinstance(metadata, dict) else None
        if not preview and isinstance(media, dict):
            preview = media.get('url')
        if not isinstance(preview, str):
            return None
        if preview.startswith(('http://', 'https://', 'data:image/', '/')):
            return preview
        return None

    def _workflow_share_preview_html(self, request, share):
        share_id = self._safe_share_id(share.get('share_id'))
        package = share.get('package', {}) if isinstance(share, dict) else {}
        metadata = package.get('metadata', {}) if isinstance(package, dict) else {}
        manifest = package.get('manifest', {}) if isinstance(package, dict) else {}
        studio = metadata.get('studio', {}) if isinstance(metadata, dict) else {}
        template = manifest.get('template', {}) if isinstance(manifest, dict) else {}
        provenance = manifest.get('provenance', {}) if isinstance(manifest, dict) else {}
        frontend_url = f"/?share={quote(share_id)}"
        json_url = f"/workflows/share/{quote(share_id)}?format=json"
        preview = self._share_preview_url(package)
        title = studio.get('prompt') or template.get('templateLabel') or f"MoDiff workflow {share_id}"
        prompt = studio.get('prompt') or ''
        mode = studio.get('mode') or 'workflow'
        model_type = studio.get('modelType') or 'unknown model'
        exported_at = metadata.get('exportedAt') or manifest.get('exportedAt') or share.get('createdAt') or ''
        media_hash = None
        if isinstance(provenance, dict):
            frontend = provenance.get('frontend', {})
            backend = provenance.get('backend', {})
            if isinstance(frontend, dict):
                media_hash = frontend.get('mediaHash')
            if not media_hash and isinstance(backend, dict):
                media_hash = backend.get('mediaHash')

        def esc(value):
            return html.escape(str(value or ''), quote=True)

        preview_html = ''
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
        <dt>Template</dt><dd>{esc(template.get('templateLabel') if isinstance(template, dict) else '')}</dd>
        <dt>Media hash</dt><dd>{esc(media_hash)}</dd>
        <dt>Prompt</dt><dd>{esc(prompt)}</dd>
      </dl>
    </section>
  </main>
</body>
</html>"""

    async def workflow_shares_list(self, request):
        limit = min(max(int(request.query.get('limit', 50)), 1), 200)
        shares_dir = self._workflow_shares_dir()
        summaries = []

        if shares_dir.exists():
            for share_file in shares_dir.glob('*.json'):
                try:
                    with open(share_file, 'r', encoding='utf-8') as f:
                        share = json.load(f)
                    summaries.append(self._share_summary(share))
                except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
                    logger.error(f"Error reading workflow share {share_file}: {e}")

        summaries.sort(key=lambda item: item.get('createdAt') or '', reverse=True)
        return web.json_response({
            'error': False,
            'count': len(summaries),
            'shares': summaries[:limit],
            'path': str(shares_dir),
        })

    async def workflow_share_post(self, request):
        try:
            package = await request.json()
        except json.JSONDecodeError:
            return web.json_response({'error': True, 'message': 'Invalid JSON body.'}, status=400)

        if not isinstance(package, dict):
            return web.json_response({'error': True, 'message': 'Workflow share package must be a JSON object.'}, status=400)

        share_id = self._safe_share_id(package.get('share_id') or package.get('shareId'))
        package, persisted_media = self._persist_share_preview_media(share_id, package)
        created_at = package.get('createdAt') or int(time.time() * 1000)
        url = f"/workflows/share/{share_id}"
        share = {
            'share_id': share_id,
            'createdAt': created_at,
            'updatedAt': int(time.time() * 1000),
            'url': url,
            'package': package,
        }
        if persisted_media:
            share['persistedMedia'] = persisted_media

        shares_dir = self._workflow_shares_dir()
        shares_dir.mkdir(parents=True, exist_ok=True)
        share_file = self._workflow_share_file(share_id)
        temp_file = share_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(share, f, ensure_ascii=False)
            temp_file.replace(share_file)
        except OSError as e:
            logger.error(f"Error saving workflow share {share_id}: {e}")
            return web.json_response({'error': True, 'message': str(e)}, status=500)

        return web.json_response({
            'error': False,
            'share_id': share_id,
            'url': url,
            'path': str(share_file),
            'persistedMedia': persisted_media,
            'share': share,
        })

    async def workflow_share_media_get(self, request):
        share_id = self._safe_share_id(request.match_info.get('share_id'))
        filename = self._safe_media_filename(request.match_info.get('filename'))
        media_dir = self._workflow_share_media_dir(share_id)
        media_path = (media_dir / filename).resolve()

        if not self._path_within(media_path, media_dir):
            return web.json_response({'error': True, 'message': 'Invalid workflow share media path.'}, status=403)
        if not media_path.exists() or not media_path.is_file():
            return web.json_response({'error': True, 'message': f'Workflow share media {filename} was not found.'}, status=404)

        resp = web.FileResponse(media_path)
        resp.headers['Content-Disposition'] = f'inline; filename="{filename}"'
        resp.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
        return resp

    async def workflow_share_get(self, request):
        share_id = request.match_info.get('share_id')
        if not share_id:
            return web.json_response({'error': True, 'message': 'Missing workflow share id.'}, status=400)

        share_file = self._workflow_share_file(share_id)
        if not share_file.exists():
            return web.json_response({'error': True, 'message': f'Workflow share {share_id} was not found.'}, status=404)

        try:
            with open(share_file, 'r', encoding='utf-8') as f:
                share = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            logger.error(f"Error reading workflow share {share_id}: {e}")
            return web.json_response({'error': True, 'message': str(e)}, status=500)

        wants_html = (
            request.query.get('format') != 'json'
            and 'text/html' in request.headers.get('Accept', '')
        )
        if wants_html:
            return web.Response(
                text=self._workflow_share_preview_html(request, share),
                content_type='text/html',
            )

        return web.json_response({
            'error': False,
            **share,
        })

    """
    ╭────────────────────────╮
       Main graph execution
    ╰────────────────────────╯
    """

    async def graph(self, request):
        graph = await request.json()
        sid = graph.get("sid")
        #if not sid:
        #    return web.json_response({"error": True, "message": "Missing session id"}, status=400)

        task_id = await self.queue_task(self.execute_graph, (graph,), None, sid, name=f"Graph execution")
        return web.json_response({
            "error": False,
            "message": "Graph queued for processing",
            "sid": sid,
            "task_id": task_id,
        })

    def _exception_chain(self, e):
        chain = []
        seen = set()
        current = e
        while current is not None and id(current) not in seen:
            chain.append(current)
            seen.add(id(current))
            current = getattr(current, '__cause__', None) or getattr(current, '__context__', None)
        return chain

    def _loader_diagnostics_snapshot(self):
        diagnostics = {}
        runtime_hints = self.current_task.get("runtimeHints") if self.current_task else None
        runtime_hint_offload = runtime_hints.get("offloadMode") if isinstance(runtime_hints, dict) else None
        runtime_hint_resource_mode = runtime_hints.get("resourceMode") if isinstance(runtime_hints, dict) else None
        runtime_hint_resolved_mode = runtime_hints.get("resolvedResourceMode") if isinstance(runtime_hints, dict) else None

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

    def _current_run_identity_payload(self):
        runtime_hints = self.current_task.get("runtimeHints") if self.current_task else None
        if not isinstance(runtime_hints, dict):
            return {}
        payload = {}
        client_run_id = runtime_hints.get("clientRunId")
        run_input_hash = runtime_hints.get("runInputHash")
        if client_run_id:
            payload["client_run_id"] = client_run_id
        if run_input_hash:
            payload["run_input_hash"] = run_input_hash
        return payload

    def _exception_payload(self, e, task_id=None, sid=None, node_id=None, node_name=None, traceback_text=None):
        exception_type = type(e).__name__
        message = str(e) or exception_type
        classification = self._classify_exception(e, message=message, exception_type=exception_type)
        if classification['error_code'] == 'missing_prompt_embeddings':
            message = 'Prompt embeddings are missing from Encode Prompt. Update or recreate the Studio graph after node definitions finish refreshing.'
        elif classification.get('message'):
            message = classification['message']
        oom = classification['category'] == 'oom'
        memory_summary = None
        if oom:
            memory_summary = message.split('\n')[0]

        return {
            "task_id": task_id,
            "sid": sid,
            "node": node_id,
            "node_name": node_name,
            "message": message,
            "exception_type": exception_type,
            "traceback": traceback_text,
            "category": classification['category'],
            "error_code": classification['error_code'],
            "recovery_hint": classification['recovery_hint'],
            "oom": oom,
            "memory_summary": memory_summary,
            "cuda_memory_snapshot": self._cuda_memory_snapshot(),
            "gpu_processes": self._gpu_process_snapshot(),
            "runtime_hints": self.current_task.get("runtimeHints") if self.current_task else None,
            "runtime_budget": self.current_task.get("runtimeBudget") if self.current_task else None,
            "loader_diagnostics": self._loader_diagnostics_snapshot(),
        }

    def _classify_exception(self, e, message=None, exception_type=None):
        exception_type = exception_type or type(e).__name__
        message = message or str(e) or exception_type
        explicit_error_code = getattr(e, 'modiff_error_code', None)
        if explicit_error_code:
            return {
                'category': getattr(e, 'modiff_category', 'runtime'),
                'error_code': explicit_error_code,
                'message': message,
                'recovery_hint': getattr(e, 'modiff_recovery_hint', None),
            }
        chain = self._exception_chain(e)
        chain_text = ' | '.join(f'{type(item).__name__} {str(item) or type(item).__name__}' for item in chain)
        normalized = f'{exception_type} {message} {chain_text}'.lower()

        if (
            'outofmemory' in normalized
            or 'out of memory' in normalized
            or 'cuda out of memory' in normalized
            or 'cublas_status_alloc_failed' in normalized
            or 'cusolver_status_alloc_failed' in normalized
        ):
            return {
                'category': 'oom',
                'error_code': 'cuda_oom',
                'message': next((str(item) for item in reversed(chain) if 'out of memory' in (str(item) or '').lower() or 'outofmemory' in type(item).__name__.lower()), message),
                'recovery_hint': 'Release accelerator cache, close other accelerator-heavy apps, apply the Low VRAM preset, or switch to a smaller compatible model.',
            }

        if any(isinstance(item, MissingConnectedOutputError) for item in chain):
            return {
                'category': 'graph_incomplete',
                'error_code': 'missing_connected_output',
                'recovery_hint': 'An upstream node did not produce a connected output. Update or recreate the graph; if this followed a loader failure, inspect loader diagnostics.',
            }

        if 'illegal memory access' in normalized or 'cudaerrorillegaladdress' in normalized:
            return {
                'category': 'cuda_context',
                'error_code': 'cuda_context_poisoned',
                'message': next((str(item) for item in reversed(chain) if 'illegal memory access' in (str(item) or '').lower()), message),
                'recovery_hint': 'CUDA reported an illegal memory access. Stop this run, restart the backend process, and retry with a safer execution plan so the current Python CUDA context is not reused.',
            }

        if 'cublas_status_not_supported' in normalized or 'cublasltmatmulalgogetheuristic' in normalized:
            return {
                'category': 'cuda_kernel',
                'error_code': 'cuda_kernel_unsupported',
                'message': next((str(item) for item in reversed(chain) if 'cublas' in (str(item) or '').lower()), message),
                'recovery_hint': 'The quantized CUDA kernel used by this model path is not supported by the current PyTorch/bitsandbytes/CUDA combination. Try a non-quantized smaller model path, update the CUDA/PyTorch/bitsandbytes stack, or use a backend path that provides compatible Qwen weights.',
            }

        if (
            (isinstance(e, KeyError) and str(e).strip("'\"") == 'embeddings')
            or "keyerror 'embeddings'" in normalized
            or 'keyerror "embeddings"' in normalized
        ):
            return {
                'category': 'graph_incomplete',
                'error_code': 'missing_prompt_embeddings',
                'recovery_hint': 'Prompt embeddings were not ready or connected. Update or recreate the Studio graph after node definitions finish refreshing, then retry.',
            }

        if isinstance(e, (ModuleNotFoundError, ImportError)) or 'no module named' in normalized or 'cannot import name' in normalized:
            return {
                'category': 'missing_dependency',
                'error_code': 'missing_dependency',
                'recovery_hint': 'Install or repair the missing Python package, then restart the backend.',
            }

        if (
            isinstance(e, FileNotFoundError)
            or 'model not found' in normalized
            or 'missing model' in normalized
            or 'no such file or directory' in normalized
            or 'localentrynotfound' in normalized
            or 'entrynotfound' in normalized
            or 'repo not found' in normalized
            or 'repository not found' in normalized
        ):
            return {
                'category': 'missing_model',
                'error_code': 'missing_model',
                'recovery_hint': 'Open Setup, refresh model indexes, then install or relink the missing model package.',
            }

        if (
            isinstance(e, ConnectionError)
            or 'connection refused' in normalized
            or 'connection reset' in normalized
            or 'backend unavailable' in normalized
            or 'server disconnected' in normalized
        ):
            return {
                'category': 'backend_unavailable',
                'error_code': 'backend_unavailable',
                'recovery_hint': 'Check that the backend is still running, then retry the workflow.',
            }

        if isinstance(e, PermissionError) or 'permission denied' in normalized or 'access is denied' in normalized:
            return {
                'category': 'permission',
                'error_code': 'permission_denied',
                'recovery_hint': 'Check file permissions and whether another process is locking the target path.',
            }

        if isinstance(e, asyncio.CancelledError) or 'cancelled' in normalized or 'interrupted' in normalized:
            return {
                'category': 'interrupted',
                'error_code': 'run_interrupted',
                'recovery_hint': 'The run was interrupted. Retry when the backend queue is idle.',
            }

        return {
            'category': 'runtime_error',
            'error_code': 'runtime_error',
            'recovery_hint': 'Review the run details, fix the referenced node or input, and retry.',
        }

    def _runtime_fingerprint(self):
        packages = {
            'python': sys.version.split(' ')[0],
            'platform': platform.platform(),
        }
        for package_name in ('diffusers', 'transformers', 'accelerate', 'bitsandbytes'):
            try:
                packages[package_name] = metadata.version(package_name)
            except Exception:
                packages[package_name] = None
        try:
            hardware = get_hardware_snapshot(self.data_dir)
            torch_metadata = hardware.get('torch') if isinstance(hardware.get('torch'), dict) else {}
            legacy_status = legacy_torch_status(hardware)
            if torch_metadata.get('available'):
                packages['torch'] = torch_metadata.get('version')
                torch_state = {
                    'cuda_available': legacy_status.get('cuda_available', False),
                    'cuda_device_count': legacy_status.get('cuda_device_count', 0),
                    'cuda_device_name': legacy_status.get('cuda_device_name'),
                    'cudnn_version': torch_metadata.get('cudnn_version'),
                    'cudnn_deterministic': torch_metadata.get('cudnn_deterministic'),
                    'cudnn_benchmark': torch_metadata.get('cudnn_benchmark'),
                    'deterministic_algorithms': torch_metadata.get('deterministic_algorithms'),
                }
            else:
                errors = torch_metadata.get('errors') if isinstance(torch_metadata.get('errors'), dict) else {}
                torch_state = {'error': errors.get('import') or 'torch is unavailable'}

            if torch_state.get('cuda_available') and torch_state.get('cuda_device_count', 0) > 0:
                torch_state.update({
                    'cuda_device_total_memory_bytes': legacy_status.get('cuda_device_total_memory_bytes'),
                    'cuda_memory_free_bytes': legacy_status.get('cuda_memory_free_bytes'),
                    'cuda_memory_total_bytes': legacy_status.get('cuda_memory_total_bytes'),
                })
                try:
                    torch = import_module('torch')
                    capability = torch.cuda.get_device_capability(0)
                    torch_state['cuda_device_capability'] = '.'.join(str(item) for item in capability)
                except Exception as capability_error:
                    torch_state['cuda_device_capability_error'] = str(capability_error)
        except Exception as hardware_error:
            hardware = {'error': str(hardware_error)}
            torch_state = {'error': str(hardware_error)}

        fingerprint_payload = {
            'packages': packages,
            'torch': torch_state,
            'work_dir': str(self.work_dir),
            'data_dir': str(self.data_dir),
        }
        fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True, default=str).encode('utf-8')).hexdigest()
        return {
            'fingerprint': f'sha256:{fingerprint}',
            **fingerprint_payload,
            'hardware': hardware,
        }

    def _extract_graph_seed(self, graph, deterministic_options):
        seed = deterministic_options.get('seed') if isinstance(deterministic_options, dict) else None
        if seed is not None:
            try:
                return int(seed)
            except (TypeError, ValueError):
                return None

        for node in graph.get('nodes', {}).values():
            if not isinstance(node, dict):
                continue
            params = node.get('params', {})
            if not isinstance(params, dict):
                continue
            for key, param in params.items():
                if key != 'seed' or not isinstance(param, dict):
                    continue
                value = param.get('value')
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue

        return None

    def _deterministic_warnings(self, graph):
        warnings = []
        for node_id, node in graph.get('nodes', {}).items():
            if not isinstance(node, dict):
                continue
            params = node.get('params', {})
            if not isinstance(params, dict):
                continue
            for key, param in params.items():
                if not isinstance(param, dict):
                    continue
                if param.get('display') == 'random':
                    warnings.append(f'{node_id}.{key} is still marked random in the API graph.')
                if key == 'seed' and param.get('value') in (None, '', -1):
                    warnings.append(f'{node_id}.seed is not locked.')
        return warnings

    def _apply_deterministic_mode(self, graph):
        options = graph.get('deterministicMode')
        if options is True:
            options = {'enabled': True}
        if not isinstance(options, dict) or not options.get('enabled'):
            return None

        seed = self._extract_graph_seed(graph, options)
        applied = {
            'enabled': True,
            'seed': seed,
            'strict': bool(options.get('strict', True)),
            'warnings': self._deterministic_warnings(graph),
            'settings': {
                'python_random': False,
                'numpy_random': False,
                'torch_manual_seed': False,
                'torch_cuda_manual_seed_all': False,
                'torch_deterministic_algorithms': False,
                'cudnn_benchmark': None,
                'cudnn_deterministic': None,
                'allow_tf32': None,
            },
        }

        if seed is None:
            applied['warnings'].append('No fixed seed found for deterministic execution.')
        else:
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            applied['settings']['python_random'] = True

            try:
                import numpy as np
                np.random.seed(seed % (2 ** 32))
                applied['settings']['numpy_random'] = True
            except Exception as e:
                applied['warnings'].append(f'NumPy seed was not applied: {e}')

            try:
                import torch
                torch.manual_seed(seed)
                applied['settings']['torch_manual_seed'] = True
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                    applied['settings']['torch_cuda_manual_seed_all'] = True
                if hasattr(torch, 'use_deterministic_algorithms'):
                    torch.use_deterministic_algorithms(True, warn_only=True)
                    applied['settings']['torch_deterministic_algorithms'] = True
                if hasattr(torch.backends, 'cudnn'):
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                    applied['settings']['cudnn_benchmark'] = False
                    applied['settings']['cudnn_deterministic'] = True
                if hasattr(torch.backends, 'cuda'):
                    torch.backends.cuda.matmul.allow_tf32 = False
                    applied['settings']['allow_tf32'] = False
                if hasattr(torch.backends, 'cudnn'):
                    torch.backends.cudnn.allow_tf32 = False
            except Exception as e:
                applied['warnings'].append(f'Torch deterministic settings were not fully applied: {e}')

        return applied

    def _coerce_runtime_hints(self, value):
        if not isinstance(value, dict):
            return None

        allowed = {
            'source',
            'device',
            'cudaIndex',
            'cudaMemoryFreeBytes',
            'cudaMemoryTotalBytes',
            'modelFamily',
            'modelType',
            'modelRepo',
            'modelName',
            'resolvedModelRepo',
            'resolvedArtifact',
            'executionPath',
            'pipelineClass',
            'dtype',
            'resourceMode',
            'resolvedResourceMode',
            'quantizationMode',
            'quantizedComponents',
            'autoOffload',
            'offloadMode',
            'offloadDiskPath',
            'supportedOffloadModes',
            'resourcePlan',
            'autoResourcePlan',
            'autoResourceCandidates',
            'autoResourceProofStatus',
            'autoResourceCandidateId',
            'resourceRetryModes',
            'resourceRetryPlans',
            'resourceRetryAttempt',
            'resourceRetryHistory',
            'resourceRetryLastError',
            'resourceRetryLastCode',
            'cudaBudgetPolicy',
            'enforceCudaBudget',
            'compatibilityProbe',
            'compatibilityStatus',
            'lowVramMode',
            'requestedCudaReserveBytes',
            'requestedCudaBudgetBytes',
            'clientRunId',
            'runInputHash',
        }
        hints = {key: value.get(key) for key in allowed if key in value}

        for key in (
            'source',
            'device',
            'modelFamily',
            'modelType',
            'modelRepo',
            'modelName',
            'resolvedModelRepo',
            'resolvedArtifact',
            'executionPath',
            'pipelineClass',
            'dtype',
            'resourceMode',
            'resolvedResourceMode',
            'quantizationMode',
            'offloadMode',
            'offloadDiskPath',
            'resourceRetryLastError',
            'resourceRetryLastCode',
            'cudaBudgetPolicy',
            'compatibilityStatus',
            'autoResourceProofStatus',
            'autoResourceCandidateId',
            'clientRunId',
            'runInputHash',
        ):
            if key in hints and hints[key] is not None and not isinstance(hints[key], str):
                hints[key] = str(hints[key])

        for key in ('quantizedComponents', 'supportedOffloadModes', 'resourceRetryModes'):
            if key in hints and hints[key] is not None:
                if isinstance(hints[key], list):
                    hints[key] = [str(item) for item in hints[key] if item is not None]
                else:
                    hints.pop(key, None)

        if 'resourcePlan' in hints and hints['resourcePlan'] is not None and not isinstance(hints['resourcePlan'], dict):
            hints.pop('resourcePlan', None)

        if 'autoResourcePlan' in hints and hints['autoResourcePlan'] is not None and not isinstance(hints['autoResourcePlan'], dict):
            hints.pop('autoResourcePlan', None)

        if 'autoResourceCandidates' in hints and hints['autoResourceCandidates'] is not None and not isinstance(hints['autoResourceCandidates'], list):
            hints.pop('autoResourceCandidates', None)

        if 'resourceRetryHistory' in hints and hints['resourceRetryHistory'] is not None and not isinstance(hints['resourceRetryHistory'], list):
            hints.pop('resourceRetryHistory', None)

        if 'resourceRetryPlans' in hints and hints['resourceRetryPlans'] is not None:
            if isinstance(hints['resourceRetryPlans'], list):
                plans = []
                for item in hints['resourceRetryPlans']:
                    if isinstance(item, dict):
                        plans.append(deepcopy(item))
                hints['resourceRetryPlans'] = plans
            else:
                hints.pop('resourceRetryPlans', None)

        if 'compatibilityProbe' in hints and hints['compatibilityProbe'] is not None and not isinstance(hints['compatibilityProbe'], dict):
            hints.pop('compatibilityProbe', None)

        for key in ('cudaIndex', 'cudaMemoryFreeBytes', 'cudaMemoryTotalBytes', 'requestedCudaReserveBytes', 'requestedCudaBudgetBytes', 'resourceRetryAttempt'):
            if key in hints and hints[key] is not None:
                try:
                    hints[key] = int(hints[key])
                except (TypeError, ValueError):
                    hints.pop(key, None)

        for key in ('autoOffload', 'lowVramMode'):
            if key in hints and hints[key] is not None:
                hints[key] = bool(hints[key])
        if 'enforceCudaBudget' in hints and hints['enforceCudaBudget'] is not None:
            hints['enforceCudaBudget'] = bool(hints['enforceCudaBudget'])
        if hints.get('cudaBudgetPolicy') not in (None, 'advisory', 'enforced'):
            hints.pop('cudaBudgetPolicy', None)

        return hints

    def _cuda_index_from_runtime_hints(self, hints):
        if not hints:
            return None
        if isinstance(hints.get('cudaIndex'), int):
            return hints.get('cudaIndex')

        device = str(hints.get('device') or '').strip().lower()
        match = re.match(r'^cuda(?::(\d+))?$', device)
        if not match:
            return None
        return int(match.group(1) or 0)

    def _apply_cuda_runtime_budget(self, runtime_hints):
        result = {
            'applied': False,
            'reason': 'No CUDA runtime hints were provided.',
        }
        cuda_index = self._cuda_index_from_runtime_hints(runtime_hints)
        if cuda_index is None:
            if runtime_hints:
                result['reason'] = 'Runtime hints did not target a CUDA device.'
            return result

        try:
            torch = import_module('torch')
        except Exception as e:
            return {
                **result,
                'reason': f'torch import failed: {e}',
                'cuda_index': cuda_index,
            }

        if not torch.cuda.is_available():
            return {
                **result,
                'reason': 'CUDA is not available in this backend process.',
                'cuda_index': cuda_index,
            }

        device_count = int(torch.cuda.device_count())
        if cuda_index < 0 or cuda_index >= device_count:
            return {
                **result,
                'reason': f'CUDA device {cuda_index} is not available.',
                'cuda_index': cuda_index,
                'device_count': device_count,
            }

        enforce_budget = (
            (runtime_hints or {}).get('enforceCudaBudget') is True
            or (runtime_hints or {}).get('cudaBudgetPolicy') == 'enforced'
        )
        if not enforce_budget:
            try:
                torch.cuda.set_per_process_memory_fraction(1.0, cuda_index)
                reset_reason = 'CUDA budget is advisory; reset PyTorch process memory fraction to full device.'
            except Exception as e:
                reset_reason = f'CUDA budget is advisory; could not reset PyTorch process memory fraction: {e}'
            return {
                **result,
                'reason': reset_reason,
                'cuda_index': cuda_index,
                'cuda_budget_policy': 'advisory',
                'fraction': 1.0,
                'model_repo': runtime_hints.get('modelRepo') if runtime_hints else None,
                'model_name': runtime_hints.get('modelName') if runtime_hints else None,
                'execution_path': runtime_hints.get('executionPath') if runtime_hints else None,
                'offload_mode': runtime_hints.get('offloadMode') if runtime_hints else None,
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
                'reason': f'CUDA memory info unavailable: {e}',
                'cuda_index': cuda_index,
            }

        gib = 1024 ** 3
        requested_reserve = runtime_hints.get('requestedCudaReserveBytes') if runtime_hints else None
        reserve_bytes = requested_reserve if isinstance(requested_reserve, int) and requested_reserve > 0 else max(gib, int(total_bytes * 0.1))
        reserve_bytes = min(reserve_bytes, max(total_bytes - 1, 0))

        free_budget = max(0, free_bytes - reserve_bytes)
        total_budget = max(0, total_bytes - reserve_bytes)
        requested_budget = runtime_hints.get('requestedCudaBudgetBytes') if runtime_hints else None
        budget_candidates = [free_budget, total_budget]
        if isinstance(requested_budget, int) and requested_budget > 0:
            budget_candidates.append(requested_budget)
        applied_budget = min(candidate for candidate in budget_candidates if candidate > 0) if any(candidate > 0 for candidate in budget_candidates) else 0

        if applied_budget <= 0:
            return {
                **result,
                'reason': 'No CUDA budget remained after reserve calculation.',
                'cuda_index': cuda_index,
                'free_bytes': free_bytes,
                'total_bytes': total_bytes,
                'reserve_bytes': reserve_bytes,
            }

        fraction = max(0.05, min(1.0, applied_budget / total_bytes))
        try:
            torch.cuda.set_per_process_memory_fraction(fraction, cuda_index)
        except Exception as e:
            return {
                **result,
                'reason': f'Could not apply CUDA memory fraction: {e}',
                'cuda_index': cuda_index,
                'free_bytes': free_bytes,
                'total_bytes': total_bytes,
                'reserve_bytes': reserve_bytes,
                'applied_budget_bytes': applied_budget,
                'fraction': fraction,
            }

        return {
            'applied': True,
            'reason': 'Applied CUDA memory fraction from runtime hints.',
            'cuda_index': cuda_index,
            'free_bytes': free_bytes,
            'total_bytes': total_bytes,
            'reserve_bytes': reserve_bytes,
            'applied_budget_bytes': applied_budget,
            'fraction': fraction,
            'model_repo': runtime_hints.get('modelRepo') if runtime_hints else None,
            'model_name': runtime_hints.get('modelName') if runtime_hints else None,
            'dtype': runtime_hints.get('dtype') if runtime_hints else None,
            'quantization_mode': runtime_hints.get('quantizationMode') if runtime_hints else None,
            'auto_offload': runtime_hints.get('autoOffload') if runtime_hints else None,
            'offload_mode': runtime_hints.get('offloadMode') if runtime_hints else None,
            'low_vram_mode': runtime_hints.get('lowVramMode') if runtime_hints else None,
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
        requested = runtime_hints.get('resourceRetryModes')
        modes = requested if isinstance(requested, list) else allowed
        modes = [mode for mode in modes if mode in allowed]
        current_mode = runtime_hints.get('offloadMode')
        if current_mode in modes:
            return modes[modes.index(current_mode) + 1:]
        return modes

    def _coerce_retry_plan_list(self, runtime_hints):
        if not runtime_hints:
            return []

        raw_plans = runtime_hints.get('resourceRetryPlans')
        if isinstance(raw_plans, list):
            plans = []
            for index, raw_plan in enumerate(raw_plans):
                if not isinstance(raw_plan, dict):
                    continue
                plan = deepcopy(raw_plan)
                plan.setdefault('index', index)
                plan.setdefault('reason', f'retry_plan_{index + 1}')
                plans.append(plan)
            return plans

        return [
            {
                'index': index,
                'reason': f'{mode}_after_oom',
                'offloadMode': mode,
                'onCategories': ['oom'],
            }
            for index, mode in enumerate(self._resource_retry_modes(runtime_hints))
        ]

    def _set_param_value_if_present(self, node, key, value):
        params = node.get('params') if isinstance(node, dict) else None
        if not isinstance(params, dict) or key not in params or not isinstance(params[key], dict):
            return False
        params[key]['value'] = value
        return True

    def _set_model_repo_if_present(self, node, key, repo):
        params = node.get('params') if isinstance(node, dict) else None
        if not repo or not isinstance(params, dict) or key not in params or not isinstance(params[key], dict):
            return False
        current = params[key].get('value')
        if isinstance(current, dict):
            params[key]['value'] = {**current, 'value': repo}
        else:
            params[key]['value'] = {'source': 'hub', 'value': repo}
        return True

    def _retry_plan_matches(self, plan, classification):
        if not isinstance(plan, dict) or not isinstance(classification, dict):
            return False

        error_code = classification.get('error_code')
        category = classification.get('category')
        on_error_codes = plan.get('onErrorCodes')
        on_categories = plan.get('onCategories')

        if isinstance(on_error_codes, list) and error_code in on_error_codes:
            return True
        if isinstance(on_categories, list) and category in on_categories:
            return True
        if on_error_codes is None and on_categories is None:
            return category == 'oom'
        return False

    def _next_retry_plan_index(self, plans, current_index, classification):
        for index in range(current_index + 1, len(plans)):
            if self._retry_plan_matches(plans[index], classification):
                return index
        return None

    def _sanitize_retry_plan_for_hints(self, plan):
        if not isinstance(plan, dict):
            return None
        allowed = {
            'index',
            'reason',
            'executionPath',
            'modelRepo',
            'resolvedArtifact',
            'quantizationMode',
            'quantizedComponents',
            'bnb4ComputeDtype',
            'dtype',
            'pipelineClass',
            'offloadMode',
            'generation',
            'onCategories',
            'onErrorCodes',
        }
        return {key: deepcopy(plan.get(key)) for key in allowed if key in plan}

    def _apply_resource_retry_to_graph(self, graph, offload_mode):
        nodes = graph.get('nodes', {})
        if not isinstance(nodes, dict):
            return []

        updated = []
        for node_id, node in nodes.items():
            if not isinstance(node, dict):
                continue
            action = node.get('action')
            module = node.get('module')
            compatible_loader = (
                (module == 'modules.ModularDiffusers' and action in ('ModelsLoader', 'DynamicPipelineLoader'))
                or (module == 'modules.QwenImage' and action in ('LoadInpaintPipeline', 'LoadPipeline'))
                or (module == 'modules.WanVACE' and action == 'LoadPipeline')
                or (module in ('modules.DiffusersImage', 'modules.DiffusersAudio') and action == 'LoadPipeline')
            )
            if not compatible_loader:
                continue
            changed = self._set_param_value_if_present(node, 'offload_mode', offload_mode)
            changed = self._set_param_value_if_present(node, 'auto_offload', offload_mode != OFFLOAD_MODE_NONE) or changed
            if changed:
                updated.append(str(node_id))

        return updated

    def _apply_resource_retry_plan_to_graph(self, graph, plan):
        nodes = graph.get('nodes', {})
        if not isinstance(nodes, dict) or not isinstance(plan, dict):
            return []

        offload_mode = plan.get('offloadMode')
        model_repo = plan.get('modelRepo') or plan.get('resolvedArtifact')
        quantization_mode = plan.get('quantizationMode')
        quantized_components = plan.get('quantizedComponents')
        compute_dtype = plan.get('bnb4ComputeDtype')
        dtype = plan.get('dtype')
        generation = plan.get('generation') if isinstance(plan.get('generation'), dict) else {}

        updated = []
        for node_id, node in nodes.items():
            if not isinstance(node, dict):
                continue
            action = node.get('action')
            module = node.get('module')
            compatible_loader = (
                (module == 'modules.ModularDiffusers' and action in ('ModelsLoader', 'DynamicPipelineLoader'))
                or (module == 'modules.QwenImage' and action in ('LoadInpaintPipeline', 'LoadPipeline'))
                or (module == 'modules.WanVACE' and action == 'LoadPipeline')
                or (module in ('modules.DiffusersImage', 'modules.DiffusersAudio') and action == 'LoadPipeline')
            )
            if not compatible_loader:
                continue

            changed = False
            if isinstance(offload_mode, str):
                changed = self._set_param_value_if_present(node, 'offload_mode', offload_mode) or changed
                changed = self._set_param_value_if_present(node, 'auto_offload', offload_mode != OFFLOAD_MODE_NONE) or changed
            if isinstance(model_repo, str) and model_repo:
                changed = self._set_model_repo_if_present(node, 'model_id', model_repo) or changed
                changed = self._set_model_repo_if_present(node, 'repo_id', model_repo) or changed
            if isinstance(plan.get('pipelineClass'), str):
                changed = self._set_param_value_if_present(node, 'pipeline_class', plan.get('pipelineClass')) or changed
            if isinstance(dtype, str):
                changed = self._set_param_value_if_present(node, 'dtype', dtype) or changed
            if module == 'modules.QwenImage' and action == 'LoadPipeline':
                if isinstance(quantization_mode, str):
                    changed = self._set_param_value_if_present(node, 'quantization_mode', quantization_mode) or changed
                if isinstance(quantized_components, list):
                    changed = self._set_param_value_if_present(node, 'quantized_components', [str(item) for item in quantized_components]) or changed
                if isinstance(compute_dtype, str):
                    changed = self._set_param_value_if_present(node, 'bnb_4bit_compute_dtype', compute_dtype) or changed
                if model_repo == QWEN_IMAGE_2512_PREQUANTIZED_REPO:
                    changed = self._set_param_value_if_present(node, 'quantization_mode', 'none') or changed
                    changed = self._set_param_value_if_present(node, 'quantized_components', []) or changed
            if changed:
                updated.append(str(node_id))

            if module == 'modules.QwenImage' and action in ('Generate', 'InpaintGenerate'):
                generation_changed = False
                for plan_key, param_keys in (
                    ('width', ('width',)),
                    ('height', ('height',)),
                    ('steps', ('num_inference_steps', 'steps')),
                    ('guidanceScale', ('true_cfg_scale', 'guidance_scale', 'guidance')),
                    ('negativePrompt', ('negative_prompt',)),
                    ('maxSequenceLength', ('max_sequence_length',)),
                ):
                    if plan_key not in generation:
                        continue
                    for param_key in param_keys:
                        generation_changed = self._set_param_value_if_present(node, param_key, generation[plan_key]) or generation_changed
                if generation_changed:
                    updated.append(str(node_id))
            if module in ('modules.DiffusersImage', 'modules.DiffusersAudio') and action in ('Generate', 'Edit', 'Inpaint', 'ControlGenerate'):
                generation_changed = False
                for plan_key, param_keys in (
                    ('width', ('width',)),
                    ('height', ('height',)),
                    ('steps', ('num_inference_steps', 'steps')),
                    ('guidanceScale', ('guidance_scale', 'true_cfg_scale', 'guidance')),
                    ('negativePrompt', ('negative_prompt',)),
                    ('maxSequenceLength', ('max_sequence_length',)),
                    ('audioDuration', ('audio_duration',)),
                    ('shift', ('shift',)),
                ):
                    if plan_key not in generation:
                        continue
                    for param_key in param_keys:
                        generation_changed = self._set_param_value_if_present(node, param_key, generation[plan_key]) or generation_changed
                if generation_changed:
                    updated.append(str(node_id))

        return updated

    def _auto_resource_candidate_is_proven(self, candidate):
        if not isinstance(candidate, dict):
            return False
        proof = candidate.get('proof')
        status = proof.get('status') if isinstance(proof, dict) else None
        return status in PROVEN_PROOF_STATUSES

    def _auto_resource_requires_proven_candidate(self, runtime_hints):
        if not isinstance(runtime_hints, dict) or runtime_hints.get('resourceMode') != 'auto':
            return False
        return True

    def _assert_auto_resource_candidate_ready(self, runtime_hints):
        if not self._auto_resource_requires_proven_candidate(runtime_hints):
            return
        auto_plan = runtime_hints.get('autoResourcePlan') if isinstance(runtime_hints, dict) else None
        if self._auto_resource_candidate_is_proven(auto_plan):
            return

        candidates = runtime_hints.get('autoResourceCandidates') if isinstance(runtime_hints, dict) else None
        proven_candidates = [
            candidate for candidate in candidates
            if self._auto_resource_candidate_is_proven(candidate)
        ] if isinstance(candidates, list) else []
        if proven_candidates:
            runtime_hints['autoResourcePlan'] = proven_candidates[0]
            runtime_hints['autoResourceCandidateId'] = proven_candidates[0].get('id')
            proof = proven_candidates[0].get('proof') if isinstance(proven_candidates[0].get('proof'), dict) else {}
            runtime_hints['autoResourceProofStatus'] = proof.get('status')
            return

        status = runtime_hints.get('compatibilityStatus') or runtime_hints.get('autoResourceProofStatus') or 'unproven'
        model_name = runtime_hints.get('modelName') or runtime_hints.get('modelType') or 'this workflow'
        error = RuntimeError(
            f"Auto resource plan is not ready for {model_name}. Refresh the Auto plan or choose Expert settings before executing this workflow."
        )
        setattr(error, 'modiff_error_code', 'auto_resource_unproven')
        setattr(error, 'modiff_category', 'auto_resource')
        setattr(error, 'modiff_recovery_hint', (
            "Auto has not found a runnable artifact in the local model/cache and hardware metadata. "
            "Install the suggested compatible artifact or switch to Expert if you want to choose the configuration yourself."
        ))
        setattr(error, 'modiff_auto_resource_status', status)
        raise error

    def _record_auto_resource_success(self, runtime_hints, runtime_fingerprint):
        try:
            record_auto_resource_success(
                self.data_dir,
                runtime_fingerprint=runtime_fingerprint if isinstance(runtime_fingerprint, dict) else self._runtime_fingerprint(),
                runtime_hints=runtime_hints,
            )
        except Exception as exc:
            logger.debug(f"Could not record Auto resource success: {exc}")

    def _record_auto_resource_failure(self, error, classification=None):
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

    def _release_runtime_caches_for_retry(self):
        errors = []
        released = {
            'nodes': len(self.node_cache),
            'models': 0,
            'diffusers_components': 0,
            'offload_files': 0,
        }

        try:
            self.node_cache.clear()
        except Exception as e:
            errors.append(f'node cache: {e}')

        try:
            released['models'] = memory_manager.clear()
        except Exception as e:
            errors.append(f'memory manager: {e}')
            try:
                memory_manager.cache.clear()
            except Exception as clear_error:
                errors.append(f'memory manager fallback: {clear_error}')

        released['diffusers_components'], diffusers_errors = self._release_modular_diffusers_components()
        errors.extend(diffusers_errors)
        released['offload_files'], offload_errors = self._release_diffusers_offload_cache()
        errors.extend(offload_errors)

        try:
            gc.collect()
        except Exception as e:
            errors.append(f'gc.collect: {e}')
        errors.extend(self._best_effort_device_cache_clear())

        return {
            'released': released,
            'errors': errors,
        }

    def execute_graph(self, graph):
        sid = graph['sid']
        nodes = graph['nodes']
        paths = graph['paths']

        graph_execution_time = time.time()
        base_runtime_hints = self._coerce_runtime_hints(graph.get('runtimeHints'))
        retry_plans = self._coerce_retry_plan_list(base_runtime_hints)
        retry_history = []
        attempt_index = 0
        retry_plan_index = -1

        while True:
            if self.current_task:
                self.current_task["attempt_index"] = attempt_index
                self.current_task["progress"] = 0
            runtime_hints = deepcopy(base_runtime_hints) if base_runtime_hints else None
            active_retry_plan = retry_plans[retry_plan_index] if retry_plan_index >= 0 and retry_plan_index < len(retry_plans) else None
            if self.current_task and runtime_hints is not None:
                self.current_task['runtimeHints'] = runtime_hints
            if runtime_hints and active_retry_plan is None and attempt_index == 0:
                self._assert_auto_resource_candidate_ready(runtime_hints)
                auto_plan = runtime_hints.get('autoResourcePlan')
                if isinstance(auto_plan, dict) and self._auto_resource_candidate_is_proven(auto_plan):
                    updated_nodes = self._apply_resource_retry_plan_to_graph(graph, auto_plan)
                    if updated_nodes:
                        self.queue_message({
                            "type": "auto_resource_plan_applied",
                            "sid": sid,
                            "task_id": self.current_task.get("task_id") if self.current_task else None,
                            "attempt": attempt_index,
                            "attempt_index": attempt_index,
                            "candidateId": auto_plan.get('id'),
                            "updatedNodes": updated_nodes,
                            "message": "Applied the proven Auto resource plan before execution.",
                        }, sid)
            if runtime_hints and active_retry_plan:
                updated_nodes = self._apply_resource_retry_plan_to_graph(graph, active_retry_plan)
                retry_mode = active_retry_plan.get('offloadMode')
                if isinstance(retry_mode, str):
                    runtime_hints['offloadMode'] = retry_mode
                    runtime_hints['autoOffload'] = retry_mode != OFFLOAD_MODE_NONE
                    runtime_hints['offloadDiskPath'] = 'data/offload/diffusers' if retry_mode == OFFLOAD_MODE_GROUP_DISK else None
                if isinstance(active_retry_plan.get('modelRepo'), str):
                    runtime_hints['modelRepo'] = active_retry_plan['modelRepo']
                    runtime_hints['resolvedModelRepo'] = active_retry_plan['modelRepo']
                if isinstance(active_retry_plan.get('resolvedArtifact'), str):
                    runtime_hints['resolvedArtifact'] = active_retry_plan['resolvedArtifact']
                elif isinstance(active_retry_plan.get('modelRepo'), str):
                    runtime_hints['resolvedArtifact'] = active_retry_plan['modelRepo']
                if isinstance(active_retry_plan.get('executionPath'), str):
                    runtime_hints['executionPath'] = active_retry_plan['executionPath']
                if isinstance(active_retry_plan.get('quantizationMode'), str):
                    runtime_hints['quantizationMode'] = active_retry_plan['quantizationMode']
                if isinstance(active_retry_plan.get('quantizedComponents'), list):
                    runtime_hints['quantizedComponents'] = [str(item) for item in active_retry_plan['quantizedComponents']]
                runtime_hints['resourceRetryAttempt'] = attempt_index
                runtime_hints['resourceRetryHistory'] = retry_history
                plan = runtime_hints.get('resourcePlan')
                if isinstance(plan, dict):
                    plan['activeRetryPlan'] = self._sanitize_retry_plan_for_hints(active_retry_plan)
                    if isinstance(retry_mode, str):
                        plan['offloadMode'] = retry_mode
                        plan['autoOffload'] = retry_mode != OFFLOAD_MODE_NONE
                self.queue_message({
                    "type": "resource_retry",
                    "sid": sid,
                    "task_id": self.current_task.get("task_id") if self.current_task else None,
                    "attempt": attempt_index,
                    "attempt_index": attempt_index,
                    "offloadMode": retry_mode,
                    "retryPlan": self._sanitize_retry_plan_for_hints(active_retry_plan),
                    "updatedNodes": updated_nodes,
                    "message": f"Retrying with {str(retry_mode or active_retry_plan.get('reason') or 'safer plan').replace('_', '-')} after {retry_history[-1]['errorCode'] if retry_history else 'resource pressure'}.",
                    "history": retry_history,
                }, sid)

            runtime_budget = self._apply_cuda_runtime_budget(runtime_hints)
            deterministic = self._apply_deterministic_mode(graph) if attempt_index == 0 else None
            runtime_fingerprint = self._runtime_fingerprint()
            if self.current_task:
                self.current_task['runtimeFingerprint'] = runtime_fingerprint.get('fingerprint')
                if deterministic is not None:
                    self.current_task['deterministicMode'] = deterministic
                self.current_task['runtimeHints'] = runtime_hints
                self.current_task['runtimeBudget'] = runtime_budget

            if deterministic:
                self.queue_message({
                    "type": "deterministic_execution",
                    "sid": sid,
                    "task_id": self.current_task.get("task_id") if self.current_task else None,
                    "deterministicMode": deterministic,
                    "runtimeFingerprint": runtime_fingerprint,
                }, sid)

            node_weights = {
                id: node_execution_weight(nodes[id].get('module', ''), nodes[id].get('action', ''))
                for path in paths
                for id in path
                if id in nodes
            }
            total_task_weight = sum(node_weights.values()) or 1
            task_progress = 0.0

            try:
                for path in paths:
                    for id in path:
                        if self.interrupt_flag:
                            if self.current_task:
                                self.current_task["interrupt_requested"] = True
                            return

                        if self.current_task:
                            node = nodes.get(id, {})
                            module = node.get('module', '')
                            action = node.get('action', '')
                            self.current_task.update({
                                "updated_at": time.time(),
                                "current_node": id,
                                "current_node_name": f"{module}.{action}",
                                "node_progress": -1,
                                "phase": node_execution_phase(module, action),
                                "message": node_execution_message(module, action, node_execution_phase(module, action)),
                                "current_step": None,
                                "total_steps": None,
                                "completed_progress": task_progress,
                                "current_node_weight": node_weights.get(id, 1.0) / total_task_weight * 100,
                            })
                        self.execute_node(id, nodes[id], sid)

                        # broadcast the task progress
                        if self.current_task:
                            task_progress += node_weights.get(id, 1.0) / total_task_weight * 100
                            self.current_task['progress'] = int(task_progress)
                            self.current_task['completed_progress'] = task_progress
                            self.current_task['node_progress'] = 100
                            self.current_task['updated_at'] = time.time()
                        self.queue_message({
                            "type": "task_progress",
                            "task_id": self.current_task["task_id"],
                            "attempt_index": self.current_task.get("attempt_index"),
                            **self._current_run_identity_payload(),
                            "progress": self.current_task['progress'],
                        })
            except Exception as e:
                classification = self._classify_exception(e)
                next_retry_plan_index = None
                if classification['error_code'] != 'cuda_context_poisoned':
                    next_retry_plan_index = self._next_retry_plan_index(retry_plans, retry_plan_index, classification)
                can_retry = next_retry_plan_index is not None
                if not can_retry:
                    raise

                retry_history.append({
                    'attempt': attempt_index,
                    'offloadMode': runtime_hints.get('offloadMode') if runtime_hints else None,
                    'retryPlan': self._sanitize_retry_plan_for_hints(active_retry_plan),
                    'category': classification.get('category'),
                    'errorCode': classification.get('error_code'),
                    'error': str(e) or type(e).__name__,
                    'node': getattr(e, 'modiff_node_id', None) or getattr(e, 'mellon_node_id', None),
                    'nodeName': getattr(e, 'modiff_node_name', None) or getattr(e, 'mellon_node_name', None),
                    'loaderDiagnostics': self._loader_diagnostics_snapshot(),
                    'nextRetryPlan': self._sanitize_retry_plan_for_hints(retry_plans[next_retry_plan_index]),
                })
                if runtime_hints is not None:
                    runtime_hints['resourceRetryLastError'] = str(e) or type(e).__name__
                    runtime_hints['resourceRetryLastCode'] = classification.get('error_code')
                    runtime_hints['resourceRetryHistory'] = retry_history
                    if self.current_task:
                        self.current_task['runtimeHints'] = runtime_hints

                cleanup = self._release_runtime_caches_for_retry()
                self.queue_message({
                    "type": "resource_retry_cleanup",
                    "sid": sid,
                    "task_id": self.current_task.get("task_id") if self.current_task else None,
                    "attempt": attempt_index,
                    "attempt_index": attempt_index,
                    **cleanup,
                }, sid)
                attempt_index += 1
                retry_plan_index = next_retry_plan_index
                continue

            # the graph has completed
            self._record_auto_resource_success(runtime_hints, runtime_fingerprint)
            self.queue_message({
                "type": "graph_completed",
                "sid": sid,
                "task_id": self.current_task.get("task_id") if self.current_task else None,
                **self._current_run_identity_payload(),
                "executionTime": time.time() - graph_execution_time,
                "runtimeFingerprint": runtime_fingerprint,
                "deterministicMode": deterministic,
                "runtimeHints": runtime_hints,
                "runtimeBudget": runtime_budget,
                "resourceRetryHistory": retry_history,
            }, sid)
            return

    async def stop_execution(self, _):
        # check if there is a current task or any queued task
        if not self.current_task and not self.queued_tasks:
            return web.json_response({
                "error": True,
                "message": "Nothing to do. No task is currently running or queued.",
            })

        if self.interrupt_flag:
            return web.json_response({
                "error": True,
                "message": "Execution is already set for interruption.",
            })

        self.interrupt_flag = True
        if self.current_task:
            self.current_task["interrupt_requested"] = True
            self.current_task["updated_at"] = time.time()
            self.current_task["message"] = "Stopping after the current model step"

        # set the interrupt flag for all the nodes in the cache
        for node in self.node_cache:
            self.node_cache[node]._interrupt = True
            active_pipeline = getattr(self.node_cache[node], "_active_pipeline", None)
            if active_pipeline is not None and hasattr(active_pipeline, "_interrupt"):
                active_pipeline._interrupt = True

        return web.json_response({
            "error": False,
            "message": "Execution set for interruption.",
        })

    def _connected_output_value(self, *, target_node_id, target_node_name, target_param, source_node_id, source_key):
        source_node = self.node_cache.get(source_node_id)
        if source_node is None:
            error = MissingConnectedOutputError(
                f"Connected input '{target_param}' on {target_node_name} expected output '{source_key}' "
                f"from upstream node {source_node_id}, but that node has not executed."
            )
            setattr(error, 'modiff_node_id', source_node_id)
            setattr(error, 'modiff_node_name', 'Unknown upstream node')
            setattr(error, 'modiff_target_node_id', target_node_id)
            setattr(error, 'modiff_target_node_name', target_node_name)
            setattr(error, 'mellon_target_node_id', target_node_id)
            setattr(error, 'mellon_target_node_name', target_node_name)
            raise error

        output = getattr(source_node, 'output', None)
        source_name = f"{getattr(source_node, 'module_name', 'unknown')}.{getattr(source_node, 'class_name', 'unknown')}"
        if not isinstance(output, dict) or source_key not in output or output.get(source_key) is None:
            available = sorted(output.keys()) if isinstance(output, dict) else []
            error = MissingConnectedOutputError(
                f"Connected input '{target_param}' on {target_node_name} expected output '{source_key}' "
                f"from upstream node {source_name}, but that output was not produced. "
                f"Available outputs: {available or 'none'}."
            )
            setattr(error, 'modiff_node_id', source_node_id)
            setattr(error, 'modiff_node_name', source_name)
            setattr(error, 'modiff_target_node_id', target_node_id)
            setattr(error, 'modiff_target_node_name', target_node_name)
            setattr(error, 'mellon_target_node_id', target_node_id)
            setattr(error, 'mellon_target_node_name', target_node_name)
            raise error

        return output[source_key]

    def execute_node(self, id, node, sid, quiet=False):
        module = node['module']
        action = node['action']
        params = node['params']

        if module not in self.modules:
            raise ValueError(f"Invalid module: {module}")

        if action not in self.modules[module]:
            raise ValueError(f"Invalid action: {action}")

        # get the arguments values
        args = {}
        ui_fields = {}

        for p in params:
            data_source_id = params[p].get('sourceId')
            data_param_key = params[p].get('sourceKey')

            # the field is a UI element, used mostly to display the data in the UI
            if 'display' in params[p] and params[p]['display'] in ['ui_group', 'ui_text', 'ui_image', 'ui_imagecompare', 'ui_areaselect', 'ui_audio', 'ui_video', 'ui_3d', 'ui_label', 'ui_button']:
                ui_fields[p] = data_param_key if data_param_key else None

            # the field is an input that gets its value from an output of another node
            elif data_source_id and data_param_key:
                # spawn field handling
                #if '>>>' in p or self.modules[module][action]['params'][p].get('spawn'):
                if params[p].get('spawn'):
                    spawn_key = p.split('>>>')[0]
                    if not spawn_key in args:
                        args[spawn_key] = []

                    args[spawn_key].append(self._connected_output_value(
                        target_node_id=id,
                        target_node_name=f"{module}.{action}",
                        target_param=p,
                        source_node_id=data_source_id,
                        source_key=data_param_key,
                    ))
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
                args[p] = params[p].get('value')

        if not quiet:
            reset_memory_stats()
            start_time = time.time()
            phase = node_execution_phase(module, action)

            # tell the client that the node is running
            self.queue_message({
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
                "progress": -1, # -1 sets the progress to indeterminate
            }, sid)

        # if the node is not in the cache, initialize it
        if id not in self.node_cache:
            work_module = import_module(f"{module}.main")
            work_action = getattr(work_module, action)
            self.node_cache[id] = work_action(id)

        if not callable(self.node_cache[id]):
            raise TypeError(f"The class `{module}.{action}` is not callable. Make sure the class has a `__call__` method or extends `NodeBase`.")

        # set the session id, it can be used to send messages from the node back to the client
        self.node_cache[id]._sid = sid

        # *** execute the node ***
        try:
            self.node_cache[id](**args)
        except Exception as e:
            traceback_text = traceback.format_exc()
            logger.error(f"Error executing node {id} ({module}.{action})")
            logger.error(traceback_text)
            setattr(e, 'modiff_node_id', id)
            setattr(e, 'modiff_node_name', f"{module}.{action}")
            setattr(e, 'modiff_traceback', traceback_text)
            setattr(e, 'mellon_node_id', id)
            setattr(e, 'mellon_node_name', f"{module}.{action}")
            setattr(e, 'mellon_traceback', traceback_text)
            self.queue_message({
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
            }, sid)
            if not quiet:
                self.queue_message({
                    "type": "progress",
                    "node": id,
                    "task_id": self.current_task.get("task_id") if self.current_task else None,
                    "attempt_index": self.current_task.get("attempt_index") if self.current_task else None,
                    **self._current_run_identity_payload(),
                    "status": "failed",
                    "phase": node_execution_phase(module, action),
                    "message": f"{module}.{action} failed",
                    "progress": 0,
                }, sid)
            raise e

        if not quiet:
            execution_time = time.time() - start_time
            self.node_cache[id]._execution_time['last'] = execution_time
            self.node_cache[id]._execution_time['min'] = min(self.node_cache[id]._execution_time['min'], execution_time) if self.node_cache[id]._execution_time['min'] is not None else execution_time
            self.node_cache[id]._execution_time['max'] = max(self.node_cache[id]._execution_time['max'], execution_time) if self.node_cache[id]._execution_time['max'] is not None else execution_time

            memory_stats = get_memory_stats()
            if memory_stats:
                self.node_cache[id]._memory_usage['last'] = memory_stats['peak']
                self.node_cache[id]._memory_usage['min'] = min(self.node_cache[id]._memory_usage['min'], memory_stats['peak']) if self.node_cache[id]._memory_usage['min'] is not None else memory_stats['peak']
                self.node_cache[id]._memory_usage['max'] = max(self.node_cache[id]._memory_usage['max'], memory_stats['peak']) if self.node_cache[id]._memory_usage['max'] is not None else memory_stats['peak']

            # the node has completed
            self.queue_message({
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
                "memoryUsage": self.node_cache[id]._memory_usage
            }, sid)

        for ui_key, data_key in ui_fields.items():
            message = None

            # skip for button and group fields
            if self.modules[module][action]['params'][ui_key].get('display') in ['ui_button', 'ui_group']:
                continue

            else:
                # if the data key is an output, get the value from the output otherwise from the params
                if data_key in self.node_cache[id].output:
                    source_value = self.node_cache[id].output[data_key]
                elif data_key in self.node_cache[id].params:
                    source_value = self.node_cache[id].params[data_key]
                else:
                    continue

            data_type = self.modules[module][action]['params'][data_key].get('type') # data type of the source field
            data_format = self.modules[module][action]['params'][ui_key].get('type', 'text') # format of the returned value: text, raw, url
            fieldOptions = self.modules[module][action]['params'][ui_key].get('fieldOptions', {})

            source_value = source_value if isinstance(source_value, list) else [source_value]
            artifacts = None
            if data_format == 'url':
                if is_image_data_type(data_type):
                    image_format = fieldOptions.get('format', 'WEBP')
                    image_quality = fieldOptions.get('quality', 100)
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
                        artifacts.append(attach_run_identity_to_artifact(
                            cache_image_artifact(id, data_key, i, url, source_value[i], image_format),
                            task_id=task_id,
                            attempt_index=attempt_index,
                            runtime_hints=runtime_hints,
                        ))
                else:
                    data_value = [f"/cache/{id}/{data_key}/{i}?t={time.time()}"
                                  for i in range(len(source_value)) if source_value[i] is not None]
            elif data_format == 'raw':
                data_value = [to_bytes(data_type, item, fieldOptions) for item in source_value if item is not None]
            else:
                data_value  = [to_base64(data_type, item, fieldOptions) for item in source_value if item is not None]

            message = {
                'client_id': sid,
                'type': 'update_value',
                'node': id,
                'key': ui_key,
                'data_type': data_type,
                'value': data_value,
                'task_id': self.current_task.get("task_id") if self.current_task else None,
                'attempt_index': self.current_task.get("attempt_index") if self.current_task else None,
                **self._current_run_identity_payload(),
                'runtimeFingerprint': self.current_task.get("runtimeFingerprint") if self.current_task else None,
            }
            if artifacts is not None:
                message['artifacts'] = artifacts

            if message:
                self.queue_message(message, sid)


    def trigger_node(self, source_id, output, sid):
        if not self.current_task:
            return

        graph = self.current_task['args'][0]
        nodes = graph['nodes']

        for id in nodes:
            params = nodes[id]['params']

            for p in params:
                data_source_id = params[p].get('sourceId')
                data_param_key = params[p].get('sourceKey')
                if data_source_id == source_id and data_param_key == output:
                    self.execute_node(id, nodes[id], sid, quiet=True)


    """
    ╭────────────────╮
       Hugging Face
    ╰────────────────╯
    """

    async def hf_cache(self, request):
        refresh = request.query.get('refresh', False)
        id = request.match_info.get('id', None)
        class_name = request.query.get('className', None)
        compact = request.query.get('compact', False)
        return_type = "compact" if compact else "full"

        if refresh:
            modelstore.update_hf()

        models = modelstore.get_hf_models(id, class_name, return_type)

        return web.json_response(models)

    def _package_status(self, module_name, distribution_name=None):
        package = {
            'available': False,
            'module': module_name,
            'distribution': distribution_name or module_name,
        }

        try:
            package['version'] = metadata.version(distribution_name or module_name)
        except Exception:
            pass

        try:
            module = import_module(module_name)
            package['available'] = True
            package['version'] = getattr(module, '__version__', package.get('version'))
        except Exception as e:
            package['error'] = str(e)

        return package

    async def system_stats(self, _request):
        return web.json_response(get_hardware_snapshot(self.data_dir))

    async def runtime_status(self, request):
        hardware = get_hardware_snapshot(self.data_dir)
        packages = {
            'aiohttp': self._package_status('aiohttp'),
            'aiohttp_cors': self._package_status('aiohttp_cors', 'aiohttp-cors'),
            'torch': self._package_status('torch'),
            'diffusers': self._package_status('diffusers'),
            'transformers': self._package_status('transformers'),
            'huggingface_hub': self._package_status('huggingface_hub', 'huggingface-hub'),
            'accelerate': self._package_status('accelerate'),
            'safetensors': self._package_status('safetensors'),
        }
        packages['torch'].update(legacy_torch_status(hardware))
        required = ['aiohttp', 'aiohttp_cors', 'torch', 'diffusers', 'huggingface_hub']
        missing_required = [name for name in required if not packages.get(name, {}).get('available')]

        current_task = None
        if self.current_task:
            current_task = {
                'task_id': self.current_task.get('task_id'),
                'name': self.current_task.get('name'),
                'sid': self.current_task.get('sid'),
                'started_at': self.current_task.get('started_at'),
                'progress': self.current_task.get('progress'),
            }

        return web.json_response({
            'error': False,
            'ready': len(missing_required) == 0,
            'instance': self.instance,
            'server': {
                'host': self.host,
                'port': self.port,
                'scheme': 'https' if self.ssl_context else 'http',
                'work_dir': self.work_dir,
                'data_dir': self.data_dir,
                'client_max_size': self.client_max_size,
            },
            'python': {
                'version': sys.version,
                'executable': sys.executable,
                'platform': platform.platform(),
                'cwd': os.getcwd(),
            },
            'config': {
                'hf_cache_dir': CONFIG.hf.get('cache_dir'),
                'hf_online_status': CONFIG.hf.get('online_status'),
                'hf_token_configured': bool(CONFIG.hf.get('token')),
                'pytorch_cuda_alloc_conf': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
                'paths': CONFIG.paths,
            },
            'packages': packages,
            'hardware': hardware,
            'missing_required_packages': missing_required,
            'modules': {
                'registered_count': len(self.modules),
                'module_map_count': len(MODULE_MAP),
            },
            'queue': {
                'current': current_task,
                'queued_count': len(self.queued_tasks),
                'main_queue_size': self.main_queue.qsize(),
                'background_queue_size': self.background_queue.qsize(),
                'interrupt_requested': self.interrupt_flag,
            },
        })

    def _cuda_memory_snapshot(self):
        snapshot = {
            'available': False,
            'device_count': 0,
            'devices': [],
        }
        try:
            torch = import_module('torch')
            cuda_available = bool(torch.cuda.is_available())
            snapshot['available'] = cuda_available
            if cuda_available:
                device_count = int(torch.cuda.device_count())
                snapshot['device_count'] = device_count
                devices = []
                for index in range(device_count):
                    device = {
                        'index': index,
                        'name': torch.cuda.get_device_name(index),
                    }
                    try:
                        properties = torch.cuda.get_device_properties(index)
                        device['total_memory'] = int(getattr(properties, 'total_memory', 0))
                    except Exception as properties_error:
                        device['properties_error'] = str(properties_error)
                    try:
                        try:
                            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                        except TypeError:
                            with torch.cuda.device(index):
                                free_bytes, total_bytes = torch.cuda.mem_get_info()
                        device['free_bytes'] = int(free_bytes)
                        device['total_bytes'] = int(total_bytes)
                    except Exception as memory_error:
                        device['mem_get_info_error'] = str(memory_error)
                    try:
                        device['allocated_bytes'] = int(torch.cuda.memory_allocated(index))
                        device['reserved_bytes'] = int(torch.cuda.memory_reserved(index))
                        device['max_allocated_bytes'] = int(torch.cuda.max_memory_allocated(index))
                        device['max_reserved_bytes'] = int(torch.cuda.max_memory_reserved(index))
                    except Exception as stats_error:
                        device['memory_stats_error'] = str(stats_error)
                    devices.append(device)

                snapshot['devices'] = devices
                if devices:
                    first_device = devices[0]
                    snapshot['free_bytes'] = first_device.get('free_bytes')
                    snapshot['total_bytes'] = first_device.get('total_bytes')
                    snapshot['allocated_bytes'] = first_device.get('allocated_bytes')
                    snapshot['reserved_bytes'] = first_device.get('reserved_bytes')
                    snapshot['device_name'] = first_device.get('name')
        except Exception as e:
            snapshot['error'] = str(e)
        return snapshot

    def _gpu_process_snapshot(self):
        nvidia_smi = shutil.which('nvidia-smi')
        if not nvidia_smi:
            return {
                'available': False,
                'reason': 'nvidia-smi not found',
                'gpus': [],
                'processes': [],
            }

        snapshot = {
            'available': True,
            'gpus': [],
            'processes': [],
        }

        try:
            gpu_result = subprocess.run(
                [
                    nvidia_smi,
                    '--query-gpu=index,name,memory.used,memory.free,memory.total',
                    '--format=csv,noheader,nounits',
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
                    snapshot['gpus'].append({
                        'index': int(index) if index.isdigit() else index,
                        'name': name,
                        'memory_used_mb': self._safe_int(used_mb),
                        'memory_free_mb': self._safe_int(free_mb),
                        'memory_total_mb': self._safe_int(total_mb),
                    })
            else:
                snapshot['gpu_query_error'] = gpu_result.stderr.strip() or gpu_result.stdout.strip()
        except Exception as e:
            snapshot['gpu_query_error'] = str(e)

        try:
            process_result = subprocess.run(
                [
                    nvidia_smi,
                    '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                    '--format=csv,noheader,nounits',
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
                    snapshot['processes'].append({
                        'gpu_uuid': gpu_uuid,
                        'pid': self._safe_int(pid),
                        'process_name': process_name,
                        'used_memory_mb': self._safe_int(used_memory_mb),
                    })
            else:
                snapshot['process_query_error'] = process_result.stderr.strip() or process_result.stdout.strip()
        except Exception as e:
            snapshot['process_query_error'] = str(e)

        return snapshot

    def _safe_int(self, value):
        try:
            return int(str(value).strip())
        except Exception:
            return None

    async def runtime_gpu_processes(self, request):
        return web.json_response({
            'error': False,
            'cuda_memory_snapshot': self._cuda_memory_snapshot(),
            'gpu_processes': self._gpu_process_snapshot(),
        })

    def _best_effort_device_cache_clear(self):
        errors = []
        try:
            torch = import_module('torch')
        except Exception as e:
            return [f'torch import failed: {e}']

        if torch.cuda.is_available():
            for label, callback in (
                ('torch.cuda.empty_cache', torch.cuda.empty_cache),
                ('torch.cuda.ipc_collect', torch.cuda.ipc_collect),
            ):
                try:
                    callback()
                except Exception as e:
                    logger.debug(f"{label} failed during accelerator cleanup", exc_info=True)
                    errors.append(f'{label}: {e}')

        mps = getattr(torch, 'mps', None)
        if mps is not None:
            try:
                if mps.is_available():
                    mps.empty_cache()
            except Exception as e:
                logger.debug("torch.mps.empty_cache failed during accelerator cleanup", exc_info=True)
                errors.append(f'torch.mps.empty_cache: {e}')

        return errors

    def _release_modular_diffusers_components(self):
        try:
            modular_diffusers = import_module('modules.ModularDiffusers')
            manager = getattr(modular_diffusers, 'components', None)
        except Exception as e:
            return 0, [f'Modular Diffusers components unavailable: {e}']

        if manager is None:
            return 0, []

        components_dict = getattr(manager, 'components', None)
        released_count = len(components_dict) if components_dict is not None else 0
        errors = []

        try:
            torch = import_module('torch')
        except Exception:
            torch = None

        hooks = list(getattr(manager, 'model_hooks', None) or [])
        for hook in hooks:
            for label, callback in (
                ('offload', getattr(hook, 'offload', None)),
                ('remove', getattr(hook, 'remove', None)),
            ):
                if callback is None:
                    continue
                try:
                    callback()
                except Exception as e:
                    logger.debug(f"Modular Diffusers hook {label} failed during cleanup", exc_info=True)
                    errors.append(f'Modular Diffusers hook {label}: {e}')

        try:
            manager.model_hooks = None
            manager._auto_offload_enabled = False
            if hasattr(manager, '_auto_offload_device'):
                manager._auto_offload_device = None
        except Exception as e:
            errors.append(f'Modular Diffusers offload reset: {e}')

        if components_dict is not None:
            for component_id, component in list(components_dict.items()):
                try:
                    if torch is not None and isinstance(component, torch.nn.Module):
                        component.to('cpu')
                except Exception as e:
                    logger.debug(f"Could not move Modular Diffusers component {component_id} to CPU", exc_info=True)
                    errors.append(f'Modular Diffusers component {component_id}: {e}')

            try:
                components_dict.clear()
            except Exception as e:
                errors.append(f'Modular Diffusers component clear: {e}')

        for attr in ('added_time', 'collections'):
            try:
                value = getattr(manager, attr, None)
                if value is not None:
                    value.clear()
            except Exception as e:
                errors.append(f'Modular Diffusers {attr} clear: {e}')

        return released_count, errors

    def _release_diffusers_offload_cache(self):
        offload_path = Path('data') / 'offload' / 'diffusers'
        if not offload_path.exists():
            return 0, []

        errors = []
        released_count = 0
        try:
            released_count = sum(1 for item in offload_path.rglob('*') if item.is_file())
            shutil.rmtree(offload_path)
        except Exception as e:
            logger.debug("Diffusers disk offload cache cleanup failed", exc_info=True)
            errors.append(f'Diffusers disk offload cache: {e}')
        return released_count, errors

    async def runtime_gpu_cleanup(self, request):
        before = self._cuda_memory_snapshot()
        cleanup_errors = []
        released_nodes = len(self.node_cache)
        released_models = 0
        released_diffusers_components = 0
        released_offload_files = 0

        try:
            self.node_cache.clear()
        except Exception as e:
            logger.debug("Failed to clear node cache during accelerator cleanup", exc_info=True)
            cleanup_errors.append(f'node cache: {e}')

        try:
            released_models = memory_manager.clear()
        except Exception as e:
            logger.debug("Failed to clear managed models during accelerator cleanup", exc_info=True)
            cleanup_errors.append(f'memory manager: {e}')
            try:
                released_models = len(memory_manager.cache)
                memory_manager.cache.clear()
            except Exception as clear_error:
                cleanup_errors.append(f'memory manager fallback: {clear_error}')

        released_diffusers_components, diffusers_errors = self._release_modular_diffusers_components()
        cleanup_errors.extend(diffusers_errors)

        released_offload_files, offload_cache_errors = self._release_diffusers_offload_cache()
        cleanup_errors.extend(offload_cache_errors)

        try:
            gc.collect()
        except Exception as e:
            cleanup_errors.append(f'gc.collect: {e}')

        cleanup_errors.extend(self._best_effort_device_cache_clear())
        after = self._cuda_memory_snapshot()

        message = (
            f'Accelerator cleanup complete. Released {released_nodes} cached node object(s), '
            f'{released_models} managed model(s), {released_diffusers_components} Modular Diffusers component(s), '
            f'and {released_offload_files} Diffusers disk offload file(s).'
        )
        if cleanup_errors:
            message += ' Some cleanup calls reported errors but references were dropped where possible.'

        return web.json_response({
            'error': False,
            'message': message,
            'released_node_count': released_nodes,
            'released_model_count': released_models,
            'released_diffusers_component_count': released_diffusers_components,
            'released_diffusers_offload_file_count': released_offload_files,
            'cleanup_errors': cleanup_errors,
            'before': before,
            'after': after,
        })

    async def model_capabilities(self, request):
        query = str(request.query.get('q', '')).lower().strip()
        capabilities = list(STUDIO_MODEL_CAPABILITIES.values())
        if query:
            capabilities = [
                capability
                for capability in capabilities
                if query in capability.get('modelType', '').lower()
                or query in capability.get('label', '').lower()
                or query in capability.get('family', '').lower()
                or query in capability.get('defaultRepo', '').lower()
            ]

        return web.json_response({
            'error': False,
            'count': len(capabilities),
            'capabilities': capabilities,
            'diffusersExecutionProfiles': public_execution_profiles(),
            'source': 'modiff-backend',
        })

    async def auto_resource_plan(self, request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        try:
            plan = build_auto_resource_plan(
                payload if isinstance(payload, dict) else {},
                runtime_fingerprint=self._runtime_fingerprint(),
                local_models=get_local_models(),
                data_dir=self.data_dir,
                history=read_auto_resource_history(self.data_dir),
            )
            return web.json_response(plan)
        except Exception as exc:
            return web.json_response({
                'error': True,
                'status': 'needs_setup',
                'statusLabel': 'Needs setup',
                'message': str(exc) or type(exc).__name__,
            }, status=500)

    async def auto_resource_plans(self, request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        try:
            result = build_auto_resource_plans(
                payload if isinstance(payload, dict) else {},
                runtime_fingerprint=self._runtime_fingerprint(),
                local_models=get_local_models(),
                data_dir=self.data_dir,
            )
            return web.json_response(result)
        except Exception as exc:
            return web.json_response({
                'error': True,
                'message': str(exc) or type(exc).__name__,
                'plans': [],
                'count': 0,
            }, status=500)

    async def auto_resource_history(self, _request):
        return web.json_response({
            'error': False,
            'history': read_auto_resource_history(self.data_dir),
        })

    async def auto_resource_history_clear(self, request):
        model_type = request.query.get('modelType') or None
        mode = request.query.get('mode') or None
        artifact = request.query.get('artifact') or None
        removed = clear_auto_resource_history(
            self.data_dir,
            model_type=model_type,
            mode=mode,
            artifact=artifact,
        )
        return web.json_response({
            'error': False,
            'removed': removed,
            'history': read_auto_resource_history(self.data_dir),
        })

    def _timestamp_seconds(self, value):
        if hasattr(value, 'timestamp'):
            return int(value.timestamp())
        if isinstance(value, (int, float)):
            return int(value)
        return None

    def _model_revision_records(self, model):
        revisions = []
        for index, revision in enumerate(model.get('revisions') or []):
            commit_hash = revision.get('hash') if isinstance(revision, dict) else None
            if not commit_hash:
                continue
            revisions.append({
                'hash': commit_hash,
                'size': revision.get('size', 0),
                'lastModified': self._timestamp_seconds(revision.get('last_modified')),
                'order': index,
            })
        revisions.sort(key=lambda item: ((item.get('lastModified') or 0), item.get('order') or 0))
        return revisions

    def _model_fingerprint(self, repo_id, revisions):
        payload = {
            'repoId': repo_id,
            'revisions': [revision.get('hash') for revision in revisions],
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()
        return f"sha256:{digest}"

    async def model_fingerprints(self, request):
        query = str(request.query.get('q', '')).lower().strip()
        model_type = str(request.query.get('modelType', '')).strip()
        repo_query = str(request.query.get('repo') or request.query.get('repoId') or '').strip()
        capability_by_model = {
            capability.get('modelType'): capability
            for capability in STUDIO_MODEL_CAPABILITIES.values()
        }
        repo_to_model_types = {}
        for capability in STUDIO_MODEL_CAPABILITIES.values():
            repo_id = capability.get('defaultRepo')
            if repo_id:
                repo_to_model_types.setdefault(repo_id, []).append(capability.get('modelType'))

        requested_repos = []
        if model_type and model_type in capability_by_model:
            default_repo = capability_by_model[model_type].get('defaultRepo')
            if default_repo:
                requested_repos.append(default_repo)
        if repo_query:
            requested_repos.append(repo_query)
        requested_repo_set = {repo.lower() for repo in requested_repos}

        models = []
        found_repo_ids = set()
        for model in get_local_models():
            repo_id = model.get('id')
            if not repo_id:
                continue
            repo_id_lower = repo_id.lower()
            if requested_repo_set and repo_id_lower not in requested_repo_set:
                continue
            if query and query not in repo_id_lower and not any(query in str(name).lower() for name in model.get('class_names', [])):
                continue

            revisions = self._model_revision_records(model)
            selected_revision = revisions[-1]['hash'] if revisions else None
            found_repo_ids.add(repo_id_lower)
            models.append({
                'repoId': repo_id,
                'modelTypes': repo_to_model_types.get(repo_id, []),
                'installed': True,
                'selectedRevision': selected_revision,
                'modelRevision': f"{repo_id}@{selected_revision}" if selected_revision else None,
                'fingerprint': self._model_fingerprint(repo_id, revisions),
                'size': model.get('size', 0),
                'classNames': model.get('class_names', []),
                'cacheDirs': model.get('cache_dirs') or ([model.get('cache_dir')] if model.get('cache_dir') else []),
                'revisions': revisions,
            })

        for repo_id in requested_repos:
            if repo_id.lower() in found_repo_ids:
                continue
            models.append({
                'repoId': repo_id,
                'modelTypes': repo_to_model_types.get(repo_id, [model_type] if model_type else []),
                'installed': False,
                'selectedRevision': None,
                'modelRevision': None,
                'fingerprint': None,
                'size': 0,
                'classNames': [],
                'cacheDirs': [],
                'revisions': [],
            })

        runtime_fingerprint = self._runtime_fingerprint()
        return web.json_response({
            'error': False,
            'count': len(models),
            'models': models,
            'runtimeFingerprint': runtime_fingerprint.get('fingerprint'),
            'source': 'modiff-backend',
        })

    async def model_cache_diagnostics(self, request):
        refresh = str(request.query.get('refresh', '')).lower() in ('1', 'true', 'yes')
        if refresh:
            modelstore.actualize()

        return web.json_response(get_cache_diagnostics())

    def _custom_modules_root(self):
        root = Path('custom').resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _disabled_custom_modules_root(self):
        root = (self._custom_modules_root() / '.disabled').resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _safe_custom_module_name(self, value):
        name = str(value or '').strip()
        if not name:
            raise ValueError('Module name is required.')
        if not re.match(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$', name):
            raise ValueError('Module name may only contain letters, numbers, dot, underscore, and dash, and must not start with a dot.')
        return name

    def _derive_custom_module_name(self, source):
        source_text = str(source or '').strip().rstrip('/\\')
        if not source_text:
            raise ValueError('Module source is required.')
        source_text = source_text[:-4] if source_text.endswith('.git') else source_text
        name = re.split(r'[/\\:]', source_text)[-1]
        name = re.sub(r'[^A-Za-z0-9_.-]+', '-', name).strip('.-')
        return self._safe_custom_module_name(name)

    def _custom_module_path(self, name, disabled=False):
        safe_name = self._safe_custom_module_name(name)
        root = self._disabled_custom_modules_root() if disabled else self._custom_modules_root()
        target = (root / safe_name).resolve()
        if target.parent != root:
            raise ValueError('Resolved custom module path escaped the custom module directory.')
        return target

    def _is_git_source(self, source):
        source = str(source or '').strip().lower()
        return source.startswith(('https://', 'http://', 'ssh://', 'git@')) or source.endswith('.git')

    def _run_git(self, args, cwd=None, timeout=300):
        git_bin = shutil.which('git')
        if not git_bin:
            raise RuntimeError('git is not available in the MoDiff backend environment.')

        completed = subprocess.run(
            [git_bin, *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        result = {
            'returncode': completed.returncode,
            'stdout': completed.stdout.strip(),
            'stderr': completed.stderr.strip(),
        }
        if completed.returncode != 0:
            message = result['stderr'] or result['stdout'] or f'git exited with {completed.returncode}'
            raise RuntimeError(message)
        return result

    def _git_value(self, module_path, args):
        try:
            return self._run_git(args, cwd=module_path, timeout=10).get('stdout', '')
        except Exception:
            return ''

    def _custom_module_git_info(self, module_path):
        is_git = bool(self._git_value(module_path, ['rev-parse', '--is-inside-work-tree']))
        if not is_git:
            return {
                'hasGit': False,
                'canUpdate': False,
            }
        return {
            'hasGit': True,
            'canUpdate': True,
            'remote': self._git_value(module_path, ['config', '--get', 'remote.origin.url']),
            'branch': self._git_value(module_path, ['rev-parse', '--abbrev-ref', 'HEAD']),
            'commit': self._git_value(module_path, ['rev-parse', '--short', 'HEAD']),
        }

    def _custom_module_info(self, name, module_path, enabled=True):
        module_key = f'custom.{name}'
        node_actions = sorted((self.modules.get(module_key) or {}).keys()) if enabled else []
        git_info = self._custom_module_git_info(module_path)
        return {
            'name': name,
            'moduleKey': module_key,
            'source': 'custom',
            'enabled': enabled,
            'status': 'enabled' if enabled else 'disabled',
            'path': str(module_path),
            'hasInit': (module_path / '__init__.py').exists(),
            'hasMain': (module_path / 'main.py').exists(),
            'nodeCount': len(node_actions),
            'nodes': node_actions,
            'canDisable': enabled,
            'canEnable': not enabled,
            **git_info,
        }

    def _list_custom_modules(self):
        root = self._custom_modules_root()
        disabled_root = self._disabled_custom_modules_root()
        modules = []

        for entry in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if not entry.is_dir() or entry.name.startswith('.') or entry.name == '__pycache__':
                continue
            modules.append(self._custom_module_info(entry.name, entry, enabled=True))

        for entry in sorted(disabled_root.iterdir(), key=lambda item: item.name.lower()):
            if not entry.is_dir() or entry.name.startswith('.') or entry.name == '__pycache__':
                continue
            modules.append(self._custom_module_info(entry.name, entry, enabled=False))

        return modules

    def _refresh_custom_module_registry(self):
        for key in list(MODULE_MAP.keys()):
            if key.startswith('custom.'):
                MODULE_MAP.pop(key, None)

        for key in list(sys.modules.keys()):
            if key == 'custom' or key.startswith('custom.'):
                sys.modules.pop(key, None)

        invalidate_caches()
        custom_root = self._custom_modules_root()
        if custom_root.exists():
            parse_module_map('custom')

        self.modules = MODULE_MAP
        self.instance = nanoid.generate(size=10)
        return self._list_custom_modules()

    def _prune_custom_node_cache(self, module_key=None):
        removed = []
        for node_id, cached_node in list(self.node_cache.items()):
            cached_module = getattr(cached_node, 'module_name', '')
            if module_key is None:
                should_remove = str(cached_module).startswith('custom.')
            else:
                should_remove = cached_module == module_key
            if should_remove:
                self.node_cache.pop(node_id, None)
                removed.append(node_id)
        return removed

    def _custom_modules_payload(self):
        modules = self._list_custom_modules()
        return {
            'error': False,
            'root': str(self._custom_modules_root()),
            'disabledRoot': str(self._disabled_custom_modules_root()),
            'count': len(modules),
            'modules': modules,
        }

    async def custom_modules_list(self, request):
        return web.json_response(self._custom_modules_payload())

    async def custom_modules_refresh(self, request):
        self._prune_custom_node_cache()
        modules = self._refresh_custom_module_registry()
        return web.json_response({
            'error': False,
            'message': 'Custom module registry refreshed.',
            'instance': self.instance,
            'count': len(modules),
            'modules': modules,
        })

    async def custom_modules_install(self, request):
        try:
            data = await request.json()
            source = str(data.get('source') or data.get('url') or '').strip()
            name = self._safe_custom_module_name(data.get('name') or self._derive_custom_module_name(source))
            target = self._custom_module_path(name)
            disabled_target = self._custom_module_path(name, disabled=True)

            if target.exists() or disabled_target.exists():
                return web.json_response({'error': True, 'message': f'Custom module `{name}` already exists.'}, status=409)

            if self._is_git_source(source):
                self._run_git(['clone', source, str(target)], timeout=900)
            else:
                source_path = Path(source).expanduser().resolve()
                if not source_path.is_dir():
                    return web.json_response({'error': True, 'message': 'Source must be a Git URL or an existing local directory.'}, status=400)
                shutil.copytree(source_path, target, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache', '.mypy_cache'))

            modules = self._refresh_custom_module_registry()
            return web.json_response({
                'error': False,
                'message': f'Custom module `{name}` installed.',
                'module': next((item for item in modules if item['name'] == name), None),
                'modules': modules,
                'instance': self.instance,
            })
        except Exception as e:
            logger.error(f"Error installing custom module: {e}", exc_info=True)
            return web.json_response({'error': True, 'message': str(e)}, status=500)

    async def custom_modules_update(self, request):
        try:
            name = self._safe_custom_module_name(request.match_info.get('name'))
            enabled_path = self._custom_module_path(name)
            disabled_path = self._custom_module_path(name, disabled=True)
            module_path = enabled_path if enabled_path.exists() else disabled_path
            if not module_path.exists():
                return web.json_response({'error': True, 'message': f'Custom module `{name}` was not found.'}, status=404)

            if not (module_path / '.git').exists():
                return web.json_response({'error': True, 'message': f'Custom module `{name}` is not a Git checkout.'}, status=400)

            git_result = self._run_git(['pull', '--ff-only'], cwd=module_path, timeout=900)
            removed_cache_nodes = self._prune_custom_node_cache(f'custom.{name}')
            modules = self._refresh_custom_module_registry() if enabled_path.exists() else self._list_custom_modules()
            return web.json_response({
                'error': False,
                'message': f'Custom module `{name}` updated.',
                'git': git_result,
                'removedCacheNodes': removed_cache_nodes,
                'module': next((item for item in modules if item['name'] == name), None),
                'modules': modules,
                'instance': self.instance,
            })
        except Exception as e:
            logger.error(f"Error updating custom module: {e}", exc_info=True)
            return web.json_response({'error': True, 'message': str(e)}, status=500)

    async def custom_modules_disable(self, request):
        try:
            name = self._safe_custom_module_name(request.match_info.get('name'))
            source = self._custom_module_path(name)
            target = self._custom_module_path(name, disabled=True)
            if not source.exists():
                return web.json_response({'error': True, 'message': f'Custom module `{name}` is not enabled.'}, status=404)
            if target.exists():
                return web.json_response({'error': True, 'message': f'Disabled custom module `{name}` already exists.'}, status=409)

            shutil.move(str(source), str(target))
            removed_cache_nodes = self._prune_custom_node_cache(f'custom.{name}')
            modules = self._refresh_custom_module_registry()
            return web.json_response({
                'error': False,
                'message': f'Custom module `{name}` disabled.',
                'removedCacheNodes': removed_cache_nodes,
                'module': next((item for item in modules if item['name'] == name), None),
                'modules': modules,
                'instance': self.instance,
            })
        except Exception as e:
            logger.error(f"Error disabling custom module: {e}", exc_info=True)
            return web.json_response({'error': True, 'message': str(e)}, status=500)

    async def custom_modules_enable(self, request):
        try:
            name = self._safe_custom_module_name(request.match_info.get('name'))
            source = self._custom_module_path(name, disabled=True)
            target = self._custom_module_path(name)
            if not source.exists():
                return web.json_response({'error': True, 'message': f'Custom module `{name}` is not disabled.'}, status=404)
            if target.exists():
                return web.json_response({'error': True, 'message': f'Enabled custom module `{name}` already exists.'}, status=409)

            shutil.move(str(source), str(target))
            modules = self._refresh_custom_module_registry()
            return web.json_response({
                'error': False,
                'message': f'Custom module `{name}` enabled.',
                'module': next((item for item in modules if item['name'] == name), None),
                'modules': modules,
                'instance': self.instance,
            })
        except Exception as e:
            logger.error(f"Error enabling custom module: {e}", exc_info=True)
            return web.json_response({'error': True, 'message': str(e)}, status=500)

    async def hf_cache_delete(self, request):
        hashes = request.match_info.get('hash').split(',')
        if not hashes:
            return web.json_response({"error": "Incorrect request, `hash` is required."}, status=400)

        result = delete_model(*hashes)
        return web.json_response({"error": not result})

    # TODO: not yet implemented
    async def hf_hub(self, request):
        query = request.query.get('q', '')
        sid = request.query.get('sid')

        future = asyncio.Future()
        await self.queue_task(search_hub, query, future, sid, name=f"Hugging Face search")

        try:
            result = await future
            return web.json_response(result)
        except Exception as e:
            logger.error(f"Error in hf_hub endpoint: {e}")
            return web.json_response({"error": str(e)}, status=500)

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
            else:
                progress_value = float(progress or 0)
                message["progress"] = progress_value

            if "status" not in message:
                message["status"] = "downloading" if progress_value is None or progress_value < 1 else "complete"

            for download_sid in list(entry.get("sids", [])):
                self.queue_message(message, download_sid)

        for download_sid in list(entry.get("sids", [])):
            self.queue_message({
                "type": "hf_download_progress",
                "repo_id": repo_id,
                "task_id": task_id,
                "download_id": task_id,
                "progress": 0,
                "status": "queued",
                "phase": "queued",
                "started_at": entry.get("started_at"),
                "updated_at": time.time(),
            }, download_sid)

        async with self.hf_download_semaphore:
            for download_sid in list(entry.get("sids", [])):
                self.queue_message({
                    "type": "hf_download_progress",
                    "repo_id": repo_id,
                    "task_id": task_id,
                    "download_id": task_id,
                    "progress": 0,
                    "status": "planning",
                    "phase": "planning",
                    "started_at": entry.get("started_at"),
                    "updated_at": time.time(),
                }, download_sid)

            result = await self.loop.run_in_executor(None, partial(download_hub_model, repo_id, progress_cb, bool(entry.get("repair"))))
            if result:
                modelstore.actualize()
            return result

    async def hf_download(self, request):
        repo_id = request.query.get('repo_id')
        sid = request.query.get('sid')
        repair = str(request.query.get('repair') or '').lower() in {'1', 'true', 'yes'}

        if not repo_id:
            return web.json_response({"error": "Incorrect request, `repo_id` is required."}, status=400)

        if repo_id in self.hf_download_tasks:
            entry = self.hf_download_tasks[repo_id]
            if sid:
                entry["sids"].add(sid)
                self.queue_message({
                    "type": "hf_download_progress",
                    "repo_id": repo_id,
                    "task_id": entry["task_id"],
                    "download_id": entry["task_id"],
                    "status": "joined",
                    "phase": "queued",
                    "progress": None,
                    "started_at": entry.get("started_at"),
                    "updated_at": time.time(),
                }, sid)
        else:
            task_id = nanoid.generate(size=12)
            entry = {
                "task_id": task_id,
                "sids": set([sid] if sid else []),
                "started_at": time.time(),
                "repair": repair,
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
                return web.json_response({
                    "error": reason,
                    "result": result,
                    "task_id": entry["task_id"],
                    "repo_id": repo_id,
                    "repair_required": True,
                }, status=409)
            return web.json_response({"error": False, "result": result, "task_id": entry["task_id"], "repo_id": repo_id})
        except Exception as e:
            logger.error(f"Error in hf_download endpoint: {e}")
            return web.json_response({"error": str(e)}, status=500)
        finally:
            if repo_id in self.hf_download_tasks and self.hf_download_tasks[repo_id].get("future") is entry.get("future") and entry["future"].done():
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
        return web.json_response({
            "error": False,
            "token_configured": True,
            "account_type": identity.get("type") if isinstance(identity, dict) else None,
        })


    """
    ╭───────────────╮
        Websocket
    ╰───────────────╯
    """
    async def websocket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        sid = request.query.get('sid')
        if not sid:
            sid = nanoid.generate(size=10)
        if sid in self.ws_sessions:
            # close the connection and remove the old session
            logger.debug(f"Websocket session {sid} already exists, closing the old session.")
            #await self.ws_sessions[sid].close()
            sid = nanoid.generate(size=10)
            #del self.ws_sessions[sid]

        self.ws_sessions[sid] = ws
        logger.debug(f"Websocket connection opened: {sid}")

        # Update the session id in all cached nodes
        for node in self.node_cache:
            # check if the node has a _sid attribute
            if hasattr(self.node_cache[node], '_sid') and self.node_cache[node]._sid != sid:
                self.node_cache[node]._sid = sid

        # Restore global execution truth as part of the connection handshake so
        # panels never become responsible for discovering active work.
        queued_tasks, current_task = self._get_queue()
        await self.broadcast({
            "type": "welcome",
            "instance": self.instance,
            "sid": sid,
            "cachedNodes": list(self.node_cache.keys()),
            "queued": queued_tasks,
            "current": current_task,
            "recent": self.recent_tasks,
        }, sid)

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    data = json.loads(msg.data)

                    if data['type'] == 'close':
                        await ws.close()
                        break
                    elif data['type'] == 'ping':
                        await self.broadcast({"type": "pong"}, sid)
                    elif data['type'] == 'signal_value':
                        request_id = data.get('request_id')
                        if not request_id:
                            logger.warning("[Websocket] signal_value received without request_id")
                            continue
                        # Resolve the pending request future if present
                        try:
                            promised_sid = data.get('sid', None)
                            future = self.pending_ws_requests.pop(request_id, None)
                            if future is None:
                                logger.debug(f"[Websocket] signal_value received for unknown request_id: {request_id}")
                                continue
                            if not future.done():
                                result = data.get('value')
                                if promised_sid != sid:
                                    result = { '__MODIFF_ERROR': 'sid_mismatch' }
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
            try:
                if session in self.ws_sessions and not self.ws_sessions[session].closed:
                    if isinstance(message, dict):
                        await self.ws_sessions[session].send_json(message)
                    else:
                        await self.ws_sessions[session].send_bytes(message)
            except Exception as e:
                logger.error(f"[Websocket] Error broadcasting message: {e}")
                pass

    def queue_message(self, message: dict | bytes, sid: list[str] | str = None, exclude: list[str] | str = None):
        if self.loop.is_running() and not self._shutdown_event.is_set():
            asyncio.run_coroutine_threadsafe(
                self.background_queue.put((self.broadcast, (message, sid, exclude))),
                self.loop
            )

    def get_signal_value(self, node: str, field: str, sid: str, timeout: int = 2):
        try:
            if not sid or sid not in self.ws_sessions or self.ws_sessions[sid].closed:
                return { '__MODIFF_ERROR': 'invalid_sid' }

            if not getattr(self, 'loop', None) or not self.loop.is_running():
                return { '__MODIFF_ERROR': 'server_not_running' }

            try:
                running_loop = asyncio.get_running_loop()
                if running_loop is self.loop:
                    logger.warning("[Server] get_signal_value called from event loop thread; returning None to avoid deadlock.")
                    return { '__MODIFF_ERROR': 'called_from_event_loop' }
            except RuntimeError:
                # No running loop in this thread; safe to proceed
                pass

            # Create a future bound to the server loop and register it
            request_id = nanoid.generate(size=12)
            future = self.loop.create_future()
            self.pending_ws_requests[request_id] = future

            # Send the request to the target client
            self.queue_message({
                "type": "get_signal_value",
                "request_id": request_id,
                "node": node,
                "field": field,
                "sid": sid
            }, sid)

            # Await the future result from outside the event loop thread
            # Use run_coroutine_threadsafe to wait with a timeout safely
            wrapped = asyncio.wait_for(future, timeout=timeout)
            cfut = asyncio.run_coroutine_threadsafe(wrapped, self.loop)
            try:
                result = cfut.result(timeout=timeout + 0.5)
            except Exception:
                result = { '__MODIFF_ERROR': 'timeout' }
            finally:
                # Cleanup any leftover pending entry
                self.pending_ws_requests.pop(request_id, None)

            return result
        except Exception as e:
            logger.error(f"[Server] get_signal_value error: {e}")
            return { '__MODIFF_ERROR': 'exception' }


def to_base64(type, value, options={}):
    import io
    import base64

    out = value

    if type == 'image':
        format = options.get('format', 'WEBP').upper()
        quality = options.get('quality')
        if format == 'WEBP' and not quality:
            quality = 100
        elif format == 'JPEG' and not quality:
            quality = 75
        elif format == 'PNG' and not quality:
            quality = None
        mime_type = f"image/{format.lower()}"

        byte_arr = io.BytesIO()
        save_kwargs = {"format": format}
        if quality is not None:
            save_kwargs["quality"] = int(quality)
        value.save(byte_arr, **save_kwargs)
        # TODO: check shutil.copyfile
        header = f"data:{mime_type};base64,"
        out = header + base64.b64encode(byte_arr.getvalue()).decode('utf-8')

    return out

def to_bytes(data_type, value, options={}):
    import io
    from PIL import Image

    out = value

    if isinstance(value, Image.Image):
        format = options.get('format', 'WEBP').upper()
        quality = options.get('quality')
        if format == 'WEBP' and not quality:
            quality = 100
        elif format == 'JPEG' and not quality:
            quality = 75
        elif format == 'PNG' and not quality:
            quality = None

        byte_arr = io.BytesIO()
        save_kwargs = {"format": format}
        if quality is not None:
            save_kwargs["quality"] = int(quality)
        value.save(byte_arr, **save_kwargs)
        out = byte_arr.getvalue()
    elif isinstance(value, str):
        out = value.encode('utf-8')

    return out

server = WebServer(
    MODULE_MAP,
    host=CONFIG.server['host'],
    port=CONFIG.server['port'],
    secure=CONFIG.server['secure'],
    certfile=CONFIG.server['certfile'],
    keyfile=CONFIG.server['keyfile'],
    cors=CONFIG.server['cors'],
    cors_routes=CONFIG.server['cors_routes'],
    client_max_size=CONFIG.server['client_max_size'],
    work_dir=CONFIG.paths['work_dir'],
    data_dir=CONFIG.paths['data']
)
