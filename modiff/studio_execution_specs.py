from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from modiff.diffusers_offload_modes import (
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_NONE,
    OFFLOAD_MODE_SEQUENTIAL_CPU,
)
from modiff.model_artifact_catalog import require_catalog_revision


STUDIO_EXECUTION_SPEC_SCHEMA_VERSION = 1
STUDIO_EXECUTION_SPEC_CANONICALIZATION_VERSION = 1

FLUX_SCHNELL_REPO = "black-forest-labs/FLUX.1-schnell"
FLUX_DEV_REPO = "black-forest-labs/FLUX.1-dev"
FLUX_DEV_FP8_REPO = "black-forest-labs/FLUX.1-dev-FP8"
FLUX_KREA_REPO = "black-forest-labs/FLUX.1-Krea-dev"
FLUX_DEPTH_REPO = "black-forest-labs/FLUX.1-Depth-dev"
FLUX_CANNY_REPO = "black-forest-labs/FLUX.1-Canny-dev"
FLUX_CANNY_VERIFIED_REPAIR_REPO = "fuliucansheng/FLUX.1-Canny-dev-diffusers"
FLUX_REDUX_REPO = "black-forest-labs/FLUX.1-Redux-dev"
FLUX_KONTEXT_REPO = "black-forest-labs/FLUX.1-Kontext-dev"
FLUX_KONTEXT_NVFP4_REPO = "black-forest-labs/FLUX.1-Kontext-dev-NVFP4"
FLUX_FILL_REPO = "black-forest-labs/FLUX.1-Fill-dev"
FLUX2_KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
SDXL_BASE_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_TURBO_REPO = "stabilityai/sdxl-turbo"
SDXL_INSTRUCT_PIX2PIX_REPO = "diffusers/sdxl-instructpix2pix-768"
SDXL_CONTROLNET_CANNY_REPO = "diffusers/controlnet-canny-sdxl-1.0"
SDXL_T2I_ADAPTER_CANNY_REPO = "TencentARC/t2i-adapter-canny-sdxl-1.0"
SD15_BASE_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD15_CONTROLNET_CANNY_REPO = "lllyasviel/control_v11p_sd15_canny"
SANA_REPO = "Efficient-Large-Model/Sana_600M_1024px_diffusers"
SANA_SPRINT_REPO = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
PIXART_SIGMA_REPO = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
AURAFLOW_V03_REPO = "fal/AuraFlow-v0.3"
CHROMA1_HD_REPO = "lodestones/Chroma1-HD"
DREAMLITE_BASE_REPO = "carlofkl/DreamLite-base"
DREAMLITE_MOBILE_REPO = "carlofkl/DreamLite-mobile"
LCM_DREAMSHAPER_REPO = "SimianLuo/LCM_Dreamshaper_v7"
MARIGOLD_DEPTH_LCM_REPO = "prs-eth/marigold-depth-lcm-v1-0"
WHISPER_TINY_REPO = "openai/whisper-tiny"
WAN_22_I2V_A14B_REPO = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
WAN_22_TI2V_5B_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN_T2V_1_3B_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
LTX_VIDEO_REPO = "Lightricks/LTX-Video-0.9.8-13B-distilled"
LTX_VIDEO_FALLBACK_REPO = "Lightricks/LTX-Video"
ACE_STEP_REPO = "ACE-Step/acestep-v15-xl-turbo-diffusers"
ACE_STEP_LORA_BASE_REPO = "Runware/acestep-v15-turbo-diffusers"
STABLE_AUDIO_REPO = "stabilityai/stable-audio-open-1.0"
LONGCAT_AUDIO_DIT_REPO = "ruixiangma/LongCat-AudioDiT-1B-Diffusers"
AUDIO_LDM2_REPO = "cvssp/audioldm2"
SHAP_E_REPO = "openai/shap-e"
WAN_22_T2V_A14B_REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
WAN_ANIMATE_REPO = "Wan-AI/Wan2.2-Animate-14B-Diffusers"
WAN_FLF_REPO = "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"
LTX2_REPO = "Lightricks/LTX-2"
FRAMEPACK_REPO = "lllyasviel/FramePackI2V_HY"
STABLE_VIDEO_DIFFUSION_REPO = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
ANIMATEDIFF_MOTION_REPO = "guoyww/animatediff-motion-adapter-v1-5-2"
ANIMATELCM_MOTION_REPO = "wangfuyun/AnimateLCM"
COGVIDEOX_2B_REPO = "zai-org/CogVideoX-2b"
QWEN_CONTROLNET_REPO = "InstantX/Qwen-Image-ControlNet-Union"
QWEN_IMAGE_2512_REPO = "Qwen/Qwen-Image-2512"
Z_IMAGE_REPO = "Tongyi-MAI/Z-Image-Turbo"
DDPM_CIFAR10_REPO = "google/ddpm-cifar10-32"
CONSISTENCY_IMAGENET64_REPO = "openai/diffusers-cd_imagenet64_l2"

_STUDIO_MODEL_DEPENDENCY_REQUIREMENTS = {
    ("AnimateDiffPipeline", "text_to_video"): (
        {
            "id": "animatediff-motion-adapter-v1-5-2",
            "label": "AnimateDiff SD1.5 v2 MotionAdapter",
            "repo": ANIMATEDIFF_MOTION_REPO,
            "revision": require_catalog_revision(ANIMATEDIFF_MOTION_REPO),
            "kind": "adapter",
            "requiredForModes": ["text_to_video"],
            "description": "Exact fp16 safetensors motion module for the reviewed AnimateDiff SD1.5 recipe.",
        },
    ),
    ("AnimateLCMPipeline", "text_to_video"): (
        {
            "id": "animatelcm-motion-adapter-and-lora",
            "label": "AnimateLCM MotionAdapter and spatial LoRA",
            "repo": ANIMATELCM_MOTION_REPO,
            "revision": require_catalog_revision(ANIMATELCM_MOTION_REPO),
            "kind": "adapter",
            "requiredForModes": ["text_to_video"],
            "description": "Exact fp16 safetensors motion module and named safetensors LoRA for the LCM recipe.",
        },
    ),
    ("StableDiffusionXLAdapterPipeline", "control_image"): (
        {
            "id": "sdxl-t2i-adapter-canny",
            "label": "Stable Diffusion XL Canny T2I Adapter",
            "repo": SDXL_T2I_ADAPTER_CANNY_REPO,
            "revision": require_catalog_revision(SDXL_T2I_ADAPTER_CANNY_REPO),
            "kind": "t2i_adapter",
            "requiredForModes": ["control_image"],
            "description": "Required by the generic SDXL Canny T2I-Adapter workflow.",
        },
    ),
    ("StableDiffusionXLControlNetPipeline", "control_image"): (
        {
            "id": "sdxl-controlnet-canny",
            "label": "Stable Diffusion XL Canny ControlNet",
            "repo": SDXL_CONTROLNET_CANNY_REPO,
            "revision": require_catalog_revision(SDXL_CONTROLNET_CANNY_REPO),
            "kind": "controlnet",
            "requiredForModes": ["control_image"],
            "description": "Required by the generic SDXL Canny ControlNet workflow.",
        },
    ),
    ("StableDiffusionPipeline", "control_image"): (
        {
            "id": "sd15-controlnet-canny",
            "label": "Stable Diffusion 1.5 Canny ControlNet",
            "repo": SD15_CONTROLNET_CANNY_REPO,
            "revision": require_catalog_revision(SD15_CONTROLNET_CANNY_REPO),
            "kind": "controlnet",
            "requiredForModes": ["control_image"],
            "description": "Required by the generic SD1.5 Canny ControlNet workflow.",
        },
    ),
    ("QwenImageModularPipeline", "control_image"): (
        {
            "id": "qwen-controlnet-union",
            "label": "Qwen ControlNet Union",
            "repo": QWEN_CONTROLNET_REPO,
            "revision": require_catalog_revision(QWEN_CONTROLNET_REPO),
            "kind": "controlnet",
            "requiredForModes": ["control_image"],
            "description": "Required for Qwen Image Control image workflows.",
        },
    ),
    ("FluxReduxPipeline", "edit_image"): (
        {
            "id": "flux-redux-base",
            "label": "FLUX.1-dev base pipeline",
            "repo": FLUX_DEV_REPO,
            "revision": require_catalog_revision(FLUX_DEV_REPO, model_type="FluxDevPipeline"),
            "kind": "base",
            "requiredForModes": ["edit_image"],
            "description": "Redux supplies reference embeddings to the app-installed FLUX.1-dev base pipeline.",
        },
    ),
}


def studio_model_requirements_for_pair(model_type: str, mode: str) -> list[dict[str, Any]]:
    return deepcopy(list(_STUDIO_MODEL_DEPENDENCY_REQUIREMENTS.get((model_type, mode), ())))


def studio_model_dependencies_for_pair(model_type: str, mode: str) -> list[dict[str, str]]:
    return [
        {key: requirement[key] for key in ("id", "kind", "repo", "revision")}
        for requirement in studio_model_requirements_for_pair(model_type, mode)
    ]

_GIB = 1024**3
_HIGH_MEMORY_FULL_RESIDENCY = {
    "accelerator": "cuda",
    "vramBytes": 80 * _GIB,
    "systemRamBytes": 64 * _GIB,
}
_DIRECT_OFFLOAD_MODES = (
    OFFLOAD_MODE_NONE,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_SEQUENTIAL_CPU,
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
)

_GRAPH_ROLES = (
    ("diffusersQuantization", "modules.DiffusersRuntime.PipelineQuantizationConfigV2", -1280, -80),
    ("diffusersRecipe", "modules.DiffusersRuntime.DiffusersExecutionRecipe", -900, -80),
    ("diffusersImagePipeline", "modules.DiffusersImage.LoadPipeline", -520, -80),
    ("diffusersImageGenerate", "modules.DiffusersImage.Generate", -120, -80),
    ("preview", "modules.Image.Preview", 980, -80),
)
_GRAPH_EDGES = (
    ("diffusersQuantization", "quantization_config", "diffusersRecipe", "quantization_config"),
    ("diffusersRecipe", "execution_recipe", "diffusersImagePipeline", "execution_recipe"),
    ("diffusersImagePipeline", "pipeline", "diffusersImageGenerate", "pipeline"),
    ("diffusersImageGenerate", "images", "preview", "image"),
)
_IMAGE_PIPELINE_BINDINGS = (
    ("diffusersQuantization", "backend", "quantizationMode"),
    ("diffusersQuantization", "components", "quantizedComponents"),
    ("diffusersQuantization", "dtype", "dtype"),
    ("diffusersRecipe", "device_map", "deviceMapNone"),
    ("diffusersRecipe", "offload_mode", "offloadMode"),
    ("diffusersRecipe", "device", "device"),
    ("diffusersRecipe", "attention_backend", "attentionBackend"),
    ("diffusersRecipe", "attention_components", "empty"),
    ("diffusersRecipe", "vae_slicing", "true"),
    ("diffusersRecipe", "vae_tiling", "true"),
    ("diffusersRecipe", "regional_compile", "regionalCompile"),
    ("diffusersRecipe", "denoiser_cache", "denoiserCache"),
    ("diffusersRecipe", "layerwise_casting", "layerwiseCasting"),
    ("diffusersRecipe", "channels_last", "channelsLast"),
    ("diffusersImagePipeline", "model_id", "artifact"),
    ("diffusersImagePipeline", "pipeline_class", "pipelineClass"),
    ("diffusersImagePipeline", "mode", "mode"),
    ("diffusersImagePipeline", "dtype", "dtype"),
    ("diffusersImagePipeline", "device", "device"),
    ("diffusersImagePipeline", "quantization_mode", "quantizationMode"),
    ("diffusersImagePipeline", "quantized_components", "pipelineQuantizedComponents"),
    ("diffusersImagePipeline", "auto_offload", "autoOffload"),
    ("diffusersImagePipeline", "offload_mode", "offloadMode"),
)
_GRAPH_BINDINGS = _IMAGE_PIPELINE_BINDINGS + (
    ("diffusersImageGenerate", "prompt", "prompt"),
    ("diffusersImageGenerate", "negative_prompt", "negativePrompt"),
    ("diffusersImageGenerate", "width", "width"),
    ("diffusersImageGenerate", "height", "height"),
    ("diffusersImageGenerate", "seed", "seed"),
    ("diffusersImageGenerate", "num_inference_steps", "steps"),
    ("diffusersImageGenerate", "guidance_scale", "guidanceScale"),
    ("diffusersImageGenerate", "strength", "strength"),
    ("diffusersImageGenerate", "output_type", "outputType"),
    ("diffusersImageGenerate", "max_sequence_length", "maxSequenceLength"),
)
_SDXL_GRAPH_BINDINGS = _GRAPH_BINDINGS + (
    ("diffusersImagePipeline", "revision", "defaultRevision"),
)
_PAG_GRAPH_BINDINGS = _SDXL_GRAPH_BINDINGS + (
    ("diffusersImageGenerate", "pag_scale", "pagScale"),
    ("diffusersImageGenerate", "pag_adaptive_scale", "pagAdaptiveScale"),
)
_PERCEPTION_GRAPH_ROLES = (
    ("diffusersQuantization", "modules.DiffusersRuntime.PipelineQuantizationConfigV2", -1280, -80),
    ("diffusersRecipe", "modules.DiffusersRuntime.DiffusersExecutionRecipe", -900, -80),
    ("diffusersImagePipeline", "modules.DiffusersImage.LoadPipeline", -520, -80),
    ("loadImage", "modules.Image.Load", -520, 300),
    ("diffusersPredictMap", "modules.DiffusersImage.PredictMap", -120, -80),
    ("preview", "modules.Image.Preview", 980, -80),
)
_PERCEPTION_GRAPH_EDGES = (
    ("diffusersQuantization", "quantization_config", "diffusersRecipe", "quantization_config"),
    ("diffusersRecipe", "execution_recipe", "diffusersImagePipeline", "execution_recipe"),
    ("diffusersImagePipeline", "pipeline", "diffusersPredictMap", "pipeline"),
    ("loadImage", "image", "diffusersPredictMap", "image"),
    ("diffusersPredictMap", "preview_images", "preview", "image"),
)
_PERCEPTION_GRAPH_BINDINGS = _IMAGE_PIPELINE_BINDINGS + (
    ("diffusersImagePipeline", "revision", "defaultRevision"),
    ("loadImage", "file", "referenceImages"),
    ("loadImage", "alpha_channel", "alphaMode"),
    ("diffusersPredictMap", "prediction_kind", "depth"),
    ("diffusersPredictMap", "seed", "seed"),
    ("diffusersPredictMap", "num_inference_steps", "steps"),
    ("diffusersPredictMap", "processing_resolution", "processingResolution"),
    ("diffusersPredictMap", "match_input_resolution", "matchInputResolution"),
)
_SPEECH_GRAPH_ROLES = (
    ("speechModel", "modules.HuggingFaceSpeech.LoadSpeechRecognitionModel", -720, -80),
    ("loadAudio", "modules.Audio.Load", -720, 280),
    ("transcribeAudio", "modules.HuggingFaceSpeech.TranscribeAudio", -240, -80),
    ("transcriptPreview", "modules.Primitive.DataViewer", 240, -80),
)
_SPEECH_GRAPH_EDGES = (
    ("speechModel", "model", "transcribeAudio", "model"),
    ("loadAudio", "audio", "transcribeAudio", "audio"),
    ("transcribeAudio", "transcript", "transcriptPreview", "value"),
)
_SPEECH_TRANSCRIPTION_GRAPH_BINDINGS = (
    ("speechModel", "model_id", "artifact"),
    ("speechModel", "revision", "defaultRevision"),
    ("speechModel", "dtype", "dtype"),
    ("speechModel", "device", "device"),
    ("loadAudio", "file", "sourceAudio"),
    ("transcribeAudio", "task", "transcribe"),
    ("transcribeAudio", "language", "speechLanguage"),
    ("transcribeAudio", "timestamps", "speechTimestamps"),
    ("transcribeAudio", "chunk_length_seconds", "speechChunkSeconds"),
    ("transcribeAudio", "stride_length_seconds", "speechStrideSeconds"),
)
_SPEECH_TRANSLATION_GRAPH_BINDINGS = tuple(
    (role, param, "translate" if source == "transcribe" else source)
    for role, param, source in _SPEECH_TRANSCRIPTION_GRAPH_BINDINGS
)
_MODULAR_EDIT_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -720, -80),
    ("prompt", "modules.ModularDiffusers.EncodePrompt", -360, -240),
    ("loadImage", "modules.Image.Load", -720, 320),
    ("imageEncode", "modules.ModularDiffusers.ImageEncode", -360, 320),
    ("denoise", "modules.ModularDiffusers.Denoise", 80, -80),
    ("decode", "modules.ModularDiffusers.DecodeLatents", 440, -80),
    ("preview", "modules.Image.Preview", 800, -80),
)
_MODULAR_EDIT_GRAPH_EDGES = (
    ("models", "text_encoders", "prompt", "text_encoders"),
    ("models", "unet_out", "denoise", "unet"),
    ("models", "scheduler", "denoise", "scheduler"),
    ("models", "vae_out", "imageEncode", "vae"),
    ("models", "vae_out", "decode", "vae"),
    ("loadImage", "image", "prompt", "image"),
    ("loadImage", "image", "imageEncode", "image"),
    ("prompt", "embeddings", "denoise", "embeddings"),
    ("imageEncode", "image_latents", "denoise", "image_latents"),
    ("imageEncode", "route_state_out", "denoise", "route_state_in"),
    ("denoise", "latents", "decode", "latents"),
    ("denoise", "route_state_out", "decode", "route_state_in"),
    ("decode", "images", "preview", "image"),
)
_MODULAR_EDIT_GRAPH_BINDINGS = (
    ("models", "model_type", "pipelineClass"),
    ("models", "repo_id", "artifact"),
    ("models", "dtype", "dtype"),
    ("models", "device", "device"),
    ("models", "auto_offload", "autoOffload"),
    ("models", "offload_mode", "offloadMode"),
    ("models", "trust_remote_code", "false"),
    ("loadImage", "file", "referenceImages"),
    ("loadImage", "alpha_channel", "alphaMode"),
    ("prompt", "prompt", "prompt"),
    ("prompt", "negative_prompt", "negativePrompt"),
    ("imageEncode", "seed", "seed"),
    ("denoise", "seed", "seed"),
    ("denoise", "num_inference_steps", "steps"),
    ("denoise", "guidance_scale", "guidanceScale"),
)
_MODULAR_LAYERED_GRAPH_EDGES = tuple(
    edge for edge in _MODULAR_EDIT_GRAPH_EDGES if edge[1] != "route_state_out"
)
_MODULAR_LAYERED_GRAPH_BINDINGS = (
    ("models", "model_type", "pipelineClass"),
    ("models", "repo_id", "artifact"),
    ("models", "dtype", "dtype"),
    ("models", "device", "device"),
    ("models", "auto_offload", "autoOffload"),
    ("models", "offload_mode", "offloadMode"),
    ("models", "trust_remote_code", "false"),
    ("loadImage", "file", "referenceImages"),
    ("loadImage", "alpha_channel", "addAlpha"),
    ("prompt", "prompt", "prompt"),
    ("prompt", "negative_prompt", "negativePrompt"),
    ("prompt", "max_sequence_length", "maxSequenceLength"),
    ("imageEncode", "seed", "seed"),
    ("denoise", "seed", "seed"),
    ("denoise", "num_inference_steps", "steps"),
    ("denoise", "guidance_scale", "guidanceScale"),
    ("denoise", "layers", "layers"),
)
_MODULAR_CONTROL_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -720, -80),
    ("prompt", "modules.ModularDiffusers.EncodePrompt", -360, -240),
    ("loadImage", "modules.Image.Load", -720, 320),
    ("controlnetModel", "modules.ModularDiffusers.AutoModelLoader", -360, 520),
    ("controlnet", "modules.ModularDiffusers.Controlnet", 80, 320),
    ("denoise", "modules.ModularDiffusers.Denoise", 80, -80),
    ("decode", "modules.ModularDiffusers.DecodeLatents", 440, -80),
    ("preview", "modules.Image.Preview", 800, -80),
)
_MODULAR_CONTROL_GRAPH_EDGES = (
    ("models", "text_encoders", "prompt", "text_encoders"),
    ("models", "unet_out", "denoise", "unet"),
    ("models", "scheduler", "denoise", "scheduler"),
    ("models", "vae_out", "controlnet", "vae"),
    ("models", "vae_out", "decode", "vae"),
    ("loadImage", "image", "controlnet", "control_image"),
    ("controlnetModel", "model", "controlnet", "controlnet"),
    ("prompt", "embeddings", "denoise", "embeddings"),
    ("controlnet", "controlnet_bundle", "denoise", "controlnet_bundle"),
    ("controlnet", "route_state_out", "denoise", "route_state_in"),
    ("denoise", "latents", "decode", "latents"),
    ("denoise", "route_state_out", "decode", "route_state_in"),
    ("decode", "images", "preview", "image"),
)
_MODULAR_CONTROL_GRAPH_BINDINGS = (
    ("models", "model_type", "pipelineClass"),
    ("models", "repo_id", "artifact"),
    ("models", "dtype", "dtype"),
    ("models", "device", "device"),
    ("models", "auto_offload", "autoOffload"),
    ("models", "offload_mode", "offloadMode"),
    ("models", "trust_remote_code", "false"),
    ("loadImage", "file", "controlImage"),
    ("loadImage", "alpha_channel", "alphaMode"),
    ("controlnetModel", "model_type", "kind"),
    ("controlnetModel", "model_id", "repo"),
    ("controlnetModel", "dtype", "dtype"),
    ("controlnetModel", "subfolder", "empty"),
    ("controlnetModel", "variant", "empty"),
    ("controlnetModel", "trust_remote_code", "false"),
    ("controlnetModel", "revision", "revision"),
    ("controlnetModel", "device", "device"),
    ("controlnetModel", "auto_offload", "autoOffload"),
    ("controlnetModel", "offload_mode", "offloadMode"),
    ("prompt", "prompt", "prompt"),
    ("prompt", "negative_prompt", "negativePrompt"),
    ("controlnet", "model_type", "pipelineClass"),
    ("controlnet", "width", "width"),
    ("controlnet", "height", "height"),
    ("controlnet", "seed", "seed"),
    ("controlnet", "controlnet_conditioning_scale", "conditioningScale"),
    ("denoise", "width", "width"),
    ("denoise", "height", "height"),
    ("denoise", "seed", "seed"),
    ("denoise", "num_inference_steps", "steps"),
    ("denoise", "guidance_scale", "guidanceScale"),
    ("denoise", "strength", "strength"),
)
_CONTROL_GRAPH_ROLES = (
    ("diffusersQuantization", "modules.DiffusersRuntime.PipelineQuantizationConfigV2", -1280, -80),
    ("diffusersRecipe", "modules.DiffusersRuntime.DiffusersExecutionRecipe", -900, -80),
    ("diffusersImagePipeline", "modules.DiffusersImage.LoadPipeline", -520, -80),
    ("loadImage", "modules.Image.Load", -520, 300),
    ("diffusersImageControl", "modules.DiffusersImage.ControlGenerate", -120, -80),
    ("preview", "modules.Image.Preview", 980, -80),
)
_CONTROL_GRAPH_EDGES = (
    ("diffusersQuantization", "quantization_config", "diffusersRecipe", "quantization_config"),
    ("diffusersRecipe", "execution_recipe", "diffusersImagePipeline", "execution_recipe"),
    ("diffusersImagePipeline", "pipeline", "diffusersImageControl", "pipeline"),
    ("loadImage", "image", "diffusersImageControl", "control_image"),
    ("diffusersImageControl", "images", "preview", "image"),
)
_CONDITIONED_CONTROL_GRAPH_ROLES = (
    ("diffusersQuantization", "modules.DiffusersRuntime.PipelineQuantizationConfigV2", -1280, -80),
    ("diffusersRecipe", "modules.DiffusersRuntime.DiffusersExecutionRecipe", -900, -80),
    ("diffusersImagePipeline", "modules.DiffusersImage.LoadPipeline", -520, -80),
    ("loadImage", "modules.Image.Load", -900, 300),
    ("controlPreprocessor", "modules.ImageFilters.Canny", -520, 300),
    ("diffusersImageControl", "modules.DiffusersImage.ControlGenerate", -120, -80),
    ("preview", "modules.Image.Preview", 980, -80),
)
_CONDITIONED_CONTROL_GRAPH_EDGES = (
    ("diffusersQuantization", "quantization_config", "diffusersRecipe", "quantization_config"),
    ("diffusersRecipe", "execution_recipe", "diffusersImagePipeline", "execution_recipe"),
    ("diffusersImagePipeline", "pipeline", "diffusersImageControl", "pipeline"),
    ("loadImage", "image", "controlPreprocessor", "image"),
    ("controlPreprocessor", "output", "diffusersImageControl", "control_image"),
    ("diffusersImageControl", "images", "preview", "image"),
)
_CONTROL_GRAPH_BINDINGS = _IMAGE_PIPELINE_BINDINGS + (
    ("loadImage", "file", "controlImage"),
    ("loadImage", "alpha_channel", "alphaMode"),
    ("diffusersImageControl", "prompt", "prompt"),
    ("diffusersImageControl", "negative_prompt", "negativePrompt"),
    ("diffusersImageControl", "width", "width"),
    ("diffusersImageControl", "height", "height"),
    ("diffusersImageControl", "seed", "seed"),
    ("diffusersImageControl", "num_inference_steps", "steps"),
    ("diffusersImageControl", "guidance_scale", "guidanceScale"),
    ("diffusersImageControl", "strength", "strength"),
    ("diffusersImageControl", "output_type", "outputType"),
    ("diffusersImageControl", "max_sequence_length", "maxSequenceLength"),
)
_CONDITIONED_CONTROL_GRAPH_BINDINGS = _IMAGE_PIPELINE_BINDINGS + (
    ("diffusersImagePipeline", "revision", "defaultRevision"),
    ("diffusersImagePipeline", "conditioning_kind", "kind"),
    ("diffusersImagePipeline", "conditioning_model_id", "repo"),
    ("diffusersImagePipeline", "conditioning_revision", "revision"),
    ("loadImage", "file", "controlImage"),
    ("loadImage", "alpha_channel", "alphaMode"),
    ("controlPreprocessor", "low_threshold", "cannyLowThreshold"),
    ("controlPreprocessor", "high_threshold", "cannyHighThreshold"),
    ("controlPreprocessor", "device", "device"),
    ("diffusersImageControl", "prompt", "prompt"),
    ("diffusersImageControl", "negative_prompt", "negativePrompt"),
    ("diffusersImageControl", "width", "width"),
    ("diffusersImageControl", "height", "height"),
    ("diffusersImageControl", "seed", "seed"),
    ("diffusersImageControl", "num_inference_steps", "steps"),
    ("diffusersImageControl", "guidance_scale", "guidanceScale"),
    ("diffusersImageControl", "conditioning_scale", "conditioningScale"),
    ("diffusersImageControl", "output_type", "outputType"),
    ("diffusersImageControl", "max_sequence_length", "maxSequenceLength"),
)
_EDIT_GRAPH_ROLES = (
    ("diffusersQuantization", "modules.DiffusersRuntime.PipelineQuantizationConfigV2", -1280, -80),
    ("diffusersRecipe", "modules.DiffusersRuntime.DiffusersExecutionRecipe", -900, -80),
    ("diffusersImagePipeline", "modules.DiffusersImage.LoadPipeline", -520, -80),
    ("loadImage", "modules.Image.Load", -520, 300),
    ("diffusersImageEdit", "modules.DiffusersImage.Edit", -120, -80),
    ("preview", "modules.Image.Preview", 980, -80),
)
_EDIT_GRAPH_EDGES = (
    ("diffusersQuantization", "quantization_config", "diffusersRecipe", "quantization_config"),
    ("diffusersRecipe", "execution_recipe", "diffusersImagePipeline", "execution_recipe"),
    ("diffusersImagePipeline", "pipeline", "diffusersImageEdit", "pipeline"),
    ("loadImage", "image", "diffusersImageEdit", "image"),
    ("diffusersImageEdit", "images", "preview", "image"),
)
_EDIT_GRAPH_BINDINGS = _IMAGE_PIPELINE_BINDINGS + (
    ("loadImage", "file", "referenceImages"),
    ("loadImage", "alpha_channel", "alphaMode"),
    ("diffusersImageEdit", "prompt", "prompt"),
    ("diffusersImageEdit", "negative_prompt", "negativePrompt"),
    ("diffusersImageEdit", "width", "width"),
    ("diffusersImageEdit", "height", "height"),
    ("diffusersImageEdit", "seed", "seed"),
    ("diffusersImageEdit", "num_inference_steps", "steps"),
    ("diffusersImageEdit", "guidance_scale", "guidanceScale"),
    ("diffusersImageEdit", "strength", "strength"),
    ("diffusersImageEdit", "reference_strength", "conditioningScale"),
    ("diffusersImageEdit", "output_type", "outputType"),
    ("diffusersImageEdit", "max_sequence_length", "maxSequenceLength"),
)
_SDXL_EDIT_GRAPH_BINDINGS = _EDIT_GRAPH_BINDINGS + (
    ("diffusersImagePipeline", "revision", "defaultRevision"),
)
_PAG_EDIT_GRAPH_BINDINGS = _SDXL_EDIT_GRAPH_BINDINGS + (
    ("diffusersImageEdit", "pag_scale", "pagScale"),
    ("diffusersImageEdit", "pag_adaptive_scale", "pagAdaptiveScale"),
)
_SDXL_INSTRUCT_EDIT_GRAPH_BINDINGS = _SDXL_EDIT_GRAPH_BINDINGS + (
    ("diffusersImageEdit", "image_guidance_scale", "conditioningScale"),
)
_DREAMLITE_GRAPH_BINDINGS = tuple(
    item for item in _SDXL_GRAPH_BINDINGS if item[:2] != ("diffusersImageGenerate", "strength")
)
_DREAMLITE_EDIT_GRAPH_BINDINGS = tuple(
    item
    for item in _SDXL_EDIT_GRAPH_BINDINGS
    if item[:2]
    not in {
        ("diffusersImageEdit", "strength"),
        ("diffusersImageEdit", "reference_strength"),
        ("diffusersImageEdit", "max_sequence_length"),
    }
) + (("diffusersImageEdit", "image_guidance_scale", "conditioningScale"),)
_DREAMLITE_MOBILE_GRAPH_BINDINGS = tuple(
    item
    for item in _DREAMLITE_GRAPH_BINDINGS
    if item[:2]
    not in {
        ("diffusersImageGenerate", "negative_prompt"),
        ("diffusersImageGenerate", "guidance_scale"),
    }
)
_DREAMLITE_MOBILE_EDIT_GRAPH_BINDINGS = tuple(
    item
    for item in _SDXL_EDIT_GRAPH_BINDINGS
    if item[:2]
    not in {
        ("diffusersImageEdit", "negative_prompt"),
        ("diffusersImageEdit", "guidance_scale"),
        ("diffusersImageEdit", "strength"),
        ("diffusersImageEdit", "reference_strength"),
        ("diffusersImageEdit", "max_sequence_length"),
    }
)
_INPAINT_GRAPH_ROLES = (
    ("diffusersQuantization", "modules.DiffusersRuntime.PipelineQuantizationConfigV2", -1280, -80),
    ("diffusersRecipe", "modules.DiffusersRuntime.DiffusersExecutionRecipe", -900, -80),
    ("diffusersImagePipeline", "modules.DiffusersImage.LoadPipeline", -520, -80),
    ("loadImage", "modules.Image.Load", -520, 300),
    ("loadMask", "modules.Image.Load", -520, 560),
    ("diffusersImageInpaint", "modules.DiffusersImage.Inpaint", -120, -80),
    ("preview", "modules.Image.Preview", 980, -80),
)
_INPAINT_GRAPH_EDGES = (
    ("diffusersQuantization", "quantization_config", "diffusersRecipe", "quantization_config"),
    ("diffusersRecipe", "execution_recipe", "diffusersImagePipeline", "execution_recipe"),
    ("diffusersImagePipeline", "pipeline", "diffusersImageInpaint", "pipeline"),
    ("loadImage", "image", "diffusersImageInpaint", "image"),
    ("loadMask", "image", "diffusersImageInpaint", "mask_image"),
    ("diffusersImageInpaint", "images", "preview", "image"),
)
_INPAINT_GRAPH_BINDINGS = _IMAGE_PIPELINE_BINDINGS + (
    ("loadImage", "file", "referenceImages"),
    ("loadImage", "alpha_channel", "alphaMode"),
    ("loadMask", "file", "maskImage"),
    ("loadMask", "alpha_channel", "removeAlpha"),
    ("diffusersImageInpaint", "prompt", "prompt"),
    ("diffusersImageInpaint", "negative_prompt", "negativePrompt"),
    ("diffusersImageInpaint", "width", "width"),
    ("diffusersImageInpaint", "height", "height"),
    ("diffusersImageInpaint", "seed", "seed"),
    ("diffusersImageInpaint", "num_inference_steps", "steps"),
    ("diffusersImageInpaint", "guidance_scale", "guidanceScale"),
    ("diffusersImageInpaint", "strength", "strength"),
    ("diffusersImageInpaint", "reference_strength", "conditioningScale"),
    ("diffusersImageInpaint", "output_type", "outputType"),
    ("diffusersImageInpaint", "max_sequence_length", "maxSequenceLength"),
)
_SDXL_INPAINT_GRAPH_BINDINGS = _INPAINT_GRAPH_BINDINGS + (
    ("diffusersImagePipeline", "revision", "defaultRevision"),
)
_PAG_INPAINT_GRAPH_BINDINGS = _SDXL_INPAINT_GRAPH_BINDINGS + (
    ("diffusersImageInpaint", "pag_scale", "pagScale"),
    ("diffusersImageInpaint", "pag_adaptive_scale", "pagAdaptiveScale"),
)
_QWEN_OUTPAINT_GRAPH_ROLES = (
    ("diffusersQuantization", "modules.DiffusersRuntime.PipelineQuantizationConfigV2", -1280, -80),
    ("diffusersRecipe", "modules.DiffusersRuntime.DiffusersExecutionRecipe", -900, -80),
    ("diffusersImagePipeline", "modules.DiffusersImage.LoadPipeline", -520, -80),
    ("loadImage", "modules.Image.Load", -520, 300),
    ("qwenOutpaintCanvas", "modules.DiffusersImage.OutpaintCanvas", -520, 300),
    ("diffusersImageInpaint", "modules.DiffusersImage.Inpaint", -120, -80),
    ("preview", "modules.Image.Preview", 980, -80),
)
_QWEN_OUTPAINT_GRAPH_EDGES = (
    ("diffusersQuantization", "quantization_config", "diffusersRecipe", "quantization_config"),
    ("diffusersRecipe", "execution_recipe", "diffusersImagePipeline", "execution_recipe"),
    ("diffusersImagePipeline", "pipeline", "diffusersImageInpaint", "pipeline"),
    ("loadImage", "image", "qwenOutpaintCanvas", "image"),
    ("qwenOutpaintCanvas", "canvas", "diffusersImageInpaint", "image"),
    ("qwenOutpaintCanvas", "mask_image", "diffusersImageInpaint", "mask_image"),
    ("diffusersImageInpaint", "images", "preview", "image"),
)
_QWEN_OUTPAINT_GRAPH_BINDINGS = tuple(item for item in _INPAINT_GRAPH_BINDINGS if item[0] != "loadMask") + (
    ("qwenOutpaintCanvas", "width", "width"),
    ("qwenOutpaintCanvas", "height", "height"),
    ("qwenOutpaintCanvas", "left", "outpaintLeft"),
    ("qwenOutpaintCanvas", "right", "outpaintRight"),
    ("qwenOutpaintCanvas", "top", "outpaintTop"),
    ("qwenOutpaintCanvas", "bottom", "outpaintBottom"),
    ("qwenOutpaintCanvas", "overlap", "outpaintOverlap"),
    ("qwenOutpaintCanvas", "feather", "outpaintFeather"),
    ("qwenOutpaintCanvas", "fill_color", "outpaintFillColor"),
)
_VIDEO_GRAPH_ROLES = (
    ("diffusersQuantization", "modules.DiffusersRuntime.PipelineQuantizationConfigV2", -1280, -80),
    ("diffusersRecipe", "modules.DiffusersRuntime.DiffusersExecutionRecipe", -900, -80),
    ("wanPipeline", "modules.DiffusersVideo.LoadPipeline", -520, -80),
    ("wanGenerate", "modules.DiffusersVideo.Generate", 220, -80),
    ("videoExport", "modules.Video.Export", 640, -80),
)
_VIDEO_GRAPH_EDGES = (
    ("diffusersQuantization", "quantization_config", "diffusersRecipe", "quantization_config"),
    ("diffusersRecipe", "execution_recipe", "wanPipeline", "execution_recipe"),
    ("wanPipeline", "pipeline", "wanGenerate", "pipeline"),
    ("wanGenerate", "video_out", "videoExport", "video"),
)
_VIDEO_GRAPH_BINDINGS = (
    ("diffusersQuantization", "backend", "quantizationMode"),
    ("diffusersQuantization", "components", "quantizedComponents"),
    ("diffusersQuantization", "dtype", "dtype"),
    ("diffusersRecipe", "device_map", "deviceMapNone"),
    ("diffusersRecipe", "offload_mode", "offloadMode"),
    ("diffusersRecipe", "device", "device"),
    ("diffusersRecipe", "attention_backend", "nativeFlashAttention"),
    ("diffusersRecipe", "attention_components", "transformer"),
    ("diffusersRecipe", "vae_slicing", "true"),
    ("diffusersRecipe", "vae_tiling", "videoVaeTiling"),
    ("diffusersRecipe", "regional_compile", "regionalCompile"),
    ("diffusersRecipe", "denoiser_cache", "denoiserCache"),
    ("diffusersRecipe", "layerwise_casting", "layerwiseCasting"),
    ("diffusersRecipe", "channels_last", "channelsLast"),
    ("wanPipeline", "model_id", "artifact"),
    ("wanPipeline", "pipeline_class", "pipelineClass"),
    ("wanPipeline", "revision", "empty"),
    ("wanPipeline", "dtype", "dtype"),
    ("wanPipeline", "device", "device"),
    ("wanPipeline", "auto_offload", "autoOffload"),
    ("wanPipeline", "offload_mode", "offloadMode"),
    ("wanGenerate", "prompt", "prompt"),
    ("wanGenerate", "mode", "mode"),
    ("wanGenerate", "negative_prompt", "negativePrompt"),
    ("wanGenerate", "width", "width"),
    ("wanGenerate", "height", "height"),
    ("wanGenerate", "seed", "seed"),
    ("wanGenerate", "num_frames", "numFrames"),
    ("wanGenerate", "num_inference_steps", "steps"),
    ("wanGenerate", "guidance_scale", "guidanceScale"),
    ("wanGenerate", "scheduler_flow_shift", "shift"),
    ("wanGenerate", "conditioning_scale", "conditioningScale"),
    ("wanGenerate", "strength", "strength"),
    ("wanGenerate", "denoise_strength", "strength"),
    ("wanGenerate", "frame_rate", "fps"),
    ("wanGenerate", "guidance_scale_2", "guidanceScale2"),
    ("wanGenerate", "use_guidance_scale_2", "useGuidanceScale2"),
    ("wanGenerate", "output_type", "outputType"),
    ("wanGenerate", "max_sequence_length", "maxSequenceLength"),
    ("wanGenerate", "attention_kwargs_json", "attentionKwargsJson"),
    ("videoExport", "fps", "fps"),
)
_WAN_VACE_GRAPH_BINDINGS = tuple(
    (role, param, "wanVaceRevision") if role == "wanPipeline" and param == "revision" else (role, param, source)
    for role, param, source in _VIDEO_GRAPH_BINDINGS
)
_I2V_GRAPH_ROLES = _VIDEO_GRAPH_ROLES + (
    ("loadImage", "modules.Image.Load", -520, 300),
)
_I2V_GRAPH_EDGES = _VIDEO_GRAPH_EDGES + (
    ("loadImage", "image", "wanGenerate", "reference_images"),
)
_I2V_GRAPH_BINDINGS = tuple(
    (
        role,
        param,
        "dualQuantizedComponents"
        if role == "diffusersQuantization" and param == "components"
        else "dualTransformer"
        if role == "diffusersRecipe" and param == "attention_components"
        else "true"
        if role == "diffusersRecipe" and param == "vae_tiling"
        else source,
    )
    for role, param, source in _VIDEO_GRAPH_BINDINGS
    if not (role == "wanGenerate" and param == "scheduler_flow_shift")
) + (
    ("loadImage", "file", "referenceImages"),
    ("loadImage", "alpha_channel", "alphaMode"),
)
_V2V_GRAPH_ROLES = _VIDEO_GRAPH_ROLES + (
    ("loadVideo", "modules.Video.Load", -520, 260),
    ("normalizeVideo", "modules.VideoConditioning.Normalize", -160, 260),
)
_V2V_GRAPH_EDGES = _VIDEO_GRAPH_EDGES + (
    ("loadVideo", "video", "normalizeVideo", "video"),
    ("normalizeVideo", "output", "wanGenerate", "video"),
)
_V2V_GRAPH_BINDINGS = _VIDEO_GRAPH_BINDINGS + (
    ("loadVideo", "file", "sourceVideo"),
    ("normalizeVideo", "width", "width"),
    ("normalizeVideo", "height", "height"),
    ("normalizeVideo", "num_frames", "numFrames"),
)
_VACE_INPAINT_GRAPH_ROLES = _V2V_GRAPH_ROLES + (
    ("loadMaskVideo", "modules.Video.Load", -520, 520),
    ("alignMaskVideo", "modules.VideoConditioning.AlignMask", -160, 520),
)
_VACE_INPAINT_GRAPH_EDGES = _V2V_GRAPH_EDGES + (
    ("normalizeVideo", "output", "alignMaskVideo", "video"),
    ("loadMaskVideo", "video", "alignMaskVideo", "mask"),
    ("alignMaskVideo", "output", "wanGenerate", "mask"),
)
_VACE_INPAINT_GRAPH_BINDINGS = tuple(
    (role, param, "wanVaceRevision") if role == "wanPipeline" and param == "revision" else (role, param, source)
    for role, param, source in _V2V_GRAPH_BINDINGS
) + (
    ("loadMaskVideo", "file", "maskVideo"),
    ("alignMaskVideo", "threshold", "maskThreshold127"),
    ("alignMaskVideo", "grow_pixels", "inpaintMaskGrow96"),
)
_VACE_OUTPAINT_GRAPH_BINDINGS = tuple(
    (role, param, "outpaintMaskGrow0") if role == "alignMaskVideo" and param == "grow_pixels" else (role, param, source)
    for role, param, source in _VACE_INPAINT_GRAPH_BINDINGS
)
_VACE_CONTROL_GRAPH_ROLES = _VIDEO_GRAPH_ROLES + (
    ("loadControlVideo", "modules.Video.Load", -520, 260),
    ("normalizeVideo", "modules.VideoConditioning.Normalize", -160, 260),
)
_VACE_CONTROL_GRAPH_EDGES = _VIDEO_GRAPH_EDGES + (
    ("loadControlVideo", "video", "normalizeVideo", "video"),
    ("normalizeVideo", "output", "wanGenerate", "video"),
)
_VACE_CONTROL_GRAPH_BINDINGS = _WAN_VACE_GRAPH_BINDINGS + (
    ("loadControlVideo", "file", "controlVideo"),
    ("normalizeVideo", "width", "width"),
    ("normalizeVideo", "height", "height"),
    ("normalizeVideo", "num_frames", "numFrames"),
)
_LTX_T2V_GRAPH_BINDINGS = tuple(
    (
        role,
        param,
        "nativeMath"
        if role == "diffusersRecipe" and param == "attention_backend"
        else "empty"
        if role == "diffusersRecipe" and param == "attention_components"
        else source,
    )
    for role, param, source in _VIDEO_GRAPH_BINDINGS
    if not (role == "wanGenerate" and param == "scheduler_flow_shift")
)
_LTX_I2V_GRAPH_BINDINGS = _LTX_T2V_GRAPH_BINDINGS + (
    ("loadImage", "file", "referenceImages"),
    ("loadImage", "alpha_channel", "alphaMode"),
)
_LTX_V2V_GRAPH_BINDINGS = tuple(
    (
        role,
        param,
        "conditioningScale" if role == "wanGenerate" and param == "strength" else source,
    )
    for role, param, source in _LTX_T2V_GRAPH_BINDINGS
) + (
    ("loadVideo", "file", "sourceVideo"),
    ("normalizeVideo", "width", "width"),
    ("normalizeVideo", "height", "height"),
    ("normalizeVideo", "num_frames", "numFrames"),
)
_VIDEO_REVISION_GRAPH_BINDINGS = tuple(
    (role, param, "defaultRevision") if role == "wanPipeline" and param == "revision" else (role, param, source)
    for role, param, source in _VIDEO_GRAPH_BINDINGS
)
_I2V_REVISION_GRAPH_BINDINGS = tuple(
    (role, param, "defaultRevision") if role == "wanPipeline" and param == "revision" else (role, param, source)
    for role, param, source in _I2V_GRAPH_BINDINGS
)
_STABLE_VIDEO_DIFFUSION_GRAPH_BINDINGS = tuple(
    (
        role,
        param,
        "empty"
        if (role, param) in {
            ("diffusersQuantization", "components"),
            ("diffusersRecipe", "attention_components"),
        }
        else "nativeMath"
        if (role, param) == ("diffusersRecipe", "attention_backend")
        else "false"
        if (role, param) in {
            ("diffusersRecipe", "regional_compile"),
            ("diffusersRecipe", "denoiser_cache"),
            ("diffusersRecipe", "layerwise_casting"),
            ("diffusersRecipe", "channels_last"),
        }
        else "empty"
        if role == "wanGenerate" and param in {"prompt", "negative_prompt"}
        else source,
    )
    for role, param, source in _I2V_REVISION_GRAPH_BINDINGS
)
_ANIMATEDIFF_GRAPH_BINDINGS = tuple(
    (
        role,
        param,
        "empty"
        if (role, param)
        in {
            ("diffusersQuantization", "components"),
            ("diffusersRecipe", "attention_components"),
        }
        else "nativeMath"
        if (role, param) == ("diffusersRecipe", "attention_backend")
        else "false"
        if (role, param)
        in {
            ("diffusersRecipe", "vae_tiling"),
            ("diffusersRecipe", "regional_compile"),
            ("diffusersRecipe", "denoiser_cache"),
            ("diffusersRecipe", "layerwise_casting"),
            ("diffusersRecipe", "channels_last"),
        }
        else source,
    )
    for role, param, source in _VIDEO_REVISION_GRAPH_BINDINGS
    if not (role == "wanGenerate" and param == "scheduler_flow_shift")
) + (
    ("wanPipeline", "motion_adapter_id", "repo"),
    ("wanPipeline", "motion_adapter_revision", "revision"),
)
_COGVIDEOX_GRAPH_BINDINGS = tuple(
    (
        role,
        param,
        "empty"
        if (role, param)
        in {
            ("diffusersQuantization", "components"),
            ("diffusersRecipe", "attention_components"),
        }
        else "nativeMath"
        if (role, param) == ("diffusersRecipe", "attention_backend")
        else "false"
        if (role, param)
        in {
            ("diffusersRecipe", "vae_tiling"),
            ("diffusersRecipe", "regional_compile"),
            ("diffusersRecipe", "denoiser_cache"),
            ("diffusersRecipe", "layerwise_casting"),
            ("diffusersRecipe", "channels_last"),
        }
        else source,
    )
    for role, param, source in _VIDEO_REVISION_GRAPH_BINDINGS
    if not (role == "wanGenerate" and param == "scheduler_flow_shift")
)
_WAN_ANIMATE_GRAPH_ROLES = _VIDEO_GRAPH_ROLES + (
    ("loadImage", "modules.Image.Load", -520, 300),
    ("loadPoseVideo", "modules.Video.Load", -520, 520),
    ("loadFaceVideo", "modules.Video.Load", -160, 520),
)
_WAN_ANIMATE_GRAPH_EDGES = _VIDEO_GRAPH_EDGES + (
    ("loadImage", "image", "wanGenerate", "reference_images"),
    ("loadPoseVideo", "video", "wanGenerate", "pose_video"),
    ("loadFaceVideo", "video", "wanGenerate", "face_video"),
)
_WAN_ANIMATE_GRAPH_BINDINGS = _VIDEO_REVISION_GRAPH_BINDINGS + (
    ("loadImage", "file", "referenceImages"),
    ("loadImage", "alpha_channel", "alphaMode"),
    ("loadPoseVideo", "file", "poseVideo"),
    ("loadFaceVideo", "file", "faceVideo"),
    ("wanGenerate", "segment_frame_length", "segmentFrameLength77"),
    ("wanGenerate", "previous_conditioning_frames", "previousConditioningFrames1"),
    ("wanGenerate", "motion_encode_batch_size", "motionEncodeBatchSize1"),
)
_WAN_REPLACE_GRAPH_ROLES = _WAN_ANIMATE_GRAPH_ROLES + (
    ("loadBackgroundVideo", "modules.Video.Load", 220, 520),
    ("loadMaskVideo", "modules.Video.Load", 600, 520),
)
_WAN_REPLACE_GRAPH_EDGES = _WAN_ANIMATE_GRAPH_EDGES + (
    ("loadBackgroundVideo", "video", "wanGenerate", "background_video"),
    ("loadMaskVideo", "video", "wanGenerate", "mask"),
)
_WAN_REPLACE_GRAPH_BINDINGS = _WAN_ANIMATE_GRAPH_BINDINGS + (
    ("loadBackgroundVideo", "file", "backgroundVideo"),
    ("loadMaskVideo", "file", "maskVideo"),
)
_LTX_LONG_GRAPH_BINDINGS = tuple(
    (role, param, "defaultRevision") if role == "wanPipeline" and param == "revision" else (role, param, source)
    for role, param, source in _LTX_I2V_GRAPH_BINDINGS
) + (
    ("wanGenerate", "temporal_tile_size", "temporalTileSize80"),
    ("wanGenerate", "temporal_overlap", "temporalOverlap24"),
    ("wanGenerate", "temporal_overlap_condition_strength", "temporalOverlapConditionStrength05"),
    ("wanGenerate", "adain_factor", "adainFactor025"),
    ("wanGenerate", "prompt_segments_json", "empty"),
)
_LTX2_GRAPH_ROLES = tuple(
    (role, "modules.DiffusersVideo.GenerateVideoAudio" if role == "wanGenerate" else node_key, x, y)
    for role, node_key, x, y in _VIDEO_GRAPH_ROLES
    if role != "videoExport"
) + (("videoExport", "modules.Video.ExportWithAudio", 640, -80),)
_LTX2_GRAPH_EDGES = tuple(edge for edge in _VIDEO_GRAPH_EDGES if edge[2] != "videoExport") + (
    ("wanGenerate", "video_out", "videoExport", "video"),
    ("wanGenerate", "audio", "videoExport", "audio"),
)
_LTX2_GRAPH_BINDINGS = tuple(
    (role, param, "defaultRevision") if role == "wanPipeline" and param == "revision" else (role, param, source)
    for role, param, source in _LTX_T2V_GRAPH_BINDINGS
)
_LTX2_I2V_GRAPH_ROLES = _LTX2_GRAPH_ROLES + (("loadImage", "modules.Image.Load", -520, 300),)
_LTX2_I2V_GRAPH_EDGES = _LTX2_GRAPH_EDGES + (("loadImage", "image", "wanGenerate", "reference_images"),)
_LTX2_I2V_GRAPH_BINDINGS = _LTX2_GRAPH_BINDINGS + (
    ("loadImage", "file", "referenceImages"),
    ("loadImage", "alpha_channel", "alphaMode"),
)
_LTX2_V2V_GRAPH_ROLES = _LTX2_GRAPH_ROLES + (
    ("loadVideo", "modules.Video.Load", -520, 260),
    ("normalizeVideo", "modules.VideoConditioning.Normalize", -160, 260),
)
_LTX2_V2V_GRAPH_EDGES = _LTX2_GRAPH_EDGES + (
    ("loadVideo", "video", "normalizeVideo", "video"),
    ("normalizeVideo", "output", "wanGenerate", "video"),
)
_LTX2_V2V_GRAPH_BINDINGS = _LTX2_GRAPH_BINDINGS + (
    ("loadVideo", "file", "sourceVideo"),
    ("normalizeVideo", "width", "width"),
    ("normalizeVideo", "height", "height"),
    ("normalizeVideo", "num_frames", "numFrames"),
)
_FRAMEPACK_GRAPH_ROLES = _I2V_GRAPH_ROLES
_FRAMEPACK_GRAPH_EDGES = _I2V_GRAPH_EDGES
_FRAMEPACK_GRAPH_BINDINGS = _I2V_REVISION_GRAPH_BINDINGS + (
    ("wanGenerate", "framepack_sampling", "framepackSampling"),
    ("wanGenerate", "latent_window_size", "latentWindowSize9"),
    ("wanGenerate", "true_cfg_scale", "trueCfgScale1"),
)
_WAN_FLF_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -1080, -80),
    ("prompt", "modules.ModularDiffusers.EncodePrompt", -700, -240),
    ("loadImage", "modules.Image.Load", -1080, 300),
    ("loadLastImage", "modules.Image.Load", -1080, 560),
    ("imageEmbeddings", "modules.ModularDiffusers.ImageEmbeddings", -700, 300),
    ("imageEncode", "modules.ModularDiffusers.ImageEncode", -300, 300),
    ("denoise", "modules.ModularDiffusers.Denoise", 100, -80),
    ("decode", "modules.ModularDiffusers.DecodeLatents", 500, -80),
    ("videoExport", "modules.Video.Export", 900, -80),
)
_WAN_FLF_GRAPH_EDGES = (
    ("models", "text_encoders", "prompt", "text_encoders"),
    ("models", "image_encoder", "imageEmbeddings", "image_encoder"),
    ("models", "vae_out", "imageEncode", "vae"),
    ("models", "unet_out", "denoise", "unet"),
    ("models", "scheduler", "denoise", "scheduler"),
    ("models", "vae_out", "denoise", "vae"),
    ("models", "vae_out", "decode", "vae"),
    ("loadImage", "image", "imageEmbeddings", "image"),
    ("loadImage", "image", "imageEncode", "image"),
    ("loadLastImage", "image", "imageEmbeddings", "last_image"),
    ("loadLastImage", "image", "imageEncode", "last_image"),
    ("prompt", "embeddings", "denoise", "embeddings"),
    ("imageEmbeddings", "image_embeds", "denoise", "image_embeds"),
    ("imageEmbeddings", "route_state_out", "imageEncode", "route_state_in"),
    ("imageEncode", "image_condition_latents", "denoise", "image_condition_latents"),
    ("imageEncode", "route_state_out", "denoise", "route_state_in"),
    ("denoise", "latents", "decode", "latents"),
    ("denoise", "route_state_out", "decode", "route_state_in"),
    ("decode", "videos", "videoExport", "video"),
)
_WAN_FLF_GRAPH_BINDINGS = (
    ("models", "model_type", "pipelineClass"),
    ("models", "repo_id", "artifact"),
    ("models", "revision", "defaultRevision"),
    ("models", "dtype", "dtype"),
    ("models", "device", "device"),
    ("models", "auto_offload", "autoOffload"),
    ("models", "offload_mode", "offloadMode"),
    ("models", "trust_remote_code", "false"),
    ("loadImage", "file", "referenceImages"),
    ("loadImage", "alpha_channel", "alphaMode"),
    ("loadLastImage", "file", "lastImage"),
    ("loadLastImage", "alpha_channel", "alphaMode"),
    ("prompt", "prompt", "prompt"),
    ("prompt", "negative_prompt", "negativePrompt"),
    ("imageEmbeddings", "width", "width"),
    ("imageEmbeddings", "height", "height"),
    ("imageEncode", "width", "width"),
    ("imageEncode", "height", "height"),
    ("imageEncode", "num_frames", "numFrames"),
    ("imageEncode", "seed", "seed"),
    ("denoise", "width", "width"),
    ("denoise", "height", "height"),
    ("denoise", "num_frames", "numFrames"),
    ("denoise", "seed", "seed"),
    ("denoise", "num_inference_steps", "steps"),
    ("denoise", "guidance_scale", "guidanceScale"),
    ("decode", "output_type", "outputType"),
    ("videoExport", "fps", "fps"),
)
_AUDIO_GRAPH_ROLES = (
    ("diffusersQuantization", "modules.DiffusersRuntime.PipelineQuantizationConfigV2", -1280, -80),
    ("diffusersRecipe", "modules.DiffusersRuntime.DiffusersExecutionRecipe", -900, -80),
    ("audioPipeline", "modules.DiffusersAudio.LoadPipeline", -520, -80),
    ("audioGenerate", "modules.DiffusersAudio.Generate", -120, -80),
    ("audioExport", "modules.Audio.Export", 1060, -80),
)
_AUDIO_GRAPH_EDGES = (
    ("diffusersQuantization", "quantization_config", "diffusersRecipe", "quantization_config"),
    ("diffusersRecipe", "execution_recipe", "audioPipeline", "execution_recipe"),
    ("audioPipeline", "pipeline", "audioGenerate", "pipeline"),
    ("audioGenerate", "audio", "audioExport", "audio"),
)
_AUDIO_GRAPH_BINDINGS = (
    ("diffusersQuantization", "backend", "quantizationMode"),
    ("diffusersQuantization", "components", "quantizedComponents"),
    ("diffusersQuantization", "dtype", "dtype"),
    ("diffusersRecipe", "device_map", "deviceMapNone"),
    ("diffusersRecipe", "offload_mode", "offloadMode"),
    ("diffusersRecipe", "device", "device"),
    ("diffusersRecipe", "attention_backend", "attentionBackend"),
    ("diffusersRecipe", "attention_components", "empty"),
    ("diffusersRecipe", "vae_slicing", "true"),
    ("diffusersRecipe", "vae_tiling", "true"),
    ("diffusersRecipe", "regional_compile", "regionalCompile"),
    ("diffusersRecipe", "denoiser_cache", "denoiserCache"),
    ("diffusersRecipe", "layerwise_casting", "layerwiseCasting"),
    ("diffusersRecipe", "channels_last", "channelsLast"),
    ("audioPipeline", "model_id", "artifact"),
    ("audioPipeline", "pipeline_class", "pipelineClass"),
    ("audioPipeline", "mode", "mode"),
    ("audioPipeline", "dtype", "dtype"),
    ("audioPipeline", "device", "device"),
    ("audioPipeline", "auto_offload", "autoOffload"),
    ("audioPipeline", "offload_mode", "offloadMode"),
    ("audioGenerate", "task_type", "text2music"),
    ("audioGenerate", "prompt", "prompt"),
    ("audioGenerate", "negative_prompt", "negativePrompt"),
    ("audioGenerate", "lyrics", "lyrics"),
    ("audioGenerate", "audio_duration", "audioDuration"),
    ("audioGenerate", "extension_duration", "extensionDuration"),
    ("audioGenerate", "vocal_language", "vocalLanguage"),
    ("audioGenerate", "seed", "seed"),
    ("audioGenerate", "num_inference_steps", "steps"),
    ("audioGenerate", "guidance_scale", "guidanceScale"),
    ("audioGenerate", "shift", "shift"),
    ("audioGenerate", "bpm", "bpmNormalized"),
    ("audioGenerate", "keyscale", "keyscale"),
    ("audioGenerate", "timesignature", "timesignature"),
    ("audioGenerate", "repainting_start", "repaintingStart"),
    ("audioGenerate", "repainting_end", "repaintingEnd"),
    ("audioGenerate", "audio_cover_strength", "audioCoverStrength"),
    ("audioGenerate", "return_continuation_tail", "false"),
    ("audioGenerate", "sample_rate", "sampleRate48000"),
    ("audioExport", "sample_rate", "sampleRate48000"),
)
_STABLE_AUDIO_GRAPH_BINDINGS = (
    ("diffusersQuantization", "backend", "quantizationMode"),
    ("diffusersQuantization", "components", "quantizedComponents"),
    ("diffusersQuantization", "dtype", "dtype"),
    ("diffusersRecipe", "device_map", "deviceMapNone"),
    ("diffusersRecipe", "offload_mode", "offloadMode"),
    ("diffusersRecipe", "device", "device"),
    ("diffusersRecipe", "attention_backend", "attentionBackend"),
    ("diffusersRecipe", "attention_components", "empty"),
    ("diffusersRecipe", "vae_slicing", "true"),
    ("diffusersRecipe", "vae_tiling", "true"),
    ("diffusersRecipe", "regional_compile", "regionalCompile"),
    ("diffusersRecipe", "denoiser_cache", "denoiserCache"),
    ("diffusersRecipe", "layerwise_casting", "layerwiseCasting"),
    ("diffusersRecipe", "channels_last", "channelsLast"),
    ("audioPipeline", "model_id", "artifact"),
    ("audioPipeline", "pipeline_class", "pipelineClass"),
    ("audioPipeline", "mode", "mode"),
    ("audioPipeline", "revision", "defaultRevision"),
    ("audioPipeline", "dtype", "dtype"),
    ("audioPipeline", "device", "device"),
    ("audioPipeline", "auto_offload", "autoOffload"),
    ("audioPipeline", "offload_mode", "offloadMode"),
    ("audioGenerate", "task_type", "text2audio"),
    ("audioGenerate", "prompt", "prompt"),
    ("audioGenerate", "negative_prompt", "negativePrompt"),
    ("audioGenerate", "audio_duration", "audioDuration"),
    ("audioGenerate", "seed", "seed"),
    ("audioGenerate", "stable_audio_steps", "steps"),
    ("audioGenerate", "stable_audio_guidance", "guidanceScale"),
    ("audioGenerate", "num_waveforms", "numWaveforms1"),
    ("audioGenerate", "sample_rate", "sampleRate48000"),
    ("audioExport", "sample_rate", "sampleRate48000"),
)
_LONGCAT_AUDIO_DIT_GRAPH_BINDINGS = tuple(
    (role, param, "sampleRate24000") if param == "sample_rate" else (role, param, source)
    for role, param, source in _STABLE_AUDIO_GRAPH_BINDINGS
    if param != "num_waveforms"
)
_AUDIO_LDM2_GRAPH_BINDINGS = tuple(
    (
        role,
        param,
        "sampleRate16000"
        if param == "sample_rate"
        else "numWaveforms3"
        if param == "num_waveforms"
        else source,
    )
    for role, param, source in _STABLE_AUDIO_GRAPH_BINDINGS
)
_AUDIO_VARIATION_GRAPH_ROLES = (
    ("loadAudio", "modules.Audio.Load", -520, 300),
    *_AUDIO_GRAPH_ROLES,
)
_AUDIO_VARIATION_GRAPH_EDGES = (
    ("diffusersQuantization", "quantization_config", "diffusersRecipe", "quantization_config"),
    ("diffusersRecipe", "execution_recipe", "audioPipeline", "execution_recipe"),
    ("audioPipeline", "pipeline", "audioGenerate", "pipeline"),
    ("loadAudio", "audio", "audioGenerate", "source_audio"),
    ("audioGenerate", "audio", "audioExport", "audio"),
)
_AUDIO_VARIATION_GRAPH_BINDINGS = (
    ("loadAudio", "file", "sourceAudio"),
    *(
        (role, param, "cover") if role == "audioGenerate" and param == "task_type" else (role, param, source)
        for role, param, source in _AUDIO_GRAPH_BINDINGS
    ),
)
_AUDIO_CONTINUATION_GRAPH_ROLES = (
    *_AUDIO_VARIATION_GRAPH_ROLES,
    ("audioLoudnessMatch", "modules.Audio.MatchLoudness", 300, -80),
    ("audioJoin", "modules.Audio.Join", 680, -80),
)
_AUDIO_CONTINUATION_GRAPH_EDGES = (
    ("diffusersQuantization", "quantization_config", "diffusersRecipe", "quantization_config"),
    ("diffusersRecipe", "execution_recipe", "audioPipeline", "execution_recipe"),
    ("audioPipeline", "pipeline", "audioGenerate", "pipeline"),
    ("loadAudio", "audio", "audioGenerate", "source_audio"),
    ("audioGenerate", "audio", "audioLoudnessMatch", "audio"),
    ("loadAudio", "audio", "audioLoudnessMatch", "reference"),
    ("audioLoudnessMatch", "output", "audioJoin", "continuation"),
    ("loadAudio", "audio", "audioJoin", "source"),
    ("audioJoin", "output", "audioExport", "audio"),
)
_AUDIO_CONTINUATION_GRAPH_BINDINGS = (
    ("loadAudio", "file", "sourceAudio"),
    *(
        (
            role,
            param,
            "continuation"
            if role == "audioGenerate" and param == "task_type"
            else "true"
            if role == "audioGenerate" and param == "return_continuation_tail"
            else source,
        )
        for role, param, source in _AUDIO_GRAPH_BINDINGS
    ),
    ("audioLoudnessMatch", "reference_window_seconds", "referenceWindow15"),
    ("audioLoudnessMatch", "target_peak_dbfs", "targetPeakMinus1"),
    ("audioLoudnessMatch", "max_adjustment_db", "maxAdjustment12"),
    ("audioJoin", "boundary_fade_seconds", "boundaryFade001"),
)
_AUDIO_REPAINT_GRAPH_ROLES = _AUDIO_VARIATION_GRAPH_ROLES
_AUDIO_REPAINT_GRAPH_EDGES = _AUDIO_VARIATION_GRAPH_EDGES
_AUDIO_REPAINT_GRAPH_BINDINGS = (
    ("loadAudio", "file", "sourceAudio"),
    *(
        (role, param, "repaint") if role == "audioGenerate" and param == "task_type" else (role, param, source)
        for role, param, source in _AUDIO_GRAPH_BINDINGS
    ),
)
_THREE_D_GRAPH_ROLES = (
    ("diffusersQuantization", "modules.DiffusersRuntime.PipelineQuantizationConfigV2", -1280, -80),
    ("diffusersRecipe", "modules.DiffusersRuntime.DiffusersExecutionRecipe", -900, -80),
    ("diffusersThreeDPipeline", "modules.DiffusersThreeD.LoadPipeline", -520, -80),
    ("diffusersThreeDGenerate", "modules.DiffusersThreeD.GenerateRenderedArtifact", -120, -80),
    ("videoExport", "modules.Video.Export", 420, -80),
)
_THREE_D_GRAPH_EDGES = (
    ("diffusersQuantization", "quantization_config", "diffusersRecipe", "quantization_config"),
    ("diffusersRecipe", "execution_recipe", "diffusersThreeDPipeline", "execution_recipe"),
    ("diffusersThreeDPipeline", "pipeline", "diffusersThreeDGenerate", "pipeline"),
    ("diffusersThreeDGenerate", "video", "videoExport", "video"),
)
_THREE_D_GRAPH_BINDINGS = (
    ("diffusersQuantization", "backend", "quantizationMode"),
    ("diffusersQuantization", "components", "empty"),
    ("diffusersQuantization", "dtype", "dtype"),
    ("diffusersRecipe", "device_map", "deviceMapNone"),
    ("diffusersRecipe", "offload_mode", "offloadMode"),
    ("diffusersRecipe", "device", "device"),
    ("diffusersRecipe", "attention_backend", "attentionBackend"),
    ("diffusersRecipe", "attention_components", "empty"),
    ("diffusersRecipe", "vae_slicing", "false"),
    ("diffusersRecipe", "vae_tiling", "false"),
    ("diffusersRecipe", "regional_compile", "false"),
    ("diffusersRecipe", "denoiser_cache", "false"),
    ("diffusersRecipe", "layerwise_casting", "false"),
    ("diffusersRecipe", "channels_last", "false"),
    ("diffusersThreeDPipeline", "model_id", "artifact"),
    ("diffusersThreeDPipeline", "pipeline_class", "pipelineClass"),
    ("diffusersThreeDPipeline", "mode", "mode"),
    ("diffusersThreeDPipeline", "revision", "defaultRevision"),
    ("diffusersThreeDPipeline", "dtype", "dtype"),
    ("diffusersThreeDPipeline", "device", "device"),
    ("diffusersThreeDPipeline", "auto_offload", "autoOffload"),
    ("diffusersThreeDPipeline", "offload_mode", "offloadMode"),
    ("diffusersThreeDGenerate", "prompt", "prompt"),
    ("diffusersThreeDGenerate", "seed", "seed"),
    ("diffusersThreeDGenerate", "num_inference_steps", "steps"),
    ("diffusersThreeDGenerate", "guidance_scale", "guidanceScale"),
    ("diffusersThreeDGenerate", "frame_size", "width"),
    ("videoExport", "fps", "fps"),
)
_UNCONDITIONAL_GRAPH_ROLES = (
    ("diffusersQuantization", "modules.DiffusersRuntime.PipelineQuantizationConfigV2", -1080, -80),
    ("diffusersRecipe", "modules.DiffusersRuntime.DiffusersExecutionRecipe", -720, -80),
    ("diffusersImagePipeline", "modules.DiffusersImage.LoadPipeline", -360, -80),
    ("diffusersUnconditionalGenerate", "modules.DiffusersImage.UnconditionalGenerate", 80, -80),
    ("preview", "modules.Image.Preview", 520, -80),
)
_UNCONDITIONAL_GRAPH_EDGES = (
    ("diffusersQuantization", "quantization_config", "diffusersRecipe", "quantization_config"),
    ("diffusersRecipe", "execution_recipe", "diffusersImagePipeline", "execution_recipe"),
    ("diffusersImagePipeline", "pipeline", "diffusersUnconditionalGenerate", "pipeline"),
    ("diffusersUnconditionalGenerate", "images", "preview", "image"),
)
_UNCONDITIONAL_GRAPH_BINDINGS = (
    ("diffusersQuantization", "backend", "quantizationMode"),
    ("diffusersQuantization", "components", "quantizedComponents"),
    ("diffusersQuantization", "dtype", "dtype"),
    ("diffusersRecipe", "device_map", "deviceMapNone"),
    ("diffusersRecipe", "offload_mode", "offloadMode"),
    ("diffusersRecipe", "device", "device"),
    ("diffusersRecipe", "attention_backend", "nativeMath"),
    ("diffusersRecipe", "attention_components", "empty"),
    ("diffusersRecipe", "vae_slicing", "false"),
    ("diffusersRecipe", "vae_tiling", "false"),
    ("diffusersRecipe", "regional_compile", "false"),
    ("diffusersRecipe", "denoiser_cache", "false"),
    ("diffusersRecipe", "layerwise_casting", "false"),
    ("diffusersRecipe", "channels_last", "false"),
    ("diffusersImagePipeline", "model_id", "artifact"),
    ("diffusersImagePipeline", "pipeline_class", "pipelineClass"),
    ("diffusersImagePipeline", "mode", "mode"),
    ("diffusersImagePipeline", "revision", "defaultRevision"),
    ("diffusersImagePipeline", "dtype", "dtype"),
    ("diffusersImagePipeline", "device", "device"),
    ("diffusersImagePipeline", "quantization_mode", "quantizationMode"),
    ("diffusersImagePipeline", "quantized_components", "empty"),
    ("diffusersImagePipeline", "auto_offload", "autoOffload"),
    ("diffusersImagePipeline", "offload_mode", "offloadMode"),
    ("diffusersUnconditionalGenerate", "batch_size", "batchSize"),
    ("diffusersUnconditionalGenerate", "seed", "seed"),
    ("diffusersUnconditionalGenerate", "num_inference_steps", "steps"),
    ("diffusersUnconditionalGenerate", "eta", "eta"),
    ("diffusersUnconditionalGenerate", "class_label", "classLabel"),
    ("diffusersUnconditionalGenerate", "output_type", "outputType"),
)
_AUTO_FIELDS = (
    "resolvedArtifact",
    "artifact",
    "installTarget.repo",
    "modelRepo",
    "pipelineClass",
    "dtype",
    "offloadMode",
    "quantizedComponents",
    "attentionBackend",
    "regionalCompile",
    "denoiserCache",
    "layerwiseCasting",
    "channelsLast",
)
_BINDING_SOURCES = frozenset(
    item[2]
    for item in (
        *_GRAPH_BINDINGS,
        *_SDXL_GRAPH_BINDINGS,
        *_PAG_GRAPH_BINDINGS,
        *_PERCEPTION_GRAPH_BINDINGS,
        *_SPEECH_TRANSCRIPTION_GRAPH_BINDINGS,
        *_SPEECH_TRANSLATION_GRAPH_BINDINGS,
        *_SDXL_EDIT_GRAPH_BINDINGS,
        *_MODULAR_EDIT_GRAPH_BINDINGS,
        *_MODULAR_LAYERED_GRAPH_BINDINGS,
        *_MODULAR_CONTROL_GRAPH_BINDINGS,
        *_CONTROL_GRAPH_BINDINGS,
        *_CONDITIONED_CONTROL_GRAPH_BINDINGS,
        *_EDIT_GRAPH_BINDINGS,
        *_INPAINT_GRAPH_BINDINGS,
        *_QWEN_OUTPAINT_GRAPH_BINDINGS,
        *_VIDEO_GRAPH_BINDINGS,
        *_WAN_VACE_GRAPH_BINDINGS,
        *_I2V_GRAPH_BINDINGS,
        *_V2V_GRAPH_BINDINGS,
        *_VACE_INPAINT_GRAPH_BINDINGS,
        *_VACE_OUTPAINT_GRAPH_BINDINGS,
        *_VACE_CONTROL_GRAPH_BINDINGS,
        *_LTX_T2V_GRAPH_BINDINGS,
        *_LTX_I2V_GRAPH_BINDINGS,
        *_LTX_V2V_GRAPH_BINDINGS,
        *_WAN_ANIMATE_GRAPH_BINDINGS,
        *_WAN_REPLACE_GRAPH_BINDINGS,
        *_LTX_LONG_GRAPH_BINDINGS,
        *_LTX2_GRAPH_BINDINGS,
        *_LTX2_I2V_GRAPH_BINDINGS,
        *_LTX2_V2V_GRAPH_BINDINGS,
        *_FRAMEPACK_GRAPH_BINDINGS,
        *_WAN_FLF_GRAPH_BINDINGS,
        *_STABLE_AUDIO_GRAPH_BINDINGS,
        *_LONGCAT_AUDIO_DIT_GRAPH_BINDINGS,
        *_AUDIO_LDM2_GRAPH_BINDINGS,
        *_AUDIO_GRAPH_BINDINGS,
        *_AUDIO_VARIATION_GRAPH_BINDINGS,
        *_AUDIO_CONTINUATION_GRAPH_BINDINGS,
        *_AUDIO_REPAINT_GRAPH_BINDINGS,
        *_THREE_D_GRAPH_BINDINGS,
        *_UNCONDITIONAL_GRAPH_BINDINGS,
    )
)

_MODULAR_EDIT_PLUS_PROFILE = {
    "id": "qwen-edit-plus:modular",
    "model_type": "QwenImageEditPlusModularPipeline",
    "modes": ("edit_image", "multi_image_reference_edit"),
    "loader_module": "modules.ModularDiffusers",
    "loader_action": "ModelsLoader",
    "execution_path": "modular-diffusers",
    "pipeline_class": "QwenImageEditPlusModularPipeline",
    "default_repo": "Qwen/Qwen-Image-Edit-2511",
    "fallback_repo": None,
    "quantizable_components": ("transformer", "text_encoder"),
    "default_quantized_components": ("transformer", "text_encoder"),
    "supported_offload_modes": (
        OFFLOAD_MODE_NONE,
        OFFLOAD_MODE_MODEL_CPU,
        OFFLOAD_MODE_GROUP_CPU,
        OFFLOAD_MODE_GROUP_DISK,
    ),
    "retry_offload_modes": (OFFLOAD_MODE_GROUP_DISK,),
    "max_low_memory_side": 768,
    "max_low_memory_steps": 24,
    "live_proof": False,
    "compatible_repos": (),
}
_MODULAR_LAYERED_PROFILE = {
    "id": "qwen-layered:modular",
    "model_type": "QwenImageLayeredModularPipeline",
    "modes": ("layer_decomposition",),
    "loader_module": "modules.ModularDiffusers",
    "loader_action": "ModelsLoader",
    "execution_path": "modular-diffusers",
    "pipeline_class": "QwenImageLayeredModularPipeline",
    "default_repo": "Qwen/Qwen-Image-Layered",
    "fallback_repo": None,
    "quantizable_components": ("transformer", "text_encoder"),
    "default_quantized_components": ("transformer", "text_encoder"),
    "supported_offload_modes": (
        OFFLOAD_MODE_NONE,
        OFFLOAD_MODE_MODEL_CPU,
        OFFLOAD_MODE_GROUP_CPU,
        OFFLOAD_MODE_GROUP_DISK,
    ),
    "retry_offload_modes": (OFFLOAD_MODE_GROUP_DISK,),
    "max_low_memory_side": 768,
    "max_low_memory_steps": 30,
    "live_proof": False,
    "compatible_repos": (),
}
_MODULAR_CONTROL_PROFILE = {
    "id": "qwen-image:modular",
    "model_type": "QwenImageModularPipeline",
    "modes": ("control_image",),
    "loader_module": "modules.ModularDiffusers",
    "loader_action": "ModelsLoader",
    "execution_path": "modular-diffusers",
    "pipeline_class": "QwenImageModularPipeline",
    "default_repo": "Qwen/Qwen-Image-2512",
    "fallback_repo": None,
    "quantizable_components": ("transformer", "text_encoder"),
    "default_quantized_components": ("transformer", "text_encoder"),
    "supported_offload_modes": (
        OFFLOAD_MODE_NONE,
        OFFLOAD_MODE_MODEL_CPU,
        OFFLOAD_MODE_GROUP_CPU,
        OFFLOAD_MODE_GROUP_DISK,
    ),
    "retry_offload_modes": (OFFLOAD_MODE_GROUP_DISK,),
    "max_low_memory_side": 768,
    "max_low_memory_steps": 28,
    "live_proof": False,
    "compatible_repos": (),
}
_AUTO_FIELD_ALLOWLIST = frozenset(_AUTO_FIELDS)
_EXPERT_IMAGE_QUANTIZATION_MODES = (
    "bnb_4bit",
    "bnb_8bit",
    "quanto_float8",
    "torchao_float8",
)


def _profile(
    profile_id: str,
    model_type: str,
    repo: str,
    *,
    default_quantized_components: tuple[str, ...],
    supported_offload_modes: tuple[str, ...],
    retry_offload_modes: tuple[str, ...],
    max_low_memory_side: int,
    max_low_memory_steps: int,
    compatible_repos: tuple[str, ...] = (),
    mode: str = "text_to_image",
    pipeline_class: str = "FluxPipeline",
) -> dict[str, Any]:
    return {
        "id": profile_id,
        "model_type": model_type,
        "modes": (mode,),
        "loader_module": "modules.DiffusersImage",
        "loader_action": "LoadPipeline",
        "execution_path": "direct-diffusers-image",
        "pipeline_class": pipeline_class,
        "default_repo": repo,
        "fallback_repo": None,
        "quantizable_components": ("transformer", "text_encoder_2"),
        "default_quantized_components": default_quantized_components,
        "supported_offload_modes": supported_offload_modes,
        "retry_offload_modes": retry_offload_modes,
        "max_low_memory_side": max_low_memory_side,
        "max_low_memory_steps": max_low_memory_steps,
        "live_proof": False,
        "compatible_repos": compatible_repos,
    }


def _capability(
    model_type: str,
    label: str,
    display_name: str,
    repo: str,
    *,
    width: int,
    steps: int,
    guidance: float,
    low_vram_mode: str,
    alternate_artifact: str | None = None,
    low_vram_width: int | None = None,
    low_vram_steps: int | None = None,
    execution_status: str = "supported_with_model",
) -> dict[str, Any]:
    return {
        "modelType": model_type,
        "label": label,
        "displayName": display_name,
        "family": "FLUX Image",
        "defaultRepo": repo,
        **({"alternateArtifact": alternate_artifact} if alternate_artifact else {}),
        "artifactLabel": "Diffusers repo",
        "defaultDtype": "bfloat16",
        "defaultSize": {"width": width, "height": width, "aspectRatio": "1:1"},
        "recommendedSteps": steps,
        "recommendedGuidance": guidance,
        "guidanceLabel": "Guidance",
        "supportsImageInput": False,
        "supportsMask": False,
        "supportsMultiImage": False,
        "supportsControlImage": False,
        "supportsLayers": False,
        "supportsLora": True,
        "offloadSupport": {
            "default": OFFLOAD_MODE_MODEL_CPU,
            "lowVram": OFFLOAD_MODE_MODEL_CPU,
            "emergency": OFFLOAD_MODE_GROUP_DISK,
            "modes": list(_DIRECT_OFFLOAD_MODES),
        },
        "lowVram": {
            "dtype": "bfloat16",
            "autoOffload": True,
            "offloadMode": low_vram_mode,
            "steps": low_vram_steps if low_vram_steps is not None else steps,
            "width": low_vram_width if low_vram_width is not None else width,
            "height": low_vram_width if low_vram_width is not None else width,
        },
        "modes": ["text_to_image"],
        "executionStatus": execution_status,
    }


STUDIO_EXECUTION_SPEC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "flux-schnell:text-to-image:v1": {
        "modelType": "FluxSchnellPipeline",
        "mode": "text_to_image",
        "profile": _profile(
            "flux-schnell:direct",
            "FluxSchnellPipeline",
            FLUX_SCHNELL_REPO,
            default_quantized_components=(),
            supported_offload_modes=_DIRECT_OFFLOAD_MODES,
            retry_offload_modes=(
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            max_low_memory_side=1024,
            max_low_memory_steps=4,
        ),
        "capability": _capability(
            "FluxSchnellPipeline",
            "FLUX.1 schnell",
            "FLUX.1-schnell",
            FLUX_SCHNELL_REPO,
            width=1024,
            steps=4,
            guidance=0.0,
            low_vram_mode=OFFLOAD_MODE_MODEL_CPU,
        ),
        "autoRequirements": {
            "supportedTasks": ["text_to_image"],
            "defaultRepo": FLUX_SCHNELL_REPO,
            "qualityDefaults": {
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "guidanceScale": 0,
                "maxSequenceLength": 256,
            },
            "minimum": {
                "accelerator": "cuda",
                "vramBytes": 12 * _GIB,
                "systemRamBytes": 24 * _GIB,
                "diskFreeBytes": 25 * _GIB,
            },
            "recommended": {
                "accelerator": "cuda",
                "vramBytes": 16 * _GIB,
                "systemRamBytes": 32 * _GIB,
                "diskFreeBytes": 35 * _GIB,
            },
            "fullResidency": _HIGH_MEMORY_FULL_RESIDENCY,
            "supportedOffloadModes": [
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
                OFFLOAD_MODE_NONE,
            ],
            "requiredPackages": ["diffusers", "transformers", "accelerate", "torch"],
        },
    },
    "flux-dev:text-to-image:v1": {
        "modelType": "FluxDevPipeline",
        "mode": "text_to_image",
        "profile": _profile(
            "flux-dev:direct",
            "FluxDevPipeline",
            FLUX_DEV_REPO,
            default_quantized_components=("transformer",),
            supported_offload_modes=(
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            retry_offload_modes=(OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK),
            max_low_memory_side=768,
            max_low_memory_steps=20,
            compatible_repos=(FLUX_DEV_FP8_REPO,),
        ),
        "capability": {
            **_capability(
                "FluxDevPipeline",
                "FLUX.1 dev",
                "FLUX.1-dev",
                FLUX_DEV_REPO,
                width=768,
                steps=20,
                guidance=3.5,
                low_vram_mode=OFFLOAD_MODE_GROUP_DISK,
                alternate_artifact=FLUX_DEV_FP8_REPO,
            ),
            "notes": ["Auto prefers the FP8 artifact on 16 GB CUDA when available."],
            "revisionCandidates": [
                require_catalog_revision(FLUX_DEV_REPO, model_type="FluxDevPipeline")
            ],
            "supportsImageInput": True,
            "supportsMask": True,
            "modes": ["text_to_image", "edit_image", "inpaint"],
            "modeRequirements": {
                "edit_image": {
                    "requiredImages": ["referenceImages"],
                    "note": "Requires one source image for image-to-image transformation.",
                },
                "inpaint": {
                    "requiredImages": ["referenceImages", "maskImage"],
                    "note": "Requires one source image and one mask image for inpainting.",
                }
            },
        },
        "autoRequirements": {
            "supportedTasks": ["text_to_image"],
            "defaultRepo": FLUX_DEV_REPO,
            "preferredLowerMemoryRepo": FLUX_DEV_FP8_REPO,
            "qualityDefaults": {
                "width": 768,
                "height": 768,
                "steps": 20,
                "guidanceScale": 3.5,
                "maxSequenceLength": 256,
            },
            "minimum": {
                "accelerator": "cuda",
                "vramBytes": 24 * _GIB,
                "systemRamBytes": 48 * _GIB,
                "diskFreeBytes": 45 * _GIB,
            },
            "recommended": {
                "accelerator": "cuda",
                "vramBytes": 32 * _GIB,
                "systemRamBytes": 64 * _GIB,
                "diskFreeBytes": 60 * _GIB,
            },
            "fullResidency": _HIGH_MEMORY_FULL_RESIDENCY,
            "lowerMemory": {
                "accelerator": "cuda",
                "vramBytes": 16 * _GIB,
                "systemRamBytes": 32 * _GIB,
                "diskFreeBytes": 45 * _GIB,
                "quantizationMode": "quanto_float8",
                "quantizedComponents": ["transformer", "text_encoder_2"],
            },
            "supportedOffloadModes": [
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
                OFFLOAD_MODE_NONE,
            ],
            "requiredPackages": [
                "diffusers",
                "transformers",
                "accelerate",
                "torch",
                "optimum-quanto",
            ],
        },
    },
    "flux-krea:text-to-image:v1": {
        "modelType": "FluxKreaPipeline",
        "mode": "text_to_image",
        "profile": _profile(
            "flux-krea:direct",
            "FluxKreaPipeline",
            FLUX_KREA_REPO,
            default_quantized_components=("transformer",),
            supported_offload_modes=(
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            retry_offload_modes=(OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK),
            max_low_memory_side=768,
            max_low_memory_steps=24,
        ),
        "capability": _capability(
            "FluxKreaPipeline",
            "FLUX.1 Krea dev",
            "FLUX.1-Krea-dev",
            FLUX_KREA_REPO,
            width=1024,
            steps=28,
            guidance=3.5,
            low_vram_mode=OFFLOAD_MODE_GROUP_DISK,
            low_vram_width=768,
            low_vram_steps=20,
            execution_status="expert_only",
        ),
        "autoRequirements": {
            "supportedTasks": ["text_to_image"],
            "defaultRepo": FLUX_KREA_REPO,
            "qualityDefaults": {
                "width": 768,
                "height": 768,
                "steps": 24,
                "guidanceScale": 3.5,
                "maxSequenceLength": 256,
            },
            "minimum": {
                "accelerator": "cuda",
                "vramBytes": 24 * _GIB,
                "systemRamBytes": 48 * _GIB,
                "diskFreeBytes": 45 * _GIB,
            },
            "recommended": {
                "accelerator": "cuda",
                "vramBytes": 32 * _GIB,
                "systemRamBytes": 64 * _GIB,
                "diskFreeBytes": 60 * _GIB,
            },
            "fullResidency": _HIGH_MEMORY_FULL_RESIDENCY,
            "onLoadQuantization": {
                "accelerator": "cuda",
                "vramBytes": 16 * _GIB,
                "systemRamBytes": 32 * _GIB,
                "diskFreeBytes": 45 * _GIB,
                "quantizationMode": "quanto_float8",
                "quantizedComponents": ["transformer", "text_encoder_2"],
            },
            "supportedOffloadModes": [
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
                OFFLOAD_MODE_NONE,
            ],
            "requiredPackages": [
                "diffusers",
                "transformers",
                "accelerate",
                "torch",
                "optimum-quanto",
            ],
            "guardedReason": "FLUX Krea has broad guarded Auto coverage through on-load float8 quantization and Diffusers offload.",
        },
    },
    "flux-depth:control-image:v1": {
        "modelType": "FluxDepthPipeline",
        "mode": "control_image",
        "profile": _profile(
            "flux-depth:direct",
            "FluxDepthPipeline",
            FLUX_DEPTH_REPO,
            default_quantized_components=("transformer",),
            supported_offload_modes=(
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            retry_offload_modes=(OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK),
            max_low_memory_side=768,
            max_low_memory_steps=24,
            mode="control_image",
            pipeline_class="FluxControlPipeline",
        ),
        "capability": {
            **_capability(
                "FluxDepthPipeline",
                "FLUX.1 Depth dev",
                "FLUX.1-Depth-dev",
                FLUX_DEPTH_REPO,
                width=1024,
                steps=28,
                guidance=3.5,
                low_vram_mode=OFFLOAD_MODE_GROUP_DISK,
                low_vram_width=768,
                low_vram_steps=20,
                execution_status="expert_only",
            ),
            "supportsImageInput": True,
            "supportsControlImage": True,
            "modes": ["control_image"],
        },
        "autoRequirements": {
            "supportedTasks": ["control_image"],
            "defaultRepo": FLUX_DEPTH_REPO,
            "qualityDefaults": {
                "width": 768,
                "height": 768,
                "steps": 24,
                "guidanceScale": 10,
                "maxSequenceLength": 256,
            },
            "minimum": {
                "accelerator": "cuda",
                "vramBytes": 24 * _GIB,
                "systemRamBytes": 48 * _GIB,
                "diskFreeBytes": 45 * _GIB,
            },
            "recommended": {
                "accelerator": "cuda",
                "vramBytes": 32 * _GIB,
                "systemRamBytes": 64 * _GIB,
                "diskFreeBytes": 60 * _GIB,
            },
            "fullResidency": _HIGH_MEMORY_FULL_RESIDENCY,
            "onLoadQuantization": {
                "accelerator": "cuda",
                "vramBytes": 16 * _GIB,
                "systemRamBytes": 32 * _GIB,
                "diskFreeBytes": 45 * _GIB,
                "quantizationMode": "quanto_float8",
                "quantizedComponents": ["transformer", "text_encoder_2"],
            },
            "supportedOffloadModes": [
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
                OFFLOAD_MODE_NONE,
            ],
            "requiredPackages": [
                "diffusers",
                "transformers",
                "accelerate",
                "torch",
                "optimum-quanto",
            ],
            "guardedReason": "FLUX Depth has guarded Auto coverage through generic control-image Diffusers nodes.",
        },
        "roles": _CONTROL_GRAPH_ROLES,
        "edges": _CONTROL_GRAPH_EDGES,
        "bindings": _CONTROL_GRAPH_BINDINGS,
    },
    "flux-canny:control-image:v1": {
        "modelType": "FluxCannyPipeline",
        "mode": "control_image",
        "profile": _profile(
            "flux-canny:direct",
            "FluxCannyPipeline",
            FLUX_CANNY_REPO,
            default_quantized_components=("transformer",),
            supported_offload_modes=(
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            retry_offload_modes=(OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK),
            max_low_memory_side=768,
            max_low_memory_steps=24,
            compatible_repos=(FLUX_CANNY_VERIFIED_REPAIR_REPO,),
            mode="control_image",
            pipeline_class="FluxControlPipeline",
        ),
        "capability": {
            **_capability(
                "FluxCannyPipeline",
                "FLUX.1 Canny dev",
                "FLUX.1-Canny-dev",
                FLUX_CANNY_REPO,
                width=1024,
                steps=28,
                guidance=3.5,
                low_vram_mode=OFFLOAD_MODE_GROUP_DISK,
                low_vram_width=768,
                low_vram_steps=20,
                execution_status="expert_only",
            ),
            "artifactCandidates": [
                FLUX_CANNY_REPO,
                FLUX_CANNY_VERIFIED_REPAIR_REPO,
            ],
            "verifiedRepairSources": [
                {
                    "repo": FLUX_CANNY_VERIFIED_REPAIR_REPO,
                    "verification": "matching filename, size, and LFS SHA-256 plus local byte verification",
                }
            ],
            "supportsImageInput": True,
            "supportsControlImage": True,
            "modes": ["control_image"],
        },
        "autoRequirements": {
            "supportedTasks": ["control_image"],
            "defaultRepo": FLUX_CANNY_REPO,
            "qualityDefaults": {
                "width": 768,
                "height": 768,
                "steps": 24,
                "guidanceScale": 10,
                "maxSequenceLength": 256,
            },
            "minimum": {
                "accelerator": "cuda",
                "vramBytes": 24 * _GIB,
                "systemRamBytes": 48 * _GIB,
                "diskFreeBytes": 45 * _GIB,
            },
            "recommended": {
                "accelerator": "cuda",
                "vramBytes": 32 * _GIB,
                "systemRamBytes": 64 * _GIB,
                "diskFreeBytes": 60 * _GIB,
            },
            "fullResidency": _HIGH_MEMORY_FULL_RESIDENCY,
            "onLoadQuantization": {
                "accelerator": "cuda",
                "vramBytes": 16 * _GIB,
                "systemRamBytes": 32 * _GIB,
                "diskFreeBytes": 45 * _GIB,
                "quantizationMode": "quanto_float8",
                "quantizedComponents": ["transformer", "text_encoder_2"],
            },
            "supportedOffloadModes": [
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
                OFFLOAD_MODE_NONE,
            ],
            "requiredPackages": [
                "diffusers",
                "transformers",
                "accelerate",
                "torch",
                "optimum-quanto",
            ],
            "guardedReason": "FLUX Canny has guarded Auto coverage through generic control-image Diffusers nodes.",
        },
        "roles": _CONTROL_GRAPH_ROLES,
        "edges": _CONTROL_GRAPH_EDGES,
        "bindings": _CONTROL_GRAPH_BINDINGS,
    },
    "flux-redux:edit-image:v1": {
        "modelType": "FluxReduxPipeline",
        "mode": "edit_image",
        "profile": _profile(
            "flux-redux:direct",
            "FluxReduxPipeline",
            FLUX_REDUX_REPO,
            default_quantized_components=("transformer",),
            supported_offload_modes=(
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            retry_offload_modes=(OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK),
            max_low_memory_side=768,
            max_low_memory_steps=24,
            mode="edit_image",
            pipeline_class="FluxReduxPipeline",
        ),
        "capability": {
            **_capability(
                "FluxReduxPipeline",
                "FLUX.1 Redux dev",
                "FLUX.1-Redux-dev",
                FLUX_REDUX_REPO,
                width=1024,
                steps=28,
                guidance=3.5,
                low_vram_mode=OFFLOAD_MODE_GROUP_DISK,
                low_vram_width=768,
                low_vram_steps=20,
                execution_status="expert_only",
            ),
            "artifactCandidates": [FLUX_REDUX_REPO, FLUX_DEV_REPO],
            "supportsImageInput": True,
            "modes": ["edit_image"],
            "additionalRequirements": studio_model_requirements_for_pair(
                "FluxReduxPipeline", "edit_image"
            ),
            "modeRequirements": {
                "edit_image": {
                    "modelRequirements": studio_model_requirements_for_pair(
                        "FluxReduxPipeline", "edit_image"
                    ),
                    "requiredImages": ["referenceImages"],
                    "note": "Requires reference images plus the reviewed FLUX.1-dev base pipeline.",
                }
            },
        },
        "autoRequirements": {
            "supportedTasks": ["edit_image"],
            "defaultRepo": FLUX_REDUX_REPO,
            "qualityDefaults": {
                "width": 768,
                "height": 768,
                "steps": 24,
                "guidanceScale": 3.5,
                "maxSequenceLength": 256,
            },
            "minimum": {
                "accelerator": "cuda",
                "vramBytes": 24 * _GIB,
                "systemRamBytes": 48 * _GIB,
                "diskFreeBytes": 45 * _GIB,
            },
            "recommended": {
                "accelerator": "cuda",
                "vramBytes": 32 * _GIB,
                "systemRamBytes": 64 * _GIB,
                "diskFreeBytes": 60 * _GIB,
            },
            "fullResidency": _HIGH_MEMORY_FULL_RESIDENCY,
            "onLoadQuantization": {
                "accelerator": "cuda",
                "vramBytes": 16 * _GIB,
                "systemRamBytes": 32 * _GIB,
                "diskFreeBytes": 45 * _GIB,
                "quantizationMode": "quanto_float8",
                "quantizedComponents": ["transformer", "text_encoder_2"],
            },
            "supportedOffloadModes": [
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
                OFFLOAD_MODE_NONE,
            ],
            "requiredPackages": [
                "diffusers",
                "transformers",
                "accelerate",
                "torch",
                "optimum-quanto",
            ],
            "guardedReason": "FLUX Redux has guarded Auto coverage through generic Diffusers image/reference nodes.",
        },
        "roles": _EDIT_GRAPH_ROLES,
        "edges": _EDIT_GRAPH_EDGES,
        "bindings": _EDIT_GRAPH_BINDINGS,
    },
    "flux-kontext:edit-image:v1": {
        "modelType": "FluxKontextPipeline",
        "mode": "edit_image",
        "profile": {
            "id": "flux-kontext:direct",
            "model_type": "FluxKontextPipeline",
            "modes": ("edit_image", "multi_image_reference_edit"),
            "loader_module": "modules.DiffusersImage",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-image",
            "pipeline_class": "FluxKontextPipeline",
            "default_repo": FLUX_KONTEXT_REPO,
            "fallback_repo": None,
            "quantizable_components": ("transformer", "text_encoder_2"),
            "default_quantized_components": ("transformer",),
            "supported_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "retry_offload_modes": (OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK),
            "max_low_memory_side": 768,
            "max_low_memory_steps": 24,
            "live_proof": False,
            "compatible_repos": (FLUX_KONTEXT_NVFP4_REPO,),
        },
        "autoRequirements": {
            "supportedTasks": ["edit_image"],
            "defaultRepo": FLUX_KONTEXT_REPO,
            "preferredLowerMemoryRepo": FLUX_KONTEXT_NVFP4_REPO,
            "executionPath": "direct-diffusers-image",
            "pipelineClass": "FluxKontextPipeline",
            "qualityDefaults": {
                "width": 768,
                "height": 768,
                "steps": 24,
                "guidanceScale": 3.5,
                "maxSequenceLength": 256,
            },
            "minimum": {
                "accelerator": "cuda",
                "vramBytes": 24 * _GIB,
                "systemRamBytes": 48 * _GIB,
                "diskFreeBytes": 45 * _GIB,
            },
            "recommended": {
                "accelerator": "cuda",
                "vramBytes": 32 * _GIB,
                "systemRamBytes": 64 * _GIB,
                "diskFreeBytes": 60 * _GIB,
            },
            "fullResidency": _HIGH_MEMORY_FULL_RESIDENCY,
            "lowerMemory": {
                "accelerator": "cuda",
                "vramBytes": 16 * _GIB,
                "systemRamBytes": 32 * _GIB,
                "diskFreeBytes": 45 * _GIB,
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
            "guardedReason": (
                "FLUX Kontext uses the NVFP4 lower-memory artifact when available; failures are remembered for "
                "this machine."
            ),
        },
        "roles": _EDIT_GRAPH_ROLES,
        "edges": _EDIT_GRAPH_EDGES,
        "bindings": _EDIT_GRAPH_BINDINGS,
    },
    "flux-kontext:multi-image-reference-edit:v1": {
        "modelType": "FluxKontextPipeline",
        "mode": "multi_image_reference_edit",
        "profile": {
            "id": "flux-kontext:direct",
            "model_type": "FluxKontextPipeline",
            "modes": ("edit_image", "multi_image_reference_edit"),
            "loader_module": "modules.DiffusersImage",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-image",
            "pipeline_class": "FluxKontextPipeline",
            "default_repo": FLUX_KONTEXT_REPO,
            "fallback_repo": None,
            "quantizable_components": ("transformer", "text_encoder_2"),
            "default_quantized_components": ("transformer",),
            "supported_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "retry_offload_modes": (OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK),
            "max_low_memory_side": 768,
            "max_low_memory_steps": 24,
            "live_proof": False,
            "compatible_repos": (FLUX_KONTEXT_NVFP4_REPO,),
        },
        "roles": _EDIT_GRAPH_ROLES,
        "edges": _EDIT_GRAPH_EDGES,
        "bindings": _EDIT_GRAPH_BINDINGS,
    },
    "flux-fill:inpaint:v1": {
        "modelType": "FluxFillPipeline",
        "mode": "inpaint",
        "profile": {
            "id": "flux-fill:direct",
            "model_type": "FluxFillPipeline",
            "modes": ("inpaint", "outpaint"),
            "loader_module": "modules.DiffusersImage",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-image",
            "pipeline_class": "FluxFillPipeline",
            "default_repo": FLUX_FILL_REPO,
            "fallback_repo": None,
            "quantizable_components": ("transformer", "text_encoder_2"),
            "default_quantized_components": ("transformer",),
            "supported_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "retry_offload_modes": (OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK),
            "max_low_memory_side": 768,
            "max_low_memory_steps": 24,
            "live_proof": False,
            "compatible_repos": (),
        },
        "autoRequirements": {
            "supportedTasks": ["inpaint", "outpaint"],
            "defaultRepo": FLUX_FILL_REPO,
            "executionPath": "direct-diffusers-image",
            "pipelineClass": "FluxFillPipeline",
            "qualityDefaults": {
                "width": 768,
                "height": 768,
                "steps": 24,
                "guidanceScale": 30,
                "maxSequenceLength": 256,
            },
            "minimum": {
                "accelerator": "cuda",
                "vramBytes": 24 * _GIB,
                "systemRamBytes": 48 * _GIB,
                "diskFreeBytes": 45 * _GIB,
            },
            "recommended": {
                "accelerator": "cuda",
                "vramBytes": 32 * _GIB,
                "systemRamBytes": 64 * _GIB,
                "diskFreeBytes": 60 * _GIB,
            },
            "fullResidency": _HIGH_MEMORY_FULL_RESIDENCY,
            "onLoadQuantization": {
                "accelerator": "cuda",
                "vramBytes": 16 * _GIB,
                "systemRamBytes": 32 * _GIB,
                "diskFreeBytes": 45 * _GIB,
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
            "guardedReason": (
                "FLUX Fill has guarded Auto coverage through generic Diffusers inpaint/outpaint nodes and "
                "on-load quantization."
            ),
        },
        "roles": _INPAINT_GRAPH_ROLES,
        "edges": _INPAINT_GRAPH_EDGES,
        "bindings": _INPAINT_GRAPH_BINDINGS,
    },
    "flux-fill:outpaint:v1": {
        "modelType": "FluxFillPipeline",
        "mode": "outpaint",
        "profile": {
            "id": "flux-fill:direct",
            "model_type": "FluxFillPipeline",
            "modes": ("inpaint", "outpaint"),
            "loader_module": "modules.DiffusersImage",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-image",
            "pipeline_class": "FluxFillPipeline",
            "default_repo": FLUX_FILL_REPO,
            "fallback_repo": None,
            "quantizable_components": ("transformer", "text_encoder_2"),
            "default_quantized_components": ("transformer",),
            "supported_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "retry_offload_modes": (OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK),
            "max_low_memory_side": 768,
            "max_low_memory_steps": 24,
            "live_proof": False,
            "compatible_repos": (),
        },
        "roles": _INPAINT_GRAPH_ROLES,
        "edges": _INPAINT_GRAPH_EDGES,
        "bindings": _INPAINT_GRAPH_BINDINGS,
    },
    "flux2-klein:text-to-image:v1": {
        "modelType": "Flux2KleinPipeline",
        "mode": "text_to_image",
        "profile": {
            "id": "flux2-klein:direct",
            "model_type": "Flux2KleinPipeline",
            "modes": ("text_to_image", "edit_image", "multi_image_reference_edit"),
            "loader_module": "modules.DiffusersImage",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-image",
            "pipeline_class": "Flux2KleinPipeline",
            "default_repo": FLUX2_KLEIN_REPO,
            "fallback_repo": None,
            "quantizable_components": ("transformer", "text_encoder_2"),
            "default_quantized_components": ("transformer",),
            "supported_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "retry_offload_modes": (OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK),
            "max_low_memory_side": 768,
            "max_low_memory_steps": 24,
            "live_proof": True,
            "compatible_repos": (),
        },
        "capability": {
            "modelType": "Flux2KleinPipeline",
            "label": "FLUX.2 Klein 4B",
            "displayName": "FLUX.2-klein-4B",
            "family": "FLUX Image",
            "defaultRepo": FLUX2_KLEIN_REPO,
            "artifactLabel": "Diffusers repo",
            "defaultDtype": "bfloat16",
            "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
            "recommendedSteps": 4,
            "recommendedGuidance": 1.0,
            "guidanceLabel": "Guidance",
            "supportsImageInput": True,
            "supportsMask": False,
            "supportsMultiImage": True,
            "supportsControlImage": False,
            "supportsLayers": False,
            "supportsLora": True,
            "offloadSupport": {
                "default": OFFLOAD_MODE_MODEL_CPU,
                "lowVram": OFFLOAD_MODE_MODEL_CPU,
                "emergency": OFFLOAD_MODE_GROUP_DISK,
                "modes": list(_DIRECT_OFFLOAD_MODES),
            },
            "lowVram": {
                "dtype": "bfloat16",
                "autoOffload": True,
                "offloadMode": OFFLOAD_MODE_MODEL_CPU,
                "steps": 4,
                "width": 768,
                "height": 768,
            },
            "modes": ["text_to_image", "edit_image", "multi_image_reference_edit"],
            "executionStatus": "supported_with_model",
            "modeRequirements": {
                "edit_image": {
                    "requiredImages": ["referenceImages"],
                    "note": "Requires one source/reference image.",
                },
                "multi_image_reference_edit": {
                    "requiredImages": ["referenceImages"],
                    "note": "Requires two or more reference images.",
                },
            },
            "notes": [
                "Qualified through the generic Diffusers image facade for text, single-reference, and multi-reference generation."
            ],
        },
        "autoRequirements": {
            "supportedTasks": ["text_to_image", "edit_image", "multi_image_reference_edit"],
            "defaultRepo": FLUX2_KLEIN_REPO,
            "executionPath": "direct-diffusers-image",
            "pipelineClass": "Flux2KleinPipeline",
            "qualityDefaults": {
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "guidanceScale": 1,
                "maxSequenceLength": 512,
            },
            "minimum": {
                "accelerator": "cuda",
                "vramBytes": 13 * _GIB,
                "systemRamBytes": 24 * _GIB,
                "diskFreeBytes": 25 * _GIB,
            },
            "recommended": {
                "accelerator": "cuda",
                "vramBytes": 20 * _GIB,
                "systemRamBytes": 32 * _GIB,
                "diskFreeBytes": 35 * _GIB,
            },
            "fullResidency": _HIGH_MEMORY_FULL_RESIDENCY,
            "supportedOffloadModes": [
                OFFLOAD_MODE_NONE,
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ],
            "requiredPackages": ["diffusers", "transformers", "accelerate", "torch"],
        },
        "roles": _GRAPH_ROLES,
        "edges": _GRAPH_EDGES,
        "bindings": _GRAPH_BINDINGS,
    },
    "flux2-klein:edit-image:v1": {
        "modelType": "Flux2KleinPipeline",
        "mode": "edit_image",
        "profile": {
            "id": "flux2-klein:direct",
            "model_type": "Flux2KleinPipeline",
            "modes": ("text_to_image", "edit_image", "multi_image_reference_edit"),
            "loader_module": "modules.DiffusersImage",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-image",
            "pipeline_class": "Flux2KleinPipeline",
            "default_repo": FLUX2_KLEIN_REPO,
            "fallback_repo": None,
            "quantizable_components": ("transformer", "text_encoder_2"),
            "default_quantized_components": ("transformer",),
            "supported_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "retry_offload_modes": (OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK),
            "max_low_memory_side": 768,
            "max_low_memory_steps": 24,
            "live_proof": True,
            "compatible_repos": (),
        },
        "roles": _EDIT_GRAPH_ROLES,
        "edges": _EDIT_GRAPH_EDGES,
        "bindings": _EDIT_GRAPH_BINDINGS,
    },
    "flux2-klein:multi-image-reference-edit:v1": {
        "modelType": "Flux2KleinPipeline",
        "mode": "multi_image_reference_edit",
        "profile": {
            "id": "flux2-klein:direct",
            "model_type": "Flux2KleinPipeline",
            "modes": ("text_to_image", "edit_image", "multi_image_reference_edit"),
            "loader_module": "modules.DiffusersImage",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-image",
            "pipeline_class": "Flux2KleinPipeline",
            "default_repo": FLUX2_KLEIN_REPO,
            "fallback_repo": None,
            "quantizable_components": ("transformer", "text_encoder_2"),
            "default_quantized_components": ("transformer",),
            "supported_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "retry_offload_modes": (OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK),
            "max_low_memory_side": 768,
            "max_low_memory_steps": 24,
            "live_proof": True,
            "compatible_repos": (),
        },
        "roles": _EDIT_GRAPH_ROLES,
        "edges": _EDIT_GRAPH_EDGES,
        "bindings": _EDIT_GRAPH_BINDINGS,
    },
    "wan-22-i2v-a14b:image-to-video:v1": {
        "modelType": "WanImageToVideoPipeline",
        "mode": "image_to_video",
        "profile": {
            "id": "wan-22-image-to-video:direct",
            "model_type": "WanImageToVideoPipeline",
            "modes": ("image_to_video",),
            "loader_module": "modules.DiffusersVideo",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-video",
            "pipeline_class": "WanImageToVideoPipeline",
            "default_repo": WAN_22_I2V_A14B_REPO,
            "fallback_repo": None,
            "quantizable_components": ("transformer", "transformer_2", "text_encoder"),
            "default_quantized_components": ("transformer", "transformer_2"),
            "supported_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_DISK),
            "max_low_memory_side": 832,
            "max_low_memory_steps": 40,
            "live_proof": False,
            "compatible_repos": (),
        },
        "capability": {
            "modelType": "WanImageToVideoPipeline",
            "label": "Wan 2.2 I2V A14B",
            "displayName": "Wan2.2-I2V-A14B-Diffusers",
            "family": "Wan Video",
            "supportTier": "supported",
            "qualificationStatus": "graph-qualified-execution-pending",
            "qualifiedModes": [],
            "defaultRepo": WAN_22_I2V_A14B_REPO,
            "artifactLabel": "Diffusers repo",
            "defaultDtype": "bfloat16",
            "defaultSize": {"width": 832, "height": 480, "aspectRatio": "16:9"},
            "recommendedSteps": 40,
            "recommendedGuidance": 3.5,
            "guidanceLabel": "High-noise guidance",
            "supportsImageInput": True,
            "supportsMask": False,
            "supportsMultiImage": True,
            "supportsControlImage": False,
            "supportsLayers": False,
            "supportsLora": False,
            "supportsVideoInput": False,
            "supportsVideoMask": False,
            "outputKind": "video",
            "recommendedFrames": 81,
            "recommendedFps": 16,
            "conditioningScale": 1.0,
            "offloadSupport": {
                "default": OFFLOAD_MODE_MODEL_CPU,
                "lowVram": OFFLOAD_MODE_MODEL_CPU,
                "emergency": OFFLOAD_MODE_GROUP_DISK,
                "modes": list(_DIRECT_OFFLOAD_MODES),
            },
            "lowVram": {
                "dtype": "bfloat16",
                "autoOffload": True,
                "offloadMode": OFFLOAD_MODE_MODEL_CPU,
                "steps": 40,
                "width": 832,
                "height": 480,
                "numFrames": 81,
            },
            "modes": ["image_to_video"],
            "executionStatus": "supported_with_model",
            "notes": [
                "Uses the generic Diffusers video facade with the official dual-expert WanImageToVideoPipeline.",
                "The quality workflow quantizes both denoising experts to Quanto INT8 and runs five-second shots sequentially.",
                "Human review remains required before generated examples are promoted to the gallery.",
            ],
            "modeRequirements": {
                "image_to_video": {
                    "requiredImages": ["referenceImages"],
                    "note": "The story workflow requires one ordered opening keyframe per shot.",
                },
            },
        },
        "autoRequirements": {
            "supportedTasks": ["image_to_video"],
            "defaultRepo": WAN_22_I2V_A14B_REPO,
            "executionPath": "direct-diffusers-video",
            "pipelineClass": "WanImageToVideoPipeline",
            "qualityDefaults": {
                "width": 832,
                "height": 480,
                "steps": 40,
                "guidanceScale": 3.5,
                "numFrames": 81,
            },
            "minimum": {
                "accelerator": "cuda",
                "vramBytes": 24 * _GIB,
                "systemRamBytes": 64 * _GIB,
                "diskFreeBytes": 140 * _GIB,
            },
            "recommended": {
                "accelerator": "cuda",
                "vramBytes": 80 * _GIB,
                "systemRamBytes": 64 * _GIB,
                "diskFreeBytes": 140 * _GIB,
            },
            "fullResidency": _HIGH_MEMORY_FULL_RESIDENCY,
            "supportedOffloadModes": [
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ],
            "requiredPackages": ["diffusers", "transformers", "accelerate", "torch"],
            "guardedReason": "Wan 2.2 I2V A14B uses the generic Diffusers video graph with a dual-transformer execution contract.",
        },
        "roles": _I2V_GRAPH_ROLES,
        "edges": _I2V_GRAPH_EDGES,
        "bindings": _I2V_GRAPH_BINDINGS,
    },
    "wan-22-ti2v-5b:text-to-video:v1": {
        "modelType": "WanTI2VPipeline",
        "mode": "text_to_video",
        "profile": {
            "id": "wan-22-ti2v-5b:direct",
            "model_type": "WanTI2VPipeline",
            "modes": ("text_to_video",),
            "loader_module": "modules.DiffusersVideo",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-video",
            "pipeline_class": "WanTI2VPipeline",
            "default_repo": WAN_22_TI2V_5B_REPO,
            "fallback_repo": None,
            "quantizable_components": ("transformer", "text_encoder"),
            "default_quantized_components": (),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "max_low_memory_side": 1280,
            "max_low_memory_steps": 50,
            "live_proof": False,
            "compatible_repos": (),
        },
        "capability": {
            "modelType": "WanTI2VPipeline",
            "label": "Wan 2.2 TI2V 5B",
            "displayName": "Wan2.2-TI2V-5B-Diffusers",
            "family": "Wan Video",
            "supportTier": "supported",
            "qualificationStatus": "graph-qualified-execution-pending",
            "qualifiedModes": [],
            "defaultRepo": WAN_22_TI2V_5B_REPO,
            "artifactLabel": "Diffusers repo",
            "defaultDtype": "bfloat16",
            "defaultSize": {"width": 1280, "height": 704, "aspectRatio": "16:9"},
            "recommendedSteps": 50,
            "recommendedGuidance": 5.0,
            "guidanceLabel": "Guidance",
            "supportsImageInput": True,
            "supportsMask": False,
            "supportsMultiImage": False,
            "supportsControlImage": False,
            "supportsLayers": False,
            "supportsLora": True,
            "supportsVideoInput": False,
            "supportsVideoMask": False,
            "outputKind": "video",
            "recommendedFrames": 121,
            "recommendedFps": 24,
            "conditioningScale": 1.0,
            "offloadSupport": {
                "default": OFFLOAD_MODE_MODEL_CPU,
                "lowVram": OFFLOAD_MODE_MODEL_CPU,
                "emergency": OFFLOAD_MODE_GROUP_DISK,
                "modes": list(_DIRECT_OFFLOAD_MODES),
            },
            "lowVram": {
                "dtype": "bfloat16",
                "autoOffload": True,
                "offloadMode": OFFLOAD_MODE_MODEL_CPU,
                "steps": 50,
                "width": 1280,
                "height": 704,
                "numFrames": 121,
            },
            "modes": ["text_to_video"],
            "executionStatus": "supported_with_model",
            "notes": [
                "Uses the official dense Wan 2.2 5B high-compression video model for five-second 720p shots.",
                "The current Diffusers WanPipeline exposes text-to-video; A14B remains the image-to-video adapter.",
                "Human review remains required before generated examples are promoted to the gallery.",
            ],
            "modeRequirements": {},
        },
        "autoRequirements": {
            "supportedTasks": ["text_to_video"],
            "defaultRepo": WAN_22_TI2V_5B_REPO,
            "executionPath": "direct-diffusers-video",
            "pipelineClass": "WanTI2VPipeline",
            "qualityDefaults": {
                "width": 1280,
                "height": 704,
                "steps": 50,
                "guidanceScale": 5,
                "numFrames": 121,
            },
            "minimum": {
                "accelerator": "cuda",
                "vramBytes": 24 * _GIB,
                "systemRamBytes": 48 * _GIB,
                "diskFreeBytes": 45 * _GIB,
            },
            "recommended": {
                "accelerator": "cuda",
                "vramBytes": 40 * _GIB,
                "systemRamBytes": 64 * _GIB,
                "diskFreeBytes": 45 * _GIB,
            },
            "fullResidency": _HIGH_MEMORY_FULL_RESIDENCY,
            "supportedOffloadModes": list(_DIRECT_OFFLOAD_MODES),
            "requiredPackages": ["diffusers", "transformers", "accelerate", "torch"],
            "guardedReason": "Wan 2.2 TI2V 5B uses the generic Diffusers video graph with an exact direct loader contract.",
        },
        "roles": _VIDEO_GRAPH_ROLES,
        "edges": _VIDEO_GRAPH_EDGES,
        "bindings": _VIDEO_GRAPH_BINDINGS,
    },
    "wan-21-t2v-1.3b:text-to-video:v1": {
        "modelType": "WanVideoPipeline",
        "mode": "text_to_video",
        "autoRequirementKey": "WanVideoPipeline:text_to_video",
        "profile": {
            "id": "wan-text-to-video:direct",
            "model_type": "WanVideoPipeline",
            "modes": ("text_to_video",),
            "loader_module": "modules.DiffusersVideo",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-video",
            "pipeline_class": "WanPipeline",
            "default_repo": WAN_T2V_1_3B_REPO,
            "fallback_repo": None,
            "quantizable_components": (),
            "default_quantized_components": (),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_DISK),
            "max_low_memory_side": 832,
            "max_low_memory_steps": 30,
            "live_proof": True,
            "compatible_repos": (),
        },
        "autoRequirements": {
            "supportedTasks": ["text_to_video"],
            "defaultRepo": WAN_T2V_1_3B_REPO,
            "executionPath": "direct-diffusers-video",
            "pipelineClass": "WanPipeline",
            "qualityDefaults": {
                "width": 832,
                "height": 480,
                "steps": 30,
                "guidanceScale": 5,
                "numFrames": 81,
            },
            "minimum": {"accelerator": "cuda", "vramBytes": 10 * _GIB, "systemRamBytes": 24 * _GIB},
            "recommended": {"accelerator": "cuda", "vramBytes": 12 * _GIB, "systemRamBytes": 32 * _GIB},
            "highQuality": {"accelerator": "cuda", "vramBytes": 24 * _GIB, "systemRamBytes": 48 * _GIB},
            "supportedOffloadModes": [
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
                OFFLOAD_MODE_NONE,
            ],
        },
        "roles": _VIDEO_GRAPH_ROLES,
        "edges": _VIDEO_GRAPH_EDGES,
        "bindings": _VIDEO_GRAPH_BINDINGS,
    },
    "wan-21-t2v-1.3b:video-to-video:v1": {
        "modelType": "WanVideoPipeline",
        "mode": "video_to_video",
        "profile": {
            "id": "wan-video-to-video:direct",
            "model_type": "WanVideoPipeline",
            "modes": ("video_to_video", "video_color_edit"),
            "loader_module": "modules.DiffusersVideo",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-video",
            "pipeline_class": "WanVideoToVideoPipeline",
            "default_repo": WAN_T2V_1_3B_REPO,
            "fallback_repo": None,
            "quantizable_components": (),
            "default_quantized_components": (),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_DISK),
            "max_low_memory_side": 832,
            "max_low_memory_steps": 30,
            "live_proof": False,
            "compatible_repos": (),
        },
        "roles": _V2V_GRAPH_ROLES,
        "edges": _V2V_GRAPH_EDGES,
        "bindings": _V2V_GRAPH_BINDINGS,
    },
    "wan-21-t2v-1.3b:video-color-edit:v1": {
        "modelType": "WanVideoPipeline",
        "mode": "video_color_edit",
        "profile": {
            "id": "wan-video-to-video:direct",
            "model_type": "WanVideoPipeline",
            "modes": ("video_to_video", "video_color_edit"),
            "loader_module": "modules.DiffusersVideo",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-video",
            "pipeline_class": "WanVideoToVideoPipeline",
            "default_repo": WAN_T2V_1_3B_REPO,
            "fallback_repo": None,
            "quantizable_components": (),
            "default_quantized_components": (),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_DISK),
            "max_low_memory_side": 832,
            "max_low_memory_steps": 30,
            "live_proof": False,
            "compatible_repos": (),
        },
        "roles": _V2V_GRAPH_ROLES,
        "edges": _V2V_GRAPH_EDGES,
        "bindings": _V2V_GRAPH_BINDINGS,
    },
    "ltx-video-0.9.8-13b-distilled:text-to-video:v1": {
        "modelType": "LTXVideoPipeline",
        "mode": "text_to_video",
        "profile": {
            "id": "ltx-video:direct",
            "model_type": "LTXVideoPipeline",
            "modes": ("text_to_video", "image_to_video", "video_to_video", "reference_to_video"),
            "loader_module": "modules.DiffusersVideo",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-video",
            "pipeline_class": "LTXConditionPipeline",
            "default_repo": LTX_VIDEO_REPO,
            "fallback_repo": None,
            "quantizable_components": ("transformer", "text_encoder"),
            "default_quantized_components": (),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_DISK),
            "max_low_memory_side": 704,
            "max_low_memory_steps": 8,
            "live_proof": False,
            "compatible_repos": (),
        },
        "roles": _VIDEO_GRAPH_ROLES,
        "edges": _VIDEO_GRAPH_EDGES,
        "bindings": _LTX_T2V_GRAPH_BINDINGS,
    },
    "ltx-video-0.9.8-13b-distilled:image-to-video:v1": {
        "modelType": "LTXVideoPipeline",
        "mode": "image_to_video",
        "profile": {
            "id": "ltx-video:direct",
            "model_type": "LTXVideoPipeline",
            "modes": ("text_to_video", "image_to_video", "video_to_video", "reference_to_video"),
            "loader_module": "modules.DiffusersVideo",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-video",
            "pipeline_class": "LTXConditionPipeline",
            "default_repo": LTX_VIDEO_REPO,
            "fallback_repo": None,
            "quantizable_components": ("transformer", "text_encoder"),
            "default_quantized_components": (),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_DISK),
            "max_low_memory_side": 704,
            "max_low_memory_steps": 8,
            "live_proof": False,
            "compatible_repos": (),
        },
        "roles": _I2V_GRAPH_ROLES,
        "edges": _I2V_GRAPH_EDGES,
        "bindings": _LTX_I2V_GRAPH_BINDINGS,
    },
    "ltx-video-0.9.8-13b-distilled:video-to-video:v1": {
        "modelType": "LTXVideoPipeline",
        "mode": "video_to_video",
        "profile": {
            "id": "ltx-video:direct",
            "model_type": "LTXVideoPipeline",
            "modes": ("text_to_video", "image_to_video", "video_to_video", "reference_to_video"),
            "loader_module": "modules.DiffusersVideo",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-video",
            "pipeline_class": "LTXConditionPipeline",
            "default_repo": LTX_VIDEO_REPO,
            "fallback_repo": None,
            "quantizable_components": ("transformer", "text_encoder"),
            "default_quantized_components": (),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_DISK),
            "max_low_memory_side": 704,
            "max_low_memory_steps": 8,
            "live_proof": False,
            "compatible_repos": (),
        },
        "roles": _V2V_GRAPH_ROLES,
        "edges": _V2V_GRAPH_EDGES,
        "bindings": _LTX_V2V_GRAPH_BINDINGS,
    },
    "ltx-video-0.9.8-13b-distilled:reference-to-video:v1": {
        "modelType": "LTXVideoPipeline",
        "mode": "reference_to_video",
        "profile": {
            "id": "ltx-video:direct",
            "model_type": "LTXVideoPipeline",
            "modes": ("text_to_video", "image_to_video", "video_to_video", "reference_to_video"),
            "loader_module": "modules.DiffusersVideo",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-video",
            "pipeline_class": "LTXConditionPipeline",
            "default_repo": LTX_VIDEO_REPO,
            "fallback_repo": None,
            "quantizable_components": ("transformer", "text_encoder"),
            "default_quantized_components": (),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_DISK),
            "max_low_memory_side": 704,
            "max_low_memory_steps": 8,
            "live_proof": False,
            "compatible_repos": (),
        },
        "roles": _I2V_GRAPH_ROLES,
        "edges": _I2V_GRAPH_EDGES,
        "bindings": _LTX_I2V_GRAPH_BINDINGS,
    },
    "ace-step-v1.5-xl-turbo:text-to-audio:v1": {
        "modelType": "AceStepAudioPipeline",
        "mode": "text_to_audio",
        "profile": {
            "id": "ace-step-audio:direct",
            "model_type": "AceStepAudioPipeline",
            "modes": ("text_to_audio", "audio_variation", "audio_continuation", "audio_repaint"),
            "loader_module": "modules.DiffusersAudio",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-audio",
            "pipeline_class": "AceStepPipeline",
            "default_repo": ACE_STEP_REPO,
            "fallback_repo": None,
            "quantizable_components": (),
            "default_quantized_components": (),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "max_low_memory_side": None,
            "max_low_memory_steps": 8,
            "live_proof": False,
            "compatible_repos": (ACE_STEP_LORA_BASE_REPO,),
        },
        "roles": _AUDIO_GRAPH_ROLES,
        "edges": _AUDIO_GRAPH_EDGES,
        "bindings": _AUDIO_GRAPH_BINDINGS,
    },
    "ace-step-v1.5-xl-turbo:audio-variation:v1": {
        "modelType": "AceStepAudioPipeline",
        "mode": "audio_variation",
        "profile": {
            "id": "ace-step-audio:direct",
            "model_type": "AceStepAudioPipeline",
            "modes": ("text_to_audio", "audio_variation", "audio_continuation", "audio_repaint"),
            "loader_module": "modules.DiffusersAudio",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-audio",
            "pipeline_class": "AceStepPipeline",
            "default_repo": ACE_STEP_REPO,
            "fallback_repo": None,
            "quantizable_components": (),
            "default_quantized_components": (),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "max_low_memory_side": None,
            "max_low_memory_steps": 8,
            "live_proof": False,
            "compatible_repos": (ACE_STEP_LORA_BASE_REPO,),
        },
        "roles": _AUDIO_VARIATION_GRAPH_ROLES,
        "edges": _AUDIO_VARIATION_GRAPH_EDGES,
        "bindings": _AUDIO_VARIATION_GRAPH_BINDINGS,
    },
    "ace-step-v1.5-xl-turbo:audio-continuation:v1": {
        "modelType": "AceStepAudioPipeline",
        "mode": "audio_continuation",
        "profile": {
            "id": "ace-step-audio:direct",
            "model_type": "AceStepAudioPipeline",
            "modes": ("text_to_audio", "audio_variation", "audio_continuation", "audio_repaint"),
            "loader_module": "modules.DiffusersAudio",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-audio",
            "pipeline_class": "AceStepPipeline",
            "default_repo": ACE_STEP_REPO,
            "fallback_repo": None,
            "quantizable_components": (),
            "default_quantized_components": (),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "max_low_memory_side": None,
            "max_low_memory_steps": 8,
            "live_proof": False,
            "compatible_repos": (ACE_STEP_LORA_BASE_REPO,),
        },
        "roles": _AUDIO_CONTINUATION_GRAPH_ROLES,
        "edges": _AUDIO_CONTINUATION_GRAPH_EDGES,
        "bindings": _AUDIO_CONTINUATION_GRAPH_BINDINGS,
    },
    "ace-step-v1.5-xl-turbo:audio-repaint:v1": {
        "modelType": "AceStepAudioPipeline",
        "mode": "audio_repaint",
        "profile": {
            "id": "ace-step-audio:direct",
            "model_type": "AceStepAudioPipeline",
            "modes": ("text_to_audio", "audio_variation", "audio_continuation", "audio_repaint"),
            "loader_module": "modules.DiffusersAudio",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-audio",
            "pipeline_class": "AceStepPipeline",
            "default_repo": ACE_STEP_REPO,
            "fallback_repo": None,
            "quantizable_components": (),
            "default_quantized_components": (),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "max_low_memory_side": None,
            "max_low_memory_steps": 8,
            "live_proof": False,
            "compatible_repos": (ACE_STEP_LORA_BASE_REPO,),
        },
        "roles": _AUDIO_REPAINT_GRAPH_ROLES,
        "edges": _AUDIO_REPAINT_GRAPH_EDGES,
        "bindings": _AUDIO_REPAINT_GRAPH_BINDINGS,
    },
    "qwen-image-edit:inpaint:v1": {
        "modelType": "QwenImageEditModularPipeline",
        "mode": "inpaint",
        "profile": {
            "id": "qwen-edit:direct-inpaint",
            "model_type": "QwenImageEditModularPipeline",
            "modes": ("inpaint", "outpaint"),
            "loader_module": "modules.DiffusersImage",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-image",
            "pipeline_class": "QwenImageEditInpaintPipeline",
            "default_repo": "Qwen/Qwen-Image-Edit",
            "fallback_repo": None,
            "quantizable_components": ("transformer", "text_encoder"),
            "default_quantized_components": ("transformer", "text_encoder"),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "max_low_memory_side": 768,
            "max_low_memory_steps": 24,
            "live_proof": False,
            "compatible_repos": (),
        },
        "roles": _INPAINT_GRAPH_ROLES,
        "edges": _INPAINT_GRAPH_EDGES,
        "bindings": _INPAINT_GRAPH_BINDINGS,
    },
    "wan-vace-1.3b:text-to-video:v1": {
        "modelType": "WanVACEPipeline",
        "mode": "text_to_video",
        "profile": {
            "id": "wan-vace:direct",
            "model_type": "WanVACEPipeline",
            "modes": ("text_to_video", "video_inpaint", "video_outpaint", "control_to_video"),
            "loader_module": "modules.DiffusersVideo",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-wan-vace",
            "pipeline_class": "WanVACEPipeline",
            "default_repo": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
            "fallback_repo": None,
            "quantizable_components": (),
            "default_quantized_components": (),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK),
            "max_low_memory_side": 832,
            "max_low_memory_steps": 24,
            "live_proof": False,
            "compatible_repos": (),
        },
        "roles": _VIDEO_GRAPH_ROLES,
        "edges": _VIDEO_GRAPH_EDGES,
        "bindings": _WAN_VACE_GRAPH_BINDINGS,
    },
    "wan-vace-1.3b:video-inpaint:v1": {
        "modelType": "WanVACEPipeline",
        "mode": "video_inpaint",
        "profile": {
            "id": "wan-vace:direct",
            "model_type": "WanVACEPipeline",
            "modes": ("text_to_video", "video_inpaint", "video_outpaint", "control_to_video"),
            "loader_module": "modules.DiffusersVideo",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-wan-vace",
            "pipeline_class": "WanVACEPipeline",
            "default_repo": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
            "fallback_repo": None,
            "quantizable_components": (),
            "default_quantized_components": (),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK),
            "max_low_memory_side": 832,
            "max_low_memory_steps": 24,
            "live_proof": False,
            "compatible_repos": (),
        },
        "roles": _VACE_INPAINT_GRAPH_ROLES,
        "edges": _VACE_INPAINT_GRAPH_EDGES,
        "bindings": _VACE_INPAINT_GRAPH_BINDINGS,
    },
    "wan-vace-1.3b:video-outpaint:v1": {
        "modelType": "WanVACEPipeline",
        "mode": "video_outpaint",
        "profile": {
            "id": "wan-vace:direct",
            "model_type": "WanVACEPipeline",
            "modes": ("text_to_video", "video_inpaint", "video_outpaint", "control_to_video"),
            "loader_module": "modules.DiffusersVideo",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-wan-vace",
            "pipeline_class": "WanVACEPipeline",
            "default_repo": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
            "fallback_repo": None,
            "quantizable_components": (),
            "default_quantized_components": (),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK),
            "max_low_memory_side": 832,
            "max_low_memory_steps": 24,
            "live_proof": False,
            "compatible_repos": (),
        },
        "roles": _VACE_INPAINT_GRAPH_ROLES,
        "edges": _VACE_INPAINT_GRAPH_EDGES,
        "bindings": _VACE_OUTPAINT_GRAPH_BINDINGS,
    },
    "wan-vace-1.3b:control-to-video:v1": {
        "modelType": "WanVACEPipeline",
        "mode": "control_to_video",
        "profile": {
            "id": "wan-vace:direct",
            "model_type": "WanVACEPipeline",
            "modes": ("text_to_video", "video_inpaint", "video_outpaint", "control_to_video"),
            "loader_module": "modules.DiffusersVideo",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-wan-vace",
            "pipeline_class": "WanVACEPipeline",
            "default_repo": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
            "fallback_repo": None,
            "quantizable_components": (),
            "default_quantized_components": (),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK),
            "max_low_memory_side": 832,
            "max_low_memory_steps": 24,
            "live_proof": False,
            "compatible_repos": (),
        },
        "roles": _VACE_CONTROL_GRAPH_ROLES,
        "edges": _VACE_CONTROL_GRAPH_EDGES,
        "bindings": _VACE_CONTROL_GRAPH_BINDINGS,
    },
    "qwen-image-edit:outpaint:v1": {
        "modelType": "QwenImageEditModularPipeline",
        "mode": "outpaint",
        "profile": {
            "id": "qwen-edit:direct-inpaint",
            "model_type": "QwenImageEditModularPipeline",
            "modes": ("inpaint", "outpaint"),
            "loader_module": "modules.DiffusersImage",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-image",
            "pipeline_class": "QwenImageEditInpaintPipeline",
            "default_repo": "Qwen/Qwen-Image-Edit",
            "fallback_repo": None,
            "quantizable_components": ("transformer", "text_encoder"),
            "default_quantized_components": ("transformer", "text_encoder"),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "max_low_memory_side": 768,
            "max_low_memory_steps": 24,
            "live_proof": False,
            "compatible_repos": (),
        },
        "roles": _QWEN_OUTPAINT_GRAPH_ROLES,
        "edges": _QWEN_OUTPAINT_GRAPH_EDGES,
        "bindings": _QWEN_OUTPAINT_GRAPH_BINDINGS,
    },
    "z-image:text-to-image:v1": {
        "modelType": "ZImageModularPipeline",
        "mode": "text_to_image",
        "profile": {
            "id": "z-image:auto",
            "model_type": "ZImageModularPipeline",
            "modes": ("text_to_image",),
            "loader_module": "modules.DiffusersImage",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-image",
            "pipeline_class": "ZImagePipeline",
            "default_repo": Z_IMAGE_REPO,
            "fallback_repo": None,
            "quantizable_components": (),
            "default_quantized_components": (),
            "supported_offload_modes": (
                OFFLOAD_MODE_NONE,
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_DISK),
            "max_low_memory_side": 1024,
            "max_low_memory_steps": 8,
            "live_proof": False,
            "compatible_repos": (),
        },
    },
    "z-image:edit-image:v1": {
        "modelType": "ZImageModularPipeline",
        "mode": "edit_image",
        "profile": {
            "id": "z-image:img2img-direct",
            "model_type": "ZImageModularPipeline",
            "modes": ("edit_image",),
            "loader_module": "modules.DiffusersImage",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-image",
            "pipeline_class": "ZImageImg2ImgPipeline",
            "default_repo": Z_IMAGE_REPO,
            "fallback_repo": None,
            "quantizable_components": (),
            "default_quantized_components": (),
            "supported_offload_modes": (
                OFFLOAD_MODE_NONE,
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_DISK),
            "max_low_memory_side": 1024,
            "max_low_memory_steps": 8,
            "live_proof": False,
            "compatible_repos": (),
        },
        "roles": _EDIT_GRAPH_ROLES,
        "edges": _EDIT_GRAPH_EDGES,
        "bindings": _EDIT_GRAPH_BINDINGS
        + (("diffusersImagePipeline", "revision", "defaultRevision"),),
    },
    "qwen-image-2512:text-to-image:v1": {
        "modelType": "QwenImageModularPipeline",
        "mode": "text_to_image",
        "profile": {
            "id": "qwen-image:t2i-direct",
            "model_type": "QwenImageModularPipeline",
            "modes": ("text_to_image",),
            "loader_module": "modules.DiffusersImage",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-image",
            "pipeline_class": "QwenImagePipeline",
            "default_repo": QWEN_IMAGE_2512_REPO,
            "fallback_repo": "unsloth/Qwen-Image-2512-unsloth-bnb-4bit",
            "quantizable_components": ("transformer", "text_encoder"),
            "default_quantized_components": (),
            "supported_offload_modes": (
                OFFLOAD_MODE_NONE,
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "retry_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "max_low_memory_side": 1328,
            "max_low_memory_steps": 50,
            "live_proof": False,
            "compatible_repos": (),
        },
    },
    "qwen-image-2512:edit-image:v1": {
        "modelType": "QwenImageModularPipeline",
        "mode": "edit_image",
        "profile": {
            "id": "qwen-image:img2img-direct",
            "model_type": "QwenImageModularPipeline",
            "modes": ("edit_image",),
            "loader_module": "modules.DiffusersImage",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-image",
            "pipeline_class": "QwenImageImg2ImgPipeline",
            "default_repo": QWEN_IMAGE_2512_REPO,
            "fallback_repo": None,
            "quantizable_components": ("transformer", "text_encoder"),
            "default_quantized_components": (),
            "supported_offload_modes": (
                OFFLOAD_MODE_NONE,
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "retry_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "max_low_memory_side": 1328,
            "max_low_memory_steps": 50,
            "live_proof": False,
            "compatible_repos": (),
        },
        "roles": _EDIT_GRAPH_ROLES,
        "edges": _EDIT_GRAPH_EDGES,
        "bindings": _EDIT_GRAPH_BINDINGS
        + (("diffusersImagePipeline", "revision", "defaultRevision"),),
    },
    "qwen-image-2512:inpaint:v1": {
        "modelType": "QwenImageModularPipeline",
        "mode": "inpaint",
        "profile": {
            "id": "qwen-image:inpaint-direct",
            "model_type": "QwenImageModularPipeline",
            "modes": ("inpaint",),
            "loader_module": "modules.DiffusersImage",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-image",
            "pipeline_class": "QwenImageInpaintPipeline",
            "default_repo": QWEN_IMAGE_2512_REPO,
            "fallback_repo": None,
            "quantizable_components": ("transformer", "text_encoder"),
            "default_quantized_components": (),
            "supported_offload_modes": (
                OFFLOAD_MODE_NONE,
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "retry_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "max_low_memory_side": 1328,
            "max_low_memory_steps": 50,
            "live_proof": False,
            "compatible_repos": (),
        },
        "roles": _INPAINT_GRAPH_ROLES,
        "edges": _INPAINT_GRAPH_EDGES,
        "bindings": _INPAINT_GRAPH_BINDINGS
        + (("diffusersImagePipeline", "revision", "defaultRevision"),),
    },
    "qwen-image-edit:edit-image:v1": {
        "modelType": "QwenImageEditModularPipeline",
        "mode": "edit_image",
        "profile": {
            "id": "qwen-edit:modular",
            "model_type": "QwenImageEditModularPipeline",
            "modes": ("edit_image",),
            "loader_module": "modules.ModularDiffusers",
            "loader_action": "ModelsLoader",
            "execution_path": "modular-diffusers",
            "pipeline_class": "QwenImageEditModularPipeline",
            "default_repo": "Qwen/Qwen-Image-Edit",
            "fallback_repo": None,
            "quantizable_components": ("transformer", "text_encoder"),
            "default_quantized_components": ("transformer", "text_encoder"),
            "supported_offload_modes": (
                OFFLOAD_MODE_NONE,
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "retry_offload_modes": (OFFLOAD_MODE_GROUP_DISK,),
            "max_low_memory_side": 768,
            "max_low_memory_steps": 24,
            "live_proof": False,
            "compatible_repos": (),
        },
        "roles": _MODULAR_EDIT_GRAPH_ROLES,
        "edges": _MODULAR_EDIT_GRAPH_EDGES,
        "bindings": _MODULAR_EDIT_GRAPH_BINDINGS,
    },
    "qwen-image-edit-plus:edit-image:v1": {
        "modelType": "QwenImageEditPlusModularPipeline",
        "mode": "edit_image",
        "profile": _MODULAR_EDIT_PLUS_PROFILE,
        "roles": _MODULAR_EDIT_GRAPH_ROLES,
        "edges": _MODULAR_EDIT_GRAPH_EDGES,
        "bindings": _MODULAR_EDIT_GRAPH_BINDINGS,
    },
    "qwen-image-edit-plus:multi-image-reference-edit:v1": {
        "modelType": "QwenImageEditPlusModularPipeline",
        "mode": "multi_image_reference_edit",
        "profile": _MODULAR_EDIT_PLUS_PROFILE,
        "roles": _MODULAR_EDIT_GRAPH_ROLES,
        "edges": _MODULAR_EDIT_GRAPH_EDGES,
        "bindings": _MODULAR_EDIT_GRAPH_BINDINGS,
    },
    "qwen-image-layered:layer-decomposition:v1": {
        "modelType": "QwenImageLayeredModularPipeline",
        "mode": "layer_decomposition",
        "profile": _MODULAR_LAYERED_PROFILE,
        "roles": _MODULAR_EDIT_GRAPH_ROLES,
        "edges": _MODULAR_LAYERED_GRAPH_EDGES,
        "bindings": _MODULAR_LAYERED_GRAPH_BINDINGS,
    },
    "qwen-image-2512:control-image:v1": {
        "modelType": "QwenImageModularPipeline",
        "mode": "control_image",
        "profile": _MODULAR_CONTROL_PROFILE,
        "roles": _MODULAR_CONTROL_GRAPH_ROLES,
        "edges": _MODULAR_CONTROL_GRAPH_EDGES,
        "bindings": _MODULAR_CONTROL_GRAPH_BINDINGS,
    },
    "stable-audio-open-1.0:text-to-audio:v1": {
        "modelType": "StableAudioPipeline",
        "mode": "text_to_audio",
        "profile": {
            "id": "stable-audio:direct",
            "model_type": "StableAudioPipeline",
            "modes": ("text_to_audio",),
            "loader_module": "modules.DiffusersAudio",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-audio",
            "pipeline_class": "StableAudioPipeline",
            "default_repo": STABLE_AUDIO_REPO,
            "fallback_repo": None,
            "quantizable_components": (),
            "default_quantized_components": (),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "max_low_memory_side": None,
            "max_low_memory_steps": 100,
            "live_proof": False,
            "compatible_repos": (),
        },
        "capability": {
            "modelType": "StableAudioPipeline",
            "label": "Stable Audio Open 1.0",
            "displayName": "stable-audio-open-1.0",
            "family": "Stable Audio",
            "defaultRepo": STABLE_AUDIO_REPO,
            "artifactLabel": "Diffusers audio repo",
            "defaultDtype": "bfloat16",
            "defaultSize": {"width": 0, "height": 0, "aspectRatio": "custom"},
            "recommendedSteps": 100,
            "recommendedGuidance": 7.0,
            "guidanceLabel": "Guidance",
            "supportsImageInput": False,
            "supportsMask": False,
            "supportsMultiImage": False,
            "supportsControlImage": False,
            "supportsLayers": False,
            "supportsLora": False,
            "supportsAudioInput": False,
            "outputKind": "audio",
            "recommendedSampleRate": 48000,
            "recommendedDuration": 30,
            "offloadSupport": {
                "default": OFFLOAD_MODE_MODEL_CPU,
                "lowVram": OFFLOAD_MODE_MODEL_CPU,
                "emergency": OFFLOAD_MODE_GROUP_DISK,
                "modes": list(_DIRECT_OFFLOAD_MODES),
            },
            "lowVram": {
                "dtype": "bfloat16",
                "autoOffload": True,
                "offloadMode": OFFLOAD_MODE_MODEL_CPU,
                "steps": 100,
            },
            "modes": ["text_to_audio"],
            "executionStatus": "expert_only",
            "revisionCandidates": [require_catalog_revision(STABLE_AUDIO_REPO)],
            "notes": [
                "Stable Audio uses the generic Diffusers audio loader and generator nodes.",
                "Auto and Gallery remain disabled until a live resource and output qualification exists.",
            ],
        },
        "roles": _AUDIO_GRAPH_ROLES,
        "edges": _AUDIO_GRAPH_EDGES,
        "bindings": _STABLE_AUDIO_GRAPH_BINDINGS,
    },
    "longcat-audio-dit-1b:text-to-audio:v1": {
        "modelType": "LongCatAudioDiTPipeline",
        "mode": "text_to_audio",
        "profile": {
            "id": "longcat-audio-dit-1b:direct",
            "model_type": "LongCatAudioDiTPipeline",
            "modes": ("text_to_audio",),
            "loader_module": "modules.DiffusersAudio",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-audio",
            "pipeline_class": "LongCatAudioDiTPipeline",
            "default_repo": LONGCAT_AUDIO_DIT_REPO,
            "fallback_repo": None,
            "quantizable_components": (),
            "default_quantized_components": (),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "max_low_memory_side": None,
            "max_low_memory_steps": 16,
            "live_proof": False,
            "compatible_repos": (),
        },
        "capability": {
            "modelType": "LongCatAudioDiTPipeline",
            "label": "LongCat AudioDiT 1B",
            "displayName": "LongCat-AudioDiT-1B-Diffusers",
            "family": "LongCat AudioDiT",
            "supportTier": "supported",
            "qualificationStatus": "graph-qualified-execution-pending",
            "qualifiedModes": [],
            "defaultRepo": LONGCAT_AUDIO_DIT_REPO,
            "artifactLabel": "Reviewed Diffusers-format safetensors conversion",
            "defaultDtype": "bfloat16",
            "defaultSize": {"width": 0, "height": 0, "aspectRatio": "custom"},
            "recommendedSteps": 16,
            "recommendedGuidance": 4.0,
            "guidanceLabel": "Guidance",
            "supportsImageInput": False,
            "supportsMask": False,
            "supportsMultiImage": False,
            "supportsControlImage": False,
            "supportsLayers": False,
            "supportsLora": False,
            "supportsAudioInput": False,
            "outputKind": "audio",
            "recommendedSampleRate": 24000,
            "recommendedDuration": 5,
            "offloadSupport": {
                "default": OFFLOAD_MODE_MODEL_CPU,
                "lowVram": OFFLOAD_MODE_SEQUENTIAL_CPU,
                "emergency": OFFLOAD_MODE_GROUP_DISK,
                "modes": list(_DIRECT_OFFLOAD_MODES),
            },
            "lowVram": {
                "dtype": "bfloat16",
                "autoOffload": True,
                "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
                "steps": 16,
            },
            "modes": ["text_to_audio"],
            "executionStatus": "expert_only",
            "revisionCandidates": [require_catalog_revision(LONGCAT_AUDIO_DIT_REPO)],
            "autoEligible": False,
            "templateEligible": True,
            "galleryEligible": False,
            "notes": [
                "The immutable reviewed Diffusers-format conversion contains only safetensors weights and no repository Python.",
                "The approximately 5.70 GB snapshot follows the upstream MIT license and runs a five-second, 16-step, guidance-4 recipe at 24 kHz.",
                "Auto and Gallery remain disabled pending remote runtime, output-quality, provenance, and rights review.",
            ],
        },
        "roles": _AUDIO_GRAPH_ROLES,
        "edges": _AUDIO_GRAPH_EDGES,
        "bindings": _LONGCAT_AUDIO_DIT_GRAPH_BINDINGS,
    },
    "audioldm2-base:text-to-audio:v1": {
        "modelType": "AudioLDM2Pipeline",
        "mode": "text_to_audio",
        "profile": {
            "id": "audioldm2-base:direct",
            "model_type": "AudioLDM2Pipeline",
            "modes": ("text_to_audio",),
            "loader_module": "modules.DiffusersAudio",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-audio",
            "pipeline_class": "AudioLDM2Pipeline",
            "default_repo": AUDIO_LDM2_REPO,
            "fallback_repo": None,
            "quantizable_components": (),
            "default_quantized_components": (),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "max_low_memory_side": None,
            "max_low_memory_steps": 200,
            "live_proof": False,
            "compatible_repos": (),
        },
        "capability": {
            "modelType": "AudioLDM2Pipeline",
            "label": "AudioLDM2 Base",
            "displayName": "audioldm2",
            "family": "AudioLDM2",
            "supportTier": "supported",
            "qualificationStatus": "graph-qualified-execution-pending",
            "qualifiedModes": [],
            "defaultRepo": AUDIO_LDM2_REPO,
            "artifactLabel": "Diffusers safetensors repo",
            "defaultDtype": "float16",
            "defaultSize": {"width": 0, "height": 0, "aspectRatio": "custom"},
            "recommendedSteps": 200,
            "recommendedGuidance": 3.5,
            "guidanceLabel": "Guidance",
            "supportsImageInput": False,
            "supportsMask": False,
            "supportsMultiImage": False,
            "supportsControlImage": False,
            "supportsLayers": False,
            "supportsLora": False,
            "supportsAudioInput": False,
            "outputKind": "audio",
            "recommendedSampleRate": 16000,
            "recommendedDuration": 10,
            "offloadSupport": {
                "default": OFFLOAD_MODE_MODEL_CPU,
                "lowVram": OFFLOAD_MODE_SEQUENTIAL_CPU,
                "emergency": OFFLOAD_MODE_GROUP_DISK,
                "modes": list(_DIRECT_OFFLOAD_MODES),
            },
            "lowVram": {
                "dtype": "float16",
                "autoOffload": True,
                "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
                "steps": 200,
            },
            "modes": ["text_to_audio"],
            "executionStatus": "expert_only",
            "revisionCandidates": [require_catalog_revision(AUDIO_LDM2_REPO)],
            "autoEligible": False,
            "templateEligible": True,
            "galleryEligible": False,
            "notes": [
                "The immutable official base snapshot has safetensors equivalents for every weighted component; loading rejects all co-published legacy pickle files.",
                "The selected safe envelope is approximately 4.48 GB before cache overhead and follows the reviewed ten-second, 200-step, guidance-3.5, three-waveform quality recipe at 16 kHz.",
                "CC-BY-NC-SA-4.0 applies; Auto and Gallery remain disabled pending remote runtime and output-quality review.",
            ],
        },
        "roles": _AUDIO_GRAPH_ROLES,
        "edges": _AUDIO_GRAPH_EDGES,
        "bindings": _AUDIO_LDM2_GRAPH_BINDINGS,
    },
    "shap-e:text-to-3d:v1": {
        "modelType": "ShapEPipeline",
        "mode": "text_to_3d",
        "profile": {
            "id": "shap-e:direct",
            "model_type": "ShapEPipeline",
            "modes": ("text_to_3d",),
            "loader_module": "modules.DiffusersThreeD",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-three-d",
            "pipeline_class": "ShapEPipeline",
            "default_repo": SHAP_E_REPO,
            "fallback_repo": None,
            "quantizable_components": (),
            "default_quantized_components": (),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "max_low_memory_side": 256,
            "max_low_memory_steps": 64,
            "live_proof": False,
            "compatible_repos": (),
        },
        "capability": {
            "modelType": "ShapEPipeline",
            "label": "Shap-E Rendered 3D",
            "displayName": "shap-e",
            "family": "Shap-E",
            "supportTier": "supported",
            "qualificationStatus": "graph-qualified-execution-pending",
            "qualifiedModes": [],
            "defaultRepo": SHAP_E_REPO,
            "artifactLabel": "Explicit safe-component Diffusers assembly",
            "defaultDtype": "float16",
            "defaultSize": {"width": 256, "height": 256, "aspectRatio": "1:1"},
            "recommendedSteps": 64,
            "recommendedGuidance": 15.0,
            "guidanceLabel": "Guidance",
            "supportsImageInput": False,
            "supportsMask": False,
            "supportsMultiImage": False,
            "supportsControlImage": False,
            "supportsLayers": False,
            "supportsLora": False,
            "supportsVideoInput": False,
            "supportsAudioInput": False,
            "outputKind": "video",
            "recommendedFrames": 20,
            "recommendedFps": 12,
            "offloadSupport": {
                "default": OFFLOAD_MODE_SEQUENTIAL_CPU,
                "lowVram": OFFLOAD_MODE_SEQUENTIAL_CPU,
                "emergency": OFFLOAD_MODE_GROUP_DISK,
                "modes": list(_DIRECT_OFFLOAD_MODES),
            },
            "lowVram": {
                "dtype": "float16",
                "autoOffload": True,
                "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
                "steps": 64,
                "width": 256,
                "height": 256,
                "numFrames": 20,
            },
            "modes": ["text_to_3d"],
            "executionStatus": "expert_only",
            "revisionCandidates": [require_catalog_revision(SHAP_E_REPO)],
            "autoEligible": False,
            "templateEligible": True,
            "galleryEligible": False,
            "notes": [
                "The generic contract returns a bounded rendered orbit; it does not expose latent or mesh serialization.",
                "The loader explicitly assembles only the official fp16 safetensors prior, CLIP encoder, and pre-rename renderer component, avoiding the co-published legacy renderer pickle.",
                "MIT applies; Auto and Gallery remain disabled pending remote runtime, output-quality, and export review.",
            ],
        },
        "roles": _THREE_D_GRAPH_ROLES,
        "edges": _THREE_D_GRAPH_EDGES,
        "bindings": _THREE_D_GRAPH_BINDINGS,
    },
    "flux-dev:edit-image:v1": {
        "modelType": "FluxDevPipeline",
        "mode": "edit_image",
        "profile": _profile(
            "flux-dev:img2img-direct",
            "FluxDevPipeline",
            FLUX_DEV_REPO,
            mode="edit_image",
            pipeline_class="FluxImg2ImgPipeline",
            default_quantized_components=("transformer",),
            supported_offload_modes=(
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            retry_offload_modes=(OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK),
            max_low_memory_side=768,
            max_low_memory_steps=20,
            compatible_repos=(FLUX_DEV_FP8_REPO,),
        ),
        "roles": _EDIT_GRAPH_ROLES,
        "edges": _EDIT_GRAPH_EDGES,
        "bindings": _EDIT_GRAPH_BINDINGS
        + (("diffusersImagePipeline", "revision", "defaultRevision"),),
    },
    "flux-dev:inpaint:v1": {
        "modelType": "FluxDevPipeline",
        "mode": "inpaint",
        "profile": _profile(
            "flux-dev:inpaint-direct",
            "FluxDevPipeline",
            FLUX_DEV_REPO,
            mode="inpaint",
            pipeline_class="FluxInpaintPipeline",
            default_quantized_components=("transformer",),
            supported_offload_modes=(
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            retry_offload_modes=(OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK),
            max_low_memory_side=768,
            max_low_memory_steps=20,
            compatible_repos=(FLUX_DEV_FP8_REPO,),
        ),
        "roles": _INPAINT_GRAPH_ROLES,
        "edges": _INPAINT_GRAPH_EDGES,
        "bindings": _INPAINT_GRAPH_BINDINGS
        + (("diffusersImagePipeline", "revision", "defaultRevision"),),
    },
    "sdxl-base:text-to-image:v1": {
        "modelType": "StableDiffusionXLPipeline",
        "mode": "text_to_image",
        "profile": {
            "id": "sdxl-base:direct",
            "model_type": "StableDiffusionXLPipeline",
            "modes": ("text_to_image",),
            "loader_module": "modules.DiffusersImage",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-image",
            "pipeline_class": "StableDiffusionXLPipeline",
            "default_repo": SDXL_BASE_REPO,
            "fallback_repo": None,
            "quantizable_components": ("unet", "text_encoder", "text_encoder_2"),
            "default_quantized_components": (),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "max_low_memory_side": 768,
            "max_low_memory_steps": 30,
            "live_proof": False,
            "compatible_repos": (),
        },
        "bindings": _SDXL_GRAPH_BINDINGS,
        "capability": {
            "modelType": "StableDiffusionXLPipeline",
            "label": "Stable Diffusion XL 1.0",
            "displayName": "stable-diffusion-xl-base-1.0",
            "family": "Stable Diffusion XL",
            "defaultRepo": SDXL_BASE_REPO,
            "artifactLabel": "Diffusers repo",
            "defaultDtype": "bfloat16",
            "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
            "recommendedSteps": 30,
            "recommendedGuidance": 5.0,
            "guidanceLabel": "Guidance",
            "supportsImageInput": True,
            "supportsMask": True,
            "supportsMultiImage": False,
            "supportsControlImage": False,
            "supportsLayers": False,
            "supportsLora": True,
            "offloadSupport": {
                "default": OFFLOAD_MODE_MODEL_CPU,
                "lowVram": OFFLOAD_MODE_SEQUENTIAL_CPU,
                "emergency": OFFLOAD_MODE_GROUP_DISK,
                "modes": list(_DIRECT_OFFLOAD_MODES),
            },
            "lowVram": {
                "dtype": "bfloat16",
                "autoOffload": True,
                "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
                "steps": 24,
                "width": 768,
                "height": 768,
            },
            "modes": ["text_to_image", "edit_image", "inpaint"],
            "modeRequirements": {
                "edit_image": {
                    "requiredImages": ["referenceImages"],
                    "note": "Requires one source image for image-to-image transformation.",
                },
                "inpaint": {
                    "requiredImages": ["referenceImages", "maskImage"],
                    "note": "Requires one source image and one mask image.",
                },
            },
            "executionStatus": "expert_only",
            "qualificationStatus": "graph-qualified-execution-pending",
            "revisionCandidates": [require_catalog_revision(SDXL_BASE_REPO)],
            "autoEligible": False,
            "templateEligible": True,
            "galleryEligible": False,
        },
    },
    "sdxl-base:edit-image:v1": {
        "modelType": "StableDiffusionXLPipeline",
        "mode": "edit_image",
        "profile": {
            "id": "sdxl-base:img2img-direct",
            "model_type": "StableDiffusionXLPipeline",
            "modes": ("edit_image",),
            "loader_module": "modules.DiffusersImage",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-image",
            "pipeline_class": "StableDiffusionXLImg2ImgPipeline",
            "default_repo": SDXL_BASE_REPO,
            "fallback_repo": None,
            "quantizable_components": ("unet", "text_encoder", "text_encoder_2"),
            "default_quantized_components": (),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "max_low_memory_side": 768,
            "max_low_memory_steps": 30,
            "live_proof": False,
            "compatible_repos": (),
        },
        "roles": _EDIT_GRAPH_ROLES,
        "edges": _EDIT_GRAPH_EDGES,
        "bindings": _SDXL_EDIT_GRAPH_BINDINGS,
    },
    "sdxl-base:inpaint:v1": {
        "modelType": "StableDiffusionXLPipeline",
        "mode": "inpaint",
        "profile": {
            "id": "sdxl-base:inpaint-direct",
            "model_type": "StableDiffusionXLPipeline",
            "modes": ("inpaint",),
            "loader_module": "modules.DiffusersImage",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-image",
            "pipeline_class": "StableDiffusionXLInpaintPipeline",
            "default_repo": SDXL_BASE_REPO,
            "fallback_repo": None,
            "quantizable_components": ("unet", "text_encoder", "text_encoder_2"),
            "default_quantized_components": (),
            "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
            "retry_offload_modes": (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "max_low_memory_side": 768,
            "max_low_memory_steps": 30,
            "live_proof": False,
            "compatible_repos": (),
        },
        "roles": _INPAINT_GRAPH_ROLES,
        "edges": _INPAINT_GRAPH_EDGES,
        "bindings": _SDXL_INPAINT_GRAPH_BINDINGS,
    },
}


def _planning_video_profile(
    profile_id: str,
    model_type: str,
    modes: tuple[str, ...],
    pipeline_class: str,
    repo: str,
    *,
    loader_module: str = "modules.DiffusersVideo",
    loader_action: str = "LoadPipeline",
    execution_path: str = "direct-diffusers-video",
) -> dict[str, Any]:
    return {
        "id": profile_id,
        "model_type": model_type,
        "modes": modes,
        "loader_module": loader_module,
        "loader_action": loader_action,
        "execution_path": execution_path,
        "pipeline_class": pipeline_class,
        "default_repo": repo,
        "fallback_repo": None,
        "quantizable_components": (),
        "default_quantized_components": (),
        "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
        "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_DISK),
        "max_low_memory_side": None,
        "max_low_memory_steps": 50,
        "live_proof": False,
        "compatible_repos": (),
    }


def _planning_video_capability(
    model_type: str,
    label: str,
    family: str,
    repo: str,
    modes: tuple[str, ...],
    input_contracts: dict[str, dict[str, list[str]]] | None = None,
) -> dict[str, Any]:
    return {
        "modelType": model_type,
        "label": label,
        "displayName": label,
        "family": family,
        "supportTier": "supported",
        "qualificationStatus": "graph-qualified-execution-pending",
        "qualifiedModes": [],
        "defaultRepo": repo,
        "artifactLabel": "Diffusers video repo",
        "defaultDtype": "bfloat16",
        "defaultSize": {"width": 768, "height": 512, "aspectRatio": "custom"},
        "recommendedSteps": 30,
        "recommendedGuidance": 3.0,
        "guidanceLabel": "Guidance",
        "supportsImageInput": any(
            contract.get("requiredImages") for contract in (input_contracts or {}).values()
        ),
        "supportsMask": False,
        "supportsMultiImage": False,
        "supportsControlImage": False,
        "supportsLayers": False,
        "supportsLora": False,
        "supportsVideoInput": any(
            contract.get("requiredVideos") for contract in (input_contracts or {}).values()
        ),
        "supportsVideoMask": False,
        "outputKind": "video",
        "recommendedFrames": 81,
        "recommendedFps": 24,
        "offloadSupport": {
            "default": OFFLOAD_MODE_MODEL_CPU,
            "lowVram": OFFLOAD_MODE_MODEL_CPU,
            "emergency": OFFLOAD_MODE_GROUP_DISK,
            "modes": list(_DIRECT_OFFLOAD_MODES),
        },
        "lowVram": {
            "dtype": "bfloat16",
            "autoOffload": True,
            "offloadMode": OFFLOAD_MODE_MODEL_CPU,
            "steps": 20,
            "width": 768,
            "height": 512,
            "numFrames": 49,
        },
        "modes": list(modes),
        "executionStatus": "expert_only",
        "revisionCandidates": [require_catalog_revision(repo)],
        "modeRequirements": input_contracts or {},
        "notes": [
            "This generic planning graph is Expert-only pending remote video execution qualification.",
            "Auto and Gallery publication remain disabled until an exact live receipt is reviewed.",
        ],
    }


def _animatediff_capability(model_type: str, *, lcm: bool) -> dict[str, Any]:
    requirements = studio_model_requirements_for_pair(model_type, "text_to_video")
    return {
        "modelType": model_type,
        "label": "AnimateLCM SD1.5" if lcm else "AnimateDiff SD1.5 v2",
        "displayName": "AnimateLCM" if lcm else "AnimateDiff motion adapter v1.5.2",
        "family": "AnimateDiff",
        "supportTier": "supported",
        "qualificationStatus": "graph-qualified-execution-pending",
        "qualifiedModes": [],
        "defaultRepo": SD15_BASE_REPO,
        "artifactLabel": "SD1.5 safetensors base plus pinned fp16 safetensors motion adapter",
        "defaultDtype": "float16",
        "defaultSize": {"width": 512, "height": 512, "aspectRatio": "1:1"},
        "recommendedSteps": 6 if lcm else 25,
        "recommendedGuidance": 1.5 if lcm else 7.5,
        "guidanceLabel": "Guidance",
        "supportsImageInput": False,
        "supportsMask": False,
        "supportsMultiImage": False,
        "supportsControlImage": False,
        "supportsLayers": False,
        "supportsLora": False,
        "supportsVideoInput": False,
        "supportsVideoMask": False,
        "outputKind": "video",
        "recommendedFrames": 16,
        "recommendedFps": 8,
        "offloadSupport": {
            "default": OFFLOAD_MODE_MODEL_CPU,
            "lowVram": OFFLOAD_MODE_MODEL_CPU,
            "emergency": OFFLOAD_MODE_GROUP_DISK,
            "modes": list(_DIRECT_OFFLOAD_MODES),
        },
        "lowVram": {
            "dtype": "float16",
            "autoOffload": True,
            "offloadMode": OFFLOAD_MODE_MODEL_CPU,
            "steps": 4 if lcm else 16,
            "width": 512,
            "height": 512,
            "numFrames": 8,
        },
        "modes": ["text_to_video"],
        "modeRequirements": {
            "text_to_video": {
                "modelRequirements": requirements,
                "note": (
                    "Uses the exact AnimateLCM motion module, linear-beta LCM scheduler, and named spatial LoRA."
                    if lcm
                    else "Uses the exact AnimateDiff SD1.5 v2 motion module and documented linear-beta DDIM scheduler."
                ),
            }
        },
        "executionStatus": "expert_only",
        "revisionCandidates": [require_catalog_revision(SD15_BASE_REPO, model_type="StableDiffusionPipeline")],
        "autoEligible": False,
        "templateEligible": True,
        "galleryEligible": False,
        "notes": [
            "The SD1.5 base and motion component revisions are pinned independently and load only safetensors weights.",
            "The motion repository does not declare a weight license; users must establish authorization before use.",
            "Auto and Gallery publication remain disabled until exact remote runtime, quality, and rights proof is reviewed.",
        ],
    }


def _cogvideox_capability() -> dict[str, Any]:
    return {
        "modelType": "CogVideoXPipeline",
        "label": "CogVideoX-2B",
        "displayName": "CogVideoX-2B",
        "family": "CogVideoX",
        "supportTier": "supported",
        "qualificationStatus": "graph-qualified-execution-pending",
        "qualifiedModes": [],
        "defaultRepo": COGVIDEOX_2B_REPO,
        "artifactLabel": "Official Apache-2.0 safetensors Diffusers repo",
        "defaultDtype": "float16",
        "defaultSize": {"width": 720, "height": 480, "aspectRatio": "custom"},
        "recommendedSteps": 25,
        "recommendedGuidance": 6.0,
        "recommendedMaxSequenceLength": 226,
        "guidanceLabel": "Guidance",
        "supportsImageInput": False,
        "supportsMask": False,
        "supportsMultiImage": False,
        "supportsControlImage": False,
        "supportsLayers": False,
        "supportsLora": False,
        "supportsVideoInput": False,
        "supportsVideoMask": False,
        "outputKind": "video",
        "recommendedFrames": 25,
        "recommendedFps": 8,
        "offloadSupport": {
            "default": OFFLOAD_MODE_MODEL_CPU,
            "lowVram": OFFLOAD_MODE_MODEL_CPU,
            "emergency": OFFLOAD_MODE_GROUP_DISK,
            "modes": list(_DIRECT_OFFLOAD_MODES),
        },
        "lowVram": {
            "dtype": "float16",
            "autoOffload": True,
            "offloadMode": OFFLOAD_MODE_MODEL_CPU,
            "steps": 16,
            "width": 720,
            "height": 480,
            "numFrames": 9,
        },
        "modes": ["text_to_video"],
        "modeRequirements": {
            "text_to_video": {
                "note": "Uses the exact CogVideoX-2B safetensors snapshot with VAE tiling and model CPU offload."
            }
        },
        "executionStatus": "expert_only",
        "revisionCandidates": [require_catalog_revision(COGVIDEOX_2B_REPO)],
        "autoEligible": False,
        "templateEligible": True,
        "galleryEligible": False,
        "notes": [
            "The admitted source graph is bounded to 9-25 frames in 4k+1 form at the native 720x480 size.",
            "The publisher's representative quality recipe remains 49 frames and 50 steps; this shorter graph needs remote output review.",
            "Auto and Gallery publication remain disabled until exact remote runtime and quality proof is reviewed.",
        ],
    }


_WAN_ANIMATE_MODES = ("character_animate", "character_replace")
_LTX2_MODES = ("text_to_video", "image_to_video", "video_to_video", "reference_to_video")
_P2_VIDEO_PROFILES = {
    "wan22": _planning_video_profile(
        "wan-22-a14b:direct", "Wan22Pipeline", ("text_to_video",), "Wan22Pipeline", WAN_22_T2V_A14B_REPO
    ),
    "animate": _planning_video_profile(
        "wan-animate:direct", "WanAnimatePipeline", _WAN_ANIMATE_MODES, "WanAnimatePipeline", WAN_ANIMATE_REPO
    ),
    "ltx-long": _planning_video_profile(
        "ltx-long:direct",
        "LTXI2VLongMultiPromptPipeline",
        ("image_to_video",),
        "LTXI2VLongMultiPromptPipeline",
        LTX_VIDEO_REPO,
    ),
    "ltx2": _planning_video_profile(
        "ltx2:direct", "LTX2ConditionPipeline", _LTX2_MODES, "LTX2ConditionPipeline", LTX2_REPO
    ),
    "framepack": _planning_video_profile(
        "framepack:direct",
        "HunyuanVideoFramepackPipeline",
        ("image_to_video",),
        "HunyuanVideoFramepackPipeline",
        FRAMEPACK_REPO,
    ),
    "stable-video-diffusion": _planning_video_profile(
        "stable-video-diffusion:direct",
        "StableVideoDiffusionPipeline",
        ("image_to_video",),
        "StableVideoDiffusionPipeline",
        STABLE_VIDEO_DIFFUSION_REPO,
    ),
    "animatediff": _planning_video_profile(
        "animatediff-sd15-v2:direct",
        "AnimateDiffPipeline",
        ("text_to_video",),
        "AnimateDiffPipeline",
        SD15_BASE_REPO,
    ),
    "animatelcm": _planning_video_profile(
        "animatelcm-sd15:direct",
        "AnimateLCMPipeline",
        ("text_to_video",),
        "AnimateLCMPipeline",
        SD15_BASE_REPO,
    ),
    "cogvideox-2b": _planning_video_profile(
        "cogvideox-2b:direct",
        "CogVideoXPipeline",
        ("text_to_video",),
        "CogVideoXPipeline",
        COGVIDEOX_2B_REPO,
    ),
    "wan-flf": _planning_video_profile(
        "wan-flf:modular",
        "WanImage2VideoModularPipeline",
        ("image_to_video",),
        "WanImage2VideoModularPipeline",
        WAN_FLF_REPO,
        loader_module="modules.ModularDiffusers",
        loader_action="ModelsLoader",
        execution_path="modular-diffusers",
    ),
}
_WAN_ANIMATE_INPUTS = {
    "character_animate": {
        "requiredImages": ["referenceImages"],
        "requiredVideos": ["poseVideo", "faceVideo"],
    },
    "character_replace": {
        "requiredImages": ["referenceImages"],
        "requiredVideos": ["poseVideo", "faceVideo", "backgroundVideo", "maskVideo"],
    },
}
_LTX2_INPUTS = {
    "image_to_video": {"requiredImages": ["referenceImages"]},
    "reference_to_video": {"requiredImages": ["referenceImages"]},
    "video_to_video": {"requiredVideos": ["sourceVideo"]},
}
STUDIO_EXECUTION_SPEC_DEFINITIONS.update(
    {
        "wan-22-a14b:text-to-video:v1": {
            "modelType": "Wan22Pipeline",
            "mode": "text_to_video",
            "profile": _P2_VIDEO_PROFILES["wan22"],
            "capability": _planning_video_capability(
                "Wan22Pipeline", "Wan 2.2 T2V A14B", "Wan Video", WAN_22_T2V_A14B_REPO, ("text_to_video",)
            ),
            "roles": _VIDEO_GRAPH_ROLES,
            "edges": _VIDEO_GRAPH_EDGES,
            "bindings": _VIDEO_REVISION_GRAPH_BINDINGS,
        },
        "wan-animate:character-animate:v1": {
            "modelType": "WanAnimatePipeline",
            "mode": "character_animate",
            "profile": _P2_VIDEO_PROFILES["animate"],
            "capability": _planning_video_capability(
                "WanAnimatePipeline", "Wan 2.2 Animate", "Wan Video", WAN_ANIMATE_REPO, _WAN_ANIMATE_MODES,
                _WAN_ANIMATE_INPUTS,
            ),
            "roles": _WAN_ANIMATE_GRAPH_ROLES,
            "edges": _WAN_ANIMATE_GRAPH_EDGES,
            "bindings": _WAN_ANIMATE_GRAPH_BINDINGS,
        },
        "wan-animate:character-replace:v1": {
            "modelType": "WanAnimatePipeline",
            "mode": "character_replace",
            "profile": _P2_VIDEO_PROFILES["animate"],
            "roles": _WAN_REPLACE_GRAPH_ROLES,
            "edges": _WAN_REPLACE_GRAPH_EDGES,
            "bindings": _WAN_REPLACE_GRAPH_BINDINGS,
        },
        "ltx-long:image-to-video:v1": {
            "modelType": "LTXI2VLongMultiPromptPipeline",
            "mode": "image_to_video",
            "profile": _P2_VIDEO_PROFILES["ltx-long"],
            "capability": _planning_video_capability(
                "LTXI2VLongMultiPromptPipeline", "LTX long-prompt I2V", "LTX Video", LTX_VIDEO_REPO,
                ("image_to_video",), {"image_to_video": {"requiredImages": ["referenceImages"]}},
            ),
            "roles": _I2V_GRAPH_ROLES,
            "edges": _I2V_GRAPH_EDGES,
            "bindings": _LTX_LONG_GRAPH_BINDINGS,
        },
        "ltx2:text-to-video:v1": {
            "modelType": "LTX2ConditionPipeline",
            "mode": "text_to_video",
            "profile": _P2_VIDEO_PROFILES["ltx2"],
            "capability": _planning_video_capability(
                "LTX2ConditionPipeline", "LTX-2 video and audio", "LTX Video", LTX2_REPO, _LTX2_MODES,
                _LTX2_INPUTS,
            ),
            "roles": _LTX2_GRAPH_ROLES,
            "edges": _LTX2_GRAPH_EDGES,
            "bindings": _LTX2_GRAPH_BINDINGS,
        },
        "ltx2:image-to-video:v1": {
            "modelType": "LTX2ConditionPipeline",
            "mode": "image_to_video",
            "profile": _P2_VIDEO_PROFILES["ltx2"],
            "roles": _LTX2_I2V_GRAPH_ROLES,
            "edges": _LTX2_I2V_GRAPH_EDGES,
            "bindings": _LTX2_I2V_GRAPH_BINDINGS,
        },
        "ltx2:reference-to-video:v1": {
            "modelType": "LTX2ConditionPipeline",
            "mode": "reference_to_video",
            "profile": _P2_VIDEO_PROFILES["ltx2"],
            "roles": _LTX2_I2V_GRAPH_ROLES,
            "edges": _LTX2_I2V_GRAPH_EDGES,
            "bindings": _LTX2_I2V_GRAPH_BINDINGS,
        },
        "ltx2:video-to-video:v1": {
            "modelType": "LTX2ConditionPipeline",
            "mode": "video_to_video",
            "profile": _P2_VIDEO_PROFILES["ltx2"],
            "roles": _LTX2_V2V_GRAPH_ROLES,
            "edges": _LTX2_V2V_GRAPH_EDGES,
            "bindings": _LTX2_V2V_GRAPH_BINDINGS,
        },
        "framepack:image-to-video:v1": {
            "modelType": "HunyuanVideoFramepackPipeline",
            "mode": "image_to_video",
            "profile": _P2_VIDEO_PROFILES["framepack"],
            "capability": _planning_video_capability(
                "HunyuanVideoFramepackPipeline", "Hunyuan FramePack I2V", "Wan Video", FRAMEPACK_REPO,
                ("image_to_video",), {"image_to_video": {"requiredImages": ["referenceImages"]}},
            ),
            "roles": _FRAMEPACK_GRAPH_ROLES,
            "edges": _FRAMEPACK_GRAPH_EDGES,
            "bindings": _FRAMEPACK_GRAPH_BINDINGS,
        },
        "stable-video-diffusion:image-to-video:v1": {
            "modelType": "StableVideoDiffusionPipeline",
            "mode": "image_to_video",
            "profile": _P2_VIDEO_PROFILES["stable-video-diffusion"],
            "capability": {
                **_planning_video_capability(
                    "StableVideoDiffusionPipeline",
                    "Stable Video Diffusion XT 1.1",
                    "Stable Video Diffusion",
                    STABLE_VIDEO_DIFFUSION_REPO,
                    ("image_to_video",),
                    {"image_to_video": {"requiredImages": ["referenceImages"]}},
                ),
                "defaultDtype": "float16",
                "defaultSize": {"width": 1024, "height": 576, "aspectRatio": "16:9"},
                "recommendedSteps": 25,
                "recommendedGuidance": 3.0,
                "recommendedFrames": 25,
                "recommendedFps": 7,
                "autoEligible": False,
                "galleryEligible": False,
                "lowVram": {
                    "dtype": "float16",
                    "autoOffload": True,
                    "offloadMode": OFFLOAD_MODE_MODEL_CPU,
                    "steps": 25,
                    "width": 1024,
                    "height": 576,
                    "numFrames": 8,
                },
                "notes": [
                    "This exact gated source requires prior acceptance of the publisher's Hub access terms.",
                    "The reviewed recipe uses safetensors-only fp16 loading, model CPU offload, UNet forward chunking, and decode chunks of two.",
                    "Auto and Gallery publication remain disabled until an exact live receipt is reviewed.",
                ],
            },
            "roles": _I2V_GRAPH_ROLES,
            "edges": _I2V_GRAPH_EDGES,
            "bindings": _STABLE_VIDEO_DIFFUSION_GRAPH_BINDINGS,
        },
        "animatediff:text-to-video:v1": {
            "modelType": "AnimateDiffPipeline",
            "mode": "text_to_video",
            "profile": _P2_VIDEO_PROFILES["animatediff"],
            "capability": _animatediff_capability("AnimateDiffPipeline", lcm=False),
            "roles": _VIDEO_GRAPH_ROLES,
            "edges": _VIDEO_GRAPH_EDGES,
            "bindings": _ANIMATEDIFF_GRAPH_BINDINGS,
        },
        "animatelcm:text-to-video:v1": {
            "modelType": "AnimateLCMPipeline",
            "mode": "text_to_video",
            "profile": _P2_VIDEO_PROFILES["animatelcm"],
            "capability": _animatediff_capability("AnimateLCMPipeline", lcm=True),
            "roles": _VIDEO_GRAPH_ROLES,
            "edges": _VIDEO_GRAPH_EDGES,
            "bindings": _ANIMATEDIFF_GRAPH_BINDINGS,
        },
        "cogvideox-2b:text-to-video:v1": {
            "modelType": "CogVideoXPipeline",
            "mode": "text_to_video",
            "profile": _P2_VIDEO_PROFILES["cogvideox-2b"],
            "capability": _cogvideox_capability(),
            "roles": _VIDEO_GRAPH_ROLES,
            "edges": _VIDEO_GRAPH_EDGES,
            "bindings": _COGVIDEOX_GRAPH_BINDINGS,
        },
        "wan-flf:image-to-video:v1": {
            "modelType": "WanImage2VideoModularPipeline",
            "mode": "image_to_video",
            "profile": _P2_VIDEO_PROFILES["wan-flf"],
            "capability": _planning_video_capability(
                "WanImage2VideoModularPipeline", "Wan first/last-frame video", "Wan Video", WAN_FLF_REPO,
                ("image_to_video",),
                {"image_to_video": {"requiredImages": ["referenceImages", "lastImage"]}},
            ),
            "roles": _WAN_FLF_GRAPH_ROLES,
            "edges": _WAN_FLF_GRAPH_EDGES,
            "bindings": _WAN_FLF_GRAPH_BINDINGS,
        },
    }
)


def _unconditional_profile(
    profile_id: str,
    pipeline_class: str,
    repo: str,
    *,
    max_steps: int,
) -> dict[str, Any]:
    return {
        "id": profile_id,
        "model_type": pipeline_class,
        "modes": ("unconditional_image",),
        "loader_module": "modules.DiffusersImage",
        "loader_action": "LoadPipeline",
        "execution_path": "direct-diffusers-image",
        "pipeline_class": pipeline_class,
        "default_repo": repo,
        "fallback_repo": None,
        "quantizable_components": (),
        "default_quantized_components": (),
        "supported_offload_modes": (OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU),
        "retry_offload_modes": (OFFLOAD_MODE_NONE,),
        "max_low_memory_side": 64,
        "max_low_memory_steps": max_steps,
        "live_proof": True,
        "compatible_repos": (),
    }


def _unconditional_capability(
    pipeline_class: str,
    label: str,
    family: str,
    repo: str,
    *,
    side: int,
    steps: int,
) -> dict[str, Any]:
    return {
        "modelType": pipeline_class,
        "label": label,
        "displayName": label,
        "family": family,
        "supportTier": "supported",
        "qualificationStatus": "graph-qualified-execution-pending",
        "qualifiedModes": [],
        "defaultRepo": repo,
        "artifactLabel": "Diffusers unconditional image repo",
        "defaultDtype": "float32",
        "defaultSize": {"width": side, "height": side, "aspectRatio": "1:1"},
        "recommendedSteps": steps,
        "recommendedGuidance": 0.0,
        "guidanceLabel": "Not used",
        "supportsImageInput": False,
        "supportsMask": False,
        "supportsMultiImage": False,
        "supportsControlImage": False,
        "supportsLayers": False,
        "supportsLora": False,
        "outputKind": "image",
        "offloadSupport": {
            "default": OFFLOAD_MODE_NONE,
            "lowVram": OFFLOAD_MODE_NONE,
            "emergency": OFFLOAD_MODE_MODEL_CPU,
            "modes": [OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU],
        },
        "lowVram": {
            "dtype": "float32",
            "autoOffload": False,
            "offloadMode": OFFLOAD_MODE_NONE,
            "steps": steps,
            "width": side,
            "height": side,
        },
        "modes": ["unconditional_image"],
        "executionStatus": "expert_only",
        "revisionCandidates": [require_catalog_revision(repo, model_type=pipeline_class)],
        "autoEligible": False,
        "templateEligible": True,
        "galleryEligible": False,
        "notes": [
            "This model uses the generic unconditional image loader and sampler nodes.",
            "Auto and Gallery remain disabled until exact live output qualification is reviewed.",
        ],
    }


_P3_UNCONDITIONAL_DEFINITIONS = (
    (
        "ddpm-cifar10:unconditional-image:v1",
        "ddpm-cifar10:direct",
        "DDPMPipeline",
        "DDPM CIFAR-10 32x32",
        "DDPM",
        DDPM_CIFAR10_REPO,
        32,
        1000,
    ),
    (
        "ddim-cifar10:unconditional-image:v1",
        "ddim-cifar10:direct",
        "DDIMPipeline",
        "DDIM CIFAR-10 32x32",
        "DDIM",
        DDPM_CIFAR10_REPO,
        32,
        50,
    ),
    (
        "consistency-imagenet64:unconditional-image:v1",
        "consistency-imagenet64:direct",
        "ConsistencyModelPipeline",
        "Consistency Model ImageNet 64x64",
        "Consistency Models",
        CONSISTENCY_IMAGENET64_REPO,
        64,
        1,
    ),
)
STUDIO_EXECUTION_SPEC_DEFINITIONS.update(
    {
        spec_id: {
            "modelType": pipeline_class,
            "mode": "unconditional_image",
            "profile": _unconditional_profile(profile_id, pipeline_class, repo, max_steps=steps),
            "capability": _unconditional_capability(
                pipeline_class,
                label,
                family,
                repo,
                side=side,
                steps=steps,
            ),
            "roles": _UNCONDITIONAL_GRAPH_ROLES,
            "edges": _UNCONDITIONAL_GRAPH_EDGES,
            "bindings": _UNCONDITIONAL_GRAPH_BINDINGS,
        }
        for spec_id, profile_id, pipeline_class, label, family, repo, side, steps in _P3_UNCONDITIONAL_DEFINITIONS
    }
)


def _sd15_profile(profile_id: str, mode: str, pipeline_class: str) -> dict[str, Any]:
    return {
        "id": profile_id,
        "model_type": "StableDiffusionPipeline",
        "modes": (mode,),
        "loader_module": "modules.DiffusersImage",
        "loader_action": "LoadPipeline",
        "execution_path": "direct-diffusers-image",
        "pipeline_class": pipeline_class,
        "default_repo": SD15_BASE_REPO,
        "fallback_repo": None,
        "quantizable_components": (),
        "default_quantized_components": (),
        "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
        "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
        "max_low_memory_side": 512,
        "max_low_memory_steps": 30,
        "live_proof": True,
        "compatible_repos": (),
    }


_SD15_CAPABILITY = {
    "modelType": "StableDiffusionPipeline",
    "label": "Stable Diffusion 1.5",
    "displayName": "stable-diffusion-v1-5",
    "family": "Stable Diffusion 1.x",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": SD15_BASE_REPO,
    "artifactLabel": "Diffusers safetensors repo",
    "defaultDtype": "float32",
    "defaultSize": {"width": 512, "height": 512, "aspectRatio": "1:1"},
    "recommendedSteps": 30,
    "recommendedGuidance": 7.5,
    "guidanceLabel": "Guidance",
    "supportsImageInput": True,
    "supportsMask": True,
    "supportsMultiImage": False,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": True,
    "outputKind": "image",
    "offloadSupport": {
        "default": OFFLOAD_MODE_NONE,
        "lowVram": OFFLOAD_MODE_MODEL_CPU,
        "emergency": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "modes": list(_DIRECT_OFFLOAD_MODES),
    },
    "lowVram": {
        "dtype": "float32",
        "autoOffload": False,
        "offloadMode": OFFLOAD_MODE_NONE,
        "steps": 20,
        "width": 512,
        "height": 512,
    },
    "modes": ["text_to_image", "edit_image", "inpaint"],
    "modeRequirements": {
        "edit_image": {
            "requiredImages": ["referenceImages"],
            "note": "Requires one source image.",
        },
        "inpaint": {
            "requiredImages": ["referenceImages", "maskImage"],
            "note": "Requires one source image and one mask image.",
        },
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [require_catalog_revision(SD15_BASE_REPO, model_type="StableDiffusionPipeline")],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "All modes reuse the immutable Stable Diffusion 1.5 safetensors base through generic image nodes.",
        "Auto and Gallery remain disabled until exact live output qualification is reviewed.",
    ],
}

_P3_SD15_DEFINITIONS = (
    (
        "sd15-base:text-to-image:v1",
        "sd15-base:direct",
        "text_to_image",
        "StableDiffusionPipeline",
        _GRAPH_ROLES,
        _GRAPH_EDGES,
        _SDXL_GRAPH_BINDINGS,
    ),
    (
        "sd15-base:edit-image:v1",
        "sd15-base:img2img-direct",
        "edit_image",
        "StableDiffusionImg2ImgPipeline",
        _EDIT_GRAPH_ROLES,
        _EDIT_GRAPH_EDGES,
        _SDXL_EDIT_GRAPH_BINDINGS,
    ),
    (
        "sd15-base:inpaint:v1",
        "sd15-base:inpaint-direct",
        "inpaint",
        "StableDiffusionInpaintPipeline",
        _INPAINT_GRAPH_ROLES,
        _INPAINT_GRAPH_EDGES,
        _SDXL_INPAINT_GRAPH_BINDINGS,
    ),
)
STUDIO_EXECUTION_SPEC_DEFINITIONS.update(
    {
        spec_id: {
            "modelType": "StableDiffusionPipeline",
            "mode": mode,
            "profile": _sd15_profile(profile_id, mode, pipeline_class),
            "capability": deepcopy(_SD15_CAPABILITY),
            "roles": roles,
            "edges": edges,
            "bindings": bindings,
        }
        for spec_id, profile_id, mode, pipeline_class, roles, edges, bindings in _P3_SD15_DEFINITIONS
    }
)

_SD15_CONTROLNET_CAPABILITY = deepcopy(_SD15_CAPABILITY)
_SD15_CONTROLNET_CAPABILITY.update(
    {
        "supportsControlImage": True,
        "modes": ["text_to_image", "edit_image", "inpaint", "control_image"],
        "modeRequirements": {
            **_SD15_CONTROLNET_CAPABILITY["modeRequirements"],
            "control_image": {
                "modelRequirements": studio_model_requirements_for_pair(
                    "StableDiffusionPipeline", "control_image"
                ),
                "requiredImages": ["controlImage"],
                "note": "Requires one control image and the immutable Canny ControlNet component.",
            },
        },
        "notes": [
            "Base, img2img, inpaint, and Canny ControlNet reuse generic Diffusers image nodes.",
            "The ControlNet component is pinned independently and requires safetensors during assembly.",
            "Auto and Gallery remain disabled until exact live output qualification is reviewed.",
        ],
    }
)
for _sd15_spec_id, *_unused in _P3_SD15_DEFINITIONS:
    STUDIO_EXECUTION_SPEC_DEFINITIONS[_sd15_spec_id]["capability"] = deepcopy(
        _SD15_CONTROLNET_CAPABILITY
    )
_SD15_CONTROLNET_PROFILE = _sd15_profile(
    "sd15-controlnet-canny:direct",
    "control_image",
    "StableDiffusionControlNetPipeline",
)
_SD15_CONTROLNET_PROFILE["live_proof"] = False
STUDIO_EXECUTION_SPEC_DEFINITIONS["sd15-controlnet-canny:control-image:v1"] = {
    "modelType": "StableDiffusionPipeline",
    "mode": "control_image",
    "profile": _SD15_CONTROLNET_PROFILE,
    "capability": _SD15_CONTROLNET_CAPABILITY,
    "roles": _CONDITIONED_CONTROL_GRAPH_ROLES,
    "edges": _CONDITIONED_CONTROL_GRAPH_EDGES,
    "bindings": _CONDITIONED_CONTROL_GRAPH_BINDINGS,
}


_SDXL_TURBO_PROFILE = {
    "id": "sdxl-turbo:direct",
    "model_type": "StableDiffusionXLTurboPipeline",
    "modes": ("text_to_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "StableDiffusionXLTurboPipeline",
    "default_repo": SDXL_TURBO_REPO,
    "fallback_repo": None,
    "quantizable_components": ("unet", "text_encoder", "text_encoder_2"),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 512,
    "max_low_memory_steps": 4,
    "live_proof": False,
    "compatible_repos": (),
}
_SDXL_TURBO_CAPABILITY = {
    "modelType": "StableDiffusionXLTurboPipeline",
    "label": "Stable Diffusion XL Turbo",
    "displayName": "SDXL Turbo",
    "family": "Stable Diffusion XL",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": SDXL_TURBO_REPO,
    "artifactLabel": "Diffusers safetensors repo",
    "defaultDtype": "float16",
    "defaultSize": {"width": 512, "height": 512, "aspectRatio": "1:1"},
    "recommendedSteps": 1,
    "recommendedGuidance": 0.0,
    "guidanceLabel": "Guidance disabled",
    "supportsImageInput": False,
    "supportsMask": False,
    "supportsMultiImage": False,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": False,
    "outputKind": "image",
    "offloadSupport": {
        "default": OFFLOAD_MODE_MODEL_CPU,
        "lowVram": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "emergency": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "modes": list(_DIRECT_OFFLOAD_MODES),
    },
    "lowVram": {
        "dtype": "float16",
        "autoOffload": True,
        "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "steps": 1,
        "width": 512,
        "height": 512,
    },
    "modes": ["text_to_image"],
    "modeRequirements": {},
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(SDXL_TURBO_REPO, model_type="StableDiffusionXLTurboPipeline")
    ],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The reviewed fp16 safetensors variant runs at 512px in one to four steps with guidance fixed to zero.",
        "Auto and Gallery remain disabled until exact live output and license-surface review are complete.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["sdxl-turbo:text-to-image:v1"] = {
    "modelType": "StableDiffusionXLTurboPipeline",
    "mode": "text_to_image",
    "profile": _SDXL_TURBO_PROFILE,
    "capability": _SDXL_TURBO_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _SDXL_GRAPH_BINDINGS,
}


_SDXL_INSTRUCT_PROFILE = {
    "id": "sdxl-instruct-pix2pix:direct",
    "model_type": "StableDiffusionXLInstructPix2PixPipeline",
    "modes": ("edit_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "StableDiffusionXLInstructPix2PixPipeline",
    "default_repo": SDXL_INSTRUCT_PIX2PIX_REPO,
    "fallback_repo": None,
    "quantizable_components": ("unet", "text_encoder", "text_encoder_2"),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 768,
    "max_low_memory_steps": 30,
    "live_proof": False,
    "compatible_repos": (),
}
_SDXL_INSTRUCT_CAPABILITY = {
    "modelType": "StableDiffusionXLInstructPix2PixPipeline",
    "label": "Stable Diffusion XL InstructPix2Pix",
    "displayName": "SDXL InstructPix2Pix 768",
    "family": "Stable Diffusion XL",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": SDXL_INSTRUCT_PIX2PIX_REPO,
    "artifactLabel": "Diffusers safetensors repo",
    "defaultDtype": "float16",
    "defaultSize": {"width": 768, "height": 768, "aspectRatio": "1:1"},
    "recommendedSteps": 30,
    "recommendedGuidance": 3.0,
    "guidanceLabel": "Text guidance",
    "conditioningScale": 1.5,
    "supportsImageInput": True,
    "supportsMask": False,
    "supportsMultiImage": False,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": False,
    "outputKind": "image",
    "offloadSupport": {
        "default": OFFLOAD_MODE_MODEL_CPU,
        "lowVram": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "emergency": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "modes": list(_DIRECT_OFFLOAD_MODES),
    },
    "lowVram": {
        "dtype": "float16",
        "autoOffload": True,
        "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "steps": 30,
        "width": 768,
        "height": 768,
    },
    "modes": ["edit_image"],
    "modeRequirements": {
        "edit_image": {
            "requiredImages": ["referenceImages"],
            "note": "Requires one source image and a text edit instruction.",
        }
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(
            SDXL_INSTRUCT_PIX2PIX_REPO,
            model_type="StableDiffusionXLInstructPix2PixPipeline",
        )
    ],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The reviewed experimental checkpoint uses a 768px, 30-step recipe with text guidance 3 and image guidance 1.5.",
        "Auto and Gallery remain disabled until exact live output review is complete.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["sdxl-instruct-pix2pix:edit-image:v1"] = {
    "modelType": "StableDiffusionXLInstructPix2PixPipeline",
    "mode": "edit_image",
    "profile": _SDXL_INSTRUCT_PROFILE,
    "capability": _SDXL_INSTRUCT_CAPABILITY,
    "roles": _EDIT_GRAPH_ROLES,
    "edges": _EDIT_GRAPH_EDGES,
    "bindings": _SDXL_INSTRUCT_EDIT_GRAPH_BINDINGS,
}


_SDXL_CONTROLNET_CAPABILITY = {
    "modelType": "StableDiffusionXLControlNetPipeline",
    "label": "Stable Diffusion XL ControlNet",
    "displayName": "SDXL ControlNet Canny",
    "family": "Stable Diffusion XL",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": SDXL_BASE_REPO,
    "artifactLabel": "Diffusers fp16 safetensors assembly",
    "defaultDtype": "float16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 50,
    "recommendedGuidance": 5.0,
    "guidanceLabel": "Guidance",
    "conditioningScale": 0.5,
    "supportsImageInput": False,
    "supportsMask": False,
    "supportsMultiImage": False,
    "supportsControlImage": True,
    "supportsLayers": False,
    "supportsLora": False,
    "outputKind": "image",
    "offloadSupport": {
        "default": OFFLOAD_MODE_MODEL_CPU,
        "lowVram": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "emergency": OFFLOAD_MODE_GROUP_DISK,
        "modes": list(_DIRECT_OFFLOAD_MODES),
    },
    "lowVram": {
        "dtype": "float16",
        "autoOffload": True,
        "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "steps": 50,
        "width": 1024,
        "height": 1024,
    },
    "modes": ["control_image"],
    "modeRequirements": {
        "control_image": {
            "modelRequirements": studio_model_requirements_for_pair(
                "StableDiffusionXLControlNetPipeline", "control_image"
            ),
            "requiredImages": ["controlImage"],
            "note": "Requires one control image and the immutable SDXL Canny ControlNet component.",
        },
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(SDXL_BASE_REPO, model_type="StableDiffusionXLControlNetPipeline")
    ],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The Canny ControlNet component is pinned independently and loaded from its reviewed fp16 safetensors variant.",
        "The generic Canny preprocessor uses the same exact 0.1/0.2 threshold contract as the SD1.5 workflow.",
        "Auto and Gallery remain disabled until exact live output qualification is reviewed.",
    ],
}
_SDXL_CONTROLNET_PROFILE = {
    "id": "sdxl-controlnet-canny:direct",
    "model_type": "StableDiffusionXLControlNetPipeline",
    "modes": ("control_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "StableDiffusionXLControlNetPipeline",
    "default_repo": SDXL_BASE_REPO,
    "fallback_repo": None,
    "quantizable_components": ("unet", "text_encoder", "text_encoder_2"),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 1024,
    "max_low_memory_steps": 50,
    "live_proof": False,
    "compatible_repos": (),
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["sdxl-controlnet-canny:control-image:v1"] = {
    "modelType": "StableDiffusionXLControlNetPipeline",
    "mode": "control_image",
    "profile": _SDXL_CONTROLNET_PROFILE,
    "capability": _SDXL_CONTROLNET_CAPABILITY,
    "roles": _CONDITIONED_CONTROL_GRAPH_ROLES,
    "edges": _CONDITIONED_CONTROL_GRAPH_EDGES,
    "bindings": _CONDITIONED_CONTROL_GRAPH_BINDINGS,
}


_SDXL_T2I_ADAPTER_CAPABILITY = {
    "modelType": "StableDiffusionXLAdapterPipeline",
    "label": "Stable Diffusion XL T2I Adapter",
    "displayName": "SDXL T2I-Adapter Canny",
    "family": "Stable Diffusion XL",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": SDXL_BASE_REPO,
    "artifactLabel": "Diffusers fp16 safetensors assembly",
    "defaultDtype": "float16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 30,
    "recommendedGuidance": 7.5,
    "guidanceLabel": "Guidance",
    "conditioningScale": 0.8,
    "supportsImageInput": False,
    "supportsMask": False,
    "supportsMultiImage": False,
    "supportsControlImage": True,
    "supportsLayers": False,
    "supportsLora": False,
    "outputKind": "image",
    "offloadSupport": {
        "default": OFFLOAD_MODE_MODEL_CPU,
        "lowVram": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "emergency": OFFLOAD_MODE_GROUP_DISK,
        "modes": list(_DIRECT_OFFLOAD_MODES),
    },
    "lowVram": {
        "dtype": "float16",
        "autoOffload": True,
        "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "steps": 30,
        "width": 1024,
        "height": 1024,
    },
    "modes": ["control_image"],
    "modeRequirements": {
        "control_image": {
            "modelRequirements": studio_model_requirements_for_pair(
                "StableDiffusionXLAdapterPipeline", "control_image"
            ),
            "requiredImages": ["controlImage"],
            "note": "Requires one control image and the immutable SDXL Canny T2I-Adapter component.",
        },
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(SDXL_BASE_REPO, model_type="StableDiffusionXLAdapterPipeline")
    ],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The Canny T2I-Adapter component is pinned independently and loaded from its reviewed fp16 safetensors variant.",
        "The generic Canny preprocessor uses exact 0.1/0.2 thresholds before the documented 30-step, guidance-7.5, scale-0.8 recipe.",
        "Auto and Gallery remain disabled until exact live output qualification is reviewed.",
    ],
}
_SDXL_T2I_ADAPTER_PROFILE = {
    "id": "sdxl-t2i-adapter-canny:direct",
    "model_type": "StableDiffusionXLAdapterPipeline",
    "modes": ("control_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "StableDiffusionXLAdapterPipeline",
    "default_repo": SDXL_BASE_REPO,
    "fallback_repo": None,
    "quantizable_components": ("unet", "text_encoder", "text_encoder_2"),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 1024,
    "max_low_memory_steps": 30,
    "live_proof": False,
    "compatible_repos": (),
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["sdxl-t2i-adapter-canny:control-image:v1"] = {
    "modelType": "StableDiffusionXLAdapterPipeline",
    "mode": "control_image",
    "profile": _SDXL_T2I_ADAPTER_PROFILE,
    "capability": _SDXL_T2I_ADAPTER_CAPABILITY,
    "roles": _CONDITIONED_CONTROL_GRAPH_ROLES,
    "edges": _CONDITIONED_CONTROL_GRAPH_EDGES,
    "bindings": _CONDITIONED_CONTROL_GRAPH_BINDINGS,
}


_SDXL_PAG_PROFILE = {
    "id": "sdxl-pag:direct",
    "model_type": "StableDiffusionXLPAGPipeline",
    "modes": ("text_to_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "StableDiffusionXLPAGPipeline",
    "default_repo": SDXL_BASE_REPO,
    "fallback_repo": None,
    "quantizable_components": ("unet", "text_encoder", "text_encoder_2"),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 1024,
    "max_low_memory_steps": 50,
    "live_proof": False,
    "compatible_repos": (),
}
_SDXL_PAG_CAPABILITY = {
    "modelType": "StableDiffusionXLPAGPipeline",
    "label": "Stable Diffusion XL PAG",
    "displayName": "SDXL Perturbed-Attention Guidance",
    "family": "Stable Diffusion XL",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": SDXL_BASE_REPO,
    "artifactLabel": "Diffusers fp16 safetensors repo",
    "defaultDtype": "float16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 50,
    "recommendedGuidance": 5.0,
    "recommendedPagScale": 3.0,
    "recommendedPagAdaptiveScale": 0.0,
    "guidanceLabel": "Guidance",
    "supportsImageInput": True,
    "supportsMask": True,
    "supportsMultiImage": False,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": True,
    "outputKind": "image",
    "offloadSupport": {
        "default": OFFLOAD_MODE_MODEL_CPU,
        "lowVram": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "emergency": OFFLOAD_MODE_GROUP_DISK,
        "modes": list(_DIRECT_OFFLOAD_MODES),
    },
    "lowVram": {
        "dtype": "float16",
        "autoOffload": True,
        "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "steps": 50,
        "width": 1024,
        "height": 1024,
    },
    "modes": ["text_to_image", "edit_image", "inpaint"],
    "modeRequirements": {
        "edit_image": {
            "requiredImages": ["referenceImages"],
            "note": "Requires one source image for PAG image-to-image transformation.",
        },
        "inpaint": {
            "requiredImages": ["referenceImages", "maskImage"],
            "note": "Requires one source image and one mask image for PAG inpainting.",
        },
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(SDXL_BASE_REPO, model_type="StableDiffusionXLPAGPipeline")
    ],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "Perturbed-attention guidance reuses the immutable SDXL base without an auxiliary model artifact.",
        "The generic graph exposes PAG scale 3 and adaptive scale 0 over the upstream 1024px, 50-step, guidance-5 recipe.",
        "Auto and Gallery remain disabled until exact live output qualification is reviewed.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["sdxl-pag:text-to-image:v1"] = {
    "modelType": "StableDiffusionXLPAGPipeline",
    "mode": "text_to_image",
    "profile": _SDXL_PAG_PROFILE,
    "capability": _SDXL_PAG_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _PAG_GRAPH_BINDINGS,
}
_SDXL_PAG_IMG2IMG_PROFILE = {
    **_SDXL_PAG_PROFILE,
    "id": "sdxl-pag:img2img-direct",
    "modes": ("edit_image",),
    "pipeline_class": "StableDiffusionXLPAGImg2ImgPipeline",
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["sdxl-pag:edit-image:v1"] = {
    "modelType": "StableDiffusionXLPAGPipeline",
    "mode": "edit_image",
    "profile": _SDXL_PAG_IMG2IMG_PROFILE,
    "capability": _SDXL_PAG_CAPABILITY,
    "roles": _EDIT_GRAPH_ROLES,
    "edges": _EDIT_GRAPH_EDGES,
    "bindings": _PAG_EDIT_GRAPH_BINDINGS,
}
_SDXL_PAG_INPAINT_PROFILE = {
    **_SDXL_PAG_PROFILE,
    "id": "sdxl-pag:inpaint-direct",
    "modes": ("inpaint",),
    "pipeline_class": "StableDiffusionXLPAGInpaintPipeline",
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["sdxl-pag:inpaint:v1"] = {
    "modelType": "StableDiffusionXLPAGPipeline",
    "mode": "inpaint",
    "profile": _SDXL_PAG_INPAINT_PROFILE,
    "capability": _SDXL_PAG_CAPABILITY,
    "roles": _INPAINT_GRAPH_ROLES,
    "edges": _INPAINT_GRAPH_EDGES,
    "bindings": _PAG_INPAINT_GRAPH_BINDINGS,
}


_SANA_PROFILE = {
    "id": "sana-600m:direct",
    "model_type": "SanaPipeline",
    "modes": ("text_to_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "SanaPipeline",
    "default_repo": SANA_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 1024,
    "max_low_memory_steps": 20,
    "live_proof": False,
    "compatible_repos": (),
}
_SANA_CAPABILITY = {
    "modelType": "SanaPipeline",
    "label": "Sana 0.6B",
    "displayName": "Sana 0.6B 1024px",
    "family": "Sana",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": SANA_REPO,
    "artifactLabel": "Diffusers fp16 safetensors repo",
    "defaultDtype": "float16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 20,
    "recommendedGuidance": 4.5,
    "recommendedMaxSequenceLength": 300,
    "guidanceLabel": "Guidance",
    "supportsNegativePrompt": True,
    "supportsImageInput": False,
    "supportsMask": False,
    "supportsMultiImage": False,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": False,
    "outputKind": "image",
    "offloadSupport": {
        "default": OFFLOAD_MODE_MODEL_CPU,
        "lowVram": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "emergency": OFFLOAD_MODE_GROUP_DISK,
        "modes": list(_DIRECT_OFFLOAD_MODES),
    },
    "lowVram": {
        "dtype": "float16",
        "autoOffload": True,
        "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "steps": 20,
        "width": 1024,
        "height": 1024,
    },
    "modes": ["text_to_image"],
    "modeRequirements": {},
    "executionStatus": "expert_only",
    "revisionCandidates": [require_catalog_revision(SANA_REPO, model_type="SanaPipeline")],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The immutable 0.6B snapshot loads its reviewed fp16 variant with the text encoder and VAE placed in bfloat16 as documented upstream.",
        "The selected snapshot is approximately 7.70 GB before cache overhead; model and sequential CPU offload are bounded retry paths.",
        "Apache-2.0 applies alongside the bundled Gemma terms and prohibited-use policy; Auto and Gallery remain disabled pending live review.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["sana-600m:text-to-image:v1"] = {
    "modelType": "SanaPipeline",
    "mode": "text_to_image",
    "profile": _SANA_PROFILE,
    "capability": _SANA_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _SDXL_GRAPH_BINDINGS,
}

_SANA_SPRINT_PROFILE = {
    "id": "sana-sprint-600m:direct",
    "model_type": "SanaSprintPipeline",
    "modes": ("text_to_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "SanaSprintPipeline",
    "default_repo": SANA_SPRINT_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 1024,
    "max_low_memory_steps": 4,
    "live_proof": False,
    "compatible_repos": (),
}
_SANA_SPRINT_CAPABILITY = {
    "modelType": "SanaSprintPipeline",
    "label": "Sana Sprint 0.6B",
    "displayName": "Sana Sprint 0.6B 1024px",
    "family": "Sana",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": SANA_SPRINT_REPO,
    "artifactLabel": "Diffusers bfloat16 safetensors repo",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 2,
    "recommendedGuidance": 4.5,
    "recommendedStrength": 0.5,
    "recommendedMaxSequenceLength": 300,
    "guidanceLabel": "Guidance",
    "supportsNegativePrompt": False,
    "supportsImageInput": True,
    "supportsMask": False,
    "supportsMultiImage": False,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": False,
    "outputKind": "image",
    "offloadSupport": {
        "default": OFFLOAD_MODE_MODEL_CPU,
        "lowVram": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "emergency": OFFLOAD_MODE_GROUP_DISK,
        "modes": list(_DIRECT_OFFLOAD_MODES),
    },
    "lowVram": {
        "dtype": "bfloat16",
        "autoOffload": True,
        "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "steps": 2,
        "width": 1024,
        "height": 1024,
    },
    "modes": ["text_to_image", "edit_image"],
    "modeRequirements": {
        "edit_image": {
            "requiredImages": ["referenceImages"],
            "note": "Requires one source image for the reviewed two-step Sprint transformation.",
        },
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(SANA_SPRINT_REPO, model_type="SanaSprintPipeline")
    ],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The immutable 0.6B Sprint snapshot uses its native bfloat16 safetensors and one-to-four-step scheduler contract.",
        "The selected snapshot is approximately 7.70 GB before cache overhead; the reviewed recipe uses two steps and image strength 0.5.",
        "Apache-2.0 applies alongside the bundled Gemma terms and prohibited-use policy; Auto and Gallery remain disabled pending live review.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["sana-sprint-600m:text-to-image:v1"] = {
    "modelType": "SanaSprintPipeline",
    "mode": "text_to_image",
    "profile": _SANA_SPRINT_PROFILE,
    "capability": _SANA_SPRINT_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _SDXL_GRAPH_BINDINGS,
}
_SANA_SPRINT_IMG2IMG_PROFILE = {
    **_SANA_SPRINT_PROFILE,
    "id": "sana-sprint-600m:img2img-direct",
    "modes": ("edit_image",),
    "pipeline_class": "SanaSprintImg2ImgPipeline",
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["sana-sprint-600m:edit-image:v1"] = {
    "modelType": "SanaSprintPipeline",
    "mode": "edit_image",
    "profile": _SANA_SPRINT_IMG2IMG_PROFILE,
    "capability": _SANA_SPRINT_CAPABILITY,
    "roles": _EDIT_GRAPH_ROLES,
    "edges": _EDIT_GRAPH_EDGES,
    "bindings": _SDXL_EDIT_GRAPH_BINDINGS,
}


_PIXART_SIGMA_PROFILE = {
    "id": "pixart-sigma-1024:direct",
    "model_type": "PixArtSigmaPipeline",
    "modes": ("text_to_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "PixArtSigmaPipeline",
    "default_repo": PIXART_SIGMA_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 1024,
    "max_low_memory_steps": 20,
    "live_proof": False,
    "compatible_repos": (),
}
_PIXART_SIGMA_CAPABILITY = {
    "modelType": "PixArtSigmaPipeline",
    "label": "PixArt Sigma",
    "displayName": "PixArt Sigma XL 1024px",
    "family": "PixArt",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": PIXART_SIGMA_REPO,
    "artifactLabel": "OpenRAIL++ Diffusers safetensors repo",
    "defaultDtype": "float16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 20,
    "recommendedGuidance": 4.5,
    "recommendedMaxSequenceLength": 300,
    "guidanceLabel": "Guidance",
    "supportsNegativePrompt": True,
    "supportsImageInput": False,
    "supportsMask": False,
    "supportsMultiImage": False,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": False,
    "outputKind": "image",
    "offloadSupport": {
        "default": OFFLOAD_MODE_MODEL_CPU,
        "lowVram": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "emergency": OFFLOAD_MODE_GROUP_DISK,
        "modes": list(_DIRECT_OFFLOAD_MODES),
    },
    "lowVram": {
        "dtype": "float16",
        "autoOffload": True,
        "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "steps": 20,
        "width": 1024,
        "height": 1024,
    },
    "modes": ["text_to_image"],
    "modeRequirements": {},
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(PIXART_SIGMA_REPO, model_type="PixArtSigmaPipeline")
    ],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The immutable public snapshot contains four safetensors files and uses only package-owned Diffusers and Transformers classes.",
        "The reviewed 1024px recipe uses 20 steps, guidance 4.5, at most 300 prompt tokens, and explicit model or sequential CPU offload.",
        "The approximately 21.83 GB selected weight surface is remote-only; Auto and Gallery remain disabled pending live output review.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["pixart-sigma-1024:text-to-image:v1"] = {
    "modelType": "PixArtSigmaPipeline",
    "mode": "text_to_image",
    "profile": _PIXART_SIGMA_PROFILE,
    "capability": _PIXART_SIGMA_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _SDXL_GRAPH_BINDINGS,
}


_AURAFLOW_V03_PROFILE = {
    "id": "auraflow-v0.3:direct",
    "model_type": "AuraFlowPipeline",
    "modes": ("text_to_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "AuraFlowPipeline",
    "default_repo": AURAFLOW_V03_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 1536,
    "max_low_memory_steps": 50,
    "live_proof": False,
    "compatible_repos": (),
}
_AURAFLOW_V03_CAPABILITY = {
    "modelType": "AuraFlowPipeline",
    "label": "AuraFlow",
    "displayName": "AuraFlow v0.3 1536px",
    "family": "AuraFlow",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": AURAFLOW_V03_REPO,
    "artifactLabel": "Apache-2.0 fp16 Diffusers safetensors repo",
    "defaultDtype": "float16",
    "defaultSize": {"width": 1536, "height": 768, "aspectRatio": "custom"},
    "recommendedSteps": 50,
    "recommendedGuidance": 3.5,
    "recommendedMaxSequenceLength": 256,
    "guidanceLabel": "Guidance",
    "supportsNegativePrompt": True,
    "supportsImageInput": False,
    "supportsMask": False,
    "supportsMultiImage": False,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": False,
    "outputKind": "image",
    "offloadSupport": {
        "default": OFFLOAD_MODE_MODEL_CPU,
        "lowVram": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "emergency": OFFLOAD_MODE_GROUP_DISK,
        "modes": list(_DIRECT_OFFLOAD_MODES),
    },
    "lowVram": {
        "dtype": "float16",
        "autoOffload": True,
        "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "steps": 50,
        "width": 1536,
        "height": 768,
    },
    "modes": ["text_to_image"],
    "modeRequirements": {},
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(AURAFLOW_V03_REPO, model_type="AuraFlowPipeline")
    ],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The immutable public snapshot contains only package-owned Diffusers and Transformers code and a four-file fp16 safetensors partition.",
        "The reviewed native recipe uses a 1536x768 canvas, 50 steps, guidance 3.5, at most 256 prompt tokens, and explicit model or sequential CPU offload.",
        "The approximately 16.84 GB selected weight surface is remote-only; Auto and Gallery remain disabled pending live output review.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["auraflow-v0.3:text-to-image:v1"] = {
    "modelType": "AuraFlowPipeline",
    "mode": "text_to_image",
    "profile": _AURAFLOW_V03_PROFILE,
    "capability": _AURAFLOW_V03_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _SDXL_GRAPH_BINDINGS,
}


_CHROMA1_HD_PROFILE = {
    "id": "chroma1-hd:direct",
    "model_type": "ChromaPipeline",
    "modes": ("text_to_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "ChromaPipeline",
    "default_repo": CHROMA1_HD_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 1024,
    "max_low_memory_steps": 40,
    "live_proof": False,
    "compatible_repos": (),
}
_CHROMA1_HD_CAPABILITY = {
    "modelType": "ChromaPipeline",
    "label": "Chroma",
    "displayName": "Chroma1-HD 1024px",
    "family": "Chroma",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": CHROMA1_HD_REPO,
    "artifactLabel": "Apache-2.0 bfloat16 Diffusers safetensors repo",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 40,
    "recommendedGuidance": 3.0,
    "recommendedMaxSequenceLength": 512,
    "guidanceLabel": "Guidance",
    "supportsNegativePrompt": True,
    "supportsImageInput": False,
    "supportsMask": False,
    "supportsMultiImage": False,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": False,
    "outputKind": "image",
    "offloadSupport": {
        "default": OFFLOAD_MODE_MODEL_CPU,
        "lowVram": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "emergency": OFFLOAD_MODE_GROUP_DISK,
        "modes": list(_DIRECT_OFFLOAD_MODES),
    },
    "lowVram": {
        "dtype": "bfloat16",
        "autoOffload": True,
        "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "steps": 40,
        "width": 1024,
        "height": 1024,
    },
    "modes": ["text_to_image"],
    "modeRequirements": {},
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(CHROMA1_HD_REPO, model_type="ChromaPipeline")
    ],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The immutable public snapshot uses package-owned Diffusers and Transformers code and a five-file bfloat16 safetensors partition.",
        "The reviewed recipe is bounded to 1024x1024, 40 steps, guidance 3, at most 512 prompt tokens, and explicit model or sequential CPU offload.",
        "The approximately 27.49 GB selected weight surface is remote-only; image-to-image, Auto, and Gallery remain disabled pending live review.",
        "The upstream model card states that the model has no safety alignment, so output safety review is an explicit unresolved gate.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["chroma1-hd:text-to-image:v1"] = {
    "modelType": "ChromaPipeline",
    "mode": "text_to_image",
    "profile": _CHROMA1_HD_PROFILE,
    "capability": _CHROMA1_HD_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _SDXL_GRAPH_BINDINGS,
}


_DREAMLITE_BASE_PROFILE = {
    "id": "dreamlite-base:direct",
    "model_type": "DreamLitePipeline",
    "modes": ("text_to_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "DreamLitePipeline",
    "default_repo": DREAMLITE_BASE_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 1024,
    "max_low_memory_steps": 28,
    "live_proof": False,
    "compatible_repos": (),
}
_DREAMLITE_BASE_CAPABILITY = {
    "modelType": "DreamLitePipeline",
    "label": "DreamLite Base",
    "displayName": "DreamLite Base 1024px",
    "family": "DreamLite",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": DREAMLITE_BASE_REPO,
    "artifactLabel": "Non-commercial Diffusers safetensors repo",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 28,
    "recommendedGuidance": 3.5,
    "recommendedMaxSequenceLength": 200,
    "guidanceLabel": "Text guidance",
    "conditioningScale": 1.5,
    "supportsNegativePrompt": True,
    "supportsImageInput": True,
    "supportsMask": False,
    "supportsMultiImage": False,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": False,
    "outputKind": "image",
    "offloadSupport": {
        "default": OFFLOAD_MODE_MODEL_CPU,
        "lowVram": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "emergency": OFFLOAD_MODE_GROUP_DISK,
        "modes": list(_DIRECT_OFFLOAD_MODES),
    },
    "lowVram": {
        "dtype": "bfloat16",
        "autoOffload": True,
        "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "steps": 28,
        "width": 1024,
        "height": 1024,
    },
    "modes": ["text_to_image", "edit_image"],
    "modeRequirements": {
        "edit_image": {
            "requiredImages": ["referenceImages"],
            "note": "Requires one source image for instruction-guided dual-CFG editing.",
        },
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(DREAMLITE_BASE_REPO, model_type="DreamLitePipeline")
    ],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The immutable diffusers-branch snapshot contains only reviewed safetensors and library-owned classes; repository Python remains disabled.",
        "The 1024px recipe uses 28 steps, text guidance 3.5, image guidance 1.5 for edit mode, and at most 200 prompt tokens.",
        "CC-BY-NC-4.0 prohibits commercial use. Auto and Gallery remain disabled pending live remote and physical macOS review.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["dreamlite-base:text-to-image:v1"] = {
    "modelType": "DreamLitePipeline",
    "mode": "text_to_image",
    "profile": _DREAMLITE_BASE_PROFILE,
    "capability": _DREAMLITE_BASE_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _DREAMLITE_GRAPH_BINDINGS,
}
_DREAMLITE_BASE_EDIT_PROFILE = {
    **_DREAMLITE_BASE_PROFILE,
    "id": "dreamlite-base:edit-direct",
    "modes": ("edit_image",),
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["dreamlite-base:edit-image:v1"] = {
    "modelType": "DreamLitePipeline",
    "mode": "edit_image",
    "profile": _DREAMLITE_BASE_EDIT_PROFILE,
    "capability": _DREAMLITE_BASE_CAPABILITY,
    "roles": _EDIT_GRAPH_ROLES,
    "edges": _EDIT_GRAPH_EDGES,
    "bindings": _DREAMLITE_EDIT_GRAPH_BINDINGS,
}

_DREAMLITE_MOBILE_PROFILE = {
    "id": "dreamlite-mobile:direct",
    "model_type": "DreamLiteMobilePipeline",
    "modes": ("text_to_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "DreamLiteMobilePipeline",
    "default_repo": DREAMLITE_MOBILE_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 1024,
    "max_low_memory_steps": 8,
    "live_proof": False,
    "compatible_repos": (),
}
_DREAMLITE_MOBILE_CAPABILITY = {
    "modelType": "DreamLiteMobilePipeline",
    "label": "DreamLite Mobile",
    "displayName": "DreamLite Mobile 1024px",
    "family": "DreamLite",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": DREAMLITE_MOBILE_REPO,
    "artifactLabel": "Non-commercial Diffusers safetensors repo",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 4,
    "recommendedGuidance": 0.0,
    "recommendedMaxSequenceLength": 200,
    "guidanceLabel": "Distilled; guidance disabled",
    "conditioningScale": 0.0,
    "supportsNegativePrompt": False,
    "supportsImageInput": True,
    "supportsMask": False,
    "supportsMultiImage": False,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": False,
    "outputKind": "image",
    "offloadSupport": {
        "default": OFFLOAD_MODE_MODEL_CPU,
        "lowVram": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "emergency": OFFLOAD_MODE_GROUP_DISK,
        "modes": list(_DIRECT_OFFLOAD_MODES),
    },
    "lowVram": {
        "dtype": "bfloat16",
        "autoOffload": True,
        "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "steps": 4,
        "width": 1024,
        "height": 1024,
    },
    "modes": ["text_to_image", "edit_image"],
    "modeRequirements": {
        "edit_image": {
            "requiredImages": ["referenceImages"],
            "note": "Requires one source image for distilled instruction editing.",
        },
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(
            DREAMLITE_MOBILE_REPO,
            model_type="DreamLiteMobilePipeline",
        )
    ],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The immutable diffusers-branch snapshot contains only reviewed safetensors and library-owned classes; repository Python remains disabled.",
        "The distilled 1024px recipe supports one through eight steps and uses four by default; text and image guidance inputs are intentionally not sent because the pipeline ignores them.",
        "CC-BY-NC-4.0 prohibits commercial use. Auto and Gallery remain disabled pending live remote and physical macOS review.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["dreamlite-mobile:text-to-image:v1"] = {
    "modelType": "DreamLiteMobilePipeline",
    "mode": "text_to_image",
    "profile": _DREAMLITE_MOBILE_PROFILE,
    "capability": _DREAMLITE_MOBILE_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _DREAMLITE_MOBILE_GRAPH_BINDINGS,
}
_DREAMLITE_MOBILE_EDIT_PROFILE = {
    **_DREAMLITE_MOBILE_PROFILE,
    "id": "dreamlite-mobile:edit-direct",
    "modes": ("edit_image",),
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["dreamlite-mobile:edit-image:v1"] = {
    "modelType": "DreamLiteMobilePipeline",
    "mode": "edit_image",
    "profile": _DREAMLITE_MOBILE_EDIT_PROFILE,
    "capability": _DREAMLITE_MOBILE_CAPABILITY,
    "roles": _EDIT_GRAPH_ROLES,
    "edges": _EDIT_GRAPH_EDGES,
    "bindings": _DREAMLITE_MOBILE_EDIT_GRAPH_BINDINGS,
}


_LCM_PROFILE = {
    "id": "lcm-dreamshaper-v7:direct",
    "model_type": "LatentConsistencyModelPipeline",
    "modes": ("text_to_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "LatentConsistencyModelPipeline",
    "default_repo": LCM_DREAMSHAPER_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 512,
    "max_low_memory_steps": 4,
    "live_proof": True,
    "compatible_repos": (),
}
_LCM_CAPABILITY = {
    "modelType": "LatentConsistencyModelPipeline",
    "label": "LCM DreamShaper v7",
    "displayName": "LCM DreamShaper v7",
    "family": "Latent Consistency Models",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": LCM_DREAMSHAPER_REPO,
    "artifactLabel": "Diffusers safetensors repo",
    "defaultDtype": "float32",
    "defaultSize": {"width": 512, "height": 512, "aspectRatio": "1:1"},
    "recommendedSteps": 4,
    "recommendedGuidance": 8.5,
    "guidanceLabel": "Guidance",
    "supportsImageInput": False,
    "supportsMask": False,
    "supportsMultiImage": False,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": False,
    "outputKind": "image",
    "offloadSupport": {
        "default": OFFLOAD_MODE_NONE,
        "lowVram": OFFLOAD_MODE_MODEL_CPU,
        "emergency": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "modes": list(_DIRECT_OFFLOAD_MODES),
    },
    "lowVram": {
        "dtype": "float32",
        "autoOffload": False,
        "offloadMode": OFFLOAD_MODE_NONE,
        "steps": 4,
        "width": 512,
        "height": 512,
    },
    "modes": ["text_to_image"],
    "modeRequirements": {},
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(LCM_DREAMSHAPER_REPO, model_type="LatentConsistencyModelPipeline")
    ],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The exact LCM checkpoint supports one-to-four-step text-to-image generation through the generic image node.",
        "Auto and Gallery remain disabled until exact live output qualification is reviewed.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["lcm-dreamshaper-v7:text-to-image:v1"] = {
    "modelType": "LatentConsistencyModelPipeline",
    "mode": "text_to_image",
    "profile": _LCM_PROFILE,
    "capability": _LCM_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _SDXL_GRAPH_BINDINGS,
}

_PAG_PROFILE = {
    "id": "sd15-pag:direct",
    "model_type": "StableDiffusionPAGPipeline",
    "modes": ("text_to_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "StableDiffusionPAGPipeline",
    "default_repo": SD15_BASE_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 512,
    "max_low_memory_steps": 30,
    "live_proof": True,
    "compatible_repos": (),
}
_PAG_CAPABILITY = {
    "modelType": "StableDiffusionPAGPipeline",
    "label": "Stable Diffusion 1.5 PAG",
    "displayName": "Stable Diffusion 1.5 PAG",
    "family": "Stable Diffusion 1.x",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": SD15_BASE_REPO,
    "artifactLabel": "Diffusers safetensors repo",
    "defaultDtype": "float32",
    "defaultSize": {"width": 512, "height": 512, "aspectRatio": "1:1"},
    "recommendedSteps": 30,
    "recommendedGuidance": 7.5,
    "guidanceLabel": "Guidance",
    "supportsImageInput": False,
    "supportsMask": False,
    "supportsMultiImage": False,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": True,
    "outputKind": "image",
    "offloadSupport": {
        "default": OFFLOAD_MODE_NONE,
        "lowVram": OFFLOAD_MODE_MODEL_CPU,
        "emergency": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "modes": list(_DIRECT_OFFLOAD_MODES),
    },
    "lowVram": {
        "dtype": "float32",
        "autoOffload": False,
        "offloadMode": OFFLOAD_MODE_NONE,
        "steps": 20,
        "width": 512,
        "height": 512,
    },
    "modes": ["text_to_image"],
    "modeRequirements": {},
    "executionStatus": "expert_only",
    "revisionCandidates": [require_catalog_revision(SD15_BASE_REPO, model_type="StableDiffusionPAGPipeline")],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "Perturbed-attention guidance reuses the immutable Stable Diffusion 1.5 safetensors base.",
        "Auto and Gallery remain disabled until exact live output qualification is reviewed.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["sd15-pag:text-to-image:v1"] = {
    "modelType": "StableDiffusionPAGPipeline",
    "mode": "text_to_image",
    "profile": _PAG_PROFILE,
    "capability": _PAG_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _PAG_GRAPH_BINDINGS,
}

_MARIGOLD_DEPTH_PROFILE = {
    "id": "marigold-depth-lcm-v1-0:direct",
    "model_type": "MarigoldDepthPipeline",
    "modes": ("depth_estimation",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "MarigoldDepthPipeline",
    "default_repo": MARIGOLD_DEPTH_LCM_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 768,
    "max_low_memory_steps": 1,
    "live_proof": True,
    "compatible_repos": (),
}
_MARIGOLD_DEPTH_CAPABILITY = {
    "modelType": "MarigoldDepthPipeline",
    "label": "Marigold Depth LCM v1.0",
    "displayName": "Marigold Depth LCM v1.0",
    "family": "Marigold",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": MARIGOLD_DEPTH_LCM_REPO,
    "artifactLabel": "Diffusers safetensors repo",
    "defaultDtype": "float32",
    "defaultSize": {"width": 768, "height": 768, "aspectRatio": "1:1"},
    "recommendedSteps": 1,
    "recommendedGuidance": 0.0,
    "guidanceLabel": "Not used",
    "supportsNegativePrompt": False,
    "supportsImageInput": True,
    "supportsMask": False,
    "supportsMultiImage": False,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": False,
    "outputKind": "image",
    "offloadSupport": {
        "default": OFFLOAD_MODE_NONE,
        "lowVram": OFFLOAD_MODE_MODEL_CPU,
        "emergency": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "modes": list(_DIRECT_OFFLOAD_MODES),
    },
    "lowVram": {
        "dtype": "float32",
        "autoOffload": False,
        "offloadMode": OFFLOAD_MODE_NONE,
        "steps": 1,
        "width": 768,
        "height": 768,
    },
    "modes": ["depth_estimation"],
    "modeRequirements": {
        "depth_estimation": {
            "requiredImages": ["referenceImages"],
            "note": "Requires exactly one source image; output is a normalized relative-depth map.",
        }
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(MARIGOLD_DEPTH_LCM_REPO, model_type="MarigoldDepthPipeline")
    ],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The generic Predict Map node returns a versioned float32 NHWC relative-depth map plus a grayscale preview.",
        "Normals, intrinsics, uncertainty, Auto, and Gallery remain disabled pending separate qualification.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["marigold-depth-lcm-v1-0:depth-estimation:v1"] = {
    "modelType": "MarigoldDepthPipeline",
    "mode": "depth_estimation",
    "profile": _MARIGOLD_DEPTH_PROFILE,
    "capability": _MARIGOLD_DEPTH_CAPABILITY,
    "roles": _PERCEPTION_GRAPH_ROLES,
    "edges": _PERCEPTION_GRAPH_EDGES,
    "bindings": _PERCEPTION_GRAPH_BINDINGS,
}

_WHISPER_TINY_PROFILE = {
    "id": "whisper-tiny:direct",
    "model_type": "HuggingFaceSpeechRecognitionModel",
    "modes": ("speech_to_text", "speech_translation"),
    "loader_module": "modules.HuggingFaceSpeech",
    "loader_action": "LoadSpeechRecognitionModel",
    "execution_path": "direct-huggingface-speech",
    "pipeline_class": "AutoModelForSpeechSeq2Seq",
    "default_repo": WHISPER_TINY_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": (OFFLOAD_MODE_NONE,),
    "retry_offload_modes": (),
    "max_low_memory_side": None,
    "max_low_memory_steps": None,
    "live_proof": True,
    "compatible_repos": (),
}
_WHISPER_TINY_CAPABILITY = {
    "modelType": "HuggingFaceSpeechRecognitionModel",
    "label": "Whisper Tiny",
    "displayName": "Whisper Tiny",
    "family": "Whisper",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": WHISPER_TINY_REPO,
    "artifactLabel": "Transformers safetensors repo",
    "defaultDtype": "float32",
    "defaultSize": {"width": 1, "height": 1, "aspectRatio": "audio"},
    "recommendedSteps": 1,
    "recommendedGuidance": 0.0,
    "guidanceLabel": "Not used",
    "supportsNegativePrompt": False,
    "supportsImageInput": False,
    "supportsAudioInput": True,
    "supportsMask": False,
    "supportsMultiImage": False,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": False,
    "outputKind": "json",
    "offloadSupport": {
        "default": OFFLOAD_MODE_NONE,
        "lowVram": OFFLOAD_MODE_NONE,
        "emergency": OFFLOAD_MODE_NONE,
        "modes": [OFFLOAD_MODE_NONE],
    },
    "lowVram": {
        "dtype": "float32",
        "autoOffload": False,
        "offloadMode": OFFLOAD_MODE_NONE,
        "steps": 1,
        "width": 1,
        "height": 1,
    },
    "modes": ["speech_to_text", "speech_translation"],
    "modeRequirements": {
        "speech_to_text": {
            "requiredAudio": ["sourceAudio"],
            "note": "Requires one bounded local audio source and returns a normalized transcript.",
        },
        "speech_translation": {
            "requiredAudio": ["sourceAudio"],
            "note": "Requires one bounded local audio source and translates recognized speech to English.",
        },
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(WHISPER_TINY_REPO, model_type="HuggingFaceSpeechRecognitionModel")
    ],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The generic speech boundary returns a versioned transcript, plain text, timestamp segments, and duration.",
        "Auto and Gallery remain disabled until exact live output qualification is reviewed.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["whisper-tiny:speech-to-text:v1"] = {
    "modelType": "HuggingFaceSpeechRecognitionModel",
    "mode": "speech_to_text",
    "profile": _WHISPER_TINY_PROFILE,
    "capability": _WHISPER_TINY_CAPABILITY,
    "roles": _SPEECH_GRAPH_ROLES,
    "edges": _SPEECH_GRAPH_EDGES,
    "bindings": _SPEECH_TRANSCRIPTION_GRAPH_BINDINGS,
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["whisper-tiny:speech-translation:v1"] = {
    "modelType": "HuggingFaceSpeechRecognitionModel",
    "mode": "speech_translation",
    "profile": _WHISPER_TINY_PROFILE,
    "capability": _WHISPER_TINY_CAPABILITY,
    "roles": _SPEECH_GRAPH_ROLES,
    "edges": _SPEECH_GRAPH_EDGES,
    "bindings": _SPEECH_TRANSLATION_GRAPH_BINDINGS,
}

_EXPERT_IMAGE_QUANTIZATION_PROFILE_IDS = {
    "flux-canny:direct",
    "flux-depth:direct",
    "flux-dev:direct",
    "flux-dev:img2img-direct",
    "flux-dev:inpaint-direct",
    "flux-fill:direct",
    "flux-kontext:direct",
    "flux-krea:direct",
    "flux-redux:direct",
    "flux-schnell:direct",
    "flux2-klein:direct",
}
for _definition in STUDIO_EXECUTION_SPEC_DEFINITIONS.values():
    if _definition["profile"]["id"] in _EXPERT_IMAGE_QUANTIZATION_PROFILE_IDS:
        _definition["profile"]["expert_quantization_modes"] = _EXPERT_IMAGE_QUANTIZATION_MODES


def _ordered_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _ordered_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_ordered_value(item) for item in value]
    return value


def _stable_json(value: Any) -> str:
    return json.dumps(_ordered_value(value), ensure_ascii=False, separators=(",", ":"))


def _hash_string(value: str) -> str:
    digest = 0x811C9DC5
    encoded = value.encode("utf-16-le")
    for index in range(0, len(encoded), 2):
        digest ^= encoded[index] | encoded[index + 1] << 8
        digest = digest * 0x01000193 & 0xFFFFFFFF
    return f"{digest:08x}"


def _public_spec(spec_id: str, definition: dict[str, Any]) -> dict[str, Any]:
    profile = definition["profile"]
    payload = {
        "schemaVersion": STUDIO_EXECUTION_SPEC_SCHEMA_VERSION,
        "canonicalizationVersion": STUDIO_EXECUTION_SPEC_CANONICALIZATION_VERSION,
        "id": spec_id,
        "modelType": definition["modelType"],
        "mode": definition["mode"],
        "executionProfileId": profile["id"],
        "loaderModule": profile["loader_module"],
        "loaderAction": profile["loader_action"],
        "executionPath": profile["execution_path"],
        "pipelineClass": profile["pipeline_class"],
        "defaultRepo": profile["default_repo"],
        "roles": definition.get("roles", _GRAPH_ROLES),
        "edges": definition.get("edges", _GRAPH_EDGES),
        "bindings": definition.get("bindings", _GRAPH_BINDINGS),
        "autoFields": _AUTO_FIELDS,
        "actions": (),
    }
    payload["contentHash"] = f"studio-spec-v1-{_hash_string(_stable_json(payload))}"
    return deepcopy(payload)


def studio_execution_profile_definitions() -> dict[str, dict[str, Any]]:
    return {
        definition["profile"]["id"]: deepcopy(definition["profile"])
        for definition in STUDIO_EXECUTION_SPEC_DEFINITIONS.values()
    }


def studio_auto_model_requirements() -> dict[str, dict[str, Any]]:
    return {
        definition.get("autoRequirementKey", definition["modelType"]): deepcopy(definition["autoRequirements"])
        for definition in STUDIO_EXECUTION_SPEC_DEFINITIONS.values()
        if "autoRequirements" in definition
    }


def studio_capability_definitions() -> dict[str, dict[str, Any]]:
    return {
        definition["modelType"]: deepcopy(definition["capability"])
        for definition in STUDIO_EXECUTION_SPEC_DEFINITIONS.values()
        if "capability" in definition
    }


def studio_execution_spec_for_pair(model_type: str, mode: str) -> dict[str, Any] | None:
    matches = [
        _public_spec(spec_id, definition)
        for spec_id, definition in STUDIO_EXECUTION_SPEC_DEFINITIONS.items()
        if definition["modelType"] == model_type and definition["mode"] == mode
    ]
    return matches[0] if len(matches) == 1 else None


def _param_types(param: dict[str, Any]) -> set[str]:
    value = param.get("type")
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {item for item in value if isinstance(item, str)}
    return set()


_MODULAR_NODE_TYPES = {
    "modules.ModularDiffusers.EncodePrompt": "text_encoder",
    "modules.ModularDiffusers.ImageEmbeddings": "image_encoder",
    "modules.ModularDiffusers.ImageEncode": "vae_encoder",
    "modules.ModularDiffusers.Denoise": "denoise",
    "modules.ModularDiffusers.DecodeLatents": "decoder",
    "modules.ModularDiffusers.Controlnet": "controlnet",
}


def _execution_spec_role_params(
    public: dict[str, Any],
    node_key: str,
    node: dict[str, Any],
) -> dict[str, Any]:
    params = deepcopy(node["params"])
    node_type = _MODULAR_NODE_TYPES.get(node_key)
    if public["executionPath"] != "modular-diffusers" or node_type is None:
        return params

    # Modular action fields are backend-issued after the Models Loader selects
    # a reviewed pipeline. Validate the public receipt against that same
    # authoritative contract instead of treating the deliberately small static
    # registry definition as the executable schema.
    from modules.ModularDiffusers.modular_utils import (
        pipeline_class_from_model_type,
        require_modiff_node_contract,
    )

    pipeline_class = pipeline_class_from_model_type(public["pipelineClass"])
    _blocks, config = require_modiff_node_contract(
        pipeline_class,
        node_type,
        require_blocks=False,
        resolve_blocks=False,
    )
    params.update(config["params"])
    return params


def validate_studio_execution_specs(modules: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate every public execution-spec reference against the live node registry."""

    output = []
    pairs = set()
    profiles: dict[str, dict[str, Any]] = {}
    for spec_id, definition in STUDIO_EXECUTION_SPEC_DEFINITIONS.items():
        public = _public_spec(spec_id, definition)
        pair = (public["modelType"], public["mode"])
        profile_id = public["executionProfileId"]
        profile = definition["profile"]
        if pair in pairs or (profile_id in profiles and profiles[profile_id] != profile):
            raise ValueError("Studio execution specifications must have unique pairs and consistent execution profiles.")
        pairs.add(pair)
        profiles[profile_id] = profile

        roles = {}
        for item in public["roles"]:
            if not isinstance(item, (list, tuple)) or len(item) != 4:
                raise ValueError("Studio execution specification role is invalid.")
            role, node_key, x, y = item
            if (
                not isinstance(role, str)
                or not role
                or role in roles
                or not isinstance(node_key, str)
                or "." not in node_key
                or not isinstance(x, int)
                or not isinstance(y, int)
            ):
                raise ValueError("Studio execution specification role is invalid.")
            module, action = node_key.rsplit(".", 1)
            node = modules.get(module, {}).get(action)
            if not isinstance(node, dict) or not isinstance(node.get("params"), dict):
                raise ValueError("Studio execution specification references an unknown node.")
            roles[role] = {**node, "params": _execution_spec_role_params(public, node_key, node)}

        connections = set()
        adjacency = {role: set() for role in roles}
        for item in public["edges"]:
            if not isinstance(item, (list, tuple)) or len(item) != 4:
                raise ValueError("Studio execution specification edge is invalid.")
            source_role, source_handle, target_role, target_handle = item
            edge = tuple(item)
            if edge in connections or source_role not in roles or target_role not in roles:
                raise ValueError("Studio execution specification edge is invalid.")
            connections.add(edge)
            source_param = roles[source_role]["params"].get(source_handle)
            target_param = roles[target_role]["params"].get(target_handle)
            if (
                not isinstance(source_param, dict)
                or source_param.get("display") != "output"
                or not isinstance(target_param, dict)
                or target_param.get("display") != "input"
                or not (_param_types(source_param) & _param_types(target_param))
            ):
                raise ValueError("Studio execution specification references an incompatible handle.")
            adjacency[source_role].add(target_role)
            adjacency[target_role].add(source_role)
        visited = set()
        pending = [next(iter(roles))]
        while pending:
            role = pending.pop()
            if role in visited:
                continue
            visited.add(role)
            pending.extend(adjacency[role] - visited)
        if visited != set(roles):
            raise ValueError("Studio execution specification graph is disconnected.")

        binding_targets = set()
        for item in public["bindings"]:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise ValueError("Studio execution specification binding is invalid.")
            role, param, source = item
            target = (role, param)
            target_param = roles.get(role, {}).get("params", {}).get(param)
            if (
                role not in roles
                or not isinstance(target_param, dict)
                or target_param.get("display") == "output"
                or source not in _BINDING_SOURCES
                or target in binding_targets
            ):
                raise ValueError("Studio execution specification binding is invalid.")
            binding_targets.add(target)
        if set(public["autoFields"]) != _AUTO_FIELD_ALLOWLIST or public["actions"]:
            raise ValueError("Studio execution specification action or Auto binding is invalid.")

        expected_hash = f"studio-spec-v1-{_hash_string(_stable_json({key: value for key, value in public.items() if key != 'contentHash'}))}"
        if public["contentHash"] != expected_hash:
            raise ValueError("Studio execution specification content hash is invalid.")
        output.append(public)
    return output


def assert_studio_execution_graph(graph: dict[str, Any], runtime_hints: dict[str, Any] | None) -> None:
    """Fail closed when a submitted managed graph does not match its spec receipt."""

    if not isinstance(runtime_hints, dict):
        return
    receipt = runtime_hints.get("studioExecutionSpec")
    candidate = runtime_hints.get("autoResourcePlan")
    candidate_contract = (
        candidate.get("studioExecutionSpecContract")
        if isinstance(candidate, dict)
        else None
    )
    if receipt is None:
        if candidate_contract is not None:
            raise RuntimeError(
                "Studio execution specification receipt is required for this Auto graph. Rebuild the managed graph."
            )
        return
    if not isinstance(receipt, dict):
        raise RuntimeError("Studio execution specification receipt is invalid. Rebuild the managed graph.")
    spec_id = receipt.get("id")
    definition = STUDIO_EXECUTION_SPEC_DEFINITIONS.get(spec_id) if isinstance(spec_id, str) else None
    if definition is None:
        raise RuntimeError("Studio execution specification receipt is unknown. Rebuild the managed graph.")
    spec = _public_spec(spec_id, definition)
    expected_candidate_contract = {
        "schemaVersion": spec["schemaVersion"],
        "id": spec["id"],
        "contentHash": spec["contentHash"],
        "executionProfileId": spec["executionProfileId"],
    }
    if (
        receipt.get("schemaVersion") != STUDIO_EXECUTION_SPEC_SCHEMA_VERSION
        or receipt.get("contentHash") != spec["contentHash"]
        or runtime_hints.get("modelType") != spec["modelType"]
        or runtime_hints.get("mode") != spec["mode"]
    ):
        raise RuntimeError("Studio execution specification receipt does not match this workflow.")
    if candidate_contract is not None and candidate_contract != expected_candidate_contract:
        raise RuntimeError("Studio execution specification does not match the selected Auto graph contract.")
    if isinstance(candidate, dict) and candidate.get("executionProfileId") != spec["executionProfileId"]:
        raise RuntimeError("Studio execution specification does not match the selected Auto profile.")
    node_ids = receipt.get("nodes")
    if not isinstance(node_ids, dict) or set(node_ids) != {item[0] for item in spec["roles"]}:
        raise RuntimeError("Studio execution specification node receipt is invalid. Rebuild the managed graph.")
    nodes = graph.get("nodes")
    paths = graph.get("paths")
    if not isinstance(nodes, dict) or not isinstance(paths, list):
        raise RuntimeError("Studio execution specification graph is invalid. Rebuild the managed graph.")
    executable_ids = {
        str(node_id)
        for path in paths
        if isinstance(path, list)
        for node_id in path
    }
    for role, node_key, _x, _y in spec["roles"]:
        node_id = node_ids.get(role)
        node = nodes.get(node_id) if isinstance(node_id, str) else None
        if (
            not isinstance(node, dict)
            or str(node_id) not in executable_ids
            or f"{node.get('module')}.{node.get('action')}" != node_key
        ):
            raise RuntimeError("Studio execution specification node identity does not match the executable graph.")
    for source_role, source_handle, target_role, target_handle in spec["edges"]:
        source_id = node_ids[source_role]
        target = nodes[node_ids[target_role]]
        param = (target.get("params") or {}).get(target_handle)
        if (
            not isinstance(param, dict)
            or param.get("sourceId") != source_id
            or param.get("sourceKey") != source_handle
        ):
            raise RuntimeError("Studio execution specification edge does not match the executable graph.")
    for role, param, _source in spec["bindings"]:
        node = nodes[node_ids[role]]
        if param not in (node.get("params") or {}):
            raise RuntimeError("Studio execution specification binding is missing from the executable graph.")
